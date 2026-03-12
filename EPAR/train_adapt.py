"""
train_adapt.py — Stage 1: JEPA Adaptation (Context/Target Encoder Fine-tuning)

Supports both MVTec-3D AD and Eyecandies datasets through a single entry-point.
Class-specific hyper-parameters are stored in YAML configs under ./configs/.

Usage:
    # MVTec-3D
    python train_adapt.py --config configs/train_adapt_mvtec.yaml --class_name cookie

    # Eyecandies
    python train_adapt.py --config configs/train_adapt_eyecandies.yaml --class_name CandyCane
"""

import argparse
import os
import copy
import random
import logging
import math
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import yaml
import faiss
from tqdm import tqdm
from torch.utils.data import DataLoader

# ── Model imports ────────────────────────────────────────────────────────────
from models.feature_extractor import RGBBackbone, XYZBackbone
from models.point_feature_alignment import PointFeatureAlignment
from models import encoder_lora as vit
from models.point_transformer_v2 import TransformerPredictor
from models.encoder import VIT_EMBED_DIMS
from models.src.tensors import trunc_normal_, repeat_interleave_batch
from models.src.schedulers import WarmupCosineSchedule, CosineWDSchedule
from models.common import (
    FaissNNRGB, FaissNN3D,
    NearestNeighbourScorerRGB, NearestNeighbourScorer3D,
    RescaleSegmentor,
)
from models.cattention import InfoNCE
from models.memory_bank import MMUpdateRGB, MMUpdate3D, FocalLoss

# MaskCollator — dataset-specific variants imported together
from models.src.blocks_new_v2 import MaskCollator as MaskCollatorMVTec
from models.src.blocks_eyecandies import MaskCollator as MaskCollatorEyecandies
from models.src.apply_masks import apply_masks

import loralib as lora

# ── Utils imports ────────────────────────────────────────────────────────────
from utils.loss import CauchyLoss
from utils.metrics import calculate_au_pro
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str, class_name: str) -> SimpleNamespace:
    """Load YAML config and apply per-class overrides."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Separate per-class section
    classes = cfg.pop("classes", {})

    # Build flat namespace from defaults
    args = SimpleNamespace(**cfg)

    # Apply per-class overrides
    if class_name in classes and classes[class_name]:
        for key, val in classes[class_name].items():
            setattr(args, key, val)

    args.class_name = class_name
    return args


def get_parser():
    parser = argparse.ArgumentParser(description="Stage-1 JEPA Adaptation Trainer")
    parser.add_argument("--config",     type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument("--class_name", type=str, required=True,
                        help="Class name to train on")
    # Optionally override any config key via CLI  --key value
    parser.add_argument("--dataset_path",             type=str)
    parser.add_argument("--output_dir",               type=str)
    parser.add_argument("--visual_encoder_ckpt_pth",  type=str)
    parser.add_argument("--point_encoder_ckpt_pth",   type=str)
    parser.add_argument("--num_epochs",               type=int)
    parser.add_argument("--batch_size",               type=int)
    parser.add_argument("--lr_xyz",                   type=float)
    parser.add_argument("--start_lr_xyz",             type=float)
    parser.add_argument("--final_lr_xyz",             type=float)
    parser.add_argument("--bg_threshold",             type=float)
    parser.add_argument("--npred",                    type=int)
    parser.add_argument("--k_shot",                   type=int)
    return parser


# ─────────────────────────────────────────────────────────────────────────────
# Seed
# ─────────────────────────────────────────────────────────────────────────────

def seed_everything(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id, seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


# ─────────────────────────────────────────────────────────────────────────────
# LoRA replacement
# ─────────────────────────────────────────────────────────────────────────────

def replace_learnable(model, target_block_idxs, r=8, alpha=16,
                      dropout=0.1, mode="lora", modal="rgb",
                      init_lora_weights=True):
    block_list = list(model.blocks.children())
    for block_idx, backbone_block in enumerate(block_list):
        if block_idx not in target_block_idxs:
            continue
        if mode in ("lora", "fulllora"):
            full_match = (mode == "fulllora")
            for full_name, module in list(backbone_block.named_modules()):
                if not full_match and full_name.split('.')[0] != 'attn':
                    continue
                if not isinstance(module, nn.Linear):
                    continue
                path = full_name.split('.')
                parent = backbone_block
                for key in path[:-1]:
                    parent = getattr(parent, key)
                attr_name = path[-1]
                orig: nn.Linear = getattr(parent, attr_name)
                if orig.out_features == orig.in_features * 3:
                    lora_layer = lora.MergedLinear(
                        orig.in_features, orig.out_features,
                        r=r, lora_alpha=alpha, lora_dropout=dropout,
                        enable_lora=[True, False, True],
                        bias=(orig.bias is not None))
                elif full_match:
                    lora_layer = lora.Linear(
                        orig.in_features, orig.out_features,
                        r=r, lora_alpha=alpha, lora_dropout=dropout,
                        bias=(orig.bias is not None))
                else:
                    continue
                with torch.no_grad():
                    lora_layer.weight.copy_(orig.weight)
                    if orig.bias is not None:
                        lora_layer.bias.copy_(orig.bias)
                lora_layer.weight.requires_grad = False
                if orig.bias is not None:
                    lora_layer.bias.requires_grad = False
                if init_lora_weights:
                    lora_layer.reset_parameters()
                setattr(parent, attr_name, lora_layer)
        else:
            # plain fine-tuning
            for p in backbone_block.parameters():
                p.requires_grad = True
    return model


def reinit_selected_block(block: nn.Module, init_std: float = 0.02):
    def _init(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
    block.apply(_init)


# ─────────────────────────────────────────────────────────────────────────────
# Model init helpers
# ─────────────────────────────────────────────────────────────────────────────

def init_predictor(device, num_patches, embed_dim, pred_depth, num_heads,
                   drop_rate=0.0, pred_emb_dim=384):
    predictor = vit.__dict__['vit_predictor'](
        num_patches=num_patches,
        embed_dim=embed_dim,
        predictor_embed_dim=pred_emb_dim,
        depth=pred_depth,
        num_heads=num_heads,
    )
    def _init(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    predictor.apply(_init)
    return predictor.to(device)


def init_optimizer(encoder, predictor_list, iterations_per_epoch,
                   start_lr, ref_lr, warmup, num_epochs,
                   wd=1e-6, final_wd=1e-6, final_lr=0.0,
                   use_bfloat16=False, ipe_scale=1.0, adapter=None):
    layers = [encoder] + predictor_list
    if adapter is not None:
        layers.append(adapter)
    w_decay, wo_decay = [], []
    for layer in layers:
        for n, p in layer.named_parameters():
            if n.endswith(".bias") or p.ndim == 1:
                wo_decay.append(p)
            else:
                w_decay.append(p)
    param_groups = [
        {'params': w_decay,  'weight_decay': wd},
        {'params': wo_decay, 'weight_decay': 0.0},
    ]
    optimizer = optim.AdamW(param_groups, lr=ref_lr)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        start_lr=start_lr, ref_lr=ref_lr, final_lr=final_lr,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch))
    wd_scheduler = CosineWDSchedule(
        optimizer, ref_wd=wd, final_wd=final_wd,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch))
    scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None
    return optimizer, scaler, scheduler, wd_scheduler


# ─────────────────────────────────────────────────────────────────────────────
# Image/patch utilities
# ─────────────────────────────────────────────────────────────────────────────

def image_to_patches(image, patch_size):
    if image.shape[0] == 3:
        image = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
        image = image.unsqueeze(0)
    C, H, W = image.shape
    patches = image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.contiguous().view(C, -1, patch_size, patch_size)
    patches = patches.permute(1, 0, 2, 3)
    N, C, P, _ = patches.shape
    return patches.reshape(N, C * P * P)


def find_significant_patches(patches, bg_threshold, bg_ratio_threshold):
    N, patch_dim = patches.shape
    P = int(patch_dim ** 0.5)
    patches = patches.view(N, 1, P, P)
    bg_pixel_mask = (patches <= bg_threshold).float()
    bg_proportion = bg_pixel_mask.mean(dim=[1, 2, 3])
    return torch.where(bg_proportion < bg_ratio_threshold)[0].tolist()


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


# ─────────────────────────────────────────────────────────────────────────────
# Target-block builder (dataset-specific)
# ─────────────────────────────────────────────────────────────────────────────

def make_tgt_block_mvtec(args, bsz, patch_size, foreground_indices,
                          background_indices, pred_mask_scale):
    """MVTec variant: rectangular target region + background exclusion."""
    MaskCollator = MaskCollatorMVTec
    target_block_complements = torch.zeros(
        [bsz, args.img_size // patch_size, args.img_size // patch_size])
    min_row, max_row, min_col, max_col = 9999, 0, 9999, 0
    min_keep = (args.img_size // patch_size) ** 2
    for i in range(bsz):
        f_idx = np.array(foreground_indices[i])
        min_keep = min(int(len(f_idx) * pred_mask_scale[0]), min_keep)
        f_rows = f_idx // (args.img_size // patch_size)
        f_cols = f_idx % (args.img_size // patch_size)
        target_block_complements[i][f_rows, f_cols] = 1.0
        idxs = np.array([[j // (args.img_size // patch_size),
                           j %  (args.img_size // patch_size)]
                          for j in foreground_indices[i]])
        min_row = min(min_row, idxs[:, 0].min())
        max_row = max(max_row, idxs[:, 0].max())
        min_col = min(min_col, idxs[:, 1].min())
        max_col = max(max_col, idxs[:, 1].max())

    row_range = [min_row, max_row]
    col_range = [min_col, max_col]

    # Expand to rectangular bounding box
    for b in range(bsz):
        for i in range(row_range[0], row_range[1] + 1):
            for j in range(col_range[0], col_range[1] + 1):
                target_block_complements[b][i, j] = 1

    mask_collator = MaskCollator(
        input_size=(args.img_size, args.img_size),
        patch_size=patch_size,
        pred_mask_scale=pred_mask_scale,
        aspect_ratio=args.aspect_ratio,
        npred=args.npred,
        min_keep=min_keep,
        allow_overlap=False,
        mode="train")

    context_indices = torch.arange(
        (args.img_size // patch_size) ** 2)
    exclude_indices = torch.tensor(background_indices).flatten()
    mask = torch.ones(context_indices.shape[0], dtype=torch.bool)
    mask[exclude_indices] = False
    context_indices = context_indices[mask]

    context_blocks, target_blocks = mask_collator(
        bsz=bsz,
        row_range=row_range,
        col_range=col_range,
        target_acceptable_regions=target_block_complements,
        background_indices=background_indices,
        context_indices=context_indices)

    return (context_blocks, target_blocks,
            target_block_complements.reshape(bsz, (args.img_size // patch_size) ** 2))


def make_tgt_block_eyecandies(args, bsz, patch_size, foreground_indices,
                               pred_mask_scale):
    """Eyecandies variant: no rectangular expansion, no bg exclusion."""
    MaskCollator = MaskCollatorEyecandies
    target_block_complements = torch.zeros(
        [bsz, args.img_size // patch_size, args.img_size // patch_size])
    min_row, max_row, min_col, max_col = 9999, 0, 9999, 0
    for i in range(bsz):
        f_idx = np.array(foreground_indices[i])
        f_rows = f_idx // (args.img_size // patch_size)
        f_cols = f_idx % (args.img_size // patch_size)
        target_block_complements[i][f_rows, f_cols] = 1.0
        idxs = np.array([[j // (args.img_size // patch_size),
                           j %  (args.img_size // patch_size)]
                          for j in foreground_indices[i]])
        min_row = min(min_row, idxs[:, 0].min())
        max_row = max(max_row, idxs[:, 0].max())
        min_col = min(min_col, idxs[:, 1].min())
        max_col = max(max_col, idxs[:, 1].max())

    row_range = [min_row, max_row]
    col_range = [min_col, max_col]

    mask_collator = MaskCollator(
        input_size=(args.img_size, args.img_size),
        patch_size=patch_size,
        pred_mask_scale=pred_mask_scale,
        aspect_ratio=args.aspect_ratio,
        npred=args.npred,
        min_keep=args.min_keep,
        allow_overlap=False,
        mode="train")

    context_indices = torch.arange((args.img_size // patch_size) ** 2)

    context_blocks, target_blocks = mask_collator(
        bsz=bsz,
        row_range=row_range,
        col_range=col_range,
        target_acceptable_regions=target_block_complements,
        context_indices=context_indices)

    return (context_blocks, target_blocks,
            target_block_complements.reshape(bsz, (args.img_size // patch_size) ** 2))


# ─────────────────────────────────────────────────────────────────────────────
# Loss helpers
# ─────────────────────────────────────────────────────────────────────────────

def loss_fn(z, h, loss_type="smooth"):
    if loss_type == "smooth":
        return F.smooth_l1_loss(z, h)
    if loss_type == "cosine":
        nl = z.size(0)
        loss = 0.0
        for i in range(nl):
            z_ = F.normalize(z[i], p=2, dim=1)
            h_ = F.normalize(h[i], p=2, dim=1)
            loss += F.cosine_embedding_loss(
                z_, h_, torch.ones(z.size(1)).to(z.device))
        return loss / nl


def vic_regularizer(features_ctx, features_tgt, device, bsz):
    """VICReg: Invariance + Variance + Covariance."""
    cos_sim = nn.CosineEmbeddingLoss()
    _, npatches, d = features_ctx.shape

    # Invariance
    invar = 0.0
    for b in range(bsz):
        invar += cos_sim(
            F.normalize(features_ctx[b], dim=1),
            F.normalize(features_tgt[b], dim=1).detach(),
            torch.ones(npatches).to(device))
    invar /= bsz

    # Variance + Covariance
    var_loss, cov_loss = 0.0, 0.0
    for i in range(bsz):
        z = features_ctx[i] - features_ctx[i].mean(dim=0)
        std = torch.sqrt(z.var(dim=0) + 1e-4)
        var_loss += torch.mean(F.relu(1 - std))
        cov = (z.T @ z) / (npatches - 1)
        cov_loss += off_diagonal(cov).pow_(2).sum().div(d)
    var_loss /= bsz
    cov_loss /= bsz

    return 15 * invar + 15 * var_loss + cov_loss


# ─────────────────────────────────────────────────────────────────────────────
# Alignment projection module
# ─────────────────────────────────────────────────────────────────────────────

class Alignment(nn.Module):
    def __init__(self, rgb_dim, xyz_dim, proj_dim=768):
        super().__init__()
        mid = (rgb_dim + xyz_dim) // 2
        self.rgb_proj = nn.Sequential(
            nn.Linear(rgb_dim, mid), nn.GELU(),
            nn.Linear(mid, proj_dim), nn.LayerNorm(proj_dim))
        self.xyz_proj = nn.Sequential(
            nn.Linear(xyz_dim, mid), nn.GELU(),
            nn.Linear(mid, proj_dim), nn.LayerNorm(proj_dim))

    def forward(self, rgb, xyz):
        rgb = self.rgb_proj(rgb)
        B, P, D = xyz.shape
        H = W = int(P ** 0.5)
        xyz = xyz.permute(0, 2, 1).view(B, D, H, W)
        xyz = F.avg_pool2d(xyz, kernel_size=2, stride=2)
        B, D, H, W = xyz.shape
        xyz = xyz.permute(0, 2, 3, 1).view(B, H * W, D)
        xyz = self.xyz_proj(xyz)
        return rgb, xyz


# ─────────────────────────────────────────────────────────────────────────────
# Dataset class maps
# ─────────────────────────────────────────────────────────────────────────────

MVTEC_CLASSES = {
    "bagel": 0, "cable_gland": 1, "carrot": 2, "cookie": 3,
    "dowel": 4, "foam": 5, "peach": 6, "potato": 7, "rope": 8, "tire": 9,
}

EYECANDIES_CLASSES = {
    "CandyCane": 0, "ChocolateCookie": 1, "ChocolatePraline": 2,
    "Confetto": 3, "GummyBear": 4, "HazelnutTruffle": 5,
    "LicoriceSandwich": 6, "Lollipop": 7, "Marshmallow": 8, "PeppermintCandy": 9,
}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Parse arguments ──────────────────────────────────────────────────────
    parser = get_parser()
    cli = parser.parse_args()
    args = load_config(cli.config, cli.class_name)

    # CLI overrides (explicit args override config)
    for key in vars(cli):
        if key in ("config", "class_name"):
            continue
        val = getattr(cli, key)
        if val is not None:
            setattr(args, key, val)

    seed_everything(args.seed)

    # ── Dataset-specific setup ───────────────────────────────────────────────
    is_mvtec = (args.dataset_type == "mvtec")
    lora_mode = "full" if is_mvtec else "lora"

    if is_mvtec:
        from utils.ad_dataset_1st import TrainDataset, ValidationDataset, mvtec3d_classes, eyecandies_classes
        class_dict = MVTEC_CLASSES
    else:
        from utils.ad_dataset_1st import TrainDataset, ValidationDataset, mvtec3d_classes, eyecandies_classes
        class_dict = EYECANDIES_CLASSES

    # ── Logging ──────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    log_name = f"{args.class_name}_train_adapt_{args.dataset_type}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.output_dir, log_name)),
        ])

    # ── Device ───────────────────────────────────────────────────────────────
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ── Dataset paths ────────────────────────────────────────────────────────
    dataroute = os.path.join(args.dataset_path, args.dataset_type)
    if not os.path.exists(dataroute):
        raise FileNotFoundError(f"Dataset not found: {dataroute}")

    classes_list = mvtec3d_classes() if is_mvtec else eyecandies_classes()
    cls_idx = class_dict[args.class_name]

    trainset = TrainDataset(classes_list[cls_idx], args.img_size,
                            dataroute, k_shot=args.k_shot)
    validset = ValidationDataset(classes_list[cls_idx], args.img_size, dataroute)

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers,
                             worker_init_fn=worker_init_fn, drop_last=False,
                             pin_memory=True)
    validloader = DataLoader(validset, batch_size=1, shuffle=False,
                             num_workers=args.num_workers,
                             worker_init_fn=worker_init_fn, drop_last=False,
                             pin_memory=True)

    # ── Build models ─────────────────────────────────────────────────────────
    rgb_context = RGBBackbone(device=device,
                              rgb_checkpoint_path=args.visual_encoder_ckpt_pth)
    for p in rgb_context.parameters():
        p.requires_grad = False

    xyz_context = XYZBackbone(device=device,
                              xyz_checkpoint_path=args.point_encoder_ckpt_pth,
                              group_size=args.group_size,
                              num_group=args.num_group)
    for p in xyz_context.parameters():
        p.requires_grad = False

    # Unfreeze encoder + positional embedding in 3D backbone
    for n, p in xyz_context.named_parameters():
        part = n.split(".")[1] if len(n.split(".")) > 1 else ""
        if part in ("encoder", "pos_embed"):
            p.requires_grad = True

    # Apply LoRA (or full fine-tuning) to selected blocks
    rgb_context.rgb_backbone = replace_learnable(
        rgb_context.rgb_backbone,
        target_block_idxs=[3, 7, 11],
        r=2, alpha=8, dropout=0.1,
        init_lora_weights=True, modal="rgb", mode=lora_mode)

    xyz_context.xyz_backbone.blocks = replace_learnable(
        xyz_context.xyz_backbone.blocks,
        target_block_idxs=[3, 7, 11],
        r=2, alpha=8, dropout=0.1,
        init_lora_weights=True, modal="xyz", mode=lora_mode)

    rgb_target = copy.deepcopy(rgb_context)
    xyz_target = copy.deepcopy(xyz_context)

    for p in rgb_target.parameters():
        p.requires_grad = False
    for p in xyz_target.parameters():
        p.requires_grad = False

    rgb_context  = rgb_context.to(device)
    xyz_context  = xyz_context.to(device)
    rgb_target   = rgb_target.to(device)
    xyz_target   = xyz_target.to(device)

    n_patches = (args.img_size // args.patch_size) ** 2
    rgb_predictor = init_predictor(device, num_patches=n_patches,
                                   embed_dim=768, pred_depth=args.predictor_depth,
                                   num_heads=6, drop_rate=0.1, pred_emb_dim=384)

    xyz_predictor = TransformerPredictor(
        embed_dim=384, predictor_embed_dim=192,
        depth=args.predictor_depth, num_heads=4, mlp_ratio=4.0,
        qkv_bias=True, qk_scale=None,
        drop_rate=0.1, attn_drop_rate=0.05, drop_path_rate=0.10,
        add_pos_at_every_layer=True, add_target_pos=True).to(device)

    aligner = Alignment(rgb_dim=768 * 3, xyz_dim=384 * 3, proj_dim=768).to(device)

    # ── Optimizers ───────────────────────────────────────────────────────────
    ipe = len(trainloader)
    ipe_scale = float(getattr(args, "ipe_scale", 1.0))

    optimizer_rgb, scaler_rgb, sched_rgb, wdsched_rgb = init_optimizer(
        encoder=rgb_context, predictor_list=[rgb_predictor],
        iterations_per_epoch=ipe,
        start_lr=args.start_lr_rgb, ref_lr=args.lr_rgb,
        final_lr=args.final_lr_rgb,
        warmup=args.warmup_epochs, num_epochs=args.num_epochs,
        wd=args.weight_decay, final_wd=args.final_weight_decay,
        use_bfloat16=args.usage_half_precision, ipe_scale=ipe_scale)

    optimizer_xyz, scaler_xyz, sched_xyz, wdsched_xyz = init_optimizer(
        encoder=xyz_context, predictor_list=[xyz_predictor],
        iterations_per_epoch=ipe,
        start_lr=args.start_lr_xyz, ref_lr=args.lr_xyz,
        final_lr=args.final_lr_xyz,
        warmup=args.warmup_epochs, num_epochs=args.num_epochs,
        wd=args.weight_decay, final_wd=args.final_weight_decay,
        use_bfloat16=args.usage_half_precision, ipe_scale=ipe_scale)

    # ── EMA schedulers ───────────────────────────────────────────────────────
    ema_rgb = args.ema_params_rgb
    ema_3d  = args.ema_params_3d
    total_steps = int(ipe * args.num_epochs * ipe_scale) + 1
    momentum_scheduler_rgb = (
        ema_rgb[0] + i * (ema_rgb[1] - ema_rgb[0]) / (ipe * args.num_epochs * ipe_scale)
        for i in range(total_steps))
    momentum_scheduler_3d = (
        ema_3d[0] + i * (ema_3d[1] - ema_3d[0]) / (ipe * args.num_epochs * ipe_scale)
        for i in range(total_steps))

    cos_sim = nn.CosineEmbeddingLoss()
    vic_rate = 0.01

    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
    IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])
    pred_mask_scale = args.pred_mask_scale

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(args.num_epochs):
        print(f"@@@ Class: {args.class_name} | Epoch {epoch+1}/{args.num_epochs} @@@")

        rgb_context.train(); rgb_predictor.train(); rgb_target.train()
        xyz_context.train(); xyz_predictor.train(); xyz_target.train()

        for iteration, data in tqdm(enumerate(trainloader)):
            rgb   = data["data"][0].to(device)   # [B, 3, H, W]
            xyz   = data["data"][1]               # [B, 3, H, W]
            B, C, H, W = xyz.shape
            xyz   = xyz.view(B, C, H * W).to(device)

            # ── 3D branch ────────────────────────────────────────────────────
            (xyz_feat_ctx, center_ctx, ori_idx_ctx, center_idx_ctx,
             tgt_blocks_ctx, target_indices,
             context_tokens_ctx, context_centers_ctx,
             original_center, context_indices_xyz) = xyz_context(
                xyz, npred=args.npred, mode="context")

            xyz_feat_tgt, _, _, _ = xyz_target(xyz, mode="target")

            # Multi-level split [3 levels]
            def split_levels(feat):
                return torch.cat([
                    feat[:, :384, :].unsqueeze(0),
                    feat[:, 384:768, :].unsqueeze(0),
                    feat[:, 768:, :].unsqueeze(0)], dim=0).permute(1, 0, 3, 2)

            xyz_feat_ctx = split_levels(xyz_feat_ctx)  # [B, 3, n, d]
            xyz_feat_tgt = split_levels(xyz_feat_tgt)

            # XYZ VICReg
            b, n, p, d = xyz_feat_tgt.shape
            xyz_tgt_invar = xyz_feat_tgt.permute(0, 2, 1, 3).reshape(b, p, n*d)
            xyz_tgt_invar = xyz_tgt_invar[:, context_indices_xyz, :]
            b, n, p, d = xyz_feat_ctx.shape
            xyz_ctx_invar = xyz_feat_ctx.permute(0, 2, 1, 3).reshape(b, p, n*d)
            vic_reg_xyz = vic_regularizer(xyz_ctx_invar, xyz_tgt_invar, device, B)

            # XYZ JEPA loss
            bsz, n_l, p, d = xyz_feat_tgt.shape
            loss_xyz_jepa = 0.0
            t_l = target_indices.shape[0]
            for i in range(bsz):
                ctx_b = xyz_feat_ctx[i]  # [n_l, n, d]
                ctx_b = torch.cat([ctx_b[k].unsqueeze(0) for k in range(n_l)], dim=2)  # [1, n, d*n_l]
                tgt_b = xyz_feat_tgt[i]
                for m in range(t_l):
                    tgt_idx = target_indices[m]
                    tgt_c = original_center[i, tgt_idx, :]
                    xyz_pred = xyz_predictor(ctx_b, context_centers_ctx[i].unsqueeze(0),
                                             tgt_c.unsqueeze(0))
                    xyz_tgt_block = tgt_b[:, tgt_idx, :]
                    loss_xyz_jepa += loss_fn(xyz_pred, xyz_tgt_block.detach())
            loss_xyz_jepa /= (bsz * t_l)

            # ── RGB branch ───────────────────────────────────────────────────
            foreground_idxs, background_idxs = [], []
            for img in rgb:
                img_check = (img.clone().detach().cpu() *
                             IMAGENET_STD.view(3, 1, 1) +
                             IMAGENET_MEAN.view(3, 1, 1)) * 255.0
                img_check = img_check.to(torch.int)
                patch = image_to_patches(img_check, args.patch_size)
                f_idxs = find_significant_patches(
                    patch, args.bg_threshold, args.bg_ratio_threshold)
                all_idxs = set(range((args.img_size // args.patch_size) ** 2))
                background_idxs.append(list(all_idxs - set(f_idxs)))
                foreground_idxs.append(f_idxs)

            if is_mvtec:
                rgb_ctx_blocks, rgb_tgt_blocks, _ = make_tgt_block_mvtec(
                    args, args.batch_size, args.patch_size,
                    foreground_idxs, background_idxs, pred_mask_scale)
            else:
                rgb_ctx_blocks, rgb_tgt_blocks, _ = make_tgt_block_eyecandies(
                    args, args.batch_size, args.patch_size,
                    foreground_idxs, pred_mask_scale)

            z_rgb = rgb_context(rgb, context_mask=rgb_ctx_blocks)
            z_rgb_invar = z_rgb.clone()
            z_rgb = rgb_predictor(z_rgb, rgb_ctx_blocks, rgb_tgt_blocks, device=device)

            h_rgb = rgb_target(rgb, context_mask=None).detach()
            _, _, dim = h_rgb.shape
            h_rgb = torch.cat([
                h_rgb[:, :, :dim//3].unsqueeze(0),
                h_rgb[:, :, dim//3:(dim//3)*2].unsqueeze(0),
                h_rgb[:, :, (dim//3)*2:].unsqueeze(0)], dim=0)

            target_rgb, invar_rgb = [], []
            for i in range(len(h_rgb)):
                irgb = h_rgb[i]
                target_rgb.append(apply_masks(irgb, rgb_tgt_blocks, device=device).unsqueeze(0))
                invar_rgb.append(apply_masks(irgb, rgb_ctx_blocks, device=device).unsqueeze(0))
            target_rgb = torch.cat(target_rgb, dim=0)
            invar_rgb  = torch.cat(invar_rgb, dim=0)

            nl, bsz_r, ntgt, d = invar_rgb.shape
            invar_rgb = invar_rgb.permute(1, 2, 0, 3).reshape(bsz_r, ntgt, nl * d)
            vic_reg_rgb = vic_regularizer(z_rgb_invar, invar_rgb, device, bsz_r)

            z_rgb      = z_rgb.permute(1, 0, 2, 3)
            target_rgb = target_rgb.permute(1, 0, 2, 3)
            loss_rgb_jepa = 0.0
            for i in range(len(target_rgb)):
                loss_rgb_jepa += loss_fn(z_rgb[i], target_rgb[i].detach())
            loss_rgb_jepa /= len(target_rgb)

            step_loss_rgb = loss_rgb_jepa + vic_rate * vic_reg_rgb
            step_loss_xyz = loss_xyz_jepa + vic_rate * vic_reg_xyz

            logging.info(
                f"[{args.class_name}] epoch {epoch+1} iter {iteration} | "
                f"xyz_jepa={loss_xyz_jepa:.4f} rgb_jepa={loss_rgb_jepa:.4f} "
                f"vicreg_xyz={vic_reg_xyz:.4f} vicreg_rgb={vic_reg_rgb:.4f}")

            # ── Optimise RGB ──────────────────────────────────────────────────
            optimizer_rgb.zero_grad()
            step_loss_rgb.backward()
            optimizer_rgb.step()
            sched_rgb.step(); wdsched_rgb.step()

            # ── Optimise XYZ ──────────────────────────────────────────────────
            optimizer_xyz.zero_grad()
            step_loss_xyz.backward(retain_graph=True)
            optimizer_xyz.step()
            sched_xyz.step(); wdsched_xyz.step()

            # ── EMA update ───────────────────────────────────────────────────
            with torch.no_grad():
                m = next(momentum_scheduler_rgb)
                for q, k in zip(rgb_context.parameters(), rgb_target.parameters()):
                    k.data.mul_(m).add_((1 - m) * q.detach().data)
                m = next(momentum_scheduler_3d)
                for q, k in zip(xyz_context.parameters(), xyz_target.parameters()):
                    k.data.mul_(m).add_((1 - m) * q.detach().data)

        # ── Save checkpoint at last epoch ────────────────────────────────────
        if (epoch + 1) == args.num_epochs:
            ckpt_dir = os.path.join(args.output_dir,
                                    getattr(args, "ckpt_subdir", "1st_stage"))
            os.makedirs(ckpt_dir, exist_ok=True)
            if is_mvtec:
                rgb_name = f"{args.class_name}_best_jepa_rgb_dino_main_{args.npred}blocks_full_0.01_rpr.pth"
                xyz_name = f"{args.class_name}_best_jepa_xyz_pmae_main_{args.npred}blocks_full_0.01_rpr.pth"
            else:
                rgb_name = f"{args.class_name}_best_jepa_rgb_dino_new_rpr.pth"
                xyz_name = f"{args.class_name}_best_jepa_xyz_pmae_new_rpr.pth"
            torch.save(rgb_context.state_dict(), os.path.join(ckpt_dir, rgb_name))
            torch.save(xyz_context.state_dict(), os.path.join(ckpt_dir, xyz_name))
            print(f"Saved: {ckpt_dir}/{rgb_name}")
            print(f"Saved: {ckpt_dir}/{xyz_name}")


if __name__ == "__main__":
    main()
