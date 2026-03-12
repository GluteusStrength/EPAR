"""
train_nsn.py — Stage 2: NSN (Noise-Sensitive Normalizing) / Memory Bank Training

Supports both MVTec-3D AD and Eyecandies datasets through a single entry-point.
Class-specific hyper-parameters are stored in YAML configs under ./configs/.

Usage:
    # MVTec-3D
    python train_nsn.py --config configs/train_nsn_mvtec.yaml --class_name cookie

    # Eyecandies
    python train_nsn.py --config configs/train_nsn_eyecandies.yaml --class_name CandyCane
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
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# ── Point cloud ops ──────────────────────────────────────────────────────────
from pointnet2_ops import pointnet2_utils
from knn_cuda import KNN
from chamferdist import ChamferDistance

# ── Model imports ────────────────────────────────────────────────────────────
from models.feature_extractor import RGBBackbone, XYZBackbone
from models.point_feature_alignment_v2 import PointFeatureAlignment
from models import encoder_lora as vit
from models.point_transformer_v2 import TransformerPredictor
from models.encoder import VIT_EMBED_DIMS
from models.src.tensors import trunc_normal_, repeat_interleave_batch
from models.src.schedulers import WarmupCosineSchedule, CosineWDSchedule
from models.src.blocks_new_v2 import MaskCollator         # used only in mvtec combined mode
from models.src.apply_masks import apply_masks
from models.common import (
    FaissNNRGB, FaissNN3D,
    NearestNeighbourScorerRGB, NearestNeighbourScorer3D,
    RescaleSegmentor,
)
from models.memory_bank import MMUpdateRGB, MMUpdate3D, FocalLoss

import loralib as lora

# ── Utils imports ────────────────────────────────────────────────────────────
from utils.sampler import ApproximateGreedyCoresetSampler
from utils.metrics import calculate_au_pro

import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str, class_name: str) -> SimpleNamespace:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    classes = cfg.pop("classes", {})
    args = SimpleNamespace(**cfg)
    if class_name in classes and classes[class_name]:
        for key, val in classes[class_name].items():
            setattr(args, key, val)
    args.class_name = class_name
    return args


def get_parser():
    parser = argparse.ArgumentParser(description="Stage-2 NSN / Memory Bank Trainer")
    parser.add_argument("--config",     type=str, required=True)
    parser.add_argument("--class_name", type=str, required=True)
    parser.add_argument("--dataset_path",            type=str)
    parser.add_argument("--output_dir",              type=str)
    parser.add_argument("--visual_encoder_ckpt_pth", type=str)
    parser.add_argument("--point_encoder_ckpt_pth",  type=str)
    parser.add_argument("--num_epochs",              type=int)
    parser.add_argument("--batch_size",              type=int)
    parser.add_argument("--lr_xyz",                  type=float)
    parser.add_argument("--start_lr_xyz",            type=float)
    parser.add_argument("--final_lr_xyz",            type=float)
    parser.add_argument("--bg_threshold",            type=float)
    parser.add_argument("--npred",                   type=int)
    parser.add_argument("--k_shot",                  type=int)
    parser.add_argument("--mode",                    type=str,
                        choices=["combined", "only_jepa", "only_memory"])
    parser.add_argument("--epoch_num",               type=int)
    return parser


# ─────────────────────────────────────────────────────────────────────────────
# Seed
# ─────────────────────────────────────────────────────────────────────────────

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def worker_init_fn(worker_id, seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


# ─────────────────────────────────────────────────────────────────────────────
# LoRA replacement  (same as train_adapt)
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
            for p in backbone_block.parameters():
                p.requires_grad = True
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Model init helpers
# ─────────────────────────────────────────────────────────────────────────────

def init_predictor(device, num_patches, embed_dim, pred_depth, num_heads,
                   drop_rate=0.0, pred_emb_dim=384):
    predictor = vit.__dict__["vit_predictor"](
        num_patches=num_patches, embed_dim=embed_dim,
        predictor_embed_dim=pred_emb_dim,
        depth=pred_depth, num_heads=num_heads)
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
        {"params": w_decay,  "weight_decay": wd},
        {"params": wo_decay, "weight_decay": 0.0},
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
# Patch utilities
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
# Target-block builder (MVTec 2nd stage — no background_indices arg)
# ─────────────────────────────────────────────────────────────────────────────

def make_tgt_block(args, bsz, patch_size, foreground_indices, pred_mask_scale):
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

    row_range  = [min_row, max_row]
    col_range  = [min_col, max_col]
    context_indices = torch.arange((args.img_size // patch_size) ** 2)

    mask_collator = MaskCollator(
        input_size=(args.img_size, args.img_size),
        patch_size=patch_size,
        pred_mask_scale=pred_mask_scale,
        aspect_ratio=args.aspect_ratio,
        npred=args.npred,
        min_keep=args.min_keep,
        allow_overlap=False,
        mode="train")

    context_blocks, target_blocks = mask_collator(
        bsz=bsz,
        row_range=row_range, col_range=col_range,
        target_acceptable_regions=target_block_complements,
        context_indices=context_indices)

    return (context_blocks, target_blocks,
            target_block_complements.reshape(bsz, (args.img_size // patch_size) ** 2))


# ─────────────────────────────────────────────────────────────────────────────
# Loss helpers
# ─────────────────────────────────────────────────────────────────────────────

def smooth_l1(z, h):
    return F.smooth_l1_loss(z, h)


def build_coreset_sampler(embeddings, sampler):
    return sampler.run(embeddings)


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
# Visualisation helper
# ─────────────────────────────────────────────────────────────────────────────

def vis_quality_result(rgb, anomaly_map, gt_mask, i, classname,
                       overlay_alpha=0.5, gt_alpha=0.5, name="rgb"):
    rgb = rgb.squeeze(0).detach().cpu()
    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    rgb = (rgb * STD + MEAN).permute(1, 2, 0).numpy()
    anomaly_map = (anomaly_map - anomaly_map.min()) / (
        anomaly_map.max() - anomaly_map.min() + 1e-6)
    cmap = plt.get_cmap("jet")
    heatmap = cmap(anomaly_map)[..., :3]
    overlay = np.clip((1 - overlay_alpha) * rgb + overlay_alpha * heatmap, 0, 1)
    gt_overlay = rgb.copy()
    gt_overlay[gt_mask == 1] = (1 - gt_alpha) * gt_overlay[gt_mask == 1] + \
                                gt_alpha * np.array([1, 0, 0])
    out_dir = os.path.join("quality", classname)
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(28, 7))
    axes[0].imshow(rgb);                    axes[0].set_title("RGB");              axes[0].axis("off")
    axes[1].imshow(anomaly_map, cmap="jet");axes[1].set_title(f"{name} Heatmap");  axes[1].axis("off")
    axes[2].imshow(overlay);                axes[2].set_title(f"Overlay ({name})");axes[2].axis("off")
    axes[3].imshow(gt_overlay);             axes[3].set_title("GT Mask");          axes[3].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{i}th_{name}_image.png"))
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = get_parser()
    cli = parser.parse_args()
    args = load_config(cli.config, cli.class_name)

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
        from utils.ad_dataset_2nd import (
            TrainDataset, TestDataset, ValidationDataset,
            mvtec3d_classes, eyecandies_classes,
        )
        class_dict = MVTEC_CLASSES
    else:
        from utils.ad_dataset_eyecandies import (
            TrainDataset, TestDataset,
            mvtec3d_classes, eyecandies_classes,
        )
        class_dict = EYECANDIES_CLASSES

    # ── Logging ──────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    mode_str = getattr(args, "mode", "combined")
    log_name = f"{args.class_name}_train_nsn_{args.dataset_type}_{mode_str}.log"
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
    classes_list = mvtec3d_classes() if is_mvtec else eyecandies_classes()
    cls_idx = class_dict[args.class_name]

    trainset = TrainDataset(classes_list[cls_idx], args.img_size,
                            dataroute, k_shot=args.k_shot)
    testset  = TestDataset(classes_list[cls_idx],  args.img_size, dataroute)

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers,
                             worker_init_fn=worker_init_fn, drop_last=False,
                             pin_memory=True)
    testloader  = DataLoader(testset, batch_size=1, shuffle=False,
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
    for n, p in xyz_context.named_parameters():
        part = n.split(".")[1] if len(n.split(".")) > 1 else ""
        if part in ("encoder", "pos_embed"):
            p.requires_grad = True

    rgb_context.rgb_backbone = replace_learnable(
        rgb_context.rgb_backbone, target_block_idxs=[3, 7, 11],
        r=2, alpha=8, dropout=0.1, init_lora_weights=True, modal="rgb",
        mode=lora_mode)
    xyz_context.xyz_backbone.blocks = replace_learnable(
        xyz_context.xyz_backbone.blocks, target_block_idxs=[3, 7, 11],
        r=2, alpha=8, dropout=0.1, init_lora_weights=True, modal="xyz",
        mode=lora_mode)

    rgb_target = copy.deepcopy(rgb_context)
    xyz_target = copy.deepcopy(xyz_context)
    for p in rgb_target.parameters():
        p.requires_grad = False
    for p in xyz_target.parameters():
        p.requires_grad = False

    # ── Load Stage-1 weights ─────────────────────────────────────────────────
    stage1_dir = os.path.join(args.output_dir,
                              getattr(args, "stage1_ckpt_subdir", "1st_stage"))
    if is_mvtec:
        rgb_ckpt_name = f"{args.class_name}_best_jepa_rgb_dino_main_{args.npred}blocks_full_0.01_rpr.pth"
        xyz_ckpt_name = f"{args.class_name}_best_jepa_xyz_pmae_main_{args.npred}blocks_full_0.01_rpr.pth"
    else:
        rgb_ckpt_name = f"{args.class_name}_best_jepa_rgb_dino_new.pth"
        xyz_ckpt_name = f"{args.class_name}_best_jepa_xyz_pmae_new.pth"

    rgb_params = torch.load(os.path.join(stage1_dir, rgb_ckpt_name), map_location="cpu")
    xyz_params = torch.load(os.path.join(stage1_dir, xyz_ckpt_name), map_location="cpu")
    rgb_context.load_state_dict(rgb_params)
    xyz_context.load_state_dict(xyz_params)
    rgb_target = copy.deepcopy(rgb_context)
    xyz_target = copy.deepcopy(xyz_context)
    for p in rgb_target.parameters():
        p.requires_grad = False
    for p in xyz_target.parameters():
        p.requires_grad = False

    rgb_context = rgb_context.to(device)
    xyz_context = xyz_context.to(device)
    rgb_target  = rgb_target.to(device)
    xyz_target  = xyz_target.to(device)

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

    pfa = PointFeatureAlignment(patch_size=args.patch_size)

    # ── Memory bank & scorer ──────────────────────────────────────────────────
    anomaly_segmentor = RescaleSegmentor(
        device=torch.device("cpu"), target_size=args.img_size)
    nearest_rgb = FaissNNRGB(True, 4, mode="dist")
    nearest_xyz = FaissNN3D(True, 4, mode="dist")
    anomaly_scorer_rgb = NearestNeighbourScorerRGB(
        n_nearest_neighbours=5, nn_method_l2=nearest_rgb)
    anomaly_scorer_xyz = NearestNeighbourScorer3D(
        n_nearest_neighbours=5, nn_method_l2=nearest_xyz)

    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
    IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])
    pred_mask_scale = args.pred_mask_scale
    ipe = len(trainloader)
    ipe_scale = float(getattr(args, "ipe_scale", 1.0))

    # ── Optimizers ───────────────────────────────────────────────────────────
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

    ema_rgb = args.ema_params_rgb
    ema_3d  = args.ema_params_3d
    total_steps = int(ipe * args.num_epochs * ipe_scale) + 1
    momentum_scheduler_rgb = (
        ema_rgb[0] + i * (ema_rgb[1] - ema_rgb[0]) / (ipe * args.num_epochs * ipe_scale)
        for i in range(total_steps))
    momentum_scheduler_3d = (
        ema_3d[0] + i * (ema_3d[1] - ema_3d[0]) / (ipe * args.num_epochs * ipe_scale)
        for i in range(total_steps))

    # ── Phase 0: Pseudo-anomaly generation + feature extraction ──────────────
    rgb_original, xyz_original = [], []
    rgb_anomaly,  xyz_anomaly  = [], []
    anomaly_masks = []

    for iteration, data in tqdm(enumerate(trainloader), desc="Pseudo-anomaly generation ..."):
        rgb   = data["data"][0]
        xyz   = data["data"][1]
        anomaly_rgb = data["anomaly_data"][0] if "data" in data else data.get("anomaly_rgb")
        anomaly_3d  = data["anomaly_data"][1] if "data" in data else data.get("anomaly_3d")
        # anomaly_mask = data.get("anomaly_mask", data.get("mask"))
        anomaly_mask = data['anomaly_data'][5]

        anomaly_masks.append(anomaly_mask)
        rgb_original.append(rgb);      rgb_anomaly.append(anomaly_rgb)
        xyz_original.append(xyz);      xyz_anomaly.append(anomaly_3d)

    with torch.no_grad():
        rgb_context.eval(); xyz_context.eval()

        rgb_only_embs, rgb_anomaly_embs = [], []
        xyz_only_embs, xyz_anomaly_embs = [], []
        bg_indices_normal, bg_indices_abnormal = [], []

        rgb_original = torch.cat(rgb_original, dim=0)
        xyz_original = torch.cat(xyz_original, dim=0)
        rgb_anomaly  = torch.cat(rgb_anomaly,  dim=0)
        xyz_anomaly  = torch.cat(xyz_anomaly,  dim=0)

        for i in tqdm(range(rgb_original.shape[0]), desc="Feature extraction ..."):
            rgb_i = rgb_original[i].unsqueeze(0).to(device)
            xyz_i = xyz_original[i].unsqueeze(0).to(device)
            anom_rgb = rgb_anomaly[i].unsqueeze(0).to(device)
            anom_xyz = xyz_anomaly[i].unsqueeze(0).to(device)

            B, C, H, W = xyz_i.shape
            xyz_flat      = xyz_i.view(B, C, H * W)
            anom_xyz_flat = anom_xyz.view(B, C, H * W)

            # Valid (non-zero) point filtering
            xyz_t   = xyz_flat.permute(0, 2, 1)
            nz_mask = torch.all(xyz_t[0] != 0, dim=1)
            nz_idx  = torch.nonzero(nz_mask, as_tuple=True)[0]
            valid_pts = xyz_t[:, nz_idx, :]

            anom_xyz_t   = anom_xyz_flat.permute(0, 2, 1)
            nz_mask_anom = torch.all(anom_xyz_t[0] != 0, dim=1)
            nz_idx_anom  = torch.nonzero(nz_mask_anom, as_tuple=True)[0]
            valid_pts_anom = anom_xyz_t[:, nz_idx_anom, :]

            # RGB embeddings
            h_rgb_n = rgb_context(rgb_i,   bg_indices=None, context_mask=None)
            h_rgb_a = rgb_context(anom_rgb, bg_indices=None, context_mask=None)
            rgb_only_embs.append(h_rgb_n.detach().cpu())
            rgb_anomaly_embs.append(h_rgb_a.detach().cpu())

            # 3D embeddings + PFA
            xyz_feats, centers, _, _ = xyz_context(xyz_flat,      mode="target")
            xyz_feats_a, centers_a, _, _ = xyz_context(anom_xyz_flat, mode="target")

            def pfa_embed(feats, ctrs, pts, nz, ks):
                out = pfa(group_features=feats.permute(0,2,1),
                          group_centers=ctrs,
                          original_points=pts, nonzero_indices=nz,
                          kernel_size=ks)
                B2, D2, P2, _ = out.shape
                return out.permute(0,2,3,1).reshape(B2, P2*P2, D2)

            ks_half = args.patch_size // 2
            xyz_only_embs.append(
                pfa_embed(xyz_feats, centers, valid_pts, nz_idx, ks_half).detach().cpu())
            xyz_anomaly_embs.append(
                pfa_embed(xyz_feats_a, centers_a, valid_pts_anom, nz_idx_anom, ks_half).detach().cpu())

            # Background mask
            xyz_feat_full = pfa_embed(xyz_feats, centers, valid_pts, nz_idx, args.patch_size)
            bg_indices_normal.append((xyz_feat_full.sum(axis=-1) == 0).detach().cpu())
            xyz_feat_full_a = pfa_embed(xyz_feats_a, centers_a, valid_pts_anom, nz_idx_anom, args.patch_size)
            bg_indices_abnormal.append((xyz_feat_full_a.sum(axis=-1) == 0).detach().cpu())

        rgb_only_embs   = torch.cat(rgb_only_embs,   dim=0)
        xyz_only_embs   = torch.cat(xyz_only_embs,   dim=0)
        rgb_anomaly_embs = torch.cat(rgb_anomaly_embs, dim=0)
        xyz_anomaly_embs = torch.cat(xyz_anomaly_embs, dim=0)
        anomaly_masks   = torch.cat(anomaly_masks,   dim=0)
        bg_indices_normal   = torch.cat(bg_indices_normal,   dim=0)
        bg_indices_abnormal = torch.cat(bg_indices_abnormal, dim=0)

    # ── Coreset sampling ──────────────────────────────────────────────────────
    b, p_rgb, d_rgb = rgb_only_embs.shape
    sampler_rgb = ApproximateGreedyCoresetSampler(
        percentage=args.coreset_percentage_rgb, device=device)
    coreset_rgb, _ = build_coreset_sampler(
        rgb_only_embs.reshape(b * p_rgb, d_rgb), sampler_rgb)
    coreset_rgb = coreset_rgb.detach().cpu().numpy().astype("float32")

    b, p_xyz, d_xyz = xyz_only_embs.shape
    sampler_xyz = ApproximateGreedyCoresetSampler(
        percentage=args.coreset_percentage_xyz, device=device)
    coreset_xyz, _ = build_coreset_sampler(
        xyz_only_embs.reshape(b * p_xyz, d_xyz), sampler_xyz)
    coreset_xyz = coreset_xyz.detach().cpu().numpy().astype("float32")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # ── Phase 1: Memory bank training ────────────────────────────────────────
    if mode_str in ("combined", "only_memory"):
        rgb_bank = MMUpdateRGB(init_memory_bank=torch.from_numpy(coreset_rgb),
                               embed_dim=2304 * 2).to(device)
        xyz_bank = MMUpdate3D(init_memory_bank=torch.from_numpy(coreset_xyz),
                              embed_dim=1152 * 2).to(device)
        opt_rgb_bank = torch.optim.AdamW(rgb_bank.parameters(), lr=args.bank_lr)
        opt_xyz_bank = torch.optim.AdamW(xyz_bank.parameters(), lr=args.bank_lr)
        focal_loss   = FocalLoss()

        nsamples = xyz_only_embs.shape[0]
        for ep in tqdm(range(args.num_epochs), desc="Memory bank training ..."):
            l_rgb, l_xyz = 0.0, 0.0
            logging.info(f"[{args.class_name}] bank_lr={args.bank_lr}")
            for i in range(nsamples):
                rgb_ftr_ab = rgb_anomaly_embs[i].to(device)
                xyz_ftr_ab = xyz_anomaly_embs[i].to(device)
                bg_anom = bg_indices_abnormal[i]
                mask    = anomaly_masks[i].to(device)

                rgb_res1, _ = rgb_bank(rgb_ftr_ab)
                xyz_res1, _ = xyz_bank(xyz_ftr_ab)

                loss_rgb_b = (focal_loss(rgb_res1.reshape(1, -1), mask.reshape(1, -1)) +
                              F.smooth_l1_loss(rgb_res1.reshape(1, -1), mask.reshape(1, -1)))
                loss_xyz_b = (focal_loss(xyz_res1.reshape(1, -1), mask.reshape(1, -1)) +
                              F.smooth_l1_loss(xyz_res1.reshape(1, -1), mask.reshape(1, -1)))

                opt_rgb_bank.zero_grad(); loss_rgb_b.backward(); opt_rgb_bank.step()
                opt_xyz_bank.zero_grad(); loss_xyz_b.backward(); opt_xyz_bank.step()
                l_rgb += loss_rgb_b.item(); l_xyz += loss_xyz_b.item()

            print(f"[{args.class_name}] ep {ep} | rgb_loss={l_rgb/nsamples:.4f} "
                  f"xyz_loss={l_xyz/nsamples:.4f}")

        # Save memory bank
        stage2_dir = os.path.join(args.output_dir,
                                  getattr(args, "stage2_ckpt_subdir", "2nd_stage"))
        os.makedirs(stage2_dir, exist_ok=True)
        if is_mvtec:
            torch.save(rgb_bank.state_dict(),
                       os.path.join(stage2_dir,
                                    f"{args.class_name}_rgb_bank_main_{args.npred}blocks_full_0.01.pth"))
            torch.save(xyz_bank.state_dict(),
                       os.path.join(stage2_dir,
                                    f"{args.class_name}_xyz_bank_main_{args.npred}blocks_full_0.01.pth"))
        else:
            torch.save(rgb_bank.state_dict(),
                       os.path.join(stage2_dir, f"{args.class_name}_rgb_bank_eyecandies.pth"))
            torch.save(xyz_bank.state_dict(),
                       os.path.join(stage2_dir, f"{args.class_name}_xyz_bank_eyecandies.pth"))
    else:
        rgb_bank = xyz_bank = None

    # ── Evaluation ───────────────────────────────────────────────────────────
    anomaly_scorer_rgb.fit(detection_features=[coreset_rgb])
    anomaly_scorer_xyz.fit(detection_features=[coreset_xyz])

    anomaly_scores_rgb, anomaly_scores_xyz = [], []
    anomaly_map_rgb_all, anomaly_map_xyz_all = [], []
    anomaly_scores_mm, anomaly_map_mm_all, anomaly_labels, anomaly_gt_maps = [], [], [], []

    @torch.no_grad()
    def eval_anomaly(rgb_ftrs, xyz_ftrs):
        bsz, npr, _ = rgb_ftrs.shape
        rgb_ftrs = rgb_ftrs.reshape(bsz * npr, -1).detach().cpu()
        bsz, npx, _ = xyz_ftrs.shape
        xyz_ftrs = xyz_ftrs.reshape(bsz * npx, -1).detach().cpu()

        _, dscore_rgb, _, _ = anomaly_scorer_rgb.predict(
            [rgb_ftrs.numpy()], max_val=None, foreground_mask=None)
        _, dscore_xyz, _, _ = anomaly_scorer_xyz.predict(
            [xyz_ftrs.numpy()], max_val=None, foreground_mask=None)

        amap_rgb = anomaly_segmentor.convert_to_segmentation(
            torch.tensor(dscore_rgb).unsqueeze(0).reshape(
                -1, args.img_size // args.patch_size,
                    args.img_size // args.patch_size))[0]
        amap_rgb = ndimage.gaussian_filter(amap_rgb, sigma=4)

        amap_xyz = anomaly_segmentor.convert_to_segmentation(
            torch.tensor(dscore_xyz).unsqueeze(0).reshape(
                -1, args.img_size // (args.patch_size // 2),
                    args.img_size // (args.patch_size // 2)))[0]
        amap_xyz = ndimage.gaussian_filter(amap_xyz, sigma=4)

        if rgb_bank is not None and xyz_bank is not None:
            conf_rgb = torch.exp(rgb_bank(rgb_ftrs.to(device))[0].reshape(-1)).detach().cpu()
            conf_xyz = torch.exp(xyz_bank(xyz_ftrs.to(device))[0].reshape(-1)).detach().cpu()
            conf_rgb = conf_rgb.numpy().reshape(args.img_size, args.img_size)
            conf_xyz = conf_xyz.numpy().reshape(args.img_size, args.img_size)
            amap_rgb = amap_rgb * conf_rgb
            amap_xyz = amap_xyz * conf_xyz

        score_rgb = np.max(amap_rgb)
        score_xyz = np.max(amap_xyz)

        # normalise and fuse
        def znorm(x):
            return (x - x.mean()) / (x.std() + 1e-8)
        amap_fused = np.maximum(znorm(amap_rgb.reshape(-1)), znorm(amap_xyz.reshape(-1))).reshape(args.img_size, args.img_size)
        score_mm = np.max(amap_fused) * score_rgb * score_xyz

        return score_rgb, score_xyz, amap_rgb, amap_xyz, amap_fused, score_mm

    for iteration, data in tqdm(enumerate(testloader), desc="Inference ..."):
        rgb_context.eval(); xyz_context.eval()
        if rgb_bank is not None:
            rgb_bank.eval(); xyz_bank.eval()

        rgb   = data["data"][0].to(device)
        xyz   = data["data"][1]
        B, C, H, W = xyz.shape
        xyz_flat = xyz.view(B, C, H * W).to(device)

        xyz_t   = xyz_flat.permute(0, 2, 1)
        nz_mask = torch.all(xyz_t[0] != 0, dim=1)
        nz_idx  = torch.nonzero(nz_mask, as_tuple=True)[0]
        valid_pts = xyz_t[:, nz_idx, :]

        xyz_feats, centers, _, _ = xyz_context(xyz_flat, mode="target")

        xyz_2d = pfa(group_features=xyz_feats.permute(0, 2, 1),
                     group_centers=centers,
                     original_points=valid_pts,
                     nonzero_indices=nz_idx,
                     kernel_size=args.patch_size // 2)
        B2, D2, P2, _ = xyz_2d.shape
        xyz_features = xyz_2d.permute(0, 2, 3, 1).reshape(B2, P2 * P2, D2)

        h_rgb = rgb_context(rgb, context_mask=None)

        s_rgb, s_xyz, amap_rgb, amap_xyz, amap_fused, s_mm = eval_anomaly(h_rgb, xyz_features)

        anomaly_scores_rgb.append(s_rgb)
        anomaly_scores_xyz.append(s_xyz)
        anomaly_scores_mm.append(s_mm)
        anomaly_map_rgb_all.append(amap_rgb)
        anomaly_map_xyz_all.append(amap_xyz)
        anomaly_map_mm_all.append(amap_fused)
        anomaly_labels.append(data["label"].item())
        anomaly_gt_maps.append(data["anomaly_map"].numpy().squeeze())

    # ── Metrics ───────────────────────────────────────────────────────────────
    labels = np.array(anomaly_labels)
    auroc_rgb = roc_auc_score(labels, anomaly_scores_rgb) * 100
    auroc_xyz = roc_auc_score(labels, anomaly_scores_xyz) * 100
    auroc_mm  = roc_auc_score(labels, anomaly_scores_mm)  * 100

    gt_flat = np.concatenate([m.flatten() for m in anomaly_gt_maps])
    pred_rgb_flat = np.concatenate([m.flatten() for m in anomaly_map_rgb_all])
    pred_xyz_flat = np.concatenate([m.flatten() for m in anomaly_map_xyz_all])
    pred_mm_flat  = np.concatenate([m.flatten() for m in anomaly_map_mm_all])
    
    # P-AUROC (Pixel-level AUROC)
    pauroc_rgb = roc_auc_score(gt_flat, pred_rgb_flat) * 100
    pauroc_xyz = roc_auc_score(gt_flat, pred_xyz_flat) * 100
    pauroc_mm  = roc_auc_score(gt_flat, pred_mm_flat) * 100
    
    # AUPRO@30%
    aupro30_rgb = calculate_au_pro(anomaly_gt_maps, anomaly_map_rgb_all, integration_limit=0.3)[0][0] * 100
    aupro30_xyz = calculate_au_pro(anomaly_gt_maps, anomaly_map_xyz_all, integration_limit=0.3)[0][0] * 100
    aupro30_mm  = calculate_au_pro(anomaly_gt_maps, anomaly_map_mm_all, integration_limit=0.3)[0][0] * 100
    
    # AUPRO@1%
    aupro1_rgb = calculate_au_pro(anomaly_gt_maps, anomaly_map_rgb_all, integration_limit=0.01)[0][0] * 100
    aupro1_xyz = calculate_au_pro(anomaly_gt_maps, anomaly_map_xyz_all, integration_limit=0.01)[0][0] * 100
    aupro1_mm  = calculate_au_pro(anomaly_gt_maps, anomaly_map_mm_all, integration_limit=0.01)[0][0] * 100

    result_str = (
        "=================================================================================================\n"
        f"  Class: {args.class_name}  |  Dataset: {args.dataset_type}\n"
        "=================================================================================================\n"
        f"  I-AUROC   -> RGB: {auroc_rgb:.1f}%  |  3D: {auroc_xyz:.1f}%  |  MM: {auroc_mm:.1f}%\n"
        f"  P-AUROC   -> RGB: {pauroc_rgb:.1f}%  |  3D: {pauroc_xyz:.1f}%  |  MM: {pauroc_mm:.1f}%\n"
        f"  AUPRO@30% -> RGB: {aupro30_rgb:.1f}%  |  3D: {aupro30_xyz:.1f}%  |  MM: {aupro30_mm:.1f}%\n"
        f"  AUPRO@1%  -> RGB: {aupro1_rgb:.1f}%  |  3D: {aupro1_xyz:.1f}%  |  MM: {aupro1_mm:.1f}%\n"
        "================================================================================================="
    )
    print(result_str)
    logging.info(result_str)


if __name__ == "__main__":
    main()
