# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
This is for the visual encoder of the AD-JEPA model.
The visual encoder is basically Vision Transformer Model
    - Follow the setting of I-JEPA
    - Select the size of model: Small, base
"""

import math
from functools import partial
import numpy as np

import torch
import torch.nn as nn

from models.src.tensors import (
    trunc_normal_,
    repeat_interleave_batch
)
from models.src.apply_masks import apply_masks


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid length
    return:
    pos_embed: [grid_size, embed_dim] or [1+grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid = np.arange(grid_size, dtype=float)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega   # (D/2,)

    pos = pos.reshape(-1)   # (M,)
    out = np.einsum('m,d->md', pos, omega)   # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class ConvEmbed(nn.Module):
    """
    3x3 Convolution stems for ViT following ViTC models
    """

    def __init__(self, channels, strides, img_size=224, in_chans=3, batch_norm=True):
        super().__init__()
        # Build the stems
        stem = []
        channels = [in_chans] + channels
        for i in range(len(channels) - 2):
            stem += [nn.Conv2d(channels[i], channels[i+1], kernel_size=3,
                               stride=strides[i], padding=1, bias=(not batch_norm))]
            if batch_norm:
                stem += [nn.BatchNorm2d(channels[i+1])]
            stem += [nn.ReLU(inplace=True)]
        stem += [nn.Conv2d(channels[-2], channels[-1], kernel_size=1, stride=strides[-1])]
        self.stem = nn.Sequential(*stem)

        # Comptute the number of patches
        stride_prod = int(np.prod(strides))
        self.num_patches = (img_size[0] // stride_prod)**2

    def forward(self, x):
        p = self.stem(x)
        return p.flatten(2).transpose(1, 2)

class VisionTransformerPredictor(nn.Module):
    """ Vision Transformer """
    def __init__(
        self,
        num_patches,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        lower_projection=False,
        **kwargs
    ):
        super().__init__()
        self.lower_dim = lower_projection
        if lower_projection:
            self.proj = nn.ModuleList([
                nn.Linear(embed_dim*2, embed_dim) for _ in range(3)
            ])
        self.predictor_embed_dim = predictor_embed_dim
        self.predictor_embed = nn.ModuleList([
            nn.Linear(embed_dim, predictor_embed_dim)
            for _ in range(3)
        ])
        self.mask_token_low = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim)) # Learnable Parameter
        self.mask_token_mid = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim)) # Learnable Parameter
        self.mask_token_high = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim)) # Learnable Parameter
        # self.mask_token_last = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim)) # Learnable Parameter
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # --
        self.predictor_pos_embed = nn.Parameter(torch.zeros(1, num_patches, predictor_embed_dim),
                                                requires_grad=False)
        predictor_pos_embed = get_2d_sincos_pos_embed(self.predictor_pos_embed.shape[-1],
                                                      int(num_patches**.5),
                                                      cls_token=False)
        self.predictor_pos_embed.data.copy_(torch.from_numpy(predictor_pos_embed).float().unsqueeze(0))
        # --
        self.predictor_blocks = nn.ModuleDict({
            "low_pred": nn.ModuleList([Block(dim=predictor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                       drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                                       for i in range(depth)]),
            "mid_pred": nn.ModuleList([Block(dim=predictor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                       drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                                       for i in range(depth)]),
            "high_pred": nn.ModuleList([Block(dim=predictor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                                        for i in range(depth)])
        })
        
        self.predictor_norm_low = norm_layer(predictor_embed_dim)
        self.predictor_norm_mid = norm_layer(predictor_embed_dim)
        self.predictor_norm_high = norm_layer(predictor_embed_dim)
        # self.predictor_norm_last = norm_layer(predictor_embed_dim)

        self.predictor_proj_low = nn.Linear(predictor_embed_dim, embed_dim, bias=True)
        self.predictor_proj_mid = nn.Linear(predictor_embed_dim, embed_dim, bias=True)
        self.predictor_proj_high = nn.Linear(predictor_embed_dim, embed_dim, bias=True)
        # self.predictor_proj_last = nn.Linear(predictor_embed_dim, embed_dim, bias=True)
        # ------
        self.init_std = init_std
        trunc_normal_(self.mask_token_low, std=self.init_std)
        trunc_normal_(self.mask_token_mid, std=self.init_std)
        trunc_normal_(self.mask_token_high, std=self.init_std)
        # trunc_normal_(self.mask_token_last, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        # for layer_id, layer in enumerate(self.predictor_blocks):
        #     rescale(layer.attn.proj.weight.data, layer_id + 1)
        #     rescale(layer.mlp.fc2.weight.data, layer_id + 1)
        for branch_name, block_list in self.predictor_blocks.items():
            for layer_id, block in enumerate(block_list):
                rescale(block.attn.proj.weight.data, layer_id + 1)
                rescale(block.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, masks_x, masks, device):
        # input: [bsz, 3*n_patches, dim]
        # x: context blocks' embedding
        assert (masks is not None) and (masks_x is not None), 'Cannot run predictor without mask indices'
        # masks_x: Context block indices
        if not isinstance(masks_x, list):
            masks_x = [masks_x]
        # masks: Target block indices
        if not isinstance(masks, list):
            masks = [masks]

        # -- Batch Size
        B = len(x) // len(masks_x)
        dim = x.shape[-1]
        if self.lower_dim:
            x1 = self.proj[0](x[:,:,:1536])
            x2 = self.proj[1](x[:,:,1536:3072])
            x3 = self.proj[2](x[:,:,3072:])
            x = torch.cat([x1, x2, x3], dim=2)
        # -- map from encoder-dim to predictor-dim (Context embeddings)
        _, _, dim = x.shape
        d1, d2 = dim//3, (dim//3)*2
        x1 = x[:,:,:d1]
        x2 = x[:,:,d1:d2]
        x3 = x[:,:,d2:]
        for i, _ in enumerate(self.predictor_embed):
            if i == 0:
                x1 = self.predictor_embed[i](x1)
            if i == 1:
                x2 = self.predictor_embed[i](x2)
            if i == 2:
                x3 = self.predictor_embed[i](x3)

        # -- add positional embedding to x tokens
        x_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)
        x1 += apply_masks(x_pos_embed, masks_x, device=device) # Add positional embedding on context blocks indices
        x2 += apply_masks(x_pos_embed, masks_x, device=device) # Add positional embedding on context blocks indices
        x3 += apply_masks(x_pos_embed, masks_x, device=device) # Add positional embedding on context blocks indices
        # x4 += apply_masks(x_pos_embed, masks_x, device=device) # Add positional embedding on context blocks indices
        # Concatenate them again to the seq_len axis
        # x = torch.cat([x1,x2,x3,x4], dim=1)
        x = torch.cat([x1,x2,x3], dim=1)

        _, N_ctxt, D = x.shape

        # -- concat mask tokens to x
        pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
        pos_embs = apply_masks(pos_embs, masks, device=device) # positional embedding on target blocks indices
        pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_x))
        # -- mask tokens: learnable parameters
        pred_tokens_low = self.mask_token_low.repeat(pos_embs.size(0), pos_embs.size(1), 1)
        pred_tokens_mid = self.mask_token_mid.repeat(pos_embs.size(0), pos_embs.size(1), 1)
        pred_tokens_high = self.mask_token_high.repeat(pos_embs.size(0), pos_embs.size(1), 1)
        # pred_tokens_last = self.mask_token_last.repeat(pos_embs.size(0), pos_embs.size(1), 1)
        # --
        pred_tokens_low += pos_embs
        pred_tokens_mid += pos_embs
        pred_tokens_high += pos_embs
        # pred_tokens_last += pos_embs
        x = x.repeat(len(masks), 1, 1)
        x_low = torch.cat([x, pred_tokens_low], dim=1)
        x_mid = torch.cat([x, pred_tokens_mid], dim=1)
        x_high = torch.cat([x, pred_tokens_high], dim=1)
        # x_last = torch.cat([x, pred_tokens_last], dim=1)
        # -- add positional embedding to x tokens

        # emb_list = [x_low, x_mid, x_high, x_last]
        emb_list = [x_low, x_mid, x_high]
        # -- fwd prop (Concat between context embeddings and target blocks. Then, Forward Pass)
        pred_result = []
        for i in range(3):
            # for blk in self.predictor_blocks:
            #     x = blk(emb_list[i])
            x = emb_list[i]
            if i == 0:
                pred_blk = self.predictor_blocks["low_pred"]
                for blk in pred_blk:
                    x = blk(x)
                x = self.predictor_norm_low(x)
                # -- return preds for mask tokens
                x = x[:, N_ctxt:] # after number of context blocks: predicted target blocks
                x = self.predictor_proj_low(x)
            if i == 1:
                pred_blk = self.predictor_blocks["mid_pred"]
                for blk in pred_blk:
                    x = blk(x)
                x = self.predictor_norm_mid(x)
                # -- return preds for mask tokens
                x = x[:, N_ctxt:] # after number of context blocks: predicted target blocks
                x = self.predictor_proj_mid(x)
            if i == 2:
                pred_blk = self.predictor_blocks["high_pred"]
                for blk in pred_blk:
                    x = blk(x)
                x = self.predictor_norm_high(x)
                # -- return preds for mask tokens
                x = x[:, N_ctxt:] # after number of context blocks: predicted target blocks
                x = self.predictor_proj_high(x)
            pred_result.append(x.unsqueeze(0))
        return torch.cat(pred_result, dim=0) # 3, bsz, n_tgt, dim

class VisionTransformerPredictorMM(nn.Module):
    """ Vision Transformer """
    def __init__(
        self,
        num_patches,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        **kwargs
    ):
        super().__init__()
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # --
        self.predictor_pos_embed = nn.Parameter(torch.zeros(1, num_patches, predictor_embed_dim),
                                                requires_grad=False)
        predictor_pos_embed = get_2d_sincos_pos_embed(self.predictor_pos_embed.shape[-1],
                                                      int(num_patches**.5),
                                                      cls_token=False)
        self.predictor_pos_embed.data.copy_(torch.from_numpy(predictor_pos_embed).float().unsqueeze(0))
        # --
        self.predictor_blocks = nn.ModuleList([
            Block(
                dim=predictor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True) # RGB Projector
        # self.predictor_proj_3d = nn.Linear(predictor_embed_dim, 1152, bias=True) # 3D Projector
        # ------
        self.init_std = init_std
        trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, masks_x, masks, device):
        assert (masks is not None) and (masks_x is not None), 'Cannot run predictor without mask indices'

        if not isinstance(masks_x, list):
            masks_x = [masks_x]

        if not isinstance(masks, list):
            masks = [masks]

        # -- Batch Size
        B = len(x) // len(masks_x)

        # -- map from encoder-dim to pedictor-dim
        x = self.predictor_embed(x)

        # -- add positional embedding to x tokens
        x_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)
        x += apply_masks(x_pos_embed, masks_x, device=device)

        _, N_ctxt, D = x.shape

        # -- concat mask tokens to x
        pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
        pos_embs = apply_masks(pos_embs, masks, device=device)
        pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_x))
        # --
        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)
        # --
        pred_tokens += pos_embs
        x = x.repeat(len(masks), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        # -- fwd prop
        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        # -- return preds for mask tokens
        x = x[:, N_ctxt:]
        x = self.predictor_proj(x)

        return x

class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(
        self,
        img_size=[224],
        patch_size=16,
        in_chans=3,
        feature_dim=None,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=12,
        predictor_depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        **kwargs
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        # --
        # For AD-JEPA, Just map into the embedding dimension
        self.patch_embed = nn.Linear(feature_dim, embed_dim, bias=False) # [2304+1152, 768] 
        self.num_patches = 28*28
        # --
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1],
                                            int(self.num_patches**.5),
                                            cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        # --
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # ------
        self.init_std = init_std
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, masks=None):
        if masks is not None:
            if not isinstance(masks, list):
                masks = [masks]

        # -- patchify x
        x = self.patch_embed(x)
        # # -- add positional embedding to x
        # pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)
        # x = x + pos_embed

        # # -- mask x
        # if masks is not None:
        #     x = apply_masks(x, masks, device=x.device)
        # -- fwd prop
        for i, blk in enumerate(self.blocks):
            x = blk(x)
        
        if self.norm is not None:
            x = self.norm(x)
    
        return x

    def interpolate_pos_encoding(self, x, pos_embed):
        npatch = x.shape[1] - 1
        N = pos_embed.shape[1] - 1
        if npatch == N:
            return pos_embed
        class_emb = pos_embed[:, 0]
        pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=math.sqrt(npatch / N),
            mode='bicubic',
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_emb.unsqueeze(0), pos_embed), dim=1)

# class VisionTransformer(nn.Module):
#     """ Vision Transformer """
#     def __init__(
#         self,
#         img_size=[224],
#         patch_size=16,
#         in_chans=3,
#         feature_dim=None,
#         embed_dim=768,
#         predictor_embed_dim=384,
#         depth=12,
#         predictor_depth=12,
#         num_heads=12,
#         mlp_ratio=4.0,
#         qkv_bias=True,
#         qk_scale=None,
#         drop_rate=0.0,
#         attn_drop_rate=0.0,
#         drop_path_rate=0.0,
#         norm_layer=nn.LayerNorm,
#         init_std=0.02,
#         **kwargs
#     ):
#         super().__init__()
#         self.num_features = self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.feature_dim = feature_dim
#         # --
#         # For AD-JEPA, Just map into the embedding dimension
#         self.low_patch_embed = nn.Linear(feature_dim//3, embed_dim)
#         self.mid_patch_embed = nn.Linear(feature_dim//3, embed_dim)
#         self.final_patch_embed = nn.Linear(feature_dim//3, embed_dim)
#         self.num_patches = 28*28
#         # --
#         self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=False)
#         pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1],
#                                             int(self.num_patches**.5),
#                                             cls_token=False)
#         self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
#         # --
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
#         self.blocks = nn.ModuleList([
#             Block(
#                 dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
#                 drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
#             for i in range(depth)])
#         self.norm = norm_layer(embed_dim)
#         # ------
#         self.init_std = init_std
#         self.apply(self._init_weights)
#         self.fix_init_weight()

#     def fix_init_weight(self):
#         def rescale(param, layer_id):
#             param.div_(math.sqrt(2.0 * layer_id))

#         for layer_id, layer in enumerate(self.blocks):
#             rescale(layer.attn.proj.weight.data, layer_id + 1)
#             rescale(layer.mlp.fc2.weight.data, layer_id + 1)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=self.init_std)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             trunc_normal_(m.weight, std=self.init_std)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x, masks=None):
#         if masks is not None:
#             if not isinstance(masks, list):
#                 masks = [masks]

#         # -- patchify x
#         x1 = self.low_patch_embed(x[:,:,:self.feature_dim//3])
#         x2 = self.mid_patch_embed(x[:,:,self.feature_dim//3:(self.feature_dim//3)*2])
#         x3 = self.final_patch_embed(x[:,:,(self.feature_dim//3)*2:])
#         # -- add positional embedding to x
#         pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)
#         x1 = x1 + pos_embed
#         x2 = x2 + pos_embed
#         x3 = x3 + pos_embed

#         # -- mask x
#         if masks is not None:
#             x1 = apply_masks(x1, masks, device=x.device)
#             x2 = apply_masks(x2, masks, device=x.device)
#             x3 = apply_masks(x3, masks, device=x.device)
#         # -- fwd prop
#         x = torch.cat([x1, x2, x3], dim=1)
#         for i, blk in enumerate(self.blocks):
#             x = blk(x)
        
#         if self.norm is not None:
#             x = self.norm(x)
        
#         _, n, d = x.shape
#         m1, m2 = n//3, (n//3)*2
#         x1 = x[:,:m1, :]
#         x2 = x[:,m1:m2,:]
#         x3 = x[:,m2:,:]
#         x = torch.cat([x1, x2, x3], dim=0)
#         return x

#     def interpolate_pos_encoding(self, x, pos_embed):
#         npatch = x.shape[1] - 1
#         N = pos_embed.shape[1] - 1
#         if npatch == N:
#             return pos_embed
#         class_emb = pos_embed[:, 0]
#         pos_embed = pos_embed[:, 1:]
#         dim = x.shape[-1]
#         pos_embed = nn.functional.interpolate(
#             pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
#             scale_factor=math.sqrt(npatch / N),
#             mode='bicubic',
#         )
#         pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
#         return torch.cat((class_emb.unsqueeze(0), pos_embed), dim=1)


def vit_predictor(**kwargs):
    model = VisionTransformerPredictor(
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model

def vit_predictor_mm(**kwargs):
    model = VisionTransformerPredictorMM(
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model


def vit_tiny(patch_size=16, depth=12, feature_dim=768, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, feature_dim=feature_dim, embed_dim=192, depth=depth, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small(patch_size=16, depth=12, feature_dim=768, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, feature_dim=feature_dim, embed_dim=384, depth=depth, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(patch_size=16, depth=12, feature_dim=768, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, feature_dim=feature_dim, embed_dim=768, depth=depth, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large(patch_size=16, depth=24, feature_dim=768, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, feature_dim=feature_dim, embed_dim=1024, depth=depth, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge(patch_size=16, depth=32, feature_dim=768, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, feature_dim=feature_dim, embed_dim=1280, depth=depth, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_giant(patch_size=16, depth=40, feature_dim=768, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, feature_dim=feature_dim, embed_dim=1408, depth=depth, num_heads=16, mlp_ratio=48/11,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


VIT_EMBED_DIMS = {
    'vit_tiny': 192,
    'vit_small': 384,
    'vit_base': 768,
    'vit_large': 1024,
    'vit_huge': 1280,
    'vit_giant': 1408,
}