from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
from timm.models.layers import DropPath
from torch import nn

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        _attn = attn
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, _attn

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

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        feature_dim=1152,
        embed_dim=768,
        depth=4,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate: float | List[float] = 0.0,
        add_pos_at_every_layer=True,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=(
                        drop_path_rate[i]
                        if isinstance(drop_path_rate, list)
                        else drop_path_rate
                    ),
                )
                for i in range(depth)
            ]
        )

        # output norm
        self.norm = nn.LayerNorm(embed_dim)

        self.low_patch_embed = nn.Linear(feature_dim//3, embed_dim)
        self.mid_patch_embed = nn.Linear(feature_dim//3, embed_dim)
        self.final_patch_embed = nn.Linear(feature_dim//3, embed_dim)

        self.positional_encoding = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim),
        ) # Positional Encoding (Learnable)

        self.add_pos_at_every_layer = add_pos_at_every_layer

    def forward(
        self,
        x: torch.Tensor,
        center_x: torch.Tensor, # if target: full center, context: context center
    ):
        # -- patchify x
        # input: [BSZ, N_CTXT, DIM] => [BSZ, N_CTXT, 1152]: Full context
        x1 = self.low_patch_embed(x[:,:,:self.feature_dim//3]) # [bsz, N_CTXT, 384]
        x2 = self.mid_patch_embed(x[:,:,self.feature_dim//3:(self.feature_dim//3)*2]) # [bsz, N_CTXT, 384]
        x3 = self.final_patch_embed(x[:,:,(self.feature_dim//3)*2:]) # [bsz, N_CTXT, 384]

        pos = self.positional_encoding(center_x) # [BSZ, N_CTXT, 3] -> [BSZ, N_CTXT, DIM]
        
        if not self.add_pos_at_every_layer:
            x1 = x1 + pos
            x2 = x2 + pos
            x3 = x3 + pos
        for block in self.blocks:
            if self.add_pos_at_every_layer:
                x1 = x1 + pos
                x2 = x2 + pos
                x3 = x3 + pos
            x = torch.cat([x1, x2, x3], dim=1) # [BSZ, 3*N_CTXT, DIM] -> Multi-level attention
            x = block(x)
        
        x = self.norm(x) # Layer Normalization
        
        _, n, d = x.shape
        m1, m2 = n//3, (n//3)*2
        x1 = x[:,:m1, :]
        x2 = x[:,m1:m2,:]
        x3 = x[:,m2:,:]
        x = torch.cat([x1, x2, x3], dim=0) # [bsz * 3, N_CTXT, 384]
        
        return x

class TransformerPredictor(nn.Module):
    def __init__(
        self,
        embed_dim=384,
        predictor_embed_dim=192,
        depth=1,
        num_heads=4,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.05,
        drop_path_rate=0.25,
        add_pos_at_every_layer=True,
        add_target_pos=True,
    ):
        super().__init__()
        self.predictor_embed_low = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.predictor_embed_mid = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.predictor_embed_high = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        
        self.mask_token_low = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim)) # Learnable Parameter (Low-level)
        self.mask_token_mid = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim)) # Learnable Parameter (Mid-level)
        self.mask_token_high = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim)) # Learnable Parameter (High-level)

        # Here we use the same positional encoding as the student
        self.positional_encoding_ctx = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, predictor_embed_dim),
        )

        self.positional_encoding_tgt = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, predictor_embed_dim),
        )

        self.predictor_norm_low = nn.LayerNorm(predictor_embed_dim)
        self.predictor_norm_mid = nn.LayerNorm(predictor_embed_dim)
        self.predictor_norm_high = nn.LayerNorm(predictor_embed_dim)

        self.predictor_proj_low = nn.Linear(predictor_embed_dim, embed_dim, bias=True)
        self.predictor_proj_mid = nn.Linear(predictor_embed_dim, embed_dim, bias=True)
        self.predictor_proj_high = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.predictor_blocks = nn.ModuleDict({
            "low_pred": nn.ModuleList([Block(dim=predictor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                             drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i])
                                             for i in range(depth)]),
            "mid_pred": nn.ModuleList([Block(dim=predictor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                             drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i])
                                             for i in range(depth)]),
            "high_pred": nn.ModuleList([Block(dim=predictor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                              drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i])
                                              for i in range(depth)])
            })
        
        self.add_target_pos = add_target_pos

    def forward(self, x, center_x, center_pred):
        # x: (B, N, C)
        # center_x: (B, N, 3)
        # center_pred: (B, T, 3)

        # Bring it down to narrower dimension
        
        _, _, dim = x.shape
        x1 = self.predictor_embed_low(x[:,:,:dim//3])
        x2 = self.predictor_embed_mid(x[:,:,dim//3:(dim//3)*2])
        x3 = self.predictor_embed_high(x[:,:,(dim//3)*2:])

        # Add positional encoding
        # assert center_x, "Need center co-oridinates of Context blocks"
        
        pos = self.positional_encoding_ctx(center_x)

        x1 = x1 + pos
        x2 = x2 + pos
        x3 = x3 + pos

        x = torch.cat([x1, x2, x3], dim=1)

        _, N_ctxt, _ = x.shape
        B, N_tgt, _ = center_pred.shape

        # concate mask tokens to x
        pos_embed = self.positional_encoding_tgt(center_pred)  # (B, T, predictor_embed_dim)

        mask_tokens_low = self.mask_token_low.repeat(B, N_tgt, 1) + pos_embed # (B, T, predictor_embed_dim)
        mask_tokens_mid = self.mask_token_mid.repeat(B, N_tgt, 1) + pos_embed  # (B, T, predictor_embed_dim)
        mask_tokens_high = self.mask_token_high.repeat(B, N_tgt, 1) + pos_embed  # (B, T, predictor_embed_dim)

        x = x.repeat(B, 1, 1) # Repeat B Times
        x_low = torch.cat([x, mask_tokens_low], dim=1)
        x_mid = torch.cat([x, mask_tokens_mid], dim=1)
        x_high = torch.cat([x, mask_tokens_high], dim=1)

        emb_list = [x_low, x_mid, x_high]
        # -- fwd prop (Concat between context embeddings and target blocks. Then, Forward Pass)
        pred_result = []
        for i in range(3):
            # for blk in self.predictor_blocks:
            #     x = blk(emb_list[i])
            x = emb_list[i]
            if i == 0:
                pred_block = self.predictor_blocks["low_pred"]
                for blk in pred_block:
                    x = blk(x)
                x = self.predictor_norm_low(x)
                # -- return preds for mask tokens
                x = x[:, N_ctxt:] # after number of context blocks: predicted target blocks
                x = self.predictor_proj_low(x)
            if i == 1:
                pred_block = self.predictor_blocks["mid_pred"]
                for blk in pred_block:
                    x = blk(x)
                x = self.predictor_norm_mid(x)
                # -- return preds for mask tokens
                x = x[:, N_ctxt:] # after number of context blocks: predicted target blocks
                x = self.predictor_proj_mid(x)
            if i == 2:
                pred_block = self.predictor_blocks["high_pred"]
                for blk in pred_block:
                    x = blk(x)
                x = self.predictor_norm_high(x)
                # -- return preds for mask tokens
                x = x[:, N_ctxt:] # after number of context blocks: predicted target blocks
                x = self.predictor_proj_high(x)
            pred_result.append(x)
        
        return torch.cat(pred_result, dim=0)
