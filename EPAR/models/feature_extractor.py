import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.src.apply_masks import apply_masks
from pointnet2_ops import pointnet2_utils
from models.target_sampler import TargetSampler
from models.context_sampler import ContextSampler
from models.greedy_sequencer import PointSequencer
from utils import point_transforms as transforms
from collections import OrderedDict

import timm
from timm.models.layers import DropPath, trunc_normal_
from knn_cuda import KNN

"""
This is the feature extractor for the pretrained Point Transformer model and ViTB-8 DINO.
It is used to extract features from pretrained models.
The Point Transformer model is used to extract features from point clouds, and the ViTB-8 DINO model is used to extract features from images.

Additional Point:
- Add LoRA to the Feature Extractor
- Integrate JEPA-style Context and Target sampling into PointTransformer
"""

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

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb

class AutoEncoder(nn.Module):
    def __init__(self, embed_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, (embed_dim*2)//3),
            nn.SiLU(),
            nn.Linear((embed_dim*2)//3, latent_dim),
            nn.SiLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, (embed_dim*2)//3),
            nn.SiLU(),
            nn.Linear((embed_dim*2)//3, embed_dim)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class AdaptorBlock(nn.Module):
    def __init__(self, in_dim, proj_dim, nhead, embed_dim, ff_dim, grid_size=28):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, proj_dim)

        pos_embed = get_2d_sincos_pos_embed(embed_dim=proj_dim, grid_size=grid_size, cls_token=False)
        pos_embed = torch.from_numpy(pos_embed).float()  # [N, D]
        self.register_buffer('pos_embed', pos_embed.unsqueeze(0))  # [1, N, D]

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_layer = nn.TransformerEncoder(encoder_layer, num_layers=1)
    
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

    def forward(self, x, context_indices=None):
        """
        x: [B, N, in_dim] -> proj_dim
        pos_embed: [1, N, proj_dim]
        """
        x = self.layer_norm(x)
        x = self.proj(x)

        # pos_embed = self.pos_embed
        # B, _, _ = x.shape
        # pos_embed = pos_embed.repeat(B, 1, 1)

        # if context_indices is not None:
        #     pos_embed = apply_masks(pos_embed, context_indices, device=x.device)
    
        # if pos_embed.shape[1] != x.shape[1]:
        #     raise ValueError(f"pos_embed seq_len {pos_embed.shape[1]} != input seq_len {x.shape[1]}")

        # x = x + pos_embed  # fixed sin-cos positional embedding

        x = self.transformer_layer(x)

        return x

class RGBBackbone(torch.nn.Module):

    def __init__(self, device, rgb_backbone_name='vit_base_patch8_224_dino', out_indices=None, rgb_checkpoint_path='', num_blocks=12, pooling_dim=(32, 32)):
        super().__init__()
        self.device = device
        kwargs = {'features_only': True if out_indices else False}
        if out_indices:
            kwargs.update({'out_indices': out_indices})
        self.rgb_backbone = timm.create_model(
            model_name=rgb_backbone_name, 
            pretrained=True,
            **kwargs
        )
        # self.pooling = nn.AdaptiveAvgPool2d(pooling_dim)
        # self.linear_projector = nn.Linear(768, 768)
        self.rgb_backbone.blocks = self.rgb_backbone.blocks[:num_blocks]
        self.fetch_idx = [3,7,11]
        self.num_blocks = num_blocks

    def forward(self, rgb, abnormal_rgb=None, bg_indices=None, context_mask=None):
        rgb = self.rgb_backbone.patch_embed(rgb)
        rgb = self.rgb_backbone._pos_embed(rgb)
        rgb = self.rgb_backbone.norm_pre(rgb)
        x = rgb[:,1:,:]
        if abnormal_rgb is not None:
            abnormal_rgb = self.rgb_backbone.patch_embed(abnormal_rgb)
            abnormal_rgb = self.rgb_backbone._pos_embed(abnormal_rgb)
            abnormal_rgb = self.rgb_backbone.norm_pre(abnormal_rgb)
            x2 = abnormal_rgb[:,1:,:]
            x_prior = x.clone()
            x2_prior = x2.clone()
        if context_mask is not None:
            x = apply_masks(x, context_mask, device=x.device)
            if abnormal_rgb is not None:
                x2 = apply_masks(x2, context_mask, device=x2.device)
        feature_list = []
        feature_list_abnormal = []
        for i, block in enumerate(self.rgb_backbone.blocks):
            x = block(x)
            if abnormal_rgb is not None:
                x2 = block(x2)
            if i in self.fetch_idx:
                feature_list.append(F.layer_norm(x, (x.size(-1), )))
                if abnormal_rgb is not None:
                    feature_list_abnormal.append(F.layer_norm(x2, (x2.size(-1), )))
                
        if abnormal_rgb is not None:
            return torch.cat(feature_list, dim=2), torch.cat(feature_list_abnormal, dim=2), x_prior, x2_prior
        else:
            return torch.cat(feature_list, dim=2)


class XYZBackbone(torch.nn.Module):
    def __init__(self, device, rgb_backbone_name='vit_base_patch8_224_dino', out_indices=None, rgb_checkpoint_path='',
                 pool_last=False, xyz_backbone_name='Point_MAE', xyz_checkpoint_path='', group_size=128, num_group=1024):
        super().__init__()
        self.device = device
        if xyz_backbone_name=='Point_MAE':
            self.xyz_backbone = PointTransformer(group_size=group_size, num_group=num_group)
            self.xyz_backbone.load_model_from_ckpt(xyz_checkpoint_path)
        elif xyz_backbone_name=='Point_Bert':
            self.xyz_backbone = PointTransformer(group_size=group_size, num_group=num_group, encoder_dims=256)
            self.xyz_backbone.load_model_from_pb_ckpt(xyz_checkpoint_path)

    def forward(self, xyz, abnormal_pts=None, npred=None, mode="target"):
        return self.xyz_backbone(pts=xyz, abnormal_pts=abnormal_pts, mode=mode, npred=npred)


def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = pointnet2_utils.gather_operation(
        data.transpose(1, 2).contiguous(), fps_idx
    ).transpose(1, 2).contiguous()
    return fps_data, fps_idx


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            output:
              neighborhood: B G M 3
              center:       B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        center, center_idx = fps(xyz.contiguous(), self.num_group)
        _, idx = self.knn(xyz, center)
        ori_idx = idx
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.reshape(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.reshape(
            batch_size, self.num_group, self.group_size, 3
        ).contiguous()
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center, ori_idx, center_idx


class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([
            feature_global.expand(-1, -1, n),
            feature
        ], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, self.encoder_channel)


class Mlp(nn.Module):
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
        qkv = self.qkv(x).reshape(
            B, N, 3, self.num_heads, C // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
        drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            ) for i in range(depth)
        ])

        self.fetch_idx = [3,7,11]

    def forward(self, x, pos, mode="context"):
        feature_list = []
        for i, block in enumerate(self.blocks):
            x = block(x + pos)
            if i in self.fetch_idx:
                feature_list.append(x)

        return feature_list


class PointTransformer(nn.Module):
    def __init__(
        self, group_size=128, num_group=1024, encoder_dims=384
    ):
        super().__init__()
        self.trans_dim = 384
        self.depth = 12
        self.drop_path_rate = 0.1
        self.num_heads = 6
        self.group_size = group_size
        self.num_group = num_group
        # Farthest point sampling
        self.group_divider = Group(
            num_group=self.num_group, group_size=self.group_size
        )
        # self.linear_projector = nn.Linear(encoder_dims, encoder_dims)
        self.encoder_dims = encoder_dims
        if self.encoder_dims != self.trans_dim:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
            self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
            self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)
        # Point Encoder
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        # greedy sequencer
        self.point_sequencer = PointSequencer(method="morton")
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, self.trans_dim)
        )
        dpr = [x.item() for x in torch.linspace(
            0, self.drop_path_rate, self.depth
        )]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim, depth=self.depth,
            drop_path_rate=dpr, num_heads=self.num_heads
        )
        # Global Representation Learning
        self.norm = nn.LayerNorm(self.trans_dim)

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]
            self.load_state_dict(base_ckpt, strict=False)

    def load_model_from_pb_ckpt(self, bert_ckpt_path):
        ckpt = torch.load(bert_ckpt_path)
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        for k in list(base_ckpt.keys()):
            if k.startswith('transformer_q') and not k.startswith('transformer_q.cls_head'):
                base_ckpt[k[len('transformer_q.'):]] = base_ckpt[k]
            elif k.startswith('base_model'):
                base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
            del base_ckpt[k]
        self.load_state_dict(base_ckpt, strict=False)

    def forward(self, pts, abnormal_pts=None, npred=None, mode="context"):
        B, C, N = pts.shape
        pts = pts.transpose(-1, -2)  # B N 3
        neighborhood, center, ori_idx, center_idx = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood) # pass PointEncoder (Make into embeddiing)
        # group_input_tokens = self.linear_projector(group_input_tokens)
        if abnormal_pts is not None:
            abnormal_pts = abnormal_pts.transpose(-1, -2)
            abnormal_neighborhood, abnormal_center, abnormal_ori_idx, abnormal_center_idx = self.group_divider(abnormal_pts)
            group_input_tokens_abnormal = self.encoder(abnormal_neighborhood)
            # group_input_tokens_abnormal = self.linear_projector(group_input_tokens_abnormal)
            g1 = group_input_tokens.clone()
            g2 = group_input_tokens_abnormal.clone()
        
        if mode == "context":
            # integrate JEPA sampling
            if npred is None:
                npred = 4
            ts = TargetSampler(
                sample_method='contiguous',
                num_targets_per_sample=npred,
                sample_ratio_range=(0.15, 0.2),
                device=group_input_tokens.device
            )
            target_blocks, target_indices = ts.sample(group_input_tokens)
            # 2026.1.9 -> best choice is contiguous but take an experiment with rest
            cs = ContextSampler(
                sample_method='contiguous',
                sample_ratio_range=(0.4, 0.75),
                device=group_input_tokens.device
            )
            _, context_centers, context_indices = cs.sample(
                tokens=group_input_tokens,
                centers=center,
                target_indices=target_indices.flatten()
            ) # context centers: predictor input
            
            context_tokens = group_input_tokens[:, context_indices, :]
            if abnormal_pts is not None:
                context_tokens_abnormal = group_input_tokens_abnormal[:, context_indices, :]
            pos_ctx = self.pos_embed(context_centers)
            # if self.encoder_dims != self.trans_dim:
            #     x = torch.cat([cls_tokens, context_tokens], dim=1)
            #     pos = torch.cat([cls_pos, pos_ctx], dim=1)
            # else:
            x = context_tokens
            if abnormal_pts is not None:
                x_ab = context_tokens_abnormal
            pos = pos_ctx
            feature_list = self.blocks(x, pos, mode=mode)
            feature_list = [self.norm(x).transpose(-1, -2).contiguous() for x in feature_list]
            if abnormal_pts is not None:
                feature_list_abnormal = self.blocks(x_ab, pos, mode=mode)
                feature_list_abnormal = [self.norm(x).transpose(-1, -2).contiguous() for x in feature_list_abnormal]
            x = torch.cat((feature_list[0],feature_list[1],feature_list[2]), dim=1) #1152
            
            if abnormal_pts is None:
                return x, center, ori_idx, center_idx, \
                    target_blocks, target_indices, context_tokens, \
                    context_centers, center, context_indices
            else:
                return x, center, ori_idx, center_idx, \
                    target_blocks, target_indices, context_tokens, \
                    context_centers, center, context_indices, \
                        torch.cat((feature_list_abnormal[0],feature_list_abnormal[1],feature_list_abnormal[2]), dim=1), g1, g2
        else:
            pos = self.pos_embed(center)
            x = group_input_tokens
            if abnormal_pts is not None:
                x_ab = group_input_tokens_abnormal
            feature_list = self.blocks(x, pos, mode=mode)
            feature_list = [self.norm(x).transpose(-1, -2).contiguous() for x in feature_list]
            if abnormal_pts is not None:
                feature_list_abnormal = self.blocks(x_ab, pos, mode=mode)
                feature_list_abnormal = [self.norm(x).transpose(-1, -2).contiguous() for x in feature_list_abnormal]
            x = torch.cat((feature_list[0],feature_list[1],feature_list[2]), dim=1) #1152
            
            if abnormal_pts is not None:
                return x, center, ori_idx, center_idx, torch.cat((feature_list_abnormal[0],feature_list_abnormal[1],feature_list_abnormal[2]), dim=1) #1152
            else:
                return x, center, ori_idx, center_idx
