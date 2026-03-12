import torch
import torch.nn as nn
import torch.nn.functional as F

class AttnBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=12):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim,
                                               num_heads=num_heads,
                                               batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, emb1, emb2):
        # Cross Attention
        attn_output, _ = self.attention(query=emb1, key=emb2, value=emb2)
        emb1 = emb1 + attn_output
        emb1 = self.norm1(emb1)

        # Feed Forward + Skip
        ffn_output = self.ffn(emb1)
        emb1 = emb1 + ffn_output
        emb1 = self.norm2(emb1)
        return emb1, emb2


class CrossNet(nn.Module):
    def __init__(self, dim1, dim2, proj_dim, num_blocks=1):
        super().__init__()
        # dim1: main modality 
        # dim2: sub modality
        self.proj1 = nn.Linear(dim1, proj_dim)
        self.proj2 = nn.Linear(dim2, proj_dim)
        self.blocks = nn.ModuleList([
            AttnBlock(embed_dim=proj_dim) for _ in range(num_blocks)
        ])
        # self.pool = nn.AdaptiveAvgPool2d(pool_dim)
        self.proj = nn.Linear(proj_dim, proj_dim)

    def forward(self, emb1, emb2):
        emb1 = self.proj1(emb1)
        emb2 = self.proj2(emb2)
        for i, blk in enumerate(self.blocks):
            emb1, emb2 = blk(emb1=emb1, emb2=emb2)
        emb1 = self.proj(emb1)
        return emb1

class InfoNCE():
    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature
    
    def __call__(self, emb1, emb2):
        # Calculate InfoNCE Loss
        if emb1.dim() == 3 and emb2.dim() == 3:
            b,n,_ = emb1.shape
            emb1 = F.normalize(emb1, p=2, dim=-1)
            emb2 = F.normalize(emb2, p=2, dim=-1)
            emb1 = emb1.reshape(b*n, -1)
            emb2 = emb2.reshape(b*n, -1)
            logits = torch.matmul(emb1, emb2.t()) / self.temperature
            N = emb1.size(0)
            label = torch.arange(N, device=emb1.device)
            loss = F.cross_entropy(logits, label) + F.cross_entropy(logits.t(), label)
        else:
            emb1 = F.normalize(emb1, p=2, dim=-1)
            emb2 = F.normalize(emb2, p=2, dim=-1)
            logits = torch.matmul(emb1, emb2.t()) / self.temperature
            N = emb1.size(0)
            label = torch.arange(N, device=emb1.device)
            loss = F.cross_entropy(logits, label) + F.cross_entropy(logits.t(), label)
        return 0.5*loss # bi-directional 