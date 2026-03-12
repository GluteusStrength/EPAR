import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetDiscriminator(nn.Module):
    def __init__(self, in_ch=2304, dim1=1152, dim2=576, scale_factor=8):
        super().__init__()

        # -------- Encoder --------
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, dim1, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim1),
            nn.SiLU(),
            nn.Conv2d(dim1, dim1, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim1),
            nn.SiLU()
        )

        self.down1 = nn.Conv2d(dim1, dim2, kernel_size=4, stride=2, padding=1)  # 28 -> 14

        self.enc2 = nn.Sequential(
            nn.BatchNorm2d(dim2),
            nn.SiLU(),
            nn.Conv2d(dim2, dim2, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim2),
            nn.SiLU()
        )

        self.down2 = nn.Conv2d(dim2, dim2, kernel_size=4, stride=2, padding=1)  # 14 -> 7

        # -------- Bottleneck --------
        self.bottleneck = nn.Sequential(
            nn.BatchNorm2d(dim2),
            nn.SiLU(),
            nn.Conv2d(dim2, dim2, kernel_size=3, padding=1),
            nn.SiLU()
        )

        # -------- Decoder --------
        self.up1 = nn.ConvTranspose2d(dim2, dim2, kernel_size=4, stride=2, padding=1)  # 7 -> 14
        self.dec1 = nn.Sequential(
            nn.BatchNorm2d(dim2 * 2),
            nn.SiLU(),
            nn.Conv2d(dim2 * 2, dim2, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim2),
            nn.SiLU()
        )

        self.up2 = nn.ConvTranspose2d(dim2, dim1, kernel_size=4, stride=2, padding=1)  # 14 -> 28
        self.dec2 = nn.Sequential(
            nn.BatchNorm2d(dim1 * 2),
            nn.SiLU(),
            nn.Conv2d(dim1 * 2, in_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.SiLU()
        )

        # -------- Final upsampling to 224×224 --------
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),  # 28 -> 224
            nn.Conv2d(in_ch, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # -------- Encoder --------
        e1 = self.enc1(x)               # [B, dim1, 28, 28]
        d1 = self.down1(e1)             # [B, dim2, 14, 14]

        e2 = self.enc2(d1)              # [B, dim2, 14, 14]
        d2 = self.down2(e2)             # [B, dim2, 7, 7]

        # -------- Bottleneck --------
        b = self.bottleneck(d2)         # [B, dim2, 7, 7]

        # -------- Decoder --------
        u1 = self.up1(b)                # [B, dim2, 14, 14]
        u1 = torch.cat([u1, e2], dim=1) # Skip connection
        u1 = self.dec1(u1)

        u2 = self.up2(u1)               # [B, dim1, 28, 28]
        u2 = torch.cat([u2, e1], dim=1) # Skip connection
        u2 = self.dec2(u2)

        # -------- Output --------
        out = self.final_up(u2)         # [B, 1, 224, 224]
        return u2, out

class LinearDiscriminator(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2),
            nn.SiLU(),
            nn.Linear(embed_dim//2, 1)
        )
    def forward(self, x):
        x = self.discriminator(x)
        return x