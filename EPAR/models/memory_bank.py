import torch
import torch.nn as nn
import torch.nn.functional as F

class MMUpdateRGB(nn.Module):
    def __init__(self, init_memory_bank, embed_dim, k=1):
        super(MMUpdateRGB, self).__init__()
        self.k = k
        self.embed_dim = embed_dim
        # Memory Bank (Learnable params)
        self.memory_bank = nn.Parameter(init_memory_bank.clone(), requires_grad=False)
        self.up = nn.Sequential(
            # 28x28 → 56x56 (4608 -> 2304)
            nn.Conv2d(embed_dim, embed_dim//2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            # 56x56 → 112x112 (2304 -> 1152)
            nn.Conv2d(embed_dim//2, (embed_dim//2)//2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            # 112x112 → 224x224 (1152 -> 576)
            nn.Conv2d((embed_dim//2)//2, ((embed_dim//2)//2)//2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )

        self.seg_head = nn.Conv2d(((embed_dim//2)//2)//2, 1, kernel_size=3, padding=1)

    def forward(self, query):
        """
        Args:
            query (Tensor): [N_patches, D] -> [784, 2304]
        Returns:
            score (Tensor): [N_patches, 1]
        """
        # 1. Calculate Distances
        # query = self.query_proj(query)
        dist_l2 = torch.cdist(query, self.memory_bank, p=2)

        _, topk_indices = torch.topk(dist_l2, k=self.k, largest=False, dim=1)
        dist_l2, _ = torch.min(dist_l2, dim=1)
        dist_l2 = dist_l2 ** 2

        # 3. Fetch Top-k Representations (Gradients will flow here)
        neighbors = self.memory_bank[topk_indices] 

        # 4. Mean Pooling (Aggregate)
        neighbor_mean = torch.mean(neighbors, dim=1)
        # 5. Feature Fusion for Discrimination
        combined_features = torch.cat([query, neighbor_mean], dim=1)
        combined_features = combined_features.unsqueeze(0)

        bsz, npatches, dim = combined_features.shape
        combined_features = combined_features.permute(0,2,1).reshape(bsz, dim, int(npatches**0.5), int(npatches**0.5))
        # 6. Discrimination (n_patches, 1)
        # class_res = self.discrimination(combined_features) # [1, 1, 224 ,224]
        class_res = self.up(combined_features) # [1, 1, 224 ,224]
        class_res1 = self.seg_head(class_res)
        
        return class_res1, dist_l2

    
class MMUpdate3D(nn.Module):
    def __init__(self, init_memory_bank, embed_dim, k=1):
        super(MMUpdate3D, self).__init__()
        self.k = k
        self.embed_dim = embed_dim
    
        # Memory Bank (Learnable params)
        self.memory_bank = nn.Parameter(init_memory_bank.clone(), requires_grad=False)
        
        # Discriminator
        self.up = nn.Sequential(
            # 56x56 → 112x112 (1152 -> 576)
            nn.Conv2d(embed_dim, embed_dim//2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            # 112x112 → 224x224 (576 -> 288)
            nn.Conv2d(embed_dim//2, (embed_dim//2)//2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )
        self.seg_head = nn.Conv2d((embed_dim//2)//2, 1, kernel_size=3, padding=1)

    def forward(self, query):
        """
        Args:
            query (Tensor): [N_patches, D] -> [784, 2304]
        Returns:
            score (Tensor): [N_patches, 1]
        """
        # 1. Calculate Distances
        # query = self.query_proj(query)
        dist_l2 = torch.cdist(query, self.memory_bank, p=2)

        _, topk_indices = torch.topk(dist_l2, k=self.k, largest=False, dim=1)
        dist_l2, _ = torch.min(dist_l2, dim=1)
        dist_l2 = dist_l2 ** 2

        # 3. Fetch Top-k Representations (Gradients will flow here)
        neighbors = self.memory_bank[topk_indices] 

        # 4. Mean Pooling (Aggregate)
        neighbor_mean = torch.mean(neighbors, dim=1)
        # 5. Feature Fusion for Discrimination
        combined_features = torch.cat([query, neighbor_mean], dim=1)
        combined_features = combined_features.unsqueeze(0)

        bsz, npatches, dim = combined_features.shape
        combined_features = combined_features.permute(0,2,1).reshape(bsz, dim, int(npatches**0.5), int(npatches**0.5))
        # 6. Discrimination (n_patches, 1)
        # class_res = self.discrimination(combined_features) # [1, 1, 224 ,224]
        class_res = self.up(combined_features) # [1, 1, 224 ,224]
        class_res1 = self.seg_head(class_res)
        
        return class_res1, dist_l2

class MMUpdateMultimodal(nn.Module):
    def __init__(self, init_memory_bank, embed_dim_rgb, embed_dim_3d, k=1):
        super(MMUpdateMultimodal, self).__init__()
        self.k = k
        self.embed_dim_rgb = embed_dim_rgb
        self.embed_dim_xyz = embed_dim_3d
        self.embed_dim = (embed_dim_rgb + embed_dim_3d) * 2
        
        # self.query_proj = nn.Linear(embed_dim//2, embed_dim//2)
        # Memory Bank (Learnable params)
        self.memory_bank = nn.Parameter(init_memory_bank.clone(), requires_grad=True)
        
        # Multimodal Confidence
        self.up_mm = nn.Sequential(
            # 28x28 → 56x56 (4608 -> 1152)
            nn.Conv2d(self.embed_dim*2, self.embed_dim//2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            # 56x56 → 112x112 (1152 -> 384)
            nn.Conv2d(self.embed_dim//2, ((self.embed_dim)//2)//2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            # 112x112 → 224x224 (384 -> 128)
            nn.Conv2d(((self.embed_dim)//2)//2, (((self.embed_dim)//2)//2)//2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        ) 

        # final refine conv (optional)
        self.fcn = nn.Conv2d((((embed_dim_rgb + embed_dim_3d)//2)//2)//2, 1, kernel_size=3, padding=1)
    
    def forward(self, query):
        """
        Args:
            query (Tensor): [N_patches, D] -> [784, 2304]
        Returns:
            score (Tensor): [N_patches, 1]
        """
        # 1. Calculate Distances
        dist_l2 = torch.cdist(query, self.memory_bank, p=2)

        _, topk_indices = torch.topk(dist_l2, k=self.k, largest=False, dim=1)
        dist_l2, _ = torch.min(dist_l2, dim=1)

        neighbors = self.memory_bank[topk_indices] 

        # 4. Mean Pooling (Aggregate)
        neighbor_mean = torch.mean(neighbors, dim=1)
        # 5. Feature Fusion for Discrimination
        combined_features = torch.cat([query, neighbor_mean], dim=1)
        combined_features = combined_features.unsqueeze(0)

        bsz, npatches, dim = combined_features.shape
        combined_features = combined_features.permute(0,2,1).reshape(bsz, dim, int(npatches**0.5), int(npatches**0.5))
        # 6. Discrimination (n_patches, 1)
        class_res = self.fcn(combined_features) # [1, 1, 224 ,224]
        
        return class_res

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.95, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # controls class imbalance
        self.gamma = gamma  # focuses on hard examples
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Calculate Binary Cross-Entropy Loss for each sample
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Compute pt (model confidence on true class)
        pt = torch.exp(-BCE_loss)
        
        # Apply the focal adjustment
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        # Apply reduction (mean, sum, or no reduction)
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss