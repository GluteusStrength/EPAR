# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from pointnet2_ops import pointnet2_utils 

# def interpolating_points(xyz1, xyz2, points2):
#     """
#     Args:
#         xyz1: (B, 3, N) - Target Points 
#         xyz2: (B, 3, S) - Source Points 
#         points2: (B, Dim, S) - Source Features (Point-MAE features)
#     Returns:
#         interpolated_points: (B, Dim, N) - interpolated features
#     """
#     xyz1 = xyz1.permute(0, 2, 1).contiguous()
#     xyz2 = xyz2.permute(0, 2, 1).contiguous()
#     points2 = points2.permute(0, 2, 1).contiguous() # (B, S, Dim)

#     B, N, _ = xyz1.shape
#     _, S, _ = xyz2.shape

#     dist, idx = pointnet2_utils.three_nn(xyz1, xyz2)
#     dist = torch.clamp(dist, min=1e-10)
#     norm = torch.sum(1.0 / dist, dim=2, keepdim=True)
#     weights = (1.0 / dist) / norm # (B, N, 3)
    
#     interpolated_points = pointnet2_utils.three_interpolate(points2, idx, weights)
    
#     return interpolated_points

# class PointFeatureAlignment(nn.Module):
#     def __init__(self, image_size=224, patch_size=8):
#         super().__init__()
#         self.image_size = image_size
#         self.patch_size = patch_size  # patch size for pooling

#     def forward(self, group_features, group_centers, original_points, nonzero_indices, kernel_size):
#         """
#         Args:
#             group_features: (B, Num_Groups, Dim) - Point-MAE embeddings
#             group_centers: (B, Num_Groups, 3) - Point-MAE group centers
#             original_points: (B, Num_Valid_Points, 3) - preprocessed 3D points
#             nonzero_indices: (Num_Valid_Points,)
#         Returns:
#             feature_map_2d: (B, Dim, H, W)
#         """
#         B, G, Dim = group_features.shape
#         _, N, _ = original_points.shape
        
#         # 1. Dimension manipulation
#         xyz_target = original_points.permute(0, 2, 1).contiguous()
#         xyz_source = group_centers.permute(0, 2, 1).contiguous()
#         feat_source = group_features.permute(0, 2, 1).contiguous()

#         # 2. Interpolation (Feature Propagation)
#         interpolated_feats = interpolating_points(xyz_target, xyz_source, feat_source)

#         # 3. Projection to 2D Grid
#         full_map_flat = torch.zeros(
#             (B, Dim, self.image_size * self.image_size), 
#             dtype=group_features.dtype, 
#             device=group_features.device
#         )

#         full_map_flat[:, :, nonzero_indices] = interpolated_feats

#         # 4. Reshape to (B, Dim, H, W)
#         feature_map_2d = full_map_flat.view(B, Dim, self.image_size, self.image_size)

#         # 5. patch pooling (to [img_size // patch_size, img_size // patch_size])
#         ps = self.patch_size
#         patch_feats = F.avg_pool2d(feature_map_2d, kernel_size=kernel_size, stride=kernel_size) # 28x28

#         return patch_feats

import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils 

def interpolating_points(xyz1, xyz2, points2):
    """
    Args:
        xyz1: (B, 3, N) - Target Points Coordinates
        xyz2: (B, 3, S) - Source Points Coordinates
        points2: (B, Dim, S) - Source Features (Point-MAE features)
    Returns:
        interpolated_points: (B, Dim, N) - interpolated features
    """
    xyz1 = xyz1.permute(0, 2, 1).contiguous() # (B, N, 3)
    xyz2 = xyz2.permute(0, 2, 1).contiguous() # (B, S, 3)
    points2 = points2.contiguous()

    B, N, _ = xyz1.shape
    _, S, _ = xyz2.shape

    dist, idx = pointnet2_utils.three_nn(xyz1, xyz2)
    dist = torch.clamp(dist, min=1e-10)
    norm = torch.sum(1.0 / dist, dim=2, keepdim=True)
    weights = (1.0 / dist) / norm # (B, N, 3)
    
    # 보간 수행 (Feature: [B, Dim, S], Idx: [B, N, 3], Weight: [B, N, 3])
    interpolated_points = pointnet2_utils.three_interpolate(points2, idx, weights)
    
    return interpolated_points

class PointFeatureAlignment(nn.Module):
    def __init__(self, image_size=224, patch_size=8):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size

    def forward(self, group_features, group_centers, original_points, nonzero_indices, kernel_size):
        """
        Args:
            group_features: (B, Num_Groups, Dim)
            group_centers: (B, Num_Groups, 3)
            original_points: (B, Num_Valid_Points, 3)
            nonzero_indices: (Num_Valid_Points,)
        Returns:
            patch_feats: (B, Dim, H/ps, W/ps)
        """
        B, G, Dim = group_features.shape
        # original_points: (B, N, 3)
        
        # 1. Dimension manipulation
        xyz_target = original_points.permute(0, 2, 1).contiguous() 
        xyz_source = group_centers.permute(0, 2, 1).contiguous() 
        feat_source = group_features.permute(0, 2, 1).contiguous() 

        # 2. Interpolation
        interpolated_feats = interpolating_points(xyz_target, xyz_source, feat_source)

        # 3. Projection to 2D Grid
        full_map_flat = torch.zeros(
            (B, Dim, self.image_size * self.image_size), 
            dtype=group_features.dtype, 
            device=group_features.device
        )
        
        full_map_flat[:, :, nonzero_indices] = interpolated_feats

        # 4. Reshape to 2D Image (B, Dim, H, W)
        feature_map_2d = full_map_flat.view(B, Dim, self.image_size, self.image_size)

        # 5. Pooling
        patch_feats = F.avg_pool2d(feature_map_2d, kernel_size=kernel_size, stride=kernel_size)

        return patch_feats