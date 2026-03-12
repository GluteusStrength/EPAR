import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import matplotlib.pyplot as plt
from PIL import Image
from models.pointnet2_utils import interpolating_points
from sklearn.decomposition import PCA


class PointFeatureAlignment(nn.Module):
    def __init__(self, patch_size=8):
        super().__init__()
        self.patch_size = patch_size  # patch size for pooling

    def load_camera_parameters(self, param_file, target_size=(224, 224)):
        """
        Load and convert MVTec camera parameters to intrinsic matrix scaled to target image size
        (Camera parameter: Intrinsic informations)
        """
        with open(param_file, 'r') as f:
            params = json.load(f)
        orig_w = params["image_width"]
        orig_h = params["image_height"]
        tx, ty = target_size
        sx = params['sx']
        sy = params['sy']
        focus = params['focus']
        # scale factors
        scale_x = tx / orig_w
        scale_y = ty / orig_h
        # focal lengths in pixels
        fx = (focus / sx) * scale_x
        fy = (focus / sy) * scale_y
        cx = params['cx'] * scale_x
        cy = params['cy'] * scale_y

        K = torch.zeros(3, 3)
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy
        K[2, 2] = 1.0
        return K

    def project_to_image(self, points, K):
        """
        Project 3D points to 2D pixel coordinates
        Args:
            points: [B, N, 3]
            K: [B, 3, 3]
        Returns:
            uv: [B, N, 2] pixel coords
        """
        B, N, _ = points.shape
        # to shape [B, 3, N]
        pts = points.permute(0, 2, 1)
        # project: [B,3,3] x [B,3,N] -> [B,3,N]
        uvz = torch.bmm(K, pts)
        uvz = uvz.permute(0, 2, 1)  # [B,N,3]
        # normalize
        uv = uvz[:, :, :2] / (uvz[:, :, 2:].clamp(min=1e-8))
        return uv

    def forward(self, xyz_feats, points, centers, camera_param_file, image_size=(224,224), kernel_size=4):
        """
        Args:
            xyz_feats: [B, C, M] features per FPS center
            points: [B, N, 3] original points
            centers: [B, M, 3] FPS centers
            camera_param_file: path to camera JSON
            image_size: tuple (H, W)
        Returns:
            patch_feats: [B, C, H/ps, W/ps]
        """
        B, C, M = xyz_feats.shape
        H, W = image_size
        device = points.device
        # 1. load and batch intrinsics
        K0 = self.load_camera_parameters(camera_param_file, target_size=image_size)
        K = K0.unsqueeze(0).repeat(B, 1, 1).to(device)
        # 2. interpolate features
        interp = interpolating_points(points.permute(0,2,1), centers.permute(0,2,1), xyz_feats).permute(0,2,1)
        # 3. project points (3D -> 2D)
        uv = self.project_to_image(points, K)
        u = uv[:, :, 0].floor().clamp(0, W-1).long()
        v = uv[:, :, 1].floor().clamp(0, H-1).long()
        # # 4. scatter to 2D map
        idx_flat = u + v * W 

        feat_map = torch.zeros(B, C, H*W, device=device)   
        count    = torch.zeros(B, 1, H*W, device=device)   
        src_feat = interp.permute(0, 2, 1)
        idx_feat = idx_flat.unsqueeze(1).expand(-1, C, -1)
        feat_map.scatter_add_(2, idx_feat, src_feat)

        ones = torch.ones_like(idx_flat, dtype=feat_map.dtype)  # [B, N]
        idx_cnt = idx_flat.unsqueeze(1)                            # [B, 1, N]
        count.scatter_add_(2, idx_cnt, ones.unsqueeze(1))

        feat_map = feat_map.view(B, C, H, W)
        count = count.view(B, 1, H, W).clamp(min=1)
        feat_map = feat_map / count

        # 5. patch pooling (to [img_size // patch_size, img_size // patch_size])
        ps = self.patch_size
        patch_feats = F.avg_pool2d(feat_map, kernel_size=kernel_size, stride=kernel_size) # 28x28
        
        return patch_feats