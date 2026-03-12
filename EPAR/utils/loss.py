import torch
import torch.nn as nn

def chamfer_distance(p1, p2):
    B, N, D = p1.shape
    B, M, D = p2.shape
    # p1: (B, N, D) -> (B, N, 1, D)
    # p2: (B, M, D) -> (B, 1, M, D)
    p1_expanded = p1.unsqueeze(2)
    p2_expanded = p2.unsqueeze(1)
    dist_matrix = torch.sum((p1_expanded - p2_expanded) ** 2, dim=-1)
    min_dists_1, _ = torch.min(dist_matrix, dim=2)
    min_dists_2, _ = torch.min(dist_matrix, dim=1)
    dist1 = torch.mean(min_dists_1, dim=1)
    dist2 = torch.mean(min_dists_2, dim=1)
    chamfer_dist = dist1 + dist2
    return chamfer_dist.mean()

class CauchyLoss(nn.Module):
    def __init__(self, reduction: str = "mean", sigma: float = 0.5):
        super().__init__()
        self.reduction = reduction
        self.sigma = sigma # Scale parameter for the Cauchy distribution

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Calculate the difference between input and target
        nl, patches, dim = input.shape
        input = input.permute(1,0,2).contiguous().view(patches, nl*dim)
        target = target.permute(1,0,2).contiguous().view(patches, nl*dim)
        diff = input - target
        
        # Calculate the Cauchy loss
        # The formula for Cauchy loss is often derived from the negative log-likelihood
        # of a Cauchy distribution. A common form involves log(1 + (diff/sigma)^2)
        loss = torch.log(1 + (diff / self.sigma)**2)

        if self.reduction == "mean":
            return torch.mean(self.sigma**2 * loss)
        elif self.reduction == "sum":
            return torch.sum(self.sigma**2*loss)
        else: # reduction == "none"
            return loss