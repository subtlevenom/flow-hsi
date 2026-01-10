import torch
import torch.nn as nn
from torchmetrics import Metric


class SAM(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("sum_sam", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape, "Preds and target must have the same shape"
        
        B, C, H, W = preds.shape
        preds_flat = preds.permute(0, 2, 3, 1).reshape(B * H * W, C)
        target_flat = target.permute(0, 2, 3, 1).reshape(B * H * W, C)
        
        dot_product = (preds_flat * target_flat).sum(dim=-1)
        preds_norm = torch.norm(preds_flat, dim=-1)
        target_norm = torch.norm(target_flat, dim=-1)
        
        cos_angle = dot_product / (preds_norm * target_norm + 1e-10)
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
        sam_radians = torch.acos(cos_angle)
        sam_degrees = sam_radians * (180.0 / torch.pi)
        
        self.sum_sam += sam_degrees.sum()
        self.total += sam_degrees.numel()

    def compute(self):
        return self.sum_sam / self.total


