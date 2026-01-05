import torch
import torch.nn.functional as F
import random

def stochastic_mask(x, drop_ratio):
    """
    Binary mask M with drop_ratio of pixels set to 0.
    x: (B, C, H, W)
    """
    if drop_ratio <= 0:
        return torch.ones_like(x[:, :1])
    b, c, h, w = x.shape
    mask = torch.ones((b, 1, h, w), device=x.device)
    num_drop = int(drop_ratio * h * w)
    if num_drop > 0:
        idx = torch.randperm(h * w, device=x.device)[:num_drop]
        mask = mask.view(b, 1, -1)
        mask[:, :, idx] = 0
        mask = mask.view(b, 1, h, w)
    return mask

def add_gaussian_noise(x, std):
    if std <= 0:
        return x
    noise = torch.randn_like(x) * std
    return torch.clamp(x + noise, 0.0, 1.0)

def shuffle_residual(residual):
    """
    Shuffle spatially to break structural correlation.
    """
    b, c, h, w = residual.shape
    perm = torch.randperm(h * w, device=residual.device)
    res_flat = residual.view(b, c, -1)
    res_flat = res_flat[:, :, perm]
    return res_flat.view(b, c, h, w)