import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNAFBlock(nn.Module):
    """
    Lightweight NAF-like block (placeholder).
    Replace with your full NAFNet blocks for best performance.
    """
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.pwconv1 = nn.Conv2d(dim, dim * 2, 1)
        self.pwconv2 = nn.Conv2d(dim, dim, 1)
        self.norm2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.dwconv(x)
        x = F.gelu(self.pwconv1(x))
        x = self.pwconv2(x)
        x = x + shortcut
        x = self.norm2(x)
        return x

class SimpleNAFNet(nn.Module):
    """
    Artifact Rectification Module (ARM).
    """
    def __init__(self, dim=48, num_blocks=4):
        super().__init__()
        self.head = nn.Conv2d(3, dim, 3, padding=1)
        self.body = nn.Sequential(*[SimpleNAFBlock(dim) for _ in range(num_blocks)])
        self.tail = nn.Conv2d(dim, 3, 3, padding=1)

    def forward(self, x):
        feat = self.head(x)
        feat = self.body(feat)
        out = self.tail(feat)
        return out