import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
import random

def load_image(path, size=None):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if size is not None:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img

class PairedLowLightDataset(Dataset):
    """
    For synthetic paired data: (Ip -> N)
    root/
      low/*.png
      gt/*.png
    """
    def __init__(self, root, size=None):
        self.root = Path(root)
        self.lows = sorted((self.root / "low").glob("*"))
        self.gts = sorted((self.root / "gt").glob("*"))
        self.size = size
        assert len(self.lows) == len(self.gts), "Mismatched paired data."

    def __len__(self):
        return len(self.lows)

    def __getitem__(self, idx):
        low = load_image(self.lows[idx], self.size)
        gt = load_image(self.gts[idx], self.size)
        low = torch.from_numpy(low).permute(2, 0, 1)
        gt = torch.from_numpy(gt).permute(2, 0, 1)
        return low, gt

class UnpairedLowLightDataset(Dataset):
    """
    For real unpaired low-light data: Ir
    root/
      real/*.png
    """
    def __init__(self, root, size=None):
        self.root = Path(root)
        self.reals = sorted((self.root / "real").glob("*"))
        self.size = size

    def __len__(self):
        return len(self.reals)

    def __getitem__(self, idx):
        img = load_image(self.reals[idx], self.size)
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img