"""
Data augmentation utilities: MixUp, CutMix, RandAugment, RandomErasing.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from PIL import Image


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Apply MixUp augmentation.
    
    Args:
        x: Input batch of images [B, C, H, W]
        y: Input batch of labels [B]
        alpha: MixUp parameter (beta distribution parameter)
    
    Returns:
        mixed_x: Mixed images
        y_a: Labels for first image
        y_b: Labels for second image
        lam: Mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Apply CutMix augmentation.
    
    Args:
        x: Input batch of images [B, C, H, W]
        y: Input batch of labels [B]
        alpha: CutMix parameter (beta distribution parameter)
    
    Returns:
        mixed_x: Mixed images
        y_a: Labels for first image
        y_b: Labels for second image
        lam: Mixing coefficient (adjusted for area)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    y_a, y_b = y, y[index]
    
    # Generate random bounding box
    W, H = x.size(3), x.size(2)
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    # Random center
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    # Clamp bounding box
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    # Apply CutMix
    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
    # Adjust lambda to match actual pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    return x, y_a, y_b, lam


def mixup_criterion(criterion, pred: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float) -> torch.Tensor:
    """Compute loss for MixUp/CutMix."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class RandAugment:
    """
    RandAugment: Practical automated data augmentation with a reduced search space.
    Based on: https://arxiv.org/abs/1909.13719
    """
    
    def __init__(self, n: int = 2, m: int = 9):
        """
        Args:
            n: Number of augmentation transformations to apply
            m: Magnitude of augmentation (0-10)
        """
        self.n = n
        self.m = m
        self.augment_list = [
            self._identity,
            self._auto_contrast,
            self._equalize,
            self._rotate,
            self._posterize,
            self._solarize,
            self._color,
            self._contrast,
            self._brightness,
            self._sharpness,
            self._shear_x,
            self._shear_y,
            self._translate_x,
            self._translate_y,
        ]
    
    def __call__(self, img):
        ops = np.random.choice(len(self.augment_list), self.n, replace=False)
        for op_idx in ops:
            img = self.augment_list[op_idx](img, self.m)
        return img
    
    def _identity(self, img, m):
        return img
    
    def _auto_contrast(self, img, m):
        return F.autocontrast(img)
    
    def _equalize(self, img, m):
        return F.equalize(img)
    
    def _rotate(self, img, m):
        angle = (m / 10.0) * 30.0  # Max 30 degrees
        if np.random.random() < 0.5:
            angle = -angle
        return F.rotate(img, angle)
    
    def _posterize(self, img, m):
        bits = max(1, int(8 - (m / 10.0) * 4))  # 4-8 bits
        return F.posterize(img, bits)
    
    def _solarize(self, img, m):
        threshold = int(256 - (m / 10.0) * 128)  # 128-256
        return F.solarize(img, threshold)
    
    def _color(self, img, m):
        factor = 1.0 + (m / 10.0) * 0.4  # 1.0-1.4
        if np.random.random() < 0.5:
            factor = 1.0 / factor
        return F.adjust_saturation(img, factor)
    
    def _contrast(self, img, m):
        factor = 1.0 + (m / 10.0) * 0.4  # 1.0-1.4
        if np.random.random() < 0.5:
            factor = 1.0 / factor
        return F.adjust_contrast(img, factor)
    
    def _brightness(self, img, m):
        factor = 1.0 + (m / 10.0) * 0.4  # 1.0-1.4
        if np.random.random() < 0.5:
            factor = 1.0 / factor
        return F.adjust_brightness(img, factor)
    
    def _sharpness(self, img, m):
        factor = 1.0 + (m / 10.0) * 0.4  # 1.0-1.4
        if np.random.random() < 0.5:
            factor = 1.0 / factor
        return F.adjust_sharpness(img, factor)
    
    def _shear_x(self, img, m):
        angle = (m / 10.0) * 0.3  # Max 0.3 radians
        if np.random.random() < 0.5:
            angle = -angle
        return F.affine(img, angle=0, translate=[0, 0], scale=1.0, shear=[angle * 180 / np.pi, 0])
    
    def _shear_y(self, img, m):
        angle = (m / 10.0) * 0.3  # Max 0.3 radians
        if np.random.random() < 0.5:
            angle = -angle
        return F.affine(img, angle=0, translate=[0, 0], scale=1.0, shear=[0, angle * 180 / np.pi])
    
    def _translate_x(self, img, m):
        max_dx = (m / 10.0) * 0.3  # Max 30% translation
        dx = (np.random.random() * 2 - 1) * max_dx
        return F.affine(img, angle=0, translate=[dx * img.size[0], 0], scale=1.0, shear=[0, 0])
    
    def _translate_y(self, img, m):
        max_dy = (m / 10.0) * 0.3  # Max 30% translation
        dy = (np.random.random() * 2 - 1) * max_dy
        return F.affine(img, angle=0, translate=[0, dy * img.size[1]], scale=1.0, shear=[0, 0])


class RandomErasing:
    """
    Random Erasing data augmentation.
    Based on: https://arxiv.org/abs/1708.04896
    Works with PIL Images (before ToTensor) or Tensors (after ToTensor).
    """
    
    def __init__(self, p: float = 0.5, scale: Tuple[float, float] = (0.02, 0.33), ratio: Tuple[float, float] = (0.3, 3.3)):
        """
        Args:
            p: Probability of applying random erasing
            scale: Range of area to erase
            ratio: Range of aspect ratio
        """
        self.p = p
        self.scale = scale
        self.ratio = ratio
    
    def __call__(self, img):
        """
        Args:
            img: PIL Image or Tensor [C, H, W]
        
        Returns:
            img: Augmented image
        """
        if np.random.random() > self.p:
            return img
        
        if isinstance(img, torch.Tensor):
            # Tensor format [C, H, W]
            C, H, W = img.shape
            area = H * W
            
            for _ in range(100):  # Try up to 100 times
                erase_area = np.random.uniform(self.scale[0], self.scale[1]) * area
                aspect_ratio = np.random.uniform(self.ratio[0], self.ratio[1])
                
                h = int(np.sqrt(erase_area * aspect_ratio))
                w = int(np.sqrt(erase_area / aspect_ratio))
                
                if h < H and w < W:
                    x1 = np.random.randint(0, W - w)
                    y1 = np.random.randint(0, H - h)
                    
                    # Random value for erasing (normalized 0-1)
                    if C == 3:
                        erase_value = torch.rand(3, 1, 1)
                    else:
                        erase_value = torch.rand(1, 1, 1)
                    
                    img[:, y1:y1+h, x1:x1+w] = erase_value
                    break
        else:
            # PIL Image format
            W, H = img.size
            area = H * W
            
            for _ in range(100):  # Try up to 100 times
                erase_area = np.random.uniform(self.scale[0], self.scale[1]) * area
                aspect_ratio = np.random.uniform(self.ratio[0], self.ratio[1])
                
                h = int(np.sqrt(erase_area * aspect_ratio))
                w = int(np.sqrt(erase_area / aspect_ratio))
                
                if h < H and w < W:
                    x1 = np.random.randint(0, W - w)
                    y1 = np.random.randint(0, H - h)
                    
                    # Random RGB value for erasing
                    erase_value = tuple(np.random.randint(0, 256, 3))
                    
                    # Create a patch and paste it
                    patch = Image.new('RGB', (w, h), erase_value)
                    img.paste(patch, (x1, y1))
                    break
        
        return img
