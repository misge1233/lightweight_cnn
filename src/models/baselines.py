"""
Baseline models for wheat disease classification.
Uses transfer learning with ImageNet pretrained weights.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class ResNet18(nn.Module):
    """ResNet-18 baseline with transfer learning."""
    
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)
        
        # Replace classifier head
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.model(x)


class MobileNetV2(nn.Module):
    """MobileNetV2 baseline with transfer learning."""
    
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super(MobileNetV2, self).__init__()
        self.model = models.mobilenet_v2(pretrained=pretrained)
        
        # Replace classifier head
        num_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.model(x)


class MobileNetV3Small(nn.Module):
    """MobileNetV3-Small baseline with transfer learning."""
    
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super(MobileNetV3Small, self).__init__()
        try:
            # Try torchvision's MobileNetV3 (available in newer versions)
            self.model = models.mobilenet_v3_small(pretrained=pretrained)
            # Replace classifier head - MobileNetV3 has different structure
            # The classifier is a Sequential with [Linear, Hardswish, Dropout, Linear]
            num_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(num_features, num_classes)
        except (AttributeError, IndexError, TypeError):
            # Fallback: use timm if available
            try:
                import timm
                self.model = timm.create_model('mobilenetv3_small_100', pretrained=pretrained, num_classes=num_classes)
            except ImportError:
                raise ImportError(
                    "MobileNetV3 requires either torchvision >= 0.9 or timm. "
                    "Install with: pip install timm"
                )
    
    def forward(self, x):
        return self.model(x)


class MobileNetV3Large(nn.Module):
    """MobileNetV3-Large baseline with transfer learning."""
    
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super(MobileNetV3Large, self).__init__()
        try:
            # Try torchvision's MobileNetV3 (available in newer versions)
            self.model = models.mobilenet_v3_large(pretrained=pretrained)
            # Replace classifier head - MobileNetV3 has different structure
            # The classifier is a Sequential with [Linear, Hardswish, Dropout, Linear]
            num_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(num_features, num_classes)
        except (AttributeError, IndexError, TypeError):
            # Fallback: use timm if available
            try:
                import timm
                self.model = timm.create_model('mobilenetv3_large_100', pretrained=pretrained, num_classes=num_classes)
            except ImportError:
                raise ImportError(
                    "MobileNetV3 requires either torchvision >= 0.9 or timm. "
                    "Install with: pip install timm"
                )
    
    def forward(self, x):
        return self.model(x)


class EfficientNetB0(nn.Module):
    """EfficientNet-B0 baseline with transfer learning."""
    
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super(EfficientNetB0, self).__init__()
        try:
            # Try torchvision's EfficientNet (available in newer versions)
            self.model = models.efficientnet_b0(pretrained=pretrained)
            num_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_features, num_classes)
        except AttributeError:
            # Fallback: use timm if available, or create a simple alternative
            try:
                import timm
                self.model = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=num_classes)
            except ImportError:
                raise ImportError(
                    "EfficientNet requires either torchvision >= 0.13 or timm. "
                    "Install with: pip install timm"
                )
    
    def forward(self, x):
        return self.model(x)


class ShuffleNetV2(nn.Module):
    """ShuffleNetV2 baseline (x1.0) with transfer learning. Prefers torchvision."""
    
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super(ShuffleNetV2, self).__init__()
        try:
            # torchvision: shufflenet_v2_x1_0
            self.model = models.shufflenet_v2_x1_0(pretrained=pretrained)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, num_classes)
        except (AttributeError, TypeError):
            try:
                import timm
                self.model = timm.create_model('shufflenetv2_x1_0', pretrained=pretrained, num_classes=num_classes)
            except ImportError:
                raise ImportError(
                    "ShuffleNetV2 requires either torchvision or timm. "
                    "Install with: pip install timm"
                )
    
    def forward(self, x):
        return self.model(x)


class GhostNet(nn.Module):
    """GhostNet baseline via timm (ghostnet_100). Replaces classifier for num_classes."""
    
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super(GhostNet, self).__init__()
        try:
            import timm
            # timm: ghostnet_100; num_classes replaces head
            self.model = timm.create_model(
                "ghostnet_100",
                pretrained=pretrained,
                num_classes=num_classes
            )
        except ImportError:
            raise ImportError(
                "GhostNet requires timm. Install with: pip install timm"
            )
    
    def forward(self, x):
        return self.model(x)


def get_baseline_model(model_name: str, num_classes: int = 5, pretrained: bool = True) -> nn.Module:
    """Factory function to get baseline models."""
    model_name = model_name.lower()
    
    if model_name == 'resnet18':
        return ResNet18(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'mobilenetv2':
        return MobileNetV2(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'mobilenetv3_small' or model_name == 'mobilenetv3small':
        return MobileNetV3Small(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'mobilenetv3_large' or model_name == 'mobilenetv3large':
        return MobileNetV3Large(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'efficientnet_b0' or model_name == 'efficientnetb0':
        return EfficientNetB0(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'ghostnet' or model_name == 'ghost_net':
        return GhostNet(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'shufflenetv2' or model_name == 'shufflenet_v2':
        return ShuffleNetV2(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown baseline model: {model_name}")

