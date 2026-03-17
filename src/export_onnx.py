"""
Export trained model to ONNX format for deployment.
"""

import argparse
import torch
from pathlib import Path

from utils import load_config, get_device
from models.proposed_lightweight import get_proposed_model, proposed_model_kwargs_from_config
from models.baselines import get_baseline_model


def export_to_onnx(
    config: dict,
    checkpoint_path: str,
    output_path: str = None,
    input_size: int = 224
):
    """Export model to ONNX format."""
    device = get_device()
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_names = checkpoint.get('class_names', config['model']['classes'])
    num_classes = len(class_names)
    
    # Determine model type from checkpoint or path
    model_name = checkpoint.get('model_name', None)
    if model_name is None:
        # Infer from path
        ckpt_path_lower = checkpoint_path.lower()
        if 'resnet' in ckpt_path_lower:
            model_name = 'resnet18'
        elif 'mobilenetv3_large' in ckpt_path_lower:
            model_name = 'mobilenetv3_large'
        elif 'mobilenetv3_small' in ckpt_path_lower:
            model_name = 'mobilenetv3_small'
        elif 'mobilenet' in ckpt_path_lower:
            model_name = 'mobilenetv2'
        elif 'efficientnet' in ckpt_path_lower:
            model_name = 'efficientnet_b0'
        elif 'ghostnet' in ckpt_path_lower or 'ghost_net' in ckpt_path_lower:
            model_name = 'ghostnet'
        elif 'shufflenet' in ckpt_path_lower:
            model_name = 'shufflenetv2'
        else:
            model_name = 'proposed'
    
    # Create model (proposed: use full architecture from checkpoint/config)
    if model_name == 'proposed':
        model_config = checkpoint.get('config', config)
        kwargs = proposed_model_kwargs_from_config(model_config)
        kwargs['num_classes'] = num_classes
        input_size = kwargs.get('input_size', input_size)
        model = get_proposed_model(**kwargs)
    else:
        model = get_baseline_model(model_name, num_classes=num_classes, pretrained=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(device)
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size, input_size).to(device)
    
    # Set output path
    if output_path is None:
        output_path = Path(checkpoint_path).parent / 'model.onnx'
    else:
        output_path = Path(output_path)
    
    # Export to ONNX
    print(f"Exporting {model_name} model to ONNX...")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output path: {output_path}")
    
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        opset_version=11,
        do_constant_folding=True,
        export_params=True
    )
    
    print(f"Model successfully exported to {output_path}")
    
    # Verify ONNX model
    try:
        import onnx
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification: PASSED")
    except ImportError:
        print("ONNX not installed, skipping verification")
    except Exception as e:
        print(f"ONNX model verification: FAILED - {e}")
    
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description='Export model to ONNX')
    parser.add_argument('--config', type=str, default='src/config.yaml',
                       help='Path to config file')
    parser.add_argument('--ckpt', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default=None,
                       help='Output ONNX file path (default: same dir as checkpoint)')
    parser.add_argument('--input_size', type=int, default=224,
                       help='Input image size')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    export_to_onnx(config, args.ckpt, args.output, args.input_size)


if __name__ == '__main__':
    main()

