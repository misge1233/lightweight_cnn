"""
Quantization script for converting models to INT8.
Supports PyTorch static quantization and ONNX quantization.
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from utils import load_config, get_device, get_model_size_mb, count_parameters, measure_inference_time
from models.proposed_lightweight import get_proposed_model, proposed_model_kwargs_from_config
from train import get_data_transforms


def quantize_pytorch_static(
    model: nn.Module,
    train_loader: DataLoader,
    config: dict,
    device: torch.device
):
    """
    Apply PyTorch static quantization.
    Note: This works best with standard torchvision models.
    For custom models, ONNX quantization may be more reliable.
    """
    model.eval()
    model = model.to('cpu')  # Quantization typically done on CPU
    
    # Set quantization config
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    # Prepare model for quantization
    try:
        torch.quantization.prepare(model, inplace=True)
        
        # Calibrate with subset of training data
        calibration_samples = int(len(train_loader.dataset) * config['quantization']['calibration_samples'])
        print(f"Calibrating with {calibration_samples} samples...")
        
        count = 0
        with torch.no_grad():
            for inputs, _ in train_loader:
                if count >= calibration_samples:
                    break
                _ = model(inputs)
                count += inputs.size(0)
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(model, inplace=False)
        
        print("PyTorch static quantization complete")
        return quantized_model
    
    except Exception as e:
        print(f"PyTorch quantization failed: {e}")
        print("Falling back to ONNX quantization...")
        return None


def quantize_onnx(
    model: nn.Module,
    checkpoint_path: str,
    config: dict,
    device: torch.device,
    experiment_dir: str
):
    """
    Export to ONNX and quantize using onnxruntime.
    This is more reliable for custom architectures.
    """
    try:
        import onnx
        import onnxruntime as ort
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("ONNX/ONNXRuntime not installed. Install with:")
        print("  pip install onnx onnxruntime")
        return None
    
    model.eval()
    model = model.to(device)
    
    # Export to ONNX
    onnx_path = Path(checkpoint_path).parent / 'model.onnx'
    dummy_input = torch.randn(1, 3, config['model']['input_size'], config['model']['input_size']).to(device)
    
    print("Exporting model to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=11
    )
    print(f"ONNX model saved to {onnx_path}")
    
    # Quantize ONNX model
    quantized_onnx_path = Path(checkpoint_path).parent / 'model_quantized.onnx'
    
    print("Quantizing ONNX model...")
    try:
        # Dynamic quantization (simpler, no calibration needed)
        quantize_dynamic(
            str(onnx_path),
            str(quantized_onnx_path),
            weight_type=QuantType.QUInt8
        )
        print(f"Quantized ONNX model saved to {quantized_onnx_path}")
        return quantized_onnx_path
    except Exception as e:
        print(f"ONNX quantization failed: {e}")
        return None


def evaluate_quantized_model(
    model_path: str,
    test_loader: DataLoader,
    config: dict,
    device: torch.device,
    is_onnx: bool = True
):
    """Evaluate quantized model. Returns (accuracy, macro_f1) or (None, None)."""
    if is_onnx:
        try:
            import onnxruntime as ort
        except ImportError:
            print("ONNXRuntime not available for evaluation")
            return None, None
        
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        
        all_preds, all_labels = [], []
        for inputs, labels in test_loader:
            inputs_np = inputs.numpy()
            outputs = session.run(None, {input_name: inputs_np})
            predictions = np.argmax(outputs[0], axis=1)
            all_preds.extend(predictions.tolist())
            all_labels.extend(labels.numpy().tolist())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        accuracy = 100.0 * (all_preds == all_labels).mean()
        _, _, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
        return accuracy, float(f1) * 100
    else:
        model = torch.jit.load(model_path)
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy().tolist())
                all_labels.extend(labels.numpy().tolist())
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        accuracy = 100.0 * (all_preds == all_labels).mean()
        _, _, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
        return accuracy, float(f1) * 100


def quantize_model(config: dict, checkpoint_path: str, experiment_dir: str):
    """Main quantization function. Proposed model; uses full arch params from checkpoint config."""
    checkpoint_path = str(checkpoint_path)
    if checkpoint_path.lower().endswith('.onnx'):
        print("Error: quantize.py expects a PyTorch checkpoint (.pth), not an ONNX file.")
        print("  Pass the checkpoint used to create the ONNX, e.g.:")
        print("    --ckpt experiment/runs/<run>/best_model.pth")
        print("    --ckpt experiment/runs/pruning_sweep_xxx/ratio_0.2/pruned_model.pth")
        print("  This script loads the .pth, then exports and quantizes to produce model_quantized.onnx.")
        return None
    device = get_device()
    
    # Load model (use checkpoint config so architecture matches trained model)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint.get('config', config)
    class_names = checkpoint.get('class_names', config['model']['classes'])
    num_classes = len(class_names)
    kwargs = proposed_model_kwargs_from_config(model_config)
    kwargs['num_classes'] = num_classes
    model = get_proposed_model(**kwargs)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    fp32_size_mb = get_model_size_mb(model, 'fp32')
    print(f"Original model:")
    print(f"  Parameters: {count_parameters(model):,}")
    print(f"  Size (FP32): {fp32_size_mb:.2f} MB")
    
    # Get data loaders for calibration
    train_transform = get_data_transforms(config, 'train')
    test_transform = get_data_transforms(config, 'test')
    
    train_dataset = ImageFolder(str(Path(experiment_dir) / 'train'), transform=train_transform)
    test_dataset = ImageFolder(str(Path(experiment_dir) / 'test'), transform=test_transform)
    
    # Use smaller batch for calibration
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=2
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )
    
    # FP32 baseline metrics (for summary table)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            out = model(inputs)
            _, pred = out.max(1)
            all_preds.extend(pred.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    fp32_acc = 100.0 * (all_preds == all_labels).mean()
    _, _, fp32_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    fp32_f1 = float(fp32_f1) * 100
    input_shape = (3, config['model']['input_size'], config['model']['input_size'])
    fp32_latency_ms, _ = measure_inference_time(model, input_shape, device, warmup=5, runs=50, batch_size=1)
    
    # Try ONNX quantization first (more reliable for custom models)
    print("\nAttempting ONNX quantization...")
    quantized_path = quantize_onnx(model, checkpoint_path, config, device, experiment_dir)
    
    output_dir = Path(checkpoint_path).parent
    summary_rows = [
        {'variant': 'FP32', 'accuracy': round(fp32_acc, 2), 'macro_f1': round(fp32_f1, 2),
         'size_mb': round(fp32_size_mb, 2), 'latency_ms': round(fp32_latency_ms, 2)}
    ]
    
    if quantized_path:
        # Evaluate quantized model
        print("\nEvaluating quantized model...")
        quantized_acc, quantized_f1 = evaluate_quantized_model(
            str(quantized_path), test_loader, config, device, is_onnx=True
        )
        
        if quantized_acc is not None:
            print(f"Quantized model accuracy: {quantized_acc:.2f}%, macro_f1: {quantized_f1:.2f}%")
            original_size = get_model_size_mb(model, 'fp32')
            quantized_size_mb = original_size * 0.25  # INT8 ~4x smaller
            
            print(f"\nQuantization results:")
            print(f"  Original size (FP32): {original_size:.2f} MB")
            print(f"  Quantized size (INT8): ~{quantized_size_mb:.2f} MB")
            print(f"  Size reduction: ~{(1 - quantized_size_mb/original_size)*100:.1f}%")
            
            summary_rows.append({
                'variant': 'INT8',
                'accuracy': round(quantized_acc, 2),
                'macro_f1': round(quantized_f1, 2) if quantized_f1 is not None else '',
                'size_mb': round(quantized_size_mb, 2),
                'latency_ms': ''  # Android/ONNX Runtime timing can be filled separately
            })
            
            import json
            quant_info = {
                'quantized_model_path': str(quantized_path),
                'original_size_mb': float(original_size),
                'quantized_size_mb': float(quantized_size_mb),
                'accuracy': float(quantized_acc),
                'macro_f1': float(quantized_f1) if quantized_f1 is not None else None,
                'quantization_method': 'ONNX dynamic'
            }
            with open(output_dir / 'quantization_info.json', 'w') as f:
                json.dump(quant_info, f, indent=2)
            print(f"\nQuantization info saved to {output_dir / 'quantization_info.json'}")
    
    # Save summary CSV (FP32 vs INT8 for paper table)
    pd.DataFrame(summary_rows).to_csv(output_dir / 'quantization_summary.csv', index=False)
    print(f"Quantization summary saved to {output_dir / 'quantization_summary.csv'}")
    
    print("\nQuantization complete (or skipped if dependencies missing)")
    return quantized_path if quantized_path else None


def main():
    parser = argparse.ArgumentParser(description='Quantize trained model to INT8')
    parser.add_argument('--config', type=str, default='src/config.yaml',
                       help='Path to config file')
    parser.add_argument('--ckpt', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--experiment_dir', type=str, default='experiment',
                       help='Directory containing data splits')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    quantize_model(config, args.ckpt, args.experiment_dir)


if __name__ == '__main__':
    main()

