"""
Evaluation script for trained models.
Computes comprehensive metrics including efficiency metrics, robustness evaluation,
and uncertainty-aware inference.
"""

import argparse
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import pandas as pd
from PIL import Image as PILImage, ImageFilter
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from utils import (
    load_config, get_device, count_parameters, get_model_size_mb,
    measure_inference_time, save_metrics, set_seed
)
from models.baselines import get_baseline_model
from models.proposed_lightweight import get_proposed_model, proposed_model_kwargs_from_config
from train import get_data_transforms
try:
    from error_analysis import generate_error_report, create_failure_case_figure
    ERROR_ANALYSIS_AVAILABLE = True
except ImportError:
    ERROR_ANALYSIS_AVAILABLE = False
    create_failure_case_figure = None
    print("Warning: error_analysis module not available")


class PerturbationTransform:
    """Apply controlled perturbations to images for robustness evaluation."""
    
    @staticmethod
    def gaussian_noise(image: PILImage.Image, sigma: float = 0.1) -> PILImage.Image:
        """Add Gaussian noise to image."""
        img_array = np.array(image, dtype=np.float32) / 255.0
        noise = np.random.normal(0, sigma, img_array.shape).astype(np.float32)
        img_noisy = np.clip(img_array + noise, 0, 1)
        return PILImage.fromarray((img_noisy * 255).astype(np.uint8))
    
    @staticmethod
    def motion_blur(image: PILImage.Image, kernel_size: int = 9) -> PILImage.Image:
        """Apply motion blur to image."""
        if not CV2_AVAILABLE:
            # Fallback to PIL blur if cv2 not available
            return image.filter(ImageFilter.GaussianBlur(radius=kernel_size/3))
        img_array = np.array(image)
        # Create motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        # Apply blur
        blurred = cv2.filter2D(img_array, -1, kernel)
        return PILImage.fromarray(blurred)
    
    @staticmethod
    def reduce_brightness(image: PILImage.Image, factor: float = 0.7) -> PILImage.Image:
        """Reduce image brightness."""
        img_array = np.array(image, dtype=np.float32)
        img_array = img_array * factor
        return PILImage.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
    
    @staticmethod
    def reduce_contrast(image: PILImage.Image, factor: float = 0.7) -> PILImage.Image:
        """Reduce image contrast."""
        img_array = np.array(image, dtype=np.float32)
        mean = img_array.mean()
        img_array = mean + (img_array - mean) * factor
        return PILImage.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
    
    @staticmethod
    def jpeg_compression(image: PILImage.Image, quality: int = 30) -> PILImage.Image:
        """Apply JPEG compression. quality 1-95 (e.g. 30 for strong compression)."""
        import io
        buf = io.BytesIO()
        image.save(buf, format='JPEG', quality=quality)
        buf.seek(0)
        return PILImage.open(buf).convert(image.mode)
    
    @staticmethod
    def downsampling(image: PILImage.Image, intermediate_size: int = 112) -> PILImage.Image:
        """Downsample then resize back (e.g. to 224). Simulates low-resolution capture."""
        w, h = image.size
        small = image.resize((intermediate_size, intermediate_size), PILImage.BILINEAR)
        return small.resize((w, h), PILImage.BILINEAR)


class PerturbedDataset(torch.utils.data.Dataset):
    """Dataset wrapper that applies perturbations."""
    
    def __init__(self, base_dataset, perturbation_type: str, **kwargs):
        self.base_dataset = base_dataset
        self.perturbation_type = perturbation_type
        
        # Create perturbation function
        if perturbation_type == 'gaussian_noise':
            sigma = kwargs.get('sigma', 0.1)
            self.perturb_fn = lambda img: PerturbationTransform.gaussian_noise(img, sigma)
        elif perturbation_type == 'motion_blur':
            kernel_size = kwargs.get('kernel_size', 9)
            self.perturb_fn = lambda img: PerturbationTransform.motion_blur(img, kernel_size)
        elif perturbation_type == 'brightness':
            factor = kwargs.get('factor', 0.7)
            self.perturb_fn = lambda img: PerturbationTransform.reduce_brightness(img, factor)
        elif perturbation_type == 'contrast':
            factor = kwargs.get('factor', 0.7)
            self.perturb_fn = lambda img: PerturbationTransform.reduce_contrast(img, factor)
        elif perturbation_type == 'jpeg_compression':
            quality = kwargs.get('quality', 30)
            self.perturb_fn = lambda img: PerturbationTransform.jpeg_compression(img, quality)
        elif perturbation_type == 'downsampling':
            intermediate_size = kwargs.get('intermediate_size', 112)
            self.perturb_fn = lambda img: PerturbationTransform.downsampling(img, intermediate_size)
        else:
            raise ValueError(f"Unknown perturbation type: {perturbation_type}")
        
        # Store original transform separately
        self.original_transform = base_dataset.transform
        # Temporarily disable transform on base dataset to get raw images
        self.base_transform_backup = base_dataset.transform
        base_dataset.transform = None  # Disable to get PIL Image
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get image and label - base_dataset.transform is None, so we get PIL Image
        image, label = self.base_dataset[idx]
        
        # Ensure image is PIL Image
        if not isinstance(image, PILImage.Image):
            if isinstance(image, np.ndarray):
                image = PILImage.fromarray(image)
            else:
                image = PILImage.fromarray(np.array(image))
        
        # Apply perturbation
        image = self.perturb_fn(image)
        
        # Apply transform
        if self.original_transform:
            image = self.original_transform(image)
        
        return image, label
    
    def __del__(self):
        # Restore original transform on base dataset
        if hasattr(self, 'base_dataset') and hasattr(self, 'base_transform_backup'):
            self.base_dataset.transform = self.base_transform_backup


def evaluate_robustness(model, test_loader, device, config):
    """Evaluate model robustness under controlled perturbations."""
    perturbations = config.get('robustness', {}).get('perturbations', {
        'gaussian_noise': {'sigma': 0.1},
        'motion_blur': {'kernel_size': 9},
        'brightness': {'factor': 0.7},
        'contrast': {'factor': 0.7},
        'jpeg_compression': {'quality': 30},
        'downsampling': {'intermediate_size': 112}
    })
    
    # First evaluate on clean test set
    clean_accuracy, clean_f1 = evaluate_accuracy_f1(model, test_loader, device)
    
    results = [{
        'perturbation': 'clean',
        'accuracy': float(clean_accuracy),
        'f1_score': float(clean_f1),
        'relative_drop_acc': 0.0,
        'relative_drop_f1': 0.0
    }]
    
    print(f"\n{'='*60}")
    print("Robustness Evaluation")
    print(f"{'='*60}")
    print(f"Clean Accuracy: {clean_accuracy:.2f}%, F1: {clean_f1:.4f}")
    
    # Get original test dataset and create a copy for perturbation
    test_dataset = test_loader.dataset
    
    # Store original transform
    original_transform = test_dataset.transform
    
    # Evaluate under each perturbation
    for pert_type, pert_params in perturbations.items():
        # Reset transform before creating perturbed dataset
        test_dataset.transform = None  # Will be applied in PerturbedDataset
        
        # Create perturbed dataset
        try:
            perturbed_dataset = PerturbedDataset(test_dataset, pert_type, **pert_params)
            perturbed_loader = DataLoader(
                perturbed_dataset,
                batch_size=test_loader.batch_size,
                shuffle=False,
                num_workers=0  # Set to 0 to avoid issues with dataset modification
            )
            
            # Evaluate
            pert_accuracy, pert_f1 = evaluate_accuracy_f1(model, perturbed_loader, device)
            
            # Calculate relative performance drop
            drop_acc = ((clean_accuracy - pert_accuracy) / clean_accuracy) * 100 if clean_accuracy > 0 else 0
            drop_f1 = ((clean_f1 - pert_f1) / clean_f1) * 100 if clean_f1 > 0 else 0
            
            results.append({
                'perturbation': pert_type,
                'accuracy': float(pert_accuracy),
                'f1_score': float(pert_f1),
                'relative_drop_acc': float(drop_acc),
                'relative_drop_f1': float(drop_f1)
            })
            
            print(f"{pert_type:20s}: Acc={pert_accuracy:.2f}%, F1={pert_f1:.4f}, "
                  f"Drop Acc={drop_acc:.2f}%, Drop F1={drop_f1:.2f}%")
        except Exception as e:
            print(f"Warning: Failed to evaluate {pert_type}: {e}")
            results.append({
                'perturbation': pert_type,
                'accuracy': float(clean_accuracy),  # Use clean as fallback
                'f1_score': float(clean_f1),
                'relative_drop_acc': 0.0,
                'relative_drop_f1': 0.0
            })
        finally:
            # Restore original transform
            test_dataset.transform = original_transform
    
    return results


def evaluate_accuracy_f1(model, loader, device):
    """Evaluate accuracy and F1-score on a dataset."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100.0 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    _, _, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    
    return accuracy, f1


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error (ECE). probs: (N, num_classes), labels: (N,) integer."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(np.float64)
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            avg_confidence_in_bin = confidences[in_bin].mean()
            avg_accuracy_in_bin = accuracies[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - avg_accuracy_in_bin) * prop_in_bin
    return float(ece)


def compute_brier_score(probs: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    """Brier score (mean squared error between probs and one-hot labels)."""
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels.astype(int)] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def compute_nll(probs: np.ndarray, labels: np.ndarray) -> float:
    """Negative log-likelihood (cross-entropy) from softmax probs. labels: integer indices."""
    eps = 1e-15
    probs = np.clip(probs, eps, 1 - eps)
    nll = -np.mean(np.log(probs[np.arange(len(labels)), labels.astype(int)]))
    return float(nll)


def evaluate_uncertainty(model, test_loader, device, config):
    """Evaluate uncertainty-aware inference using softmax confidence and Monte Carlo Dropout."""
    uncertainty_config = config.get('uncertainty', {})
    mc_samples = uncertainty_config.get('monte_carlo_samples', 10)
    dropout_rate = uncertainty_config.get('dropout_rate', 0.1)
    
    model.eval()
    
    # Enable dropout for Monte Carlo sampling if model supports it
    enable_mc_dropout(model, dropout_rate)
    
    all_preds = []
    all_labels = []
    all_confidences = []  # Max softmax probability
    all_entropies = []  # Predictive entropy
    all_mc_confidences = []  # MC dropout confidence
    
    print(f"\n{'='*60}")
    print("Uncertainty-Aware Evaluation")
    print(f"{'='*60}")
    print(f"Using Monte Carlo Dropout with {mc_samples} samples")
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Standard inference (softmax confidence)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            confidences, predicted = probs.max(1)
            entropies = -(probs * torch.log(probs + 1e-10)).sum(1)
            
            # Monte Carlo Dropout inference
            mc_probs_list = []
            for _ in range(mc_samples):
                mc_outputs = model(inputs)
                mc_probs = F.softmax(mc_outputs, dim=1)
                mc_probs_list.append(mc_probs)
            
            mc_probs_mean = torch.stack(mc_probs_list).mean(0)
            mc_confidences, _ = mc_probs_mean.max(1)
            mc_entropies = -(mc_probs_mean * torch.log(mc_probs_mean + 1e-10)).sum(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
            all_entropies.extend(entropies.cpu().numpy())
            all_mc_confidences.extend(mc_confidences.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_confidences = np.array(all_confidences)
    all_entropies = np.array(all_entropies)
    all_mc_confidences = np.array(all_mc_confidences)
    
    # Calculate accuracy by confidence threshold
    thresholds = np.arange(0.0, 1.01, 0.05)
    coverage_acc_curve = []
    mc_coverage_acc_curve = []
    
    correct_mask = (all_preds == all_labels)
    
    for threshold in thresholds:
        # Standard confidence
        confident_mask = all_confidences >= threshold
        if confident_mask.sum() > 0:
            coverage = confident_mask.mean() * 100
            accuracy_at_threshold = correct_mask[confident_mask].mean() * 100 if confident_mask.sum() > 0 else 0
            coverage_acc_curve.append({
                'threshold': float(threshold),
                'coverage': float(coverage),
                'accuracy': float(accuracy_at_threshold)
            })
        
        # MC confidence
        mc_confident_mask = all_mc_confidences >= threshold
        if mc_confident_mask.sum() > 0:
            mc_coverage = mc_confident_mask.mean() * 100
            mc_accuracy_at_threshold = correct_mask[mc_confident_mask].mean() * 100 if mc_confident_mask.sum() > 0 else 0
            mc_coverage_acc_curve.append({
                'threshold': float(threshold),
                'coverage': float(mc_coverage),
                'accuracy': float(mc_accuracy_at_threshold)
            })
    
    # Confidence statistics
    correct_confidences = all_confidences[correct_mask]
    incorrect_confidences = all_confidences[~correct_mask]
    
    results = {
        'mean_confidence': float(all_confidences.mean()),
        'mean_confidence_correct': float(correct_confidences.mean()) if len(correct_confidences) > 0 else 0,
        'mean_confidence_incorrect': float(incorrect_confidences.mean()) if len(incorrect_confidences) > 0 else 0,
        'mean_mc_confidence': float(all_mc_confidences.mean()),
        'mean_entropy': float(all_entropies.mean()),
        'coverage_accuracy_curve': coverage_acc_curve,
        'mc_coverage_accuracy_curve': mc_coverage_acc_curve
    }
    
    print(f"Mean confidence (correct): {results['mean_confidence_correct']:.4f}")
    print(f"Mean confidence (incorrect): {results['mean_confidence_incorrect']:.4f}")
    print(f"Mean MC confidence: {results['mean_mc_confidence']:.4f}")
    
    return results


def enable_mc_dropout(model, dropout_rate=0.1):
    """Enable dropout layers for Monte Carlo sampling."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = dropout_rate
            module.train()  # Keep dropout active during eval


def create_model_from_config(model_name: str, num_classes: int, model_config: dict):
    """Helper function to create a model instance. Proposed model uses full arch params from config."""
    if model_name == 'proposed' or model_name is None:
        kwargs = proposed_model_kwargs_from_config(model_config)
        kwargs['num_classes'] = num_classes  # override in case len(class_names) differs
        return get_proposed_model(**kwargs)
    else:
        return get_baseline_model(model_name, num_classes=num_classes, pretrained=False)


def evaluate_model(config: dict, checkpoint_path: str, experiment_dir: str, seed: int = None):
    """Evaluate a trained model. seed sets RNG for reproducibility when provided."""
    if seed is not None:
        set_seed(seed)
    device = get_device()
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        # Fallback for older PyTorch versions (already has weights_only=False)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint.get('config', config)
    class_names = checkpoint.get('class_names', config['model']['classes'])
    
    # Create model - improved model name inference
    model_name = checkpoint.get('model_name', 'proposed')
    num_classes = len(class_names)
    
    # Better model name inference
    if model_name == 'proposed' or 'proposed' in checkpoint_path.lower():
        kwargs = proposed_model_kwargs_from_config(model_config)
        kwargs['num_classes'] = num_classes
        model = get_proposed_model(**kwargs)
    else:
        # Infer model name from checkpoint path or name
        ckpt_path_lower = checkpoint_path.lower()
        if 'resnet' in ckpt_path_lower or 'resnet' in model_name.lower():
            model_name = 'resnet18'
        elif 'mobilenetv3_large' in ckpt_path_lower or 'mobilenetv3_large' in model_name.lower():
            model_name = 'mobilenetv3_large'
        elif 'mobilenetv3_small' in ckpt_path_lower or 'mobilenetv3_small' in model_name.lower():
            model_name = 'mobilenetv3_small'
        elif 'mobilenetv2' in ckpt_path_lower or 'mobilenetv2' in model_name.lower():
            model_name = 'mobilenetv2'
        elif 'efficientnet' in ckpt_path_lower or 'efficientnet' in model_name.lower():
            model_name = 'efficientnet_b0'
        elif 'ghostnet' in ckpt_path_lower or 'ghost_net' in ckpt_path_lower or 'ghostnet' in (model_name or '').lower():
            model_name = 'ghostnet'
        elif 'shufflenet' in ckpt_path_lower or 'shufflenet' in (model_name or '').lower():
            model_name = 'shufflenetv2'
        else:
            model_name = 'resnet18'  # default
        
        model = get_baseline_model(model_name, num_classes=num_classes, pretrained=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model: {model_name}")
    print(f"Classes: {class_names}")
    
    # Get test data loader
    test_transform = get_data_transforms(config, 'test')
    test_dataset = ImageFolder(str(Path(experiment_dir) / 'test'), transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )
    
    # Evaluate on test set
    criterion = nn.CrossEntropyLoss()
    all_preds = []
    all_labels = []
    all_probs = []
    running_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    test_loss = running_loss / len(test_loader)
    
    all_probs_arr = np.array(all_probs)
    all_labels_arr = np.array(all_labels)
    all_preds_arr = np.array(all_preds)
    
    # Calibration metrics (for proposed-model-focused experiments)
    num_classes = len(class_names)
    ece = compute_ece(all_probs_arr, all_labels_arr, n_bins=15)
    brier = compute_brier_score(all_probs_arr, all_labels_arr, num_classes)
    nll = compute_nll(all_probs_arr, all_labels_arr)
    
    # Classification metrics
    accuracy = 100.0 * np.sum(all_preds_arr == all_labels_arr) / len(all_labels_arr)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels_arr, all_preds_arr, average=None, zero_division=0
    )
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    # Per-class metrics
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        per_class_metrics[class_name] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i])
        }
    
    # Classification report
    report = classification_report(
        all_labels_arr, all_preds_arr,
        target_names=class_names,
        output_dict=True
    )
    
    # Risk-coverage curve: thresholds 0.00 to 1.00 step 0.05
    max_confidences = np.max(all_probs_arr, axis=1)
    correct_mask = (all_preds_arr == all_labels_arr)
    risk_coverage_rows = []
    for t in np.arange(0.00, 1.01, 0.05):
        mask = max_confidences >= t
        if mask.sum() == 0:
            continue
        coverage = 100.0 * mask.mean()
        retained_acc = 100.0 * correct_mask[mask].mean()
        _, _, retained_f1, _ = precision_recall_fscore_support(
            all_labels_arr[mask], all_preds_arr[mask], average='macro', zero_division=0
        )
        risk = 1.0 - (correct_mask[mask].mean())
        risk_coverage_rows.append({
            'threshold': round(float(t), 2),
            'coverage': round(coverage, 4),
            'retained_accuracy': round(retained_acc, 4),
            'retained_macro_f1': round(float(retained_f1), 4),
            'risk': round(risk, 4)
        })
    
    print(f"\n{'='*60}")
    print("Classification Results")
    print(f"{'='*60}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"\nMacro-averaged metrics:")
    print(f"  Precision: {macro_precision:.4f}")
    print(f"  Recall: {macro_recall:.4f}")
    print(f"  F1-score: {macro_f1:.4f}")
    print(f"\nPer-class metrics:")
    for class_name in class_names:
        m = per_class_metrics[class_name]
        print(f"  {class_name:20s}: P={m['precision']:.4f}, R={m['recall']:.4f}, F1={m['f1']:.4f}")
    
    # Efficiency metrics
    print(f"\n{'='*60}")
    print("Efficiency Metrics")
    print(f"{'='*60}")
    
    num_params = count_parameters(model)
    model_size_fp32 = get_model_size_mb(model, precision='fp32')
    
    print(f"Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    print(f"Model size (FP32): {model_size_fp32:.2f} MB")
    
    # Inference time
    eval_cfg = config.get('evaluation', {})
    input_shape = (3, config['model']['input_size'], config['model']['input_size'])
    warmup = eval_cfg.get('inference_warmup', 10)
    runs = eval_cfg.get('inference_runs', 100)
    batch_size = eval_cfg.get('batch_size_inference', 1)
    
    mean_time, std_time = measure_inference_time(
        model, input_shape, device, warmup=warmup, runs=runs, batch_size=batch_size
    )
    
    print(f"Inference time (CPU): {mean_time:.2f} ± {std_time:.2f} ms")
    
    # FLOPs estimation (use a fresh model copy to avoid hook interference)
    flops_g = None
    try:
        from thop import profile
        # Create a fresh model copy for FLOPs calculation
        flops_model = create_model_from_config(model_name, num_classes, model_config)
        flops_model = flops_model.to(device)
        flops_model.eval()
        
        dummy_input = torch.randn(1, *input_shape).to(device)
        flops, params_thop = profile(flops_model, inputs=(dummy_input,), verbose=False)
        flops_g = flops / 1e9
        print(f"FLOPs: {flops_g:.2f} G")
        
        # Clean up flops model
        del flops_model
    except ImportError:
        try:
            from ptflops import get_model_complexity_info
            # Create a fresh model copy
            flops_model = create_model_from_config(model_name, num_classes, model_config)
            flops_model = flops_model.to(device)
            flops_model.eval()
            
            flops, params_flops = get_model_complexity_info(
                flops_model, input_shape, print_per_layer_stat=False, verbose=False
            )
            # Parse flops string (e.g., "0.123 GMac")
            if isinstance(flops, str):
                flops_g = float(flops.split()[0])
            else:
                flops_g = flops / 1e9
            print(f"FLOPs: {flops_g:.2f} G")
            
            # Clean up flops model
            del flops_model
        except ImportError:
            print("FLOPs: Not available (install thop or ptflops)")
    except Exception as e:
        print(f"FLOPs: Error during calculation ({e})")
    
    # Robustness evaluation
    robustness_results = evaluate_robustness(model, test_loader, device, config)
    
    # Recreate test_loader after robustness evaluation to ensure clean state
    # (robustness evaluation may have modified the dataset's transform)
    test_transform = get_data_transforms(config, 'test')
    test_dataset_fresh = ImageFolder(str(Path(experiment_dir) / 'test'), transform=test_transform)
    test_loader = DataLoader(
        test_dataset_fresh,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )
    
    # Uncertainty evaluation
    uncertainty_results = evaluate_uncertainty(model, test_loader, device, config)
    
    # Compile metrics
    metrics = {
        'model': model_name,
        'seed': checkpoint.get('seed'),
        'test_accuracy': float(accuracy),
        'test_loss': float(test_loss),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_f1': float(macro_f1),
        'per_class_metrics': per_class_metrics,
        'classification_report': report,
        'calibration': {
            'ece': float(ece),
            'brier_score': float(brier),
            'nll': float(nll)
        },
        'risk_coverage_curve': risk_coverage_rows,
        'efficiency': {
            'num_parameters': int(num_params),
            'num_parameters_millions': float(num_params / 1e6),
            'model_size_fp32_mb': float(model_size_fp32),
            'inference_time_ms_mean': float(mean_time),
            'inference_time_ms_std': float(std_time),
            'flops_g': float(flops_g) if flops_g is not None else None
        },
        'robustness': robustness_results,
        'uncertainty': uncertainty_results
    }
    
    # Save metrics
    output_dir = Path(checkpoint_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    
    save_metrics(metrics, str(output_dir / 'evaluation_metrics.json'))
    
    # Save results table
    results_table = {
        'Model': [model_name],
        'Accuracy (%)': [f"{accuracy:.2f}"],
        'Precision (%)': [f"{macro_precision*100:.2f}"],
        'Recall (%)': [f"{macro_recall*100:.2f}"],
        'F1-score (%)': [f"{macro_f1*100:.2f}"],
        'Params (M)': [f"{num_params/1e6:.2f}"],
        'Size (MB)': [f"{model_size_fp32:.2f}"],
        'Latency (ms)': [f"{mean_time:.2f}"]
    }
    
    try:
        df = pd.DataFrame(results_table)
        results_table_path = output_dir / 'results_table.csv'
        df.to_csv(results_table_path, index=False)
        print(f"Results table saved to: {results_table_path}")
    except Exception as e:
        print(f"Warning: Failed to save results_table.csv: {e}")
    
    # Save calibration metrics
    try:
        cal_df = pd.DataFrame([{
            'model': model_name,
            'ece': ece,
            'brier_score': brier,
            'nll': nll
        }])
        cal_df.to_csv(output_dir / 'calibration_metrics.csv', index=False)
        print(f"Calibration metrics saved to: {output_dir / 'calibration_metrics.csv'}")
    except Exception as e:
        print(f"Warning: Failed to save calibration_metrics.csv: {e}")
    
    # Save risk-coverage curve CSV and figure
    try:
        rc_df = pd.DataFrame(risk_coverage_rows)
        rc_df.to_csv(output_dir / 'risk_coverage_curve.csv', index=False)
        print(f"Risk-coverage curve saved to: {output_dir / 'risk_coverage_curve.csv'}")
        # Optional: coverage vs accuracy for compatibility
        cov_acc_df = rc_df[['threshold', 'coverage', 'retained_accuracy']].copy()
        cov_acc_df.rename(columns={'retained_accuracy': 'accuracy'}, inplace=True)
        cov_acc_df.to_csv(output_dir / 'coverage_accuracy_curve.csv', index=False)
        # Plot: x=coverage, y=risk, publication style
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(rc_df['coverage'], rc_df['risk'], 'b-', linewidth=2, label='Max softmax confidence')
        ax.set_xlabel('Coverage (%)')
        ax.set_ylabel('Risk')
        ax.set_title('Risk-Coverage Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 105)
        ax.set_ylim(-0.05, 1.05)
        fig.tight_layout()
        fig.savefig(output_dir / 'risk_coverage_curve.png', dpi=150, bbox_inches='tight')
        fig.savefig(output_dir / 'risk_coverage_curve.pdf', bbox_inches='tight')
        plt.close(fig)
        print(f"Risk-coverage figure saved to: {output_dir / 'risk_coverage_curve.png'}")
    except Exception as e:
        print(f"Warning: Failed to save risk-coverage curve/figure: {e}")
    
    # Save robustness results
    try:
        robustness_df = pd.DataFrame(robustness_results)
        robustness_df.to_csv(output_dir / 'robustness_results.csv', index=False)
        print(f"Robustness results saved to: {output_dir / 'robustness_results.csv'}")
    except Exception as e:
        print(f"Warning: Failed to save robustness_results.csv: {e}")
    
    # Save uncertainty results (coverage-accuracy curves)
    try:
        uncertainty_df = pd.DataFrame(uncertainty_results['coverage_accuracy_curve'])
        uncertainty_df.to_csv(output_dir / 'uncertainty_results.csv', index=False)
        print(f"Uncertainty results saved to: {output_dir / 'uncertainty_results.csv'}")
    except Exception as e:
        print(f"Warning: Failed to save uncertainty_results.csv: {e}")
    
    # Save confusion matrix
    try:
        cm = confusion_matrix(all_labels, all_preds)
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        cm_df.to_csv(output_dir / 'confusion_matrix.csv')
        print(f"Confusion matrix saved to: {output_dir / 'confusion_matrix.csv'}")
    except Exception as e:
        print(f"Warning: Failed to save confusion_matrix.csv: {e}")
    
    # Error analysis
    error_analysis = None
    if ERROR_ANALYSIS_AVAILABLE:
        try:
            print(f"\n{'='*60}")
            print("Error Analysis")
            print(f"{'='*60}")
            image_paths = [test_dataset.imgs[i][0] for i in range(len(test_dataset))]
            error_analysis = generate_error_report(
                np.array(all_preds), np.array(all_labels), class_names, image_paths, output_dir
            )
            metrics['error_analysis'] = error_analysis
            # Failure-case figure for paper (proposed-model-focused)
            if create_failure_case_figure is not None:
                try:
                    create_failure_case_figure(
                        np.array(all_preds), np.array(all_labels), class_names, image_paths,
                        output_dir, max_examples=6
                    )
                except Exception as e:
                    print(f"Warning: Failure-case figure failed: {e}")
        except Exception as e:
            print(f"Warning: Error analysis failed: {e}")
            error_analysis = None
    
    # Generate model card
    try:
        from model_card import generate_model_card
        model_card_path = generate_model_card(metrics, config, class_names, output_dir)
    except Exception as e:
        print(f"Warning: Model card generation failed: {e}")
        model_card_path = None
    
    print(f"\nResults saved to {output_dir}")
    if model_card_path:
        print(f"Model card saved to {model_card_path}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--config', type=str, default='src/config.yaml',
                       help='Path to config file')
    parser.add_argument('--ckpt', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--experiment_dir', type=str, default='experiment',
                       help='Directory containing test split')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility (optional).')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    evaluate_model(config, args.ckpt, args.experiment_dir, seed=args.seed)


if __name__ == '__main__':
    main()
