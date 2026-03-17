"""
Experiment runner for systematic performance improvement plan.
Executes steps sequentially and tracks results in CSV.
"""

import os
import sys
import argparse
import json
import csv
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

from utils import set_seed, setup_logging, load_config, count_parameters, get_device, save_metrics, get_model_size_mb, measure_inference_time
from models.proposed_lightweight import get_proposed_model
from augmentations import mixup_data, cutmix_data, mixup_criterion, RandAugment, RandomErasing


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


class ExperimentConfig:
    """Configuration for a single experiment run."""
    
    def __init__(self, **kwargs):
        # Model config
        self.dropout_set = kwargs.get('dropout_set', (0.3, 0.2, 0.1))
        self.activation = kwargs.get('activation', 'relu6')
        self.stem_stride = kwargs.get('stem_stride', 2)
        self.input_size = kwargs.get('input_size', 224)
        
        # Training config
        self.label_smoothing = kwargs.get('label_smoothing', 0.1)
        self.optimizer = kwargs.get('optimizer', 'adam')
        self.lr = kwargs.get('lr', 1e-4)
        self.weight_decay = kwargs.get('weight_decay', 1e-4)
        self.warmup_epochs = kwargs.get('warmup_epochs', 0)
        
        # Augmentation config
        self.mixup_alpha = kwargs.get('mixup_alpha', 0.0)
        self.cutmix_alpha = kwargs.get('cutmix_alpha', 0.0)
        self.mix_prob = kwargs.get('mix_prob', 0.5)
        self.randaugment_N = kwargs.get('randaugment_N', 0)
        self.randaugment_M = kwargs.get('randaugment_M', 0)
        self.erase_p = kwargs.get('erase_p', 0.0)
        
        # Other
        self.epochs = kwargs.get('epochs', 50)
        self.batch_size = kwargs.get('batch_size', 32)
        self.step = kwargs.get('step', 0)
        self.variant = kwargs.get('variant', 'baseline')
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for CSV/logging."""
        return {
            'step': self.step,
            'variant': self.variant,
            'label_smoothing': self.label_smoothing,
            'dropout_set': str(self.dropout_set),
            'optimizer': self.optimizer,
            'lr': self.lr,
            'weight_decay': self.weight_decay,
            'warmup_epochs': self.warmup_epochs,
            'mixup_alpha': self.mixup_alpha,
            'cutmix_alpha': self.cutmix_alpha,
            'mix_prob': self.mix_prob,
            'randaugment_N': self.randaugment_N,
            'randaugment_M': self.randaugment_M,
            'erase_p': self.erase_p,
            'input_size': self.input_size,
            'stem_stride': self.stem_stride,
            'activation': self.activation,
        }
    
    def to_yaml(self) -> Dict:
        """Convert to YAML-compatible dict."""
        return {
            'model': {
                'num_classes': 5,
                'input_size': self.input_size,
                'dropout_set': list(self.dropout_set),
                'activation': self.activation,
                'stem_stride': self.stem_stride,
            },
            'training': {
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'learning_rate': self.lr,
                'optimizer': self.optimizer,
                'optimizer_params': {
                    'weight_decay': self.weight_decay,
                },
                'scheduler': 'cosine_warmup' if self.warmup_epochs > 0 else 'cosine',
                'scheduler_params': {
                    'warmup_epochs': self.warmup_epochs,
                },
                'label_smoothing': self.label_smoothing,
                'early_stopping': {
                    'patience': 10,
                    'min_delta': 0.001,
                },
                'mixed_precision': True,
                'num_workers': 4,
                'pin_memory': True,
            },
            'augmentation': {
                'mixup_alpha': self.mixup_alpha,
                'cutmix_alpha': self.cutmix_alpha,
                'mix_prob': self.mix_prob,
                'randaugment_N': self.randaugment_N,
                'randaugment_M': self.randaugment_M,
                'erase_p': self.erase_p,
            }
        }


def get_data_transforms(config: ExperimentConfig, split: str, base_config: Dict):
    """Get data transforms with optional RandAugment and RandomErasing."""
    norm_mean = base_config['normalization']['mean']
    norm_std = base_config['normalization']['std']
    
    if split == 'train':
        aug_config = base_config['augmentation']['train']
        transforms_list = [
            transforms.Resize(aug_config['resize']),
            transforms.RandomResizedCrop(config.input_size if config.input_size > 0 else aug_config['random_resized_crop']),
            transforms.RandomHorizontalFlip(p=aug_config['random_horizontal_flip']),
            transforms.RandomVerticalFlip(p=aug_config['random_vertical_flip']),
            transforms.RandomRotation(aug_config['random_rotation']),
            transforms.ColorJitter(
                brightness=aug_config['color_jitter']['brightness'],
                contrast=aug_config['color_jitter']['contrast'],
                saturation=aug_config['color_jitter']['saturation'],
                hue=aug_config['color_jitter']['hue']
            ),
        ]
        
        # Add RandAugment if enabled (before ToTensor)
        if config.randaugment_N > 0:
            transforms_list.append(RandAugment(n=config.randaugment_N, m=config.randaugment_M))
        
        transforms_list.append(transforms.ToTensor())
        
        # Add RandomErasing if enabled (after ToTensor, before Normalize)
        if config.erase_p > 0:
            transforms_list.append(RandomErasing(p=config.erase_p))
        
        transforms_list.append(transforms.Normalize(mean=norm_mean, std=norm_std))
        
        transform = transforms.Compose(transforms_list)
    else:  # val or test
        aug_config = base_config['augmentation']['val_test']
        # For eval, always use 224 for fair comparison (unless explicitly different)
        # Step 5 option B trains at 256 but should eval at 224
        eval_size = aug_config['center_crop']  # Default to 224
        transform = transforms.Compose([
            transforms.Resize(aug_config['resize']),
            transforms.CenterCrop(eval_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std)
        ])
    
    return transform


def get_data_loaders(config: ExperimentConfig, experiment_dir: str, base_config: Dict):
    """Create data loaders."""
    exp_dir = Path(experiment_dir)
    
    train_transform = get_data_transforms(config, 'train', base_config)
    val_transform = get_data_transforms(config, 'val', base_config)
    test_transform = get_data_transforms(config, 'test', base_config)
    
    train_dataset = ImageFolder(str(exp_dir / 'train'), transform=train_transform)
    val_dataset = ImageFolder(str(exp_dir / 'valid'), transform=val_transform)
    test_dataset = ImageFolder(str(exp_dir / 'test'), transform=test_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=base_config['training']['num_workers'],
        pin_memory=base_config['training']['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=base_config['training']['num_workers'],
        pin_memory=base_config['training']['pin_memory']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=base_config['training']['num_workers'],
        pin_memory=base_config['training']['pin_memory']
    )
    
    return train_loader, val_loader, test_loader, train_dataset.classes


def train_epoch(model, loader, criterion, optimizer, device, config: ExperimentConfig, use_amp=False):
    """Train for one epoch with optional MixUp/CutMix."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    use_mixup = config.mixup_alpha > 0
    use_cutmix = config.cutmix_alpha > 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Apply MixUp or CutMix
        if use_mixup or use_cutmix:
            if np.random.random() < config.mix_prob:
                if use_cutmix and (not use_mixup or np.random.random() < 0.5):
                    # Use CutMix
                    mixed_inputs, y_a, y_b, lam = cutmix_data(inputs, labels, config.cutmix_alpha)
                else:
                    # Use MixUp
                    mixed_inputs, y_a, y_b, lam = mixup_data(inputs, labels, config.mixup_alpha)
            else:
                mixed_inputs, y_a, y_b, lam = inputs, labels, labels, 1.0
        else:
            mixed_inputs, y_a, y_b, lam = inputs, labels, labels, 1.0
        
        optimizer.zero_grad()
        
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(mixed_inputs)
                if use_mixup or use_cutmix:
                    loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
                else:
                    loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(mixed_inputs)
            if use_mixup or use_cutmix:
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        # For MixUp/CutMix, use original labels for accuracy calculation
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    
    # Calculate macro F1
    macro_f1 = f1_score(all_labels, all_preds, average='macro') * 100
    
    return epoch_loss, epoch_acc, macro_f1, all_preds, all_labels


def plot_training_curves(history: dict, save_path: str):
    """Plot training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path: str):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_experiment(config: ExperimentConfig, base_config: Dict, experiment_dir: str, output_dir: Path, logger) -> Dict:
    """Run a single experiment."""
    # Setup
    set_seed(42)
    device = get_device()
    
    # Create run directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = output_dir / f"{timestamp}_{config.step}_{config.variant}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Running experiment: Step {config.step}, Variant: {config.variant}")
    logger.info(f"Run directory: {run_dir}")
    
    # Get data loaders
    train_loader, val_loader, test_loader, class_names = get_data_loaders(
        config, experiment_dir, base_config
    )
    logger.info(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    # Create model
    model = get_proposed_model(
        num_classes=base_config['model']['num_classes'],
        input_size=config.input_size,
        dropout_set=config.dropout_set,
        activation=config.activation,
        stem_stride=config.stem_stride
    )
    model = model.to(device)
    num_params = count_parameters(model)
    logger.info(f"Model parameters: {num_params:,}")
    
    # Loss with label smoothing
    if config.label_smoothing > 0:
        criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    if config.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
    
    # Scheduler
    if config.warmup_epochs > 0:
        def lr_lambda(epoch):
            if epoch < config.warmup_epochs:
                return (epoch + 1) / config.warmup_epochs
            else:
                return 0.5 * (1 + np.cos(np.pi * (epoch - config.warmup_epochs) / (config.epochs - config.warmup_epochs)))
        scheduler = LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=base_config['training']['early_stopping']['patience'],
        min_delta=base_config['training']['early_stopping']['min_delta']
    )
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_epoch = 0
    best_model_path = run_dir / 'best_model.pth'
    use_amp = base_config['training']['mixed_precision'] and device.type == 'cuda'
    
    # Training loop
    logger.info(f"Starting training for {config.epochs} epochs...")
    
    for epoch in range(config.epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, config, use_amp
        )
        
        # Validate
        val_loss, val_acc, val_f1, _, _ = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        logger.info(
            f"Epoch {epoch+1}/{config.epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.2f}%"
        )
        
        # Save best model (tie-breaker: macro F1)
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_f1 > best_val_f1):
            best_val_acc = val_acc
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'config': config.to_dict(),
            }, best_model_path)
            logger.info(f"Saved best model (Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.2f}%)")
        
        # Early stopping check
        if early_stopping(val_loss):
            logger.info(f"Early stopping at epoch {epoch+1} (patience: {early_stopping.patience})")
            break
    
    # Load best model and evaluate on test set
    logger.info("Evaluating on test set...")
    checkpoint = torch.load(best_model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_f1, test_preds, test_labels = validate(
        model, test_loader, criterion, device
    )
    
    # Classification report
    report = classification_report(
        test_labels, test_preds,
        target_names=class_names,
        output_dict=True
    )
    
    logger.info(f"Test Accuracy: {test_acc:.2f}%")
    logger.info(f"Test F1 (macro): {test_f1:.2f}%")
    
    # Calculate efficiency metrics
    logger.info("Calculating efficiency metrics...")
    model_size_mb = get_model_size_mb(model, precision="fp32")
    params_m = num_params / 1e6  # Convert to millions
    
    # Measure inference latency
    input_size = config.input_size if config.input_size > 0 else 224
    mean_latency, std_latency = measure_inference_time(
        model, (3, input_size, input_size), device, warmup=10, runs=100, batch_size=1
    )
    
    logger.info(f"Parameters: {params_m:.2f}M, Size: {model_size_mb:.2f}MB, Latency: {mean_latency:.2f}±{std_latency:.2f}ms")
    
    # Save metrics
    metrics = {
        'config': config.to_dict(),
        'num_parameters': num_params,
        'params_m': params_m,
        'model_size_mb': model_size_mb,
        'latency_ms': mean_latency,
        'latency_std_ms': std_latency,
        'best_val_acc': best_val_acc,
        'best_val_f1': best_val_f1,
        'best_epoch': best_epoch,
        'train_acc_at_best': history['train_acc'][best_epoch - 1] if best_epoch > 0 else 0,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'test_loss': test_loss,
        'classification_report': report,
        'history': history
    }
    
    save_metrics(metrics, str(run_dir / 'metrics.json'))
    
    # Save config
    with open(run_dir / 'config.yaml', 'w') as f:
        yaml.dump(config.to_yaml(), f, default_flow_style=False)
    
    # Save classification report
    with open(run_dir / 'classification_report.txt', 'w') as f:
        f.write(classification_report(test_labels, test_preds, target_names=class_names))
    
    # Plot training curves
    plot_training_curves(history, str(run_dir / 'training_curves.png'))
    
    # Plot confusion matrix
    plot_confusion_matrix(
        test_labels, test_preds, class_names,
        str(run_dir / 'confusion_matrix.png')
    )
    
    logger.info(f"Experiment complete! Results saved to {run_dir}")
    
    return {
        'run_dir': str(run_dir),
        'best_val_acc': best_val_acc,
        'best_val_f1': best_val_f1,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'train_acc_at_best': history['train_acc'][best_epoch - 1] if best_epoch > 0 else 0,
        'params_m': params_m,
        'model_size_mb': model_size_mb,
        'latency_ms': mean_latency,
    }


def update_summary_csv(summary_path: Path, result: Dict, config: ExperimentConfig, logger=None):
    """Update or create summary CSV."""
    # Ensure directory exists
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    file_exists = summary_path.exists()
    
    fieldnames = [
        'step', 'variant', 'label_smoothing', 'dropout_set', 'optimizer', 'lr', 'weight_decay',
        'warmup_epochs', 'mixup_alpha', 'cutmix_alpha', 'mix_prob', 'randaugment_N', 'randaugment_M',
        'erase_p', 'input_size', 'stem_stride', 'activation',
        'best_val_acc', 'best_val_f1', 'test_acc', 'test_f1', 'train_acc_at_best',
        'params_m', 'size_mb', 'latency_ms'
    ]
    
    try:
        with open(summary_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            
            row = config.to_dict()
            row.update({
                'best_val_acc': f"{result['best_val_acc']:.4f}",
                'best_val_f1': f"{result['best_val_f1']:.4f}",
                'test_acc': f"{result['test_acc']:.4f}",
                'test_f1': f"{result['test_f1']:.4f}",
                'train_acc_at_best': f"{result['train_acc_at_best']:.4f}",
                'params_m': f"{result.get('params_m', 0):.2f}",
                'size_mb': f"{result.get('model_size_mb', 0):.2f}",
                'latency_ms': f"{result.get('latency_ms', 0):.2f}",
            })
            writer.writerow(row)
        
        if logger:
            logger.info(f"Results saved to summary.csv: {summary_path}")
    except Exception as e:
        error_msg = f"Failed to save summary.csv: {e}"
        if logger:
            logger.error(error_msg)
        else:
            print(error_msg)
        raise


def get_best_result_from_step(summary_path: Path, step: int) -> Optional[Dict]:
    """Get the best result from a specific step by reading CSV."""
    if not summary_path.exists():
        return None
    
    best_result = None
    best_score = -1.0
    
    with open(summary_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row['step']) == step:
                val_acc = float(row['best_val_acc'])
                val_f1 = float(row['best_val_f1'])
                score = val_acc + 0.01 * val_f1
                if score > best_score:
                    best_score = score
                    best_result = {
                        'best_val_acc': val_acc,
                        'best_val_f1': val_f1,
                        'test_acc': float(row['test_acc']),
                        'test_f1': float(row['test_f1']),
                        'variant': row['variant'],
                    }
    
    return best_result


def run_step1(base_config: Dict, experiment_dir: str, output_dir: Path, logger) -> ExperimentConfig:
    """Step 1: Reduce Regularization."""
    logger.info("=" * 80)
    logger.info("STEP 1: Reduce Regularization")
    logger.info("=" * 80)
    
    summary_path = output_dir / 'summary.csv'
    
    # Label smoothing sweep
    best_config = None
    best_result = None
    best_score = -1.0
    
    label_smoothing_values = [0.1, 0.05, 0.0]
    base_dropout = (0.3, 0.2, 0.1)
    
    logger.info("Label smoothing sweep...")
    for ls in label_smoothing_values:
        config = ExperimentConfig(
            step=1,
            variant=f'ls_{ls}',
            label_smoothing=ls,
            dropout_set=base_dropout,
            epochs=base_config['training']['epochs'],
            batch_size=base_config['training']['batch_size'],
            lr=base_config['training']['learning_rate'],
            weight_decay=base_config['training']['optimizer_params']['weight_decay'],
        )
        
        result = run_experiment(config, base_config, experiment_dir, output_dir, logger)
        update_summary_csv(summary_path, result, config, logger)
        
        score = result['best_val_acc'] + 0.01 * result['best_val_f1']  # Tie-breaker: F1
        if score > best_score:
            best_score = score
            best_config = config
            best_result = result
    
    logger.info(f"Best label smoothing: {best_config.label_smoothing} (Val Acc: {best_result['best_val_acc']:.2f}%)")
    
    # Dropout sweep with best label smoothing
    logger.info("Dropout sweep...")
    dropout_sets = [
        (0.3, 0.2, 0.1),  # Original
        (0.2, 0.1, 0.0),
        (0.1, 0.1, 0.0),
    ]
    
    for dropout_set in dropout_sets:
        config = ExperimentConfig(
            step=1,
            variant=f'dropout_{dropout_set[0]}_{dropout_set[1]}_{dropout_set[2]}',
            label_smoothing=best_config.label_smoothing,
            dropout_set=dropout_set,
            epochs=base_config['training']['epochs'],
            batch_size=base_config['training']['batch_size'],
            lr=base_config['training']['learning_rate'],
            weight_decay=base_config['training']['optimizer_params']['weight_decay'],
        )
        
        result = run_experiment(config, base_config, experiment_dir, output_dir, logger)
        update_summary_csv(summary_path, result, config, logger)
        
        score = result['best_val_acc'] + 0.01 * result['best_val_f1']
        if score > best_score:
            best_score = score
            best_config = config
            best_result = result
    
    logger.info(f"STEP 1 BEST: Val Acc: {best_result['best_val_acc']:.2f}%, Val F1: {best_result['best_val_f1']:.2f}%")
    return best_config


def run_step2(base_config: Dict, experiment_dir: str, output_dir: Path, logger, prev_config: ExperimentConfig) -> ExperimentConfig:
    """Step 2: Optimizer & LR Schedule Upgrade."""
    logger.info("=" * 80)
    logger.info("STEP 2: Optimizer & LR Schedule Upgrade")
    logger.info("=" * 80)
    
    summary_path = output_dir / 'summary.csv'
    
    best_config = None
    best_result = None
    best_score = -1.0
    
    lr_values = [1e-4, 3e-4]
    wd_values = [1e-4, 5e-4]
    
    for lr in lr_values:
        for wd in wd_values:
            config = ExperimentConfig(
                step=2,
                variant=f'adamw_lr{lr}_wd{wd}',
                label_smoothing=prev_config.label_smoothing,
                dropout_set=prev_config.dropout_set,
                optimizer='adamw',
                lr=lr,
                weight_decay=wd,
                warmup_epochs=5,
                epochs=base_config['training']['epochs'],
                batch_size=base_config['training']['batch_size'],
            )
            
            result = run_experiment(config, base_config, experiment_dir, output_dir, logger)
            update_summary_csv(summary_path, result, config)
            
            score = result['best_val_acc'] + 0.01 * result['best_val_f1']
            if score > best_score:
                best_score = score
                best_config = config
                best_result = result
    
    logger.info(f"STEP 2 BEST: Val Acc: {best_result['best_val_acc']:.2f}%, Val F1: {best_result['best_val_f1']:.2f}%")
    return best_config


def run_step3(base_config: Dict, experiment_dir: str, output_dir: Path, logger, prev_config: ExperimentConfig) -> ExperimentConfig:
    """Step 3: Data Mixing (MixUp/CutMix)."""
    logger.info("=" * 80)
    logger.info("STEP 3: Data Mixing (MixUp/CutMix)")
    logger.info("=" * 80)
    
    summary_path = output_dir / 'summary.csv'
    
    # Get best result from step 2 for comparison
    step2_best = get_best_result_from_step(summary_path, 2)
    if step2_best:
        logger.info(f"Step 2 best: Val Acc: {step2_best['best_val_acc']:.2f}%, Val F1: {step2_best['best_val_f1']:.2f}%")
    
    # Run with MixUp/CutMix
    config = ExperimentConfig(
        step=3,
        variant='mixup_cutmix',
        label_smoothing=prev_config.label_smoothing,
        dropout_set=prev_config.dropout_set,
        optimizer=prev_config.optimizer,
        lr=prev_config.lr,
        weight_decay=prev_config.weight_decay,
        warmup_epochs=prev_config.warmup_epochs,
        mixup_alpha=0.2,
        cutmix_alpha=1.0,
        mix_prob=0.5,
        epochs=base_config['training']['epochs'],
        batch_size=base_config['training']['batch_size'],
    )
    
    result = run_experiment(config, base_config, experiment_dir, output_dir, logger)
    update_summary_csv(summary_path, result, config)
    
    logger.info(f"STEP 3: Val Acc: {result['best_val_acc']:.2f}%, Val F1: {result['best_val_f1']:.2f}%")
    
    # Compare with step 2 and keep best
    if step2_best:
        step2_score = step2_best['best_val_acc'] + 0.01 * step2_best['best_val_f1']
        step3_score = result['best_val_acc'] + 0.01 * result['best_val_f1']
        if step3_score > step2_score:
            logger.info("Step 3 (MixUp/CutMix) is better than Step 2")
            return config
        else:
            logger.info("Step 2 is better, keeping Step 2 configuration")
            return prev_config
    
    return config


def run_step4(base_config: Dict, experiment_dir: str, output_dir: Path, logger, prev_config: ExperimentConfig) -> ExperimentConfig:
    """Step 4: Stronger Augmentation (RandAugment + RandomErasing)."""
    logger.info("=" * 80)
    logger.info("STEP 4: Stronger Augmentation (RandAugment + RandomErasing)")
    logger.info("=" * 80)
    
    summary_path = output_dir / 'summary.csv'
    
    config = ExperimentConfig(
        step=4,
        variant='randaug_erase',
        label_smoothing=prev_config.label_smoothing,
        dropout_set=prev_config.dropout_set,
        optimizer=prev_config.optimizer,
        lr=prev_config.lr,
        weight_decay=prev_config.weight_decay,
        warmup_epochs=prev_config.warmup_epochs,
        mixup_alpha=prev_config.mixup_alpha,
        cutmix_alpha=prev_config.cutmix_alpha,
        mix_prob=prev_config.mix_prob,
        randaugment_N=2,
        randaugment_M=9,
        erase_p=0.25,
        epochs=base_config['training']['epochs'],
        batch_size=base_config['training']['batch_size'],
    )
    
    result = run_experiment(config, base_config, experiment_dir, output_dir, logger)
    update_summary_csv(summary_path, result, config)
    
    logger.info(f"STEP 4: Val Acc: {result['best_val_acc']:.2f}%, Val F1: {result['best_val_f1']:.2f}%")
    return config


def run_step5(base_config: Dict, experiment_dir: str, output_dir: Path, logger, prev_config: ExperimentConfig) -> ExperimentConfig:
    """Step 5: Preserve Early Spatial Detail."""
    logger.info("=" * 80)
    logger.info("STEP 5: Preserve Early Spatial Detail")
    logger.info("=" * 80)
    
    summary_path = output_dir / 'summary.csv'
    
    best_config = None
    best_result = None
    best_score = -1.0
    
    # Option A: Stem stride = 1
    config_a = ExperimentConfig(
        step=5,
        variant='stem_stride1',
        label_smoothing=prev_config.label_smoothing,
        dropout_set=prev_config.dropout_set,
        optimizer=prev_config.optimizer,
        lr=prev_config.lr,
        weight_decay=prev_config.weight_decay,
        warmup_epochs=prev_config.warmup_epochs,
        mixup_alpha=prev_config.mixup_alpha,
        cutmix_alpha=prev_config.cutmix_alpha,
        mix_prob=prev_config.mix_prob,
        randaugment_N=prev_config.randaugment_N,
        randaugment_M=prev_config.randaugment_M,
        erase_p=prev_config.erase_p,
        stem_stride=1,
        epochs=base_config['training']['epochs'],
        batch_size=base_config['training']['batch_size'],
    )
    
    result_a = run_experiment(config_a, base_config, experiment_dir, output_dir, logger)
    update_summary_csv(summary_path, result_a, config_a, logger)
    
    score_a = result_a['best_val_acc'] + 0.01 * result_a['best_val_f1']
    if score_a > best_score:
        best_score = score_a
        best_config = config_a
        best_result = result_a
    
    # Option B: Higher input size (256)
    config_b = ExperimentConfig(
        step=5,
        variant='input256',
        label_smoothing=prev_config.label_smoothing,
        dropout_set=prev_config.dropout_set,
        optimizer=prev_config.optimizer,
        lr=prev_config.lr,
        weight_decay=prev_config.weight_decay,
        warmup_epochs=prev_config.warmup_epochs,
        mixup_alpha=prev_config.mixup_alpha,
        cutmix_alpha=prev_config.cutmix_alpha,
        mix_prob=prev_config.mix_prob,
        randaugment_N=prev_config.randaugment_N,
        randaugment_M=prev_config.randaugment_M,
        erase_p=prev_config.erase_p,
        input_size=256,
        epochs=base_config['training']['epochs'],
        batch_size=base_config['training']['batch_size'],
    )
    
    result_b = run_experiment(config_b, base_config, experiment_dir, output_dir, logger)
    update_summary_csv(summary_path, result_b, config_b, logger)
    
    score_b = result_b['best_val_acc'] + 0.01 * result_b['best_val_f1']
    if score_b > best_score:
        best_score = score_b
        best_config = config_b
        best_result = result_b
    
    logger.info(f"STEP 5 BEST: Val Acc: {best_result['best_val_acc']:.2f}%, Val F1: {best_result['best_val_f1']:.2f}%")
    return best_config


def run_step6(base_config: Dict, experiment_dir: str, output_dir: Path, logger, prev_config: ExperimentConfig) -> ExperimentConfig:
    """Step 6: Activation Upgrade (SiLU)."""
    logger.info("=" * 80)
    logger.info("STEP 6: Activation Upgrade (SiLU)")
    logger.info("=" * 80)
    
    summary_path = output_dir / 'summary.csv'
    
    config = ExperimentConfig(
        step=6,
        variant='silu',
        label_smoothing=prev_config.label_smoothing,
        dropout_set=prev_config.dropout_set,
        optimizer=prev_config.optimizer,
        lr=prev_config.lr,
        weight_decay=prev_config.weight_decay,
        warmup_epochs=prev_config.warmup_epochs,
        mixup_alpha=prev_config.mixup_alpha,
        cutmix_alpha=prev_config.cutmix_alpha,
        mix_prob=prev_config.mix_prob,
        randaugment_N=prev_config.randaugment_N,
        randaugment_M=prev_config.randaugment_M,
        erase_p=prev_config.erase_p,
        input_size=prev_config.input_size,
        stem_stride=prev_config.stem_stride,
        activation='silu',
        epochs=base_config['training']['epochs'],
        batch_size=base_config['training']['batch_size'],
    )
    
    result = run_experiment(config, base_config, experiment_dir, output_dir, logger)
    update_summary_csv(summary_path, result, config)
    
    logger.info(f"STEP 6: Val Acc: {result['best_val_acc']:.2f}%, Val F1: {result['best_val_f1']:.2f}%")
    return config


def main():
    parser = argparse.ArgumentParser(description='Run ablation study for performance improvement')
    parser.add_argument('--config', type=str, default='src/config.yaml',
                       help='Path to base config file')
    parser.add_argument('--experiment_dir', type=str, default='experiment',
                       help='Directory containing train/valid/test splits')
    parser.add_argument('--output_dir', type=str, default='experiment/ablation_runs',
                       help='Output directory for all runs')
    parser.add_argument('--start_step', type=int, default=1,
                       help='Starting step (1-6)')
    parser.add_argument('--end_step', type=int, default=6,
                       help='Ending step (1-6)')
    
    args = parser.parse_args()
    
    # Load base config
    base_config = load_config(args.config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(str(output_dir), 'ablation')
    
    logger.info("Starting Performance Improvement Plan")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Steps: {args.start_step} to {args.end_step}")
    
    # Run steps sequentially
    best_config = None
    
    if args.start_step <= 1 <= args.end_step:
        best_config = run_step1(base_config, args.experiment_dir, output_dir, logger)
    
    if args.start_step <= 2 <= args.end_step:
        best_config = run_step2(base_config, args.experiment_dir, output_dir, logger, best_config)
    
    if args.start_step <= 3 <= args.end_step:
        best_config = run_step3(base_config, args.experiment_dir, output_dir, logger, best_config)
    
    if args.start_step <= 4 <= args.end_step:
        best_config = run_step4(base_config, args.experiment_dir, output_dir, logger, best_config)
    
    if args.start_step <= 5 <= args.end_step:
        best_config = run_step5(base_config, args.experiment_dir, output_dir, logger, best_config)
    
    if args.start_step <= 6 <= args.end_step:
        best_config = run_step6(base_config, args.experiment_dir, output_dir, logger, best_config)
    
    # Save best config and print final summary
    if best_config:
        best_config_path = output_dir / 'best_config.yaml'
        with open(best_config_path, 'w') as f:
            yaml.dump(best_config.to_yaml(), f, default_flow_style=False)
        
        # Read summary CSV to get final results
        summary_path = output_dir / 'summary.csv'
        best_val_acc = None
        best_val_f1 = None
        final_test_acc = None
        final_test_f1 = None
        
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    # Find the row matching our best config
                    for row in reversed(rows):  # Check from most recent
                        if (int(row['step']) == best_config.step and 
                            row['variant'] == best_config.variant):
                            best_val_acc = float(row['best_val_acc'])
                            best_val_f1 = float(row['best_val_f1'])
                            final_test_acc = float(row['test_acc'])
                            final_test_f1 = float(row['test_f1'])
                            break
        
        logger.info("=" * 80)
        logger.info("FINAL BEST CONFIGURATION")
        logger.info("=" * 80)
        logger.info(f"Best configuration saved to: {best_config_path}")
        logger.info("")
        logger.info("Best Hyperparameters:")
        logger.info(f"  Label Smoothing: {best_config.label_smoothing}")
        logger.info(f"  Dropout Set: {best_config.dropout_set}")
        logger.info(f"  Optimizer: {best_config.optimizer}")
        logger.info(f"  Learning Rate: {best_config.lr}")
        logger.info(f"  Weight Decay: {best_config.weight_decay}")
        logger.info(f"  Warmup Epochs: {best_config.warmup_epochs}")
        logger.info(f"  MixUp Alpha: {best_config.mixup_alpha}")
        logger.info(f"  CutMix Alpha: {best_config.cutmix_alpha}")
        logger.info(f"  Mix Probability: {best_config.mix_prob}")
        logger.info(f"  RandAugment (N, M): ({best_config.randaugment_N}, {best_config.randaugment_M})")
        logger.info(f"  Random Erasing p: {best_config.erase_p}")
        logger.info(f"  Input Size: {best_config.input_size}")
        logger.info(f"  Stem Stride: {best_config.stem_stride}")
        logger.info(f"  Activation: {best_config.activation}")
        logger.info("")
        if best_val_acc is not None:
            logger.info(f"Best Validation Accuracy: {best_val_acc:.2f}%")
            logger.info(f"Best Validation F1 (macro): {best_val_f1:.2f}%")
            logger.info(f"Final Test Accuracy: {final_test_acc:.2f}%")
            logger.info(f"Final Test F1 (macro): {final_test_f1:.2f}%")
        logger.info("")
        logger.info("Summary CSV: experiment/ablation_runs/summary.csv")
        logger.info("=" * 80)


if __name__ == '__main__':
    main()
