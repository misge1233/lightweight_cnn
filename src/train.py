"""
Training script for wheat disease classification models.
Supports baseline models and the proposed lightweight model.
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, LambdaLR
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from utils import (
    set_seed, setup_logging, load_config, count_parameters,
    create_experiment_dir, get_device, save_metrics,
    get_model_size_mb, measure_inference_time
)
from models.baselines import get_baseline_model
from models.proposed_lightweight import get_proposed_model, proposed_model_kwargs_from_config


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


def get_data_transforms(config: dict, split: str):
    """Get data transforms for train/val/test splits."""
    norm_mean = config['normalization']['mean']
    norm_std = config['normalization']['std']
    
    if split == 'train':
        aug_config = config['augmentation']['train']
        transform = transforms.Compose([
            transforms.Resize(aug_config['resize']),
            transforms.RandomResizedCrop(aug_config['random_resized_crop']),
            transforms.RandomHorizontalFlip(p=aug_config['random_horizontal_flip']),
            transforms.RandomVerticalFlip(p=aug_config['random_vertical_flip']),
            transforms.RandomRotation(aug_config['random_rotation']),
            transforms.ColorJitter(
                brightness=aug_config['color_jitter']['brightness'],
                contrast=aug_config['color_jitter']['contrast'],
                saturation=aug_config['color_jitter']['saturation'],
                hue=aug_config['color_jitter']['hue']
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std)
        ])
    else:  # val or test
        aug_config = config['augmentation']['val_test']
        transform = transforms.Compose([
            transforms.Resize(aug_config['resize']),
            transforms.CenterCrop(aug_config['center_crop']),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std)
        ])
    
    return transform


def get_data_loaders(config: dict, experiment_dir: str):
    """Create data loaders for train/val/test splits."""
    exp_dir = Path(experiment_dir)
    
    train_transform = get_data_transforms(config, 'train')
    val_transform = get_data_transforms(config, 'val')
    test_transform = get_data_transforms(config, 'test')
    
    train_dataset = ImageFolder(str(exp_dir / 'train'), transform=train_transform)
    val_dataset = ImageFolder(str(exp_dir / 'valid'), transform=val_transform)
    test_dataset = ImageFolder(str(exp_dir / 'test'), transform=test_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory']
    )
    
    return train_loader, val_loader, test_loader, train_dataset.classes


def train_epoch(model, loader, criterion, optimizer, device, use_amp=False):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
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
    
    return epoch_loss, epoch_acc, all_preds, all_labels


def plot_training_curves(history: dict, save_path: str):
    """Plot training curves (loss and accuracy)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
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


def train_model(config: dict, model_name: str, experiment_dir: str, seed: int = None):
    """Main training function. seed overrides config['data']['seed'] when provided."""
    # Seed: CLI overrides config
    effective_seed = seed if seed is not None else config['data'].get('seed', 42)
    set_seed(effective_seed)
    device = get_device()
    
    # Create experiment directory (include seed in name when explicitly set for multi-seed runs)
    exp_dir = create_experiment_dir(
        config['output']['runs_dir'], model_name, seed=effective_seed if seed is not None else None
    )
    logger = setup_logging(str(exp_dir), model_name)
    
    logger.info(f"Training {model_name} model")
    logger.info(f"Experiment directory: {exp_dir}")
    
    # Get data loaders
    train_loader, val_loader, test_loader, class_names = get_data_loaders(
        config, experiment_dir
    )
    logger.info(f"Classes: {class_names}")
    logger.info(f"Train: {len(train_loader.dataset)}, "
                f"Val: {len(val_loader.dataset)}, "
                f"Test: {len(test_loader.dataset)}")
    
    # Create model (proposed: use all architecture params from config)
    num_classes = config['model']['num_classes']
    if model_name == 'proposed':
        kwargs = proposed_model_kwargs_from_config(config)
        model = get_proposed_model(**kwargs)
    else:
        model = get_baseline_model(model_name, num_classes=num_classes, pretrained=True)
    
    model = model.to(device)
    num_params = count_parameters(model)
    logger.info(f"Model parameters: {num_params:,}")
    
    # Loss with optional label smoothing
    label_smoothing = config['training'].get('label_smoothing', 0.0)
    if label_smoothing > 0:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        logger.info(f"Using label smoothing: {label_smoothing}")
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer: respect config (Adam vs AdamW for proposed-model best config)
    optimizer_params = config['training'].get('optimizer_params', {})
    weight_decay = optimizer_params.get('weight_decay', 0.0)
    optimizer_name = (config['training'].get('optimizer') or 'adam').strip().lower()
    if optimizer_name == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=weight_decay
        )
        logger.info(f"Using AdamW (weight_decay={weight_decay})")
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=weight_decay
        )
        if weight_decay > 0:
            logger.info(f"Using Adam (weight_decay={weight_decay})")
    
    # Scheduler: cosine, cosine_warmup, or step
    scheduler_type = (config['training'].get('scheduler') or 'cosine').strip().lower()
    scheduler_params = config['training']['scheduler_params']
    
    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs']
        )
    elif scheduler_type == 'cosine_warmup':
        # Cosine annealing with warm-up
        warmup_epochs = scheduler_params.get('warmup_epochs', 5)
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (config['training']['epochs'] - warmup_epochs)))
        scheduler = LambdaLR(optimizer, lr_lambda)
        logger.info(f"Using cosine annealing with {warmup_epochs} warmup epochs")
    else:
        scheduler = StepLR(
            optimizer,
            step_size=scheduler_params.get('step_size', 30),
            gamma=scheduler_params.get('gamma', 0.1)
        )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['training']['early_stopping']['patience'],
        min_delta=config['training']['early_stopping']['min_delta']
    )
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    best_model_path = exp_dir / 'best_model.pth'
    use_amp = config['training']['mixed_precision'] and device.type == 'cuda'
    
    # Training loop
    num_epochs = config['training']['epochs']
    logger.info(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, use_amp
        )
        
        # Validate
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        logger.info(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'model_name': model_name,
                'config': config,
                'class_names': class_names,
                'seed': effective_seed,
            }, best_model_path)
            logger.info(f"Saved best model (Val Acc: {val_acc:.2f}%)")
        
        # Early stopping
        if early_stopping(val_loss):
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model and evaluate on test set
    logger.info("Evaluating on test set...")
    checkpoint = torch.load(best_model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_preds, test_labels = validate(
        model, test_loader, criterion, device
    )
    
    # Classification report
    report = classification_report(
        test_labels, test_preds,
        target_names=class_names,
        output_dict=True
    )
    
    logger.info(f"Test Accuracy: {test_acc:.2f}%")
    logger.info(f"Test Loss: {test_loss:.4f}")
    
    # Save detailed metrics
    metrics = {
        'model': model_name,
        'seed': effective_seed,
        'num_parameters': num_params,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'classification_report': report,
        'history': history
    }
    
    save_metrics(metrics, str(exp_dir / 'metrics.json'))
    
    # Save compact results table (similar to eval.py: results_table.csv)
    try:
        # Macro-averaged precision/recall/F1 from predictions
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels, test_preds, average='macro', zero_division=0
        )

        # Model size (FP32)
        model_size_fp32 = get_model_size_mb(model, precision='fp32')

        # Inference latency configuration
        eval_cfg = config.get('evaluation', {})
        input_size = config['model'].get('input_size', 224)
        input_shape = (3, input_size, input_size)
        warmup = eval_cfg.get('inference_warmup', 10)
        runs = eval_cfg.get('inference_runs', 100)
        batch_size_inf = eval_cfg.get('batch_size_inference', 1)

        mean_time, _ = measure_inference_time(
            model,
            input_shape,
            device,
            warmup=warmup,
            runs=runs,
            batch_size=batch_size_inf
        )

        results_table = {
            'Model': [model_name],
            'Accuracy (%)': [f"{test_acc:.2f}"],
            'Precision (%)': [f"{precision * 100:.2f}"],
            'Recall (%)': [f"{recall * 100:.2f}"],
            'F1-score (%)': [f"{f1 * 100:.2f}"],
            'Params (M)': [f"{num_params / 1e6:.2f}"],
            'Size (MB)': [f"{model_size_fp32:.2f}"],
            'Latency (ms)': [f"{mean_time:.2f}"],
        }

        df = pd.DataFrame(results_table)
        results_table_path = exp_dir / 'results_table.csv'
        df.to_csv(results_table_path, index=False)
        logger.info(f"Results table saved to {results_table_path}")
    except Exception as e:
        logger.warning(f"Failed to save results_table.csv: {e}")
    
    # Save classification report as text
    with open(exp_dir / 'classification_report.txt', 'w') as f:
        f.write(classification_report(test_labels, test_preds, target_names=class_names))
    
    # Plot training curves
    plot_training_curves(history, str(exp_dir / 'training_curves.png'))
    
    # Plot confusion matrix
    plot_confusion_matrix(
        test_labels, test_preds, class_names,
        str(exp_dir / 'confusion_matrix.png')
    )
    
    logger.info(f"Training complete! Results saved to {exp_dir}")
    return exp_dir, best_model_path


def main():
    parser = argparse.ArgumentParser(description='Train wheat disease classification model')
    parser.add_argument('--config', type=str, default='src/config.yaml',
                       help='Path to config file')
    parser.add_argument('--model', type=str, required=True,
                       choices=['resnet18', 'mobilenetv2', 'mobilenetv3_small', 'mobilenetv3_large', 'efficientnet_b0', 'ghostnet', 'shufflenetv2', 'proposed'],
                       help='Model to train')
    parser.add_argument('--experiment_dir', type=str, default='experiment',
                       help='Directory containing train/valid/test splits')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed (overrides config). Use e.g. 0, 42, 123 for multi-seed runs.')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    train_model(config, args.model, args.experiment_dir, seed=args.seed)


if __name__ == '__main__':
    main()

