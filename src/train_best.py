"""
Train the proposed model with the best configuration from ablation study.
Uses the optimized hyperparameters from experiment/ablation_runs/best_config.yaml
"""

import argparse
import yaml
import subprocess
import sys
from pathlib import Path
from utils import load_config


def merge_configs(base_config: dict, best_config: dict) -> dict:
    """Merge base config with best config. Preserves all required sections."""
    merged = base_config.copy()
    for key in ('model', 'training', 'augmentation', 'data', 'normalization',
                'evaluation', 'output', 'pruning', 'quantization', 'robustness', 'uncertainty'):
        if key in best_config:
            if key not in merged:
                merged[key] = {}
            if isinstance(merged[key], dict) and isinstance(best_config[key], dict):
                merged[key] = {**merged[key], **best_config[key]}
            else:
                merged[key] = best_config[key]
    return merged


def is_standalone_config(c: dict) -> bool:
    """True if config has required keys for training."""
    return all(c.get(k) for k in ('model', 'training', 'augmentation', 'data', 'normalization'))


def main():
    parser = argparse.ArgumentParser(description='Train proposed model with best ablation configuration')
    parser.add_argument('--base_config', type=str, default='src/config.yaml',
                       help='Path to base config file')
    parser.add_argument('--config', type=str, default=None,
                       help='Use this config as-is (e.g. experiment/runs/final_best_config.yaml). Overrides base+best.')
    parser.add_argument('--best_config', type=str, default='experiment/runs/final_best_config.yaml',
                       help='Path to best config (used when --config not set)')
    parser.add_argument('--experiment_dir', type=str, default='experiment',
                       help='Directory containing train/valid/test splits')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed (e.g. 0, 42, 123). Passed to train.py.')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of epochs (optional)')
    
    args = parser.parse_args()
    
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"❌ Config not found: {args.config}")
            sys.exit(1)
        final_config = load_config(str(config_path))
        if not is_standalone_config(final_config):
            print("⚠ Config may be incomplete. Ensure model, training, augmentation, data, normalization are set.")
        print(f"Using config as-is: {args.config}")
    else:
        base_config = load_config(args.base_config)
        best_config_path = Path(args.best_config)
        if not best_config_path.exists():
            best_config_path = Path('experiment/ablation_runs/best_config.yaml')
        if not best_config_path.exists():
            print(f"❌ Best config not found. Use --config experiment/runs/final_best_config.yaml")
            sys.exit(1)
        best_config = load_config(str(best_config_path))
        final_config = merge_configs(base_config, best_config)
        if args.epochs is not None:
            final_config.setdefault('training', {})['epochs'] = args.epochs
        output_dir = Path('experiment') / 'runs'
        output_dir.mkdir(parents=True, exist_ok=True)
        merged_config_path = output_dir / 'final_best_config.yaml'
        with open(merged_config_path, 'w') as f:
            yaml.dump(final_config, f, default_flow_style=False)
        print(f"✅ Merged config saved to: {merged_config_path}")
        config_path = merged_config_path
        print(f"Loading from {args.base_config} + {best_config_path}")
    
    if args.epochs is not None and args.config:
        final_config.setdefault('training', {})['epochs'] = args.epochs
    print("")
    print("=" * 80)
    print("TRAINING PROPOSED MODEL (BEST CONFIG)")
    print("=" * 80)
    print("")
    print("Model Configuration:")
    print(f"  Input Size: {final_config['model'].get('input_size', 224)}")
    print(f"  Dropout: {final_config['model'].get('dropout_set', [0.3, 0.2, 0.1])}")
    print(f"  Activation: {final_config['model'].get('activation', 'relu6')}")
    print(f"  Stem Stride: {final_config['model'].get('stem_stride', 2)}")
    print("")
    print("Training Configuration:")
    print(f"  Optimizer: {final_config['training'].get('optimizer', 'adam')}")
    print(f"  Learning Rate: {final_config['training'].get('learning_rate', 1e-4)}")
    opt_params = final_config['training'].get('optimizer_params', {})
    print(f"  Weight Decay: {opt_params.get('weight_decay', 1e-4)}")
    sched_params = final_config['training'].get('scheduler_params', {})
    print(f"  Warmup Epochs: {sched_params.get('warmup_epochs', 0)}")
    print(f"  Label Smoothing: {final_config['training'].get('label_smoothing', 0.1)}")
    print(f"  Epochs: {final_config['training']['epochs']}")
    print(f"  Batch Size: {final_config['training']['batch_size']}")
    print("")
    print("Augmentation Configuration:")
    print(f"  MixUp Alpha: {final_config['augmentation'].get('mixup_alpha', 0.0)}")
    print(f"  CutMix Alpha: {final_config['augmentation'].get('cutmix_alpha', 0.0)}")
    print(f"  RandAugment N: {final_config['augmentation'].get('randaugment_N', 0)}")
    print(f"  RandAugment M: {final_config['augmentation'].get('randaugment_M', 0)}")
    print(f"  Random Erasing p: {final_config['augmentation'].get('erase_p', 0.0)}")
    print("")
    print("=" * 80)
    print("")
    
    # Build command to call train.py
    cmd = [
        sys.executable,
        'src/train.py',
        '--config', str(config_path),
        '--model', 'proposed',
        '--experiment_dir', args.experiment_dir
    ]
    if args.seed is not None:
        cmd += ['--seed', str(args.seed)]
    
    # Train
    print("Starting training...")
    print(f"Command: {' '.join(cmd)}")
    print("")
    
    result = subprocess.run(cmd, cwd=Path.cwd())
    
    if result.returncode != 0:
        print("")
        print("=" * 80)
        print("❌ TRAINING FAILED")
        print("=" * 80)
        print(f"Training exited with code: {result.returncode}")
        sys.exit(result.returncode)
    
    print("")
    print("=" * 80)
    print("✅ TRAINING COMPLETE")
    print("=" * 80)
    print("")
    print(f"Model trained with best configuration from ablation study.")
    print(f"Results saved to: experiment/runs/<timestamp>_proposed/")
    print("")
    print("Next steps (use same config, e.g. experiment/runs/final_best_config.yaml):")
    print("  1. Evaluate: python src/eval.py --config experiment/runs/final_best_config.yaml --ckpt experiment/runs/<run>/best_model.pth --experiment_dir experiment [--seed 42]")
    print("  2. Pruning sweep: python src/run_pruning_sweep.py --config experiment/runs/final_best_config.yaml --ckpt experiment/runs/<run>/best_model.pth --experiment_dir experiment")
    print("  3. Quantize: python src/quantize.py --config experiment/runs/final_best_config.yaml --ckpt experiment/runs/<run>/best_model.pth --experiment_dir experiment")
    print("  4. Export ONNX: python src/export_onnx.py --config experiment/runs/final_best_config.yaml --ckpt experiment/runs/<run>/best_model.pth [--output experiment/runs/<run>/model.onnx]")


if __name__ == '__main__':
    main()
