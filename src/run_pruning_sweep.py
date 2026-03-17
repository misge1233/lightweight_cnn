"""
Run pruning at multiple ratios (0.0, 0.1, 0.2, 0.3, 0.4) and collect metrics for trade-off figure.
For proposed model only. Saves pruning_sweep_summary.csv, pruning_tradeoff.png, pruning_tradeoff.pdf.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import load_config


def main():
    parser = argparse.ArgumentParser(description='Pruning sweep: run prune+eval for multiple ratios')
    parser.add_argument('--config', type=str, default='src/config.yaml', help='Path to config file')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to proposed model checkpoint')
    parser.add_argument('--experiment_dir', type=str, default='experiment',
                        help='Directory containing train/valid/test')
    parser.add_argument('--ratios', type=float, nargs='+', default=[0.0, 0.1, 0.2, 0.3, 0.4],
                        help='Pruning ratios to evaluate')
    args = parser.parse_args()
    
    config = load_config(args.config)
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)
    
    experiment_dir = Path(args.experiment_dir)
    src_dir = Path(__file__).resolve().parent
    project_root = src_dir.parent  # run subprocesses from project root so config paths resolve
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    sweep_dir = experiment_dir / 'runs' / f'pruning_sweep_{timestamp}'
    sweep_dir.mkdir(parents=True, exist_ok=True)
    
    rows = []
    for ratio in args.ratios:
        work_dir = sweep_dir / f'ratio_{ratio}'
        work_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dest = work_dir / 'best_model.pth'
        if ratio == 0:
            shutil.copy2(args.ckpt, ckpt_dest)
            eval_ckpt = ckpt_dest
        else:
            shutil.copy2(args.ckpt, ckpt_dest)
            # Write config with this pruning ratio
            config_override = dict(config)
            config_override['pruning'] = dict(config.get('pruning', {}))
            config_override['pruning']['pruning_ratio'] = ratio
            config_path = work_dir / 'config.yaml'
            import yaml
            with open(config_path, 'w') as f:
                yaml.dump(config_override, f)
            # Run prune
            cmd_prune = [
                sys.executable, str(src_dir / 'prune.py'),
                '--config', str(config_path),
                '--ckpt', str(ckpt_dest),
                '--experiment_dir', str(experiment_dir)
            ]
            subprocess.run(cmd_prune, check=True, cwd=str(project_root))
            eval_ckpt = work_dir / 'pruned_model.pth'
            if not eval_ckpt.exists():
                print(f"Warning: pruned model not found at {eval_ckpt}, skipping ratio {ratio}")
                continue
        
        # Run eval
        cmd_eval = [
            sys.executable, str(src_dir / 'eval.py'),
            '--config', str(args.config),
            '--ckpt', str(eval_ckpt),
            '--experiment_dir', str(experiment_dir)
        ]
        subprocess.run(cmd_eval, check=True, cwd=str(project_root))
        
        metrics_path = work_dir / 'evaluation_metrics.json'
        if not metrics_path.exists():
            metrics_path = work_dir / 'metrics.json'
        if not metrics_path.exists():
            print(f"Warning: no metrics at {work_dir}")
            continue
        with open(metrics_path, 'r') as f:
            m = json.load(f)
        acc = m.get('test_accuracy', m.get('test_acc', 0))
        if isinstance(acc, str):
            acc = float(acc.replace('%', ''))
        macro_f1 = m.get('macro_f1', 0)
        if macro_f1 < 1 and macro_f1 > 0:
            macro_f1 *= 100
        eff = m.get('efficiency', {})
        params_m = eff.get('num_parameters_millions') or (m.get('num_parameters', 0) / 1e6)
        size_mb = eff.get('model_size_fp32_mb') or 0
        latency_ms = eff.get('inference_time_ms_mean') or 0
        rows.append({
            'pruning_ratio': ratio,
            'accuracy': round(acc, 2),
            'macro_f1': round(macro_f1, 2),
            'params_m': round(params_m, 2),
            'size_mb': round(size_mb, 2),
            'latency_ms': round(latency_ms, 2),
        })
    
    if not rows:
        print("No results collected.")
        sys.exit(1)
    
    df = pd.DataFrame(rows)
    summary_path = sweep_dir / 'pruning_sweep_summary.csv'
    df.to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")
    
    # Plot: thin lines + dashed F1 then solid Acc so both stay visible when values are close
    lw_metric, ms = 1.0, 4
    fig, ax1 = plt.subplots(figsize=(5, 4))
    ax1.plot(df['pruning_ratio'], df['macro_f1'], 'g--s', label='Macro-F1 (%)', linewidth=lw_metric, markersize=ms, zorder=2)
    ax1.plot(df['pruning_ratio'], df['accuracy'], 'b-o', label='Accuracy (%)', linewidth=lw_metric, markersize=ms, zorder=3)
    ax1.set_xlabel('Pruning ratio')
    ax1.set_ylabel('Accuracy / Macro-F1 (%)')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    y_lo = min(df['accuracy'].min(), df['macro_f1'].min()) - 0.8
    y_hi = max(df['accuracy'].max(), df['macro_f1'].max()) + 0.8
    ax1.set_ylim(y_lo, y_hi)
    ax2 = ax1.twinx()
    ax2.plot(df['pruning_ratio'], df['latency_ms'], 'r--^', label='Latency (ms)', linewidth=1.5)
    ax2.set_ylabel('Latency (ms)')
    ax2.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(sweep_dir / 'pruning_tradeoff.png', dpi=150, bbox_inches='tight')
    fig.savefig(sweep_dir / 'pruning_tradeoff.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved pruning_tradeoff.png and .pdf to {sweep_dir}")
    print(df.to_string())


if __name__ == '__main__':
    main()
