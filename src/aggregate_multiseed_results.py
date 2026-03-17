"""
Aggregate evaluation results from multiple runs (multi-seed) into a summary CSV.
Used for proposed model, ACR uniform baseline, and MobileNetV3-Small multi-seed stability.
Generic: scans runs_dir for evaluation_metrics.json (or metrics.json), groups by model name.
"""

import argparse
import json
from pathlib import Path
import pandas as pd
from collections import defaultdict
import numpy as np


def _normalize_model_name(name: str) -> str:
    """Normalize model name for grouping (e.g. strip seed suffix)."""
    if not name:
        return "unknown"
    n = name.lower()
    # Map common variants to canonical name for grouping
    if 'proposed' in n or 'lightweight' in n:
        return 'proposed'
    if 'resnet' in n:
        return 'resnet18'
    if 'mobilenetv3_large' in n or 'mobilenetv3large' in n:
        return 'mobilenetv3_large'
    if 'mobilenetv3_small' in n or 'mobilenetv3small' in n:
        return 'mobilenetv3_small'
    if 'mobilenetv2' in n or 'mobilenet' in n:
        return 'mobilenetv2'
    if 'efficientnet' in n:
        return 'efficientnet_b0'
    if 'ghostnet' in n or 'ghost_net' in n:
        return 'ghostnet'
    if 'shufflenet' in n:
        return 'shufflenetv2'
    return n


def _display_name(model_key: str) -> str:
    """Display name for table."""
    mapping = {
        'proposed': 'Proposed',
        'resnet18': 'ResNet-18',
        'mobilenetv2': 'MobileNetV2',
        'mobilenetv3_small': 'MobileNetV3-Small',
        'mobilenetv3_large': 'MobileNetV3-Large',
        'efficientnet_b0': 'EfficientNet-B0',
        'ghostnet': 'GhostNet',
        'shufflenetv2': 'ShuffleNetV2',
    }
    return mapping.get(model_key, model_key)


def load_run_metrics(runs_dir: str) -> list:
    """Load metrics from all run directories. Returns list of (run_dir, metrics, model_name, seed)."""
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        return []
    rows = []
    for run_dir in runs_path.iterdir():
        if not run_dir.is_dir():
            continue
        for mfile in ('evaluation_metrics.json', 'metrics.json'):
            metrics_path = run_dir / mfile
            if not metrics_path.exists():
                continue
            try:
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
            except Exception:
                continue
            model_name = metrics.get('model', run_dir.name)
            seed = metrics.get('seed')
            # Prefer evaluation metrics for accuracy/F1
            acc = metrics.get('test_accuracy', metrics.get('test_acc', 0))
            if isinstance(acc, str):
                acc = float(acc.replace('%', '')) if acc else 0
            macro_f1 = metrics.get('macro_f1', 0)
            macro_precision = metrics.get('macro_precision', 0)
            macro_recall = metrics.get('macro_recall', 0)
            if (macro_f1 == 0 and 'classification_report' in metrics and 'macro avg' in metrics.get('classification_report', {})):
                r = metrics['classification_report']['macro avg']
                macro_precision = r.get('precision', 0)
                macro_recall = r.get('recall', 0)
                macro_f1 = r.get('f1-score', 0)
            if macro_f1 < 1 and macro_f1 > 0:
                macro_f1 *= 100
            if macro_precision < 1 and macro_precision > 0:
                macro_precision *= 100
            if macro_recall < 1 and macro_recall > 0:
                macro_recall *= 100
            rows.append({
                'run_dir': str(run_dir),
                'model_raw': model_name,
                'model_key': _normalize_model_name(str(model_name)),
                'seed': seed,
                'accuracy': float(acc),
                'macro_f1': float(macro_f1),
                'macro_precision': float(macro_precision),
                'macro_recall': float(macro_recall),
            })
            break
    return rows


def aggregate_multiseed(rows: list) -> pd.DataFrame:
    """Group by model_key, compute mean and std over seeds. One row per model."""
    by_model = defaultdict(list)
    for r in rows:
        by_model[r['model_key']].append(r)
    out = []
    for model_key, group in by_model.items():
        if len(group) == 0:
            continue
        accs = [x['accuracy'] for x in group]
        f1s = [x['macro_f1'] for x in group]
        precs = [x['macro_precision'] for x in group]
        recs = [x['macro_recall'] for x in group]
        seeds = [x['seed'] for x in group if x['seed'] is not None]
        if not seeds:
            seeds = [f"run_{i}" for i in range(len(group))]
        out.append({
            'Model': _display_name(model_key),
            'Seeds': ','.join(str(s) for s in seeds),
            'Accuracy Mean': round(np.mean(accs), 2),
            'Accuracy Std': round(np.std(accs), 2) if len(accs) > 1 else 0.0,
            'Macro-F1 Mean': round(np.mean(f1s), 2),
            'Macro-F1 Std': round(np.std(f1s), 2) if len(f1s) > 1 else 0.0,
            'Macro-Precision Mean': round(np.mean(precs), 2),
            'Macro-Precision Std': round(np.std(precs), 2) if len(precs) > 1 else 0.0,
            'Macro-Recall Mean': round(np.mean(recs), 2),
            'Macro-Recall Std': round(np.std(recs), 2) if len(recs) > 1 else 0.0,
        })
    return pd.DataFrame(out)


def main():
    parser = argparse.ArgumentParser(description='Aggregate multi-seed evaluation results')
    parser.add_argument('--runs_dir', type=str, default='experiment/runs',
                       help='Directory containing run subdirs with evaluation_metrics.json or metrics.json')
    parser.add_argument('--output_dir', type=str, default='experiment/results',
                       help='Output directory for multiseed_summary.csv')
    args = parser.parse_args()
    
    rows = load_run_metrics(args.runs_dir)
    if not rows:
        print(f"No metrics found under {args.runs_dir}")
        return
    
    # Group by (model_key, seed) then aggregate by model_key
    df = aggregate_multiseed(rows)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'multiseed_summary.csv'
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    print(df.to_string())


if __name__ == '__main__':
    main()
