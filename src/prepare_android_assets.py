"""
Prepare ONNX models for Android deployment and latency benchmarking.
Scans experiment/runs for model.onnx (FP32) and model_quantized.onnx (INT8),
copies them to a single directory with unique names, and writes a manifest CSV.
Use the manifest when filling android_latency_template.csv with on-device results.
"""

import argparse
import shutil
from pathlib import Path
import pandas as pd


# Map path/folder patterns to paper model names
MODEL_NAME_MAP = {
    'proposed': 'Proposed',
    'mobilenetv3_small': 'MobileNetV3-Small',
    'mobilenetv3_large': 'MobileNetV3-Large',
    'mobilenetv2': 'MobileNetV2',
    'ghostnet': 'GhostNet',
    'shufflenetv2': 'ShuffleNetV2',
    'resnet18': 'ResNet-18',
    'efficientnet_b0': 'EfficientNet-B0',
}


def _infer_model_name(run_path: Path) -> str:
    """Infer paper model name from run directory or path."""
    name = run_path.name.lower()
    for key, display in MODEL_NAME_MAP.items():
        if key in name:
            return display
    # Fallback: use a sanitized folder name (e.g. ratio_0.2 -> proposed if from proposed run)
    if 'ratio' in name or 'pruning' in name:
        return 'Proposed'
    return run_path.name.replace('_', '-')[:30]


def _discover_onnx(runs_dir: Path) -> list:
    """Return list of (run_path, onnx_path, precision)."""
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return []
    out = []
    for onnx_name in ('model.onnx', 'model_quantized.onnx'):
        precision = 'INT8' if 'quantized' in onnx_name else 'FP32'
        for path in runs_dir.rglob(onnx_name):
            if not path.is_file():
                continue
            # run_path = parent of the folder containing the onnx (e.g. ratio_0.2 or 20260310_proposed_seed42)
            run_path = path.parent
            out.append((run_path, path, precision))
    return out


def main():
    parser = argparse.ArgumentParser(
        description='Gather ONNX models from experiment runs for Android deployment'
    )
    parser.add_argument(
        '--runs_dir',
        type=str,
        default='experiment/runs',
        help='Root directory containing run folders (and optionally pruning_sweep_*)',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='deployment/android_models',
        help='Output directory for copied ONNX files and manifest.csv',
    )
    parser.add_argument(
        '--no-copy',
        action='store_true',
        help='Do not copy files; only write manifest from existing ONNX paths under output_dir.',
    )
    args = parser.parse_args()
    runs_dir = Path(args.runs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    discovered = _discover_onnx(runs_dir)
    if not discovered:
        print(f"No ONNX files found under {runs_dir}")
        return

    manifest_rows = []
    count_by_key = {}  # (model_display, precision) -> number of times seen
    for run_path, onnx_path, precision in discovered:
        model_display = _infer_model_name(run_path)
        key = (model_display, precision)
        count_by_key[key] = count_by_key.get(key, 0) + 1
        n = count_by_key[key]
        base = f"{model_display.replace(' ', '_').replace('-', '_')}_{precision.lower()}"
        dest_name = f"{base}.onnx" if n == 1 else f"{base}_{n}.onnx"
        dest_path = output_dir / dest_name
        if not args.no_copy:
            shutil.copy2(onnx_path, dest_path)
            print(f"Copied {onnx_path} -> {dest_path}")
        manifest_rows.append({
            'Model': model_display,
            'Precision': precision,
            'Source': str(onnx_path),
            'Filename': dest_name,
        })

    manifest_path = output_dir / 'manifest.csv'
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print(f"Manifest written to {manifest_path}")
    print("\nUse deployment/android_latency_template.csv and fill Median Latency (ms) and IQR (ms)")
    print("after running on-device benchmarks. Then:")
    print("  python src/merge_android_latency.py <your_filled_csv> --output experiment/results/android_latency_table.csv")


if __name__ == '__main__':
    main()
