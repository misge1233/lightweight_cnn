"""
Error analysis utilities for identifying and categorizing misclassified samples.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from collections import defaultdict
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import seaborn as sns


def categorize_errors(predictions: np.ndarray, labels: np.ndarray, 
                     class_names: List[str], image_paths: List[str]) -> Dict:
    """
    Categorize misclassified samples into error types.
    
    Error categories:
    1. Early-stage disease symptoms (weak/diffuse visual cues)
    2. Occlusion / background clutter
    3. Inter-rust disease confusion (e.g., stem rust vs yellow rust)
    4. Confusion with abiotic stress factors
    """
    errors = defaultdict(list)
    
    # Get misclassified indices
    misclassified_mask = predictions != labels
    misclassified_indices = np.where(misclassified_mask)[0]
    
    print(f"Total misclassified samples: {len(misclassified_indices)}")
    
    # Categorize errors by class confusion
    confusion_categories = defaultdict(int)
    
    for idx in misclassified_indices:
        true_label = labels[idx]
        pred_label = predictions[idx]
        true_class = class_names[true_label]
        pred_class = class_names[pred_label]
        
        # Track class confusion pairs
        confusion_pair = f"{true_class} -> {pred_class}"
        confusion_categories[confusion_pair] += 1
        
        # Categorize based on class pairs
        if 'rust' in true_class.lower() and 'rust' in pred_class.lower():
            # Inter-rust disease confusion
            error_type = 'inter_rust_confusion'
        elif true_class == 'healthy' and 'rust' in pred_class.lower():
            # Healthy misclassified as rust (could be abiotic stress)
            error_type = 'abiotic_stress_confusion'
        elif 'rust' in true_class.lower() and pred_class == 'healthy':
            # Rust misclassified as healthy (could be early-stage)
            error_type = 'early_stage_disease'
        elif true_class == 'healthy' and pred_class != 'healthy':
            # Healthy misclassified as disease
            error_type = 'abiotic_stress_confusion'
        else:
            # General confusion
            error_type = 'general_confusion'
        
        errors[error_type].append({
            'index': int(idx),
            'true_class': true_class,
            'pred_class': pred_class,
            'image_path': image_paths[idx] if idx < len(image_paths) else None,
            'confusion_pair': confusion_pair
        })
    
    # Summary statistics
    error_summary = {}
    for error_type, samples in errors.items():
        error_summary[error_type] = {
            'count': len(samples),
            'percentage': (len(samples) / len(misclassified_indices)) * 100 if len(misclassified_indices) > 0 else 0
        }
    
    return {
        'errors_by_category': dict(errors),
        'error_summary': error_summary,
        'confusion_pairs': dict(confusion_categories),
        'total_misclassified': len(misclassified_indices)
    }


def analyze_class_confusions(predictions: np.ndarray, labels: np.ndarray,
                            class_names: List[str]) -> pd.DataFrame:
    """Analyze confusion patterns between classes."""
    confusions = defaultdict(lambda: defaultdict(int))
    
    for pred, true in zip(predictions, labels):
        pred_class = class_names[pred]
        true_class = class_names[true]
        confusions[true_class][pred_class] += 1
    
    # Create confusion matrix dataframe
    confusion_matrix = []
    for true_class in class_names:
        row = {'True Class': true_class}
        for pred_class in class_names:
            row[pred_class] = confusions[true_class].get(pred_class, 0)
        confusion_matrix.append(row)
    
    return pd.DataFrame(confusion_matrix)


def save_error_examples(error_analysis: Dict, output_dir: Path, 
                       max_examples_per_category: int = 5):
    """Save example images for each error category."""
    output_dir = Path(output_dir)
    examples_dir = output_dir / 'error_examples'
    examples_dir.mkdir(parents=True, exist_ok=True)
    
    errors_by_category = error_analysis['errors_by_category']
    
    # Save error examples summary
    summary = []
    for error_type, samples in errors_by_category.items():
        if len(samples) > 0:
            # Take first N examples
            examples = samples[:max_examples_per_category]
            summary.append({
                'error_type': error_type,
                'count': len(samples),
                'examples': [s['image_path'] for s in examples]
            })
    
    with open(examples_dir / 'error_examples_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Error examples summary saved to {examples_dir / 'error_examples_summary.json'}")


def generate_error_report(predictions: np.ndarray, labels: np.ndarray,
                         class_names: List[str], image_paths: List[str],
                         output_dir: Path):
    """Generate comprehensive error analysis report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Categorize errors
    error_analysis = categorize_errors(predictions, labels, class_names, image_paths)
    
    # Save error analysis JSON
    with open(output_dir / 'error_analysis.json', 'w') as f:
        json.dump(error_analysis, f, indent=2)
    
    # Generate confusion analysis
    confusion_df = analyze_class_confusions(predictions, labels, class_names)
    confusion_df.to_csv(output_dir / 'confusion_analysis.csv', index=False)
    
    # Save error summary
    error_summary = error_analysis['error_summary']
    summary_rows = []
    for error_type, stats in error_summary.items():
        summary_rows.append({
            'Error Type': error_type.replace('_', ' ').title(),
            'Count': stats['count'],
            'Percentage (%)': f"{stats['percentage']:.2f}"
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / 'error_summary.csv', index=False)
    
    # Save confusion pairs
    confusion_pairs = error_analysis['confusion_pairs']
    pairs_rows = []
    for pair, count in sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True):
        pairs_rows.append({
            'Confusion Pair': pair,
            'Count': count
        })
    
    pairs_df = pd.DataFrame(pairs_rows)
    pairs_df.to_csv(output_dir / 'confusion_pairs.csv', index=False)
    
    # Save error examples
    save_error_examples(error_analysis, output_dir)
    
    # Print summary
    print(f"\n{'='*60}")
    print("Error Analysis Summary")
    print(f"{'='*60}")
    print(f"Total misclassified: {error_analysis['total_misclassified']}")
    print(f"\nErrors by category:")
    for error_type, stats in error_analysis['error_summary'].items():
        print(f"  {error_type:30s}: {stats['count']:4d} ({stats['percentage']:.2f}%)")
    
    print(f"\nTop confusion pairs:")
    for i, (pair, count) in enumerate(sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)[:10]):
        print(f"  {i+1:2d}. {pair:40s}: {count:4d}")
    
    print(f"\nError analysis saved to {output_dir}")
    
    return error_analysis


def create_failure_case_figure(
    predictions: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    image_paths: List[str],
    output_dir: Path,
    max_examples: int = 6,
    figsize_per_panel: Tuple[float, float] = (4, 4)
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Generate a grid figure of representative misclassified examples for paper.
    Saves failure_cases.png and failure_cases.pdf. Robust to missing image paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    misclassified_mask = predictions != labels
    misclassified_indices = np.where(misclassified_mask)[0]
    if len(misclassified_indices) == 0:
        print("No misclassified samples for failure-case figure.")
        return None, None
    
    # Build list of (idx, true_class, pred_class, error_category)
    errors_by_category = defaultdict(list)
    for idx in misclassified_indices:
        true_label = labels[idx]
        pred_label = predictions[idx]
        true_class = class_names[true_label]
        pred_class = class_names[pred_label]
        if 'rust' in true_class.lower() and 'rust' in pred_class.lower():
            error_type = 'inter_rust_confusion'
        elif true_class == 'healthy' and 'rust' in pred_class.lower():
            error_type = 'abiotic_stress_confusion'
        elif 'rust' in true_class.lower() and pred_class == 'healthy':
            error_type = 'early_stage_disease'
        elif true_class == 'healthy' and pred_class != 'healthy':
            error_type = 'abiotic_stress_confusion'
        else:
            error_type = 'general_confusion'
        errors_by_category[error_type].append({
            'index': int(idx),
            'true_class': true_class,
            'pred_class': pred_class,
            'error_category': error_type,
            'image_path': image_paths[idx] if idx < len(image_paths) else None
        })
    
    # Select representative examples across categories (up to max_examples)
    selected = []
    for cat, samples in sorted(errors_by_category.items(), key=lambda x: -len(x[1])):
        for s in samples[: max(1, max_examples // len(errors_by_category))]:
            if len(selected) >= max_examples:
                break
            selected.append(s)
        if len(selected) >= max_examples:
            break
    if len(selected) < 2:
        selected = [errors_by_category[k][0] for k in errors_by_category if errors_by_category[k]][:max_examples]
    
    n = min(len(selected), max_examples)
    if n == 0:
        return None, None
    
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, ex in enumerate(selected):
        ax = axes[i]
        img_path = ex.get('image_path')
        if img_path and Path(img_path).exists():
            try:
                img = Image.open(img_path).convert('RGB')
                ax.imshow(img)
            except Exception:
                ax.text(0.5, 0.5, 'Image unavailable', ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'Image unavailable', ha='center', va='center', transform=ax.transAxes)
        true_c = ex.get('true_class', '?').replace('_', ' ')
        pred_c = ex.get('pred_class', '?').replace('_', ' ')
        cat = ex.get('error_category', '?').replace('_', ' ')
        ax.set_title(f"True: {true_c}\nPred: {pred_c}\n[{cat}]", fontsize=9)
        ax.axis('off')
    
    for j in range(len(selected), len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    png_path = output_dir / 'failure_cases.png'
    pdf_path = output_dir / 'failure_cases.pdf'
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Failure-case figure saved to {png_path} and {pdf_path}")
    return png_path, pdf_path


def main():
    """Standalone error analysis from saved predictions."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze model errors')
    parser.add_argument('--predictions', type=str, required=True,
                       help='Path to predictions JSON or numpy file')
    parser.add_argument('--labels', type=str, required=True,
                       help='Path to labels JSON or numpy file')
    parser.add_argument('--class_names', type=str, nargs='+',
                       default=['fusarium_head_blight', 'healthy', 'septoria', 'stem_rust', 'yellow_rust'],
                       help='Class names')
    parser.add_argument('--image_paths', type=str, default=None,
                       help='Path to image paths list (JSON or text file)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for error analysis')
    
    args = parser.parse_args()
    
    # Load predictions and labels
    predictions = np.load(args.predictions) if args.predictions.endswith('.npy') else np.array(json.load(open(args.predictions)))
    labels = np.load(args.labels) if args.labels.endswith('.npy') else np.array(json.load(open(args.labels)))
    
    # Load image paths if provided
    image_paths = []
    if args.image_paths:
        if args.image_paths.endswith('.json'):
            image_paths = json.load(open(args.image_paths))
        else:
            with open(args.image_paths, 'r') as f:
                image_paths = [line.strip() for line in f.readlines()]
    
    generate_error_report(
        predictions, labels, args.class_names, image_paths, args.output_dir
    )


if __name__ == '__main__':
    main()
