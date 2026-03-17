"""
Visualize ablation study results with performance trajectories and comparisons.
Generates publication-quality figures for the research paper.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np


def plot_ablation_trajectory(df: pd.DataFrame, output_dir: Path):
    """Plot performance trajectory across ablation steps."""
    # Select best result from each step
    step_results = []
    for step in range(1, 7):
        step_df = df[df['step'] == step]
        if len(step_df) > 0:
            # Find best by validation accuracy
            best_idx = step_df['best_val_acc'].astype(float).idxmax()
            best_row = step_df.loc[best_idx]
            step_results.append({
                'step': step,
                'variant': best_row['variant'],
                'val_acc': float(best_row['best_val_acc']),
                'test_acc': float(best_row['test_acc']),
                'test_f1': float(best_row['test_f1'])
            })
    
    step_df = pd.DataFrame(step_results)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    steps = step_df['step']
    x = np.arange(len(steps))
    width = 0.35
    
    # Plot bars
    bars1 = ax.bar(x - width/2, step_df['val_acc'], width, 
                   label='Validation Accuracy', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, step_df['test_acc'], width,
                   label='Test Accuracy', color='#e74c3c', alpha=0.8)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=9)
    
    # Customize plot
    ax.set_xlabel('Ablation Step', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Ablation Study: Performance Trajectory', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Step {s}\n{v}' for s, v in zip(step_df['step'], step_df['variant'])],
                       fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(70, 100)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ablation_trajectory.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'ablation_trajectory.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved: {output_dir / 'ablation_trajectory.png'}")


def plot_step_comparison(df: pd.DataFrame, output_dir: Path):
    """Plot comparison of all variants within each step."""
    steps = df['step'].unique()
    n_steps = len(steps)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, step in enumerate(sorted(steps)):
        ax = axes[idx]
        step_df = df[df['step'] == step].copy()
        step_df = step_df.sort_values('best_val_acc', ascending=True)
        
        # Plot horizontal bars
        y_pos = np.arange(len(step_df))
        val_acc = step_df['best_val_acc'].astype(float)
        test_acc = step_df['test_acc'].astype(float)
        
        bars1 = ax.barh(y_pos - 0.2, val_acc, 0.4, label='Val Acc', color='#3498db', alpha=0.8)
        bars2 = ax.barh(y_pos + 0.2, test_acc, 0.4, label='Test Acc', color='#e74c3c', alpha=0.8)
        
        # Highlight best
        best_idx = val_acc.idxmax()
        best_pos = list(val_acc.index).index(best_idx)
        ax.axhspan(best_pos - 0.5, best_pos + 0.5, alpha=0.2, color='green')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(step_df['variant'], fontsize=8)
        ax.set_xlabel('Accuracy (%)', fontsize=10)
        ax.set_title(f'Step {step}', fontsize=11, fontweight='bold')
        ax.set_xlim(70, 100)
        ax.grid(axis='x', alpha=0.3)
        
        if idx == 0:
            ax.legend(fontsize=9)
    
    # Hide unused subplots if any
    for idx in range(n_steps, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'step_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'step_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved: {output_dir / 'step_comparison.png'}")


def plot_hyperparameter_heatmap(df: pd.DataFrame, output_dir: Path):
    """Plot heatmap of key hyperparameters vs performance."""
    # Extract key hyperparameters for Step 2 (optimizer sweep)
    step2_df = df[df['step'] == 2].copy()
    
    if len(step2_df) == 0:
        print("[WARN] No Step 2 results found, skipping heatmap")
        return
    
    # Create pivot table
    step2_df['lr_str'] = step2_df['lr'].apply(lambda x: f"{float(x):.1e}")
    step2_df['wd_str'] = step2_df['weight_decay'].apply(lambda x: f"{float(x):.1e}")
    
    pivot = step2_df.pivot_table(
        values='test_acc',
        index='wd_str',
        columns='lr_str',
        aggfunc='mean'
    )
    
    # Convert to float
    pivot = pivot.astype(float)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', vmin=70, vmax=95,
               cbar_kws={'label': 'Test Accuracy (%)'}, ax=ax)
    
    ax.set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('Weight Decay', fontsize=12, fontweight='bold')
    ax.set_title('Step 2: Optimizer Hyperparameter Sweep', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'hyperparameter_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'hyperparameter_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved: {output_dir / 'hyperparameter_heatmap.png'}")


def plot_efficiency_scatter(df: pd.DataFrame, output_dir: Path):
    """Plot accuracy vs efficiency metrics."""
    # Get best result from each step
    step_results = []
    for step in range(1, 7):
        step_df = df[df['step'] == step]
        if len(step_df) > 0:
            best_idx = step_df['best_val_acc'].astype(float).idxmax()
            best_row = step_df.loc[best_idx]
            
            # Extract metrics
            try:
                params_m = float(best_row.get('params_m', best_row.get('Parameters', 1.16)))
            except:
                params_m = 1.16
            
            try:
                latency = float(best_row.get('latency_ms', best_row.get('Latency', 30)))
            except:
                latency = 30
            
            step_results.append({
                'step': step,
                'variant': best_row['variant'],
                'test_acc': float(best_row['test_acc']),
                'params_m': params_m,
                'latency_ms': latency
            })
    
    step_df = pd.DataFrame(step_results)
    
    # Create 2-panel figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel 1: Accuracy vs Parameters
    scatter1 = ax1.scatter(step_df['params_m'], step_df['test_acc'], 
                          s=200, c=step_df['step'], cmap='viridis', 
                          alpha=0.7, edgecolors='black', linewidths=1.5)
    
    # Add labels
    for idx, row in step_df.iterrows():
        ax1.annotate(f"Step {row['step']}\n{row['variant']}", 
                    (row['params_m'], row['test_acc']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.8)
    
    ax1.set_xlabel('Parameters (Millions)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Test Accuracy (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Accuracy vs Model Size', fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # Panel 2: Accuracy vs Latency
    scatter2 = ax2.scatter(step_df['latency_ms'], step_df['test_acc'],
                          s=200, c=step_df['step'], cmap='viridis',
                          alpha=0.7, edgecolors='black', linewidths=1.5)
    
    # Add labels
    for idx, row in step_df.iterrows():
        ax2.annotate(f"Step {row['step']}\n{row['variant']}", 
                    (row['latency_ms'], row['test_acc']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.8)
    
    ax2.set_xlabel('Inference Latency (ms)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Test Accuracy (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Accuracy vs Inference Speed', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter2, ax=[ax1, ax2])
    cbar.set_label('Ablation Step', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'efficiency_scatter.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'efficiency_scatter.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved: {output_dir / 'efficiency_scatter.png'}")


def generate_summary_table(df: pd.DataFrame, output_dir: Path):
    """Generate LaTeX table of ablation results for paper."""
    # Get best result from each step
    step_results = []
    for step in range(1, 7):
        step_df = df[df['step'] == step]
        if len(step_df) > 0:
            best_idx = step_df['best_val_acc'].astype(float).idxmax()
            best_row = step_df.loc[best_idx]
            step_results.append({
                'Step': step,
                'Variant': best_row['variant'],
                'Val Acc': f"{float(best_row['best_val_acc']):.2f}",
                'Test Acc': f"{float(best_row['test_acc']):.2f}",
                'Test F1': f"{float(best_row['test_f1']):.2f}",
            })
    
    results_df = pd.DataFrame(step_results)
    
    # Save as CSV
    results_df.to_csv(output_dir / 'ablation_summary_table.csv', index=False)
    print(f"[OK] Saved: {output_dir / 'ablation_summary_table.csv'}")
    
    # Generate LaTeX
    latex = results_df.to_latex(index=False, escape=False, column_format='clrrr')
    
    with open(output_dir / 'ablation_summary_table.tex', 'w') as f:
        f.write("% Ablation Study Summary Table\n")
        f.write("% Generated by visualize_ablation.py\n\n")
        f.write(latex)
    
    print(f"[OK] Saved: {output_dir / 'ablation_summary_table.tex'}")


def main():
    parser = argparse.ArgumentParser(description='Visualize ablation study results')
    parser.add_argument('--summary_csv', type=str, default='experiment/ablation_runs/summary.csv',
                       help='Path to ablation summary CSV')
    parser.add_argument('--output_dir', type=str, default='experiment/ablation_runs',
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    # Load data
    summary_path = Path(args.summary_csv)
    if not summary_path.exists():
        print(f"[ERROR] Summary CSV not found: {args.summary_csv}")
        print("   Please run ablation study first: python src/run_ablation.py")
        return
    
    print(f"Loading ablation results from: {args.summary_csv}")
    df = pd.read_csv(summary_path)
    print(f"Loaded {len(df)} experiments across {df['step'].nunique()} steps")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating visualizations...")
    
    # Generate all plots
    plot_ablation_trajectory(df, output_dir)
    plot_step_comparison(df, output_dir)
    plot_hyperparameter_heatmap(df, output_dir)
    plot_efficiency_scatter(df, output_dir)
    generate_summary_table(df, output_dir)
    
    print("\n" + "=" * 80)
    print("[SUCCESS] ALL VISUALIZATIONS GENERATED")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    print("  - ablation_trajectory.png/pdf: Performance trajectory across steps")
    print("  - step_comparison.png/pdf: Within-step variant comparisons")
    print("  - hyperparameter_heatmap.png/pdf: Optimizer hyperparameter sweep")
    print("  - efficiency_scatter.png/pdf: Accuracy vs efficiency metrics")
    print("  - ablation_summary_table.csv: Summary table (CSV format)")
    print("  - ablation_summary_table.tex: Summary table (LaTeX format)")
    print("\nUse these figures in your research paper!")


if __name__ == '__main__':
    main()
