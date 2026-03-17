"""
Generate model card (model_card.md) for trained models.
Model cards provide comprehensive documentation of dataset, model, results, and deployment notes.
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, List
import json


def generate_model_card(metrics: Dict, config: Dict, class_names: List[str], output_dir: Path) -> Path:
    """Generate a comprehensive model card in Markdown format."""
    output_dir = Path(output_dir)
    
    model_name = metrics.get('model', 'Unknown')
    test_acc = metrics.get('test_accuracy', 0)
    macro_f1 = metrics.get('macro_f1', 0)
    macro_precision = metrics.get('macro_precision', 0)
    macro_recall = metrics.get('macro_recall', 0)
    
    efficiency = metrics.get('efficiency', {})
    num_params = efficiency.get('num_parameters_millions', 0)
    model_size = efficiency.get('model_size_fp32_mb', 0)
    latency = efficiency.get('inference_time_ms_mean', 0)
    flops = efficiency.get('flops_g', None)
    
    robustness = metrics.get('robustness', [])
    uncertainty = metrics.get('uncertainty', {})
    error_analysis = metrics.get('error_analysis', {})
    
    # Generate model card content
    card_content = f"""# Model Card: {model_name}

## Model Overview

**Model Name:** {model_name}  
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Task:** Wheat Disease Classification  
**Number of Classes:** {len(class_names)}

## Dataset Information

### Classes
{chr(10).join(f"- {class_name}" for class_name in class_names)}

### Dataset Statistics
- **Total Images:** ~{3000} (estimated based on ~600 images per class)
- **Split:** 80% train / 10% validation / 10% test
- **Image Size:** 224×224 pixels
- **Data Augmentation:** Random flips, rotations, color jitter

### Data Source
Field-collected wheat disease images from Ethiopian agricultural regions, 
capturing real-world variability in lighting, background, and plant growth stages.

## Model Architecture

### Architecture Details
- **Input Size:** {config['model']['input_size']}×{config['model']['input_size']}
- **Number of Parameters:** {num_params:.2f}M
- **Model Size (FP32):** {model_size:.2f} MB
- **FLOPs:** {flops:.2f} G if flops else "Not measured"}
- **Architecture Type:** {"Proposed Lightweight CNN" if "proposed" in model_name.lower() else "Transfer Learning Baseline"}

## Performance Metrics

### Classification Performance
- **Test Accuracy:** {test_acc:.2f}%
- **Macro Precision:** {macro_precision*100:.2f}%
- **Macro Recall:** {macro_recall*100:.2f}%
- **Macro F1-Score:** {macro_f1*100:.2f}%

### Per-Class Performance
"""
    
    # Add per-class metrics
    per_class = metrics.get('per_class_metrics', {})
    if per_class:
        card_content += "\n| Class | Precision | Recall | F1-Score |\n"
        card_content += "|-------|-----------|--------|----------|\n"
        for class_name, m in per_class.items():
            card_content += f"| {class_name} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} |\n"
    
    # Add efficiency metrics
    card_content += f"""
### Efficiency Metrics
- **Inference Latency (CPU):** {latency:.2f} ± {efficiency.get('inference_time_ms_std', 0):.2f} ms
- **Parameters:** {num_params:.2f}M
- **Model Size:** {model_size:.2f} MB

"""
    
    # Add robustness results
    if robustness:
        card_content += "### Robustness Evaluation\n\n"
        card_content += "| Perturbation | Accuracy (%) | F1-Score | Relative Drop (%) |\n"
        card_content += "|--------------|--------------|----------|-------------------|\n"
        for result in robustness:
            pert_name = result.get('perturbation', 'Unknown')
            acc = result.get('accuracy', 0)
            f1 = result.get('f1_score', 0)
            drop = result.get('relative_drop_acc', 0)
            card_content += f"| {pert_name} | {acc:.2f} | {f1:.4f} | {drop:.2f} |\n"
        card_content += "\n"
    
    # Add uncertainty results
    if uncertainty:
        card_content += "### Uncertainty-Aware Evaluation\n\n"
        mean_conf_correct = uncertainty.get('mean_confidence_correct', 0)
        mean_conf_incorrect = uncertainty.get('mean_confidence_incorrect', 0)
        mean_mc_conf = uncertainty.get('mean_mc_confidence', 0)
        
        card_content += f"- **Mean Confidence (Correct):** {mean_conf_correct:.4f}\n"
        card_content += f"- **Mean Confidence (Incorrect):** {mean_conf_incorrect:.4f}\n"
        card_content += f"- **Mean MC Dropout Confidence:** {mean_mc_conf:.4f}\n"
        card_content += "\n"
    
    # Add error analysis
    if error_analysis:
        error_summary = error_analysis.get('error_summary', {})
        if error_summary:
            card_content += "### Error Analysis\n\n"
            card_content += "| Error Category | Count | Percentage (%) |\n"
            card_content += "|----------------|-------|----------------|\n"
            for error_type, stats in error_summary.items():
                card_content += f"| {error_type.replace('_', ' ').title()} | {stats['count']} | {stats['percentage']:.2f} |\n"
            card_content += "\n"
    
    # Add training configuration
    card_content += f"""## Training Configuration

- **Optimizer:** {config['training']['optimizer']}
- **Learning Rate:** {config['training']['learning_rate']}
- **Batch Size:** {config['training']['batch_size']}
- **Epochs:** {config['training']['epochs']}
- **Scheduler:** {config['training']['scheduler']}
- **Early Stopping Patience:** {config['training']['early_stopping']['patience']}
- **Mixed Precision:** {config['training']['mixed_precision']}

"""
    
    # Add deployment notes
    card_content += """## Deployment Notes

### Deployment Requirements
- **Hardware:** CPU (mobile-friendly) or GPU for faster inference
- **Framework:** PyTorch or ONNX Runtime
- **Memory:** ~{model_size:.2f} MB for FP32 model
- **Input Format:** RGB image, 224×224 pixels, normalized with ImageNet statistics

### Inference
- Load model checkpoint and run inference on single images
- Use ONNX export for cross-platform deployment
- Apply same preprocessing (resize, normalize) as training

### Limitations
- Model trained on field-collected Ethiopian wheat disease images
- Performance may vary on images from different regions or varieties
- Early-stage disease symptoms may have lower accuracy
- Inter-rust disease confusion (stem rust vs yellow rust) can occur

### Recommendations
- Use confidence thresholding for uncertain predictions
- For deployment, use quantized (INT8) version for reduced size
- Consider ensemble with multiple models for critical applications
- Regularly retrain with new data from target deployment region

## Model Variants

### Compression Variants
1. **FP32 (Full Precision):** Original model
2. **Pruned:** Structured pruning applied (20% channels removed)
3. **Quantized (INT8):** Post-training quantization
4. **Pruned + Quantized:** Final compressed version

## Citation

If you use this model, please cite:
```
[Paper citation to be added]
```

## License

[License information to be added]

## Contact

For questions or issues, please contact: misganu.tuse@aastustudent.edu.et

---
*Model card generated automatically by the evaluation pipeline.*
"""
    
    # Save model card
    model_card_path = output_dir / 'model_card.md'
    with open(model_card_path, 'w', encoding='utf-8') as f:
        f.write(card_content)
    
    return model_card_path


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate model card')
    parser.add_argument('--metrics', type=str, required=True,
                       help='Path to evaluation_metrics.json')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config.yaml')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for model card')
    
    args = parser.parse_args()
    
    # Load metrics and config
    import yaml
    from utils import load_config
    
    with open(args.metrics, 'r') as f:
        metrics = json.load(f)
    
    config = load_config(args.config)
    class_names = config['model']['classes']
    
    model_card_path = generate_model_card(metrics, config, class_names, Path(args.output_dir))
    print(f"Model card saved to {model_card_path}")
