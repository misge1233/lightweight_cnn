# Lightweight Deep Learning for Field-Level Wheat Disease Classification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

This repository provides the implementation accompanying the paper: **"Efficient and Robust Deep Learning for Field-Level Wheat Disease Classification on Resource-Constrained Devices"**. It contains a complete pipeline for training, evaluating, and deploying lightweight deep learning models for field-level wheat disease classification on resource-constrained devices.

---

## Features

- **5-class wheat disease classification**: Fusarium head blight, healthy, septoria, stem rust, yellow rust
- **Proposed lightweight model**: MobileNetV2-style inverted residual bottleneck blocks with depthwise convolution and Adaptive Channel Reduction (ACR)
- **Baseline comparisons**: ResNet-18, MobileNetV2/V3, EfficientNet-B0, ShuffleNetV2, GhostNet with identical training protocol
- **Reproducible pipeline**: Fixed seeds, deterministic splits, 80/10/10 train/valid/test
- **6-step ablation study**: Systematic hyperparameter and augmentation tuning (e.g. AdamW, RandAugment, SiLU)
- **Model compression**: Structured pruning and INT8 quantization with ONNX export
- **Calibration**: Expected Calibration Error (ECE) and Brier score for uncertainty quantification
- **Robustness & uncertainty**: Perturbation evaluation and confidence/coverage analysis
- **Error analysis**: Categorized misclassifications and model cards for transparency
- **Android deployment**: ONNX assets and on-device latency benchmarking (e.g. Samsung Galaxy A52)

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#expected-project-structure-after-running-the-pipeline)
- [Dataset Availability](#dataset-availability)
- [Usage](#usage)
- [Configuration](#configuration)
- [Model Architecture](#model-architecture)
- [Evaluation Metrics](#evaluation-metrics)
- [Results Tables & Model Cards](#results-tables)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Installation

### Prerequisites

- **Python** 3.8 or higher
- **CUDA** (optional, for GPU training)

### Requirements

```bash
pip install -r requirements.txt
```

### Optional Dependencies

For FLOPs estimation:
```bash
pip install thop  # or ptflops
```

For ONNX quantization and inference:
```bash
pip install onnx onnxruntime
```

For robustness evaluation (motion blur):
```bash
pip install opencv-python
```

For MobileNetV3 and EfficientNet (if not available in torchvision):
```bash
pip install timm
```

---

## Quick Start

From the project root:

```bash
# 1. Prepare data (80/10/10 splits, seed=42)
python src/data_prep.py --data_dir data --out_dir experiment --seed 42

# 2. Train the proposed lightweight model
python src/train.py --config src/config.yaml --model proposed

# 3. Evaluate (replace <run_dir> with your experiment/runs/<timestamp>_proposed folder)
python src/eval.py --config src/config.yaml --ckpt experiment/runs/<run_dir>/best_model.pth
```

For best accuracy, run the [ablation study](#step-25-systematic-performance-improvement-proposed-model) then train with the saved best config: `python src/train_best.py --best_config experiment/ablation_runs/best_config.yaml`.

---

## Expected project structure after running the pipeline

```
.
├── data/                              # NOT INCLUDED — source dataset (5 classes); see Dataset Availability
│   ├── fusarium_head_blight/
│   ├── healthy/
│   ├── septoria/
│   ├── stem_rust/
│   └── yellow_rust/
├── experiment/                        # Generated after running — splits, runs, ablation outputs
│   ├── train/                         # Training split (80%)
│   ├── valid/                         # Validation split (10%)
│   ├── test/                          # Test split (10%)
│   ├── runs/                          # Training runs and results
│   │   └── <timestamp>_<modelname>/
│   │       ├── best_model.pth
│   │       ├── evaluation_metrics.json
│   │       ├── robustness_results.csv
│   │       ├── uncertainty_results.csv
│   │       ├── error_analysis.json
│   │       ├── model_card.md
│   │       └── ...
│   ├── ablation_runs/                 # Ablation study results
│   │   ├── summary.csv
│   │   ├── best_config.yaml
│   │   └── <timestamp>_<step>_<variant>/
│   ├── dataset_summary.json
│   └── dataset_summary.csv
├── deployment/                        # Optional: Android assets & benchmark docs
│   └── ANDROID_BENCHMARK_README.md
├── src/
│   ├── config.yaml                    # Main configuration
│   ├── data_prep.py                   # Dataset preparation & splits
│   ├── train.py                       # Training script
│   ├── train_best.py                  # Train with best ablation config
│   ├── run_ablation.py                # 6-step performance improvement plan
│   ├── run_pruning_sweep.py           # Pruning sweep utilities
│   ├── visualize_ablation.py          # Ablation figures & tables
│   ├── augmentations.py               # MixUp, CutMix, RandAugment, etc.
│   ├── eval.py                        # Full evaluation (metrics, robustness, uncertainty)
│   ├── prune.py                       # Structured pruning
│   ├── quantize.py                    # INT8 quantization & ONNX
│   ├── export_onnx.py                 # FP32 ONNX export
│   ├── inference.py                   # Inference with confidence thresholding
│   ├── error_analysis.py              # Error categorization
│   ├── model_card.py                  # Model card generation
│   ├── generate_results_tables.py     # Paper-style results tables
│   ├── prepare_android_assets.py       # ONNX assets for Android
│   ├── merge_android_latency.py       # Merge Android latency CSV
│   ├── aggregate_multiseed_results.py # Multi-seed result aggregation
│   ├── utils.py                       # Shared utilities
│   └── models/
│       ├── __init__.py
│       ├── baselines.py               # ResNet-18, MobileNetV2/V3, EfficientNet-B0, ShuffleNetV2, GhostNet
│       └── proposed_lightweight.py    # Proposed lightweight architecture
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dataset Availability

The dataset used in the paper is **not included** in this repository. It is **available upon request**, subject to **EIAR approval** and applicable **institutional data-sharing policies**. Please contact the authors for data access requests and terms of use.

---

## Usage

### Step 1: Prepare Dataset

Rename images and create train/valid/test splits with deterministic ordering:

```bash
python src/data_prep.py --data_dir data --out_dir experiment --seed 42
```

This will:
- Rename all images to `<class>_<index>.<ext>` format with deterministic ordering
- Create 80/10/10 train/valid/test splits (fixed seed=42)
- Ensure no data leakage (each image appears in exactly one split)
- Generate `dataset_summary.json` and `dataset_summary.csv`

### Step 2: Train Models

#### Train Baseline Models

All baseline models use identical training protocol for fair comparison:

```bash
# ResNet-18
python src/train.py --config src/config.yaml --model resnet18

# MobileNetV2
python src/train.py --config src/config.yaml --model mobilenetv2

# MobileNetV3-Small
python src/train.py --config src/config.yaml --model mobilenetv3_small

# MobileNetV3-Large
python src/train.py --config src/config.yaml --model mobilenetv3_large

# EfficientNet-B0
python src/train.py --config src/config.yaml --model efficientnet_b0

# ShuffleNetV2 (x1.0)
python src/train.py --config src/config.yaml --model shufflenetv2

# GhostNet
python src/train.py --config src/config.yaml --model ghostnet
```

#### Train Proposed Lightweight Model

```bash
python src/train.py --config src/config.yaml --model proposed
```

**Training Configuration (consistent across all models):**
- Optimizer: AdamW
- Learning Rate: 3e-4
- Weight Decay: 5e-4
- Batch Size: 32
- Scheduler: 5-epoch warmup + cosine annealing
- Loss: CrossEntropyLoss
- Early Stopping: Patience=10, monitor validation loss
- Mixed Precision: Enabled (when GPU available)
- Epochs: Up to 50 (with early stopping)

**Training Outputs** (saved to `experiment/runs/<timestamp>_<modelname>/`):
- `best_model.pth`: Best checkpoint based on validation accuracy
- `metrics.json`: Training metrics (loss, accuracy per epoch)
- `training_curves.png`: Loss and accuracy curves
- `confusion_matrix.png`: Confusion matrix heatmap
- `classification_report.txt`: Detailed classification report

### Step 2.5: Systematic Performance Improvement (Proposed Model)

For the proposed lightweight model, a systematic 6-step performance improvement plan is available to optimize accuracy while maintaining lightweight constraints:

```bash
# Run complete improvement plan (Steps 1-6)
python src/run_ablation.py --config src/config.yaml --experiment_dir experiment

# Run specific steps
python src/run_ablation.py --start_step 1 --end_step 3  # Steps 1-3 only
```
**Performance Improvement Plan:**

The system executes 6 sequential steps, each building on the previous best configuration:

1. **Step 1 - Reduce Regularization:**
   - Label smoothing sweep: [0.1, 0.05, 0.0]
   - Dropout sweep: [(0.3,0.2,0.1), (0.2,0.1,0.0), (0.1,0.1,0.0)]
   - Selects best label smoothing first, then best dropout configuration

2. **Step 2 - Optimizer & LR Schedule Upgrade:**
   - Switches to AdamW optimizer
   - Grid search: lr ∈ {1e-4, 3e-4}, weight_decay ∈ {1e-4, 5e-4}
   - Adds 5-epoch warmup + cosine annealing (min_lr=1e-6)

3. **Step 3 - Data Mixing (MixUp/CutMix):**
   - MixUp (α=0.2) and CutMix (α=1.0)
   - Mix probability = 0.5
   - Compares with Step 2 best and keeps better configuration

4. **Step 4 - Stronger Augmentation:**
   - RandAugment (N=2, M=9)
   - RandomErasing (p=0.25)

5. **Step 5 - Preserve Early Spatial Detail:**
   - Option A: Stem stride = 1 (downsample later)
   - Option B: Input size = 256 (train), eval at 224
   - Keeps the better option

6. **Step 6 - Activation Upgrade:**
   - Replaces ReLU6 with SiLU activation

**Ablation Study Outputs:**

All results are saved to `experiment/ablation_runs/`:

- `summary.csv`: Complete results table with all hyperparameters and metrics
- `best_config.yaml`: Final best configuration (YAML format)
- `<timestamp>_<step>_<variant>/`: Individual run directories containing:
  - `config.yaml`: Run-specific configuration
  - `metrics.json`: Complete metrics (val acc, val F1, test acc, test F1, params, size, latency, etc.)
  - `best_model.pth`: Best checkpoint
  - `training_curves.png`: Training/validation curves
  - `confusion_matrix.png`: Confusion matrix visualization
  - `classification_report.txt`: Detailed classification report

**Features:**
- **Reproducible**: Fixed seed=42, deterministic flags (torch, numpy, random)
- **Systematic**: Each step builds on previous best configuration
- **Tracked**: CSV summary with all hyperparameters and metrics
- **Best Model Selection**: Validation accuracy + macro F1 tie-breaker
- **Complete Logging**: Training curves, confusion matrices, classification reports

**CSV Summary Columns:**
- step, variant, label_smoothing, dropout_set, optimizer, lr, weight_decay
- warmup_epochs, mixup_alpha, cutmix_alpha, mix_prob
- randaugment_N, randaugment_M, erase_p
- input_size, stem_stride, activation
- best_val_acc, best_val_f1, test_acc, test_f1, train_acc_at_best
- params_m, size_mb, latency_ms (efficiency metrics)

After completion, the system prints:
- Best configuration (all hyperparameters)
- Best validation accuracy/F1 and final test accuracy/F1
- Summary CSV location for detailed analysis

**📊 Generate Visualizations:**

Create publication-quality figures from ablation results:

```bash
python src/visualize_ablation.py --summary_csv experiment/ablation_runs/summary.csv
```

This generates:
- `ablation_trajectory.png/pdf`: Performance trajectory across steps
- `step_comparison.png/pdf`: Within-step variant comparisons
- `hyperparameter_heatmap.png/pdf`: Optimizer hyperparameter sweep (Step 2)
- `efficiency_scatter.png/pdf`: Accuracy vs efficiency metrics
- `ablation_summary_table.csv/tex`: Summary table for paper

#### Train Final Model with Best Configuration

After completing the ablation study, train the final model using the optimized configuration:

```bash
# Train with best config from ablation study
python src/train_best.py --best_config experiment/ablation_runs/best_config.yaml

# Override epochs if needed
python src/train_best.py --best_config experiment/ablation_runs/best_config.yaml --epochs 100
```

This script:
- Loads the best configuration from the ablation study
- Merges it with the base config
- Trains the proposed model with optimized hyperparameters
- Saves results to `experiment/runs/` with all evaluation outputs

### Step 3: Comprehensive Model Evaluation

Evaluate trained models with full metrics including robustness and uncertainty:

```bash
python src/eval.py --config src/config.yaml --ckpt <path_to_checkpoint>

# Example
python src/eval.py --config src/config.yaml --ckpt experiment/runs/20260108_204617_resnet18/best_model.pth
```

**Evaluation Outputs** (saved in checkpoint directory):

1. **Classification Metrics:**
   - `evaluation_metrics.json`: Complete metrics dictionary
   - `results_table.csv`: Performance summary table
   - `confusion_matrix.csv`: Confusion matrix (CSV format)

2. **Robustness Evaluation:**
   - `robustness_results.csv`: Performance under perturbations
     - Gaussian noise (σ=0.1)
     - Motion blur (kernel_size=9)
     - Brightness reduction (factor=0.7)
     - Contrast reduction (factor=0.7)
   - Reports accuracy, F1-score, and relative performance drop

3. **Uncertainty and Calibration:**
   - `uncertainty_results.csv`: Confidence-based selective prediction (coverage–accuracy) using softmax confidence
   - Calibration metrics: Expected Calibration Error (ECE), Brier score
   - Confidence histograms for correct vs incorrect predictions
   - (Optional, if enabled in code) Monte Carlo Dropout (T=10) for additional uncertainty estimates

4. **Error Analysis:**
   - `error_analysis.json`: Categorized error analysis
   - `error_summary.csv`: Error statistics by category
   - `confusion_pairs.csv`: Most common class confusion pairs
   - `error_examples/`: Directory with error example summaries
   - Categories:
     - Early-stage disease symptoms
     - Occlusion / background clutter
     - Inter-rust disease confusion (e.g., stem rust vs yellow rust)
     - Confusion with abiotic stress factors

5. **Model Documentation:**
   - `model_card.md`: Comprehensive model documentation
     - Dataset information
     - Architecture details
     - Performance metrics
     - Robustness and uncertainty results
     - Deployment notes and limitations

6. **Efficiency Metrics:**
   - Parameter count (millions)
   - Model size (FP32, MB)
   - Inference latency (CPU, mean ± std, ms)
   - FLOPs (Giga operations, if available)

### Step 4: Model Compression (Proposed Model)

#### Prune Model

Apply structured channel pruning with L1-norm importance:

```bash
python src/prune.py --config src/config.yaml --ckpt <path_to_proposed_model_checkpoint>
```

This:
- Prunes 20% of channels (configurable via `config.yaml`)
- Fine-tunes pruned model for 10 epochs
- Creates `pruned_model.pth` in the same directory

#### Quantize Model

Convert to INT8 quantization:

```bash
python src/quantize.py --config src/config.yaml --ckpt <path_to_checkpoint>
```

This generates (in the same directory as the checkpoint):
- **`model_quantized.onnx`**: Quantized ONNX model (INT8) — **use this for Android ONNX Runtime benchmarking and deployment**
- `model.onnx`: FP32 ONNX (intermediate export)
- `quantization_info.json`: Quantization statistics
- `quantization_summary.csv`: FP32 vs INT8 comparison (accuracy, macro_f1, size_mb, latency_ms)

**Note:** For custom models, ONNX dynamic quantization is preferred over PyTorch static quantization.

### Step 5: Export to ONNX

Export the trained (or pruned) model to **FP32 ONNX** for deployment:

```bash
# Proposed model with best config (replace <run> with your run folder, e.g. 20260310_003140_proposed_seed42)
python src/export_onnx.py --config experiment/runs/final_best_config.yaml --ckpt experiment/runs/<run>/best_model.pth --output experiment/runs/<run>/model.onnx

# Example with a concrete run folder:
python src/export_onnx.py --config experiment/runs/final_best_config.yaml --ckpt experiment/runs/20260310_003140_proposed_seed42/best_model.pth --output experiment/runs/20260310_003140_proposed_seed42/model.onnx
```

This creates **`model.onnx`** (FP32) in the checkpoint directory with:
- Dynamic batch size support
- Opset version 11
- Full model parameters included

**For Android / INT8 benchmarking:** Use the **quantized** ONNX produced by `quantize.py`, not `export_onnx.py`. Run quantize first (see Step 4); it writes **`model_quantized.onnx`** in the same run folder. Use that path for ONNX Runtime on Android and for the latency table (e.g. with `merge_android_latency.py`).

### Step 6: Run Inference

Test inference on a single image with optional confidence thresholding:

```bash
# FP32 ONNX (from export_onnx.py)
python src/inference.py --onnx_model experiment/runs/<run>/model.onnx --image test_image.jpg

# INT8 quantized ONNX (from quantize.py) — use for deployment / Android
python src/inference.py --onnx_model experiment/runs/<run>/model_quantized.onnx --image test_image.jpg

# With confidence thresholding
python src/inference.py --onnx_model experiment/runs/<run>/model.onnx --image test_image.jpg --confidence_threshold 0.7

# Specify class names
python src/inference.py \
    --onnx_model experiment/runs/20260310_003140_proposed_seed42/model.onnx \
    --image test_image.jpg \
    --classes fusarium_head_blight healthy septoria stem_rust yellow_rust \
    --confidence_threshold 0.6
```

**Output:**
- Predicted class and confidence score
- All class probabilities
- Warning if confidence is below threshold (uncertain prediction)

### Mobile deployment (Android)

On-device latency is measured on **Samsung Galaxy A52** (Snapdragon 720G, 6 GB RAM, Android 13) using **ONNX Runtime Mobile v1.16** with **CPU inference, batch size 1**.

1. **Prepare ONNX assets** (copy FP32/INT8 models into one folder and get a manifest):
   ```bash
   python src/prepare_android_assets.py --runs_dir experiment/runs --output_dir deployment/android_models
   ```
2. **Deploy** the `deployment/android_models/` contents to the device (e.g. via ADB or your app assets).
3. **Run the benchmark on device**: warm-up (e.g. 50 runs), then record 100–200 inference latencies (ms); compute **median** and **IQR**.
4. **Fill** `deployment/android_latency_template.csv` with Model, Precision, Device, Runtime, Median Latency (ms), IQR (ms).
5. **Normalize for the paper table**:
   ```bash
   python src/merge_android_latency.py deployment/android_latency_results.csv --output experiment/results/android_latency_table.csv
   ```

See **`deployment/ANDROID_BENCHMARK_README.md`** for device details, CSV format, and optional Kotlin benchmark snippet.

## Configuration

Edit `src/config.yaml` to customize:

- **Data Configuration:**
  - Data paths and split ratios
  - Random seed for reproducibility

- **Training Configuration:**
  - Epochs, batch size, learning rate
  - Optimizer and scheduler settings
  - Early stopping parameters
  - Mixed precision training

- **Data Augmentation:**
  - Training augmentations (flips, rotation, color jitter)
  - Validation/test transforms
  - Advanced augmentations (MixUp, CutMix, RandAugment, RandomErasing) - configured via ablation study

- **Robustness Evaluation:**
  - Perturbation types and parameters
  - Gaussian noise sigma
  - Motion blur kernel size
  - Brightness/contrast reduction factors

- **Uncertainty and Calibration:**
  - Confidence-based selective prediction (coverage–accuracy) using softmax confidence
  - ECE and Brier score computation
  - (Optional) Monte Carlo Dropout samples (default: 10) and dropout rate for MC sampling (default: 0.1)

- **Pruning Configuration:**
  - Pruning ratio (default: 0.2 = 20%)
  - Fine-tuning epochs after pruning
  - Importance metric (L1-norm)

- **Quantization Configuration:**
  - Method (static/dynamic)
  - Calibration sample ratio
  - Export ONNX flag

- **Evaluation Configuration:**
  - Inference warmup runs
  - Inference timed runs
  - Batch size for inference timing

## Model Architecture

### Proposed Lightweight Model

The proposed architecture comprises:
- **Lightweight stem**: Initial 3×3 convolution
- **MobileNetV2-style inverted residual bottleneck blocks with depthwise convolution**: Inverted residual bottlenecks with narrower expansion ratios (1.2–2.0)
- **Adaptive Channel Reduction (ACR)**: Channel counts preserved in early and mid layers; stronger reduction in later layers
- **Global average pooling and compact classifier**: Final feature aggregation with dropout

### Baseline Models

All baselines use ImageNet pretrained weights with transfer learning:

- **ResNet-18**: Standard residual network
- **MobileNetV2**: Mobile-optimized architecture
- **MobileNetV3-Small**: NAS-optimized mobile architecture
- **MobileNetV3-Large**: NAS-optimized mobile architecture
- **EfficientNet-B0**: Compound scaling architecture
- **ShuffleNetV2**: Channel shuffle for efficient computation
- **GhostNet**: Ghost modules for efficient feature maps (via timm)

## Evaluation Metrics

### Classification Metrics
- **Accuracy**: Overall classification accuracy (%)
- **Precision, Recall, F1-score**: Per-class and macro-averaged
- **Confusion Matrix**: Visualization of class confusion patterns

### Robustness Metrics
- **Performance under perturbations**: Accuracy and F1-score under:
  - Gaussian noise
  - Motion blur
  - Brightness reduction
  - Contrast reduction
- **Relative performance drop**: Percentage decrease vs clean test set

### Uncertainty and Calibration Metrics
- **Softmax confidence**: Maximum softmax probability as confidence score
- **Selective prediction (confidence-based)**: Coverage–accuracy curves using confidence thresholds
- **Calibration**: Expected Calibration Error (ECE) and Brier score
- **Monte Carlo Dropout (optional)**: Predictive uncertainty via T forward passes
- **Confidence histograms**: Distribution of confidence for correct vs incorrect predictions

### Efficiency Metrics
- **Parameter count**: Number of trainable parameters (millions)
- **Model size**: Disk storage footprint (MB) for FP32 and INT8
- **FLOPs**: Computational cost per forward pass (Giga operations)
- **Inference latency**: Mean ± std time on CPU (batch_size=1, ms)
- **Peak memory**: Runtime memory footprint (MB)

## Error Analysis

The pipeline provides comprehensive error analysis to identify and categorize misclassified samples:

**Error Categories:**
1. **Early-stage disease symptoms**: Weak/diffuse visual cues
2. **Occlusion / background clutter**: Objects obscuring disease features
3. **Inter-rust disease confusion**: Confusion between similar rust diseases (stem rust vs yellow rust)
4. **Abiotic stress confusion**: Confusion with environmental stress factors

**Error Analysis Outputs:**
- Error count and percentage by category
- Most common confusion pairs
- Example images per error category (optional)

## Ablation Study

### Performance Improvement Ablation (Proposed Model)

See **Step 2.5** above for the systematic 6-step performance improvement plan that optimizes hyperparameters, augmentation, and architecture choices.

### Compression Ablation (Full → Pruned → Quantized)

To generate compression ablation results:

1. **Train full model:**
   ```bash
   python src/train.py --config src/config.yaml --model proposed
   ```

2. **Evaluate full model:**
   ```bash
   python src/eval.py --config src/config.yaml --ckpt <full_model_checkpoint>
   ```

3. **Prune model:**
   ```bash
   python src/prune.py --config src/config.yaml --ckpt <full_model_checkpoint>
   ```

4. **Evaluate pruned model:**
   ```bash
   python src/eval.py --config src/config.yaml --ckpt <pruned_model_checkpoint>
   ```

5. **Quantize model:**
   ```bash
   python src/quantize.py --config src/config.yaml --ckpt <pruned_model_checkpoint>
   ```

6. **Evaluate quantized model:**
   Use ONNX runtime or convert back to PyTorch

**Compare metrics across all variants:**
- Accuracy / F1-score
- Parameter count
- Model size
- Inference latency

## Results Tables

After training and evaluation, results are automatically saved. To generate aggregated results tables matching the paper's format:

```bash
python src/generate_results_tables.py --runs_dir experiment/runs --output_dir experiment
```

This generates:
- `table1_overall_performance.csv`: Table 1 from paper (all models comparison)
- `table2_per_class_performance.csv`: Table 2 from paper (per-class metrics)
- `table3_efficiency_metrics.csv`: Table 3 from paper (efficiency comparison)
- `table4_ablation_study.csv`: Table 4 from paper (ablation study)

Each model evaluation also generates individual results in its run directory.

## Model Cards

Each evaluated model automatically generates a comprehensive `model_card.md` including:
- Model overview and architecture details
- Dataset information
- Classification performance (overall and per-class)
- Robustness evaluation results
- Uncertainty-aware inference results
- Error analysis summary
- Efficiency metrics
- Training configuration
- Deployment notes and limitations
- Recommendations for use

Model cards follow best practices for model documentation and transparency.

## Notes

- **Reproducibility**: All scripts use fixed random seeds (seed=42 by default) for deterministic results
- **GPU Support**: Automatically uses GPU if available, falls back to CPU
- **Mixed Precision**: Enabled by default for faster training on compatible GPUs (NVIDIA GPUs with Tensor Cores)
- **Early Stopping**: Prevents overfitting (patience=10 epochs, monitors validation loss)
- **Data Leakage Prevention**: Deterministic splits ensure no image appears in multiple sets
- **Deterministic Ordering**: Images are sorted before renaming and splitting for reproducibility

## Troubleshooting

### EfficientNet Import Error
If EfficientNet is not available in torchvision, install timm:
```bash
pip install timm
```

### MobileNetV3 Import Error
If MobileNetV3 is not available in torchvision, install timm:
```bash
pip install timm
```

### GhostNet / ShuffleNetV2 Import Error
GhostNet requires **timm**. ShuffleNetV2 uses torchvision when available, otherwise timm:
```bash
pip install timm
```

### ONNX Export Issues
- Ensure ONNX opset version is compatible (script uses opset 11 by default)
- For older PyTorch versions, you may need to adjust opset version in `export_onnx.py`

### Robustness Evaluation Issues
- If motion blur fails, install opencv-python: `pip install opencv-python`
- The script will fall back to PIL Gaussian blur if cv2 is not available

### Memory Issues
- Reduce batch size in `config.yaml`
- Reduce number of workers for data loading
- Use gradient accumulation for effective larger batch sizes

### CUDA Out of Memory
- Reduce batch size
- Use mixed precision training (enabled by default)
- Reduce model size or use a smaller baseline model

## Citation

If you use this code or the method in your work, please cite:

```bibtex
@unpublished{wheat-field-level-2026,
  title  = {Efficient and Robust Deep Learning for Field-Level Wheat Disease Classification on Resource-Constrained Devices},
  author = {Abetu, Misganu Tuse and Abebe, Teklu Urgessa and Tune, Kula Kakeba},
  year   = {2026},
  note   = {Submitted manuscript (under review)}
}
```

## License

This repository is shared for **academic review and reproducibility**. A formal open-source license will be added once finalized; until then, please contact the authors for reuse and redistribution beyond academic use.

## Contact

- **Issues & questions:** open an issue in the repository or contact **misganu.tuse@aastustudent.edu.et**

## Acknowledgments

This work was conducted in collaboration with the **Ethiopian Institute of Agricultural Research (EIAR)** for data collection and validation. The proposed architecture is designed for resource-constrained agricultural environments in Ethiopia.
