"""
Utility functions for the wheat disease classification pipeline.
"""

import os
import json
import random
import logging
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml
from datetime import datetime


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def setup_logging(log_dir: str, name: str = "train") -> logging.Logger:
    """Setup logging to both file and console."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(name)
    return logger


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: torch.nn.Module, precision: str = "fp32") -> float:
    """Calculate model size in MB."""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        if precision == "int8":
            param_size += param.numel() * 1  # 1 byte per parameter
        else:
            param_size += param.numel() * 4  # 4 bytes per parameter (float32)
    
    for buffer in model.buffers():
        buffer_size += buffer.numel() * 4
    
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    return size_mb


def measure_inference_time(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    device: torch.device,
    warmup: int = 10,
    runs: int = 100,
    batch_size: int = 1
) -> Tuple[float, float]:
    """
    Measure inference time in milliseconds.
    Returns (mean, std) in ms.
    """
    model.eval()
    model.to(device)
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, *input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    
    # Synchronize GPU if available
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Measure inference time
    times = []
    with torch.no_grad():
        for _ in range(runs):
            if device.type == 'cuda':
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = model(dummy_input)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))  # in milliseconds
            else:
                import time
                start = time.time()
                _ = model(dummy_input)
                end = time.time()
                times.append((end - start) * 1000)  # convert to milliseconds
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    
    return mean_time, std_time


def save_metrics(metrics: Dict, filepath: str):
    """Save metrics dictionary to JSON file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)


def create_experiment_dir(base_dir: str, model_name: str, seed: Optional[int] = None) -> Path:
    """Create a timestamped experiment directory. Optionally include seed in name (e.g. ..._ghostnet_seed42)."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    name_part = model_name
    if seed is not None:
        name_part = f"{model_name}_seed{seed}"
    exp_dir = Path(base_dir) / f"{timestamp}_{name_part}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def get_device() -> torch.device:
    """Get the appropriate device (CUDA if available, else CPU)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

