"""
Data preparation script for wheat disease classification.
Renames images and creates train/valid/test splits.
"""

import os
import shutil
import argparse
import json
from pathlib import Path
from typing import Dict, List
import random
import numpy as np
from collections import defaultdict

from utils import set_seed


def get_image_files(class_dir: Path) -> List[Path]:
    """Get all image files from a directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    files = [f for f in class_dir.iterdir() if f.suffix in image_extensions and f.is_file()]
    return sorted(files)  # Deterministic ordering


def rename_images_in_class(class_dir: Path, class_name: str) -> Dict[str, str]:
    """
    Rename all images in a class directory to <class>_<index>.<ext>.
    Returns mapping of old_name -> new_name.
    """
    files = get_image_files(class_dir)
    rename_map = {}
    
    for idx, file_path in enumerate(files):
        ext = file_path.suffix.lower()
        # Normalize extension
        if ext in ['.jpg', '.jpeg']:
            ext = '.jpg'
        elif ext == '.png':
            ext = '.png'
        else:
            ext = '.jpg'  # default
        
        new_name = f"{class_name}_{idx}{ext}"
        new_path = class_dir / new_name
        
        # Handle potential duplicates
        if new_path.exists() and new_path != file_path:
            # Find next available index
            counter = idx
            while new_path.exists():
                counter += 1
                new_name = f"{class_name}_{counter}{ext}"
                new_path = class_dir / new_name
        
        if file_path != new_path:
            file_path.rename(new_path)
            rename_map[str(file_path.name)] = new_name
        else:
            rename_map[file_path.name] = new_name
    
    return rename_map


def split_class_images(
    class_dir: Path,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    seed: int
) -> Dict[str, List[str]]:
    """
    Split images in a class directory into train/valid/test.
    Returns dict with 'train', 'valid', 'test' keys containing lists of filenames.
    """
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    files = get_image_files(class_dir)  # Already sorted for deterministic ordering
    # Use set_seed for full reproducibility across modules
    random.seed(seed)
    np.random.seed(seed)
    random.shuffle(files)  # Deterministic shuffle with fixed seed
    
    n_total = len(files)
    n_train = int(n_total * train_ratio)
    n_valid = int(n_total * valid_ratio)
    # n_test = n_total - n_train - n_valid (to handle rounding)
    
    splits = {
        'train': files[:n_train],
        'valid': files[n_train:n_train + n_valid],
        'test': files[n_train + n_valid:]
    }
    
    return splits


def copy_to_splits(
    class_name: str,
    splits: Dict[str, List[Path]],
    source_dir: Path,
    output_dir: Path
):
    """Copy images to respective split directories."""
    for split_name, file_list in splits.items():
        split_dir = output_dir / split_name / class_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        for file_path in file_list:
            dest_path = split_dir / file_path.name
            shutil.copy2(file_path, dest_path)


def prepare_dataset(
    data_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
):
    """
    Main function to prepare the dataset:
    1. Rename images in each class
    2. Create train/valid/test splits
    3. Copy to output directory
    4. Generate summary statistics
    """
    set_seed(seed)
    
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    
    # Create output directory structure
    for split in ['train', 'valid', 'test']:
        (output_path / split).mkdir(parents=True, exist_ok=True)
    
    # Warn if splits already exist
    if any((output_path / split).exists() and any((output_path / split).iterdir()) for split in ['train', 'valid', 'test']):
        response = input(f"Warning: {output_path} already contains data. Continue? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Get all class directories
    class_dirs = [d for d in data_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    class_names = sorted([d.name for d in class_dirs])
    
    print(f"Found {len(class_names)} classes: {class_names}")
    
    # Statistics
    stats = {
        'classes': class_names,
        'splits': {},
        'total_images': 0
    }
    
    # Process each class
    for class_name in class_names:
        class_dir = data_path / class_name
        print(f"\nProcessing class: {class_name}")
        
        # Step 1: Rename images
        print(f"  Renaming images in {class_name}...")
        rename_map = rename_images_in_class(class_dir, class_name)
        print(f"  Renamed {len(rename_map)} images")
        
        # Step 2: Split images
        print(f"  Splitting images...")
        splits = split_class_images(class_dir, train_ratio, valid_ratio, test_ratio, seed)
        
        # Step 3: Copy to output directories
        print(f"  Copying to split directories...")
        copy_to_splits(class_name, splits, class_dir, output_path)
        
        # Update statistics
        class_stats = {
            'train': len(splits['train']),
            'valid': len(splits['valid']),
            'test': len(splits['test']),
            'total': len(splits['train']) + len(splits['valid']) + len(splits['test'])
        }
        stats['splits'][class_name] = class_stats
        stats['total_images'] += class_stats['total']
        
        print(f"  {class_name}: train={class_stats['train']}, "
              f"valid={class_stats['valid']}, test={class_stats['test']}")
    
    # Save summary
    summary_file = output_path / "dataset_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Also create CSV summary
    import csv
    csv_file = output_path / "dataset_summary.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Class', 'Train', 'Valid', 'Test', 'Total'])
        for class_name in class_names:
            s = stats['splits'][class_name]
            writer.writerow([class_name, s['train'], s['valid'], s['test'], s['total']])
        writer.writerow(['TOTAL', 
                        sum(stats['splits'][c]['train'] for c in class_names),
                        sum(stats['splits'][c]['valid'] for c in class_names),
                        sum(stats['splits'][c]['test'] for c in class_names),
                        stats['total_images']])
    
    print(f"\n{'='*60}")
    print("Dataset preparation complete!")
    print(f"{'='*60}")
    print(f"Output directory: {output_path}")
    print(f"Total images: {stats['total_images']}")
    print(f"Summary saved to: {summary_file}")
    print(f"CSV summary saved to: {csv_file}")
    print(f"\nSplit distribution:")
    for class_name in class_names:
        s = stats['splits'][class_name]
        print(f"  {class_name:20s}: train={s['train']:4d}, valid={s['valid']:4d}, test={s['test']:4d}")


def main():
    parser = argparse.ArgumentParser(description='Prepare wheat disease dataset')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Source data directory')
    parser.add_argument('--out_dir', type=str, default='experiment',
                       help='Output directory for splits')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Training set ratio')
    parser.add_argument('--valid_ratio', type=float, default=0.1,
                       help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                       help='Test set ratio')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    prepare_dataset(
        data_dir=args.data_dir,
        output_dir=args.out_dir,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )


if __name__ == '__main__':
    main()

