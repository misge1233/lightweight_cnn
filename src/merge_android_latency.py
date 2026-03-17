"""
Ingest and normalize Android ONNX Runtime latency CSV for paper tables.
Reads a CSV with columns like: Model, Precision, Device, Runtime, Median Latency (ms), IQR (ms).
Saves normalized android_latency_table.csv. Does not run Android benchmarking.
"""

import argparse
from pathlib import Path
import pandas as pd


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to standard: Model, Precision, Device, Runtime, Median Latency (ms), IQR (ms)."""
    col_map = {}
    for c in df.columns:
        c_lower = c.strip().lower()
        if c_lower == 'model':
            col_map[c] = 'Model'
        elif 'precision' in c_lower or 'dtype' in c_lower:
            col_map[c] = 'Precision'
        elif 'device' in c_lower:
            col_map[c] = 'Device'
        elif 'runtime' in c_lower:
            col_map[c] = 'Runtime'
        elif 'median' in c_lower and 'latency' in c_lower:
            col_map[c] = 'Median Latency (ms)'
        elif 'iqr' in c_lower or 'interquartile' in c_lower:
            col_map[c] = 'IQR (ms)'
        elif 'latency' in c_lower and 'median' not in c_lower:
            col_map[c] = 'Median Latency (ms)'
    if col_map:
        df = df.rename(columns=col_map)
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Merge/normalize Android latency CSV for paper tables'
    )
    parser.add_argument('input_csv', type=str,
                       help='Input CSV path (e.g. from manual Android benchmarking)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path (default: same dir as input, name android_latency_table.csv)')
    args = parser.parse_args()
    
    input_path = Path(args.input_csv)
    if not input_path.exists():
        print(f"File not found: {input_path}")
        return
    
    df = pd.read_csv(input_path)
    df = normalize_column_names(df)
    
    # Ensure expected columns exist; create if missing
    for col in ['Model', 'Precision', 'Device', 'Runtime', 'Median Latency (ms)', 'IQR (ms)']:
        if col not in df.columns:
            df[col] = ''
    
    # Select and order columns
    out_cols = ['Model', 'Precision', 'Device', 'Runtime', 'Median Latency (ms)', 'IQR (ms)']
    df = df[[c for c in out_cols if c in df.columns]]
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / 'android_latency_table.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved normalized table to {output_path}")
    print(df.to_string())


if __name__ == '__main__':
    main()
