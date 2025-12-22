"""
检查光谱数据文件中每条光谱的长度
"""
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import h5py


def check_csv_lengths(csv_path, n_samples=5):
    """检查CSV文件中每条光谱的长度（只读取前n_samples条）"""
    print(f"\nChecking CSV file: {csv_path}")
    print("=" * 60)
    
    # 只读取前几行（第一行是x-axis，接下来是光谱数据）
    data = pd.read_csv(csv_path, header=None, nrows=n_samples + 1)  # +1 for x-axis row
    print(f"Read first {n_samples} spectra")
    print(f"Data shape: {data.shape}")
    
    # 第一行是x-axis
    x_axis_length = data.shape[1]
    print(f"\nX-axis length: {x_axis_length} points")
    
    # 检查每条光谱的长度
    lengths = []
    for i in range(1, data.shape[0]):  # 跳过第一行（x-axis）
        row = data.iloc[i].values
        # 检查是否有NaN或空值
        valid_length = np.sum(~np.isnan(row))
        lengths.append(valid_length)
        print(f"  Spectrum {i}: length = {valid_length}")
    
    if len(set(lengths)) == 1:
        print(f"\n✓ All checked spectra have the same length: {lengths[0]}")
    else:
        print(f"\n⚠️  Warning: Checked spectra have different lengths!")
        print(f"  Lengths: {lengths}")
    
    return lengths


def check_h5_lengths(h5_path, n_samples=5):
    """检查HDF5文件中每条光谱的长度（只读取前n_samples条）"""
    print(f"\nChecking HDF5 file: {h5_path}")
    print("=" * 60)
    
    with h5py.File(h5_path, 'r') as f:
        # 检查主要的数据集
        print("Available datasets:")
        for key in f.keys():
            dataset = f[key]
            print(f"  {key}: shape={dataset.shape}, dtype={dataset.dtype}")
        
        # 检查spectra数据集
        if 'spectra' in f:
            spectra = f['spectra']
            print(f"\nSpectra dataset:")
            print(f"  Total shape: {spectra.shape}")
            print(f"  Total number of spectra: {spectra.shape[0]}")
            print(f"  Each spectrum length: {spectra.shape[1]}")
            
            # 只读取前n_samples条光谱
            n_check = min(n_samples, spectra.shape[0])
            print(f"\nChecking first {n_check} spectra:")
            lengths = []
            for i in range(n_check):
                sample_spectrum = spectra[i]
                valid_length = np.sum(~np.isnan(sample_spectrum))
                lengths.append(valid_length)
                print(f"  Spectrum {i+1}: valid length = {valid_length}/{spectra.shape[1]}")
            
            if len(set(lengths)) == 1:
                print(f"\n✓ All checked spectra have the same length: {lengths[0]}")
            else:
                print(f"\n⚠️  Warning: Checked spectra have different lengths!")
                print(f"  Lengths: {lengths}")
            
            return spectra.shape[0], spectra.shape[1]
        else:
            print("  No 'spectra' dataset found")
            return None, None


def main():
    parser = argparse.ArgumentParser(description='Check spectrum lengths in data files (only reads first few samples)')
    parser.add_argument('--file', type=str, required=True,
                       help='Path to CSV or HDF5 file')
    parser.add_argument('--format', type=str, choices=['auto', 'csv', 'h5'], default='auto',
                       help='File format (auto-detect if not specified)')
    parser.add_argument('--n_samples', type=int, default=5,
                       help='Number of samples to check (default: 5)')
    
    args = parser.parse_args()
    
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return
    
    # 自动检测格式
    if args.format == 'auto':
        if file_path.suffix == '.h5' or file_path.suffix == '.hdf5':
            format_type = 'h5'
        elif file_path.suffix == '.csv':
            format_type = 'csv'
        else:
            print(f"Warning: Unknown file extension {file_path.suffix}, trying CSV format")
            format_type = 'csv'
    else:
        format_type = args.format
    
    # 检查文件
    if format_type == 'csv':
        lengths = check_csv_lengths(file_path, n_samples=args.n_samples)
    elif format_type == 'h5':
        n_spectra, spectrum_length = check_h5_lengths(file_path, n_samples=args.n_samples)
        if n_spectra is not None:
            print(f"\nSummary:")
            print(f"  Total spectra in file: {n_spectra}")
            print(f"  Each spectrum length: {spectrum_length}")
    else:
        print(f"Unknown format: {format_type}")


if __name__ == '__main__':
    main()

