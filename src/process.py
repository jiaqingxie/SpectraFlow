"""
数据处理模块：将CSV文件处理成3600个点的光谱数据集
处理三个数据集：ir_broaden, uv_broaden, raman_broaden
支持保存为HDF5格式以加速后续加载
"""
import pandas as pd
import numpy as np
from scipy import interpolate
import os
from pathlib import Path
import h5py
from utils import compute_params


def process_spectrum_to_3600(spectrum, target_size=3600):
    """
    将光谱数据插值到3600个点
    
    Args:
        spectrum: 1D numpy array，原始光谱数据
        target_size: 目标长度，默认3600
    
    Returns:
        插值后的3600个点的光谱数据
    """
    if len(spectrum) == target_size:
        return spectrum
    
    # 使用线性插值
    x_old = np.linspace(0, 1, len(spectrum))
    x_new = np.linspace(0, 1, target_size)
    f = interpolate.interp1d(x_old, spectrum, kind='linear', 
                            fill_value='extrapolate', bounds_error=False)
    return f(x_new)


def load_and_process_csv(csv_path, target_size=3600):
    """
    加载CSV文件并处理成3600个点的数据集
    
    Args:
        csv_path: CSV文件路径
        target_size: 目标光谱长度，默认3600
    
    Returns:
        processed_data: 处理后的数据数组 (n_samples, 3600)
        x_axis: x轴数据
    """
    print(f"Loading CSV file: {csv_path}")
    
    # 读取CSV文件
    df = pd.read_csv(csv_path, header=None)
    
    # 第一行通常是x轴数据
    x_axis = df.iloc[0, 1:].values.astype(np.float32)
    
    # 剩余行是光谱数据
    spectra_data = df.iloc[1:, 1:].values.astype(np.float32)
    
    print(f"Original data shape: {spectra_data.shape}")
    
    # 处理每个光谱到3600个点
    processed_spectra = []
    for i, spectrum in enumerate(spectra_data):
        processed_spec = process_spectrum_to_3600(spectrum, target_size)
        processed_spectra.append(processed_spec)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(spectra_data)} spectra")
    
    processed_spectra = np.array(processed_spectra, dtype=np.float32)
    print(f"Processed data shape: {processed_spectra.shape}")
    
    # 更新x_axis到3600个点
    processed_x_axis = np.linspace(x_axis.min(), x_axis.max(), target_size)
    
    return processed_spectra, processed_x_axis


def save_processed_data(processed_spectra, x_axis, output_path):
    """
    保存处理后的数据到CSV文件
    
    Args:
        processed_spectra: 处理后的光谱数据 (n_samples, 3600)
        x_axis: x轴数据 (3600,)
        output_path: 输出文件路径
    """
    # 创建DataFrame，第一行为x_axis，后续行为光谱数据
    data_matrix = np.vstack([x_axis, processed_spectra])
    df = pd.DataFrame(data_matrix)
    
    # 保存
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    df.to_csv(output_path, index=False, header=False)
    print(f"Saved processed data to: {output_path}")


def save_processed_data_h5(processed_spectra, x_axis, output_path,
                           physical_params=None, baseline_heatmaps=None):
    """
    保存处理后的数据到HDF5文件（更快，适合大文件）
    
    Args:
        processed_spectra: 处理后的光谱数据 (n_samples, 3600)
        x_axis: x轴数据 (3600,)
        output_path: 输出文件路径
        physical_params: 物理参数 (n_samples, 7)，可选
        baseline_heatmaps: 未归一化的方形热图 (n_samples, side, side)，可选
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('spectra', data=processed_spectra, compression='gzip', compression_opts=4)
        f.create_dataset('x_axis', data=x_axis)
        f.attrs['n_samples'] = len(processed_spectra)
        f.attrs['spectrum_length'] = processed_spectra.shape[1]
        # 保存物理参数（预计算的光学特征）
        if physical_params is not None:
            f.create_dataset('physical_params', data=physical_params, compression='gzip', compression_opts=4)
        if baseline_heatmaps is not None:
            f.create_dataset('heatmaps_unscaled', data=baseline_heatmaps,
                             compression='gzip', compression_opts=4)
    
    print(f"Saved processed data to HDF5: {output_path}")


def process_dataset(input_csv, output_csv, dataset_name, target_size=3600, save_h5=False, h5_only=False,
                   export_physical=True, export_baseline_heatmaps=False):
    """
    处理单个数据集
    
    Args:
        input_csv: 输入CSV文件路径
        output_csv: 输出CSV文件路径
        dataset_name: 数据集名称（用于打印信息）
        target_size: 目标光谱长度，默认3600
        save_h5: 是否同时保存为HDF5格式（更快加载）
        h5_only: 是否只保存HDF5格式（跳过CSV）
        export_physical: 是否导出物理参数（预计算的光学特征）
        export_baseline_heatmaps: 是否额外导出未归一化的方形热图 (仅HDF5)
    """
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'='*60}")
    
    if not os.path.exists(input_csv):
        print(f"Warning: Input file not found: {input_csv}")
        print(f"Skipping {dataset_name}")
        return False
    
    try:
        processed_spectra, processed_x_axis = load_and_process_csv(input_csv, target_size)

        physical_arr = None
        baseline_heatmaps = None
        if export_physical:
            physical_list = []
            for spec in processed_spectra:
                params = compute_params(spec)
                # 物理特征向量（与训练时一致）
                physical_vec = np.array([
                    params.get('mean', 0),
                    params.get('std', 0),
                    params.get('bandwidth', 0),
                    len(params.get('peak_positions', [])),
                    params.get('max_intensity', 0),
                    params.get('energy_range', (0, 0))[0],
                    params.get('energy_range', (0, 0))[1],
                ], dtype=np.float32)
                physical_vec = np.nan_to_num(physical_vec, nan=0.0, posinf=1e6, neginf=-1e6)
                physical_vec = np.clip(physical_vec, -1e3, 1e3)
                physical_list.append(physical_vec)

            physical_arr = np.stack(physical_list, axis=0)

        if export_baseline_heatmaps:
            side = int(np.sqrt(target_size))
            if side * side != target_size:
                raise ValueError(
                    f"Target size {target_size} cannot be reshaped into a square heatmap for baseline export"
                )
            baseline_heatmaps = processed_spectra.reshape(-1, side, side).astype(np.float32)
        
        # 只保存HDF5格式（推荐，速度更快）
        if h5_only:
            output_h5 = output_csv.replace('.csv', '.h5')
            save_processed_data_h5(
                processed_spectra, processed_x_axis, output_h5,
                physical_params=physical_arr,
                baseline_heatmaps=baseline_heatmaps
            )
            print(f"Saved only HDF5 format: {output_h5}")
        else:
            # 保存CSV格式
            save_processed_data(processed_spectra, processed_x_axis, output_csv)
            
            # 如果启用，同时保存HDF5格式
            if save_h5:
                output_h5 = output_csv.replace('.csv', '.h5')
                save_processed_data_h5(
                    processed_spectra, processed_x_axis, output_h5,
                    physical_params=physical_arr,
                    baseline_heatmaps=baseline_heatmaps
                )
        
        print(f"Successfully processed {dataset_name}: {len(processed_spectra)} samples")
        return True
    except Exception as e:
        print(f"Error processing {dataset_name}: {str(e)}")
        return False


def main():
    """
    主函数：处理三个数据集
    - ir_broaden
    - uv_broaden  
    - raman_broaden
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Process spectrum data to 3600 points')
    parser.add_argument('--data_dir', type=str, default='/mnt/shared-storage-user/xiejiaqing/data/qm9s',
                       help='Directory containing input CSV files')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                       help='Directory to save processed data')
    parser.add_argument('--save_h5', action='store_true',
                       help='Also save data in HDF5 format for faster loading')
    parser.add_argument('--h5_only', action='store_true',
                       help='Only save HDF5 format (skip CSV, fastest)')
    parser.add_argument('--export_baseline_heatmaps', action='store_true',
                       help='Export unnormalized square heatmaps for baseline training/testing (HDF5 only)')
    args = parser.parse_args()
    
    # 设置数据目录
    base_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 定义要处理的数据集
    datasets = {
        'ir_broaden': {
            'input': base_dir / 'ir_boraden.csv',
            'output': output_dir / 'ir_broaden_processed.csv'
        },
        'uv_broaden': {
            'input': base_dir / 'uv_boraden.csv',
            'output': output_dir / 'uv_broaden_processed.csv'
        },
        'raman_broaden': {
            'input': base_dir / 'raman_boraden.csv',
            'output': output_dir / 'raman_broaden_processed.csv'
        }
    }
    
    # 处理每个数据集
    results = {}
    for name, paths in datasets.items():
        # 尝试不同的文件名变体
        input_file = paths['input']
        if not input_file.exists():
            # 尝试没有下划线的版本
            alt_file = base_dir / f'{name.split("_")[0]}_boraden.csv'
            if alt_file.exists():
                input_file = alt_file
                print(f"Using alternative filename: {alt_file}")
        
        success = process_dataset(
            str(input_file),
            str(paths['output']),
            name,
            target_size=3600,
            save_h5=args.save_h5,
            h5_only=args.h5_only,
            export_baseline_heatmaps=args.export_baseline_heatmaps
        )
        results[name] = success
    
    # 打印总结
    print(f"\n{'='*60}")
    print("Processing Summary:")
    print(f"{'='*60}")
    for name, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"{name}: {status}")
    
    print(f"\nProcessed data saved to: {output_dir.absolute()}")


if __name__ == '__main__':
    main()

