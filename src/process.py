"""
数据处理模块：将CSV文件处理成3600个点的光谱数据集
处理三个数据集：ir_broaden, uv_broaden, raman_broaden
"""
import pandas as pd
import numpy as np
from scipy import interpolate
import os
from pathlib import Path


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


def process_dataset(input_csv, output_csv, dataset_name, target_size=3600):
    """
    处理单个数据集
    
    Args:
        input_csv: 输入CSV文件路径
        output_csv: 输出CSV文件路径
        dataset_name: 数据集名称（用于打印信息）
        target_size: 目标光谱长度，默认3600
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
        save_processed_data(processed_spectra, processed_x_axis, output_csv)
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
    # 设置数据目录（SpectraViT项目下）
    base_dir = Path(r"E:\SpectraViT")
    output_dir = Path("data/processed")
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
        success = process_dataset(
            str(paths['input']),
            str(paths['output']),
            name,
            target_size=3600
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

