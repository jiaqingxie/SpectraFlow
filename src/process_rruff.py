"""
处理RRUFF数据格式：将txt文件转换为3600个点的光谱数据集
格式：跳过前10行，读取数据直到##END，左列为位移（x），右列为峰高（y）
"""
import numpy as np
from scipy import interpolate
import os
from pathlib import Path
import h5py
from utils import compute_params


def load_rruff_txt(txt_path):
    """
    加载RRUFF格式的txt文件
    
    Args:
        txt_path: txt文件路径
        
    Returns:
        x_data: x轴数据（位移）
        y_data: y轴数据（峰高）
    """
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 跳过前10行，从第11行开始（索引10）
    data_lines = []
    for i in range(10, len(lines)):
        line = lines[i].strip()
        if line.startswith('##END'):
            break
        if line and ',' in line:
            data_lines.append(line)
    
    if not data_lines:
        raise ValueError(f"No data found in {txt_path}")
    
    # 解析数据：左列为x，右列为y
    x_data = []
    y_data = []
    for line in data_lines:
        parts = line.split(',')
        if len(parts) >= 2:
            try:
                x_val = float(parts[0].strip())
                y_val = float(parts[1].strip())
                x_data.append(x_val)
                y_data.append(y_val)
            except ValueError:
                continue
    
    return np.array(x_data, dtype=np.float32), np.array(y_data, dtype=np.float32)


def interpolate_spectrum(y_old, target_size=3600):
    """
    将光谱数据插值到目标长度（与process.py中的process_spectrum_to_3600保持一致）
    
    Args:
        y_old: 原始y轴数据（光谱强度），1D numpy array
        target_size: 目标长度，默认3600
        
    Returns:
        y_new: 插值后的y轴数据
    """
    if len(y_old) == target_size:
        return y_old.astype(np.float32)
    
    # 使用线性插值（与原process.py完全一致的方法）
    # 将原始数据映射到[0,1]空间进行插值
    x_old = np.linspace(0, 1, len(y_old))
    x_new = np.linspace(0, 1, target_size)
    f = interpolate.interp1d(x_old, y_old, kind='linear', 
                            fill_value='extrapolate', bounds_error=False)
    return f(x_new).astype(np.float32)


def process_rruff_directory(input_dir, output_csv, target_size=3600, save_h5=True, scale_max=None):
    """
    处理RRUFF目录中的所有txt文件
    
    Args:
        input_dir: 输入目录路径（包含txt文件）
        output_csv: 输出CSV文件路径
        target_size: 目标光谱长度，默认3600
        save_h5: 是否同时保存HDF5格式
        scale_max: 将每个光谱的最大值缩放到此值（None表示不缩放）
    """
    input_path = Path(input_dir)
    txt_files = sorted(list(input_path.glob('*.txt')))
    
    if not txt_files:
        print(f"No txt files found in {input_dir}")
        return
    
    print(f"Found {len(txt_files)} txt files in {input_dir}")
    
    processed_spectra = []
    processed_x_axes = []
    physical_params_list = []
    failed_files = []
    
    for i, txt_file in enumerate(txt_files):
        try:
            # 加载数据
            x_data, y_data = load_rruff_txt(txt_file)
            
            # 插值到目标长度（只对y值插值，与原process.py一致）
            y_interp = interpolate_spectrum(y_data, target_size)
            
            # 如果指定了scale_max，将每个光谱的最大值缩放到目标值
            if scale_max is not None:
                max_val = y_interp.max()
                if max_val > 0:
                    y_interp = y_interp * (scale_max / max_val)
            
            processed_spectra.append(y_interp)
            # 保存原始x范围用于后续生成统一的x_axis
            processed_x_axes.append(x_data)
            
            # 计算物理参数
            params = compute_params(y_interp)
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
            physical_params_list.append(physical_vec)
            
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(txt_files)} files")
                
        except Exception as e:
            print(f"Error processing {txt_file.name}: {e}")
            failed_files.append(txt_file.name)
            continue
    
    if not processed_spectra:
        print("No spectra were successfully processed")
        return
    
    # 转换为numpy数组
    processed_spectra = np.array(processed_spectra, dtype=np.float32)
    physical_params = np.array(physical_params_list, dtype=np.float32)
    
    # 对于x_axis，使用第一个样本的x轴（因为都是插值到相同范围）
    # 或者可以使用所有样本的平均x轴范围
    x_min = min(x.min() for x in processed_x_axes)
    x_max = max(x.max() for x in processed_x_axes)
    processed_x_axis = np.linspace(x_min, x_max, target_size, dtype=np.float32)
    
    print(f"\nSuccessfully processed {len(processed_spectra)} spectra")
    print(f"Final shape: {processed_spectra.shape}")
    print(f"X-axis range: [{x_min:.2f}, {x_max:.2f}]")
    if scale_max is not None:
        print(f"Scaled to max value: {scale_max}")
        print(f"Actual max values - Min: {processed_spectra.max(axis=1).min():.2f}, Max: {processed_spectra.max(axis=1).max():.2f}")
    else:
        print(f"Y-axis range: [{processed_spectra.min():.2f}, {processed_spectra.max():.2f}]")
    
    if failed_files:
        print(f"\nFailed to process {len(failed_files)} files:")
        for fname in failed_files[:10]:  # 只显示前10个
            print(f"  - {fname}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")
    
    # 保存CSV格式（与之前的格式一致：第一行是x_axis，后续行是光谱数据）
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)
    data_matrix = np.vstack([processed_x_axis, processed_spectra])
    np.savetxt(output_csv, data_matrix, delimiter=',', fmt='%.6e')
    print(f"\nSaved CSV to: {output_csv}")
    
    # 保存HDF5格式（推荐，速度更快）
    if save_h5:
        output_h5 = output_csv.replace('.csv', '.h5')
        with h5py.File(output_h5, 'w') as f:
            f.create_dataset('spectra', data=processed_spectra, compression='gzip', compression_opts=4)
            f.create_dataset('x_axis', data=processed_x_axis)
            f.create_dataset('physical_params', data=physical_params, compression='gzip', compression_opts=4)
            f.attrs['n_samples'] = len(processed_spectra)
            f.attrs['spectrum_length'] = processed_spectra.shape[1]
        print(f"Saved HDF5 to: {output_h5}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Process RRUFF data format to 3600-point spectra')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory containing txt files')
    parser.add_argument('--output_csv', type=str, required=True,
                       help='Output CSV file path')
    parser.add_argument('--target_size', type=int, default=3600,
                       help='Target spectrum length (default: 3600)')
    parser.add_argument('--scale_max', type=float, default=None,
                       help='Scale each spectrum to have this maximum value (e.g., 250 for IR, 300 for Raman)')
    parser.add_argument('--no_h5', action='store_true',
                       help='Do not save HDF5 format')
    
    args = parser.parse_args()
    
    process_rruff_directory(
        args.input_dir,
        args.output_csv,
        target_size=args.target_size,
        save_h5=not args.no_h5,
        scale_max=args.scale_max
    )


if __name__ == '__main__':
    main()

