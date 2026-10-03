"""
处理LMDB数据格式：从LMDB中提取smiles, raman, ir数据，插值到3600个点并保存
"""
import numpy as np
from scipy import interpolate
import os
from pathlib import Path
import h5py
import pickle
import lmdb
from tqdm import tqdm
from utils import compute_params


def interpolate_spectrum(data, target_size=3600, x_axis=None):
    """
    将光谱数据插值到目标长度

    Args:
        data: 光谱数据，可以是：
            - 1D array: 只有y值
            - 2D array (N, 2): (x, y)对，第一列是x，第二列是y
        target_size: 目标长度，默认3600
        x_axis: 可选的x轴数组（如果data是1D且提供了x_axis）

    Returns:
        y_new: 插值后的y轴数据（target_size长度）
    """
    # 处理2D数组（x, y对）
    if isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] == 2:
        x_old = data[:, 0]
        y_old = data[:, 1]
        # 检查x轴是否单调
        if not (np.all(np.diff(x_old) >= 0) or np.all(np.diff(x_old) <= 0)):
            # 如果不是单调的，先排序
            sort_idx = np.argsort(x_old)
            x_old = x_old[sort_idx]
            y_old = y_old[sort_idx]

        # 使用实际的x轴范围进行插值
        x_min, x_max = x_old.min(), x_old.max()
        x_new = np.linspace(x_min, x_max, target_size)

        # 使用cubic spline插值可以更好地保留峰值特征
        try:
            f = interpolate.interp1d(x_old, y_old, kind='cubic',
                                    bounds_error=False, fill_value=(y_old[0], y_old[-1]))
        except ValueError:
            # 如果cubic失败（比如数据中有重复值），fallback到linear
            f = interpolate.interp1d(x_old, y_old, kind='linear',
                                    bounds_error=False, fill_value=(y_old[0], y_old[-1]))
        y_new = f(x_new)
        return y_new.astype(np.float32)

    # 处理1D数组
    y_old = np.array(data).flatten()

    if len(y_old) == target_size:
        return y_old.astype(np.float32)

    # 如果提供了x_axis，使用实际的x轴
    if x_axis is not None:
        x_old = np.array(x_axis).flatten()
        if len(x_old) != len(y_old):
            raise ValueError(f"x_axis length ({len(x_old)}) must match data length ({len(y_old)})")

        # 检查x轴是否单调
        if not (np.all(np.diff(x_old) >= 0) or np.all(np.diff(x_old) <= 0)):
            sort_idx = np.argsort(x_old)
            x_old = x_old[sort_idx]
            y_old = y_old[sort_idx]

        x_min, x_max = x_old.min(), x_old.max()
        x_new = np.linspace(x_min, x_max, target_size)
        # 使用cubic spline插值可以更好地保留峰值特征
        try:
            f = interpolate.interp1d(x_old, y_old, kind='cubic',
                                    bounds_error=False, fill_value=(y_old[0], y_old[-1]))
        except ValueError:
            # 如果cubic失败（比如数据中有重复值），fallback到linear
            f = interpolate.interp1d(x_old, y_old, kind='linear',
                                    bounds_error=False, fill_value=(y_old[0], y_old[-1]))
        y_new = f(x_new)
        return y_new.astype(np.float32)

    # 如果没有x轴信息，使用归一化插值
    # 使用cubic spline插值可以更好地保留峰值特征（比linear更好）
    x_old = np.linspace(0, 1, len(y_old))
    x_new = np.linspace(0, 1, target_size)

    # 对于较短的序列使用linear，较长的使用cubic
    if len(y_old) < 100:
        # 点数太少，使用linear避免过拟合
        f = interpolate.interp1d(x_old, y_old, kind='linear',
                                fill_value='extrapolate', bounds_error=False)
    else:
        # 点数足够，使用cubic spline可以更好地保留峰值
        try:
            f = interpolate.interp1d(x_old, y_old, kind='cubic',
                                    fill_value='extrapolate', bounds_error=False)
        except ValueError:
            # 如果cubic失败（比如数据中有重复值），fallback到linear
            f = interpolate.interp1d(x_old, y_old, kind='linear',
                                    fill_value='extrapolate', bounds_error=False)

    return f(x_new).astype(np.float32)


def process_lmdb_dataset(lmdb_path, output_dir, dataset_name, target_size=3600, save_h5=True):
    """
    处理LMDB数据集，提取smiles, raman, ir数据

    Args:
        lmdb_path: LMDB文件路径
        output_dir: 输出目录
        dataset_name: 数据集名称（用于生成输出文件名）
        target_size: 目标光谱长度，默认3600
        save_h5: 是否保存HDF5格式
    """
    # 检查LMDB路径是否存在
    lmdb_path_obj = Path(lmdb_path)
    if not lmdb_path_obj.exists():
        raise FileNotFoundError(f"LMDB path not found: {lmdb_path}")

    print(f"\nProcessing LMDB: {lmdb_path}")
    print(f"Dataset: {dataset_name}")
    print(f"Path type: {'Directory' if lmdb_path_obj.is_dir() else 'File'}")

    # 打开LMDB（LMDB可能是文件或目录）
    # 先尝试作为文件打开（subdir=False）
    try:
        if lmdb_path_obj.is_file():
            env = lmdb.open(
                str(lmdb_path),
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=256
            )
        elif lmdb_path_obj.is_dir():
            # 如果是目录，尝试作为目录打开
            env = lmdb.open(
                str(lmdb_path),
                subdir=True,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=256
            )
        else:
            raise ValueError(f"LMDB path is neither a file nor a directory: {lmdb_path}")
    except Exception as e:
        error_msg = str(e)
        if "No such device" in error_msg or "Invalid argument" in error_msg:
            # 如果作为文件打开失败，尝试作为目录打开
            print(f"  ⚠ Warning: Failed to open as file, trying as directory...")
            try:
                env = lmdb.open(
                    str(lmdb_path),
                    subdir=True,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False,
                    max_readers=256
                )
                print(f"  ✓ Successfully opened as directory")
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to open LMDB at {lmdb_path}.\n"
                    f"Error as file: {error_msg}\n"
                    f"Error as directory: {str(e2)}\n"
                    f"Please check:\n"
                    f"  1. File/directory exists and is accessible\n"
                    f"  2. File permissions are correct\n"
                    f"  3. If on network storage, ensure it's properly mounted\n"
                    f"  4. LMDB database is not corrupted"
                ) from e2
        else:
            raise

    # 获取所有keys
    with env.begin() as txn:
        keys = list(txn.cursor().iternext(values=False))

    print(f"Total entries: {len(keys)}")

    # 处理数据
    processed_raman = []
    processed_ir = []
    processed_smiles_raman = []
    processed_smiles_ir = []
    physical_params_raman = []
    physical_params_ir = []
    failed_count = 0

    # 统计原始数据信息
    original_raman_lengths = []
    original_ir_lengths = []
    original_raman_ranges = []
    original_ir_ranges = []

    # 处理每个条目
    for key in tqdm(keys, desc="Processing"):
        try:
            with env.begin() as txn:
                pickled_data = txn.get(key)
                if pickled_data is None:
                    continue
                data = pickle.loads(pickled_data)

            # 提取smiles字段
            if 'smiles' not in data:
                # 尝试其他可能的smiles字段
                if 'norm_smiles' in data:
                    smiles = data['norm_smiles']
                elif 'kekule_smiles' in data:
                    smiles = data['kekule_smiles']
                else:
                    smiles = None
            else:
                smiles = data['smiles']

            # 处理raman（支持 'raman' 和 'q_raman'）
            raman_key = None
            if 'q_raman' in data:
                raman_key = 'q_raman'
            elif 'raman' in data:
                raman_key = 'raman'

            if raman_key and smiles is not None:
                raman_data = data[raman_key]
                # 转换为numpy数组（处理torch tensor或其他类型）
                if hasattr(raman_data, 'cpu'):  # torch tensor
                    raman_data = raman_data.cpu().numpy()
                elif not isinstance(raman_data, np.ndarray):
                    raman_data = np.array(raman_data)

                # 检查是否有x轴信息（k_raman）
                raman_x_axis = None
                if 'k_raman' in data:
                    raman_x_axis_data = data['k_raman']
                    if hasattr(raman_x_axis_data, 'cpu'):
                        raman_x_axis = raman_x_axis_data.cpu().numpy()
                    elif isinstance(raman_x_axis_data, np.ndarray):
                        raman_x_axis = raman_x_axis_data
                    else:
                        raman_x_axis = np.array(raman_x_axis_data)
                    raman_x_axis = raman_x_axis.flatten()

                # 统计原始数据信息
                if raman_data.ndim == 2 and raman_data.shape[1] == 2:
                    # 2D数组，可能是(x, y)对
                    original_raman_lengths.append(raman_data.shape[0])
                    original_raman_ranges.append((raman_data[:, 1].min(), raman_data[:, 1].max()))
                else:
                    # 1D数组
                    raman_y_flat = raman_data.flatten()
                    original_raman_lengths.append(len(raman_y_flat))
                    original_raman_ranges.append((raman_y_flat.min(), raman_y_flat.max()))

                # 插值到target_size（如果提供了x_axis会使用它）
                raman_interp = interpolate_spectrum(raman_data, target_size, x_axis=raman_x_axis)
                processed_raman.append(raman_interp)
                processed_smiles_raman.append(smiles)

                # 计算物理参数
                params_raman = compute_params(raman_interp)
                physical_vec_raman = np.array([
                    params_raman.get('mean', 0),
                    params_raman.get('std', 0),
                    params_raman.get('bandwidth', 0),
                    len(params_raman.get('peak_positions', [])),
                    params_raman.get('max_intensity', 0),
                    params_raman.get('energy_range', (0, 0))[0],
                    params_raman.get('energy_range', (0, 0))[1],
                ], dtype=np.float32)
                physical_params_raman.append(physical_vec_raman)

            # 处理ir（支持 'ir' 和 'q_ir'）
            ir_key = None
            if 'q_ir' in data:
                ir_key = 'q_ir'
            elif 'ir' in data:
                ir_key = 'ir'

            if ir_key and smiles is not None:
                ir_data = data[ir_key]
                # 转换为numpy数组（处理torch tensor或其他类型）
                if hasattr(ir_data, 'cpu'):  # torch tensor
                    ir_data = ir_data.cpu().numpy()
                elif not isinstance(ir_data, np.ndarray):
                    ir_data = np.array(ir_data)

                # 检查是否有x轴信息（k_ir）
                ir_x_axis = None
                if 'k_ir' in data:
                    ir_x_axis_data = data['k_ir']
                    if hasattr(ir_x_axis_data, 'cpu'):
                        ir_x_axis = ir_x_axis_data.cpu().numpy()
                    elif isinstance(ir_x_axis_data, np.ndarray):
                        ir_x_axis = ir_x_axis_data
                    else:
                        ir_x_axis = np.array(ir_x_axis_data)
                    ir_x_axis = ir_x_axis.flatten()

                # 统计原始数据信息
                if ir_data.ndim == 2 and ir_data.shape[1] == 2:
                    # 2D数组，可能是(x, y)对
                    original_ir_lengths.append(ir_data.shape[0])
                    original_ir_ranges.append((ir_data[:, 1].min(), ir_data[:, 1].max()))
                else:
                    # 1D数组
                    ir_y_flat = ir_data.flatten()
                    original_ir_lengths.append(len(ir_y_flat))
                    original_ir_ranges.append((ir_y_flat.min(), ir_y_flat.max()))

                # 插值到target_size（如果提供了x_axis会使用它）
                ir_interp = interpolate_spectrum(ir_data, target_size, x_axis=ir_x_axis)
                processed_ir.append(ir_interp)
                processed_smiles_ir.append(smiles)

                # 计算物理参数
                params_ir = compute_params(ir_interp)
                physical_vec_ir = np.array([
                    params_ir.get('mean', 0),
                    params_ir.get('std', 0),
                    params_ir.get('bandwidth', 0),
                    len(params_ir.get('peak_positions', [])),
                    params_ir.get('max_intensity', 0),
                    params_ir.get('energy_range', (0, 0))[0],
                    params_ir.get('energy_range', (0, 0))[1],
                ], dtype=np.float32)
                physical_params_ir.append(physical_vec_ir)

        except Exception as e:
            failed_count += 1
            continue

    env.close()

    print(f"\nProcessed {len(processed_raman)} raman spectra, {len(processed_ir)} ir spectra")
    print(f"Failed entries: {failed_count}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 处理raman数据
    if processed_raman:
        print(f"\nSaving Raman data...")
        raman_spectra = np.array(processed_raman, dtype=np.float32)
        raman_physical = np.array(physical_params_raman, dtype=np.float32)

        # 生成x_axis（使用归一化范围，因为原始x轴未知）
        raman_x_axis = np.linspace(0, 1, target_size, dtype=np.float32)

        # 保存CSV
        raman_csv = os.path.join(output_dir, f"{dataset_name}_raman_processed.csv")
        data_matrix = np.vstack([raman_x_axis, raman_spectra])
        np.savetxt(raman_csv, data_matrix, delimiter=',', fmt='%.6e')
        print(f"  Saved CSV: {raman_csv}")

        # 保存HDF5
        if save_h5:
            raman_h5 = os.path.join(output_dir, f"{dataset_name}_raman_processed.h5")
            with h5py.File(raman_h5, 'w') as f:
                f.create_dataset('spectra', data=raman_spectra, compression='gzip', compression_opts=4)
                f.create_dataset('x_axis', data=raman_x_axis)
                f.create_dataset('physical_params', data=raman_physical, compression='gzip', compression_opts=4)
                f.attrs['n_samples'] = len(raman_spectra)
                f.attrs['spectrum_length'] = raman_spectra.shape[1]
            print(f"  Saved HDF5: {raman_h5}")

        print(f"  Shape: {raman_spectra.shape}")
        if original_raman_lengths:
            print(f"  Original data statistics:")
            print(f"    Length - Min: {min(original_raman_lengths)}, Max: {max(original_raman_lengths)}, Mean: {np.mean(original_raman_lengths):.1f}")
            print(f"    Y-range (original) - Min: [{min(r[0] for r in original_raman_ranges):.2f}, {min(r[1] for r in original_raman_ranges):.2f}], "
                  f"Max: [{max(r[0] for r in original_raman_ranges):.2f}, {max(r[1] for r in original_raman_ranges):.2f}]")

    # 处理ir数据
    if processed_ir:
        print(f"\nSaving IR data...")
        ir_spectra = np.array(processed_ir, dtype=np.float32)
        ir_physical = np.array(physical_params_ir, dtype=np.float32)

        # 生成x_axis
        ir_x_axis = np.linspace(0, 1, target_size, dtype=np.float32)

        # 保存CSV
        ir_csv = os.path.join(output_dir, f"{dataset_name}_ir_processed.csv")
        data_matrix = np.vstack([ir_x_axis, ir_spectra])
        np.savetxt(ir_csv, data_matrix, delimiter=',', fmt='%.6e')
        print(f"  Saved CSV: {ir_csv}")

        # 保存HDF5
        if save_h5:
            ir_h5 = os.path.join(output_dir, f"{dataset_name}_ir_processed.h5")
            with h5py.File(ir_h5, 'w') as f:
                f.create_dataset('spectra', data=ir_spectra, compression='gzip', compression_opts=4)
                f.create_dataset('x_axis', data=ir_x_axis)
                f.create_dataset('physical_params', data=ir_physical, compression='gzip', compression_opts=4)
                f.attrs['n_samples'] = len(ir_spectra)
                f.attrs['spectrum_length'] = ir_spectra.shape[1]
            print(f"  Saved HDF5: {ir_h5}")

        print(f"  Shape: {ir_spectra.shape}")
        if original_ir_lengths:
            print(f"  Original data statistics:")
            print(f"    Length - Min: {min(original_ir_lengths)}, Max: {max(original_ir_lengths)}, Mean: {np.mean(original_ir_lengths):.1f}")
            print(f"    Y-range (original) - Min: [{min(r[0] for r in original_ir_ranges):.2f}, {min(r[1] for r in original_ir_ranges):.2f}], "
                  f"Max: [{max(r[0] for r in original_ir_ranges):.2f}, {max(r[1] for r in original_ir_ranges):.2f}]")

    # 保存smiles（如果有）
    if processed_smiles_raman:
        smiles_file = os.path.join(output_dir, f"{dataset_name}_raman_smiles.txt")
        with open(smiles_file, 'w') as f:
            for smiles in processed_smiles_raman:
                f.write(f"{smiles}\n")
        print(f"  Saved Raman smiles to: {smiles_file}")

    if processed_smiles_ir:
        smiles_file = os.path.join(output_dir, f"{dataset_name}_ir_smiles.txt")
        with open(smiles_file, 'w') as f:
            for smiles in processed_smiles_ir:
                f.write(f"{smiles}\n")
        print(f"  Saved IR smiles to: {smiles_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Process LMDB dataset to extract smiles, raman, ir')
    parser.add_argument('--lmdb_path', type=str, required=True,
                       help='Path to LMDB file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for processed data')
    parser.add_argument('--dataset_name', type=str, required=True,
                       help='Dataset name (for output filenames: {name}_raman_processed.csv, etc.)')
    parser.add_argument('--target_size', type=int, default=3600,
                       help='Target spectrum length (default: 3600)')
    parser.add_argument('--no_h5', action='store_true',
                       help='Do not save HDF5 format')

    args = parser.parse_args()

    process_lmdb_dataset(
        args.lmdb_path,
        args.output_dir,
        args.dataset_name,
        target_size=args.target_size,
        save_h5=not args.no_h5
    )


if __name__ == '__main__':
    main()
