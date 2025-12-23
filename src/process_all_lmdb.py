"""
批量处理多个LMDB数据集：zinc15, peptide, pahs, nist_ir, mols, peptide_mod, geom
"""
import os
from pathlib import Path
from process_lmdb import process_lmdb_dataset


def process_all_datasets(data_root, output_root, datasets=['zinc15', 'peptide', 'pahs', 'nist_ir', 'mols', 'peptide_mod', 'geom'], 
                         target_size=3600, save_h5=True):
    """
    批量处理所有数据集
    
    Args:
        data_root: LMDB文件所在根目录
        output_root: 输出根目录
        datasets: 要处理的数据集列表
        target_size: 目标光谱长度
        save_h5: 是否保存HDF5格式
    """
    data_root = Path(data_root)
    output_root = Path(output_root)
    
    print(f"Data root: {data_root}")
    print(f"Output root: {output_root}")
    print(f"Datasets to process: {datasets}")
    print(f"=" * 60)
    
    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*60}")
        
        # 构建LMDB文件路径
        # 只查找test文件
        lmdb_paths = []
        
        # 尝试不同的路径格式（只找test）
        possible_paths = [
            data_root / dataset_name / f"{dataset_name}_test.lmdb",
            data_root / dataset_name / f"{dataset_name}.lmdb",
            data_root / f"{dataset_name}_test.lmdb",
            data_root / f"{dataset_name}.lmdb",
        ]
        
        found = False
        for lmdb_path in possible_paths:
            if lmdb_path.exists() and lmdb_path.is_file():
                lmdb_paths.append(lmdb_path)
                found = True
        
        if not found:
            print(f"  ⚠ Warning: No LMDB file found for {dataset_name}")
            print(f"    Tried paths:")
            for p in possible_paths:
                print(f"      - {p}")
            continue
        
        # 处理每个找到的LMDB文件
        for lmdb_path in lmdb_paths:
            print(f"\n  Processing: {lmdb_path.name}")
            try:
                # 如果文件名包含_test，添加到dataset_name
                if '_test' in lmdb_path.stem:
                    output_name = f"{dataset_name}_test"
                else:
                    output_name = dataset_name
                
                process_lmdb_dataset(
                    str(lmdb_path),
                    str(output_root),
                    output_name,
                    target_size=target_size,
                    save_h5=save_h5
                )
                print(f"  ✓ Successfully processed {output_name}")
            except Exception as e:
                print(f"  ✗ Failed to process {lmdb_path}: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("All datasets processed!")
    print(f"{'='*60}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch process multiple LMDB datasets')
    parser.add_argument('--data_root', type=str, default='/mnt/shared-storage-user/xiejiaqing/data/vibench',
                       help='Root directory containing LMDB files (default: /mnt/shared-storage-user/xiejiaqing/data/vibench)')
    parser.add_argument('--output_root', type=str, required=True,
                       help='Root directory for output processed data')
    parser.add_argument('--datasets', type=str, nargs='+', 
                       default=['zinc15', 'peptide', 'pahs', 'nist_ir', 'mols', 'peptide_mod', 'geom'],
                       help='List of dataset names to process (default: zinc15 peptide pahs nist_ir mols peptide_mod geom)')
    parser.add_argument('--target_size', type=int, default=3600,
                       help='Target spectrum length (default: 3600)')
    parser.add_argument('--no_h5', action='store_true',
                       help='Do not save HDF5 format')
    
    args = parser.parse_args()
    
    process_all_datasets(
        args.data_root,
        args.output_root,
        datasets=args.datasets,
        target_size=args.target_size,
        save_h5=not args.no_h5
    )


if __name__ == '__main__':
    main()

