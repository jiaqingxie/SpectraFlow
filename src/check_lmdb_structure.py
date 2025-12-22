"""
检查LMDB数据结构，特别是raman和ir数据的形状和长度
"""
import pickle
import lmdb
import numpy as np
from pathlib import Path


def check_lmdb_structure(lmdb_path, n_samples=5):
    """检查LMDB数据的详细结构"""
    print(f"\nChecking LMDB structure: {lmdb_path}")
    print("=" * 60)
    
    env = lmdb.open(
        str(lmdb_path),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256
    )
    
    with env.begin() as txn:
        keys = list(txn.cursor().iternext(values=False))
    
    print(f"Total entries: {len(keys)}")
    print(f"\nExamining first {n_samples} entries:")
    print("=" * 60)
    
    raman_lengths = []
    ir_lengths = []
    raman_shapes = []
    ir_shapes = []
    
    for i, key in enumerate(keys[:n_samples]):
        print(f"\nEntry {i+1}:")
        try:
            with env.begin() as txn:
                pickled_data = txn.get(key)
                data = pickle.loads(pickled_data)
            
            print(f"  Keys in data: {list(data.keys())}")
            
            # 检查raman（支持 'raman' 和 'q_raman'）
            raman_key = None
            if 'q_raman' in data:
                raman_key = 'q_raman'
            elif 'raman' in data:
                raman_key = 'raman'
            
            if raman_key:
                raman_data = data[raman_key]
                print(f"  Found raman data in key: '{raman_key}'")
                print(f"  raman type: {type(raman_data)}")
                if hasattr(raman_data, 'shape'):
                    print(f"  raman shape: {raman_data.shape}")
                    raman_shapes.append(raman_data.shape)
                    if isinstance(raman_data, np.ndarray):
                        if raman_data.ndim == 1:
                            raman_lengths.append(len(raman_data))
                            print(f"    -> 1D array, length: {len(raman_data)}")
                            print(f"    -> min: {raman_data.min():.6f}, max: {raman_data.max():.6f}, mean: {raman_data.mean():.6f}")
                        elif raman_data.ndim == 2:
                            print(f"    -> 2D array (possibly x, y pairs)")
                            print(f"    -> First few values: {raman_data[:min(5, len(raman_data))]}")
                            # 检查是否是(x, y)对
                            if raman_data.shape[1] == 2:
                                print(f"    -> Appears to be (x, y) pairs")
                                x_vals = raman_data[:, 0]
                                y_vals = raman_data[:, 1]
                                print(f"    -> X range: [{x_vals.min():.6f}, {x_vals.max():.6f}]")
                                print(f"    -> Y range: [{y_vals.min():.6f}, {y_vals.max():.6f}]")
                                raman_lengths.append(len(y_vals))
                else:
                    print(f"  raman value: {raman_data}")
            
            # 检查ir（支持 'ir' 和 'q_ir'）
            ir_key = None
            if 'q_ir' in data:
                ir_key = 'q_ir'
            elif 'ir' in data:
                ir_key = 'ir'
            
            if ir_key:
                ir_data = data[ir_key]
                print(f"  Found ir data in key: '{ir_key}'")
                print(f"  ir type: {type(ir_data)}")
                if hasattr(ir_data, 'shape'):
                    print(f"  ir shape: {ir_data.shape}")
                    ir_shapes.append(ir_data.shape)
                    if isinstance(ir_data, np.ndarray):
                        if ir_data.ndim == 1:
                            ir_lengths.append(len(ir_data))
                            print(f"    -> 1D array, length: {len(ir_data)}")
                            print(f"    -> min: {ir_data.min():.6f}, max: {ir_data.max():.6f}, mean: {ir_data.mean():.6f}")
                        elif ir_data.ndim == 2:
                            print(f"    -> 2D array (possibly x, y pairs)")
                            print(f"    -> First few values: {ir_data[:min(5, len(ir_data))]}")
                            if ir_data.shape[1] == 2:
                                print(f"    -> Appears to be (x, y) pairs")
                                x_vals = ir_data[:, 0]
                                y_vals = ir_data[:, 1]
                                print(f"    -> X range: [{x_vals.min():.6f}, {x_vals.max():.6f}]")
                                print(f"    -> Y range: [{y_vals.min():.6f}, {y_vals.max():.6f}]")
                                ir_lengths.append(len(y_vals))
                else:
                    print(f"  ir value: {ir_data}")
        
        except Exception as e:
            print(f"  Error processing entry: {e}")
            import traceback
            traceback.print_exc()
    
    env.close()
    
    # 统计信息
    print(f"\n{'='*60}")
    print("Summary Statistics:")
    print(f"{'='*60}")
    
    if raman_lengths:
        print(f"\nRaman lengths:")
        print(f"  Unique lengths: {set(raman_lengths)}")
        print(f"  Min: {min(raman_lengths)}, Max: {max(raman_lengths)}")
        if len(set(raman_lengths)) == 1:
            print(f"  ✓ All raman spectra have same length: {raman_lengths[0]}")
        else:
            print(f"  ⚠️  Raman spectra have different lengths!")
    
    if ir_lengths:
        print(f"\nIR lengths:")
        print(f"  Unique lengths: {set(ir_lengths)}")
        print(f"  Min: {min(ir_lengths)}, Max: {max(ir_lengths)}")
        if len(set(ir_lengths)) == 1:
            print(f"  ✓ All IR spectra have same length: {ir_lengths[0]}")
        else:
            print(f"  ⚠️  IR spectra have different lengths!")
    
    if raman_shapes:
        print(f"\nRaman shapes: {set(raman_shapes)}")
    if ir_shapes:
        print(f"IR shapes: {set(ir_shapes)}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Check LMDB data structure')
    parser.add_argument('lmdb_path', type=str, help='Path to LMDB file')
    parser.add_argument('--n_samples', type=int, default=5, help='Number of samples to examine')
    
    args = parser.parse_args()
    
    check_lmdb_structure(args.lmdb_path, args.n_samples)

