"""
检查 geom.lmdb 文件的数据类型和结构
"""
import os
import sys
from pathlib import Path
import numpy as np

def inspect_lmdb(file_path):
    """检查lmdb文件的内容"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return
    
    print(f"File: {file_path}")
    if file_path.is_dir():
        # LMDB 通常是一个目录
        total_size = sum(f.stat().st_size for f in file_path.rglob('*') if f.is_file())
        print(f"Size: {total_size / (1024*1024):.2f} MB (directory)")
    else:
        print(f"Size: {file_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"\n{'='*60}")
    
    # 首先尝试 LMDB
    print("\n1. Trying to load as LMDB (Lightning Memory-Mapped Database)...")
    try:
        import lmdb
        print(f"   ✓ lmdb module imported successfully")
        
        # LMDB 通常是一个目录，检查路径
        if file_path.is_file():
            print(f"   Note: {file_path} is a file, but LMDB expects a directory.")
            print(f"   Trying parent directory or checking for .lmdb directory...")
            # 尝试同名的目录（去掉扩展名）
            lmdb_dir = file_path.parent / file_path.stem
            if lmdb_dir.exists() and lmdb_dir.is_dir():
                print(f"   Found directory: {lmdb_dir}")
                file_path = lmdb_dir
            else:
                # 或者尝试直接使用文件所在目录
                print(f"   Trying to use parent directory: {file_path.parent}")
                # 检查是否有 data.mdb 或 lock.mdb 文件（LMDB 的标志文件）
                if (file_path.parent / "data.mdb").exists():
                    print(f"   Found data.mdb in parent directory, using parent as LMDB root")
                    file_path = file_path.parent
                else:
                    raise ValueError(f"LMDB path should be a directory, but got a file: {file_path}")
        elif file_path.is_dir():
            print(f"   ✓ Path is a directory (as expected for LMDB)")
        else:
            raise ValueError(f"Path does not exist: {file_path}")
        
        print(f"   Opening LMDB at: {file_path}")
        
        # 如果是目录，列出内容
        if file_path.is_dir():
            print(f"\n   Directory contents:")
            for item in sorted(file_path.iterdir()):
                if item.is_file():
                    size = item.stat().st_size / (1024*1024)
                    print(f"     {item.name} ({size:.2f} MB)")
                else:
                    print(f"     {item.name}/ (directory)")
        
        env = lmdb.open(str(file_path), readonly=True, lock=False, readahead=False, meminit=False)
        
        print("   ✓ Successfully opened as LMDB file!")
        
        with env.begin() as txn:
            stat = txn.stat()
            print(f"\n   Database statistics:")
            print(f"     Entries: {stat['entries']}")
            print(f"     Depth: {stat['depth']}")
            print(f"     Branch pages: {stat['branch_pages']}")
            print(f"     Leaf pages: {stat['leaf_pages']}")
            print(f"     Overflow pages: {stat['overflow_pages']}")
            
            # 获取前几个键值对作为示例
            cursor = txn.cursor()
            print(f"\n   Sample keys (first 10):")
            count = 0
            for key, value in cursor:
                if count < 10:
                    try:
                        # 尝试解码key
                        if isinstance(key, bytes):
                            key_str = key.decode('utf-8', errors='ignore')
                        else:
                            key_str = str(key)
                        
                        # 尝试解析value
                        try:
                            import pickle
                            value_obj = pickle.loads(value)
                            print(f"\n     Key: {key_str}")
                            print(f"       Value type: {type(value_obj)}")
                            if isinstance(value_obj, dict):
                                print(f"       Dict keys: {list(value_obj.keys())[:10]}")
                                for k, v in list(value_obj.items())[:5]:
                                    print(f"         {k}: {type(v)}", end="")
                                    if isinstance(v, np.ndarray):
                                        print(f", shape={v.shape}, dtype={v.dtype}")
                                    elif isinstance(v, (list, tuple)):
                                        print(f", len={len(v)}")
                                    else:
                                        print(f", value={v}")
                            elif isinstance(value_obj, np.ndarray):
                                print(f"       Shape: {value_obj.shape}")
                                print(f"       Dtype: {value_obj.dtype}")
                                print(f"       Min: {value_obj.min()}, Max: {value_obj.max()}")
                            elif isinstance(value_obj, (list, tuple)):
                                print(f"       Length: {len(value_obj)}")
                                if len(value_obj) > 0:
                                    print(f"       First item type: {type(value_obj[0])}")
                            else:
                                print(f"       Value: {value_obj}")
                        except:
                            print(f"\n     Key: {key_str}")
                            print(f"       Value (raw bytes, length={len(value)}): {value[:100]}...")
                        count += 1
                    except Exception as e:
                        print(f"     Error parsing key/value: {e}")
                        count += 1
                else:
                    break
            
            # 统计所有键的类型
            print(f"\n   Analyzing all keys...")
            cursor = txn.cursor()
            key_types = {}
            value_sizes = []
            for key, value in cursor:
                try:
                    if isinstance(key, bytes):
                        key_str = key.decode('utf-8', errors='ignore')
                    else:
                        key_str = str(key)
                    key_types[key_str[:20]] = key_types.get(key_str[:20], 0) + 1
                    value_sizes.append(len(value))
                except:
                    pass
            
            print(f"   Unique key patterns: {len(key_types)}")
            print(f"   Value sizes - Min: {min(value_sizes) if value_sizes else 0}, "
                  f"Max: {max(value_sizes) if value_sizes else 0}, "
                  f"Mean: {np.mean(value_sizes) if value_sizes else 0:.2f}")
        
        env.close()
        return
    except ImportError as e:
        print(f"   ✗ lmdb module not installed. Install with: pip install lmdb")
        print(f"   Error details: {e}")
        import sys
        print(f"   Python path: {sys.executable}")
        print(f"   Trying to check if lmdb is available...")
        try:
            import importlib.util
            spec = importlib.util.find_spec("lmdb")
            if spec is None:
                print("   ✗ lmdb module not found in Python path")
            else:
                print(f"   ! lmdb module found at: {spec.origin}")
                print("   But import failed, this is strange...")
        except:
            pass
    except Exception as e:
        print(f"   ✗ Not LMDB or error: {type(e).__name__}: {e}")
        import traceback
        print(f"   Traceback:")
        traceback.print_exc()
    
    # 尝试不同的加载方式
    print("\n2. Trying to load as HDF5...")
    try:
        import h5py
        with h5py.File(file_path, 'r') as f:
            print("   ✓ Successfully opened as HDF5 file!")
            print(f"\n   Keys in file:")
            def print_structure(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"     Dataset: {name}")
                    print(f"       Shape: {obj.shape}")
                    print(f"       Dtype: {obj.dtype}")
                    if obj.size < 20:  # 如果数据量小，显示内容
                        print(f"       Data: {obj[:]}")
                    else:
                        print(f"       Min: {obj[:].min()}, Max: {obj[:].max()}")
                elif isinstance(obj, h5py.Group):
                    print(f"     Group: {name}")
            f.visititems(print_structure)
            return
    except Exception as e:
        print(f"   ✗ Not HDF5: {e}")
    
    print("\n2. Trying to load as NumPy .npz...")
    try:
        data = np.load(file_path, allow_pickle=True)
        print("   ✓ Successfully opened as .npz file!")
        print(f"\n   Keys in file:")
        for key in data.keys():
            arr = data[key]
            print(f"     {key}:")
            print(f"       Type: {type(arr)}")
            if isinstance(arr, np.ndarray):
                print(f"       Shape: {arr.shape}")
                print(f"       Dtype: {arr.dtype}")
                if arr.size < 20:
                    print(f"       Data: {arr}")
                else:
                    print(f"       Min: {arr.min()}, Max: {arr.max()}")
            else:
                print(f"       Value: {arr}")
        return
    except Exception as e:
        print(f"   ✗ Not .npz: {e}")
    
    print("\n3. Trying to load as Pickle...")
    try:
        import pickle
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print("   ✓ Successfully opened as Pickle file!")
        print(f"\n   Type: {type(data)}")
        if isinstance(data, dict):
            print(f"   Keys: {list(data.keys())}")
            for key, value in data.items():
                print(f"\n     {key}:")
                print(f"       Type: {type(value)}")
                if isinstance(value, np.ndarray):
                    print(f"       Shape: {value.shape}")
                    print(f"       Dtype: {value.dtype}")
                    if value.size < 20:
                        print(f"       Data: {value}")
                    else:
                        print(f"       Min: {value.min()}, Max: {value.max()}")
                elif isinstance(value, (list, tuple)):
                    print(f"       Length: {len(value)}")
                    if len(value) < 10:
                        print(f"       Items: {value}")
                else:
                    print(f"       Value: {value}")
        elif isinstance(data, (list, tuple)):
            print(f"   Length: {len(data)}")
            if len(data) > 0:
                print(f"   First item type: {type(data[0])}")
                if len(data) < 10:
                    print(f"   Items: {data}")
        else:
            print(f"   Value: {data}")
        return
    except Exception as e:
        print(f"   ✗ Not Pickle: {e}")
    
    print("\n4. Trying to read as text (first 100 lines)...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[:100]
            print(f"   ✓ File is readable as text!")
            print(f"   First 20 lines:")
            for i, line in enumerate(lines[:20]):
                print(f"     {i+1:3d}: {line.rstrip()}")
            return
    except Exception as e:
        print(f"   ✗ Not readable as text: {e}")
    
    print("\n5. Checking file header (first 16 bytes)...")
    try:
        with open(file_path, 'rb') as f:
            header = f.read(16)
            print(f"   Hex: {header.hex()}")
            print(f"   ASCII (if printable): {''.join([chr(b) if 32 <= b < 127 else '.' for b in header])}")
    except Exception as e:
        print(f"   ✗ Error reading header: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Inspect .lmdb file structure')
    parser.add_argument('file_path', type=str, 
                       help='Path to .lmdb file or directory')
    
    args = parser.parse_args()
    inspect_lmdb(args.file_path)


if __name__ == '__main__':
    main()

