"""简单测试 lmdb 导入"""
import sys

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

try:
    import lmdb
    print(f"✓ lmdb imported successfully")
    print(f"  lmdb version: {lmdb.__version__ if hasattr(lmdb, '__version__') else 'unknown'}")
    print(f"  lmdb location: {lmdb.__file__}")
except ImportError as e:
    print(f"✗ Failed to import lmdb: {e}")
    import importlib.util
    spec = importlib.util.find_spec("lmdb")
    if spec is None:
        print("  lmdb module not found in Python path")
    else:
        print(f"  But found at: {spec.origin}")
