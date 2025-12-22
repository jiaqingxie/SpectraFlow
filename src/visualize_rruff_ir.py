"""
可视化RRUFF IR原始数据，查看所有光谱的原始样子
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def load_rruff_txt(txt_path):
    """加载RRUFF格式的txt文件"""
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
        return None, None
    
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
    
    return np.array(x_data), np.array(y_data)


def visualize_all_spectra(input_dir, output_dir='rruff_ir_plots', max_spectra=None):
    """
    可视化所有IR光谱，每个光谱保存为独立的PNG文件
    
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径（每个光谱会保存为独立的PNG文件）
        max_spectra: 最多显示多少个光谱（None表示全部）
    """
    input_path = Path(input_dir)
    txt_files = sorted(list(input_path.glob('*.txt')))
    
    if not txt_files:
        print(f"No txt files found in {input_dir}")
        return
    
    if max_spectra:
        txt_files = txt_files[:max_spectra]
    
    print(f"Loading {len(txt_files)} IR spectra...")
    
    all_spectra = []
    all_x_axes = []
    successful_files = []  # 保存成功加载的文件，用于后续命名
    failed_files = []
    
    for txt_file in txt_files:
        try:
            x_data, y_data = load_rruff_txt(txt_file)
            if x_data is not None and len(x_data) > 0:
                all_spectra.append(y_data)
                all_x_axes.append(x_data)
                successful_files.append(txt_file)
        except Exception as e:
            print(f"Error loading {txt_file.name}: {e}")
            failed_files.append(txt_file.name)
    
    if not all_spectra:
        print("No spectra loaded")
        return
    
    print(f"Successfully loaded {len(all_spectra)} spectra")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_path}")
    
    # 为每个光谱单独创建并保存PNG文件
    for i, (x_data, y_data) in enumerate(zip(all_x_axes, all_spectra)):
        # 创建单个图形
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x_data, y_data, linewidth=1.5, color='blue')
        
        # 获取文件名（去掉路径和扩展名）
        label = successful_files[i].name.replace('.txt', '')
        
        ax.set_title(label, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=11)
        ax.set_ylabel('Intensity', fontsize=11)
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存文件（使用原始文件名）
        output_file = output_path / f"{label}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        if (i + 1) % 50 == 0:
            print(f"  Saved {i + 1}/{len(all_spectra)} spectra...")
    
    print(f"\nSaved all {len(all_spectra)} spectra to: {output_path}")
    
    # 打印统计信息
    print("\nStatistics:")
    print(f"  Number of spectra: {len(all_spectra)}")
    print(f"  X-axis range: [{min(x.min() for x in all_x_axes):.2f}, {max(x.max() for x in all_x_axes):.2f}]")
    print(f"  Y-axis range: [{min(y.min() for y in all_spectra):.2f}, {max(y.max() for y in all_spectra):.2f}]")
    
    # 统计每个光谱的最大值
    max_values = [y.max() for y in all_spectra]
    print(f"  Max intensity - Min: {min(max_values):.2f}, Max: {max(max_values):.2f}, Mean: {np.mean(max_values):.2f}")
    
    if failed_files:
        print(f"\nFailed to load {len(failed_files)} files")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize RRUFF IR raw spectra (each as separate PNG)')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory containing IR txt files')
    parser.add_argument('--output_dir', type=str, default='rruff_ir_plots',
                       help='Output directory for PNG files (each spectrum saved separately)')
    parser.add_argument('--max_spectra', type=int, default=None,
                       help='Maximum number of spectra to visualize (None = all)')
    
    args = parser.parse_args()
    
    visualize_all_spectra(args.input_dir, args.output_dir, args.max_spectra)


if __name__ == '__main__':
    main()

