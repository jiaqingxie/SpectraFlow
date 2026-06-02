"""
可视化 Flow Matching 的采样过程
展示从 source 到 target 的路径和速度场
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def create_2d_gaussian(center, cov, size=100):
    """创建2D高斯分布"""
    x = np.linspace(-3, 3, size)
    y = np.linspace(-3, 3, size)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    
    # 计算多元高斯PDF
    det = np.linalg.det(cov)
    inv_cov = np.linalg.inv(cov)
    diff = pos - center
    exp_term = np.exp(-0.5 * np.einsum('ijk,kl,ijl->ij', diff, inv_cov, diff))
    pdf = (1.0 / (2 * np.pi * np.sqrt(det))) * exp_term
    
    return X, Y, pdf

def create_multimodal_distribution(size=100):
    """创建多峰分布（模拟target）"""
    x = np.linspace(-3, 3, size)
    y = np.linspace(-3, 3, size)
    X, Y = np.meshgrid(x, y)
    
    # 两个高斯峰的混合
    center1 = np.array([1.5, 1.5])
    center2 = np.array([-1.5, -1.5])
    cov1 = np.array([[0.4, 0.1], [0.1, 0.4]])
    cov2 = np.array([[0.4, -0.1], [-0.1, 0.4]])
    
    _, _, pdf1 = create_2d_gaussian(center1, cov1, size)
    _, _, pdf2 = create_2d_gaussian(center2, cov2, size)
    pdf = 0.5 * pdf1 + 0.5 * pdf2
    
    return X, Y, pdf

def compute_velocity_field(x_t, x0, x1):
    """
    计算速度场（Flow Matching 的目标速度）
    v_t = x1 - x0 (线性路径的导数)
    """
    return x1 - x0

def create_sample_image(size=32, mode='noise'):
    """创建示例图像（噪声或结构化图案）"""
    if mode == 'noise':
        # 随机噪声图（模拟 IR 光谱热图）
        img = np.random.rand(size, size)
        # 添加一些结构
        x = np.linspace(0, 4*np.pi, size)
        y = np.linspace(0, 4*np.pi, size)
        X, Y = np.meshgrid(x, y)
        img += 0.3 * np.sin(X) * np.cos(Y)
        img = (img - img.min()) / (img.max() - img.min())
    else:
        # 结构化图案（模拟 Raman 光谱热图）
        img = np.zeros((size, size))
        x = np.linspace(0, 4*np.pi, size)
        y = np.linspace(0, 4*np.pi, size)
        X, Y = np.meshgrid(x, y)
        img = 0.5 + 0.5 * np.sin(2*X) * np.cos(2*Y)
        # 添加一些峰值
        center = size // 2
        for i in range(size):
            for j in range(size):
                dist = np.sqrt((i-center)**2 + (j-center)**2)
                img[i, j] += 0.3 * np.exp(-dist**2 / (2 * (size/6)**2))
        img = np.clip(img, 0, 1)
    return img

def visualize_flow_matching():
    """可视化 Flow Matching 采样过程 - 左右分布 + 下方小图风格"""
    # 创建主图和子图
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[1, 1], 
                          hspace=0.3, wspace=0.2)
    
    # 主分布图（上半部分，跨两列）
    ax_main = fig.add_subplot(gs[0, :])
    
    # 定义 source (IR) 和 target (Raman) 位置
    x0_center = np.array([-2.5, 0.0])  # IR 分布中心（左侧）
    x1_center = np.array([2.5, 0.0])   # Raman 分布中心（右侧）
    
    # 创建坐标网格
    size = 200
    x = np.linspace(-5, 5, size)
    y = np.linspace(-3, 3, size)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    
    # 创建 IR 分布（单峰，圆形，与 Raman 区分开）
    ir_cov = np.array([[0.6, 0.0], [0.0, 0.6]])  # 圆形单峰
    diff_ir = pos - x0_center
    inv_cov_ir = np.linalg.inv(ir_cov)
    det_ir = np.linalg.det(ir_cov)
    exp_ir = np.exp(-0.5 * np.einsum('ijk,kl,ijl->ij', diff_ir, inv_cov_ir, diff_ir))
    ir_pdf = (1.0 / (2 * np.pi * np.sqrt(det_ir))) * exp_ir
    ir_pdf = ir_pdf / ir_pdf.max()
    
    # 创建 Raman 分布（多峰，红色调）
    raman_center1 = x1_center + np.array([1.0, 0.8])
    raman_center2 = x1_center + np.array([-1.0, -0.8])
    raman_cov1 = np.array([[0.5, 0.18], [0.18, 0.5]])
    raman_cov2 = np.array([[0.5, -0.18], [-0.18, 0.5]])
    
    diff_raman1 = pos - raman_center1
    diff_raman2 = pos - raman_center2
    inv_cov_raman1 = np.linalg.inv(raman_cov1)
    inv_cov_raman2 = np.linalg.inv(raman_cov2)
    det_raman1 = np.linalg.det(raman_cov1)
    det_raman2 = np.linalg.det(raman_cov2)
    
    exp_raman1 = np.exp(-0.5 * np.einsum('ijk,kl,ijl->ij', diff_raman1, inv_cov_raman1, diff_raman1))
    exp_raman2 = np.exp(-0.5 * np.einsum('ijk,kl,ijl->ij', diff_raman2, inv_cov_raman2, diff_raman2))
    raman_pdf = 0.5 * (1.0 / (2 * np.pi * np.sqrt(det_raman1))) * exp_raman1 + \
                0.5 * (1.0 / (2 * np.pi * np.sqrt(det_raman2))) * exp_raman2
    raman_pdf = raman_pdf / raman_pdf.max()
    
    # 绘制 IR 分布（蓝色梯田效果，左侧）
    ir_levels = np.linspace(0.15, 0.95, 10)
    cs_ir = ax_main.contourf(X, Y, ir_pdf, levels=ir_levels, 
                             colors=plt.cm.Blues(np.linspace(0.4, 0.95, len(ir_levels)-1)),
                             alpha=0.6, zorder=1)
    cs_ir_lines = ax_main.contour(X, Y, ir_pdf, levels=ir_levels, 
                                   colors='darkblue', linewidths=1.0, alpha=0.6, zorder=2)
    
    # 绘制 Raman 分布（红色梯田效果，右侧）
    raman_levels = np.linspace(0.15, 0.95, 10)
    cs_raman = ax_main.contourf(X, Y, raman_pdf, levels=raman_levels,
                                colors=plt.cm.Reds(np.linspace(0.4, 0.95, len(raman_levels)-1)),
                                alpha=0.6, zorder=1)
    cs_raman_lines = ax_main.contour(X, Y, raman_pdf, levels=raman_levels,
                                     colors='darkred', linewidths=1.0, alpha=0.6, zorder=2)
    
    # 在分布上标注采样点（黑点）
    # IR 分布的采样点（在中心附近）
    ir_sample = x0_center + np.array([0.3, 0.2])
    ax_main.plot(ir_sample[0], ir_sample[1], 'ko', markersize=8, zorder=10)
    
    # Raman 分布的采样点（在第二个峰的内层）
    raman_sample = raman_center2 + np.array([-0.2, -0.1])
    ax_main.plot(raman_sample[0], raman_sample[1], 'ko', markersize=8, zorder=10)
    
    # 设置主图坐标轴
    ax_main.set_xlim(-5, 5)
    ax_main.set_ylim(-3, 3)
    ax_main.set_xlabel('Latent Space', fontsize=12, fontweight='bold')
    ax_main.set_ylabel('', fontsize=12)
    ax_main.set_xticks([])
    ax_main.set_yticks([])
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)
    ax_main.spines['bottom'].set_visible(False)
    ax_main.spines['left'].set_visible(False)
    
    # 创建下方两个小图
    ax_ir_img = fig.add_subplot(gs[1, 0])
    ax_raman_img = fig.add_subplot(gs[1, 1])
    
    # 生成示例图像
    ir_img = create_sample_image(size=32, mode='noise')
    raman_img = create_sample_image(size=32, mode='structured')
    
    # 绘制 IR 图像
    ax_ir_img.imshow(ir_img, cmap='viridis', aspect='equal', interpolation='nearest')
    ax_ir_img.set_title(r'$X_0$', fontsize=16, fontweight='bold', pad=10)
    ax_ir_img.axis('off')
    
    # 绘制 Raman 图像
    ax_raman_img.imshow(raman_img, cmap='viridis', aspect='equal', interpolation='nearest')
    ax_raman_img.set_title(r'$X_1$', fontsize=16, fontweight='bold', pad=10)
    ax_raman_img.axis('off')
    
    
    plt.suptitle('Flow Matching: IR → Raman', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig('flow_matching_ir_raman.png', dpi=300, bbox_inches='tight')
    print("Saved visualization to: flow_matching_ir_raman.png")
    plt.show()

if __name__ == '__main__':
    visualize_flow_matching()
