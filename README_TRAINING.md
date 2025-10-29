# 训练指南

## 训练流程

### 步骤1: 数据预处理

首先需要将原始CSV文件处理成3600个点的格式：

```bash
python src/process.py
```

这会处理以下三个数据集：
- `ir_boraden.csv` → `data/processed/ir_broaden_processed.csv`
- `uv_boraden.csv` → `data/processed/uv_broaden_processed.csv`
- `raman_boraden.csv` → `data/processed/raman_broaden_processed.csv`

**注意**: 确保原始CSV文件在 `E:\SpectraViT\` 目录下。

### 步骤2: 训练模型

#### 基本训练命令

```bash
python src/train.py
```

#### 完整参数示例

```bash
python src/train.py \
    --data_dir data/processed \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 4e-4 \
    --beta_max 0.01 \
    --latent_dim 128 \
    --hidden_channels 128 \
    --save_dir checkpoints \
    --cpu  # 如果要用CPU而不是GPU，加上这个参数
```

#### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | `data/processed` | 处理后的数据目录 |
| `--epochs` | `20` | 训练轮数 |
| `--batch_size` | `32` | 批次大小 |
| `--learning_rate` | `4e-4` | 学习率 |
| `--beta_max` | `0.01` | KL散度最大权重（物理先验） |
| `--latent_dim` | `128` | 潜在空间维度 |
| `--hidden_channels` | `128` | 隐藏层通道数 |
| `--save_dir` | `checkpoints` | 模型保存目录 |
| `--cpu` | `False` | 是否使用CPU（默认使用GPU） |

### 步骤3: 查看训练过程

训练时会自动训练所有模态对：
- IR → UV
- IR → Raman
- UV → Raman
- UV → IR
- Raman → IR
- Raman → UV

每个模态对的训练会显示：
- 训练损失（总损失、重建损失、KL损失）
- 验证损失
- 最佳模型会自动保存

### 训练输出

模型会保存在 `checkpoints/` 目录下，文件名格式：
```
vae_{source_mode}2{target_mode}_best.pt
```

例如：
- `vae_ir2uv_best.pt`
- `vae_raman2ir_best.pt`

### 步骤4: 测试模型

训练完成后，可以测试模型：

```bash
python src/test.py \
    --source_mode ir \
    --target_mode uv \
    --checkpoint_dir checkpoints \
    --data_dir data/processed
```

## 训练特点

### 1. 物理先验
- 自动计算光谱物理参数（展宽、峰值等）
- 使用物理先验优化KL散度损失
- 参考：https://github.com/ymzhu19eee/Raman-generation

### 2. 自动数据分割
- 训练集：70%
- 验证集：15%
- 测试集：15%

### 3. 模型检查点
- 每个模态对独立保存最佳模型
- 基于验证集重建损失选择最佳模型

## 常见问题

### Q: 如何只训练特定的模态对？
A: 目前代码会训练所有模态对。如果想只训练特定对，可以注释掉 `train.py` 中 `mode_pairs` 列表里不需要的对。

### Q: 训练很慢怎么办？
A: 
- 确保使用GPU（去掉 `--cpu` 参数）
- 减小 `--batch_size`
- 减小 `--latent_dim` 和 `--hidden_channels`

### Q: 如何恢复训练？
A: 目前代码不支持断点续训，但可以加载已保存的模型继续训练（需要修改代码添加加载检查点的功能）。

## 数据格式要求

处理的CSV文件格式：
- 第一行：x轴数据（波数/波长）
- 后续行：每个样本的光谱数据
- 列：数据点

处理后的数据：
- 每个光谱统一为3600个点
- 转换为热图格式 (60x60) 用于训练


