# SpectraFlow 复现核查与投稿准备

核查日期为 2026 年 10 月 3 日。依据是最新 Overleaf 稿件和当前共享目录中的代码、数据与 checkpoint。论文快照对应提交 `b42f38bdc48aa32b04a250409a0ef2df00a43508`；本地代码仓库 HEAD 为 `e74b94733960a46307b1474ad07dbff522b7b7f5`，本次新增的审查脚本尚未提交。

**主要 checkpoint 可以运行，并能在小批样本上复现保存的预测；目前不能称为主测试集全量重跑或四种子统计复现。投稿前最紧急的问题是 NIST 实验配对错误，以及现有预处理与论文所述波数顺序不一致。** 我建议先修复这两项、补齐运行证据，再决定是否投 Nature Computational Science。以下投稿优先级是根据当前证据作出的审稿判断，不代表期刊的录用承诺。

## 已经验证的结果

扫描了 184 个 checkpoint。六个主 SpectraDiT checkpoint 均成功严格加载，使用八步 Euler 和 8 核 CPU 各运行了 32 个固定测试样本，共 192 个模型输出。每组前 8 个输出与原保存预测逐点比较，共 48 个完整光谱；最大归一化绝对差为 `1.407e-5`，低于预设 `1e-4` 容差。CPU 是按用户后续允许的备选方案使用的。

另外，从保存的预测与目标 CSV **全量重新计算**了六组主结果及十六组 Flow 域迁移结果，共 260,386 个样本结果。重算的逐样本 R² 与保存值最大差约为 `1.2e-6`。这验证了结果文件内部的数值一致性，与从权重重新推理是两种证据。

| 数据集 | 方向 | 保存预测的有效样本数 | 全量复算 R² | 全量复算 Pearson | 全量复算 RMSE |
|---|---|---:|---:|---:|---:|
| QM9S | IR → Raman | 19,474 | 0.938876 | 0.975790 | 5.129666 |
| QM9S | Raman → IR | 19,474 | 0.830359 | 0.925525 | 7.007172 |
| QMe14S | IR → Raman | 27,660 | 0.858851 | 0.938168 | 6949.087912 |
| QMe14S | Raman → IR | 27,660 | 0.755306 | 0.888217 | 3.630140 |
| ViBench Full | IR → Raman | 13,793 | 0.899785 | 0.955147 | 0.022338 |
| ViBench Full | Raman → IR | 13,793 | 0.864547 | 0.936105 | 0.022185 |

这些中心值接近论文主表。RMSE 使用原评估器的目标强度回标协议，三个数据集的单位与尺度不同，不宜直接横向比较；目标 extrema 并未输入模型，但没有目标模态时不能依赖这些 extrema 恢复绝对强度。

QMe14S 的固定测试划分共有 27,916 行。保存文件在两个方向均排除了同样的 256 行。核对原 HDF5 后，**这 256 行的 IR 和 Raman 都是全零谱**，没有非有限数值。原评估器遇到未定义的 Pearson 等指标后跳过整行。[原始排除行号](reproduction_audit/qme14s_excluded_test_rows.csv) 已保存。应预先规定这一数据质量排除，公开有效分母，并对所有方法统一应用。

详细证据见 [主结果复算](reproduction_audit/recomputed_summary.csv)、[逐个权重的小批运行](reproduction_audit/checkpoint_smoke_cpu/checkpoint_summary.csv)、[域迁移复算](reproduction_audit/ood_recomputed/recomputed_summary.csv) 与 [checkpoint 清单](reproduction_audit/checkpoint_inventory.json)。小批均值不是全测试集估计，不应替换主表。

## NIST 配对必须重做

原分析从 `../data/IR_broaden.zip` 按位置恢复所谓 QM9S SMILES，但这个文件实际是 QMe14S 的光谱包。证据包括：

- ZIP 含 186,102 个分子文件；首个分子是 `CCS(=O)(=O)CCO`。
- 首个原始光谱经同样插值后，与 QMe14S HDF5 首行逐点完全一致；与 QM9S HDF5 首行最大差为 71.0635。
- 原匹配表的 QM9S 第 1153 行被标为 `CC=CCCl`，而真实 QM9S 映射和原始分子文件均为 `CC(C)CCCO`。第 1214 行也被错误标为含三枚 Br 的分子。
- 对 84 个目标行逐条核对原始 QM9S 分子文件，再用 RDKit 比较分子图：**立体化学匹配为 0/84，忽略立体化学后的连接关系匹配同样为 0/84**。

因此，虽然原 84 对的 Pearson 中位数 0.421 可以数值复算，原 NIST 结果及其挑选案例、波数误差曲线、bootstrap 区间不能作为身份匹配实验的证据。需要重建配对后重新推理及出图。该错误不改变主光谱重建 R² 的数值，但凡使用这个 ZIP 或 `recover_qm9s_aligned_smiles.py` 恢复分子身份的结构图、scaffold 分析及标签，都应追查来源；目前没有据此判定所有这些分析均错误。

已使用真正的 QM9S `mapping.txt` 重新生成匹配，得到 **38 个 NIST CID、39 个 QM9S 匹配行**。一条 CID 对应多个候选，不能直接当作独立分子重复计数。固定 seed 2 下，这 39 行中训练身份占 31 行、验证占 3 行、测试占 5 行。

本地已有输入光谱覆盖 33 行、32 个 CID，我已使用原 QM9S checkpoint 对这 33 行重新推理。结果为 Pearson 均值 0.382586、中位数 0.380460，平均 R² 为 −1.034415。该子集含 27 个训练身份行、2 个验证行、4 个测试行，不能称为新分子外推验证。输入轴沿用论文的“399–4000 cm⁻¹ 升序”假设；对应实验 IR 与计算 IR 的 Pearson 中位数升序为 0.139、反向为 −0.025，支持继续核查升序协议，但不能代替原来源的波数轴、单位及测量条件记录。

剩余缺失的原始光谱 CID 为 `107131`、`20309777`、`21702841`、`529340`、`565593`、`637978`。补齐后仍需处理身份歧义、固定分母及分子层面的统计。

证据见 [逐对身份核查](reproduction_audit/experimental/nist_identity_audit.csv)、[正确匹配清单](reproduction_audit/experimental/corrected_nist_qm9s_identity_matches.csv)、[当前可用输入清单](reproduction_audit/experimental/corrected_nist_available_pairs.csv)、[波数方向敏感性](reproduction_audit/experimental/corrected_nist_source_axis_sensitivity.csv) 与 [新推理结果](reproduction_audit/checkpoint_experimental_cpu/nist_corrected_available_ir2raman_seed2/summary.json)。这些文件没有覆盖原结果。

## RRUFF 实验结果可以复现

147 个外部测试对已从对应 checkpoint **全部重新推理**。平均 R² 为 −1.041316，Pearson 均值为 0.192593、中位数为 0.309273，与保存结果一致。

保存的光谱需要恢复物理波数顺序：ordered 训练结果在原评估器中又经过了 legacy inverse heatmap 转换。恢复后，目标与原 HDF5 的最大差为 `5.0e-8`。在真实波数顺序上按论文参数做 AsLS 基线校正，Pearson 中位数为 0.339756，复现正文的 0.34。

训练池为 4720 对、590 个 RRUFF ID、379 个矿物名。外部测试为 147 对、147 个 ID、135 个矿物名，与开发池没有 ID 交集，但有 72 个矿物名交集。这些统计与最新版正文一致。Rutile、cassiterite、staurolite 三个所选案例的原始相关系数也能对上：0.953、0.942、0.852。

这验证了现有实验结果，没有改善实验表现。平均 R² 为负，且高相关案例含明显宽背景，仍需评估基线之外的峰恢复、峰位与峰强误差，并与简单实验域基线和重复测量的一致性比较。内部 checkpoint-selection 划分按构造对随机拆分、共享 RRUFF ID；建议按 ID 重做验证集划分，另报矿物名不重叠的测试。跨库共享 ID 也不保证同一试样或同一测量条件。

见 [完整新推理](reproduction_audit/checkpoint_experimental_cpu/rruff_external_ir2raman_seed42/summary.json)、[逐对复算及基线校正](reproduction_audit/experimental/rruff_recomputed_per_pair.csv) 与 [实验核查汇总](reproduction_audit/experimental/experimental_summary.json)。

## 代码与论文协议仍有差距

**波数邻接关系。** `src/train.py` 的 `preserve_spectral_order` 默认为 false，现有主结果复现路径沿用 patch 拼图。60×60 时，flatten 后前二十个物理索引为 `0…9, 100…109`，不是连续波数。`train_flow.py` 的导数、曲率与局部 OT 直接作用于这个 flatten 序列，而论文写的是相邻波数上的约束。R² 等总体指标在共同排列下不受影响，但对局部光谱损失及波数位置编码的解释受影响。应先明确旧实验实际使用的排列；若要验证论文所述有序机制，应在保持物理顺序的预处理下重训并做匹配对照，不能直接给旧权重切换输入顺序。

**训练种子与数据划分。** 主 SpectraDiT 权重目前只有 seed 2；ViBench OOD SpectraDiT 有 seed 0 和 1。较多其他四种子 Flow 文件实际属于旧 UNet backbone，不能作为主 SpectraDiT 四种子证据。184 个 checkpoint 中仅两个 Direct 文件包含运行 config，所有文件都没有内嵌 split IDs。当前 `get_paired_loaders` 使用同一个 seed 产生划分，缺少独立的 split seed 与初始化 seed。需要找回四组原运行及相同划分清单；找不到时，就重新固定划分、独立改变初始化，并重算 SD 与统计检验。不能根据文件名猜测四种子均值或拿分子间 SD 代替训练种子 SD。

**域迁移评估范围。** ViBench QM9 的 supplied test HDF5 有 26,687 行，保存 Flow 结果只有 4,004 行，恰为再次随机拆分后的约 15%。最新版论文写全 supplied test 文件评估；因此这份保存结果不满足论文协议。其他保存 OOD 均值与主文种子平均表亦不完全对应，可能来自不同 checkpoint 或 endpoint 设置，需提供运行映射。复现脚本明确区分 `--protocol archived` 与 `--protocol paper`。

ViBench Full 的本地矩阵为 91,949 行，13,793 是其随机测试划分大小。不能把数据资源总规模、过滤后矩阵和本次训练集规模混用。ViBench 的 Mols 本身是 QM9 与 ZINC15 的组合，并非独立的全新化学域；应先按分子图、立体异构体、构象及 scaffold 做交叉资源重叠核查，再解释 OOD 汇总。[ViBench 原论文的数据定义](https://arxiv.org/html/2503.07014v4)

**下游证据。** 保存的五划分 ridge 汇总支持部分表示收益，但 GEOM LogP 的原始 Raman R² 为 0.5148，Flow 为 0.6779，与正文“约 0.11 到约 0.70”的起点不符，应追踪正文对应的运行。十个描述符、八个数据集上的 Flow+FP 相对 FP 有 56 个正变化、24 个负变化，平均绝对 R² 变化 0.0255。应公开逐任务差异，不宜泛化成所有任务受益。NIST-IR 的 fingerprint 表现及部分 raw-spectrum 极负 R² 需要核查样本对齐、异常值和正则化；当前证据不足以确定原因。UMAP 脚本采用的 heatmap patch 候选还与训练预处理不同，应统一后再解释 baseline 表示。

本次没有重新训练四种子模型、重算主表 Wilcoxon/Holm 显著性、验证全部 baseline 的运行对应关系，也没有复核所有峰分析数值。公开代码库已有峰分析入口，但当前本地归档还缺少这些结果对应的可追溯运行与统计产物。

## 面向 Nature Computational Science 的补充优先级

期刊范围关注具有实质贡献的计算方法和计算带来的科学认识，化学在其覆盖范围内。下列建议着重解决本文的证据与贡献问题。[期刊范围](https://www.nature.com/natcomputsci/about/aims)

| 优先级 | 需要完成的工作 | 完成标准 |
|---|---|---|
| P0 | 修复 NIST 分子配对并追踪相关结构标签来源 | 正确分子图与原光谱行一致，处理歧义，补齐输入来源和轴信息，重算结果与图 |
| P0 | 统一预处理、方法描述、测试分母与运行来源 | 每个表格数字能对应 config、划分 IDs、权重 SHA256、代码版本和逐分子预测；统一记录全零谱排除 |
| P0 | 补齐固定划分的四种子主模型及 paired statistics | 同一划分、独立初始化；报告种子间 SD，明确 molecule-level 配对方式和多重校正 |
| P1 | 匹配预算的 Flow 与 Direct 对照 | 相同 backbone、数据、损失系数、训练与调参预算；比较 NFE 1/2/4/8/16 的误差与实测成本，并做主要损失消融 |
| P1 | 严格独立的化学与实验验证 | 分子身份及 scaffold 重叠审计；RRUFF 按 ID 选 checkpoint，另报矿物名独立测试；优先补同试样、条件匹配的 IR/Raman |
| P1 | 展示具有实际用途的表示收益 | 在公平调参的 encoder/contrastive baselines 下测试；增加检索、结构候选排序或实验性质任务，并证明预测补全确实提高决策结果 |
| P2 | 不确定性、失败条件与鲁棒性 | 区分不可辨识性与模型误差；对峰缺失、频移、背景、噪声和来源变化给出校准及拒绝机制 |

最新版已经承认 Direct 的训练预算不匹配、目标强度回标限制、数值轨迹不是物理过程、nonnegative 项在 clamp 后无独立惩罚、下游 probe 在每个域分别拟合，以及实验跨库配对局限。这些说明应保留；现在需要补对应证据和实验，不必重复增加相同的免责声明。

最能提升说服力的补充是：**先修正有序光谱预处理，再做预算匹配的 Flow/Direct 实验，并展示在严格独立的实验任务上带来的可量化收益。** 仅凭现有 VAE/Transformer 优势，还无法隔离 flow matching 的贡献。Vib2Mol 已覆盖光谱表示、检索和结构生成，因此本文需要清楚区分任务，说明翻译训练带来的额外价值，而不是把一般的谱学表示学习当作新增贡献。[Vib2Mol](https://arxiv.org/abs/2503.07014)

发布层面，现有 GitHub README 明确不跟踪数据、权重与结果；需要另外提供有版本的可访问 checkpoint、划分和最小演示。Nature 的计算工具与软件指南强调清楚的预处理和划分、版本与依赖、运行说明、演示数据及预期输出。受第三方许可限制的数据可以提供合法获取方式、标识符及处理清单，不能统一改写为作者自有数据许可。[代码仓库](https://github.com/jiaqingxie/SpectraFlow)、[计算工具报告指南](https://www.nature.com/documents/Computational_tools_reporting_guidelines.pdf)、[软件提交指南](https://media.nature.com/full/nature-cms/documents/GuidelinesCodePublication.pdf)

## 重画的图与运行入口

四张审查图按 Chen Liu 的 [scientific figure making skill](https://github.com/ChenLiu-1996/figures4papers/tree/main/scientific-figure-making) 整理。使用统一语义颜色、简洁坐标、可编辑文字，并提供 PDF、SVG、300 dpi PNG 和数值源文件。图是可审阅的本地草稿，没有覆盖 Overleaf 原图，也没有把未验证的四种子误差条画出来。

| 图 | 当前内容 | PDF |
|---|---|---|
| Results 1 | 保存 seed 2 的主结果中心值，以及归档单次 OOD Flow/VAE 对照 | [翻译结果](figures/reproduction/result_1_translation.pdf) |
| Results 2 | QM9S nMAE 50% 与 90% 分位附近的实际样本及残差 | [分位数案例](figures/reproduction/result_2_quantile_examples.pdf) |
| Results 3 | 十个描述符的平均表现与 Flow+FP 相对 FP 的有符号变化 | [下游结果](figures/reproduction/result_3_downstream.pdf) |
| Results 4 | RRUFF 全体 147 对的相关性分布及明确标注的高相关案例 | [实验结果](figures/reproduction/result_4_rruff_external.pdf) |

原 NIST 图暂不放入有效实验图。正确重配后的 33 行是初步核查子集，不能无说明地替换原 84 行图。每幅图的数据、选择规则与证据范围见 [图清单](figures/reproduction/figure_manifest.json)。

计算池当前使用工作区挂载的服务身份 `group-ailab-chemagent-chemagent-cpu`，不是个人账号 `xiejiaqing`。GPU 请求被这个服务身份的 workload 权限拒绝；这没有验证或否定个人账号的池子权限。`brainctl login --username xiejiaqing` 还需要 SSO issuer 等登录配置。仅替换用户名不能切换凭据。没有成功启动的 GPU 作业。

在已通过个人账号认证、拥有池子使用权的终端执行以下命令，即可开始主测试集全量重跑。环境默认使用共享目录中已有的 PyTorch 2.6.0 + CUDA 12.4 conda 环境，可以用 `REPRO_PYTHON` 指向其他已准备的环境。

```bash
bash run_reproduction_audit_gpu.sh --suite id \
  --output-dir /mnt/shared-storage-user/xiejiaqing/spectrogen_v2/reproduction_audit/checkpoint_full_gpu
```

核查论文所写的完整域迁移协议：

```bash
bash run_reproduction_audit_gpu.sh --suite ood --protocol paper \
  --output-dir /mnt/shared-storage-user/xiejiaqing/spectrogen_v2/reproduction_audit/checkpoint_ood_paper_gpu
```

如需对齐旧 QM9 归档的再次随机拆分，则改用 `--protocol archived`，并在结果中明确它不是论文当前所写的完整 supplied test 协议。OOD 注意力 head 数需要从原运行配置核实，默认 6 不能从权重形状推断。

本次有限 CPU 运行命令：

```bash
/mnt/shared-storage-user/xiejiaqing/miniconda3/envs/azr/bin/python \
  src/reproduce_paper_checkpoints.py --suite id --device cpu \
  --threads 8 --batch-size 8 --limit 32 \
  --output-dir reproduction_audit/checkpoint_smoke_cpu

/mnt/shared-storage-user/xiejiaqing/miniconda3/envs/azr/bin/python \
  src/reproduce_paper_checkpoints.py --suite experimental --device cpu \
  --threads 8 --batch-size 8 \
  --output-dir reproduction_audit/checkpoint_experimental_cpu
```

脚本保存逐样本指标、原 HDF5 行号、首批预测、checkpoint 校验值与求解器设置。`--suite direct` 可检查已保存的一步模型；当前没有把它作为隔离积分作用的公平对照。
