# CIFAR-10 图像分类（当代人工智能作业三）

基于 PyTorch 的 CIFAR-10 图像分类完整实现，覆盖作业基础要求：自定义 CNN、训练/验证集划分、数据增强与正则化、训练曲线、混淆矩阵、每类准确率、误分类样本分析，以及可复现的运行说明。

## 1. 基础要求完成情况

| 作业要求 | 实现位置 | 产出 |
|---|---|---|
| 设计一个 CNN 模型 | `src/model.py` | 6 层卷积的 VGG 风格 CNN（BN + Dropout + GAP） |
| 自行划分训练集/验证集 | `src/data.py` | 45,000 训练 / 5,000 验证（按类别分层，固定种子） |
| 在测试集上评估准确率 | `train.py`、`evaluate.py` | 10,000 张测试集的总体/每类准确率 |
| 数据增强 | `src/data.py` | RandomCrop(32, padding=4) + RandomHorizontalFlip；`--no-augment` 可关闭增强做对比 |
| 正则化 | `src/model.py`、`train.py` | BatchNorm、Dropout、weight decay、梯度裁剪 |
| 训练过程可视化 | `src/visualization.py` | `training_curves.png`、`history.csv/json` |
| 混淆矩阵与每类/总准确率 | `src/metrics.py`、`src/evaluation.py` | `confusion_matrix*.png/csv`、`per_class_accuracy.png` |
| 误分类样本分析 | `src/visualization.py`、`src/evaluation.py` | `misclassified_samples.png/csv`、置信度分布、Top 混淆对 |
| 代码规范与 README | 全项目 | 模块化、类型标注、中文注释、本文件 |

## 2. 目录结构

```
第三次作业/
├── train.py                  # 训练入口：数据、训练、验证、测试、报告素材
├── evaluate.py               # 加载 checkpoint，单独复现测试集评估
├── src/
│   ├── __init__.py
│   ├── data.py               # CIFAR-10 加载、分层划分、数据增强、DataLoader
│   ├── model.py              # CIFARCNN 模型、可选 SE 注意力与模型工厂
│   ├── engine.py             # 训练/验证循环、checkpoint
│   ├── metrics.py            # 混淆矩阵、每类指标、Top-k、Top 混淆对
│   ├── visualization.py      # 曲线、混淆矩阵、置信度、误分类网格
│   ├── evaluation.py         # 测试集评估与所有报告文件输出
│   └── utils.py              # 随机种子、设备、AverageMeter、JSON/checkpoint
├── requirements.txt
├── scripts/
│   ├── compare_runs.py        # 两个 run 的 McNemar / 配对 bootstrap / 每类差异
│   ├── analyze_interaction.py # 调度 × SE 的 2×2 交互与联合 bootstrap
│   └── analyze_se.py          # SE 门控统计（通道/样本/交互项）
└── README.md           
```

运行后自动生成（每次实验一个子目录；下例为 100 epoch 最终模型）：

```
data/                        # CIFAR-10 数据（~170 MB，已加入 .gitignore）
outputs/
└── final100/                 # 100 epoch 最终模型；其他实验使用各自的子目录
    ├── args.json            # 本次运行的完整参数
    ├── split_indices.json   # 训练/验证划分索引（保证可复现）
    ├── history.json / .csv  # 每个 epoch 的 loss / acc / lr
    ├── training_curves.png
    ├── best.pt / last.pt    # 最佳验证权重 / 最新权重
    ├── run_summary.json
    ├── test_metrics.json
    ├── classification_report.txt
    ├── confusion_matrix.csv
    ├── confusion_matrix.png
    ├── confusion_matrix_normalized.png
    ├── per_class_accuracy.png
    ├── confidence_distribution.png
    └── misclassified_samples.png / .csv
```

> `final100/` 是训练目录；`final100_eval/` 是 `evaluate.py` 的独立评估目录，配对检验所需的 `test_predictions.npz` 位于后者。`final30/`、`final30_eval/` 同理对应 30 epoch 最终配置及其评估。
>
> `outputs/` 下所有实验/调试子目录的配置、结果和用途见 **3.1 节**。

## 3. 输出文件说明

### 3.1 outputs 子目录一览

`outputs/` 下每个子目录对应一次训练或独立评估。`test_skipped=true` 表示训练阶段没有在测试集上评估，测试结果需要看对应的 `_eval/` 目录；百分数为 CIFAR-10 测试集准确率，`val` 表示验证集最佳准确率。

| 目录 | 配置 / 实验 | 结果 | 用途与备注 |
|---|---|---|---|
| `bs64/` | 30 epoch，batch size 64，默认增强 + cosine | test 91.28% | batch size 超参对比 |
| `final100/` | 100 epoch，默认增强 + cosine，最终配置 | test 92.66% | **最终模型**，报告主结果 |
| `final100_eval/` | 独立评估 `final100/best.pt` | test 92.66% | 生成 `test_predictions.npz`，供配对检验使用 |
| `final30/` | 30 epoch，默认增强 + cosine | test 91.27% | 30 epoch 短程基线 |
| `final30_eval/` | 独立评估 `final30/best.pt` | test 91.27% | 30 epoch 基线的独立评估目录 |
| `lr001/` | 30 epoch，学习率 0.01，默认增强 + cosine | test 86.25% | 学习率超参对比 |
| `no_aug100/` | 100 epoch，关闭数据增强，cosine，训练时跳过测试 | val 90.92% | 数据增强消融训练目录 |
| `no_aug100_eval/` | 独立评估 `no_aug100/best.pt` | test 90.72% | 数据增强消融的测试结果 |
| `no_aug_no_sched_eval/` | 空目录，无评估文件 | — | 评估未完成；暂无可用结果，不应引用 |
| `no_aug_no_sched_no_se/` | 100 epoch，关闭增强、关闭调度、无 SE | val 84.34%，test 83.46% | 最小基线，已完成 |
| `no_aug_no_sched_no_se_eval/` | 独立评估 `no_aug_no_sched_no_se/best.pt` | test 83.46% | 最小基线的独立评估与配对预测 |
| `no_aug_no_sched_se/` | 100 epoch，关闭增强、关闭调度，SE 全 stage，`r=8` | val 85.38%，test 84.64% | 无增强 + 无调度条件下的 SE 对比 |
| `no_aug_no_sched_se_eval/` | 独立评估 `no_aug_no_sched_se/best.pt` | test 84.64% | SE 对比的独立评估与配对预测 |
| `no_sched100/` | 100 epoch，默认增强，`scheduler=none`，训练时跳过测试 | val 85.82% | 学习率调度消融训练目录 |
| `no_sched100_eval/` | 独立评估 `no_sched100/best.pt` | test 85.19% | 无调度对照的测试结果 |
| `no_sched100_se/` | 100 epoch，默认增强，`scheduler=none`，SE 全 stage，`r=8`，训练时跳过测试 | val 87.38% | 补全“增强开启、无调度”条件下的 SE 对照 |
| `no_sched100_se_eval/` | 独立评估 `no_sched100_se/best.pt` | test 86.46% | SE×调度 2×2 的独立评估与配对预测 |
| `se/` | 30 epoch，SE 全 stage，`r=16` | test 91.21% | 早期 SE 初筛实验 |
| `se100_r8/` | 100 epoch，SE 全 stage，`r=8` | test 92.61% | 最终 SE 对比模型 |
| `se100_r8_eval/` | 独立评估 `se100_r8/best.pt` | test 92.61% | SE 与最终模型的配对比较目录 |
| `screen/` | 筛选实验根目录，无直接训练结果 | — | 存放只使用验证集的短程筛选实验 |
| `screen/se_s12_r8_b2/` | 30 epoch，SE stage1,2，`r=8`，`se_init_bias=2.0`，跳过测试 | val 91.40% | SE 插入位置/初始化筛选；不做测试集评估 |
| `smoke_noaug/` | 1 epoch，`max_batches=1`，关闭增强，跳过测试 | 无参考价值 | 冒烟测试，仅验证 `--no-augment` 路径、日志和 checkpoint 能正常写出 |

`outputs/` 根目录还有八个配对比较/交互结果文件，它们不是训练目录：

| 文件 | 内容 |
|---|---|
| `compare_final100_vs_noaug.json` | `final100` 与 `no_aug100` 的 McNemar / 配对 bootstrap 对比 |
| `compare_final100_vs_no_aug_no_sched.json` | `final100` 与 `no_aug_no_sched_no_se` 的 McNemar / 配对 bootstrap 对比 |
| `compare_final100_vs_se.json` | `final100` 与 `se100_r8` 的 McNemar / 配对 bootstrap 对比 |
| `compare_final100_vs_no_sched.json` | `final100` 与 `no_sched100` 的 McNemar / 配对 bootstrap 对比 |
| `compare_no_aug_no_sched_vs_se.json` | `no_aug_no_sched_no_se` 与 `no_aug_no_sched_se` 的 McNemar / 配对 bootstrap 对比 |
| `compare_no_sched_vs_no_sched_se.json` | `no_sched100` 与 `no_sched100_se` 的 McNemar / 配对 bootstrap 对比 |
| `compare_se100_vs_no_sched_se.json` | `se100_r8` 与 `no_sched100_se` 的 McNemar / 配对 bootstrap 对比 |
| `compare_interaction_se_sched.json` | 固定增强开启时“调度 × SE”的 2×2 交互、联合 bootstrap CI 与 p 值 |

其中 `compare_interaction_se_sched.json` 汇总的 2×2 结果为：SE 效应在无调度/有调度下分别为 +1.27/−0.05 pp，交互 +1.32 pp（95% CI [0.50, 2.18]），说明 SE 与调度部分替代。

训练目录通常包含 `args.json`、`split_indices.json`、`history.*`、checkpoint 和训练曲线；独立 `_eval/` 目录只保存评估指标、图表与 `test_predictions.npz`，不重复保存模型。

### 3.2 单个实验目录内的文件

| 文件 | 用途 |
|---|---|
| `history.csv/json` | 每 epoch 的 train/val loss、accuracy、learning rate |
| `training_curves.png` | 训练/验证 loss 与 accuracy 曲线，标出最佳 epoch |
| `test_metrics.json` | 总准确率、Top-3、macro/weighted P/R/F1、每类指标、Top 混淆对、置信度统计 |
| `test_predictions.npz` | 逐样本 `targets` / `preds` / `probs`，用于 McNemar 与配对 bootstrap |
| `classification_report.txt` | 可直接放入报告的每类分类报告 |
| `confusion_matrix.png` | 混淆矩阵计数图 |
| `confusion_matrix_normalized.png` | 按真实类别归一化的混淆矩阵（更适合比较类别） |
| `per_class_accuracy.png` | 每类测试准确率柱状图（含 10% 随机基线） |
| `confidence_distribution.png` | 正确/错误预测的最大 softmax 概率分布 |
| `misclassified_samples.csv` | 全部误分类样本：索引、真实类、预测类、置信度、真实类概率 |
| `misclassified_samples.png` | 置信度最高的 24 个误分类样本网格 |

报告中分析误分类时，建议结合（以下文件均位于 `outputs/final100/`）：

1. `confusion_matrix_normalized.png` 找出最易混淆的类别对；
2. `misclassified_samples.png` 观察高置信度错误（通常反映标注歧义或类内差异）；
3. `test_metrics.json` 中 `top_confusions` 与 `mean_confidence_correct/incorrect` 做定量说明；
4. `misclassified_samples.csv` 中 `image_index` 可回到原始测试集定位样本。

### SE 门控诊断与配对比较

逐样本预测由 `train.py` / `evaluate.py` 自动生成。两个 run 需要先在同一测试集上评估，然后可以：

```bash
# McNemar 精确检验 + 配对 bootstrap + 每类差异
python scripts/compare_runs.py --run-a outputs/final100_eval --run-b outputs/se100_r8_eval

# 固定增强开启，估计“调度 × SE”的 2×2 交互与联合 bootstrap
python scripts/analyze_interaction.py --output outputs/compare_interaction_se_sched.json

# 统计 SE 门控均值/通道间 std/样本间 std，默认在验证集上运行
python scripts/analyze_se.py --checkpoint outputs/se100_r8/best.pt --split val
```

`analyze_se.py` 默认输出 `se_gate_stats_val.json`；如果 `channel_std` 远大于 `sample_std`，说明门控主要学成静态通道缩放；如果某 stage 的 `mean` 接近 1 且 `fraction_gt_0.9` 接近 1，说明该 stage 近似恒等映射。

## 4. 环境依赖

- Python >= 3.10
- PyTorch >= 2.4、torchvision >= 0.19
- numpy、matplotlib、tqdm

安装：

```bash
pip install -r requirements.txt
```

GPU 用户请按 [PyTorch 官网](https://pytorch.org/get-started/locally/) 选择对应 CUDA 版本的安装命令，例如：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

当前项目已在 `Python 3.13 + torch 2.9.0 + torchvision 0.24.0` 下通过语法检查、模型前向/反向测试，以及用合成 CIFAR 格式小数据完成的端到端快速验证（覆盖数据划分、训练/验证、checkpoint、断点续训、测试评估与全部图表生成）。此外，已在完整官方 CIFAR-10（50,000 训练 / 10,000 测试）上跑通 1 epoch 全流程作为真实数据校验（本机 CPU 测试准确率 50.34%，仅证明流程正确）；正式精度请以实际多 epoch 训练输出为准。

## 5. 数据准备

### 方式一：自动下载（推荐）

```bash
python train.py --download
```

首次运行会下载约 170 MB 的 `cifar-10-python.tar.gz` 并解压到 `data/cifar-10-batches-py/`。

### 方式二：手动放置

若自动下载超时（部分校园网/代理环境可能无法访问 `cs.toronto.edu`）：

1. 从 CIFAR-10 官网或镜像下载 `cifar-10-python.tar.gz`；
2. 放到 `data/` 目录下，确保结构为：
   ```
   data/cifar-10-batches-py/
   ├── batches.meta
   ├── data_batch_1 ... data_batch_5
   └── test_batch
   ```
3. 不加 `--download` 直接训练：`python train.py`

> 注意：`cifar-10-python.tar.gz` **不只是测试集**，它同时包含 5 个训练 batch（共 50,000 张）和 1 个测试 batch（10,000 张）。解压后 `train=True` 读取训练集，`train=False` 读取测试集。

官方压缩包 MD5 为 `c58f30108f718f92721af3b95e74349a`，可用于校验。解压后可用以下命令确认数据完整：

```bash
python -c "from torchvision.datasets import CIFAR10; tr=CIFAR10('data', train=True); te=CIFAR10('data', train=False); print('train', len(tr), 'test', len(te), tr.classes)"
```

预期输出 `train 50000 test 10000` 及 10 个类别名。

## 6. 快速开始

完整训练（默认 100 epoch，单卡 GPU 推荐）：

```bash
python train.py --download --epochs 100 --batch-size 128 --lr 0.1 --output-dir outputs/final100
```

训练结束后会自动用验证集最优的 `best.pt` 在测试集上评估，结果和全部图表输出到 `outputs/final100/`。

单独复现测试集评估：

```bash
python evaluate.py --checkpoint outputs/final100/best.pt
```

快速调试（只跑 2 个 batch，验证流程是否跑通，结果无意义）：

```bash
python train.py --epochs 1 --max-batches 2 --batch-size 32 --num-workers 0 --output-dir outputs/smoke
```

断点续训：

```bash
python train.py --resume outputs/final100/last.pt --output-dir outputs/final100
```

> 续训时需保持 `--seed`、`--val-ratio`、`--epochs`、`--model` 与原实验一致，程序会校验并在不一致时退出。

## 7. 主要参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--data-root` | `data` | 数据集根目录 |
| `--output-dir` | `outputs` | 结果输出目录 |
| `--download` | 关闭 | 自动下载 CIFAR-10 |
| `--epochs` | 100 | 训练轮数 |
| `--batch-size` | 128 | batch size |
| `--lr` | 0.1 | SGD 初始学习率 |
| `--momentum` | 0.9 | SGD momentum（Nesterov） |
| `--weight-decay` | 5e-4 | L2 正则化 |
| `--label-smoothing` | 0.0 | 可选 label smoothing（加分实验用） |
| `--grad-clip` | 5.0 | 梯度裁剪阈值，0 关闭 |
| `--val-ratio` | 0.1 | 验证集比例（分层抽样） |
| `--seed` | 42 | 随机种子 |
| `--deterministic` | 关闭 | 开启 cuDNN 确定性模式 |
| `--num-workers` | 4 | DataLoader 进程数，Windows 出错时设 0 |
| `--no-augment` | 关闭 | 关闭训练集的 RandomCrop / RandomHorizontalFlip，仅保留标准化 |
| `--device` | `auto` | `auto` / `cpu` / `cuda` / `cuda:1` |
| `--amp` | 关闭 | CUDA 混合精度训练 |
| `--scheduler` | `cosine` | 学习率调度器：`cosine` / `step` / `none` |
| `--scheduler-step-size` | 30 | StepLR 的衰减间隔，仅 `--scheduler step` 生效 |
| `--scheduler-gamma` | 0.1 | StepLR 的学习率衰减系数，仅 `--scheduler step` 生效 |
| `--max-batches` | 0 | 每轮最多 batch 数，仅调试用 |
| `--use-se` | 关闭 | 在 CIFARCNN 中按 `--se-stages` 选择 stage 加入 SE 通道注意力 |
| `--se-reduction` | 16 | SE 通道压缩比（仅 `--use-se` 生效，需 ≥ 1） |
| `--se-stages` | `1,2,3` | 在哪些 stage 插入 SE（1-based，逗号分隔），如 `1`、`1,2` |
| `--se-init-bias` | 0.0 | SE 最后一层 bias 初值；0 初始门控 0.5，2.0 约 0.88 |
| `--skip-test` | 关闭 | 跳过训练结束后的测试集评估，仅用于筛选/消融阶段 |

> `--output-dir` 默认仍为 `outputs`；正式实验建议显式指定 `outputs/<exp_name>`（如 `outputs/final100`），避免不同实验互相覆盖。

数据增强对比示例（Windows 建议写成单行）：

```bash
python train.py --no-augment --epochs 30 --output-dir outputs/no_aug
python train.py --epochs 30 --output-dir outputs/with_aug
```

学习率调度示例（Windows 建议写成单行）：

```bash
# 默认 cosine 调度（最终模型）
python train.py --epochs 100 --output-dir outputs/final100

# 固定学习率对照
python train.py --scheduler none --epochs 100 --output-dir outputs/no_sched100
```

SE 消融示例：

```bash
# 只在 stage1 加 SE，并使用近恒等初始化
python train.py --use-se --se-stages 1 --se-reduction 8 --se-init-bias 2.0 --output-dir outputs/se100_s1

# 补全“仅增强、无调度”条件下的 SE 对照（与 no_sched100 只差 SE）
python train.py --scheduler none --use-se --se-reduction 8 --se-stages 1,2,3 --epochs 100 --skip-test --output-dir outputs/no_sched100_se

# 筛选阶段只看验证集，避免反复使用测试集
python train.py --use-se --se-stages 1,2 --se-reduction 8 --se-init-bias 2.0 --epochs 30 --skip-test --output-dir outputs/screen/se_s12
```

- `--scheduler cosine`：余弦退火（默认，与旧实验一致）；
- `--scheduler step`：StepLR，可配合 `--scheduler-step-size`、`--scheduler-gamma`；
- `--scheduler none`：固定学习率，可用于做“有无调度”的对照。

## 8. 模型结构

`CIFARCNN`（`src/model.py`，约 **1.15 M** 可训练参数）：

| Stage | 操作 | 输出尺寸 |
|---|---|---|
| 输入 | — | 3 × 32 × 32 |
| Stage 1 | 2 × [Conv3x3(3→64) + BN + ReLU]，MaxPool，Dropout 0.1 | 64 × 16 × 16 |
| Stage 2 | 2 × [Conv3x3(64→128) + BN + ReLU]，MaxPool，Dropout 0.2 | 128 × 8 × 8 |
| Stage 3 | 2 × [Conv3x3(128→256) + BN + ReLU]，MaxPool，Dropout 0.3 | 256 × 4 × 4 |
| Head | GlobalAvgPool → Dropout 0.5 → Linear(256→10) | 10 |

可选 `--use-se` 会按 `--se-stages` 在指定 Stage 的 `ConvBlock` 后插入 SE 通道注意力模块：先全局平均池化，再用两层 1×1 卷积学习通道权重并乘回特征图。默认关闭，因此基线模型结构和已有 checkpoint 完全兼容。参数增量取决于 `--se-reduction`：`r=16` 约增加 1.1 万参数（约 1.16 M），`r=8` 约增加 2.2 万参数（约 1.17 M）。`--se-stages 1` / `1,2` 可用于结构消融，`--se-init-bias 2.0` 会让初始门控约为 0.88，接近恒等映射。

设计要点：

- 全部使用 3×3 小卷积核，每个 stage 堆叠两层，兼顾感受野与参数量；
- BatchNorm 加速收敛并提供正则效果；Dropout 随 stage 递增，分类头使用 0.5；
- 全局平均池化替代大尺寸全连接层，减少参数、抑制过拟合；
- Kaiming 初始化，SGD + Nesterov + CosineAnnealingLR 训练。

## 9. 训练配置与损失

- 损失函数：CrossEntropyLoss（可选 label smoothing）；
- 优化器：SGD（lr=0.1, momentum=0.9, Nesterov, weight_decay=5e-4）；
- 学习率调度：默认 CosineAnnealingLR，`T_max=epochs`，`eta_min=lr×0.01`（默认 lr=0.1 时为 0.001）；可通过 `--scheduler step` 使用 StepLR，或 `--scheduler none` 关闭调度；
- 梯度裁剪：max-norm = 5.0；
- 验证集用于选择最优 checkpoint，测试集只在训练结束后使用一次，避免信息泄漏。

## 10. 可复现性

- `set_seed(seed)` 固定 Python / NumPy / PyTorch 随机种子；
- 训练/验证划分使用 `numpy.random.default_rng(seed)` 分层抽样，索引保存到对应实验目录（如 `outputs/final100/split_indices.json`）；
- 对应实验目录下的 `args.json` 保存本次运行的全部超参数；
- DataLoader 使用 `worker_init_fn` 固定每个 worker 的随机状态；
- 需要更强确定性时加 `--deterministic`（会降低 cuDNN 速度）。

注意：GPU 上部分卷积/原子操作仍可能存在非确定性；即使固定种子，不同 GPU 型号或 CUDA 版本的结果也可能有小数点后的差异。报告中应记录硬件与软件版本。

## 11. 预期性能

默认配置（100 epoch、RandomCrop + Flip、SGD + Cosine）在单卡 GPU 上训练，验证/测试准确率通常在 **90%–93%** 区间。实际数值受硬件、随机种子和训练轮数影响，请以 `run_summary.json` / `test_metrics.json` 中的记录为准，不要直接引用本段估计值。

CPU 训练可以跑通，但速度明显慢于 GPU。以本机 14 核 20 线程 CPU、`batch-size=128`、`num-workers=0` 实测：短程约 190 秒/epoch，持续训练因笔记本降频约 8–9 分钟/epoch，100 epoch 预计 **8–14 小时**；不同机器差异很大。时间有限时建议减少 `--epochs`（如 20–30）得到基线，或改用 GPU。

## 12. 常见问题

**Q: 下载 CIFAR-10 超时？**  
A: 参考第 5 节手动放置数据；或配置代理后重试。程序在数据缺失时会给出明确路径提示。

**Q: Windows 下 DataLoader 报错或卡住？**  
A: 将 `--num-workers` 设为 `0`。所有多进程入口均已放在 `if __name__ == "__main__":` 下。

**Q: 显存不足（CUDA out of memory）？**  
A: 减小 `--batch-size`（如 64 或 32）；本模型仅约 1.15 M 参数，通常 4 GB 显存足够。

**Q: 想直接看测试结果，不想重新训练？**  
A: `python evaluate.py --checkpoint outputs/final100/best.pt`。

**Q: 如何用于报告中的对比实验？**  
A: 每次训练使用不同的 `--output-dir`（如 `outputs/final100`、`outputs/no_aug100`），再汇总各自的 `run_summary.json` 与 `test_metrics.json`。已完成的加分项见第 13 节。

## 13. 报告与加分项

- 最终实验报告：[`CIFAR10实验报告.docx`](CIFAR10实验报告.docx)（6 页，含最小基线、三项方法原理、七组 100 epoch 数据对比、SE 收益边界与 SE×调度交互、结论）；
- 实验报告的结构、每章要点、图表取数位置与提交检查清单见 [`REPORT_GUIDE.md`](REPORT_GUIDE.md)；
- 已完成的加分项：**数据增强、学习率调度、SE**（SE 通道注意力）。对应配置、命令和结果见第 3.1 节；SE 通过 `--use-se` / `--se-stages` / `--se-init-bias` 控制，并提供 `scripts/analyze_se.py`、`scripts/compare_runs.py` 与 `scripts/analyze_interaction.py`。
