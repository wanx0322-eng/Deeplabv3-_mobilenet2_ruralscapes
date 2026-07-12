# DeeplabV3+ MobileNetV2 乡村场景语义分割系统

基于 PyTorch 实现的 DeepLabV3+ 语义分割项目，主干网络采用轻量级的 MobileNetV2（亦支持 Xception），针对**无人机航拍乡村场景**（建筑 / 天空 / 树木 / 道路）进行像素级分割。仓库内已包含训练好的权重、示例数据集与可视化结果，开箱即用。

---

## 目录

- [项目简介](#项目简介)
- [核心特性](#核心特性)
- [环境依赖](#环境依赖)
- [目录结构](#目录结构)
- [数据集说明](#数据集说明)
- [模型架构](#模型架构)
- [快速开始](#快速开始)
- [训练自己的数据集](#训练自己的数据集)
- [预测与可视化](#预测与可视化)
- [模型评估（mIoU）](#模型评估miou)
- [关键参数说明](#关键参数说明)
- [训练结果](#训练结果)
- [常见问题](#常见问题)
- [致谢](#致谢)

---

## 项目简介

本项目实现了完整的语义分割流水线：数据处理 → 模型构建 → 训练 → 评估 → 推理 → 部署（ONNX）。模型选用 **DeepLabV3+** 架构，结合 **MobileNetV2** 轻量级主干，在保证精度的同时显著降低参数量与计算量，适合在算力受限的边缘设备上部署。

**应用场景**：无人机航拍的乡村/田园场景语义分割，可用于：

- 村庄建筑识别与统计
- 道路网络提取
- 植被覆盖分析
- 土地利用分类

---

## 核心特性

- **双主干支持**：MobileNetV2（轻量，默认）/ Xception（高精度）
- **完整训练流水线**：冻结-解冻两阶段训练、学习率余弦退火、混合精度（FP16）、多卡 DP/DDP
- **丰富数据增强**：随机缩放/扭曲、左右翻转、高斯模糊、随机旋转、HSV 色域变换
- **多损失函数**：Cross Entropy（默认）/ Focal Loss / Dice Loss，可任意组合
- **多模式推理**：单张图片、视频/摄像头、批量文件夹、FPS 测试、ONNX 导出
- **完整评估体系**：mIoU、mPA、Precision、Recall、混淆矩阵、F1-Score，自动绘制曲线
- **TensorBoard 可视化**：训练过程 loss 与 mIoU 曲线实时记录
- **断点续训**：支持加载部分权重，按 key 名匹配，自动跳过不匹配层

---

## 环境依赖

| 依赖            | 建议版本        | 说明                       |
| --------------- | --------------- | -------------------------- |
| Python          | 3.8+            | 推荐 3.8 / 3.9 / 3.10      |
| PyTorch         | 1.7.1+          | FP16 需要 1.7.1 以上       |
| torchvision     | 0.8+            | 与 PyTorch 版本匹配        |
| numpy           | ≥ 1.19          | 数值计算                   |
| opencv-python   | ≥ 4.5           | 图像处理                   |
| Pillow          | ≥ 8.0           | 图像读写                   |
| matplotlib      | ≥ 3.3           | 训练曲线绘制               |
| scipy           | ≥ 1.5           | 平滑曲线                   |
| tqdm            | ≥ 4.50          | 进度条                     |
| thop            | ≥ 0.0.5         | FLOPs/参数量统计（summary）|
| torchsummary    | ≥ 1.5           | 网络结构打印（summary）    |
| tensorboard     | ≥ 2.4           | 训练日志可视化             |
| onnx / onnxsim  | 可选            | ONNX 导出                  |

一键安装：

```bash
pip install torch torchvision numpy opencv-python pillow matplotlib scipy tqdm thop torchsummary tensorboard
```

---

## 目录结构

```
Deeplabv3-_mobilenet2_ruralscapes-master/
├── deeplab.py                  # 推理引擎：DeeplabV3 类（加载模型、detect_image、FPS、ONNX 导出）
├── train.py                    # 训练入口：参数配置、数据加载、训练循环
├── predict.py                  # 预测入口：5 种模式（predict/video/fps/dir_predict/export_onnx）
├── get_miou.py                 # mIoU 评估入口：生成预测结果 + 计算 mIoU
├── voc_annotation.py           # 数据集划分（train/val/test.txt）+ 标签格式自检
├── summary.py                  # 打印网络结构、统计 FLOPs 与参数量
├── process_dataset.ipynb       # 数据预处理：从 labelme 标注 (img.png/label.png) 转 VOC 格式
│
├── nets/                       # 网络定义
│   ├── deeplabv3_plus.py       # DeepLab 主模型 + ASPP 模块 + 解码器
│   ├── mobilenetv2.py          # MobileNetV2 主干（含 ImageNet 预训练权重下载）
│   ├── xception.py             # Xception 主干（含 ImageNet 预训练权重下载）
│   └── deeplabv3_training.py   # 损失函数（CE/Focal/Dice）+ 学习率调度 + 权重初始化
│
├── utils/                      # 工具函数
│   ├── dataloader.py           # DeeplabDataset + 数据增强 + collate_fn
│   ├── utils.py                # 通用工具：cvtColor/resize/seed/preprocess_input/download_weights
│   ├── utils_fit.py            # 单 epoch 训练逻辑（前向/反向/验证/保存）
│   ├── utils_metrics.py        # f_score / fast_hist / compute_mIoU / show_results
│   └── callbacks.py            # LossHistory（TensorBoard+曲线图）+ EvalCallback（验证集 mIoU）
│
├── model_data/                 # 预训练权重
│   └── deeplab_mobilenetv2.pth # 整个模型的初始权重（基于 ImageNet 主干迁移）
│
├── logs/                       # 训练输出目录
│   ├── best_epoch_weights.pth  # 验证集 loss 最优权重
│   ├── last_epoch_weights.pth  # 最新 epoch 权重
│   ├── ep0XX-lossX.XXX-val_lossX.XXX.pth   # 每 save_period 个 epoch 保存一次
│   └── loss_YYYY_MM_DD_HH_MM_SS/           # 每次训练的 loss/miou 曲线与 txt 日志
│
├── VOCdevkit/                  # VOC 格式数据集
│   └── VOC2007/
│       ├── JPEGImages/         # 输入图片（.png，313 张）
│       ├── SegmentationClass/  # 分割标签（.png，像素值=类别索引）
│       └── ImageSets/Segmentation/
│           ├── train.txt       # 训练集图片名（115 张）
│           ├── val.txt         # 验证集图片名（50 张）
│           ├── trainval.txt    # 训练+验证（165 张）
│           └── test.txt        # 测试集（空，本库将 val 当作 test 使用）
│
├── img/                        # 预测输入图片文件夹
├── img_out/                    # 预测输出文件夹（dir_predict 模式）
├── miou_out/                   # mIoU 评估输出（含 mIoU/mPA/Precision/Recall 图与混淆矩阵）
└── .idea/                      # PyCharm 工程配置（可忽略）
```

---

## 数据集说明

### 数据格式（PASC VOC）

本仓库要求 **VOC 格式**数据集，组织如下：

```
VOCdevkit/VOC2007/
├── JPEGImages/         # 输入图片（jpg 或 png，无需固定尺寸）
│   ├── 0.png
│   ├── 1.png
│   └── ...
├── SegmentationClass/  # 标签图片（png，灰度或 8 位彩图）
│   ├── 0.png           # 像素值 = 类别索引（0,1,2,...）
│   └── ...
└── ImageSets/Segmentation/
    ├── train.txt       # 每行一个图片名（不含扩展名）
    └── val.txt
```

### 类别定义

本项目针对乡村航拍场景，共定义 **5 个类别**（含背景）：

| 索引 | 类名          | 中文   | 颜色 (R,G,B)   |
| ---- | ------------- | ------ | -------------- |
| 0    | `_background_`| 背景   | (0, 0, 0)      |
| 1    | `building`    | 建筑   | (128, 0, 0)    |
| 2    | `sky`         | 天空   | (0, 128, 0)    |
| 3    | `tree`        | 树木   | (128, 128, 0)  |
| 4    | `way`         | 道路   | (0, 0, 128)    |

> ⚠️ **标签格式注意**：标签图片每个像素的值必须是该像素所属类别索引（0、1、2、3、4）。网络上下载的标签常使用 0/255 二值化格式，**会导致训练无效果**。若标签格式有误，可参考 [segmentation-format-fix](https://github.com/bubbliiiing/segmentation-format-fix) 工具修正。

### 数据来源与预处理

原始数据来自 labelme 标注，结构为 `pictures/json/train_X_json/` 下含 `img.png` 与 `label.png`。运行 `process_dataset.ipynb` 可批量重命名并整理为 VOC 格式：

```python
# 节选自 process_dataset.ipynb
img_save_path  = r"Data\images"
label_save_path = r"Data\labels"
for i in range(len(img_list)):
    Image.open(img_list[i]).save(f"{img_save_path}\\{i}.png")
    Image.open(label_list[i]).save(f"{label_save_path}\\{i}.png")
```

随后运行 `voc_annotation.py` 自动生成 train/val 划分并校验标签格式。

---

## 模型架构

DeepLabV3+ 是 Google 提出的经典语义分割网络，由 **主干特征提取 + ASPP 加强特征提取 + 解码器特征融合** 三部分组成。本项目实现的整体前向流程如下：

```
输入图像 (B, 3, 256, 256)
        │
        ▼
┌────────────────────────────────────────────┐
│  主干网络 (MobileNetV2 / Xception)         │
│  ├─ 浅层特征 low_level_features            │  ← 用于解码器细节恢复
│  │    MobileNetV2: (B, 24, 64, 64)         │
│  │    Xception:     (B, 256, 128, 128)     │
│  └─ 深层特征 x                             │  ← 送入 ASPP
│       MobileNetV2: (B, 320, 32, 32)        │
│       Xception:     (B, 2048, 30, 30)      │
└────────────────────────────────────────────┘
        │                              │
        │                              ▼
        │              ┌───────────────────────────────────┐
        │              │  ASPP 模块 (5 个并行分支)         │
        │              │  ├─ 1×1 卷积                      │
        │              │  ├─ 3×3 膨胀卷积 (rate=6)         │
        │              │  ├─ 3×3 膨胀卷积 (rate=12)        │
        │              │  ├─ 3×3 膨胀卷积 (rate=18)        │
        │              │  └─ 全局平均池化 + 1×1 卷积       │
        │              │  → 拼接 + 1×1 卷积整合            │
        │              │  输出: (B, 256, H, W)             │
        │              └───────────────────────────────────┘
        │                              │
        ▼                              ▼
┌──────────────────┐         ┌──────────────────────┐
│  shortcut_conv   │         │  双线性上采样         │
│  1×1 卷积降维    │         │  对齐到浅层特征尺寸   │
│  → (B, 48, H, W) │         └──────────────────────┘
└──────────────────┘                  │
        │                             │
        └──────────┬──────────────────┘
                   ▼
        ┌─────────────────────────┐
        │  cat_conv (3×3 卷积×2)  │  ← 拼接浅层与深层特征
        │  Dropout 防过拟合        │
        │  输出: (B, 256, H, W)   │
        └─────────────────────────┘
                   │
                   ▼
        ┌─────────────────────────┐
        │  cls_conv (1×1 卷积)    │  ← 分类头
        │  输出: (B, num_classes, │
        │         H, W)           │
        └─────────────────────────┘
                   │
                   ▼
        ┌─────────────────────────┐
        │  双线性上采样到原图尺寸  │
        │  输出: (B, num_classes, │
        │         256, 256)        │
        └─────────────────────────┘
                   │
                   ▼
            像素级分类结果
```

### 关键模块说明

**1. MobileNetV2 主干**（`nets/mobilenetv2.py`）

- 采用 **倒残差结构 (Inverted Residual)** + **深度可分离卷积**，参数量约 2.2M
- 通过 `_nostride_dilate` 将下采样层的 stride 改为 1 并引入膨胀卷积，在保持感受野的同时提升特征图分辨率
- 支持 `downsample_factor=8`（更高精度，更耗显存）或 `16`（更快）

**2. ASPP 模块**（`nets/deeplabv3_plus.py`）

- 5 路并行：1×1 卷积 + 3 个不同膨胀率（6/12/18）的 3×3 膨胀卷积 + 全局平均池化分支
- 有效捕获多尺度上下文信息，解决分割中的尺度变化问题

**3. 解码器**（`nets/deeplabv3_plus.py`）

- 将 ASPP 输出上采样后与主干浅层特征拼接
- 浅层特征经 1×1 卷积降维至 48 通道（减少计算量）
- 两层 3×3 卷积融合特征后输出分类结果

---

## 快速开始

### 1. 环境准备

```bash
git clone <repo_url>
cd Deeplabv3-_mobilenet2_ruralscapes-master
pip install -r requirements.txt   # 若无 requirements.txt，参考"环境依赖"手动安装
```

### 2. 使用预训练权重进行预测

仓库已附带训练好的权重 `logs/best_epoch_weights.pth`，可直接推理：

```bash
python predict.py
```

默认 `mode = "dir_predict"`，会遍历 `img/` 文件夹下所有图片，分割结果保存到 `img_out/`。

### 3. 单张图片预测

修改 `predict.py`：

```python
mode = "predict"
name_classes = ["_background_", "building", "sky", "tree", "way"]
```

运行后输入图片路径即可查看分割结果。

### 4. 查看网络结构

```bash
python summary.py
```

输出模型的层结构、总 FLOPs 与参数量。

---

## 训练自己的数据集

### 步骤 1：准备数据集

按 VOC 格式组织数据（见 [数据集说明](#数据集说明)），确保：

- `JPEGImages/` 中是输入图片（jpg/png）
- `SegmentationClass/` 中是对应标签（png，像素值=类别索引）
- 标签必须为**灰度图或 8 位彩图**，每个像素的值代表类别（0,1,2,...）

### 步骤 2：划分训练集/验证集

```bash
python voc_annotation.py
```

该脚本会：

1. 按 `trainval_percent`（默认 1.0）和 `train_percent`（默认 0.7）随机划分
2. 生成 `train.txt` / `val.txt` / `trainval.txt` / `test.txt`
3. 自动检查标签格式，统计各类别像素数量，识别 0/255 二值化错误

### 步骤 3：修改训练参数

编辑 `train.py`，**必须修改**以下参数：

```python
num_classes     = 5        # 你的类别数 + 1（含背景）
backbone        = "mobilenet"   # 或 "xception"
model_path      = "model_data/deeplab_mobilenetv2.pth"  # 预训练权重路径
input_shape     = [256, 256]    # 输入尺寸
VOCdevkit_path  = 'VOCdevkit'   # 数据集路径
```

其他可选参数：

```python
Cuda            = True       # 是否使用 GPU
distributed     = False      # 是否使用 DDP 多卡
fp16            = False      # 是否混合精度训练
Freeze_Train    = True       # 是否冻结主干先训练
Init_Epoch      = 0
Freeze_Epoch    = 50         # 冻结阶段 epoch 数
UnFreeze_Epoch  = 80         # 总训练 epoch 数
Freeze_batch_size   = 4
Unfreeze_batch_size = 4
optimizer_type  = "sgd"      # 或 "adam"
Init_lr         = 7e-3       # SGD 推荐 7e-3，Adam 推荐 5e-4
lr_decay_type   = 'cos'      # 或 'step'
dice_loss       = False      # 是否使用 Dice Loss
focal_loss      = False      # 是否使用 Focal Loss
```

### 步骤 4：开始训练

```bash
python train.py
```

**多卡训练**（DDP 模式，仅 Linux）：

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
```

**多卡训练**（DP 模式，Windows/Linux 通用）：

```bash
# 在 train.py 中设置 distributed = False
CUDA_VISIBLE_DEVICES=0,1 python train.py
```

### 步骤 5：训练过程监控

训练过程中会自动生成：

- `logs/loss_YYYY_MM_DD_HH_MM_SS/epoch_loss.txt`：每个 epoch 的训练 loss
- `logs/loss_YYYY_MM_DD_HH_MM_SS/epoch_val_loss.txt`：验证 loss
- `logs/loss_YYYY_MM_DD_HH_MM_SS/epoch_miou.txt`：验证集 mIoU
- `logs/loss_YYYY_MM_DD_HH_MM_SS/epoch_loss.png`：loss 曲线图
- `logs/loss_YYYY_MM_DD_HH_MM_SS/epoch_miou.png`：mIoU 曲线图

使用 TensorBoard 实时查看：

```bash
tensorboard --logdir logs/
```

### 训练策略说明

本项目采用**冻结-解冻两阶段训练**：

| 阶段     | Epoch 范围          | 主干状态 | batch_size        | 显存占用 | 说明                       |
| -------- | ------------------- | -------- | ----------------- | -------- | -------------------------- |
| 冻结阶段 | 0 → Freeze_Epoch    | 冻结     | Freeze_batch_size | 较小     | 仅训练解码器，快速收敛     |
| 解冻阶段 | Freeze_Epoch → 终点 | 解冻     | Unfreeze_batch_size | 较大   | 全网络微调，提升精度       |

**学习率自适应**：根据 batch_size 按 `nbs=16` 基准自动缩放，避免显存不足时调小 batch_size 导致学习率过大。

---

## 预测与可视化

`predict.py` 支持 5 种模式，通过 `mode` 变量切换：

### 模式 1：单张图片预测 (`predict`)

```python
mode = "predict"
count = True                      # 是否统计各类别像素占比
name_classes = ["_background_", "building", "sky", "tree", "way"]
```

运行后输入图片路径，弹出分割结果窗口。`count=True` 时会打印各类别像素数与占比表格。

### 模式 2：视频/摄像头检测 (`video`)

```python
mode = "video"
video_path = 0                    # 0 表示摄像头；或 "xxx.mp4"
video_save_path = "result.mp4"    # 留空则不保存
video_fps = 25.0
```

### 模式 3：FPS 测试 (`fps`)

```python
mode = "fps"
test_interval = 100               # 测试次数，越大越准
fps_image_path = "img/street.jpg"
```

### 模式 4：批量文件夹预测 (`dir_predict`)

```python
mode = "dir_predict"
dir_origin_path = "img/"          # 输入文件夹
dir_save_path   = "img_out/"      # 输出文件夹
```

### 模式 5：导出 ONNX (`export_onnx`)

```python
mode = "export_onnx"
simplify = True                   # 是否使用 onnx-simplifier 简化
onnx_save_path = "model_data/models.onnx"
```

### 可视化效果（mix_type）

在 `deeplab.py` 的 `_defaults` 中设置 `mix_type`：

| mix_type | 效果                                   |
| -------- | -------------------------------------- |
| 0        | 原图与分割图混合（默认，0.7 比例混合） |
| 1        | 仅输出彩色分割图                       |
| 2        | 仅保留目标，扣除背景                   |

---

## 模型评估（mIoU）

### 评估流程

```bash
python get_miou.py
```

`miou_mode` 控制评估内容：

| miou_mode | 行为                                   |
| --------- | -------------------------------------- |
| 0         | 完整流程：生成预测结果 + 计算 mIoU     |
| 1         | 仅生成预测结果（保存到 miou_out/）     |
| 2         | 仅计算 mIoU（需已生成预测结果）        |

### 评估指标

`utils/utils_metrics.py` 实现以下指标：

- **mIoU**（平均交并比）：各类别 IoU 的均值，语义分割核心指标
- **mPA**（平均像素准确率）：各类别像素准确率的均值
- **Precision**（精确率）：每类 TP / (TP + FP)
- **Recall**（召回率）：每类 TP / (TP + FN)
- **F-Score**（F1 分数）：训练过程中实时计算
- **混淆矩阵**：导出为 CSV，可视化分类错误分布

### 评估输出

运行后 `miou_out/` 目录包含：

```
miou_out/
├── detection-results/      # 验证集每张图的预测结果
├── mIoU.png                # 各类别 IoU 柱状图
├── mPA.png                 # 各类别 PA 柱状图
├── Precision.png           # 各类别 Precision 柱状图
├── Recall.png              # 各类别 Recall 柱状图
└── confusion_matrix.csv    # 混淆矩阵
```

### 训练时评估

`train.py` 中设置 `eval_flag = True` 会在每 `eval_period` 个 epoch 后自动评估验证集 mIoU，并保存到 `logs/.../epoch_miou.txt` 与曲线图。

---

## 关键参数说明

### `deeplab.py` 中的预测参数

| 参数                | 默认值                          | 说明                                      |
| ------------------- | ------------------------------- | ----------------------------------------- |
| `model_path`        | `logs/best_epoch_weights.pth`   | 权重文件路径                              |
| `num_classes`       | 5                               | 类别数（含背景）                          |
| `backbone`          | `"mobilenet"`                   | 主干网络：`mobilenet` / `xception`        |
| `input_shape`       | `[256, 256]`                    | 输入图片尺寸                              |
| `downsample_factor` | 8                               | 下采样倍数：8（精度高）/ 16（速度快）     |
| `mix_type`          | 0                               | 可视化方式：0 混合 / 1 仅分割 / 2 扣背景  |
| `cuda`              | True                            | 是否使用 GPU                              |

### `train.py` 中的训练参数

| 参数                  | 默认值                        | 说明                                  |
| --------------------- | ----------------------------- | ------------------------------------- |
| `num_classes`         | 5                             | 类别数（含背景）                      |
| `backbone`            | `"mobilenet"`                 | 主干网络                              |
| `pretrained`          | False                         | 是否加载主干 ImageNet 预训练权重      |
| `model_path`          | `model_data/deeplab_mobilenetv2.pth` | 整个模型的初始权重              |
| `downsample_factor`   | 8                             | 下采样倍数                            |
| `input_shape`         | `[256, 256]`                  | 输入尺寸                              |
| `Init_Epoch`          | 0                             | 起始 epoch（断点续训用）              |
| `Freeze_Epoch`        | 50                            | 冻结阶段结束 epoch                    |
| `UnFreeze_Epoch`      | 80                            | 总训练 epoch                          |
| `Freeze_batch_size`   | 4                             | 冻结阶段 batch_size                   |
| `Unfreeze_batch_size` | 4                             | 解冻阶段 batch_size                   |
| `Freeze_Train`        | True                          | 是否启用冻结训练                      |
| `Init_lr`             | 7e-3                          | 初始学习率（SGD: 7e-3, Adam: 5e-4）   |
| `Min_lr`              | Init_lr × 0.01                | 最小学习率                            |
| `optimizer_type`      | `"sgd"`                       | 优化器：`sgd` / `adam`                |
| `lr_decay_type`       | `'cos'`                       | 学习率衰减：`cos` / `step`            |
| `save_period`         | 5                             | 每 N 个 epoch 保存一次权重            |
| `eval_flag`           | True                          | 训练时是否评估 mIoU                    |
| `eval_period`         | 5                             | 每 N 个 epoch 评估一次                |
| `dice_loss`           | False                         | 是否使用 Dice Loss                     |
| `focal_loss`          | False                         | 是否使用 Focal Loss                    |
| `cls_weights`         | `np.ones([num_classes])`      | 各类别损失权重（处理类别不平衡）      |
| `fp16`                | False                         | 混合精度训练                           |
| `num_workers`         | 0                             | 数据加载线程数（0 关闭多线程）        |

---

## 训练结果

基于本项目数据集（313 张乡村航拍图，5 类）的训练结果：

- **主干网络**：MobileNetV2
- **输入尺寸**：256 × 256
- **下采样倍数**：8
- **优化器**：SGD（Init_lr=7e-3, momentum=0.9, weight_decay=1e-4）
- **学习率衰减**：余弦退火
- **训练 Epoch**：80（冻结 50 + 解冻 30）
- **Batch Size**：4

**关键指标**（来自 `logs/loss_2024_03_24_15_18_29/`）：

| 指标                | 数值       |
| ------------------- | ---------- |
| 最终训练 Loss       | ≈ 0.38     |
| 最终验证 Loss       | ≈ 0.62     |
| 最佳验证 mIoU       | ≈ 57.4%    |
| 模型权重大小        | ≈ 23 MB    |

> mIoU 在 epoch 16 左右达到峰值约 57.4%，随后出现轻微过拟合趋势。可通过增加数据增强、调整 `dice_loss`/`focal_loss`、扩充数据集等方式进一步提升。

---

## 常见问题

### Q1：训练时提示 "shape 不匹配"

确保训练时的 `model_path`、`backbone`、`num_classes`、`downsample_factor` 与预测时一致。切换 backbone 必须重新训练或使用对应权重。

### Q2：显存不足（OOM / CUDA out of memory）

- 调小 `Freeze_batch_size` 和 `Unfreeze_batch_size`（最小为 2，受 BatchNorm 限制）
- 降低 `input_shape`（如 256 → 128）
- 设置 `downsample_factor = 16`（特征图更小）
- 开启 `fp16 = True` 混合精度训练

### Q3：训练 loss 不下降 / mIoU 很低

- 检查标签格式：运行 `voc_annotation.py` 自带的格式检查
- 确认标签像素值为类别索引（0,1,2,...），而非 0/255
- 确认 `num_classes` 设置正确（类别数 + 1）
- 尝试加载预训练权重（`pretrained = True` 或指定 `model_path`）

### Q4：预测结果全黑

- 检查 `model_path` 是否指向正确的权重文件
- 确认 `backbone` 与训练时一致
- 确认 `num_classes` 与训练时一致

### Q5：如何从 0 开始训练（无预训练权重）

```python
model_path = ""
pretrained = False
Freeze_Train = False
```

> ⚠️ 不推荐从 0 训练，主干权值过于随机会导致特征提取效果差。建议至少加载主干 ImageNet 预训练权重（`pretrained = True`）。

### Q6：如何断点续训

将 `model_path` 指向 `logs/` 下已训练的权重文件，设置 `Init_Epoch` 为已训练的 epoch 数，调整 `Freeze_Epoch` 和 `UnFreeze_Epoch` 保证 epoch 连续性。

### Q7：Windows 下多卡训练

Windows 不支持 DDP，使用 DP 模式：

```python
# train.py 中
distributed = False
Cuda = True
```

```bash
set CUDA_VISIBLE_DEVICES=0,1
python train.py
```

### Q8：如何更换为主干 Xception

```python
# train.py 与 deeplab.py 中
backbone = "xception"
model_path = "model_data/deeplab_xception.pth"   # 需自行训练或下载
```

Xception 参数量更大（约 40M+），精度通常更高但速度更慢。

---

## 致谢

本项目基于以下开源工作：

- **DeepLabV3+** 原论文：[Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611) (Chen et al., ECCV 2018)
- **MobileNetV2** 原论文：[MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381) (Sandler et al., CVPR 2018)
- **Xception** 原论文：[Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357) (Chollet, CVPR 2017)
- 代码框架参考：[bubbliiiing/deeplabv3-plus-pytorch](https://github.com/bubbliiiing/deeplabv3-plus-pytorch)

预训练主干权重下载地址：

- MobileNetV2 (ImageNet)：`https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/mobilenet_v2.pth.tar`
- Xception (ImageNet)：`https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/xception_pytorch_imagenet.pth`

---

## License

本项目仅供学习与研究使用。商业使用请联系原作者并遵守相关开源协议。
