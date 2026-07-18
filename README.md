# 乡村街景语义分割系统（DeepLabV3+ / SegFormer）

对乡村街景影像做 5 类语义分割：`_background_ / building / sky / tree / way`。
包含图形工作站（数据管理、标注、训练、预测、评估）与命令行工具两套入口。

## 模型库

验证集 50 张固定不变，指标口径统一：**完整混淆矩阵计算逐类 IoU，背景展示但不计入平均**。

| 权重 | 架构 | 前景 mIoU | 速度 | 说明 |
|---|---|---|---|---|
| `logs_segformer_b2/best_segformer.pth` | SegFormer-B2 | **73.45** | 15.2 ms | **系统默认**（精度档） |
| `logs_segformer_b0/best_segformer.pth` | SegFormer-B0 | 72.15 | 9.6 ms | 速度档（3.7M 参数） |
| `logs_v2_E/best_epoch_weights.pth` | DeepLabV3+ MobileNetV2 + EMA | 69.09（+TTA 70.07） | 22 ms | way 类最优的 DeepLab 权重 |
| `logs_v2_B/best_epoch_weights.pth` | DeepLabV3+ MobileNetV2 | 69.11 | 22 ms | DeepLab 路线最优 |
| `logs/best_epoch_weights.pth` | DeepLabV3+ MobileNetV2 | 62.52 | 22 ms | 历史基线（仅存档对照） |
| `model_data/deeplab_mobilenetv2.pth` | 主干预训练 | — | — | DeepLab 重训练的初始权值 |

SegFormer 权重需选择对应的 `segformer-*` 主干加载；预测页的 **TTA（高质量模式）** 对
DeepLab 约 +0.7 mIoU，对 SegFormer-B2 无收益（可不开）。

## 快速开始

```bash
pip install -r requirements.txt        # PyTorch 按自己的 CUDA 版本安装，见文件内注释

python check_env.py                    # 环境自检：依赖/CUDA/配置/数据集/权值/冒烟测试
python -m pytest tests -q              # 测试套件

# 图形工作站（推荐入口，功能见 工作站使用说明.md）
python run_workstation.py              # 或双击 启动工作站.bat
python run_studio.py                   # QML 外壳（同一套后端，标注页仍在 Widgets 工作站）

# 命令行（DeepLab 路线）
python predict.py                      # mode 见文件头注释：单张/批量/视频/fps/onnx
python get_miou.py                     # 验证集 mIoU 评估
python train.py                        # DeepLab 训练
python voc_annotation.py               # 重新生成数据划分（自动跳过非索引图标签）
```

## tools/ 实用脚本

| 脚本 | 用途 |
|---|---|
| `train_segformer.py` | 训练 SegFormer（`--model` 选 b0/b1/b2；支持两阶段，见下） |
| `compare_models.py` | 多个 DeepLab 权重同口径对比，`--tta` 开测试时增强 |
| `import_external.py` | 把 RuralScapes / UAVid 等外部数据集转成本项目 VOC 格式 |
| `external_datasets.py` | 外部数据集 → 本项目类别的映射表（可直接运行查看） |
| `fix_rgb_masks.py` | 把 RGB 三通道标签转成索引图（原图备份到 `VOCdevkit/_mask_backup_rgb/`） |
| `normalize_ext.py` | 统一图片扩展名为小写（跨平台兼容） |
| `rebuild_splits.py` | 重建 train/val 划分：验证集不动、剔除与验证集像素重复的泄漏图 |
| `count_classes.py` | 统计各类像素占比，给出三档可直接填进配置的 `cls_weights` |

### 两阶段训练（领域内预训练）

> **已验证结论：用 UAVid 做领域预训练无效，不要重复这个实验。**
> 实测两阶段 73.89 vs 单阶段基线 73.45，配对 bootstrap 95% 区间 [−0.33, +1.47]
> **跨过 0，不显著**。原因：UAVid 是 50m 高空斜下视角俯瞰城市街区，与本项目的
> 地面视角村落场景特征差异过大；且 UAVid 无 sky 类（转换后占比 0.00%）；
> 而 SegFormer 的 ADE20K 初始权重本就覆盖大量地面建筑/植被/道路场景，
> 相关性高于 UAVid。唯一正向迹象是 way 类 +2.07，但不足以支撑整体显著性。
>
> RuralScapes 条件更好（含 sky、纯乡村、5 万帧），但同属航拍视角，
> 主要障碍依旧存在，预期收益应保守估计。下方流程可直接复用于验证。

流程如下（以 RuralScapes 为例）：

```bash
# 0) RuralScapes 的掩码配色未公开，先扫描实际颜色并填写模板
python tools/import_external.py --discover /data/ruralscapes/labels
#    编辑生成的 palette_template.json，把颜色填上源类别名
#    （forest/land/hill/sky/residential/road/water/person/church/haystack/fence/car）

# 1) 转换成 VOC 格式（UAVid 配色已内置，可省略 --palette）
python tools/import_external.py \
    --images /data/ruralscapes/frames --labels /data/ruralscapes/labels \
    --dataset ruralscapes --scheme base5 --palette palette_template.json \
    --out VOCdevkit_ruralscapes

# 2) 阶段一：外部数据预训练
python tools/train_segformer.py --voc-root VOCdevkit_ruralscapes \
    --num-classes 5 --save-dir logs_sf_stage1 --epochs 60

# 3) 阶段二：本项目数据微调（学习率调小）
python tools/train_segformer.py --init-from logs_sf_stage1/best_segformer.pth \
    --save-dir logs_sf_stage2 --epochs 200 --lr 3e-5
```

`--scheme` 可选 `base5`（当前 5 类，与现有权重兼容）或 `ext7`（把背景拆出
`farmland` / `water`，针对背景类 IoU 仅 43 的问题）。用 `ext7` 时两个阶段都要加
`--num-classes 7`，且本项目标注需相应扩充。

## 数据集

`VOCdevkit/VOC2007/`：`JPEGImages/`（256×256 png 原图 313 张）、
`SegmentationClass/`（P 模式索引 png，像素值=类别下标）、
`ImageSets/Segmentation/`（当前划分：train 245 / val 50）。
`VOCdevkit/_mask_backup_rgb/` 是 148 张原始 RGB 标签的备份，勿删。

### 外部数据（体积大，不要提交到版本库）

| 目录 | 内容 | 来源与许可 |
|---|---|---|
| `data_external/uavid/` | UAVid 原始下载，270 对 4K 图/标签，3.9 GB | HuggingFace `dronefreak/UAVid-2020` 镜像，**CC BY-NC-SA 4.0（非商业）** |
| `VOCdevkit_uavid/` | 转换后的 VOC 格式（512px，train 230 / val 40） | 同上 |

保留这两份仅为可复现性（上面的两阶段实验）。若不打算复现，删除可释放约 4 GB。

**许可提醒**：UAVid 为 CC BY-NC-SA 4.0，禁止商业用途，且演绎作品（含用它预训练
的模型权重）分发时需沿用同一许可。商业项目请勿使用该路线。

UAVid 官方站点（uavid.nl）的 EOStore 下载链路证书已于 2024-08 过期、百度网盘需账号，
故改用 HuggingFace 镜像。注意该镜像的标签是**类别下标编码**（`(1,1,1)`）而非官方
彩色编码（`(128,0,0)`），`import_external.py` 会自动识别，无需手动指定。

标注新图请用工作站「图像标注」页：**AI 预标注（SegFormer-B2）+ SAM2 边界精修** 打底，
人工只做修正；标签保存即为可直接训练的索引 PNG。

## 注意事项

- SegFormer / SAM2 首次加载需联网从 HuggingFace 下载配置与预训练文件，之后走本地缓存。
- 评估时若自定义 `remove_classes`，它只影响"哪些类参与平均"，混淆矩阵永远是完整的
  ——不要从矩阵里删行列，那会把背景相关的错误一并丢掉，虚高前景指标（历史教训：+15.7）。
- 训练最佳权重按验证集 mIoU 挑选（不是 val_loss）。
