# DeepLabV3+ MobileNetV2 Rural-Scene Semantic Segmentation

This repository contains a PyTorch implementation of DeepLabV3+ for pixel-level semantic segmentation of **rural scenes captured by drones**. The default lightweight backbone is MobileNetV2; Xception is also supported. The included trained weights, sample dataset, and visualized results make the project ready to run.

## Contents

- [Features](#features)
- [Requirements](#requirements)
- [Repository layout](#repository-layout)
- [Dataset](#dataset)
- [Model architecture](#model-architecture)
- [Quick start](#quick-start)
- [Training on a custom dataset](#training-on-a-custom-dataset)
- [Inference and visualization](#inference-and-visualization)
- [Evaluation](#evaluation)
- [Key configuration](#key-configuration)
- [Results](#results)
- [FAQ](#faq)
- [Acknowledgements](#acknowledgements)

## Features

- **Two backbones:** MobileNetV2 (lightweight and default) and Xception (higher accuracy).
- **Complete training workflow:** freeze/unfreeze training, cosine learning-rate decay, mixed precision (FP16), and multi-GPU DP/DDP support.
- **Data augmentation:** random scaling/distortion, horizontal flips, Gaussian blur, random rotation, and HSV color augmentation.
- **Loss functions:** Cross Entropy, Focal Loss, and Dice Loss, which can be combined.
- **Inference modes:** single image, video/webcam, batch folder, FPS benchmark, and ONNX export.
- **Evaluation metrics:** mIoU, mPA, Precision, Recall, confusion matrix, and F1 score.
- **TensorBoard logging** for training loss and mIoU curves.
- **Resumable training:** compatible weights are loaded by key and incompatible layers are skipped automatically.

## Requirements

| Dependency | Recommended version | Notes |
| --- | --- | --- |
| Python | 3.8+ | Python 3.8, 3.9, or 3.10 is recommended |
| PyTorch | 1.7.1+ | Version 1.7.1+ is required for FP16 |
| torchvision | 0.8+ | Match the installed PyTorch version |
| numpy | 1.19+ | Numerical computing |
| opencv-python | 4.5+ | Image processing |
| Pillow | 8.0+ | Image I/O |
| matplotlib | 3.3+ | Training curves |
| scipy | 1.5+ | Curve smoothing |
| tqdm | 4.50+ | Progress bars |
| thop / torchsummary | optional | FLOPs, parameter counts, and model summaries |
| tensorboard | 2.4+ | Training-log visualization |
| onnx / onnxsim | optional | ONNX export |

Install the common dependencies with:

```bash
pip install torch torchvision numpy opencv-python pillow matplotlib scipy tqdm thop torchsummary tensorboard
```

## Repository Layout

```text
Deeplabv3-_mobilenet2_ruralscapes-master/
├── deeplab.py                  # Inference engine: model loading, image detection, FPS, ONNX export
├── train.py                    # Training entry point and configuration
├── predict.py                  # Prediction entry point: image/video/FPS/folder/ONNX modes
├── get_miou.py                 # Prediction generation and mIoU evaluation
├── voc_annotation.py           # Dataset split generation and label-format validation
├── summary.py                  # Network summary, FLOPs, and parameter count
├── process_dataset.ipynb       # LabelMe-to-VOC preprocessing notebook
├── nets/                       # Network definitions and training utilities
├── utils/                      # Dataset, metrics, callbacks, and general helpers
├── model_data/                 # Pretrained weights
├── logs/                       # Training outputs and checkpoints
├── VOCdevkit/VOC2007/          # PASCAL VOC-style dataset
├── img/                        # Input images for prediction
├── img_out/                    # Batch prediction outputs
├── miou_out/                   # Evaluation outputs
└── .idea/                      # PyCharm configuration (optional)
```

## Dataset

The project expects a dataset in **PASCAL VOC format**:

```text
VOCdevkit/VOC2007/
├── JPEGImages/                 # Input images (JPG or PNG; no fixed size required)
├── SegmentationClass/          # PNG label images
└── ImageSets/Segmentation/
    ├── train.txt               # One image name without extension per line
    └── val.txt
```

The rural aerial dataset defines five classes, including background:

| Index | Class | Description | RGB color |
| --- | --- | --- | --- |
| 0 | `_background_` | Background | (0, 0, 0) |
| 1 | `building` | Building | (128, 0, 0) |
| 2 | `sky` | Sky | (0, 128, 0) |
| 3 | `tree` | Tree / vegetation | (128, 128, 0) |
| 4 | `way` | Road / path | (0, 0, 128) |

Each label pixel must contain the class index (`0, 1, 2, 3, 4`). Labels encoded as binary `0/255` masks will prevent effective training. See [segmentation-format-fix](https://github.com/bubbliiiing/segmentation-format-fix) if label conversion is necessary.

The original annotations were produced with LabelMe. Run `process_dataset.ipynb` to organize `img.png` and `label.png` files into VOC format, then run `voc_annotation.py` to generate the train/validation split and validate labels.

## Model Architecture

DeepLabV3+ combines three stages:

1. A **backbone** (MobileNetV2 or Xception) extracts low-level and high-level features.
2. **ASPP** (Atrous Spatial Pyramid Pooling) collects multi-scale context through parallel 1x1 and dilated 3x3 convolutions (rates 6, 12, and 18), plus global average pooling.
3. A **decoder** upsamples ASPP features, fuses them with low-level backbone features, applies two 3x3 convolutions, and produces per-pixel class logits through a 1x1 classification head.

MobileNetV2 uses inverted residual blocks and depthwise separable convolutions (about 2.2M parameters). `downsample_factor=8` offers higher accuracy at greater memory cost, while `16` is faster.

## Quick Start

```bash
git clone <repository-url>
cd Deeplabv3-_mobilenet2_ruralscapes-master
pip install -r requirements.txt  # If unavailable, install the dependencies listed above.
```

The repository includes trained weights at `logs/best_epoch_weights.pth`. Run batch inference with:

```bash
python predict.py
```

The default mode is `dir_predict`: images in `img/` are processed and results are written to `img_out/`.

For one image, edit `predict.py`:

```python
mode = "predict"
name_classes = ["_background_", "building", "sky", "tree", "way"]
```

To inspect the network, FLOPs, and parameter count:

```bash
python summary.py
```

## Training on a Custom Dataset

1. Arrange the images and labels in the VOC layout above. Images belong in `JPEGImages/`; matching PNG labels whose pixel values are class indices belong in `SegmentationClass/`.
2. Generate the split and validate labels:

   ```bash
   python voc_annotation.py
   ```

3. Edit the required configuration in `train.py`:

   ```python
   num_classes = 5
   backbone = "mobilenet"       # or "xception"
   model_path = "model_data/deeplab_mobilenetv2.pth"
   input_shape = [256, 256]
   VOCdevkit_path = "VOCdevkit"
   ```

   Useful optional settings include `Cuda`, `distributed`, `fp16`, `Freeze_Train`, `Freeze_Epoch`, `UnFreeze_Epoch`, `optimizer_type`, `Init_lr`, `lr_decay_type`, `dice_loss`, and `focal_loss`.

4. Start training:

   ```bash
   python train.py
   ```

For Linux DDP training:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
```

For Windows or simple data-parallel training, set `distributed = False` in `train.py` and run:

```bash
set CUDA_VISIBLE_DEVICES=0,1
python train.py
```

Training creates checkpoints and loss/mIoU logs in `logs/loss_YYYY_MM_DD_HH_MM_SS/`. View them in real time with:

```bash
tensorboard --logdir logs/
```

Training uses two stages: first freeze the backbone to train the decoder efficiently, then unfreeze the full network for fine-tuning. The learning rate is automatically scaled relative to a nominal batch size of 16.

## Inference and Visualization

Set `mode` in `predict.py` to one of the following:

| Mode | Purpose |
| --- | --- |
| `predict` | Run prediction on one image; optional class-pixel statistics |
| `video` | Process a webcam or video file |
| `fps` | Benchmark inference speed |
| `dir_predict` | Process every image in an input directory |
| `export_onnx` | Export the model to ONNX |

For batch prediction, use:

```python
mode = "dir_predict"
dir_origin_path = "img/"
dir_save_path = "img_out/"
```

Set `mix_type` in `deeplab.py` to control visual output: `0` overlays the segmentation on the source image, `1` outputs the color segmentation only, and `2` keeps foreground objects while removing the background.

## Evaluation

Run:

```bash
python get_miou.py
```

`miou_mode` controls the workflow:

| Value | Behavior |
| --- | --- |
| 0 | Generate predictions and calculate mIoU |
| 1 | Generate predictions only (`miou_out/`) |
| 2 | Calculate mIoU using existing predictions |

The project reports mIoU, mean pixel accuracy (mPA), Precision, Recall, F1 score, and a confusion matrix. The `miou_out/` directory contains per-image predictions, metric charts, and `confusion_matrix.csv`. Set `eval_flag = True` in `train.py` to evaluate validation mIoU every `eval_period` epochs during training.

## Key Configuration

Important inference settings in `deeplab.py` are `model_path`, `num_classes`, `backbone`, `input_shape`, `downsample_factor`, `mix_type`, and `cuda`.

Important training settings in `train.py` are `num_classes`, `backbone`, `pretrained`, `model_path`, `input_shape`, `Freeze_Epoch`, `UnFreeze_Epoch`, `Freeze_batch_size`, `Unfreeze_batch_size`, `Init_lr`, `optimizer_type`, `lr_decay_type`, `save_period`, `eval_flag`, `dice_loss`, `focal_loss`, `cls_weights`, `fp16`, and `num_workers`.

## Results

On the included dataset (313 rural aerial images and five classes), the MobileNetV2 configuration uses a 256x256 input, downsample factor 8, SGD (`Init_lr=7e-3`, momentum `0.9`, weight decay `1e-4`), cosine decay, 80 epochs (50 frozen and 30 unfrozen), and batch size 4.

The logs in `logs/loss_2024_03_24_15_18_29/` report approximately:

| Metric | Value |
| --- | --- |
| Final training loss | 0.38 |
| Final validation loss | 0.62 |
| Best validation mIoU | 57.4% |
| Model-weight size | 23 MB |

Validation mIoU peaks near epoch 16, then shows mild overfitting. More data augmentation, Dice/Focal Loss, or additional data may improve results.

## FAQ

### Shape mismatch during training

Ensure that `model_path`, `backbone`, `num_classes`, and `downsample_factor` match between training and inference. Changing the backbone requires compatible weights or retraining.

### CUDA out of memory

Reduce `Freeze_batch_size` and `Unfreeze_batch_size`, reduce `input_shape` (for example, 256 to 128), set `downsample_factor = 16`, or enable `fp16 = True`.

### Loss does not decrease or mIoU is very low

Run the label check in `voc_annotation.py`, verify that label pixels are class indices rather than `0/255`, verify `num_classes`, and consider pretrained weights.

### Prediction is entirely black

Verify that `model_path` points to the intended weights and that `backbone` and `num_classes` match the training configuration.

### Training from scratch

```python
model_path = ""
pretrained = False
Freeze_Train = False
```

Training from scratch is not recommended; use ImageNet-pretrained backbone weights when possible.

### Resuming training

Set `model_path` to an existing checkpoint under `logs/`, set `Init_Epoch` to the completed epoch count, and adjust `Freeze_Epoch` and `UnFreeze_Epoch` to keep epochs continuous.

### Switching to Xception

```python
backbone = "xception"
model_path = "model_data/deeplab_xception.pth"
```

Xception has roughly 40M+ parameters and often provides higher accuracy at lower speed.

## Acknowledgements

- [DeepLabV3+: Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611), Chen et al., ECCV 2018.
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381), Sandler et al., CVPR 2018.
- [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357), Chollet, CVPR 2017.
- Reference implementation: [bubbliiiing/deeplabv3-plus-pytorch](https://github.com/bubbliiiing/deeplabv3-plus-pytorch).

Pretrained backbone downloads:

- MobileNetV2 (ImageNet): `https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/mobilenet_v2.pth.tar`
- Xception (ImageNet): `https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/xception_pytorch_imagenet.pth`

## License

This project is provided for learning and research. For commercial use, contact the original author and comply with the applicable open-source licenses.
