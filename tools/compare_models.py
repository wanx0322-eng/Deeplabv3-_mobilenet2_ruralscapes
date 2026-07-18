"""在验证集上横向对比多个权重文件，统一使用修复后的 mIoU 口径。

指标在【完整】混淆矩阵上计算，背景类展示但不计入平均（与 get_miou.py 一致）。

用法:
    python tools/compare_models.py logs/best_epoch_weights.pth logs_v2_B/best_epoch_weights.pth
    python tools/compare_models.py --tta logs_v2_E/best_epoch_weights.pth

仅适用于 DeepLabV3+ 权重；SegFormer 权重的评估请用工作站「精度评估」页
或 tools/train_segformer.py（训练结束会输出同口径指标）。
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from nets.deeplabv3_plus import DeepLab
from utils.utils import cvtColor, preprocess_input, resize_image
from utils.utils_metrics import (mean_metric, per_class_iu, per_class_PA_Recall,
                                 per_class_Precision, per_Accuracy)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOC = os.path.join(ROOT, "VOCdevkit", "VOC2007")
NAMES = ["_background_", "building", "sky", "tree", "way"]
REMOVE = [0]


def _forward_prob(net, img, shape, device, flip=False):
    """单次前向：letterbox -> 网络 -> softmax -> 去灰条 -> 还原原尺寸的概率图"""
    ow, oh = img.size
    src = img.transpose(Image.FLIP_LEFT_RIGHT) if flip else img
    data, nw, nh = resize_image(src, (shape[1], shape[0]))
    x = np.expand_dims(np.transpose(
        preprocess_input(np.array(data, np.float32)), (2, 0, 1)), 0)
    pr = net(torch.from_numpy(x).to(device))[0]
    pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
    pr = pr[(shape[0] - nh) // 2:(shape[0] - nh) // 2 + nh,
            (shape[1] - nw) // 2:(shape[1] - nw) // 2 + nw]
    pr = cv2.resize(pr, (ow, oh), interpolation=cv2.INTER_LINEAR)
    if flip:
        pr = pr[:, ::-1]
    return pr


def evaluate(ckpt, num_classes, backbone, downsample_factor, input_shape, device,
             tta=False):
    net = DeepLab(num_classes=num_classes, backbone=backbone,
                  downsample_factor=downsample_factor, pretrained=False)
    net.load_state_dict(torch.load(ckpt, map_location="cpu"))
    net = net.eval().to(device)

    #   TTA：水平翻转 × 多尺度（0.75 / 1.0 / 1.25 倍，取 32 的倍数），
    #   对概率图取平均后再 argmax。纯推理端增强，不改训练。
    if tta:
        base = input_shape[0]
        scales = sorted({max(64, int(round(base * s / 32)) * 32)
                         for s in (0.75, 1.0, 1.25)})
        views = [((s, s), f) for s in scales for f in (False, True)]
    else:
        views = [(tuple(input_shape), False)]

    val = open(os.path.join(VOC, "ImageSets/Segmentation/val.txt")).read().split()
    hist = np.zeros((num_classes, num_classes), np.int64)

    with torch.no_grad():
        for stem in val:
            img = cvtColor(Image.open(os.path.join(VOC, "JPEGImages", stem + ".png")))
            prob = None
            for shape, flip in views:
                p = _forward_prob(net, img, shape, device, flip)
                prob = p if prob is None else prob + p
            pr = prob.argmax(-1)

            gt = np.array(Image.open(os.path.join(VOC, "SegmentationClass", stem + ".png")))
            a, b = gt.flatten(), pr.flatten()
            k = (a >= 0) & (a < num_classes)
            hist += np.bincount(num_classes * a[k].astype(int) + b[k],
                                minlength=num_classes ** 2).reshape(num_classes, num_classes)

    return hist, per_class_iu(hist), per_class_PA_Recall(hist), per_class_Precision(hist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--num-classes", type=int, default=5)
    ap.add_argument("--backbone", default="mobilenet")
    ap.add_argument("--downsample-factor", type=int, default=8)
    ap.add_argument("--input-shape", type=int, nargs=2, default=[256, 256])
    ap.add_argument("--tta", action="store_true",
                    help="测试时增强：水平翻转 + 多尺度 (0.75/1.0/1.25) 概率平均")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = args.num_classes

    hdr = f"{'checkpoint':<42}{'mIoU':>8}{'mPA':>8}{'Acc':>8}   " + \
          "".join(f"{name:>13}" for name in NAMES[1:])
    print(hdr)
    print("-" * len(hdr))

    for ck in args.ckpts:
        if not os.path.exists(ck):
            print(f"{ck:<42}  (缺失)")
            continue
        hist, iou, rec, pre = evaluate(ck, n, args.backbone,
                                       args.downsample_factor, args.input_shape, device,
                                       tta=args.tta)
        miou = mean_metric(iou, n, REMOVE) * 100
        mpa = mean_metric(rec, n, REMOVE) * 100
        acc = per_Accuracy(hist) * 100
        label = os.path.relpath(ck, ROOT)
        print(f"{label:<42}{miou:8.2f}{mpa:8.2f}{acc:8.2f}   " +
              "".join(f"{iou[i]*100:13.2f}" for i in range(1, n)) +
              f"   [bg {iou[0]*100:.1f}]")


if __name__ == "__main__":
    main()
