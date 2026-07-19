"""统一的评估协议：逐图混淆矩阵 -> 指标。

口径（全项目唯一，不要在别处重新实现）：
  * 指标在【完整】混淆矩阵上计算；
  * remove_classes 只影响求平均与打印，绝不从矩阵里删行列
    —— 删行列会丢掉跨背景的 FP/FN，本数据集上前景 IoU 虚高约 15.7 个点。
    见 tests/test_metrics.py 的回归锁。

返回逐图矩阵 (N, C, C) 而不是直接聚合，是为了让配对 bootstrap
（tools/paired_bootstrap.py）能按图重采样。求和即得聚合矩阵。
"""
import os

import numpy as np
from PIL import Image

from .engine import is_segformer, predict_mask


def read_split(voc_root, split="val"):
    """读取 ImageSets/Segmentation/<split>.txt 的图像 stem 列表。"""
    path = os.path.join(voc_root, "VOC2007", "ImageSets", "Segmentation",
                        split + ".txt")
    with open(path, "r") as handle:
        return handle.read().split()


def per_image_hists(net, stems, voc_root, num_classes, input_shape, device,
                    backbone="mobilenet", tta=False, progress=None,
                    save_pred_dir=None):
    """对每张图算一个 num_classes×num_classes 混淆矩阵，返回 (N, C, C)。

    net 可以是刚构建的推理网络，也可以是训练中的模型（含 EMA 副本）——
    训练循环的周期评估复用同一条路径，不再自带一份前向。

    save_pred_dir 非空时顺带把预测 png 落盘（工作站评估页要展示）。
    """
    from utils.utils import cvtColor

    segformer = is_segformer(backbone)
    image_dir = os.path.join(voc_root, "VOC2007", "JPEGImages")
    mask_dir = os.path.join(voc_root, "VOC2007", "SegmentationClass")
    if save_pred_dir:
        os.makedirs(save_pred_dir, exist_ok=True)

    hists = np.zeros((len(stems), num_classes, num_classes), np.int64)
    was_training = getattr(net, "training", False)
    if was_training:
        net.eval()
    try:
        for i, stem in enumerate(stems):
            image = cvtColor(Image.open(os.path.join(image_dir, stem + ".png")))
            pred = predict_mask(net, image, input_shape, device,
                                segformer=segformer, tta=tta)
            if save_pred_dir:
                Image.fromarray(pred).save(
                    os.path.join(save_pred_dir, stem + ".png"))

            gt = np.array(Image.open(os.path.join(mask_dir, stem + ".png")))
            if gt.size != pred.size:
                #   与 compute_mIoU 一致：尺寸对不上的样本跳过而不是崩溃
                print("Skipping: gt %d vs pred %d, %s"
                      % (gt.size, pred.size, stem))
                continue
            a, b = gt.flatten(), pred.flatten()
            k = (a >= 0) & (a < num_classes)
            hists[i] = np.bincount(num_classes * a[k].astype(int) + b[k],
                                   minlength=num_classes ** 2
                                   ).reshape(num_classes, num_classes)
            if progress is not None:
                progress(i + 1, len(stems))
    finally:
        if was_training:
            net.train()
    return hists


def evaluate_hist(hist, num_classes, remove_classes=(0,)):
    """聚合矩阵 -> (fg_miou, fg_mpa, accuracy, per-class dict)，均为百分数。"""
    from utils.utils_metrics import (mean_metric, per_Accuracy, per_class_iu,
                                     per_class_PA_Recall, per_class_Precision)

    remove = list(remove_classes) if remove_classes else None
    iou = per_class_iu(hist)
    recall = per_class_PA_Recall(hist)
    precision = per_class_Precision(hist)
    return {
        "miou": float(mean_metric(iou, num_classes, remove) * 100),
        "mpa": float(mean_metric(recall, num_classes, remove) * 100),
        "accuracy": float(per_Accuracy(hist) * 100),
        "iou": iou,
        "recall": recall,
        "precision": precision,
    }


__all__ = ["evaluate_hist", "per_image_hists", "read_split"]
