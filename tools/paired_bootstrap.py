"""两个权重在同一验证集上的配对 bootstrap 显著性检验。

为什么需要它：验证集只有 50 张，单次训练的 mIoU 差 1 个点以内完全可能是噪声。
UAVid 那次两阶段实验点估计 +0.44，配对 bootstrap 95% 区间 [-0.33, +1.47] 跨过 0，
结论是"无效"。没有这一步就会把噪声当成提升。

配对的含义：每次重采样对两个模型使用【同一批】图像，差值的方差因此显著小于
两个独立区间各自的方差 —— 这也是为什么不能拿两个模型各自的置信区间比重叠。

    python tools/paired_bootstrap.py \\
        --a logs_sf_ade/best_segformer.pth  --a-backbone segformer-b2 \\
        --b logs_sf_city/best_segformer.pth --b-backbone segformer-b2

DeepLab 权重用 --a-backbone mobilenet / xception。
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from utils.utils import cvtColor, preprocess_input, resize_image
from utils.utils_metrics import mean_metric, per_class_iu
from workstation.core.engine import SEGFORMER_HUB, is_segformer

VOC = os.path.join(ROOT, "VOCdevkit", "VOC2007")
NAMES = ["_background_", "building", "sky", "tree", "way"]
REMOVE = [0]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_net(ckpt, backbone, num_classes, downsample_factor, device):
    if is_segformer(backbone):
        from transformers import SegformerForSemanticSegmentation
        net = SegformerForSemanticSegmentation.from_pretrained(
            SEGFORMER_HUB[backbone], num_labels=num_classes,
            ignore_mismatched_sizes=True)
    else:
        from nets.deeplabv3_plus import DeepLab
        net = DeepLab(num_classes=num_classes, backbone=backbone,
                      downsample_factor=downsample_factor, pretrained=False)
    net.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return net.eval().to(device)


@torch.no_grad()
def per_image_hists(net, backbone, stems, input_shape, num_classes, device):
    """返回 shape=(N, C, C) 的逐图混淆矩阵 —— 配对重采样需要按图聚合。"""
    segformer = is_segformer(backbone)
    hists = np.zeros((len(stems), num_classes, num_classes), np.int64)
    for i, stem in enumerate(stems):
        img = cvtColor(Image.open(os.path.join(VOC, "JPEGImages", stem + ".png")))
        ow, oh = img.size
        data, nw, nh = resize_image(img, (input_shape[1], input_shape[0]))
        arr = preprocess_input(np.array(data, np.float32))
        if segformer:
            arr = (arr - MEAN) / STD
        x = torch.from_numpy(np.expand_dims(np.transpose(arr, (2, 0, 1)), 0)).to(device)

        if segformer:
            logits = net(pixel_values=x).logits
            logits = F.interpolate(logits, size=tuple(input_shape),
                                   mode="bilinear", align_corners=False)[0]
        else:
            logits = net(x)[0]
        prob = F.softmax(logits.permute(1, 2, 0), dim=-1).cpu().numpy()
        prob = prob[(input_shape[0] - nh) // 2:(input_shape[0] - nh) // 2 + nh,
                    (input_shape[1] - nw) // 2:(input_shape[1] - nw) // 2 + nw]
        pred = cv2.resize(prob, (ow, oh), interpolation=cv2.INTER_LINEAR).argmax(-1)

        gt = np.array(Image.open(os.path.join(VOC, "SegmentationClass", stem + ".png")))
        a, b = gt.flatten(), pred.flatten()
        k = (a >= 0) & (a < num_classes)
        hists[i] = np.bincount(num_classes * a[k].astype(int) + b[k],
                               minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hists


def fg_miou(hist, num_classes):
    return mean_metric(per_class_iu(hist), num_classes, REMOVE) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="基线权重")
    ap.add_argument("--b", required=True, help="对照权重")
    ap.add_argument("--a-backbone", default="segformer-b2")
    ap.add_argument("--b-backbone", default="segformer-b2")
    ap.add_argument("--a-label", default="A")
    ap.add_argument("--b-label", default="B")
    ap.add_argument("--num-classes", type=int, default=5)
    ap.add_argument("--downsample-factor", type=int, default=8)
    ap.add_argument("--input-size", type=int, default=256)
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shape = [args.input_size, args.input_size]
    stems = open(os.path.join(VOC, "ImageSets/Segmentation/val.txt")).read().split()
    print("验证集 %d 张，input %s，device %s\n" % (len(stems), shape, device))

    hists = {}
    for key, ckpt, backbone in (("a", args.a, args.a_backbone),
                                ("b", args.b, args.b_backbone)):
        net = load_net(ckpt, backbone, args.num_classes,
                       args.downsample_factor, device)
        hists[key] = per_image_hists(net, backbone, stems, shape,
                                     args.num_classes, device)
        del net
        torch.cuda.empty_cache()

    point_a = fg_miou(hists["a"].sum(0), args.num_classes)
    point_b = fg_miou(hists["b"].sum(0), args.num_classes)
    delta = point_b - point_a

    print("=" * 64)
    print("%-28s %s" % (args.a_label, "fg mIoU %.2f" % point_a))
    print("%-28s %s" % (args.b_label, "fg mIoU %.2f" % point_b))
    print("%-28s %+.2f" % ("点估计差值 (B - A)", delta))
    print("=" * 64)

    print("\n逐类 IoU：")
    iou_a = per_class_iu(hists["a"].sum(0)) * 100
    iou_b = per_class_iu(hists["b"].sum(0)) * 100
    for i, name in enumerate(NAMES[:args.num_classes]):
        tag = "   (不计入平均)" if i in REMOVE else ""
        print("  %-14s %6.2f -> %6.2f  (%+.2f)%s"
              % (name, iou_a[i], iou_b[i], iou_b[i] - iou_a[i], tag))

    #   配对 bootstrap：每次对两个模型抽【同一批】图
    rng = np.random.default_rng(args.seed)
    n = len(stems)
    deltas = np.empty(args.iters)
    for it in range(args.iters):
        idx = rng.integers(0, n, n)
        deltas[it] = (fg_miou(hists["b"][idx].sum(0), args.num_classes)
                      - fg_miou(hists["a"][idx].sum(0), args.num_classes))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p_better = float((deltas > 0).mean())

    print("\n配对 bootstrap（%d 次重采样，seed %d）：" % (args.iters, args.seed))
    print("  差值均值      %+.2f" % deltas.mean())
    print("  95%% 置信区间  [%+.2f, %+.2f]" % (lo, hi))
    print("  P(B > A)      %.3f" % p_better)
    if lo <= 0 <= hi:
        print("\n结论：区间跨过 0，**差异不显著**。"
              "\n      不能据此认为其中一个底座更好 —— 参考 README 里 UAVid 那次的处理方式。")
    else:
        print("\n结论：区间不跨 0，差异显著（%s 更好）。"
              % (args.b_label if lo > 0 else args.a_label))
        print("      注意：这只是单次训练对单次训练。要下强结论，"
              "两臂各跑多个 seed 再比更稳妥。")


if __name__ == "__main__":
    main()
