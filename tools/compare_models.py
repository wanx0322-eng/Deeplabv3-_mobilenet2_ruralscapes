"""在验证集上横向对比多个权重文件，统一使用修复后的 mIoU 口径。

指标在【完整】混淆矩阵上计算，背景类展示但不计入平均（与 get_miou.py 一致）。
前向与评估协议来自 segcore，DeepLabV3+ 与 SegFormer 都支持。

用法:
    python tools/compare_models.py logs_v2_B/best_epoch_weights.pth
    python tools/compare_models.py --tta logs_v2_E/best_epoch_weights.pth
    # 混合对比：每个权重用 权重路径:主干名 指定各自的主干
    python tools/compare_models.py \\
        logs_v2_B/best_epoch_weights.pth:mobilenet \\
        logs_segformer_b2/best_segformer.pth:segformer-b2
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch

from segcore import evaluate_hist, load_net, per_image_hists, read_split

NAMES = ["_background_", "building", "sky", "tree", "way"]
REMOVE = [0]


def parse_target(spec, default_backbone):
    """'路径' 或 '路径:主干名' -> (路径, 主干名)"""
    #   Windows 盘符里的冒号不能当分隔符，从右边找且要求右段不含路径分隔符
    if ":" in spec:
        head, tail = spec.rsplit(":", 1)
        if head and not any(sep in tail for sep in ("/", "\\")):
            return head, tail
    return spec, default_backbone


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+", help="权重路径，可写成 路径:主干名")
    ap.add_argument("--num-classes", type=int, default=5)
    ap.add_argument("--backbone", default="mobilenet",
                    help="未在路径里单独指定时使用的主干")
    ap.add_argument("--downsample-factor", type=int, default=8)
    ap.add_argument("--input-shape", type=int, nargs=2, default=[256, 256])
    ap.add_argument("--voc-root", default="VOCdevkit")
    ap.add_argument("--split", default="val")
    ap.add_argument("--tta", action="store_true",
                    help="测试时增强：水平翻转 + 多尺度 (0.75/1.0/1.25) 概率平均")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = args.num_classes
    voc = os.path.join(ROOT, args.voc_root)
    stems = read_split(voc, args.split)
    print("验证集 %s：%d 张，input %s，TTA=%s\n"
          % (args.split, len(stems), args.input_shape, args.tta))

    hdr = f"{'checkpoint':<44}{'mIoU':>8}{'mPA':>8}{'Acc':>8}   " + \
          "".join(f"{name:>13}" for name in NAMES[1:n])
    print(hdr)
    print("-" * len(hdr))

    for spec in args.ckpts:
        ckpt, backbone = parse_target(spec, args.backbone)
        if not os.path.exists(ckpt):
            print(f"{ckpt:<44}  (缺失)")
            continue
        net = load_net(ckpt, backbone, n, args.downsample_factor, device)
        hists = per_image_hists(net, stems, voc, n, args.input_shape, device,
                                backbone=backbone, tta=args.tta)
        m = evaluate_hist(hists.sum(0), n, REMOVE)
        iou = m["iou"]
        label = os.path.relpath(ckpt, ROOT)
        print(f"{label:<44}{m['miou']:8.2f}{m['mpa']:8.2f}{m['accuracy']:8.2f}   " +
              "".join(f"{iou[i]*100:13.2f}" for i in range(1, n)) +
              f"   [bg {iou[0]*100:.1f}]")
        del net
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
