"""把 SegmentationClass 里残留的 RGB 三通道标签转成 8 位调色板索引图。

数据集里有一部分标签是直接存成 RGB 的（每个像素是颜色，不是类别索引），
DeeplabDataset 只认索引图，这些文件一旦进入划分就会在 one-hot 处崩溃。
本脚本按 VOC 调色板做颜色 -> 类别索引的映射，杂色（抗锯齿/压缩产生）
按最近邻颜色归类。原文件先备份到 VOCdevkit/_mask_backup_rgb/。

用法: python tools/fix_rgb_masks.py [--dry-run]
"""
import argparse
import os
import shutil
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VOC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "VOCdevkit")
MASK_DIR = os.path.join(VOC, "VOC2007", "SegmentationClass")
BACKUP_DIR = os.path.join(VOC, "_mask_backup_rgb")

CLASS_COLORS = [
    (0, 0, 0),          # 0 _background_
    (128, 0, 0),        # 1 building
    (0, 128, 0),        # 2 sky
    (128, 128, 0),      # 3 tree
    (0, 0, 128),        # 4 way
]
CLASS_NAMES = ["_background_", "building", "sky", "tree", "way"]


def voc_palette():
    """标准 VOC 调色板，前 5 项即 CLASS_COLORS。"""
    palette = [0] * 768
    for i, (r, g, b) in enumerate(CLASS_COLORS):
        palette[i * 3:i * 3 + 3] = [r, g, b]
    return palette


def rgb_to_index(arr):
    """(H,W,3) uint8 -> (H,W) uint8 类别索引，按最近欧氏距离归类。"""
    colors = np.array(CLASS_COLORS, np.int16)
    flat = arr.reshape(-1, 3).astype(np.int16)
    # (N, 5) 距离矩阵
    dist = ((flat[:, None, :] - colors[None, :, :]) ** 2).sum(-1)
    idx = dist.argmin(1).astype(np.uint8)
    exact = (dist.min(1) == 0)
    return idx.reshape(arr.shape[:2]), int((~exact).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只报告，不写文件")
    args = ap.parse_args()

    targets = []
    for fn in sorted(os.listdir(MASK_DIR)):
        if not fn.lower().endswith(".png"):
            continue
        with Image.open(os.path.join(MASK_DIR, fn)) as im:
            if im.mode != "P":
                targets.append(fn)

    print(f"需要转换的非索引标签: {len(targets)} / "
          f"{len(os.listdir(MASK_DIR))}")
    if not targets:
        print("没有需要转换的文件。")
        return

    if not args.dry_run:
        os.makedirs(BACKUP_DIR, exist_ok=True)

    palette = voc_palette()
    total_snapped = 0
    hist = np.zeros(5, np.int64)

    for fn in targets:
        path = os.path.join(MASK_DIR, fn)
        arr = np.array(Image.open(path).convert("RGB"), np.uint8)
        idx, snapped = rgb_to_index(arr)
        total_snapped += snapped
        hist += np.bincount(idx.reshape(-1), minlength=5)

        if args.dry_run:
            continue

        shutil.copy2(path, os.path.join(BACKUP_DIR, fn))
        out = Image.fromarray(idx, mode="P")
        out.putpalette(palette)
        out.save(path, optimize=True)

    print(f"非精确匹配（就近归类）的像素: {total_snapped:,}")
    print("转换后类别像素分布:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"   {i} {name:<14}{hist[i] / hist.sum() * 100:7.2f}%")
    if args.dry_run:
        print("\n[dry-run] 未写入任何文件。")
    else:
        print(f"\n原文件已备份到 {BACKUP_DIR}")
        print(f"已转换 {len(targets)} 个标签为 P 模式索引图。")


if __name__ == "__main__":
    main()
