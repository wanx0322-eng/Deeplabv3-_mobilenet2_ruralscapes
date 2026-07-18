"""重建 train/val 划分：验证集保持不变，新增图片并入训练集，并剔除数据泄漏。

数据集里存在 18 组【像素完全相同】的重复图片（同一张图被标注了两次，
存成了两个文件名）。如果重复的另一半落到验证集里，训练就等于见过验证图 ——
mIoU 会被虚高。本脚本：

  1. 验证集 val.txt 原样保留（保证与历史结果可比）
  2. 其余所有【有索引图标签】的图片进入训练集
  3. 剔除任何与验证集图片像素重复的训练图片（泄漏）
  4. 训练集内部再按像素去重

用法: python tools/rebuild_splits.py [--dry-run]
"""
import argparse
import hashlib
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOC = os.path.join(ROOT, "VOCdevkit", "VOC2007")
IMG_DIR = os.path.join(VOC, "JPEGImages")
MASK_DIR = os.path.join(VOC, "SegmentationClass")
SPLIT_DIR = os.path.join(VOC, "ImageSets", "Segmentation")


def img_hash(stem):
    path = os.path.join(IMG_DIR, stem + ".png")
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"))
    return hashlib.md5(arr.tobytes()).hexdigest()


def is_index_label(stem):
    path = os.path.join(MASK_DIR, stem + ".png")
    if not os.path.exists(path):
        return False
    with Image.open(path) as im:
        return im.mode in ("P", "L")


def read_split(name):
    path = os.path.join(SPLIT_DIR, name + ".txt")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def write_split(name, stems):
    with open(os.path.join(SPLIT_DIR, name + ".txt"), "w") as f:
        for s in stems:
            f.write(s + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    val = read_split("val")
    old_train = read_split("train")
    all_stems = sorted((f[:-4] for f in os.listdir(IMG_DIR) if f.endswith(".png")),
                       key=lambda s: int(s) if s.isdigit() else 10 ** 9)

    labelled = [s for s in all_stems if is_index_label(s)]
    print(f"图片总数 {len(all_stems)}  有索引图标签 {len(labelled)}")
    print(f"原 train {len(old_train)}  原 val {len(val)}")

    val_set = set(val)
    hashes = {s: img_hash(s) for s in labelled}
    val_hashes = {hashes[s] for s in val if s in hashes}

    # ---- 先自检：原训练集里有没有已经泄漏的图 ----
    pre_leak = [s for s in old_train if hashes.get(s) in val_hashes]
    if pre_leak:
        print(f"\n[!] 原训练集中已存在与验证集重复的图片: {sorted(pre_leak, key=int)}")
    else:
        print("\n原训练集与验证集之间没有像素级重复。")

    # ---- 构建新训练集 ----
    candidates = [s for s in labelled if s not in val_set]
    train, seen, leaked, dup_in_train = [], set(), [], []
    for s in candidates:
        h = hashes[s]
        if h in val_hashes:
            leaked.append(s)          # 与验证集重复 -> 丢弃（泄漏）
            continue
        if h in seen:
            dup_in_train.append(s)    # 训练集内部重复 -> 丢弃（冗余）
            continue
        seen.add(h)
        train.append(s)

    train.sort(key=lambda s: int(s) if s.isdigit() else 10 ** 9)

    print(f"\n候选训练图 {len(candidates)}")
    print(f"  剔除：与验证集像素重复（泄漏） {len(leaked)} -> {sorted(leaked, key=int)}")
    print(f"  剔除：训练集内部重复           {len(dup_in_train)} -> {sorted(dup_in_train, key=int)}")
    print(f"\n最终  train={len(train)}   val={len(val)}   (train 原为 {len(old_train)})")

    if args.dry_run:
        print("\n[dry-run] 未写入任何文件。")
        return

    write_split("train", train)
    write_split("val", val)                     # 原样写回，保持不变
    write_split("trainval", train + val)
    write_split("test", [])
    print("\n已写入 train.txt / val.txt / trainval.txt / test.txt")


if __name__ == "__main__":
    main()
