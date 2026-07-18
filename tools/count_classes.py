# -*- coding: utf-8 -*-
"""统计各类别像素占比，给出可直接填进 workstation_config.json 的 cls_weights。

    python tools/count_classes.py
    python tools/count_classes.py --split train --voc-root VOCdevkit_uavid

来源：F 盘工程化分支的 count_classes.py。这里补了 sqrt 逆频率一档 ——
本项目 workstation_config.json 里在用的 cls_weights 就是这个口径，
不补的话输出的三档权重没有一档能和现有配置对上。
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def load_stems(voc, split):
    seg_dir = os.path.join(voc, "VOC2007", "ImageSets", "Segmentation")
    if split == "all":
        mask_dir = os.path.join(voc, "VOC2007", "SegmentationClass")
        return sorted(os.path.splitext(f)[0] for f in os.listdir(mask_dir)
                      if f.lower().endswith(".png"))
    files = ["%s.txt" % split] if split != "trainval" else ["trainval.txt"]
    if split == "trainval" and not os.path.exists(
            os.path.join(seg_dir, "trainval.txt")):
        files = ["train.txt", "val.txt"]
    stems = []
    for name in files:
        with open(os.path.join(seg_dir, name), "r") as handle:
            stems += [line.strip() for line in handle if line.strip()]
    return stems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train",
                        choices=["train", "val", "trainval", "all"],
                        help="默认 train —— 权重应当只按训练集统计，"
                             "把验证集算进来是信息泄漏")
    parser.add_argument("--voc-root", default="VOCdevkit")
    args = parser.parse_args()

    voc = os.path.join(ROOT, args.voc_root)
    mask_dir = os.path.join(voc, "VOC2007", "SegmentationClass")
    stems = load_stems(voc, args.split)
    print("[信息] split=%s，标签数量=%d\n" % (args.split, len(stems)))

    pixel_count = np.zeros(256, dtype=np.int64)
    image_count = np.zeros(256, dtype=np.int64)
    n_ok, n_rgb = 0, 0
    for stem in stems:
        path = os.path.join(mask_dir, stem + ".png")
        if not os.path.exists(path):
            continue
        arr = np.array(Image.open(path))
        if arr.ndim == 3:
            #   RGB 标签不能直接当类别下标用，统计结果会是错的
            n_rgb += 1
            continue
        values, counts = np.unique(arr, return_counts=True)
        pixel_count[values] += counts
        image_count[values] += 1
        n_ok += 1

    if n_rgb:
        print("[警告] 跳过 %d 个 RGB 三通道标签，先跑 python tools/fix_rgb_masks.py\n"
              % n_rgb)
    if n_ok == 0:
        raise SystemExit("没有可用的索引图标签")

    present = np.where(pixel_count > 0)[0]
    total = int(pixel_count.sum())
    print("[信息] 成功读取 %d 张标签，总像素 %s" % (n_ok, format(total, ",")))
    print("[结果] 出现的类别值: %s" % present.tolist())
    print("[结果] num_classes 建议设为 %d\n" % (int(present.max()) + 1))

    #   类别名从配置里取，输出直接能看懂是哪个类
    try:
        from workstation.config import Config
        names = Config().dataset["class_names"]
    except Exception:
        names = []

    print("-" * 74)
    print("%6s %-14s %16s %11s %10s" % ("类别值", "类别名", "像素数", "像素占比", "图像数"))
    print("-" * 74)
    for value in present:
        name = names[value] if value < len(names) else "?"
        print("%6d %-14s %16s %10.4f%% %10d"
              % (value, name, format(int(pixel_count[value]), ","),
                 pixel_count[value] / total * 100, image_count[value]))
    print("-" * 74)

    freq = pixel_count[present] / total
    w_medfreq = np.median(freq) / freq
    w_inv = 1.0 / freq
    w_inv = w_inv / w_inv.mean()
    #   1/sqrt(频率) 归一化 —— 当前 workstation_config.json 用的就是这一档
    w_sqrt = 1.0 / np.sqrt(freq)
    w_sqrt = w_sqrt / w_sqrt.mean()

    def fmt(arr):
        return "[" + ", ".join("%.4f" % x for x in arr) + "]"

    print("\n[建议] 1/sqrt(频率) 归一化（本项目当前口径，最稳健）:")
    print('  "cls_weights": %s' % fmt(w_sqrt))
    print("\n[建议] median frequency balancing:")
    print('  "cls_weights": %s' % fmt(w_medfreq))
    print("\n[建议] 逆频率归一化（最激进，小类容易过拟合）:")
    print('  "cls_weights": %s' % fmt(w_inv))

    try:
        current = Config().train.get("cls_weights")
        if current:
            print("\n[对照] workstation_config.json 现值: %s"
                  % json.dumps([round(float(x), 4) for x in current]))
    except Exception:
        pass
    print("\n[提示] 权重按类别值从小到大排列，长度必须等于 num_classes。"
          "\n       背景类若不想参与加权可手动调小。")


if __name__ == "__main__":
    main()
