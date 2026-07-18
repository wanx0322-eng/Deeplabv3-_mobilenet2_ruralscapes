"""把外部数据集（RuralScapes / UAVid 等）转换成本项目的 VOC 目录结构。

产出与 VOCdevkit 完全同构，可直接喂给 tools/train_segformer.py 做第一阶段
领域内预训练：

    VOCdevkit_<name>/VOC2007/
        JPEGImages/            *.png
        SegmentationClass/     *.png  (P 模式索引图，像素值=目标类别下标)
        ImageSets/Segmentation/train.txt val.txt

重要：同一数据集的不同再分发版本，标签编码可能完全不同。
UAVid 官方发布的是彩色标签（building=[128,0,0] 等），而 HuggingFace 上的
dronefreak/UAVid-2020 镜像预先转成了类别下标（building=[1,1,1]）。
本脚本会先采样实际颜色、与各内置变体比对覆盖率后自动选择；覆盖率低于 90%
直接报错而不是硬转——配色选错会静默地把整个数据集标错，是最危险的失败模式。

典型流程
--------
1) 先看源标签用了哪些颜色（RuralScapes 未公开配色，必须先扫）：
       python tools/import_external.py --discover /data/ruralscapes/labels

   输出一份 JSON 模板，把每个颜色填上 RuralScapes 的源类别名
   （forest/land/hill/sky/residential/road/water/person/church/haystack/fence/car）。

2) 转换：
       python tools/import_external.py \
           --images /data/ruralscapes/frames --labels /data/ruralscapes/labels \
           --dataset ruralscapes --scheme base5 \
           --palette my_palette.json --out VOCdevkit_ruralscapes

   UAVid 的官方配色已内置，可省略 --palette：
       python tools/import_external.py --images ... --labels ... \
           --dataset uavid --scheme base5 --out VOCdevkit_uavid
"""
import argparse
import json
import os
import random
import sys
from collections import Counter

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.external_datasets import (PALETTE_VARIANTS, SOURCE_PALETTES,  # noqa: E402
                                     resolve)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def voc_palette(names):
    """目标类别的调色板，与本项目 VOCdevkit 保持一致"""
    base = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128),
            (128, 0, 128), (0, 128, 128), (128, 128, 128), (64, 0, 0)]
    palette = [0] * 768
    for i in range(len(names)):
        r, g, b = base[i % len(base)]
        palette[i * 3:i * 3 + 3] = [r, g, b]
    return palette


def discover(label_dir, top=40):
    """扫描标签目录里出现的颜色，输出待填写的调色板模板"""
    counter = Counter()
    files = [f for f in sorted(os.listdir(label_dir))
             if f.lower().endswith(IMAGE_EXTS)]
    sample = files[:200]
    print(f"扫描 {len(sample)} / {len(files)} 个标签文件…")
    for fn in sample:
        arr = np.asarray(Image.open(os.path.join(label_dir, fn)).convert("RGB"))
        cols, cnts = np.unique(arr.reshape(-1, 3), axis=0, return_counts=True)
        for c, n in zip(cols, cnts):
            counter[tuple(int(x) for x in c)] += int(n)

    total = sum(counter.values())
    print(f"\n共发现 {len(counter)} 种颜色，按占比排序（前 {top}）：")
    template = {}
    for col, n in counter.most_common(top):
        print(f"  {str(col):<20} {n / total * 100:7.3f}%")
        template[",".join(map(str, col))] = "<填写源类别名>"

    out = "palette_template.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)
    print(f"\n模板已写入 {out}，填好源类别名后用 --palette 传入。")
    print("注意：占比极小的颜色多半是抗锯齿杂色，可留空，转换时会就近归入背景。")


def sample_colors(label_dir, n=12):
    """采样若干标签，返回实际出现的颜色集合"""
    files = [f for f in sorted(os.listdir(label_dir))
             if f.lower().endswith(IMAGE_EXTS)][:n]
    seen = set()
    for fn in files:
        arr = np.asarray(Image.open(os.path.join(label_dir, fn)).convert("RGB"))
        for c in np.unique(arr.reshape(-1, 3), axis=0):
            seen.add(tuple(int(x) for x in c))
    return seen


def load_palette(dataset, palette_path, label_dir=None):
    if palette_path:
        with open(palette_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        out = {}
        for key, name in raw.items():
            if not name or name.startswith("<"):
                continue                      # 未填写的条目跳过
            rgb = tuple(int(x) for x in key.replace(" ", "").split(","))
            out[rgb] = name
        if not out:
            raise ValueError(f"{palette_path} 中没有任何已填写的类别名")
        return out

    #-------------------------------------------------------------------#
    #   同一数据集在不同再分发版本里可能是彩色标签或索引编码标签。
    #   用实际出现的颜色去匹配各变体，选覆盖率最高的那个 —— 配色选错会
    #   静默地把整个数据集标错，必须先验证再转换。
    #-------------------------------------------------------------------#
    variants = PALETTE_VARIANTS.get(dataset, [dataset])
    if label_dir:
        present = sample_colors(label_dir)
        best, best_cov = None, -1.0
        for v in variants:
            if v not in SOURCE_PALETTES:
                continue
            cov = len(present & set(SOURCE_PALETTES[v])) / max(len(present), 1)
            print(f"  调色板候选 {v:<14} 颜色覆盖率 {cov*100:5.1f}%")
            if cov > best_cov:
                best, best_cov = v, cov
        if best is None:
            raise ValueError(f"{dataset} 没有内置调色板，请用 --discover 扫描后通过 --palette 提供")
        if best_cov < 0.9:
            raise ValueError(
                f"内置调色板与实际标签不匹配（最高覆盖率仅 {best_cov*100:.1f}%）。\n"
                f"请用 --discover 扫描实际颜色，填好模板后通过 --palette 传入。")
        print(f"  -> 采用 {best}（覆盖率 {best_cov*100:.1f}%）")
        return SOURCE_PALETTES[best]

    if dataset in SOURCE_PALETTES:
        return SOURCE_PALETTES[dataset]
    raise ValueError(f"{dataset} 没有内置调色板，请用 --discover 扫描后通过 --palette 提供")


def build_lut(palette, mapping, names):
    """(源RGB -> 源类别名) + (源类别名 -> 目标下标) 合成一张查找表"""
    colors, targets, unknown = [], [], []
    for rgb, src_name in palette.items():
        if src_name not in mapping:
            unknown.append(src_name)
            continue
        colors.append(rgb)
        targets.append(mapping[src_name])
    if unknown:
        print(f"[警告] 调色板中这些类别不在映射表里，将并入背景: {sorted(set(unknown))}")
    #   必须 int32：颜色距离平方和最大 3*255^2=195075，int16 会溢出成负数，
    #   argmin 随之选错类别
    return np.array(colors, np.int32), np.array(targets, np.uint8)


def remap_rgb(lbl, colors, targets):
    """RGB 掩码 -> 目标类别索引图。
    先做精确匹配（绝大多数像素），剩余杂色再按最近颜色归类。"""
    arr = np.asarray(lbl, np.int32)
    h, w = arr.shape[:2]
    packed = (arr[..., 0] << 16) | (arr[..., 1] << 8) | arr[..., 2]

    idx = np.zeros((h, w), np.uint8)
    matched = np.zeros((h, w), bool)
    for color, target in zip(colors, targets):
        key = (int(color[0]) << 16) | (int(color[1]) << 8) | int(color[2])
        hit = packed == key
        idx[hit] = target
        matched |= hit

    leftover = ~matched
    if leftover.any():
        px = arr[leftover].astype(np.int32)                  # (M, 3)
        dist = ((px[:, None, :] - colors[None, :, :]) ** 2).sum(-1)
        idx[leftover] = targets[dist.argmin(1)]
    return idx, int(leftover.sum())


def convert(args):
    mapping, names = resolve(args.dataset, args.scheme)
    palette = load_palette(args.dataset, args.palette, args.labels)
    colors, targets = build_lut(palette, mapping, names)

    out_root = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    img_dst = os.path.join(out_root, "VOC2007", "JPEGImages")
    lbl_dst = os.path.join(out_root, "VOC2007", "SegmentationClass")
    split_dst = os.path.join(out_root, "VOC2007", "ImageSets", "Segmentation")
    for d in (img_dst, lbl_dst, split_dst):
        os.makedirs(d, exist_ok=True)

    labels = {os.path.splitext(f)[0]: f for f in os.listdir(args.labels)
              if f.lower().endswith(IMAGE_EXTS)}
    images = {os.path.splitext(f)[0]: f for f in os.listdir(args.images)
              if f.lower().endswith(IMAGE_EXTS)}
    stems = sorted(set(labels) & set(images))
    if not stems:
        raise SystemExit("图片与标签没有同名文件，请检查 --images / --labels 路径")
    if args.limit:
        stems = stems[:args.limit]
    print(f"配对成功 {len(stems)} 组（图片 {len(images)} / 标签 {len(labels)}）")

    tgt_palette = voc_palette(names)
    hist = np.zeros(len(names), np.int64)
    snapped_total = 0

    for i, stem in enumerate(stems, 1):
        img = Image.open(os.path.join(args.images, images[stem])).convert("RGB")
        lbl = Image.open(os.path.join(args.labels, labels[stem])).convert("RGB")
        if args.size:
            img = img.resize((args.size, args.size), Image.BICUBIC)
            lbl = lbl.resize((args.size, args.size), Image.NEAREST)
        elif lbl.size != img.size:
            lbl = lbl.resize(img.size, Image.NEAREST)

        idx, n_snapped = remap_rgb(lbl, colors, targets)
        snapped_total += n_snapped
        hist += np.bincount(idx.reshape(-1), minlength=len(names))

        img.save(os.path.join(img_dst, stem + ".png"))
        out = Image.fromarray(idx, mode="P")
        out.putpalette(tgt_palette)
        out.save(os.path.join(lbl_dst, stem + ".png"), optimize=True)
        if i % 100 == 0 or i == len(stems):
            print(f"  {i}/{len(stems)}")

    random.seed(args.seed)
    shuffled = stems[:]
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * args.val_percent))
    val, train = sorted(shuffled[:n_val]), sorted(shuffled[n_val:])
    for name, group in (("train", train), ("val", val), ("trainval", stems), ("test", [])):
        with open(os.path.join(split_dst, name + ".txt"), "w") as f:
            f.write("\n".join(group) + ("\n" if group else ""))

    print(f"\n完成 -> {out_root}")
    print(f"  train {len(train)} / val {len(val)}")
    if snapped_total:
        print(f"  非精确匹配（就近归类）像素: {snapped_total:,}"
              f"  —— 占比过高说明调色板不完整，请复查 --discover 的输出")
    print("  类别像素占比:")
    for i, n in enumerate(names):
        print(f"    {i} {n:<14}{hist[i] / hist.sum() * 100:7.2f}%")
    print(f"\n下一步（第一阶段预训练）:")
    print(f"  python tools/train_segformer.py --voc-root {args.out} "
          f"--num-classes {len(names)} --save-dir logs_sf_stage1 --epochs 60")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--discover", metavar="LABEL_DIR",
                    help="扫描标签目录的颜色分布并生成调色板模板")
    ap.add_argument("--images", help="源图片目录")
    ap.add_argument("--labels", help="源标签目录（RGB 掩码）")
    ap.add_argument("--dataset", default="ruralscapes",
                    choices=["ruralscapes", "uavid"])
    ap.add_argument("--scheme", default="base5", choices=["base5", "ext7"])
    ap.add_argument("--palette", help="颜色->源类别名 的 JSON（RuralScapes 必填）")
    ap.add_argument("--out", default="VOCdevkit_external", help="输出数据集根目录")
    ap.add_argument("--size", type=int, default=512,
                    help="统一缩放到 NxN，0 表示保持原尺寸（默认 512）")
    ap.add_argument("--val-percent", type=float, default=0.1)
    ap.add_argument("--limit", type=int, help="只转换前 N 组（先小样本验证流程）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.discover:
        discover(args.discover)
        return
    if not (args.images and args.labels):
        ap.error("需要 --images 与 --labels（或用 --discover 先扫描颜色）")
    convert(args)


if __name__ == "__main__":
    main()
