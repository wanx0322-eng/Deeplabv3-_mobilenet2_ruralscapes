"""VOC 数据集管理：条目枚举、导入、删除、划分、校验"""
import os
import random
import shutil
import time

import numpy as np
from PIL import Image

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


class DatasetEntry:
    __slots__ = ("stem", "image_path", "label_path", "split")

    def __init__(self, stem, image_path, label_path, split):
        self.stem = stem
        self.image_path = image_path
        self.label_path = label_path
        self.split = split  # train / val / test / 未划分


class DatasetManager:
    """封装对 VOCdevkit/VOC2007 目录的全部操作"""

    def __init__(self, voc2007_dir):
        self.root = voc2007_dir

    @property
    def image_dir(self):
        return os.path.join(self.root, "JPEGImages")

    @property
    def label_dir(self):
        return os.path.join(self.root, "SegmentationClass")

    @property
    def split_dir(self):
        return os.path.join(self.root, "ImageSets", "Segmentation")

    def ensure_dirs(self):
        for d in (self.image_dir, self.label_dir, self.split_dir):
            os.makedirs(d, exist_ok=True)

    # ---------------- 枚举 ----------------
    def read_split(self, name):
        path = os.path.join(self.split_dir, name + ".txt")
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def write_split(self, name, stems):
        self.ensure_dirs()
        path = os.path.join(self.split_dir, name + ".txt")
        with open(path, "w", encoding="utf-8") as f:
            for stem in stems:
                f.write(stem + "\n")

    def list_entries(self):
        """列出所有 图片-标签 条目及其所属划分"""
        self.ensure_dirs()
        split_of = {}
        for split in ("train", "val", "test"):
            for stem in self.read_split(split):
                split_of[stem] = split

        images = {}
        for fn in sorted(os.listdir(self.image_dir)):
            stem, ext = os.path.splitext(fn)
            if ext.lower() in IMAGE_EXTS:
                images[stem] = os.path.join(self.image_dir, fn)

        entries = []
        for stem, img_path in images.items():
            label_path = os.path.join(self.label_dir, stem + ".png")
            if not os.path.exists(label_path):
                label_path = None
            entries.append(DatasetEntry(stem, img_path, label_path,
                                        split_of.get(stem, "未划分")))
        return entries

    def counts(self):
        entries = self.list_entries()
        result = {"total": len(entries), "train": 0, "val": 0, "test": 0,
                  "未划分": 0, "no_label": 0}
        for e in entries:
            result[e.split] = result.get(e.split, 0) + 1
            if e.label_path is None:
                result["no_label"] += 1
        return result

    # ---------------- 导入 ----------------
    def import_pairs(self, image_files, label_dir=None):
        """导入图片（可选：从 label_dir 中按同名 stem 匹配标签）。
        返回 (导入数, 匹配到标签数, 跳过列表)"""
        self.ensure_dirs()
        imported, labeled, skipped = 0, 0, []
        for src in image_files:
            stem, ext = os.path.splitext(os.path.basename(src))
            if ext.lower() not in IMAGE_EXTS:
                skipped.append((src, "不支持的图片格式"))
                continue
            dst = os.path.join(self.image_dir, stem + ext.lower())
            if os.path.exists(dst):
                skipped.append((src, "同名图片已存在"))
                continue
            shutil.copy2(src, dst)
            imported += 1
            if label_dir:
                for label_ext in (".png",):
                    label_src = os.path.join(label_dir, stem + label_ext)
                    if os.path.exists(label_src):
                        shutil.copy2(label_src,
                                     os.path.join(self.label_dir, stem + ".png"))
                        labeled += 1
                        break
        return imported, labeled, skipped

    def import_labels(self, label_files):
        """单独导入标签 png，返回 (导入数, 跳过列表)"""
        self.ensure_dirs()
        imported, skipped = 0, []
        for src in label_files:
            stem, ext = os.path.splitext(os.path.basename(src))
            if ext.lower() != ".png":
                skipped.append((src, "标签必须为 png"))
                continue
            shutil.copy2(src, os.path.join(self.label_dir, stem + ".png"))
            imported += 1
        return imported, skipped

    # ---------------- 删除 ----------------
    def delete_entries(self, stems):
        """把选中条目移入回收目录（不直接删除），并从划分中移除"""
        trash = os.path.join(os.path.dirname(self.root), "_trash",
                             time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(trash, exist_ok=True)
        stems = set(stems)
        for e in self.list_entries():
            if e.stem not in stems:
                continue
            shutil.move(e.image_path, os.path.join(trash, os.path.basename(e.image_path)))
            if e.label_path and os.path.exists(e.label_path):
                shutil.move(e.label_path, os.path.join(trash, "label_" + os.path.basename(e.label_path)))
        for split in ("train", "val", "test", "trainval"):
            remain = [s for s in self.read_split(split) if s not in stems]
            self.write_split(split, remain)
        return trash

    # ---------------- 划分 ----------------
    def is_index_label(self, path):
        """标签是否为可用的索引图（P / L 模式）。RGB 标签进训练集会崩。"""
        try:
            with Image.open(path) as im:
                return im.mode in ("P", "L")
        except Exception:
            return False

    def random_split(self, trainval_percent=1.0, train_percent=0.7, seed=0):
        """按 voc_annotation.py 的逻辑重新随机划分。
        只纳入【有标签且标签是索引图】的条目 —— RGB 三通道标签会让训练在
        one-hot 处直接崩溃，必须先用 tools/fix_rgb_masks.py 转换。"""
        entries = [e for e in self.list_entries()
                   if e.label_path and self.is_index_label(e.label_path)]
        stems = sorted(e.stem for e in entries)
        random.seed(seed)
        num = len(stems)
        tv = int(num * trainval_percent)
        tr = int(tv * train_percent)
        trainval_idx = set(random.sample(range(num), tv))
        train_idx = set(random.sample(sorted(trainval_idx), tr))

        train, val, test, trainval = [], [], [], []
        for i, stem in enumerate(stems):
            if i in trainval_idx:
                trainval.append(stem)
                if i in train_idx:
                    train.append(stem)
                else:
                    val.append(stem)
            else:
                test.append(stem)
        self.write_split("trainval", trainval)
        self.write_split("train", train)
        self.write_split("val", val)
        self.write_split("test", test)
        return len(train), len(val), len(test)

    def assign_split(self, stems, split):
        """手动把条目划到 train/val/test"""
        stems = set(stems)
        current = {name: [s for s in self.read_split(name) if s not in stems]
                   for name in ("train", "val", "test")}
        if split in current:
            current[split].extend(sorted(stems))
        for name, values in current.items():
            self.write_split(name, values)
        self.write_split("trainval", current["train"] + current["val"])

    # ---------------- 校验 ----------------
    def check_labels(self, num_classes, progress_cb=None):
        """统计标签像素值分布，返回 (counts[256], 问题列表)"""
        entries = [e for e in self.list_entries() if e.label_path]
        counts = np.zeros(256, np.int64)
        problems = []
        for i, e in enumerate(entries):
            try:
                png = np.array(Image.open(e.label_path), np.uint8)
            except Exception as exc:
                problems.append(f"{e.stem}: 标签无法读取 ({exc})")
                continue
            if png.ndim > 2:
                problems.append(f"{e.stem}: 标签不是灰度/8位彩图 shape={png.shape}")
                continue
            counts += np.bincount(png.reshape(-1), minlength=256)
            if progress_cb:
                progress_cb(i + 1, len(entries))
        used = np.nonzero(counts)[0]
        invalid = [int(v) for v in used if v >= num_classes and v != 255]
        if invalid:
            problems.append(f"存在超出类别数({num_classes})的像素值: {invalid}")
        if counts[255] > 0 and counts[0] > 0 and counts[1:255].sum() == 0:
            problems.append("标签只含 0 和 255，二分类应将目标像素改为 1")
        return counts, problems
