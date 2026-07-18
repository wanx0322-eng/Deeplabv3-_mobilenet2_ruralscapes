"""标签导出：索引 PNG / 调色板 PNG / 彩色 PNG / 叠加 PNG / LabelMe JSON / YOLO-seg TXT"""
import json
import os

import cv2
import numpy as np
from PIL import Image

FORMATS = [
    ("index_png", "索引 PNG（像素值=类别号，VOC 训练格式）"),
    ("palette_png", "调色板 PNG（索引值+颜色显示，推荐）"),
    ("color_png", "彩色 PNG（纯 RGB 着色图）"),
    ("overlay_png", "叠加 PNG（原图+半透明标签）"),
    ("labelme_json", "LabelMe JSON（多边形，可再编辑）"),
    ("yolo_txt", "YOLO-seg TXT（归一化多边形）"),
]


def make_palette(class_colors):
    palette = np.zeros((256, 3), np.uint8)
    for i, c in enumerate(class_colors[:256]):
        palette[i] = c
    return palette


def save_palette_png(mask, class_colors, path):
    """P 模式 PNG：像素值仍是类别索引，但自带调色板可直接查看"""
    img = Image.fromarray(mask, mode="P")
    img.putpalette(make_palette(class_colors).reshape(-1).tolist())
    img.save(path)


def save_index_png(mask, path):
    Image.fromarray(mask, mode="L").save(path)


def save_color_png(mask, class_colors, path):
    Image.fromarray(make_palette(class_colors)[mask]).save(path)


def save_overlay_png(mask, class_colors, image, path, alpha=0.6):
    seg = Image.fromarray(make_palette(class_colors)[mask])
    Image.blend(image.convert("RGB"), seg, alpha).save(path)


def mask_to_polygons(mask, class_index, epsilon_ratio=0.002, min_area=16.0):
    """提取某一类的多边形轮廓（外轮廓，近似简化）"""
    binary = (mask == class_index).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        epsilon = epsilon_ratio * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) >= 3:
            polygons.append(approx.reshape(-1, 2).astype(float))
    return polygons


def save_labelme_json(mask, class_names, image_name, path):
    h, w = mask.shape
    shapes = []
    for ci in range(1, len(class_names)):        # 跳过背景
        for poly in mask_to_polygons(mask, ci):
            shapes.append({
                "label": class_names[ci],
                "points": [[float(x), float(y)] for x, y in poly],
                "group_id": None,
                "shape_type": "polygon",
                "flags": {},
            })
    data = {
        "version": "5.3.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_yolo_txt(mask, num_classes, path):
    """YOLO 分割格式：每行 `class_id x1 y1 x2 y2 ...`（归一化，前景类 id 从 0 起）"""
    h, w = mask.shape
    lines = []
    for ci in range(1, num_classes):
        for poly in mask_to_polygons(mask, ci):
            coords = []
            for x, y in poly:
                coords.append(f"{x / w:.6f}")
                coords.append(f"{y / h:.6f}")
            lines.append(f"{ci - 1} " + " ".join(coords))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def export_one(stem, mask, image_path, out_dir, formats, class_names,
               class_colors, alpha=0.6):
    """按所选格式导出一张标签，各格式放各自子目录。返回输出文件数"""
    count = 0
    image = None
    for fmt in formats:
        sub = os.path.join(out_dir, fmt)
        os.makedirs(sub, exist_ok=True)
        if fmt == "index_png":
            save_index_png(mask, os.path.join(sub, stem + ".png"))
        elif fmt == "palette_png":
            save_palette_png(mask, class_colors, os.path.join(sub, stem + ".png"))
        elif fmt == "color_png":
            save_color_png(mask, class_colors, os.path.join(sub, stem + ".png"))
        elif fmt == "overlay_png":
            if image is None:
                image = Image.open(image_path)
            save_overlay_png(mask, class_colors, image,
                             os.path.join(sub, stem + ".png"), alpha)
        elif fmt == "labelme_json":
            save_labelme_json(mask, class_names,
                              os.path.basename(image_path),
                              os.path.join(sub, stem + ".json"))
        elif fmt == "yolo_txt":
            save_yolo_txt(mask, len(class_names),
                          os.path.join(sub, stem + ".txt"))
            classes_txt = os.path.join(sub, "classes.txt")
            if not os.path.exists(classes_txt):
                with open(classes_txt, "w", encoding="utf-8") as f:
                    f.write("\n".join(class_names[1:]))
        count += 1
    return count
