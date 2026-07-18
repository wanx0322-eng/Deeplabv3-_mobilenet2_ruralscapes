"""外部数据集 -> 本项目类别体系的映射表。

用途：领域内预训练。本项目只有 245 张训练图，直接从 ADE20K（通用场景）
微调跨度太大；先在 RuralScapes / UAVid 这类"乡村+航拍"数据上训练一轮，
再用本项目数据微调，通常比继续调超参有效得多。

两套目标体系：
  BASE_5   当前体系，与现有权重/标注完全兼容
  EXT_7    扩展体系，把"背景"垃圾桶拆成 farmland / water
           （背景类 IoU 长期只有 43，是拆分的主要动机）

映射按【源类别名】书写，与具体文件格式无关；颜色 -> 源类别名的对应关系
由各数据集的调色板负责（见 SOURCE_PALETTES / tools/import_external.py）。
"""

# --------------------------------------------------------------------------
# 目标类别体系
# --------------------------------------------------------------------------
BASE_5 = ["_background_", "building", "sky", "tree", "way"]

EXT_7 = ["_background_", "building", "sky", "tree", "way", "farmland", "water"]

# --------------------------------------------------------------------------
# RuralScapes（12 类，4K 无人机视频，欧洲乡村，50,835 精标帧）
#   区域类：forest land hill sky residential road water
#   物体类：person church haystack fence car
# --------------------------------------------------------------------------
RURALSCAPES_TO_BASE5 = {
    "forest":      "tree",
    "residential": "building",
    "church":      "building",       # 教堂是建筑，合并
    "sky":         "sky",
    "road":        "way",
    "land":        "_background_",
    "hill":        "_background_",
    "water":       "_background_",
    "person":      "_background_",   # 本体系无人物类
    "haystack":    "_background_",
    "fence":       "_background_",
    "car":         "_background_",
}

RURALSCAPES_TO_EXT7 = dict(RURALSCAPES_TO_BASE5, **{
    "land":  "farmland",             # 农田/耕地单独成类
    "water": "water",
})

# --------------------------------------------------------------------------
# UAVid（8 类，4096×2160，50m 斜视角）
#   注意：UAVid 几乎不含天空（俯视角），预训练后 sky 类收益有限
# --------------------------------------------------------------------------
UAVID_TO_BASE5 = {
    "building":           "building",
    "road":               "way",
    "tree":               "tree",
    "low vegetation":     "_background_",
    "static car":         "_background_",
    "moving car":         "_background_",
    "human":              "_background_",
    "background clutter": "_background_",
}

UAVID_TO_EXT7 = dict(UAVID_TO_BASE5, **{
    "low vegetation": "farmland",
})

# --------------------------------------------------------------------------
# 源数据集调色板：RGB -> 源类别名
#
# UAVid 的官方配色是公开且稳定的，直接内置。
# RuralScapes 的掩码配色未在文中列出，请用
#     python tools/import_external.py --discover <标签目录>
# 扫描实际颜色，再写进 JSON 传给 --palette（模板见 --discover 的输出）。
# --------------------------------------------------------------------------
SOURCE_PALETTES = {
    # UAVid 官方发布的彩色标签
    "uavid": {
        (0, 0, 0):       "background clutter",
        (128, 0, 0):     "building",
        (128, 64, 128):  "road",
        (192, 0, 192):   "static car",
        (0, 128, 0):     "tree",
        (128, 128, 0):   "low vegetation",
        (64, 64, 0):     "human",
        (64, 0, 128):    "moving car",
    },
    # 部分再分发版本（如 HuggingFace dronefreak/UAVid-2020）把标签预先转成了
    # 类别下标，存成 (i,i,i) 形式。下标顺序见其 data.yaml，与官方配色一一对应。
    "uavid_index": {
        (0, 0, 0):    "background clutter",
        (1, 1, 1):    "building",
        (2, 2, 2):    "road",
        (3, 3, 3):    "static car",
        (4, 4, 4):    "tree",
        (5, 5, 5):    "low vegetation",
        (6, 6, 6):    "human",
        (7, 7, 7):    "moving car",
    },
}

#   同一数据集的多种编码变体，供 import_external.py 自动识别
PALETTE_VARIANTS = {
    "uavid": ["uavid", "uavid_index"],
}

MAPPINGS = {
    ("ruralscapes", "base5"): RURALSCAPES_TO_BASE5,
    ("ruralscapes", "ext7"):  RURALSCAPES_TO_EXT7,
    ("uavid", "base5"):       UAVID_TO_BASE5,
    ("uavid", "ext7"):        UAVID_TO_EXT7,
}

TARGET_SCHEMES = {"base5": BASE_5, "ext7": EXT_7}


def resolve(dataset, scheme):
    """返回 (源类别名 -> 目标类别下标, 目标类别名列表)"""
    key = (dataset, scheme)
    if key not in MAPPINGS:
        raise KeyError(f"未知组合 {key}，可选 {sorted(MAPPINGS)}")
    names = TARGET_SCHEMES[scheme]
    name_to_idx = {n: i for i, n in enumerate(names)}
    return {src: name_to_idx[dst] for src, dst in MAPPINGS[key].items()}, names


def print_table(dataset, scheme):
    mapping, names = resolve(dataset, scheme)
    width = max(len(s) for s in mapping)
    print(f"{dataset} -> {scheme}  ({len(names)} 类: {', '.join(names)})")
    for src, idx in sorted(mapping.items(), key=lambda kv: (kv[1], kv[0])):
        print(f"  {src:<{width}}  ->  {idx}  {names[idx]}")


if __name__ == "__main__":
    import sys
    ds = sys.argv[1] if len(sys.argv) > 1 else "ruralscapes"
    sc = sys.argv[2] if len(sys.argv) > 2 else "base5"
    print_table(ds, sc)
