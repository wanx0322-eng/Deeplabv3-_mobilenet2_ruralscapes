"""把数据集里的 .PNG 扩展名统一成小写 .png。

代码各处都是按 name + ".png" 拼路径的。Windows 文件系统不区分大小写，
所以 .PNG 文件也能被打开；但在 Linux / macOS 上会直接 FileNotFoundError。
这里做一次性归一化，消除这个移植性隐患。

Windows 上仅改大小写的重命名需要经过一个临时名，否则会被当成同名文件。

用法: python tools/normalize_ext.py [--dry-run]
"""
import argparse
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIRS = [
    os.path.join(ROOT, "VOCdevkit", "VOC2007", "JPEGImages"),
    os.path.join(ROOT, "VOCdevkit", "VOC2007", "SegmentationClass"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    total = 0
    for d in DIRS:
        if not os.path.isdir(d):
            continue
        targets = []
        for fn in os.listdir(d):
            stem, ext = os.path.splitext(fn)
            if ext and ext != ext.lower():
                targets.append((fn, stem + ext.lower()))

        print(f"{os.path.relpath(d, ROOT)}: 需要归一化 {len(targets)} 个")
        for old, new in targets:
            if args.dry_run:
                continue
            src = os.path.join(d, old)
            dst = os.path.join(d, new)
            tmp = os.path.join(d, old + ".tmp_rename")
            os.rename(src, tmp)      # 两步，绕开 Windows 的大小写不敏感
            os.rename(tmp, dst)
        total += len(targets)

    if args.dry_run:
        print(f"\n[dry-run] 共 {total} 个文件待重命名，未改动。")
    else:
        print(f"\n已重命名 {total} 个文件为小写扩展名。")


if __name__ == "__main__":
    main()
