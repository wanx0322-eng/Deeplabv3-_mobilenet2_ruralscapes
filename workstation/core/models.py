"""权值文件管理：扫描 model_data 与 logs 下的 .pth / .onnx"""
import os
import shutil
import time

from workstation.config import PROJECT_ROOT

WEIGHT_EXTS = (".pth", ".pt", ".onnx")


def weight_dirs():
    """model_data + 项目根目录下所有 logs 开头的目录（logs / logs_v2_B / logs_segformer_b0 …）"""
    dirs = ["model_data"]
    for name in sorted(os.listdir(PROJECT_ROOT)):
        if name.startswith("logs") and os.path.isdir(os.path.join(PROJECT_ROOT, name)):
            dirs.append(name)
    return dirs


def scan_weights():
    """返回 [{name, rel_path, abs_path, size_mb, mtime_str}]，按修改时间倒序"""
    found = []
    for d in weight_dirs():
        base = os.path.join(PROJECT_ROOT, d)
        if not os.path.isdir(base):
            continue
        for dirpath, _, filenames in os.walk(base):
            for fn in filenames:
                if os.path.splitext(fn)[1].lower() in WEIGHT_EXTS:
                    abs_path = os.path.join(dirpath, fn)
                    stat = os.stat(abs_path)
                    found.append({
                        "name": fn,
                        "rel_path": os.path.relpath(abs_path, PROJECT_ROOT).replace("\\", "/"),
                        "abs_path": abs_path,
                        "size_mb": stat.st_size / 1024 / 1024,
                        "mtime": stat.st_mtime,
                        "mtime_str": time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime)),
                    })
    found.sort(key=lambda item: item["mtime"], reverse=True)
    return found


def import_weight(src, target_dir="model_data"):
    dst_dir = os.path.join(PROJECT_ROOT, target_dir)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, os.path.basename(src))
    if os.path.abspath(src) == os.path.abspath(dst):
        return dst
    shutil.copy2(src, dst)
    return dst
