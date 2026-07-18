# -*- coding: utf-8 -*-
"""训练前环境自检：python check_env.py

依次检查 Python/依赖/CUDA/数据集/权值/端到端冒烟测试，最后给 PASS-FAIL 汇总。
尺寸与类别数从 workstation_config.json 读取，不写死。

来源：F 盘工程化分支的 check_env.py，按本项目实际情况改写
（256 输入、SegFormer 分支、新增的 pydantic/pytest 依赖、权值清单检查）。
"""
import os
import platform
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

results = []          # (name, ok, detail)


def check(name, fn):
    try:
        ok, detail = fn()
    except Exception as exc:
        ok, detail = False, "%s: %s" % (type(exc).__name__, exc)
    results.append((name, ok, detail))
    print("[%s] %s: %s" % ("PASS" if ok else "FAIL", name, detail))
    return ok


print("=" * 70)
print("Python %s  |  %s %s  |  %s"
      % (sys.version.split()[0], platform.system(), platform.release(),
         platform.machine()))
print("=" * 70)


# ---------- 1. 依赖包 ----------
def _imp(mod, attr="__version__"):
    def probe():
        module = __import__(mod)
        return True, str(getattr(module, attr, "?"))
    return probe


has_torch = check("torch", _imp("torch"))
check("torchvision", _imp("torchvision"))
check("cv2 (opencv)", _imp("cv2"))
check("numpy", _imp("numpy"))
check("PIL (pillow)", lambda: (True, __import__("PIL").__version__))
check("scipy", _imp("scipy"))
check("tqdm", _imp("tqdm"))
check("matplotlib", _imp("matplotlib"))
check("PySide6 (工作站)", _imp("PySide6"))
#   SegFormer 权重是当前系统默认，transformers 缺失会让预测/评估直接失败
check("transformers (SegFormer)", _imp("transformers"))
#   配置校验层与测试套件依赖
check("pydantic (配置校验)", _imp("pydantic"))
check("pytest (测试)", _imp("pytest"))


# ---------- 2. 配置 ----------
def config_check():
    from workstation.config import Config

    config = Config()
    errors = config.validate()
    detail = ("num_classes=%d, 训练输入=%s, 预测权重=%s"
              % (config.num_classes, config.train["input_shape"],
                 config.predict["model_path"]))
    if errors:
        return False, detail + "\n  配置问题: " + "; ".join(errors)
    return True, detail


check("workstation_config.json", config_check)


# ---------- 3. CUDA / GPU ----------
if has_torch:
    import torch

    def cuda_check():
        if not torch.cuda.is_available():
            return False, ("torch.cuda.is_available() == False"
                           "（装成 CPU 版 torch，或驱动/CUDA 不匹配）")
        count = torch.cuda.device_count()
        lines = ["%d 张 GPU" % count]
        total = 0.0
        blackwell = False
        for i in range(count):
            prop = torch.cuda.get_device_properties(i)
            gb = prop.total_memory / 1024 ** 3
            total += gb
            if prop.major >= 12:
                blackwell = True
            lines.append("  GPU%d: %s  %.1fGB  算力 sm_%d%d"
                         % (i, prop.name, gb, prop.major, prop.minor))
        detail = "\n" + "\n".join(lines) + "\n  合计显存 %.1fGB" % total
        try:
            arch_list = torch.cuda.get_arch_list()
        except Exception:
            arch_list = []
        if blackwell and not any("sm_120" in a for a in arch_list):
            detail += ("\n  [警告] 检测到 Blackwell 卡但 torch 支持架构=%s，"
                       "可能报 sm_120 not supported，请装 cu128 版 torch" % arch_list)
        return True, detail

    check("CUDA / 显卡", cuda_check)


# ---------- 4. 数据集 ----------
def dataset_check():
    """用 ruralscape_studio 的审计器，比单纯数文件严格得多。"""
    from ruralscape_studio.dataset import inspect_dataset
    from workstation.config import Config

    config = Config()
    profile = inspect_dataset(config.abs_path(config.dataset["voc_root"]))
    detail = ("图 %d / 掩膜 %d，划分 %s，类别值 %s"
              % (profile.total_images, profile.total_masks,
                 dict(profile.split_counts), list(profile.class_values)))
    if profile.issues:
        codes = sorted({issue.code for issue in profile.issues})
        return False, detail + "\n  %d 处问题: %s" % (len(profile.issues), codes)
    if profile.split_counts.get("train", 0) == 0:
        return False, detail + "\n  train.txt 为空，先跑 python voc_annotation.py"
    return True, detail


check("数据集 VOCdevkit", dataset_check)


# ---------- 5. 权值清单 ----------
def weights_check():
    from workstation.core.models import scan_weights

    weights = scan_weights()
    if not weights:
        return False, "没有找到任何 .pth/.onnx —— model_data 与 logs* 目录都是空的"
    lines = ["%d 个权值文件" % len(weights)]
    for item in weights[:6]:
        lines.append("  %s  %.1f MB  %s"
                     % (item["rel_path"], item["size_mb"], item["mtime_str"]))
    if len(weights) > 6:
        lines.append("  ... 其余 %d 个略" % (len(weights) - 6))
    return True, "\n" + "\n".join(lines)


check("权值清单", weights_check)


# ---------- 6. 端到端冒烟测试 ----------
if has_torch:
    import torch

    def smoke():
        from nets.deeplabv3_plus import DeepLab
        from workstation.config import Config

        config = Config()
        num_classes = config.num_classes
        size = int(config.train["input_shape"][0])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = DeepLab(num_classes=num_classes, backbone="mobilenet",
                      downsample_factor=config.train["downsample_factor"],
                      pretrained=False).to(device)
        net.train()
        images = torch.randn(2, 3, size, size, device=device)
        labels = torch.randint(0, num_classes, (2, size, size), device=device)
        use_amp = device.type == "cuda"
        if use_amp:
            torch.cuda.reset_peak_memory_stats()
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = net(images)
            loss = torch.nn.functional.cross_entropy(out, labels)
        loss.backward()
        expected = (2, num_classes, size, size)
        if tuple(out.shape) != expected:
            return False, "输出 shape=%s，期望 %s" % (tuple(out.shape), expected)
        detail = "输出 shape=%s, loss=%.4f" % (tuple(out.shape), loss.item())
        if use_amp:
            detail += (", 峰值显存 %.2fGB (batch=2, %dx%d)"
                       % (torch.cuda.max_memory_allocated() / 1024 ** 3, size, size))
        return True, detail

    check("DeepLab 前向/反向冒烟", smoke)


# ---------- 汇总 ----------
print("=" * 70)
n_pass = sum(1 for _, ok, _ in results if ok)
n_fail = len(results) - n_pass
print("汇总: %d PASS / %d FAIL" % (n_pass, n_fail))
if n_fail == 0:
    print("环境就绪：python run_workstation.py（Widgets）或 python run_studio.py（QML）")
else:
    print("未通过:", [name for name, ok, _ in results if not ok])
    sys.exit(1)
