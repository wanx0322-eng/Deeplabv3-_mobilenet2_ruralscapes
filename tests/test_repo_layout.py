"""仓库布局金丝雀。

已两次观察到 logs_* 权重目录被【外部因素】整体移动进 logs/ 里
（2026-07-18 20:50 与 2026-07-19 07:49，均非项目代码所为——代码里
不存在会移动这些目录的逻辑）。这会打断 workstation_config.json 里
的 model_path 与 README 的模型库路径。

这个测试的作用：目录再被移动时，测试套件第一时间变红，
而不是等到加载权重时才发现。若你是有意重排目录，请同步更新
workstation_config.json、README 模型库表格与本测试。
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#   README「模型库」表格里承诺的权重路径（LFS 跟踪）
EXPECTED_WEIGHTS = (
    "model_data/deeplab_mobilenetv2.pth",
    "logs/best_epoch_weights.pth",
    "logs_v2_B/best_epoch_weights.pth",
    "logs_v2_E/best_epoch_weights.pth",
    "logs_segformer_b0/best_segformer.pth",
    "logs_segformer_b2/best_segformer.pth",
    "logs_sf_ade/best_segformer.pth",
    "logs_sf_city/best_segformer.pth",
)


def test_weight_files_are_at_their_documented_paths():
    missing = [rel for rel in EXPECTED_WEIGHTS if not (ROOT / rel).is_file()]
    assert not missing, (
        "以下权重不在文档承诺的路径上（历史上发生过 logs_* 目录被外部"
        "移动进 logs/ 的事件，先检查是不是又发生了）：%s" % missing)


def test_weight_dirs_are_not_nested_inside_logs():
    nested = sorted(p.name for p in (ROOT / "logs").iterdir()
                    if p.is_dir() and p.name.startswith("logs"))
    assert nested == [], (
        "logs/ 里出现了嵌套的权重目录 %s —— 又被移动了，请移回顶层" % nested)


def test_configured_model_path_exists():
    config = json.loads((ROOT / "workstation_config.json").read_text("utf-8"))
    model_path = config.get("predict", {}).get("model_path")
    if model_path:
        assert (ROOT / model_path).is_file(), (
            "workstation_config.json 的 predict.model_path 指向不存在的文件: %s"
            % model_path)
