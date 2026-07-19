"""segcore 公共引擎的测试。

重点不是"前向算得对不对"（那要靠真实权重的数值对拍，见 README 模型库表格），
而是锁住这次重构的三个成果：
  1. 前向实现全项目只剩一份 —— 谁再复制一份，这里会红；
  2. 分层正确：segcore 不依赖 workstation（否则 utils/callbacks 会形成反向依赖）；
  3. SegFormer 的 ImageNet 归一化由引擎统一处理 —— 这一行以前散在 4 个文件里，
     漏写不报错，只会让 mIoU 无声下降。
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from segcore import evaluate_hist, is_segformer, per_image_hists, read_split, tta_views
from segcore.engine import IMAGENET_MEAN, IMAGENET_STD, SEGFORMER_HUB

ROOT = Path(__file__).resolve().parents[1]


# ---------------- 结构：拷贝数量 ----------------

#   前向流水线的特征签名：softmax 作用在 permute(1,2,0) 之后
FORWARD_SIGNATURE = re.compile(r"F\.softmax\([^)]*permute\(1,\s*2,\s*0\)")


def _project_py_files():
    skip = {".venv", "__pycache__", ".git", "tests", "ruralscape_studio"}
    for path in ROOT.rglob("*.py"):
        if not any(part in skip for part in path.parts):
            yield path


def test_forward_pass_has_exactly_one_implementation():
    """历史上这条流水线有 8 份拷贝，'SegFormer 要归一化'漏写 1 处就无声掉点。"""
    hits = [p.relative_to(ROOT).as_posix() for p in _project_py_files()
            if FORWARD_SIGNATURE.search(p.read_text(encoding="utf-8"))]
    assert hits == ["segcore/engine.py"], (
        "前向实现应当只在 segcore/engine.py 里存在一份，实际出现在：%s" % hits)


def test_imagenet_normalisation_constants_live_only_in_segcore():
    """归一化常数也只应有一处定义（train_segformer 训练侧的 tensor 版除外）。"""
    pattern = re.compile(r"0\.485,\s*0\.456,\s*0\.406")
    hits = [p.relative_to(ROOT).as_posix() for p in _project_py_files()
            if pattern.search(p.read_text(encoding="utf-8"))]
    assert set(hits) <= {"segcore/engine.py", "tools/train_segformer.py"}, (
        "ImageNet 归一化常数散落到了额外的文件：%s" % hits)


# ---------------- 分层与惰性导入 ----------------

def test_segcore_does_not_depend_on_workstation():
    """utils/callbacks.py 要用 segcore，若 segcore 反过来依赖 workstation 就成环。"""
    result = subprocess.run(
        [sys.executable, "-c",
         "import segcore, sys; "
         "print(any(m.startswith('workstation') for m in sys.modules))"],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_importing_segcore_does_not_load_torch():
    """torch 惰性导入，界面/CLI 的启动阶段不该为此付几秒。"""
    result = subprocess.run(
        [sys.executable, "-c", "import segcore, sys; print('torch' in sys.modules)"],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_legacy_engine_import_path_still_works():
    """workstation.core.engine 是兼容层，旧引用不能断。"""
    from workstation.core import engine as legacy

    assert legacy.SegEngine is not None
    assert legacy.is_segformer("segformer-b2") is True
    #   miou_worker 曾用私有名引用归一化常数
    np.testing.assert_array_equal(legacy._IMAGENET_MEAN, IMAGENET_MEAN)
    np.testing.assert_array_equal(legacy._IMAGENET_STD, IMAGENET_STD)


# ---------------- 引擎的纯函数部分 ----------------

def test_is_segformer_routing():
    assert is_segformer("segformer-b0") is True
    assert is_segformer("segformer-b2") is True
    assert is_segformer("mobilenet") is False
    assert is_segformer("xception") is False


def test_segformer_hub_covers_every_supported_model():
    assert set(SEGFORMER_HUB) == {"segformer-b0", "segformer-b1", "segformer-b2"}
    for hub in SEGFORMER_HUB.values():
        assert hub.startswith("nvidia/segformer-")


def test_tta_views_shape():
    assert tta_views([256, 256], tta=False) == [((256, 256), False)]
    views = tta_views([256, 256], tta=True)
    #   3 个尺度 × 翻转与否
    assert len(views) == 6
    assert {flip for _shape, flip in views} == {False, True}
    scales = sorted({shape[0] for shape, _flip in views})
    assert scales == [192, 256, 320]
    assert all(s % 32 == 0 for s in scales)


def test_normalisation_constants_are_float32():
    #   float64 会让 preprocess_input 的结果被提升类型，多一次隐式拷贝
    assert IMAGENET_MEAN.dtype == np.float32
    assert IMAGENET_STD.dtype == np.float32


# ---------------- 评估协议 ----------------

class _ConstantNet:
    """总是预测同一张类别图的假网络，用于验证评估协议本身。"""

    training = False

    def __init__(self, pred, num_classes):
        self.pred = pred
        self.num_classes = num_classes

    def __call__(self, x):
        import torch

        n, _c, h, w = x.shape
        logits = torch.zeros(n, self.num_classes, h, w)
        target = torch.from_numpy(np.asarray(self.pred)).long()
        target = target[None].expand(n, -1, -1)
        logits.scatter_(1, target[:, None], 10.0)
        return logits

    def eval(self):
        return self

    def train(self):
        return self


@pytest.fixture
def tiny_voc(tmp_path):
    """两张 4x4 的合成图，gt 与预测各半对错。"""
    voc = tmp_path / "VOC2007"
    (voc / "JPEGImages").mkdir(parents=True)
    (voc / "SegmentationClass").mkdir()
    (voc / "ImageSets" / "Segmentation").mkdir(parents=True)
    gt = np.array([[0, 0, 1, 1]] * 4, np.uint8)
    for stem in ("a", "b"):
        Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(
            voc / "JPEGImages" / f"{stem}.png")
        Image.fromarray(gt).save(voc / "SegmentationClass" / f"{stem}.png")
    (voc / "ImageSets" / "Segmentation" / "val.txt").write_text("a\nb\n")
    return tmp_path


def test_read_split(tiny_voc):
    assert read_split(str(tiny_voc), "val") == ["a", "b"]


def test_per_image_hists_returns_one_matrix_per_image(tiny_voc):
    import torch

    #   全预测为类别 1：gt 里一半是 0 一半是 1
    net = _ConstantNet(np.ones((4, 4), np.int64), 2)
    hists = per_image_hists(net, ["a", "b"], str(tiny_voc), 2, [4, 4],
                            torch.device("cpu"), backbone="mobilenet")
    assert hists.shape == (2, 2, 2)
    #   每张图 16 像素：8 个 gt=0 全被判成 1，8 个 gt=1 判对
    expected = np.array([[0, 8], [0, 8]], np.int64)
    np.testing.assert_array_equal(hists[0], expected)
    np.testing.assert_array_equal(hists[1], expected)
    #   逐图返回是为了让配对 bootstrap 能按图重采样
    np.testing.assert_array_equal(hists.sum(0), expected * 2)


def test_per_image_hists_restores_training_mode(tiny_voc):
    """训练循环里调用评估后必须回到 train 模式，否则 BN 行为会错。"""
    import torch

    net = _ConstantNet(np.ones((4, 4), np.int64), 2)
    net.training = True
    per_image_hists(net, ["a"], str(tiny_voc), 2, [4, 4],
                    torch.device("cpu"), backbone="mobilenet")
    assert net.training is True


def test_evaluate_hist_matches_metric_convention():
    #   与 tests/test_metrics.py 用同一个可手算矩阵
    hist = np.array([[1, 1, 0], [0, 4, 0], [1, 0, 1]], np.int64)
    m = evaluate_hist(hist, 3, remove_classes=(0,))
    assert m["miou"] == pytest.approx(65.0)         # (0.8 + 0.5) / 2
    assert m["accuracy"] == pytest.approx(75.0)     # 6/8
    np.testing.assert_allclose(m["iou"], [1 / 3, 4 / 5, 1 / 2])
    #   不移除时是三类平均
    assert evaluate_hist(hist, 3, remove_classes=None)["miou"] == pytest.approx(
        (1 / 3 + 4 / 5 + 1 / 2) / 3 * 100)
