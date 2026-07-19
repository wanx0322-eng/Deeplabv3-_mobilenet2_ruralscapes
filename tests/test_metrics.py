"""utils/utils_metrics.py 的回归测试。

这个模块承载着本项目最贵的教训：曾经把 remove_classes 对应的行列从混淆矩阵里
删掉再算指标，把"背景误判成前景"(FP) 和"前景误判成背景"(FN) 整体丢弃，
前景 IoU 虚高约 15.7 个点。修复后 remove_classes 只影响求平均，矩阵永远完整。
在这些测试出现之前，防止它被改回去的只有一段注释。

下面的断言全部基于一个可以手算的 8 像素合成场景（3 类：bg / A / B）：

    gt   = [[0, 0, 1, 1],      pred = [[0, 1, 1, 1],
            [2, 2, 1, 1]]              [2, 0, 1, 1]]

    完整混淆矩阵（行=gt，列=pred）:
        [[1, 1, 0],     bg: 1 对, 1 错成 A
         [0, 4, 0],     A : 4 全对
         [1, 0, 1]]     B : 1 对, 1 错成 bg

    完整矩阵下:  IoU_bg = 1/3,  IoU_A = 4/5,  IoU_B = 1/2
    删掉 bg 行列: IoU_A = 1.0,  IoU_B = 1.0   <- 这就是当年虚高的来源
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from utils.utils_metrics import (
    compute_mIoU,
    fast_hist,
    keep_classes,
    mean_metric,
    per_Accuracy,
    per_class_iu,
    per_class_PA_Recall,
    per_class_Precision,
)

GT = np.array([[0, 0, 1, 1],
               [2, 2, 1, 1]], np.uint8)
PRED = np.array([[0, 1, 1, 1],
                 [2, 0, 1, 1]], np.uint8)
FULL_HIST = np.array([[1, 1, 0],
                      [0, 4, 0],
                      [1, 0, 1]], np.int64)


def test_fast_hist_builds_full_confusion_matrix():
    hist = fast_hist(GT.flatten(), PRED.flatten(), 3)
    assert hist.shape == (3, 3)
    np.testing.assert_array_equal(hist, FULL_HIST)
    #   全部 8 个像素都要在矩阵里，一个都不能丢
    assert hist.sum() == GT.size


def test_per_class_metrics_on_full_matrix():
    iou = per_class_iu(FULL_HIST)
    np.testing.assert_allclose(iou, [1 / 3, 4 / 5, 1 / 2])
    np.testing.assert_allclose(per_class_PA_Recall(FULL_HIST), [1 / 2, 1.0, 1 / 2])
    np.testing.assert_allclose(per_class_Precision(FULL_HIST), [1 / 2, 4 / 5, 1.0])
    assert per_Accuracy(FULL_HIST) == pytest.approx(6 / 8)


def test_removing_background_rows_would_inflate_foreground_iou():
    """把当年的错误算法演示一遍：删行列后前景 IoU 从 0.65 涨到 1.0。

    这个测试存在的意义是给下一个想"简化"矩阵的人看：两种算法的差距
    不是舍入误差，而是把跨背景的错误整体丢掉造成的系统性虚高。
    """
    deleted = np.delete(np.delete(FULL_HIST, [0], axis=0), [0], axis=1)
    inflated = per_class_iu(deleted)
    honest = per_class_iu(FULL_HIST)[1:]
    #   删行列后 A、B 都变成满分 —— 错误被"洗掉"了
    np.testing.assert_allclose(inflated, [1.0, 1.0])
    assert float(np.mean(inflated)) > float(np.mean(honest)) + 0.3


def test_keep_classes_and_mean_metric_only_affect_the_average():
    assert keep_classes(5, [0]) == [1, 2, 3, 4]
    assert keep_classes(3, None) == [0, 1, 2]
    assert keep_classes(3, []) == [0, 1, 2]

    iou = per_class_iu(FULL_HIST)
    #   前景平均 = (0.8 + 0.5) / 2，背景的 1/3 不参与
    assert mean_metric(iou, 3, [0]) == pytest.approx(0.65)
    #   不移除时是三类的普通平均
    assert mean_metric(iou, 3, None) == pytest.approx((1 / 3 + 4 / 5 + 1 / 2) / 3)


def test_mean_metric_ignores_nan_classes():
    #   某个前景类在 gt 和 pred 中都不出现时，别让 NaN 污染平均
    values = np.array([0.4, 0.8, np.nan])
    assert mean_metric(values, 3, [0]) == pytest.approx(0.8)


def test_compute_miou_end_to_end_keeps_matrix_complete(tmp_path):
    """文件级端到端：写出 png，验证返回的矩阵是完整的、指标按完整矩阵算。"""
    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    gt_dir.mkdir()
    pred_dir.mkdir()
    #   拆成两张 1x4 的图，顺带覆盖多图累加路径
    for i in range(2):
        Image.fromarray(GT[i:i + 1]).save(gt_dir / f"img{i}.png")
        Image.fromarray(PRED[i:i + 1]).save(pred_dir / f"img{i}.png")

    hist, ious, pa_recall, precision = compute_mIoU(
        str(gt_dir), str(pred_dir), ["img0", "img1"], 3,
        ["bg", "A", "B"], remove_classes=[0])

    #   核心断言：remove_classes 传了 [0]，矩阵仍然是完整的 3x3，
    #   背景行列原封不动 —— 它只影响 mean_metric 的平均口径。
    assert hist.shape == (3, 3)
    np.testing.assert_array_equal(hist, FULL_HIST)
    np.testing.assert_allclose(ious, [1 / 3, 4 / 5, 1 / 2])
    assert mean_metric(ious, 3, [0]) == pytest.approx(0.65)


def test_compute_miou_skips_size_mismatched_pairs(tmp_path):
    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    gt_dir.mkdir()
    pred_dir.mkdir()
    Image.fromarray(GT).save(gt_dir / "ok.png")
    Image.fromarray(PRED).save(pred_dir / "ok.png")
    #   尺寸不一致的对子应被跳过而不是崩溃或污染矩阵
    Image.fromarray(GT).save(gt_dir / "bad.png")
    Image.fromarray(PRED[:, :2]).save(pred_dir / "bad.png")

    hist, _, _, _ = compute_mIoU(
        str(gt_dir), str(pred_dir), ["ok", "bad"], 3, None, [0])
    np.testing.assert_array_equal(hist, FULL_HIST)
