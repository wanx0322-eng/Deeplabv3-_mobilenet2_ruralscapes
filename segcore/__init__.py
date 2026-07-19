"""语义分割的公共引擎层：模型构建 + 前向 + 评估协议。

存在的理由：这条流水线
    letterbox -> 前向 -> softmax -> 去灰条 -> resize 回原尺寸 -> argmax
以前在项目里有 8 份各自独立的拷贝（deeplab.py 三份、EvalCallback、
SegEngine、miou_worker、compare_models、train_segformer、paired_bootstrap）。
其中"SegFormer 输入必须做 ImageNet 归一化"这个约定散布在 4 处，
漏写不会报错，只会让 mIoU 无声地低几个点；而 compare_models / get_miou.py
因为各自持有 DeepLab-only 的前向，压根没法评估系统默认的 SegFormer 权重。

现在所有消费方都调用这里。新增引擎只改这一个包。

分层：segcore 依赖 utils / nets（低层），不依赖 workstation（GUI 层）。
torch、cv2、transformers 全部惰性导入，import segcore 本身很轻。
"""

from .engine import (
    SEGFORMER_HUB,
    SegEngine,
    build_net,
    compose_view,
    forward_prob,
    is_segformer,
    load_net,
    mask_statistics,
    predict_mask,
    predict_prob,
    tta_views,
)
from .evaluate import evaluate_hist, per_image_hists, read_split

__all__ = [
    "SEGFORMER_HUB",
    "SegEngine",
    "build_net",
    "compose_view",
    "evaluate_hist",
    "forward_prob",
    "is_segformer",
    "load_net",
    "mask_statistics",
    "per_image_hists",
    "predict_mask",
    "predict_prob",
    "read_split",
    "tta_views",
]
