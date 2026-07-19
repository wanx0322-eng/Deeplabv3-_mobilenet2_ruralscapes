"""兼容层：引擎实现已移至顶层 segcore 包。

移动原因：utils/callbacks.py（低层）也需要同一份前向，而 utils -> workstation
是反向依赖。segcore 只依赖 utils / nets，各层都能安全引用。

新代码请直接 `from segcore import ...`。
"""
from segcore.engine import (  # noqa: F401
    IMAGENET_MEAN as _IMAGENET_MEAN,
    IMAGENET_STD as _IMAGENET_STD,
    SEGFORMER_HUB,
    SegEngine,
    build_net,
    compose_view,
    forward_prob,
    is_segformer,
    load_net,
    mask_statistics,
    predict_mask,
    tta_views,
)

__all__ = [
    "SEGFORMER_HUB",
    "SegEngine",
    "build_net",
    "compose_view",
    "forward_prob",
    "is_segformer",
    "load_net",
    "mask_statistics",
    "predict_mask",
    "tta_views",
]
