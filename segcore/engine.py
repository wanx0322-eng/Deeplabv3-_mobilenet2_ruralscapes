"""模型构建与前向推理 —— 全项目唯一一份实现。

前向协议（与历史各处拷贝逐位一致，不要随意改动）:
    cvtColor -> letterbox 到 input_shape（灰条填充 128）
    -> /255（SegFormer 额外做 ImageNet 归一化）
    -> 网络前向 -> softmax
    -> 裁掉灰条 -> cv2 双线性 resize 回原尺寸 -> argmax

注意 resize 的是【概率图】而不是类别图。先 argmax 再 resize 会在类别边界
产生锯齿，指标会变。历史上所有拷贝都是先 resize 后 argmax，保持不变。
"""
import os

import numpy as np
from PIL import Image

#   backbone 字符串 -> HuggingFace 模型 id。"segformer-b0/b1/b2" 走 SegFormer，
#   其余（mobilenet / xception）走 DeepLabV3+。
SEGFORMER_HUB = {
    "segformer-b0": "nvidia/segformer-b0-finetuned-ade-512-512",
    "segformer-b1": "nvidia/segformer-b1-finetuned-ade-512-512",
    "segformer-b2": "nvidia/segformer-b2-finetuned-ade-512-512",
}
#   SegFormer 预训练主干要求 ImageNet 归一化；DeepLab 分支只做 /255。
#   这一行以前散落在 4 个文件里，漏写不报错、只是指标无声下降。
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def is_segformer(backbone):
    return str(backbone).lower().startswith("segformer")


def build_net(backbone, num_classes, downsample_factor=8, pretrained=False):
    """按 backbone 构建【未加载权值】的网络。"""
    if is_segformer(backbone):
        try:
            from transformers import SegformerForSemanticSegmentation
        except ImportError as exc:
            raise RuntimeError(
                "使用 SegFormer 需要安装 transformers：pip install transformers") from exc
        hub = SEGFORMER_HUB.get(str(backbone).lower())
        if hub is None:
            raise RuntimeError("未知的 SegFormer 型号: %s，可选 %s"
                               % (backbone, sorted(SEGFORMER_HUB)))
        return SegformerForSemanticSegmentation.from_pretrained(
            hub, num_labels=num_classes, ignore_mismatched_sizes=True)

    from nets.deeplabv3_plus import DeepLab
    return DeepLab(num_classes=num_classes, backbone=backbone,
                   downsample_factor=downsample_factor, pretrained=pretrained)


def load_net(model_path, backbone, num_classes, downsample_factor=8, device=None):
    """构建网络并载入权值，返回 eval 模式的网络。"""
    import torch

    if not os.path.exists(model_path):
        raise FileNotFoundError("权值文件不存在: %s" % model_path)
    net = build_net(backbone, num_classes, downsample_factor)
    state = torch.load(model_path, map_location="cpu")
    try:
        net.load_state_dict(state)
    except RuntimeError as exc:
        raise RuntimeError(
            "权值与当前配置不匹配（请检查 类别数/主干网络/下采样倍数 是否与训练时一致，"
            "SegFormer 权重需选择对应的 segformer-* 主干）\n\n" + str(exc)[:800]) from exc
    net = net.eval()
    if device is not None:
        net = net.to(device)
    return net


def tta_views(input_shape, tta=False):
    """返回 [(shape, flip), ...]。tta 时做水平翻转 × 多尺度 0.75/1.0/1.25。"""
    if not tta:
        return [(tuple(input_shape), False)]
    base = input_shape[0]
    scales = sorted({max(64, int(round(base * s / 32)) * 32)
                     for s in (0.75, 1.0, 1.25)})
    return [((s, s), flip) for s in scales for flip in (False, True)]


def forward_prob(net, image, shape, device, segformer=False, flip=False):
    """单次前向，返回【原图尺寸】的概率图 (H, W, C)。

    image 需为 PIL RGB 图（调用方先过 cvtColor）。
    """
    import cv2
    import torch
    import torch.nn.functional as F

    from utils.utils import preprocess_input, resize_image

    orig_w, orig_h = image.size
    src = image.transpose(Image.FLIP_LEFT_RIGHT) if flip else image
    data, nw, nh = resize_image(src, (shape[1], shape[0]))
    #   preprocess_input 是原地除法，必须传入新建的 float32 数组
    arr = preprocess_input(np.array(data, np.float32))
    if segformer:
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.expand_dims(np.transpose(arr, (2, 0, 1)), 0)

    with torch.no_grad():
        images = torch.from_numpy(arr).to(device)
        if segformer:
            logits = net(pixel_values=images).logits          # 1/4 分辨率
            logits = F.interpolate(logits, size=tuple(shape),
                                   mode="bilinear", align_corners=False)
            pr = logits[0]
        else:
            pr = net(images)[0]
        pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
        #   裁掉 letterbox 的灰条
        pr = pr[(shape[0] - nh) // 2: (shape[0] - nh) // 2 + nh,
                (shape[1] - nw) // 2: (shape[1] - nw) // 2 + nw]
        pr = cv2.resize(pr, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    if flip:
        pr = pr[:, ::-1]
    return pr


def predict_prob(net, image, input_shape, device, segformer=False, tta=False):
    """（多视角平均后的）原图尺寸概率图。"""
    from utils.utils import cvtColor

    image = cvtColor(image)
    prob = None
    for shape, flip in tta_views(input_shape, tta):
        p = forward_prob(net, image, shape, device, segformer=segformer, flip=flip)
        prob = p if prob is None else prob + p
    return prob


def predict_mask(net, image, input_shape, device, segformer=False, tta=False):
    """原图尺寸的类别索引 mask (np.uint8)。"""
    return predict_prob(net, image, input_shape, device,
                        segformer=segformer, tta=tta).argmax(axis=-1).astype(np.uint8)


class SegEngine:
    """带权值缓存的推理引擎（工作站/QML 壳用）。

    签名一致时复用已加载的模型，避免每次预测都重新建网。
    """

    def __init__(self):
        self.net = None
        self.cfg = None

    def loaded_signature(self):
        if self.cfg is None:
            return None
        return (self.cfg["model_path"], self.cfg["backbone"],
                self.cfg["num_classes"], self.cfg["downsample_factor"],
                tuple(self.cfg["input_shape"]), self.cfg["cuda"])

    def load(self, model_path, backbone, num_classes, downsample_factor,
             input_shape, cuda):
        """加载权值。签名一致时直接复用，返回是否真的重新加载过。"""
        import torch

        sig = (model_path, backbone, num_classes, downsample_factor,
               tuple(input_shape), cuda)
        if self.net is not None and self.loaded_signature() == sig:
            return False

        use_cuda = bool(cuda) and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        self.net = load_net(model_path, backbone, num_classes,
                            downsample_factor, device)
        self.cfg = {"model_path": model_path, "backbone": backbone,
                    "num_classes": num_classes,
                    "downsample_factor": downsample_factor,
                    "input_shape": list(input_shape), "cuda": use_cuda,
                    "device": device}
        return True

    def predict_mask(self, image, tta=False):
        """输入 PIL Image，输出与原图同尺寸的类别索引 mask。

        tta=True 时质量更高但耗时约 6 倍，适合单张精细预测；批量/视频建议关闭。
        """
        return predict_mask(self.net, image, self.cfg["input_shape"],
                            self.cfg["device"],
                            segformer=is_segformer(self.cfg["backbone"]),
                            tta=tta)


def compose_view(image, mask, colors, mode, alpha=0.7):
    """根据 mask 合成可视化图。mode: 0 混合 / 1 仅分割图 / 2 扣除背景 / 3 原图"""
    image = image.convert("RGB")
    if mode == 3:
        return image
    palette = np.array(colors, np.uint8)
    if len(palette) < 256:
        palette = np.vstack([palette, np.zeros((256 - len(palette), 3), np.uint8)])
    seg = palette[mask]
    if mode == 1:
        return Image.fromarray(seg)
    if mode == 2:
        arr = (np.expand_dims(mask != 0, -1) * np.array(image, np.float32)).astype(np.uint8)
        return Image.fromarray(arr)
    return Image.blend(image, Image.fromarray(seg), alpha)


def mask_statistics(mask, class_names):
    """返回 [(类名, 像素数, 占比)]"""
    total = mask.size
    counts = np.bincount(mask.reshape(-1), minlength=len(class_names))
    rows = []
    for i, name in enumerate(class_names):
        if i < len(counts):
            rows.append((name, int(counts[i]), counts[i] / total * 100.0))
    return rows


__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "SEGFORMER_HUB",
    "SegEngine",
    "build_net",
    "compose_view",
    "forward_prob",
    "is_segformer",
    "load_net",
    "mask_statistics",
    "predict_mask",
    "predict_prob",
    "tta_views",
]
