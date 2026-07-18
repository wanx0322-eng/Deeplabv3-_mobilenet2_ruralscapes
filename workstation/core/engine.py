"""预测引擎：在后台线程中加载 DeepLabV3+ 模型并推理。

torch 只在本模块函数被调用时才导入，保证主界面秒开。
返回的是类别索引 mask，各种可视化由调用方合成。
"""
import os

import numpy as np
from PIL import Image


#   backbone 字符串 -> 架构。"segformer-b0" 等走 HuggingFace SegFormer，
#   其余（mobilenet / xception）走原有 DeepLabV3+。
SEGFORMER_HUB = {
    "segformer-b0": "nvidia/segformer-b0-finetuned-ade-512-512",
    "segformer-b1": "nvidia/segformer-b1-finetuned-ade-512-512",
    "segformer-b2": "nvidia/segformer-b2-finetuned-ade-512-512",
}
#   SegFormer 预训练主干要求 ImageNet 归一化（DeepLab 分支只做 /255）
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def is_segformer(backbone):
    return str(backbone).lower().startswith("segformer")


class SegEngine:
    def __init__(self):
        self.net = None
        self.cfg = None          # 当前加载配置 dict
        self._torch = None

    def loaded_signature(self):
        if self.cfg is None:
            return None
        return (self.cfg["model_path"], self.cfg["backbone"],
                self.cfg["num_classes"], self.cfg["downsample_factor"],
                tuple(self.cfg["input_shape"]), self.cfg["cuda"])

    def load(self, model_path, backbone, num_classes, downsample_factor,
             input_shape, cuda):
        """加载权值。签名一致时直接复用。抛出异常给调用方展示。"""
        sig = (model_path, backbone, num_classes, downsample_factor,
               tuple(input_shape), cuda)
        if self.net is not None and self.loaded_signature() == sig:
            return False  # 未重新加载

        import torch
        self._torch = torch

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"权值文件不存在: {model_path}")

        use_cuda = bool(cuda) and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        if is_segformer(backbone):
            try:
                from transformers import SegformerForSemanticSegmentation
            except ImportError as exc:
                raise RuntimeError(
                    "使用 SegFormer 需要安装 transformers：pip install transformers") from exc
            hub = SEGFORMER_HUB.get(str(backbone).lower())
            if hub is None:
                raise RuntimeError(f"未知的 SegFormer 型号: {backbone}，"
                                   f"可选 {sorted(SEGFORMER_HUB)}")
            net = SegformerForSemanticSegmentation.from_pretrained(
                hub, num_labels=num_classes, ignore_mismatched_sizes=True)
        else:
            from nets.deeplabv3_plus import DeepLab
            net = DeepLab(num_classes=num_classes, backbone=backbone,
                          downsample_factor=downsample_factor, pretrained=False)

        state = torch.load(model_path, map_location="cpu")
        try:
            net.load_state_dict(state)
        except RuntimeError as exc:
            raise RuntimeError(
                "权值与当前配置不匹配（请检查 类别数/主干网络/下采样倍数 是否与训练时一致，"
                "SegFormer 权重需选择对应的 segformer-* 主干）\n\n"
                + str(exc)[:800]) from exc
        net = net.eval().to(device)

        self.net = net
        self.cfg = {"model_path": model_path, "backbone": backbone,
                    "num_classes": num_classes,
                    "downsample_factor": downsample_factor,
                    "input_shape": list(input_shape), "cuda": use_cuda,
                    "device": device}
        return True

    def _forward_prob(self, image, shape, flip=False):
        """单次前向：letterbox -> 网络 -> softmax -> 去灰条 -> 原尺寸概率图"""
        import cv2
        import torch.nn.functional as F
        from utils.utils import preprocess_input, resize_image

        torch = self._torch
        segformer = is_segformer(self.cfg["backbone"])
        orig_w, orig_h = image.size
        src = image.transpose(1) if flip else image      # 1 = FLIP_LEFT_RIGHT
        image_data, nw, nh = resize_image(src, (shape[1], shape[0]))
        image_data = preprocess_input(np.array(image_data, np.float32))
        if segformer:
            image_data = (image_data - _IMAGENET_MEAN) / _IMAGENET_STD
        image_data = np.expand_dims(np.transpose(image_data, (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data).to(self.cfg["device"])
            if segformer:
                logits = self.net(pixel_values=images).logits      # 1/4 分辨率
                logits = F.interpolate(logits, size=tuple(shape),
                                       mode="bilinear", align_corners=False)
                pr = logits[0]
            else:
                pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            pr = pr[int((shape[0] - nh) // 2): int((shape[0] - nh) // 2 + nh),
                    int((shape[1] - nw) // 2): int((shape[1] - nw) // 2 + nw)]
            pr = cv2.resize(pr, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        if flip:
            pr = pr[:, ::-1]
        return pr

    def predict_mask(self, image, tta=False):
        """输入 PIL Image，输出与原图同尺寸的类别索引 mask (np.uint8)。

        tta=True 时做测试时增强（水平翻转 × 多尺度 0.75/1.0/1.25 概率平均），
        质量更高但耗时约 6 倍，适合单张精细预测；批量/视频建议关闭。
        """
        from utils.utils import cvtColor

        input_shape = self.cfg["input_shape"]
        image = cvtColor(image)

        if tta:
            base = input_shape[0]
            scales = sorted({max(64, int(round(base * s / 32)) * 32)
                             for s in (0.75, 1.0, 1.25)})
            views = [((s, s), f) for s in scales for f in (False, True)]
        else:
            views = [(tuple(input_shape), False)]

        prob = None
        for shape, flip in views:
            p = self._forward_prob(image, shape, flip)
            prob = p if prob is None else prob + p
        return prob.argmax(axis=-1).astype(np.uint8)


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
