"""mIoU 评估子进程：python -m workstation.workers.miou_worker --config xxx.json

输出结构化行（@@ + JSON）：progress / result / error / done
结果图与混淆矩阵保存到 miou_out 目录。
"""
import argparse
import json
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")


def emit(msg_type, **payload):
    payload["type"] = msg_type
    print("@@" + json.dumps(payload, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image

    from utils.utils import cvtColor, preprocess_input, resize_image
    from utils.utils_metrics import (compute_mIoU, mean_metric, per_Accuracy,
                                     show_results)
    from workstation.core.engine import (SEGFORMER_HUB, _IMAGENET_MEAN,
                                         _IMAGENET_STD, is_segformer)

    num_classes = cfg["num_classes"]
    name_classes = cfg["class_names"]
    input_shape = cfg["input_shape"]
    voc_path = cfg["voc_root"]
    split = cfg.get("split", "val")
    cuda = bool(cfg["cuda"]) and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    split_file = os.path.join(voc_path, f"VOC2007/ImageSets/Segmentation/{split}.txt")
    with open(split_file, "r") as f:
        image_ids = [line.strip() for line in f if line.strip()]
    if not image_ids:
        emit("error", message=f"{split}.txt 为空，无法评估")
        sys.exit(2)

    gt_dir = os.path.join(voc_path, "VOC2007/SegmentationClass/")
    miou_out = cfg.get("miou_out", "miou_out")
    pred_dir = os.path.join(miou_out, "detection-results")
    os.makedirs(pred_dir, exist_ok=True)

    emit("status", message=f"加载模型 {cfg['model_path']}")
    backbone = cfg["backbone"]
    segformer = is_segformer(backbone)
    if segformer:
        from transformers import SegformerForSemanticSegmentation
        net = SegformerForSemanticSegmentation.from_pretrained(
            SEGFORMER_HUB[str(backbone).lower()], num_labels=num_classes,
            ignore_mismatched_sizes=True)
    else:
        from nets.deeplabv3_plus import DeepLab
        net = DeepLab(num_classes=num_classes, backbone=backbone,
                      downsample_factor=cfg["downsample_factor"], pretrained=False)
    try:
        net.load_state_dict(torch.load(cfg["model_path"], map_location="cpu"))
    except RuntimeError as exc:
        emit("error", message="权值与配置不匹配（检查类别数/主干/下采样倍数，"
                              "SegFormer 权重需选择对应的 segformer-* 主干）: " + str(exc)[:500])
        sys.exit(2)
    net = net.eval().to(device)

    emit("status", message=f"对 {len(image_ids)} 张 {split} 图片生成预测…")
    with torch.no_grad():
        for i, image_id in enumerate(image_ids):
            image = Image.open(os.path.join(voc_path, "VOC2007/JPEGImages/" + image_id + ".png"))
            image = cvtColor(image)
            orig_w, orig_h = image.size
            image_data, nw, nh = resize_image(image, (input_shape[1], input_shape[0]))
            image_data = preprocess_input(np.array(image_data, np.float32))
            if segformer:
                image_data = (image_data - _IMAGENET_MEAN) / _IMAGENET_STD
            image_data = np.expand_dims(np.transpose(image_data, (2, 0, 1)), 0)
            images = torch.from_numpy(image_data).to(device)
            if segformer:
                logits = net(pixel_values=images).logits
                logits = F.interpolate(logits, size=tuple(input_shape),
                                       mode="bilinear", align_corners=False)
                pr = logits[0]
            else:
                pr = net(images)[0]
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            pr = pr[int((input_shape[0] - nh) // 2): int((input_shape[0] - nh) // 2 + nh),
                    int((input_shape[1] - nw) // 2): int((input_shape[1] - nw) // 2 + nw)]
            pr = cv2.resize(pr, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR).argmax(axis=-1)
            Image.fromarray(np.uint8(pr)).save(os.path.join(pred_dir, image_id + ".png"))
            if (i + 1) % 10 == 0 or i + 1 == len(image_ids):
                emit("progress", current=i + 1, total=len(image_ids))

    emit("status", message="计算 mIoU…")
    #-------------------------------------------------------------------#
    #   与 get_miou.py / 训练中的 EvalCallback 统一口径：
    #   指标在完整混淆矩阵上计算，背景类展示但不计入平均。
    #-------------------------------------------------------------------#
    remove_classes = cfg.get("remove_classes", [0])
    hist, IoUs, PA_Recall, Precision = compute_mIoU(
        gt_dir, pred_dir, image_ids, num_classes, None, remove_classes)

    rows = []
    for i, name in enumerate(name_classes):
        rows.append({
            "name": name,
            "iou": None if np.isnan(IoUs[i]) else round(float(IoUs[i]) * 100, 2),
            "recall": None if np.isnan(PA_Recall[i]) else round(float(PA_Recall[i]) * 100, 2),
            "precision": None if np.isnan(Precision[i]) else round(float(Precision[i]) * 100, 2),
            "averaged": i not in (remove_classes or []),
        })
    emit("result",
         miou=round(float(mean_metric(IoUs, num_classes, remove_classes)) * 100, 2),
         mpa=round(float(mean_metric(PA_Recall, num_classes, remove_classes)) * 100, 2),
         accuracy=round(float(per_Accuracy(hist)) * 100, 2),
         classes=rows, out_dir=miou_out, num_images=len(image_ids), split=split)

    try:
        show_results(miou_out, hist, IoUs, PA_Recall, Precision, name_classes,
                     remove_classes=remove_classes)
    except Exception as exc:
        emit("status", message=f"结果图保存失败: {exc}")

    emit("done", message="评估完成")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        import traceback
        emit("error", message=str(exc), traceback=traceback.format_exc()[-2000:])
        sys.exit(1)
