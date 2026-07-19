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

    import numpy as np
    import torch

    from segcore import evaluate_hist, load_net, per_image_hists
    from utils.utils_metrics import show_results

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

    miou_out = cfg.get("miou_out", "miou_out")
    pred_dir = os.path.join(miou_out, "detection-results")
    os.makedirs(pred_dir, exist_ok=True)

    emit("status", message=f"加载模型 {cfg['model_path']}")
    backbone = cfg["backbone"]
    try:
        net = load_net(cfg["model_path"], backbone, num_classes,
                       cfg["downsample_factor"], device)
    except (RuntimeError, FileNotFoundError) as exc:
        emit("error", message=str(exc)[:600])
        sys.exit(2)

    emit("status", message=f"对 {len(image_ids)} 张 {split} 图片生成预测…")
    #   前向与评估协议来自 segcore（全项目唯一实现），
    #   预测 png 仍然落盘，工作站评估页要用。
    hists = per_image_hists(
        net, image_ids, voc_path, num_classes, input_shape, device,
        backbone=backbone, save_pred_dir=pred_dir,
        progress=lambda cur, total: (
            emit("progress", current=cur, total=total)
            if cur % 10 == 0 or cur == total else None))

    emit("status", message="计算 mIoU…")
    #-------------------------------------------------------------------#
    #   与 get_miou.py / 训练中的周期评估统一口径：
    #   指标在完整混淆矩阵上计算，背景类展示但不计入平均。
    #-------------------------------------------------------------------#
    remove_classes = cfg.get("remove_classes", [0])
    hist = hists.sum(0)
    metrics = evaluate_hist(hist, num_classes, remove_classes)
    IoUs, PA_Recall, Precision = (metrics["iou"], metrics["recall"],
                                  metrics["precision"])

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
         miou=round(metrics["miou"], 2),
         mpa=round(metrics["mpa"], 2),
         accuracy=round(metrics["accuracy"], 2),
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
