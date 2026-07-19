"""训练 SegFormer。与 DeepLabV3+ 使用完全相同的划分与 mIoU 口径
（完整混淆矩阵 + 背景不计入平均），结果可直接横向对比。

初始权重取 ADE20K 微调版（nvidia/segformer-b*-finetuned-ade-512-512），
对分割任务的迁移效果好于仅 ImageNet 分类预训练。

单阶段（默认，训练本项目数据）:
    python tools/train_segformer.py
    python tools/train_segformer.py --model nvidia/segformer-b2-finetuned-ade-512-512

底座对照（ADE20K vs Cityscapes，尚未验证）:
    Cityscapes 是街景域、类别体系与本项目更近（road/building/vegetation/sky）；
    ADE20K 场景更杂但覆盖面广。两个 checkpoint 都已验证可在 256px / 5 类下加载。
    对照时只改 --model 与 --save-dir，其余参数与 --seed 必须保持一致：
        python tools/train_segformer.py --model nvidia/segformer-b2-finetuned-ade-512-512 \
            --save-dir logs_sf_ade
        python tools/train_segformer.py --model nvidia/segformer-b2-finetuned-cityscapes-1024-1024 \
            --save-dir logs_sf_city
    单次训练的 mIoU 差异 1 个点以内不足以下结论 —— UAVid 那次的教训是
    要做配对 bootstrap 看区间是否跨 0，见 README「两阶段训练」一节。

两阶段（领域内预训练 -> 微调，适合本项目训练集只有 245 张的情况）:
    # 阶段一：在外部乡村/航拍数据上训练（数据用 tools/import_external.py 转换）
    python tools/train_segformer.py --voc-root VOCdevkit_ruralscapes \
        --num-classes 5 --save-dir logs_sf_stage1 --epochs 60

    # 阶段二：载入阶段一权重，在本项目数据上微调（学习率调小）
    python tools/train_segformer.py --init-from logs_sf_stage1/best_segformer.pth \
        --save-dir logs_sf_stage2 --epochs 200 --lr 3e-5

两阶段的类别数必须一致；若阶段一用了 ext7 体系，两条命令都要 --num-classes 7。
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch
from torch.utils.data import DataLoader

from segcore import evaluate_hist, per_image_hists, read_split
from utils.dataloader import DeeplabDataset, deeplab_dataset_collate
from utils.utils import seed_everything

NAMES = ["_background_", "building", "sky", "tree", "way"]
EXT7_NAMES = NAMES + ["farmland", "water"]
REMOVE = [0]
# ImageNet 归一化：预训练主干的输入约定（DeeplabDataset 只做了 /255）。
# 训练侧仍需自己做；推理/评估侧由 segcore 统一处理。
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def evaluate(model, val_ids, input_shape, num_classes, device, voc_root,
             backbone="segformer-b2"):
    """评估协议来自 segcore，与 get_miou.py / compare_models 逐位一致。"""
    hists = per_image_hists(model, val_ids, voc_root, num_classes,
                            input_shape, device, backbone=backbone)
    model.train()
    metrics = evaluate_hist(hists.sum(0), num_classes, REMOVE)
    return metrics["miou"], metrics["iou"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nvidia/segformer-b0-finetuned-ade-512-512")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=6e-5, help="编码器学习率（解码头 ×10）")
    ap.add_argument("--input-size", type=int, default=256)
    ap.add_argument("--eval-period", type=int, default=5)
    ap.add_argument("--save-dir", default=os.path.join(ROOT, "logs_segformer_b0"))
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--voc-root", default="VOCdevkit",
                    help="数据集根目录（含 VOC2007/），阶段一指向外部数据集")
    ap.add_argument("--num-classes", type=int, default=len(NAMES),
                    help="类别数，base5=5 / ext7=7；两阶段必须一致")
    ap.add_argument("--init-from",
                    help="从已有 .pth 初始化（两阶段训练的第二阶段用），"
                         "优先于 --model 的 HuggingFace 预训练权重")
    args = ap.parse_args()

    from transformers import SegformerForSemanticSegmentation

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = args.num_classes
    names = EXT7_NAMES if num_classes == len(EXT7_NAMES) else NAMES
    if len(names) != num_classes:
        names = [f"class_{i}" for i in range(num_classes)]
    input_shape = [args.input_size, args.input_size]
    os.makedirs(args.save_dir, exist_ok=True)

    #   voc_root 指向含 VOC2007/ 的目录（segcore 的约定），不要指到 VOC2007 里面
    voc_root = os.path.join(ROOT, args.voc_root)
    split_dir = os.path.join(voc_root, "VOC2007", "ImageSets", "Segmentation")
    train_lines = open(os.path.join(split_dir, "train.txt")).readlines()
    val_ids = read_split(voc_root, "val")
    print(f"数据集 {args.voc_root}  train {len(train_lines)} / val {len(val_ids)}  "
          f"input {input_shape}  classes {num_classes}  device {device}")

    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model, num_labels=num_classes, ignore_mismatched_sizes=True)
    if args.init_from:
        #   第二阶段：载入阶段一的领域内预训练权重，覆盖 HF 的 ADE20K 初始化
        state = torch.load(args.init_from, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[警告] 权重未完全匹配 missing={len(missing)} unexpected={len(unexpected)}"
                  f"（类别数不一致时分类头不匹配属正常）")
        print(f"已从 {args.init_from} 初始化")
    model = model.to(device).train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model {args.model}  params {n_params/1e6:.2f}M")

    #   DeeplabDataset 按相对 CWD 拼路径，这里给绝对路径以免换工作目录就失效
    train_ds = DeeplabDataset(train_lines, input_shape, num_classes, True,
                              os.path.join(ROOT, args.voc_root))
    gen = DataLoader(train_ds, shuffle=True, batch_size=args.batch_size,
                     num_workers=0, pin_memory=True, drop_last=True,
                     collate_fn=deeplab_dataset_collate)
    steps_per_epoch = len(train_lines) // args.batch_size
    total_iters = steps_per_epoch * args.epochs

    #   SegFormer 论文配方：AdamW + poly 衰减，解码头 10 倍学习率
    head_params, enc_params = [], []
    for n, p in model.named_parameters():
        (head_params if "decode_head" in n else enc_params).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": enc_params, "lr": args.lr},
         {"params": head_params, "lr": args.lr * 10}], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda it: (1 - it / total_iters) ** 1.0)

    mean_d, std_d = MEAN.to(device), STD.to(device)
    best_miou, best_epoch, it = -1.0, -1, 0
    best_path = os.path.join(args.save_dir, "best_segformer.pth")

    for epoch in range(args.epochs):
        total_loss = 0.0
        for step, (imgs, pngs, _) in enumerate(gen):
            if step >= steps_per_epoch:
                break
            imgs = ((imgs.to(device) - mean_d) / std_d)
            pngs = pngs.to(device)
            out = model(pixel_values=imgs, labels=pngs)   # HF 内部上采样 logits 并算 CE
            optimizer.zero_grad()
            out.loss.backward()
            optimizer.step()
            sched.step()
            it += 1
            total_loss += out.loss.item()

        if (epoch + 1) % args.eval_period == 0 or epoch + 1 == args.epochs:
            miou, iou = evaluate(model, val_ids, input_shape, num_classes, device, voc_root)
            tag = ""
            if miou >= best_miou:
                best_miou, best_epoch = miou, epoch + 1
                torch.save(model.state_dict(), best_path)
                tag = "  <- best"
            print(f"epoch {epoch+1:3d}/{args.epochs}  loss {total_loss/steps_per_epoch:.4f}  "
                  f"mIoU {miou:.2f}{tag}", flush=True)

    print("\n==== final ====")
    model.load_state_dict(torch.load(best_path, map_location=device))
    miou, iou = evaluate(model, val_ids, input_shape, num_classes, device, voc_root)
    print(f"best @ epoch {best_epoch}: fg mIoU {miou:.2f}")
    for i, name in enumerate(names):
        extra = "   (不计入平均)" if i in REMOVE else ""
        print(f"  {name:<14}{iou[i]*100:7.2f}{extra}")
    print(f"weights -> {best_path}")


if __name__ == "__main__":
    main()
