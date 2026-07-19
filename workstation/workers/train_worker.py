"""训练子进程：python -m workstation.workers.train_worker --config xxx.json

向 stdout 输出结构化进度行（前缀 @@ + JSON），供图形界面解析：
  step  每若干个迭代一次（loss / f_score / lr / 进度）
  epoch 每个 epoch 结束（train_loss / val_loss）
  miou  周期性评估结果
  saved 权值保存事件
  error / done
在 log_dir 写入 stop.flag 可让训练在当前 epoch 结束后安全停止。
"""
import argparse
import datetime
import json
import os
import sys
from functools import partial

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
    import torch.backends.cudnn as cudnn
    import torch.optim as optim
    from torch.utils.data import DataLoader

    from nets.deeplabv3_plus import DeepLab
    from nets.deeplabv3_training import (CE_Loss, Dice_loss, Focal_Loss,
                                         get_lr_scheduler, set_optimizer_lr,
                                         weights_init)
    from utils.dataloader import DeeplabDataset, deeplab_dataset_collate
    from utils.utils import download_weights, get_lr, seed_everything, worker_init_fn
    from utils.utils_metrics import f_score

    num_classes = cfg["num_classes"]
    # 评估 mIoU 时不计入平均的类别（默认排除背景），与 get_miou.py 口径一致
    remove_classes = cfg.get("remove_classes", [0])
    backbone = cfg["backbone"]
    input_shape = cfg["input_shape"]
    downsample_factor = cfg["downsample_factor"]
    voc_path = cfg["voc_root"]
    cuda = bool(cfg["cuda"]) and torch.cuda.is_available()
    fp16 = bool(cfg["fp16"]) and cuda
    seed = cfg.get("seed", 11)

    seed_everything(seed)
    device = torch.device("cuda" if cuda else "cpu")
    emit("status", message=f"torch {torch.__version__} | device={device} | fp16={fp16}")

    # ---------------- 构建模型 ----------------
    if cfg.get("pretrained") and not cfg.get("model_path"):
        emit("status", message="下载主干预训练权重…")
        try:
            download_weights(backbone)
        except Exception as exc:
            emit("error", message=f"预训练权重下载失败: {exc}")
            sys.exit(2)

    model = DeepLab(num_classes=num_classes, backbone=backbone,
                    downsample_factor=downsample_factor,
                    pretrained=bool(cfg.get("pretrained")))
    if not cfg.get("pretrained"):
        weights_init(model)

    model_path = cfg.get("model_path") or ""
    if model_path:
        if not os.path.exists(model_path):
            emit("error", message=f"权值文件不存在: {model_path}")
            sys.exit(2)
        emit("status", message=f"加载权值 {model_path}")
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location="cpu")
        loaded, skipped = [], []
        for k, v in pretrained_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                model_dict[k] = v
                loaded.append(k)
            else:
                skipped.append(k)
        model.load_state_dict(model_dict)
        emit("status", message=f"成功加载 {len(loaded)} 个权值键，跳过 {len(skipped)} 个（head 不匹配属正常现象）")

    # ---------------- 日志目录 ----------------
    save_dir = cfg["save_dir"]
    time_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    log_dir = os.path.join(save_dir, "loss_" + time_str)
    os.makedirs(log_dir, exist_ok=True)
    stop_flag = os.path.join(save_dir, "stop.flag")
    if os.path.exists(stop_flag):
        os.remove(stop_flag)
    emit("status", message=f"日志目录 {log_dir}", log_dir=log_dir)

    losses_hist, val_losses_hist, miou_hist = [], [], []
    best_miou = -1.0          # 最佳权重按 mIoU 挑选，见训练循环末尾

    def record_epoch_loss(loss, val_loss):
        losses_hist.append(loss)
        val_losses_hist.append(val_loss)
        with open(os.path.join(log_dir, "epoch_loss.txt"), "a") as f:
            f.write(f"{loss}\n")
        with open(os.path.join(log_dir, "epoch_val_loss.txt"), "a") as f:
            f.write(f"{val_loss}\n")

    # ---------------- 数据 ----------------
    seg_dir = os.path.join(voc_path, "VOC2007/ImageSets/Segmentation")
    with open(os.path.join(seg_dir, "train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(seg_dir, "val.txt"), "r") as f:
        val_lines = f.readlines()
    num_train, num_val = len(train_lines), len(val_lines)
    emit("status", message=f"训练集 {num_train} 张 / 验证集 {num_val} 张")

    dice_loss = bool(cfg["dice_loss"])
    focal_loss = bool(cfg["focal_loss"])
    cls_weights = np.ones([num_classes], np.float32)
    if cfg.get("cls_weights"):
        cls_weights = np.array(cfg["cls_weights"], np.float32)

    optimizer_type = cfg["optimizer_type"]
    momentum = cfg["momentum"]
    weight_decay = cfg["weight_decay"]
    Init_lr = cfg["init_lr"]
    Min_lr = Init_lr * cfg.get("min_lr_ratio", 0.01)
    Init_Epoch = cfg["init_epoch"]
    Freeze_Epoch = cfg["freeze_epoch"]
    UnFreeze_Epoch = cfg["unfreeze_epoch"]
    Freeze_batch_size = cfg["freeze_batch_size"]
    Unfreeze_batch_size = cfg["unfreeze_batch_size"]
    Freeze_Train = bool(cfg["freeze_train"])
    lr_decay_type = cfg["lr_decay_type"]
    save_period = cfg["save_period"]
    eval_flag = bool(cfg["eval_flag"])
    eval_period = cfg["eval_period"]
    num_workers = cfg.get("num_workers", 0)

    model_train = model.train()
    if cuda:
        #   与 train.py 保持一致：cudnn.benchmark = True 会覆盖 seed_everything
        #   设下的 deterministic，让上面的 seed 形同虚设。只有明确不要求
        #   可复现时才打开它换取速度（配置 train.deterministic = false）。
        cudnn.benchmark = not bool(cfg.get("deterministic", True))
        model_train = model_train.to(device)

    # ---------------- EMA（指数滑动平均权重） ----------------
    #   评估与最佳权重保存均使用 EMA 副本。小数据集上 EMA 能平滑
    #   训练末期的权值震荡，通常带来 0.5~1.5 mIoU 的稳定提升。
    #   参数与 buffer（含 BN 统计量）一并做 lerp，同 timm ModelEmaV2。
    use_ema = bool(cfg.get("ema", False))
    ema_model = None
    ema_updates = 0
    if use_ema:
        import copy as _copy
        ema_model = _copy.deepcopy(model).to(device).eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)
        emit("status", message="EMA 已启用（decay=0.999，带预热）")

    def ema_update():
        nonlocal ema_updates
        ema_updates += 1
        d = min(0.999, (1 + ema_updates) / (10 + ema_updates))   # 预热期 decay 较小
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in ema_model.state_dict().items():
                if v.dtype.is_floating_point:
                    v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
                else:
                    v.copy_(msd[k])

    scaler = None
    if fp16:
        scaler = torch.cuda.amp.GradScaler()

    if Freeze_Train:
        for param in model.backbone.parameters():
            param.requires_grad = False
    batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

    def adjusted_lr(batch_size):
        nbs = 16
        lr_limit_max = 5e-4 if optimizer_type == "adam" else 1e-1
        lr_limit_min = 3e-4 if optimizer_type == "adam" else 5e-4
        if backbone == "xception":
            lr_limit_max = 1e-4 if optimizer_type == "adam" else 1e-1
            lr_limit_min = 1e-4 if optimizer_type == "adam" else 5e-4
        init_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        min_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
        return init_fit, min_fit

    Init_lr_fit, Min_lr_fit = adjusted_lr(batch_size)
    optimizer = {
        "adam": optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999),
                           weight_decay=weight_decay),
        "sgd": optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum,
                         nesterov=True, weight_decay=weight_decay),
    }[optimizer_type]
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

    epoch_step = num_train // batch_size
    #   验证批数向上取整（drop_last=False），才能覆盖全部验证图 —— 与 train.py 一致
    epoch_step_val = -(-num_val // batch_size)
    if epoch_step == 0 or epoch_step_val == 0:
        emit("error", message="数据集过小或 batch_size 过大，无法训练（每个 epoch 的步数为 0）")
        sys.exit(2)

    train_dataset = DeeplabDataset(train_lines, input_shape, num_classes, True, voc_path)
    val_dataset = DeeplabDataset(val_lines, input_shape, num_classes, False, voc_path)

    def make_loaders(batch_size):
        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size,
                         num_workers=num_workers, pin_memory=True, drop_last=True,
                         collate_fn=deeplab_dataset_collate,
                         worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))
        #   验证集不打乱、不丢尾批 —— 否则每个 epoch 的 val_loss 是在随机的
        #   48/50 张上算的，轮次之间不可比（train.py 已修，此处曾漏同步）。
        #   eval_flag 关闭时最佳权重按 val_loss 挑选，这一条直接影响选型。
        gen_val = DataLoader(val_dataset, shuffle=False, batch_size=batch_size,
                             num_workers=num_workers, pin_memory=True, drop_last=False,
                             collate_fn=deeplab_dataset_collate,
                             worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))
        return gen, gen_val

    gen, gen_val = make_loaders(batch_size)

    # ---------------- 周期评估（mIoU） ----------------
    val_ids = [line.split()[0] for line in val_lines]

    def evaluate_miou():
        import cv2
        import shutil
        import torch.nn.functional as F
        from PIL import Image
        from utils.utils import cvtColor, preprocess_input, resize_image
        from utils.utils_metrics import compute_mIoU, mean_metric

        #   EMA 开启时评估 EMA 副本 —— 最佳权重也按它的 mIoU 挑选
        eval_net = ema_model if use_ema else model_train

        miou_out = os.path.join(log_dir, ".temp_miou")
        pred_dir = os.path.join(miou_out, "detection-results")
        os.makedirs(pred_dir, exist_ok=True)
        gt_dir = os.path.join(voc_path, "VOC2007/SegmentationClass/")
        eval_net.eval()
        with torch.no_grad():
            for image_id in val_ids:
                image = Image.open(os.path.join(voc_path, "VOC2007/JPEGImages/" + image_id + ".png"))
                image = cvtColor(image)
                orig_w, orig_h = image.size
                image_data, nw, nh = resize_image(image, (input_shape[1], input_shape[0]))
                image_data = np.expand_dims(np.transpose(
                    preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)
                images = torch.from_numpy(image_data).to(device)
                pr = eval_net(images)[0]
                pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
                pr = pr[int((input_shape[0] - nh) // 2): int((input_shape[0] - nh) // 2 + nh),
                        int((input_shape[1] - nw) // 2): int((input_shape[1] - nw) // 2 + nw)]
                pr = cv2.resize(pr, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR).argmax(axis=-1)
                Image.fromarray(np.uint8(pr)).save(os.path.join(pred_dir, image_id + ".png"))
        model_train.train()
        # 与 get_miou.py / 评估页统一：完整混淆矩阵算指标，背景不计入平均
        _, IoUs, _, _ = compute_mIoU(gt_dir, pred_dir, val_ids, num_classes,
                                     None, remove_classes)
        shutil.rmtree(miou_out, ignore_errors=True)
        return float(mean_metric(IoUs, num_classes, remove_classes) * 100)

    # ---------------- 训练循环 ----------------
    weights = torch.from_numpy(cls_weights).to(device)
    UnFreeze_flag = False
    report_every = max(1, epoch_step // 25)

    for epoch in range(Init_Epoch, UnFreeze_Epoch):
        if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
            batch_size = Unfreeze_batch_size
            Init_lr_fit, Min_lr_fit = adjusted_lr(batch_size)
            lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
            for param in model.backbone.parameters():
                param.requires_grad = True
            epoch_step = num_train // batch_size
            epoch_step_val = -(-num_val // batch_size)
            if epoch_step == 0 or epoch_step_val == 0:
                emit("error", message="解冻后步数为 0，无法继续训练")
                sys.exit(2)
            gen, gen_val = make_loaders(batch_size)
            report_every = max(1, epoch_step // 25)
            UnFreeze_flag = True
            emit("status", message=f"第 {epoch + 1} 个 epoch：解冻主干网络，batch_size={batch_size}")

        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        total_loss, total_f = 0.0, 0.0
        model_train.train()
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            imgs, pngs, labels = batch
            imgs, pngs, labels = imgs.to(device), pngs.to(device), labels.to(device)
            optimizer.zero_grad()
            if not fp16:
                outputs = model_train(imgs)
                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)
                if dice_loss:
                    loss = loss + Dice_loss(outputs, labels)
                with torch.no_grad():
                    _f = f_score(outputs, labels)
                loss.backward()
                optimizer.step()
                if use_ema:
                    ema_update()
            else:
                with torch.cuda.amp.autocast():
                    outputs = model_train(imgs)
                    if focal_loss:
                        loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
                    else:
                        loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)
                    if dice_loss:
                        loss = loss + Dice_loss(outputs, labels)
                    with torch.no_grad():
                        _f = f_score(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if use_ema:
                    ema_update()

            total_loss += loss.item()
            total_f += _f.item()
            if (iteration + 1) % report_every == 0 or iteration + 1 == epoch_step:
                emit("step", phase="train", epoch=epoch + 1, total_epoch=UnFreeze_Epoch,
                     step=iteration + 1, total_step=epoch_step,
                     loss=round(total_loss / (iteration + 1), 4),
                     f_score=round(total_f / (iteration + 1), 4),
                     lr=get_lr(optimizer))

        # ---------- 验证 ----------
        val_loss, val_f = 0.0, 0.0
        model_train.eval()
        with torch.no_grad():
            for iteration, batch in enumerate(gen_val):
                if iteration >= epoch_step_val:
                    break
                imgs, pngs, labels = batch
                imgs, pngs, labels = imgs.to(device), pngs.to(device), labels.to(device)
                outputs = model_train(imgs)
                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)
                if dice_loss:
                    loss = loss + Dice_loss(outputs, labels)
                _f = f_score(outputs, labels)
                val_loss += loss.item()
                val_f += _f.item()

        train_loss_avg = total_loss / epoch_step
        val_loss_avg = val_loss / epoch_step_val
        record_epoch_loss(train_loss_avg, val_loss_avg)
        emit("epoch", epoch=epoch + 1, total_epoch=UnFreeze_Epoch,
             train_loss=round(train_loss_avg, 4), val_loss=round(val_loss_avg, 4),
             lr=get_lr(optimizer))

        # ---------- 周期 mIoU ----------
        miou = None
        if eval_flag and (epoch + 1) % eval_period == 0:
            emit("status", message="正在计算验证集 mIoU…")
            try:
                miou = evaluate_miou()
                miou_hist.append((epoch + 1, miou))
                with open(os.path.join(log_dir, "epoch_miou.txt"), "a") as f:
                    f.write(f"{miou}\n")
                emit("miou", epoch=epoch + 1, miou=round(miou, 2))
            except Exception as exc:
                emit("status", message=f"mIoU 评估失败: {exc}")
                miou = None

        # ---------- 保存 ----------
        if (epoch + 1) % save_period == 0 or epoch + 1 == UnFreeze_Epoch:
            path = os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth"
                                % (epoch + 1, train_loss_avg, val_loss_avg))
            torch.save(model.state_dict(), path)
            emit("saved", path=path, kind="period")

        #-------------------------------------------------------------------#
        #   最佳权重按【mIoU】挑选，而不是 val_loss。
        #   实测 val_loss 最低的权重并不是 mIoU 最高的那个（差约 1.7 个点），
        #   val_loss 只反映逐像素交叉熵，和分割质量并不同向。
        #   只有在关闭周期评估时才退回按 val_loss 选。
        #-------------------------------------------------------------------#
        #   EMA 开启时，mIoU 是在 EMA 副本上测的，best/last 相应保存 EMA 权重
        #  （周期存档仍保存原始权重，便于断点续训）
        deploy_state = ema_model.state_dict() if use_ema else model.state_dict()
        best_path = os.path.join(save_dir, "best_epoch_weights.pth")
        if miou is not None:
            if miou >= best_miou:
                best_miou = miou
                torch.save(deploy_state, best_path)
                emit("saved", path=best_path, kind="best",
                     metric="miou", value=round(miou, 2))
        elif not eval_flag:
            if val_loss_avg <= min(val_losses_hist):
                torch.save(deploy_state, best_path)
                emit("saved", path=best_path, kind="best",
                     metric="val_loss", value=round(val_loss_avg, 4))
        torch.save(deploy_state, os.path.join(save_dir, "last_epoch_weights.pth"))

        if os.path.exists(stop_flag):
            os.remove(stop_flag)
            emit("status", message="收到停止指令，已在 epoch 结束后安全停止")
            break

    emit("done", message="训练结束")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        import traceback
        emit("error", message=str(exc), traceback=traceback.format_exc()[-2000:])
        sys.exit(1)
