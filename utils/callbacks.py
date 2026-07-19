import os

import matplotlib
import torch

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import scipy.signal

from torch.utils.tensorboard import SummaryWriter


class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        self.log_dir    = log_dir
        self.losses     = []
        self.val_loss   = []
        
        os.makedirs(self.log_dir)
        self.writer     = SummaryWriter(self.log_dir)
        try:
            dummy_input     = torch.randn(2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    def append_loss(self, epoch, loss, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")

        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")

class EvalCallback():
    def __init__(self, net, input_shape, num_classes, image_ids, dataset_path, log_dir, cuda, \
            miou_out_path=".temp_miou_out", eval_flag=True, period=1, remove_classes=(0,), \
            backbone="mobilenet"):
        super(EvalCallback, self).__init__()

        self.net                = net
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.image_ids          = image_ids
        self.dataset_path       = dataset_path
        self.log_dir            = log_dir
        self.cuda               = cuda
        #   miou_out_path 保留仅为兼容既有调用方；评估已不再落盘中间 png
        self.miou_out_path      = miou_out_path
        self.eval_flag          = eval_flag
        self.period             = period
        self.backbone           = backbone
        #-------------------------------------------------------------------#
        #   与 get_miou.py / 工作站评估页保持同一口径：默认只对前景类求平均。
        #-------------------------------------------------------------------#
        self.remove_classes     = list(remove_classes) if remove_classes else None

        self.image_ids          = [image_id.split()[0] for image_id in image_ids]
        self.mious      = [0]
        self.epoches    = [0]
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write(str(0))
                f.write("\n")

    def on_epoch_end(self, epoch, model_eval):
        if epoch % self.period == 0 and self.eval_flag:
            #---------------------------------------------------------------#
            #   前向与评估协议来自 segcore（全项目唯一实现），与 get_miou.py /
            #   工作站评估页 / train_worker 的周期评估逐位同口径。
            #   这里以前自带一份 get_miou_png —— 那是 8 份拷贝之一。
            #---------------------------------------------------------------#
            from segcore import evaluate_hist, per_image_hists

            self.net    = model_eval
            print("Get miou.")
            device = next(model_eval.parameters()).device
            hists = per_image_hists(model_eval, self.image_ids, self.dataset_path,
                                    self.num_classes, self.input_shape, device,
                                    backbone=self.backbone)
            print("Calculate miou.")
            temp_miou = evaluate_hist(hists.sum(0), self.num_classes,
                                      self.remove_classes)["miou"]

            self.mious.append(temp_miou)
            self.epoches.append(epoch)

            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write(str(temp_miou))
                f.write("\n")
            
            plt.figure()
            plt.plot(self.epoches, self.mious, 'red', linewidth = 2, label='train miou')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Miou')
            plt.title('A Miou Curve')
            plt.legend(loc="upper right")

            plt.savefig(os.path.join(self.log_dir, "epoch_miou.png"))
            plt.cla()
            plt.close("all")

            print("Get miou done.")
            shutil.rmtree(self.miou_out_path)
