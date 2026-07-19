import colorsys
import time

import numpy as np
import torch
from PIL import Image

from segcore import is_segformer, load_net, predict_prob
from utils.utils import cvtColor, show_config


#-----------------------------------------------------------------------------------#
#   使用自己训练好的模型预测需要修改3个参数
#   model_path、backbone和num_classes都需要修改！
#   如果出现shape不匹配，一定要注意训练时的model_path、backbone和num_classes的修改
#-----------------------------------------------------------------------------------#
class DeeplabV3(object):
    _defaults = {
        #-------------------------------------------------------------------#
        #   model_path指向logs文件夹下的权值文件
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表miou较高，仅代表该权值在验证集上泛化性能较好。
        #-------------------------------------------------------------------#
        # "model_path"        : 'model_data/deeplab_mobilenetv2.pth',
        #   logs_v2_B：245 张训练图 + 类别加权 + dice loss，按 mIoU 选出的最佳权重
        #   验证集（50 张，与历史一致）前景 mIoU 69.11（旧权重为 62.52）
        "model_path"        : r'logs_v2_B\best_epoch_weights.pth',
        #----------------------------------------#
        #   所需要区分的类的个数+1
        #----------------------------------------#
        "num_classes"       : 5,
        #----------------------------------------#
        #   所使用的的主干网络：
        #   mobilenet / xception          -> DeepLabV3+
        #   segformer-b0/b1/b2            -> SegFormer
        #   （SegFormer 权重必须搭配对应型号的主干名，
        #     且输入会自动做 ImageNet 归一化，见 segcore.engine）
        #----------------------------------------#
        "backbone"          : "mobilenet",
        #----------------------------------------#
        #   输入图片的大小
        #----------------------------------------#
        "input_shape"       : [256, 256],
        #----------------------------------------#
        #   下采样的倍数，一般可选的为8和16
        #   与训练时设置的一样即可
        #----------------------------------------#
        "downsample_factor" : 8,
        #-------------------------------------------------#
        #   mix_type参数用于控制检测结果的可视化方式
        #
        #   mix_type = 0的时候代表原图与生成的图进行混合
        #   mix_type = 1的时候代表仅保留生成的图
        #   mix_type = 2的时候代表仅扣去背景，仅保留原图中的目标
        #-------------------------------------------------#
        "mix_type"          : 0,
        #-------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        #-------------------------------#
        "cuda"              : True,
    }

    #---------------------------------------------------#
    #   初始化Deeplab
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        #---------------------------------------------------#
        #   没有可用的 GPU 时自动退回 CPU，否则 .cuda() 会直接抛错
        #---------------------------------------------------#
        self.cuda = bool(self.cuda) and torch.cuda.is_available()
        #---------------------------------------------------#
        #   画框设置不同的颜色
        #---------------------------------------------------#
        if self.num_classes <= 21:
            self.colors = [ (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), 
                            (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
                            (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), 
                            (128, 64, 12)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        #---------------------------------------------------#
        #   获得模型
        #---------------------------------------------------#
        self.generate()

        #   打印【实际生效】的配置，而不是类默认值
        #   （原来是 show_config(**self._defaults)，传入的 kwargs 根本不会显示）
        show_config(**{k: getattr(self, k) for k in self._defaults})
                    
    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def generate(self, onnx=False):
        #-------------------------------#
        #   载入模型与权值（构建与加载统一走 segcore）
        #-------------------------------#
        self.device = torch.device('cuda' if (self.cuda and not onnx) else 'cpu')
        self.net = load_net(self.model_path, self.backbone, self.num_classes,
                            self.downsample_factor, self.device)
        self.segformer = is_segformer(self.backbone)
        print('{} model, and classes loaded.'.format(self.model_path))

    #---------------------------------------------------#
    #   统一前向：letterbox -> 网络 -> softmax -> 去灰条 -> 还原尺寸
    #   detect_image / get_FPS / get_miou_png 以前各有一份拷贝，
    #   现在都走 segcore.predict_prob 这一条路径。
    #---------------------------------------------------#
    def _predict(self, image):
        return predict_prob(self.net, image, self.input_shape, self.device,
                            segformer=self.segformer)

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image, count=False, name_classes=None):
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        old_img     = image.copy()
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]

        pr = self._predict(image).argmax(axis=-1)

        #---------------------------------------------------------#
        #   计数
        #---------------------------------------------------------#
        if count:
            classes_nums        = np.zeros([self.num_classes])
            total_points_num    = orininal_h * orininal_w
            print('-' * 63)
            print("|%25s | %15s | %15s|"%("Key", "Value", "Ratio"))
            print('-' * 63)
            for i in range(self.num_classes):
                num     = np.sum(pr == i)
                ratio   = num / total_points_num * 100
                if num > 0:
                    print("|%25s | %15s | %14.2f%%|"%(str(name_classes[i]), str(num), ratio))
                    print('-' * 63)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
    
        if self.mix_type == 0:
            # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
            # for c in range(self.num_classes):
            #     seg_img[:, :, 0] += ((pr[:, :] == c ) * self.colors[c][0]).astype('uint8')
            #     seg_img[:, :, 1] += ((pr[:, :] == c ) * self.colors[c][1]).astype('uint8')
            #     seg_img[:, :, 2] += ((pr[:, :] == c ) * self.colors[c][2]).astype('uint8')
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            #------------------------------------------------#
            #   将新图片转换成Image的形式
            #------------------------------------------------#
            image   = Image.fromarray(np.uint8(seg_img))
            #------------------------------------------------#
            #   将新图与原图及进行混合
            #------------------------------------------------#
            image   = Image.blend(old_img, image, 0.7)

        elif self.mix_type == 1:
            # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
            # for c in range(self.num_classes):
            #     seg_img[:, :, 0] += ((pr[:, :] == c ) * self.colors[c][0]).astype('uint8')
            #     seg_img[:, :, 1] += ((pr[:, :] == c ) * self.colors[c][1]).astype('uint8')
            #     seg_img[:, :, 2] += ((pr[:, :] == c ) * self.colors[c][2]).astype('uint8')
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            #------------------------------------------------#
            #   将新图片转换成Image的形式
            #------------------------------------------------#
            image   = Image.fromarray(np.uint8(seg_img))

        elif self.mix_type == 2:
            seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype('uint8')
            #------------------------------------------------#
            #   将新图片转换成Image的形式
            #------------------------------------------------#
            image = Image.fromarray(np.uint8(seg_img))
        
        return image

    def get_FPS(self, image, test_interval):
        image = cvtColor(image)
        #   先跑一次预热（CUDA 首次调用含上下文初始化，计入会严重高估耗时）
        self._predict(image)
        t1 = time.time()
        for _ in range(test_interval):
            self._predict(image)
        t2 = time.time()
        return (t2 - t1) / test_interval

    def convert_to_onnx(self, simplify, model_path):
        import onnx
        if is_segformer(self.backbone):
            raise RuntimeError("ONNX 导出目前只支持 DeepLabV3+ 主干"
                               "（SegFormer 的前向签名是 pixel_values=）")
        self.generate(onnx=True)

        im                  = torch.zeros(1, 3, *self.input_shape).to('cpu')  # image size(1, 3, 512, 512) BCHW
        input_layer_names   = ["images"]
        output_layer_names  = ["output"]
        
        # Export the model
        print(f'Starting export with onnx {onnx.__version__}.')
        torch.onnx.export(self.net,
                        im,
                        f               = model_path,
                        verbose         = False,
                        opset_version   = 12,
                        training        = torch.onnx.TrainingMode.EVAL,
                        do_constant_folding = True,
                        input_names     = input_layer_names,
                        output_names    = output_layer_names,
                        dynamic_axes    = None)

        # Checks
        model_onnx = onnx.load(model_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Simplify onnx
        if simplify:
            import onnxsim
            print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=False,
                input_shapes=None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, model_path)

        print('Onnx model save as {}'.format(model_path))
    
    def get_miou_png(self, image):
        pr = self._predict(cvtColor(image)).argmax(axis=-1)
        return Image.fromarray(np.uint8(pr))
