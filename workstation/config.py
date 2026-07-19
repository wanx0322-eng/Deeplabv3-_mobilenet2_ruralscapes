"""工作站全局配置：以 JSON 形式持久化到项目根目录 workstation_config.json"""
import copy
import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "workstation_config.json")

DEFAULT_CONFIG = {
    "dataset": {
        # VOC 数据集根目录（内部为 VOC2007 结构）
        "voc_root": "VOCdevkit",
        # 类别名，第 0 个为背景。num_classes = len(class_names)
        "class_names": ["_background_", "building", "sky", "tree", "way"],
        # 每个类别的显示颜色 RGB
        "class_colors": [
            [0, 0, 0],
            [128, 0, 0],
            [0, 128, 0],
            [128, 128, 0],
            [0, 0, 128],
        ],
        # 随机划分参数
        "trainval_percent": 1.0,
        "train_percent": 0.7,
        "split_seed": 0,
        # 评估 mIoU / mPA 时不计入平均的类别（默认排除背景）。
        # 注意：这些类别的像素仍然留在混淆矩阵里充当其它类别的 FP / FN，
        # 只是不参与求平均 —— 绝不能从矩阵里删掉行列，否则前景 IoU 会被高估。
        "remove_classes": [0],
    },
    "train": {
        "cuda": True,
        "seed": 11,
        # True = 可复现（cudnn.benchmark 关闭，稍慢）；
        # False = 允许 cudnn 自动寻优，更快但同 seed 结果不可完全复现
        "deterministic": True,
        "fp16": False,
        "backbone": "mobilenet",         # mobilenet / xception
        "pretrained": False,
        "model_path": "model_data/deeplab_mobilenetv2.pth",
        "downsample_factor": 8,          # 8 / 16
        "input_shape": [256, 256],
        "init_epoch": 0,
        "freeze_epoch": 50,
        "freeze_batch_size": 4,
        "unfreeze_epoch": 80,
        "unfreeze_batch_size": 4,
        "freeze_train": True,
        "init_lr": 7e-3,
        "min_lr_ratio": 0.01,
        "optimizer_type": "sgd",         # sgd / adam
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "lr_decay_type": "cos",          # cos / step
        "save_period": 5,
        "save_dir": "logs",
        "eval_flag": True,
        "eval_period": 5,
        "dice_loss": False,
        "focal_loss": False,
        # 类别损失权重（长度 = num_classes）。留空表示各类等权。
        # 本数据集 way 只占约 7% 像素、building 占 50%，等权会让 way 学不好。
        # 下面是按 1/sqrt(频率) 归一化得到的权重。
        "cls_weights": [0.943, 0.535, 1.146, 0.980, 1.396],
        # EMA 指数滑动平均权重：评估与最佳权重保存使用 EMA 副本，
        # 平滑训练末期震荡，小数据集上通常稳定 +0.5~1.5 mIoU
        "ema": True,
        "num_workers": 0,
    },
    "predict": {
        # SegFormer-B2：前景 mIoU 73.45（DeepLabV3+ MobileNetV2 为 69.11）
        # 备选：logs_segformer_b0（72.15，9.6ms 最快）/ logs_v2_B（69.11，DeepLab）
        "model_path": "logs_segformer_b2/best_segformer.pth",
        "backbone": "segformer-b2",
        "downsample_factor": 8,
        "input_shape": [256, 256],
        "cuda": True,
        "mix_type": 0,                    # 0 混合 / 1 仅分割图 / 2 扣除背景
        "blend_alpha": 0.7,
        # 测试时增强（翻转×多尺度概率平均）：质量更高，单张耗时约 6 倍
        "tta": False,
        "save_dir": "img_out",
    },
}


def _merge(base, override):
    """递归合并：以 base 为模板，用 override 中已有的值覆盖"""
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


class Config:
    """简单的 JSON 配置存取，页面之间共享同一实例"""

    def __init__(self, path=CONFIG_PATH):
        self.path = path
        self.data = copy.deepcopy(DEFAULT_CONFIG)
        self.load()

    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.data = _merge(DEFAULT_CONFIG, json.load(f))
            except (json.JSONDecodeError, OSError):
                pass
        #-----------------------------------------------------------------#
        #   持久化的配置可能被手改坏。这里只告警不抛异常 —— 启动阶段直接崩
        #   会让工作站打不开，反而更难修。真正的拦截在 validate_or_raise()，
        #   由训练/推理页在启动任务前调用。
        #-----------------------------------------------------------------#
        for message in self.validate():
            print("[配置告警] %s" % message)

    # ---------------- 校验 ----------------
    def validate(self):
        """返回错误信息列表；空列表表示配置合法。"""
        from workstation.config_schema import validate_config
        return validate_config(self.data)

    def validate_or_raise(self):
        """配置非法时抛 ValueError，消息可直接弹给用户。"""
        errors = self.validate()
        if errors:
            raise ValueError("配置有 %d 处问题：\n\n%s"
                             % (len(errors), "\n".join("· " + e for e in errors)))

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    # 常用快捷访问
    @property
    def dataset(self):
        return self.data["dataset"]

    @property
    def train(self):
        return self.data["train"]

    @property
    def predict(self):
        return self.data["predict"]

    @property
    def num_classes(self):
        return len(self.dataset["class_names"])

    def abs_path(self, rel):
        if os.path.isabs(rel):
            return rel
        return os.path.join(PROJECT_ROOT, rel)

    def voc2007_dir(self):
        return os.path.join(self.abs_path(self.dataset["voc_root"]), "VOC2007")
