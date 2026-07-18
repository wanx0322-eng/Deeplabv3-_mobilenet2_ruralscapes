"""workstation_config.json 的校验层。

把 config.py 里的裸 dict 映射到 ruralscape_studio.domain 的 Pydantic 配置类上，
在训练/推理真正启动【之前】把配置错误变成一条人话错误信息。

设计约束：
  * 不改变 Config.data 的结构 —— 各页面仍然按 cfg.train["xxx"] 读写，
    这里只做只读校验，映射失败不会影响既有流程。
  * domain 里的配置类只覆盖了公共字段（类别数/权重/轮次/输入尺寸/学习率），
    D 项目特有的字段（优化器、下采样倍数、EMA、mix_type 等）在本模块补充校验。
"""
import os

from ruralscape_studio.domain import (
    DeepLabTrainingConfig,
    EvaluationConfig,
    InferenceConfig,
    ModelEngine,
)

#   与 workstation/core/engine.py 的 SEGFORMER_HUB / DeepLab 分支保持一致
DEEPLAB_BACKBONES = ("mobilenet", "xception")
SEGFORMER_BACKBONES = ("segformer-b0", "segformer-b1", "segformer-b2")
KNOWN_BACKBONES = DEEPLAB_BACKBONES + SEGFORMER_BACKBONES


def engine_of(backbone):
    """backbone 字符串 -> ModelEngine 枚举。"""
    return (ModelEngine.SEGFORMER_B2 if str(backbone).lower().startswith("segformer")
            else ModelEngine.DEEPLAB_V3_PLUS)


def _class_weights(train, num_classes):
    """cls_weights 留空表示各类等权，补成全 1 以满足 domain 的长度校验。"""
    weights = train.get("cls_weights") or []
    if not weights:
        return tuple(1.0 for _ in range(num_classes))
    return tuple(float(w) for w in weights)


def _check_dataset(dataset, errors):
    names = dataset.get("class_names") or []
    colors = dataset.get("class_colors") or []
    if len(names) < 2:
        errors.append("dataset.class_names 至少要有背景 + 1 个前景类")
    if len(colors) != len(names):
        errors.append(
            "dataset.class_colors 有 %d 项，class_names 有 %d 项，两者必须等长"
            % (len(colors), len(names)))
    for i, color in enumerate(colors):
        if len(color) != 3 or any(not 0 <= int(c) <= 255 for c in color):
            errors.append("dataset.class_colors[%d] 不是合法的 RGB 三元组" % i)
    for i in dataset.get("remove_classes") or []:
        if not 0 <= int(i) < len(names):
            errors.append(
                "dataset.remove_classes 里的下标 %d 超出类别范围 0..%d"
                % (i, len(names) - 1))
    percent = dataset.get("train_percent", 0.7)
    if not 0 < float(percent) < 1:
        errors.append("dataset.train_percent 必须在 (0, 1) 之间，当前 %r" % percent)


def _check_train_extras(train, num_classes, errors):
    """domain.DeepLabTrainingConfig 覆盖不到的 D 项目特有字段。"""
    backbone = str(train.get("backbone", ""))
    if backbone not in KNOWN_BACKBONES:
        errors.append("train.backbone=%r 未知，可选 %s"
                      % (backbone, list(KNOWN_BACKBONES)))
    if int(train.get("downsample_factor", 8)) not in (8, 16):
        errors.append("train.downsample_factor 只能是 8 或 16，当前 %r"
                      % train.get("downsample_factor"))
    if str(train.get("optimizer_type", "")) not in ("sgd", "adam"):
        errors.append("train.optimizer_type 只能是 sgd 或 adam，当前 %r"
                      % train.get("optimizer_type"))
    if str(train.get("lr_decay_type", "")) not in ("cos", "step"):
        errors.append("train.lr_decay_type 只能是 cos 或 step，当前 %r"
                      % train.get("lr_decay_type"))
    if not 0 < float(train.get("min_lr_ratio", 0.01)) <= 1:
        errors.append("train.min_lr_ratio 必须在 (0, 1] 之间，当前 %r"
                      % train.get("min_lr_ratio"))
    if int(train.get("freeze_batch_size", 4)) < 2:
        errors.append("train.freeze_batch_size 至少为 2（BatchNorm 需要）")
    for key in ("save_period", "eval_period"):
        if int(train.get(key, 1)) < 1:
            errors.append("train.%s 必须 >= 1，当前 %r" % (key, train.get(key)))
    weights = train.get("cls_weights") or []
    if weights and len(weights) != num_classes:
        errors.append(
            "train.cls_weights 有 %d 项，但类别数是 %d —— 长度必须相等（留空表示等权）"
            % (len(weights), num_classes))


def build_training_config(data):
    """构造 domain 的训练配置对象。校验失败时抛 ValueError。"""
    train = data["train"]
    num_classes = len(data["dataset"]["class_names"])
    freeze_train = bool(train.get("freeze_train", True))
    #   不冻结训练时 freeze_epoch 不参与流程，domain 的
    #   start_epoch < freeze_epoch <= total_epochs 约束不适用，
    #   用 init_epoch + 1 占位并单独校验首尾轮次。
    freeze_epoch = int(train["freeze_epoch"]) if freeze_train else int(train["init_epoch"]) + 1
    return DeepLabTrainingConfig(
        num_classes=num_classes,
        class_weights=_class_weights(train, num_classes),
        batch_size=int(train["unfreeze_batch_size"]),
        start_epoch=int(train["init_epoch"]),
        freeze_epoch=freeze_epoch,
        total_epochs=int(train["unfreeze_epoch"]),
        input_size=tuple(train["input_shape"]),
        learning_rate=float(train["init_lr"]),
    )


def build_inference_config(data, project_root=""):
    predict = data["predict"]
    model_path = predict["model_path"]
    return InferenceConfig(
        engine=engine_of(predict.get("backbone", "mobilenet")),
        model_path=model_path,
        input_size=tuple(predict["input_shape"]),
        output_dir=os.path.join(project_root, predict["save_dir"]) if project_root
        else predict["save_dir"],
        #   confidence 是 domain 里的置信度阈值，本项目的推理走 argmax、
        #   没有这个概念，保持默认值。blend_alpha（混合不透明度）语义不同，
        #   不要塞进来，它在 validate_config 里单独校验。
        blend=int(predict.get("mix_type", 0)) == 0,
    )


def build_evaluation_config(data, split="val"):
    names = tuple(data["dataset"]["class_names"])
    return EvaluationConfig(
        num_classes=len(names),
        class_names=names,
        input_size=tuple(data["predict"]["input_shape"]),
        split=split,
    )


def validate_config(data):
    """返回错误信息列表；空列表表示配置合法。不抛异常。"""
    errors = []
    try:
        dataset = data["dataset"]
        train = data["train"]
        predict = data["predict"]
    except (KeyError, TypeError):
        return ["配置缺少 dataset / train / predict 段落"]

    _check_dataset(dataset, errors)
    num_classes = len(dataset.get("class_names") or [])
    if num_classes >= 2:
        _check_train_extras(train, num_classes, errors)

    #   交给 domain 的 Pydantic 校验：类别数范围、权重长度与非负、
    #   轮次顺序、输入尺寸为正、学习率为正
    for name, builder in (("train", build_training_config),
                          ("predict", build_inference_config),
                          ("evaluation", build_evaluation_config)):
        try:
            builder(data)
        except Exception as exc:                      # pydantic ValidationError 等
            errors.append("[%s] %s" % (name, _first_message(exc)))

    if str(predict.get("backbone", "")) not in KNOWN_BACKBONES:
        errors.append("predict.backbone=%r 未知，可选 %s"
                      % (predict.get("backbone"), list(KNOWN_BACKBONES)))
    if int(predict.get("mix_type", 0)) not in (0, 1, 2, 3):
        errors.append("predict.mix_type 只能是 0/1/2/3，当前 %r"
                      % predict.get("mix_type"))
    if not 0 <= float(predict.get("blend_alpha", 0.7)) <= 1:
        errors.append("predict.blend_alpha 必须在 [0, 1] 之间，当前 %r"
                      % predict.get("blend_alpha"))
    return errors


def _first_message(exc):
    """把 pydantic 的多行报错压成一行，界面弹窗里能看清。"""
    text = str(exc).replace("\n", " ")
    #   pydantic 会带上 "For further information visit https://..." 尾巴
    marker = "For further information"
    if marker in text:
        text = text.split(marker)[0]
    return " ".join(text.split())[:300]


__all__ = [
    "DEEPLAB_BACKBONES",
    "KNOWN_BACKBONES",
    "SEGFORMER_BACKBONES",
    "build_evaluation_config",
    "build_inference_config",
    "build_training_config",
    "engine_of",
    "validate_config",
]
