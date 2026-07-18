"""workstation_config.json 校验层的测试。"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ruralscape_studio.domain import ModelEngine
from workstation.config import DEFAULT_CONFIG, Config
from workstation.config_schema import (
    build_evaluation_config,
    build_inference_config,
    build_training_config,
    engine_of,
    validate_config,
)


def _cfg():
    return copy.deepcopy(DEFAULT_CONFIG)


def test_default_config_is_valid():
    assert validate_config(_cfg()) == []


def test_shipped_config_file_is_valid():
    """项目根目录里实际在用的 workstation_config.json 必须合法。"""
    path = Path(__file__).resolve().parents[1] / "workstation_config.json"
    if not path.exists():
        pytest.skip("workstation_config.json 不存在")
    from workstation.config import _merge

    data = _merge(DEFAULT_CONFIG, json.loads(path.read_text(encoding="utf-8")))
    assert validate_config(data) == []


def test_cls_weights_length_mismatch_is_caught():
    data = _cfg()
    data["train"]["cls_weights"] = [1.0, 1.0, 1.0]      # 类别数是 5
    errors = validate_config(data)
    assert any("cls_weights" in e for e in errors)


def test_class_colors_must_match_class_names():
    data = _cfg()
    data["dataset"]["class_colors"] = data["dataset"]["class_colors"][:3]
    assert any("class_colors" in e for e in validate_config(data))


def test_remove_classes_out_of_range_is_caught():
    data = _cfg()
    data["dataset"]["remove_classes"] = [9]
    assert any("remove_classes" in e for e in validate_config(data))


def test_epoch_order_is_enforced():
    data = _cfg()
    data["train"]["freeze_epoch"] = 90                  # > unfreeze_epoch(80)
    assert validate_config(data)


def test_freeze_epoch_ignored_when_freeze_train_off():
    """不冻结训练时 freeze_epoch 不参与流程，不该因为它报错。"""
    data = _cfg()
    data["train"]["freeze_train"] = False
    data["train"]["freeze_epoch"] = 999
    assert validate_config(data) == []


def test_unknown_backbone_is_caught():
    data = _cfg()
    data["train"]["backbone"] = "resnet50"
    assert any("backbone" in e for e in validate_config(data))


def test_bad_downsample_factor_is_caught():
    data = _cfg()
    data["train"]["downsample_factor"] = 32
    assert any("downsample_factor" in e for e in validate_config(data))


def test_bad_optimizer_is_caught():
    data = _cfg()
    data["train"]["optimizer_type"] = "lion"
    assert any("optimizer_type" in e for e in validate_config(data))


def test_negative_learning_rate_is_caught():
    data = _cfg()
    data["train"]["init_lr"] = -1.0
    assert validate_config(data)


def test_batch_size_below_two_is_caught():
    data = _cfg()
    data["train"]["unfreeze_batch_size"] = 1
    assert validate_config(data)


def test_blend_alpha_out_of_range_is_caught():
    data = _cfg()
    data["predict"]["blend_alpha"] = 1.5
    assert any("blend_alpha" in e for e in validate_config(data))


def test_mix_type_out_of_range_is_caught():
    data = _cfg()
    data["predict"]["mix_type"] = 7
    assert any("mix_type" in e for e in validate_config(data))


def test_missing_sections_reported_without_crashing():
    assert validate_config({}) == ["配置缺少 dataset / train / predict 段落"]


def test_engine_mapping():
    assert engine_of("mobilenet") is ModelEngine.DEEPLAB_V3_PLUS
    assert engine_of("xception") is ModelEngine.DEEPLAB_V3_PLUS
    assert engine_of("segformer-b2") is ModelEngine.SEGFORMER_B2


def test_builders_map_fields_faithfully():
    data = _cfg()
    train = build_training_config(data)
    assert train.num_classes == 5
    assert train.class_weights == tuple(data["train"]["cls_weights"])
    assert train.batch_size == data["train"]["unfreeze_batch_size"]
    assert train.start_epoch == data["train"]["init_epoch"]
    assert train.freeze_epoch == data["train"]["freeze_epoch"]
    assert train.total_epochs == data["train"]["unfreeze_epoch"]
    assert train.input_size == tuple(data["train"]["input_shape"])
    assert train.learning_rate == data["train"]["init_lr"]

    inference = build_inference_config(data)
    assert inference.engine is ModelEngine.SEGFORMER_B2   # 默认权重是 segformer-b2
    assert inference.model_path == data["predict"]["model_path"]

    evaluation = build_evaluation_config(data)
    assert evaluation.class_names == tuple(data["dataset"]["class_names"])
    assert evaluation.num_classes == 5


def test_empty_cls_weights_means_equal_weights():
    data = _cfg()
    data["train"]["cls_weights"] = []
    assert build_training_config(data).class_weights == (1.0,) * 5


def test_validate_or_raise_message_lists_every_problem(tmp_path):
    config = Config(path=str(tmp_path / "missing.json"))
    config.data["train"]["backbone"] = "resnet50"
    config.data["train"]["optimizer_type"] = "lion"
    with pytest.raises(ValueError) as excinfo:
        config.validate_or_raise()
    message = str(excinfo.value)
    assert "backbone" in message and "optimizer_type" in message
    assert "2 处问题" in message


def test_valid_config_passes_validate_or_raise(tmp_path):
    Config(path=str(tmp_path / "missing.json")).validate_or_raise()


def test_load_warns_but_does_not_crash_on_broken_config(tmp_path, capsys):
    """启动阶段遇到坏配置只告警，不能让工作站打不开。"""
    path = tmp_path / "workstation_config.json"
    path.write_text(json.dumps({"train": {"backbone": "resnet50"}}), encoding="utf-8")
    config = Config(path=str(path))
    assert config.data["train"]["backbone"] == "resnet50"
    assert "[配置告警]" in capsys.readouterr().out
