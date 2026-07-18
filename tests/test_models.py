from __future__ import annotations

import json
from dataclasses import fields
from datetime import datetime, timezone

import pytest

import ruralscape_studio as studio


PUBLIC_TYPES = {
    "ModelEngine",
    "Project",
    "ClassDefinition",
    "DatasetProfile",
    "DatasetIssue",
    "DeepLabTrainingConfig",
    "SegFormerTrainingConfig",
    "InferenceConfig",
    "EvaluationConfig",
    "AnnotationDocument",
    "AnnotationVersion",
    "ModelArtifact",
    "TaskCommand",
    "TaskEvent",
    "PackageManifest",
}


def test_public_domain_types_are_exported() -> None:
    assert PUBLIC_TYPES <= set(dir(studio))


def test_model_engines_use_stable_serialized_values() -> None:
    assert studio.ModelEngine.DEEPLAB_V3_PLUS.value == "deeplab_v3_plus"
    assert studio.ModelEngine.SEGFORMER_B2.value == "segformer_b2"


@pytest.mark.parametrize("class_id", [-1, 256])
def test_class_definition_rejects_ids_outside_byte_range(class_id: int) -> None:
    with pytest.raises(ValueError, match="0..255"):
        studio.ClassDefinition(class_id=class_id, name="invalid", color=(0, 0, 0))


def test_class_definition_rejects_invalid_rgb_color() -> None:
    with pytest.raises(ValueError, match="color"):
        studio.ClassDefinition(class_id=1, name="road", color=(0, 0, 300))


@pytest.mark.parametrize("batch_size", [0, 1])
def test_training_configs_require_batch_size_at_least_two(batch_size: int) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        studio.DeepLabTrainingConfig(num_classes=2, class_weights=[1, 1], batch_size=batch_size)

    with pytest.raises(ValueError, match="batch_size"):
        studio.SegFormerTrainingConfig(num_classes=2, class_weights=[1, 1], batch_size=batch_size)


def test_training_configs_require_one_weight_per_class() -> None:
    with pytest.raises(ValueError, match="class_weights"):
        studio.DeepLabTrainingConfig(num_classes=3, class_weights=[1, 1])

    with pytest.raises(ValueError, match="class_weights"):
        studio.SegFormerTrainingConfig(num_classes=3, class_weights=[1, 1])


def test_deeplab_epoch_order_is_validated() -> None:
    with pytest.raises(ValueError, match="start_epoch.*freeze_epoch.*total_epochs"):
        studio.DeepLabTrainingConfig(
            num_classes=2,
            class_weights=[1, 1],
            start_epoch=10,
            freeze_epoch=5,
            total_epochs=20,
        )


def test_segformer_epoch_order_is_validated() -> None:
    with pytest.raises(ValueError, match="start_epoch.*total_epochs"):
        studio.SegFormerTrainingConfig(
            num_classes=2,
            class_weights=[1, 1],
            start_epoch=10,
            total_epochs=10,
        )


@pytest.mark.parametrize("input_size", [(0, 512), (512, -1), (512,), (512, 512, 3)])
def test_invalid_input_sizes_are_rejected(input_size: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="input_size"):
        studio.InferenceConfig(
            engine=studio.ModelEngine.DEEPLAB_V3_PLUS,
            model_path="model.pth",
            input_size=input_size,
        )


def test_timestamps_must_be_timezone_aware() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        studio.Project(id="p1", name="demo", root_path="/repo", created_at=datetime.now())


def test_domain_objects_convert_to_json_serializable_dictionaries() -> None:
    project = studio.Project(
        id="p1",
        name="demo",
        root_path="/repo",
        engine=studio.ModelEngine.SEGFORMER_B2,
        created_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    class_definition = studio.ClassDefinition(0, "background", (0, 0, 0))
    manifest = studio.PackageManifest(
        schema_version=1,
        project=project,
        classes=[class_definition],
    )

    encoded = json.loads(json.dumps(manifest.to_dict(), allow_nan=False))

    assert encoded["project"]["engine"] == "segformer_b2"
    assert encoded["project"]["created_at"] == "2026-01-02T00:00:00+00:00"
    assert encoded["classes"][0]["color"] == [0, 0, 0]


def test_all_dataclass_timestamp_defaults_are_timezone_aware() -> None:
    project = studio.Project(id="p1", name="demo", root_path="/repo")
    profile = studio.DatasetProfile(root_path="/dataset")
    version = studio.AnnotationVersion(id="v1", version=1, mask_path="mask.png")
    document = studio.AnnotationDocument(id="a1", project_id="p1", image_path="image.png")
    artifact = studio.ModelArtifact(
        id="m1",
        project_id="p1",
        run_id="r1",
        engine=studio.ModelEngine.DEEPLAB_V3_PLUS,
        path="model.pth",
        num_classes=2,
    )
    command = studio.TaskCommand(id="t1", project_id="p1", command="train")
    event = studio.TaskEvent(task_id="t1", sequence=1, status="running")
    manifest = studio.PackageManifest(schema_version=1, project=project)

    timestamps = (
        project.created_at,
        project.updated_at,
        profile.indexed_at,
        version.created_at,
        document.created_at,
        document.updated_at,
        artifact.created_at,
        command.created_at,
        command.updated_at,
        event.timestamp,
        manifest.created_at,
    )
    for value in timestamps:
        assert value.tzinfo is not None
        assert value.utcoffset() is not None


@pytest.mark.parametrize(
    "factory",
    [
        lambda value: studio.Project(
            id="p1", name="demo", root_path="/repo", created_at=value
        ),
        lambda value: studio.Project(
            id="p1", name="demo", root_path="/repo", updated_at=value
        ),
        lambda value: studio.DatasetProfile(root_path="/dataset", indexed_at=value),
        lambda value: studio.AnnotationVersion(
            id="v1", version=1, mask_path="mask.png", created_at=value
        ),
        lambda value: studio.AnnotationDocument(
            id="a1", project_id="p1", image_path="image.png", created_at=value
        ),
        lambda value: studio.AnnotationDocument(
            id="a1", project_id="p1", image_path="image.png", updated_at=value
        ),
        lambda value: studio.ModelArtifact(
            id="m1",
            project_id="p1",
            run_id="r1",
            engine=studio.ModelEngine.DEEPLAB_V3_PLUS,
            path="model.pth",
            num_classes=2,
            created_at=value,
        ),
        lambda value: studio.TaskCommand(
            id="t1", project_id="p1", command="train", created_at=value
        ),
        lambda value: studio.TaskCommand(
            id="t1", project_id="p1", command="train", updated_at=value
        ),
        lambda value: studio.TaskEvent(
            task_id="t1", sequence=1, status="running", timestamp=value
        ),
        lambda value: studio.PackageManifest(
            schema_version=1,
            project=studio.Project(id="p1", name="demo", root_path="/repo"),
            created_at=value,
        ),
    ],
)
def test_all_timestamp_fields_reject_naive_values(factory: object) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        factory(datetime.now())  # type: ignore[operator]


def test_required_dataclasses_have_only_serializable_field_kinds() -> None:
    project = studio.Project(id="p1", name="demo", root_path="/repo")
    assert {field.name for field in fields(project)} >= {"id", "name", "root_path", "created_at"}
    json.dumps(project.to_dict())

def test_failed_reassignment_cannot_corrupt_validated_state() -> None:
    config = studio.DeepLabTrainingConfig(num_classes=2, class_weights=[1, 1])

    with pytest.raises((AttributeError, ValueError)):
        config.start_epoch = config.total_epochs  # type: ignore[misc]

    assert config.start_epoch == 0


def test_domain_instances_do_not_gain_a_mixin_dict() -> None:
    definition = studio.ClassDefinition(0, "background", (0, 0, 0))
    assert not hasattr(definition, "__dict__")


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), float("-inf")])
def test_persisted_float_fields_reject_non_finite_values(non_finite: float) -> None:
    with pytest.raises(ValueError):
        studio.ClassDefinition(0, "background", (0, 0, 0), weight=non_finite)

    with pytest.raises(ValueError):
        studio.DeepLabTrainingConfig(num_classes=2, class_weights=[1, non_finite])

    with pytest.raises(ValueError):
        studio.SegFormerTrainingConfig(num_classes=2, class_weights=[1, non_finite])

    with pytest.raises(ValueError):
        studio.DeepLabTrainingConfig(
            num_classes=2, class_weights=[1, 1], learning_rate=non_finite
        )

    with pytest.raises(ValueError):
        studio.SegFormerTrainingConfig(
            num_classes=2, class_weights=[1, 1], learning_rate=non_finite
        )

    with pytest.raises(ValueError):
        studio.InferenceConfig(
            engine=studio.ModelEngine.DEEPLAB_V3_PLUS,
            model_path="model.pth",
            confidence=non_finite,
        )

    with pytest.raises(ValueError):
        studio.Project(
            id="p1",
            name="demo",
            root_path="/repo",
            metadata={"value": non_finite},
        )


def test_legacy_models_module_reexports_identical_public_types() -> None:
    from ruralscape_studio import models

    for name in PUBLIC_TYPES:
        assert getattr(models, name) is getattr(studio, name)

def test_sequence_fields_are_deeply_immutable_but_accept_lists() -> None:
    issue = studio.DatasetIssue(code="missing-mask", message="Mask is missing")
    version = studio.AnnotationVersion(id="v1", version=1, mask_path="mask.png")
    artifact = studio.ModelArtifact(
        id="m1",
        project_id="p1",
        run_id="r1",
        engine=studio.ModelEngine.DEEPLAB_V3_PLUS,
        path="model.pth",
        num_classes=2,
    )
    definition = studio.ClassDefinition(0, "background", (0, 0, 0))
    profile = studio.DatasetProfile(
        root_path="/dataset", class_values=[0, 1], issues=[issue]
    )
    document = studio.AnnotationDocument(
        id="a1", project_id="p1", image_path="image.png", versions=[version]
    )
    manifest = studio.PackageManifest(
        schema_version=1,
        project=studio.Project(id="p1", name="demo", root_path="/repo"),
        classes=[definition],
        artifacts=[artifact],
        annotations=[document],
    )
    sequences = (
        studio.DeepLabTrainingConfig(num_classes=2, class_weights=[1, 1]).class_weights,
        studio.SegFormerTrainingConfig(num_classes=2, class_weights=[1, 1]).class_weights,
        studio.EvaluationConfig(num_classes=2, class_names=["background", "road"]).class_names,
        profile.class_values,
        profile.issues,
        document.versions,
        manifest.classes,
        manifest.artifacts,
        manifest.annotations,
    )

    for sequence in sequences:
        assert isinstance(sequence, tuple)
        with pytest.raises(AttributeError):
            sequence.append(None)  # type: ignore[attr-defined]


def test_mapping_fields_and_nested_json_values_are_deeply_immutable() -> None:
    project = studio.Project(
        id="p1",
        name="demo",
        root_path="/repo",
        metadata={"nested": {"count": 1}, "items": ["a", "b"]},
    )
    profile = studio.DatasetProfile(root_path="/dataset", split_counts={"train": 3})
    artifact = studio.ModelArtifact(
        id="m1",
        project_id="p1",
        run_id="r1",
        engine=studio.ModelEngine.DEEPLAB_V3_PLUS,
        path="model.pth",
        num_classes=2,
        metadata={"metrics": {"miou": 0.5}},
    )
    command = studio.TaskCommand(
        id="t1", project_id="p1", command="train", payload={"epochs": [1, 2]}
    )
    event = studio.TaskEvent(
        task_id="t1", sequence=1, status="running", payload={"progress": {"step": 1}}
    )

    for mapping in (
        project.metadata,
        profile.split_counts,
        artifact.metadata,
        command.payload,
        event.payload,
    ):
        with pytest.raises(TypeError):
            mapping["new"] = "value"  # type: ignore[index]

    with pytest.raises(TypeError):
        project.metadata["nested"]["count"] = 2  # type: ignore[index]
    with pytest.raises(AttributeError):
        project.metadata["items"].append("c")  # type: ignore[union-attr]


def test_immutable_collections_serialize_as_strict_ordinary_json() -> None:
    project = studio.Project(
        id="p1",
        name="demo",
        root_path="/repo",
        metadata={"nested": {"values": [1, 2, 3]}},
    )
    profile = studio.DatasetProfile(
        root_path="/dataset", split_counts={"train": 3}, class_values=[0, 1]
    )
    config = studio.DeepLabTrainingConfig(num_classes=2, class_weights=[1, 0.5])

    encoded = json.loads(
        json.dumps(
            {
                "project": project.to_dict(),
                "profile": profile.to_dict(),
                "config": config.to_dict(),
            },
            allow_nan=False,
        )
    )

    assert encoded["project"]["metadata"] == {"nested": {"values": [1, 2, 3]}}
    assert encoded["profile"]["split_counts"] == {"train": 3}
    assert encoded["profile"]["class_values"] == [0, 1]
    assert encoded["config"]["class_weights"] == [1.0, 0.5]

