"""Pydantic-validated, JSON-serializable domain types for RuralScape Studio."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import field, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import Annotated, cast

from pydantic import AfterValidator, ConfigDict, FiniteFloat, JsonValue, model_validator
from pydantic.dataclasses import dataclass



_CONFIG = ConfigDict(validate_default=True, allow_inf_nan=False)


def utc_now() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""

    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


def _validate_class_id(value: int) -> int:
    if not 0 <= value <= 255:
        raise ValueError("class_id must be in the range 0..255")
    return value


def _validate_color(value: tuple[int, int, int]) -> tuple[int, int, int]:
    if any(not 0 <= channel <= 255 for channel in value):
        raise ValueError("color must contain three RGB byte values")
    return value


def _validate_input_size(value: tuple[int, int]) -> tuple[int, int]:
    if any(part <= 0 for part in value):
        raise ValueError("input_size must contain two positive integers")
    return value


def _validate_batch_size(value: int) -> int:
    if value < 2:
        raise ValueError("batch_size must be at least 2")
    return value


def _validate_num_classes(value: int) -> int:
    if not 1 <= value <= 256:
        raise ValueError("num_classes must be between 1 and 256")
    return value


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return cast(Mapping[str, object], _freeze_json(value))


AwareDateTime = Annotated[datetime, AfterValidator(_as_utc)]
ClassId = Annotated[int, AfterValidator(_validate_class_id)]
RgbColor = Annotated[tuple[int, int, int], AfterValidator(_validate_color)]
InputSize = Annotated[tuple[int, int], AfterValidator(_validate_input_size)]
BatchSize = Annotated[int, AfterValidator(_validate_batch_size)]
NumClasses = Annotated[int, AfterValidator(_validate_num_classes)]
ImmutableJsonMapping = Annotated[
    Mapping[str, JsonValue], AfterValidator(_freeze_mapping)
]
ImmutableIntMapping = Annotated[Mapping[str, int], AfterValidator(_freeze_mapping)]


def _serialize(value: object) -> JsonValue:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, float) and not isfinite(value):
        raise ValueError("persisted floats must be finite")
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _serialize(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    return cast(JsonValue, value)


class SerializableDataclass:
    """Supply a JSON-compatible dictionary for every public domain dataclass."""

    __slots__ = ()

    def to_dict(self) -> dict[str, JsonValue]:
        return cast(dict[str, JsonValue], _serialize(self))


class ModelEngine(str, Enum):
    """Supported semantic-segmentation engines and their stable wire values."""

    DEEPLAB_V3_PLUS = "deeplab_v3_plus"
    SEGFORMER_B2 = "segformer_b2"


@dataclass(config=_CONFIG, slots=True, frozen=True)
class ClassDefinition(SerializableDataclass):
    class_id: ClassId
    name: str
    color: RgbColor
    weight: FiniteFloat = 1.0

    @model_validator(mode="after")
    def validate_definition(self) -> "ClassDefinition":
        if not self.name.strip():
            raise ValueError("name must not be blank")
        if self.weight < 0:
            raise ValueError("weight must be non-negative")
        return self


@dataclass(config=_CONFIG, slots=True, frozen=True)
class Project(SerializableDataclass):
    id: str
    name: str
    root_path: str
    app_data_path: str | None = None
    engine: ModelEngine = ModelEngine.DEEPLAB_V3_PLUS
    metadata: ImmutableJsonMapping = field(default_factory=dict)
    created_at: AwareDateTime = field(default_factory=utc_now)
    updated_at: AwareDateTime = field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_identity(self) -> "Project":
        if not self.id.strip():
            raise ValueError("id must not be blank")
        if not self.name.strip():
            raise ValueError("name must not be blank")
        if not self.root_path.strip():
            raise ValueError("root_path must not be blank")
        return self


@dataclass(config=_CONFIG, slots=True, frozen=True)
class DatasetIssue(SerializableDataclass):
    code: str
    message: str
    severity: str = "warning"
    stem: str | None = None
    image_path: str | None = None
    mask_path: str | None = None


@dataclass(config=_CONFIG, slots=True, frozen=True)
class DatasetProfile(SerializableDataclass):
    root_path: str
    total_images: int = 0
    total_masks: int = 0
    assigned_count: int = 0
    unassigned_count: int = 0
    split_counts: ImmutableIntMapping = field(default_factory=dict)
    class_values: tuple[int, ...] = field(default_factory=tuple)
    issues: tuple[DatasetIssue, ...] = field(default_factory=tuple)
    indexed_at: AwareDateTime = field(default_factory=utc_now)

    @property
    def image_count(self) -> int:
        return self.total_images

    @property
    def mask_count(self) -> int:
        return self.total_masks

    @property
    def train_count(self) -> int:
        return self.split_counts.get("train", 0)

    @property
    def val_count(self) -> int:
        return self.split_counts.get("val", 0)

    @property
    def test_count(self) -> int:
        return self.split_counts.get("test", 0)


@dataclass(config=_CONFIG, slots=True, frozen=True)
class DeepLabTrainingConfig(SerializableDataclass):
    num_classes: NumClasses
    class_weights: tuple[FiniteFloat, ...]
    batch_size: BatchSize = 4
    start_epoch: int = 0
    freeze_epoch: int = 50
    total_epochs: int = 100
    input_size: InputSize = (512, 512)
    learning_rate: FiniteFloat = 7e-3

    @model_validator(mode="after")
    def validate_training_config(self) -> "DeepLabTrainingConfig":
        if len(self.class_weights) != self.num_classes:
            raise ValueError("class_weights length must equal num_classes")
        if any(weight < 0 for weight in self.class_weights):
            raise ValueError("class_weights must be non-negative")
        if not 0 <= self.start_epoch < self.freeze_epoch <= self.total_epochs:
            raise ValueError("start_epoch < freeze_epoch <= total_epochs is required")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        return self


@dataclass(config=_CONFIG, slots=True, frozen=True)
class SegFormerTrainingConfig(SerializableDataclass):
    num_classes: NumClasses
    class_weights: tuple[FiniteFloat, ...]
    batch_size: BatchSize = 4
    start_epoch: int = 0
    total_epochs: int = 100
    input_size: InputSize = (512, 512)
    learning_rate: FiniteFloat = 6e-5

    @model_validator(mode="after")
    def validate_training_config(self) -> "SegFormerTrainingConfig":
        if len(self.class_weights) != self.num_classes:
            raise ValueError("class_weights length must equal num_classes")
        if any(weight < 0 for weight in self.class_weights):
            raise ValueError("class_weights must be non-negative")
        if not 0 <= self.start_epoch < self.total_epochs:
            raise ValueError("start_epoch < total_epochs is required")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        return self


@dataclass(config=_CONFIG, slots=True, frozen=True)
class InferenceConfig(SerializableDataclass):
    engine: ModelEngine
    model_path: str
    input_size: InputSize = (512, 512)
    output_dir: str | None = None
    confidence: FiniteFloat = 0.5
    blend: bool = True

    @model_validator(mode="after")
    def validate_inference_config(self) -> "InferenceConfig":
        if not self.model_path.strip():
            raise ValueError("model_path must not be blank")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        return self


@dataclass(config=_CONFIG, slots=True, frozen=True)
class EvaluationConfig(SerializableDataclass):
    num_classes: NumClasses
    class_names: tuple[str, ...]
    input_size: InputSize = (512, 512)
    split: str = "val"
    output_dir: str | None = None

    @model_validator(mode="after")
    def validate_evaluation_config(self) -> "EvaluationConfig":
        if len(self.class_names) != self.num_classes:
            raise ValueError("class_names length must equal num_classes")
        if self.split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        return self


@dataclass(config=_CONFIG, slots=True, frozen=True)
class AnnotationVersion(SerializableDataclass):
    id: str
    version: int
    mask_path: str
    author: str | None = None
    parent_version_id: str | None = None
    notes: str = ""
    created_at: AwareDateTime = field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_version(self) -> "AnnotationVersion":
        if self.version < 1:
            raise ValueError("version must be at least 1")
        return self


@dataclass(config=_CONFIG, slots=True, frozen=True)
class AnnotationDocument(SerializableDataclass):
    id: str
    project_id: str
    image_path: str
    versions: tuple[AnnotationVersion, ...] = field(default_factory=tuple)
    active_version_id: str | None = None
    created_at: AwareDateTime = field(default_factory=utc_now)
    updated_at: AwareDateTime = field(default_factory=utc_now)


@dataclass(config=_CONFIG, slots=True, frozen=True)
class ModelArtifact(SerializableDataclass):
    id: str
    project_id: str
    run_id: str
    engine: ModelEngine
    path: str
    num_classes: NumClasses
    metadata: ImmutableJsonMapping = field(default_factory=dict)
    created_at: AwareDateTime = field(default_factory=utc_now)


@dataclass(config=_CONFIG, slots=True, frozen=True)
class TaskCommand(SerializableDataclass):
    id: str
    project_id: str
    command: str
    payload: ImmutableJsonMapping = field(default_factory=dict)
    status: str = "queued"
    created_at: AwareDateTime = field(default_factory=utc_now)
    updated_at: AwareDateTime = field(default_factory=utc_now)


@dataclass(config=_CONFIG, slots=True, frozen=True)
class TaskEvent(SerializableDataclass):
    task_id: str
    sequence: int
    status: str
    message: str = ""
    payload: ImmutableJsonMapping = field(default_factory=dict)
    timestamp: AwareDateTime = field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_event(self) -> "TaskEvent":
        if self.sequence < 0:
            raise ValueError("sequence must not be negative")
        return self


@dataclass(config=_CONFIG, slots=True, frozen=True)
class PackageManifest(SerializableDataclass):
    schema_version: int
    project: Project
    classes: tuple[ClassDefinition, ...] = field(default_factory=tuple)
    artifacts: tuple[ModelArtifact, ...] = field(default_factory=tuple)
    annotations: tuple[AnnotationDocument, ...] = field(default_factory=tuple)
    created_at: AwareDateTime = field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_manifest(self) -> "PackageManifest":
        if self.schema_version < 1:
            raise ValueError("schema_version must be at least 1")
        return self


__all__ = [
    "AnnotationDocument",
    "AnnotationVersion",
    "ClassDefinition",
    "DatasetIssue",
    "DatasetProfile",
    "DeepLabTrainingConfig",
    "EvaluationConfig",
    "InferenceConfig",
    "ModelArtifact",
    "ModelEngine",
    "PackageManifest",
    "Project",
    "SegFormerTrainingConfig",
    "TaskCommand",
    "TaskEvent",
]




