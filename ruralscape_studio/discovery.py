"""Read-only discovery of legacy RuralScape workspace capabilities."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .domain import ModelEngine, Project


_COMPONENTS = ("VOCdevkit", "logs", "model_data", "img", "img_out", "segformer")
_DIGEST_LENGTH = 12
_MAX_PATH_COMPONENT_LENGTH = 255


@dataclass(frozen=True, slots=True)
class CapabilityFlags:
    vocdevkit: bool = False
    logs: bool = False
    model_data: bool = False
    img: bool = False
    img_out: bool = False
    segformer: bool = False

    def to_dict(self) -> dict[str, bool]:
        return {
            "VOCdevkit": self.vocdevkit,
            "logs": self.logs,
            "model_data": self.model_data,
            "img": self.img,
            "img_out": self.img_out,
            "segformer": self.segformer,
        }


@dataclass(frozen=True, slots=True)
class WorkspaceDiscovery:
    project: Project
    capabilities: CapabilityFlags
    detected_paths: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "detected_paths", MappingProxyType(dict(self.detected_paths))
        )


def _is_within(candidate: Path, parent: Path) -> bool:
    try:
        candidate.relative_to(parent)
    except ValueError:
        return False
    return True


def _project_id(root: Path) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", root.name.casefold()).strip("-") or "project"
    max_slug_length = _MAX_PATH_COMPONENT_LENGTH - _DIGEST_LENGTH - 1
    slug = slug[:max_slug_length].rstrip("-") or "project"
    platform_normalized_root = os.path.normcase(str(root))
    digest = hashlib.sha256(platform_normalized_root.encode("utf-8")).hexdigest()[
        :_DIGEST_LENGTH
    ]
    return f"{slug}-{digest}"


def discover_workspace(
    repo_root: str | Path, app_data_root: str | Path
) -> WorkspaceDiscovery:
    """Inspect known directories and propose a project without creating anything."""

    root = Path(repo_root).expanduser().resolve()
    app_root = Path(app_data_root).expanduser().resolve()
    detected: dict[str, str] = {}
    for name in _COMPONENTS:
        component = root / name
        if not component.is_dir():
            continue
        resolved_component = component.resolve()
        if _is_within(resolved_component, root):
            detected[name] = str(resolved_component)

    flags = CapabilityFlags(
        vocdevkit="VOCdevkit" in detected,
        logs="logs" in detected,
        model_data="model_data" in detected,
        img="img" in detected,
        img_out="img_out" in detected,
        segformer="segformer" in detected,
    )
    project_id = _project_id(root)
    engine = (
        ModelEngine.SEGFORMER_B2
        if flags.segformer
        else ModelEngine.DEEPLAB_V3_PLUS
    )
    app_data_path = (app_root / project_id).resolve()
    if not _is_within(app_data_path, app_root):
        raise ValueError("app_data_path resolves outside app_data_root")
    project = Project(
        id=project_id,
        name=root.name or project_id,
        root_path=str(root),
        app_data_path=str(app_data_path),
        engine=engine,
        metadata={"capabilities": flags.to_dict()},
    )
    return WorkspaceDiscovery(project, flags, detected)


DiscoveryResult = WorkspaceDiscovery

__all__ = [
    "CapabilityFlags",
    "DiscoveryResult",
    "WorkspaceDiscovery",
    "discover_workspace",
]
