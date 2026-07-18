from __future__ import annotations

import re
from pathlib import Path

from ruralscape_studio import ModelEngine, Project
from ruralscape_studio.discovery import CapabilityFlags, discover_workspace


def test_discovery_detects_known_workspace_components_without_mutation(tmp_path) -> None:
    repo = tmp_path / "My Rural Repo"
    app_data = tmp_path / "application-data"
    repo.mkdir()
    for name in ("VOCdevkit", "logs", "model_data", "img", "segformer"):
        (repo / name).mkdir()
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    result = discover_workspace(repo, app_data)

    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert isinstance(result.project, Project)
    assert re.fullmatch(r"my-rural-repo-[0-9a-f]{12}", result.project.id)
    assert result.project.name == "My Rural Repo"
    assert result.project.root_path == str(repo.resolve())
    assert result.project.app_data_path == str(
        (app_data / result.project.id).resolve()
    )
    assert result.project.engine is ModelEngine.SEGFORMER_B2
    assert result.capabilities == CapabilityFlags(
        vocdevkit=True,
        logs=True,
        model_data=True,
        img=True,
        img_out=False,
        segformer=True,
    )
    assert set(result.detected_paths) == {
        "VOCdevkit",
        "logs",
        "model_data",
        "img",
        "segformer",
    }
    assert before == after
    assert not app_data.exists()


def test_discovery_ignores_files_named_like_expected_directories(tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "logs").write_text("not a directory", encoding="utf-8")

    result = discover_workspace(repo, tmp_path / "data")

    assert result.capabilities == CapabilityFlags()
    assert result.project.engine is ModelEngine.DEEPLAB_V3_PLUS


def test_discovery_project_id_is_stable_and_has_a_nonempty_fallback(tmp_path) -> None:
    repo = tmp_path / "---"
    repo.mkdir()

    first = discover_workspace(repo, tmp_path / "data")
    second = discover_workspace(Path(str(repo)), tmp_path / "data")

    assert first.project.id == second.project.id
    assert first.project.id


def test_capability_flags_offer_named_mapping_for_serialization() -> None:
    flags = CapabilityFlags(vocdevkit=True, img_out=True)

    assert flags.to_dict() == {
        "VOCdevkit": True,
        "logs": False,
        "model_data": False,
        "img": False,
        "img_out": True,
        "segformer": False,
    }
