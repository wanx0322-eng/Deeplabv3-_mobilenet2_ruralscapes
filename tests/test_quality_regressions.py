from __future__ import annotations

import importlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from PIL import Image

import ruralscape_studio as studio
from ruralscape_studio import dataset as dataset_module
from ruralscape_studio.discovery import discover_workspace
from ruralscape_studio.storage import StudioRepository


NOW = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def _project(project_id: str = "p1", **changes: object) -> studio.Project:
    values: dict[str, object] = {
        "id": project_id,
        "name": f"Project {project_id}",
        "root_path": f"/work/{project_id}",
        "app_data_path": f"/data/{project_id}",
        "engine": studio.ModelEngine.SEGFORMER_B2,
        "metadata": {"labels": ["sky", "road"]},
        "created_at": NOW,
        "updated_at": NOW,
    }
    values.update(changes)
    return studio.Project(**values)


def _make_voc(root: Path) -> Path:
    voc = root / "VOCdevkit" / "VOC2007"
    (voc / "JPEGImages").mkdir(parents=True)
    (voc / "SegmentationClass").mkdir()
    (voc / "ImageSets" / "Segmentation").mkdir(parents=True)
    return voc


def _save_pair(voc: Path, stem: str, size: tuple[int, int] = (4, 3)) -> None:
    Image.new("RGB", size, (10, 20, 30)).save(voc / "JPEGImages" / f"{stem}.png")
    Image.new("L", size, 0).save(voc / "SegmentationClass" / f"{stem}.png")


def test_same_basename_roots_get_distinct_readable_ids_and_app_paths(tmp_path) -> None:
    first_root = tmp_path / "one" / "shared-name"
    second_root = tmp_path / "two" / "shared-name"
    first_root.mkdir(parents=True)
    second_root.mkdir(parents=True)

    first = discover_workspace(first_root, tmp_path / "data")
    second = discover_workspace(second_root, tmp_path / "data")

    assert first.project.id.startswith("shared-name-")
    assert second.project.id.startswith("shared-name-")
    assert first.project.id != second.project.id
    assert Path(first.project.app_data_path).name == first.project.id
    assert Path(second.project.app_data_path).name == second.project.id


def test_discovery_ignores_capability_symlink_that_escapes_repo(tmp_path) -> None:
    repo = tmp_path / "repo"
    outside = tmp_path / "outside-logs"
    repo.mkdir()
    outside.mkdir()
    try:
        (repo / "logs").symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks unavailable: {error}")

    result = discover_workspace(repo, tmp_path / "data")

    assert result.capabilities.logs is False
    assert "logs" not in result.detected_paths


def test_discovery_rejects_app_project_path_symlink_escape(tmp_path) -> None:
    repo = tmp_path / "repo"
    app_data = tmp_path / "data"
    outside = tmp_path / "outside"
    repo.mkdir()
    app_data.mkdir()
    outside.mkdir()
    project_id = discover_workspace(repo, app_data).project.id
    try:
        (app_data / project_id).symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks unavailable: {error}")

    with pytest.raises(ValueError, match="app_data_path.*outside app_data_root"):
        discover_workspace(repo, app_data)


def test_equal_sequence_task_event_cannot_overwrite_snapshot(tmp_path) -> None:
    repository = StudioRepository(tmp_path / "studio.sqlite3")
    accepted = studio.TaskEvent(
        task_id="task-1", sequence=2, status="complete", message="accepted", timestamp=NOW
    )
    duplicate = studio.TaskEvent(
        task_id="task-1",
        sequence=2,
        status="failed",
        message="must not replace",
        timestamp=NOW + timedelta(seconds=1),
    )

    assert repository.upsert_task_event(accepted) is True
    assert repository.upsert_task_event(duplicate) is False
    assert repository.get_task("task-1") == accepted


def test_project_upsert_does_not_swallow_unrelated_integrity_errors(tmp_path) -> None:
    repository = StudioRepository(tmp_path / "studio.sqlite3")
    with sqlite3.connect(repository.db_path) as connection:
        connection.executescript(
            """
            CREATE TRIGGER reject_blocked_project
            BEFORE INSERT ON projects
            WHEN NEW.id = 'blocked'
            BEGIN
                SELECT RAISE(ABORT, 'blocked by policy');
            END;
            """
        )

    with pytest.raises(sqlite3.IntegrityError, match="blocked by policy"):
        repository.upsert_project(_project("blocked"))


def test_project_upsert_inserts_then_updates_with_one_public_operation(tmp_path) -> None:
    repository = StudioRepository(tmp_path / "studio.sqlite3")
    first = _project()
    changed = _project(name="Renamed", updated_at=NOW + timedelta(minutes=1))

    assert repository.upsert_project(first) == first
    assert repository.upsert_project(changed) == changed
    assert repository.get_project("p1") == changed


@pytest.mark.parametrize(
    ("snapshot", "message"),
    [
        ('{"id":"p1","id":"p2"}', "duplicate JSON key"),
        (json.dumps({**_project().to_dict(), "unexpected": True}), "unknown fields"),
        ("{not-json", "Expecting property name"),
        ("[]", "JSON object"),
    ],
)
def test_project_reads_reject_corrupt_or_ambiguous_json(
    tmp_path, snapshot: str, message: str
) -> None:
    repository = StudioRepository(tmp_path / "studio.sqlite3")
    repository.create_project(_project())
    with sqlite3.connect(repository.db_path) as connection:
        connection.execute(
            "UPDATE projects SET snapshot_json = ? WHERE id = 'p1'", (snapshot,)
        )

    with pytest.raises(ValueError, match=message):
        repository.get_project("p1")


def test_task_read_rejects_payload_with_wrong_json_shape(tmp_path) -> None:
    repository = StudioRepository(tmp_path / "studio.sqlite3")
    event = studio.TaskEvent(task_id="t1", sequence=1, status="running", timestamp=NOW)
    repository.upsert_task_event(event)
    malformed = event.to_dict()
    malformed["payload"] = []
    with sqlite3.connect(repository.db_path) as connection:
        connection.execute(
            "UPDATE tasks SET snapshot_json = ? WHERE task_id = 't1'",
            (json.dumps(malformed),),
        )

    with pytest.raises(ValueError):
        repository.get_task("t1")


def test_mask_value_helper_processes_multiple_bounded_chunks(tmp_path) -> None:
    mask_path = tmp_path / "mask.png"
    mask = Image.new("L", (9, 3), 0)
    mask.putdata([index % 4 for index in range(27)])
    mask.save(mask_path)

    with Image.open(mask_path) as opened:
        chunks = list(dataset_module._iter_mask_value_chunks(opened, max_pixels=7))

    assert len(chunks) > 1
    assert all(len(chunk) <= 7 for chunk in chunks)
    assert set().union(*(set(chunk) for chunk in chunks)) == {0, 1, 2, 3}


def test_dataset_scan_does_not_materialize_get_flattened_data(tmp_path, monkeypatch) -> None:
    voc = _make_voc(tmp_path)
    _save_pair(voc, "one")

    def forbidden_flatten(*args, **kwargs):
        raise AssertionError("full flattened mask allocation is forbidden")

    monkeypatch.setattr(Image.Image, "get_flattened_data", forbidden_flatten)

    profile = studio.inspect_dataset(voc)

    assert profile.class_values == (0,)


def test_dataset_reports_cross_split_leakage_and_unknown_members(tmp_path) -> None:
    voc = _make_voc(tmp_path)
    for stem in ("shared", "train-only", "unused"):
        _save_pair(voc, stem)
    split_dir = voc / "ImageSets" / "Segmentation"
    (split_dir / "train.txt").write_text("shared\ntrain-only\nghost\n", encoding="utf-8")
    (split_dir / "val.txt").write_text("shared\n", encoding="utf-8")

    profile = studio.inspect_dataset(voc)
    issue_pairs = {(issue.code, issue.stem) for issue in profile.issues}

    assert ("cross-split-leakage", "shared") in issue_pairs
    assert ("unknown-split-member", "ghost") in issue_pairs
    assert profile.assigned_count == 2
    assert profile.unassigned_count == 1


def test_dataset_fully_decodes_images_before_accepting_them(tmp_path) -> None:
    voc = _make_voc(tmp_path)
    _save_pair(voc, "truncated", size=(20, 20))
    image_path = voc / "JPEGImages" / "truncated.png"
    Image.new("RGB", (20, 20), (10, 20, 30)).save(image_path, format="BMP")
    image_path.write_bytes(image_path.read_bytes()[:54])

    profile = studio.inspect_dataset(voc)

    assert ("unreadable-image", "truncated") in {
        (issue.code, issue.stem) for issue in profile.issues
    }


def test_package_import_executes_and_compatibility_aliases_are_exported() -> None:
    imported = importlib.import_module("ruralscape_studio")

    assert imported.SQLiteRepository is imported.StudioRepository
    assert imported.DiscoveryResult is imported.WorkspaceDiscovery
    assert imported.scan_dataset is imported.inspect_dataset
