from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest

from ruralscape_studio import ModelEngine, Project, TaskEvent
from ruralscape_studio.storage import StudioRepository


NOW = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def make_project(project_id: str = "p1", **changes: object) -> Project:
    values: dict[str, object] = {
        "id": project_id,
        "name": f"Project {project_id}",
        "root_path": f"/work/{project_id}",
        "app_data_path": f"/data/{project_id}",
        "engine": ModelEngine.SEGFORMER_B2,
        "metadata": {"labels": ["sky", "road"]},
        "created_at": NOW,
        "updated_at": NOW,
    }
    values.update(changes)
    return Project(**values)


def test_repository_initializes_all_tables_and_sqlite_pragmas(tmp_path) -> None:
    database = tmp_path / "state" / "studio.sqlite3"

    StudioRepository(database)

    with sqlite3.connect(database) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]

    assert {"projects", "tasks", "runs", "artifacts", "annotations"} <= tables
    assert journal_mode.lower() == "wal"


def test_project_crud_round_trips_domain_objects_and_strict_json(tmp_path) -> None:
    repository = StudioRepository(tmp_path / "studio.sqlite3")
    first = make_project()
    second = make_project(
        "p2",
        name="Second",
        updated_at=NOW + timedelta(seconds=1),
        metadata={"ratio": 0.5, "nested": {"enabled": True}},
    )

    repository.create_project(first)
    repository.create_project(second)

    assert repository.get_project("p1") == first
    assert repository.list_projects() == [second, first]

    changed = make_project(
        name="Renamed",
        engine=ModelEngine.DEEPLAB_V3_PLUS,
        updated_at=NOW + timedelta(minutes=1),
    )
    assert repository.update_project(changed) is True
    assert repository.get_project("p1") == changed
    assert repository.delete_project("p1") is True
    assert repository.delete_project("missing") is False
    assert repository.get_project("p1") is None

    with sqlite3.connect(repository.db_path) as connection:
        stored = connection.execute(
            "SELECT snapshot_json FROM projects WHERE id = 'p2'"
        ).fetchone()[0]
    assert json.loads(stored) == second.to_dict()
    assert "NaN" not in stored


def test_create_project_rejects_duplicate_id(tmp_path) -> None:
    repository = StudioRepository(tmp_path / "studio.sqlite3")
    repository.create_project(make_project())

    with pytest.raises(sqlite3.IntegrityError):
        repository.create_project(make_project())


def test_strict_json_encoder_rejects_non_finite_generic_snapshots(tmp_path) -> None:
    repository = StudioRepository(tmp_path / "studio.sqlite3")

    with pytest.raises(ValueError, match="JSON"):
        repository.persist_run("r1", {"loss": float("nan")})


def test_task_events_are_upserted_and_stale_snapshots_cannot_regress(tmp_path) -> None:
    repository = StudioRepository(tmp_path / "studio.sqlite3")
    repository.create_project(make_project())
    old = TaskEvent(
        task_id="task-1",
        sequence=1,
        status="running",
        message="epoch 1",
        payload={"progress": 0.1},
        timestamp=NOW,
    )
    new = TaskEvent(
        task_id="task-1",
        sequence=2,
        status="complete",
        message="done",
        payload={"progress": 1.0},
        timestamp=NOW + timedelta(seconds=2),
    )

    assert repository.upsert_task_event(old, project_id="p1") is True
    assert repository.upsert_task_event(new, project_id="p1") is True
    assert repository.upsert_task_event(old, project_id="p1") is False

    assert repository.get_task("task-1") == new
    assert repository.list_tasks(project_id="p1") == [new]


def test_tasks_are_ordered_newest_with_deterministic_ties_and_limit(tmp_path) -> None:
    repository = StudioRepository(tmp_path / "studio.sqlite3")
    events = [
        TaskEvent(task_id="b", sequence=0, status="queued", timestamp=NOW),
        TaskEvent(
            task_id="c",
            sequence=0,
            status="queued",
            timestamp=NOW + timedelta(seconds=1),
        ),
        TaskEvent(task_id="a", sequence=0, status="queued", timestamp=NOW),
    ]
    for event in events:
        repository.upsert_task_event(event)

    assert [event.task_id for event in repository.list_tasks()] == ["c", "a", "b"]
    assert [event.task_id for event in repository.list_tasks(limit=2)] == ["c", "a"]


def test_connection_per_operation_supports_concurrent_writers(tmp_path) -> None:
    repository = StudioRepository(tmp_path / "studio.sqlite3")

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda index: repository.create_project(make_project(f"p{index}")), range(20)))

    assert len(repository.list_projects()) == 20


def test_foreign_keys_are_enabled_for_every_operation(tmp_path) -> None:
    repository = StudioRepository(tmp_path / "studio.sqlite3")

    with pytest.raises(sqlite3.IntegrityError):
        repository.upsert_task_event(
            TaskEvent(task_id="orphan", sequence=0, status="queued"),
            project_id="missing-project",
        )
