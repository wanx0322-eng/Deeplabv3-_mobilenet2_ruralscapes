"""SQLite persistence for RuralScape Studio application state."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, TypeVar

from .domain import Project, TaskEvent


_SnapshotType = TypeVar("_SnapshotType", Project, TaskEvent)


def _strict_json(value: object) -> str:
    """Encode a deterministic snapshot and reject JSON's non-finite extensions."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    decoded: dict[str, object] = {}
    for key, value in pairs:
        if key in decoded:
            raise ValueError(f"duplicate JSON key: {key}")
        decoded[key] = value
    return decoded


def _reject_non_finite_json(value: str) -> object:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _load_snapshot(value: str, model_type: type[_SnapshotType]) -> _SnapshotType:
    decoded = json.loads(
        value,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_non_finite_json,
    )
    if not isinstance(decoded, dict):
        raise ValueError("snapshot must be a JSON object")
    allowed_fields = {item.name for item in fields(model_type)}
    unknown_fields = sorted(set(decoded) - allowed_fields)
    if unknown_fields:
        raise ValueError(f"snapshot contains unknown fields: {', '.join(unknown_fields)}")
    return model_type(**decoded)


class StudioRepository:
    """A small thread-safe repository using one SQLite connection per operation."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    @contextmanager
    def _operation(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        """Create the schema idempotently and select WAL journaling."""

        with self._operation() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    root_path TEXT NOT NULL,
                    engine TEXT NOT NULL,
                    snapshot_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    project_id TEXT REFERENCES projects(id) ON DELETE SET NULL,
                    sequence INTEGER NOT NULL CHECK (sequence >= 0),
                    status TEXT NOT NULL,
                    snapshot_json TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    project_id TEXT REFERENCES projects(id) ON DELETE CASCADE,
                    snapshot_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS artifacts (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    run_id TEXT REFERENCES runs(id) ON DELETE SET NULL,
                    snapshot_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS annotations (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    snapshot_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_projects_updated
                    ON projects(updated_at DESC, id ASC);
                CREATE INDEX IF NOT EXISTS idx_tasks_timestamp
                    ON tasks(timestamp DESC, task_id ASC);
                CREATE INDEX IF NOT EXISTS idx_tasks_project_timestamp
                    ON tasks(project_id, timestamp DESC, task_id ASC);
                """
            )

    @staticmethod
    def _project_values(project: Project) -> tuple[object, ...]:
        return (
            project.id,
            project.name,
            project.root_path,
            project.engine.value,
            _strict_json(project.to_dict()),
            project.created_at.isoformat(),
            project.updated_at.isoformat(),
        )

    def create_project(self, project: Project) -> Project:
        with self._operation() as connection:
            connection.execute(
                """
                INSERT INTO projects
                    (id, name, root_path, engine, snapshot_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                self._project_values(project),
            )
        return project

    def get_project(self, project_id: str) -> Project | None:
        with self._operation() as connection:
            row = connection.execute(
                "SELECT snapshot_json FROM projects WHERE id = ?", (project_id,)
            ).fetchone()
        return None if row is None else _load_snapshot(row["snapshot_json"], Project)

    def list_projects(self) -> list[Project]:
        with self._operation() as connection:
            rows = connection.execute(
                """
                SELECT snapshot_json FROM projects
                ORDER BY updated_at DESC, id ASC
                """
            ).fetchall()
        return [_load_snapshot(row["snapshot_json"], Project) for row in rows]

    def update_project(self, project: Project) -> bool:
        snapshot = _strict_json(project.to_dict())
        with self._operation() as connection:
            cursor = connection.execute(
                """
                UPDATE projects
                SET name = ?, root_path = ?, engine = ?, snapshot_json = ?,
                    created_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    project.name,
                    project.root_path,
                    project.engine.value,
                    snapshot,
                    project.created_at.isoformat(),
                    project.updated_at.isoformat(),
                    project.id,
                ),
            )
            return cursor.rowcount == 1

    def delete_project(self, project_id: str) -> bool:
        with self._operation() as connection:
            cursor = connection.execute(
                "DELETE FROM projects WHERE id = ?", (project_id,)
            )
            return cursor.rowcount == 1

    def upsert_project(self, project: Project) -> Project:
        """Insert or update a project with one conflict-targeted statement."""

        with self._operation() as connection:
            connection.execute(
                """
                INSERT INTO projects
                    (id, name, root_path, engine, snapshot_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    root_path = excluded.root_path,
                    engine = excluded.engine,
                    snapshot_json = excluded.snapshot_json,
                    created_at = excluded.created_at,
                    updated_at = excluded.updated_at
                """,
                self._project_values(project),
            )
        return project

    def upsert_task_event(
        self, event: TaskEvent, *, project_id: str | None = None
    ) -> bool:
        """Store a task snapshot only when its sequence strictly advances."""

        snapshot = _strict_json(event.to_dict())
        with self._operation() as connection:
            cursor = connection.execute(
                """
                INSERT INTO tasks
                    (task_id, project_id, sequence, status, snapshot_json, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    project_id = COALESCE(excluded.project_id, tasks.project_id),
                    sequence = excluded.sequence,
                    status = excluded.status,
                    snapshot_json = excluded.snapshot_json,
                    timestamp = excluded.timestamp
                WHERE excluded.sequence > tasks.sequence
                """,
                (
                    event.task_id,
                    project_id,
                    event.sequence,
                    event.status,
                    snapshot,
                    event.timestamp.isoformat(),
                ),
            )
            return cursor.rowcount == 1

    def get_task(self, task_id: str) -> TaskEvent | None:
        with self._operation() as connection:
            row = connection.execute(
                "SELECT snapshot_json FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
        return None if row is None else _load_snapshot(row["snapshot_json"], TaskEvent)

    def list_tasks(
        self, *, project_id: str | None = None, limit: int | None = None
    ) -> list[TaskEvent]:
        if limit is not None and limit < 0:
            raise ValueError("limit must not be negative")
        where = "" if project_id is None else "WHERE project_id = ?"
        parameters: list[object] = [] if project_id is None else [project_id]
        limit_sql = ""
        if limit is not None:
            limit_sql = "LIMIT ?"
            parameters.append(limit)
        with self._operation() as connection:
            rows = connection.execute(
                f"""
                SELECT snapshot_json FROM tasks
                {where}
                ORDER BY timestamp DESC, task_id ASC
                {limit_sql}
                """,
                parameters,
            ).fetchall()
        return [_load_snapshot(row["snapshot_json"], TaskEvent) for row in rows]

    def persist_run(
        self,
        run_id: str,
        snapshot: Mapping[str, Any],
        *,
        project_id: str | None = None,
    ) -> None:
        """Persist a generic run snapshot with the same strict JSON policy."""

        encoded = _strict_json(snapshot)
        now = datetime.now(timezone.utc).isoformat()
        with self._operation() as connection:
            connection.execute(
                """
                INSERT INTO runs (id, project_id, snapshot_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    project_id = COALESCE(excluded.project_id, runs.project_id),
                    snapshot_json = excluded.snapshot_json,
                    updated_at = excluded.updated_at
                """,
                (run_id, project_id, encoded, now, now),
            )


SQLiteRepository = StudioRepository

__all__ = ["SQLiteRepository", "StudioRepository"]
