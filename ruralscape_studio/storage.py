"""SQLite persistence public API for RuralScape Studio."""

from __future__ import annotations

from pathlib import Path

from ._storage_impl import StudioRepository as _FileStudioRepository


class StudioRepository(_FileStudioRepository):
    """File-backed repository using one SQLite connection per operation."""

    def __init__(self, db_path: str | Path) -> None:
        if str(db_path) == ":memory:":
            raise ValueError(":memory: databases are not supported; pass a filesystem path")
        super().__init__(db_path)


SQLiteRepository = StudioRepository

__all__ = ["SQLiteRepository", "StudioRepository"]
