"""Recoverable model-file trash batches."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable


@dataclass(frozen=True, slots=True)
class TrashEntry:
    source: Path
    trashed: Path


@dataclass(frozen=True, slots=True)
class TrashBatch:
    batch_id: str
    entries: tuple[TrashEntry, ...]


class TrashManager:
    def __init__(
        self,
        models_root: str | Path,
        *,
        timestamp_factory: Callable[[], str] | None = None,
        trash_root: str | Path | None = None,
    ) -> None:
        self.models_root = Path(models_root).resolve()
        self.trash_root = Path(trash_root).resolve() if trash_root else self.models_root / "_trash"
        self.timestamp_factory = timestamp_factory or (
            lambda: datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        )
        self._latest: TrashBatch | None = None

    def move_batch(self, paths: Iterable[str | Path]) -> TrashBatch:
        sources = tuple(Path(path).resolve() for path in paths)
        if not sources:
            raise ValueError("trash batch cannot be empty")
        for source in sources:
            if not source.is_file():
                raise FileNotFoundError(source)
            if self.models_root not in source.parents:
                raise ValueError(f"model file escapes configured root: {source}")
        batch_id = self.timestamp_factory()
        batch_root = self.trash_root / batch_id
        batch_root.mkdir(parents=True, exist_ok=False)
        entries = []
        for source in sources:
            relative = source.relative_to(self.models_root)
            target = batch_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
            entries.append(TrashEntry(source, target))
        self._latest = TrashBatch(batch_id, tuple(entries))
        return self._latest

    def undo_latest(self) -> bool:
        batch = self._latest
        if batch is None:
            return False
        for entry in batch.entries:
            if entry.source.exists():
                raise FileExistsError(entry.source)
        for entry in batch.entries:
            entry.source.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(entry.trashed), str(entry.source))
        self._latest = None
        return True
