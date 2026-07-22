"""Stale-safe asynchronous image preview loading with a bounded LRU."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from threading import RLock
from typing import Any

from PIL import Image
from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal


class ImageRequestGate:
    def __init__(self) -> None:
        self._sequence = 0
        self._active = 0

    def begin(self) -> int:
        self._sequence += 1
        self._active = self._sequence
        return self._active

    def accept(self, request_id: int) -> bool:
        return request_id == self._active


class BoundedImageCache:
    def __init__(self, *, max_bytes: int = 128 * 1024 * 1024) -> None:
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        self.max_bytes = max_bytes
        self.total_bytes = 0
        self._items: OrderedDict[Any, tuple[Any, int]] = OrderedDict()
        self._lock = RLock()

    def get(self, key: Any) -> Any:
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            self._items.move_to_end(key)
            return item[0]

    def put(self, key: Any, value: Any, *, cost: int) -> None:
        cost = max(0, int(cost))
        with self._lock:
            previous = self._items.pop(key, None)
            if previous is not None:
                self.total_bytes -= previous[1]
            if cost > self.max_bytes:
                return
            self._items[key] = (value, cost)
            self.total_bytes += cost
            while self.total_bytes > self.max_bytes and self._items:
                _old_key, (_old_value, old_cost) = self._items.popitem(last=False)
                self.total_bytes -= old_cost

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self.total_bytes = 0


class _TaskSignals(QObject):
    ready = Signal(int, object, object, object, int)
    failed = Signal(int, str)


class _PreviewTask(QRunnable):
    def __init__(
        self,
        request_id: int,
        key,
        image_path: str,
        label_path: str | None,
        max_size: tuple[int, int],
        signals: _TaskSignals,
    ) -> None:
        super().__init__()
        self.request_id = request_id
        self.key = key
        self.image_path = image_path
        self.label_path = label_path
        self.max_size = max_size
        self.signals = signals

    def run(self) -> None:
        try:
            image = Image.open(self.image_path).convert("RGB")
            image.thumbnail(self.max_size, Image.Resampling.LANCZOS)
            mask = None
            if self.label_path and Path(self.label_path).is_file():
                mask = Image.open(self.label_path).copy()
                mask.thumbnail(self.max_size, Image.Resampling.NEAREST)
            cost = image.width * image.height * 4
            if mask is not None:
                cost += mask.width * mask.height
            self.signals.ready.emit(self.request_id, self.key, image, mask, cost)
        except Exception as error:
            self.signals.failed.emit(self.request_id, str(error))


class PreviewLoader(QObject):
    ready = Signal(int, object, object)
    failed = Signal(int, str)

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        max_cache_bytes: int = 128 * 1024 * 1024,
        pool: QThreadPool | None = None,
    ) -> None:
        super().__init__(parent)
        self.gate = ImageRequestGate()
        self.cache = BoundedImageCache(max_bytes=max_cache_bytes)
        self.pool = pool or QThreadPool.globalInstance()
        self._signals = _TaskSignals()
        self._signals.ready.connect(self._deliver)
        self._signals.failed.connect(self._fail)

    def load(
        self,
        image_path: str,
        label_path: str | None = None,
        *,
        max_size: tuple[int, int] = (2048, 2048),
    ) -> int:
        request_id = self.gate.begin()
        key = (image_path, label_path, max_size)
        cached = self.cache.get(key)
        if cached is not None:
            QTimer.singleShot(
                0, lambda: self.ready.emit(request_id, cached[0], cached[1])
            )
            return request_id
        self.pool.start(
            _PreviewTask(
                request_id, key, image_path, label_path, max_size, self._signals
            )
        )
        return request_id

    def _deliver(self, request_id, key, image, mask, cost) -> None:
        self.cache.put(key, (image, mask), cost=cost)
        if self.gate.accept(request_id):
            self.ready.emit(request_id, image, mask)

    def _fail(self, request_id: int, message: str) -> None:
        if self.gate.accept(request_id):
            self.failed.emit(request_id, message)
