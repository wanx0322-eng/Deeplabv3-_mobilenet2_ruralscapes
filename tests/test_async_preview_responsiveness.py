from __future__ import annotations

import time

from PIL import Image
from PySide6.QtCore import QEventLoop, Qt, QTimer


def test_4k_preview_decode_keeps_gui_heartbeat_below_100ms(qapp, tmp_path) -> None:
    from workstation.async_images import PreviewLoader

    image_path = tmp_path / "4k.png"
    Image.new("RGB", (4096, 4096), (30, 90, 60)).save(image_path)
    loader = PreviewLoader(max_cache_bytes=32 * 1024 * 1024)
    loop = QEventLoop()
    heartbeats: list[float] = []
    timer = QTimer()
    timer.setTimerType(Qt.PreciseTimer)
    timer.setInterval(10)
    timer.timeout.connect(lambda: heartbeats.append(time.perf_counter()))
    loader.ready.connect(lambda *_args: loop.quit())
    QTimer.singleShot(10_000, loop.quit)

    timer.start()
    loader.load(str(image_path), max_size=(1600, 1600))
    loop.exec()
    timer.stop()

    assert len(heartbeats) >= 2
    assert max(b - a for a, b in zip(heartbeats, heartbeats[1:])) < 0.100


def test_preview_loader_drops_an_older_request(qapp, tmp_path) -> None:
    from workstation.async_images import PreviewLoader

    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (2048, 2048), "red").save(first)
    Image.new("RGB", (16, 16), "blue").save(second)
    loader = PreviewLoader()
    loop = QEventLoop()
    delivered: list[int] = []
    loader.ready.connect(lambda request_id, *_args: (delivered.append(request_id), loop.quit()))
    QTimer.singleShot(10_000, loop.quit)

    loader.load(str(first))
    latest = loader.load(str(second))
    loop.exec()

    assert delivered == [latest]


import os
import pytest


@pytest.fixture
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])
