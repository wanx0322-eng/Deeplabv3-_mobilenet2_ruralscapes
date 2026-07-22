"""Token-tinted SVG icon provider with DPR-aware caching."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QByteArray, QRectF
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

from workstation.theme import DARK_TOKENS


class IconProvider:
    def __init__(self, asset_root: str | Path | None = None) -> None:
        self.asset_root = Path(asset_root or Path(__file__).with_name("assets") / "icons")
        self._cache: dict[tuple[str, str, int, float], QIcon] = {}

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def icon(
        self,
        name: str,
        *,
        size: int = DARK_TOKENS.ICON_MD,
        color: str = DARK_TOKENS.CONTENT_SECONDARY,
        dpr: float = 1.0,
    ) -> QIcon:
        normalized = QColor(color).name(QColor.HexArgb)
        key = (name, normalized, int(size), float(dpr))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        source = self.asset_root / f"{name}.svg"
        if not source.is_file():
            raise KeyError(f"Unknown icon: {name}")
        svg = source.read_text(encoding="utf-8").replace("currentColor", normalized)
        pixels = max(1, round(size * dpr))
        image = QImage(pixels, pixels, QImage.Format_ARGB32_Premultiplied)
        image.fill(QColor(0, 0, 0, 0))
        image.setDevicePixelRatio(dpr)
        painter = QPainter(image)
        renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
        renderer.render(painter, QRectF(0, 0, size, size))
        painter.end()
        icon = QIcon(QPixmap.fromImage(image))
        self._cache[key] = icon
        return icon

    def clear(self) -> None:
        self._cache.clear()


ICONS = IconProvider()
