"""Qt-free annotation state and bounded undo history."""

from __future__ import annotations

from collections import deque

import numpy as np


class AnnotationState:
    TOOLS = frozenset({"brush", "eraser", "polygon", "fill"})

    def __init__(self, *, max_undo: int = 20) -> None:
        if max_undo < 1:
            raise ValueError("max_undo must be positive")
        self.max_undo = max_undo
        self.mask: np.ndarray | None = None
        self.dirty = False
        self.tool = "brush"
        self._undo: deque[np.ndarray] = deque(maxlen=max_undo)
        self._redo: deque[np.ndarray] = deque(maxlen=max_undo)

    def load(self, mask: np.ndarray | None) -> None:
        self.mask = self._copy_mask(mask)
        self._undo.clear()
        self._redo.clear()
        self.dirty = False

    def set_tool(self, tool: str) -> None:
        if tool not in self.TOOLS:
            raise ValueError(f"Unsupported annotation tool: {tool}")
        self.tool = tool

    def replace_mask(self, mask: np.ndarray) -> None:
        replacement = self._copy_mask(mask)
        if replacement is None:
            raise ValueError("mask cannot be None")
        if self.mask is not None:
            self._undo.append(self.mask.copy())
        self.mask = replacement
        self._redo.clear()
        self.dirty = True

    def undo(self) -> bool:
        if not self._undo or self.mask is None:
            return False
        self._redo.append(self.mask.copy())
        self.mask = self._undo.pop()
        self.dirty = True
        return True

    def redo(self) -> bool:
        if not self._redo or self.mask is None:
            return False
        self._undo.append(self.mask.copy())
        self.mask = self._redo.pop()
        self.dirty = True
        return True

    def mark_saved(self) -> None:
        self.dirty = False

    def snapshot(self) -> np.ndarray | None:
        return self._copy_mask(self.mask)

    @staticmethod
    def _copy_mask(mask: np.ndarray | None) -> np.ndarray | None:
        if mask is None:
            return None
        array = np.asarray(mask)
        if array.ndim != 2:
            raise ValueError("annotation mask must be a 2D array")
        return array.astype(np.uint8, copy=True)
