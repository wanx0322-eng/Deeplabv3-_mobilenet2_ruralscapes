"""GUI-free orchestration for annotation engines and async runners."""

from __future__ import annotations

from typing import Any

from .state import AnnotationState


class AnnotationController:
    def __init__(
        self,
        *,
        state: AnnotationState | None = None,
        engine: Any = None,
        dataset_manager: Any = None,
        sam2_runner: Any = None,
        export_runner: Any = None,
    ) -> None:
        self.state = state or AnnotationState()
        self.engine = engine
        self.dataset_manager = dataset_manager
        self.sam2_runner = sam2_runner
        self.export_runner = export_runner
        self.status = "idle"
        self.error = ""
        self._request_sequence = 0
        self._active_request: int | None = None
        self._cancelled: set[int] = set()

    @property
    def active_request(self) -> int | None:
        return self._active_request

    def begin_auto_annotation(self) -> int:
        self._request_sequence += 1
        self._active_request = self._request_sequence
        self.status = "running"
        self.error = ""
        return self._request_sequence

    def accept_auto_result(self, request_id: int, mask) -> bool:
        if (
            request_id != self._active_request
            or request_id in self._cancelled
            or self.status != "running"
        ):
            return False
        self.state.replace_mask(mask)
        self.status = "completed"
        self._active_request = None
        return True

    def fail_auto_annotation(self, request_id: int, error: Exception | str) -> bool:
        if request_id != self._active_request or request_id in self._cancelled:
            return False
        self.status = "failed"
        self.error = str(error)
        self._active_request = None
        return True

    def cancel_auto_annotation(self) -> bool:
        request_id = self._active_request
        if request_id is None:
            return False
        self._cancelled.add(request_id)
        runner = self.sam2_runner
        if runner is not None and hasattr(runner, "cancel"):
            runner.cancel(request_id)
        self.status = "cancelled"
        self._active_request = None
        return True
