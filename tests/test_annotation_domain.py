from __future__ import annotations

import inspect

import numpy as np


def test_annotation_state_tracks_dirty_tools_and_undo_redo() -> None:
    from workstation.pages.annotate.state import AnnotationState

    state = AnnotationState(max_undo=3)
    initial = np.zeros((4, 4), dtype=np.uint8)
    state.load(initial)
    state.set_tool("polygon")
    edited = initial.copy()
    edited[1:3, 1:3] = 2
    state.replace_mask(edited)

    assert state.tool == "polygon"
    assert state.dirty is True
    assert state.undo() is True
    assert np.array_equal(state.mask, initial)
    assert state.redo() is True
    assert np.array_equal(state.mask, edited)
    state.mark_saved()
    assert state.dirty is False


class FakeRunner:
    def __init__(self) -> None:
        self.cancelled: list[int] = []

    def cancel(self, request_id: int) -> None:
        self.cancelled.append(request_id)


def test_annotation_controller_discards_stale_results_and_supports_cancel() -> None:
    from workstation.pages.annotate.controller import AnnotationController
    from workstation.pages.annotate.state import AnnotationState

    state = AnnotationState()
    runner = FakeRunner()
    controller = AnnotationController(
        state=state,
        engine=object(),
        dataset_manager=object(),
        sam2_runner=runner,
        export_runner=object(),
    )
    first = controller.begin_auto_annotation()
    second = controller.begin_auto_annotation()

    assert controller.accept_auto_result(first, np.ones((2, 2), dtype=np.uint8)) is False
    result = np.full((2, 2), 3, dtype=np.uint8)
    assert controller.accept_auto_result(second, result) is True
    assert np.array_equal(state.mask, result)

    third = controller.begin_auto_annotation()
    assert controller.cancel_auto_annotation() is True
    assert runner.cancelled == [third]
    assert controller.accept_auto_result(third, result) is False


def test_annotation_controller_reports_failures_without_qtwidgets_dependency() -> None:
    from workstation.pages.annotate import controller as module
    from workstation.pages.annotate.controller import AnnotationController

    controller = AnnotationController()
    request_id = controller.begin_auto_annotation()
    assert controller.fail_auto_annotation(request_id, RuntimeError("boom")) is True
    assert controller.status == "failed"
    assert controller.error == "boom"
    assert "QtWidgets" not in inspect.getsource(module)
