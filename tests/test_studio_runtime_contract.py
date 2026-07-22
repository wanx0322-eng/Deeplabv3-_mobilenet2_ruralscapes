from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject


ROOT = Path(__file__).resolve().parents[1]


def test_studio_runtime_exposes_fixed_connected_contract() -> None:
    from ruralscape_studio.runtime import RuntimeConsole, StudioRuntime

    controller_names = (
        "projectController",
        "datasetController",
        "annotationController",
        "trainingController",
        "inferenceController",
        "evaluationController",
        "modelController",
        "taskManager",
    )
    backend_names = (
        "datasetBackend",
        "trainingBackend",
        "evaluationBackend",
        "inferenceBackend",
        "modelBackend",
    )
    controllers = {name: QObject() for name in controller_names}
    backends = {name: QObject() for name in backend_names}
    console = RuntimeConsole()
    settings = QObject()
    runtime = StudioRuntime(
        controllers, backends, console, settings, mode="connected"
    )

    assert runtime.mode == "connected"
    assert runtime.isDemo is False
    assert all(getattr(runtime, name) is controllers[name] for name in controller_names)
    assert all(getattr(runtime, name) is backends[name] for name in backend_names)
    assert runtime.console is console
    assert runtime.appSettings is settings


def test_demo_backend_reports_every_qml_action() -> None:
    from ruralscape_studio.runtime import DemoBackend, RuntimeConsole

    console = RuntimeConsole()
    backend = DemoBackend("testBackend", console)
    feedback: list[str] = []
    backend.feedback.connect(feedback.append)
    backend.useProjectDataset()
    backend.scan()
    backend.start("")
    backend.requestStop()
    backend.refresh()

    assert len(console.lines) == 5
    assert feedback == console.lines
    assert all("Demo mode" in line and "not connected" in line for line in feedback)


def test_qml_buttons_call_runtime_without_guard_or_corrupt_arguments() -> None:
    source = (ROOT / "ruralscape_studio" / "qml" / "Main.qml").read_text(
        encoding="utf-8"
    )
    assert "typeof" not in source
    assert 'evaluationBackend.start("")' in source
    for backend in (
        "datasetBackend",
        "trainingBackend",
        "evaluationBackend",
        "inferenceBackend",
        "modelBackend",
    ):
        assert f"root.runtime.{backend}" in source
