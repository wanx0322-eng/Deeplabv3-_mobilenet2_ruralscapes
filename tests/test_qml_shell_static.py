from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "ruralscape_studio"
QML = PACKAGE / "qml"

NAVIGATION = (
    "项目概览",
    "数据集",
    "标注工作台",
    "训练中心",
    "识别工作台",
    "评估报告",
    "模型与导出",
    "任务记录",
)

COMPONENTS = (
    "StudioButton.qml",
    "StudioField.qml",
    "JournalCard.qml",
    "NavItem.qml",
    "StatusPill.qml",
    "StudioTable.qml",
    "StudioTableRow.qml",
    "StudioProgress.qml",
    "StudioDialog.qml",
    "ErrorBanner.qml",
)

PAGES = {
    "ProjectOverview.qml": ("项目概览", "新建项目", "打开项目"),
    "DatasetPage.qml": ("数据集", "选择数据集目录", "扫描数据集"),
    "AnnotationWorkbench.qml": ("标注工作台", "选择图像", "保存版本"),
    "TrainingCenter.qml": ("训练中心", "DeepLab V3+", "SegFormer B2"),
    "InferenceWorkbench.qml": ("识别工作台", "选择模型", "选择输入"),
    "EvaluationReport.qml": ("评估报告", "验证集", "开始评估"),
    "ModelsExport.qml": ("模型与导出", "模型产物", "导出"),
    "TaskHistory.qml": ("任务记录", "等待任务", "清空筛选"),
}


def test_native_bootstrap_and_launch_entry_are_local_qt_only() -> None:
    bootstrap = PACKAGE / "native_app.py"
    entry = PACKAGE / "__main__.py"
    assert bootstrap.is_file()
    assert entry.is_file()

    source = bootstrap.read_text(encoding="utf-8")
    assert "QQmlApplicationEngine" in source
    assert "QGuiApplication" in source
    assert "torch" not in source
    for forbidden in ("QtWebEngine", "http://", "https://", "node", "server"):
        assert forbidden not in source


def test_qml_shell_has_native_window_layout_and_exact_navigation() -> None:
    main = QML / "Main.qml"
    assert main.is_file()
    source = main.read_text(encoding="utf-8")

    for label in NAVIGATION:
        assert label in source
    for marker in (
        "Qt.FramelessWindowHint",
        "minimumWidth: 1024",
        "width: 260",
        "1280",
        "startSystemMove()",
        "showMinimized()",
        "showMaximized()",
        "showNormal()",
        "close()",
        "reducedMotion ? 0 : Theme.motionDuration",
    ):
        assert marker in source


def test_required_qml_components_exist_with_design_tokens() -> None:
    for filename in COMPONENTS:
        assert (QML / "components" / filename).is_file(), filename

    card = (QML / "components" / "JournalCard.qml").read_text(encoding="utf-8")
    assert "radius: Theme.cardRadius" in card
    assert "-4" in card
    assert "Easing.InOutQuad" in card

    button = (QML / "components" / "StudioButton.qml").read_text(encoding="utf-8")
    assert "radius: Theme.buttonRadius" in button
    assert "Theme.motionDuration" in button


def test_all_pages_exist_and_contain_meaningful_product_controls() -> None:
    for filename, strings in PAGES.items():
        page = QML / "pages" / filename
        assert page.is_file(), filename
        source = page.read_text(encoding="utf-8")
        for value in strings:
            assert value in source, f"{filename}: {value}"
        assert "SectionHeader" in source

def test_theme_singleton_centralizes_exact_foundation_tokens() -> None:
    theme = QML / "theme" / "Theme.qml"
    qmldir = QML / "theme" / "qmldir"
    assert theme.is_file()
    assert qmldir.is_file()

    source = theme.read_text(encoding="utf-8")
    for token in ("#F2EBDD", "#173C2D", "#C7F464", "#111827", "#6B7280"):
        assert token in source
    assert "pragma Singleton" in source
    assert "Qt.application.font.family" in source
    assert "singleton Theme 1.0 Theme.qml" in qmldir.read_text(encoding="utf-8")

    qml_files = [qml_file for qml_file in QML.rglob("*.qml") if qml_file != theme]
    for qml_file in qml_files:
        qml_source = qml_file.read_text(encoding="utf-8")
        assert 'import "theme"' in qml_source or 'import "../theme"' in qml_source
        assert "#111827" not in qml_source
        assert "#6B7280" not in qml_source
        assert re.search(r"#[0-9A-Fa-f]{6,8}", qml_source) is None






