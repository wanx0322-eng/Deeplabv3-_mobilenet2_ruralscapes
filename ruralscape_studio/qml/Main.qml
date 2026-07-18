import "theme"
import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import "components" as Components
import "pages" as Pages

ApplicationWindow {
    id: root
    visible: true
    width: 1440
    height: 900
    minimumWidth: 1024
    minimumHeight: 700
    color: Theme.paper
    title: "RuralScape Studio"
    flags: Qt.Window | Qt.FramelessWindowHint

    property int currentIndex: 0
    readonly property bool reducedMotion: appSettings.reducedMotion
    readonly property int motionDuration: reducedMotion ? 0 : Theme.motionDuration
    readonly property var navigation: [
        { "number": "01", "label": "项目概览" },
        { "number": "02", "label": "数据集" },
        { "number": "03", "label": "标注工作台" },
        { "number": "04", "label": "训练中心" },
        { "number": "05", "label": "识别工作台" },
        { "number": "06", "label": "评估报告" },
        { "number": "07", "label": "模型与导出" },
        { "number": "08", "label": "任务记录" }
    ]

    Rectangle {
        id: titleBar
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.top: parent.top
        height: 48
        color: Theme.field
        z: 10

        RowLayout {
            anchors.fill: parent
            anchors.leftMargin: 16
            spacing: 10

            Rectangle {
                Layout.preferredWidth: 26
                Layout.preferredHeight: 26
                radius: Theme.buttonRadius
                color: Theme.acid
                Text {
                    anchors.centerIn: parent
                    text: "RS"
                    color: Theme.field
                    font.family: Theme.fontFamily
                    font.pixelSize: 10
                    font.weight: Font.Black
                }
            }

            Text {
                text: "RuralScape Studio"
                color: Theme.fieldForeground
                font.family: Theme.fontFamily
                font.pixelSize: 13
                font.weight: Font.DemiBold
            }

            Rectangle {
                Layout.preferredWidth: 1
                Layout.preferredHeight: 16
                color: Theme.fieldMuted
            }

            Text {
                text: root.navigation[root.currentIndex].label
                color: Theme.fieldForegroundMuted
                font.family: Theme.fontFamily
                font.pixelSize: 12
            }

            Item { Layout.fillWidth: true }

            Components.WindowControl {
                kind: "minimize"
                onClicked: root.showMinimized()
            }
            Components.WindowControl {
                kind: "maximize"
                onClicked: root.visibility === Window.Maximized ? root.showNormal() : root.showMaximized()
            }
            Components.WindowControl {
                kind: "close"
                onClicked: root.close()
            }
        }

        DragHandler {
            target: null
            onActiveChanged: if (active) root.startSystemMove()
        }
        TapHandler {
            acceptedButtons: Qt.LeftButton
            onDoubleTapped: root.visibility === Window.Maximized ? root.showNormal() : root.showMaximized()
        }
    }

    Rectangle {
        id: sidebar
        anchors.left: parent.left
        anchors.top: titleBar.bottom
        anchors.bottom: parent.bottom
        width: 260
        color: Theme.field

        ColumnLayout {
            anchors.fill: parent
            anchors.margins: 18
            spacing: 8

            ColumnLayout {
                Layout.fillWidth: true
                Layout.topMargin: 12
                Layout.bottomMargin: 18
                spacing: 6
                Text {
                    text: projectController.hasProject ? projectController.projectName : "未打开项目"
                    color: Theme.fieldForeground
                    font.family: Theme.fontFamily
                    font.pixelSize: 17
                    font.weight: Font.DemiBold
                    elide: Text.ElideRight
                    Layout.fillWidth: true
                }
                Text {
                    text: projectController.hasProject ? projectController.rootPath : "选择项目后开始工作"
                    color: Theme.fieldForegroundMuted
                    font.family: Theme.fontFamily
                    font.pixelSize: 11
                    elide: Text.ElideMiddle
                    Layout.fillWidth: true
                }
            }

            Repeater {
                model: root.navigation
                delegate: Components.NavItem {
                    required property int index
                    required property var modelData
                    Layout.fillWidth: true
                    number: modelData.number
                    label: modelData.label
                    current: root.currentIndex === index
                    reducedMotion: root.reducedMotion
                    onClicked: root.currentIndex = index
                }
            }

            Item { Layout.fillHeight: true }

            Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: 1
                color: Theme.fieldMuted
            }

            RowLayout {
                Layout.fillWidth: true
                Layout.topMargin: 10
                spacing: 10
                Rectangle {
                    Layout.preferredWidth: 8
                    Layout.preferredHeight: 8
                    radius: 4
                    color: Theme.acid
                }
                Text {
                    text: "本地工作台"
                    color: Theme.fieldForegroundMuted
                    font.family: Theme.fontFamily
                    font.pixelSize: 12
                }
                Item { Layout.fillWidth: true }
                Text {
                    text: taskManager.runningCount > 0 ? "任务 " + taskManager.runningCount : "空闲"
                    color: Theme.fieldForegroundMuted
                    font.family: Theme.fontFamily
                    font.pixelSize: 11
                }
            }
        }
    }

    Rectangle {
        anchors.left: sidebar.right
        anchors.right: parent.right
        anchors.top: titleBar.bottom
        anchors.bottom: parent.bottom
        color: Theme.paper

        Item {
            anchors.fill: parent
            anchors.leftMargin: 28
            anchors.rightMargin: 28

            Item {
                width: Math.min(parent.width, 1280)
                height: parent.height
                anchors.horizontalCenter: parent.horizontalCenter

                StackLayout {
                    anchors.fill: parent
                    currentIndex: root.currentIndex

                    Pages.ProjectOverview {
                        controller: projectController
                        reducedMotion: root.reducedMotion
                    }
                    Pages.DatasetPage {
                        controller: datasetController
                        reducedMotion: root.reducedMotion
                        // 后端由 workstation/studio_app.py 注入；用 native_app.py
                        // 单独加载外壳时不存在，这里判空后再调用。
                        onChooseDirectoryRequested: if (typeof datasetBackend !== "undefined") datasetBackend.useProjectDataset()
                        onScanRequested: if (typeof datasetBackend !== "undefined") datasetBackend.scan()
                    }
                    Pages.AnnotationWorkbench {
                        controller: annotationController
                        reducedMotion: root.reducedMotion
                    }
                    Pages.TrainingCenter {
                        controller: trainingController
                        reducedMotion: root.reducedMotion
                        onStartRequested: if (typeof trainingBackend !== "undefined") trainingBackend.start()
                        onStopRequested: if (typeof trainingBackend !== "undefined") trainingBackend.requestStop()
                    }
                    Pages.InferenceWorkbench {
                        controller: inferenceController
                        reducedMotion: root.reducedMotion
                        onStartRequested: if (typeof inferenceBackend !== "undefined") inferenceBackend.start()
                    }
                    Pages.EvaluationReport {
                        controller: evaluationController
                        reducedMotion: root.reducedMotion
                        onStartRequested: if (typeof evaluationBackend !== "undefined") evaluationBackend.start("")
                    }
                    Pages.ModelsExport {
                        controller: modelController
                        reducedMotion: root.reducedMotion
                        onChooseModelRequested: if (typeof modelBackend !== "undefined") modelBackend.refresh()
                    }
                    Pages.TaskHistory {
                        controller: taskManager
                        reducedMotion: root.reducedMotion
                    }
                }
            }
        }
    }
}



