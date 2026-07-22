pragma ComponentBehavior: Bound

import "theme"
import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import "components" as Components
import "pages" as Pages

ApplicationWindow {
    id: root
    required property var runtime
    visible: true
    width: 1440
    height: 900
    minimumWidth: 1024
    minimumHeight: 700
    color: Theme.paper
    title: "RuralScape Studio"
    flags: Qt.Window | Qt.FramelessWindowHint

    property int currentIndex: 0
    readonly property bool reducedMotion: runtime.appSettings.reducedMotion
    readonly property int motionDuration: reducedMotion ? 0 : Theme.motionDuration
    readonly property var navigation: [
        { "number": "01", "label": qsTr("项目概览") },
        { "number": "02", "label": qsTr("数据集") },
        { "number": "03", "label": qsTr("标注工作台") },
        { "number": "04", "label": qsTr("训练中心") },
        { "number": "05", "label": qsTr("识别工作台") },
        { "number": "06", "label": qsTr("评估报告") },
        { "number": "07", "label": qsTr("模型与导出") },
        { "number": "08", "label": qsTr("任务记录") }
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
                    text: root.runtime.projectController.hasProject ? root.runtime.projectController.projectName : qsTr("未打开项目")
                    color: Theme.fieldForeground
                    font.family: Theme.fontFamily
                    font.pixelSize: 17
                    font.weight: Font.DemiBold
                    elide: Text.ElideRight
                    Layout.fillWidth: true
                }
                Text {
                    text: root.runtime.projectController.hasProject ? root.runtime.projectController.rootPath : qsTr("选择项目后开始工作")
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
                    text: qsTr("本地工作台")
                    color: Theme.fieldForegroundMuted
                    font.family: Theme.fontFamily
                    font.pixelSize: 12
                }
                Item { Layout.fillWidth: true }
                Text {
                    text: root.runtime.taskManager.runningCount > 0 ? qsTr("任务 ") + root.runtime.taskManager.runningCount : qsTr("空闲")
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
                        controller: root.runtime.projectController
                        reducedMotion: root.reducedMotion
                    }
                    Pages.DatasetPage {
                        controller: root.runtime.datasetController
                        reducedMotion: root.reducedMotion
                        onChooseDirectoryRequested: root.runtime.datasetBackend.useProjectDataset()
                        onScanRequested: root.runtime.datasetBackend.scan()
                    }
                    Pages.AnnotationWorkbench {
                        controller: root.runtime.annotationController
                        reducedMotion: root.reducedMotion
                    }
                    Pages.TrainingCenter {
                        controller: root.runtime.trainingController
                        reducedMotion: root.reducedMotion
                        onStartRequested: root.runtime.trainingBackend.start()
                        onStopRequested: root.runtime.trainingBackend.requestStop()
                    }
                    Pages.InferenceWorkbench {
                        controller: root.runtime.inferenceController
                        reducedMotion: root.reducedMotion
                        onStartRequested: root.runtime.inferenceBackend.start()
                    }
                    Pages.EvaluationReport {
                        controller: root.runtime.evaluationController
                        reducedMotion: root.reducedMotion
                        onStartRequested: root.runtime.evaluationBackend.start("")
                    }
                    Pages.ModelsExport {
                        controller: root.runtime.modelController
                        reducedMotion: root.reducedMotion
                        onChooseModelRequested: root.runtime.modelBackend.refresh()
                    }
                    Pages.TaskHistory {
                        controller: root.runtime.taskManager
                        reducedMotion: root.reducedMotion
                    }
                }
            }
        }
    }
    Rectangle {
        visible: root.runtime.isDemo
        anchors.top: titleBar.bottom
        anchors.right: parent.right
        anchors.margins: 12
        width: demoLabel.implicitWidth + 20
        height: 30
        radius: Theme.buttonRadius
        color: Theme.acid
        z: 100

        Text {
            id: demoLabel
            anchors.centerIn: parent
            text: qsTr("实验预览版 · DEMO")
            color: Theme.field
            font.bold: true
        }
    }
}



