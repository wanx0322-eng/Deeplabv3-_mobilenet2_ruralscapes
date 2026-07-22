import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import "../components" as Components

Flickable {
    id: root
    property var controller
    property bool reducedMotion: false
    signal chooseModelRequested()
    signal chooseInputRequested()
    signal chooseOutputRequested()
    signal startRequested()

    clip: true
    contentWidth: width
    contentHeight: content.implicitHeight + 64
    boundsBehavior: Flickable.StopAtBounds
    ScrollBar.vertical: ScrollBar { }

    ColumnLayout {
        id: content
        x: 32
        y: 28
        width: root.width - 64
        spacing: 20

        Components.SectionHeader {
            Layout.fillWidth: true
            number: "05"
            title: qsTr("识别工作台")
            description: qsTr("选择模型与输入图像，确认输出目录后创建本地识别任务。")
        }

        Components.ErrorBanner {
            Layout.fillWidth: true
            message: root.controller ? root.controller.errorMessage : ""
            onDismissed: if (root.controller) root.controller.clearError()
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 16

            Components.JournalCard {
                Layout.fillWidth: true
                title: qsTr("识别输入")
                subtitle: qsTr("模型和图像均保留在本地工作目录。")
                reducedMotion: root.reducedMotion

                ColumnLayout {
                    Layout.fillWidth: true
                    spacing: 12
                    Components.StudioField {
                        Layout.fillWidth: true
                        label: qsTr("模型文件")
                        text: root.controller ? root.controller.modelPath : ""
                        placeholderText: qsTr("尚未选择模型")
                        readOnly: true
                    }
                    Components.StudioButton {
                        text: qsTr("选择模型")
                        reducedMotion: root.reducedMotion
                        onClicked: root.chooseModelRequested()
                    }
                    Components.StudioField {
                        Layout.fillWidth: true
                        label: qsTr("输入图像或目录")
                        text: root.controller ? root.controller.inputPath : ""
                        placeholderText: qsTr("尚未选择输入")
                        readOnly: true
                    }
                    Components.StudioButton {
                        text: qsTr("选择输入")
                        accent: false
                        reducedMotion: root.reducedMotion
                        onClicked: root.chooseInputRequested()
                    }
                    Components.StudioField {
                        Layout.fillWidth: true
                        label: qsTr("输出目录")
                        text: root.controller ? root.controller.outputPath : ""
                        placeholderText: qsTr("尚未选择输出目录")
                        readOnly: true
                    }
                    Components.StudioButton {
                        text: qsTr("选择输出目录")
                        quiet: true
                        reducedMotion: root.reducedMotion
                        onClicked: root.chooseOutputRequested()
                    }
                }
            }

            Components.JournalCard {
                Layout.preferredWidth: 340
                title: qsTr("识别状态")
                subtitle: root.controller && root.controller.running ? qsTr("本地识别任务正在运行") : qsTr("等待模型与输入")
                reducedMotion: root.reducedMotion

                ColumnLayout {
                    Layout.fillWidth: true
                    spacing: 14
                    Components.StatusPill {
                        text: root.controller && root.controller.running ? qsTr("处理中") : qsTr("未开始")
                        tone: root.controller && root.controller.running ? "success" : "neutral"
                    }
                    Components.StudioProgress {
                        Layout.fillWidth: true
                        value: root.controller ? root.controller.progress : 0
                        reducedMotion: root.reducedMotion
                    }
                    Components.StudioButton {
                        Layout.fillWidth: true
                        text: qsTr("开始识别")
                        enabled: root.controller && root.controller.modelPath.length > 0 && root.controller.inputPath.length > 0 && !root.controller.running
                        reducedMotion: root.reducedMotion
                        onClicked: root.startRequested()
                    }
                }
            }
        }
    }
}


