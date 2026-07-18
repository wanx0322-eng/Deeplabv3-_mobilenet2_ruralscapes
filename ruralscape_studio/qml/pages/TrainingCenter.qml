import "../theme"
import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import "../components" as Components

Flickable {
    id: root
    property var controller
    property bool reducedMotion: false
    signal startRequested()
    signal stopRequested()

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
            number: "04"
            title: "训练中心"
            description: "选择分割引擎并填写训练参数；启动前不加载模型运行时。"
        }

        Components.ErrorBanner {
            Layout.fillWidth: true
            message: controller ? controller.errorMessage : ""
            onDismissed: if (controller) controller.clearError()
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 16

            Components.JournalCard {
                Layout.fillWidth: true
                title: "训练配置"
                subtitle: "参数确认后才能创建训练任务。"
                reducedMotion: root.reducedMotion

                ColumnLayout {
                    Layout.fillWidth: true
                    spacing: 14
                    Text {
                        text: "模型引擎"
                        color: Theme.primaryText
                        font.family: Theme.fontFamily
                        font.pixelSize: 12
                        font.weight: Font.DemiBold
                    }
                    ComboBox {
                        id: engineSelect
                        Layout.fillWidth: true
                        model: ["DeepLab V3+", "SegFormer B2"]
                        currentIndex: controller && controller.engine === "segformer_b2" ? 1 : 0
                        onActivated: if (controller) controller.selectEngine(currentIndex === 0 ? "deeplab_v3_plus" : "segformer_b2")
                    }
                    RowLayout {
                        Layout.fillWidth: true
                        Components.StudioField {
                            id: classCount
                            Layout.fillWidth: true
                            label: "类别数量"
                            placeholderText: "输入类别数量"
                        }
                        Components.StudioField {
                            id: batchSize
                            Layout.fillWidth: true
                            label: "批大小"
                            placeholderText: "至少为 2"
                        }
                    }
                    Components.StudioField {
                        id: datasetPath
                        Layout.fillWidth: true
                        label: "数据集目录"
                        placeholderText: "从数据集页面选择目录"
                        readOnly: true
                    }
                }
            }

            Components.JournalCard {
                Layout.preferredWidth: 340
                title: "任务状态"
                subtitle: controller && controller.running ? "训练任务正在运行" : "尚未启动训练任务"
                reducedMotion: root.reducedMotion

                ColumnLayout {
                    Layout.fillWidth: true
                    spacing: 14
                    Components.StatusPill {
                        text: controller && controller.running ? "训练中" : "待启动"
                        tone: controller && controller.running ? "success" : "neutral"
                    }
                    Components.StudioProgress {
                        Layout.fillWidth: true
                        value: controller ? controller.progress : 0
                        reducedMotion: root.reducedMotion
                    }
                    Components.StudioButton {
                        Layout.fillWidth: true
                        text: "开始训练"
                        enabled: classCount.text.length > 0 && batchSize.text.length > 0 && controller && !controller.running
                        reducedMotion: root.reducedMotion
                        onClicked: root.startRequested()
                    }
                    Components.StudioButton {
                        Layout.fillWidth: true
                        text: "停止训练"
                        accent: false
                        enabled: controller && controller.running
                        reducedMotion: root.reducedMotion
                        onClicked: root.stopRequested()
                    }
                    Components.StudioButton {
                        Layout.fillWidth: true
                        text: "重置状态"
                        quiet: true
                        enabled: controller && (controller.running || controller.status !== "idle")
                        reducedMotion: root.reducedMotion
                        onClicked: if (controller) controller.reset()
                    }
                }
            }
        }
    }
}


