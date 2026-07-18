import "../theme"
import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import "../components" as Components

Flickable {
    id: root
    property var controller
    property bool reducedMotion: false
    property string statusFilter: "all"

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
            number: "08"
            title: "任务记录"
            description: "查看本次工作期间实际创建的扫描、训练、识别与评估任务。"
        }

        Components.JournalCard {
            Layout.fillWidth: true
            title: "任务筛选"
            subtitle: controller && controller.runningCount > 0 ? "有任务正在运行" : "当前没有运行中的任务"
            reducedMotion: root.reducedMotion

            RowLayout {
                Layout.fillWidth: true
                spacing: 12
                ComboBox {
                    id: filterSelect
                    Layout.preferredWidth: 180
                    model: ["全部状态", "等待中", "运行中", "已完成", "失败"]
                    onActivated: root.statusFilter = ["all", "queued", "running", "completed", "failed"][currentIndex]
                }
                Components.StudioButton {
                    text: "清空筛选"
                    quiet: true
                    reducedMotion: root.reducedMotion
                    onClicked: {
                        filterSelect.currentIndex = 0
                        root.statusFilter = "all"
                    }
                }
                Item { Layout.fillWidth: true }
                Components.StatusPill {
                    text: controller ? "运行中 " + controller.runningCount : "运行中 0"
                    tone: controller && controller.runningCount > 0 ? "success" : "neutral"
                }
            }
        }

        Components.StudioTable {
            Layout.fillWidth: true
            title: "任务"

            Components.StudioTableRow {
                visible: !controller || controller.tasks.length === 0
                primaryText: "等待任务"
                secondaryText: "执行数据扫描、训练、识别或评估后，真实任务记录会显示在这里。"
                trailingText: "空"
            }

            Repeater {
                model: controller ? controller.tasks : []
                delegate: Components.StudioTableRow {
                    visible: root.statusFilter === "all" || modelData.status === root.statusFilter
                    primaryText: modelData.title
                    secondaryText: modelData.kind
                    trailingText: modelData.status
                }
            }
        }
    }
}


