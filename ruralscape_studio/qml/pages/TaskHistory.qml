pragma ComponentBehavior: Bound

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
            title: qsTr("任务记录")
            description: qsTr("查看本次工作期间实际创建的扫描、训练、识别与评估任务。")
        }

        Components.JournalCard {
            Layout.fillWidth: true
            title: qsTr("任务筛选")
            subtitle: root.controller && root.controller.runningCount > 0 ? qsTr("有任务正在运行") : qsTr("当前没有运行中的任务")
            reducedMotion: root.reducedMotion

            RowLayout {
                Layout.fillWidth: true
                spacing: 12
                ComboBox {
                    id: filterSelect
                    Layout.preferredWidth: 180
                    model: [qsTr("全部状态"), qsTr("等待中"), qsTr("运行中"), qsTr("已完成"), qsTr("失败")]
                    onActivated: root.statusFilter = ["all", "queued", "running", "completed", "failed"][currentIndex]
                }
                Components.StudioButton {
                    text: qsTr("清空筛选")
                    quiet: true
                    reducedMotion: root.reducedMotion
                    onClicked: {
                        filterSelect.currentIndex = 0
                        root.statusFilter = "all"
                    }
                }
                Item { Layout.fillWidth: true }
                Components.StatusPill {
                    text: root.controller ? qsTr("运行中 ") + root.controller.runningCount : qsTr("运行中 0")
                    tone: root.controller && root.controller.runningCount > 0 ? "success" : "neutral"
                }
            }
        }

        Components.StudioTable {
            Layout.fillWidth: true
            title: qsTr("任务")

            Components.StudioTableRow {
                visible: !root.controller || root.controller.tasks.length === 0
                primaryText: qsTr("等待任务")
                secondaryText: qsTr("执行数据扫描、训练、识别或评估后，真实任务记录会显示在这里。")
                trailingText: qsTr("空")
            }

            Repeater {
                model: root.controller ? root.controller.tasks : []
                delegate: Components.StudioTableRow {
                    required property var modelData
                    visible: root.statusFilter === "all" || modelData.status === root.statusFilter
                    primaryText: modelData.title
                    secondaryText: modelData.kind
                    trailingText: modelData.status
                }
            }
        }
    }
}


