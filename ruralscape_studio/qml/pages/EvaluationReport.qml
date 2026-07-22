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
            number: "06"
            title: qsTr("评估报告")
            description: qsTr("在指定数据划分上执行评估，并从实际结果生成报告。")
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 16

            Components.JournalCard {
                Layout.preferredWidth: 360
                title: qsTr("评估设置")
                subtitle: qsTr("选择数据划分后创建评估任务。")
                reducedMotion: root.reducedMotion

                ColumnLayout {
                    Layout.fillWidth: true
                    spacing: 12
                    Text {
                        text: qsTr("数据划分")
                        color: Theme.primaryText
                        font.family: Theme.fontFamily
                        font.pixelSize: 12
                        font.weight: Font.DemiBold
                    }
                    ComboBox {
                        Layout.fillWidth: true
                        model: [qsTr("训练集"), qsTr("验证集"), qsTr("测试集")]
                        currentIndex: root.controller && root.controller.split === "train" ? 0 : root.controller && root.controller.split === "test" ? 2 : 1
                        onActivated: if (root.controller) root.controller.selectSplit(currentIndex === 0 ? "train" : currentIndex === 2 ? "test" : "val")
                    }
                    Components.StudioButton {
                        Layout.fillWidth: true
                        text: qsTr("开始评估")
                        enabled: root.controller && !root.controller.running
                        reducedMotion: root.reducedMotion
                        onClicked: root.startRequested()
                    }
                }
            }

            Components.JournalCard {
                Layout.fillWidth: true
                title: qsTr("报告内容")
                subtitle: root.controller && root.controller.hasReport ? root.controller.reportPath : qsTr("尚无评估报告")
                reducedMotion: root.reducedMotion

                ColumnLayout {
                    Layout.fillWidth: true
                    spacing: 14
                    Components.StatusPill {
                        text: root.controller && root.controller.running ? qsTr("评估中") : root.controller && root.controller.hasReport ? qsTr("报告已生成") : qsTr("等待评估")
                        tone: root.controller && root.controller.hasReport ? "success" : "neutral"
                    }
                    Text {
                        Layout.fillWidth: true
                        text: root.controller && root.controller.hasReport ? qsTr("报告将显示真实的类别指标、混淆信息与导出路径。") : qsTr("完成评估后，这里显示实际计算结果；当前不展示示例指标。")
                        color: Theme.secondaryText
                        font.family: Theme.fontFamily
                        font.pixelSize: 14
                        wrapMode: Text.WordWrap
                    }
                    Components.StudioProgress {
                        Layout.fillWidth: true
                        visible: root.controller && root.controller.running
                        value: root.controller ? root.controller.progress : 0
                        reducedMotion: root.reducedMotion
                    }
                }
            }
        }
    }
}


