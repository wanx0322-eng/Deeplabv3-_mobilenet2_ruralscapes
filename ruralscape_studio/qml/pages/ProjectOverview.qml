import "../theme"
import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import "../components" as Components

Flickable {
    id: root
    property var controller
    property bool reducedMotion: false
    signal createRequested()
    signal openRequested()

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
            number: "01"
            title: qsTr("项目概览")
            description: qsTr("建立项目工作目录，并确认当前分割引擎与数据位置。")
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 16

            Components.JournalCard {
                Layout.fillWidth: true
                title: qsTr("当前项目")
                subtitle: root.controller && root.controller.hasProject ? qsTr("项目已就绪") : qsTr("尚未打开项目")
                reducedMotion: root.reducedMotion

                ColumnLayout {
                    Layout.fillWidth: true
                    spacing: 12
                    Components.StatusPill {
                        text: root.controller && root.controller.hasProject ? qsTr("已打开") : qsTr("未选择")
                        tone: root.controller && root.controller.hasProject ? "success" : "neutral"
                    }
                    Text {
                        Layout.fillWidth: true
                        text: root.controller && root.controller.hasProject ? root.controller.projectName : qsTr("选择已有项目，或创建新的项目目录。")
                        color: Theme.primaryText
                        font.family: Theme.fontFamily
                        font.pixelSize: 15
                        wrapMode: Text.WordWrap
                    }
                    Text {
                        Layout.fillWidth: true
                        visible: root.controller && root.controller.hasProject
                        text: root.controller ? root.controller.rootPath : ""
                        color: Theme.secondaryText
                        font.family: Theme.fontFamily
                        font.pixelSize: 12
                        elide: Text.ElideMiddle
                    }
                }
            }

            Components.JournalCard {
                Layout.preferredWidth: 360
                title: qsTr("项目操作")
                subtitle: qsTr("项目数据保存在所选工作目录内。")
                reducedMotion: root.reducedMotion

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 10
                    Components.StudioButton {
                        text: qsTr("新建项目")
                        reducedMotion: root.reducedMotion
                        onClicked: root.createRequested()
                    }
                    Components.StudioButton {
                        text: qsTr("打开项目")
                        accent: false
                        reducedMotion: root.reducedMotion
                        onClicked: root.openRequested()
                    }
                }
            }
        }

        Components.JournalCard {
            Layout.fillWidth: true
            title: qsTr("开始工作")
            subtitle: qsTr("项目就绪后，可从左侧依次检查数据集、标注、训练与评估。")
            reducedMotion: root.reducedMotion

            Text {
                Layout.fillWidth: true
                text: root.controller && root.controller.hasProject ? qsTr("继续前往数据集，核对图像、掩膜与划分。") : qsTr("打开项目后，工作台会显示真实的目录状态与任务记录。")
                color: Theme.primaryText
                font.family: Theme.fontFamily
                font.pixelSize: 14
                wrapMode: Text.WordWrap
            }
        }
    }
}


