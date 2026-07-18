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
            title: "项目概览"
            description: "建立项目工作目录，并确认当前分割引擎与数据位置。"
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 16

            Components.JournalCard {
                Layout.fillWidth: true
                title: "当前项目"
                subtitle: controller && controller.hasProject ? "项目已就绪" : "尚未打开项目"
                reducedMotion: root.reducedMotion

                ColumnLayout {
                    Layout.fillWidth: true
                    spacing: 12
                    Components.StatusPill {
                        text: controller && controller.hasProject ? "已打开" : "未选择"
                        tone: controller && controller.hasProject ? "success" : "neutral"
                    }
                    Text {
                        Layout.fillWidth: true
                        text: controller && controller.hasProject ? controller.projectName : "选择已有项目，或创建新的项目目录。"
                        color: Theme.primaryText
                        font.family: Theme.fontFamily
                        font.pixelSize: 15
                        wrapMode: Text.WordWrap
                    }
                    Text {
                        Layout.fillWidth: true
                        visible: controller && controller.hasProject
                        text: controller ? controller.rootPath : ""
                        color: Theme.secondaryText
                        font.family: Theme.fontFamily
                        font.pixelSize: 12
                        elide: Text.ElideMiddle
                    }
                }
            }

            Components.JournalCard {
                Layout.preferredWidth: 360
                title: "项目操作"
                subtitle: "项目数据保存在所选工作目录内。"
                reducedMotion: root.reducedMotion

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 10
                    Components.StudioButton {
                        text: "新建项目"
                        reducedMotion: root.reducedMotion
                        onClicked: root.createRequested()
                    }
                    Components.StudioButton {
                        text: "打开项目"
                        accent: false
                        reducedMotion: root.reducedMotion
                        onClicked: root.openRequested()
                    }
                }
            }
        }

        Components.JournalCard {
            Layout.fillWidth: true
            title: "开始工作"
            subtitle: "项目就绪后，可从左侧依次检查数据集、标注、训练与评估。"
            reducedMotion: root.reducedMotion

            Text {
                Layout.fillWidth: true
                text: controller && controller.hasProject ? "继续前往数据集，核对图像、掩膜与划分。" : "打开项目后，工作台会显示真实的目录状态与任务记录。"
                color: Theme.primaryText
                font.family: Theme.fontFamily
                font.pixelSize: 14
                wrapMode: Text.WordWrap
            }
        }
    }
}


