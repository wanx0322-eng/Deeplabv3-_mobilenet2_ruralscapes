import "../theme"
import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import "../components" as Components

Flickable {
    id: root
    property var controller
    property bool reducedMotion: false
    signal chooseModelRequested()
    signal exportRequested()

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
            number: "07"
            title: "模型与导出"
            description: "查看项目模型产物，选择检查点并导出可交付文件。"
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 16
            Components.JournalCard {
                Layout.preferredWidth: 300
                title: "模型产物"
                subtitle: "当前项目已登记的模型数量"
                reducedMotion: root.reducedMotion
                Text {
                    text: controller ? controller.artifactCount : 0
                    color: Theme.field
                    font.family: Theme.fontFamily
                    font.pixelSize: 36
                    font.weight: Font.Bold
                }
            }
            Components.JournalCard {
                Layout.fillWidth: true
                title: "已选模型"
                subtitle: controller && controller.selectedModelPath.length > 0 ? controller.selectedModelPath : "尚未选择模型文件"
                reducedMotion: root.reducedMotion
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 10
                    Components.StudioButton {
                        text: "选择模型"
                        reducedMotion: root.reducedMotion
                        onClicked: root.chooseModelRequested()
                    }
                    Components.StudioButton {
                        text: "导出"
                        accent: false
                        enabled: controller && controller.selectedModelPath.length > 0
                        reducedMotion: root.reducedMotion
                        onClicked: root.exportRequested()
                    }
                }
            }
        }

        Components.StudioTable {
            Layout.fillWidth: true
            title: "模型列表"
            Components.StudioTableRow {
                primaryText: controller && controller.artifactCount > 0 ? "模型产物已登记" : "当前项目没有模型产物"
                secondaryText: controller && controller.artifactCount > 0 ? "选择实际模型文件以查看和导出。" : "完成训练或导入已有检查点后，模型会出现在这里。"
                trailingText: controller && controller.artifactCount > 0 ? "选择" : "空"
            }
        }
    }
}


