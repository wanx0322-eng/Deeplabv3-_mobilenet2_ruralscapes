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
            title: qsTr("模型与导出")
            description: qsTr("查看项目模型产物，选择检查点并导出可交付文件。")
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 16
            Components.JournalCard {
                Layout.preferredWidth: 300
                title: qsTr("模型产物")
                subtitle: qsTr("当前项目已登记的模型数量")
                reducedMotion: root.reducedMotion
                Text {
                    text: root.controller ? root.controller.artifactCount : 0
                    color: Theme.field
                    font.family: Theme.fontFamily
                    font.pixelSize: 36
                    font.weight: Font.Bold
                }
            }
            Components.JournalCard {
                Layout.fillWidth: true
                title: qsTr("已选模型")
                subtitle: root.controller && root.controller.selectedModelPath.length > 0 ? root.controller.selectedModelPath : qsTr("尚未选择模型文件")
                reducedMotion: root.reducedMotion
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 10
                    Components.StudioButton {
                        text: qsTr("选择模型")
                        reducedMotion: root.reducedMotion
                        onClicked: root.chooseModelRequested()
                    }
                    Components.StudioButton {
                        text: qsTr("导出")
                        accent: false
                        enabled: root.controller && root.controller.selectedModelPath.length > 0
                        reducedMotion: root.reducedMotion
                        onClicked: root.exportRequested()
                    }
                }
            }
        }

        Components.StudioTable {
            Layout.fillWidth: true
            title: qsTr("模型列表")
            Components.StudioTableRow {
                primaryText: root.controller && root.controller.artifactCount > 0 ? qsTr("模型产物已登记") : qsTr("当前项目没有模型产物")
                secondaryText: root.controller && root.controller.artifactCount > 0 ? qsTr("选择实际模型文件以查看和导出。") : qsTr("完成训练或导入已有检查点后，模型会出现在这里。")
                trailingText: root.controller && root.controller.artifactCount > 0 ? qsTr("选择") : qsTr("空")
            }
        }
    }
}


