import "../theme"
import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import "../components" as Components

Flickable {
    id: root
    property var controller
    property bool reducedMotion: false
    signal chooseDirectoryRequested()
    signal scanRequested()

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
            number: "02"
            title: qsTr("数据集")
            description: qsTr("选择语义分割数据目录，扫描图像、掩膜、类别值与数据划分。")
        }

        Components.JournalCard {
            Layout.fillWidth: true
            title: qsTr("数据来源")
            subtitle: root.controller && root.controller.datasetPath.length > 0 ? root.controller.datasetPath : qsTr("尚未选择数据集目录")
            reducedMotion: root.reducedMotion

            RowLayout {
                Layout.fillWidth: true
                spacing: 12
                Components.StudioButton {
                    text: qsTr("选择数据集目录")
                    reducedMotion: root.reducedMotion
                    onClicked: root.chooseDirectoryRequested()
                }
                Components.StudioButton {
                    text: qsTr("扫描数据集")
                    accent: false
                    enabled: root.controller && root.controller.datasetPath.length > 0
                    reducedMotion: root.reducedMotion
                    onClicked: root.scanRequested()
                }
                Item { Layout.fillWidth: true }
                Components.StatusPill {
                    text: root.controller && root.controller.indexState === "ready" ? qsTr("索引完成") : qsTr("尚未索引")
                    tone: root.controller && root.controller.indexState === "ready" ? "success" : "neutral"
                }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 16
            Components.JournalCard {
                Layout.fillWidth: true
                title: qsTr("图像")
                subtitle: qsTr("已识别的原始图像")
                reducedMotion: root.reducedMotion
                Text {
                    text: root.controller ? root.controller.imageCount : 0
                    color: Theme.field
                    font.family: Theme.fontFamily
                    font.pixelSize: 32
                    font.weight: Font.Bold
                }
            }
            Components.JournalCard {
                Layout.fillWidth: true
                title: qsTr("掩膜")
                subtitle: qsTr("已匹配的标注掩膜")
                reducedMotion: root.reducedMotion
                Text {
                    text: root.controller ? root.controller.maskCount : 0
                    color: Theme.field
                    font.family: Theme.fontFamily
                    font.pixelSize: 32
                    font.weight: Font.Bold
                }
            }
            Components.JournalCard {
                Layout.fillWidth: true
                title: qsTr("待处理问题")
                subtitle: qsTr("缺失、重复或无法读取")
                reducedMotion: root.reducedMotion
                Text {
                    text: root.controller ? root.controller.issueCount : 0
                    color: Theme.field
                    font.family: Theme.fontFamily
                    font.pixelSize: 32
                    font.weight: Font.Bold
                }
            }
        }

        Components.StudioTable {
            Layout.fillWidth: true
            title: qsTr("数据检查")
            Components.StudioTableRow {
                primaryText: root.controller && root.controller.indexState === "ready" ? qsTr("索引结果可供检查") : qsTr("等待扫描数据集")
                secondaryText: root.controller && root.controller.indexState === "ready" ? qsTr("查看类别值和数据划分后再进入训练。") : qsTr("选择目录后执行扫描，这里将显示真实检查结果。")
                trailingText: root.controller && root.controller.indexState === "ready" ? qsTr("查看") : qsTr("未开始")
            }
        }
    }
}


