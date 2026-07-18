import "../theme"
import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import "../components" as Components

Flickable {
    id: root
    property var controller
    property bool reducedMotion: false
    signal chooseImageRequested()
    signal saveVersionRequested()

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
            number: "03"
            title: "标注工作台"
            description: "在原图与掩膜之间切换，保存可追溯的标注版本。"
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 16

            Components.JournalCard {
                Layout.fillWidth: true
                title: "标注画布"
                subtitle: controller && controller.activeImagePath.length > 0 ? controller.activeImagePath : "尚未选择图像"
                reducedMotion: root.reducedMotion

                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 360
                    radius: 12
                    color: Theme.mutedSurface
                    border.width: 1
                    border.color: Theme.border
                    Text {
                        anchors.centerIn: parent
                        width: parent.width - 48
                        text: controller && controller.activeImagePath.length > 0 ? "图像已选择，等待加载标注内容。" : "选择图像后，在这里显示原图与掩膜。"
                        color: Theme.secondaryText
                        font.family: Theme.fontFamily
                        font.pixelSize: 14
                        horizontalAlignment: Text.AlignHCenter
                        wrapMode: Text.WordWrap
                    }
                }
            }

            Components.JournalCard {
                Layout.preferredWidth: 300
                title: "工具与版本"
                subtitle: controller ? "已保存版本：" + controller.versionCount : "已保存版本：0"
                reducedMotion: root.reducedMotion

                ColumnLayout {
                    Layout.fillWidth: true
                    spacing: 12
                    Components.StudioButton {
                        Layout.fillWidth: true
                        text: "选择图像"
                        reducedMotion: root.reducedMotion
                        onClicked: root.chooseImageRequested()
                    }
                    Components.StudioField {
                        Layout.fillWidth: true
                        label: "画笔尺寸"
                        placeholderText: "输入像素大小"
                        enabled: controller && controller.activeImagePath.length > 0
                    }
                    Components.StudioButton {
                        Layout.fillWidth: true
                        text: "保存版本"
                        accent: false
                        enabled: controller && controller.activeImagePath.length > 0 && controller.dirty
                        reducedMotion: root.reducedMotion
                        onClicked: root.saveVersionRequested()
                    }
                    Components.StatusPill {
                        text: controller && controller.dirty ? "有未保存修改" : "无未保存修改"
                        tone: controller && controller.dirty ? "warning" : "neutral"
                    }
                }
            }
        }
    }
}


