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
            title: qsTr("标注工作台")
            description: qsTr("在原图与掩膜之间切换，保存可追溯的标注版本。")
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 16

            Components.JournalCard {
                Layout.fillWidth: true
                title: qsTr("标注画布")
                subtitle: root.controller && root.controller.activeImagePath.length > 0 ? root.controller.activeImagePath : qsTr("尚未选择图像")
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
                        text: root.controller && root.controller.activeImagePath.length > 0 ? qsTr("图像已选择，等待加载标注内容。") : qsTr("选择图像后，在这里显示原图与掩膜。")
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
                title: qsTr("工具与版本")
                subtitle: root.controller ? qsTr("已保存版本：") + root.controller.versionCount : qsTr("已保存版本：0")
                reducedMotion: root.reducedMotion

                ColumnLayout {
                    Layout.fillWidth: true
                    spacing: 12
                    Components.StudioButton {
                        Layout.fillWidth: true
                        text: qsTr("选择图像")
                        reducedMotion: root.reducedMotion
                        onClicked: root.chooseImageRequested()
                    }
                    Components.StudioField {
                        Layout.fillWidth: true
                        label: qsTr("画笔尺寸")
                        placeholderText: qsTr("输入像素大小")
                        enabled: root.controller && root.controller.activeImagePath.length > 0
                    }
                    Components.StudioButton {
                        Layout.fillWidth: true
                        text: qsTr("保存版本")
                        accent: false
                        enabled: root.controller && root.controller.activeImagePath.length > 0 && root.controller.dirty
                        reducedMotion: root.reducedMotion
                        onClicked: root.saveVersionRequested()
                    }
                    Components.StatusPill {
                        text: root.controller && root.controller.dirty ? qsTr("有未保存修改") : qsTr("无未保存修改")
                        tone: root.controller && root.controller.dirty ? "warning" : "neutral"
                    }
                }
            }
        }
    }
}


