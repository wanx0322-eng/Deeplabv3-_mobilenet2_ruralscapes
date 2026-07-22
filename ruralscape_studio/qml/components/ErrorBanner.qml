import "../theme"
import QtQuick 2.15

Rectangle {
    id: root
    property string message: ""
    signal dismissed()

    visible: root.message.length > 0
    implicitHeight: visible ? 48 : 0
    radius: Theme.buttonRadius
    color: Theme.errorSurface
    border.width: 1
    border.color: Theme.errorBorder

    Text {
        anchors.left: parent.left
        anchors.right: dismiss.left
        anchors.verticalCenter: parent.verticalCenter
        anchors.leftMargin: 14
        anchors.rightMargin: 10
        text: root.message
        color: Theme.errorText
        font.family: Theme.fontFamily
        font.pixelSize: 13
        elide: Text.ElideRight
    }

    Text {
        id: dismiss
        anchors.right: parent.right
        anchors.rightMargin: 14
        anchors.verticalCenter: parent.verticalCenter
        text: qsTr("关闭")
        color: Theme.errorText
        font.family: Theme.fontFamily
        font.pixelSize: 12
        font.weight: Font.DemiBold
        TapHandler { onTapped: root.dismissed() }
    }
}


