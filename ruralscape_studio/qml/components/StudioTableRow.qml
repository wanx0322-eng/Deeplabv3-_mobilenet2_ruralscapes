import "../theme"
import QtQuick 2.15
import QtQuick.Layouts 1.15

Item {
    id: root
    property string primaryText: ""
    property string secondaryText: ""
    property string trailingText: ""

    implicitHeight: 58
    Layout.fillWidth: true

    Rectangle {
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        height: 1
        color: Theme.divider
    }

    RowLayout {
        anchors.fill: parent
        anchors.leftMargin: 20
        anchors.rightMargin: 20
        spacing: 16

        ColumnLayout {
            Layout.fillWidth: true
            spacing: 3
            Text {
                text: root.primaryText
                color: Theme.primaryText
                font.family: Theme.fontFamily
                font.pixelSize: 14
                font.weight: Font.DemiBold
            }
            Text {
                visible: root.secondaryText.length > 0
                text: root.secondaryText
                color: Theme.secondaryText
                font.family: Theme.fontFamily
                font.pixelSize: 12
            }
        }
        Text {
            text: root.trailingText
            color: Theme.fieldMuted
            font.family: Theme.fontFamily
            font.pixelSize: 12
            font.weight: Font.DemiBold
        }
    }
}


