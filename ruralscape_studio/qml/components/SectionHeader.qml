import "../theme"
import QtQuick 2.15
import QtQuick.Layouts 1.15

Item {
    id: root
    property string number: "01"
    property string title: ""
    property string description: ""

    implicitHeight: 92

    RowLayout {
        anchors.fill: parent
        spacing: 20

        Rectangle {
            Layout.preferredWidth: 58
            Layout.fillHeight: true
            radius: Theme.cardRadius
            color: Theme.field

            Text {
                anchors.left: parent.left
                anchors.leftMargin: 12
                anchors.bottom: parent.bottom
                anchors.bottomMargin: 10
                text: root.number
                color: Theme.acid
                font.family: Theme.fontFamily
                font.pixelSize: 26
                font.weight: Font.Black
            }
        }

        ColumnLayout {
            Layout.fillWidth: true
            spacing: 7

            Text {
                text: root.title
                color: Theme.primaryText
                font.family: Theme.fontFamily
                font.pixelSize: 30
                font.weight: Font.Bold
            }
            Text {
                text: root.description
                color: Theme.secondaryText
                font.family: Theme.fontFamily
                font.pixelSize: 14
                wrapMode: Text.WordWrap
                Layout.fillWidth: true
            }
        }
    }
}


