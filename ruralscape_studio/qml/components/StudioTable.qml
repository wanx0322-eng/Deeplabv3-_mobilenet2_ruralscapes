import "../theme"
import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

Control {
    id: root
    default property alias rows: rowColumn.data
    property string title: ""

    padding: 0
    implicitHeight: rowColumn.implicitHeight

    background: Rectangle {
        radius: Theme.cardRadius
        color: Theme.card
        border.width: 1
        border.color: Theme.border
    }

    contentItem: ColumnLayout {
        id: rowColumn
        spacing: 0

        Text {
            visible: root.title.length > 0
            text: root.title
            color: Theme.primaryText
            font.family: Theme.fontFamily
            font.pixelSize: 14
            font.weight: Font.DemiBold
            Layout.fillWidth: true
            Layout.leftMargin: 20
            Layout.rightMargin: 20
            Layout.topMargin: 16
            Layout.bottomMargin: 12
        }
    }
}


