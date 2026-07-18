import "../theme"
import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

Control {
    id: root
    default property alias cardContent: contentColumn.data
    property string title: ""
    property string subtitle: ""
    property bool reducedMotion: false

    padding: 24
    implicitWidth: 320
    hoverEnabled: true

    background: Rectangle {
        radius: Theme.cardRadius
        color: Theme.card
        border.width: 1
        border.color: root.hovered ? Theme.borderStrong : Theme.border
    }

    contentItem: ColumnLayout {
        id: contentColumn
        spacing: 14

        Text {
            visible: root.title.length > 0
            text: root.title
            color: Theme.primaryText
            font.family: Theme.fontFamily
            font.pixelSize: 18
            font.weight: Font.DemiBold
            Layout.fillWidth: true
        }

        Text {
            visible: root.subtitle.length > 0
            text: root.subtitle
            color: Theme.secondaryText
            font.family: Theme.fontFamily
            font.pixelSize: 13
            wrapMode: Text.WordWrap
            Layout.fillWidth: true
        }
    }

    HoverHandler { id: hover }
    transform: Translate {
        y: hover.hovered && root.enabled ? -4 : 0
        Behavior on y {
            NumberAnimation {
                duration: root.reducedMotion ? 0 : Theme.motionDuration
                easing.type: Easing.InOutQuad
            }
        }
    }
}


