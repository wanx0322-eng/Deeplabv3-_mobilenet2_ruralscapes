import "../theme"
import QtQuick 2.15
import QtQuick.Controls 2.15

Control {
    id: root
    property string number: "01"
    property string label: ""
    property bool current: false
    property bool reducedMotion: false
    signal clicked()

    implicitHeight: 48
    hoverEnabled: true

    background: Rectangle {
        radius: Theme.buttonRadius
        color: root.current ? Theme.fieldLight : root.hovered ? Theme.fieldHover : "transparent"
        Behavior on color {
            ColorAnimation {
                duration: root.reducedMotion ? 0 : Theme.motionDuration
                easing.type: Easing.InOutQuad
            }
        }
    }

    contentItem: Row {
        leftPadding: 14
        rightPadding: 14
        spacing: 12

        Text {
            width: 28
            anchors.verticalCenter: parent.verticalCenter
            text: root.number
            color: root.current ? Theme.acid : Theme.fieldForegroundMuted
            font.family: Theme.fontFamily
            font.pixelSize: 11
            font.weight: Font.Bold
        }
        Text {
            anchors.verticalCenter: parent.verticalCenter
            text: root.label
            color: root.current ? Theme.fieldForeground : Theme.fieldForeground
            font.family: Theme.fontFamily
            font.pixelSize: 14
            font.weight: root.current ? Font.DemiBold : Font.Normal
        }
    }

    TapHandler { onTapped: root.clicked() }
}


