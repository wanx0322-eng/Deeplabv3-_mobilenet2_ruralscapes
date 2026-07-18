import "../theme"
import QtQuick 2.15
import QtQuick.Controls 2.15

Control {
    id: root
    property real value: 0.0
    property bool reducedMotion: false

    implicitHeight: 8
    implicitWidth: 220

    background: Rectangle {
        radius: 4
        color: Theme.divider
    }

    contentItem: Item {
        Rectangle {
            width: parent.width * Math.max(0, Math.min(1, root.value))
            height: parent.height
            radius: 4
            color: Theme.accentDeep
            Behavior on width {
                NumberAnimation {
                    duration: root.reducedMotion ? 0 : Theme.motionDuration
                    easing.type: Easing.InOutQuad
                }
            }
        }
    }
}


