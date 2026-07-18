import "../theme"
import QtQuick 2.15
import QtQuick.Controls 2.15

Button {
    id: control
    property bool accent: true
    property bool quiet: false
    property bool reducedMotion: false

    implicitHeight: 40
    implicitWidth: Math.max(104, contentItem.implicitWidth + 32)
    leftPadding: 16
    rightPadding: 16
    hoverEnabled: true
    font.family: Theme.fontFamily
    font.pixelSize: 14
    font.weight: Font.DemiBold

    contentItem: Text {
        text: control.text
        color: !control.enabled ? Theme.disabledText : control.quiet ? Theme.fieldMuted : control.accent ? Theme.field : Theme.fieldForeground
        font: control.font
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
        elide: Text.ElideRight
    }

    background: Rectangle {
        radius: Theme.buttonRadius
        color: !control.enabled ? Theme.disabledSurface : control.quiet ? (control.hovered ? Theme.quietHover : "transparent") : control.accent ? (control.hovered ? Theme.acidHover : Theme.acid) : (control.hovered ? Theme.fieldMuted : Theme.field)
        border.width: control.quiet ? 1 : 0
        border.color: Theme.borderStrong
        Behavior on color {
            ColorAnimation {
                duration: control.reducedMotion ? 0 : Theme.motionDuration
                easing.type: Easing.InOutQuad
            }
        }
    }
}


