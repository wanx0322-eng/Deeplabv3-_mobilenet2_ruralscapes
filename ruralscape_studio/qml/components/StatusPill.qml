import "../theme"
import QtQuick 2.15

Rectangle {
    id: root
    property string text: ""
    property string tone: "neutral"

    implicitWidth: label.implicitWidth + 20
    implicitHeight: 28
    radius: 14
    color: root.tone === "success" ? Theme.successSurface : root.tone === "warning" ? Theme.warningSurface : root.tone === "error" ? Theme.errorSurface : Theme.mutedSurface

    Text {
        id: label
        anchors.centerIn: parent
        text: root.text
        color: root.tone === "error" ? Theme.errorText : Theme.fieldMuted
        font.family: Theme.fontFamily
        font.pixelSize: 12
        font.weight: Font.DemiBold
    }
}


