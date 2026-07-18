import "../theme"
import QtQuick 2.15
import QtQuick.Controls 2.15

Dialog {
    id: root
    property string message: ""

    modal: true
    focus: true
    width: 440
    padding: 24
    standardButtons: Dialog.Ok | Dialog.Cancel
    closePolicy: Popup.CloseOnEscape

    background: Rectangle {
        radius: Theme.cardRadius
        color: Theme.inputSurface
        border.width: 1
        border.color: Theme.border
    }

    contentItem: Text {
        text: root.message
        color: Theme.primaryText
        font.family: Theme.fontFamily
        font.pixelSize: 14
        wrapMode: Text.WordWrap
    }
}


