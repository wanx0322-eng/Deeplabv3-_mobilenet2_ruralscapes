import "../theme"
import QtQuick 2.15
import QtQuick.Controls 2.15

Control {
    id: root
    property string label: ""
    property alias text: input.text
    property alias placeholderText: input.placeholderText
    property bool readOnly: false
    signal editingFinished()

    implicitHeight: root.label.length > 0 ? 70 : 44
    implicitWidth: 240

    background: null
    contentItem: Column {
        spacing: 7

        Text {
            visible: root.label.length > 0
            text: root.label
            color: Theme.primaryText
            font.family: Theme.fontFamily
            font.pixelSize: 12
            font.weight: Font.DemiBold
        }

        TextField {
            id: input
            width: parent.width
            height: 44
            enabled: root.enabled
            readOnly: root.readOnly
            selectByMouse: true
            color: Theme.primaryText
            placeholderTextColor: Theme.placeholderText
            font.family: Theme.fontFamily
            font.pixelSize: 14
            leftPadding: 14
            rightPadding: 14
            onEditingFinished: root.editingFinished()
            background: Rectangle {
                radius: Theme.buttonRadius
                color: input.enabled ? Theme.inputSurface : Theme.disabledSurface
                border.width: input.activeFocus ? 2 : 1
                border.color: input.activeFocus ? Theme.accentDeep : Theme.border
            }
        }
    }
}


