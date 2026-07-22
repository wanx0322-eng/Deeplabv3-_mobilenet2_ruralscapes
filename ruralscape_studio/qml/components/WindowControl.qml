import "../theme"
import QtQuick 2.15
import QtQuick.Controls 2.15

Button {
    id: control
    property string kind: "minimize"

    implicitWidth: 46
    implicitHeight: 40
    hoverEnabled: true
    Accessible.name: control.kind === "close" ? qsTr("关闭窗口") : control.kind === "maximize" ? qsTr("最大化或还原窗口") : qsTr("最小化窗口")

    background: Rectangle {
        color: control.hovered ? (control.kind === "close" ? Theme.destructive : Theme.fieldLight) : "transparent"
    }

    contentItem: Item {
        Rectangle {
            visible: control.kind === "minimize"
            anchors.centerIn: parent
            width: 12
            height: 1
            color: Theme.fieldForeground
        }
        Rectangle {
            visible: control.kind === "maximize"
            anchors.centerIn: parent
            width: 11
            height: 9
            color: "transparent"
            border.width: 1
            border.color: Theme.fieldForeground
        }
        Item {
            visible: control.kind === "close"
            anchors.centerIn: parent
            width: 12
            height: 12
            Rectangle {
                anchors.centerIn: parent
                width: 14
                height: 1
                rotation: 45
                color: Theme.fieldForeground
            }
            Rectangle {
                anchors.centerIn: parent
                width: 14
                height: 1
                rotation: -45
                color: Theme.fieldForeground
            }
        }
    }
}


