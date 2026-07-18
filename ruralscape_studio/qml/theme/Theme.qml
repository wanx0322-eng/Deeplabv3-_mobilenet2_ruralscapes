pragma Singleton
import QtQuick 2.15

QtObject {
    readonly property color paper: "#F2EBDD"
    readonly property color field: "#173C2D"
    readonly property color acid: "#C7F464"
    readonly property color primaryText: "#111827"
    readonly property color secondaryText: "#6B7280"

    readonly property color fieldHover: "#214B39"
    readonly property color fieldLight: "#2C5A45"
    readonly property color fieldMuted: "#3D6653"
    readonly property color fieldForeground: "#F6F8F6"
    readonly property color fieldForegroundMuted: "#A9BDB1"
    readonly property color acidHover: "#D5FF78"
    readonly property color accentDeep: "#71963E"

    readonly property color card: "#FAF6EC"
    readonly property color inputSurface: "#FFFCF5"
    readonly property color mutedSurface: "#E7E2D7"
    readonly property color divider: "#DED8CC"
    readonly property color border: "#C8C6BC"
    readonly property color borderStrong: "#A7B0A5"
    readonly property color quietHover: "#E6EED8"
    readonly property color disabledSurface: "#E4DED3"
    readonly property color disabledText: "#9CA3AF"
    readonly property color placeholderText: "#888F8B"

    readonly property color successSurface: "#E0EFD8"
    readonly property color warningSurface: "#F4E6B8"
    readonly property color errorSurface: "#F6D7D2"
    readonly property color errorText: "#7F2D27"
    readonly property color errorBorder: "#D89A91"
    readonly property color destructive: "#B84236"

    readonly property string fontFamily: Qt.application.font.family
    readonly property int cardRadius: 16
    readonly property int buttonRadius: 8
    readonly property int motionDuration: 200
}


