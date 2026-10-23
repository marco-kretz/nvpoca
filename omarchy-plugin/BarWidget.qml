import QtQuick
import Quickshell
import Quickshell.Io
import qs.Commons
import qs.Ui

Panel {
  id: root
  moduleName: "local.nvpoca"
  ipcTarget: "local.nvpoca"

  readonly property var dbus: ["org.nvpoca.Daemon", "/org/nvpoca/Daemon", "org.nvpoca.Daemon"]

  // Daemon properties (see nvpoca/__main__.py); empty while the daemon is not running.
  property var props: ({})
  readonly property bool running: props.Enabled !== undefined
  readonly property bool active: !!props.Active
  readonly property bool enabled: !!props.Enabled
  readonly property bool previewing: preview.running
  // v4l2loopback serves one reader at a time, so the preview only works while no app streams.
  readonly property bool canPreview: running && (!active || previewing)
  // filled = the real camera is open, outline = idle, crossed out = daemon down
  readonly property string icon: !running ? "\u{F0568}" : active ? "\u{F0567}" : "\u{F0BDC}"

  readonly property string statusText: !running ? "Daemon not running"
    : active ? (previewing ? "Preview" : "In use") + " · " + Math.round(props.Fps || 0) + " fps · " + (props.Latency || 0).toFixed(0) + " ms"
    : props.Loaded ? "Idle, camera off · models in VRAM" : "Idle, camera off · VRAM free"

  function refresh() { if (!getAll.running) getAll.running = true }

  function set(name, signature, value) {
    Quickshell.execDetached(["busctl", "--user", "set-property"].concat(dbus, [name, signature, String(value)]))
  }

  implicitWidth: button.implicitWidth
  implicitHeight: button.implicitHeight

  onOpenedChanged: if (opened) refresh()

  Process {
    id: getAll
    command: ["busctl", "--user", "--json=short", "call", "org.nvpoca.Daemon", "/org/nvpoca/Daemon",
      "org.freedesktop.DBus.Properties", "GetAll", "s", "org.nvpoca.Daemon"]
    stdout: StdioCollector {
      onStreamFinished: {
        var next = {}
        try {
          var data = JSON.parse(text).data[0]
          for (var k in data) next[k] = data[k].data
        } catch (e) {}  // daemon not running: busctl prints nothing to stdout
        root.props = next
      }
    }
  }

  // Event-driven: every PropertiesChanged / owner change line from the daemon triggers a re-read.
  Process {
    id: watch
    running: true
    command: ["gdbus", "monitor", "--session", "--dest", "org.nvpoca.Daemon", "--object-path", "/org/nvpoca/Daemon"]
    stdout: SplitParser { onRead: root.refresh() }
    onRunningChanged: if (!running) restartWatch.start()
  }

  Timer {
    id: restartWatch
    interval: 5000
    onTriggered: watch.running = true
  }

  // Fallback for a missed event, e.g. right after a shell reload.
  Timer {
    interval: 15000
    running: true
    repeat: true
    triggeredOnStart: true
    onTriggered: root.refresh()
  }

  Process {
    id: preview
    command: ["mpv", "--profile=low-latency", "--untimed", "--no-cache", "--no-osc", "--title=nvpoca preview",
      "--demuxer-lavf-format=video4linux2", "--demuxer-lavf-o=video_size=1920x1080,input_format=yuyv422",
      "av://v4l2:/dev/video10"]
  }

  WidgetButton {
    id: button
    anchors.fill: parent
    bar: root.bar
    text: root.icon
    opacity: root.running ? 1 : 0.5
    active: root.active
    activeColor: Color.accent
    tooltipText: ""
    onPressed: root.toggle()
  }

  KeyboardPanel {
    id: panel
    anchorItem: button
    owner: root
    bar: root.bar
    open: root.opened
    focusTarget: keyCatcher
    contentWidth: panel.fittedContentWidth(Style.space(340))
    contentHeight: panel.fittedContentHeight(column.implicitHeight)

    PanelKeyCatcher {
      id: keyCatcher
      anchors.fill: parent
      onCloseRequested: root.close()
      onTabRequested: function(direction) { root.switchPanel(direction) }

      Column {
        id: column
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.top: parent.top
        spacing: Style.space(14)

        PanelHero {
          foreground: root.bar.foreground
          fontFamily: root.bar.fontFamily
          title: "Background effect"
          meta: root.statusText
          iconComponent: Text {
            text: root.icon
            color: root.active ? Color.accent : root.bar.foreground
            font.family: root.bar.fontFamily
            font.pixelSize: Style.font.display
          }
          trailingControl: ToggleSwitch {
            checked: root.enabled
            interactive: root.running
            foreground: root.bar.foreground
            onToggled: root.set("Enabled", "b", !root.enabled)
          }
        }

        Column {
          width: parent.width
          spacing: Style.spacing.labelGap
          visible: root.running && root.enabled

          // Image only makes sense once a background is set (busctl ... Background s <path>).
          ButtonGroup {
            options: [{ value: "blur", label: "Blur" }, { value: "green", label: "Green" }]
              .concat(root.props.Background ? [{ value: "replace", label: "Image" }] : [])
            value: root.props.Mode || "blur"
            foreground: root.bar.foreground
            fontFamily: root.bar.fontFamily
            onChanged: function(value) { root.set("Mode", "s", value) }
          }

          Row {
            width: parent.width
            visible: (root.props.Mode || "blur") === "blur"
            spacing: Style.space(8)

            InfoText { id: blurLabel; text: "Blur"; opacity: 0.6; anchors.verticalCenter: parent.verticalCenter }
            PanelSlider {
              id: blurSlider
              bar: root.bar
              width: parent.width - blurLabel.implicitWidth - blurValue.implicitWidth - parent.spacing * 2
              minimum: 1
              maximum: 20
              step: 0.5
              value: root.props.Blur || 4
              onReleased: function(v) { root.set("Blur", "d", Math.round(v * 2) / 2) }
            }
            InfoText { id: blurValue; text: (Math.round(blurSlider.liveValue * 2) / 2).toFixed(1); anchors.verticalCenter: parent.verticalCenter }
          }

          Row {
            width: parent.width
            spacing: Style.space(8)

            InfoText { id: smoothLabel; text: "Steady edges"; opacity: 0.6; anchors.verticalCenter: parent.verticalCenter }
            Item { width: parent.width - smoothLabel.implicitWidth - smoothSwitch.width - parent.spacing * 2; height: 1 }
            ToggleSwitch {
              id: smoothSwitch
              checked: (root.props.Smooth || 0) > 0
              foreground: root.bar.foreground
              onToggled: root.set("Smooth", "d", checked ? 0 : 0.15)
            }
          }

          Row {
            width: parent.width
            spacing: Style.space(8)

            InfoText { id: qualityLabel; text: "High quality"; opacity: 0.6; anchors.verticalCenter: parent.verticalCenter }
            Item { width: parent.width - qualityLabel.implicitWidth - qualitySwitch.width - parent.spacing * 2; height: 1 }
            ToggleSwitch {
              id: qualitySwitch
              checked: !!root.props.Quality
              foreground: root.bar.foreground
              onToggled: root.set("Quality", "b", !checked)
            }
          }
        }

        PanelSeparator { foreground: root.bar.foreground }

        Button {
          width: parent.width
          text: root.previewing ? "Close preview" : "Preview"
          iconText: "\u{F06D0}"
          enabled: root.canPreview
          opacity: enabled ? 1 : 0.5
          tooltipText: root.canPreview ? "" : (root.running ? "An app is using the camera" : "Start the nvpoca daemon first")
          foreground: root.bar.foreground
          fontFamily: root.bar.fontFamily
          // SIGKILL: mpv's demuxer can sit in a blocking v4l2 read and then ignores SIGTERM
          onClicked: if (root.previewing) preview.signal(9); else preview.running = true
        }
      }
    }
  }

  component InfoText: Text {
    textFormat: Text.PlainText
    color: root.bar.foreground
    font.family: root.bar.fontFamily
    font.pixelSize: Style.font.bodySmall
  }
}
