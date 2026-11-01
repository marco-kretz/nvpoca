--!nonstrict
-- Live controls. Values go to the daemon over D-Bus; the service's state update re-renders.

local props = noctalia.state.get("nvpoca") or {}
-- Light = 0 means off, so the strength to restore when switching back on lives here
local lightStrength = 0.5
local MODES = { "blur", "green", "replace" }

local function set(name, signature, value)
  noctalia.runAsync({ "busctl", "--user", "set-property", "org.nvpoca.Daemon", "/org/nvpoca/Daemon",
    "org.nvpoca.Daemon", name, signature, tostring(value) })
end

local function section(title, control)
  return ui.column({ gap = 6 }, {
    ui.label({ text = title, fontWeight = "medium", color = "on_surface_variant" }),
    control,
  })
end

local function render()
  local running = props.Enabled ~= nil
  local status = not running and "Daemon not running"
    or props.Active and `In use · {math.round(props.Fps or 0)} fps · {math.round(props.Latency or 0)} ms`
    or props.Loaded and "Idle, camera off · models in VRAM"
    or "Idle, camera off · VRAM free"
  -- image needs a background (busctl ... Background s <path.jpg>)
  local modes = props.Background and props.Background ~= "" and MODES or { "blur", "green" }
  local modeIndex = (table.find(modes, props.Mode) or 1) - 1
  local light = props.Light or 0
  if light > 0 then
    lightStrength = light
  end

  panel.render(ui.column({ flexGrow = 1, gap = 14 }, {
    ui.label({ text = "nvpoca camera", fontSize = 16, fontWeight = "bold", color = "on_surface" }),
    ui.label({ text = status, color = props.Active and "primary" or "on_surface_variant" }),
    ui.row({ gap = 12, align = "center" }, {
      ui.toggle({ checked = props.Enabled == true, enabled = running, onChange = function(v) set("Enabled", "b", v) end }),
      ui.label({ text = "Background effect", flexGrow = 1 }),
    }),
    section("Mode", ui.select({ options = modes, selectedIndex = modeIndex, enabled = running and props.Enabled == true,
      onChange = function(i) set("Mode", "s", modes[(tonumber(i) or 0) + 1]) end })),
    section(`Blur · {props.Blur or 0}`, ui.slider({ min = 1, max = 20, step = 0.5, value = props.Blur or 4,
      enabled = running and props.Mode == "blur",
      onChange = function(v) set("Blur", "d", v) end })),
    ui.row({ gap = 12, align = "center" }, {
      ui.toggle({ checked = (props.Smooth or 0) > 0, enabled = running,
        onChange = function(v) set("Smooth", "d", v == "true" and 0.15 or 0) end }),
      ui.label({ text = "Steady edges", flexGrow = 1 }),
    }),
    ui.row({ gap = 12, align = "center" }, {
      ui.toggle({ checked = light > 0, enabled = running,
        onChange = function(v) set("Light", "d", v == "true" and lightStrength or 0) end }),
      ui.label({ text = "Key light", flexGrow = 1 }),
    }),
    ui.slider({ min = 0.1, max = 1, step = 0.1, value = lightStrength, enabled = running and light > 0,
      onChange = function(v) set("Light", "d", v) end }),
    -- v4l2loopback serves one reader at a time, so the preview only works while no app streams
    ui.button({ text = "Preview", glyph = "player-play", enabled = running and not props.Active,
      onClick = function()
        noctalia.runAsync({ "ffplay", "-loglevel", "error", "-fflags", "nobuffer", "-window_title", "nvpoca preview",
          "-f", "v4l2", "-input_format", "yuyv422", "/dev/video10" })
      end }),
  }))
end

noctalia.state.watch("nvpoca", function(value)
  props = value or {}
  render()
end)

function onOpen(_context)
  props = noctalia.state.get("nvpoca") or {}
  render()
end
