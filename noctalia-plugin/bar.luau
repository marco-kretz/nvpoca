--!nonstrict
-- Camera glyph: crossed out = daemon not running, outline = idle, filled + primary = an app
-- uses the camera. Left click opens the panel, right click toggles the effect.

local props = noctalia.state.get("nvpoca") or {}

local function render()
  local running = props.Enabled ~= nil
  barWidget.setGlyph(not running and "video-off" or props.Active and "video-filled" or "video")
  barWidget.setGlyphColor(props.Active and "primary" or running and "on_surface" or "on_surface_variant")
  if not running then
    barWidget.setTooltip("nvpoca: daemon not running")
  elseif props.Active then
    barWidget.setTooltip(`nvpoca: in use · {math.round(props.Fps or 0)} fps · {math.round(props.Latency or 0)} ms`)
  else
    barWidget.setTooltip(props.Loaded and "nvpoca: idle · models in VRAM" or "nvpoca: idle · VRAM free")
  end
end

noctalia.state.watch("nvpoca", function(value)
  props = value or {}
  render()
end)

function update()
  noctalia.setUpdateInterval(60000)
  render()
end

function onClick()
  noctalia.togglePanel("marco/nvpoca:panel")
end

function onRightClick()
  if props.Enabled ~= nil then
    noctalia.runAsync({ "busctl", "--user", "set-property", "org.nvpoca.Daemon", "/org/nvpoca/Daemon",
      "org.nvpoca.Daemon", "Enabled", "b", tostring(not props.Enabled) })
  end
end
