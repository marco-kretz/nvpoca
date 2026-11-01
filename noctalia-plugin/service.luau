--!nonstrict
-- Mirrors org.nvpoca.Daemon's properties into noctalia.state["nvpoca"]; an empty table means
-- the daemon is not running.

local function refresh()
  noctalia.runAsync({ "busctl", "--user", "--json=short", "call", "org.nvpoca.Daemon", "/org/nvpoca/Daemon",
    "org.freedesktop.DBus.Properties", "GetAll", "s", "org.nvpoca.Daemon" }, function(res)
    local props = {}
    local reply = res.exitCode == 0 and noctalia.json.decode(res.stdout)
    if reply then
      for k, v in reply.data[1] do
        props[k] = v.data
      end
    end
    noctalia.state.set("nvpoca", props)
  end, 2000)
end

-- event-driven: every PropertiesChanged / name owner change line triggers a re-read
noctalia.runStream("gdbus monitor --session --dest org.nvpoca.Daemon --object-path /org/nvpoca/Daemon", function()
  refresh()
end)

-- fallback for a missed event, e.g. right after a shell reload
function update()
  noctalia.setUpdateInterval(15000)
  refresh()
end
