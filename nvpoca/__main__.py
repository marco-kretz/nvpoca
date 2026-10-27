"""nvpoca supervisor: keeps the virtual camera ready, starts the GPU worker (nvpoca.worker)
when an app reads it, stops the worker after --unload-after seconds without readers (frees
the VRAM), and serves the settings over D-Bus (org.nvpoca.Daemon on the session bus).
No torch/CUDA in this process, so it stays small while idle."""

import asyncio
import json
import os
import select
import signal
import struct
import sys
import threading
from pathlib import Path

from dbus_fast import BusType, DBusError, ErrorType, RequestNameReply
from dbus_fast.aio import MessageBus
from dbus_fast.service import PropertyAccess, ServiceInterface, dbus_property
from linuxpy.video.device import BufferType, Device

from nvpoca.common import finish_args, parser

BUS_NAME = INTERFACE = "org.nvpoca.Daemon"
OBJECT_PATH = "/org/nvpoca/Daemon"
CLIENT_USAGE = 0x08000000 + 0x08E00000 + 1  # v4l2loopback V4L2_EVENT_PRI_CLIENT_USAGE (v4l2loopback.c)
SETTINGS = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "nvpoca" / "settings.json"
# saved setting -> CLI flag that overrides it when given explicitly
CLI_FLAG = {"enabled": "mode", "mode": "mode", "blur": "blur", "smooth": "smooth", "light": "light", "bg": "bg"}


def load_settings(args, explicit: set[str]) -> dict:
    settings = {
        "enabled": args.mode != "passthrough",
        "mode": args.mode if args.mode != "passthrough" else "blur",
        "blur": args.blur,
        "smooth": args.smooth or 0.0,
        "light": args.light or 0.0,
        "bg": str(args.bg) if args.bg else "",
    }
    try:
        saved = json.loads(SETTINGS.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        saved = {}
    for key in settings:
        if key in saved and CLI_FLAG[key] not in explicit:
            settings[key] = saved[key]
    return settings


def watch_readers(out: Device, on_change) -> None:
    """Thread: v4l2loopback reports 1 when an app starts streaming, 0 when it stops or dies."""
    out.subscribe_event(CLIENT_USAGE)
    poll = select.poll()
    poll.register(out.fileno(), select.POLLPRI)
    while True:
        poll.poll()
        on_change(struct.unpack_from("I", bytes(out.deque_event().u))[0])


class Supervisor:
    def __init__(self, args, worker_argv: list[str], settings: dict, out: Device, black: bytes):
        self.args, self.worker_argv, self.settings = args, worker_argv, settings
        self.out, self.out_fd, self.black = out, out.fileno(), black
        self.worker = None
        self.spawning = False
        self.unload = None
        self.readers = 0
        self.loaded = self.active = False
        self.fps = self.latency = 0.0
        self.changed = lambda *names: None  # replaced once the D-Bus object exists

    def update(self, key: str, value) -> None:
        self.settings[key] = value
        SETTINGS.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS.write_text(json.dumps(self.settings, indent=2))
        self.send(settings=self.settings)

    def send(self, **msg) -> None:
        if self.worker:
            self.worker.stdin.write((json.dumps(msg) + "\n").encode())

    async def on_readers(self, count: int) -> None:
        self.readers = count
        if count and not self.worker and not self.spawning:
            await self.start_worker()
        self.sync()

    def sync(self) -> None:
        """Bring the worker in line with the reader count; idempotent, so event order doesn't matter."""
        if not self.worker:
            return
        self.send(stream=bool(self.readers))
        if self.readers and self.unload:
            self.unload.cancel()
            self.unload = None
        elif not self.readers and not self.unload:
            self.unload = asyncio.get_running_loop().call_later(self.args.unload_after, self.stop_worker)

    async def start_worker(self) -> None:
        self.spawning = True
        try:
            self.worker = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "nvpoca.worker", *self.worker_argv, "--out-fd", str(self.out_fd),
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, pass_fds=(self.out_fd,),
            )
        finally:
            self.spawning = False
        print(f"worker {self.worker.pid} started", flush=True)
        self.loaded = True
        self.changed("Loaded")
        self.send(settings=self.settings)
        asyncio.create_task(self.follow(self.worker))

    def stop_worker(self) -> None:
        self.unload = None
        if self.worker:
            print(f"no reader for {self.args.unload_after:g} s, stopping worker {self.worker.pid}", flush=True)
            self.worker.stdin.close()  # EOF = quit; follow() notices the exit
            self.worker = None

    async def follow(self, proc) -> None:
        async for line in proc.stdout:
            msg = json.loads(line)
            if "active" in msg:
                self.active = msg["active"]
                self.fps = self.latency = 0.0
                self.changed("Active", "Fps", "Latency")
            if "fps" in msg:
                self.fps, self.latency = float(msg["fps"]), float(msg["latency"])
                self.changed("Fps", "Latency")
        rc = await proc.wait()
        print(f"worker {proc.pid} exited ({rc})", flush=True)
        if self.worker is proc:  # crashed rather than stopped by us
            self.worker = None
            if self.unload:
                self.unload.cancel()
                self.unload = None
        if not self.worker:
            os.write(self.out_fd, self.black)  # don't leave a reader on the last processed frame
            self.loaded = self.active = False
            self.fps = self.latency = 0.0
            self.changed("Loaded", "Active", "Fps", "Latency")
        if rc != 0 and self.readers:
            await asyncio.sleep(2)
            await self.on_readers(self.readers)


def invalid(msg: str) -> DBusError:
    return DBusError(ErrorType.INVALID_ARGS, msg)


class Daemon(ServiceInterface):
    def __init__(self, sup: Supervisor):
        super().__init__(INTERFACE)
        self.sup = sup

    def changed(self, *names: str) -> None:
        self.emit_properties_changed({n: getattr(self, n) for n in names})

    @dbus_property()
    def Enabled(self) -> "b":
        return self.sup.settings["enabled"]

    @Enabled.setter
    def Enabled(self, value: "b"):
        self.sup.update("enabled", value)

    @dbus_property()
    def Mode(self) -> "s":
        return self.sup.settings["mode"]

    @Mode.setter
    def Mode(self, value: "s"):
        if value not in ("blur", "replace", "green"):
            raise invalid("Mode must be 'blur', 'replace' or 'green'")
        self.sup.update("mode", value)

    @dbus_property()
    def Blur(self) -> "d":
        return self.sup.settings["blur"]

    @Blur.setter
    def Blur(self, value: "d"):
        if not 0.5 <= value <= 40:
            raise invalid("Blur must be within 0.5..40")
        self.sup.update("blur", value)

    @dbus_property()
    def Smooth(self) -> "d":
        return self.sup.settings["smooth"]

    @Smooth.setter
    def Smooth(self, value: "d"):
        if not 0 <= value <= 1:
            raise invalid("Smooth must be within 0..1 (0 = off)")
        self.sup.update("smooth", value)

    @dbus_property()
    def Light(self) -> "d":
        return self.sup.settings["light"]

    @Light.setter
    def Light(self, value: "d"):
        if not 0 <= value <= 1:
            raise invalid("Light must be within 0..1 (0 = off)")
        self.sup.update("light", value)

    @dbus_property()
    def Background(self) -> "s":
        return self.sup.settings["bg"]

    @Background.setter
    def Background(self, value: "s"):
        path = Path(value).expanduser() if value else None
        if path and not path.is_file():  # decoding happens in the worker, which falls back to blur
            raise invalid(f"no such file: {path}")
        self.sup.update("bg", str(path or ""))

    @dbus_property(access=PropertyAccess.READ)
    def Active(self) -> "b":
        """The real camera is open because an app reads the virtual one."""
        return self.sup.active

    @dbus_property(access=PropertyAccess.READ)
    def Loaded(self) -> "b":
        """The GPU worker is running (models in VRAM)."""
        return self.sup.loaded

    @dbus_property(access=PropertyAccess.READ)
    def Fps(self) -> "d":
        return self.sup.fps

    @dbus_property(access=PropertyAccess.READ)
    def Latency(self) -> "d":
        """Capture-to-output latency in ms."""
        return self.sup.latency


async def serve(sup: Supervisor) -> None:
    bus = await MessageBus(bus_type=BusType.SESSION).connect()
    daemon = Daemon(sup)
    bus.export(OBJECT_PATH, daemon)
    if await bus.request_name(BUS_NAME) != RequestNameReply.PRIMARY_OWNER:
        raise SystemExit(f"{BUS_NAME} is already taken: is another nvpoca running?")
    sup.changed = daemon.changed
    loop = asyncio.get_running_loop()
    threading.Thread(
        target=watch_readers,
        args=(sup.out, lambda n: loop.call_soon_threadsafe(asyncio.ensure_future, sup.on_readers(n))),
        daemon=True, name="readers",
    ).start()
    print(f"ready: {sup.args.output} {sup.args.width}x{sup.args.height} YUYV; "
          f"GPU worker starts with the first reader, stops {sup.args.unload_after:g} s after the last", flush=True)
    await bus.wait_for_disconnect()


def on_sigterm(*_):
    raise KeyboardInterrupt  # systemd stop: same clean exit path as Ctrl+C


def main():
    p = parser()
    args = finish_args(p, p.parse_args())
    defaults = p.parse_args([])
    explicit = {k for k in set(CLI_FLAG.values()) if getattr(args, k) != getattr(defaults, k)}
    settings = load_settings(args, explicit)
    signal.signal(signal.SIGTERM, on_sigterm)

    w, h = args.width, args.height
    black = bytes([16, 128]) * (w * h)  # YUYV black
    with Device(args.output) as out:
        out.set_format(BufferType.VIDEO_OUTPUT, w, h, "YUYV")
        out.set_fps(BufferType.VIDEO_OUTPUT, args.fps)
        os.write(out.fileno(), black)  # a frame up front so a new reader gets an image while the worker loads
        sup = Supervisor(args, sys.argv[1:], settings, out, black)
        try:
            asyncio.run(serve(sup))
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
