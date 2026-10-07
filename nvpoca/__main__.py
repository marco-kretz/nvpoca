"""nvpoca daemon: keeps the virtual camera ready, opens the real camera only while an app
reads it, and takes its settings over D-Bus (org.nvpoca.Daemon on the session bus)."""

import asyncio
import json
import os
import select
import signal
import struct
import threading
import time
import traceback
from collections import defaultdict, deque
from pathlib import Path

import torch
from dbus_fast import BusType, DBusError, ErrorType, RequestNameReply
from dbus_fast.aio import MessageBus
from dbus_fast.service import PropertyAccess, ServiceInterface, dbus_property
from linuxpy.video.device import BufferType, Device, VideoCapture

from nvpoca import gpu
from nvpoca.core import Effect, configure_camera, decode, finish_args, load_image, parser

BUS_NAME = INTERFACE = "org.nvpoca.Daemon"
OBJECT_PATH = "/org/nvpoca/Daemon"
CLIENT_USAGE = 0x08000000 + 0x08E00000 + 1  # v4l2loopback V4L2_EVENT_PRI_CLIENT_USAGE (v4l2loopback.c)
SETTINGS = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "nvpoca" / "settings.json"
# saved setting -> CLI flag that overrides it when given explicitly
CLI_FLAG = {"enabled": "mode", "mode": "mode", "blur": "blur", "smooth": "smooth", "bg": "bg"}


class State:
    """Shared by the D-Bus side (asyncio, main thread) and the video thread. Only the video thread
    touches the GPU pipeline; single attribute writes are atomic under the GIL."""

    def __init__(self, settings: dict):
        self.settings = settings
        self.bg_image = load_image(Path(settings["bg"])) if settings["bg"] else None
        self.dirty = threading.Event()
        self.active = False
        self.fps = 0.0
        self.latency_ms = 0.0
        self.notify = lambda *names: None  # replaced once the D-Bus side is up

    def update(self, key: str, value) -> None:
        self.settings[key] = value
        SETTINGS.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS.write_text(json.dumps(self.settings, indent=2))
        self.dirty.set()

    def apply(self, effect: Effect) -> None:
        if not self.dirty.is_set():
            return
        self.dirty.clear()
        s = self.settings
        effect.mode = s["mode"] if s["enabled"] else "passthrough"
        effect.set_blur(s["blur"])
        if (effect.smoother.min_rate if effect.smoother else 0) != s["smooth"]:
            effect.set_smooth(s["smooth"] or None)
        effect.set_background(self.bg_image)


def load_settings(args, explicit: set[str]) -> dict:
    settings = {
        "enabled": args.mode != "passthrough",
        "mode": args.mode if args.mode != "passthrough" else "blur",
        "blur": args.blur,
        "smooth": args.smooth or 0.0,
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


def pending_readers(out: Device, poll: select.poll, timeout_ms: int | None) -> int | None:
    """Latest v4l2loopback reader count (0/1) if an event arrived, else None."""
    count = None
    while poll.poll(timeout_ms):
        count = struct.unpack_from("I", bytes(out.deque_event().u))[0]
        timeout_ms = 0
    return count


def video(args, state: State, recent: deque | None) -> None:
    w, h = args.width, args.height
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_stream(torch.cuda.Stream())  # TensorRT syncs extra on the legacy default stream
    effect = Effect(args)
    state.dirty.set()
    out_buf = torch.empty(h, w * 2, dtype=torch.uint8, pin_memory=True)
    black = torch.tensor([16, 128], dtype=torch.uint8).repeat(h, w).numpy()

    with Device(args.output) as out:
        out.set_format(BufferType.VIDEO_OUTPUT, w, h, "YUYV")
        out.set_fps(BufferType.VIDEO_OUTPUT, args.fps)
        # a frame up front so a reader that starts before the camera is up gets an image at once
        out.write(black)
        out.subscribe_event(CLIENT_USAGE)
        poll = select.poll()
        poll.register(out.fileno(), select.POLLPRI)
        print(f"ready: {args.output} {w}x{h} YUYV, camera {args.input} opens on demand", flush=True)

        while True:
            if pending_readers(out, poll, None) != 1:
                continue
            state.active = True
            state.notify("Active")
            try:
                stream(args, state, effect, out, out_buf, poll, recent)
            finally:
                state.active, state.fps, state.latency_ms = False, 0.0, 0.0
                effect.reset()
                out.write(black)
                state.notify("Active", "Fps", "Latency")
            print("reader gone, camera closed", flush=True)


def stream(args, state: State, effect: Effect, out: Device, out_buf: torch.Tensor, poll, recent) -> None:
    with Device(args.input) as cam:
        configure_camera(cam, args.width, args.height, args.fps)
        print(f"reader attached, camera {args.input} open", flush=True)
        stats, frames, dropped = defaultdict(float), 0, 0
        report_at = time.monotonic() + 1
        with VideoCapture(cam, size=2) as capture:
            t = time.perf_counter()
            for frame in capture:
                stage = {}

                def lap(name, sync=True):
                    nonlocal t
                    if sync:
                        torch.cuda.synchronize()
                    now = time.perf_counter()
                    stage[name] = now - t
                    t = now

                lap("capture", sync=False)
                state.apply(effect)
                raw = frame.data
                if recent is not None:
                    recent.append(raw)
                try:
                    src = decode(torch.frombuffer(bytearray(raw), dtype=torch.uint8))  # JPEG upload happens inside
                except RuntimeError as e:  # corrupt MJPEG frames happen on USB hiccups
                    print(f"skip frame: {e}", flush=True)
                    dropped += 1
                    t = time.perf_counter()
                    continue
                lap("decode")
                fgr, pha, _ = effect.matte(src)
                lap("infer")
                yuyv = gpu.to_yuyv(effect.compose(src, fgr, pha))
                lap("compose")
                out_buf.copy_(yuyv)
                lap("download")
                out.write(out_buf.numpy())
                lap("write", sync=False)
                # frame timestamp is CLOCK_MONOTONIC at USB start-of-frame -> true capture-to-output latency
                stage["latency"] = time.clock_gettime(time.CLOCK_MONOTONIC) - frame.timestamp

                for k, v in stage.items():
                    stats[k] += v
                frames += 1
                if time.monotonic() >= report_at:
                    report_at += 1
                    state.fps, state.latency_ms = frames, stats["latency"] / frames * 1000
                    state.notify("Fps", "Latency")
                    line = " ".join(f"{k}={v / frames * 1000:.1f}" for k, v in stats.items())
                    print(f"fps={frames:2d} drop={dropped} ms: {line}", flush=True)
                    stats.clear()
                    frames = dropped = 0
                if pending_readers(out, poll, 0) == 0:
                    return


def invalid(msg: str) -> DBusError:
    return DBusError(ErrorType.INVALID_ARGS, msg)


class Daemon(ServiceInterface):
    def __init__(self, state: State):
        super().__init__(INTERFACE)
        self.state = state

    def changed(self, *names: str) -> None:
        self.emit_properties_changed({n: getattr(self, n) for n in names})

    @dbus_property()
    def Enabled(self) -> "b":
        return self.state.settings["enabled"]

    @Enabled.setter
    def Enabled(self, value: "b"):
        self.state.update("enabled", value)

    @dbus_property()
    def Mode(self) -> "s":
        return self.state.settings["mode"]

    @Mode.setter
    def Mode(self, value: "s"):
        if value not in ("blur", "replace"):
            raise invalid("Mode must be 'blur' or 'replace'")
        self.state.update("mode", value)

    @dbus_property()
    def Blur(self) -> "d":
        return self.state.settings["blur"]

    @Blur.setter
    def Blur(self, value: "d"):
        if not 0.5 <= value <= 40:
            raise invalid("Blur must be within 0.5..40")
        self.state.update("blur", value)

    @dbus_property()
    def Smooth(self) -> "d":
        return self.state.settings["smooth"]

    @Smooth.setter
    def Smooth(self, value: "d"):
        if not 0 <= value <= 1:
            raise invalid("Smooth must be within 0..1 (0 = off)")
        self.state.update("smooth", value)

    @dbus_property()
    def Background(self) -> "s":
        return self.state.settings["bg"]

    @Background.setter
    def Background(self, value: "s"):
        path = Path(value).expanduser() if value else None
        try:
            self.state.bg_image = load_image(path) if path else None
        except (OSError, RuntimeError) as e:
            raise invalid(f"cannot load background image: {e}") from e
        self.state.update("bg", str(path or ""))

    @dbus_property(access=PropertyAccess.READ)
    def Active(self) -> "b":
        """The real camera is open because an app reads the virtual one."""
        return self.state.active

    @dbus_property(access=PropertyAccess.READ)
    def Fps(self) -> "d":
        return float(self.state.fps)

    @dbus_property(access=PropertyAccess.READ)
    def Latency(self) -> "d":
        """Capture-to-output latency in ms."""
        return float(self.state.latency_ms)


async def serve(state: State) -> None:
    bus = await MessageBus(bus_type=BusType.SESSION).connect()
    daemon = Daemon(state)
    bus.export(OBJECT_PATH, daemon)
    if await bus.request_name(BUS_NAME) != RequestNameReply.PRIMARY_OWNER:
        raise SystemExit(f"{BUS_NAME} is already taken: is another nvpoca running?")
    loop = asyncio.get_running_loop()
    state.notify = lambda *names: loop.call_soon_threadsafe(daemon.changed, *names)
    await bus.wait_for_disconnect()


def on_sigterm(*_):
    raise KeyboardInterrupt  # systemd stop: same clean exit path as Ctrl+C


def main():
    p = parser()
    args = finish_args(p, p.parse_args())
    defaults = p.parse_args([])
    explicit = {k for k in set(CLI_FLAG.values()) if getattr(args, k) != getattr(defaults, k)}
    state = State(load_settings(args, explicit))
    if args.mode == "passthrough":
        args.mode = "blur"  # keep the model loaded so the effect can be switched on later
    recent = deque(maxlen=10 * args.fps) if args.record else None

    def run_video():
        try:
            video(args, state, recent)
        except BaseException:
            traceback.print_exc()
            os._exit(1)  # without the video thread the daemon is useless; let systemd restart it

    signal.signal(signal.SIGTERM, on_sigterm)
    threading.Thread(target=run_video, daemon=True, name="video").start()
    try:
        asyncio.run(serve(state))
    except KeyboardInterrupt:
        pass
    finally:
        if recent:
            args.record.parent.mkdir(parents=True, exist_ok=True)
            args.record.write_bytes(b"".join(recent))  # same format as compare --record
            print(f"wrote last {len(recent)} frames to {args.record}", flush=True)


if __name__ == "__main__":
    main()
