"""GPU worker, started on demand by the supervisor (__main__) and stopped when idle so the
VRAM is freed (a CUDA context only goes away with its process).

stdin  (JSON lines from the supervisor): {"settings": {...}}, {"stream": true|false}; EOF = quit
stdout (JSON lines to the supervisor):   {"ready": true}, {"active": bool}, {"fps": n, "latency": ms}
Logs go to stderr. Frames are written to the loopback fd inherited from the supervisor
(--out-fd), so they go through the supervisor's v4l2loopback opener.
"""

import time

STARTED = time.monotonic()  # before the heavy imports, so "ready in" covers torch/TensorRT loading

import json
import os
import signal
import sys
import threading
from collections import defaultdict, deque
from pathlib import Path

import torch
from linuxpy.video.device import Device, VideoCapture

from nvpoca import gpu
from nvpoca.common import configure_camera, finish_args, parser
from nvpoca.core import Effect, decode, load_image


def send(**msg) -> None:
    try:
        print(json.dumps(msg), flush=True)
    except BrokenPipeError:  # supervisor already gone (shutdown); stdin EOF stops us right after
        sys.stdout = open(os.devnull, "w")


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


class Control:
    """Commands from the supervisor; read on a thread, applied by the main (GPU) thread."""

    def __init__(self):
        self.cond = threading.Condition()
        self.settings = None
        self.dirty = False
        self.streaming = False
        self.quit = False
        self.bg_path = None
        self.bg_image = None

    def listen(self) -> None:
        for line in sys.stdin:
            msg = json.loads(line)
            with self.cond:
                if "settings" in msg:
                    self.settings, self.dirty = msg["settings"], True
                if "stream" in msg:
                    self.streaming = msg["stream"]
                self.cond.notify_all()
        self.stop()

    def stop(self, *_) -> None:
        with self.cond:
            self.quit, self.streaming = True, False
            self.cond.notify_all()

    def apply(self, effect: Effect) -> None:
        with self.cond:
            if not self.dirty:
                return
            self.dirty, s = False, self.settings
        effect.mode = s["mode"] if s["enabled"] else "passthrough"
        effect.set_blur(s["blur"])
        if (effect.smoother.min_rate if effect.smoother else 0) != s["smooth"]:
            effect.set_smooth(s["smooth"] or None)
        if s["bg"] != self.bg_path:
            self.bg_path = s["bg"]
            try:
                self.bg_image = load_image(Path(s["bg"])) if s["bg"] else None
            except (OSError, RuntimeError) as e:
                log(f"background {s['bg']}: {e}; falling back to blur")
                self.bg_image = None
            effect.set_background(self.bg_image)


def write_frame(fd: int, frame) -> None:
    if os.write(fd, frame) != frame.nbytes:
        raise OSError("short write to the loopback device")


def stream(args, ctl: Control, effect: Effect, out_buf: torch.Tensor, recent: deque | None) -> None:
    with Device(args.input) as cam:
        configure_camera(cam, args.width, args.height, args.fps)
        log(f"camera {args.input} open, {time.monotonic() - STARTED:.1f} s after worker start")
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
                ctl.apply(effect)
                raw = frame.data
                if recent is not None:
                    recent.append(raw)
                try:
                    src = decode(torch.frombuffer(bytearray(raw), dtype=torch.uint8))  # JPEG upload happens inside
                except RuntimeError as e:  # corrupt MJPEG frames happen on USB hiccups
                    log(f"skip frame: {e}")
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
                write_frame(args.out_fd, out_buf.numpy())
                lap("write", sync=False)
                # frame timestamp is CLOCK_MONOTONIC at USB start-of-frame -> true capture-to-output latency
                stage["latency"] = time.clock_gettime(time.CLOCK_MONOTONIC) - frame.timestamp

                for k, v in stage.items():
                    stats[k] += v
                frames += 1
                if time.monotonic() >= report_at:
                    report_at += 1
                    send(fps=frames, latency=stats["latency"] / frames * 1000)
                    line = " ".join(f"{k}={v / frames * 1000:.1f}" for k, v in stats.items())
                    log(f"fps={frames:2d} drop={dropped} ms: {line}")
                    stats.clear()
                    frames = dropped = 0
                if not ctl.streaming:
                    return


def main():
    p = parser()
    p.add_argument("--out-fd", type=int, required=True)
    args = finish_args(p, p.parse_args())
    if args.mode == "passthrough":
        args.mode = "blur"  # the supervisor switches modes at runtime; keep the model loaded
    ctl = Control()
    signal.signal(signal.SIGTERM, ctl.stop)  # systemd stops the whole cgroup: still write --record
    threading.Thread(target=ctl.listen, daemon=True, name="control").start()

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_stream(torch.cuda.Stream())  # TensorRT syncs extra on the legacy default stream
    effect = Effect(args)
    out_buf = torch.empty(args.height, args.width * 2, dtype=torch.uint8, pin_memory=True)
    black = torch.tensor([16, 128], dtype=torch.uint8).repeat(args.height, args.width).numpy()
    recent = deque(maxlen=10 * args.fps) if args.record else None
    log(f"worker ready in {time.monotonic() - STARTED:.1f} s")
    send(ready=True)

    try:
        while True:
            with ctl.cond:
                ctl.cond.wait_for(lambda: ctl.streaming or ctl.quit, timeout=1)
                if ctl.quit:
                    break
                if not ctl.streaming:
                    continue
            send(active=True)
            try:
                stream(args, ctl, effect, out_buf, recent)
            finally:
                effect.reset()
                write_frame(args.out_fd, black)
                send(active=False)
                log("camera closed")
    finally:
        if recent:
            args.record.parent.mkdir(parents=True, exist_ok=True)
            args.record.write_bytes(b"".join(recent))  # same format as compare --record
            log(f"wrote last {len(recent)} frames to {args.record}")


if __name__ == "__main__":
    main()
