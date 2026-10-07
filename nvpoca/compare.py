"""Record a camera clip once, replay it through several configs, write a labelled grid video.

    python -m nvpoca.compare clips/chair.mjpeg --record 10
    python -m nvpoca.compare clips/chair.mjpeg -c "" -c="--variant resnet50" -c="--smooth" [--alpha]

(use -c="..." so argparse does not read the flags inside as its own)

Prints two scores per config (x1000, lower is better):
  flicker: mean |alpha change| where the camera image did not change (jitter not explained by motion)
  lag:     mean |smoothed - raw alpha| where the image did change (ghosting from --smooth)
  soft:    share of semi-transparent pixels (0.05 < alpha < 0.95); catches half-kept
           background objects (e.g. a chair headrest at alpha 0.3) that flicker misses
"""

import argparse
import math
import re
import shlex
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from linuxpy.video.device import Device, VideoCapture

from nvpoca.core import Effect, configure_camera, decode, finish_args, parser

SOI, EOI = b"\xff\xd8", b"\xff\xd9"


def record(clip: Path, seconds: float, device: str, w: int, h: int) -> None:
    clip.parent.mkdir(parents=True, exist_ok=True)
    with Device(device) as cam, clip.open("wb") as f:
        configure_camera(cam, w, h, 30)
        with VideoCapture(cam, size=2) as stream:
            print(f"recording {seconds:g}s to {clip} ...", flush=True)
            end = time.monotonic() + seconds
            for frame in stream:
                f.write(frame.data)  # concatenated JPEGs; ffplay -f mjpeg plays this directly
                if time.monotonic() >= end:
                    break


def load(clip: Path) -> list[torch.Tensor]:
    # UVC MJPEG frames carry no embedded thumbnails, so EOI+SOI only occurs at frame boundaries
    parts = clip.read_bytes().split(EOI + SOI)
    parts = [(b"" if i == 0 else SOI) + p + (b"" if i == len(parts) - 1 else EOI) for i, p in enumerate(parts)]
    return [torch.frombuffer(bytearray(p), dtype=torch.uint8) for p in parts]


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> float:
    return (x * mask).sum().item() / max(mask.sum().item(), 1)


class Scores:
    """Computed 4x downscaled, which averages out sensor noise."""

    def __init__(self):
        self.prev, self.flicker, self.lag, self.soft, self.n = None, 0.0, 0.0, 0.0, 0

    def __call__(self, gray: torch.Tensor, static: torch.Tensor | None, pha: torch.Tensor, raw: torch.Tensor) -> None:
        a, r = (F.interpolate(t.float(), size=gray.shape[-2:], mode="area") for t in (pha, raw))
        if self.prev is not None and static is not None:
            self.flicker += masked_mean((a - self.prev).abs(), static)
            self.lag += masked_mean((a - r).abs(), 1 - static)
            self.soft += ((a > 0.05) & (a < 0.95)).float().mean().item()
            self.n += 1
        self.prev = a

    def __str__(self) -> str:
        n = max(self.n, 1)
        return f"flicker={1000 * self.flicker / n:6.3f}  lag={1000 * self.lag / n:6.3f}  soft={1000 * self.soft / n:5.1f}"


def label(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9 ._=/-]", "", s) or "default"


def main():
    ap = argparse.ArgumentParser(prog="nvpoca.compare", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", type=Path)
    ap.add_argument("--record", type=float, metavar="SECONDS", help="record a new clip first")
    ap.add_argument("-c", "--config", action="append", default=[], help='nvpoca flags, e.g. "--variant resnet50 --smooth"')
    ap.add_argument("--alpha", action="store_true", help="render the alpha matte instead of the composite")
    ap.add_argument("--input", default="/dev/video0")
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--out", type=Path, help="output video (default: next to the clip)")
    a = ap.parse_args()

    if a.record:
        w, h = map(int, a.size.split("x"))
        record(a.clip, a.record, a.input, w, h)
    if not a.config:
        return

    torch.cuda.set_stream(torch.cuda.Stream())
    frames = load(a.clip)
    first = decode(frames[0])
    h, w = first.shape[-2:]
    p = parser()
    effects = []
    for cfg in a.config:
        args = p.parse_args([*shlex.split(cfg), "--size", f"{w}x{h}"])
        effects.append((label(cfg), Effect(finish_args(p, args)), Scores()))

    cols = math.ceil(math.sqrt(len(effects)))
    rows = math.ceil(len(effects) / cols)
    out = a.out or a.clip.with_name(f"{a.clip.stem}-{'alpha' if a.alpha else 'comp'}-{time.strftime('%H%M%S')}.mp4")
    text = ",".join(
        f"drawtext=text='{name}':x={(i % cols) * w + 20}:y={(i // cols) * h + 20}:fontsize=40:fontcolor=white:box=1:boxcolor=black@0.6"
        for i, (name, _, _) in enumerate(effects)
    )
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{cols * w}x{rows * h}",
         "-r", "30", "-i", "-", "-vf", text, "-c:v", "hevc_nvenc", "-cq", "21", "-pix_fmt", "yuv420p", str(out)],
        stdin=subprocess.PIPE,
    )
    grid = torch.zeros(rows * h, cols * w, 3, dtype=torch.uint8, device="cuda")
    host = torch.empty_like(grid, device="cpu").pin_memory()
    prev_gray = None
    for data in frames:
        try:
            src = decode(data)
        except RuntimeError:
            continue
        gray = F.interpolate(src.float().mean(1, keepdim=True), scale_factor=0.25, mode="area")
        static = None if prev_gray is None else ((gray - prev_gray).abs() < 3 / 255).float()
        prev_gray = gray
        for i, (_, effect, scores) in enumerate(effects):
            if effect.smoother:
                fgr, raw = effect.matter(src)
                pha = effect.smoother(raw, src)
            else:
                fgr, pha = effect.matte(src)
                raw = pha
            scores(gray, static, pha, raw)
            img = pha.expand(-1, 3, -1, -1) if a.alpha else effect.compose(src, fgr, pha)
            y, x = (i // cols) * h, (i % cols) * w
            grid[y : y + h, x : x + w] = (img[0].permute(1, 2, 0).float() * 255).round().clamp(0, 255).byte()
        host.copy_(grid)
        ffmpeg.stdin.write(host.numpy().tobytes())
    ffmpeg.stdin.close()
    ffmpeg.wait()

    print(f"{len(frames)} frames -> {out}")
    for name, _, scores in effects:
        print(f"  {scores}  {name}")


if __name__ == "__main__":
    main()
