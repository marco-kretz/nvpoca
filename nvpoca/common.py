"""Torch-free pieces shared by the supervisor, the GPU worker and compare."""

import argparse
from pathlib import Path

from linuxpy.video.device import BufferType, Device


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nvpoca", description="GPU background effects -> v4l2loopback")
    p.add_argument("--mode", choices=["blur", "replace", "green", "passthrough"], default="blur")
    p.add_argument("--bg", type=Path, help="background image for --mode replace")
    p.add_argument("--backend", choices=["trt", "torch"], default="trt")
    p.add_argument("--variant", choices=["mobilenetv3", "resnet50"], default="mobilenetv3")
    p.add_argument("--ratio", type=float, help="RVM downsample_ratio (default: 480px on the long side, 0.25 @1080p)")
    p.add_argument("--smooth", type=float, nargs="?", const=0.15, default=0.15, metavar="MIN_RATE",
                   help="damp alpha flicker where the image is static; MIN_RATE in (0, 1], lower = steadier (default 0.15)")
    p.add_argument("--no-smooth", dest="smooth", action="store_const", const=None, help="raw RVM alpha, for comparison")
    p.add_argument("--quality", action="store_true",
                   help="SAM2Matting on top of RVM: solid headset/hair, no chair; ~18 ms GPU per frame instead of ~1.5")
    p.add_argument("--keep-chair", action="store_true", help="keep what sits at your depth (chair) via a depth model")
    p.add_argument("--blur", type=float, default=4, help="blur sigma in output pixels at 1080p (subtle 4, portrait ~5-10, heavy 16+)")
    p.add_argument("--record", type=Path, metavar="CLIP", help="keep the last 10 s of raw camera MJPEG, write to CLIP on exit")
    p.add_argument("--input", default="/dev/video0")
    p.add_argument("--output", default="/dev/video10")
    p.add_argument("--size", default="1920x1080")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--unload-after", type=float, default=60, metavar="SECONDS",
                   help="stop the GPU worker (frees VRAM) this long after the last app stopped reading (default 60)")
    return p


def finish_args(p: argparse.ArgumentParser, args: argparse.Namespace) -> argparse.Namespace:
    if args.mode == "replace" and not args.bg:
        p.error("--mode replace needs --bg")
    args.width, args.height = map(int, args.size.split("x"))
    if args.ratio is None:
        args.ratio = min(1.0, 480 / max(args.width, args.height))
    return args


def configure_camera(cam: Device, w: int, h: int, fps: int) -> None:
    cam.set_format(BufferType.VIDEO_CAPTURE, w, h, "MJPG")
    cam.set_fps(BufferType.VIDEO_CAPTURE, fps)
    if "exposure_dynamic_framerate" in cam.controls:
        cam.controls.exposure_dynamic_framerate.value = False  # keep 30 fps in low light
    fmt = cam.get_format(BufferType.VIDEO_CAPTURE)
    if (fmt.width, fmt.height) != (w, h):
        raise SystemExit(f"camera refused {w}x{h} MJPG, got {fmt.width}x{fmt.height}")
