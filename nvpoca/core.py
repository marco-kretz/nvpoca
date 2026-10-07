"""Shared by the live loop (__main__) and the offline comparison (compare)."""

import argparse
from pathlib import Path

import torch
from linuxpy.video.device import BufferType, Device
from torchvision.io import ImageReadMode, decode_image, decode_jpeg, read_file

from nvpoca import gpu
from nvpoca.rvm.model import MattingNetwork

MODELS = Path(__file__).resolve().parent.parent / "models"


def decode(data: torch.Tensor) -> torch.Tensor:
    """MJPEG bytes (CPU uint8) -> (1, 3, H, W) half on GPU via nvJPEG. Raises RuntimeError on corrupt data."""
    # torchvision's nvJPEG decoder runs on its own stream and can write into allocator blocks our
    # stream still uses; without this sync, frames silently corrupt once the GPU queue gets long
    torch.cuda.current_stream().synchronize()
    return decode_jpeg(data, device="cuda")[None].half().div_(255)


class Matter:
    """RVM with its recurrent state carried across frames."""

    def __init__(self, variant: str, ratio: float, h: int, w: int):
        self.model = MattingNetwork(variant).eval().cuda()
        self.model.load_state_dict(torch.load(MODELS / f"rvm_{variant}.pth", map_location="cuda"))
        self.model.half()
        self.ratio, self.rec = ratio, [None] * 4
        # warm up cuDNN autotuning before the camera streams, else the first frames queue up for seconds
        for _ in range(3):
            self(torch.zeros(1, 3, h, w, device="cuda", dtype=torch.half))
        self.rec = [None] * 4

    @torch.inference_mode()
    def __call__(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fgr, pha, *self.rec = self.model(src, *self.rec, downsample_ratio=self.ratio)
        return fgr, pha


class Effect:
    def __init__(self, args: argparse.Namespace):
        h, w = args.height, args.width
        self.matter = None
        if args.mode != "passthrough":
            if args.backend == "trt":
                from nvpoca.trt import TrtMatter

                self.matter = TrtMatter(MODELS, args.variant, args.ratio, h, w)
            else:
                self.matter = Matter(args.variant, args.ratio, h, w)
        self.smoother = gpu.AlphaSmoother(args.smooth) if args.smooth else None
        self.bg = None
        if args.mode == "replace":
            img = decode_image(read_file(str(args.bg)), mode=ImageReadMode.RGB).cuda()
            self.bg = gpu.cover(img[None].half() / 255, h, w)
        self.sigma = args.blur * h / 1080

    def matte(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(fgr, pha); passthrough returns src with full alpha."""
        if not self.matter:
            return src, torch.ones_like(src[:, :1])
        fgr, pha = self.matter(src)
        if self.smoother:
            pha = self.smoother(pha, src)
        return fgr, pha

    def compose(self, src: torch.Tensor, fgr: torch.Tensor, pha: torch.Tensor) -> torch.Tensor:
        if not self.matter:
            return src
        back = self.bg if self.bg is not None else gpu.blurred_background(src, pha, self.sigma)
        return gpu.composite(fgr, pha, back)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nvpoca", description="GPU background effects -> v4l2loopback")
    p.add_argument("--mode", choices=["blur", "replace", "passthrough"], default="blur")
    p.add_argument("--bg", type=Path, help="background image for --mode replace")
    p.add_argument("--backend", choices=["trt", "torch"], default="trt")
    p.add_argument("--variant", choices=["mobilenetv3", "resnet50"], default="mobilenetv3")
    p.add_argument("--ratio", type=float, help="RVM downsample_ratio (default: 480px on the long side, 0.25 @1080p)")
    p.add_argument("--smooth", type=float, nargs="?", const=0.15, metavar="MIN_RATE",
                   help="damp alpha flicker where the image is static; MIN_RATE in (0, 1], lower = steadier (default 0.15)")
    p.add_argument("--blur", type=float, default=7, help="blur sigma in output pixels at 1080p (portrait ~5-10, heavy 16+)")
    p.add_argument("--record", type=Path, metavar="CLIP", help="keep the last 10 s of raw camera MJPEG, write to CLIP on exit")
    p.add_argument("--input", default="/dev/video0")
    p.add_argument("--output", default="/dev/video10")
    p.add_argument("--size", default="1920x1080")
    p.add_argument("--fps", type=int, default=30)
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
