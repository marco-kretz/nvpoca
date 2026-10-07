"""Shared by the live loop (__main__) and the offline comparison (compare)."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
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


def load_image(path: Path) -> torch.Tensor:
    """(3, H, W) uint8 RGB on CPU; raises on unreadable files."""
    return decode_image(read_file(str(path)), mode=ImageReadMode.RGB)


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

    def reset(self) -> None:
        self.rec = [None] * 4

    @torch.inference_mode()
    def __call__(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fgr, pha, *self.rec = self.model(src, *self.rec, downsample_ratio=self.ratio)
        return fgr, pha


class Depth:
    """Depth Anything V2 Small: relative disparity at a fixed ~518 px input."""

    SIZE = (294, 518)  # multiples of the 14 px ViT patch, ~16:9

    def __init__(self):
        from nvpoca.depth.dpt import DepthAnythingV2

        self.model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
        self.model.load_state_dict(torch.load(MODELS / "depth_anything_v2_vits.pth", map_location="cpu"))
        self.model = self.model.eval().cuda().half()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device="cuda", dtype=torch.half).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device="cuda", dtype=torch.half).view(1, 3, 1, 1)

    @torch.inference_mode()
    def __call__(self, src: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(src, size=self.SIZE, mode="bilinear", antialias=True)
        return self.model((x - self.mean) / self.std).unsqueeze(1).float()


class Effect:
    """Matting + composition. mode/blur/smooth/background can change between frames (daemon)."""

    def __init__(self, args: argparse.Namespace):
        self.h, self.w = args.height, args.width
        self.matter = None
        if args.mode != "passthrough":
            if args.backend == "trt":
                from nvpoca.trt import TrtMatter

                self.matter = TrtMatter(MODELS, args.variant, args.ratio, self.h, self.w)
            else:
                self.matter = Matter(args.variant, args.ratio, self.h, self.w)
        self.depth = Depth() if args.keep_chair and self.matter else None
        self.mode = args.mode
        self.set_smooth(args.smooth)
        self.set_blur(args.blur)
        self.set_background(load_image(args.bg) if args.bg else None)

    def set_smooth(self, min_rate: float | None) -> None:
        self.smoother = gpu.AlphaSmoother(min_rate) if min_rate else None

    def set_blur(self, sigma_1080p: float) -> None:
        self.sigma = sigma_1080p * self.h / 1080

    def set_background(self, img: torch.Tensor | None) -> None:
        """img: (3, H, W) uint8 on any device, or None."""
        self.bg = gpu.cover(img.cuda()[None].half() / 255, self.h, self.w) if img is not None else None

    def reset(self) -> None:
        """Forget temporal state, e.g. when the camera restarts."""
        if self.matter:
            self.matter.reset()
        if self.smoother:
            self.smoother.prev = None

    def matte(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(fgr, pha, raw pha before smoothing); passthrough returns src with full alpha."""
        if not self.matter or self.mode == "passthrough":
            one = torch.ones_like(src[:, :1])
            return src, one, one
        fgr, pha = self.matter(src)
        if self.depth:
            fgr, pha = self.add_near(src, fgr, pha)
        raw = pha
        if self.smoother:
            pha = self.smoother(pha, src)
        return fgr, pha, raw

    def add_near(self, src, fgr, pha):
        """Union the person with whatever sits at their depth (chair); the chair uses camera pixels,
        since RVM's foreground colour is only meaningful on the person."""
        disp = self.depth(src)
        person = F.interpolate(pha.float(), size=disp.shape[-2:], mode="area")
        refs = gpu.depth_refs(disp, person)
        if refs is None:
            return fgr, pha
        near = gpu.near_mask(disp, person, *refs)
        gray = src.float().mean(1, keepdim=True)
        near = gpu.guided_filter(F.interpolate(gray, size=disp.shape[-2:], mode="area"), near, gray).clamp(0, 1)
        alpha = torch.maximum(pha.float(), near)
        fgr = torch.lerp(src.float(), fgr.float(), pha.float() / alpha.clamp(min=1e-3))
        return fgr.to(src.dtype), alpha.to(src.dtype)

    def compose(self, src: torch.Tensor, fgr: torch.Tensor, pha: torch.Tensor) -> torch.Tensor:
        if not self.matter or self.mode == "passthrough":
            return src
        if self.mode == "replace" and self.bg is not None:
            back = self.bg
        else:
            back = gpu.blurred_background(src, pha, self.sigma)
        return gpu.composite(fgr, pha, back)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nvpoca", description="GPU background effects -> v4l2loopback")
    p.add_argument("--mode", choices=["blur", "replace", "passthrough"], default="blur")
    p.add_argument("--bg", type=Path, help="background image for --mode replace")
    p.add_argument("--backend", choices=["trt", "torch"], default="trt")
    p.add_argument("--variant", choices=["mobilenetv3", "resnet50"], default="mobilenetv3")
    p.add_argument("--ratio", type=float, help="RVM downsample_ratio (default: 480px on the long side, 0.25 @1080p)")
    p.add_argument("--smooth", type=float, nargs="?", const=0.15, default=0.15, metavar="MIN_RATE",
                   help="damp alpha flicker where the image is static; MIN_RATE in (0, 1], lower = steadier (default 0.15)")
    p.add_argument("--no-smooth", dest="smooth", action="store_const", const=None, help="raw RVM alpha, for comparison")
    p.add_argument("--keep-chair", action="store_true", help="keep what sits at your depth (chair) via a depth model")
    p.add_argument("--blur", type=float, default=4, help="blur sigma in output pixels at 1080p (subtle 4, portrait ~5-10, heavy 16+)")
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
