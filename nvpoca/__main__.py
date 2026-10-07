import argparse
import time
from collections import defaultdict
from pathlib import Path

import torch
from linuxpy.video.device import BufferType, Device, VideoCapture
from torchvision.io import ImageReadMode, decode_image, decode_jpeg, read_file

from nvpoca import gpu
from nvpoca.rvm.model import MattingNetwork

MODELS = Path(__file__).resolve().parent.parent / "models"


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


def parse_args():
    p = argparse.ArgumentParser(prog="nvpoca", description="GPU background effects -> v4l2loopback")
    p.add_argument("--mode", choices=["blur", "replace", "passthrough"], default="blur")
    p.add_argument("--bg", type=Path, help="background image for --mode replace")
    p.add_argument("--variant", choices=["mobilenetv3", "resnet50"], default="mobilenetv3")
    p.add_argument("--ratio", type=float, help="RVM downsample_ratio (default: ~480px on the long side)")
    p.add_argument("--smooth", action="store_true", help="adaptive temporal smoothing of alpha")
    p.add_argument("--blur", type=float, default=16, help="blur sigma in output pixels at 1080p")
    p.add_argument("--input", default="/dev/video0")
    p.add_argument("--output", default="/dev/video10")
    p.add_argument("--size", default="1920x1080")
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()
    if args.mode == "replace" and not args.bg:
        p.error("--mode replace needs --bg")
    args.width, args.height = map(int, args.size.split("x"))
    if args.ratio is None:
        args.ratio = min(1.0, 480 / max(args.width, args.height))
    return args


def main():
    args = parse_args()
    w, h = args.width, args.height
    torch.backends.cudnn.benchmark = True

    matter = Matter(args.variant, args.ratio, h, w) if args.mode != "passthrough" else None
    smoother = gpu.AlphaSmoother() if args.smooth else None
    bg = None
    if args.mode == "replace":
        img = decode_image(read_file(str(args.bg)), mode=ImageReadMode.RGB).cuda()
        bg = gpu.cover(img[None].half() / 255, h, w)
    sigma = args.blur * h / 1080
    out_buf = torch.empty(h, w * 2, dtype=torch.uint8, pin_memory=True)

    with Device(args.input) as cam, Device(args.output) as out:
        cam.set_format(BufferType.VIDEO_CAPTURE, w, h, "MJPG")
        cam.set_fps(BufferType.VIDEO_CAPTURE, args.fps)
        if "exposure_dynamic_framerate" in cam.controls:
            cam.controls.exposure_dynamic_framerate.value = False  # keep 30 fps in low light
        fmt = cam.get_format(BufferType.VIDEO_CAPTURE)
        if (fmt.width, fmt.height) != (w, h):
            raise SystemExit(f"camera refused {w}x{h} MJPG, got {fmt.width}x{fmt.height}")
        out.set_format(BufferType.VIDEO_OUTPUT, w, h, "YUYV")
        out.set_fps(BufferType.VIDEO_OUTPUT, args.fps)
        print(f"{args.input} {w}x{h} MJPG -> {args.output} YUYV | mode={args.mode} variant={args.variant} "
              f"ratio={args.ratio:.3f} smooth={args.smooth}", flush=True)

        stats, frames, dropped = defaultdict(float), 0, 0
        report_at = time.monotonic() + 1
        with VideoCapture(cam, size=2) as stream:
            t = time.perf_counter()
            for frame in stream:
                stage = {}

                def lap(name, sync=True):
                    nonlocal t
                    if sync:
                        torch.cuda.synchronize()
                    now = time.perf_counter()
                    stage[name] = now - t
                    t = now

                lap("capture", sync=False)
                data = torch.frombuffer(bytearray(frame.data), dtype=torch.uint8)
                try:
                    img = decode_jpeg(data, device="cuda")  # nvJPEG; the JPEG upload happens inside
                except RuntimeError as e:  # corrupt MJPEG frames happen on USB hiccups
                    print(f"skip frame: {e}", flush=True)
                    dropped += 1
                    t = time.perf_counter()
                    continue
                src = img[None].half().div_(255)
                lap("decode")

                if matter:
                    fgr, pha = matter(src)
                    lap("infer")
                    if smoother:
                        pha = smoother(pha)
                    back = bg if bg is not None else gpu.blurred_background(src, pha, sigma)
                    res = gpu.composite(fgr, pha, back)
                else:
                    res = src
                yuyv = gpu.to_yuyv(res)
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
                    line = " ".join(f"{k}={v / frames * 1000:.1f}" for k, v in stats.items())
                    print(f"fps={frames:2d} drop={dropped} ms: {line}", flush=True)
                    stats.clear()
                    frames = dropped = 0


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
