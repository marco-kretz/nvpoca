import time
from collections import defaultdict, deque

import torch
from linuxpy.video.device import BufferType, Device, VideoCapture

from nvpoca import gpu
from nvpoca.core import Effect, configure_camera, decode, finish_args, parser


def main():
    p = parser()
    args = finish_args(p, p.parse_args())
    w, h = args.width, args.height
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_stream(torch.cuda.Stream())  # TensorRT syncs extra on the legacy default stream

    effect = Effect(args)
    out_buf = torch.empty(h, w * 2, dtype=torch.uint8, pin_memory=True)
    recent = deque(maxlen=10 * args.fps) if args.record else None
    try:
        run(args, effect, out_buf, recent)
    except KeyboardInterrupt:
        pass
    finally:
        if recent:
            args.record.parent.mkdir(parents=True, exist_ok=True)
            args.record.write_bytes(b"".join(recent))  # same format as compare --record
            print(f"wrote last {len(recent)} frames to {args.record}", flush=True)


def run(args, effect, out_buf, recent):
    w, h = args.width, args.height

    with Device(args.input) as cam, Device(args.output) as out:
        configure_camera(cam, w, h, args.fps)
        out.set_format(BufferType.VIDEO_OUTPUT, w, h, "YUYV")
        out.set_fps(BufferType.VIDEO_OUTPUT, args.fps)
        print(f"{args.input} {w}x{h} MJPG -> {args.output} YUYV | mode={args.mode} backend={args.backend} variant={args.variant} "
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
                raw = frame.data
                if recent is not None:
                    recent.append(raw)
                data = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
                try:
                    src = decode(data)  # the JPEG upload happens inside
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
                    line = " ".join(f"{k}={v / frames * 1000:.1f}" for k, v in stats.items())
                    print(f"fps={frames:2d} drop={dropped} ms: {line}", flush=True)
                    stats.clear()
                    frames = dropped = 0


if __name__ == "__main__":
    main()
