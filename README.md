# nvpoca

nVIDIA POstprocessing CAmera: real-time background blur/replacement for a webcam on
Linux, exposed as a v4l2loopback virtual camera. NVIDIA-only (CUDA, nvJPEG, TensorRT).

Pipeline: MJPEG capture → nvJPEG decode → [RVM](https://github.com/PeterL1n/RobustVideoMatting)
FP16 (TensorRT) with recurrent state and Deep Guided Filter refiner → composite with RVM's
decontaminated foreground → BT.601 YUYV on GPU → one download → `/dev/video10`.

## Setup (Arch / Omarchy)

Requirements: NVIDIA driver with CUDA 13 support, `uv`, kernel headers for the running kernel.

```bash
# virtual camera
sudo pacman -S v4l2loopback-dkms v4l2loopback-utils
echo 'options v4l2loopback devices=1 video_nr=10 card_label="nvpoca Cam" exclusive_caps=1 max_buffers=2' \
  | sudo tee /etc/modprobe.d/v4l2loopback.conf
echo v4l2loopback | sudo tee /etc/modules-load.d/v4l2loopback.conf
sudo modprobe v4l2loopback

# python env (torch cu130, TensorRT 11 wheels; no system CUDA/cuDNN/TensorRT needed)
uv sync

# RVM weights (GPL-3.0, same as the vendored model code in nvpoca/rvm)
mkdir -p models
for f in rvm_mobilenetv3.pth rvm_resnet50.pth; do
  curl -L -o models/$f https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/$f
done
```

## Usage

```bash
uv run python -m nvpoca                                  # blur, 1080p30, TensorRT, ResNet50, ratio 0.4
uv run python -m nvpoca --smooth                         # recommended: steadier edges (see Phase 2)
uv run python -m nvpoca --mode replace --bg ~/bg.jpg
uv run python -m nvpoca --variant mobilenetv3 --blur 24
uv run python -m nvpoca --help
```

The first run per (variant, resolution, ratio) builds a TensorRT engine into `models/`
(~20–40 s). `--backend torch` skips TensorRT.

Every second a line with per-stage milliseconds is logged. `capture` is time spent waiting
for the next camera frame (≈ idle), `decode` includes the JPEG upload, `latency` is from the
camera's USB start-of-frame timestamp to the frame being written to the loopback device.

## Comparing configs

```bash
uv run python -m nvpoca.compare clips/chair.mjpeg --record 10      # record a clip once
uv run python -m nvpoca.compare clips/chair.mjpeg --alpha \
  -c "" -c="--variant mobilenetv3" -c="--smooth"                  # labelled grid video + scores
```

Each `-c` is a set of normal `nvpoca` flags (write it as `-c="..."`). Output is a grid
video next to the clip (`--alpha` shows the matte instead of the composite) and two scores
(x1000, lower is better): `flicker` = alpha change where the camera image did not change,
`lag` = how far smoothing trails the raw alpha where the image did change.

## Measurements

### Phase 1 (RTX 5070 Ti, Logitech C920 MJPEG, 30 fps, ratio 0.25 @1080p / 0.375 @720p)

Steady-state means, ms:

| Config | decode | infer | compose | download | write | latency |
|---|---|---|---|---|---|---|
| 1080p passthrough | 1.2 | – | 0.3 | 0.1 | 0.2 | 10.4 |
| 1080p blur, MobileNetV3, TRT | 1.2 | 0.9 | 1.1 | 0.1 | 0.2 | 12.3 |
| 1080p blur, MobileNetV3, TRT, smooth | 1.3 | 0.9 | 1.2 | 0.1 | 0.2 | 12.6 |
| 1080p replace, MobileNetV3, TRT | 1.2 | 0.9 | 0.3 | 0.1 | 0.1 | 11.4 |
| 1080p blur, MobileNetV3, torch | 1.2 | 3.0 | 1.1 | 0.1 | 0.2 | 13.9 |
| 1080p blur, ResNet50, TRT | 1.2 | 1.1 | 1.1 | 0.1 | 0.1 | 12.3 |
| 1080p blur, ResNet50, torch | 1.2 | 3.0 | 1.1 | 0.1 | 0.2 | 14.2 |
| 720p blur, MobileNetV3, TRT | 0.7 | 0.8 | 0.5 | 0.1 | 0.1 | 9.1 |
| 720p blur, ResNet50, TRT | 0.7 | 1.0 | 0.5 | 0.1 | 0.1 | 9.1 |

All configs hold 30 fps with no drops. Processing adds ~2–4 ms; the rest of `latency` is the
USB transfer of the MJPEG frame (see passthrough). TensorRT vs torch on 60 real frames:
mean |Δalpha| ≈ 5e-5.

### Phase 2 (quality, clip: sitting in front of a dark office chair, 10 s)

| Config | flicker | lag | infer ms (live) |
|---|---|---|---|
| MobileNetV3, ratio 0.25 (Phase 1 default) | 0.627 | – | 0.9 |
| ResNet50, ratio 0.25 | 0.506 | – | 1.1 |
| ResNet50, ratio 0.33 | 0.462 | – | |
| ResNet50, ratio 0.4 (new default) | 0.442 | – | |
| ResNet50, ratio 0.25, smooth 0.3 | 0.344 | 0.85 | |
| ResNet50, ratio 0.4, smooth 0.3 | 0.303 | 0.76 | 2.3 |
| ResNet50, ratio 0.4, smooth 0.5 | 0.327 | 0.65 | |
| ResNet50, ratio 0.4, smooth 0.15 | 0.295 | 0.83 | |

ResNet50 at 0.4 + `--smooth`: 1080p30, end-to-end latency ~15 ms. Below `--smooth 0.3`
flicker barely improves while lag keeps growing. Remaining flicker sits on dark-on-dark
edges (hair against the dark acoustic panel, shoulder against the dark shelf).
