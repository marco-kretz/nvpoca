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
uv run python -m nvpoca                                  # blur, 1080p30, TensorRT, MobileNetV3, ratio 0.25
uv run python -m nvpoca --smooth                         # recommended: steadier edges (see Phase 2)
uv run python -m nvpoca --mode replace --bg ~/bg.jpg
uv run python -m nvpoca --blur 16                         # stronger blur (default 7 = portrait-like)
uv run python -m nvpoca --smooth --record clips/x.mjpeg  # keeps last 10 s raw; written on Ctrl+C
uv run python -m nvpoca --help
```

The first run per (variant, resolution, ratio) builds a TensorRT engine into `models/`
(~20–40 s). `--backend torch` skips TensorRT.

Every second a line with per-stage milliseconds is logged. `capture` is time spent waiting
for the next camera frame (≈ idle), `decode` includes the JPEG upload, `latency` is from the
camera's USB start-of-frame timestamp to the frame being written to the loopback device.

## Comparing configs

A clip can also come from a live session (`--record`), so a problem can be captured the
moment it is visible.

```bash
uv run python -m nvpoca.compare clips/chair.mjpeg --record 10      # record a clip once
uv run python -m nvpoca.compare clips/chair.mjpeg --alpha \
  -c "" -c="--variant resnet50" -c="--smooth"                     # labelled grid video + scores
```

Each `-c` is a set of normal `nvpoca` flags (write it as `-c="..."`). Output is a grid
video next to the clip (`--alpha` shows the matte instead of the composite) and three scores
(x1000, lower is better): `flicker` = alpha change where the camera image did not change,
`lag` = how far smoothing trails the raw alpha where the image did change, `soft` = share of
semi-transparent pixels, which exposes background objects kept as a faint ghost.

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

### Phase 2 (quality, two 10 s clips in front of a dark office chair)

`chair`: hair/shoulder edge in front of the chair. `chair2`: head right next to the headrest.

| Config | chair flicker / lag / soft | chair2 flicker / lag / soft |
|---|---|---|
| MobileNetV3 0.25 (default) | 0.627 / – / 16.3 | 0.568 / – / 15.4 |
| MobileNetV3 0.25 `--smooth` | 0.452 / 0.72 / 16.4 | **0.345** / 0.71 / **15.5** |
| ResNet50 0.25 | 0.506 / – / 18.3 | 0.742 / – / 37.9 |
| ResNet50 0.25 `--smooth` | 0.344 / 0.85 / 18.3 | 0.320 / 0.81 / 38.0 |
| ResNet50 0.4 | 0.442 / – / 14.6 | 1.089 / – / 32.4 |
| ResNet50 0.4 `--smooth` | **0.303** / 0.76 / **14.6** | 0.569 / 0.76 / 32.4 |

ResNet50 gives slightly calmer hair edges on `chair`, but on `chair2` it keeps the headrest
as a pulsing alpha 0.2–0.5 ghost (soft ×2.4) that smoothing cannot remove. MobileNetV3 cuts
the headrest cleanly, so it stays the default.

The table above used the first smoother (rate from the size of the alpha change). It can't
tell a flickering headrest from real motion, since both are large jumps. `--smooth` now
steers the rate by camera-image motion instead: static image → alpha changes are flicker
and get damped (min rate 0.15); moving image → alpha follows at once. Third clip `brille`:
glasses frame and headrest at the right side of the head.

| `--smooth` (MobileNetV3 0.25) | chair | chair2 | brille |
|---|---|---|---|
| alpha-driven (old) | 0.452 / 0.72 | 0.345 / 0.71 | 0.554 / 1.24 |
| image-driven, no noise floor | 0.457 / 0.21 | 0.304 / 0.50 | 0.574 / 0.21 |
| **image-driven, noise 2/255, motion 6/255** | **0.362 / 0.67** | **0.169** / 1.24 | **0.499 / 0.79** |
| image-driven, noise 3/255, motion 10/255 | 0.304 / 1.84 | 0.125 / 2.27 | 0.446 / 2.57 |

(flicker / lag). Live: 30 fps, latency ~12 ms. Ratio 0.4 keeps the thin glasses rim more
consistently (brille flicker 0.50 vs 0.57 unsmoothed) but is neutral or worse on the chair
clips, so 0.25 stays the default.
