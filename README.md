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

# SAM2Matting Tiny for --quality (CC BY-NC 4.0: non-commercial only, so its code is fetched into
# nvpoca/sam2m from FudanCVL/SAM2Matting 73dd721 rather than included). Optionally build its
# TensorRT engines up front, else the first switch to quality freezes the camera for about a minute:
contrib/fetch-sam2matting.sh
curl -L -o models/sam2matting_tiny.pt \
  https://huggingface.co/FudanCVL/SAM2Matting/resolve/main/checkpoints/SAM2Matting-SAM2.1Tiny.pt
uv run python -m nvpoca.sam clips/chair.mjpeg   # any recorded clip; prints timing and TRT-vs-torch error

# Depth Anything V2 Small for --keep-chair (Apache-2.0, code vendored in nvpoca/depth)
curl -L -o models/depth_anything_v2_vits.pth \
  https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth
```

## Usage

`python -m nvpoca` is a small supervisor (no torch/CUDA, ~20 MB RAM): it keeps `/dev/video10`
ready and, when an app (browser, Zoom, the preview) starts reading it, launches the GPU worker
(`nvpoca.worker`), which loads the models and opens the real camera. `--unload-after` seconds
(default 60) after the last reader the worker exits, which frees all of its VRAM; a CUDA
context only goes away with its process. Cold start to the first camera frame ~1.6 s (the app
sees black meanwhile), warm ~0.3 s. Settings live in `~/.config/nvpoca/settings.json` and can
be changed at runtime over D-Bus.

```bash
# run as a systemd user service (unit file in contrib/, linked from the repo)
systemctl --user link ~/Development/nvpoca/contrib/nvpoca.service
systemctl --user enable --now nvpoca.service
journalctl --user -u nvpoca -f                     # per-second timing while a reader is attached

# or in the foreground; flags given here override the saved settings
uv run python -m nvpoca --blur 10
uv run python -m nvpoca --record clips/x.mjpeg     # keeps last 10 s raw; written on Ctrl+C / stop
uv run python -m nvpoca --keep-chair               # experimental: keep the chair (depth model)
```

The first start per (variant, resolution, ratio) builds a TensorRT engine into `models/`
(~20–40 s). `--backend torch` skips TensorRT.

### D-Bus (`org.nvpoca.Daemon`, session bus)

```bash
B="busctl --user"; O="org.nvpoca.Daemon /org/nvpoca/Daemon org.nvpoca.Daemon"
$B get-property $O Active                          # is the camera in use?
$B set-property $O Enabled b false                 # passthrough
$B set-property $O Blur d 6                        # 0.5..40, sigma at 1080p
$B set-property $O Mode s green                    # blur | replace | green (#00FF00 for chroma keyers)
$B set-property $O Background s ~/Pictures/bg.jpg  # needed for replace
$B set-property $O Smooth d 0                      # 0 = off, else min rate (0.15 default)
$B set-property $O Quality b true                  # SAM2Matting on top of RVM (Phase 5)
```

Read-only: `Active` (camera open), `Loaded` (GPU worker running), `Fps`, `Latency` (ms).
Changes emit `PropertiesChanged`.

### Omarchy shell plugin

`omarchy-plugin/` is a bar widget: camera icon (highlighted while an app uses the camera),
effect on/off, blur slider, steady-edges and high-quality toggles, blur/green/image switch (image once a background is set),
and a Preview button that opens `/dev/video10` in mpv (only while no other app streams, since
v4l2loopback serves one reader at a time).

```bash
ln -sfn ~/Development/nvpoca/omarchy-plugin ~/.config/omarchy/plugins/local.nvpoca
omarchy-shell shell rescanPlugins
omarchy bar put local.nvpoca --before local.proton
```

Bar icon: crossed out = daemon not running, outline = idle (camera off), filled + accent =
camera in use. The shell watches plugins with `inotifywait -r`, which does not follow the
symlink, so after editing the plugin run `omarchy restart shell`.

Every second while a reader is attached the journal gets a line with per-stage milliseconds.
`capture` is time spent waiting for the next camera frame (≈ idle), `decode` includes the JPEG
upload, `latency` is from the camera's USB start-of-frame timestamp to the frame being written
to the loopback device.

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

### Keeping the chair (`--keep-chair`)

RVM only knows people, so the chair behind you is background and its edges flicker where
your head and shoulders meet it. `--keep-chair` adds Depth Anything V2 Small (518×294,
torch FP16, +4 ms; live latency ~17 ms) and keeps whatever sits at about your depth near
your silhouette: depth is normalised between the wall (scene median, 0) and your far side
(1); the chair measured 0.42–1.07, the desk ≤ 0.23, so 0.35–0.40 is the ramp. Things far in
front of you (>1.4) and further than ~110 px (1080p) from you are dropped. The coarse mask
gets its edges from the camera image with a guided filter; the chair uses camera pixels.

| `--smooth` vs `--smooth --keep-chair` (flicker / lag / soft) | chair | chair2 | brille |
|---|---|---|---|
| without | 0.362 / 0.67 / 16.8 | 0.169 / 1.24 / 15.9 | **0.499** / 0.79 / 17.3 |
| with | **0.302 / 0.51 / 13.0** | **0.090 / 0.68 / 13.5** | 0.727 / 0.76 / 18.2 |

Known limit (`brille`): the chair is mostly hidden there, and the black desk cabinet next to
the right shoulder sits at chair depth, so it gets kept on and off. Depth alone cannot tell
them apart. Tried and dropped: lower ramp 0.25–0.40 (gray halo over the plain wall, soft
+50%), EMA on the reference depths (no measurable effect).

### Phase 2 result

Tried replacing RVM with BiRefNet (2024, per-frame segmentation, evaluated via
`transformers`, not added to the project). Flicker with the same smoothing, 1024×576 input:

| Model | chair | chair2 | brille | infer ms (torch FP16) |
|---|---|---|---|---|
| RVM MobileNetV3 (kept) | **0.362** | **0.169** | 0.499 | ~1 (TRT) |
| BiRefNet_lite-matting | 0.398 | 0.638 | **0.387** | 19 |
| BiRefNet-matting | 0.668 | 0.185 | 0.723 | 44 |

Without temporal memory BiRefNet toggles the headrest in and out (raw flicker on `chair2`
3.3 vs RVM 0.57). RVM stays. What helped most in practice: image-motion smoothing and a
subtle blur (σ 4), where edge errors barely show. Front light instead of the window
backlight would help any model with dark hair against the dark chair.

Defaults now: MobileNetV3, ratio 0.25, `--smooth 0.15`, `--blur 4`.

Load of the default config live (`--smooth --blur 4`, 1080p30): ~3.5 ms GPU work per 33 ms
frame, whole-GPU utilisation ~6 % including the desktop, 628 MiB VRAM, ~11 % of one CPU
core, 1.8 GB RAM (mostly PyTorch/CUDA/TensorRT libraries), NVENC/NVDEC idle (nvJPEG runs on
the SMs), latency ~12 ms.

## Phase 3: daemon (Python)

Kept Python instead of a Rust rewrite: the GPU work is ~3.5 ms per frame either way; Rust
would mainly save RAM and start-up time. Reader detection uses v4l2loopback's
`V4L2_EVENT_PRI_CLIENT_USAGE` (0x10E00001), which fires on STREAMON/STREAMOFF and also when a
reader is killed. Idle: supervisor only, ~20 MB RAM, no VRAM. While loaded: the worker adds
~630 MB VRAM; it inherits the supervisor's loopback fd and talks JSON lines over stdin/stdout.
A crashed worker is restarted while an app is still reading.

## Phase 4: antialiased model input

RVM shrinks the 1080p frame to 480×270 with plain bilinear, which at ratio 0.25 reads only
2×2 of every 4×4 pixels: sensor noise and MJPEG blocks alias into the model input and show
up as edge flicker. A triangle low-pass first (what `F.interpolate(antialias=True)` does;
TensorRT has no antialiased Resize, so it runs as two small convolutions in the engine)
fixes that. Engines are cached as `*_aa.engine`; old ones can be deleted.

| MobileNetV3 0.25 `--smooth` (flicker / lag / soft) | chair | chair2 | brille |
|---|---|---|---|
| bilinear (before) | 0.361 / 0.67 / 16.7 | 0.167 / 1.24 / 15.9 | 0.498 / 0.79 / 17.3 |
| **antialiased** | **0.329 / 0.63** / 16.6 | **0.134 / 1.21** / 16.9 | **0.447 / 0.75** / 19.2 |
| ResNet50, antialiased | 0.283 / 0.43 / 21.3 | 0.144 / 0.96 / 26.3 | 0.427 / 0.62 / 22.8 |

Raw (unsmoothed) flicker drops 10–15 %. The higher `soft` is the headset: ear cups and mic
are kept more completely instead of cut away. Ratios 0.333/0.4 stay worse than 0.25 with
antialiasing too. ResNet50 now loses most of its headrest ghost (chair2 soft 38 → 26) and
scores lower flicker, but its hair edges are visibly softer (grey halo with replace/green),
so MobileNetV3 stays the default. Cost: +0.5 ms GPU per frame (TRT 0.93 → 1.47 ms).

## Phase 5: high-quality mode (`--quality`)

With the green screen, RVM's limits are plain to see: a dark chair behind dark clothes is kept
as part of the person, a headset cup turns half transparent, hair leaves a ghost. These are
model errors that no post-processing removes. NVIDIA Broadcast's model (Maxine VFX SDK) is only
free on Windows; the Linux SDK sits behind an NVIDIA AI Enterprise subscription.

`--quality` adds [SAM2Matting](https://github.com/FudanCVL/SAM2Matting) (ECCV 2026, SAM2.1-Tiny
tracker + matting heads) on top of RVM. The tracker follows the person as one object, so headset,
mic and hair stay solid, but it follows whatever it is seeded with: seeded with RVM's matte, it
faithfully tracked person + chair (clip `jacke`: dark jacket in front of the dark chair, RVM keeps
the whole backrest). So the seed comes from torchvision's Mask R-CNN (COCO separates "person" from
"chair"; ~55 ms, run every 0.5 s only while nobody is tracked; weights download once to
`~/.cache/torch`). Its masks stop short of the frame edge, so the seed's bottom rows are extended
down. With nobody detected the matte is empty (RVM shows an empty chair as a person). RVM still
runs every frame: its foreground colour is used where it sees the person, and the tracker is
re-seeded when the two disagree (IoU < 0.3; low, since RVM often includes the chair) for 1 s.

Green and image backgrounds use hard edges (alpha levels 0.4–0.6): any half-transparent edge
pixel shows as a smear or halo against a flat colour, for RVM as well. Blur keeps the soft matte. Image encoder (bf16) and
matting heads (fp16; TensorRT has no bf16 transposed conv) are TensorRT engines; memory
attention and the SAM heads run in torch, as their shapes change while the memory fills.

| live, 1080p30 | RVM | `--quality` |
|---|---|---|
| infer ms | 1.6 | ~20 |
| latency ms | ~13 | ~32 |
| worker VRAM | ~0.63 GB | ~2.6 GB |

Offline on the clips (with `--smooth`, from frame 15 so the seeding switch doesn't count),
flicker / soft:

| | chair | chair2 | brille |
|---|---|---|---|
| RVM | 0.299 / 16.3 | 0.128 / 16.9 | 0.428 / 18.7 |
| `--quality` | 0.402 / 11.3 | 0.180 / 11.1 | 0.497 / 10.6 |

(RVM-seeded; with the Mask R-CNN seed and hard edges, `jacke` scores RVM 0.558 / 34.3 vs
`--quality` 0.400 / 12.1, and an empty-chair clip 0.074 / 14.6 vs 0 / 0.)

The tracker's matte is 512² (squashed from 16:9) and its edges jitter a bit more than RVM's,
but it is far more often right about *what* belongs to the person, which is what shows. Known
limit: on `brille` it keeps a wedge of the chair next to the chin, which Mask R-CNN's mask
already contains there. Tried and
dropped: SAM2.1-B+ (~42 ms, more flicker), 7 memory frames instead of 4 (+4 ms, no better),
guided filter against the full-res image (≤ 2 % less flicker, more soft edges), torch only
(26.6 ms vs 18.2 ms with the engines; mean |Δalpha| 7.6e-4).

## License

GPL-3.0 (`LICENSE`), as required by the vendored RobustVideoMatting code in `nvpoca/rvm`.
`nvpoca/depth` is Depth Anything V2 (Apache-2.0). SAM2Matting (`--quality`) is CC BY-NC 4.0 and
not part of this repository; `contrib/fetch-sam2matting.sh` downloads it. Model weights are
downloaded separately under their own licenses.
