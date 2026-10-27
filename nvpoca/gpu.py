"""GPU tensor ops: background fill, composition, alpha smoothing, key light, YUYV packing.

Images are float tensors shaped (1, C, H, W) in [0, 1] unless noted.
"""

import math

import torch
import torch.nn.functional as F
from torchvision.transforms.v2.functional import gaussian_blur


def blurred_background(src: torch.Tensor, pha: torch.Tensor, sigma: float) -> torch.Tensor:
    """Blur only the background: normalized convolution with the person masked out,
    so the silhouette can't bleed into the blur and the area behind the person is
    filled from its surroundings."""
    h, w = src.shape[-2:]
    scale = 4  # blur at quarter resolution; visually identical for large sigma, ~16x cheaper
    weight = (1 - pha).float().clamp(0, 1) ** 4  # drop semi-transparent hair/edges out of the fill
    small = (h // scale, w // scale)
    num = F.interpolate(src.float() * weight, size=small, mode="area")
    den = F.interpolate(weight, size=small, mode="area")
    s = max(sigma / scale, 0.5)
    k = 2 * math.ceil(3 * s) + 1
    num = gaussian_blur(num, [k, k], [s, s])
    den = gaussian_blur(den, [k, k], [s, s])
    bg = num / den.clamp(min=1e-4)
    return F.interpolate(bg, size=(h, w), mode="bilinear", align_corners=False).to(src.dtype)


def composite(fgr: torch.Tensor, pha: torch.Tensor, bg: torch.Tensor) -> torch.Tensor:
    # fgr is RVM's predicted foreground colour, so background spill on edges is already removed
    return torch.lerp(bg, fgr, pha)


def cover(img: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Resize + center-crop (1, 3, H', W') to fill (h, w)."""
    ih, iw = img.shape[-2:]
    s = max(h / ih, w / iw)
    img = F.interpolate(img, size=(math.ceil(ih * s), math.ceil(iw * s)), mode="bilinear", antialias=True)
    y, x = (img.shape[-2] - h) // 2, (img.shape[-1] - w) // 2
    return img[..., y : y + h, x : x + w]


def guided_filter(lr_guide: torch.Tensor, lr_src: torch.Tensor, hr_guide: torch.Tensor, r: int = 2, eps: float = 1e-3) -> torch.Tensor:
    """Fast guided filter: fit src as a local linear function of the guide at low res, apply at high res.
    Gives coarse masks the edges of the full-res image."""
    def box(t):  # count_include_pad=False keeps borders unbiased (zero padding darkens them)
        return F.avg_pool2d(t, 2 * r + 1, stride=1, padding=r, count_include_pad=False)

    mx, my = box(lr_guide), box(lr_src)
    a = (box(lr_guide * lr_src) - mx * my) / (box(lr_guide * lr_guide) - mx * mx + eps)
    b = my - a * mx
    size = hr_guide.shape[-2:]
    return F.interpolate(a, size=size, mode="bilinear") * hr_guide + F.interpolate(b, size=size, mode="bilinear")


def depth_refs(disp: torch.Tensor, pha: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    """(person's far side, scene median) disparity; None if no person is visible."""
    person = pha > 0.9
    if person.sum() < 64:
        return None
    return torch.quantile(disp[person].float(), 0.1), disp.median()


def near_mask(disp: torch.Tensor, pha: torch.Tensor, ref: torch.Tensor, floor: torch.Tensor,
              low: float = 0.35, high: float = 0.4, cap: float = 1.4, reach: int = 30) -> torch.Tensor:
    """Soft mask of what sits at about the person's depth (e.g. the chair they sit on).

    disp: (1, 1, h, w) relative disparity (larger = closer); pha: person alpha at the same size.
    Depth is normalised so the scene median (wall, `floor`) is 0 and the person's far side
    (`ref`) is 1; values ramp in between low..high. Things beyond `cap` are in front of the
    person (desk edge right at the camera), not behind, and are dropped; the person's own
    body is covered by pha anyway. Measured on the recorded clips: chair 0.42-1.07, desk
    <= 0.23. Only pixels within `reach` px of the person count, so other near-ish objects
    elsewhere in the room stay background.
    """
    norm = (disp - floor) / (ref - floor).clamp(min=1e-3)
    near = F.max_pool2d(pha, 2 * reach + 1, stride=1, padding=reach) > 0.5
    keep = ((norm - low) / (high - low)).clamp(0, 1) * ((cap + 0.2 - norm) / 0.2).clamp(0, 1)
    return keep * near


class AlphaSmoother:
    """Per-pixel EMA steered by image motion: where the camera image is static, alpha
    changes are flicker and get damped; where it moves, alpha follows immediately."""

    def __init__(self, min_rate: float = 0.15, noise: float = 2 / 255, motion: float = 6 / 255):
        # noise/motion tuned on recorded clips (README, Phase 2): lower noise lets sensor noise
        # count as motion (no damping), higher makes slow real motion trail behind
        self.min_rate, self.noise, self.motion = min_rate, noise, motion
        self.prev = self.prev_gray = None

    def __call__(self, pha: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        gray = F.avg_pool2d(src.float().mean(1, keepdim=True), 4)  # 4x down averages out sensor noise
        if self.prev is None or self.prev.shape != pha.shape:
            self.prev, self.prev_gray = pha.clone(), gray  # clone: callers may reuse the pha buffer
            return pha
        # widen motion a little so the edges of moving objects count as moving
        move = F.max_pool2d((gray - self.prev_gray).abs(), 5, stride=1, padding=2)
        rate = ((move - self.noise) / self.motion).clamp(self.min_rate, 1)
        rate = F.interpolate(rate, size=pha.shape[-2:], mode="bilinear")
        self.prev = self.prev + rate.to(pha.dtype) * (pha - self.prev)
        self.prev_gray = gray
        return self.prev


def luma(rgb: torch.Tensor) -> torch.Tensor:
    r, g, b = rgb.unbind(1)
    return (0.299 * r + 0.587 * g + 0.114 * b).unsqueeze(1)


class KeyLight:
    """Virtual frontal key light: brightens the person when they are underexposed or darker than
    a bright background (backlight), strongest on the head and falling off softly like a large
    softbox. The gain is measured each frame and eased in, so it doesn't pump with motion."""

    def __init__(self, strength: float = 0.5, target: float = 0.42, backlit: float = 0.6, rate: float = 0.05,
                 neutral: float = 0.5):
        # untuned starting points: target = face luma below which it lights, backlit = least face
        # luma as a share of the background's, rate = per-frame ease (~0.7 s at 30 fps),
        # neutral = how white the added light is (0 = room light colour, 1 = grey)
        self.strength, self.target, self.backlit, self.rate, self.neutral = strength, target, backlit, rate, neutral
        self.gain = None

    def __call__(self, src: torch.Tensor, fgr: torch.Tensor, pha: torch.Tensor) -> torch.Tensor:
        """Lit fgr. Measured on src, as RVM's fgr is undefined on the background."""
        h, w = fgr.shape[-2:]
        small = (h // 8, w // 8)
        y = luma(F.interpolate(src.float(), size=small, mode="area"))
        a = F.interpolate(pha.float(), size=small, mode="area")
        light = self.falloff(a)
        person = (a > 0.9).float()
        face = (y * person * light).sum() / (person * light).sum().clamp(min=1e-3)
        back = (a < 0.1).float()
        bg = (y * back).sum() / back.sum().clamp(min=1)
        gain = ((self.backlit * bg).clamp(min=self.target) / face.clamp(min=0.02)).clamp(1, 1 + 2 * self.strength)
        gain = torch.where(person.sum() > 64, gain, 1.0)  # stays on the GPU: an `if` would sync every frame
        self.gain = gain if self.gain is None else self.gain + self.rate * (gain - self.gain)
        k = 1 + (self.gain - 1) * F.interpolate(light, size=(h, w), mode="bilinear")
        f = fgr.float()
        y = luma(f).clamp(0, 1)
        # ~k*y in the shadows, rolls off to 1 so highlights don't clip
        lifted = 1 - (1 - y) ** k
        # added light takes the pixel's colour pulled towards white: scaling the colour alone would
        # amplify a warm room light into orange skin
        tint = torch.lerp(f / y.clamp(min=1e-3), torch.ones_like(f), self.neutral)
        return (f + (lifted - y) * tint).clamp(0, 1).to(fgr.dtype)

    @staticmethod
    def falloff(a: torch.Tensor) -> torch.Tensor:
        """Gaussian centred just below the top of the person, at the middle of their upper part."""
        h, w = a.shape[-2:]
        body = (a > 0.5).float()
        rows = torch.arange(h, device=a.device, dtype=torch.float).view(-1, 1)
        cols = torch.arange(w, device=a.device, dtype=torch.float)
        top = body.amax(-1).flatten().argmax()  # first row with person (argmax returns the first max)
        upper = body * ((rows >= top) & (rows < top + 0.3 * h))
        cx = (upper.sum(-2).flatten() * cols).sum() / upper.sum().clamp(min=1)
        s = 0.6 * h
        gy = torch.exp(-(((rows - top - 0.2 * h) / s) ** 2) / 2)
        gx = torch.exp(-(((cols - cx) / s) ** 2) / 2)
        return (gy * gx)[None, None]


def to_yuyv(rgb: torch.Tensor) -> torch.Tensor:
    """(1, 3, H, W) RGB in [0, 1] -> (H, W * 2) uint8 YUYV, BT.601 limited range
    (what browsers and Zoom assume for YUYV webcams)."""
    r, g, b = rgb[0].float()
    y = 0.299 * r + 0.587 * g + 0.114 * b
    # chroma from the horizontal pixel-pair average (4:2:2)
    rp, bp, yp = (t.unflatten(-1, (-1, 2)).mean(-1) for t in (r, b, y))
    u = 128 + 224 * 0.5 * (bp - yp) / (1 - 0.114)
    v = 128 + 224 * 0.5 * (rp - yp) / (1 - 0.299)
    y = 16 + 219 * y
    y0, y1 = y.unflatten(-1, (-1, 2)).unbind(-1)
    out = torch.stack((y0, u, y1, v), dim=-1).round_().clamp_(0, 255).to(torch.uint8)
    return out.flatten(-2)


if __name__ == "__main__":
    dev = "cuda"
    # YUYV: white, black, pure red match the BT.601 limited-range reference values
    px = torch.tensor([[1, 1, 1], [1, 1, 1], [0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0]], device=dev, dtype=torch.float)
    yuyv = to_yuyv(px.T.reshape(1, 3, 1, 6))
    assert yuyv[0].tolist() == [235, 128, 235, 128, 16, 128, 16, 128, 81, 90, 81, 240], yuyv

    # blurred background must not contain foreground colour: red person on green background
    src = torch.zeros(1, 3, 64, 64, device=dev)
    src[:, 1] = 1
    pha = torch.zeros(1, 1, 64, 64, device=dev)
    pha[..., 16:48, 16:48] = 1
    src[..., 16:48, 16:48] = torch.tensor([1.0, 0, 0], device=dev)[:, None, None]
    bg = blurred_background(src, pha, sigma=8)
    assert bg[:, 0].max() < 0.01 and bg[:, 1].min() > 0.99, (bg[:, 0].max(), bg[:, 1].min())

    # near_mask: chair next to the person is kept, a desk at mid depth and the far wall are not
    disp = torch.zeros(1, 1, 40, 80, device=dev)
    pha = torch.zeros_like(disp)
    disp[..., 10:, 30:50], pha[..., 10:, 30:50] = 1.0, 1  # person
    disp[..., 5:, 20:30] = 0.7  # chair, left of the person
    disp[..., 25:, 50:60] = 0.25  # desk, right of the person
    disp[..., 25:, 70:] = 0.9  # something close but far from the person
    k = near_mask(disp, pha, *depth_refs(disp, pha), reach=12)
    assert k[0, 0, 20, 25] == 1 and k[0, 0, 30, 55] == 0 and k[0, 0, 2, 5] == 0 and k[0, 0, 30, 75] == 0, k
    # guided filter keeps a coarse mask's edge where the full-res image has it
    hr = torch.zeros(1, 1, 64, 64, device=dev)
    hr[..., 33:, :] = 1
    lr = F.interpolate(hr, size=(16, 16), mode="area")
    out = guided_filter(lr, lr, hr)
    assert out[0, 0, 30, 5] < 0.1 and out[0, 0, 36, 5] > 0.9 and out[0, 0, 63, 5] > 0.99, out[0, 0, :, 5]

    # RVM's TensorRT-friendly prefilter + bilinear matches torch's antialiased resize (away from the border)
    from nvpoca.rvm.model import _prefilter
    x = torch.rand(1, 3, 64, 96, device=dev)
    ours = F.interpolate(_prefilter(x, 0.25), scale_factor=0.25, mode="bilinear")
    ref = F.interpolate(x, scale_factor=0.25, mode="bilinear", antialias=True)
    assert (ours - ref)[..., 1:-1, 1:-1].abs().max() < 1e-4, (ours - ref).abs().max()

    s = AlphaSmoother()
    a = torch.zeros(1, 1, 16, 16, device=dev)
    img = torch.zeros(1, 3, 16, 16, device=dev)
    s(a, img)
    assert torch.allclose(s(a + 1, img), a + 0.15)  # static image: alpha jump is flicker, damped
    assert torch.allclose(s(a + 1, img + 0.5), a + 1)  # image moved: alpha follows at once

    # key light: dark person is lit with highlights kept, a well-lit one is left alone, backlight is lifted
    pha = torch.zeros(1, 1, 256, 256, device=dev)
    pha[..., 64:, 96:160] = 1
    src = torch.full((1, 3, 256, 256), 0.15, device=dev)
    src[..., 240:, 96:110] = 1  # white shirt
    out = KeyLight(strength=1)(src, src, pha)
    assert 0.3 < out[0, 0, 100, 128] < 0.5 and out[..., 240:, 96:110].min() > 0.99, out[0, 0, 100, 128]
    src = torch.full_like(src, 0.5)
    assert torch.allclose(KeyLight()(src, src, pha), src, atol=1e-3)
    assert torch.allclose(KeyLight()(src * 0.3, src * 0.3, pha * 0), src * 0.3)  # no person: unchanged
    src = torch.where(pha > 0, 0.45, 0.95).expand(-1, 3, -1, -1)  # bright enough, but in front of a window
    assert KeyLight()(src, src, pha)[0, 0, 100, 128] > 0.5
    src = torch.where(pha > 0, torch.tensor([0.2, 0.1, 0.05], device=dev).view(1, 3, 1, 1), 0.5)  # warm-lit skin
    r, _, b = KeyLight(strength=1)(src, src, pha)[0, :, 100, 128]
    assert r > 0.2 and r / b < 3, (r, b)  # brighter and less orange (input r/b = 4)
    print("ok")
