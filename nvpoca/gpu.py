"""GPU tensor ops: background fill, composition, alpha smoothing, YUYV packing.

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


class AlphaSmoother:
    """Per-pixel adaptive EMA: small alpha changes (flicker) are damped, large ones
    (real motion) pass through immediately."""

    def __init__(self, min_rate: float = 0.3, motion: float = 0.1):
        self.min_rate, self.motion, self.prev = min_rate, motion, None

    def __call__(self, pha: torch.Tensor) -> torch.Tensor:
        if self.prev is None or self.prev.shape != pha.shape:
            self.prev = pha.clone()  # callers may reuse the pha buffer
            return pha
        delta = pha - self.prev
        rate = (delta.abs() / self.motion).clamp(self.min_rate, 1)
        self.prev = self.prev + rate * delta
        return self.prev


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

    s = AlphaSmoother()
    a = torch.zeros(1, 1, 2, 2, device=dev)
    s(a)
    assert torch.allclose(s(a + 0.01), a + 0.003)  # flicker damped
    assert torch.allclose(s(a + 1), a + 1)  # motion passes through
    print("ok")
