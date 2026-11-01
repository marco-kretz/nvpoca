"""One-time export of RVM to the static-shape ONNX the C++ daemon builds its TensorRT engine from.

Only needs CPU torch (no CUDA, no TensorRT), e.g. in a throwaway venv:
    python -m venv /tmp/rvm && /tmp/rvm/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    /tmp/rvm/bin/pip install onnx onnxscript
    /tmp/rvm/bin/python tools/export_onnx.py --size 1920x1080
"""

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from nvpoca.rvm.model import MattingNetwork  # noqa: E402


class Fixed(nn.Module):
    # TensorRT wants static shapes, so downsample_ratio is baked in and rec states are plain inputs
    def __init__(self, model: MattingNetwork, ratio: float):
        super().__init__()
        self.model, self.ratio = model, ratio

    def forward(self, src, r1, r2, r3, r4):
        return tuple(self.model(src, r1, r2, r3, r4, downsample_ratio=self.ratio))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["mobilenetv3", "resnet50"], default="mobilenetv3")
    p.add_argument("--size", default="1920x1080")
    p.add_argument("--ratio", type=float, help="default: 480px on the long side, as the daemon")
    p.add_argument("--models", type=Path, default=ROOT / "models")
    args = p.parse_args()
    w, h = map(int, args.size.split("x"))
    ratio = args.ratio or min(1.0, 480 / max(w, h))

    model = MattingNetwork(args.variant).eval()
    model.load_state_dict(torch.load(args.models / f"rvm_{args.variant}.pth", map_location="cpu"))
    with torch.inference_mode():  # rec shapes from one float pass; CPU half kernels are not needed
        _, _, *rec = model(torch.zeros(1, 3, h, w), downsample_ratio=ratio)
    model = Fixed(model.half(), ratio)
    src = torch.zeros(1, 3, h, w, dtype=torch.half)
    out = args.models / f"rvm_{args.variant}_{w}x{h}_r{ratio:.3f}_aa.onnx"
    torch.onnx.export(
        model, (src, *[torch.zeros_like(r, dtype=torch.half) for r in rec]), out,
        input_names=["src", "r1i", "r2i", "r3i", "r4i"],
        output_names=["fgr", "pha", "r1o", "r2o", "r3o", "r4o"],
        dynamo=True, external_data=False,
    )
    print(out)


if __name__ == "__main__":
    main()
