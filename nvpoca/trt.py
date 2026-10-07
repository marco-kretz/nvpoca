"""RVM as a TensorRT FP16 engine, built once per (variant, resolution, ratio) and cached."""

from pathlib import Path

import tensorrt as trt
import torch
from torch import nn

from nvpoca.rvm.model import MattingNetwork

LOGGER = trt.Logger(trt.Logger.WARNING)


class _Fixed(nn.Module):
    # TensorRT wants static shapes, so downsample_ratio is baked in and rec states are plain inputs
    def __init__(self, model: MattingNetwork, ratio: float):
        super().__init__()
        self.model, self.ratio = model, ratio

    def forward(self, src, r1, r2, r3, r4):
        return tuple(self.model(src, r1, r2, r3, r4, downsample_ratio=self.ratio))


def _build(weights: Path, variant: str, ratio: float, h: int, w: int, engine: Path) -> None:
    model = MattingNetwork(variant).eval().cuda()
    model.load_state_dict(torch.load(weights, map_location="cuda"))
    model = _Fixed(model.half(), ratio)
    src = torch.zeros(1, 3, h, w, device="cuda", dtype=torch.half)
    with torch.inference_mode():
        _, _, *rec = model.model(src, downsample_ratio=ratio)  # discover rec shapes
    onnx = engine.with_suffix(".onnx")
    torch.onnx.export(
        model, (src, *[torch.zeros_like(r) for r in rec]), onnx,
        input_names=["src", "r1i", "r2i", "r3i", "r4i"],
        output_names=["fgr", "pha", "r1o", "r2o", "r3o", "r4o"],
        dynamo=True, external_data=False,
    )
    builder = trt.Builder(LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, LOGGER)
    if not parser.parse_from_file(str(onnx)):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT engine build failed")
    engine.write_bytes(plan)


class TrtMatter:
    """Same interface as the torch Matter: call with (1, 3, H, W) half src, get (fgr, pha)."""

    def __init__(self, models: Path, variant: str, ratio: float, h: int, w: int):
        engine_path = models / f"rvm_{variant}_{w}x{h}_r{ratio:.3f}.engine"
        if not engine_path.exists():
            print(f"building {engine_path.name} (one-time, takes a minute)...", flush=True)
            _build(models / f"rvm_{variant}.pth", variant, ratio, h, w, engine_path)
        self.engine = trt.Runtime(LOGGER).deserialize_cuda_engine(engine_path.read_bytes())
        self.ctx = self.engine.create_execution_context()
        self.stream = torch.cuda.current_stream()

        def alloc(name):
            return torch.zeros(tuple(self.engine.get_tensor_shape(name)), device="cuda", dtype=torch.half)

        self.src = alloc("src")
        self.fgr, self.pha = alloc("fgr"), alloc("pha")
        # ping-pong the recurrent state: this frame's outputs are the next frame's inputs
        self.rec_in = [alloc(f"r{i}i") for i in range(1, 5)]
        self.rec_out = [alloc(f"r{i}o") for i in range(1, 5)]
        for name, t in (("src", self.src), ("fgr", self.fgr), ("pha", self.pha)):
            self.ctx.set_tensor_address(name, t.data_ptr())

    def reset(self) -> None:
        for r in self.rec_in:
            r.zero_()

    def __call__(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.src.copy_(src)
        for i in range(4):
            self.ctx.set_tensor_address(f"r{i + 1}i", self.rec_in[i].data_ptr())
            self.ctx.set_tensor_address(f"r{i + 1}o", self.rec_out[i].data_ptr())
        if not self.ctx.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        self.rec_in, self.rec_out = self.rec_out, self.rec_in
        return self.fgr, self.pha
