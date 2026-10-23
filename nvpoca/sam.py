"""High-quality matting: SAM2Matting (ECCV 2026), a SAM2 video tracker with matting heads.

The tracker is seeded with a person mask from torchvision's Mask R-CNN (COCO knows "person" and
"chair" apart; RVM's matte as seed made the tracker follow person + chair). From then on it follows
the person as one object, so a headset, a mic and hair stay solid and the chair stays out. Until the
detector finds someone the matte is empty. RVM keeps running every frame: its foreground colour is
used where it sees the person, and when it disagrees with the tracker for a while the tracker is
re-seeded. Image encoder (bf16) and matting heads (fp16) run as TensorRT engines; the memory
attention and the small SAM heads stay in torch (shapes change while the memory fills).

    python -m nvpoca.sam CLIP    # build the engines, compare against torch, time it on a clip
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2

from nvpoca.sam2m.backbones.hieradet import Hiera
from nvpoca.sam2m.backbones.image_encoder import FpnNeck, ImageEncoder
from nvpoca.sam2m.memory_attention import MemoryAttention, MemoryAttentionLayer
from nvpoca.sam2m.memory_encoder import CXBlock, Fuser, MaskDownSampler, MemoryEncoder
from nvpoca.sam2m.position_encoding import PositionEmbeddingSine
from nvpoca.sam2m.sam.transformer import RoPEAttention
from nvpoca.sam2m.sam2matting_base import SAM2MattingBase

WEIGHTS = "sam2matting_tiny.pt"
SIZE = 1024  # the tracker's square input; the 16:9 frame is squashed, as upstream does
KEEP = 3  # previous frames in memory besides the seed; upstream's 6 scored the same on the clips, 4 ms slower
DETECT_EVERY = 15  # frames between person-detector runs (~55 ms each) while nobody is tracked
RESEED_AFTER = 30  # frames of tracker/RVM disagreement before re-seeding
RESEED_IOU = 0.3  # low: RVM often includes the chair the tracker rightly leaves out
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def build() -> SAM2MattingBase:
    """SAM2Matting-SAM2.1Tiny: upstream configs/sam2matting-sam2.1tiny.yaml plus the video predictor's
    eval overrides (build_sam2matting_video_predictor), without hydra."""
    def rope(**kw):
        return RoPEAttention(rope_theta=10000.0, feat_sizes=[64, 64], embedding_dim=256, num_heads=1,
                             downsample_rate=1, dropout=0.1, **kw)

    return SAM2MattingBase(
        image_encoder=ImageEncoder(
            scalp=1,
            trunk=Hiera(embed_dim=96, num_heads=1, stages=[1, 2, 7, 2], global_att_blocks=[5, 7, 9],
                        window_pos_embed_bkg_spatial_size=[7, 7]),
            neck=FpnNeck(position_encoding=PositionEmbeddingSine(num_pos_feats=256), d_model=256,
                         backbone_channel_list=[768, 384, 192, 96], fpn_top_down_levels=[2, 3],
                         fpn_interp_model="nearest"),
        ),
        memory_attention=MemoryAttention(
            d_model=256, pos_enc_at_input=True, num_layers=4,
            layer=MemoryAttentionLayer(
                activation="relu", dim_feedforward=2048, dropout=0.1, pos_enc_at_attn=False, d_model=256,
                pos_enc_at_cross_attn_keys=True, pos_enc_at_cross_attn_queries=False,
                self_attention=rope(), cross_attention=rope(rope_k_repeat=True, kv_in_dim=64),
            ),
        ),
        memory_encoder=MemoryEncoder(
            out_dim=64, position_encoding=PositionEmbeddingSine(num_pos_feats=64),
            mask_downsampler=MaskDownSampler(kernel_size=3, stride=2, padding=1),
            fuser=Fuser(layer=CXBlock(dim=256, kernel_size=7, padding=3, layer_scale_init_value=1e-6,
                                      use_dwconv=True), num_layers=2),
        ),
        num_maskmem=7, image_size=SIZE, sigmoid_scale_for_mem_enc=20.0, sigmoid_bias_for_mem_enc=-10.0,
        use_mask_input_as_output_without_sam=True, directly_add_no_mem_embed=True, no_obj_embed_spatial=True,
        use_high_res_features_in_sam=True, multimask_output_in_sam=True, iou_prediction_use_sigmoid=True,
        use_obj_ptrs_in_encoder=True, add_tpos_enc_to_obj_ptrs=True, proj_tpos_enc_in_obj_ptrs=True,
        use_signed_tpos_enc_to_obj_ptrs=True, only_obj_ptrs_in_the_past_for_eval=True, pred_obj_scores=True,
        pred_obj_scores_mlp=True, fixed_no_obj_ptr=True, multimask_output_for_tracking=True,
        use_multimask_token_for_obj_ptr=True, multimask_min_pt_num=0, multimask_max_pt_num=1,
        use_mlp_for_obj_ptr_proj=True, binarize_mask_from_pts_for_mem_enc=True,
        sam_mask_decoder_extra_args={"dynamic_multimask_via_stability": True,
                                     "dynamic_multimask_stability_delta": 0.05,
                                     "dynamic_multimask_stability_thresh": 0.98},
    )


def _fuse_bn(model: nn.Module) -> None:
    """Fold eval BatchNorms into the conv before them. The ONNX exporter would do it in fp32,
    leaving fp32 weights in the bf16 graph that TensorRT then rejects."""
    for seq in model.modules():
        if isinstance(seq, nn.Sequential):
            for i in range(len(seq) - 1):
                conv, bn = seq[i], seq[i + 1]
                if isinstance(conv, (nn.Conv2d, nn.ConvTranspose2d)) and isinstance(bn, nn.BatchNorm2d):
                    seq[i] = fuse_conv_bn_eval(conv, bn, transpose=isinstance(conv, nn.ConvTranspose2d))
                    seq[i + 1] = nn.Identity()


class _Encoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, img):
        return tuple(self.model.forward_image(img)["backbone_fpn"])


class _Alpha(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, mask, f0, f1, f2, img):
        return self.model._forward_alpha_heads(mask_inputs=mask, high_res_features=[f0, f1, f2], image=img)[0]


def _iou(a: torch.Tensor, b: torch.Tensor) -> float:
    union = (a | b).sum().item()
    return (a & b).sum().item() / union if union else 1.0


class SamMatter:
    """Call with the frame and RVM's alpha (both (1, ., H, W) half), get the alpha to use."""

    def __init__(self, models: Path, backend: str):
        model = build()
        model.load_state_dict(torch.load(models / WEIGHTS, map_location="cpu", weights_only=True)["model"])
        self.model = model.cuda().eval()
        _fuse_bn(self.model)
        # weights download once to ~/.cache/torch (BSD-3, COCO)
        self.detector = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1).eval().cuda()
        self.mean, self.std = MEAN.cuda(), STD.cuda()
        img = torch.zeros(1, 3, SIZE, SIZE, device="cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.model.forward_image(img)
        self.pos = out["vision_pos_enc"]  # depends on the shape only
        mask = torch.zeros(1, 1, SIZE // 4, SIZE // 4, device="cuda")
        args = {"mask": mask, **{f"f{i}": f for i, f in enumerate(out["backbone_fpn"])}, "img": img}
        if backend == "trt":
            self.encoder = self._engine(models, "encoder", _Encoder(self.model), {"img": img}, ["f0", "f1", "f2"],
                                        torch.bfloat16)
            # fp16: TensorRT has no bf16 kernel for the heads' transposed convs
            self.alpha = self._engine(models, "alpha", _Alpha(self.model), args, ["alpha"], torch.half)
        else:
            self.encoder, alpha = _Encoder(self.model), _Alpha(self.model)
            self.alpha = lambda **a: [alpha(**a)]
        self.reset()

    def _engine(self, models: Path, part: str, module: nn.Module, args: dict, outputs: list[str], dtype):
        from nvpoca.trt import Engine, build

        path = models / f"sam2matting_tiny_{part}.engine"
        if not path.exists():
            print(f"building {path.name} (one-time, takes a minute)...", flush=True)
            typed = {k: v.to(dtype) for k, v in args.items()}
            with torch.inference_mode():
                build(module.to(dtype), tuple(typed.values()), list(typed), outputs, path)
            self.model.float()  # build() converted the shared weights; torch parts run fp32 under autocast
        engine = Engine(path)
        return lambda **a: engine(**{k: v.to(dtype) for k, v in a.items()})

    def reset(self) -> None:
        self.memory = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
        self.t = 0  # frames since the camera (re)started or the last re-seed
        self.seeded = False
        self.disagree = 0

    def person(self, src: torch.Tensor) -> torch.Tensor | None:
        """The most confident person's mask at the tracker's input size, or None."""
        r = self.detector(list(F.interpolate(src.float(), scale_factor=0.5, mode="area")))[0]
        keep = (r["labels"] == 1) & (r["scores"] > 0.7)  # COCO label 1 = person; sorted by score
        if not keep.any():
            return None
        mask = (F.interpolate(r["masks"][keep][:1], size=(SIZE, SIZE), mode="bilinear") > 0.5).float()
        # its masks stop short of the frame edge, and the tracker would keep that green strip under the body
        edge = SIZE * 24 // 1000
        mask[..., -edge:, :] = mask[..., -edge - 1 : -edge, :]
        return mask

    @torch.inference_mode()
    def __call__(self, src: torch.Tensor, rvm: torch.Tensor) -> torch.Tensor:
        self.t += 1
        seed = None
        if not self.seeded:
            seed = self.person(src) if self.t % DETECT_EVERY == 1 else None
            if seed is None:  # nobody there; RVM would show the empty chair as a person
                return torch.zeros_like(rvm)
        img = (F.interpolate(src.float(), size=(SIZE, SIZE), mode="bilinear", antialias=True) - self.mean) / self.std
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fpn = self.encoder(img=img)
            _, feats, pos, sizes = self.model._prepare_backbone_features(
                {"backbone_fpn": [f.float() for f in fpn], "vision_pos_enc": self.pos})
            out = self.model.track_step(
                frame_idx=self.t, is_init_cond_frame=seed is not None, current_vision_feats=feats,
                current_vision_pos_embeds=pos, feat_sizes=sizes, point_inputs=None, mask_inputs=seed,
                output_dict=self.memory, num_frames=1 << 30, run_mem_encoder=True)
        out.pop("pred_masks_high_res")  # 1024² per frame, unused by the memory
        mask = out["pred_masks"] > 0
        if seed is not None:
            self.memory["cond_frame_outputs"][self.t] = out
            self.seeded = True
        else:
            self.memory["non_cond_frame_outputs"][self.t] = out
            self.memory["non_cond_frame_outputs"].pop(self.t - KEEP, None)
            # ponytail: IoU against RVM is a coarse lost/drift detector; the tracker's own object score
            # would catch "lost" earlier but not drift onto the chair
            ref = F.interpolate(rvm.float(), size=mask.shape[-2:], mode="area") > 0.5
            self.disagree = self.disagree + 1 if _iou(mask, ref) < RESEED_IOU else 0
            if self.disagree >= RESEED_AFTER:
                self.reset()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            alpha = self.alpha(mask=mask.float(), f0=fpn[0], f1=fpn[1], f2=fpn[2], img=img)[0]
        return F.interpolate(alpha.float(), size=src.shape[-2:], mode="bilinear").to(src.dtype)


def main():
    """Build the engines and check them against torch on a clip; prints mean |Δalpha| and timings."""
    from nvpoca.compare import load
    from nvpoca.core import MODELS, Matter, decode

    ap = argparse.ArgumentParser(prog="nvpoca.sam", description=main.__doc__)
    ap.add_argument("clip", type=Path)
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_stream(torch.cuda.Stream())
    srcs = [decode(d) for d in load(a.clip)[:150]]
    rvm = Matter("mobilenetv3", 0.25, *srcs[0].shape[-2:])
    trt_m, torch_m = SamMatter(MODELS, "trt"), SamMatter(MODELS, "torch")
    diff, times = 0.0, {"trt": [], "torch": []}
    for src in srcs:
        _, pha = rvm(src)
        outs = []
        for name, m in (("trt", trt_m), ("torch", torch_m)):
            torch.cuda.synchronize()
            t = time.perf_counter()
            outs.append(m(src, pha).float())
            torch.cuda.synchronize()
            times[name].append(time.perf_counter() - t)
        diff += (outs[0] - outs[1]).abs().mean().item()
    for name, ts in times.items():
        ts = sorted(ts[30:])
        print(f"{name}: median {ts[len(ts) // 2] * 1000:.1f} ms")
    print(f"mean |alpha trt - torch| = {diff / len(srcs):.2e}")


if __name__ == "__main__":
    main()
