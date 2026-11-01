// CUDA kernels of the effect pipeline. Ports of nvpoca/gpu.py; colours are floats in [0, 1].
#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// planar 8-bit RGB as nvJPEG writes it, pitch == w
struct Planes {
    const uint8_t *r, *g, *b;
    int w, h;
};

// key light state carried across frames; gain < 0 = not measured yet
struct LightState {
    float gain = -1, top = 0, cx = 0;
};

enum class Mode : int { passthrough, blur, green, replace };

struct ComposeParams {
    Mode mode;
    bool smooth;  // RVM fgr is undefined on the background: show camera pixels where alpha is held
    bool light;
};

namespace k {

// motion-adaptive temporal denoise of the planar RGB in place: static areas are averaged over
// frames (down to min_rate per frame), moving ones follow at once. acc: 3*w*h, rate: w*h floats
void denoise(uint8_t* rgb, int w, int h, float* acc, float* rate, float min_rate, bool reset, cudaStream_t s);

// (1, 3, h, w) half NCHW model input
void to_half(Planes src, __half* dst, cudaStream_t s);

// gray = 4x-down mean of src; rate = per-pixel EMA rate from |gray - prev_gray| (AlphaSmoother)
void motion_rate(Planes src, float* gray, const float* prev_gray, float* rate, float noise, float motion, float min_rate,
                 cudaStream_t s);
// pha_s += rate * (pha - pha_s); follow = take pha as is (first frame, smoothing off)
void smooth_alpha(const __half* pha, const float* rate, float* pha_s, int w, int h, bool follow, cudaStream_t s);

// measure the key light gain on 8x-down src/pha_s; scratch: 2 * (w/8) * (h/8) floats
void light_stats(Planes src, const float* pha_s, float* scratch, LightState* st, float strength, cudaStream_t s);

// background-only blur at quarter resolution into q (rgb, w unused); scratch same size
void blur_background(Planes src, const float* pha_s, float4* q, float4* scratch, float sigma, cudaStream_t s);

// background image (any size) resized to cover w x h, centre-cropped
void cover(Planes img, uchar4* dst, int w, int h, cudaStream_t s);

// composite + BT.601 limited-range YUYV (what browsers and Zoom assume for YUYV webcams)
void compose_yuyv(Planes src, const __half* fgr, const __half* pha_raw, const float* pha_s, const float4* bg_blur,
                  const uchar4* bg_img, const LightState* st, ComposeParams p, uint8_t* yuyv, cudaStream_t s);

}  // namespace k
