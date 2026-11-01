#include "kernels.hpp"

#include <cmath>
#include <cub/block/block_reduce.cuh>

namespace {

// key light tuning (untuned starting points, see gpu.KeyLight): target = face luma below which it
// lights, backlit = least face luma as a share of the background's, rate = per-frame ease
// (~0.7 s at 30 fps), neutral = how white the added light is (0 = room light colour, 1 = grey)
constexpr float kTarget = 0.42f, kBacklit = 0.6f, kRate = 0.05f, kNeutral = 0.5f;
constexpr int kMaxTaps = 129;
// denoise: mean colour change below kDenoiseNoise counts as noise, kDenoiseMotion above it as motion
constexpr float kDenoiseNoise = 2 / 255.f, kDenoiseMotion = 6 / 255.f;

struct Gauss {
    int r;
    float w[kMaxTaps / 2 + 1];  // w[0] centre, symmetric
};

__device__ float3 rgb(Planes p, int i) { return make_float3(p.r[i] / 255.f, p.g[i] / 255.f, p.b[i] / 255.f); }

__device__ float luma(float3 c) { return 0.299f * c.x + 0.587f * c.y + 0.114f * c.z; }

__device__ float3 lerp3(float3 a, float3 b, float t) {
    return make_float3(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y), a.z + t * (b.z - a.z));
}

__device__ float clamp01(float v) { return fminf(fmaxf(v, 0.f), 1.f); }

// F.interpolate(mode="area") window of output cell o of n over a source of size len
__device__ void area(int o, int n, int len, int& a, int& b) {
    a = o * len / n;
    b = ((o + 1) * len + n - 1) / n;
}

// F.interpolate(mode="bilinear", align_corners=False) source coordinate and taps
__device__ void bilin(int o, int n, int len, int& i0, int& i1, float& t) {
    float x = fmaxf((o + 0.5f) * len / n - 0.5f, 0.f);
    i0 = min(static_cast<int>(x), len - 1);
    i1 = min(i0 + 1, len - 1);
    t = x - i0;
}

template <typename T>
__device__ T sample(const T* img, int sw, int sh, int x, int y, int w, int h);

template <>
__device__ float sample(const float* img, int sw, int sh, int x, int y, int w, int h) {
    int x0, x1, y0, y1;
    float tx, ty;
    bilin(x, w, sw, x0, x1, tx);
    bilin(y, h, sh, y0, y1, ty);
    float a = img[y0 * sw + x0] + tx * (img[y0 * sw + x1] - img[y0 * sw + x0]);
    float b = img[y1 * sw + x0] + tx * (img[y1 * sw + x1] - img[y1 * sw + x0]);
    return a + ty * (b - a);
}

__device__ float3 sample_rgb(const float4* img, int sw, int sh, int x, int y, int w, int h) {
    int x0, x1, y0, y1;
    float tx, ty;
    bilin(x, w, sw, x0, x1, tx);
    bilin(y, h, sh, y0, y1, ty);
    auto at = [&](int xx, int yy) {
        float4 v = img[yy * sw + xx];
        return make_float3(v.x, v.y, v.z);
    };
    return lerp3(lerp3(at(x0, y0), at(x1, y0), tx), lerp3(at(x0, y1), at(x1, y1), tx), ty);
}

dim3 grid(int w, int h, dim3 block = dim3(32, 8)) { return dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y); }

#define XY(w, h)                                       \
    int x = blockIdx.x * blockDim.x + threadIdx.x;     \
    int y = blockIdx.y * blockDim.y + threadIdx.y;     \
    if (x >= (w) || y >= (h)) return

__global__ void to_half_k(Planes p, __half* dst) {
    XY(p.w, p.h);
    int i = y * p.w + x, n = p.w * p.h;
    float3 c = rgb(p, i);
    dst[i] = __float2half(c.x);
    dst[n + i] = __float2half(c.y);
    dst[2 * n + i] = __float2half(c.z);
}

__global__ void denoise_rate_k(const uint8_t* rgb, const float* acc, float* rate, int w, int h, float min_rate,
                               bool reset) {
    XY(w, h);
    int n = w * h;
    // 3x3 mean of the signed difference: sensor noise averages out, real change doesn't
    float d[3] = {};
    int cnt = 0;
    for (int yy = max(y - 1, 0); yy <= min(y + 1, h - 1); ++yy)
        for (int xx = max(x - 1, 0); xx <= min(x + 1, w - 1); ++xx, ++cnt)
            for (int c = 0; c < 3; ++c) d[c] += rgb[c * n + yy * w + xx] / 255.f - acc[c * n + yy * w + xx];
    float diff = (fabsf(d[0]) + fabsf(d[1]) + fabsf(d[2])) / (3 * cnt);
    rate[y * w + x] = reset ? 1 : fminf(fmaxf((diff - kDenoiseNoise) / kDenoiseMotion, min_rate), 1.f);
}

__global__ void denoise_apply_k(uint8_t* rgb, float* acc, const float* rate, int w, int h) {
    XY(w, h);
    int n = w * h, i = y * w + x;
    for (int c = 0; c < 3; ++c) {
        float& a = acc[c * n + i];
        a += rate[i] * (rgb[c * n + i] / 255.f - a);
        rgb[c * n + i] = static_cast<uint8_t>(rintf(a * 255));
    }
}

__global__ void gray4_k(Planes p, float* gray) {
    int gw = p.w / 4, gh = p.h / 4;
    XY(gw, gh);
    float sum = 0;
    for (int yy = 4 * y; yy < 4 * y + 4; ++yy)
        for (int xx = 4 * x; xx < 4 * x + 4; ++xx) {
            float3 c = rgb(p, yy * p.w + xx);
            sum += c.x + c.y + c.z;
        }
    gray[y * gw + x] = sum / 48;
}

__global__ void rate_k(const float* gray, const float* prev, float* rate, int gw, int gh, float noise, float motion,
                       float min_rate) {
    XY(gw, gh);
    // widen motion a little (5x5 max) so the edges of moving objects count as moving
    float move = 0;
    for (int yy = max(y - 2, 0); yy <= min(y + 2, gh - 1); ++yy)
        for (int xx = max(x - 2, 0); xx <= min(x + 2, gw - 1); ++xx)
            move = fmaxf(move, fabsf(gray[yy * gw + xx] - prev[yy * gw + xx]));
    rate[y * gw + x] = fminf(fmaxf((move - noise) / motion, min_rate), 1.f);
}

__global__ void smooth_k(const __half* pha, const float* rate, float* pha_s, int w, int h, bool follow) {
    XY(w, h);
    int i = y * w + x;
    float a = __half2float(pha[i]);
    pha_s[i] = follow ? a : pha_s[i] + sample(rate, w / 4, h / 4, x, y, w, h) * (a - pha_s[i]);
}

__global__ void down8_k(Planes p, const float* pha, float* y8, float* a8) {
    int w8 = p.w / 8, h8 = p.h / 8;
    XY(w8, h8);
    int x0, x1, y0, y1;
    area(x, w8, p.w, x0, x1);
    area(y, h8, p.h, y0, y1);
    float ys = 0, as = 0;
    for (int yy = y0; yy < y1; ++yy)
        for (int xx = x0; xx < x1; ++xx) {
            ys += luma(rgb(p, yy * p.w + xx));
            as += pha[yy * p.w + xx];
        }
    float n = (x1 - x0) * (y1 - y0);
    y8[y * w8 + x] = ys / n;
    a8[y * w8 + x] = as / n;
}

// Gaussian centred just below the top of the person, at the middle of their upper part (gpu.KeyLight.falloff)
__device__ float falloff(float x, float y, float top, float cx, int h8) {
    float s = 0.6f * h8, dy = (y - top - 0.2f * h8) / s, dx = (x - cx) / s;
    return expf(-dy * dy / 2) * expf(-dx * dx / 2);
}

constexpr int kLightThreads = 1024;

__global__ void light_k(const float* y8, const float* a8, int w8, int h8, LightState* st, float strength) {
    using Reduce = cub::BlockReduce<float, kLightThreads>;
    __shared__ Reduce::TempStorage tmp;
    __shared__ float top, cx;
    int n = w8 * h8;

    int first = h8;  // first row with person; 0 when there is none, as argmax over all zeros
    for (int i = threadIdx.x; i < n; i += kLightThreads)
        if (a8[i] > 0.5f) first = min(first, i / w8);
    float row = Reduce(tmp).Reduce(static_cast<float>(first), [](float a, float b) { return fminf(a, b); });
    if (threadIdx.x == 0) top = row == h8 ? 0 : row;
    __syncthreads();

    float cols = 0, count = 0;
    for (int i = threadIdx.x; i < n; i += kLightThreads) {
        int r = i / w8;
        if (a8[i] > 0.5f && r >= top && r < top + 0.3f * h8) cols += i % w8, count += 1;
    }
    cols = Reduce(tmp).Sum(cols);
    __syncthreads();
    count = Reduce(tmp).Sum(count);
    if (threadIdx.x == 0) cx = cols / fmaxf(count, 1.f);
    __syncthreads();

    float sums[5] = {};  // person*light, y*person*light, person, back, y*back
    for (int i = threadIdx.x; i < n; i += kLightThreads) {
        float a = a8[i], y = y8[i];
        if (a > 0.9f) {
            float l = falloff(i % w8, i / w8, top, cx, h8);
            sums[0] += l, sums[1] += y * l, sums[2] += 1;
        }
        if (a < 0.1f) sums[3] += 1, sums[4] += y;
    }
    for (float& v : sums) {
        __syncthreads();
        v = Reduce(tmp).Sum(v);
    }
    if (threadIdx.x != 0) return;
    float face = sums[1] / fmaxf(sums[0], 1e-3f), bg = sums[4] / fmaxf(sums[3], 1.f);
    float gain = fminf(fmaxf(fmaxf(kBacklit * bg, kTarget) / fmaxf(face, 0.02f), 1.f), 1 + 2 * strength);
    if (sums[2] <= 64) gain = 1;
    st->gain = st->gain < 0 ? gain : st->gain + kRate * (gain - st->gain);
    st->top = top;
    st->cx = cx;
}

__global__ void blur_prep_k(Planes p, const float* pha, float4* q) {
    int qw = p.w / 4, qh = p.h / 4;
    XY(qw, qh);
    int x0, x1, y0, y1;
    area(x, qw, p.w, x0, x1);
    area(y, qh, p.h, y0, y1);
    float4 acc = {};
    for (int yy = y0; yy < y1; ++yy)
        for (int xx = x0; xx < x1; ++xx) {
            int i = yy * p.w + xx;
            float wt = clamp01(1 - pha[i]);
            wt *= wt;
            wt *= wt;  // ^4 drops semi-transparent hair/edges out of the fill
            float3 c = rgb(p, i);
            acc.x += c.x * wt, acc.y += c.y * wt, acc.z += c.z * wt, acc.w += wt;
        }
    float n = (x1 - x0) * (y1 - y0);
    q[y * qw + x] = make_float4(acc.x / n, acc.y / n, acc.z / n, acc.w / n);
}

__device__ int reflect(int i, int n) { return i < 0 ? -i : i >= n ? 2 * n - 2 - i : i; }

// separable Gaussian with reflect padding (torchvision gaussian_blur); the vertical pass normalises
__global__ void blur_k(const float4* in, float4* out, int w, int h, Gauss g, bool vertical) {
    XY(w, h);
    float4 acc = {};
    for (int d = -g.r; d <= g.r; ++d) {
        int i = vertical ? reflect(y + d, h) * w + x : y * w + reflect(x + d, w);
        float wt = g.w[abs(d)];
        float4 v = in[i];
        acc.x += wt * v.x, acc.y += wt * v.y, acc.z += wt * v.z, acc.w += wt * v.w;
    }
    if (vertical) {
        float den = fmaxf(acc.w, 1e-4f);
        acc = make_float4(acc.x / den, acc.y / den, acc.z / den, 1);
    }
    out[y * w + x] = acc;
}

__global__ void cover_k(Planes img, uchar4* dst, int w, int h) {
    XY(w, h);
    float s = fmaxf(static_cast<float>(h) / img.h, static_cast<float>(w) / img.w);
    int sw = ceilf(img.w * s), sh = ceilf(img.h * s);
    float fx = (x + (sw - w) / 2 + 0.5f) / s - 0.5f, fy = (y + (sh - h) / 2 + 0.5f) / s - 0.5f;
    float3 c;
    if (s < 1) {  // downscale: box over the source footprint (antialiased)
        int x0 = max(static_cast<int>(floorf(fx + 0.5f - 0.5f / s)), 0), x1 = min(static_cast<int>(ceilf(fx + 0.5f + 0.5f / s)), img.w);
        int y0 = max(static_cast<int>(floorf(fy + 0.5f - 0.5f / s)), 0), y1 = min(static_cast<int>(ceilf(fy + 0.5f + 0.5f / s)), img.h);
        c = {};
        for (int yy = y0; yy < y1; ++yy)
            for (int xx = x0; xx < x1; ++xx) {
                float3 v = rgb(img, yy * img.w + xx);
                c.x += v.x, c.y += v.y, c.z += v.z;
            }
        float n = fmaxf((x1 - x0) * (y1 - y0), 1);
        c = make_float3(c.x / n, c.y / n, c.z / n);
    } else {
        fx = fminf(fmaxf(fx, 0.f), img.w - 1.f), fy = fminf(fmaxf(fy, 0.f), img.h - 1.f);
        int x0 = fx, y0 = fy, x1 = min(x0 + 1, img.w - 1), y1 = min(y0 + 1, img.h - 1);
        float tx = fx - x0, ty = fy - y0;
        c = lerp3(lerp3(rgb(img, y0 * img.w + x0), rgb(img, y0 * img.w + x1), tx),
                  lerp3(rgb(img, y1 * img.w + x0), rgb(img, y1 * img.w + x1), tx), ty);
    }
    dst[y * w + x] = make_uchar4(rintf(c.x * 255), rintf(c.y * 255), rintf(c.z * 255), 255);
}

struct ComposeIn {
    Planes src;
    const __half *fgr, *pha_raw;
    const float* pha_s;
    const float4* bg_blur;
    const uchar4* bg_img;
    const LightState* st;
    ComposeParams p;
};

__device__ float3 pixel(const ComposeIn& in, int x, int y) {
    int w = in.src.w, h = in.src.h, i = y * w + x, n = w * h;
    float3 s = rgb(in.src, i);
    if (in.p.mode == Mode::passthrough) return s;
    float3 f = make_float3(__half2float(in.fgr[i]), __half2float(in.fgr[n + i]), __half2float(in.fgr[2 * n + i]));
    if (in.p.smooth) f = lerp3(s, f, fminf(__half2float(in.pha_raw[i]) / 0.05f, 1.f));
    float a = in.pha_s[i];
    if (in.p.light) {
        int w8 = w / 8, h8 = h / 8;
        float lx = fminf(fmaxf((x + 0.5f) * w8 / w - 0.5f, 0.f), w8 - 1.f);
        float ly = fminf(fmaxf((y + 0.5f) * h8 / h - 0.5f, 0.f), h8 - 1.f);
        float k = 1 + (in.st->gain - 1) * falloff(lx, ly, in.st->top, in.st->cx, h8);
        float yl = clamp01(luma(f));
        float lift = 1 - powf(1 - yl, k) - yl;  // ~k*y in the shadows, rolls off to 1 so highlights don't clip
        float inv = 1 / fmaxf(yl, 1e-3f);
        // added light takes the pixel's colour pulled towards white, so a warm room light doesn't turn skin orange
        f = make_float3(clamp01(f.x + lift * (f.x * inv + kNeutral * (1 - f.x * inv))),
                        clamp01(f.y + lift * (f.y * inv + kNeutral * (1 - f.y * inv))),
                        clamp01(f.z + lift * (f.z * inv + kNeutral * (1 - f.z * inv))));
    }
    float3 back;
    if (in.p.mode == Mode::blur) {
        back = sample_rgb(in.bg_blur, w / 4, h / 4, x, y, w, h);
    } else {
        // a flat or new background shows every half-transparent edge pixel as a halo: hard edges, as a keyer would cut them
        a = clamp01((a - 0.4f) / 0.2f);
        if (in.p.mode == Mode::green) {
            back = make_float3(0, 1, 0);  // pure #00FF00, OBS's default chroma key colour
        } else {
            uchar4 b = in.bg_img[i];
            back = make_float3(b.x / 255.f, b.y / 255.f, b.z / 255.f);
        }
    }
    return lerp3(back, f, a);
}

__global__ void compose_k(ComposeIn in, uint8_t* yuyv) {
    int w = in.src.w;
    XY(w / 2, in.src.h);
    float3 c0 = pixel(in, 2 * x, y), c1 = pixel(in, 2 * x + 1, y);
    float y0 = luma(c0), y1 = luma(c1);
    // chroma from the horizontal pixel-pair average (4:2:2)
    float r = (c0.x + c1.x) / 2, b = (c0.z + c1.z) / 2, ym = (y0 + y1) / 2;
    float u = 128 + 224 * 0.5f * (b - ym) / (1 - 0.114f), v = 128 + 224 * 0.5f * (r - ym) / (1 - 0.299f);
    auto q = [](float t) { return static_cast<uint8_t>(fminf(fmaxf(rintf(t), 0.f), 255.f)); };
    uchar4 out = {q(16 + 219 * y0), q(u), q(16 + 219 * y1), q(v)};
    reinterpret_cast<uchar4*>(yuyv)[y * (w / 2) + x] = out;
}

}  // namespace

namespace k {

void denoise(uint8_t* rgb, int w, int h, float* acc, float* rate, float min_rate, bool reset, cudaStream_t s) {
    denoise_rate_k<<<grid(w, h), dim3(32, 8), 0, s>>>(rgb, acc, rate, w, h, min_rate, reset);
    denoise_apply_k<<<grid(w, h), dim3(32, 8), 0, s>>>(rgb, acc, rate, w, h);
}

void to_half(Planes src, __half* dst, cudaStream_t s) { to_half_k<<<grid(src.w, src.h), dim3(32, 8), 0, s>>>(src, dst); }

void motion_rate(Planes src, float* gray, const float* prev_gray, float* rate, float noise, float motion, float min_rate,
                 cudaStream_t s) {
    int gw = src.w / 4, gh = src.h / 4;
    gray4_k<<<grid(gw, gh), dim3(32, 8), 0, s>>>(src, gray);
    rate_k<<<grid(gw, gh), dim3(32, 8), 0, s>>>(gray, prev_gray, rate, gw, gh, noise, motion, min_rate);
}

void smooth_alpha(const __half* pha, const float* rate, float* pha_s, int w, int h, bool follow, cudaStream_t s) {
    smooth_k<<<grid(w, h), dim3(32, 8), 0, s>>>(pha, rate, pha_s, w, h, follow);
}

void light_stats(Planes src, const float* pha_s, float* scratch, LightState* st, float strength, cudaStream_t s) {
    int w8 = src.w / 8, h8 = src.h / 8;
    float *y8 = scratch, *a8 = scratch + w8 * h8;
    down8_k<<<grid(w8, h8), dim3(32, 8), 0, s>>>(src, pha_s, y8, a8);
    light_k<<<1, kLightThreads, 0, s>>>(y8, a8, w8, h8, st, strength);
}

void blur_background(Planes src, const float* pha_s, float4* q, float4* scratch, float sigma, cudaStream_t s) {
    int qw = src.w / 4, qh = src.h / 4;
    // blur at quarter resolution: visually identical for large sigma, ~16x cheaper
    float sq = fmaxf(sigma / 4, 0.5f);
    Gauss g{std::min(static_cast<int>(std::ceil(3 * sq)), kMaxTaps / 2), {}};
    float sum = 0;
    for (int d = 0; d <= g.r; ++d) sum += (d ? 2 : 1) * (g.w[d] = std::exp(-0.5f * (d / sq) * (d / sq)));
    for (int d = 0; d <= g.r; ++d) g.w[d] /= sum;
    blur_prep_k<<<grid(qw, qh), dim3(32, 8), 0, s>>>(src, pha_s, q);
    blur_k<<<grid(qw, qh), dim3(32, 8), 0, s>>>(q, scratch, qw, qh, g, false);
    blur_k<<<grid(qw, qh), dim3(32, 8), 0, s>>>(scratch, q, qw, qh, g, true);
}

void cover(Planes img, uchar4* dst, int w, int h, cudaStream_t s) { cover_k<<<grid(w, h), dim3(32, 8), 0, s>>>(img, dst, w, h); }

void compose_yuyv(Planes src, const __half* fgr, const __half* pha_raw, const float* pha_s, const float4* bg_blur,
                  const uchar4* bg_img, const LightState* st, ComposeParams p, uint8_t* yuyv, cudaStream_t s) {
    ComposeIn in{src, fgr, pha_raw, pha_s, bg_blur, bg_img, st, p};
    compose_k<<<grid(src.w / 2, src.h), dim3(32, 8), 0, s>>>(in, yuyv);
}

}  // namespace k
