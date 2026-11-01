// Self-checks of the kernels, ported from the asserts in nvpoca/gpu.py. Exit code != 0 on failure.
#include "../src/kernels.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #cond); \
            std::exit(1);                                                 \
        }                                                                 \
    } while (0)

template <typename T>
struct Dev {
    T* p = nullptr;
    size_t n;
    explicit Dev(size_t n) : n(n) { cudaMalloc(&p, n * sizeof(T)), cudaMemset(p, 0, n * sizeof(T)); }
    Dev(const std::vector<T>& v) : Dev(v.size()) { cudaMemcpy(p, v.data(), n * sizeof(T), cudaMemcpyHostToDevice); }
    ~Dev() { cudaFree(p); }
    std::vector<T> get() const {
        std::vector<T> v(n);
        cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost);
        return v;
    }
};

struct Image {  // planar RGB on the GPU, filled per pixel from a host function returning 0..255
    int w, h;
    Dev<uint8_t> d;
    template <typename F>
    Image(int w, int h, F f) : w(w), h(h), d(fill(w, h, f)) {}
    template <typename F>
    static std::vector<uint8_t> fill(int w, int h, F f) {
        std::vector<uint8_t> v(3 * w * h);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                for (int c = 0; c < 3; ++c) v[c * w * h + y * w + x] = f(x, y, c);
        return v;
    }
    Planes planes() const { return {d.p, d.p + w * h, d.p + 2 * w * h, w, h}; }
};

std::vector<__half> halves(int n, auto f) {
    std::vector<__half> v(n);
    for (int i = 0; i < n; ++i) v[i] = __float2half(f(i));
    return v;
}

// red channel of a uniform pixel pair from YUYV (inverse of the BT.601 limited-range packing)
float red(const std::vector<uint8_t>& yuyv, int w, int x, int y) {
    const uint8_t* p = &yuyv[(y * w + (x & ~1)) * 2];
    return (p[0] - 16) / 219.f + (p[3] - 128) / 112.f * (1 - 0.299f);
}
float blue(const std::vector<uint8_t>& yuyv, int w, int x, int y) {
    const uint8_t* p = &yuyv[(y * w + (x & ~1)) * 2];
    return (p[0] - 16) / 219.f + (p[1] - 128) / 112.f * (1 - 0.114f);
}

void test_yuyv() {
    // white, black, pure red match the BT.601 limited-range reference values
    Image img(6, 1, [](int x, int, int c) { return x < 2 || (x >= 4 && c == 0) ? 255 : 0; });
    Dev<uint8_t> out(12);
    k::compose_yuyv(img.planes(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, {Mode::passthrough, false, false},
                    out.p, 0);
    std::vector<uint8_t> want = {235, 128, 235, 128, 16, 128, 16, 128, 81, 90, 81, 240};
    CHECK(out.get() == want);
}

void test_blur() {
    // blurred background must not contain foreground colour: red person on green background
    auto person = [](int x, int y) { return x >= 16 && x < 48 && y >= 16 && y < 48; };
    Image img(64, 64, [&](int x, int y, int c) { return c == (person(x, y) ? 0 : 1) ? 255 : 0; });
    std::vector<float> pha(64 * 64);
    for (int i = 0; i < 64 * 64; ++i) pha[i] = person(i % 64, i / 64);
    Dev<float> pha_d(pha);
    Dev<float4> q(16 * 16), tmp(16 * 16);
    k::blur_background(img.planes(), pha_d.p, q.p, tmp.p, 8, 0);
    for (float4 v : q.get()) CHECK(v.x < 0.01f && v.y > 0.99f);
}

void test_smoother() {
    Image still(16, 16, [](int, int, int) { return 0; }), moved(16, 16, [](int, int, int) { return 128; });
    Dev<float> g0(16), g1(16), rate(16), pha_s(256);
    Dev<__half> zero(halves(256, [](int) { return 0.f; })), one(halves(256, [](int) { return 1.f; }));
    k::motion_rate(still.planes(), g0.p, g1.p, rate.p, 2 / 255.f, 6 / 255.f, 0.15f, 0);
    k::smooth_alpha(zero.p, rate.p, pha_s.p, 16, 16, true, 0);
    k::motion_rate(still.planes(), g1.p, g0.p, rate.p, 2 / 255.f, 6 / 255.f, 0.15f, 0);
    k::smooth_alpha(one.p, rate.p, pha_s.p, 16, 16, false, 0);
    for (float a : pha_s.get()) CHECK(std::fabs(a - 0.15f) < 1e-5f);  // static image: alpha jump is flicker, damped
    k::motion_rate(moved.planes(), g0.p, g1.p, rate.p, 2 / 255.f, 6 / 255.f, 0.15f, 0);
    k::smooth_alpha(one.p, rate.p, pha_s.p, 16, 16, false, 0);
    for (float a : pha_s.get()) CHECK(std::fabs(a - 1) < 1e-5f);  // image moved: alpha follows at once
}

// key light on a 256x256 frame with a person at x 96..159, y 64..255; returns the YUYV output
std::vector<uint8_t> lit(auto colour, bool person, float strength, LightState* st_out = nullptr) {
    constexpr int n = 256;
    auto inside = [&](int x, int y) { return person && x >= 96 && x < 160 && y >= 64; };
    Image img(n, n, [&](int x, int y, int c) { return colour(x, y, c, inside(x, y)); });
    std::vector<float> pha(n * n);
    for (int i = 0; i < n * n; ++i) pha[i] = inside(i % n, i / n);
    Dev<float> pha_s(pha), scratch(2 * 32 * 32);
    std::vector<__half> fgr(3 * n * n);
    auto host = Image::fill(n, n, [&](int x, int y, int c) { return colour(x, y, c, inside(x, y)); });
    for (int i = 0; i < 3 * n * n; ++i) fgr[i] = __float2half(host[i] / 255.f);
    Dev<__half> fgr_d(fgr);
    Dev<LightState> st(std::vector<LightState>{LightState{}});
    k::light_stats(img.planes(), pha_s.p, scratch.p, st.p, strength, 0);
    Dev<uint8_t> out(2 * n * n);
    // green: the person's pixels show the lit fgr unchanged by any background
    k::compose_yuyv(img.planes(), fgr_d.p, nullptr, pha_s.p, nullptr, nullptr, st.p, {Mode::green, false, true}, out.p, 0);
    if (st_out) *st_out = st.get()[0];
    return out.get();
}

void test_light() {
    // dark person is lit with highlights kept
    auto dark = [](int x, int y, int, bool) { return x >= 96 && x < 110 && y >= 240 ? 255 : 38; };  // white shirt
    auto out = lit(dark, true, 1);
    float r = red(out, 256, 128, 100);
    CHECK(r > 0.3f && r < 0.5f);
    CHECK(red(out, 256, 100, 250) > 0.99f);
    // a well-lit person is left alone
    out = lit([](int, int, int, bool) { return 128; }, true, 0.5f);
    CHECK(std::fabs(red(out, 256, 128, 100) - 128 / 255.f) < 0.01f);
    // no person: gain stays 1
    LightState st;
    lit([](int, int, int, bool) { return 38; }, false, 0.5f, &st);
    CHECK(st.gain == 1);
    // bright enough, but in front of a window: lifted
    out = lit([](int, int, int, bool p) { return p ? 115 : 242; }, true, 0.5f);
    CHECK(red(out, 256, 128, 100) > 0.5f);
    // warm-lit skin: brighter and less orange (input r/b = 4)
    out = lit([](int, int, int c, bool p) { return p ? (c == 0 ? 51 : c == 1 ? 26 : 13) : 128; }, true, 1);
    float rr = red(out, 256, 128, 100), bb = blue(out, 256, 128, 100);
    CHECK(rr > 0.2f && rr / bb < 3);
}

int main() {
    test_yuyv();
    test_blur();
    test_smoother();
    test_light();
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    std::puts("ok");
}
