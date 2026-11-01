#include "pipeline.hpp"

#include <NvOnnxParser.h>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <vector>

namespace {

#define CK(call)                                                                                  \
    do {                                                                                          \
        if (cudaError_t e_ = (call); e_ != cudaSuccess)                                            \
            throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(e_));          \
    } while (0)
#define CKJ(call)                                                                                 \
    do {                                                                                          \
        if (nvjpegStatus_t s_ = (call); s_ != NVJPEG_STATUS_SUCCESS)                               \
            throw std::runtime_error(std::string(#call) + ": nvjpeg status " + std::to_string(s_)); \
    } while (0)

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) std::fprintf(stderr, "TensorRT: %s\n", msg);
    }
} logger;

std::vector<char> read_file(const std::filesystem::path& path, size_t limit) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + path.string());
    if (std::filesystem::file_size(path) > limit) throw std::runtime_error(path.string() + " is too large");
    return {std::istreambuf_iterator<char>(f), {}};
}

void build_engine(const std::filesystem::path& onnx, const std::filesystem::path& engine) {
    std::fprintf(stderr, "building %s (one-time, takes a minute)...\n", engine.filename().c_str());
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0));  // strongly typed in TRT 11
    std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
    if (!parser->parseFromFile(onnx.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::string err;
        for (int i = 0; i < parser->getNbErrors(); ++i) err += std::string(parser->getError(i)->desc()) + "\n";
        throw std::runtime_error("parsing " + onnx.string() + " failed:\n" + err);
    }
    std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    std::unique_ptr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
    if (!plan) throw std::runtime_error("TensorRT engine build failed");
    auto tmp = engine;
    tmp += ".tmp";
    std::ofstream(tmp, std::ios::binary).write(static_cast<const char*>(plan->data()), plan->size());
    std::filesystem::rename(tmp, engine);
}

size_t volume(const nvinfer1::Dims& d) {
    size_t n = 1;
    for (int i = 0; i < d.nbDims; ++i) n *= d.d[i];
    return n;
}

}  // namespace

template <typename T>
T* Pipeline::alloc(size_t n) {
    void* p;
    CK(cudaMalloc(&p, n * sizeof(T)));
    allocs_.push_back(p);
    CK(cudaMemset(p, 0, n * sizeof(T)));
    return static_cast<T*>(p);
}

Planes Pipeline::planes() const { return {rgb_, rgb_ + w_ * h_, rgb_ + 2 * w_ * h_, w_, h_}; }

Pipeline::Pipeline(const std::filesystem::path& onnx, const std::filesystem::path& engine, int w, int h) : w_(w), h_(h) {
    CK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    CKJ(nvjpegCreateSimple(&jpeg_));
    CKJ(nvjpegJpegStateCreate(jpeg_, &jpeg_state_));

    if (!std::filesystem::exists(engine)) build_engine(onnx, engine);
    auto plan = read_file(engine, 1ull << 30);
    runtime_.reset(nvinfer1::createInferRuntime(logger));
    engine_.reset(runtime_->deserializeCudaEngine(plan.data(), plan.size()));
    if (!engine_) throw std::runtime_error("cannot load " + engine.string() + "; delete it to rebuild");
    ctx_.reset(engine_->createExecutionContext());
    auto src = engine_->getTensorShape("src");
    if (src.nbDims != 4 || src.d[2] != h || src.d[3] != w)
        throw std::runtime_error(engine.string() + " was built for another resolution");

    size_t n = static_cast<size_t>(w) * h;
    rgb_ = alloc<uint8_t>(3 * n);
    src_ = alloc<__half>(3 * n);
    fgr_ = alloc<__half>(3 * n);
    pha_ = alloc<__half>(n);
    for (int i = 0; i < 4; ++i) {
        auto name = "r" + std::to_string(i + 1);
        rec_bytes_[i] = volume(engine_->getTensorShape((name + "i").c_str())) * sizeof(__half);
        rec_in_[i] = alloc<uint8_t>(rec_bytes_[i]);
        rec_out_[i] = alloc<uint8_t>(rec_bytes_[i]);
    }
    ctx_->setTensorAddress("src", src_);
    ctx_->setTensorAddress("fgr", fgr_);
    ctx_->setTensorAddress("pha", pha_);
    denoise_acc_ = alloc<float>(3 * n);
    denoise_rate_ = alloc<float>(n);
    pha_s_ = alloc<float>(n);
    gray_ = alloc<float>(n / 16);
    prev_gray_ = alloc<float>(n / 16);
    rate_ = alloc<float>(n / 16);
    light_scratch_ = alloc<float>(2 * (w / 8) * (h / 8));
    blur_ = alloc<float4>(n / 16);
    blur_tmp_ = alloc<float4>(n / 16);
    light_ = alloc<LightState>(1);
    yuyv_ = alloc<uint8_t>(2 * n);
    CK(cudaMallocHost(&host_yuyv_, 2 * n));
    reset();
}

Pipeline::~Pipeline() {
    cudaStreamSynchronize(stream_);
    for (void* p : allocs_) cudaFree(p);
    cudaFreeHost(host_yuyv_);
    ctx_.reset();
    engine_.reset();
    runtime_.reset();
    nvjpegJpegStateDestroy(jpeg_state_);
    nvjpegDestroy(jpeg_);
    cudaStreamDestroy(stream_);
}

void Pipeline::reset() {
    for (int i = 0; i < 4; ++i) CK(cudaMemsetAsync(rec_in_[i], 0, rec_bytes_[i], stream_));
    LightState fresh;
    CK(cudaMemcpyAsync(light_, &fresh, sizeof fresh, cudaMemcpyHostToDevice, stream_));
    CK(cudaStreamSynchronize(stream_));
    first_ = denoise_first_ = true;
}

bool Pipeline::decode(const uint8_t* jpeg, size_t size) {
    int comps, widths[NVJPEG_MAX_COMPONENT], heights[NVJPEG_MAX_COMPONENT];
    nvjpegChromaSubsampling_t sub;
    if (nvjpegGetImageInfo(jpeg_, jpeg, size, &comps, &sub, widths, heights) != NVJPEG_STATUS_SUCCESS ||
        widths[0] != w_ || heights[0] != h_)
        return false;
    nvjpegImage_t out{};
    for (int c = 0; c < 3; ++c) {
        out.channel[c] = rgb_ + c * w_ * h_;
        out.pitch[c] = w_;
    }
    return nvjpegDecode(jpeg_, jpeg_state_, jpeg, size, NVJPEG_OUTPUT_RGB, &out, stream_) == NVJPEG_STATUS_SUCCESS;
}

bool Pipeline::set_background(const std::string& path) {
    bg_ = nullptr;
    if (path.empty()) return true;
    try {
        auto data = read_file(path, 64 << 20);
        auto bytes = reinterpret_cast<const uint8_t*>(data.data());
        int comps, widths[NVJPEG_MAX_COMPONENT], heights[NVJPEG_MAX_COMPONENT];
        nvjpegChromaSubsampling_t sub;
        CKJ(nvjpegGetImageInfo(jpeg_, bytes, data.size(), &comps, &sub, widths, heights));
        int iw = widths[0], ih = heights[0];
        if (iw <= 0 || ih <= 0 || iw > 16384 || ih > 16384) throw std::runtime_error("unsupported image size");
        uint8_t* tmp;
        CK(cudaMalloc(&tmp, 3ull * iw * ih));
        nvjpegImage_t out{};
        for (int c = 0; c < 3; ++c) {
            out.channel[c] = tmp + static_cast<size_t>(c) * iw * ih;
            out.pitch[c] = iw;
        }
        auto st = nvjpegDecode(jpeg_, jpeg_state_, bytes, data.size(), NVJPEG_OUTPUT_RGB, &out, stream_);
        if (st == NVJPEG_STATUS_SUCCESS) {
            if (!bg_storage_) bg_storage_ = alloc<uchar4>(static_cast<size_t>(w_) * h_);
            k::cover({tmp, tmp + iw * ih, tmp + 2 * iw * ih, iw, ih}, bg_storage_, w_, h_, stream_);
        }
        CK(cudaStreamSynchronize(stream_));
        cudaFree(tmp);
        CKJ(st);
        bg_ = bg_storage_;
        return true;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "background %s: %s; falling back to blur\n", path.c_str(), e.what());
        return false;
    }
}

void Pipeline::process(const EffectParams& p) {
    Mode mode = p.mode == Mode::replace && !bg_ ? Mode::blur : p.mode;
    if (mode != Mode::passthrough && p.denoise > 0) {
        // strength 1 averages over ~10 frames where the image is static
        k::denoise(rgb_, w_, h_, denoise_acc_, denoise_rate_, 1 - 0.9f * p.denoise, denoise_first_, stream_);
        denoise_first_ = false;
    } else {
        denoise_first_ = true;
    }
    if (mode != Mode::passthrough) {
        k::to_half(planes(), src_, stream_);
        for (int i = 0; i < 4; ++i) {
            auto name = "r" + std::to_string(i + 1);
            ctx_->setTensorAddress((name + "i").c_str(), rec_in_[i]);
            ctx_->setTensorAddress((name + "o").c_str(), rec_out_[i]);
        }
        if (!ctx_->enqueueV3(stream_)) throw std::runtime_error("TensorRT execution failed");
        // ping-pong the recurrent state: this frame's outputs are the next frame's inputs
        std::swap(rec_in_, rec_out_);

        if (p.smooth != last_smooth_) first_ = true;  // a new smoother, as gpu.AlphaSmoother
        last_smooth_ = p.smooth;
        if (p.smooth > 0) {
            // noise/motion tuned on recorded clips (README, Phase 2)
            k::motion_rate(planes(), gray_, prev_gray_, rate_, 2 / 255.f, 6 / 255.f, p.smooth, stream_);
            std::swap(gray_, prev_gray_);
        }
        k::smooth_alpha(pha_, rate_, pha_s_, w_, h_, first_ || p.smooth <= 0, stream_);
        first_ = false;

        if (p.light != last_light_) {
            LightState fresh;
            CK(cudaMemcpyAsync(light_, &fresh, sizeof fresh, cudaMemcpyHostToDevice, stream_));
        }
        last_light_ = p.light;
        if (p.light > 0) k::light_stats(planes(), pha_s_, light_scratch_, light_, p.light, stream_);
        if (mode == Mode::blur) k::blur_background(planes(), pha_s_, blur_, blur_tmp_, p.sigma, stream_);
    }
    k::compose_yuyv(planes(), fgr_, pha_, pha_s_, blur_, bg_, light_, {mode, p.smooth > 0, p.light > 0}, yuyv_, stream_);
    CK(cudaMemcpyAsync(host_yuyv_, yuyv_, yuyv_size(), cudaMemcpyDeviceToHost, stream_));
    CK(cudaStreamSynchronize(stream_));
    CK(cudaGetLastError());
}
