// GPU side: nvJPEG decode, RVM (TensorRT), effects, YUYV download. Owns all CUDA state.
#pragma once

#include "kernels.hpp"

#include <NvInfer.h>
#include <filesystem>
#include <memory>
#include <nvjpeg.h>
#include <string>
#include <vector>

struct EffectParams {
    Mode mode;     // passthrough when disabled; replace falls back to blur without a background
    float sigma;   // blur sigma in output pixels
    float smooth;  // min EMA rate, 0 = off
    float light;   // key light strength, 0 = off
};

class Pipeline {
public:
    // builds the engine from `onnx` into `engine` on first use (~20-40 s)
    Pipeline(const std::filesystem::path& onnx, const std::filesystem::path& engine, int w, int h);
    ~Pipeline();
    Pipeline(const Pipeline&) = delete;

    // false on corrupt or wrong-size MJPEG (USB hiccups)
    bool decode(const uint8_t* jpeg, size_t size);
    // effects + YUYV into yuyv(); blocks until the frame is downloaded
    void process(const EffectParams& p);
    const uint8_t* yuyv() const { return host_yuyv_; }
    size_t yuyv_size() const { return static_cast<size_t>(w_) * h_ * 2; }
    // JPEG only (nvJPEG); false and no background on failure
    bool set_background(const std::string& path);
    // forget temporal state, e.g. when the camera restarts
    void reset();

private:
    template <typename T>
    T* alloc(size_t n);
    Planes planes() const;

    int w_, h_;
    cudaStream_t stream_{};
    nvjpegHandle_t jpeg_{};
    nvjpegJpegState_t jpeg_state_{};
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> ctx_;
    std::vector<void*> allocs_;

    uint8_t* rgb_;  // 3 planes
    __half *src_, *fgr_, *pha_;
    void *rec_in_[4], *rec_out_[4];
    size_t rec_bytes_[4];
    float *pha_s_, *gray_, *prev_gray_, *rate_, *light_scratch_;
    float4 *blur_, *blur_tmp_;
    uchar4* bg_storage_ = nullptr;
    uchar4* bg_ = nullptr;  // bg_storage_ while a background is loaded
    LightState* light_;
    uint8_t* yuyv_;
    uint8_t* host_yuyv_ = nullptr;

    bool first_ = true;  // no alpha history yet
    float last_smooth_ = 0, last_light_ = 0;
};
