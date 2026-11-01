// Real camera (MJPEG capture) and the v4l2loopback output device.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

struct Frame {
    int index;
    const uint8_t* data;
    size_t size;
    double timestamp;  // CLOCK_MONOTONIC seconds at USB start-of-frame
};

class Camera {
public:
    // throws std::system_error (errno EBUSY while another app streams the camera)
    Camera(const std::string& path, int w, int h, int fps);
    ~Camera();
    Camera(const Camera&) = delete;
    // false on timeout
    bool next(Frame& f, int timeout_ms);
    void requeue(const Frame& f);

private:
    void release();

    int fd_;
    void* bufs_[2] = {};
    size_t lens_[2] = {};
};

class Loopback {
public:
    Loopback(const std::string& path, int w, int h, int fps);
    ~Loopback();
    int fd() const { return fd_; }
    void write(const uint8_t* frame, size_t size) const;
    void set_fps(int fps) const;
    // v4l2loopback reports the reader count when an app starts or stops streaming (or dies)
    void subscribe_readers() const;
    // pending reader-count events; returns the latest count
    int read_readers() const;

private:
    int fd_;
};

double monotonic_now();
