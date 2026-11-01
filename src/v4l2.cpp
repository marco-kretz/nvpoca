#include "v4l2.hpp"

#include <cerrno>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <system_error>
#include <unistd.h>

namespace {

// v4l2loopback V4L2_EVENT_PRI_CLIENT_USAGE (v4l2loopback.c)
constexpr uint32_t kClientUsage = 0x08000000 + 0x08E00000 + 1;

void xioctl(int fd, unsigned long req, void* arg, const char* what) {
    int r;
    do r = ioctl(fd, req, arg);
    while (r < 0 && errno == EINTR);
    if (r < 0) throw std::system_error(errno, std::generic_category(), what);
}

int open_dev(const std::string& path) {
    int fd = open(path.c_str(), O_RDWR | O_CLOEXEC);
    if (fd < 0) throw std::system_error(errno, std::generic_category(), path);
    return fd;
}

void set_parm(int fd, v4l2_buf_type type, int fps) {
    v4l2_streamparm parm{};
    parm.type = type;
    auto& tpf = type == V4L2_BUF_TYPE_VIDEO_CAPTURE ? parm.parm.capture.timeperframe : parm.parm.output.timeperframe;
    tpf = {1, static_cast<uint32_t>(fps)};
    xioctl(fd, VIDIOC_S_PARM, &parm, "VIDIOC_S_PARM");
}

}  // namespace

double monotonic_now() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

Camera::Camera(const std::string& path, int w, int h, int fps) : fd_(open_dev(path)) {
    try {
        v4l2_format fmt{};
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.fmt.pix.width = w;
        fmt.fmt.pix.height = h;
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
        xioctl(fd_, VIDIOC_S_FMT, &fmt, "VIDIOC_S_FMT");
        if (fmt.fmt.pix.width != static_cast<uint32_t>(w) || fmt.fmt.pix.height != static_cast<uint32_t>(h) ||
            fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_MJPEG)
            throw std::system_error(EINVAL, std::generic_category(), path + " refused " + std::to_string(w) + "x" +
                                                                         std::to_string(h) + " MJPG");
        set_parm(fd_, V4L2_BUF_TYPE_VIDEO_CAPTURE, fps);
        // keep the frame rate in low light (exposure_dynamic_framerate); not every camera has it
        v4l2_control ctl{V4L2_CID_EXPOSURE_AUTO_PRIORITY, 0};
        ioctl(fd_, VIDIOC_S_CTRL, &ctl);

        v4l2_requestbuffers req{};
        req.count = 2;
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_MMAP;
        xioctl(fd_, VIDIOC_REQBUFS, &req, "VIDIOC_REQBUFS");
        if (req.count != 2) throw std::system_error(ENOMEM, std::generic_category(), "VIDIOC_REQBUFS");
        for (int i = 0; i < 2; ++i) {
            v4l2_buffer buf{};
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index = i;
            xioctl(fd_, VIDIOC_QUERYBUF, &buf, "VIDIOC_QUERYBUF");
            bufs_[i] = mmap(nullptr, buf.length, PROT_READ, MAP_SHARED, fd_, buf.m.offset);
            if (bufs_[i] == MAP_FAILED) {
                bufs_[i] = nullptr;
                throw std::system_error(errno, std::generic_category(), "mmap");
            }
            lens_[i] = buf.length;
            xioctl(fd_, VIDIOC_QBUF, &buf, "VIDIOC_QBUF");
        }
        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        xioctl(fd_, VIDIOC_STREAMON, &type, "VIDIOC_STREAMON");
    } catch (...) {
        release();
        throw;
    }
}

Camera::~Camera() { release(); }

void Camera::release() {
    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ioctl(fd_, VIDIOC_STREAMOFF, &type);
    for (int i = 0; i < 2; ++i)
        if (bufs_[i]) munmap(bufs_[i], lens_[i]);
    close(fd_);
}

bool Camera::next(Frame& f, int timeout_ms) {
    pollfd p{fd_, POLLIN, 0};
    int r = poll(&p, 1, timeout_ms);
    if (r < 0 && errno != EINTR) throw std::system_error(errno, std::generic_category(), "poll camera");
    if (r <= 0) return false;
    v4l2_buffer buf{};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    xioctl(fd_, VIDIOC_DQBUF, &buf, "VIDIOC_DQBUF");
    f = {static_cast<int>(buf.index), static_cast<const uint8_t*>(bufs_[buf.index]),
         std::min<size_t>(buf.bytesused, lens_[buf.index]), buf.timestamp.tv_sec + buf.timestamp.tv_usec * 1e-6};
    return true;
}

void Camera::requeue(const Frame& f) {
    v4l2_buffer buf{};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = f.index;
    xioctl(fd_, VIDIOC_QBUF, &buf, "VIDIOC_QBUF");
}

Loopback::Loopback(const std::string& path, int w, int h, int fps) : fd_(open_dev(path)) {
    v4l2_format fmt{};
    fmt.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
    fmt.fmt.pix.width = w;
    fmt.fmt.pix.height = h;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;
    fmt.fmt.pix.bytesperline = w * 2;
    fmt.fmt.pix.sizeimage = w * h * 2;
    xioctl(fd_, VIDIOC_S_FMT, &fmt, "VIDIOC_S_FMT (loopback)");
    set_fps(fps);
}

Loopback::~Loopback() { close(fd_); }

void Loopback::write(const uint8_t* frame, size_t size) const {
    ssize_t n = ::write(fd_, frame, size);
    if (n < 0) throw std::system_error(errno, std::generic_category(), "write to the loopback device");
    if (static_cast<size_t>(n) != size) throw std::system_error(EIO, std::generic_category(), "short write to the loopback device");
}

void Loopback::set_fps(int fps) const { set_parm(fd_, V4L2_BUF_TYPE_VIDEO_OUTPUT, fps); }

void Loopback::subscribe_readers() const {
    v4l2_event_subscription sub{};
    sub.type = kClientUsage;
    sub.flags = V4L2_EVENT_SUB_FL_SEND_INITIAL;
    xioctl(fd_, VIDIOC_SUBSCRIBE_EVENT, &sub, "VIDIOC_SUBSCRIBE_EVENT");
}

int Loopback::read_readers() const {
    v4l2_event ev{};
    uint32_t count;
    do {
        xioctl(fd_, VIDIOC_DQEVENT, &ev, "VIDIOC_DQEVENT");
        std::memcpy(&count, ev.u.data, sizeof count);
    } while (ev.pending);
    return static_cast<int>(count);
}
