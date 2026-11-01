// nvpoca daemon: keeps the v4l2loopback camera ready, runs the GPU pipeline while an app reads it,
// frees the VRAM --unload-after seconds after the last reader, and serves the settings over D-Bus
// (org.nvpoca.Daemon on the session bus).
#include "pipeline.hpp"
#include "v4l2.hpp"

#include <NvInferVersion.h>
#include <algorithm>
#include <cerrno>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <sys/eventfd.h>
#include <sys/epoll.h>
#include <systemd/sd-bus.h>
#include <systemd/sd-event.h>
#include <thread>
#include <unistd.h>

namespace fs = std::filesystem;

namespace {

constexpr const char* kBusName = "org.nvpoca.Daemon";
constexpr const char* kPath = "/org/nvpoca/Daemon";

struct Settings {
    bool enabled = true;
    std::string mode = "blur";
    double blur = 4, smooth = 0.15, light = 0;
    std::string bg;
};

struct Args {
    Settings settings;
    std::set<std::string> explicit_;  // settings given on the command line override the saved ones
    std::string input = "/dev/video0", output = "/dev/video10", variant = "mobilenetv3";
    fs::path models = NVPOCA_MODELS;
    int w = 1920, h = 1080, fps = 30;
    double ratio = 0, unload_after = 60;
};

void log(const char* fmt, auto... args) {
    std::fprintf(stderr, fmt, args...);
    std::fputc('\n', stderr);
}

// nullopt = valid; shared by the command line, the settings file and D-Bus
std::optional<std::string> invalid(const std::string& key, double v) {
    if (key == "blur" && !(v >= 0.5 && v <= 40)) return "Blur must be within 0.5..40";
    if (key == "smooth" && !(v >= 0 && v <= 1)) return "Smooth must be within 0..1 (0 = off)";
    if (key == "light" && !(v >= 0 && v <= 1)) return "Light must be within 0..1 (0 = off)";
    return std::nullopt;
}

bool valid_mode(const std::string& m) { return m == "blur" || m == "replace" || m == "green"; }

std::string expand_home(const std::string& p) {
    const char* home = std::getenv("HOME");
    return p.starts_with("~/") && home ? home + p.substr(1) : p;
}

std::optional<std::string> invalid_background(const std::string& path) {
    if (path.empty()) return std::nullopt;
    if (!fs::is_regular_file(path)) return "no such file: " + path;
    std::string ext = fs::path(path).extension();
    std::ranges::transform(ext, ext.begin(), ::tolower);
    if (ext != ".jpg" && ext != ".jpeg") return "background must be a JPEG: " + path;
    return std::nullopt;
}

const char* kUsage = R"(usage: nvpoca [options]
  --mode blur|replace|green|passthrough   (default blur)
  --bg PATH            JPEG background for --mode replace
  --blur SIGMA         blur sigma in output pixels at 1080p, 0.5..40 (default 4)
  --smooth [MIN_RATE]  damp alpha flicker where the image is static, 0..1 (default 0.15)
  --no-smooth          raw RVM alpha
  --light [STRENGTH]   virtual key light when underexposed or backlit, 0..1 (bare: 0.5)
  --variant NAME       mobilenetv3 | resnet50 (default mobilenetv3)
  --ratio R            RVM downsample ratio (default: 480 px on the long side)
  --input DEV          real camera (default /dev/video0)
  --output DEV         v4l2loopback device (default /dev/video10)
  --size WxH           (default 1920x1080)
  --fps N              (default 30)
  --unload-after SEC   free the GPU this long after the last reader (default 60)
  --models DIR         ONNX/engine directory)";

[[noreturn]] void usage(const std::string& msg) {
    std::fprintf(stderr, "%s\nnvpoca: %s\n", kUsage, msg.c_str());
    std::exit(2);
}

double number(const char* s, const std::string& flag) {
    char* end;
    double v = std::strtod(s, &end);
    if (end == s || *end) usage(flag + ": not a number: " + s);
    return v;
}

Args parse(int argc, char** argv) {
    Args a;
    auto& s = a.settings;
    for (int i = 1; i < argc; ++i) {
        std::string f = argv[i];
        auto value = [&]() -> const char* {
            if (i + 1 >= argc) usage(f + " needs a value");
            return argv[++i];
        };
        auto optional = [&](double bare) {  // --smooth / --light take an optional number
            return i + 1 < argc && argv[i + 1][0] != '-' ? number(argv[++i], f) : bare;
        };
        if (f == "-h" || f == "--help") {
            std::puts(kUsage);
            std::exit(0);
        } else if (f == "--mode") {
            std::string m = value();
            if (m != "passthrough" && !valid_mode(m)) usage("--mode: " + m);
            s.enabled = m != "passthrough";
            s.mode = s.enabled ? m : "blur";
            a.explicit_.insert({"enabled", "mode"});
        } else if (f == "--bg") {
            s.bg = fs::absolute(expand_home(value()));
            a.explicit_.insert("bg");
        } else if (f == "--blur") {
            s.blur = number(value(), f);
            a.explicit_.insert("blur");
        } else if (f == "--smooth") {
            s.smooth = optional(0.15);
            a.explicit_.insert("smooth");
        } else if (f == "--no-smooth") {
            s.smooth = 0;
            a.explicit_.insert("smooth");
        } else if (f == "--light") {
            s.light = optional(0.5);
            a.explicit_.insert("light");
        } else if (f == "--variant") {
            a.variant = value();
            if (a.variant != "mobilenetv3" && a.variant != "resnet50") usage("--variant: " + a.variant);
        } else if (f == "--ratio") {
            a.ratio = number(value(), f);
            if (!(a.ratio > 0 && a.ratio <= 1)) usage("--ratio must be within (0, 1]");
        } else if (f == "--input") {
            a.input = value();
        } else if (f == "--output") {
            a.output = value();
        } else if (f == "--size") {
            if (std::sscanf(value(), "%dx%d", &a.w, &a.h) != 2 || a.w < 64 || a.h < 64 || a.w % 8 || a.h % 8)
                usage("--size must be WxH, multiples of 8");
        } else if (f == "--fps") {
            a.fps = static_cast<int>(number(value(), f));
            if (a.fps < 1) usage("--fps must be positive");
        } else if (f == "--unload-after") {
            a.unload_after = number(value(), f);
        } else if (f == "--models") {
            a.models = value();
        } else {
            usage("unknown option " + f);
        }
    }
    for (const char* key : {"blur", "smooth", "light"})
        if (auto err = invalid(key, key[0] == 'b' ? s.blur : key[0] == 's' ? s.smooth : s.light)) usage(*err);
    if (s.mode == "replace" && s.bg.empty()) usage("--mode replace needs --bg");
    if (auto err = invalid_background(s.bg)) usage(*err);
    if (a.ratio == 0) a.ratio = std::min(1.0, 480.0 / std::max(a.w, a.h));
    return a;
}

fs::path settings_path() {
    const char* xdg = std::getenv("XDG_CONFIG_HOME");
    return (xdg && *xdg ? fs::path(xdg) : fs::path(std::getenv("HOME")) / ".config") / "nvpoca" / "settings.conf";
}

// key=value lines; unknown keys and invalid values are ignored
void load_settings(Args& a) {
    std::ifstream f(settings_path());
    std::string line;
    auto& s = a.settings;
    while (std::getline(f, line)) {
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = line.substr(0, eq), v = line.substr(eq + 1);
        if (a.explicit_.contains(key)) continue;
        char* end;
        double d = std::strtod(v.c_str(), &end);
        bool num = !v.empty() && !*end && !invalid(key, d);
        if (key == "enabled" && (v == "true" || v == "false")) s.enabled = v == "true";
        else if (key == "mode" && valid_mode(v)) s.mode = v;
        else if (key == "blur" && num) s.blur = d;
        else if (key == "smooth" && num) s.smooth = d;
        else if (key == "light" && num) s.light = d;
        else if (key == "bg" && !invalid_background(v)) s.bg = v;
    }
}

void save_settings(const Settings& s) {
    auto path = settings_path(), tmp = path;
    tmp += ".tmp";
    std::error_code ec;
    fs::create_directories(path.parent_path(), ec);
    {
        std::ofstream f(tmp);
        f << "enabled=" << (s.enabled ? "true" : "false") << "\nmode=" << s.mode << "\nblur=" << s.blur
          << "\nsmooth=" << s.smooth << "\nlight=" << s.light << "\nbg=" << s.bg << "\n";
        if (!f) return log("cannot write %s", tmp.c_str());
    }
    fs::rename(tmp, path, ec);
    if (ec) log("cannot write %s: %s", path.c_str(), ec.message().c_str());
}

enum Prop : unsigned { kActive = 1, kLoaded = 2, kFps = 4, kLatency = 8 };

// shared between the main (D-Bus/event) thread and the GPU thread
struct State {
    std::mutex m;
    std::condition_variable cv;
    Settings settings;
    bool dirty = true;  // settings changed since the GPU thread applied them
    int readers = 0;
    bool unload = false, quit = false;
    // reported by the GPU thread
    bool loaded = false, active = false, failed = false;
    double fps = 0, latency = 0;
    unsigned changed = 0;
    int efd = -1;

    void report(unsigned props) {  // with m held; wakes the main thread to emit PropertiesChanged
        changed |= props;
        uint64_t one = 1;
        (void)!write(efd, &one, sizeof one);
    }
};

struct Daemon {
    Args args;
    State st;
    std::unique_ptr<Loopback> out;
    std::vector<uint8_t> black;
    fs::path onnx, engine;
    sd_event* ev = nullptr;
    sd_bus* bus = nullptr;
    sd_event_source* unload_timer = nullptr;
};

// ---- GPU thread ----

struct Applied {
    EffectParams params{Mode::blur, 4, 0, 0};
    std::optional<std::string> bg;  // nullopt = not loaded into this pipeline yet
};

void apply(Daemon& d, Pipeline& p, Applied& a) {
    Settings s;
    {
        std::lock_guard lk(d.st.m);
        if (!d.st.dirty) return;
        d.st.dirty = false;
        s = d.st.settings;
    }
    Mode mode = s.mode == "green" ? Mode::green : s.mode == "replace" ? Mode::replace : Mode::blur;
    a.params = {s.enabled ? mode : Mode::passthrough, static_cast<float>(s.blur * d.args.h / 1080),
                static_cast<float>(s.smooth), static_cast<float>(s.light)};
    if (a.bg != s.bg) {
        a.bg = s.bg;
        p.set_background(s.bg);
    }
}

void session(Daemon& d, Pipeline& p, Applied& a) {
    auto& st = d.st;
    bool opened = false;
    try {
        Camera cam(d.args.input, d.args.w, d.args.h, d.args.fps);
        opened = true;
        {
            std::lock_guard lk(st.m);
            st.active = true;
            st.report(kActive);
        }
        log("camera %s open", d.args.input.c_str());
        std::map<std::string, double> stats;
        int frames = 0, dropped = 0;
        double t = monotonic_now(), report_at = t + 1;
        for (;;) {
            {
                std::lock_guard lk(st.m);
                if (!st.readers || st.quit) break;
            }
            apply(d, p, a);
            Frame f;
            if (!cam.next(f, 1000)) continue;
            double t1 = monotonic_now();
            if (!p.decode(f.data, f.size)) {  // corrupt MJPEG frames happen on USB hiccups
                cam.requeue(f);
                ++dropped;
                t = monotonic_now();
                continue;
            }
            double t2 = monotonic_now();
            p.process(a.params);
            cam.requeue(f);
            double t3 = monotonic_now();
            d.out->write(p.yuyv(), p.yuyv_size());
            double t4 = monotonic_now();
            stats["capture"] += t1 - t;
            stats["decode"] += t2 - t1;
            stats["gpu"] += t3 - t2;
            stats["write"] += t4 - t3;
            // frame timestamp is CLOCK_MONOTONIC at USB start-of-frame -> true capture-to-output latency
            stats["latency"] += t4 - f.timestamp;
            ++frames;
            t = t4;
            if (t4 >= report_at) {
                report_at = t4 + 1;
                std::string line;
                char buf[32];
                for (auto& [k, v] : stats) {
                    std::snprintf(buf, sizeof buf, " %s=%.1f", k.c_str(), v / frames * 1000);
                    line += buf;
                }
                log("fps=%2d drop=%d ms:%s", frames, dropped, line.c_str());
                {
                    std::lock_guard lk(st.m);
                    st.fps = frames;
                    st.latency = stats["latency"] / frames * 1000;
                    st.report(kFps | kLatency);
                }
                stats.clear();
                frames = dropped = 0;
            }
        }
    } catch (const std::system_error& e) {
        // e.g. the browser still holds the real camera while switching to ours; readers get black meanwhile
        log("%s%s, retrying", e.what(), e.code().value() == EBUSY ? " (another app has the camera open)" : "");
        std::unique_lock lk(st.m);
        st.cv.wait_for(lk, std::chrono::seconds(1), [&] { return !st.readers || st.quit; });
    }
    p.reset();
    d.out->write(d.black.data(), d.black.size());  // don't leave a reader on the last processed frame
    if (opened) {
        std::lock_guard lk(st.m);
        st.active = false;
        st.fps = st.latency = 0;
        st.report(kActive | kFps | kLatency);
        log("camera closed");
    }
}

void gpu_main(Daemon& d) {
    auto& st = d.st;
    std::unique_ptr<Pipeline> p;
    Applied applied;
    try {
        for (;;) {
            {
                std::unique_lock lk(st.m);
                st.cv.wait(lk, [&] { return st.quit || st.readers > 0 || (p && st.unload); });
                if (st.quit) break;
                if (!st.readers) {
                    st.unload = false;
                    lk.unlock();
                    log("no reader for %g s, freeing the GPU", d.args.unload_after);
                    p.reset();
                    cudaDeviceReset();  // the CUDA context holds ~300 MB of VRAM on its own
                    lk.lock();
                    st.loaded = false;
                    st.report(kLoaded);
                    continue;
                }
            }
            if (!p) {
                double t = monotonic_now();
                p = std::make_unique<Pipeline>(d.onnx, d.engine, d.args.w, d.args.h);
                applied = {};
                log("GPU ready in %.2f s", monotonic_now() - t);
                std::lock_guard lk(st.m);
                st.dirty = st.loaded = true;
                st.report(kLoaded);
            }
            session(d, *p, applied);
        }
    } catch (const std::exception& e) {
        log("fatal: %s", e.what());
        std::lock_guard lk(st.m);
        st.failed = true;
        st.report(0);
    }
}

// ---- main thread: D-Bus and events ----

int get_prop(sd_bus*, const char*, const char*, const char* prop, sd_bus_message* reply, void* ud, sd_bus_error*) {
    auto& st = static_cast<Daemon*>(ud)->st;
    std::lock_guard lk(st.m);
    std::string_view p = prop;
    const auto& s = st.settings;
    if (p == "Enabled") return sd_bus_message_append(reply, "b", s.enabled);
    if (p == "Mode") return sd_bus_message_append(reply, "s", s.mode.c_str());
    if (p == "Blur") return sd_bus_message_append(reply, "d", s.blur);
    if (p == "Smooth") return sd_bus_message_append(reply, "d", s.smooth);
    if (p == "Light") return sd_bus_message_append(reply, "d", s.light);
    if (p == "Background") return sd_bus_message_append(reply, "s", s.bg.c_str());
    if (p == "Active") return sd_bus_message_append(reply, "b", st.active);
    if (p == "Loaded") return sd_bus_message_append(reply, "b", st.loaded);
    if (p == "Fps") return sd_bus_message_append(reply, "d", st.fps);
    return sd_bus_message_append(reply, "d", st.latency);
}

int set_prop(sd_bus* bus, const char*, const char*, const char* prop, sd_bus_message* value, void* ud, sd_bus_error* err) {
    auto& d = *static_cast<Daemon*>(ud);
    std::string_view p = prop;
    Settings s;
    {
        std::lock_guard lk(d.st.m);
        s = d.st.settings;
    }
    int r;
    if (p == "Enabled") {
        int b;
        if ((r = sd_bus_message_read(value, "b", &b)) < 0) return r;
        s.enabled = b;
    } else if (p == "Mode" || p == "Background") {
        const char* v;
        if ((r = sd_bus_message_read(value, "s", &v)) < 0) return r;
        if (p == "Mode") {
            if (!valid_mode(v)) return sd_bus_error_set(err, SD_BUS_ERROR_INVALID_ARGS, "Mode must be 'blur', 'replace' or 'green'");
            s.mode = v;
        } else {
            std::string path = *v ? fs::absolute(expand_home(v)).string() : "";
            if (auto e = invalid_background(path)) return sd_bus_error_set(err, SD_BUS_ERROR_INVALID_ARGS, e->c_str());
            s.bg = path;
        }
    } else {
        double v;
        if ((r = sd_bus_message_read(value, "d", &v)) < 0) return r;
        std::string key = p == "Blur" ? "blur" : p == "Smooth" ? "smooth" : "light";
        if (auto e = invalid(key, v)) return sd_bus_error_set(err, SD_BUS_ERROR_INVALID_ARGS, e->c_str());
        (key == "blur" ? s.blur : key == "smooth" ? s.smooth : s.light) = v;
    }
    {
        std::lock_guard lk(d.st.m);
        d.st.settings = s;
        d.st.dirty = true;
    }
    save_settings(s);
    sd_bus_emit_properties_changed(bus, kPath, kBusName, prop, nullptr);
    return 1;
}

constexpr uint64_t kEmits = SD_BUS_VTABLE_PROPERTY_EMITS_CHANGE;
const sd_bus_vtable kVtable[] = {
    SD_BUS_VTABLE_START(0),
    SD_BUS_WRITABLE_PROPERTY("Enabled", "b", get_prop, set_prop, 0, kEmits),
    SD_BUS_WRITABLE_PROPERTY("Mode", "s", get_prop, set_prop, 0, kEmits),
    SD_BUS_WRITABLE_PROPERTY("Blur", "d", get_prop, set_prop, 0, kEmits),
    SD_BUS_WRITABLE_PROPERTY("Smooth", "d", get_prop, set_prop, 0, kEmits),
    SD_BUS_WRITABLE_PROPERTY("Light", "d", get_prop, set_prop, 0, kEmits),
    SD_BUS_WRITABLE_PROPERTY("Background", "s", get_prop, set_prop, 0, kEmits),
    SD_BUS_PROPERTY("Active", "b", get_prop, 0, kEmits),  // the real camera is open because an app reads ours
    SD_BUS_PROPERTY("Loaded", "b", get_prop, 0, kEmits),  // models in VRAM
    SD_BUS_PROPERTY("Fps", "d", get_prop, 0, kEmits),
    SD_BUS_PROPERTY("Latency", "d", get_prop, 0, kEmits),  // capture-to-output, ms
    SD_BUS_VTABLE_END,
};

int on_unload(sd_event_source*, uint64_t, void* ud) {
    auto& st = static_cast<Daemon*>(ud)->st;
    std::lock_guard lk(st.m);
    st.unload = true;
    st.cv.notify_all();
    return 0;
}

int on_readers(sd_event_source*, int, uint32_t, void* ud) {
    auto& d = *static_cast<Daemon*>(ud);
    int n;
    try {
        n = d.out->read_readers();
        // a reader's S_PARM sticks to the device (Chromium sets 10000/300000), and PipeWire then
        // fails to match a client's 30/1 ("no more input formats")
        if (!n) d.out->set_fps(d.args.fps);
    } catch (const std::system_error& e) {
        log("%s", e.what());
        return 0;
    }
    {
        std::lock_guard lk(d.st.m);
        d.st.readers = n;
        if (n) d.st.unload = false;
        d.st.cv.notify_all();
    }
    uint64_t usec = static_cast<uint64_t>(d.args.unload_after * 1e6);
    if (n) {
        if (d.unload_timer) sd_event_source_set_enabled(d.unload_timer, SD_EVENT_OFF);
    } else if (!d.unload_timer) {
        sd_event_add_time_relative(d.ev, &d.unload_timer, CLOCK_MONOTONIC, usec, 0, on_unload, &d);
    } else {
        sd_event_source_set_time_relative(d.unload_timer, usec);
        sd_event_source_set_enabled(d.unload_timer, SD_EVENT_ONESHOT);
    }
    return 0;
}

int on_report(sd_event_source*, int fd, uint32_t, void* ud) {
    auto& d = *static_cast<Daemon*>(ud);
    uint64_t n;
    (void)!read(fd, &n, sizeof n);
    unsigned changed;
    bool failed;
    {
        std::lock_guard lk(d.st.m);
        changed = std::exchange(d.st.changed, 0);
        failed = d.st.failed;
    }
    std::vector<const char*> names;
    for (auto [bit, name] : {std::pair{kActive, "Active"}, {kLoaded, "Loaded"}, {kFps, "Fps"}, {kLatency, "Latency"}})
        if (changed & bit) names.push_back(name);
    if (!names.empty()) {
        names.push_back(nullptr);
        sd_bus_emit_properties_changed_strv(d.bus, kPath, kBusName, const_cast<char**>(names.data()));
    }
    if (failed) sd_event_exit(d.ev, EXIT_FAILURE);  // systemd restarts us
    return 0;
}

int fail(const char* what, int r) {
    log("%s: %s", what, std::strerror(-r));
    return EXIT_FAILURE;
}

}  // namespace

int main(int argc, char** argv) {
    Daemon d;
    d.args = parse(argc, argv);
    load_settings(d.args);
    d.st.settings = d.args.settings;
    auto& a = d.args;

    char name[128];
    std::snprintf(name, sizeof name, "rvm_%s_%dx%d_r%.3f_aa", a.variant.c_str(), a.w, a.h, a.ratio);
    d.onnx = a.models / (std::string(name) + ".onnx");
    // an engine only deserializes in the TensorRT version that built it; after an upgrade, build anew
    d.engine = a.models / (std::string(name) + "_trt" + std::to_string(NV_TENSORRT_MAJOR) + "." +
                           std::to_string(NV_TENSORRT_MINOR) + "." + std::to_string(NV_TENSORRT_PATCH) + "." +
                           std::to_string(NV_TENSORRT_BUILD) + ".engine");
    if (!fs::exists(d.engine) && !fs::exists(d.onnx)) {
        log("missing %s; export it once with tools/export_onnx.py --size %dx%d", d.onnx.c_str(), a.w, a.h);
        return EXIT_FAILURE;
    }

    try {
        d.out = std::make_unique<Loopback>(a.output, a.w, a.h, a.fps);
        d.black.resize(static_cast<size_t>(a.w) * a.h * 2);
        for (size_t i = 0; i < d.black.size(); i += 2) d.black[i] = 16, d.black[i + 1] = 128;  // YUYV black
        d.out->write(d.black.data(), d.black.size());  // a frame up front so a new reader gets an image while the GPU loads
        d.out->subscribe_readers();
    } catch (const std::system_error& e) {
        log("%s", e.what());
        return EXIT_FAILURE;
    }

    sigset_t mask;
    sigemptyset(&mask);
    sigaddset(&mask, SIGTERM);
    sigaddset(&mask, SIGINT);
    pthread_sigmask(SIG_BLOCK, &mask, nullptr);  // before the GPU thread starts, so it inherits the mask
    d.st.efd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);

    int r;
    if ((r = sd_event_default(&d.ev)) < 0) return fail("sd_event_default", r);
    sd_event_add_signal(d.ev, nullptr, SIGTERM, nullptr, nullptr);  // default handler: leave the loop
    sd_event_add_signal(d.ev, nullptr, SIGINT, nullptr, nullptr);
    if ((r = sd_bus_open_user(&d.bus)) < 0) return fail("session bus", r);
    if ((r = sd_bus_add_object_vtable(d.bus, nullptr, kPath, kBusName, kVtable, &d)) < 0) return fail("vtable", r);
    if ((r = sd_bus_request_name(d.bus, kBusName, 0)) < 0) {
        log("%s is already taken: is another nvpoca running?", kBusName);
        return EXIT_FAILURE;
    }
    if ((r = sd_bus_attach_event(d.bus, d.ev, 0)) < 0) return fail("sd_bus_attach_event", r);
    if ((r = sd_event_add_io(d.ev, nullptr, d.out->fd(), EPOLLPRI, on_readers, &d)) < 0) return fail("loopback events", r);
    if ((r = sd_event_add_io(d.ev, nullptr, d.st.efd, EPOLLIN, on_report, &d)) < 0) return fail("eventfd", r);

    std::thread gpu(gpu_main, std::ref(d));
    log("ready: %s %dx%d YUYV; the GPU loads with the first reader, frees %g s after the last", a.output.c_str(), a.w,
        a.h, a.unload_after);
    r = sd_event_loop(d.ev);
    {
        std::lock_guard lk(d.st.m);
        d.st.quit = true;
        d.st.cv.notify_all();
    }
    gpu.join();
    sd_bus_flush_close_unref(d.bus);
    sd_event_unref(d.ev);
    return r < 0 ? EXIT_FAILURE : r;
}
