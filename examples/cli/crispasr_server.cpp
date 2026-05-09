// crispasr_server.cpp — HTTP server with persistent model for all backends.
//
// Keeps the model loaded in memory between requests. Accepts audio via
// POST /inference (multipart file upload) and returns JSON transcription.
//
// Usage:
//   crispasr --server -m model.gguf [--port 8080] [--host 127.0.0.1]
//
// Endpoints:
//   POST /inference                   — transcribe (native JSON)
//   POST /v1/audio/transcriptions     — OpenAI-compatible endpoint
//   POST /v1/audio/speech             — TTS (OpenAI-compatible; CAP_TTS only)
//   POST /load                        — hot-swap model
//   GET  /health                      — server status
//   GET  /backends                    — list available backends
//   GET  /v1/models                   — OpenAI-compatible model list
//   GET  /v1/voices                   — list voices in --voice-dir (CAP_TTS only)
//
// Adapted from examples/server/server.cpp for multi-backend support.

#include "crispasr_backend.h"
#include "crispasr_lid.h"
#include "crispasr_lid_cli.h"
#include "crispasr_output.h"
#include "crispasr_model_mgr_cli.h"
#include "crispasr_vad_cli.h"
#include "whisper_params.h"

#include "common-crispasr.h" // read_audio_data
#include "crispasr_tts_chunking.h"
#include "crispasr_wav_writer.h"
#include "crispasr_worker_session.h" // PARAKEET_WORKER_ISOLATION subprocess model
#include "../server/httplib.h"
#include "../json.hpp"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <io.h> // _mktemp_s
#include <windows.h>
#else
#include <unistd.h> // mkstemp, close, unlink
#endif

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Create a temporary file securely via mkstemp (POSIX) or _mktemp_s (Win).
// Writes `data` to it and returns the path. On failure returns "".
// The caller is responsible for calling std::remove() on the returned path.
static std::string write_temp_audio(const char* data, size_t size) {
#ifdef _WIN32
    char tmp_dir[MAX_PATH];
    if (!GetTempPathA(MAX_PATH, tmp_dir))
        return "";
    char tmp_path[MAX_PATH];
    if (!GetTempFileNameA(tmp_dir, "cra", 0, tmp_path))
        return "";
    std::ofstream f(tmp_path, std::ios::binary);
    if (!f)
        return "";
    f.write(data, (std::streamsize)size);
    f.close();
    return std::string(tmp_path);
#else
    char tmpl[] = "/tmp/crispasr-XXXXXX";
    int fd = mkstemp(tmpl);
    if (fd < 0)
        return "";
    // Write all data; retry on partial write.
    const char* p = data;
    size_t remaining = size;
    while (remaining > 0) {
        ssize_t n = ::write(fd, p, remaining);
        if (n < 0) {
            if (errno == EINTR)
                continue;
            ::close(fd);
            ::unlink(tmpl);
            return "";
        }
        p += n;
        remaining -= (size_t)n;
    }
    ::close(fd);
    return std::string(tmpl);
#endif
}

// Read a form field as a trimmed string, or return a default.
static std::string form_string(const httplib::Request& req, const std::string& key, const std::string& def = "") {
    std::string v;
    if (req.has_file(key)) {
        v = req.get_file_value(key).content;
    } else if (req.has_param(key)) {
        v = req.get_param_value(key);
    } else {
        return def;
    }
    // Trim whitespace.
    while (!v.empty() && (v.front() == ' ' || v.front() == '\t'))
        v.erase(v.begin());
    while (!v.empty() && (v.back() == ' ' || v.back() == '\t'))
        v.pop_back();
    return v.empty() ? def : v;
}

static std::string trim_copy(std::string v) {
    while (!v.empty() && (v.front() == ' ' || v.front() == '\t' || v.front() == '\r' || v.front() == '\n'))
        v.erase(v.begin());
    while (!v.empty() && (v.back() == ' ' || v.back() == '\t' || v.back() == '\r' || v.back() == '\n'))
        v.pop_back();
    return v;
}

static std::vector<std::string> split_api_keys(const std::string& csv) {
    std::vector<std::string> keys;
    std::stringstream ss(csv);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item = trim_copy(item);
        if (!item.empty())
            keys.push_back(item);
    }
    return keys;
}

static bool fixed_time_equal(const std::string& a, const std::string& b) {
    unsigned char diff = (unsigned char)(a.size() ^ b.size());
    const size_t n = a.size() < b.size() ? a.size() : b.size();
    for (size_t i = 0; i < n; ++i)
        diff |= (unsigned char)(a[i] ^ b[i]);
    return diff == 0 && a.size() == b.size();
}

static std::string request_api_key(const httplib::Request& req) {
    if (req.has_header("Authorization")) {
        const std::string value = trim_copy(req.get_header_value("Authorization"));
        const std::string prefix = "Bearer ";
        if (value.rfind(prefix, 0) == 0)
            return trim_copy(value.substr(prefix.size()));
    }
    if (req.has_header("X-API-Key"))
        return trim_copy(req.get_header_value("X-API-Key"));
    return "";
}

static bool is_authorized(const httplib::Request& req, const std::vector<std::string>& api_keys) {
    if (api_keys.empty())
        return true;
    const std::string key = request_api_key(req);
    if (key.empty())
        return false;
    for (const std::string& expected : api_keys)
        if (fixed_time_equal(key, expected))
            return true;
    return false;
}

// Parse a form field as float, returning `def` on missing or parse error.
static float form_float(const httplib::Request& req, const std::string& key, float def) {
    if (!req.has_file(key) && !req.has_param(key))
        return def;
    const std::string v = req.has_file(key) ? req.get_file_value(key).content : req.get_param_value(key);
    try {
        size_t pos = 0;
        float f = std::stof(v, &pos);
        // Reject trailing garbage like "0.5abc".
        if (pos != v.size())
            return def;
        return f;
    } catch (...) {
        return def;
    }
}

// JSON error response helper. Shape matches OpenAI's:
//   { "error": { "message": ..., "type": ..., "code": ..., "param": ... } }
// `code` is a stable machine-readable enum-string the client can switch on
// (e.g. "voice_not_found", "input_too_long"); `param` is the offending
// request field name (e.g. "voice", "input"). Both default to "" and are
// omitted from the JSON body when empty so the on-wire shape stays
// minimal for non-OpenAI consumers.
static void json_error(httplib::Response& res, int status, const std::string& message, const std::string& code = "",
                       const std::string& param = "") {
    res.status = status;
    std::string body =
        "{\"error\": {\"message\": \"" + crispasr_json_escape(message) + "\", \"type\": \"invalid_request_error\"";
    if (!code.empty())
        body += ", \"code\": \"" + crispasr_json_escape(code) + "\"";
    if (!param.empty())
        body += ", \"param\": \"" + crispasr_json_escape(param) + "\"";
    body += "}}";
    res.set_content(body, "application/json");
}

static void auth_error(httplib::Response& res) {
    res.status = 401;
    res.set_header("WWW-Authenticate", "Bearer");
    res.set_content("{\"error\": {\"message\": \"invalid or missing API key\", \"type\": \"invalid_api_key\"}}",
                    "application/json");
}

// Shared transcription result.
struct transcription_result {
    bool ok = false;
    std::string error;
    std::vector<crispasr_segment> segs;
    std::string language;
    double duration_s = 0.0;
    double elapsed_s = 0.0;
    bool preempted = false; // true if cancel_url signalled mid-transcription
};

// Poll an external cancel URL between chunks; treats unreachable / malformed
// URL as "not cancelled" so a flaky signal endpoint can never block STT.
// koblem signals between-chunk preemption when a higher-priority service
// (TTS/vision) needs the GPU — see api/src/stt.rs::apply_params().
static bool check_cancel(const std::string& cancel_url) {
    if (cancel_url.empty())
        return false;
    // Parse "http(s)://host:port/path" — httplib::Client wants scheme+host
    // separately from path. Cheap manual split (no URL escaping inside path
    // expected from koblem's signed-token format).
    auto scheme_end = cancel_url.find("://");
    if (scheme_end == std::string::npos)
        return false;
    auto host_start = scheme_end + 3;
    auto path_start = cancel_url.find('/', host_start);
    std::string host = (path_start == std::string::npos) ? cancel_url.substr(0, std::string::npos)
                                                         : cancel_url.substr(0, path_start);
    std::string path = (path_start == std::string::npos) ? "/" : cancel_url.substr(path_start);
    httplib::Client cli(host);
    cli.set_connection_timeout(2);
    cli.set_read_timeout(2);
    auto r = cli.Get(path);
    if (!r || r->status != 200)
        return false;
    try {
        auto body = nlohmann::json::parse(r->body);
        return body.value("cancelled", false);
    } catch (...) {
        return false;
    }
}

// Decode audio from a path on disk and transcribe it. Caller is responsible
// for placing audio at `audio_path` (either a real file_path or a temp file
// written from a multipart upload). Acquires model_mutex internally.
//
// Between chunks the loop polls cancel_url; on a cancel signal it sets
// preempted=true and returns whatever has transcribed so far. Useful when
// a higher-priority GPU consumer (TTS, vision) preempts the STT job and we
// don't want to discard partial work.
//
// Exactly one of `backend` / `worker` is set. `worker`-mode forwards the
// transcribe call over IPC to a subprocess that owns the CUDA context;
// the chunking + cancel polling stays here so /unload (SIGKILL of the
// worker) reclaims VRAM the same way regardless of mode.
static transcription_result do_transcribe(const std::string& audio_path, CrispasrBackend* backend,
                                          crispasr::WorkerSession* worker, std::mutex& model_mutex,
                                          whisper_params rp, const std::string& cancel_url) {
    transcription_result result;
    result.language = rp.language;

    if (rp.verbose)
        fprintf(stderr, "crispasr-server: processing '%s'\n", audio_path.c_str());

    // Decode audio directly from the provided path.
    std::vector<float> pcmf32;
    std::vector<std::vector<float>> pcmf32s;
    if (!read_audio_data(audio_path, pcmf32, pcmf32s, rp.diarize)) {
        result.error = "failed to decode audio (unsupported format or corrupt file)";
        return result;
    }

    if (pcmf32.empty()) {
        result.error = "audio file contains no samples";
        return result;
    }

    result.duration_s = (double)pcmf32.size() / 16000.0;

    // Auto-chunk long audio to prevent OOM (#27).
    // Most backends have O(T²) attention in the encoder — 30s chunks keep
    // memory bounded. The CLI does this via --vad / --chunk-seconds.
    const int SR = 16000;
    const int max_chunk_samples = rp.chunk_seconds * SR; // default 30s = 480000
    const int n_samples = (int)pcmf32.size();

    // run_one returns segments + a worker-died flag. In worker mode, after
    // each call we sample worker->is_alive() to distinguish "real empty
    // result" (silent chunk) from "worker died and the call returned {}".
    // Without this distinction the chunked path used to silently swallow
    // worker death and return a partial transcript flagged ok=true.
    auto run_one = [&](const float* p, int n, int64_t t_off_cs, bool* worker_died) {
        std::vector<crispasr_segment> segs =
            worker ? worker->transcribe(p, n, t_off_cs, rp) : backend->transcribe(p, n, t_off_cs, rp);
        *worker_died = worker && !worker->is_alive();
        return segs;
    };

    {
        std::lock_guard<std::mutex> lock(model_mutex);
        auto t0 = std::chrono::steady_clock::now();

        bool worker_died = false;
        if (n_samples <= max_chunk_samples) {
            // Short audio — single pass; no cancel checkpoint mid-chunk
            // (the encoder owns the GPU for the duration of one transcribe
            // call and there's no cooperative preemption inside it).
            result.segs = run_one(pcmf32.data(), n_samples, 0, &worker_died);
            if (worker_died) {
                result.error = "worker died mid-transcribe: " + worker->last_error();
                result.elapsed_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                return result;
            }
        } else {
            // Chunk long audio into fixed segments. Between chunks we yield
            // control long enough to poll cancel_url; if koblem signals
            // preemption we keep the partial transcript and bail.
            int n_chunks = (n_samples + max_chunk_samples - 1) / max_chunk_samples;
            fprintf(stderr, "crispasr-server: chunking %.1fs audio into %d × %ds segments\n", result.duration_s,
                    n_chunks, rp.chunk_seconds);
            int chunk_idx = 0;
            for (int offset = 0; offset < n_samples; offset += max_chunk_samples) {
                if (check_cancel(cancel_url)) {
                    fprintf(stderr, "crispasr-server: cancel_url signalled preemption at chunk %d/%d\n", chunk_idx,
                            n_chunks);
                    result.preempted = true;
                    break;
                }
                int chunk_len = std::min(max_chunk_samples, n_samples - offset);
                int64_t t_offset_cs = (int64_t)((double)offset / SR * 100.0);
                auto tc0 = std::chrono::steady_clock::now();
                auto chunk_segs = run_one(pcmf32.data() + offset, chunk_len, t_offset_cs, &worker_died);
                auto tc1 = std::chrono::steady_clock::now();
                double chunk_s = std::chrono::duration<double>(tc1 - tc0).count();
                result.segs.insert(result.segs.end(), chunk_segs.begin(), chunk_segs.end());
                chunk_idx++;
                fprintf(stderr, "crispasr-server: chunk %d/%d done (%.1fs audio in %.1fs)%s\n", chunk_idx, n_chunks,
                        chunk_len / (double)SR, chunk_s, worker_died ? " [WORKER DIED]" : "");
                if (worker_died) {
                    result.error = "worker died at chunk " + std::to_string(chunk_idx) + "/" +
                                   std::to_string(n_chunks) + ": " + worker->last_error();
                    result.elapsed_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                    return result;
                }
            }
        }

        auto t1 = std::chrono::steady_clock::now();
        result.elapsed_s = std::chrono::duration<double>(t1 - t0).count();
    }

    result.ok = true;
    return result;
}

// crispasr_make_wav_int16 lives in crispasr_wav_writer.h so the unit
// tests can exercise it without linking the server translation unit.

// ---------------------------------------------------------------------------
// Server entry point
// ---------------------------------------------------------------------------

int crispasr_run_server(whisper_params& params, const std::string& host, int port) {
    using namespace httplib;

    std::vector<std::string> api_keys = split_api_keys(params.server_api_keys);
    if (const char* env_keys = getenv("CRISPASR_API_KEYS")) {
        std::vector<std::string> more = split_api_keys(env_keys);
        api_keys.insert(api_keys.end(), more.begin(), more.end());
    }

    std::unique_ptr<CrispasrBackend> backend;
    std::mutex model_mutex;
    std::atomic<bool> ready{false};
    // model_loaded distinguishes "we have a backend instance holding GPU
    // memory" from ready (which goes false during /load swaps as well).
    // /unload sets both to false; lazy-load enters service with both false.
    std::atomic<bool> model_loaded{false};
    std::atomic<int64_t> last_request_ms{0};
    std::string backend_name = params.backend;

    // Worker-isolation: when PARAKEET_WORKER_ISOLATION=1, the parent process
    // never touches CUDA. A subprocess holds the model + ggml-cuda primary
    // context; on /unload we SIGKILL it and the entire context (cuBLAS
    // workspace, cubin/PTX cache, driver state) goes with it — that's the
    // only way to return nvidia-smi to the pre-load floor without exiting
    // the parent. Mirrors qwen3-tts.cpp QWEN3_TTS_WORKER_ISOLATION semantics.
    std::unique_ptr<crispasr::WorkerSession> worker;
    bool worker_iso = false;
    if (const char* env = std::getenv("PARAKEET_WORKER_ISOLATION")) {
        worker_iso = env[0] && env[0] != '0';
    }
    // Forward CLI args (sans `--worker`) so the child sees the same model /
    // backend / -t / etc. options. Note: the parent's argv isn't visible
    // here; the child re-runs cli.cpp main() and re-parses these. We can't
    // round-trip every CLI flag without argv, so the worker leans entirely
    // on LOAD_REQ params for its config — extra_argv is empty. The worker
    // mode in cli.cpp branches before whisper_params_parse, so flags we
    // don't propagate are simply skipped (params come over LOAD_REQ).
    std::vector<std::string> worker_extra_argv;
    if (worker_iso) {
        // argv[0] for the spawn must point at the running binary on disk.
        // /proc/self/exe is the canonical Linux source; readlink at runtime
        // because $0 / argv[0] aren't reliable when invoked through a
        // shell wrapper (Dockerfile uses `/bin/sh -c "exec /usr/local/bin/crispasr ..."`).
        char self_path[4096] = {0};
        ssize_t n = -1;
#if !defined(_WIN32)
        n = ::readlink("/proc/self/exe", self_path, sizeof(self_path) - 1);
#endif
        if (n <= 0) {
            // Fallback: hard-code the docker entrypoint path. If it doesn't
            // exist the worker fails-soft on next /inference (no isolation).
            std::strncpy(self_path, "/usr/local/bin/crispasr", sizeof(self_path) - 1);
        }
        worker = std::make_unique<crispasr::WorkerSession>(self_path, worker_extra_argv);
        fprintf(stderr, "crispasr-server: PARAKEET_WORKER_ISOLATION=1 — model loads in subprocess (argv0=%s)\n",
                self_path);
    }

    // Ecosystem env knobs — keep load semantics matching the python parakeet-stt
    // service (kobbler/docker/parakeet-stt/) so koblem's GPU gate sees the same
    // load/idle behavior on either backend.
    const bool lazy_load = []() {
        const char* v = getenv("PARAKEET_LAZY_LOAD");
        if (!v)
            return false;
        std::string s = v;
        return s == "1" || s == "true" || s == "TRUE" || s == "yes" || s == "YES";
    }();
    const int idle_unload_seconds = []() {
        const char* v = getenv("PARAKEET_IDLE_UNLOAD_SECONDS");
        if (!v || !*v)
            return 300;
        try {
            return std::stoi(v);
        } catch (...) {
            return 300;
        }
    }();

    // Resolve the model + backend name without committing GPU memory.
    // Splitting resolve from init lets lazy mode advertise the model via
    // /v1/models even when nothing is loaded yet.
    auto resolve_model_into_params = [&](whisper_params& p, std::string& out_backend_name) -> bool {
        const bool model_is_auto = p.model == "auto" || p.model == "default";
        if (out_backend_name.empty() || out_backend_name == "auto") {
            if (model_is_auto) {
                out_backend_name = "whisper";
                if (!p.no_prints)
                    fprintf(stderr, "crispasr-server: -m auto with no backend — defaulting to whisper\n");
            } else {
                out_backend_name = crispasr_detect_backend_from_gguf(p.model);
            }
        }
        if (out_backend_name.empty()) {
            fprintf(stderr, "crispasr-server: cannot detect backend from '%s'\n", p.model.c_str());
            return false;
        }
        const std::string resolved = crispasr_resolve_model_cli(p.model, out_backend_name, p.no_prints, p.cache_dir,
                                                                p.auto_download || model_is_auto, p.model_quant);
        if (resolved.empty()) {
            fprintf(stderr, "crispasr-server: failed to resolve model '%s' for backend '%s'\n", p.model.c_str(),
                    out_backend_name.c_str());
            return false;
        }
        p.model = resolved;
        return true;
    };

    auto build_worker_cfg = [&]() {
        crispasr::WorkerLoadConfig cfg;
        cfg.backend = backend_name;
        cfg.model = params.model;
        cfg.model_quant = params.model_quant;
        cfg.verbose = params.verbose;
        cfg.use_gpu = params.use_gpu;
        cfg.gpu_device = params.gpu_device;
        cfg.flash_attn = params.flash_attn;
        cfg.init_params = params;
        return cfg;
    };

    // Materialize the backend (allocates GPU memory). Caller holds model_mutex.
    // Builds a fresh backend before dropping any existing one so a failed
    // load on hot-swap leaves the prior model in place. The `ready` flag
    // is only flipped during the brief window where backend is being
    // assigned; new requests block on model_mutex through that window.
    //
    // In worker-isolation mode this path runs ensure_loaded() on the worker
    // session, which spawns the subprocess (or kills+respawns if cfg
    // changed) and sends LOAD_REQ. The parent stays CUDA-free.
    auto load_backend_locked = [&]() -> bool {
        if (worker) {
            ready.store(false);
            if (!worker->ensure_loaded(build_worker_cfg())) {
                fprintf(stderr, "crispasr-server: worker LOAD_REQ failed: %s\n", worker->last_error().c_str());
                return false;
            }
            model_loaded.store(true);
            ready.store(true);
            fprintf(stderr, "crispasr-server: worker pid=%d loaded backend '%s', model '%s'\n", (int)worker->pid(),
                    worker->backend_name().c_str(), params.model.c_str());
            return true;
        }
        auto nb = crispasr_create_backend(backend_name);
        if (!nb || !nb->init(params)) {
            fprintf(stderr, "crispasr-server: failed to init backend '%s'\n", backend_name.c_str());
            return false;
        }
        ready.store(false);
        backend = std::move(nb); // dtor of previous backend frees prior GPU memory
        model_loaded.store(true);
        ready.store(true);
        fprintf(stderr, "crispasr-server: backend '%s' loaded, model '%s'\n", backend_name.c_str(),
                params.model.c_str());
        return true;
    };

    // Tear down the backend and free GPU memory. Caller holds model_mutex.
    //
    // - In-process mode: CrispasrBackend's dtor calls shutdown() / the
    //   backend's free routine, releasing the explicit CUDA allocations.
    //   The CUDA primary context (cuBLAS workspace, cubin/PTX cache, driver
    //   runtime state — 100-400 MiB) sticks until this process exits;
    //   that's the residual visible on nvidia-smi after /unload.
    // - Worker-isolation mode: SIGKILL the subprocess. The kernel reaps
    //   the entire CUDA context with the process; nvidia-smi returns to
    //   the pre-load floor instantly.
    auto free_backend_locked = [&]() {
        ready.store(false);
        if (worker) {
            worker->shutdown();
        } else {
            backend.reset();
        }
        model_loaded.store(false);
        fprintf(stderr, "crispasr-server: backend unloaded%s\n", worker ? " (worker SIGKILLed)" : "");
    };

    if (!resolve_model_into_params(params, backend_name))
        return 1;

    if (lazy_load) {
        fprintf(stderr, "crispasr-server: PARAKEET_LAZY_LOAD=1 — deferring model load until first request\n");
    } else {
        std::lock_guard<std::mutex> lock(model_mutex);
        if (!load_backend_locked())
            return 1;
    }

    auto touch_last_request = [&]() {
        last_request_ms.store(
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch())
                .count());
    };

    // Ensure backend is loaded (lazy path). Returns true if ready to serve.
    // Sets a 503 on `res` and returns false on failure so callers can early-out.
    //
    // Worker mode caveat: model_loaded is a parent-side hint; if the worker
    // dies out-of-band (SIGKILL, SEGV, OOM) the hint stays true. So in
    // worker mode we additionally check worker->is_alive() (which actively
    // reaps a dead child) and force a respawn. Without this, the next
    // request would dispatch to a stale fd and silently return empty.
    auto ensure_loaded = [&](Response& res) -> bool {
        const bool worker_dead = worker && !worker->is_alive();
        if (model_loaded.load() && ready.load() && !worker_dead)
            return true;
        if (!lazy_load && !worker_dead) {
            // Non-lazy server: model is genuinely still loading or failed.
            json_error(res, 503, "model is still loading");
            return false;
        }
        std::lock_guard<std::mutex> lock(model_mutex);
        // Re-check under lock — another request may have respawned.
        if (model_loaded.load() && ready.load() && (!worker || worker->is_alive()))
            return true;
        if (worker && !worker->is_alive()) {
            // Force a clean respawn: model_loaded is stale.
            model_loaded.store(false);
            ready.store(false);
        }
        if (!load_backend_locked()) {
            json_error(res, 503, "failed to load model");
            return false;
        }
        return true;
    };

    Server svr;

    // CORS support — opt-in via --cors-origin. Browser clients calling our
    // /v1/* endpoints from a different origin need:
    //   1. Access-Control-Allow-Origin on every response (set on each route)
    //   2. A 204 reply to OPTIONS preflights with Allow-{Methods,Headers}
    // The pre-routing handler runs on every request before route dispatch;
    // we use it to attach the response headers and short-circuit the
    // OPTIONS preflight without touching individual routes.
    if (!params.server_cors_origin.empty()) {
        const std::string cors_origin = params.server_cors_origin;
        svr.set_pre_routing_handler([cors_origin](const Request& req, Response& res) {
            res.set_header("Access-Control-Allow-Origin", cors_origin);
            res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
            res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-API-Key");
            res.set_header("Access-Control-Max-Age", "86400");
            if (req.method == "OPTIONS") {
                res.status = 204;
                return Server::HandlerResponse::Handled;
            }
            return Server::HandlerResponse::Unhandled;
        });
        fprintf(stderr, "crispasr-server: CORS enabled (Allow-Origin: %s)\n", cors_origin.c_str());
    }

    auto require_auth = [&](const Request& req, Response& res) -> bool {
        if (is_authorized(req, api_keys))
            return true;
        auth_error(res);
        return false;
    };

    // -----------------------------------------------------------------------
    // POST /inference — native CrispASR transcription endpoint
    // -----------------------------------------------------------------------
    svr.Post("/inference", [&](const Request& req, Response& res) {
        if (!require_auth(req, res))
            return;
        touch_last_request();
        if (!ensure_loaded(res))
            return;

        // Audio source: either multipart upload (`file`) or server-side path
        // (`file_path`, kobbler-style — koblem mounts kobbler media into the
        // container and passes the path to skip multi-GB multipart uploads).
        std::string audio_path;
        bool tmp_owned = false;
        const std::string file_path = form_string(req, "file_path");
        if (!file_path.empty()) {
            audio_path = file_path;
        } else if (req.has_file("file")) {
            auto audio_file = req.get_file_value("file");
            fprintf(stderr, "crispasr-server: /inference received '%s' (%zu bytes)\n", audio_file.filename.c_str(),
                    audio_file.content.size());
            audio_path = write_temp_audio(audio_file.content.data(), audio_file.content.size());
            if (audio_path.empty()) {
                json_error(res, 500, "failed to create temporary file for audio");
                return;
            }
            tmp_owned = true;
        } else {
            json_error(res, 400, "missing audio: provide multipart 'file' or form 'file_path'");
            return;
        }

        // Per-request parameter overrides.
        whisper_params rp = params;
        rp.language = form_string(req, "language", rp.language);

        const std::string cancel_url = form_string(req, "cancel_url");
        auto result = do_transcribe(audio_path, backend.get(), worker.get(), model_mutex, rp, cancel_url);
        if (tmp_owned)
            std::remove(audio_path.c_str());

        if (!result.ok) {
            json_error(res, 400, result.error);
            return;
        }

        fprintf(stderr, "crispasr-server: transcribed %.1fs audio in %.2fs (%.1fx realtime)%s\n", result.duration_s,
                result.elapsed_s, result.elapsed_s > 0 ? result.duration_s / result.elapsed_s : 0.0,
                result.preempted ? " [preempted]" : "");

        std::string json = crispasr_segments_to_native_json(result.segs, backend_name, result.duration_s);
        res.set_content(json, "application/json");
        touch_last_request();
    });

    // -----------------------------------------------------------------------
    // POST /v1/audio/transcriptions — OpenAI-compatible endpoint
    //
    // Accepts OpenAI fields plus kobbler-ecosystem extensions:
    //   file                       (optional) — audio file (multipart upload)
    //   file_path                  (optional, kobbler) — server-side path
    //   model                      (optional) — ignored (we use the loaded model)
    //   language                   (optional) — ISO-639-1 code
    //   prompt                     (optional) — initial prompt / context
    //   response_format            (optional) — json|verbose_json|text|srt|vtt|kobbler
    //   temperature                (optional) — sampling temperature
    //   timestamp_granularities[]  (optional) — word|segment (verbose_json)
    //   timestamp_granularity      (optional, kobbler) — segment|word|char
    //   beam_size                  (optional, kobbler) — silently ignored by parakeet (greedy TDT)
    //   cancel_url                 (optional, kobbler) — between-chunks GPU preemption signal
    //
    // One of `file` or `file_path` is required. Kobbler clients use file_path
    // to skip multi-GB multipart uploads when the audio is already mounted
    // into the container.
    // -----------------------------------------------------------------------
    svr.Post("/v1/audio/transcriptions", [&](const Request& req, Response& res) {
        if (!require_auth(req, res))
            return;
        touch_last_request();
        if (!ensure_loaded(res))
            return;

        // Audio source: file_path (kobbler) takes precedence over multipart file.
        std::string audio_path;
        bool tmp_owned = false;
        const std::string file_path = form_string(req, "file_path");
        if (!file_path.empty()) {
            audio_path = file_path;
            fprintf(stderr, "crispasr-server: /v1/audio/transcriptions file_path='%s'\n", audio_path.c_str());
        } else if (req.has_file("file")) {
            auto audio_file = req.get_file_value("file");
            fprintf(stderr, "crispasr-server: /v1/audio/transcriptions received '%s' (%zu bytes)\n",
                    audio_file.filename.c_str(), audio_file.content.size());
            audio_path = write_temp_audio(audio_file.content.data(), audio_file.content.size());
            if (audio_path.empty()) {
                json_error(res, 500, "failed to create temporary file for audio");
                return;
            }
            tmp_owned = true;
        } else {
            json_error(res, 400, "missing audio: provide multipart 'file' or form 'file_path'");
            return;
        }

        // Parse OpenAI form fields.
        std::string language = form_string(req, "language", params.language);
        std::string prompt = form_string(req, "prompt", "");
        float temperature = form_float(req, "temperature", params.temperature);

        // Kobbler extensions. Koblem clients identify themselves implicitly via
        // these fields — koblem's stt.rs::apply_params only ever populates them
        // — so we use their presence to select the kobbler response shape when
        // response_format isn't explicitly set. OpenAI clients (which send
        // multipart `file` and never these extras) get the OpenAI default.
        std::string granularity = form_string(req, "timestamp_granularity", "segment");
        const std::string cancel_url = form_string(req, "cancel_url");
        const bool kobbler_hint = !file_path.empty() || !cancel_url.empty() ||
                                  req.has_param("timestamp_granularity") || req.has_file("timestamp_granularity") ||
                                  req.has_param("beam_size") || req.has_file("beam_size");
        std::string response_format = form_string(req, "response_format", kobbler_hint ? "kobbler" : "json");

        // Validate response_format early.
        if (response_format != "json" && response_format != "verbose_json" && response_format != "text" &&
            response_format != "srt" && response_format != "vtt" && response_format != "kobbler") {
            if (tmp_owned)
                std::remove(audio_path.c_str());
            json_error(res, 400,
                       "invalid response_format '" + response_format +
                           "'; must be one of: json, verbose_json, text, srt, vtt, kobbler");
            return;
        }

        // Build per-request params.
        whisper_params rp = params;
        rp.language = language;
        rp.temperature = temperature;
        if (!prompt.empty())
            rp.prompt = prompt;

        auto result = do_transcribe(audio_path, backend.get(), worker.get(), model_mutex, rp, cancel_url);
        if (tmp_owned)
            std::remove(audio_path.c_str());

        if (!result.ok) {
            json_error(res, 400, result.error);
            return;
        }

        fprintf(stderr, "crispasr-server: transcribed %.1fs audio in %.2fs (%.1fx realtime), format=%s%s\n",
                result.duration_s, result.elapsed_s, result.elapsed_s > 0 ? result.duration_s / result.elapsed_s : 0.0,
                response_format.c_str(), result.preempted ? " [preempted]" : "");

        // Format response.
        if (response_format == "text") {
            res.set_content(crispasr_segments_to_text(result.segs), "text/plain; charset=utf-8");
        } else if (response_format == "srt") {
            res.set_content(crispasr_segments_to_srt(result.segs), "application/x-subrip; charset=utf-8");
        } else if (response_format == "vtt") {
            res.set_content(crispasr_segments_to_vtt(result.segs), "text/vtt; charset=utf-8");
        } else if (response_format == "verbose_json") {
            std::string task = rp.translate ? "translate" : "transcribe";
            res.set_content(
                crispasr_segments_to_openai_verbose_json(result.segs, result.duration_s, language, task, temperature),
                "application/json");
        } else if (response_format == "kobbler") {
            res.set_content(crispasr_segments_to_kobbler_json(result.segs, granularity, language, result.preempted),
                            "application/json");
        } else {
            // Default: json — {"text": "..."}
            res.set_content(crispasr_segments_to_openai_json(result.segs), "application/json");
        }
        touch_last_request();
    });

    // -----------------------------------------------------------------------
    // POST /load — hot-swap model
    // -----------------------------------------------------------------------
    svr.Post("/load", [&](const Request& req, Response& res) {
        if (!require_auth(req, res))
            return;
        std::lock_guard<std::mutex> lock(model_mutex);

        std::string new_model = form_string(req, "model");
        std::string new_backend = form_string(req, "backend");

        if (new_model.empty()) {
            json_error(res, 400, "no 'model' field");
            return;
        }

        if (new_backend.empty())
            new_backend = crispasr_detect_backend_from_gguf(new_model);

        const bool new_model_is_auto = new_model == "auto" || new_model == "default";
        if (new_backend.empty() && new_model_is_auto)
            new_backend = "whisper";
        if (new_backend.empty()) {
            json_error(res, 400, "cannot detect backend for model '" + new_model + "'");
            return;
        }

        // Snapshot prior config so we can roll back if the new backend fails.
        const std::string prior_backend_name = backend_name;
        const std::string prior_model = params.model;
        whisper_params new_params = params;
        new_params.model = new_model;
        new_params.backend = new_backend;
        std::string new_backend_name = new_backend;

        if (!resolve_model_into_params(new_params, new_backend_name)) {
            json_error(res, 500, "failed to resolve model '" + new_model + "' for backend '" + new_backend + "'");
            return;
        }

        // Commit the new params + backend_name; load_backend_locked builds the
        // new backend before dropping the prior one, so a failed init leaves
        // `backend` pointing at the still-alive previous model — we just need
        // to restore params/backend_name so /v1/models reflects reality.
        params = new_params;
        backend_name = new_backend_name;
        if (!load_backend_locked()) {
            params.model = prior_model;
            backend_name = prior_backend_name;
            json_error(res, 500, "failed to load model '" + new_params.model + "' with backend '" + new_backend + "'");
            return;
        }

        fprintf(stderr, "crispasr-server: hot-swapped to '%s' backend, model '%s'\n", backend_name.c_str(),
                params.model.c_str());
        res.set_content("{\"status\": \"ok\", \"backend\": \"" + crispasr_json_escape(backend_name) + "\"}",
                        "application/json");
    });

    // -----------------------------------------------------------------------
    // POST /unload — free model + GPU memory
    //
    // Matches kobbler/docker/parakeet-stt/server.py:/unload semantics so the
    // GPU gate (api/src/gpu_lock.rs) can ask either backend to release VRAM
    // when a higher-priority service needs it. Sets model_loaded=false; lazy
    // mode means the next inference request reloads.
    // -----------------------------------------------------------------------
    svr.Post("/unload", [&](const Request& req, Response& res) {
        if (!require_auth(req, res))
            return;
        std::lock_guard<std::mutex> lock(model_mutex);
        free_backend_locked();
        res.set_content("{\"status\": \"unloaded\"}", "application/json");
    });

    // -----------------------------------------------------------------------
    // GET /health
    //
    // Shape matches what koblem's SttClient (api/src/stt.rs::HealthInfo)
    // tolerates plus extras: {status, model_loaded, backend, model}.
    // status="loading" only during the brief window between server bind and
    // initial-load completion (eager mode); lazy mode reports "ok" with
    // model_loaded=false until the first request triggers load.
    // -----------------------------------------------------------------------
    svr.Get("/health", [&](const Request&, Response& res) {
        // In worker mode, the authoritative liveness signal is the worker
        // process itself — model_loaded is a parent-side hint that gets
        // stale if the worker dies out-of-band. Probing worker->is_alive()
        // here actively reaps a dead child so /health stops lying about
        // model_loaded after a crash.
        bool loaded = model_loaded.load();
        if (worker && !worker->is_alive())
            loaded = false;
        const bool serving = ready.load() || lazy_load;
        std::ostringstream js;
        js << "{\"status\": \"" << (serving ? "ok" : "loading") << "\"";
        js << ", \"model_loaded\": " << (loaded ? "true" : "false");
        js << ", \"backend\": \"" << crispasr_json_escape(backend_name) << "\"";
        js << ", \"model\": \"" << crispasr_json_escape(params.model) << "\"";
        if (worker) {
            js << ", \"worker_isolation\": true";
            js << ", \"worker_pid\": " << (worker->is_alive() ? (int)worker->pid() : 0);
        }
        js << "}";
        if (!serving)
            res.status = 503;
        res.set_content(js.str(), "application/json");
    });

    // -----------------------------------------------------------------------
    // GET /backends
    // -----------------------------------------------------------------------
    svr.Get("/backends", [&](const Request& req, Response& res) {
        if (!require_auth(req, res))
            return;
        auto names = crispasr_list_backends();
        std::ostringstream js;
        js << "{\"backends\": [";
        for (size_t i = 0; i < names.size(); i++) {
            if (i)
                js << ", ";
            js << "\"" << crispasr_json_escape(names[i]) << "\"";
        }
        js << "], \"active\": \"" << crispasr_json_escape(backend_name) << "\"}";
        res.set_content(js.str(), "application/json");
    });

    // -----------------------------------------------------------------------
    // GET /v1/models — OpenAI-compatible model list
    // -----------------------------------------------------------------------
    svr.Get("/v1/models", [&](const Request& req, Response& res) {
        if (!require_auth(req, res))
            return;
        std::ostringstream js;
        js << "{\"object\": \"list\", \"data\": [{";
        js << "\"id\": \"" << crispasr_json_escape(params.model) << "\", ";
        js << "\"object\": \"model\", ";
        js << "\"owned_by\": \"crispasr\", ";
        js << "\"backend\": \"" << crispasr_json_escape(backend_name) << "\"";
        js << "}]}";
        res.set_content(js.str(), "application/json");
    });

    // -----------------------------------------------------------------------
    // POST /v1/audio/speech — OpenAI-compatible TTS endpoint
    //
    // Body: application/json
    //   {
    //     "input":           "TEXT to synthesize",       (required)
    //     "model":           "<model id>",               (optional, ignored — we serve the loaded one)
    //     "voice":           "<name in --voice-dir>",    (optional)
    //     "instructions":    "<voice direction prose>",  (optional, applied via params.tts_instruct)
    //     "speed":           0.25 .. 4.0,                (optional, default 1.0)
    //     "response_format": "wav" | "pcm" | "f32"       (optional, default "wav")
    //   }
    //
    // Returns:
    //   200 audio/wav                 — 16-bit PCM int16 RIFF, 24 kHz mono (default)
    //   200 audio/pcm                 — raw int16 LE PCM, 24 kHz mono (OpenAI spec)
    //   200 application/octet-stream  — raw float32 PCM (crispasr-specific f32)
    //
    //   400 — backend lacks CAP_TTS, missing/empty input, input too long,
    //         malformed body, unknown response_format, speed out of range
    //   500 — backend->synthesize returned empty (e.g. unknown voice)
    //   503 — model still loading
    //
    // OpenAI compatibility notes:
    //   - `model` is read but not validated — clients always send it; we
    //     serve whatever was loaded via -m or POST /load. Surfaced in
    //     the synth log line for diagnostics.
    //   - `pcm` is OpenAI's 24 kHz signed 16-bit LE mono raw byte
    //     stream (no header). `f32` is the crispasr extension that
    //     emits raw float32 for downstream DSP.
    //   - `instructions` maps to params.tts_instruct (qwen3-tts
    //     VoiceDesign). On non-VoiceDesign backends it's silently
    //     ignored — OpenAI clients don't expect it to ever 4xx.
    //   - `speed` is applied as a post-synth linear resampler. Native
    //     backend duration knobs are a future improvement.
    //
    // Voice handling: the `voice` field is passed through to
    // params.tts_voice verbatim. Each backend interprets it on its
    // own terms — qwen3-tts CustomVoice resolves it as a speaker
    // name, orpheus resolves "tara"/"leah" as presets, qwen3-tts
    // Base resolves it as a path or (with --voice-dir set) as a
    // bare name relative to the voice-dir. When "voice" is omitted
    // the request inherits whatever was set at server startup via
    // --voice / --instruct.
    // -----------------------------------------------------------------------
    svr.Post("/v1/audio/speech", [&](const Request& req, Response& res) {
        if (!require_auth(req, res))
            return;
        touch_last_request();
        if (!ensure_loaded(res))
            return;
        const uint32_t active_caps = worker ? worker->capabilities() : (backend ? backend->capabilities() : 0u);
        const bool have_active = worker ? worker->is_alive() : (backend != nullptr);
        if (!have_active || !(active_caps & CAP_TTS)) {
            json_error(res, 400,
                       "loaded backend '" + backend_name +
                           "' does not support TTS (no CAP_TTS); load a TTS backend "
                           "(e.g. qwen3-tts, kokoro, vibevoice, orpheus) via POST /load");
            return;
        }

        nlohmann::json body;
        try {
            body = nlohmann::json::parse(req.body);
        } catch (...) {
            json_error(res, 400, "invalid JSON body", "invalid_json");
            return;
        }

        std::string text = body.value("input", "");
        if (text.empty()) {
            json_error(res, 400, "missing or empty 'input' field", "missing_required_field", "input");
            return;
        }
        if (params.tts_max_input_chars > 0 && (int)text.size() > params.tts_max_input_chars) {
            json_error(res, 400,
                       "'input' length " + std::to_string(text.size()) + " exceeds the configured limit of " +
                           std::to_string(params.tts_max_input_chars) +
                           " chars; raise --tts-max-input-chars or split the input client-side",
                       "input_too_long", "input");
            return;
        }

        // Read but don't validate `model` — we serve whatever was loaded.
        // Surfaced in the log line below for diagnostics.
        std::string requested_model = body.value("model", "");

        std::string voice_name = body.value("voice", "");
        std::string instructions = body.value("instructions", "");
        std::string response_format = body.value("response_format", std::string("wav"));
        if (response_format != "wav" && response_format != "pcm" && response_format != "f32") {
            json_error(res, 400, "response_format must be one of 'wav', 'pcm', or 'f32'", "unsupported_response_format",
                       "response_format");
            return;
        }

        float speed = body.value("speed", 1.0f);
        if (!(speed >= 0.25f && speed <= 4.0f)) {
            json_error(res, 400, "'speed' must be between 0.25 and 4.0 (got " + std::to_string(speed) + ")",
                       "invalid_speed", "speed");
            return;
        }

        // Per-request param overrides — copy then mutate. The voice
        // string is passed through verbatim; the backend adapter owns
        // the interpretation (speaker name, preset, path, or bare name
        // relative to --voice-dir). rp.tts_voice_dir already carries
        // the server's configured dir for adapters that want to do
        // bare-name resolution.
        //
        // `instructions` maps to params.tts_instruct (qwen3-tts
        // VoiceDesign). Non-VoiceDesign backends silently ignore it;
        // we don't 4xx because OpenAI clients always include the field
        // when they're using gpt-4o-mini-tts and shouldn't see errors
        // when pointed at a base TTS server.
        whisper_params rp = params;
        if (!voice_name.empty())
            rp.tts_voice = voice_name;
        if (!instructions.empty())
            rp.tts_instruct = instructions;

        // Long-form chunking (PLAN §75d / issue #66): split input on
        // sentence boundaries before dispatching to the backend so each
        // synth stays inside the talker's healthy training horizon.
        // Single-sentence input becomes a 1-element vector; the per-call
        // overhead is one std::vector<float> move.
        //
        // VibeVoice Base voice cloning relies on the continuous prompt +
        // generated-text context to maintain speaker identity and prosody.
        // Chunking degrades it, so keep the request as one synthesis call.
        auto t0 = std::chrono::steady_clock::now();
        const std::vector<std::string> sentences = crispasr_tts_plan_chunks_for_backend(text, backend->name());

        std::vector<std::vector<float>> chunks;
        chunks.reserve(sentences.size());
        {
            std::lock_guard<std::mutex> lock(model_mutex);
            for (const auto& sent : sentences) {
                std::vector<float> chunk = worker ? worker->synthesize(sent, rp) : backend->synthesize(sent, rp);
                if (!chunk.empty())
                    chunks.push_back(std::move(chunk));
            }
        }
        // 200 ms silence at 24 kHz between chunks. Inaudible click
        // suppression at boundaries; long enough that the listener
        // perceives a natural sentence pause without dragging.
        std::vector<float> pcm = crispasr_tts_concat_with_silence(chunks, 4800);
        auto t1 = std::chrono::steady_clock::now();

        if (pcm.empty()) {
            json_error(res, 500, "synthesis failed (backend returned empty audio)", "synthesis_failed");
            return;
        }

        // Apply speed via linear-interpolation resampler. speed=1.0 is a
        // no-op. Quality loss vs a sinc resampler is minimal at modest
        // speeds (0.5x .. 2.0x) for speech; backends that grow native
        // duration knobs will plumb through `rp.tts_speed` directly and
        // bypass this path.
        if (speed != 1.0f) {
            const int in_n = (int)pcm.size();
            const int out_n = std::max(1, (int)((float)in_n / speed));
            std::vector<float> resampled((size_t)out_n);
            for (int i = 0; i < out_n; i++) {
                const float src = (float)i * speed;
                const int s0 = (int)src;
                const int s1 = std::min(s0 + 1, in_n - 1);
                const float frac = src - (float)s0;
                resampled[i] = pcm[s0] * (1.0f - frac) + pcm[s1] * frac;
            }
            pcm = std::move(resampled);
        }

        const double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
        const double audio_s = (double)pcm.size() / 24000.0;
        fprintf(stderr,
                "crispasr-server: synthesized %.1fs audio in %.2fs (RTF=%.2f) "
                "voice='%s' speed=%.2f format=%s model='%s' chunks=%zu\n",
                audio_s, elapsed_s, elapsed_s > 0 ? elapsed_s / audio_s : 0.0,
                voice_name.empty() ? "<startup>" : voice_name.c_str(), speed, response_format.c_str(),
                requested_model.empty() ? "<unset>" : requested_model.c_str(), chunks.size());

        if (response_format == "f32") {
            std::string buf((const char*)pcm.data(), pcm.size() * sizeof(float));
            res.set_content(std::move(buf), "application/octet-stream");
        } else if (response_format == "pcm") {
            // OpenAI's pcm: 24 kHz signed 16-bit LE mono raw bytes, no
            // header. Content-Type is documented as audio/pcm; clients
            // know the rate out-of-band from the spec.
            std::string raw = crispasr_make_pcm_int16_le(pcm.data(), (int)pcm.size());
            res.set_content(std::move(raw), "audio/pcm");
        } else {
            std::string wav = crispasr_make_wav_int16(pcm.data(), (int)pcm.size(), 24000);
            res.set_content(std::move(wav), "audio/wav");
        }
    });

    // -----------------------------------------------------------------------
    // GET /v1/voices — list voices in --voice-dir (CAP_TTS only)
    // Returns: {"voices": [{"name": "<stem>", "format": "wav"|"gguf"}, ...]}
    // -----------------------------------------------------------------------
    svr.Get("/v1/voices", [&](const Request& req, Response& res) {
        if (!require_auth(req, res))
            return;
        const bool have_active = worker ? worker->is_alive() : (backend != nullptr);
        if (!model_loaded.load() || !have_active) {
            json_error(res, 503, "no model loaded");
            return;
        }
        const uint32_t active_caps = worker ? worker->capabilities() : backend->capabilities();
        if (!(active_caps & CAP_TTS)) {
            json_error(res, 400,
                       "loaded backend '" + backend_name +
                           "' does not support TTS (no CAP_TTS); load a TTS backend "
                           "(e.g. qwen3-tts, kokoro, vibevoice, orpheus) via POST /load");
            return;
        }

        std::ostringstream js;
        js << "{\"voices\": [";
        bool first = true;
        if (!params.tts_voice_dir.empty()) {
            std::error_code ec;
            for (const auto& entry : std::filesystem::directory_iterator(params.tts_voice_dir, ec)) {
                if (ec)
                    break;
                if (!entry.is_regular_file())
                    continue;
                const auto& path = entry.path();
                const std::string ext = path.extension().string();
                if (ext != ".wav" && ext != ".gguf")
                    continue;
                const std::string stem = path.stem().string();
                const char* fmt = (ext == ".wav") ? "wav" : "gguf";
                if (!first)
                    js << ", ";
                js << "{\"name\": \"" << crispasr_json_escape(stem) << "\", \"format\": \"" << fmt << "\"}";
                first = false;
            }
        }
        js << "]}";
        res.set_content(js.str(), "application/json");
    });

    // -----------------------------------------------------------------------
    // Catch unmatched routes. cpp-httplib invokes the error handler for any
    // 4xx/5xx response, including ones our own route handlers produced via
    // json_error() — so guard on `res.body.empty()` to avoid clobbering the
    // structured error bodies the route handlers already set. Empty body
    // here means no route matched (or a matched route forgot to call
    // set_content), so falling back to the legacy "not found" payload is
    // safe.
    svr.set_error_handler([&](const Request& req, Response& res) {
        if (!res.body.empty())
            return;
        fprintf(stderr, "crispasr-server: %s %s → %d (no matching route)\n", req.method.c_str(), req.path.c_str(),
                res.status);
        res.set_content("{\"error\": \"not found. Use POST /v1/audio/transcriptions\"}", "application/json");
    });

    // Start
    // -----------------------------------------------------------------------
    // backend (or worker) may be null in lazy mode; gate the TTS-banner on
    // whether we actually have a loaded backend, otherwise just suppress
    // the TTS lines until the first /load (which can swap to a TTS backend
    // at runtime).
    const uint32_t boot_caps = worker ? worker->capabilities() : (backend ? backend->capabilities() : 0u);
    const bool tts = (boot_caps & CAP_TTS) != 0;
    fprintf(stderr, "\ncrispasr-server: listening on %s:%d\n", host.c_str(), port);
    fprintf(stderr, "  POST /inference                  — upload audio (native JSON)\n");
    fprintf(stderr, "  POST /v1/audio/transcriptions    — OpenAI-compatible API\n");
    if (tts) {
        fprintf(stderr, "  POST /v1/audio/speech            — TTS (OpenAI-compatible)\n");
    }
    fprintf(stderr, "  POST /load                       — hot-swap model\n");
    fprintf(stderr, "  POST /unload                     — free model + GPU memory\n");
    fprintf(stderr, "  GET  /health                     — server status\n");
    fprintf(stderr, "  GET  /backends                   — list backends\n");
    fprintf(stderr, "  GET  /v1/models                  — model info\n");
    if (tts) {
        fprintf(stderr, "  GET  /v1/voices                  — list voices in --voice-dir\n");
        if (params.tts_voice_dir.empty()) {
            fprintf(stderr, "crispasr-server: warning: --voice-dir not set; /v1/voices will return empty "
                            "and /v1/audio/speech will reject requests with a 'voice' field\n");
        }
    }
    fprintf(stderr, "\n");
    if (!api_keys.empty())
        fprintf(stderr, "crispasr-server: API key authentication enabled\n");
    if (worker)
        fprintf(stderr, "crispasr-server: worker-isolation enabled (subprocess holds GPU; /unload SIGKILLs it)\n");
    if (lazy_load)
        fprintf(stderr, "crispasr-server: lazy-load enabled (model loads on first request)\n");
    if (idle_unload_seconds > 0)
        fprintf(stderr, "crispasr-server: idle-unload after %ds of inactivity\n", idle_unload_seconds);

    // Idle-unload watchdog. Wakes once a second; when the model is loaded
    // and last_request_ms is older than the threshold, frees the backend.
    // Stops cleanly via watchdog_stop on listen() return — we'd rather have
    // a tidy join() on shutdown than detach() and race container teardown.
    std::atomic<bool> watchdog_stop{false};
    std::condition_variable watchdog_cv;
    std::mutex watchdog_cv_mutex;
    std::thread watchdog;
    if (idle_unload_seconds > 0) {
        watchdog = std::thread([&]() {
            for (;;) {
                std::unique_lock<std::mutex> lk(watchdog_cv_mutex);
                if (watchdog_cv.wait_for(lk, std::chrono::seconds(1), [&]() { return watchdog_stop.load(); }))
                    return;
                lk.unlock();
                if (!model_loaded.load())
                    continue;
                const int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                           std::chrono::steady_clock::now().time_since_epoch())
                                           .count();
                const int64_t last = last_request_ms.load();
                if (last == 0)
                    continue; // never received a request — leave loaded
                if ((now_ms - last) >= (int64_t)idle_unload_seconds * 1000) {
                    fprintf(stderr, "crispasr-server: idle %ds since last request — unloading\n", idle_unload_seconds);
                    std::lock_guard<std::mutex> lock(model_mutex);
                    free_backend_locked();
                }
            }
        });
    }

    svr.listen(host, port);

    if (watchdog.joinable()) {
        {
            std::lock_guard<std::mutex> lk(watchdog_cv_mutex);
            watchdog_stop.store(true);
        }
        watchdog_cv.notify_all();
        watchdog.join();
    }
    return 0;
}
