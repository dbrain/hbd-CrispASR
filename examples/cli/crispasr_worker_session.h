// crispasr_worker_session.h — parent-side handle on a subprocess worker
// that owns the GPU-bearing CrispasrBackend, plus the child-side dispatch
// loop. See HANDOFF-perf-* in kobbler/docker/stt-parakeet-cpp for context.
//
// Purpose: when the parent's HTTP /unload SIGKILLs the worker, the entire
// CUDA primary context (cuBLAS workspace, cubin/PTX cache, driver state)
// is reclaimed alongside the weight buffers. In-process unload only
// frees explicit allocations, leaving 100-400 MiB of "stuck" residual on
// nvidia-smi until the parent exits — that's the visible symptom this
// module fixes.
//
// Design mirrors qwen3-tts.cpp/src/worker_session.{h,cpp}. The IPC is in
// crispasr_worker_ipc.{h,cpp}; this header owns the parent->worker
// lifecycle, request marshalling, and the worker's main loop.

#pragma once

#include "crispasr_backend.h"
#include "crispasr_worker_ipc.h"
#include "whisper_params.h"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace crispasr {

// Subset of whisper_params the worker needs at LOAD time. We send the
// full params blob anyway (so per-call overrides like language/temperature
// work without a second roundtrip), but the load-time identity (model
// path + backend name) drives the "can we keep the existing worker, or
// do we need to respawn?" decision in WorkerSession::ensure_loaded.
struct WorkerLoadConfig {
    std::string backend; // "parakeet", "whisper", "qwen3-tts", ...
    std::string model;   // resolved local path (parent has already run resolve_model_into_params)
    std::string model_quant;
    bool        verbose   = false;
    bool        use_gpu   = true;
    int32_t     gpu_device = 0;
    bool        flash_attn = true;

    // Params blob the worker passes to backend->init(). Serialized as JSON.
    // Carries every relevant whisper_params field. Cached on the parent so
    // subsequent transcribe() calls can override per-call fields without
    // re-sending the whole struct.
    whisper_params init_params;
};

// Parent-side handle. One instance per server process. Thread-safe:
// every method that touches the socket takes io_mutex_ for the full
// request/response round-trip.
class WorkerSession {
public:
    WorkerSession(const char* argv0, std::vector<std::string> extra_argv = {});
    ~WorkerSession();

    // Spawn the worker if it isn't running, send LOAD_REQ if its loaded
    // config differs from `cfg`. Returns true on success; sets last_error_.
    // On a config mismatch, the existing worker is SIGKILLed first so the
    // new one starts with a fresh CUDA context.
    bool ensure_loaded(const WorkerLoadConfig& cfg);

    // SIGKILL + waitpid. Idempotent. Subsequent ensure_loaded() respawns.
    // This is the lever that reclaims VRAM cleanly.
    void shutdown();

    // Active liveness check. Reaps a dead child via waitpid(WNOHANG) so a
    // worker that died out-of-band (external SIGKILL, SEGV, GGML_ABORT,
    // OOM-killer) is detected before the next request and the parent's
    // ensure_loaded() respawns instead of dispatching to a stale fd. The
    // const-cast is intentional: this is logically idempotent state
    // reconciliation, not a mutation of observable behaviour.
    bool        is_alive() const;
    pid_t       pid() const { return pid_; }
    int32_t     sample_rate() const { return sample_rate_; }
    uint32_t    capabilities() const { return capabilities_; }
    const std::string& backend_name() const { return backend_name_; }
    const std::string& last_error() const { return last_error_; }

    // GPU placement: default card (UUID) for un-targeted requests + a pending
    // per-request override. The next ensure_loaded() relocates if it differs.
    void set_default_gpu(std::string gpu) { default_gpu_ = std::move(gpu); }
    void set_next_gpu(const std::string& gpu) {
        if (!gpu.empty()) { std::lock_guard<std::mutex> lk(io_mutex_); next_gpu_ = gpu; }
    }

    // STT — mirror CrispasrBackend::transcribe. `params` is the per-call
    // params (with overrides applied by the HTTP handler). On IPC failure,
    // the worker is reaped (kill+wait) and the call returns an empty vector;
    // last_error_ carries the detail. The next ensure_loaded() respawns.
    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& params);

    std::vector<crispasr_segment> transcribe_stereo(const float* left, const float* right,
                                                    int n_samples_per_channel, int64_t t_offset_cs,
                                                    const whisper_params& params);

    // TTS — mirror CrispasrBackend::synthesize. Returns 24 kHz mono PCM
    // (or whatever the loaded backend's native rate is — the value is
    // backend-specific; see CrispasrBackend::synthesize comment). On IPC
    // failure returns an empty vector and sets last_error_.
    std::vector<float> synthesize(const std::string& text, const whisper_params& params);

private:
    bool send_load_req_locked(const WorkerLoadConfig& cfg);
    void kill_worker_locked();

    std::vector<crispasr_segment> do_transcribe_locked(WorkerFrame frame_type, const std::vector<uint8_t>& payload);

    std::string                argv0_;
    std::vector<std::string>   extra_argv_;
    WorkerLoadConfig           loaded_cfg_;
    bool                       loaded_ok_ = false;
    std::string                default_gpu_;   // CVD for un-targeted spawns
    std::string                next_gpu_;      // pending per-request target
    std::string                worker_gpu_;    // GPU the live worker is pinned to

    pid_t                      pid_  = -1;
    int                        fd_   = -1;
    int32_t                    sample_rate_ = 0;
    uint32_t                   capabilities_ = 0;
    std::string                backend_name_;
    std::string                resolved_model_;
    mutable std::mutex         io_mutex_;
    std::string                last_error_;
    std::atomic<uint32_t>      next_req_id_{1};
};

// Worker-side dispatch loop. Called from main() when --worker <fd> is
// passed. Owns the CrispasrBackend instance, services LOAD_REQ +
// TRANSCRIBE_REQ + TRANSCRIBE_STEREO_REQ + SYNTH_REQ + SHUTDOWN. Exits on
// EOF or SHUTDOWN.
//
// Returns the process exit code (0 on clean shutdown).
int run_worker_loop(int fd);

} // namespace crispasr
