// crispasr_worker_ipc.h — length-prefixed frame protocol over Unix-domain
// socketpair, used by the crispasr-server subprocess-worker model.
//
// Goal: when the parent process calls /unload on a CUDA-backed backend,
// SIGKILLing the worker tears down the CUDA primary context (cuBLAS
// workspace, compiled cubin/PTX cache, driver runtime state) so VRAM
// returns to the pre-load floor. In-process unload only frees weight
// buffers + KV slabs; the 100-400 MiB primary-context residual sticks
// until the parent exits — that's why parakeet-stt looks "still loaded"
// on nvidia-smi after a /unload.
//
// Adapted from qwen3-tts.cpp/src/worker_ipc.h. Frame types renamed to
// reflect the dual STT/TTS surface: TRANSCRIBE_* for backend->transcribe,
// SYNTH_* for backend->synthesize. The wire format is identical so
// future work can collapse the two.
//
// Parent role: HTTP server, model resolver, no GPU. Owns the parent end
// of the socket. Spawns child via fork()+execv("--worker <fd>").
//
// Worker role: owns CrispasrBackend, ggml-cuda context, model weights.
// Started via fork()+execv from parent so the address space is fresh
// (no copy-on-write inheritance of parent state into the CUDA process).
//
// Protocol: fixed 12-byte header followed by `payload_len` bytes.
//
//   [u32 frame_type][u32 payload_len][u32 req_id][u8 payload[payload_len]]
//
// Strings + structured payloads use nlohmann::json. TRANSCRIBE_REQ /
// SYNTH_RESP carry raw float32 sample bytes alongside JSON metadata in
// a [u32 json_len][json][raw bytes] envelope so we don't base64-stuff
// multi-MiB audio through JSON.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#if defined(_WIN32)
// Worker-isolation is POSIX-only (fork + socketpair). On Windows the
// parent uses the in-process backend and CrispasrWorkerSession::ensure_loaded
// returns false. The header still compiles so callers don't need ifdefs
// at every call site.
typedef int pid_t;
#else
#include <sys/types.h>
#endif

namespace crispasr {

enum class WorkerFrame : uint32_t {
    HELLO                 = 0x01, // W→P  {"pid": int, "role": "crispasr-worker"}
    LOAD_REQ              = 0x10, // P→W  {"params": <whisper_params json>, "backend": str}
    LOAD_RESP             = 0x11, // W→P  {"ok": bool, "error": str, "backend_name": str,
                                  //       "capabilities": uint, "model": str, "sample_rate": int}
    TRANSCRIBE_REQ        = 0x20, // P→W  json header + raw f32 samples
    TRANSCRIBE_RESP       = 0x21, // W→P  json segments
    TRANSCRIBE_STEREO_REQ = 0x22, // P→W  json header + raw f32 L samples + raw f32 R samples
    SYNTH_REQ             = 0x30, // P→W  {"text": str, "params": json}
    SYNTH_RESP            = 0x31, // W→P  json header + raw f32 audio
    ERR_RESP              = 0x2F, // W→P  {"error": str}  — generic per-request failure
    PING                  = 0x40, // either {"t_send_ns": u64}
    PONG                  = 0x41,
    SHUTDOWN              = 0xFF, // P→W  exit cleanly (parent normally just SIGKILLs)
};

struct FrameHeader {
    uint32_t type;     // WorkerFrame
    uint32_t len;      // payload bytes that follow
    uint32_t req_id;   // 0 = unsolicited / no correlation
};
static_assert(sizeof(FrameHeader) == 12, "FrameHeader must stay 12 bytes");

inline constexpr size_t HEADER_BYTES      = sizeof(FrameHeader);
inline constexpr size_t MAX_FRAME_PAYLOAD = 256u * 1024u * 1024u; // 256 MiB safety cap (long audio chunks)

// Errors. Return-coded so worker dispatch loop can log + continue
// or shutdown deterministically.
enum class IpcError {
    OK = 0,
    EofClean,        // peer closed cleanly mid-read (no bytes received)
    EofMidFrame,     // peer closed in the middle of a frame
    SocketError,     // read/write returned -1 with non-recoverable errno
    ProtocolError,   // bad header
    PayloadTooBig,
};

const char* ipc_error_str(IpcError e);

// Blocking. Returns OK iff `len` bytes were fully received.
IpcError read_exact(int fd, void* buf, size_t len);
IpcError write_exact(int fd, const void* buf, size_t len);

// Send / receive a full frame. send_frame coalesces header+payload into
// a single writev() to keep latency tight on small frames.
IpcError send_frame(int fd, WorkerFrame type, uint32_t req_id, const void* payload, size_t payload_len);
IpcError send_frame(int fd, WorkerFrame type, uint32_t req_id, const std::string& json);
IpcError send_frame(int fd, WorkerFrame type, uint32_t req_id, const std::vector<uint8_t>& payload);

// Receive a full frame. On OK, `out_payload` contains exactly hdr.len bytes.
IpcError recv_frame(int fd, FrameHeader* out_hdr, std::vector<uint8_t>* out_payload);

// Pack/unpack a JSON-meta + raw f32 audio envelope: [u32 json_len][json][f32 bytes].
std::vector<uint8_t> pack_audio_payload(const std::string& json_meta, const float* samples, size_t n_samples);
bool unpack_audio_payload(const std::vector<uint8_t>& payload, std::string* out_meta, std::vector<float>* out_samples);

// Zero-copy fast path for TRANSCRIBE_REQ-style payloads: header(u32 json_len),
// json bytes, and raw f32 sample bytes go to the kernel as separate iovs in
// one writev() — no allocate-and-memcpy of the audio buffer first. Caller
// owns `samples` for the duration of the call.
//
// Wire format produced is identical to send_frame(..., pack_audio_payload(..)),
// so the receiver path (recv_frame + unpack_audio_payload) is unchanged.
IpcError send_frame_audio(int fd, WorkerFrame type, uint32_t req_id, const std::string& json_meta,
                          const float* samples, size_t n_samples);

// Pack/unpack a JSON-meta + two-channel f32 envelope:
//   [u32 json_len][json][f32 left[n]][f32 right[n]]
std::vector<uint8_t> pack_stereo_payload(const std::string& json_meta, const float* left, const float* right,
                                         size_t n_samples_per_channel);
bool unpack_stereo_payload(const std::vector<uint8_t>& payload, std::string* out_meta, std::vector<float>* out_left,
                           std::vector<float>* out_right);

// Spawn helper: socketpair() + fork() + execv(argv[0], "--worker <fd>").
// `extra_argv` is appended after "--worker N" so the child sees the same
// load-time arguments (-m, --backend, --auto-download, etc.) it would
// have seen if invoked directly. Returns the child pid on success and
// writes the parent-side fd to `*out_parent_fd`. Returns -1 on failure.
pid_t spawn_worker(const char* self_argv0, const std::vector<std::string>& extra_argv, int* out_parent_fd,
                   const std::string& cuda_visible_devices = "");

} // namespace crispasr
