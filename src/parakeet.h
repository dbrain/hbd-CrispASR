// parakeet.h — public C API for nvidia/parakeet-tdt-0.6b-v3 ggml runtime
//
// Multilingual ASR (25 European languages) using FastConformer encoder +
// Token-and-Duration Transducer (TDT) decoder. Word-level timestamps come
// for free from the duration head — no separate CTC alignment needed.
//
// Models are loaded from GGUF files produced by:
//   python models/convert-parakeet-to-gguf.py --nemo X.nemo --output X.gguf

#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct parakeet_context;

struct parakeet_context_params {
    int n_threads;
    bool use_flash; // flash attention in encoder (default: false)
    int verbosity;  // 0=silent 1=normal 2=verbose
    bool use_gpu;   // false => force CPU backend
};

struct parakeet_context_params parakeet_context_default_params(void);

// Load model from GGUF (produced by convert-parakeet-to-gguf.py)
struct parakeet_context* parakeet_init_from_file(const char* path_model, struct parakeet_context_params params);

void parakeet_free(struct parakeet_context* ctx);

// ---- Per-token data returned by parakeet_transcribe_ex() ----

struct parakeet_token_data {
    int id;        // SentencePiece token id (0 .. vocab_size-1)
    char text[48]; // decoded text (SentencePiece '▁' converted to ' ')
    int64_t t0;    // start time, centiseconds (absolute, includes t_offset_cs)
    int64_t t1;    // end time,   centiseconds (start + duration*frame_dur_cs)
    float p;       // softmax probability of the emitted token [0,1]
};

// Word-level data: sub-word tokens grouped at SentencePiece '▁' boundaries.
struct parakeet_word_data {
    char text[64]; // word text (no leading space)
    int64_t t0;    // start time, centiseconds (from first sub-word)
    int64_t t1;    // end time,   centiseconds (from last sub-word)
    float p;       // mean softmax probability across the word's sub-word tokens [0,1]
};

struct parakeet_result {
    char* text;                         // full transcript (malloc'd, caller owns)
    struct parakeet_token_data* tokens; // per-token timing (malloc'd)
    int n_tokens;
    struct parakeet_word_data* words; // grouped word timings (malloc'd)
    int n_words;
};

void parakeet_result_free(struct parakeet_result* r);

// Transcribe raw 16 kHz mono PCM, returning a malloc'd UTF-8 string.
char* parakeet_transcribe(struct parakeet_context* ctx, const float* samples, int n_samples);

// Like parakeet_transcribe but returns per-token TDT timestamps.
//
// t_offset_cs: absolute start of this audio slice in centiseconds.
//   Token t0/t1 = t_offset_cs + (encoder_frame * frame_dur_cs).
//   For long audio with VAD, pass (vad_segment_t0_seconds * 100).
//
// Unlike Cohere's cross-attention DTW path, these timestamps come directly
// from the TDT decoder's duration head and are accurate to one encoder
// frame (~80 ms for parakeet-tdt-0.6b-v3).
struct parakeet_result* parakeet_transcribe_ex(struct parakeet_context* ctx, const float* samples, int n_samples,
                                               int64_t t_offset_cs);

// Vocabulary helpers
int parakeet_n_vocab(struct parakeet_context* ctx);
int parakeet_blank_id(struct parakeet_context* ctx);
const char* parakeet_token_to_str(struct parakeet_context* ctx, int token_id);

// Sampling: when temperature > 0, the TDT decoder draws each non-blank
// token via stable-softmax(logits / temperature) instead of argmax.
// Temperature == 0 (the default) keeps the bit-identical pure-greedy
// path. Set per-call as needed; the setting is sticky on the context
// until the next call. seed == 0 means time-based RNG.
void parakeet_set_temperature(struct parakeet_context* ctx, float temperature, uint64_t seed);

// ---- Windowed streaming (buffered, bounded window) ----
//
// Audio is cut on a fixed grid of chunk_sec. Each chunk is encoded inside a window of
// [left_sec of history | chunk | right_sec of lookahead] with full attention, only the chunk's
// frames are decoded, and the TDT predictor state carries across chunks, so text is final once
// returned. Latency ~= chunk + right context + compute. Accuracy falls off sharply below ~0.64 s
// of right context. The caller serialises calls per context; several streams may share one
// context (each keeps its own predictor state).
// Optional speech gate: called before a chunk is encoded with the audio [chunk start - up to 1 s
// of context, end of lookahead). Return 0 for "no speech" and the chunk is skipped (no encoder
// pass; equivalent to decoding it as all-blank, so the predictor state is unaffected). Decoding
// resumes as soon as speech appears in a chunk's lookahead, and every encoded window carries
// left_sec of history, so a skipped stretch never cuts off the start of the next utterance.
// After the last speech chunk, hangover_chunks more are decoded regardless.
typedef int (*parakeet_speech_fn)(const float* samples, int n_samples, void* user);

struct parakeet_stream_params {
    float chunk_sec; // decode grid (default 0.64)
    float right_sec; // lookahead the chunk sees (default 0.64)
    float left_sec;  // history the chunk sees (default 10.0)
    parakeet_speech_fn speech_fn; // NULL = decode every chunk
    void* speech_user;
    int hangover_chunks; // default 2
};
struct parakeet_stream;
struct parakeet_stream_params parakeet_stream_default_params(void);
struct parakeet_stream* parakeet_stream_begin(struct parakeet_context* ctx, struct parakeet_stream_params params);
// Append 16 kHz mono PCM. Returns the text/tokens committed by this call (possibly none; times
// are absolute from the stream start, text keeps its leading space so deltas concatenate). Words
// lag by up to one word: the trailing word is reported once the next word starts, or on the
// flush, so a word cut by a chunk edge arrives whole. final=1 flushes the tail.
// partial_mode 1: *out_partial receives a malloc'd provisional transcript of the audio not yet
// committed, from the last window's lookahead (free). partial_mode 2: same, but re-encodes a
// window ending at the newest sample first (one extra encoder pass) so the text reaches "now";
// ask for it when the speaker goes quiet. 0: no partial. Returns NULL on error.
struct parakeet_result* parakeet_stream_feed(struct parakeet_context* ctx, struct parakeet_stream* st,
                                             const float* samples, int n_samples, int final, int partial_mode,
                                             char** out_partial);
void parakeet_stream_free(struct parakeet_stream* st);
// Windows encoded vs skipped by the speech gate so far.
void parakeet_stream_stats(const struct parakeet_stream* st, int* n_encoded, int* n_skipped);

// Hyper-parameters needed by callers (frame duration for stamping etc.)
int parakeet_frame_dur_cs(struct parakeet_context* ctx); // centiseconds per encoder frame
int parakeet_n_mels(struct parakeet_context* ctx);
int parakeet_sample_rate(struct parakeet_context* ctx);

// ---- Stage-level entry points (for crispasr-diff testing) ----
// These let the diff harness compare intermediate activations against
// a PyTorch reference. They are NOT needed by the normal transcribe
// path — use parakeet_transcribe(_ex) for inference.
//
// Returns a malloc'd F32 buffer that the caller must free(). Shape is
// reported via the out_* parameters. Returns nullptr on failure.

// Log-mel spectrogram of raw 16 kHz mono PCM.
// Output layout: row-major (n_mels, T_mel).
float* parakeet_compute_mel(struct parakeet_context* ctx, const float* samples, int n_samples, int* out_n_mels,
                            int* out_T_mel);

// Run just the audio encoder on a mel spectrogram. Takes the output of
// parakeet_compute_mel() (or any externally-produced reference mel with
// the same layout) and returns the encoder hidden state.
// Output layout: row-major (T_enc, d_model).
float* parakeet_run_encoder(struct parakeet_context* ctx, const float* mel, int n_mels, int T_mel, int* out_T_enc,
                            int* out_d_model);

// Internal smoke test: build encoder graph on a zero mel of `T_mel` frames,
// run it, and report the output T_enc. Returns T_enc on success or -1.
int parakeet_test_encoder(struct parakeet_context* ctx, int T_mel);

// Run the encoder and capture per-layer intermediates for diff testing.
// Caller passes pre-allocated row-major (T_enc, d_model) buffers in `out`:
//   out[0]    : after pre-encode (subsampling + projection)
//   out[1..N] : after each conformer layer (where N = n_layers)
// `out_count` must be at least n_layers+1; extra slots are ignored.
// Sizes are reported back via *out_T_enc / *out_d_model. Returns 0 on
// success, non-zero on failure. The C-side allocates and frees its own
// scratch; the caller-provided buffers are written into directly.
int parakeet_run_encoder_dump(struct parakeet_context* ctx, const float* mel, int n_mels, int T_mel, float** out,
                              int out_count, int* out_T_enc, int* out_d_model);

// Internal smoke test: take raw 16 kHz mono PCM, run mel + encoder, print
// encoder-output statistics. Returns T_enc on success or -1.
int parakeet_test_audio(struct parakeet_context* ctx, const float* samples, int n_samples);

// Per-stage encoder profiler (lap-5 lever 1).
// Runs the encoder once through the sched path with eval-callback breakpoints
// at each conformer block's ff1/attn/conv/ff2/block_end, plus pre_encode and
// the final-LN. Each break forces a backend synchronize so chrono captures
// real GPU time. Sched-path inflates total cost vs the lap-4 gallocr cache,
// but per-stage *ratios* are accurate. Prints a per-layer table to stderr.
// Returns 0 on success, non-zero on failure. Use PARAKEET_PROFILE_ENC=1 for
// auto-trigger from parakeet_transcribe_ex (runs once and continues).
int parakeet_profile_encoder(struct parakeet_context* ctx, const float* mel, int n_mels, int T_mel);

#ifdef __cplusplus
}
#endif
