// crispasr_worker_session.cpp — see crispasr_worker_session.h for design.

#include "crispasr_worker_session.h"

#include "../json.hpp"

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>

#if !defined(_WIN32)
#include <sys/wait.h>
#include <unistd.h>
#if defined(__linux__)
#include <sys/prctl.h>
#endif
#endif

using nlohmann::json;

namespace crispasr {

// ────────────────────────── whisper_params <-> json ──────────────────────────
//
// We send a comprehensive serialization (every backend-readable field) so
// per-call overrides land on the worker without bespoke per-field plumbing.
// `grammar_parsed` (the GBNF parser state) is intentionally skipped — it's
// non-trivially copyable; the worker re-parses `grammar` when needed.

static json params_to_json(const whisper_params& p) {
    json j = {
        {"n_threads", p.n_threads},
        {"n_processors", p.n_processors},
        {"offset_t_ms", p.offset_t_ms},
        {"offset_n", p.offset_n},
        {"duration_ms", p.duration_ms},
        {"progress_step", p.progress_step},
        {"max_context", p.max_context},
        {"max_len", p.max_len},
        {"split_on_punct", p.split_on_punct},
        {"best_of", p.best_of},
        {"beam_size", p.beam_size},
        {"audio_ctx", p.audio_ctx},
        {"word_thold", p.word_thold},
        {"entropy_thold", p.entropy_thold},
        {"logprob_thold", p.logprob_thold},
        {"no_speech_thold", p.no_speech_thold},
        {"grammar_penalty", p.grammar_penalty},
        {"temperature", p.temperature},
        {"temperature_inc", p.temperature_inc},
        {"debug_mode", p.debug_mode},
        {"translate", p.translate},
        {"detect_language", p.detect_language},
        {"diarize", p.diarize},
        {"tinydiarize", p.tinydiarize},
        {"split_on_word", p.split_on_word},
        {"no_fallback", p.no_fallback},
        {"output_txt", p.output_txt},
        {"output_vtt", p.output_vtt},
        {"output_srt", p.output_srt},
        {"output_wts", p.output_wts},
        {"output_csv", p.output_csv},
        {"output_jsn", p.output_jsn},
        {"output_jsn_full", p.output_jsn_full},
        {"output_lrc", p.output_lrc},
        {"no_prints", p.no_prints},
        {"verbose", p.verbose},
        {"print_special", p.print_special},
        {"print_colors", p.print_colors},
        {"print_confidence", p.print_confidence},
        {"print_progress", p.print_progress},
        {"no_timestamps", p.no_timestamps},
        {"log_score", p.log_score},
        {"use_gpu", p.use_gpu},
        {"flash_attn", p.flash_attn},
        {"gpu_device", p.gpu_device},
        {"gpu_backend", p.gpu_backend},
        {"suppress_nst", p.suppress_nst},
        {"carry_initial_prompt", p.carry_initial_prompt},
        {"language", p.language},
        {"prompt", p.prompt},
        {"ask", p.ask},
        {"font_path", p.font_path},
        {"model", p.model},
        {"model_quant", p.model_quant},
        {"grammar", p.grammar},
        {"grammar_rule", p.grammar_rule},
        {"tdrz_speaker_turn", p.tdrz_speaker_turn},
        {"suppress_regex", p.suppress_regex},
        {"openvino_encode_device", p.openvino_encode_device},
        {"dtw", p.dtw},
        {"vad", p.vad},
        {"vad_model", p.vad_model},
        {"vad_threshold", p.vad_threshold},
        {"vad_min_speech_duration_ms", p.vad_min_speech_duration_ms},
        {"vad_min_silence_duration_ms", p.vad_min_silence_duration_ms},
        {"vad_max_speech_duration_s", p.vad_max_speech_duration_s},
        {"vad_speech_pad_ms", p.vad_speech_pad_ms},
        {"vad_samples_overlap", p.vad_samples_overlap},
        {"vad_stitch", p.vad_stitch},
        {"backend", p.backend},
        {"source_lang", p.source_lang},
        {"target_lang", p.target_lang},
        {"punctuation", p.punctuation},
        {"punc_model", p.punc_model},
        {"flush_after", p.flush_after},
        {"show_alternatives", p.show_alternatives},
        {"n_alternatives", p.n_alternatives},
        {"aligner_model", p.aligner_model},
        {"force_aligner", p.force_aligner},
        {"no_auto_aligner", p.no_auto_aligner},
        {"max_new_tokens", p.max_new_tokens},
        {"chunk_seconds", p.chunk_seconds},
        {"lid_backend", p.lid_backend},
        {"lid_model", p.lid_model},
        {"diarize_method", p.diarize_method},
        {"sherpa_bin", p.sherpa_bin},
        {"sherpa_segment_model", p.sherpa_segment_model},
        {"sherpa_embedding_model", p.sherpa_embedding_model},
        {"sherpa_num_clusters", p.sherpa_num_clusters},
        {"stream", p.stream},
        {"mic", p.mic},
        {"stream_continuous", p.stream_continuous},
        {"stream_monitor", p.stream_monitor},
        {"server", p.server},
        {"server_host", p.server_host},
        {"server_port", p.server_port},
        {"server_api_keys", p.server_api_keys},
        {"stream_step_ms", p.stream_step_ms},
        {"stream_length_ms", p.stream_length_ms},
        {"stream_keep_ms", p.stream_keep_ms},
        {"auto_download", p.auto_download},
        {"dry_run_resolve", p.dry_run_resolve},
        {"dry_run_ignore_cache", p.dry_run_ignore_cache},
        {"cache_dir", p.cache_dir},
        {"tts_text", p.tts_text},
        {"tts_output", p.tts_output},
        {"tts_voice", p.tts_voice},
        {"tts_steps", p.tts_steps},
        {"tts_codec_model", p.tts_codec_model},
        {"tts_codec_quant", p.tts_codec_quant},
        {"tts_ref_text", p.tts_ref_text},
        {"tts_instruct", p.tts_instruct},
        {"tts_trim_silence", p.tts_trim_silence},
        {"tts_voice_dir", p.tts_voice_dir},
        {"tts_max_input_chars", p.tts_max_input_chars},
        {"tts_speed", p.tts_speed},
        {"server_cors_origin", p.server_cors_origin},
        {"text_input", p.text_input},
        {"translate_max_tokens", p.translate_max_tokens},
        {"translate_source_lang", p.translate_source_lang},
        {"translate_target_lang", p.translate_target_lang},
        {"fname_inp", p.fname_inp},
        {"fname_out", p.fname_out},
    };
    return j;
}

static whisper_params params_from_json(const json& j) {
    whisper_params p;
    p.n_threads = j.value("n_threads", p.n_threads);
    p.n_processors = j.value("n_processors", p.n_processors);
    p.offset_t_ms = j.value("offset_t_ms", p.offset_t_ms);
    p.offset_n = j.value("offset_n", p.offset_n);
    p.duration_ms = j.value("duration_ms", p.duration_ms);
    p.progress_step = j.value("progress_step", p.progress_step);
    p.max_context = j.value("max_context", p.max_context);
    p.max_len = j.value("max_len", p.max_len);
    p.split_on_punct = j.value("split_on_punct", p.split_on_punct);
    p.best_of = j.value("best_of", p.best_of);
    p.beam_size = j.value("beam_size", p.beam_size);
    p.audio_ctx = j.value("audio_ctx", p.audio_ctx);
    p.word_thold = j.value("word_thold", p.word_thold);
    p.entropy_thold = j.value("entropy_thold", p.entropy_thold);
    p.logprob_thold = j.value("logprob_thold", p.logprob_thold);
    p.no_speech_thold = j.value("no_speech_thold", p.no_speech_thold);
    p.grammar_penalty = j.value("grammar_penalty", p.grammar_penalty);
    p.temperature = j.value("temperature", p.temperature);
    p.temperature_inc = j.value("temperature_inc", p.temperature_inc);
    p.debug_mode = j.value("debug_mode", p.debug_mode);
    p.translate = j.value("translate", p.translate);
    p.detect_language = j.value("detect_language", p.detect_language);
    p.diarize = j.value("diarize", p.diarize);
    p.tinydiarize = j.value("tinydiarize", p.tinydiarize);
    p.split_on_word = j.value("split_on_word", p.split_on_word);
    p.no_fallback = j.value("no_fallback", p.no_fallback);
    p.output_txt = j.value("output_txt", p.output_txt);
    p.output_vtt = j.value("output_vtt", p.output_vtt);
    p.output_srt = j.value("output_srt", p.output_srt);
    p.output_wts = j.value("output_wts", p.output_wts);
    p.output_csv = j.value("output_csv", p.output_csv);
    p.output_jsn = j.value("output_jsn", p.output_jsn);
    p.output_jsn_full = j.value("output_jsn_full", p.output_jsn_full);
    p.output_lrc = j.value("output_lrc", p.output_lrc);
    p.no_prints = j.value("no_prints", p.no_prints);
    p.verbose = j.value("verbose", p.verbose);
    p.print_special = j.value("print_special", p.print_special);
    p.print_colors = j.value("print_colors", p.print_colors);
    p.print_confidence = j.value("print_confidence", p.print_confidence);
    p.print_progress = j.value("print_progress", p.print_progress);
    p.no_timestamps = j.value("no_timestamps", p.no_timestamps);
    p.log_score = j.value("log_score", p.log_score);
    p.use_gpu = j.value("use_gpu", p.use_gpu);
    p.flash_attn = j.value("flash_attn", p.flash_attn);
    p.gpu_device = j.value("gpu_device", p.gpu_device);
    p.gpu_backend = j.value("gpu_backend", p.gpu_backend);
    p.suppress_nst = j.value("suppress_nst", p.suppress_nst);
    p.carry_initial_prompt = j.value("carry_initial_prompt", p.carry_initial_prompt);
    p.language = j.value("language", p.language);
    p.prompt = j.value("prompt", p.prompt);
    p.ask = j.value("ask", p.ask);
    p.font_path = j.value("font_path", p.font_path);
    p.model = j.value("model", p.model);
    p.model_quant = j.value("model_quant", p.model_quant);
    p.grammar = j.value("grammar", p.grammar);
    p.grammar_rule = j.value("grammar_rule", p.grammar_rule);
    p.tdrz_speaker_turn = j.value("tdrz_speaker_turn", p.tdrz_speaker_turn);
    p.suppress_regex = j.value("suppress_regex", p.suppress_regex);
    p.openvino_encode_device = j.value("openvino_encode_device", p.openvino_encode_device);
    p.dtw = j.value("dtw", p.dtw);
    p.vad = j.value("vad", p.vad);
    p.vad_model = j.value("vad_model", p.vad_model);
    p.vad_threshold = j.value("vad_threshold", p.vad_threshold);
    p.vad_min_speech_duration_ms = j.value("vad_min_speech_duration_ms", p.vad_min_speech_duration_ms);
    p.vad_min_silence_duration_ms = j.value("vad_min_silence_duration_ms", p.vad_min_silence_duration_ms);
    p.vad_max_speech_duration_s = j.value("vad_max_speech_duration_s", p.vad_max_speech_duration_s);
    p.vad_speech_pad_ms = j.value("vad_speech_pad_ms", p.vad_speech_pad_ms);
    p.vad_samples_overlap = j.value("vad_samples_overlap", p.vad_samples_overlap);
    p.vad_stitch = j.value("vad_stitch", p.vad_stitch);
    p.backend = j.value("backend", p.backend);
    p.source_lang = j.value("source_lang", p.source_lang);
    p.target_lang = j.value("target_lang", p.target_lang);
    p.punctuation = j.value("punctuation", p.punctuation);
    p.punc_model = j.value("punc_model", p.punc_model);
    p.flush_after = j.value("flush_after", p.flush_after);
    p.show_alternatives = j.value("show_alternatives", p.show_alternatives);
    p.n_alternatives = j.value("n_alternatives", p.n_alternatives);
    p.aligner_model = j.value("aligner_model", p.aligner_model);
    p.force_aligner = j.value("force_aligner", p.force_aligner);
    p.no_auto_aligner = j.value("no_auto_aligner", p.no_auto_aligner);
    p.max_new_tokens = j.value("max_new_tokens", p.max_new_tokens);
    p.chunk_seconds = j.value("chunk_seconds", p.chunk_seconds);
    p.lid_backend = j.value("lid_backend", p.lid_backend);
    p.lid_model = j.value("lid_model", p.lid_model);
    p.diarize_method = j.value("diarize_method", p.diarize_method);
    p.sherpa_bin = j.value("sherpa_bin", p.sherpa_bin);
    p.sherpa_segment_model = j.value("sherpa_segment_model", p.sherpa_segment_model);
    p.sherpa_embedding_model = j.value("sherpa_embedding_model", p.sherpa_embedding_model);
    p.sherpa_num_clusters = j.value("sherpa_num_clusters", p.sherpa_num_clusters);
    p.stream = j.value("stream", p.stream);
    p.mic = j.value("mic", p.mic);
    p.stream_continuous = j.value("stream_continuous", p.stream_continuous);
    p.stream_monitor = j.value("stream_monitor", p.stream_monitor);
    p.server = j.value("server", p.server);
    p.server_host = j.value("server_host", p.server_host);
    p.server_port = j.value("server_port", p.server_port);
    p.server_api_keys = j.value("server_api_keys", p.server_api_keys);
    p.stream_step_ms = j.value("stream_step_ms", p.stream_step_ms);
    p.stream_length_ms = j.value("stream_length_ms", p.stream_length_ms);
    p.stream_keep_ms = j.value("stream_keep_ms", p.stream_keep_ms);
    p.auto_download = j.value("auto_download", p.auto_download);
    p.dry_run_resolve = j.value("dry_run_resolve", p.dry_run_resolve);
    p.dry_run_ignore_cache = j.value("dry_run_ignore_cache", p.dry_run_ignore_cache);
    p.cache_dir = j.value("cache_dir", p.cache_dir);
    p.tts_text = j.value("tts_text", p.tts_text);
    p.tts_output = j.value("tts_output", p.tts_output);
    p.tts_voice = j.value("tts_voice", p.tts_voice);
    p.tts_steps = j.value("tts_steps", p.tts_steps);
    p.tts_codec_model = j.value("tts_codec_model", p.tts_codec_model);
    p.tts_codec_quant = j.value("tts_codec_quant", p.tts_codec_quant);
    p.tts_ref_text = j.value("tts_ref_text", p.tts_ref_text);
    p.tts_instruct = j.value("tts_instruct", p.tts_instruct);
    p.tts_trim_silence = j.value("tts_trim_silence", p.tts_trim_silence);
    p.tts_voice_dir = j.value("tts_voice_dir", p.tts_voice_dir);
    p.tts_max_input_chars = j.value("tts_max_input_chars", p.tts_max_input_chars);
    p.tts_speed = j.value("tts_speed", p.tts_speed);
    p.server_cors_origin = j.value("server_cors_origin", p.server_cors_origin);
    p.text_input = j.value("text_input", p.text_input);
    p.translate_max_tokens = j.value("translate_max_tokens", p.translate_max_tokens);
    p.translate_source_lang = j.value("translate_source_lang", p.translate_source_lang);
    p.translate_target_lang = j.value("translate_target_lang", p.translate_target_lang);
    if (j.contains("fname_inp"))
        p.fname_inp = j.at("fname_inp").get<std::vector<std::string>>();
    if (j.contains("fname_out"))
        p.fname_out = j.at("fname_out").get<std::vector<std::string>>();
    return p;
}

// ────────────────────────── crispasr_segment <-> json ──────────────────────────

static json segment_to_json(const crispasr_segment& s) {
    json js = {
        {"text", s.text},
        {"t0", s.t0},
        {"t1", s.t1},
        {"speaker", s.speaker},
        {"speaker_turn_next", s.speaker_turn_next},
    };
    if (!s.words.empty()) {
        json jw = json::array();
        for (const auto& w : s.words) {
            jw.push_back({{"text", w.text}, {"t0", w.t0}, {"t1", w.t1}});
        }
        js["words"] = std::move(jw);
    }
    if (!s.tokens.empty()) {
        json jt = json::array();
        for (const auto& t : s.tokens) {
            json one = {
                {"text", t.text},
                {"confidence", t.confidence},
                {"t0", t.t0},
                {"t1", t.t1},
                {"id", t.id},
                {"t_dtw", t.t_dtw},
                {"is_special", t.is_special},
            };
            if (!t.alts.empty()) {
                json ja = json::array();
                for (const auto& a : t.alts) {
                    ja.push_back({{"text", a.text}, {"prob", a.prob}, {"id", a.id}});
                }
                one["alts"] = std::move(ja);
            }
            jt.push_back(std::move(one));
        }
        js["tokens"] = std::move(jt);
    }
    return js;
}

static crispasr_segment segment_from_json(const json& js) {
    crispasr_segment s;
    s.text = js.value("text", std::string{});
    s.t0 = js.value("t0", (int64_t)0);
    s.t1 = js.value("t1", (int64_t)0);
    s.speaker = js.value("speaker", std::string{});
    s.speaker_turn_next = js.value("speaker_turn_next", false);
    if (js.contains("words")) {
        for (const auto& jw : js.at("words")) {
            crispasr_word w;
            w.text = jw.value("text", std::string{});
            w.t0 = jw.value("t0", (int64_t)0);
            w.t1 = jw.value("t1", (int64_t)0);
            s.words.push_back(std::move(w));
        }
    }
    if (js.contains("tokens")) {
        for (const auto& jt : js.at("tokens")) {
            crispasr_token t;
            t.text = jt.value("text", std::string{});
            t.confidence = jt.value("confidence", -1.0f);
            t.t0 = jt.value("t0", (int64_t)-1);
            t.t1 = jt.value("t1", (int64_t)-1);
            t.id = jt.value("id", (int32_t)-1);
            t.t_dtw = jt.value("t_dtw", (int64_t)-1);
            t.is_special = jt.value("is_special", false);
            if (jt.contains("alts")) {
                for (const auto& ja : jt.at("alts")) {
                    crispasr_token_alt a;
                    a.text = ja.value("text", std::string{});
                    a.prob = ja.value("prob", 0.0f);
                    a.id = ja.value("id", (int32_t)-1);
                    t.alts.push_back(std::move(a));
                }
            }
            s.tokens.push_back(std::move(t));
        }
    }
    return s;
}

static json segments_to_json(const std::vector<crispasr_segment>& segs) {
    json out = json::array();
    for (const auto& s : segs)
        out.push_back(segment_to_json(s));
    return out;
}

static std::vector<crispasr_segment> segments_from_json(const json& j) {
    std::vector<crispasr_segment> out;
    if (!j.is_array())
        return out;
    out.reserve(j.size());
    for (const auto& je : j)
        out.push_back(segment_from_json(je));
    return out;
}

// ──────────────────────────── WorkerSession (parent) ────────────────────────────

WorkerSession::WorkerSession(const char* argv0, std::vector<std::string> extra_argv)
    : argv0_(argv0 ? argv0 : ""), extra_argv_(std::move(extra_argv)) {
}

WorkerSession::~WorkerSession() {
    shutdown();
}

bool WorkerSession::is_alive() const {
#if defined(_WIN32)
    return false;
#else
    // try_lock so /health doesn't stall behind a cold load (~1s fork+exec
    // +model). If another caller is mid-IPC, fall back to the cached pid_:
    // the in-flight call will detect death itself (recv_frame EofMidFrame
    // → kill_worker_locked) and clear pid_ before releasing the lock, so
    // the next is_alive() observes the truth. The race we're protecting
    // against is two parallel out-of-band-death detectors closing the
    // same fd_; that requires both passing the lock.
    std::unique_lock<std::mutex> lock(io_mutex_, std::try_to_lock);
    if (!lock.owns_lock())
        return pid_ > 0;
    if (pid_ <= 0)
        return false;
    int status = 0;
    pid_t r = ::waitpid(pid_, &status, WNOHANG);
    if (r == pid_ || r < 0) {
        // r == pid_: child died out-of-band (external SIGKILL, SEGV,
        // GGML_ABORT, OOM-killer) and we just reaped it.
        // r < 0:    ECHILD — someone else already reaped (e.g. an init
        //           subreaper or a parallel is_alive() that won the race).
        // Either way the worker is gone — drop our handle so the next
        // ensure_loaded() respawns instead of dispatching to a stale fd.
        WorkerSession* self = const_cast<WorkerSession*>(this);
        if (self->fd_ >= 0) {
            ::close(self->fd_);
            self->fd_ = -1;
        }
        self->pid_ = -1;
        self->loaded_ok_ = false;
        if (r == pid_) {
            std::fprintf(stderr, "crispasr-worker: detected external child exit (status=0x%x)\n", status);
        }
        return false;
    }
    return true;
#endif
}

void WorkerSession::kill_worker_locked() {
#if !defined(_WIN32)
    if (pid_ > 0) {
        // SIGKILL is the headline feature: it tears down the CUDA primary
        // context and reclaims ALL VRAM, no graceful-shutdown handshake
        // needed.
        ::kill(pid_, SIGKILL);
        int wstat = 0;
        ::waitpid(pid_, &wstat, 0);
        std::fprintf(stderr, "crispasr-worker: killed worker pid=%d (wstat=0x%x)\n", (int)pid_, wstat);
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
#endif
    pid_ = -1;
    loaded_ok_ = false;
    sample_rate_ = 0;
    capabilities_ = 0;
    backend_name_.clear();
    resolved_model_.clear();
    loaded_cfg_ = {};
}

void WorkerSession::shutdown() {
    std::lock_guard<std::mutex> lock(io_mutex_);
    kill_worker_locked();
}

bool WorkerSession::send_load_req_locked(const WorkerLoadConfig& cfg) {
    json req = {
        {"backend", cfg.backend},
        {"model", cfg.model},
        {"params", params_to_json(cfg.init_params)},
    };
    IpcError e = send_frame(fd_, WorkerFrame::LOAD_REQ, 0, req.dump());
    if (e != IpcError::OK) {
        last_error_ = std::string("LOAD_REQ send failed: ") + ipc_error_str(e);
        return false;
    }
    FrameHeader hdr{};
    std::vector<uint8_t> payload;
    e = recv_frame(fd_, &hdr, &payload);
    if (e != IpcError::OK) {
        last_error_ = std::string("LOAD_RESP recv failed: ") + ipc_error_str(e);
        return false;
    }
    if (hdr.type != static_cast<uint32_t>(WorkerFrame::LOAD_RESP)) {
        last_error_ = std::string("expected LOAD_RESP, got type=0x") + std::to_string(hdr.type);
        return false;
    }
    json resp;
    try {
        resp = json::parse(std::string(payload.begin(), payload.end()));
    } catch (const std::exception& ex) {
        last_error_ = std::string("LOAD_RESP json parse: ") + ex.what();
        return false;
    }
    if (!resp.value("ok", false)) {
        last_error_ = std::string("worker load failed: ") + resp.value("error", std::string{"(no msg)"});
        return false;
    }
    backend_name_ = resp.value("backend_name", std::string{});
    capabilities_ = resp.value("capabilities", (uint32_t)0);
    sample_rate_ = resp.value("sample_rate", 0);
    resolved_model_ = resp.value("model", cfg.model);
    return true;
}

bool WorkerSession::ensure_loaded(const WorkerLoadConfig& cfg) {
#if defined(_WIN32)
    (void)cfg;
    last_error_ = "worker isolation not supported on Windows";
    return false;
#else
    std::lock_guard<std::mutex> lock(io_mutex_);

    // If config matches and worker is alive, we're good. Match on
    // backend + model + the few load-time GPU-affecting flags. Per-call
    // overrides (language/temperature/etc.) ride along in the request
    // params, so they don't gate the cached worker.
    if (pid_ > 0 && loaded_ok_ && loaded_cfg_.backend == cfg.backend && loaded_cfg_.model == cfg.model &&
        loaded_cfg_.use_gpu == cfg.use_gpu && loaded_cfg_.gpu_device == cfg.gpu_device &&
        loaded_cfg_.flash_attn == cfg.flash_attn && loaded_cfg_.model_quant == cfg.model_quant) {
        return true;
    }

    if (pid_ > 0)
        kill_worker_locked();

    pid_t child = spawn_worker(argv0_.c_str(), extra_argv_, &fd_);
    if (child < 0) {
        last_error_ = "spawn_worker failed";
        return false;
    }
    pid_ = child;

    // Expect HELLO before LOAD_REQ.
    FrameHeader hdr{};
    std::vector<uint8_t> payload;
    IpcError e = recv_frame(fd_, &hdr, &payload);
    if (e != IpcError::OK || hdr.type != static_cast<uint32_t>(WorkerFrame::HELLO)) {
        last_error_ = std::string("worker HELLO failed: ") + ipc_error_str(e);
        kill_worker_locked();
        return false;
    }
    std::fprintf(stderr, "crispasr-worker: child pid=%d HELLO: %.*s\n", (int)pid_, (int)payload.size(),
                 (const char*)payload.data());

    if (!send_load_req_locked(cfg)) {
        kill_worker_locked();
        return false;
    }
    loaded_cfg_ = cfg;
    loaded_ok_ = true;
    return true;
#endif
}

std::vector<crispasr_segment> WorkerSession::do_transcribe_locked(WorkerFrame frame_type,
                                                                  const std::vector<uint8_t>& payload) {
    if (pid_ <= 0 || fd_ < 0 || !loaded_ok_) {
        last_error_ = "worker not ready (call ensure_loaded first)";
        return {};
    }

    uint32_t req_id = next_req_id_.fetch_add(1);
    // Note: the zero-copy fast path bypasses this overload by calling
    // send_frame_audio directly; this code path is for stereo (two audio
    // buffers) where pre-packing into one vector keeps the IPC simple.
    IpcError e = send_frame(fd_, frame_type, req_id, payload);
    if (e != IpcError::OK) {
        last_error_ = std::string("TRANSCRIBE_REQ send failed: ") + ipc_error_str(e);
        kill_worker_locked();
        return {};
    }

    FrameHeader hdr{};
    std::vector<uint8_t> resp;
    e = recv_frame(fd_, &hdr, &resp);
    if (e != IpcError::OK) {
        last_error_ = std::string("TRANSCRIBE_RESP recv failed: ") + ipc_error_str(e);
        kill_worker_locked();
        return {};
    }
    if (hdr.type == static_cast<uint32_t>(WorkerFrame::ERR_RESP)) {
        try {
            json j = json::parse(std::string(resp.begin(), resp.end()));
            last_error_ = j.value("error", std::string{"unknown worker error"});
        } catch (...) {
            last_error_ = "worker reported error (unparseable)";
        }
        return {};
    }
    if (hdr.type != static_cast<uint32_t>(WorkerFrame::TRANSCRIBE_RESP)) {
        last_error_ = std::string("expected TRANSCRIBE_RESP, got 0x") + std::to_string(hdr.type);
        kill_worker_locked();
        return {};
    }
    try {
        json j = json::parse(std::string(resp.begin(), resp.end()));
        return segments_from_json(j.value("segments", json::array()));
    } catch (const std::exception& ex) {
        last_error_ = std::string("TRANSCRIBE_RESP parse: ") + ex.what();
        return {};
    }
}

std::vector<crispasr_segment> WorkerSession::transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                                        const whisper_params& params) {
    // Per-call overrides only, not the whole 120-field whisper_params. The
    // worker has init_params from LOAD_REQ; per-request handlers in the
    // server only mutate {language, temperature, prompt}, so re-shipping
    // the rest is wasted JSON serialization on every call. See
    // crispasr_server.cpp `rp.{language,temperature,prompt} =` sites.
    json meta = {
        {"n_samples", n_samples},
        {"t_offset_cs", t_offset_cs},
        {"o", {
            {"language", params.language},
            {"temperature", params.temperature},
            {"prompt", params.prompt},
        }},
    };
    std::string meta_str = meta.dump();
    std::lock_guard<std::mutex> lock(io_mutex_);
    if (pid_ <= 0 || fd_ < 0 || !loaded_ok_) {
        last_error_ = "worker not ready (call ensure_loaded first)";
        return {};
    }
    // Zero-copy send: header + json + raw f32 samples handed to writev() as
    // separate iovs. Saves a 1.9 MiB memcpy on the parent's hot path vs the
    // pack_audio_payload route. Wire format is identical, so the worker's
    // unpack_audio_payload still parses it correctly.
    uint32_t req_id = next_req_id_.fetch_add(1);
    IpcError e = send_frame_audio(fd_, WorkerFrame::TRANSCRIBE_REQ, req_id, meta_str, samples, (size_t)n_samples);
    if (e != IpcError::OK) {
        last_error_ = std::string("TRANSCRIBE_REQ send failed: ") + ipc_error_str(e);
        kill_worker_locked();
        return {};
    }
    // Reuse the existing recv path by passing an empty payload (already sent).
    // do_transcribe_locked's send_frame would re-send if payload non-empty, so
    // we replicate just the recv portion here.
    FrameHeader hdr{};
    std::vector<uint8_t> resp;
    e = recv_frame(fd_, &hdr, &resp);
    if (e != IpcError::OK) {
        last_error_ = std::string("TRANSCRIBE_RESP recv failed: ") + ipc_error_str(e);
        kill_worker_locked();
        return {};
    }
    if (hdr.type == static_cast<uint32_t>(WorkerFrame::ERR_RESP)) {
        try {
            json j = json::parse(std::string(resp.begin(), resp.end()));
            last_error_ = j.value("error", std::string{"unknown worker error"});
        } catch (...) {
            last_error_ = "worker reported error (unparseable)";
        }
        return {};
    }
    if (hdr.type != static_cast<uint32_t>(WorkerFrame::TRANSCRIBE_RESP)) {
        last_error_ = std::string("expected TRANSCRIBE_RESP, got 0x") + std::to_string(hdr.type);
        kill_worker_locked();
        return {};
    }
    try {
        json j = json::parse(std::string(resp.begin(), resp.end()));
        return segments_from_json(j.value("segments", json::array()));
    } catch (const std::exception& ex) {
        last_error_ = std::string("TRANSCRIBE_RESP parse: ") + ex.what();
        return {};
    }
}

std::vector<crispasr_segment> WorkerSession::transcribe_stereo(const float* left, const float* right,
                                                               int n_samples_per_channel, int64_t t_offset_cs,
                                                               const whisper_params& params) {
    // Same lean override schema as mono — see WorkerSession::transcribe.
    json meta = {
        {"n_samples", n_samples_per_channel},
        {"t_offset_cs", t_offset_cs},
        {"o", {
            {"language", params.language},
            {"temperature", params.temperature},
            {"prompt", params.prompt},
        }},
    };
    auto payload = pack_stereo_payload(meta.dump(), left, right, (size_t)n_samples_per_channel);
    std::lock_guard<std::mutex> lock(io_mutex_);
    return do_transcribe_locked(WorkerFrame::TRANSCRIBE_STEREO_REQ, payload);
}

std::vector<float> WorkerSession::synthesize(const std::string& text, const whisper_params& params) {
    std::lock_guard<std::mutex> lock(io_mutex_);
    if (pid_ <= 0 || fd_ < 0 || !loaded_ok_) {
        last_error_ = "worker not ready (call ensure_loaded first)";
        return {};
    }
    // Lean override schema: only the per-call mutable TTS fields (voice
    // selection, VoiceDesign instructions). See `rp.tts_voice` /
    // `rp.tts_instruct` mutations in crispasr_server.cpp.
    json req = {
        {"text", text},
        {"o", {
            {"tts_voice", params.tts_voice},
            {"tts_instruct", params.tts_instruct},
            {"temperature", params.temperature},
        }},
    };
    uint32_t req_id = next_req_id_.fetch_add(1);
    IpcError e = send_frame(fd_, WorkerFrame::SYNTH_REQ, req_id, req.dump());
    if (e != IpcError::OK) {
        last_error_ = std::string("SYNTH_REQ send failed: ") + ipc_error_str(e);
        kill_worker_locked();
        return {};
    }

    FrameHeader hdr{};
    std::vector<uint8_t> resp;
    e = recv_frame(fd_, &hdr, &resp);
    if (e != IpcError::OK) {
        last_error_ = std::string("SYNTH_RESP recv failed: ") + ipc_error_str(e);
        kill_worker_locked();
        return {};
    }
    if (hdr.type == static_cast<uint32_t>(WorkerFrame::ERR_RESP)) {
        try {
            json j = json::parse(std::string(resp.begin(), resp.end()));
            last_error_ = j.value("error", std::string{"unknown worker error"});
        } catch (...) {
            last_error_ = "worker reported error (unparseable)";
        }
        return {};
    }
    if (hdr.type != static_cast<uint32_t>(WorkerFrame::SYNTH_RESP)) {
        last_error_ = std::string("expected SYNTH_RESP, got 0x") + std::to_string(hdr.type);
        kill_worker_locked();
        return {};
    }

    std::string meta_str;
    std::vector<float> pcm;
    if (!unpack_audio_payload(resp, &meta_str, &pcm)) {
        last_error_ = "SYNTH_RESP unpack failed";
        return {};
    }
    return pcm;
}

// ──────────────────────────── run_worker_loop (child) ────────────────────────────

#if defined(_WIN32)

int run_worker_loop(int /*fd*/) {
    std::fprintf(stderr, "crispasr-worker: worker isolation not supported on Windows\n");
    return 1;
}

#else

int run_worker_loop(int fd) {
    setvbuf(stderr, nullptr, _IONBF, 0);

#if defined(__linux__)
    // PR_SET_PDEATHSIG: the kernel sends SIGTERM if the parent dies. Belt-
    // and-braces against an orphaned worker holding VRAM after the parent
    // crashes. (We can't fall back to `if (getppid() == 1) bail`: in a
    // container the parent IS pid 1, so that check fires on every boot.)
    if (prctl(PR_SET_PDEATHSIG, SIGTERM) != 0) {
        std::fprintf(stderr, "crispasr-worker: prctl(PR_SET_PDEATHSIG) failed: %s (continuing)\n",
                     std::strerror(errno));
    }
#endif

    std::fprintf(stderr, "crispasr-worker[%d]: alive on fd=%d ppid=%d\n", (int)getpid(), fd, (int)getppid());

    json hello = {
        {"pid", (int)getpid()},
        {"role", "crispasr-worker"},
    };
    if (send_frame(fd, WorkerFrame::HELLO, 0, hello.dump()) != IpcError::OK) {
        std::fprintf(stderr, "crispasr-worker: HELLO send failed; bailing\n");
        return 2;
    }

    std::unique_ptr<CrispasrBackend> backend;
    whisper_params init_params; // params from the most recent successful LOAD_REQ
    std::string backend_name;

    auto err_resp = [&](uint32_t req_id, const std::string& msg) {
        json j = {{"error", msg}};
        return send_frame(fd, WorkerFrame::ERR_RESP, req_id, j.dump());
    };

    while (true) {
        FrameHeader hdr{};
        std::vector<uint8_t> payload;
        IpcError e = recv_frame(fd, &hdr, &payload);
        if (e == IpcError::EofClean) {
            std::fprintf(stderr, "crispasr-worker: parent EOF, exiting cleanly\n");
            return 0;
        }
        if (e != IpcError::OK) {
            std::fprintf(stderr, "crispasr-worker: recv_frame failed: %s\n", ipc_error_str(e));
            return 3;
        }

        switch (static_cast<WorkerFrame>(hdr.type)) {
        case WorkerFrame::SHUTDOWN: {
            std::fprintf(stderr, "crispasr-worker: SHUTDOWN, exiting\n");
            return 0;
        }
        case WorkerFrame::PING: {
            send_frame(fd, WorkerFrame::PONG, hdr.req_id, payload);
            break;
        }
        case WorkerFrame::LOAD_REQ: {
            std::string err_msg;
            bool ok = false;
            backend_name.clear();

            try {
                json req = json::parse(std::string(payload.begin(), payload.end()));
                backend_name = req.value("backend", std::string{});
                init_params = params_from_json(req.value("params", json::object()));

                // The parent has already resolved + downloaded the model;
                // init_params.model carries the local path.
                auto nb = crispasr_create_backend(backend_name);
                if (!nb) {
                    err_msg = "crispasr_create_backend('" + backend_name + "') returned null";
                } else if (!nb->init(init_params)) {
                    err_msg = "backend->init() failed (" + backend_name + ", " + init_params.model + ")";
                } else {
                    backend = std::move(nb);
                    ok = true;
                }
            } catch (const std::exception& ex) {
                err_msg = std::string("LOAD_REQ parse/init: ") + ex.what();
            }

            uint32_t caps = (ok && backend) ? backend->capabilities() : 0;
            const char* nm = (ok && backend) ? backend->name() : "";
            json resp = {
                {"ok", ok},
                {"error", err_msg},
                {"backend_name", std::string(nm)},
                {"capabilities", caps},
                {"model", init_params.model},
                {"sample_rate", 0}, // populated below if backend exposes one
            };
            if (send_frame(fd, WorkerFrame::LOAD_RESP, hdr.req_id, resp.dump()) != IpcError::OK) {
                std::fprintf(stderr, "crispasr-worker: LOAD_RESP send failed\n");
                return 4;
            }
            break;
        }
        case WorkerFrame::TRANSCRIBE_REQ: {
            if (!backend) {
                err_resp(hdr.req_id, "no model loaded (LOAD_REQ first)");
                break;
            }
            std::string meta_str;
            std::vector<float> samples;
            if (!unpack_audio_payload(payload, &meta_str, &samples)) {
                err_resp(hdr.req_id, "TRANSCRIBE_REQ unpack failed");
                break;
            }
            json meta;
            try {
                meta = json::parse(meta_str);
            } catch (const std::exception& ex) {
                err_resp(hdr.req_id, std::string("TRANSCRIBE_REQ meta parse: ") + ex.what());
                break;
            }
            int n_samples = meta.value("n_samples", 0);
            int64_t t_offset_cs = meta.value("t_offset_cs", (int64_t)0);
            whisper_params rp = init_params;
            // Lean per-call override schema (preferred); fall back to the
            // legacy "params" full-blob if a pre-update parent is talking to
            // a post-update worker (mismatched bin via /load swap, etc).
            if (meta.contains("o")) {
                const auto& ov = meta.at("o");
                rp.language = ov.value("language", rp.language);
                rp.temperature = ov.value("temperature", rp.temperature);
                rp.prompt = ov.value("prompt", rp.prompt);
            } else if (meta.contains("params")) {
                rp = params_from_json(meta.at("params"));
            }
            if (n_samples != (int)samples.size()) {
                err_resp(hdr.req_id, "TRANSCRIBE_REQ n_samples mismatch");
                break;
            }
            std::vector<crispasr_segment> segs;
            try {
                segs = backend->transcribe(samples.data(), n_samples, t_offset_cs, rp);
            } catch (const std::exception& ex) {
                err_resp(hdr.req_id, std::string("transcribe threw: ") + ex.what());
                break;
            }
            json out = {{"segments", segments_to_json(segs)}};
            if (send_frame(fd, WorkerFrame::TRANSCRIBE_RESP, hdr.req_id, out.dump()) != IpcError::OK) {
                std::fprintf(stderr, "crispasr-worker: TRANSCRIBE_RESP send failed\n");
                return 5;
            }
            break;
        }
        case WorkerFrame::TRANSCRIBE_STEREO_REQ: {
            if (!backend) {
                err_resp(hdr.req_id, "no model loaded (LOAD_REQ first)");
                break;
            }
            std::string meta_str;
            std::vector<float> left, right;
            if (!unpack_stereo_payload(payload, &meta_str, &left, &right)) {
                err_resp(hdr.req_id, "TRANSCRIBE_STEREO_REQ unpack failed");
                break;
            }
            json meta;
            try {
                meta = json::parse(meta_str);
            } catch (const std::exception& ex) {
                err_resp(hdr.req_id, std::string("STEREO_REQ meta parse: ") + ex.what());
                break;
            }
            int n_per = meta.value("n_samples", 0);
            int64_t t_offset_cs = meta.value("t_offset_cs", (int64_t)0);
            whisper_params rp = init_params;
            if (meta.contains("o")) {
                const auto& ov = meta.at("o");
                rp.language = ov.value("language", rp.language);
                rp.temperature = ov.value("temperature", rp.temperature);
                rp.prompt = ov.value("prompt", rp.prompt);
            } else if (meta.contains("params")) {
                rp = params_from_json(meta.at("params"));
            }
            if (n_per != (int)left.size() || n_per != (int)right.size()) {
                err_resp(hdr.req_id, "STEREO_REQ n_samples mismatch");
                break;
            }
            std::vector<crispasr_segment> segs;
            try {
                segs = backend->transcribe_stereo(left.data(), right.data(), n_per, t_offset_cs, rp);
            } catch (const std::exception& ex) {
                err_resp(hdr.req_id, std::string("transcribe_stereo threw: ") + ex.what());
                break;
            }
            json out = {{"segments", segments_to_json(segs)}};
            if (send_frame(fd, WorkerFrame::TRANSCRIBE_RESP, hdr.req_id, out.dump()) != IpcError::OK) {
                std::fprintf(stderr, "crispasr-worker: TRANSCRIBE_RESP (stereo) send failed\n");
                return 5;
            }
            break;
        }
        case WorkerFrame::SYNTH_REQ: {
            if (!backend) {
                err_resp(hdr.req_id, "no model loaded (LOAD_REQ first)");
                break;
            }
            if (!(backend->capabilities() & CAP_TTS)) {
                err_resp(hdr.req_id, std::string("backend '") + backend->name() + "' has no CAP_TTS");
                break;
            }
            std::string text;
            whisper_params rp = init_params;
            try {
                json req = json::parse(std::string(payload.begin(), payload.end()));
                text = req.value("text", std::string{});
                if (req.contains("o")) {
                    const auto& ov = req.at("o");
                    rp.tts_voice = ov.value("tts_voice", rp.tts_voice);
                    rp.tts_instruct = ov.value("tts_instruct", rp.tts_instruct);
                    rp.temperature = ov.value("temperature", rp.temperature);
                } else if (req.contains("params")) {
                    rp = params_from_json(req.at("params"));
                }
            } catch (const std::exception& ex) {
                err_resp(hdr.req_id, std::string("SYNTH_REQ parse: ") + ex.what());
                break;
            }
            std::vector<float> pcm;
            try {
                pcm = backend->synthesize(text, rp);
            } catch (const std::exception& ex) {
                err_resp(hdr.req_id, std::string("synthesize threw: ") + ex.what());
                break;
            }
            json meta = {{"n_samples", (uint64_t)pcm.size()}};
            auto out = pack_audio_payload(meta.dump(), pcm.data(), pcm.size());
            if (send_frame(fd, WorkerFrame::SYNTH_RESP, hdr.req_id, out) != IpcError::OK) {
                std::fprintf(stderr, "crispasr-worker: SYNTH_RESP send failed\n");
                return 6;
            }
            break;
        }
        default:
            std::fprintf(stderr, "crispasr-worker: unexpected frame type=0x%x len=%u\n", hdr.type, hdr.len);
            break;
        }
    }
}

#endif // _WIN32

} // namespace crispasr
