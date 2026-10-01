// crispasr_backend_parakeet.cpp — adapter for nvidia/parakeet-tdt-0.6b-v3.
//
// Wraps parakeet_init_from_file + parakeet_transcribe_ex and converts the
// native parakeet_result into a std::vector<crispasr_segment>. One segment
// per transcribe() call, with word-level data attached (parakeet emits word
// timestamps for free via its TDT duration head).

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "whisper_params.h"

#include "parakeet.h"
#include "firered_vad.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <map>
#include <memory>

namespace {

class ParakeetBackend : public CrispasrBackend {
public:
    ParakeetBackend() = default;
    ~ParakeetBackend() override { ParakeetBackend::shutdown(); }

    const char* name() const override { return "parakeet"; }

    uint32_t capabilities() const override {
        // CAP_LANGUAGE_DETECT intentionally NOT declared: the parakeet
        // backend has no native LID code path. Declaring the cap would
        // disable the framework's pre-step LID gate
        // (crispasr_run.cpp:`!has_native_lid`), so users wanting LID
        // get nothing. With the cap absent, `-dl` correctly routes
        // through the whisper-tiny pre-step.
        return CAP_TIMESTAMPS_NATIVE | CAP_WORD_TIMESTAMPS | CAP_TOKEN_CONFIDENCE | CAP_FLASH_ATTN |
               CAP_PUNCTUATION_TOGGLE | CAP_TEMPERATURE | CAP_DIARIZE | CAP_PARALLEL_PROCESSORS | CAP_AUTO_DOWNLOAD | CAP_STREAM;
    }

    bool init(const whisper_params& p) override {
        parakeet_context_params cp = parakeet_context_default_params();
        cp.n_threads = p.n_threads;
        cp.use_flash = p.flash_attn;
        cp.verbosity = p.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(p);

        ctx_ = parakeet_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[parakeet]: failed to load model '%s'\n", p.model.c_str());
            return false;
        }
        return true;
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& params) override {
        std::vector<crispasr_segment> out;
        if (!ctx_)
            return out;

        // Sticky per-call sampling state. The setter just stores the
        // value on the parakeet_context, so subsequent transcribe calls
        // re-pick it up. We zero it on the first temp==0 call so a user
        // who toggles --temperature back off doesn't keep the previous
        // sampling state from a prior file.
        parakeet_set_temperature(ctx_, params.temperature, /*seed=*/0);

        parakeet_result* r = parakeet_transcribe_ex(ctx_, samples, n_samples, t_offset_cs);
        if (!r)
            return out;

        crispasr_segment seg;
        seg.t0 = t_offset_cs;
        seg.t1 = t_offset_cs;
        seg.text = r->text ? r->text : "";

        // Words
        seg.words.reserve(r->n_words);
        for (int i = 0; i < r->n_words; i++) {
            const auto& w = r->words[i];
            crispasr_word cw;
            cw.text = w.text;
            cw.t0 = w.t0;
            cw.t1 = w.t1;
            seg.words.push_back(std::move(cw));
        }

        // Tokens (sub-word pieces with their own timing + softmax confidence)
        seg.tokens.reserve(r->n_tokens);
        for (int i = 0; i < r->n_tokens; i++) {
            const auto& t = r->tokens[i];
            crispasr_token ct;
            ct.text = t.text;
            ct.id = t.id;
            ct.t0 = t.t0;
            ct.t1 = t.t1;
            ct.confidence = t.p;
            seg.tokens.push_back(std::move(ct));
        }

        // Segment t0/t1 bracketed by first/last word when available.
        if (!seg.words.empty()) {
            seg.t0 = seg.words.front().t0;
            seg.t1 = seg.words.back().t1;
        } else if (!seg.tokens.empty()) {
            seg.t0 = seg.tokens.front().t0;
            seg.t1 = seg.tokens.back().t1;
        }

        parakeet_result_free(r);
        out.push_back(std::move(seg));
        return out;
    }

    bool stream_begin(uint32_t sid, const crispasr_stream_opts& o) override {
        if (!ctx_)
            return false;
        stream_end(sid);
        parakeet_stream_params sp = parakeet_stream_default_params();
        sp.chunk_sec = o.chunk_sec;
        sp.right_sec = o.right_sec;
        sp.left_sec = o.left_sec;
        auto gate = std::make_unique<SpeechGate>();
        if (!vad_ && !vad_missing_)
            vad_missing_ = !(vad_ = load_vad());
        gate->activity_threshold = o.vad_threshold > 0.0f ? o.vad_threshold : activity_threshold();
        if (o.vad_threshold > 0.0f && vad_) {
            gate->vad = vad_;
            gate->threshold = o.vad_threshold;
            sp.speech_fn = &SpeechGate::call;
            sp.speech_user = gate.get();
        }
        parakeet_stream* st = parakeet_stream_begin(ctx_, sp);
        if (!st)
            return false;
        streams_[sid] = st;
        gates_[sid] = std::move(gate);
        return true;
    }

    crispasr_stream_delta stream_feed(uint32_t sid, const float* samples, int n_samples, bool final,
                                      int partial) override {
        crispasr_stream_delta d;
        auto it = streams_.find(sid);
        if (!ctx_ || it == streams_.end()) {
            d.error = "unknown stream";
            return d;
        }
        char* ptext = nullptr;
        parakeet_result* r = parakeet_stream_feed(ctx_, it->second, samples, n_samples, final ? 1 : 0, partial, &ptext);
        if (!r) {
            if (final)
                stream_end(sid);
            free(ptext);
            d.error = "stream decode failed";
            return d;
        }
        d.ok = true;
        parakeet_stream_stats(it->second, &d.n_encoded, &d.n_skipped);
        SpeechGate& g = *gates_[sid];
        d.gated = g.vad != nullptr;
        track_activity(g, samples, n_samples);
        d.audio_sec = (double)g.total / 16000.0;
        d.text = r->text ? r->text : "";
        for (int i = 0; i < r->n_words; i++) {
            d.words.push_back({r->words[i].text, r->words[i].t0, r->words[i].t1});
            // Committed words also count as speech: quiet talkers the VAD misses still get
            // transcribed (they just arrive ~1 s late).
            g.speech_end = std::max(g.speech_end, r->words[i].t1 / 100.0);
        }
        d.speech_end = g.speech_end;
        if (ptext)
            d.partial = ptext;
        free(ptext);
        parakeet_result_free(r);
        if (final)
            stream_end(sid);
        return d;
    }

    void stream_end(uint32_t sid) override {
        auto it = streams_.find(sid);
        if (it != streams_.end()) {
            parakeet_stream_free(it->second);
            streams_.erase(it);
        }
        gates_.erase(sid);
    }

    void shutdown() override {
        for (auto& [sid, st] : streams_)
            parakeet_stream_free(st);
        streams_.clear();
        gates_.clear();
        if (vad_) {
            firered_vad_free(vad_);
            vad_ = nullptr;
        }
        if (ctx_) {
            parakeet_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    // Speech gate for live streams: FireRedVAD (DFSMN, 588K params, CPU) over the chunk + lookahead.
    struct SpeechGate {
        firered_vad_context* vad = nullptr; // set only when gating
        float threshold = 0.0f;
        // Activity tracking (always on when a VAD model is present).
        float activity_threshold = 0.3f;
        std::vector<float> ctx; // last kActivityCtx samples, re-scanned with each feed
        int64_t total = 0;      // samples fed
        double speech_end = -1.0;
        static int call(const float* samples, int n, void* user) {
            auto* g = static_cast<SpeechGate*>(user);
            firered_vad_segment* segs = nullptr;
            int n_segs = 0;
            if (firered_vad_detect(g->vad, samples, n, &segs, &n_segs, g->threshold, 0.05f, 0.1f) != 0)
                return 1; // VAD failure: decode rather than drop speech
            free(segs);
            return n_segs > 0;
        }
    };

    static constexpr size_t kActivityCtx = 16000; // 1 s: FireRedVAD needs context to place edges

    // PARAKEET_STREAM_ACTIVITY_THRESHOLD (default 0.3): VAD threshold for speech_end when the
    // stream has no gate threshold of its own.
    static float activity_threshold() {
        static const float t = []() {
            const char* e = std::getenv("PARAKEET_STREAM_ACTIVITY_THRESHOLD");
            return e && *e ? (float)std::atof(e) : 0.3f;
        }();
        return t;
    }

    // Scans [1 s of earlier audio + this feed] and moves speech_end to the end of the latest
    // speech found. A segment reaching the buffer end means speech is still going.
    void track_activity(SpeechGate& g, const float* samples, int n) {
        if (n <= 0)
            return;
        const int64_t base = g.total - (int64_t)g.ctx.size();
        g.ctx.insert(g.ctx.end(), samples, samples + n);
        g.total += n;
        if (vad_) {
            firered_vad_segment* segs = nullptr;
            int n_segs = 0;
            if (firered_vad_detect(vad_, g.ctx.data(), (int)g.ctx.size(), &segs, &n_segs, g.activity_threshold,
                                   0.05f, 0.1f) == 0) {
                const double buf_sec = (double)g.ctx.size() / 16000.0;
                for (int i = 0; i < n_segs; i++) {
                    const double e = segs[i].end_sec >= buf_sec - 0.05 ? buf_sec : segs[i].end_sec;
                    g.speech_end = std::max(g.speech_end, (double)base / 16000.0 + e);
                }
                free(segs);
            }
        }
        if (g.ctx.size() > kActivityCtx)
            g.ctx.erase(g.ctx.begin(), g.ctx.end() - kActivityCtx);
    }

    // PARAKEET_STREAM_VAD_MODEL, default /models/firered-vad.gguf. Missing → streams run ungated.
    static firered_vad_context* load_vad() {
        const char* env = std::getenv("PARAKEET_STREAM_VAD_MODEL");
        const char* path = env && *env ? env : "/models/firered-vad.gguf";
        firered_vad_context* v = firered_vad_init(path);
        if (!v)
            fprintf(stderr, "crispasr[parakeet]: stream speech gate and activity off — no VAD model at '%s'\n", path);
        return v;
    }

    parakeet_context* ctx_ = nullptr;
    std::map<uint32_t, parakeet_stream*> streams_;
    std::map<uint32_t, std::unique_ptr<SpeechGate>> gates_;
    firered_vad_context* vad_ = nullptr;
    bool vad_missing_ = false;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_parakeet_backend() {
    return std::unique_ptr<CrispasrBackend>(new ParakeetBackend());
}
