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

#include <cstdio>
#include <cstring>
#include <map>

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
        parakeet_stream* st = parakeet_stream_begin(ctx_, sp);
        if (!st)
            return false;
        streams_[sid] = st;
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
        if (final)
            stream_end(sid);
        if (!r) {
            free(ptext);
            d.error = "stream decode failed";
            return d;
        }
        d.ok = true;
        d.text = r->text ? r->text : "";
        for (int i = 0; i < r->n_words; i++)
            d.words.push_back({r->words[i].text, r->words[i].t0, r->words[i].t1});
        if (ptext)
            d.partial = ptext;
        free(ptext);
        parakeet_result_free(r);
        return d;
    }

    void stream_end(uint32_t sid) override {
        auto it = streams_.find(sid);
        if (it != streams_.end()) {
            parakeet_stream_free(it->second);
            streams_.erase(it);
        }
    }

    void shutdown() override {
        for (auto& [sid, st] : streams_)
            parakeet_stream_free(st);
        streams_.clear();
        if (ctx_) {
            parakeet_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    parakeet_context* ctx_ = nullptr;
    std::map<uint32_t, parakeet_stream*> streams_;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_parakeet_backend() {
    return std::unique_ptr<CrispasrBackend>(new ParakeetBackend());
}
