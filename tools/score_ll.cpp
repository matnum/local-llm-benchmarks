// score_ll.cpp — loglikelihood scorer for multiple-choice evaluation.
//
// Why this exists: FIN-bench-v2 scores multiple-choice tasks by the summed
// loglikelihood of the full choice text as a continuation. The pinned
// llama-server cannot produce those (it refuses `echo` outright —
// tools/server/server-common.cpp: "Only no echo is supported" — and the native
// /completion endpoint returns next-token probabilities only, 1 entry for 22
// evaluated prompt tokens). Linking against the pinned build's own libllama
// gives the logits, so the loglikelihoods can be computed exactly and cheaply:
// one context pass per item, then one short pass per choice.
//
// Protocol (delimited by TAB; every field is JSON-escaped by the caller with
// json.dumps(..., ensure_ascii=False), so UTF-8 passes through and only
// \n \t \r \" \\ need unescaping):
//
//   stdin   id \t context \t choice1 \t choice2 \t ...
//   stdout  id \t ll1 \t ntok1 \t ll2 \t ntok2 \t ...
//
// ll = sum of log P(token_t | context + choice[:t]) over the choice's tokens.
// The character length the harness divides by for acc_norm is NOT emitted here
// — Python's len() counts characters while bytes differ for Finnish ä/ö, so the
// caller computes it.
//
// Build (against the pinned build; no other dependency):
//   L=~/.local/opt/llama.cpp
//   g++ -O2 -std=c++17 -I$L/include -I$L/ggml/include tools/score_ll.cpp
//     -L$L/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$L/build/bin -o tools/score_ll
//
// Usage:
//   tools/score_ll -m model.gguf [-ngl 99] [-c 4096] [-t 16] [--fa] < items.tsv

#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

static void quiet_log(enum ggml_log_level, const char *, void *) {}

static std::vector<std::string> split_tab(const std::string & line) {
    std::vector<std::string> out;
    size_t start = 0;
    while (true) {
        size_t pos = line.find('\t', start);
        out.push_back(line.substr(start, pos == std::string::npos ? pos : pos - start));
        if (pos == std::string::npos) break;
        start = pos + 1;
    }
    return out;
}

// Minimal unescape for the escapes json.dumps(ensure_ascii=False) emits.
static std::string unescape(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            switch (s[++i]) {
                case 'n':  out += '\n'; break;
                case 't':  out += '\t'; break;
                case 'r':  out += '\r'; break;
                case '"':  out += '"';  break;
                case '\\': out += '\\'; break;
                case '/':  out += '/';  break;
                case 'b':  out += '\b'; break;
                case 'f':  out += '\f'; break;
                default:   out += s[i]; break;
            }
        } else {
            out += s[i];
        }
    }
    return out;
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text,
                                         bool add_special) {
    const int32_t n = -llama_tokenize(vocab, text.data(), (int32_t) text.size(), nullptr, 0,
                                      add_special, true);
    std::vector<llama_token> toks(n);
    const int32_t got = llama_tokenize(vocab, text.data(), (int32_t) text.size(), toks.data(),
                                       (int32_t) toks.size(), add_special, true);
    if (got < 0) {
        std::fprintf(stderr, "score_ll: tokenize failed\n");
        std::exit(2);
    }
    toks.resize(got);
    return toks;
}

// log softmax over the whole vocabulary, read at `target`.
static double logprob_of(const float * logits, int32_t n_vocab, llama_token target) {
    float mx = logits[0];
    for (int32_t i = 1; i < n_vocab; ++i) mx = std::max(mx, logits[i]);
    double sum = 0.0;
    for (int32_t i = 0; i < n_vocab; ++i) sum += std::exp((double) logits[i] - (double) mx);
    return (double) logits[target] - (double) mx - std::log(sum);
}

static bool decode(llama_context * ctx, const std::vector<llama_token> & toks,
                   const std::vector<int32_t> & pos, const std::vector<int8_t> & want_logits) {
    llama_batch batch = llama_batch_init((int32_t) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        batch.token[i]      = toks[i];
        batch.pos[i]        = pos[i];
        batch.n_seq_id[i]   = 1;
        batch.seq_id[i][0]  = 0;
        batch.logits[i]     = want_logits[i];
    }
    batch.n_tokens = (int32_t) toks.size();
    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return rc == 0;
}

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    int ngl = 99, n_ctx = 4096, n_threads = 16, n_batch = 2048;
    bool use_fa = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) { std::fprintf(stderr, "score_ll: %s needs a value\n", argv[i]); std::exit(2); }
            return argv[++i];
        };
        if      (a == "-m")    model_path = next();
        else if (a == "-ngl")  ngl = std::atoi(next());
        else if (a == "-c")    n_ctx = std::atoi(next());
        else if (a == "-t")    n_threads = std::atoi(next());
        else if (a == "-b")    n_batch = std::atoi(next());
        else if (a == "--fa")  use_fa = true;
        else { std::fprintf(stderr, "score_ll: unknown arg %s\n", a.c_str()); std::exit(2); }
    }
    if (!model_path) { std::fprintf(stderr, "score_ll: -m <model.gguf> is required\n"); return 2; }

    llama_log_set(quiet_log, nullptr);
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) { std::fprintf(stderr, "score_ll: failed to load %s\n", model_path); return 3; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = (uint32_t) n_ctx;
    cparams.n_batch   = (uint32_t) n_batch;
    cparams.n_ubatch  = (uint32_t) std::min(n_batch, 512);
    cparams.n_seq_max = 1;
    cparams.n_threads = n_threads;
    cparams.n_threads_batch = n_threads;
    if (use_fa) cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { std::fprintf(stderr, "score_ll: failed to create context\n"); return 3; }
    llama_memory_t mem = llama_get_memory(ctx);

    std::fprintf(stderr, "score_ll: %s | n_ctx=%d ngl=%d threads=%d vocab=%d\n",
                 model_path, n_ctx, ngl, n_threads, n_vocab);

    std::string line;
    long items = 0, prompt_tokens = 0;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        const std::vector<std::string> f = split_tab(line);
        if (f.size() < 3) { std::fprintf(stderr, "score_ll: bad input line (need id, context, >=1 choice)\n"); return 2; }
        const std::string & id = f[0];
        const std::string context = unescape(f[1]);

        std::vector<std::vector<llama_token>> cont;
        cont.reserve(f.size() - 2);
        for (size_t i = 2; i < f.size(); ++i) cont.push_back(tokenize(vocab, unescape(f[i]), false));
        const std::vector<llama_token> ctxt = tokenize(vocab, context, true);

        size_t max_cont = 0;
        for (const auto & c : cont) max_cont = std::max(max_cont, c.size());
        if (ctxt.empty() || ctxt.size() + max_cont + 2 > (size_t) n_ctx) {
            // One oversized item must not kill a multi-hour run: emit a skip marker
            // (ntok = -1) and let the caller exclude it and report the count.
            std::fprintf(stderr, "score_ll: SKIP %s: does not fit n_ctx=%d (%zu + %zu)\n",
                         id.c_str(), n_ctx, ctxt.size(), max_cont);
            printf("%s", id.c_str());
            for (size_t i = 0; i < cont.size(); ++i) printf("\t-1e9\t-1");
            printf("\n");
            std::fflush(stdout);
            continue;
        }

        // One context pass per item; every choice reuses it.
        llama_memory_clear(mem, true);
        std::vector<int32_t> cpos(ctxt.size());
        std::vector<int8_t>  cwant(ctxt.size(), 0);
        for (size_t i = 0; i < ctxt.size(); ++i) cpos[i] = (int32_t) i;
        cwant[ctxt.size() - 1] = 1;                         // logits at the last context position
        if (!decode(ctx, ctxt, cpos, cwant)) { std::fprintf(stderr, "score_ll: context decode failed\n"); return 5; }

        // P(choice[0] | context) — same distribution for every choice, so read it
        // now: the logits pointer is invalidated by the next decode.
        std::vector<double> first_lp(cont.size(), 0.0);
        const float * ctx_logits = llama_get_logits_ith(ctx, (int32_t) ctxt.size() - 1);
        for (size_t i = 0; i < cont.size(); ++i) {
            first_lp[i] = cont[i].empty() ? 0.0 : logprob_of(ctx_logits, n_vocab, cont[i][0]);
        }

        printf("%s", id.c_str());
        for (size_t i = 0; i < cont.size(); ++i) {
            const size_t k = cont[i].size();
            double ll = first_lp[i];
            if (k > 1) {
                // Decode choice[0 .. k-2] at positions C..C+k-2; the logits at
                // position C+j predict choice[j+1].
                std::vector<llama_token> toks(cont[i].begin(), cont[i].end() - 1);
                std::vector<int32_t> pos(toks.size());
                std::vector<int8_t>  want(toks.size(), 1);
                for (size_t j = 0; j < toks.size(); ++j) pos[j] = (int32_t) (ctxt.size() + j);
                if (!decode(ctx, toks, pos, want)) { std::fprintf(stderr, "score_ll: choice decode failed\n"); return 5; }
                for (size_t j = 0; j < toks.size(); ++j) {
                    ll += logprob_of(llama_get_logits_ith(ctx, (int32_t) j), n_vocab, cont[i][j + 1]);
                }
                // Drop the choice from the KV so the next choice reuses the context.
                llama_memory_seq_rm(mem, 0, (llama_pos) ctxt.size(), -1);
            }
            printf("\t%.6f\t%zu", ll, k);
        }
        printf("\n");
        std::fflush(stdout);

        items++;
        prompt_tokens += (long) ctxt.size();
        if (items % 200 == 0) std::fprintf(stderr, "score_ll: %ld items\n", items);
    }

    std::fprintf(stderr, "score_ll: done, %ld items, %ld context tokens\n", items, prompt_tokens);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
