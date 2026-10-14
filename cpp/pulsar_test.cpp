// PULSAR C++ acceptance test:
//   1) exactness on a fixed prompt (first emitted token vs target row)
//   2) sequence-level agreement with an independently seeded AR run (unigram TV)
//   3) cost-model speedup > 1.1 and measured work <= M6 bound
//   4) theta fixed point lands in the sane bracket
#include "pulsar_engine.hpp"
#include <cstdio>
#include <cmath>
#include <chrono>
using namespace pulsar;

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "inputs/model.bin";
    Model m;
    if (!loadModel(m, path)) return 2;
    std::mt19937_64 probe_rng(20261012);
    Engine eng(m);
    const int V = m.p.V;

    // -- (1) fixed-prompt exactness: 250k weak-collapse probes of depth 0 --
    std::vector<int> hist(V, 0);
    int N1 = 250000;
    for (int i = 0; i < N1; ++i) {
        std::vector<int> ctx = {5, 3, 9};
        auto s = eng.step(ctx, probe_rng);
        ++hist[s.emitted[0]];
    }
    const double* p = m.prow(m.ctx3(std::vector<int>{5, 3, 9}));
    double tv = 0, zmax = 0;
    for (int v = 0; v < V; ++v) {
        double e = p[v] * N1;
        tv += std::fabs(hist[v] / (double)N1 - p[v]);
        zmax = std::max(zmax, std::fabs(hist[v] - e) / std::sqrt(std::max(e, 1.0)));
    }
    tv *= 0.5;
    bool ok1 = tv < 0.006 && zmax < 6.0;

    // -- (2) sequence-level: independent runs, equal start and token limit ---
    std::vector<int> histE(V, 0), histA(V, 0);
    const std::vector<int> initial_ctx = {1, 4, 7};
    std::vector<int> ctx = initial_ctx;
    std::vector<int> ctxA = initial_ctx;
    std::mt19937_64 speculative_rng(0x5EED5EEDULL);
    std::mt19937_64 ar_rng(0xA17A17A1ULL);
    double totCost = 0; int totTok = 0, steps = 0; double maxStep = 0;
    int N2 = 300000;
    for (int i = 0; i < N2; ++i) {
        auto s = eng.step(ctx, speculative_rng);
        for (int x : s.emitted) {
            if (totTok >= N2) break;
            ++histE[x];
            ++totTok;
            ctx.push_back(x);
        }
        totCost += s.cost; ++steps;
        maxStep = std::max(maxStep, s.cost);
        if (totTok >= N2) break;
    }
    for (int i = 0; i < N2; ++i) {
        int x = arStep(m, ctxA, ar_rng);
        ++histA[x];
        ctxA.push_back(x);
    }
    double tvSeq = 0;
    for (int v = 0; v < V; ++v)
        tvSeq += std::fabs(histE[v] / (double)totTok - histA[v] / (double)N2);
    tvSeq *= 0.5;
    // Independent long-run histograms have sampling noise; conditional laws
    // receive the tighter, context-stratified gate in Stage 2's test suite.
    bool ok2 = tvSeq < 0.007;

    // -- (3) cost-model speedup & M6 bound -----------------------------------
    double speedup = totTok / totCost;
    double bound = m.p.cDraft * m.p.Hmax + m.p.cTick + 1.0
                 + m.p.cV1 * m.p.maxNodes;
    bool ok3 = speedup > 1.10 && maxStep <= bound + 1e-9;

    // -- (4) theta in sane range ---------------------------------------------
    double th = eng.theta();
    bool ok4 = th > 1e-4 && th < 0.5;

    std::printf("PULSAR C++ verification\n");
    std::printf("  (1) fixed-prompt exactness   TV=%.5f  maxZ=%.2f   [%s]\n",
                tv, zmax, ok1 ? "PASS" : "FAIL");
    std::printf("  (2) independent-seed AR unigram (300k tok) TV=%.5f [%s]\n",
                tvSeq, ok2 ? "PASS" : "FAIL");
    std::printf("  (3) speedup=%.3f (cost-model), maxstep=%.2f <= bound=%.2f [%s]\n",
                speedup, maxStep, bound, ok3 ? "PASS" : "FAIL");
    std::printf("  (4) theta=%.4f in (1e-4, 0.5)                       [%s]\n",
                th, ok4 ? "PASS" : "FAIL");
    std::printf("  mean emit/step=%.3f  steps=%d\n", totTok / (double)steps, steps);
    bool all = ok1 && ok2 && ok3 && ok4;
    std::printf("%s\n", all ? "PULSAR_TEST: PASS" : "PULSAR_TEST: FAIL");
    return all ? 0 : 1;
}
