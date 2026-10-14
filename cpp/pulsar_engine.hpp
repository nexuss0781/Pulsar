// ============================================================
//  PULSAR reference engine -- C++17, header-only, no deps.
//  Legacy experimental engine. Its exact-sampling primitive is shared with
//  the frozen v1 core in pulsar_core_v1.hpp.
//    M1  top-B candidate generation from the draft row
//    M2  candidate prefix tree (probability path weights, not amplitudes)
//    M4a rotor: fixed-batch readout ticks (bounded per-tick work)
//    M4c trainable confidence field: logistic + isotonic calibrator
//    M4b pulse: water-level A(u) >= theta, A = prod q*ahat,
//               theta = theta_mult * c_v1 * S   (self-consistent, M7-A)
//    M4d collapse: sequential categorical draw x ~ pi_u (v1 uses pi=q)
//    M3  verifier: accept min(1, p(x)/pi(x)); residual norm(max(p-pi,0))
//    M5  commit: emitted prefix + residual/bonus token
//    M6  pulse clock: all step work <= c_draft*levels + c_tick*ticks
//        + 1 + c_v1*n_nodes  (compile-time-bounded tree budget)
//  Exactness invariant: verification samples from and tests the same pi.
//  Optional research modules below are not part of the algorithm_v1 contract.
// ============================================================
#pragma once
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>
#include <limits>
#include "pulsar_core_v1.hpp"
#include <algorithm>
#include <cmath>

namespace pulsar {

struct Params {
    int V, Hmax, B, maxNodes, wDim, nIso;
    double cDraft, cV1, cTick, thetaMult, ema, thetaFloor, bias;
};

struct Model {
    Params p;
    std::vector<double> w, isoK, isoV;      // M4c
    std::vector<double> P;                  // (V^3, V) target rows
    std::vector<double> Q;                  // (V^2, V) draft rows
    std::vector<double> qEnt, qTop;         // cached stats

    static double interp(const std::vector<double>& xs,
                         const std::vector<double>& ys, double x) {
        if (x <= xs.front()) return ys.front();
        if (x >= xs.back())  return ys.back();
        auto it = std::lower_bound(xs.begin(), xs.end(), x);
        size_t i = it - xs.begin();
        double t = (x - xs[i - 1]) / (xs[i] - xs[i - 1] + 1e-300);
        return ys[i - 1] + t * (ys[i] - ys[i - 1]);
    }

    double predictA(const double* x, int dim) const {           // M4c
        double z = p.bias;
        for (int i = 0; i < dim; ++i) z += w[i] * x[i];
        double a = interp(isoK, isoV, z);                       // calibrated
        return a < 1e-6 ? 1e-6 : (a > 1 ? 1 : a);
    }

    int ctx2(const std::vector<int>& c) const {
        int b = c.size() >= 2 ? c[c.size() - 2] : 0;
        int v = c.size() >= 1 ? c[c.size() - 1] : 0;
        return b * p.V + v;
    }
    int ctx3(const std::vector<int>& c) const {
        int a = c.size() >= 3 ? c[c.size() - 3] : 0;
        int b = c.size() >= 2 ? c[c.size() - 2] : 0;
        int v = c.size() >= 1 ? c[c.size() - 1] : 0;
        return (a * p.V + b) * p.V + v;
    }
    const double* qrow(int i2) const { return &Q[(size_t)i2 * p.V]; }
    const double* prow(int i3) const { return &P[(size_t)i3 * p.V]; }

    double predictArow(const double* q, int x, int B, int rank) const {
        double ent = 0, mx = 0;
        for (int v = 0; v < p.V; ++v) {
            double pv = q[v];
            ent -= (pv > 1e-300 ? pv * std::log(pv) : 0.0);
            if (pv > mx) mx = pv;
        }
        double lq = std::log(std::max(q[x], 1e-300));
        double t1 = std::log(std::max(mx, 1e-300));
        double f[6] = {1.0, lq, -ent, lq - t1, t1,
                       (double)rank / std::max(B - 1, 1)};
        return predictA(f, 6);
    }
};

inline bool loadModel(Model& m, const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::printf("cannot open %s\n", path); return false; }
    const auto readExact = [f](void* dst, size_t size, size_t count) {
        return std::fread(dst, size, count, f) == count;
    };
    const auto fail = [f, path]() {
        std::fclose(f);
        std::printf("invalid or truncated model file: %s\n", path);
        return false;
    };
    int64_t h[6]{};
    if (!readExact(h, sizeof(int64_t), 6)) return fail();
    for (int64_t dimension : h) {
        if (dimension < 0 || dimension > std::numeric_limits<int>::max())
            return fail();
    }
    if (h[0] <= 0 || h[1] <= 0 || h[2] <= 0 || h[3] <= 0 ||
        h[2] > h[0] || h[4] <= 0 || h[5] <= 0) return fail();
    m.p.V = h[0]; m.p.Hmax = h[1]; m.p.B = h[2]; m.p.maxNodes = h[3];
    m.p.wDim = h[4]; m.p.nIso = h[5];
    double d[7]{};
    if (!readExact(d, sizeof(double), 7)) return fail();
    m.p.cDraft = d[0]; m.p.cV1 = d[1]; m.p.cTick = d[2];
    m.p.thetaMult = d[3]; m.p.ema = d[4]; m.p.thetaFloor = d[5]; m.p.bias = d[6];
    m.w.resize(m.p.wDim);
    m.isoK.resize(m.p.nIso);
    m.isoV.resize(m.p.nIso);
    const size_t v = static_cast<size_t>(m.p.V);
    const size_t maxSize = std::numeric_limits<size_t>::max();
    if (v > maxSize / v) return fail();
    const size_t v2 = v * v;
    if (v2 > maxSize / v) return fail();
    const size_t v3 = v2 * v;
    if (v3 > maxSize / v) return fail();
    const size_t np = v3 * v;
    const size_t nq = v3;
    m.P.resize(np); m.Q.resize(nq);
    if (!readExact(m.w.data(), sizeof(double), m.w.size()) ||
        !readExact(m.isoK.data(), sizeof(double), m.isoK.size()) ||
        !readExact(m.isoV.data(), sizeof(double), m.isoV.size()) ||
        !readExact(m.P.data(), sizeof(double), m.P.size()) ||
        !readExact(m.Q.data(), sizeof(double), m.Q.size())) return fail();
    std::fclose(f);
    return true;
}

struct StepResult {
    std::vector<int> emitted;
    double cost;
    int nAccepted;
    int nBlock;
    double theta;
};

class Engine {
public:
    explicit Engine(const Model& m) : m_(m), Lema_(1.0) {}

    double theta() const {                                    // M7-A
        double S = 1.0 / std::max(Lema_, 1e-6);
        return std::max(m_.p.thetaFloor, m_.p.cV1 * m_.p.thetaMult * S);
    }

    StepResult step(std::vector<int>& ctx, std::mt19937_64& rng) {
        StepResult out;
        const int V = m_.p.V, B = m_.p.B, H = m_.p.Hmax;
        const double th = theta();

        // ---- node arena (M2 field) --------------------------------------
        struct Node { int tok, par, dep, c2, c3; double q, A; };
        std::vector<Node> nd(m_.p.maxNodes + 16);
        int nN = 1;
        nd[0] = {-1, -1, 0, m_.ctx2(ctx), m_.ctx3(ctx), 0.0, 1.0};
        std::vector<int> front = {0};

        // ---- M1 + M4c + M4b: grow with water-level pruning ----------------
        int levels = 0;
        for (int dep = 1; dep <= H && !front.empty() && nN + (int)front.size()*B < (int)nd.size(); ++dep) {
            std::vector<int> nfront;
            for (int u : front) {
                const double* q = m_.qrow(nd[u].c2);
                std::vector<int> top(B, 0);
                std::vector<double> tq(B, -1e300);
                for (int v = 0; v < V; ++v) {
                    for (int j = 0; j < B; ++j) {
                        if (q[v] > tq[j]) {                        // insertion top-B (sorted)
                            for (int k = B - 1; k > j; --k) { tq[k] = tq[k-1]; top[k] = top[k-1]; }
                            tq[j] = q[v]; top[j] = v;
                            break;
                        }
                    }
                }
                for (int r = 0; r < B; ++r) {
                    double qc = tq[r];
                    if (qc <= 0) continue;
                    double ah = m_.predictArow(q, top[r], B, r);  // M4c
                    double A = nd[u].A * qc * ah;                 // superposition mass density
                    if (A < th) continue;                          // water level (M4b)
                    int id = nN++;
                    nd[id] = {top[r], u, dep,
                              (nd[u].c2 % V) * V + top[r],
                              (nd[u].c3 % (V*V)) * V + top[r],
                              qc, A};
                    nfront.push_back(id);
                    if (nN + B >= (int)nd.size()) break;
                }
                if (nN + B >= (int)nd.size()) break;
            }
            if (nfront.empty()) break;
            front.swap(nfront);
            ++levels;
        }

        // ---- weak pulse (single verified probe) ---------------------------
        if (nN == 1) {
            const double* q = m_.qrow(nd[0].c2);
            const double* p = m_.prow(nd[0].c3);
            const auto sample = v1::sample_verified_token(
                v1::DistributionView(p, V), v1::DistributionView(q, V), rng);
            out.cost = 1.0; out.nBlock = 0; out.theta = th;
            out.emitted = {static_cast<int>(sample.emitted_token)};
            out.nAccepted = sample.accepted ? 1 : 0;
            Lema_ = (1 - m_.p.ema) * Lema_ + m_.p.ema * out.cost;
            return out;
        }

        // ---- M4d fall-through collapse + M3 verification -------------------
        int u = 0, plen = 0;
        bool ok = true;
        while (true) {
            if (plen >= H) break;
            const double* q = m_.qrow(nd[u].c2);
            const double* p = m_.prow(nd[u].c3);
            const auto sample = v1::sample_verified_token(
                v1::DistributionView(p, V), v1::DistributionView(q, V), rng);
            const int x = static_cast<int>(sample.proposal_token);
            ++plen;
            int child = -1;
            for (size_t v = 1; v < (size_t)nN; ++v)
                if (nd[v].par == u && nd[v].tok == x) { child = (int)v; break; }
            if (ok && sample.accepted) {
                out.emitted.push_back(static_cast<int>(sample.emitted_token));
                ++out.nAccepted;
                if (child < 0) break;                              // fall-through
                u = child;
            } else {
                ok = false;
                out.emitted.push_back(static_cast<int>(sample.emitted_token));
                break;
            }
        }
        if (ok && plen == H) {                                     // bonus token
            std::vector<int> c2 = ctx;
            for (int x : out.emitted) c2.push_back(x);
            const double* pb = m_.prow(m_.ctx3(c2));
            std::discrete_distribution<int> db(pb, pb + V);
            out.emitted.push_back(db(rng));
        }
        int ticks = std::max(1, (nN - 1 + 63) / 64);          // M6 tick budget
        out.cost = m_.p.cDraft * levels + m_.p.cTick * ticks
                 + 1.0 + m_.p.cV1 * (nN - 1);
        out.nBlock = nN - 1;
        out.theta = th;
        Lema_ = (1 - m_.p.ema) * Lema_
              + m_.p.ema * (out.cost / std::max((size_t)1, out.emitted.size()));
        return out;
    }

    double L() const { return Lema_; }

private:
    const Model& m_;
    double Lema_;
};

// vanilla AR sampler (reference distribution)
inline int arStep(const Model& m, std::vector<int>& ctx, std::mt19937_64& rng) {
    const double* p = m.prow(m.ctx3(ctx));
    std::discrete_distribution<int> d(p, p + m.p.V);
    return d(rng);
}

}  // namespace pulsar
