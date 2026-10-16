// Pulsar v1 mathematical core: exact categorical rejection sampling.
// Only probability weights are used here; path products are not amplitudes.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <random>
#include <stdexcept>
#include <vector>

namespace pulsar::v1 {

using Token = std::size_t;
using Weights = std::vector<double>;

// A non-owning view of non-negative categorical weights. The represented law
// is the weights normalized by their sum; inputs need not sum to exactly 1.
// The caller must keep the pointed-to storage alive for the duration of use.
struct DistributionView {
    const double* data;
    std::size_t size;

    DistributionView(const double* values, std::size_t count)
        : data(values), size(count) {}

    explicit DistributionView(const Weights& values)
        : data(values.data()), size(values.size()) {}
};

namespace detail {

inline long double total_mass(DistributionView distribution) {
    if (distribution.data == nullptr || distribution.size == 0) {
        throw std::invalid_argument("distribution must be non-empty");
    }

    long double sum = 0.0L;
    for (std::size_t i = 0; i < distribution.size; ++i) {
        const double weight = distribution.data[i];
        if (!std::isfinite(weight) || weight < 0.0) {
            throw std::invalid_argument(
                "distribution weights must be finite and non-negative");
        }
        sum += static_cast<long double>(weight);
    }
    if (!(sum > 0.0L) || !std::isfinite(sum)) {
        throw std::invalid_argument("distribution must have positive finite mass");
    }
    return sum;
}

struct DistributionFeatures {
    double maximum_probability = 0.0;
    double top_two_margin = 0.0;
    double normalized_entropy = 0.0;
};

inline DistributionFeatures features(DistributionView distribution) {
    const long double mass = total_mass(distribution);
    double maximum = 0.0;
    double second = 0.0;
    long double entropy = 0.0L;
    for (std::size_t i = 0; i < distribution.size; ++i) {
        const double probability = static_cast<double>(
            static_cast<long double>(distribution.data[i]) / mass);
        if (probability > maximum) {
            second = maximum;
            maximum = probability;
        } else if (probability > second) {
            second = probability;
        }
        if (probability > 0.0) {
            entropy -= static_cast<long double>(probability) * std::log(probability);
        }
    }
    const double normalizer = distribution.size > 1
        ? std::log(static_cast<double>(distribution.size)) : 1.0;
    return {maximum, maximum - second,
            distribution.size > 1 ? static_cast<double>(entropy) / normalizer : 0.0};
}

template <class URBG>
inline Token draw(DistributionView distribution, URBG& rng) {
    std::discrete_distribution<Token> categorical(
        distribution.data, distribution.data + distribution.size);
    return categorical(rng);
}

}  // namespace detail

struct VerifiedToken {
    Token proposal_token;
    Token emitted_token;
    bool accepted;
    // Diagnostic view of the acceptance probability, rounded to double.
    double acceptance_probability;
};

// Return Z = sum_x max(p[x] - pi[x], 0) after normalizing both input weight
// vectors. This is TV(p, pi) in exact arithmetic. Zero means there is no
// rejection branch; callers must not use a residual sampler in that case.
inline long double residual_mass(DistributionView target,
                                 DistributionView effective_pi) {
    if (target.size != effective_pi.size) {
        throw std::invalid_argument("target and effective policy sizes differ");
    }
    const long double target_mass = detail::total_mass(target);
    const long double pi_mass = detail::total_mass(effective_pi);

    long double mass = 0.0L;
    for (std::size_t i = 0; i < target.size; ++i) {
        const long double p = static_cast<long double>(target.data[i]) / target_mass;
        const long double pi =
            static_cast<long double>(effective_pi.data[i]) / pi_mass;
        mass += std::max(0.0L, p - pi);
    }
    return mass;
}

// Return the normalized residual law. No epsilon threshold is used: every
// positive residual representable by the working arithmetic is retained. A
// zero residual is an error by design: when p == pi, rejection has probability
// zero and this function is not needed. A positive residual too small to be
// represented by the platform's floating-point arithmetic cannot be recovered.
inline Weights residual_distribution(DistributionView target,
                                     DistributionView effective_pi) {
    if (target.size != effective_pi.size) {
        throw std::invalid_argument("target and effective policy sizes differ");
    }
    const long double target_mass = detail::total_mass(target);
    const long double pi_mass = detail::total_mass(effective_pi);

    std::vector<long double> positive_difference(target.size, 0.0L);
    long double residual_total = 0.0L;
    for (std::size_t i = 0; i < target.size; ++i) {
        const long double p = static_cast<long double>(target.data[i]) / target_mass;
        const long double pi =
            static_cast<long double>(effective_pi.data[i]) / pi_mass;
        positive_difference[i] = std::max(0.0L, p - pi);
        residual_total += positive_difference[i];
    }
    if (!(residual_total > 0.0L)) {
        throw std::logic_error("residual is undefined when target equals policy");
    }

    Weights residual(target.size, 0.0);
    for (std::size_t i = 0; i < target.size; ++i) {
        residual[i] = static_cast<double>(positive_difference[i] / residual_total);
    }
    return residual;
}

// Draw x from the effective collapse law pi, then accept with min(1, p[x]/pi[x]).
// On rejection, draw from normalized max(p - pi, 0). `target` and `effective_pi`
// must have the same non-empty vocabulary and finite, non-negative weights of
// positive total mass. Zero weights are allowed. Any standard UniformRandomBitGenerator
// can be supplied; the same effective_pi law is used for proposal and ratio.
template <class URBG>
inline VerifiedToken sample_verified_token(DistributionView target,
                                           DistributionView effective_pi,
                                           URBG& rng) {
    if (target.size != effective_pi.size) {
        throw std::invalid_argument("target and effective policy sizes differ");
    }
    const long double target_mass = detail::total_mass(target);
    const long double pi_mass = detail::total_mass(effective_pi);
    const Token proposal = detail::draw(effective_pi, rng);
    const long double pi_x =
        static_cast<long double>(effective_pi.data[proposal]) / pi_mass;
    if (!(pi_x > 0.0L)) {
        throw std::logic_error("categorical sampler returned a zero-mass token");
    }
    const long double p_x =
        static_cast<long double>(target.data[proposal]) / target_mass;
    const long double accept_probability = std::min(1.0L, p_x / pi_x);

    std::uniform_real_distribution<long double> uniform(0.0L, 1.0L);
    if (uniform(rng) < accept_probability) {
        return {proposal, proposal, true,
                static_cast<double>(accept_probability)};
    }

    const Weights residual = residual_distribution(target, effective_pi);
    const Token emitted = detail::draw(DistributionView(residual), rng);
    return {proposal, emitted, false, static_cast<double>(accept_probability)};
}

// Candidate construction may use q independently. The retained-token list
// controls continuation only: it cannot filter, renormalize, or replace pi.
struct PulseToken {
    Token proposal_token;
    Token emitted_token;
    bool accepted;
    bool continue_pulse;
    double acceptance_probability;
};

template <class URBG>
inline PulseToken sample_pulse_token(DistributionView target,
                                     DistributionView effective_pi,
                                     const std::vector<Token>& retained_children,
                                     URBG& rng) {
    const VerifiedToken verified =
        sample_verified_token(target, effective_pi, rng);
    const bool has_child = std::find(retained_children.begin(),
                                     retained_children.end(),
                                     verified.proposal_token) != retained_children.end();
    const bool continue_pulse = verified.accepted && has_child;
    return {verified.proposal_token, verified.emitted_token,
            verified.accepted, continue_pulse, verified.acceptance_probability};
}

// Deterministic top-k candidate set from q, with token-id tie breaking.
// This is a tree-budget helper only; it does not define the collapse law.
inline std::vector<Token> top_k_candidates(DistributionView q, std::size_t k) {
    (void)detail::total_mass(q);
    std::vector<Token> tokens(q.size);
    for (std::size_t i = 0; i < q.size; ++i) {
        tokens[i] = i;
    }
    std::stable_sort(tokens.begin(), tokens.end(), [q](Token a, Token b) {
        if (q.data[a] != q.data[b]) {
            return q.data[a] > q.data[b];
        }
        return a < b;
    });
    if (k < tokens.size()) {
        tokens.resize(k);
    }
    return tokens;
}

}  // namespace pulsar::v1
