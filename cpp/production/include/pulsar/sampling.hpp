#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace pulsar::runtime {

using Token = std::int32_t;
using Probabilities = std::vector<double>;

struct SamplingConfig {
    // temperature == 0 selects the deterministic maximum-logit token.
    double temperature = 0.8;
    std::size_t top_k = 40;
    double top_p = 0.95;
    double repetition_penalty = 1.0;
    std::size_t repetition_window = 64;
};

inline void validate_sampling_config(const SamplingConfig& config) {
    if (!std::isfinite(config.temperature) || config.temperature < 0.0) {
        throw std::invalid_argument("temperature must be finite and non-negative");
    }
    if (!std::isfinite(config.top_p) || config.top_p <= 0.0 || config.top_p > 1.0) {
        throw std::invalid_argument("top_p must be in (0, 1]");
    }
    if (!std::isfinite(config.repetition_penalty) ||
        config.repetition_penalty < 1.0) {
        throw std::invalid_argument("repetition_penalty must be finite and >= 1");
    }
}

inline Probabilities make_distribution(const std::vector<float>& logits,
                                       const std::vector<Token>& history,
                                       const SamplingConfig& config) {
    validate_sampling_config(config);
    if (logits.empty()) {
        throw std::invalid_argument("logit row must be non-empty");
    }
    if (logits.size() > static_cast<std::size_t>(std::numeric_limits<Token>::max())) {
        throw std::length_error("vocabulary is too large for the runtime token type");
    }

    std::vector<long double> adjusted(logits.size());
    for (std::size_t i = 0; i < logits.size(); ++i) {
        if (!std::isfinite(logits[i])) {
            throw std::invalid_argument("model logits must be finite");
        }
        adjusted[i] = static_cast<long double>(logits[i]);
    }

    const std::size_t history_begin = history.size() > config.repetition_window
        ? history.size() - config.repetition_window
        : 0;
    if (config.repetition_penalty != 1.0 && config.repetition_window != 0) {
        std::vector<bool> seen(logits.size(), false);
        for (std::size_t i = history_begin; i < history.size(); ++i) {
            const Token token = history[i];
            if (token < 0 || static_cast<std::size_t>(token) >= logits.size()) {
                throw std::invalid_argument("history contains a token outside the vocabulary");
            }
            const std::size_t id = static_cast<std::size_t>(token);
            if (seen[id]) {
                continue;
            }
            seen[id] = true;
            if (adjusted[id] > 0.0L) {
                adjusted[id] /= static_cast<long double>(config.repetition_penalty);
            } else {
                adjusted[id] *= static_cast<long double>(config.repetition_penalty);
            }
        }
    }

    Probabilities probabilities(logits.size(), 0.0);
    if (config.temperature == 0.0) {
        std::size_t best = 0;
        for (std::size_t i = 1; i < adjusted.size(); ++i) {
            // Iteration order gives the lower token ID a deterministic tie break.
            if (adjusted[i] > adjusted[best]) {
                best = i;
            }
        }
        probabilities[best] = 1.0;
        return probabilities;
    }

    const long double temperature = static_cast<long double>(config.temperature);
    for (long double& value : adjusted) {
        value /= temperature;
        if (!std::isfinite(value)) {
            throw std::invalid_argument("temperature scaling overflowed model logits");
        }
    }

    std::vector<std::size_t> ranked(logits.size());
    std::iota(ranked.begin(), ranked.end(), std::size_t{0});
    const auto by_logit = [&adjusted](std::size_t left, std::size_t right) {
        if (adjusted[left] != adjusted[right]) {
            return adjusted[left] > adjusted[right];
        }
        return left < right;
    };

    std::vector<bool> retained(logits.size(), true);
    if (config.top_k != 0 && config.top_k < ranked.size()) {
        std::partial_sort(ranked.begin(), ranked.begin() + config.top_k,
                          ranked.end(), by_logit);
        std::fill(retained.begin(), retained.end(), false);
        for (std::size_t i = 0; i < config.top_k; ++i) {
            retained[ranked[i]] = true;
        }
    } else {
        std::sort(ranked.begin(), ranked.end(), by_logit);
    }

    long double maximum = -std::numeric_limits<long double>::infinity();
    for (std::size_t i = 0; i < adjusted.size(); ++i) {
        if (retained[i]) {
            maximum = std::max(maximum, adjusted[i]);
        }
    }

    std::vector<long double> weights(logits.size(), 0.0L);
    long double total = 0.0L;
    for (std::size_t i = 0; i < adjusted.size(); ++i) {
        if (retained[i]) {
            weights[i] = std::exp(adjusted[i] - maximum);
            total += weights[i];
        }
    }
    if (!(total > 0.0L) || !std::isfinite(total)) {
        throw std::runtime_error("softmax normalization failed");
    }
    for (std::size_t i = 0; i < weights.size(); ++i) {
        probabilities[i] = static_cast<double>(weights[i] / total);
    }

    if (config.top_p < 1.0) {
        std::iota(ranked.begin(), ranked.end(), std::size_t{0});
        std::sort(ranked.begin(), ranked.end(), [&probabilities](std::size_t left,
                                                                 std::size_t right) {
            if (probabilities[left] != probabilities[right]) {
                return probabilities[left] > probabilities[right];
            }
            return left < right;
        });
        long double cumulative = 0.0L;
        std::vector<bool> nucleus(probabilities.size(), false);
        for (std::size_t id : ranked) {
            if (probabilities[id] == 0.0) {
                continue;
            }
            nucleus[id] = true;
            cumulative += static_cast<long double>(probabilities[id]);
            if (cumulative >= static_cast<long double>(config.top_p)) {
                break;
            }
        }
        long double nucleus_mass = 0.0L;
        for (std::size_t i = 0; i < probabilities.size(); ++i) {
            if (!nucleus[i]) {
                probabilities[i] = 0.0;
            } else {
                nucleus_mass += static_cast<long double>(probabilities[i]);
            }
        }
        if (!(nucleus_mass > 0.0L)) {
            throw std::runtime_error("top-p nucleus has no probability mass");
        }
        for (double& probability : probabilities) {
            probability = static_cast<double>(static_cast<long double>(probability) /
                                              nucleus_mass);
        }
    }

    return probabilities;
}

template <class URBG>
inline Token sample_categorical(const Probabilities& probabilities, URBG& rng) {
    if (probabilities.empty()) {
        throw std::invalid_argument("categorical distribution must be non-empty");
    }
    long double total = 0.0L;
    for (double probability : probabilities) {
        if (!std::isfinite(probability) || probability < 0.0) {
            throw std::invalid_argument("categorical weights must be finite and non-negative");
        }
        total += static_cast<long double>(probability);
    }
    if (!(total > 0.0L) || !std::isfinite(total)) {
        throw std::invalid_argument("categorical distribution must have positive mass");
    }

    std::uniform_real_distribution<long double> uniform(0.0L, total);
    const long double draw = uniform(rng);
    long double cumulative = 0.0L;
    std::size_t last_positive = probabilities.size();
    for (std::size_t i = 0; i < probabilities.size(); ++i) {
        if (probabilities[i] > 0.0) {
            last_positive = i;
        }
        cumulative += static_cast<long double>(probabilities[i]);
        if (draw < cumulative) {
            return static_cast<Token>(i);
        }
    }
    // Handles the possible upper endpoint of uniform_real_distribution and
    // harmless floating-point summation differences.
    if (last_positive == probabilities.size()) {
        throw std::logic_error("categorical distribution lost all positive mass");
    }
    return static_cast<Token>(last_positive);
}

}  // namespace pulsar::runtime
