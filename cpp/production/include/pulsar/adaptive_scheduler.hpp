#pragma once

#include "pulsar_tree_v3.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace pulsar::runtime {

struct CandidateFeatures {
    double max_probability = 0.0;
    double top_two_margin = 0.0;
    double normalized_entropy = 0.0;
};

inline CandidateFeatures candidate_features(const std::vector<double>& probabilities) {
    if (probabilities.empty()) {
        throw std::invalid_argument("candidate feature row must be non-empty");
    }
    long double total = 0.0L;
    for (double value : probabilities) {
        if (!std::isfinite(value) || value < 0.0) {
            throw std::invalid_argument("candidate feature probabilities must be finite and non-negative");
        }
        total += static_cast<long double>(value);
    }
    if (!(total > 0.0L) || !std::isfinite(total)) {
        throw std::invalid_argument("candidate feature row must have positive finite mass");
    }
    long double entropy = 0.0L;
    double first = 0.0;
    double second = 0.0;
    for (double value : probabilities) {
        const double probability = static_cast<double>(static_cast<long double>(value) / total);
        if (probability > first) {
            second = first;
            first = probability;
        } else if (probability > second) {
            second = probability;
        }
        if (probability > 0.0) {
            entropy -= static_cast<long double>(probability) * std::log(probability);
        }
    }
    const double normalizer = probabilities.size() > 1
        ? std::log(static_cast<double>(probabilities.size())) : 1.0;
    return {first, first - second,
            probabilities.size() > 1 ? static_cast<double>(entropy) / normalizer : 0.0};
}

struct TreeSchedule {
    std::size_t branching = 1;
    std::size_t depth = 1;
};

struct RuntimeTreeObservation {
    std::uint64_t request_id = 0;
    std::size_t batch_size = 1;
    std::size_t branching = 1;
    std::size_t configured_depth = 1;
    std::size_t pulse_depth = 1;
    std::size_t pulse_index = 0;
    std::size_t depth = 0;
    CandidateFeatures root_features;
    CandidateFeatures node_features;
    double proposal_probability = 0.0;
    bool accepted = false;
    bool continuation_label_available = false;
    bool continued = false;
};

namespace detail {

inline std::vector<std::string> split_tsv(const std::string& line) {
    std::vector<std::string> fields;
    std::size_t begin = 0;
    while (true) {
        const std::size_t end = line.find('\t', begin);
        fields.push_back(line.substr(begin, end == std::string::npos ? end : end - begin));
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return fields;
}

inline double parse_finite_double(const std::string& text, const std::string& name) {
    std::size_t consumed = 0;
    const double value = std::stod(text, &consumed);
    if (consumed != text.size() || !std::isfinite(value)) {
        throw std::invalid_argument("invalid finite value for " + name);
    }
    return value;
}

inline std::size_t parse_size_value(const std::string& text, const std::string& name) {
    std::size_t consumed = 0;
    const unsigned long long value = std::stoull(text, &consumed, 10);
    if (consumed != text.size()) {
        throw std::invalid_argument("invalid integer for " + name);
    }
    return static_cast<std::size_t>(value);
}

}  // namespace detail

struct CalibrationCellKey {
    std::size_t batch_size = 0;
    std::size_t branching = 0;
    std::size_t depth = 0;
    std::string outcome;

    bool operator<(const CalibrationCellKey& other) const noexcept {
        return std::tie(batch_size, branching, depth, outcome) <
               std::tie(other.batch_size, other.branching, other.depth, other.outcome);
    }
};

struct LogisticCalibrationCell {
    std::size_t observations = 0;
    std::array<double, 3> means{};
    std::array<double, 3> scales{{1.0, 1.0, 1.0}};
    std::array<double, 4> coefficients{};
};

class ConfidenceCalibrator {
public:
    void add_cell(const CalibrationCellKey& key, const LogisticCalibrationCell& cell) {
        if (key.batch_size == 0 || key.branching == 0 || key.depth == 0 ||
            (key.outcome != "accepted" && key.outcome != "continued") ||
            cell.observations == 0) {
            throw std::invalid_argument("calibration cell key or sample count is invalid");
        }
        for (double scale : cell.scales) {
            if (!(scale > 0.0) || !std::isfinite(scale)) {
                throw std::invalid_argument("calibration feature scale must be positive and finite");
            }
        }
        for (double value : cell.means) {
            if (!std::isfinite(value)) throw std::invalid_argument("calibration mean is non-finite");
        }
        for (double value : cell.coefficients) {
            if (!std::isfinite(value)) throw std::invalid_argument("calibration coefficient is non-finite");
        }
        if (!cells_.emplace(key, cell).second) {
            throw std::invalid_argument("duplicate calibration cell");
        }
    }

    static ConfidenceCalibrator load_tsv(const std::string& path) {
        std::ifstream input(path);
        if (!input) throw std::runtime_error("cannot open calibrator file: " + path);
        ConfidenceCalibrator model;
        std::string line;
        std::size_t line_number = 0;
        while (std::getline(input, line)) {
            ++line_number;
            if (line.empty() || line.front() == '#') continue;
            const std::vector<std::string> fields = detail::split_tsv(line);
            if (fields.size() != 15) {
                throw std::runtime_error("calibrator TSV row must have 15 fields at line " +
                                         std::to_string(line_number));
            }
            CalibrationCellKey key;
            key.batch_size = detail::parse_size_value(fields[0], "batch_size");
            key.branching = detail::parse_size_value(fields[1], "branching");
            key.depth = detail::parse_size_value(fields[2], "depth");
            key.outcome = fields[3];
            LogisticCalibrationCell cell;
            cell.observations = detail::parse_size_value(fields[4], "observations");
            for (std::size_t i = 0; i < 3; ++i) {
                cell.means[i] = detail::parse_finite_double(fields[5 + 2 * i], "feature mean");
                cell.scales[i] = detail::parse_finite_double(fields[6 + 2 * i], "feature scale");
            }
            for (std::size_t i = 0; i < 4; ++i) {
                cell.coefficients[i] = detail::parse_finite_double(fields[11 + i], "coefficient");
            }
            model.add_cell(key, cell);
        }
        if (model.cells_.empty()) throw std::runtime_error("calibrator file contains no models");
        return model;
    }

    double predict(const std::string& outcome, std::size_t batch_size,
                   std::size_t branching, std::size_t depth,
                   const CandidateFeatures& features) const {
        const CalibrationCellKey key{batch_size, branching, depth, outcome};
        const auto found = cells_.find(key);
        if (found == cells_.end()) {
            throw std::out_of_range("no calibrated model for requested workload/action/depth/outcome");
        }
        const std::array<double, 3> values{{features.max_probability,
                                            features.top_two_margin,
                                            features.normalized_entropy}};
        double score = found->second.coefficients[0];
        for (std::size_t i = 0; i < values.size(); ++i) {
            score += found->second.coefficients[i + 1] *
                ((values[i] - found->second.means[i]) / found->second.scales[i]);
        }
        if (score >= 0.0) {
            return 1.0 / (1.0 + std::exp(-score));
        }
        const double exponential = std::exp(score);
        return exponential / (1.0 + exponential);
    }

    bool contains(const CalibrationCellKey& key) const noexcept {
        return cells_.find(key) != cells_.end();
    }

    const std::map<CalibrationCellKey, LogisticCalibrationCell>& cells() const noexcept {
        return cells_;
    }

private:
    std::map<CalibrationCellKey, LogisticCalibrationCell> cells_;
};

struct EmaState {
    bool initialized = false;
    std::size_t updates = 0;
    double value = 0.0;

    void update(bool observation, double alpha = 0.20) {
        if (!(alpha > 0.0 && alpha <= 1.0) || !std::isfinite(alpha)) {
            throw std::invalid_argument("EMA alpha must be finite and in (0,1]");
        }
        const double sample = observation ? 1.0 : 0.0;
        value = initialized ? (1.0 - alpha) * value + alpha * sample : sample;
        initialized = true;
        ++updates;
    }
};

struct ActionCostKey {
    std::size_t batch_size = 0;
    std::size_t branching = 0;
    std::size_t depth = 0;

    bool operator<(const ActionCostKey& other) const noexcept {
        return std::tie(batch_size, branching, depth) <
               std::tie(other.batch_size, other.branching, other.depth);
    }
};

inline std::size_t adaptive_action_node_count(const TreeSchedule& action) {
    std::size_t total = 0;
    std::size_t level = 1;
    for (std::size_t depth = 0; depth < action.depth; ++depth) {
        total += level;
        level *= action.branching;
    }
    return total;
}

inline std::map<ActionCostKey, double> load_action_costs_tsv(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open action cost file: " + path);
    std::map<ActionCostKey, double> costs;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty() || line.front() == '#') continue;
        const std::vector<std::string> fields = detail::split_tsv(line);
        if (fields.size() != 4) {
            throw std::runtime_error("action-cost TSV row must have 4 fields at line " +
                                     std::to_string(line_number));
        }
        const ActionCostKey key{detail::parse_size_value(fields[0], "batch_size"),
                                detail::parse_size_value(fields[1], "branching"),
                                detail::parse_size_value(fields[2], "depth")};
        const double cost = detail::parse_finite_double(fields[3], "pulse cost");
        if (key.batch_size == 0 || key.branching == 0 || key.depth < 2 || !(cost > 0.0) ||
            !costs.emplace(key, cost).second) {
            throw std::runtime_error("invalid or duplicate action-cost row");
        }
    }
    if (costs.empty()) throw std::runtime_error("action cost file contains no rows");
    return costs;
}

class AdaptiveTreeScheduler {
public:
    AdaptiveTreeScheduler(ConfidenceCalibrator calibrator,
                          std::map<ActionCostKey, double> action_costs,
                          double ema_alpha = 0.20,
                          double calibrated_weight = 0.50)
        : calibrator_(std::move(calibrator)), action_costs_(std::move(action_costs)),
          ema_alpha_(ema_alpha), calibrated_weight_(calibrated_weight) {
        if (!(ema_alpha_ > 0.0 && ema_alpha_ <= 1.0) ||
            !(calibrated_weight_ >= 0.0 && calibrated_weight_ <= 1.0)) {
            throw std::invalid_argument("adaptive scheduler weights are outside their valid ranges");
        }
    }

    void reset() { ema_.clear(); }

    TreeSchedule choose(std::uint64_t request_id, std::size_t batch_size,
                        const CandidateFeatures& root_features,
                        std::size_t remaining_tokens,
                        std::size_t context_positions) const {
        if (remaining_tokens == 0 || context_positions == 0) {
            throw std::invalid_argument("scheduler requires a positive remaining pulse budget");
        }
        constexpr std::array<TreeSchedule, 3> actions{{{1, 2}, {2, 2}, {2, 3}}};
        bool have_best = false;
        TreeSchedule best{};
        double best_score = -1.0;
        std::size_t best_nodes = 0;
        for (const TreeSchedule action : actions) {
            const auto cost_it = action_costs_.find(
                ActionCostKey{batch_size, action.branching, action.depth});
            if (cost_it == action_costs_.end() || !(cost_it->second > 0.0)) {
                throw std::out_of_range("no tuning-only pulse cost for adaptive action");
            }
            const std::size_t usable_depth = std::min({action.depth, remaining_tokens,
                                                       context_positions});
            double expected_emitted_tokens = 1.0;
            double survival = 1.0;
            for (std::size_t depth = 1; depth < usable_depth; ++depth) {
                const double calibrated = calibrator_.predict(
                    "continued", batch_size, action.branching, depth,
                    root_features);
                const auto state = ema_.find(EmaKey{request_id, batch_size,
                                                     action.branching, depth});
                const double continuation = state == ema_.end() || !state->second.initialized
                    ? calibrated
                    : calibrated_weight_ * calibrated +
                      (1.0 - calibrated_weight_) * state->second.value;
                survival *= continuation;
                expected_emitted_tokens += survival;
            }
            const double score = expected_emitted_tokens / cost_it->second;
            const std::size_t node_count = adaptive_action_node_count(action);
            const bool better = !have_best || score > best_score + 1e-12 ||
                (std::abs(score - best_score) <= 1e-12 &&
                 std::tie(node_count, action.depth, action.branching) <
                 std::tie(best_nodes, best.depth, best.branching));
            if (better) {
                have_best = true;
                best = action;
                best_score = score;
                best_nodes = node_count;
            }
        }
        return best;
    }

    void observe(const RuntimeTreeObservation& observation) {
        if (!observation.continuation_label_available) return;
        EmaKey key{observation.request_id, observation.batch_size,
                   observation.branching, observation.depth + 1};
        ema_[key].update(observation.continued, ema_alpha_);
    }

    std::size_t update_count(std::uint64_t request_id, std::size_t batch_size,
                             std::size_t branching, std::size_t depth) const {
        const auto found = ema_.find(EmaKey{request_id, batch_size, branching, depth});
        return found == ema_.end() ? 0 : found->second.updates;
    }

private:
    struct EmaKey {
        std::uint64_t request_id = 0;
        std::size_t batch_size = 0;
        std::size_t branching = 0;
        std::size_t depth = 0;

        bool operator<(const EmaKey& other) const noexcept {
            return std::tie(request_id, batch_size, branching, depth) <
                   std::tie(other.request_id, other.batch_size, other.branching, other.depth);
        }
    };

    ConfidenceCalibrator calibrator_;
    std::map<ActionCostKey, double> action_costs_;
    double ema_alpha_;
    double calibrated_weight_;
    mutable std::map<EmaKey, EmaState> ema_;
};

}  // namespace pulsar::runtime
