#include "pulsar_speculative_v2.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using pulsar::v2::Context;
using pulsar::v2::Token;
using pulsar::v2::Weights;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_each_first_rejection_position() {
    constexpr std::size_t gamma = 5;
    const Weights q{1.0, 0.0, 0.0};
    std::vector<Token> proposals(gamma, 0);
    std::vector<Weights> q_rows(gamma, q);
    std::vector<Weights> target_rows(gamma, q);
    const Weights cached_bonus{0.0, 0.0, 1.0};

    for (std::size_t first_reject = 0; first_reject < gamma; ++first_reject) {
        target_rows.assign(gamma, q);
        target_rows[first_reject] = Weights{0.0, 1.0, 0.0};
        std::mt19937_64 rng(0xA110000ULL + first_reject);
        const auto result = pulsar::v2::verify_speculative_block(
            proposals, q_rows, target_rows, std::optional<Weights>(cached_bonus), rng);

        require(result.proposed_tokens == proposals,
                "verification must retain the fully drafted block");
        require(result.rejected_position == first_reject,
                "first rejection position was not reported correctly");
        require(result.accepted_count == first_reject,
                "accepted prefix length differs from first rejection position");
        require(result.emitted_tokens.size() == first_reject + 1,
                "first rejection must append one residual and stop");
        for (std::size_t i = 0; i < first_reject; ++i) {
            require(result.emitted_tokens[i] == 0,
                    "accepted proposal was not emitted unchanged");
        }
        require(result.emitted_tokens.back() == 1,
                "rejection did not emit the target-minus-q residual token");
        require(!result.bonus_emitted,
                "a rejection must suppress the target bonus token");
    }
    std::cout << "  first rejection at every block position [PASS]\n";
}

void test_all_accepted_bonus_and_missing_bonus() {
    constexpr std::size_t gamma = 3;
    const std::vector<Token> proposals(gamma, 0);
    const std::vector<Weights> q_rows(gamma, Weights{1.0, 0.0, 0.0});
    const std::vector<Weights> target_rows = q_rows;
    const Weights bonus{0.0, 1.0, 0.0};

    std::mt19937_64 with_bonus_rng(77);
    const auto with_bonus = pulsar::v2::verify_speculative_block(
        proposals, q_rows, target_rows, std::optional<Weights>(bonus), with_bonus_rng);
    require(with_bonus.accepted_count == gamma && !with_bonus.rejected_position,
            "identical q and p rows must accept the entire block");
    require(with_bonus.emitted_tokens.size() == gamma + 1 && with_bonus.bonus_emitted,
            "cached target row must supply exactly one bonus token");
    require(with_bonus.emitted_tokens.back() == 1,
            "bonus token must be drawn from the target row");

    std::mt19937_64 without_bonus_rng(78);
    const auto without_bonus = pulsar::v2::verify_speculative_block(
        proposals, q_rows, target_rows, std::nullopt, without_bonus_rng);
    require(without_bonus.accepted_count == gamma && !without_bonus.bonus_emitted,
            "absent cached row must not produce a bonus token");
    require(without_bonus.emitted_tokens.size() == gamma,
            "block without a cached target row must emit only accepted proposals");

    std::mt19937_64 zero_gamma_rng(79);
    const auto zero_gamma = pulsar::v2::verify_speculative_block(
        {}, {}, {}, std::optional<Weights>(bonus), zero_gamma_rng);
    require(zero_gamma.emitted_tokens == std::vector<Token>{1} &&
                zero_gamma.bonus_emitted,
            "zero-proposal block should sample one available target bonus");
    std::cout << "  all-accepted bonus and missing-row behavior [PASS]\n";
}

struct MarkovFixture {
    std::array<Weights, 3> p{{
        Weights{0.65, 0.25, 0.10},
        Weights{0.10, 0.60, 0.30},
        Weights{0.25, 0.15, 0.60},
    }};
    std::array<Weights, 3> q{{
        Weights{0.50, 0.30, 0.20},
        Weights{0.30, 0.50, 0.20},
        Weights{0.20, 0.40, 0.40},
    }};

    static std::size_t state(const Context& context) {
        return context.empty() ? 0 : context.back() % 3;
    }
    Weights target_row(const Context& context) const { return p[state(context)]; }
    Weights draft_row(const Context& context) const { return q[state(context)]; }
    std::optional<Weights> cached_bonus_row(const Context& context) const {
        // This toy model's complete target table is resident, so this is a
        // genuine cache hit. Production callers may return nullopt instead.
        return target_row(context);
    }
};

std::vector<Token> speculative_sequence(const MarkovFixture& model,
                                        const Context& initial_context,
                                        std::size_t token_count,
                                        std::size_t configured_gamma,
                                        std::mt19937_64& rng) {
    Context context = initial_context;
    std::vector<Token> result;
    result.reserve(token_count);
    while (result.size() < token_count) {
        const std::size_t remaining = token_count - result.size();
        const std::size_t gamma = std::min(configured_gamma, remaining);
        const bool bonus_fits = remaining > gamma;
        const auto block = pulsar::v2::sample_speculative_block(
            context, gamma,
            [&model](const Context& c) { return model.draft_row(c); },
            [&model](const Context& c) { return model.target_row(c); },
            [&model, bonus_fits](const Context& c) -> std::optional<Weights> {
                return bonus_fits ? model.cached_bonus_row(c) : std::nullopt;
            },
            rng);
        require(!block.emitted_tokens.empty(),
                "a positive remaining-token budget must make progress");
        require(block.emitted_tokens.size() <= remaining,
                "speculative block exceeded the fixed sequence stopping limit");
        for (Token token : block.emitted_tokens) {
            result.push_back(token);
            context.push_back(token);
        }
    }
    return result;
}

void test_independent_seed_sequence_and_conditional_laws() {
    const MarkovFixture model;
    constexpr std::size_t sequence_count = 6000;
    constexpr std::size_t tokens_per_sequence = 32;
    constexpr std::size_t gamma = 4;
    std::array<std::array<std::size_t, 3>, 3> spec_counts{};
    std::array<std::array<std::size_t, 3>, 3> ar_counts{};

    // Independent sampler runs use independent seeds; this compares conditional
    // next-token (bigram) laws, not two histograms from one correlated stream.
    std::mt19937_64 speculative_rng(0x5EED5EEDULL);
    std::mt19937_64 autoregressive_rng(0xA17A17A1ULL);
    for (std::size_t sequence = 0; sequence < sequence_count; ++sequence) {
        const Context initial{sequence % 3};
        const std::vector<Token> spec = speculative_sequence(
            model, initial, tokens_per_sequence, gamma, speculative_rng);
        const std::vector<Token> ar = pulsar::v2::sample_autoregressive(
            initial, tokens_per_sequence,
            [&model](const Context& c) { return model.target_row(c); },
            autoregressive_rng);
        require(spec.size() == tokens_per_sequence && ar.size() == tokens_per_sequence,
                "AR and speculative reference must share the fixed token limit");

        Token spec_previous = initial.back();
        Token ar_previous = initial.back();
        for (std::size_t i = 0; i < tokens_per_sequence; ++i) {
            ++spec_counts[spec_previous][spec[i]];
            ++ar_counts[ar_previous][ar[i]];
            spec_previous = spec[i];
            ar_previous = ar[i];
        }
    }

    constexpr long double max_conditional_abs_error = 0.018L;
    constexpr long double max_spec_vs_ar_tv = 0.035L;
    for (std::size_t state = 0; state < 3; ++state) {
        const std::size_t n_spec = spec_counts[state][0] + spec_counts[state][1]
                                 + spec_counts[state][2];
        const std::size_t n_ar = ar_counts[state][0] + ar_counts[state][1]
                               + ar_counts[state][2];
        require(n_spec > 10000 && n_ar > 10000,
                "each tested conditional context needs adequate coverage");
        long double spec_tv = 0.0L;
        long double ar_tv = 0.0L;
        for (std::size_t token = 0; token < 3; ++token) {
            const long double p = model.p[state][token];
            const long double spec_rate =
                static_cast<long double>(spec_counts[state][token]) / n_spec;
            const long double ar_rate =
                static_cast<long double>(ar_counts[state][token]) / n_ar;
            require(std::abs(spec_rate - p) < max_conditional_abs_error,
                    "speculative conditional next-token law differs from target p");
            require(std::abs(ar_rate - p) < max_conditional_abs_error,
                    "plain AR conditional next-token law differs from target p");
            spec_tv += std::abs(spec_rate - p);
            ar_tv += std::abs(ar_rate - p);
        }
        spec_tv *= 0.5L;
        ar_tv *= 0.5L;
        long double between_samplers_tv = 0.0L;
        for (std::size_t token = 0; token < 3; ++token) {
            const long double spec_rate =
                static_cast<long double>(spec_counts[state][token]) / n_spec;
            const long double ar_rate =
                static_cast<long double>(ar_counts[state][token]) / n_ar;
            between_samplers_tv += std::abs(spec_rate - ar_rate);
        }
        between_samplers_tv *= 0.5L;
        require(between_samplers_tv < max_spec_vs_ar_tv,
                "independent speculative and AR conditional laws disagree");
        std::cout << "  context " << state << ": n(spec/ar)=" << n_spec << '/' << n_ar
                  << " conditional TV(spec,p)=" << static_cast<double>(spec_tv)
                  << " TV(ar,p)=" << static_cast<double>(ar_tv)
                  << " TV(spec,ar)=" << static_cast<double>(between_samplers_tv)
                  << " [PASS]\n";
    }
}

void test_context_aware_draft_and_target_rows() {
    const MarkovFixture model;
    std::mt19937_64 rng(314159);
    const auto block = pulsar::v2::sample_speculative_block(
        Context{1}, 4,
        [&model](const Context& c) { return model.draft_row(c); },
        [&model](const Context& c) { return model.target_row(c); },
        [](const Context&) -> std::optional<Weights> { return std::nullopt; }, rng);
    require(block.proposed_tokens.size() == 4,
            "context-aware driver did not draft the requested gamma proposals");
    require(block.emitted_tokens.size() >= 1 && block.emitted_tokens.size() <= 4,
            "context-aware block must stop at rejection or gamma without bonus");

    int bonus_lookups = 0;
    std::mt19937_64 rejection_rng(271828);
    const auto rejected = pulsar::v2::sample_speculative_block(
        Context{}, 2,
        [](const Context&) { return Weights{1.0, 0.0}; },
        [](const Context&) { return Weights{0.0, 1.0}; },
        [&bonus_lookups](const Context&) -> std::optional<Weights> {
            ++bonus_lookups;
            return Weights{0.0, 1.0};
        },
        rejection_rng);
    require(rejected.rejected_position == 0 && bonus_lookups == 0,
            "first rejection must stop before any bonus-row cache lookup");
    std::cout << "  context-aware q drafting and p row lookup [PASS]\n";
}

}  // namespace

int main() {
    try {
        test_each_first_rejection_position();
        test_all_accepted_bonus_and_missing_bonus();
        test_context_aware_draft_and_target_rows();
        test_independent_seed_sequence_and_conditional_laws();
        std::cout << "PULSAR_SPECULATIVE_V2_TEST: PASS\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "PULSAR_SPECULATIVE_V2_TEST: FAIL: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
