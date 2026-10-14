#include "pulsar_core_v1.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using pulsar::v1::DistributionView;
using pulsar::v1::Token;
using pulsar::v1::Weights;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::vector<long double> normalized(const Weights& weights) {
    long double total = 0.0L;
    for (double weight : weights) {
        total += static_cast<long double>(weight);
    }
    require(total > 0.0L, "test fixture must have positive total mass");
    std::vector<long double> result;
    result.reserve(weights.size());
    for (double weight : weights) {
        result.push_back(static_cast<long double>(weight) / total);
    }
    return result;
}

void test_invalid_input() {
    const Weights empty;
    bool rejected = false;
    try {
        (void)pulsar::v1::detail::total_mass(DistributionView(empty));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "empty distributions must be rejected");

    const Weights negative{0.5, -0.1, 0.6};
    rejected = false;
    try {
        (void)pulsar::v1::detail::total_mass(DistributionView(negative));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "negative weights must be rejected");

    const Weights nonfinite{0.5, std::numeric_limits<double>::infinity()};
    rejected = false;
    try {
        (void)pulsar::v1::detail::total_mass(DistributionView(nonfinite));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "non-finite weights must be rejected");
}

void check_algebra(const std::string& label, const Weights& p_weights,
                   const Weights& pi_weights) {
    require(p_weights.size() == pi_weights.size(),
            label + ": test fixture vocabulary mismatch");
    const std::vector<long double> p = normalized(p_weights);
    const std::vector<long double> pi = normalized(pi_weights);
    const long double z = pulsar::v1::residual_mass(
        DistributionView(p_weights), DistributionView(pi_weights));

    if (z == 0.0L) {
        bool undefined = false;
        try {
            (void)pulsar::v1::residual_distribution(
                DistributionView(p_weights), DistributionView(pi_weights));
        } catch (const std::logic_error&) {
            undefined = true;
        }
        require(undefined, label + ": zero residual must not create a fallback law");
    } else {
        const Weights residual = pulsar::v1::residual_distribution(
            DistributionView(p_weights), DistributionView(pi_weights));
        long double residual_total = 0.0L;
        for (double value : residual) {
            residual_total += static_cast<long double>(value);
        }
        require(std::abs(residual_total - 1.0L) < 2e-15L,
                label + ": residual law must normalize to one");

        // Every positive residual must be represented on its positive support;
        // there is intentionally no epsilon-based equality shortcut.
        for (std::size_t i = 0; i < p.size(); ++i) {
            if (p[i] > pi[i]) {
                require(residual[i] > 0.0,
                        label + ": positive residual support was discarded");
            } else {
                require(residual[i] == 0.0,
                        label + ": residual assigned mass outside max(p-pi,0)");
            }
        }

        for (std::size_t i = 0; i < p.size(); ++i) {
            const long double acceptance_probability = pi[i] > 0.0L
                ? std::min(1.0L, p[i] / pi[i])
                : 0.0L;
            const long double accepted_mass = pi[i] * acceptance_probability;
            const long double reconstructed = accepted_mass
                + z * static_cast<long double>(residual[i]);
            const long double error = std::abs(reconstructed - p[i]);
            if (error > 3e-15L) {
                std::ostringstream message;
                message << label << ": pointwise law mismatch at token " << i
                        << " (error=" << static_cast<double>(error) << ')';
                throw std::runtime_error(message.str());
            }
        }
    }
}

void test_deterministic_algebra_fixtures() {
    check_algebra("identical laws", Weights{1.0, 2.0, 7.0},
                  Weights{1.0, 2.0, 7.0});
    check_algebra("proportional unnormalized weights", Weights{0.25, 0.5, 1.75},
                  Weights{1.0, 2.0, 7.0});
    check_algebra("zero proposal support where p is positive",
                  Weights{0.4, 0.3, 0.3}, Weights{0.5, 0.5, 0.0});
    check_algebra("concentrated target", Weights{1.0, 0.0, 0.0},
                  Weights{0.0, 1.0, 0.0});
    check_algebra("nearly disjoint support",
                  Weights{0.499999, 0.5, 0.000001, 0.0},
                  Weights{0.0, 0.000001, 0.499999, 0.5});

    const Weights near_p{0.5000000000005, 0.4999999999995};
    const Weights near_pi{0.4999999999995, 0.5000000000005};
    check_algebra("small positive floating residual", near_p, near_pi);
    const long double small_z = pulsar::v1::residual_mass(
        DistributionView(near_p), DistributionView(near_pi));
    require(small_z > 0.0L && std::abs(small_z - 1e-12L) < 2e-16L,
            "small positive residual must not be treated as zero");

    const Weights p{0.10, 0.20, 0.70};
    const Weights q{0.80, 0.10, 0.10};
    const Weights pi{0.25, 0.25, 0.50};
    (void)q;  // Draft q is deliberately not an input to the one-token verifier.
    check_algebra("p != pi (and independent draft q)", p, pi);
    std::cout << "  deterministic pointwise algebra fixtures [PASS]\n";
}

void test_randomized_algebra_cases() {
    std::mt19937_64 rng(0x51A6E1ULL);
    std::uniform_int_distribution<int> vocab_size(2, 16);
    std::uniform_int_distribution<int> weight(0, 100);

    for (int trial = 0; trial < 2000; ++trial) {
        const int size = vocab_size(rng);
        Weights p(static_cast<std::size_t>(size));
        Weights pi(static_cast<std::size_t>(size));
        for (int i = 0; i < size; ++i) {
            p[static_cast<std::size_t>(i)] = weight(rng);
            pi[static_cast<std::size_t>(i)] = weight(rng);
        }
        if (trial % 5 == 0) {
            pi = p;  // Explicit equality cases among randomized vectors.
        }
        if (std::all_of(p.begin(), p.end(), [](double x) { return x == 0.0; })) {
            p[0] = 1.0;
        }
        if (std::all_of(pi.begin(), pi.end(), [](double x) { return x == 0.0; })) {
            pi[static_cast<std::size_t>(size - 1)] = 1.0;
        }
        check_algebra("randomized algebra trial " + std::to_string(trial), p, pi);
    }
    std::cout << "  randomized pointwise algebra (2000 cases) [PASS]\n";
}

struct MonteCarloFixture {
    std::string label;
    Weights p;
    Weights pi;
};

void append_monte_carlo_fixture(std::vector<MonteCarloFixture>& fixtures,
                                const std::string& label, const Weights& p,
                                const Weights& pi) {
    check_algebra(label, p, pi);
    fixtures.push_back({label, p, pi});
}

void test_monte_carlo_confidence_bounds() {
    std::vector<MonteCarloFixture> fixtures;
    append_monte_carlo_fixture(fixtures, "MC p=pi",
                               Weights{0.11, 0.23, 0.29, 0.37},
                               Weights{0.11, 0.23, 0.29, 0.37});
    append_monte_carlo_fixture(fixtures, "MC zero proposal support",
                               Weights{0.4, 0.3, 0.3},
                               Weights{0.5, 0.5, 0.0});
    append_monte_carlo_fixture(fixtures, "MC concentrated",
                               Weights{1.0, 0.0, 0.0, 0.0},
                               Weights{0.0, 1.0, 0.0, 0.0});
    append_monte_carlo_fixture(fixtures, "MC nearly disjoint",
                               Weights{0.499, 0.5, 0.001, 0.0},
                               Weights{0.0, 0.001, 0.499, 0.5});
    append_monte_carlo_fixture(fixtures, "MC small nonzero overlap",
                               Weights{0.499, 0.499, 0.002},
                               Weights{0.002, 0.499, 0.499});

    // A tiny 3-state Markov model: each row is a transition law p(.|state),
    // and the corresponding pi row is the proposal policy for that state.
    const std::vector<Weights> markov_p{
        {0.70, 0.20, 0.10},
        {0.10, 0.75, 0.15},
        {0.25, 0.15, 0.60},
    };
    const std::vector<Weights> markov_pi{
        {0.50, 0.35, 0.15},
        {0.20, 0.60, 0.20},
        {0.10, 0.30, 0.60},
    };
    for (std::size_t state = 0; state < markov_p.size(); ++state) {
        append_monte_carlo_fixture(fixtures,
                                   "Markov transition row from state "
                                       + std::to_string(state),
                                   markov_p[state], markov_pi[state]);
    }

    std::mt19937_64 fixture_rng(0xC0FFEEULL);
    std::uniform_int_distribution<int> size_dist(2, 8);
    std::uniform_int_distribution<int> weight_dist(0, 100);
    for (int i = 0; i < 12; ++i) {
        const int size = size_dist(fixture_rng);
        Weights p(static_cast<std::size_t>(size));
        Weights pi(static_cast<std::size_t>(size));
        for (int token = 0; token < size; ++token) {
            p[static_cast<std::size_t>(token)] = weight_dist(fixture_rng);
            pi[static_cast<std::size_t>(token)] = weight_dist(fixture_rng);
        }
        if (std::all_of(p.begin(), p.end(), [](double x) { return x == 0.0; })) {
            p[0] = 1.0;
        }
        if (std::all_of(pi.begin(), pi.end(), [](double x) { return x == 0.0; })) {
            pi[0] = 1.0;
        }
        append_monte_carlo_fixture(fixtures, "random MC case " + std::to_string(i),
                                   p, pi);
    }

    constexpr int samples_per_fixture = 40000;
    constexpr long double family_alpha = 0.01L;
    std::size_t total_bins = 0;
    for (const MonteCarloFixture& fixture : fixtures) {
        total_bins += fixture.p.size();
    }
    // Hoeffding's inequality with a union bound over every token in every
    // fixture gives a simultaneous (at least 99%) confidence envelope.
    const long double tolerance = std::sqrt(
        std::log(2.0L * static_cast<long double>(total_bins) / family_alpha)
        / (2.0L * samples_per_fixture));

    std::mt19937_64 rng(0xBADC0DEULL);
    for (const MonteCarloFixture& fixture : fixtures) {
        const std::vector<long double> expected = normalized(fixture.p);
        std::vector<int> histogram(fixture.p.size(), 0);
        for (int sample = 0; sample < samples_per_fixture; ++sample) {
            const pulsar::v1::VerifiedToken result =
                pulsar::v1::sample_verified_token(
                    DistributionView(fixture.p), DistributionView(fixture.pi), rng);
            ++histogram[result.emitted_token];
        }
        for (std::size_t token = 0; token < expected.size(); ++token) {
            const long double observed =
                static_cast<long double>(histogram[token]) / samples_per_fixture;
            const long double error = std::abs(observed - expected[token]);
            if (error > tolerance) {
                std::ostringstream message;
                message << fixture.label << ": token " << token
                        << " exceeded simultaneous Hoeffding bound (error="
                        << static_cast<double>(error) << ", bound="
                        << static_cast<double>(tolerance) << ')';
                throw std::runtime_error(message.str());
            }
        }
    }
    std::cout << "  Monte Carlo: " << fixtures.size() << " fixtures x "
              << samples_per_fixture << " draws; 99% simultaneous Hoeffding "
              << "bound +/- " << static_cast<double>(tolerance) << " [PASS]\n";
}

void test_generic_random_engine() {
    const Weights p{0.2, 0.3, 0.5};
    const Weights pi{0.4, 0.4, 0.2};
    std::minstd_rand rng(12345);
    for (int i = 0; i < 1000; ++i) {
        const auto result = pulsar::v1::sample_verified_token(
            DistributionView(p), DistributionView(pi), rng);
        require(result.proposal_token < p.size() && result.emitted_token < p.size(),
                "generic URBG produced an out-of-range token");
    }
    std::cout << "  generic standard URBG API [PASS]\n";
}

void test_nonidentical_policy_and_fallthrough() {
    using namespace pulsar::v1;

    // The retained tree is built from q, while collapse samples from pi != q.
    const Weights p{0.10, 0.20, 0.70};
    const Weights q{0.80, 0.10, 0.10};
    const Weights pi{0.25, 0.25, 0.50};
    const std::vector<Token> retained = top_k_candidates(DistributionView(q), 1);
    require(retained == std::vector<Token>{0}, "unexpected q top-1 candidate");

    constexpr int samples = 300000;
    std::vector<int> histogram(p.size(), 0);
    int accepted = 0;
    int accepted_off_tree = 0;
    int residuals = 0;
    std::mt19937_64 rng(20261012);

    for (int i = 0; i < samples; ++i) {
        const PulseToken result = sample_pulse_token(
            DistributionView(p), DistributionView(pi), retained, rng);
        ++histogram[result.emitted_token];
        accepted += result.accepted ? 1 : 0;

        if (!result.accepted) {
            ++residuals;
            require(result.emitted_token == 2,
                    "residual must be normalized max(p - pi, 0)");
            require(!result.continue_pulse,
                    "a rejected token must stop the pulse");
        } else if (result.proposal_token == 0) {
            require(result.emitted_token == 0 && result.continue_pulse,
                    "accepted retained token must be emitted and may continue");
        } else {
            ++accepted_off_tree;
            require(result.emitted_token == result.proposal_token,
                    "accepted off-tree token must not be replaced");
            require(!result.continue_pulse,
                    "accepted off-tree token must stop pulse extension");
        }
    }

    double tv = 0.0;
    for (std::size_t token = 0; token < p.size(); ++token) {
        tv += std::abs(histogram[token] / static_cast<double>(samples) - p[token]);
    }
    tv *= 0.5;
    const double accepted_rate = accepted / static_cast<double>(samples);
    const double off_tree_rate = accepted_off_tree / static_cast<double>(samples);
    const double residual_rate = residuals / static_cast<double>(samples);

    require(tv < 0.008, "empirical output law differs from p beyond MC tolerance");
    require(std::abs(accepted_rate - 0.8) < 0.006,
            "measured acceptance rate differs from sum min(p,pi)");
    require(std::abs(off_tree_rate - 0.7) < 0.006,
            "accepted off-tree mass differs from its exact probability");
    require(std::abs(residual_rate - 0.2) < 0.006,
            "rejection rate differs from TV(p,pi)");

    std::cout << "  pi!=q fall-through: TV=" << tv
              << " acceptance=" << accepted_rate
              << " off-tree-accepted=" << off_tree_rate
              << " residual=" << residual_rate << " [PASS]\n";
}

void test_equal_laws_never_reject() {
    using namespace pulsar::v1;
    const Weights p{0.125, 0.375, 0.50};
    const std::vector<Token> no_retained_children;
    std::mt19937_64 rng(314159);

    for (int i = 0; i < 50000; ++i) {
        const PulseToken result = sample_pulse_token(
            DistributionView(p), DistributionView(p), no_retained_children, rng);
        require(result.accepted, "p == pi must never reject");
        require(result.emitted_token == result.proposal_token,
                "accepted token must be emitted unchanged");
        require(!result.continue_pulse,
                "empty candidate tree only stops continuation");
    }

    bool residual_rejected = false;
    try {
        (void)residual_distribution(DistributionView(p), DistributionView(p));
    } catch (const std::logic_error&) {
        residual_rejected = true;
    }
    require(residual_rejected,
            "zero-mass residual must be explicit, not replaced by target");
    std::cout << "  p==pi: no rejections and zero residual is undefined [PASS]\n";
}

}  // namespace

int main() {
    try {
        test_invalid_input();
        test_deterministic_algebra_fixtures();
        test_randomized_algebra_cases();
        test_generic_random_engine();
        test_monte_carlo_confidence_bounds();
        test_nonidentical_policy_and_fallthrough();
        test_equal_laws_never_reject();
        std::cout << "PULSAR_V1_CORE_TEST: PASS\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "PULSAR_V1_CORE_TEST: FAIL: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
