#include "pulsar_tree_v3.hpp"
#include "pulsar_speculative_v2.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using pulsar::v3::CandidateTree;
using pulsar::v3::Context;
using pulsar::v3::StopReason;
using pulsar::v3::Token;
using pulsar::v3::Weights;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <class Fn>
void require_invalid_argument(Fn&& fn, const std::string& message) {
    bool threw = false;
    try {
        fn();
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, message);
}

void test_tree_is_bounded_and_keeps_full_q_rows() {
    std::size_t q_calls = 0;
    const auto tree = pulsar::v3::build_candidate_tree(
        Context{}, 2, 3,
        [&q_calls](const Context& context) {
            ++q_calls;
            if (context.empty()) {
                return Weights{0.5, 0.3, 0.2};
            }
            return context.back() % 2 == 0 ? Weights{0.2, 0.7, 0.1}
                                            : Weights{0.6, 0.1, 0.3};
        });

    require(tree.nodes.size() <= 1 + 2 + 4,
            "tree exceeded the fixed branching/depth node bound");
    require(q_calls == tree.nodes.size(),
            "each retained node must receive exactly one context-specific q row");
    require(tree.nodes.front().q == Weights({0.5, 0.3, 0.2}),
            "root q row was truncated to the retained children");
    require(tree.nodes.front().children.size() == 2,
            "root did not retain the configured top-B positive-q children");

    for (std::size_t i = 0; i < tree.nodes.size(); ++i) {
        const auto& node = tree.nodes[i];
        require(node.q.size() == 3, "full q row was not retained at a node");
        if (node.depth + 1 < tree.max_depth) {
            require(node.children.size() <= tree.branch_factor,
                    "nonterminal node retained more than B children");
        } else {
            require(node.children.empty(), "terminal-depth node must not expand");
        }
        for (std::size_t child_index : node.children) {
            require(child_index > i && child_index < tree.nodes.size(),
                    "tree is not a valid forward-indexed prefix tree");
            const auto& child = tree.nodes[child_index];
            Context expected = node.context;
            expected.push_back(child.incoming_token);
            require(child.parent_index == i && child.depth == node.depth + 1 &&
                        child.context == expected,
                    "child metadata/context does not match its exact parent path");
        }
    }
    std::cout << "  bounded prefix tree retains complete q rows [PASS]\n";
}

void test_accepted_off_tree_token_falls_through_unchanged() {
    const Context root{0};
    const auto tree = pulsar::v3::build_candidate_tree(
        root, 1, 4,
        [](const Context&) { return Weights{0.9, 0.1, 0.0}; });
    require(tree.nodes.front().children.size() == 1 &&
                tree.nodes[tree.nodes.front().children.front()].incoming_token == 0,
            "fixture did not prune token 2 from the retained root children");

    std::size_t target_calls = 0;
    std::mt19937_64 rng(991);
    const auto result = pulsar::v3::sample_candidate_tree(
        tree, root,
        [&target_calls](const Context&) {
            ++target_calls;
            return Weights{0.0, 0.0, 1.0};
        },
        [](const Context&) { return Weights{0.0, 0.0, 1.0}; }, rng);

    require(result.proposed_tokens == std::vector<Token>{2},
            "effective proposal was restricted to retained q children");
    require(result.emitted_tokens == std::vector<Token>{2} &&
                result.accepted_count == 1 && !result.rejected_position,
            "accepted off-tree proposal was not emitted unchanged");
    require(result.stop_reason == StopReason::no_retained_child,
            "accepted off-tree token did not stop continuation");
    require(target_calls == 1,
            "tree sampler continued evaluating target rows after fall-through");
    std::cout << "  accepted off-tree token is emitted unchanged and stops [PASS]\n";
}

void test_default_pi_q_uses_full_unpruned_row() {
    const Context root{1};
    const auto tree = pulsar::v3::build_candidate_tree(
        root, 1, 2,
        [](const Context&) { return Weights{0.6, 0.4}; });
    constexpr std::size_t draws = 20000;
    std::size_t token_zero = 0;
    std::mt19937_64 rng(0xF0117ULL);
    for (std::size_t i = 0; i < draws; ++i) {
        const auto result = pulsar::v3::sample_candidate_tree(
            tree, root,
            [](const Context&) { return Weights{0.6, 0.4}; }, rng);
        require(!result.proposed_tokens.empty() &&
                    result.emitted_tokens.front() == result.proposed_tokens.front(),
                "pi=q path changed an accepted proposal");
        token_zero += result.proposed_tokens.front() == 0;
    }
    const long double rate = static_cast<long double>(token_zero) / draws;
    require(std::abs(rate - 0.6L) < 0.02L,
            "default pi=q draw was truncated/renormalized to retained children");
    std::cout << "  default pi=q retains the full proposal law: P(0)="
              << static_cast<double>(rate) << " [PASS]\n";
}

void test_rejection_uses_residual_and_stops() {
    const Context root{2};
    const auto tree = pulsar::v3::build_candidate_tree(
        root, 1, 3,
        [](const Context&) { return Weights{1.0, 0.0}; });
    std::size_t target_calls = 0;
    std::mt19937_64 rng(1234);
    const auto result = pulsar::v3::sample_candidate_tree(
        tree, root,
        [&target_calls](const Context&) {
            ++target_calls;
            return Weights{0.0, 1.0};
        },
        [](const Context&) { return Weights{1.0, 0.0}; }, rng);

    require(result.proposed_tokens == std::vector<Token>{0},
            "rejection fixture did not sample the effective proposal");
    require(result.emitted_tokens == std::vector<Token>{1} &&
                result.accepted_count == 0 && result.rejected_position == 0,
            "rejection did not emit the verifier residual");
    require(result.stop_reason == StopReason::rejected && target_calls == 1,
            "rejection must stop before any continuation");
    std::cout << "  rejection emits the effective-pi residual and stops [PASS]\n";
}

void test_stale_root_and_child_contexts_are_rejected() {
    const Context root{0};
    const auto tree = pulsar::v3::build_candidate_tree(
        root, 1, 2,
        [](const Context&) { return Weights{1.0, 0.0}; });
    const auto p = [](const Context&) { return Weights{1.0, 0.0}; };
    const auto pi = [](const Context&) { return Weights{1.0, 0.0}; };

    std::mt19937_64 stale_root_rng(1);
    require_invalid_argument(
        [&] { (void)pulsar::v3::sample_candidate_tree(
                  tree, Context{1}, p, pi, stale_root_rng); },
        "sampler accepted a stale initial context");

    CandidateTree stale_child = tree;
    require(stale_child.nodes.size() > 1, "stale-child fixture needs a child");
    stale_child.nodes[1].context = Context{7};
    std::mt19937_64 stale_child_rng(2);
    require_invalid_argument(
        [&] { (void)pulsar::v3::sample_candidate_tree(
                  stale_child, root, p, pi, stale_child_rng); },
        "sampler followed a child whose stored context was stale");
    std::cout << "  stale root and stale child contexts cannot continue [PASS]\n";
}

void test_effective_proposal_verifier_identity() {
    const std::vector<std::pair<Weights, Weights>> laws{
        {{0.65, 0.25, 0.10}, {0.20, 0.50, 0.30}},
        {{0.0, 0.7, 0.3}, {0.4, 0.0, 0.6}},
        {{0.2, 0.3, 0.5}, {0.2, 0.3, 0.5}},
    };
    constexpr long double tolerance = 3.0e-15L;
    for (const auto& law : laws) {
        const Weights& p_weights = law.first;
        const Weights& pi_weights = law.second;
        const auto p_view = pulsar::v1::DistributionView(p_weights);
        const auto pi_view = pulsar::v1::DistributionView(pi_weights);
        const long double p_sum = pulsar::v1::detail::total_mass(p_view);
        const long double pi_sum = pulsar::v1::detail::total_mass(pi_view);
        const long double z = pulsar::v1::residual_mass(p_view, pi_view);
        Weights residual(p_weights.size(), 0.0);
        if (z > 0.0L) {
            residual = pulsar::v1::residual_distribution(p_view, pi_view);
        }
        for (std::size_t token = 0; token < p_weights.size(); ++token) {
            const long double p = static_cast<long double>(p_weights[token]) / p_sum;
            const long double pi = static_cast<long double>(pi_weights[token]) / pi_sum;
            const long double acceptance = pi > 0.0L ? std::min(1.0L, p / pi) : 0.0L;
            const long double lhs = pi * acceptance + z * residual[token];
            require(std::abs(lhs - p) <= tolerance,
                    "pi*a + Z*r = p failed for an effective-proposal row");
        }
    }
    std::cout << "  deterministic effective-pi verifier identity [PASS]\n";
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
    std::array<Weights, 3> pi{{
        Weights{0.40, 0.40, 0.20},
        Weights{0.15, 0.55, 0.30},
        Weights{0.30, 0.20, 0.50},
    }};

    static std::size_t state(const Context& context) {
        return context.empty() ? 0 : context.back() % 3;
    }
    Weights target_row(const Context& context) const { return p[state(context)]; }
    Weights draft_row(const Context& context) const { return q[state(context)]; }
    Weights effective_pi_row(const Context& context) const { return pi[state(context)]; }
};

std::vector<Token> tree_sequence(const MarkovFixture& model,
                                 const Context& initial_context,
                                 std::size_t token_count,
                                 std::size_t branch_factor,
                                 std::size_t configured_depth,
                                 std::mt19937_64& rng) {
    Context context = initial_context;
    std::vector<Token> result;
    result.reserve(token_count);
    while (result.size() < token_count) {
        const std::size_t remaining = token_count - result.size();
        const std::size_t depth = std::min(configured_depth, remaining);
        const CandidateTree tree = pulsar::v3::build_candidate_tree(
            context, branch_factor, depth,
            [&model](const Context& c) { return model.draft_row(c); });
        const auto block = pulsar::v3::sample_candidate_tree(
            tree, context,
            [&model](const Context& c) { return model.target_row(c); },
            [&model](const Context& c) { return model.effective_pi_row(c); }, rng);
        require(!block.emitted_tokens.empty(),
                "a positive output budget must make progress");
        require(block.emitted_tokens.size() <= remaining,
                "tree block exceeded the fixed output-token limit");
        for (Token token : block.emitted_tokens) {
            result.push_back(token);
            context.push_back(token);
        }
    }
    return result;
}

void test_conditional_tree_law_matches_target_and_ar() {
    const MarkovFixture model;
    constexpr std::size_t sequence_count = 5000;
    constexpr std::size_t tokens_per_sequence = 30;
    constexpr std::size_t branch_factor = 1;
    constexpr std::size_t depth = 4;
    std::array<std::array<std::size_t, 3>, 3> tree_counts{};
    std::array<std::array<std::size_t, 3>, 3> ar_counts{};

    // Independent seeds compare conditional transition laws, not a shared stream.
    std::mt19937_64 tree_rng(0x7AEE003ULL);
    std::mt19937_64 ar_rng(0xA770003ULL);
    for (std::size_t sequence = 0; sequence < sequence_count; ++sequence) {
        const Context initial{sequence % 3};
        const std::vector<Token> tree = tree_sequence(
            model, initial, tokens_per_sequence, branch_factor, depth, tree_rng);
        const std::vector<Token> ar = pulsar::v2::sample_autoregressive(
            initial, tokens_per_sequence,
            [&model](const Context& c) { return model.target_row(c); }, ar_rng);
        require(tree.size() == tokens_per_sequence && ar.size() == tokens_per_sequence,
                "tree and AR references must share the fixed sequence token limit");

        Token tree_previous = initial.back();
        Token ar_previous = initial.back();
        for (std::size_t i = 0; i < tokens_per_sequence; ++i) {
            ++tree_counts[tree_previous][tree[i]];
            ++ar_counts[ar_previous][ar[i]];
            tree_previous = tree[i];
            ar_previous = ar[i];
        }
    }

    constexpr long double max_conditional_abs_error = 0.018L;
    constexpr long double max_tree_vs_ar_tv = 0.035L;
    for (std::size_t state = 0; state < 3; ++state) {
        const std::size_t n_tree = tree_counts[state][0] + tree_counts[state][1] +
                                   tree_counts[state][2];
        const std::size_t n_ar = ar_counts[state][0] + ar_counts[state][1] +
                                 ar_counts[state][2];
        require(n_tree > 7000 && n_ar > 7000,
                "each tested conditional context needs adequate coverage");
        long double tree_tv = 0.0L;
        long double ar_tv = 0.0L;
        long double between_tv = 0.0L;
        for (std::size_t token = 0; token < 3; ++token) {
            const long double target = model.p[state][token];
            const long double tree_rate =
                static_cast<long double>(tree_counts[state][token]) / n_tree;
            const long double ar_rate =
                static_cast<long double>(ar_counts[state][token]) / n_ar;
            require(std::abs(tree_rate - target) < max_conditional_abs_error,
                    "tree-mode conditional next-token law differs from target p");
            require(std::abs(ar_rate - target) < max_conditional_abs_error,
                    "AR conditional next-token law differs from target p");
            tree_tv += std::abs(tree_rate - target);
            ar_tv += std::abs(ar_rate - target);
            between_tv += std::abs(tree_rate - ar_rate);
        }
        tree_tv *= 0.5L;
        ar_tv *= 0.5L;
        between_tv *= 0.5L;
        require(between_tv < max_tree_vs_ar_tv,
                "independent tree and AR conditional laws disagree");
        std::cout << "  context " << state << ": n(tree/ar)=" << n_tree << '/' << n_ar
                  << " conditional TV(tree,p)=" << static_cast<double>(tree_tv)
                  << " TV(ar,p)=" << static_cast<double>(ar_tv)
                  << " TV(tree,ar)=" << static_cast<double>(between_tv) << " [PASS]\n";
    }
}

}  // namespace

int main() {
    try {
        test_tree_is_bounded_and_keeps_full_q_rows();
        test_accepted_off_tree_token_falls_through_unchanged();
        test_default_pi_q_uses_full_unpruned_row();
        test_rejection_uses_residual_and_stops();
        test_stale_root_and_child_contexts_are_rejected();
        test_effective_proposal_verifier_identity();
        test_conditional_tree_law_matches_target_and_ar();
        std::cout << "PULSAR_TREE_V3_TEST: PASS\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "PULSAR_TREE_V3_TEST: FAIL: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
