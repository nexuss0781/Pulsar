// Pulsar Stage 3: bounded candidate-tree continuation with exact fall-through.
// The tree is built from full q rows; retention controls continuation only.
// Every verifier draw and ratio use the same explicit effective proposal pi.
#pragma once

#include "pulsar_core_v1.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pulsar::v3 {

using Token = v1::Token;
using Weights = v1::Weights;
using Context = std::vector<Token>;

inline constexpr std::size_t no_parent = std::numeric_limits<std::size_t>::max();

struct CandidateTreeNode {
    Context context;
    std::size_t depth = 0;
    std::size_t parent_index = no_parent;
    Token incoming_token = 0;
    // Keep the complete q row: child pruning must never truncate the verifier law.
    Weights q;
    std::vector<std::size_t> children;
};

struct CandidateTree {
    Context root_context;
    std::size_t branch_factor = 0;
    // Number of verifier positions, including the root position. A value of 1
    // allows one draw at the root; max_depth must therefore be positive.
    std::size_t max_depth = 0;
    std::size_t vocabulary_size = 0;
    std::vector<CandidateTreeNode> nodes;
};

enum class StopReason {
    rejected,
    no_retained_child,
    depth_limit,
    terminal_token,
};

struct CandidateTreeResult {
    std::vector<Token> proposed_tokens;
    // Accepted proposals unchanged, or one residual replacement at rejection.
    std::vector<Token> emitted_tokens;
    std::size_t accepted_count = 0;
    // Position of the rejected proposal within this pulse, if any.
    std::optional<std::size_t> rejected_position;
    StopReason stop_reason = StopReason::depth_limit;
    struct Decision {
        std::size_t depth = 0;
        double q_max_probability = 0.0;
        double q_top_two_margin = 0.0;
        double q_normalized_entropy = 0.0;
        double proposal_probability = 0.0;
        bool accepted = false;
        bool continuation_label_available = false;
        bool continued = false;
    };
    std::vector<Decision> decisions;
};

namespace detail {

inline std::vector<Token> retained_tokens(v1::DistributionView q,
                                          std::size_t branch_factor) {
    (void)v1::detail::total_mass(q);
    std::vector<Token> ranked = v1::top_k_candidates(q, q.size);
    ranked.erase(std::remove_if(ranked.begin(), ranked.end(), [q](Token token) {
                     return !(q.data[token] > 0.0);
                 }),
                 ranked.end());
    if (ranked.size() > branch_factor) {
        ranked.resize(branch_factor);
    }
    return ranked;
}

// This implementation accepts a two-argument accessor so the default overload
// can return the exact full q row stored on the current node (pi = q).
template <class TargetRowFn, class EffectivePiAccessor, class URBG,
 class TerminalTokenFn>
inline CandidateTreeResult sample_candidate_tree_impl(
    const CandidateTree& tree,
    const Context& initial_context,
    TargetRowFn&& target_row,
    EffectivePiAccessor&& effective_pi_at_node,
    URBG& rng,
    TerminalTokenFn&& is_terminal_token,
    bool capture_decisions = false) {
    if (tree.nodes.empty() || tree.max_depth == 0 || tree.vocabulary_size == 0) {
        throw std::invalid_argument("candidate tree must have a root and positive depth");
    }
    if (initial_context != tree.root_context || tree.nodes.front().context != tree.root_context) {
        throw std::invalid_argument("initial context does not match the candidate tree root");
    }
    if (tree.nodes.front().depth != 0 || tree.nodes.front().parent_index != no_parent) {
        throw std::invalid_argument("candidate tree root metadata is invalid");
    }

    CandidateTreeResult result;
    result.proposed_tokens.reserve(tree.max_depth);
    result.emitted_tokens.reserve(tree.max_depth);
    Context context = initial_context;
    std::size_t node_index = 0;

    for (std::size_t step = 0; step < tree.max_depth; ++step) {
        if (node_index >= tree.nodes.size()) {
            throw std::invalid_argument("candidate-tree path references a missing node");
        }
        const CandidateTreeNode& node = tree.nodes[node_index];
        if (node.depth != step || node.context != context) {
            throw std::invalid_argument("candidate-tree path has a stale context or depth");
        }
        if (node.q.size() != tree.vocabulary_size) {
            throw std::invalid_argument("candidate-tree q row has an inconsistent vocabulary");
        }
        (void)v1::detail::total_mass(v1::DistributionView(node.q));

        Weights target = target_row(context);
        Weights effective_pi = effective_pi_at_node(context, node);
        if (target.size() != tree.vocabulary_size ||
            effective_pi.size() != tree.vocabulary_size) {
            throw std::invalid_argument("target, q, and effective pi must share one vocabulary");
        }

        const v1::VerifiedToken checked = v1::sample_verified_token(
            v1::DistributionView(target), v1::DistributionView(effective_pi), rng);
        if (capture_decisions) {
            const v1::detail::DistributionFeatures q_features = v1::detail::features(
                v1::DistributionView(node.q));
            const long double effective_pi_mass = v1::detail::total_mass(
                v1::DistributionView(effective_pi));
            const double proposal_probability = static_cast<double>(
                static_cast<long double>(effective_pi[checked.proposal_token]) /
                effective_pi_mass);
            result.decisions.push_back(CandidateTreeResult::Decision{
                step, q_features.maximum_probability, q_features.top_two_margin,
                q_features.normalized_entropy, proposal_probability, checked.accepted,
                step + 1 < tree.max_depth, false});
        }
        result.proposed_tokens.push_back(checked.proposal_token);
        if (!checked.accepted) {
            result.rejected_position = step;
            result.emitted_tokens.push_back(checked.emitted_token);
            result.stop_reason = StopReason::rejected;
            return result;
        }

        // The accepted draw always remains eligible, even when pruning omitted it.
        result.emitted_tokens.push_back(checked.proposal_token);
        ++result.accepted_count;

        if (is_terminal_token(checked.proposal_token)) {
            result.stop_reason = StopReason::terminal_token;
            return result;
        }

        if (step + 1 == tree.max_depth) {
            result.stop_reason = StopReason::depth_limit;
            return result;
        }

        const std::vector<Token> expected = retained_tokens(
            v1::DistributionView(node.q), tree.branch_factor);
        if (node.children.size() != expected.size()) {
            throw std::invalid_argument("candidate-tree child count disagrees with retained q tokens");
        }

        std::size_t selected_child = no_parent;
        for (std::size_t child_position = 0; child_position < node.children.size();
             ++child_position) {
            const std::size_t child_index = node.children[child_position];
            if (child_index <= node_index || child_index >= tree.nodes.size()) {
                throw std::invalid_argument("candidate-tree child index is invalid");
            }
            const CandidateTreeNode& child = tree.nodes[child_index];
            const Token retained_token = expected[child_position];
            Context expected_context = context;
            expected_context.push_back(retained_token);
            if (child.parent_index != node_index || child.depth != step + 1 ||
                child.incoming_token != retained_token || child.context != expected_context ||
                child.q.size() != tree.vocabulary_size) {
                throw std::invalid_argument("candidate-tree child is non-child or has stale context");
            }
            (void)v1::detail::total_mass(v1::DistributionView(child.q));
            if (retained_token == checked.proposal_token) {
                if (selected_child != no_parent) {
                    throw std::invalid_argument("candidate tree contains duplicate retained children");
                }
                selected_child = child_index;
            }
        }

        if (selected_child == no_parent) {
            result.stop_reason = StopReason::no_retained_child;
            return result;
        }
        if (capture_decisions) result.decisions.back().continued = true;
        context.push_back(checked.proposal_token);
        node_index = selected_child;
    }

    // max_depth is positive, so the loop always returns through a defined stop.
    throw std::logic_error("unreachable candidate-tree sampler state");
}

}  // namespace detail

// Build a breadth-bounded prefix tree. Every node stores its complete q row;
// at non-terminal positions it retains at most B positive-q children using
// deterministic probability-descending/token-id tie order. max_depth is the
// maximum number of verifier positions in one pulse (root included).
template <class DraftRowFn>
inline CandidateTree build_candidate_tree(const Context& root_context,
                                          std::size_t branch_factor,
                                          std::size_t max_depth,
                                          DraftRowFn&& draft_row) {
    if (max_depth == 0) {
        throw std::invalid_argument("candidate-tree max_depth must be positive");
    }

    CandidateTree tree;
    tree.root_context = root_context;
    tree.branch_factor = branch_factor;
    tree.max_depth = max_depth;

    Weights root_q = draft_row(root_context);
    (void)v1::detail::total_mass(v1::DistributionView(root_q));
    tree.vocabulary_size = root_q.size();
    tree.nodes.push_back(CandidateTreeNode{
        root_context, 0, no_parent, 0, std::move(root_q), {}});

    // Appending nodes breadth-first makes every child index greater than its
    // parent index and bounds generated rows by 1 + B + ... + B^(max_depth-1).
    for (std::size_t node_index = 0; node_index < tree.nodes.size(); ++node_index) {
        const std::size_t depth = tree.nodes[node_index].depth;
        if (depth + 1 >= max_depth) {
            continue;
        }
        const std::vector<Token> retained = detail::retained_tokens(
            v1::DistributionView(tree.nodes[node_index].q), branch_factor);
        for (Token token : retained) {
            Context child_context = tree.nodes[node_index].context;
            child_context.push_back(token);
            Weights child_q = draft_row(child_context);
            if (child_q.size() != tree.vocabulary_size) {
                throw std::invalid_argument("all candidate-tree q rows must share one vocabulary");
            }
            (void)v1::detail::total_mass(v1::DistributionView(child_q));
            if (tree.nodes.size() == tree.nodes.max_size()) {
                throw std::length_error("candidate tree exceeds container capacity");
            }
            const std::size_t child_index = tree.nodes.size();
            tree.nodes.emplace_back(CandidateTreeNode{
                std::move(child_context), depth + 1, node_index, token,
                std::move(child_q), {}});
            tree.nodes[node_index].children.push_back(child_index);
        }
    }
    return tree;
}

// Use an explicitly specified effective proposal pi. It may differ from q;
// the verifier both samples and evaluates this same pi law.
template <class TargetRowFn, class EffectivePiRowFn, class URBG>
inline CandidateTreeResult sample_candidate_tree(
    const CandidateTree& tree,
    const Context& initial_context,
    TargetRowFn&& target_row,
    EffectivePiRowFn&& effective_pi_row,
    URBG& rng) {
    auto accessor = [&effective_pi_row](const Context& context,
                                        const CandidateTreeNode&) {
        return effective_pi_row(context);
    };
    return detail::sample_candidate_tree_impl(
        tree, initial_context, std::forward<TargetRowFn>(target_row), accessor,
        rng, [](Token) { return false; });
}

template <class TargetRowFn, class EffectivePiRowFn, class URBG,
          class TerminalTokenFn>
inline CandidateTreeResult sample_candidate_tree_with_terminal(
    const CandidateTree& tree,
    const Context& initial_context,
    TargetRowFn&& target_row,
    EffectivePiRowFn&& effective_pi_row,
    URBG& rng,
    TerminalTokenFn&& is_terminal_token,
    bool capture_decisions = false) {
    auto accessor = [&effective_pi_row](const Context& context,
                                        const CandidateTreeNode&) {
        return effective_pi_row(context);
    };
    return detail::sample_candidate_tree_impl(
        tree, initial_context, std::forward<TargetRowFn>(target_row), accessor,
        rng, std::forward<TerminalTokenFn>(is_terminal_token), capture_decisions);
}

// Convenience path for the unprocessed-proposal case pi = the full stored q.
template <class TargetRowFn, class URBG>
inline CandidateTreeResult sample_candidate_tree(const CandidateTree& tree,
                                                  const Context& initial_context,
                                                  TargetRowFn&& target_row,
                                                  URBG& rng) {
    auto accessor = [](const Context&, const CandidateTreeNode& node) {
        return node.q;
    };
    return detail::sample_candidate_tree_impl(
        tree, initial_context, std::forward<TargetRowFn>(target_row), accessor,
        rng, [](Token) { return false; });
}

template <class TargetRowFn, class URBG, class TerminalTokenFn>
inline CandidateTreeResult sample_candidate_tree_with_terminal(
    const CandidateTree& tree,
    const Context& initial_context,
    TargetRowFn&& target_row,
    URBG& rng,
    TerminalTokenFn&& is_terminal_token,
    bool capture_decisions = false) {
    auto accessor = [](const Context&, const CandidateTreeNode& node) {
        return node.q;
    };
    return detail::sample_candidate_tree_impl(
        tree, initial_context, std::forward<TargetRowFn>(target_row), accessor,
        rng, std::forward<TerminalTokenFn>(is_terminal_token), capture_decisions);
}

}  // namespace pulsar::v3
