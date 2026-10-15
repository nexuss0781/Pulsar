#pragma once

#include "pulsar/batch_generation.hpp"
#include "pulsar_tree_v3.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pulsar::runtime {

inline std::size_t candidate_tree_node_bound(std::size_t branch_factor,
                                             std::size_t max_depth) {
    if (max_depth == 0) {
        throw std::invalid_argument("candidate-tree max_depth must be positive");
    }
    if (branch_factor == 0) {
        return 1;
    }
    if (branch_factor == 1) {
        return max_depth;
    }
    std::size_t total = 1;
    std::size_t level_nodes = 1;
    for (std::size_t depth = 1; depth < max_depth; ++depth) {
        if (branch_factor == 0) {
            break;
        }
        if (level_nodes > std::numeric_limits<std::size_t>::max() / branch_factor) {
            throw std::length_error("candidate-tree node bound overflows this platform");
        }
        level_nodes *= branch_factor;
        if (total > std::numeric_limits<std::size_t>::max() - level_nodes) {
            throw std::length_error("candidate-tree node bound overflows this platform");
        }
        total += level_nodes;
    }
    return total;
}

namespace detail {

using TreeContext = pulsar::v3::Context;
using TreeToken = pulsar::v3::Token;

inline TreeContext to_tree_context(const std::vector<Token>& context) {
    TreeContext converted;
    converted.reserve(context.size());
    for (Token token : context) {
        if (token < 0) {
            throw std::invalid_argument("runtime context contains a negative token ID");
        }
        converted.push_back(static_cast<TreeToken>(token));
    }
    return converted;
}

inline Token to_runtime_token(TreeToken token) {
    if (static_cast<std::uintmax_t>(token) >
        static_cast<std::uintmax_t>(std::numeric_limits<Token>::max())) {
        throw std::invalid_argument("tree verifier token exceeds the runtime token ID range");
    }
    return static_cast<Token>(token);
}

inline std::vector<Token> to_runtime_context(const TreeContext& context) {
    std::vector<Token> converted;
    converted.reserve(context.size());
    for (TreeToken token : context) {
        converted.push_back(to_runtime_token(token));
    }
    return converted;
}

inline std::size_t tree_sequence_id(std::size_t request_index,
                                    std::size_t node_index,
                                    std::size_t request_count,
                                    std::size_t node_bound) {
    if (node_index == 0) {
        return request_index;
    }
    return request_count + request_index * (node_bound - 1) + (node_index - 1);
}

inline std::vector<float> decode_branch_token(Model& model,
                                              std::size_t sequence_id,
                                              std::size_t position,
                                              Token token) {
    std::vector<std::vector<float>> rows;
    model.decode_batch({SequenceInput{sequence_id, position, token}}, rows);
    if (rows.size() != 1 || rows.front().size() != model.vocabulary_size()) {
        throw std::runtime_error("backend returned an invalid candidate-tree decode row");
    }
    return std::move(rows.front());
}

inline void validate_model_pair(Model& target, Model& draft) {
    if (target.vocabulary_size() == 0 ||
        target.vocabulary_size() != draft.vocabulary_size()) {
        throw std::invalid_argument("target and draft must have the same non-empty vocabulary size");
    }
    if (target.vocabulary_size() > static_cast<std::size_t>(std::numeric_limits<Token>::max())) {
        throw std::invalid_argument("vocabulary exceeds the runtime token ID range");
    }
    for (std::size_t index = 0; index < target.vocabulary_size(); ++index) {
        const Token token = static_cast<Token>(index);
        if (target.is_end_token(token) != draft.is_end_token(token)) {
            throw std::invalid_argument("target and draft end-token IDs are not aligned");
        }
    }
}

}  // namespace detail

// Exact fixed candidate-tree decoding. The retained tree is built from each
// node's complete processed draft row; it only controls whether an accepted
// proposal can continue within this pulse. The verifier always draws from the
// full stored q law, and rejection uses the exact p-q residual. Each tree node
// has its own sequence ID and cloned KV prefix; the verified final prefix is
// copied back to the request's root sequence before scratch branches are cleared.
inline std::vector<GenerationResult> generate_candidate_tree_batch(
    Model& target,
    Model& draft,
    const std::vector<std::string>& prompts,
    const GenerationOptions& options,
    std::size_t branch_factor,
    std::size_t max_depth,
    std::uint64_t first_request_id = 0) {
    if (options.batch_size == 0 || prompts.size() > options.batch_size) {
        throw std::invalid_argument("request count must be within the configured positive batch size");
    }
    if (options.max_context_tokens == 0) {
        throw std::invalid_argument("max_context_tokens must be positive");
    }
    validate_sampling_config(options.sampling);
    detail::validate_model_pair(target, draft);
    const std::size_t node_bound = candidate_tree_node_bound(branch_factor, max_depth);
    if (prompts.size() != 0 && node_bound > std::numeric_limits<std::size_t>::max() / prompts.size()) {
        throw std::length_error("candidate-tree batch sequence requirement overflows this platform");
    }
    const std::size_t required_sequences = prompts.size() * node_bound;
    if (required_sequences > target.max_sequence_count() ||
        required_sequences > draft.max_sequence_count()) {
        throw std::length_error("candidate-tree batch exceeds backend sequence capacity; reduce batch size, branching, or depth");
    }

    const auto started = std::chrono::steady_clock::now();
    std::vector<GenerationResult> results(prompts.size());
    std::vector<std::vector<Token>> histories;
    histories.reserve(prompts.size());
    std::vector<std::mt19937_64> random_engines;
    random_engines.reserve(prompts.size());
    for (std::size_t i = 0; i < prompts.size(); ++i) {
        std::vector<Token> target_tokens = target.tokenize(prompts[i]);
        std::vector<Token> draft_tokens = draft.tokenize(prompts[i]);
        if (target_tokens.empty() || draft_tokens.empty()) {
            throw std::invalid_argument("prompt tokenized to an empty sequence");
        }
        if (target_tokens != draft_tokens) {
            throw std::invalid_argument("target and draft tokenizers produced different IDs for a prompt");
        }
        if (target_tokens.size() > options.max_context_tokens) {
            throw std::length_error("prompt exceeds max_context_tokens");
        }
        for (Token token : target_tokens) {
            if (token < 0 || static_cast<std::size_t>(token) >= target.vocabulary_size()) {
                throw std::invalid_argument("prompt token is outside the model vocabulary");
            }
        }
        histories.push_back(std::move(target_tokens));
        results[i].prompt_tokens = histories.back().size();
        random_engines.emplace_back(splitmix64(options.seed + first_request_id + i));
    }

    if (prompts.empty() || options.max_new_tokens == 0) {
        const double elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        for (GenerationResult& result : results) {
            result.elapsed_ms = elapsed;
            result.finish_reason = FinishReason::max_tokens;
        }
        return results;
    }

    target.reset();
    draft.reset();
    std::vector<std::vector<float>> target_logits;
    std::vector<std::vector<float>> draft_logits;
    target.prefill(histories, target_logits);
    draft.prefill(histories, draft_logits);
    if (target_logits.size() != prompts.size() || draft_logits.size() != prompts.size()) {
        throw std::runtime_error("backend returned an incorrect candidate-tree prefill row count");
    }
    for (std::size_t i = 0; i < prompts.size(); ++i) {
        if (target_logits[i].size() != target.vocabulary_size() ||
            draft_logits[i].size() != draft.vocabulary_size()) {
            throw std::runtime_error("backend returned an incorrect candidate-tree vocabulary size");
        }
    }

    std::vector<bool> finished(prompts.size(), false);
    std::size_t active_count = prompts.size();
    while (active_count != 0) {
        for (std::size_t request = 0; request < prompts.size(); ++request) {
            if (finished[request]) {
                continue;
            }
            GenerationResult& result = results[request];
            const std::size_t remaining = options.max_new_tokens - result.generated_tokens;
            const std::size_t base_length = histories[request].size();
            std::size_t context_positions = 1;
            if (base_length < options.max_context_tokens) {
                const std::size_t slots = options.max_context_tokens - base_length;
                const std::size_t slots_plus_final =
                    slots == std::numeric_limits<std::size_t>::max() ? slots : slots + 1;
                context_positions = slots_plus_final;
            }
            const std::size_t pulse_depth = std::min({max_depth, remaining, context_positions});
            if (pulse_depth == 0) {
                throw std::logic_error("active candidate-tree request has no verifier positions");
            }

            const std::vector<Token>& root_runtime_context = histories[request];
            const detail::TreeContext root_tree_context =
                detail::to_tree_context(root_runtime_context);
            std::map<detail::TreeContext, std::size_t> node_indices;
            std::size_t next_node_index = 0;
            const auto request_sequence_id = [&](std::size_t node_index) {
                return detail::tree_sequence_id(request, node_index,
                                                prompts.size(), node_bound);
            };
            const auto draft_row = [&](const detail::TreeContext& tree_context) {
                const std::vector<Token> context = detail::to_runtime_context(tree_context);
                if (tree_context == root_tree_context) {
                    if (!node_indices.emplace(tree_context, 0).second || next_node_index != 0) {
                        throw std::logic_error("candidate-tree root context was registered more than once");
                    }
                    next_node_index = 1;
                    return make_distribution(draft_logits[request], context, options.sampling);
                }
                if (tree_context.empty()) {
                    throw std::logic_error("candidate-tree child context is empty");
                }
                detail::TreeContext parent_context = tree_context;
                const Token incoming_token = detail::to_runtime_token(parent_context.back());
                parent_context.pop_back();
                const auto parent = node_indices.find(parent_context);
                if (parent == node_indices.end()) {
                    throw std::logic_error("candidate-tree parent KV state is unavailable");
                }
                if (next_node_index >= node_bound) {
                    throw std::length_error("candidate tree exceeded its fixed node bound");
                }
                const std::size_t node_index = next_node_index++;
                const std::size_t parent_id = request_sequence_id(parent->second);
                const std::size_t node_id = request_sequence_id(node_index);
                draft.copy_sequence(parent_id, node_id);
                const std::vector<float> logits = detail::decode_branch_token(
                    draft, node_id, context.size() - 1, incoming_token);
                if (!node_indices.emplace(tree_context, node_index).second) {
                    throw std::logic_error("candidate tree contains a duplicate context path");
                }
                return make_distribution(logits, context, options.sampling);
            };

            const auto tree_build_started = std::chrono::steady_clock::now();
            const pulsar::v3::CandidateTree tree = pulsar::v3::build_candidate_tree(
                root_tree_context, branch_factor, pulse_depth, draft_row);
            result.candidate_tree_build_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - tree_build_started).count();
            ++result.candidate_tree_pulses;
            result.candidate_tree_nodes += tree.nodes.size();
            detail::TreeContext last_target_context;
            std::size_t last_target_node = 0;
            const auto target_row = [&](const detail::TreeContext& tree_context) {
                const auto current = node_indices.find(tree_context);
                if (current == node_indices.end()) {
                    throw std::invalid_argument("target verifier requested a context outside the candidate tree");
                }
                last_target_context = tree_context;
                last_target_node = current->second;
                const std::vector<Token> context = detail::to_runtime_context(tree_context);
                if (current->second == 0) {
                    return make_distribution(target_logits[request], context, options.sampling);
                }
                if (tree_context.empty()) {
                    throw std::invalid_argument("candidate-tree verifier context is empty");
                }
                detail::TreeContext parent_context = tree_context;
                const Token incoming_token = detail::to_runtime_token(parent_context.back());
                parent_context.pop_back();
                const auto parent = node_indices.find(parent_context);
                if (parent == node_indices.end()) {
                    throw std::invalid_argument("candidate-tree verifier parent context is stale");
                }
                const std::size_t parent_id = request_sequence_id(parent->second);
                const std::size_t node_id = request_sequence_id(current->second);
                target.copy_sequence(parent_id, node_id);
                const std::vector<float> logits = detail::decode_branch_token(
                    target, node_id, context.size() - 1, incoming_token);
                return make_distribution(logits, context, options.sampling);
            };

            const auto tree_verify_started = std::chrono::steady_clock::now();
            const pulsar::v3::CandidateTreeResult block =
                pulsar::v3::sample_candidate_tree_with_terminal(
                    tree, root_tree_context, target_row, random_engines[request],
                    [&target](detail::TreeToken token) {
                        return target.is_end_token(detail::to_runtime_token(token));
                    });
            result.candidate_tree_verification_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - tree_verify_started).count();
            if (block.emitted_tokens.empty()) {
                throw std::logic_error("candidate-tree verifier made no generation progress");
            }
            result.draft_proposals += block.proposed_tokens.size();
            result.accepted_proposals += block.accepted_count;
            if (block.rejected_position) {
                ++result.rejected_blocks;
            }

            bool request_finished = false;
            for (std::size_t offset = 0; offset < block.emitted_tokens.size(); ++offset) {
                const Token token = detail::to_runtime_token(block.emitted_tokens[offset]);
                const std::size_t token_context_length = base_length + offset;
                ++result.generated_tokens;
                if (target.is_end_token(token)) {
                    result.finish_reason = FinishReason::end_of_generation;
                    request_finished = true;
                } else {
                    result.text += target.token_to_piece(token);
                    if (result.generated_tokens >= options.max_new_tokens) {
                        result.finish_reason = FinishReason::max_tokens;
                        request_finished = true;
                    } else if (token_context_length >= options.max_context_tokens) {
                        result.finish_reason = FinishReason::context_limit;
                        request_finished = true;
                    } else {
                        if (histories[request].size() != token_context_length) {
                            throw std::logic_error("candidate-tree emitted context length is inconsistent");
                        }
                        histories[request].push_back(token);
                    }
                }
                if (request_finished) {
                    finished[request] = true;
                    --active_count;
                    break;
                }
            }

            if (!request_finished) {
                const std::vector<Token> last_runtime_context =
                    detail::to_runtime_context(last_target_context);
                const Token final_token = detail::to_runtime_token(block.emitted_tokens.back());
                if (last_runtime_context.size() + 1 != histories[request].size() ||
                    last_target_node >= tree.nodes.size() ||
                    tree.nodes[last_target_node].context != last_target_context ||
                    !std::equal(last_runtime_context.begin(), last_runtime_context.end(),
                                histories[request].begin()) ||
                    histories[request].back() != final_token) {
                    throw std::logic_error("candidate-tree final KV context does not match the emitted prefix");
                }
                const std::size_t source_id = request_sequence_id(last_target_node);
                target.copy_sequence(source_id, request);
                draft.copy_sequence(source_id, request);
                target_logits[request] = detail::decode_branch_token(
                    target, request, last_runtime_context.size(), final_token);
                draft_logits[request] = detail::decode_branch_token(
                    draft, request, last_runtime_context.size(), final_token);
            }

            // Every non-root node slot is scratch: commit the selected prefix
            // first, then remove all branch KV state before the next pulse.
            for (std::size_t node_index = 1; node_index < node_bound; ++node_index) {
                const std::size_t sequence_id = request_sequence_id(node_index);
                target.rollback(sequence_id, 0);
                draft.rollback(sequence_id, 0);
            }
            if (request_finished) {
                result.elapsed_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - started).count();
            }
        }
    }

    const double elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    for (GenerationResult& result : results) {
        if (result.elapsed_ms == 0.0) {
            result.elapsed_ms = elapsed;
        }
    }
    return results;
}

}  // namespace pulsar::runtime
