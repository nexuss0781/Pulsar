#pragma once

#include "pulsar/batch_generation.hpp"
#include "pulsar_speculative_v2.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pulsar::runtime {

inline std::vector<GenerationResult> generate_speculative_batch(
    Model& target,
    Model& draft,
    const std::vector<std::string>& prompts,
    const GenerationOptions& options,
    std::size_t speculative_tokens,
    std::uint64_t first_request_id = 0) {
    if (options.batch_size == 0 || prompts.size() > options.batch_size) {
        throw std::invalid_argument("request count must be within the configured positive batch size");
    }
    if (options.max_context_tokens == 0) {
        throw std::invalid_argument("max_context_tokens must be positive");
    }
    if (speculative_tokens == 0) {
        throw std::invalid_argument("speculative_tokens must be positive");
    }
    if (target.vocabulary_size() == 0 || target.vocabulary_size() != draft.vocabulary_size()) {
        throw std::invalid_argument("target and draft must have the same non-empty vocabulary size");
    }
    const std::size_t vocabulary_size = target.vocabulary_size();
    if (vocabulary_size > static_cast<std::size_t>(std::numeric_limits<Token>::max())) {
        throw std::invalid_argument("vocabulary exceeds the runtime token ID range");
    }
    for (std::size_t index = 0; index < vocabulary_size; ++index) {
        const Token token = static_cast<Token>(index);
        if (target.is_end_token(token) != draft.is_end_token(token)) {
            throw std::invalid_argument("target and draft end-token IDs are not aligned");
        }
    }
    validate_sampling_config(options.sampling);

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
            throw std::invalid_argument(
                "target and draft tokenizers produced different IDs for a prompt");
        }
        if (target_tokens.size() > options.max_context_tokens) {
            throw std::length_error("prompt exceeds max_context_tokens");
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
        throw std::runtime_error("backend returned an incorrect speculative prefill row count");
    }
    for (std::size_t i = 0; i < prompts.size(); ++i) {
        if (target_logits[i].size() != target.vocabulary_size() ||
            draft_logits[i].size() != draft.vocabulary_size()) {
            throw std::runtime_error("backend returned an incorrect speculative vocabulary size");
        }
    }

    std::vector<bool> finished(prompts.size(), false);
    std::size_t active_count = prompts.size();
    while (active_count != 0) {
        if (is_cancellation_requested(options)) {
            const double elapsed = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count();
            mark_unfinished_cancelled(results, finished, active_count, elapsed);
            break;
        }
        const std::size_t count = prompts.size();
        std::vector<std::size_t> base_lengths(count, 0);
        std::vector<std::size_t> planned(count, 0);
        std::vector<std::vector<Token>> proposals(count);
        std::vector<std::vector<Probabilities>> draft_rows(count);
        std::vector<bool> draft_ended(count, false);

        // A full context still permits one final token sampled from its cached
        // next-token row. It cannot be fed back into a KV cache at that limit.
        for (std::size_t i = 0; i < count; ++i) {
            if (finished[i]) continue;
            GenerationResult& result = results[i];
            const std::size_t remaining = options.max_new_tokens - result.generated_tokens;
            if (remaining == 0) {
                finished[i] = true;
                --active_count;
                result.finish_reason = FinishReason::max_tokens;
                result.elapsed_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - started).count();
                continue;
            }
            base_lengths[i] = histories[i].size();
            if (base_lengths[i] >= options.max_context_tokens) {
                const Probabilities law = make_distribution(target_logits[i], histories[i],
                                                            options.sampling);
                const Token token = sample_categorical(law, random_engines[i]);
                ++result.generated_tokens;
                if (target.is_end_token(token)) {
                    result.finish_reason = FinishReason::end_of_generation;
                } else {
                    result.text += target.token_to_piece(token);
                    result.finish_reason = result.generated_tokens >= options.max_new_tokens
                        ? FinishReason::max_tokens
                        : FinishReason::context_limit;
                }
                finished[i] = true;
                --active_count;
                result.elapsed_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - started).count();
                continue;
            }
            const std::size_t context_slots = options.max_context_tokens - base_lengths[i];
            planned[i] = std::min({speculative_tokens, remaining, context_slots});
            if (planned[i] == 0) {
                throw std::logic_error("speculative planner produced an empty active block");
            }
            proposals[i].reserve(planned[i]);
            draft_rows[i].reserve(planned[i]);
        }
        if (active_count == 0) break;

        // Draft all proposal paths first. Each depth is batched across requests;
        // every proposal law is retained in full for the exact q-based verifier.
        for (std::size_t depth = 0; depth < speculative_tokens; ++depth) {
            std::vector<SequenceInput> draft_inputs;
            std::vector<std::size_t> input_sequences;
            for (std::size_t i = 0; i < count; ++i) {
                if (finished[i] || draft_ended[i] || proposals[i].size() >= planned[i]) {
                    continue;
                }
                std::vector<Token> proposal_context = histories[i];
                proposal_context.insert(proposal_context.end(), proposals[i].begin(),
                                        proposals[i].end());
                Probabilities q = make_distribution(draft_logits[i], proposal_context,
                                                    options.sampling);
                const Token proposal = sample_categorical(q, random_engines[i]);
                if (target.is_end_token(proposal) != draft.is_end_token(proposal)) {
                    throw std::invalid_argument(
                        "target and draft end-token IDs are not aligned");
                }
                draft_rows[i].push_back(std::move(q));
                proposals[i].push_back(proposal);
                if (draft.is_end_token(proposal)) {
                    draft_ended[i] = true;
                    continue;
                }
                const std::size_t position = base_lengths[i] + proposals[i].size() - 1;
                draft_inputs.push_back(SequenceInput{i, position, proposal});
                input_sequences.push_back(i);
            }
            if (!draft_inputs.empty()) {
                std::vector<std::vector<float>> decoded;
                draft.decode_batch(draft_inputs, decoded);
                if (decoded.size() != draft_inputs.size()) {
                    throw std::runtime_error("draft backend returned an incorrect decode row count");
                }
                for (std::size_t row = 0; row < decoded.size(); ++row) {
                    if (decoded[row].size() != draft.vocabulary_size()) {
                        throw std::runtime_error("draft backend returned an incorrect decode vocabulary size");
                    }
                    draft_logits[input_sequences[row]] = std::move(decoded[row]);
                }
            }
        }

        // Target consumes every proposal in one logical batch. Its row before
        // proposal j is the prefill row for j=0, otherwise the output after j-1.
        std::vector<SequenceInput> target_inputs;
        std::vector<std::pair<std::size_t, std::size_t>> target_output_map;
        for (std::size_t i = 0; i < count; ++i) {
            if (finished[i]) continue;
            for (std::size_t j = 0; j < proposals[i].size(); ++j) {
                target_inputs.push_back(SequenceInput{
                    i, base_lengths[i] + j, proposals[i][j]});
                target_output_map.emplace_back(i, j);
            }
        }
        if (!target_inputs.empty()) {
            std::vector<std::vector<float>> decoded;
            target.decode_batch(target_inputs, decoded);
            if (decoded.size() != target_inputs.size()) {
                throw std::runtime_error("target backend returned an incorrect verify row count");
            }
            std::vector<std::vector<std::vector<float>>> rows_after(count);
            for (std::size_t i = 0; i < count; ++i) {
                rows_after[i].resize(proposals[i].size());
            }
            for (std::size_t row = 0; row < decoded.size(); ++row) {
                if (decoded[row].size() != target.vocabulary_size()) {
                    throw std::runtime_error("target backend returned an incorrect verify vocabulary size");
                }
                const auto [sequence, offset] = target_output_map[row];
                rows_after[sequence][offset] = std::move(decoded[row]);
            }

            std::vector<SequenceInput> repair_target_inputs;
            std::vector<std::size_t> repair_sequences;
            std::vector<std::size_t> repair_positions;
            std::vector<Token> repair_tokens;
            for (std::size_t i = 0; i < count; ++i) {
                if (finished[i] || proposals[i].empty()) continue;
                std::vector<Probabilities> target_rows;
                target_rows.reserve(proposals[i].size());
                std::vector<Token> proposal_context = histories[i];
                for (std::size_t j = 0; j < proposals[i].size(); ++j) {
                    const std::vector<float>& raw = j == 0 ? target_logits[i]
                                                           : rows_after[i][j - 1];
                    target_rows.push_back(make_distribution(raw, proposal_context,
                                                            options.sampling));
                    proposal_context.push_back(proposals[i][j]);
                }

                std::optional<Probabilities> bonus_row;
                const std::size_t remaining = options.max_new_tokens -
                                              results[i].generated_tokens;
                if (!draft_ended[i] && remaining > proposals[i].size()) {
                    bonus_row = make_distribution(rows_after[i].back(), proposal_context,
                                                  options.sampling);
                }

                std::vector<pulsar::v2::Token> verifier_proposals;
                verifier_proposals.reserve(proposals[i].size());
                for (Token token : proposals[i]) {
                    verifier_proposals.push_back(static_cast<pulsar::v2::Token>(token));
                }
                const std::optional<std::vector<double>> optional_bonus = bonus_row;
                results[i].draft_proposals += proposals[i].size();
                const auto verified = pulsar::v2::verify_speculative_block(
                    verifier_proposals, draft_rows[i], target_rows, optional_bonus,
                    random_engines[i]);
                const bool rejected = verified.rejected_position.has_value();
                const std::size_t accepted = verified.accepted_count;
                results[i].accepted_proposals += accepted;
                if (rejected) {
                    ++results[i].rejected_blocks;
                }
                const Token residual = rejected
                    ? static_cast<Token>(verified.emitted_tokens.back())
                    : Token{-1};

                bool emitted_end_token = false;
                bool bonus_was_emitted = false;
                for (std::size_t j = 0; j < verified.emitted_tokens.size(); ++j) {
                    const Token token = static_cast<Token>(verified.emitted_tokens[j]);
                    const bool is_bonus = !rejected && verified.bonus_emitted &&
                                          j + 1 == verified.emitted_tokens.size();
                    GenerationResult& result = results[i];
                    ++result.generated_tokens;
                    if (target.is_end_token(token)) {
                        result.finish_reason = FinishReason::end_of_generation;
                        emitted_end_token = true;
                        finished[i] = true;
                        --active_count;
                        result.elapsed_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - started).count();
                        break;
                    }
                    result.text += target.token_to_piece(token);
                    histories[i].push_back(token);
                    if (is_bonus) {
                        bonus_was_emitted = true;
                    }
                    if (result.generated_tokens >= options.max_new_tokens) {
                        result.finish_reason = FinishReason::max_tokens;
                        finished[i] = true;
                        --active_count;
                        result.elapsed_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - started).count();
                        break;
                    }
                    if (is_bonus && histories[i].size() >= options.max_context_tokens) {
                        result.finish_reason = FinishReason::context_limit;
                        finished[i] = true;
                        --active_count;
                        result.elapsed_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - started).count();
                        break;
                    }
                }

                if (rejected) {
                    const std::size_t rollback_length = base_lengths[i] + accepted;
                    target.rollback(i, rollback_length);
                    draft.rollback(i, rollback_length);
                    if (!finished[i]) {
                        if (target.is_end_token(residual)) {
                            throw std::logic_error("end token should have terminated a rejected pulse");
                        }
                        // The residual replacement is committed after the accepted
                        // prefix and must be consumed by both models for next logits.
                        const std::size_t position = rollback_length;
                        repair_target_inputs.push_back(SequenceInput{i, position, residual});
                        repair_sequences.push_back(i);
                        repair_positions.push_back(position);
                        repair_tokens.push_back(residual);
                    }
                } else if (!finished[i]) {
                    if (verified.bonus_emitted) {
                        if (!bonus_was_emitted) {
                            throw std::logic_error("verified bonus was not present in emitted tokens");
                        }
                        const Token bonus_token = static_cast<Token>(verified.emitted_tokens.back());
                        const std::size_t position = base_lengths[i] + proposals[i].size();
                        repair_target_inputs.push_back(SequenceInput{i, position, bonus_token});
                        repair_sequences.push_back(i);
                        repair_positions.push_back(position);
                        repair_tokens.push_back(bonus_token);
                    } else {
                        target_logits[i] = std::move(rows_after[i].back());
                        // draft_logits already holds the row after its last ordinary proposal.
                    }
                }

                (void)emitted_end_token;
            }

            if (!repair_target_inputs.empty()) {
                // Rejected residuals and accepted bonus tokens form the new
                // committed suffix. Feed the identical tokens to both backends.
                std::vector<std::vector<float>> target_repair_logits;
                std::vector<std::vector<float>> draft_repair_logits;
                target.decode_batch(repair_target_inputs, target_repair_logits);
                draft.decode_batch(repair_target_inputs, draft_repair_logits);
                if (target_repair_logits.size() != repair_target_inputs.size() ||
                    draft_repair_logits.size() != repair_target_inputs.size()) {
                    throw std::runtime_error("cache-repair decode returned an incorrect row count");
                }
                for (std::size_t row = 0; row < repair_target_inputs.size(); ++row) {
                    const std::size_t sequence = repair_sequences[row];
                    if (target_repair_logits[row].size() != target.vocabulary_size() ||
                        draft_repair_logits[row].size() != draft.vocabulary_size()) {
                        throw std::runtime_error("cache-repair decode returned an incorrect vocabulary size");
                    }
                    target_logits[sequence] = std::move(target_repair_logits[row]);
                    draft_logits[sequence] = std::move(draft_repair_logits[row]);
                    (void)repair_positions[row];
                    (void)repair_tokens[row];
                }
            }
        }
    }

    return results;
}

}  // namespace pulsar::runtime
