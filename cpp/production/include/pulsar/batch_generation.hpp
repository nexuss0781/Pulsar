#pragma once

#include "pulsar/sampling.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace pulsar::runtime {

struct SequenceInput {
    std::size_t sequence_id = 0;
    std::size_t position = 0;
    Token token = 0;
};

// Backend contract for bounded independent sequences. Implementors must return
// next-token logits in request/input order and isolate sequence KV by sequence_id.
// copy_sequence() replaces its destination with an exact source-prefix copy;
// rollback() removes a suffix from exactly the requested token.
class Model {
public:
    virtual ~Model() = default;
    virtual std::size_t vocabulary_size() const noexcept = 0;
    virtual std::size_t max_sequence_count() const noexcept = 0;
    virtual std::vector<Token> tokenize(const std::string& text) const = 0;
    virtual std::string token_to_piece(Token token) const = 0;
    virtual bool is_end_token(Token token) const noexcept = 0;
    virtual void reset() = 0;
    virtual void prefill(const std::vector<std::vector<Token>>& prompts,
                        std::vector<std::vector<float>>& next_logits) = 0;
    virtual void decode_batch(const std::vector<SequenceInput>& inputs,
                              std::vector<std::vector<float>>& next_logits) = 0;
    virtual void copy_sequence(std::size_t source_id,
                               std::size_t destination_id) = 0;
    virtual void rollback(std::size_t sequence_id, std::size_t new_length) = 0;
};

struct GenerationOptions {
    std::size_t max_new_tokens = 128;
    std::size_t max_context_tokens = 4096;
    std::size_t batch_size = 1;
    std::uint64_t seed = 1;
    SamplingConfig sampling{};
};

enum class FinishReason {
    end_of_generation,
    max_tokens,
    context_limit,
};

struct GenerationResult {
    std::size_t prompt_tokens = 0;
    std::size_t generated_tokens = 0;
    std::size_t draft_proposals = 0;
    std::size_t accepted_proposals = 0;
    std::size_t rejected_blocks = 0;
    std::string text;
    FinishReason finish_reason = FinishReason::max_tokens;
    double elapsed_ms = 0.0;
};

inline const char* to_string(FinishReason reason) noexcept {
    switch (reason) {
        case FinishReason::end_of_generation: return "end_of_generation";
        case FinishReason::max_tokens: return "max_tokens";
        case FinishReason::context_limit: return "context_limit";
    }
    return "unknown";
}

inline std::uint64_t splitmix64(std::uint64_t value) noexcept {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

inline std::vector<GenerationResult> generate_batch(
    Model& model,
    const std::vector<std::string>& prompts,
    const GenerationOptions& options,
    std::uint64_t first_request_id = 0) {
    if (options.batch_size == 0 || prompts.size() > options.batch_size) {
        throw std::invalid_argument("request count must be within the configured positive batch size");
    }
    if (options.max_context_tokens == 0) {
        throw std::invalid_argument("max_context_tokens must be positive");
    }
    validate_sampling_config(options.sampling);

    const auto started = std::chrono::steady_clock::now();
    std::vector<GenerationResult> results(prompts.size());
    std::vector<std::vector<Token>> histories;
    histories.reserve(prompts.size());
    std::vector<std::mt19937_64> random_engines;
    random_engines.reserve(prompts.size());

    for (std::size_t i = 0; i < prompts.size(); ++i) {
        histories.push_back(model.tokenize(prompts[i]));
        if (histories.back().empty()) {
            throw std::invalid_argument("prompt tokenized to an empty sequence");
        }
        if (histories.back().size() > options.max_context_tokens) {
            throw std::length_error("prompt exceeds max_context_tokens");
        }
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

    model.reset();
    std::vector<std::vector<float>> next_logits;
    model.prefill(histories, next_logits);
    if (next_logits.size() != prompts.size()) {
        throw std::runtime_error("backend returned an incorrect prefill row count");
    }
    for (const auto& row : next_logits) {
        if (row.size() != model.vocabulary_size()) {
            throw std::runtime_error("backend returned an incorrect prefill vocabulary size");
        }
    }

    std::vector<bool> finished(prompts.size(), false);
    std::size_t active_count = prompts.size();
    while (active_count != 0) {
        std::vector<SequenceInput> decode_inputs;
        std::vector<std::size_t> decode_sequences;
        decode_inputs.reserve(active_count);
        decode_sequences.reserve(active_count);

        for (std::size_t i = 0; i < prompts.size(); ++i) {
            if (finished[i]) {
                continue;
            }
            const Probabilities law = make_distribution(next_logits[i], histories[i],
                                                        options.sampling);
            const Token token = sample_categorical(law, random_engines[i]);
            GenerationResult& result = results[i];
            ++result.generated_tokens;

            if (model.is_end_token(token)) {
                finished[i] = true;
                --active_count;
                result.finish_reason = FinishReason::end_of_generation;
                result.elapsed_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - started).count();
                continue;
            }

            result.text += model.token_to_piece(token);
            if (result.generated_tokens >= options.max_new_tokens) {
                finished[i] = true;
                --active_count;
                result.finish_reason = FinishReason::max_tokens;
                result.elapsed_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - started).count();
                continue;
            }

            // The sampled token remains a valid completion at a full context;
            // only KV insertion is stopped to keep all positions in bounds.
            if (histories[i].size() >= options.max_context_tokens) {
                finished[i] = true;
                --active_count;
                result.finish_reason = FinishReason::context_limit;
                result.elapsed_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - started).count();
                continue;
            }

            const std::size_t position = histories[i].size();
            histories[i].push_back(token);
            decode_inputs.push_back(SequenceInput{i, position, token});
            decode_sequences.push_back(i);
        }

        if (decode_inputs.empty()) {
            continue;
        }
        std::vector<std::vector<float>> decoded_logits;
        model.decode_batch(decode_inputs, decoded_logits);
        if (decoded_logits.size() != decode_inputs.size()) {
            throw std::runtime_error("backend returned an incorrect decode row count");
        }
        for (std::size_t row = 0; row < decoded_logits.size(); ++row) {
            if (decoded_logits[row].size() != model.vocabulary_size()) {
                throw std::runtime_error("backend returned an incorrect decode vocabulary size");
            }
            next_logits[decode_sequences[row]] = std::move(decoded_logits[row]);
        }
    }

    return results;
}

}  // namespace pulsar::runtime
