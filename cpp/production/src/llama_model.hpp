#pragma once

#include "pulsar/batch_generation.hpp"

#include <cstddef>
#include <memory>
#include <string>

namespace pulsar::runtime {

struct LlamaModelOptions {
    std::string model_path;
    std::size_t max_context_tokens = 4096;
    std::size_t max_sequences = 1;
    std::size_t batch_tokens = 512;
    int threads = 0;  // 0 selects the llama.cpp default.
};

// Real GGUF-backed CPU inference through the llama.cpp C API. One model and one
// context are reused across bounded request microbatches; sequence IDs isolate
// each request's KV state during batched prefill and token decoding.
class LlamaModel final : public Model {
public:
    explicit LlamaModel(const LlamaModelOptions& options);
    ~LlamaModel() override;

    LlamaModel(const LlamaModel&) = delete;
    LlamaModel& operator=(const LlamaModel&) = delete;
    LlamaModel(LlamaModel&&) noexcept;
    LlamaModel& operator=(LlamaModel&&) noexcept;

    std::size_t vocabulary_size() const noexcept override;
    std::size_t max_sequence_count() const noexcept override;
    std::vector<Token> tokenize(const std::string& text) const override;
    std::string token_to_piece(Token token) const override;
    bool is_end_token(Token token) const noexcept override;
    void reset() override;
    void prefill(const std::vector<std::vector<Token>>& prompts,
                 std::vector<std::vector<float>>& next_logits) override;
    void decode_batch(const std::vector<SequenceInput>& inputs,
                      std::vector<std::vector<float>>& next_logits) override;
    void copy_sequence(std::size_t source_id,
                       std::size_t destination_id) override;
    void rollback(std::size_t sequence_id, std::size_t new_length) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace pulsar::runtime
