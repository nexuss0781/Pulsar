#include "llama_model.hpp"

#include <llama.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace pulsar::runtime {
namespace {

struct BatchGuard {
    explicit BatchGuard(std::int32_t capacity)
        : value(llama_batch_init(capacity, 0, 1)) {
        if (value.token == nullptr || value.pos == nullptr || value.n_seq_id == nullptr ||
            value.seq_id == nullptr || value.logits == nullptr) {
            llama_batch_free(value);
            throw std::runtime_error("llama.cpp could not allocate a token batch");
        }
    }
    ~BatchGuard() { llama_batch_free(value); }
    BatchGuard(const BatchGuard&) = delete;
    BatchGuard& operator=(const BatchGuard&) = delete;
    llama_batch value{};
};

void validate_decode_status(std::int32_t status, const char* operation) {
    if (status != 0) {
        throw std::runtime_error(std::string("llama.cpp ") + operation +
                                 " failed with status " + std::to_string(status));
    }
}

std::mutex backend_mutex;
std::size_t backend_users = 0;

void acquire_backend() {
    std::lock_guard<std::mutex> lock(backend_mutex);
    if (backend_users == 0) {
        llama_backend_init();
    }
    ++backend_users;
}

void release_backend() noexcept {
    std::lock_guard<std::mutex> lock(backend_mutex);
    if (backend_users != 0 && --backend_users == 0) {
        llama_backend_free();
    }
}

}  // namespace

struct LlamaModel::Impl {
    explicit Impl(LlamaModelOptions requested) : options(std::move(requested)) {}

    ~Impl() {
        if (context != nullptr) {
            llama_free(context);
        }
        if (model != nullptr) {
            llama_model_free(model);
        }
        if (backend_initialized) {
            release_backend();
        }
    }

    void open() {
        if (options.model_path.empty()) {
            throw std::invalid_argument("model_path must not be empty");
        }
        if (options.max_context_tokens == 0 || options.max_sequences == 0) {
            throw std::invalid_argument("context size and sequence count must be positive");
        }
        if (options.threads < 0) {
            throw std::invalid_argument("threads must be zero (automatic) or positive");
        }
        if (options.batch_tokens == 0) {
            throw std::invalid_argument("batch_tokens must be positive");
        }
        if (options.max_context_tokens > static_cast<std::size_t>(INT32_MAX) ||
            options.max_sequences > static_cast<std::size_t>(UINT32_MAX)) {
            throw std::length_error("requested context or sequence count exceeds llama.cpp limits");
        }
        if (options.max_sequences > llama_max_parallel_sequences()) {
            throw std::invalid_argument("requested batch size exceeds llama.cpp's parallel-sequence limit");
        }
        if (options.max_context_tokens >
            std::numeric_limits<std::uint32_t>::max() / options.max_sequences) {
            throw std::length_error("total KV context capacity exceeds llama.cpp limits");
        }

        const std::size_t total_context = options.max_context_tokens * options.max_sequences;
        if (total_context > static_cast<std::size_t>(UINT32_MAX)) {
            throw std::length_error("total KV context capacity exceeds llama.cpp limits");
        }
        const std::size_t requested_batch = std::max(options.batch_tokens,
                                                     options.max_sequences);
        const std::size_t bounded_batch = std::min(requested_batch, total_context);
        if (bounded_batch > static_cast<std::size_t>(INT32_MAX)) {
            throw std::length_error("decode batch exceeds llama.cpp limits");
        }

        acquire_backend();
        backend_initialized = true;

        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0;
        model_params.devices = nullptr;
        model = llama_model_load_from_file(options.model_path.c_str(), model_params);
        if (model == nullptr) {
            throw std::runtime_error("could not load GGUF model: " + options.model_path);
        }
        if (llama_model_has_encoder(model)) {
            throw std::invalid_argument(
                "encoder-decoder GGUF models are not supported by this text-generation runtime");
        }

        vocabulary = llama_model_get_vocab(model);
        if (vocabulary == nullptr) {
            throw std::runtime_error("loaded model has no tokenizer vocabulary");
        }
        vocab_size = llama_vocab_n_tokens(vocabulary);
        if (vocab_size <= 0) {
            throw std::runtime_error("loaded model has an invalid vocabulary size");
        }

        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = static_cast<std::uint32_t>(total_context);
        context_params.n_batch = static_cast<std::uint32_t>(bounded_batch);
        context_params.n_ubatch = static_cast<std::uint32_t>(
            std::min<std::size_t>(bounded_batch, 512));
        context_params.n_seq_max = static_cast<std::uint32_t>(options.max_sequences);
        context_params.n_outputs_max = context_params.n_batch;
        context_params.n_outputs_max_per_seq = context_params.n_outputs_max;
        context_params.embeddings = false;
        context_params.offload_kqv = false;
        context_params.op_offload = false;
        context_params.kv_unified = false;
        if (options.threads > 0) {
            context_params.n_threads = options.threads;
            context_params.n_threads_batch = options.threads;
        }

        context = llama_init_from_model(model, context_params);
        if (context == nullptr) {
            throw std::runtime_error("llama.cpp could not create an inference context");
        }
        actual_context_per_sequence = llama_n_ctx_seq(context);
        if (actual_context_per_sequence < options.max_context_tokens) {
            throw std::runtime_error(
                "llama.cpp allocated fewer context tokens per sequence than requested");
        }
        actual_batch_tokens = llama_n_batch(context);
        sequence_lengths.assign(options.max_sequences, 0);
    }

    void append_token(llama_batch& batch,
                      std::size_t sequence_id,
                      std::size_t position,
                      Token token,
                      bool output_logits) const {
        if (batch.n_tokens >= static_cast<std::int32_t>(actual_batch_tokens)) {
            throw std::length_error("internal llama.cpp batch capacity exceeded");
        }
        if (sequence_id >= options.max_sequences || position > static_cast<std::size_t>(INT32_MAX)) {
            throw std::out_of_range("sequence id or token position exceeds backend bounds");
        }
        const std::int32_t index = batch.n_tokens++;
        batch.token[index] = static_cast<llama_token>(token);
        batch.pos[index] = static_cast<llama_pos>(position);
        batch.n_seq_id[index] = 1;
        batch.seq_id[index][0] = static_cast<llama_seq_id>(sequence_id);
        batch.logits[index] = output_logits ? 1 : 0;
    }

    void flush_prefill_batch(llama_batch& batch,
                             const std::vector<std::size_t>& output_sequences,
                             const std::vector<std::size_t>& output_token_indices,
                             std::vector<std::vector<float>>& next_logits) {
        if (batch.n_tokens == 0) {
            return;
        }
        if (output_sequences.size() != output_token_indices.size()) {
            throw std::logic_error("prefill output sequence/index map is inconsistent");
        }
        validate_decode_status(llama_decode(context, batch), "prompt prefill");
        for (std::size_t output = 0; output < output_sequences.size(); ++output) {
            float* logits = llama_get_logits_ith(
                context, static_cast<std::int32_t>(output_token_indices[output]));
            if (logits == nullptr) {
                throw std::runtime_error("llama.cpp returned a null prefill logit row");
            }
            auto& row = next_logits.at(output_sequences[output]);
            row.assign(logits, logits + vocab_size);
        }
        batch.n_tokens = 0;
    }

    LlamaModelOptions options;
    llama_model* model = nullptr;
    llama_context* context = nullptr;
    const llama_vocab* vocabulary = nullptr;
    std::int32_t vocab_size = 0;
    std::uint32_t actual_context_per_sequence = 0;
    std::uint32_t actual_batch_tokens = 0;
    std::vector<std::size_t> sequence_lengths;
    bool backend_initialized = false;
};

LlamaModel::LlamaModel(const LlamaModelOptions& options)
    : impl_(std::make_unique<Impl>(options)) {
    impl_->open();
}

LlamaModel::~LlamaModel() = default;
LlamaModel::LlamaModel(LlamaModel&&) noexcept = default;
LlamaModel& LlamaModel::operator=(LlamaModel&&) noexcept = default;

std::size_t LlamaModel::vocabulary_size() const noexcept {
    return impl_ == nullptr ? 0 : static_cast<std::size_t>(impl_->vocab_size);
}

std::size_t LlamaModel::max_sequence_count() const noexcept {
    return impl_ == nullptr ? 0 : impl_->sequence_lengths.size();
}

std::size_t LlamaModel::kv_cache_bytes_per_token_estimate() const noexcept {
    if (impl_ == nullptr || impl_->model == nullptr) {
        return 0;
    }
    const std::int32_t layers = llama_model_n_layer(impl_->model);
    const std::int32_t embedding = llama_model_n_embd(impl_->model);
    const std::int32_t heads = llama_model_n_head(impl_->model);
    const std::int32_t kv_heads = llama_model_n_head_kv(impl_->model);
    if (layers <= 0 || embedding <= 0 || heads <= 0 || kv_heads <= 0 || embedding % heads != 0) {
        return 0;
    }
    std::size_t values_per_token = static_cast<std::size_t>(layers);
    const std::size_t factors[] = {
        static_cast<std::size_t>(kv_heads),
        static_cast<std::size_t>(embedding / heads),
    };
    for (const std::size_t factor : factors) {
        if (values_per_token > std::numeric_limits<std::size_t>::max() / factor) {
            return 0;
        }
        values_per_token *= factor;
    }
    if (values_per_token > std::numeric_limits<std::size_t>::max() / 4U) {
        return 0;
    }
    // The pinned runtime leaves type_k/type_v at llama.cpp's F16 defaults:
    // 2 bytes each for one key and one value element per cached position.
    return values_per_token * 4U;
}

std::vector<Token> LlamaModel::tokenize(const std::string& text) const {
    if (impl_ == nullptr || impl_->vocabulary == nullptr) {
        throw std::logic_error("model backend is not initialized");
    }
    if (text.size() > static_cast<std::size_t>(INT32_MAX)) {
        throw std::length_error("prompt exceeds llama.cpp tokenizer input limits");
    }

    std::int32_t capacity = static_cast<std::int32_t>(
        std::max<std::size_t>(8, std::min<std::size_t>(text.size() + 8, 256)));
    std::vector<llama_token> token_buffer(static_cast<std::size_t>(capacity));
    const char* input = text.empty() ? "" : text.data();
    std::int32_t count = llama_tokenize(impl_->vocabulary, input,
                                        static_cast<std::int32_t>(text.size()),
                                        token_buffer.data(), capacity, true, false);
    if (count < 0) {
        if (count == INT32_MIN) {
            throw std::length_error("tokenized prompt exceeds llama.cpp limits");
        }
        capacity = -count;
        token_buffer.resize(static_cast<std::size_t>(capacity));
        count = llama_tokenize(impl_->vocabulary, input,
                               static_cast<std::int32_t>(text.size()),
                               token_buffer.data(), capacity, true, false);
    }
    if (count < 0) {
        throw std::runtime_error("llama.cpp tokenization failed");
    }

    std::vector<Token> tokens;
    tokens.reserve(static_cast<std::size_t>(count));
    for (std::int32_t i = 0; i < count; ++i) {
        tokens.push_back(static_cast<Token>(token_buffer[static_cast<std::size_t>(i)]));
    }
    return tokens;
}

std::string LlamaModel::token_to_piece(Token token) const {
    if (impl_ == nullptr || impl_->vocabulary == nullptr) {
        throw std::logic_error("model backend is not initialized");
    }
    if (token < 0 || token >= impl_->vocab_size) {
        throw std::out_of_range("token id is outside the model vocabulary");
    }

    std::int32_t capacity = 64;
    std::string piece(static_cast<std::size_t>(capacity), '\0');
    std::int32_t count = llama_token_to_piece(impl_->vocabulary,
                                              static_cast<llama_token>(token),
                                              piece.data(), capacity, 0, false);
    if (count < 0) {
        if (count == INT32_MIN) {
            throw std::length_error("token piece exceeds llama.cpp limits");
        }
        capacity = -count;
        piece.resize(static_cast<std::size_t>(capacity));
        count = llama_token_to_piece(impl_->vocabulary,
                                     static_cast<llama_token>(token),
                                     piece.data(), capacity, 0, false);
    }
    if (count < 0) {
        throw std::runtime_error("llama.cpp token-to-piece conversion failed");
    }
    piece.resize(static_cast<std::size_t>(count));
    return piece;
}

bool LlamaModel::is_end_token(Token token) const noexcept {
    return impl_ != nullptr && impl_->vocabulary != nullptr && token >= 0 &&
           token < impl_->vocab_size &&
           llama_vocab_is_eog(impl_->vocabulary, static_cast<llama_token>(token));
}

void LlamaModel::reset() {
    if (impl_ == nullptr || impl_->context == nullptr) {
        throw std::logic_error("model backend is not initialized");
    }
    llama_memory_clear(llama_get_memory(impl_->context), true);
    std::fill(impl_->sequence_lengths.begin(), impl_->sequence_lengths.end(), 0);
}

void LlamaModel::prefill(const std::vector<std::vector<Token>>& prompts,
                         std::vector<std::vector<float>>& next_logits) {
    if (impl_ == nullptr || impl_->context == nullptr) {
        throw std::logic_error("model backend is not initialized");
    }
    if (prompts.empty() || prompts.size() > impl_->options.max_sequences) {
        throw std::invalid_argument("prefill request count is outside configured sequence capacity");
    }
    reset();
    next_logits.assign(prompts.size(), {});
    for (const auto& prompt : prompts) {
        if (prompt.empty()) {
            throw std::invalid_argument("each prefill prompt must contain at least one token");
        }
        if (prompt.size() > impl_->actual_context_per_sequence) {
            throw std::length_error("prefill prompt exceeds the allocated per-sequence context");
        }
        for (Token token : prompt) {
            if (token < 0 || token >= impl_->vocab_size) {
                throw std::invalid_argument("prefill contains a token outside the model vocabulary");
            }
        }
    }

    BatchGuard batch(static_cast<std::int32_t>(impl_->actual_batch_tokens));
    std::vector<std::size_t> output_sequences;
    std::vector<std::size_t> output_token_indices;
    output_sequences.reserve(prompts.size());
    output_token_indices.reserve(prompts.size());
    for (std::size_t sequence = 0; sequence < prompts.size(); ++sequence) {
        const auto& prompt = prompts[sequence];
        for (std::size_t position = 0; position < prompt.size(); ++position) {
            const bool output = position + 1 == prompt.size();
            const std::size_t batch_index = static_cast<std::size_t>(batch.value.n_tokens);
            impl_->append_token(batch.value, sequence, position, prompt[position], output);
            if (output) {
                output_sequences.push_back(sequence);
                output_token_indices.push_back(batch_index);
            }
            if (batch.value.n_tokens == static_cast<std::int32_t>(impl_->actual_batch_tokens)) {
                impl_->flush_prefill_batch(batch.value, output_sequences,
                                           output_token_indices, next_logits);
                output_sequences.clear();
                output_token_indices.clear();
            }
        }
    }
    impl_->flush_prefill_batch(batch.value, output_sequences,
                               output_token_indices, next_logits);

    for (std::size_t sequence = 0; sequence < prompts.size(); ++sequence) {
        if (next_logits[sequence].size() != static_cast<std::size_t>(impl_->vocab_size)) {
            throw std::runtime_error("llama.cpp did not return the final prompt logits for every sequence");
        }
        impl_->sequence_lengths[sequence] = prompts[sequence].size();
    }
}

void LlamaModel::decode_batch(const std::vector<SequenceInput>& inputs,
                              std::vector<std::vector<float>>& next_logits) {
    if (impl_ == nullptr || impl_->context == nullptr) {
        throw std::logic_error("model backend is not initialized");
    }
    next_logits.clear();
    next_logits.reserve(inputs.size());
    if (inputs.empty()) {
        throw std::invalid_argument("decode batch must not be empty");
    }

    const std::vector<std::size_t> initial_lengths = impl_->sequence_lengths;
    std::vector<std::size_t> consumed(impl_->options.max_sequences, 0);
    std::size_t begin = 0;
    while (begin < inputs.size()) {
        const std::size_t chunk_size = std::min<std::size_t>(
            impl_->actual_batch_tokens, inputs.size() - begin);
        BatchGuard batch(static_cast<std::int32_t>(impl_->actual_batch_tokens));
        std::vector<std::size_t> chunk_appended(impl_->options.max_sequences, 0);
        for (std::size_t offset = 0; offset < chunk_size; ++offset) {
            const SequenceInput& input = inputs[begin + offset];
            if (input.sequence_id >= impl_->options.max_sequences) {
                throw std::invalid_argument("decode sequence ID is out of range");
            }
            if (input.position != initial_lengths[input.sequence_id] + consumed[input.sequence_id]) {
                throw std::invalid_argument("decode positions must extend each sequence KV cache in order");
            }
            if (input.position >= impl_->actual_context_per_sequence) {
                throw std::length_error("decode would exceed the per-sequence context limit");
            }
            if (input.token < 0 || input.token >= impl_->vocab_size) {
                throw std::invalid_argument("decode token is outside the model vocabulary");
            }
            ++consumed[input.sequence_id];
            ++chunk_appended[input.sequence_id];
            impl_->append_token(batch.value, input.sequence_id, input.position,
                                input.token, true);
        }

        validate_decode_status(llama_decode(impl_->context, batch.value), "token decode");
        for (std::size_t output = 0; output < chunk_size; ++output) {
            float* logits = llama_get_logits_ith(impl_->context,
                                                 static_cast<std::int32_t>(output));
            if (logits == nullptr) {
                throw std::runtime_error("llama.cpp returned a null decode logit row");
            }
            next_logits.emplace_back(logits, logits + impl_->vocab_size);
        }
        for (std::size_t sequence = 0; sequence < chunk_appended.size(); ++sequence) {
            impl_->sequence_lengths[sequence] += chunk_appended[sequence];
        }
        begin += chunk_size;
    }
}

void LlamaModel::rollback(std::size_t sequence_id, std::size_t new_length) {
    if (impl_ == nullptr || impl_->context == nullptr) {
        throw std::logic_error("model backend is not initialized");
    }
    if (sequence_id >= impl_->options.max_sequences ||
        new_length > impl_->sequence_lengths[sequence_id] ||
        new_length > static_cast<std::size_t>(INT32_MAX)) {
        throw std::out_of_range("KV rollback length or sequence ID is invalid");
    }
    const bool removed = llama_memory_seq_rm(llama_get_memory(impl_->context),
                                             static_cast<llama_seq_id>(sequence_id),
                                             static_cast<llama_pos>(new_length), -1);
    if (!removed) {
        throw std::runtime_error(
            "llama.cpp backend cannot remove a partial sequence suffix; use autoregressive mode");
    }
    impl_->sequence_lengths[sequence_id] = new_length;
}

void LlamaModel::copy_sequence(std::size_t source_id,
                               std::size_t destination_id) {
    if (impl_ == nullptr || impl_->context == nullptr) {
        throw std::logic_error("model backend is not initialized");
    }
    if (source_id >= impl_->sequence_lengths.size() ||
        destination_id >= impl_->sequence_lengths.size()) {
        throw std::out_of_range("KV copy sequence ID is out of range");
    }
    if (source_id == destination_id) {
        return;
    }
    llama_memory_t memory = llama_get_memory(impl_->context);
    const bool removed = llama_memory_seq_rm(
        memory, static_cast<llama_seq_id>(destination_id), 0, -1);
    if (!removed) {
        throw std::runtime_error("llama.cpp could not clear the destination sequence before KV copy");
    }
    llama_memory_seq_cp(memory, static_cast<llama_seq_id>(source_id),
                        static_cast<llama_seq_id>(destination_id), 0, -1);
    impl_->sequence_lengths[destination_id] = impl_->sequence_lengths[source_id];
}

}  // namespace pulsar::runtime
