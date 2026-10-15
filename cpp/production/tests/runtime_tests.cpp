#include "pulsar/batch_generation.hpp"
#include "pulsar/speculative_generation.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool near(double left, double right, double tolerance = 1e-12) {
    return std::abs(left - right) <= tolerance;
}

template <class Fn>
void require_throws(Fn&& fn, const std::string& message) {
    bool threw = false;
    try {
        fn();
    } catch (const std::exception&) {
        threw = true;
    }
    require(threw, message);
}

class FakeModel final : public pulsar::runtime::Model {
public:
    explicit FakeModel(int root_override = -1) : root_override_(root_override) {}

    std::size_t vocabulary_size() const noexcept override { return 4; }

    std::vector<pulsar::runtime::Token> tokenize(const std::string& text) const override {
        if (text == "A") return {0};
        if (text == "B") return {2};
        if (text == "AB") return {0, 2};
        return {0};
    }

    std::string token_to_piece(pulsar::runtime::Token token) const override {
        switch (token) {
            case 0: return "A";
            case 1: return "b";
            case 2: return "B";
            case 3: return "";
            default: throw std::out_of_range("bad fake token");
        }
    }

    bool is_end_token(pulsar::runtime::Token token) const noexcept override {
        return token == 3;
    }

    void reset() override {
        histories.clear();
        ++reset_count;
    }

    void prefill(const std::vector<std::vector<pulsar::runtime::Token>>& prompts,
                 std::vector<std::vector<float>>& next_logits) override {
        histories = prompts;
        next_logits.clear();
        for (const auto& history : histories) {
            next_logits.push_back(row_for(history));
        }
        ++prefill_count;
    }

    void decode_batch(const std::vector<pulsar::runtime::SequenceInput>& inputs,
                      std::vector<std::vector<float>>& next_logits) override {
        next_logits.clear();
        for (const auto& input : inputs) {
            require(input.sequence_id < histories.size(), "fake sequence ID out of range");
            require(input.position == histories[input.sequence_id].size(),
                    "fake decode position is stale");
            histories[input.sequence_id].push_back(input.token);
            next_logits.push_back(row_for(histories[input.sequence_id]));
        }
        ++decode_batch_count;
    }

    void rollback(std::size_t sequence_id, std::size_t new_length) override {
        require(sequence_id < histories.size(), "fake rollback sequence ID out of range");
        require(new_length <= histories[sequence_id].size(), "fake rollback length out of range");
        histories[sequence_id].resize(new_length);
        ++rollback_count;
    }

    std::size_t reset_count = 0;
    std::size_t prefill_count = 0;
    std::size_t decode_batch_count = 0;
    std::size_t rollback_count = 0;

private:
    std::vector<float> row_for(const std::vector<pulsar::runtime::Token>& history) const {
        if (history.size() == 1 && root_override_ >= 0) {
            std::vector<float> overridden(4, -9.0F);
            overridden[static_cast<std::size_t>(root_override_)] = 6.0F;
            return overridden;
        }
        const auto last = history.back();
        if (last == 0) return {0.0F, 4.0F, 1.0F, -9.0F};
        if (last == 1) return {-9.0F, -8.0F, -7.0F, 6.0F};
        if (last == 2) return {4.0F, 0.0F, 1.0F, -9.0F};
        return {0.0F, 0.0F, 0.0F, 5.0F};
    }

    int root_override_;
    std::vector<std::vector<pulsar::runtime::Token>> histories;
};

void test_softmax_and_top_k() {
    using namespace pulsar::runtime;
    SamplingConfig config;
    config.temperature = 1.0;
    config.top_k = 2;
    config.top_p = 1.0;
    const auto probabilities = make_distribution({1.0F, 2.0F, 3.0F, 4.0F}, {}, config);
    require(probabilities.size() == 4, "distribution size mismatch");
    require(near(probabilities[0], 0.0), "top-k retained too many tokens");
    require(near(probabilities[1], 0.0), "top-k retained too many tokens");
    require(probabilities[2] > 0.0 && probabilities[3] > probabilities[2],
            "top-k or softmax probabilities are incorrect");
    long double total = 0.0L;
    for (double probability : probabilities) total += probability;
    require(near(static_cast<double>(total), 1.0), "softmax law is not normalized");
}

void test_top_p_and_greedy_ties() {
    using namespace pulsar::runtime;
    SamplingConfig config;
    config.temperature = 1.0;
    config.top_k = 0;
    config.top_p = 0.70;
    const auto nucleus = make_distribution({4.0F, 3.0F, 2.0F, 1.0F}, {}, config);
    require(nucleus[0] > 0.0 && nucleus[1] > 0.0 && nucleus[2] == 0.0 && nucleus[3] == 0.0,
            "top-p nucleus retained the wrong support");
    require(near(nucleus[0] + nucleus[1], 1.0), "top-p nucleus was not renormalized");

    config.temperature = 0.0;
    config.top_p = 1.0;
    const auto greedy = make_distribution({3.0F, 3.0F, -1.0F}, {}, config);
    require(near(greedy[0], 1.0) && near(greedy[1], 0.0),
            "greedy tie break must select the lower token ID");
}

void test_repetition_penalty_and_validation() {
    using namespace pulsar::runtime;
    SamplingConfig config;
    config.temperature = 0.0;
    config.top_k = 0;
    config.top_p = 1.0;
    config.repetition_penalty = 2.0;
    config.repetition_window = 3;
    const auto penalized = make_distribution({2.0F, 1.5F, -2.0F}, {0}, config);
    require(near(penalized[1], 1.0), "repetition penalty did not alter the greedy choice");

    require_throws([&] { make_distribution({}, {}, config); }, "empty logits should fail");
    require_throws([&] { make_distribution({std::numeric_limits<float>::quiet_NaN()}, {}, config); },
                   "NaN logits should fail");
    require_throws([&] {
        SamplingConfig invalid = config;
        invalid.top_p = 0.0;
        make_distribution({0.0F}, {}, invalid);
    }, "invalid top-p should fail");
    require_throws([&] { make_distribution({1.0F}, {1}, config); },
                   "out-of-vocabulary repetition history should fail");
}

void test_categorical_draw_validation() {
    using namespace pulsar::runtime;
    std::mt19937_64 rng(7);
    for (int i = 0; i < 100; ++i) {
        require(sample_categorical({0.0, 1.0, 0.0}, rng) == 1,
                "categorical draw emitted zero-mass token");
    }
    require_throws([&] { sample_categorical({0.0, 0.0}, rng); },
                   "zero-mass categorical law should fail");
}

void test_batched_generation_and_cache_isolation() {
    using namespace pulsar::runtime;
    FakeModel model;
    GenerationOptions options;
    options.max_new_tokens = 8;
    options.max_context_tokens = 16;
    options.batch_size = 2;
    options.seed = 123;
    options.sampling.temperature = 0.0;
    options.sampling.top_k = 0;
    options.sampling.top_p = 1.0;

    const auto results = generate_batch(model, {"A", "B"}, options, 50);
    require(results.size() == 2, "batch result count mismatch");
    require(results[0].text == "b" && results[1].text == "Ab",
            "batched requests leaked or swapped sequence state");
    require(results[0].prompt_tokens == 1 && results[1].prompt_tokens == 1,
            "prompt token accounting mismatch");
    require(results[0].generated_tokens == 2 && results[1].generated_tokens == 3,
            "end-token accounting mismatch");
    require(results[0].finish_reason == FinishReason::end_of_generation &&
            results[1].finish_reason == FinishReason::end_of_generation,
            "end-of-generation stop reason mismatch");
    require(model.reset_count == 1 && model.prefill_count == 1,
            "one batch must reset and prefill exactly once");
    require(model.decode_batch_count == 2,
            "active sequences should be advanced together in two decode rounds");
}

void test_limits_seed_and_empty_batch() {
    using namespace pulsar::runtime;
    FakeModel model;
    GenerationOptions options;
    options.max_new_tokens = 1;
    options.max_context_tokens = 8;
    options.batch_size = 2;
    options.sampling.temperature = 0.0;
    options.sampling.top_k = 0;
    options.sampling.top_p = 1.0;
    const auto limited = generate_batch(model, {"A"}, options);
    require(limited[0].text == "b" && limited[0].generated_tokens == 1 &&
            limited[0].finish_reason == FinishReason::max_tokens,
            "max-token finish behavior mismatch");

    options.max_new_tokens = 8;
    options.max_context_tokens = 1;
    const auto context_limited = generate_batch(model, {"A"}, options);
    require(context_limited[0].text == "b" && context_limited[0].generated_tokens == 1 &&
            context_limited[0].finish_reason == FinishReason::context_limit,
            "context-limit finish behavior mismatch");

    options.max_new_tokens = 0;
    const std::size_t prefills_before = model.prefill_count;
    const auto no_generation = generate_batch(model, {"A"}, options);
    require(no_generation[0].generated_tokens == 0 && no_generation[0].text.empty() &&
            model.prefill_count == prefills_before,
            "zero-token generation should not run model inference");
    require(generate_batch(model, {}, options).empty(), "empty input batch should return empty results");

    options.max_new_tokens = 8;
    options.max_context_tokens = 8;
    options.sampling.temperature = 0.8;
    options.sampling.top_k = 0;
    options.sampling.top_p = 1.0;
    const auto run_one = [&] { return generate_batch(model, {"A", "B"}, options, 99); };
    const auto first = run_one();
    const auto second = run_one();
    require(first[0].text == second[0].text && first[1].text == second[1].text,
            "same seed and requests must reproduce outputs");

    require_throws([&] { generate_batch(model, {"A", "B", "A"}, options); },
                   "requests beyond batch capacity should fail");
    require_throws([&] {
        GenerationOptions too_small = options;
        too_small.max_context_tokens = 0;
        generate_batch(model, {"A"}, too_small);
    }, "zero context limit should fail");
}

void test_exact_speculative_chain_and_rollback() {
    using namespace pulsar::runtime;
    GenerationOptions options;
    options.max_new_tokens = 8;
    options.max_context_tokens = 16;
    options.batch_size = 2;
    options.seed = 314;
    options.sampling.temperature = 0.0;
    options.sampling.top_k = 0;
    options.sampling.top_p = 1.0;

    FakeModel target;
    FakeModel draft;
    const auto accepted = generate_speculative_batch(target, draft, {"A", "B"},
                                                      options, 2, 10);
    require(accepted.size() == 2 && accepted[0].text == "b" && accepted[1].text == "Ab",
            "exact speculative chain changed target output on identical models");
    require(accepted[0].finish_reason == FinishReason::end_of_generation &&
            accepted[1].finish_reason == FinishReason::end_of_generation,
            "speculative EOS handling mismatch");
    require(accepted[0].draft_proposals == accepted[0].accepted_proposals &&
            accepted[1].draft_proposals == accepted[1].accepted_proposals &&
            accepted[0].rejected_blocks == 0 && accepted[1].rejected_blocks == 0,
            "identical models should accept every draft proposal");

    FakeModel different_target;
    FakeModel different_draft(2);
    const auto rejected = generate_speculative_batch(different_target, different_draft,
                                                      {"A"}, options, 2);
    require(rejected[0].text == "b" && rejected[0].finish_reason == FinishReason::end_of_generation,
            "rejected draft proposal did not use the exact target residual");
    require(rejected[0].draft_proposals == 3 && rejected[0].accepted_proposals == 1 &&
            rejected[0].rejected_blocks == 1,
            "proposal diagnostics did not record the forced rejection and EOS proposal");
    require(different_target.rollback_count == 1 && different_draft.rollback_count == 1,
            "rejection must roll back provisional KV suffixes on both models");
}

}  // namespace

int main() {
    try {
        test_softmax_and_top_k();
        test_top_p_and_greedy_ties();
        test_repetition_penalty_and_validation();
        test_categorical_draw_validation();
        test_batched_generation_and_cache_isolation();
        test_limits_seed_and_empty_batch();
        test_exact_speculative_chain_and_rollback();
        std::cout << "Pulsar production runtime tests: PASS (7 groups)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Pulsar production runtime tests: FAIL: " << error.what() << '\n';
        return 1;
    }
}
