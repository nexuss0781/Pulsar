#include "pulsar/batch_generation.hpp"
#include "pulsar/speculative_generation.hpp"
#include "pulsar/tree_generation.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
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
    std::size_t max_sequence_count() const noexcept override { return 256; }

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
            require(input.sequence_id < max_sequence_count(), "fake sequence ID out of range");
            require(input.sequence_id < histories.size(), "fake sequence has no copied or prefetched KV state");
            require(input.position == histories[input.sequence_id].size(),
                    "fake decode position is stale");
            histories[input.sequence_id].push_back(input.token);
            next_logits.push_back(row_for(histories[input.sequence_id]));
        }
        ++decode_batch_count;
    }

    void copy_sequence(std::size_t source_id, std::size_t destination_id) override {
        require(source_id < histories.size(), "fake KV copy source has no state");
        require(destination_id < max_sequence_count(), "fake KV copy destination is out of range");
        if (source_id == destination_id) return;
        if (histories.size() <= destination_id) histories.resize(destination_id + 1);
        histories[destination_id] = histories[source_id];
        ++copy_count;
    }

    void rollback(std::size_t sequence_id, std::size_t new_length) override {
        require(sequence_id < max_sequence_count(), "fake rollback sequence ID out of range");
        if (histories.size() <= sequence_id) histories.resize(sequence_id + 1);
        require(new_length <= histories[sequence_id].size(), "fake rollback length out of range");
        histories[sequence_id].resize(new_length);
        ++rollback_count;
    }

    const std::vector<pulsar::runtime::Token>& sequence_history(std::size_t id) const {
        return histories.at(id);
    }

    std::size_t reset_count = 0;
    std::size_t prefill_count = 0;
    std::size_t decode_batch_count = 0;
    std::size_t copy_count = 0;
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

void test_candidate_tree_bound_and_pointwise_verifier_identity() {
    using pulsar::v1::Weights;
    require(pulsar::runtime::candidate_tree_node_bound(0, 8) == 1,
            "zero-branch tree should contain only its root");
    require(pulsar::runtime::candidate_tree_node_bound(2, 4) == 15,
            "candidate-tree node bound is incorrect");

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
        const long double residual_mass = pulsar::v1::residual_mass(p_view, pi_view);
        Weights residual(p_weights.size(), 0.0);
        if (residual_mass > 0.0L) {
            residual = pulsar::v1::residual_distribution(p_view, pi_view);
        }
        for (std::size_t token = 0; token < p_weights.size(); ++token) {
            const long double p = static_cast<long double>(p_weights[token]) / p_sum;
            const long double pi = static_cast<long double>(pi_weights[token]) / pi_sum;
            const long double acceptance = pi > 0.0L ? std::min(1.0L, p / pi) : 0.0L;
            const long double reconstructed = pi * acceptance + residual_mass * residual[token];
            require(std::abs(reconstructed - p) <= tolerance,
                    "pi*a + Z*r = p failed for a production-tree effective proposal row");
        }
    }
}

void test_candidate_tree_rejects_at_every_depth() {
    using namespace pulsar::v3;
    const Context root{0};
    for (std::size_t reject_depth = 0; reject_depth < 4; ++reject_depth) {
        const CandidateTree tree = build_candidate_tree(
            root, 1, 4, [](const Context&) { return Weights{1.0, 0.0}; });
        std::size_t target_calls = 0;
        const auto target_row = [&](const Context& context) {
            ++target_calls;
            return context.size() == root.size() + reject_depth
                ? Weights{0.0, 1.0}
                : Weights{1.0, 0.0};
        };
        std::mt19937_64 rng(0xD3E7U + reject_depth);
        const CandidateTreeResult block = sample_candidate_tree(
            tree, root, target_row, rng);
        require(block.rejected_position == reject_depth &&
                    block.accepted_count == reject_depth &&
                    block.emitted_tokens.size() == reject_depth + 1,
                "tree rejection did not occur at the requested verifier depth");
        for (std::size_t i = 0; i < reject_depth; ++i) {
            require(block.emitted_tokens[i] == 0,
                    "accepted tree prefix changed before rejection");
        }
        require(block.emitted_tokens.back() == 1 && target_calls == reject_depth + 1,
                "tree rejection did not use the exact residual and stop the pulse");
    }
}

void test_candidate_tree_terminal_token_stops() {
    using namespace pulsar::v3;
    const Context root{0};
    const CandidateTree tree = build_candidate_tree(
        root, 1, 4, [](const Context&) { return Weights{0.0, 0.0, 0.0, 1.0}; });
    std::size_t target_calls = 0;
    std::mt19937_64 rng(17);
    const CandidateTreeResult block = sample_candidate_tree_with_terminal(
        tree, root,
        [&target_calls](const Context&) {
            ++target_calls;
            return Weights{0.0, 0.0, 0.0, 1.0};
        }, rng, [](Token token) { return token == 3; });
    require(block.emitted_tokens == std::vector<Token>{3} &&
                block.stop_reason == StopReason::terminal_token && target_calls == 1,
            "accepted end token did not stop candidate-tree continuation");
}

void test_candidate_tree_runtime_commit_and_rollback() {
    using namespace pulsar::runtime;
    GenerationOptions options;
    options.max_new_tokens = 3;
    options.max_context_tokens = 16;
    options.batch_size = 1;
    options.seed = 91;
    options.sampling.temperature = 0.0;
    options.sampling.top_k = 0;
    options.sampling.top_p = 1.0;

    FakeModel target(2);
    FakeModel draft(1);
    const auto result = generate_candidate_tree_batch(target, draft, {"A"}, options,
                                                       1, 2);
    require(result.size() == 1 && result[0].text == "BAb" &&
                result[0].generated_tokens == 3 &&
                result[0].finish_reason == FinishReason::max_tokens,
            "candidate-tree output or stopping behavior is incorrect");
    require(result[0].draft_proposals == 3 && result[0].accepted_proposals == 2 &&
                result[0].rejected_blocks == 1,
            "candidate-tree proposal diagnostics do not match rejection/acceptance");
    require(target.copy_count > 0 && draft.copy_count > 0 &&
                target.rollback_count >= 2 && draft.rollback_count >= 2,
            "tree branches were not cloned, committed, and cleared");
    require(target.sequence_history(0) == std::vector<Token>({0, 2}) &&
                draft.sequence_history(0) == std::vector<Token>({0, 2}),
            "the residual replacement was not committed into both root KV states");
    require(target.sequence_history(1).empty() && draft.sequence_history(1).empty(),
            "candidate branch KV was not rolled back after the pulse");
}

void test_candidate_tree_batch_isolation_and_capacity() {
    using namespace pulsar::runtime;
    GenerationOptions options;
    options.max_new_tokens = 8;
    options.max_context_tokens = 16;
    options.batch_size = 2;
    options.seed = 1234;
    options.sampling.temperature = 0.0;
    options.sampling.top_k = 0;
    options.sampling.top_p = 1.0;

    FakeModel autoregressive;
    const auto reference = generate_batch(autoregressive, {"A", "B"}, options, 40);
    FakeModel target;
    FakeModel draft;
    const auto tree = generate_candidate_tree_batch(target, draft, {"A", "B"},
                                                     options, 2, 3, 40);
    require(tree.size() == reference.size(), "tree batch result count mismatch");
    for (std::size_t i = 0; i < tree.size(); ++i) {
        require(tree[i].text == reference[i].text &&
                    tree[i].generated_tokens == reference[i].generated_tokens &&
                    tree[i].finish_reason == reference[i].finish_reason,
                "candidate tree leaked or swapped request state");
    }

    require_throws([&] {
        (void)generate_candidate_tree_batch(target, draft, {"A"}, options, 4, 5);
    }, "candidate tree exceeded backend sequence capacity without failing");
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
        test_candidate_tree_bound_and_pointwise_verifier_identity();
        test_candidate_tree_rejects_at_every_depth();
        test_candidate_tree_terminal_token_stops();
        test_candidate_tree_runtime_commit_and_rollback();
        test_candidate_tree_batch_isolation_and_capacity();
        std::cout << "Pulsar production runtime tests: PASS (12 groups)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Pulsar production runtime tests: FAIL: " << error.what() << '\n';
        return 1;
    }
}
