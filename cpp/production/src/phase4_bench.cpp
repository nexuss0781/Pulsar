#include "llama_model.hpp"
#include "pulsar/speculative_generation.hpp"
#include "pulsar/tree_generation.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <utility>
#include <vector>

namespace {

using pulsar::runtime::GenerationOptions;
using pulsar::runtime::GenerationResult;
using pulsar::runtime::LlamaModel;
using pulsar::runtime::LlamaModelOptions;
using pulsar::runtime::Model;
using pulsar::runtime::SequenceInput;
using pulsar::runtime::Token;

struct Prompt {
    std::size_t numeric_id = 0;
    std::string id;
    std::string text;
};

struct Options {
    std::string mode;
    std::string split;
    std::string target_path;
    std::string draft_path;
    std::string prompts_path;
    std::string warmup_path;
    std::string raw_path;
    std::string outputs_path;
    std::string observations_path;
    std::string calibrator_path;
    std::string action_costs_path;
    std::vector<std::uint64_t> seeds;
    std::size_t batch_size = 1;
    std::size_t max_new_tokens = 64;
    std::size_t context_size = 256;
    std::size_t batch_tokens = 512;
    std::size_t speculative_tokens = 4;
    std::size_t tree_branching = 2;
    std::size_t tree_depth = 3;
    std::size_t repeat_window = 64;
    int threads = 2;
    pulsar::runtime::SamplingConfig sampling{};
};

std::string require_value(int& index, int argc, char** argv, const std::string& name) {
    if (index + 1 >= argc) {
        throw std::invalid_argument("missing value for " + name);
    }
    return argv[++index];
}

std::uint64_t parse_u64(const std::string& text, const std::string& name) {
    if (text.empty() || text.front() == '-') {
        throw std::invalid_argument(name + " must be an unsigned integer");
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long long value = std::strtoull(text.c_str(), &end, 10);
    if (errno == ERANGE || end == text.c_str() || *end != '\0') {
        throw std::invalid_argument("invalid integer for " + name + ": " + text);
    }
    return static_cast<std::uint64_t>(value);
}

std::size_t parse_size(const std::string& text, const std::string& name) {
    const std::uint64_t value = parse_u64(text, name);
    if (value > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument(name + " exceeds this platform's size limit");
    }
    return static_cast<std::size_t>(value);
}

int parse_int(const std::string& text, const std::string& name) {
    const auto value = parse_size(text, name);
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(name + " is too large");
    }
    return static_cast<int>(value);
}

double parse_double(const std::string& text, const std::string& name) {
    if (text.empty()) {
        throw std::invalid_argument(name + " must not be empty");
    }
    errno = 0;
    char* end = nullptr;
    const double value = std::strtod(text.c_str(), &end);
    if (errno == ERANGE || end == text.c_str() || *end != '\0' || !std::isfinite(value)) {
        throw std::invalid_argument("invalid finite number for " + name + ": " + text);
    }
    return value;
}

std::vector<std::uint64_t> parse_seed_list(const std::string& text) {
    std::vector<std::uint64_t> values;
    std::istringstream input(text);
    std::string value;
    while (std::getline(input, value, ',')) {
        values.push_back(parse_u64(value, "--seeds"));
    }
    if (values.empty()) {
        throw std::invalid_argument("--seeds must contain at least one seed");
    }
    return values;
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::cout << "pulsar-phase4-bench --mode autoregressive|pulsar-tree|pulsar-adaptive --model PATH "
                         "--prompts TSV --raw-out FILE --outputs-out FILE [options]\n";
            std::exit(0);
        } else if (arg == "--mode") {
            options.mode = require_value(i, argc, argv, arg);
        } else if (arg == "--split") {
            options.split = require_value(i, argc, argv, arg);
        } else if (arg == "--model") {
            options.target_path = require_value(i, argc, argv, arg);
        } else if (arg == "--draft-model") {
            options.draft_path = require_value(i, argc, argv, arg);
        } else if (arg == "--prompts") {
            options.prompts_path = require_value(i, argc, argv, arg);
        } else if (arg == "--warmup-prompts") {
            options.warmup_path = require_value(i, argc, argv, arg);
        } else if (arg == "--raw-out") {
            options.raw_path = require_value(i, argc, argv, arg);
        } else if (arg == "--outputs-out") {
            options.outputs_path = require_value(i, argc, argv, arg);
        } else if (arg == "--observations-out") {
            options.observations_path = require_value(i, argc, argv, arg);
        } else if (arg == "--calibrator") {
            options.calibrator_path = require_value(i, argc, argv, arg);
        } else if (arg == "--action-costs") {
            options.action_costs_path = require_value(i, argc, argv, arg);
        } else if (arg == "--seeds") {
            options.seeds = parse_seed_list(require_value(i, argc, argv, arg));
        } else if (arg == "--batch-size") {
            options.batch_size = parse_size(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--max-new-tokens") {
            options.max_new_tokens = parse_size(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--context-size") {
            options.context_size = parse_size(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--batch-tokens") {
            options.batch_tokens = parse_size(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--threads") {
            options.threads = parse_int(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--temperature") {
            options.sampling.temperature = parse_double(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--top-k") {
            options.sampling.top_k = parse_size(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--top-p") {
            options.sampling.top_p = parse_double(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--repeat-penalty") {
            options.sampling.repetition_penalty = parse_double(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--repeat-window") {
            options.repeat_window = parse_size(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--speculative-tokens") {
            options.speculative_tokens = parse_size(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--tree-branching") {
            options.tree_branching = parse_size(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--tree-depth") {
            options.tree_depth = parse_size(require_value(i, argc, argv, arg), arg);
        } else {
            throw std::invalid_argument("unknown option: " + arg);
        }
    }
    if (options.target_path.empty() || options.prompts_path.empty() ||
        options.raw_path.empty() || options.outputs_path.empty() || options.split.empty() ||
        options.mode.empty() || options.seeds.empty()) {
        throw std::invalid_argument("mode, split, target, prompts, output files, and seeds are required");
    }
    if (options.mode != "autoregressive" && options.mode != "pulsar-tree" &&
        options.mode != "pulsar-adaptive") {
        throw std::invalid_argument("mode must be autoregressive, pulsar-tree, or pulsar-adaptive");
    }
    const bool tree_mode = options.mode != "autoregressive";
    if (tree_mode != !options.draft_path.empty()) {
        throw std::invalid_argument("tree modes require --draft-model; autoregressive must not set it");
    }
    if (tree_mode && options.observations_path.empty()) {
        throw std::invalid_argument("tree modes require --observations-out for the Phase 4 trace");
    }
    if (options.mode == "pulsar-adaptive" &&
        (options.calibrator_path.empty() || options.action_costs_path.empty())) {
        throw std::invalid_argument("pulsar-adaptive requires --calibrator and --action-costs");
    }
    if (options.batch_size == 0 || options.max_new_tokens == 0 || options.context_size == 0 ||
        options.batch_tokens == 0 || options.threads <= 0) {
        throw std::invalid_argument("batch, output, context, batch-token, and thread settings must be positive");
    }
    if (tree_mode && options.tree_depth == 0) {
        throw std::invalid_argument("tree depth must be positive");
    }
    options.sampling.repetition_window = options.repeat_window;
    pulsar::runtime::validate_sampling_config(options.sampling);
    return options;
}

std::vector<Prompt> read_prompts(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open prompt file: " + path);
    }
    std::vector<Prompt> prompts;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        const std::size_t first = line.find('\t');
        const std::size_t second = first == std::string::npos ? first : line.find('\t', first + 1);
        if (first == std::string::npos || second == std::string::npos) {
            throw std::runtime_error("prompt TSV rows require numeric_id, prompt_id, and text");
        }
        Prompt prompt;
        prompt.numeric_id = parse_size(line.substr(0, first), "prompt numeric id");
        prompt.id = line.substr(first + 1, second - first - 1);
        prompt.text = line.substr(second + 1);
        if (prompt.id.empty() || prompt.text.empty()) {
            throw std::runtime_error("prompt ID and text must not be empty");
        }
        prompts.push_back(std::move(prompt));
    }
    if (!input.eof()) {
        throw std::runtime_error("failed while reading prompt file: " + path);
    }
    if (prompts.empty()) {
        throw std::runtime_error("prompt file has no rows: " + path);
    }
    return prompts;
}

std::string json_quote(const std::string& text) {
    std::ostringstream out;
    out << '"';
    for (const unsigned char ch : text) {
        switch (ch) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (ch < 0x20) {
                    out << "\\u00" << std::hex << std::setw(2) << std::setfill('0')
                        << static_cast<unsigned int>(ch) << std::dec;
                } else {
                    out << static_cast<char>(ch);
                }
        }
    }
    out << '"';
    return out.str();
}

std::uint64_t request_seed_input(std::uint64_t root, std::size_t id) {
    if (id > std::numeric_limits<std::uint64_t>::max() - root) {
        throw std::overflow_error("seed plus prompt ID overflows");
    }
    return root + static_cast<std::uint64_t>(id);
}

std::uint64_t pulsar_rng_seed(std::uint64_t root, std::size_t id) {
    return pulsar::runtime::splitmix64(request_seed_input(root, id));
}

class InstrumentedModel final : public Model {
public:
    explicit InstrumentedModel(LlamaModel& model)
        : model_(model), sequence_lengths_(model.max_sequence_count(), 0) {}

    std::size_t vocabulary_size() const noexcept override { return model_.vocabulary_size(); }
    std::size_t max_sequence_count() const noexcept override { return model_.max_sequence_count(); }
    std::vector<Token> tokenize(const std::string& text) const override { return model_.tokenize(text); }
    std::string token_to_piece(Token token) const override { return model_.token_to_piece(token); }
    bool is_end_token(Token token) const noexcept override { return model_.is_end_token(token); }

    void clear_counters() {
        prefill_calls = 0;
        prefill_tokens = 0;
        decode_calls = 0;
        decode_tokens = 0;
        reset_calls = 0;
        copy_calls = 0;
        rollback_calls = 0;
        prefill_ms = 0.0;
        decode_ms = 0.0;
        cache_ops_ms = 0.0;
        peak_live_kv_tokens = 0;
        std::fill(sequence_lengths_.begin(), sequence_lengths_.end(), 0);
    }

    void reset() override {
        const auto started = std::chrono::steady_clock::now();
        model_.reset();
        ++reset_calls;
        cache_ops_ms += elapsed_ms(started);
        std::fill(sequence_lengths_.begin(), sequence_lengths_.end(), 0);
        update_peak();
    }

    void prefill(const std::vector<std::vector<Token>>& prompts,
                 std::vector<std::vector<float>>& next_logits) override {
        const auto started = std::chrono::steady_clock::now();
        model_.prefill(prompts, next_logits);
        ++prefill_calls;
        for (std::size_t i = 0; i < prompts.size(); ++i) {
            prefill_tokens += prompts[i].size();
            sequence_lengths_.at(i) = prompts[i].size();
        }
        prefill_ms += elapsed_ms(started);
        update_peak();
    }

    void decode_batch(const std::vector<SequenceInput>& inputs,
                      std::vector<std::vector<float>>& next_logits) override {
        const auto started = std::chrono::steady_clock::now();
        model_.decode_batch(inputs, next_logits);
        ++decode_calls;
        decode_tokens += inputs.size();
        for (const SequenceInput& input : inputs) {
            if (input.sequence_id >= sequence_lengths_.size()) {
                throw std::out_of_range("instrumented sequence ID exceeds configured capacity");
            }
            if (input.position == std::numeric_limits<std::size_t>::max()) {
                throw std::overflow_error("instrumented KV position overflows");
            }
            sequence_lengths_[input.sequence_id] = std::max(
                sequence_lengths_[input.sequence_id], input.position + 1);
        }
        decode_ms += elapsed_ms(started);
        update_peak();
    }

    void copy_sequence(std::size_t source_id, std::size_t destination_id) override {
        const auto started = std::chrono::steady_clock::now();
        model_.copy_sequence(source_id, destination_id);
        ++copy_calls;
        if (source_id != destination_id) {
            sequence_lengths_.at(destination_id) = sequence_lengths_.at(source_id);
        }
        cache_ops_ms += elapsed_ms(started);
        update_peak();
    }

    void rollback(std::size_t sequence_id, std::size_t new_length) override {
        const auto started = std::chrono::steady_clock::now();
        model_.rollback(sequence_id, new_length);
        ++rollback_calls;
        sequence_lengths_.at(sequence_id) = new_length;
        cache_ops_ms += elapsed_ms(started);
        update_peak();
    }

    std::size_t current_live_kv_tokens() const {
        return std::accumulate(sequence_lengths_.begin(), sequence_lengths_.end(), std::size_t{0});
    }

    std::size_t peak_live_kv_tokens = 0;
    std::size_t prefill_calls = 0;
    std::size_t prefill_tokens = 0;
    std::size_t decode_calls = 0;
    std::size_t decode_tokens = 0;
    std::size_t reset_calls = 0;
    std::size_t copy_calls = 0;
    std::size_t rollback_calls = 0;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    double cache_ops_ms = 0.0;

private:
    static double elapsed_ms(std::chrono::steady_clock::time_point started) {
        return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
    }

    void update_peak() {
        peak_live_kv_tokens = std::max(peak_live_kv_tokens, current_live_kv_tokens());
    }

    LlamaModel& model_;
    std::vector<std::size_t> sequence_lengths_;
};

std::size_t peak_rss_bytes() {
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0 || usage.ru_maxrss < 0) return 0;
    constexpr std::size_t bytes_per_kib = 1024;
    const auto peak_kib = static_cast<std::uint64_t>(usage.ru_maxrss);
    if (peak_kib > std::numeric_limits<std::size_t>::max() / bytes_per_kib) return 0;
    return static_cast<std::size_t>(peak_kib) * bytes_per_kib;
}

void write_model_counters(std::ostream& out, const InstrumentedModel& model,
                          std::size_t bytes_per_token) {
    out << "{\"prefill_calls\":" << model.prefill_calls
        << ",\"prefill_tokens\":" << model.prefill_tokens
        << ",\"decode_calls\":" << model.decode_calls
        << ",\"decode_tokens\":" << model.decode_tokens
        << ",\"explicit_reset_calls\":" << model.reset_calls
        << ",\"sequence_copy_calls\":" << model.copy_calls
        << ",\"sequence_rollback_calls\":" << model.rollback_calls
        << ",\"prefill_ms\":" << model.prefill_ms
        << ",\"decode_ms\":" << model.decode_ms
        << ",\"cache_operation_ms\":" << model.cache_ops_ms
        << ",\"peak_live_kv_tokens\":" << model.peak_live_kv_tokens
        << ",\"current_live_kv_tokens\":" << model.current_live_kv_tokens()
        << ",\"kv_bytes_per_token_estimate\":" << bytes_per_token
        << ",\"peak_kv_bytes_estimate\":" << model.peak_live_kv_tokens * bytes_per_token
        << '}';
}

void write_size_array(std::ostream& out, const std::vector<std::size_t>& values) {
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) out << ',';
        out << values[i];
    }
    out << ']';
}

void write_u64_array(std::ostream& out, const std::vector<std::uint64_t>& values) {
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) out << ',';
        out << values[i];
    }
    out << ']';
}

void write_double_array(std::ostream& out, const std::vector<double>& values) {
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) out << ',';
        out << values[i];
    }
    out << ']';
}

void run_one_batch(const Options& options,
                   const GenerationOptions& generation,
                   InstrumentedModel& target,
                   InstrumentedModel* draft,
                   const std::vector<Prompt>& batch,
                   std::uint64_t seed_root,
                   std::size_t trial,
                   std::size_t group,
                   std::size_t target_kv_bytes_per_token,
                   std::size_t draft_kv_bytes_per_token,
                   std::ostream& raw,
                   std::ostream& outputs,
                   pulsar::runtime::AdaptiveTreeScheduler* scheduler,
                   std::ostream* observations) {
    target.clear_counters();
    if (draft != nullptr) draft->clear_counters();
    if (scheduler != nullptr) scheduler->reset();
    std::vector<std::string> texts;
    texts.reserve(batch.size());
    for (const Prompt& prompt : batch) texts.push_back(prompt.text);

    const std::size_t first_id = batch.front().numeric_id;
    GenerationOptions current_generation = generation;
    current_generation.seed = seed_root;
    std::vector<pulsar::runtime::RuntimeTreeObservation> trace;
    const auto batch_started = std::chrono::steady_clock::now();
    const auto generation_started = std::chrono::steady_clock::now();
    const std::vector<GenerationResult> results = draft == nullptr
        ? pulsar::runtime::generate_batch(target, texts, current_generation, first_id)
        : pulsar::runtime::generate_candidate_tree_batch(
              target, *draft, texts, current_generation, options.tree_branching,
              options.tree_depth, first_id, scheduler,
              observations == nullptr ? nullptr : &trace);
    const double generation_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - generation_started).count();

    std::size_t total_tokens = 0;
    std::size_t total_proposals = 0;
    std::size_t total_accepted = 0;
    std::size_t total_rejected = 0;
    std::size_t total_tree_pulses = 0;
    std::size_t total_tree_nodes = 0;
    double tree_build_ms = 0.0;
    double tree_verification_ms = 0.0;
    std::vector<double> prompt_latencies;
    std::vector<std::size_t> prompt_tokens;
    std::vector<std::size_t> completion_tokens;
    std::vector<std::string> finish_reasons;
    std::vector<std::uint64_t> request_seed_inputs;
    std::vector<std::uint64_t> pulsar_rng_seeds;
    for (std::size_t i = 0; i < results.size(); ++i) {
        const GenerationResult& result = results[i];
        total_tokens += result.generated_tokens;
        total_proposals += result.draft_proposals;
        total_accepted += result.accepted_proposals;
        total_rejected += result.rejected_blocks;
        total_tree_pulses += result.candidate_tree_pulses;
        total_tree_nodes += result.candidate_tree_nodes;
        tree_build_ms += result.candidate_tree_build_ms;
        tree_verification_ms += result.candidate_tree_verification_ms;
        prompt_latencies.push_back(result.elapsed_ms);
        prompt_tokens.push_back(result.prompt_tokens);
        completion_tokens.push_back(result.generated_tokens);
        finish_reasons.emplace_back(pulsar::runtime::to_string(result.finish_reason));
        request_seed_inputs.push_back(request_seed_input(seed_root, batch[i].numeric_id));
        pulsar_rng_seeds.push_back(pulsar_rng_seed(seed_root, batch[i].numeric_id));
    }

    const std::string batch_id = options.split + "_b" + std::to_string(options.batch_size) +
        "_s" + std::to_string(seed_root) + "_g" + std::to_string(group);
    const auto output_started = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < results.size(); ++i) {
        outputs << "{\"batch_id\":" << json_quote(batch_id)
                << ",\"split\":" << json_quote(options.split)
                << ",\"prompt_id\":" << json_quote(batch[i].id)
                << ",\"prompt_numeric_id\":" << batch[i].numeric_id
                << ",\"prompt\":" << json_quote(batch[i].text)
                << ",\"completion\":" << json_quote(results[i].text)
                << ",\"completion_tokens\":" << results[i].generated_tokens
                << ",\"finish_reason\":"
                << json_quote(pulsar::runtime::to_string(results[i].finish_reason)) << "}\n";
    }
    outputs.flush();
    if (!outputs) throw std::runtime_error("failed to flush generated output records");
    const double output_write_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - output_started).count();
    const double batch_wall_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - batch_started).count();

    if (observations != nullptr) {
        for (const pulsar::runtime::RuntimeTreeObservation& event : trace) {
            if (event.request_id < first_id || event.request_id - first_id >= batch.size()) {
                throw std::logic_error("Phase 4 observation refers to a prompt outside its batch");
            }
            const std::size_t local = static_cast<std::size_t>(event.request_id - first_id);
            *observations << std::setprecision(9)
                << "{\"split\":" << json_quote(options.split)
                << ",\"mode\":" << json_quote(options.mode)
                << ",\"batch_id\":" << json_quote(batch_id)
                << ",\"context_group\":" << json_quote(batch[local].id)
                << ",\"prompt_numeric_id\":" << batch[local].numeric_id
                << ",\"seed_root\":" << seed_root
                << ",\"trial\":" << trial
                << ",\"batch_index\":" << group
                << ",\"batch_size\":" << event.batch_size
                << ",\"branching\":" << event.branching
                << ",\"configured_depth\":" << event.configured_depth
                << ",\"pulse_depth\":" << event.pulse_depth
                << ",\"pulse_index\":" << event.pulse_index
                << ",\"depth\":" << event.depth
                << ",\"depth_position\":" << (event.depth + 1)
                << ",\"root_q_max\":" << event.root_features.max_probability
                << ",\"root_q_margin\":" << event.root_features.top_two_margin
                << ",\"root_q_entropy\":" << event.root_features.normalized_entropy
                << ",\"node_q_max\":" << event.node_features.max_probability
                << ",\"node_q_margin\":" << event.node_features.top_two_margin
                << ",\"node_q_entropy\":" << event.node_features.normalized_entropy
                << ",\"proposal_probability\":" << event.proposal_probability
                << ",\"accepted\":" << (event.accepted ? "true" : "false")
                << ",\"continuation_label_available\":"
                << (event.continuation_label_available ? "true" : "false")
                << ",\"continued\":" << (event.continued ? "true" : "false") << "}\n";
        }
        observations->flush();
        if (!*observations) throw std::runtime_error("failed to flush verifier observations");
    }

    raw << std::setprecision(9)
        << "{\"schema_version\":1"
        << ",\"batch_id\":" << json_quote(batch_id)
        << ",\"engine\":\"pulsar\""
        << ",\"mode\":" << json_quote(options.mode)
        << ",\"split\":" << json_quote(options.split)
        << ",\"trial\":" << trial
        << ",\"seed_root\":" << seed_root
        << ",\"batch_index\":" << group
        << ",\"batch_size_configured\":" << options.batch_size
        << ",\"batch_prompt_ids\":[";
    for (std::size_t i = 0; i < batch.size(); ++i) {
        if (i != 0) raw << ',';
        raw << json_quote(batch[i].id);
    }
    raw << "]"
        << ",\"request_seed_inputs\":";
    write_u64_array(raw, request_seed_inputs);
    raw << ",\"pulsar_rng_seeds\":";
    write_u64_array(raw, pulsar_rng_seeds);
    raw << ",\"prompt_tokens\":";
    write_size_array(raw, prompt_tokens);
    raw << ",\"completion_tokens\":";
    write_size_array(raw, completion_tokens);
    raw << ",\"finish_reasons\":[";
    for (std::size_t i = 0; i < finish_reasons.size(); ++i) {
        if (i != 0) raw << ',';
        raw << json_quote(finish_reasons[i]);
    }
    raw << "]"
        << ",\"output_horizon\":" << options.max_new_tokens
        << ",\"tree_branching\":" << (draft == nullptr ? 0 : options.tree_branching)
        << ",\"tree_depth\":" << (draft == nullptr ? 0 : options.tree_depth)
        << ",\"sampling\":{\"temperature\":" << options.sampling.temperature
        << ",\"top_k\":" << options.sampling.top_k
        << ",\"top_p\":" << options.sampling.top_p
        << ",\"repetition_penalty\":" << options.sampling.repetition_penalty
        << ",\"repetition_window\":" << options.sampling.repetition_window << '}'
        << ",\"generation_wall_ms\":" << generation_ms
        << ",\"output_write_ms\":" << output_write_ms
        << ",\"batch_completion_ms\":" << batch_wall_ms
        << ",\"aggregate_tokens_per_second\":"
        << (batch_wall_ms > 0.0 ? total_tokens * 1000.0 / batch_wall_ms : 0.0)
        << ",\"request_latency_ms\":";
    write_double_array(raw, prompt_latencies);
    raw << ",\"total_completion_tokens\":" << total_tokens
        << ",\"proposed_tokens\":" << total_proposals
        << ",\"accepted_proposals\":" << total_accepted
        << ",\"rejected_blocks\":" << total_rejected
        << ",\"target_verifier_decode_tokens\":"
        << (draft == nullptr ? 0 : target.decode_tokens)
        << ",\"candidate_tree_pulses\":" << total_tree_pulses
        << ",\"candidate_tree_nodes_built\":" << total_tree_nodes
        << ",\"candidate_tree_build_inclusive_ms\":" << tree_build_ms
        << ",\"candidate_tree_verification_inclusive_ms\":" << tree_verification_ms
        << ",\"target_model_work\":";
    write_model_counters(raw, target, target_kv_bytes_per_token);
    raw << ",\"draft_model_work\":";
    if (draft == nullptr) {
        raw << "null";
    } else {
        write_model_counters(raw, *draft, draft_kv_bytes_per_token);
    }
    raw << ",\"peak_process_rss_bytes\":" << peak_rss_bytes()
        << ",\"kv_estimate_note\":\"Live sequence lengths times F16/equal-head-dimension estimate; not allocator bytes\"}\n";
    raw.flush();
    if (!raw) throw std::runtime_error("failed to flush raw benchmark record");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        const std::vector<Prompt> prompts = read_prompts(options.prompts_path);
        const std::vector<Prompt> warmup_prompts = options.warmup_path.empty()
            ? std::vector<Prompt>{prompts.front()} : read_prompts(options.warmup_path);
        if (prompts.size() % options.batch_size != 0) {
            throw std::invalid_argument("prompt count must divide evenly by configured batch size");
        }
        for (std::size_t i = 1; i < prompts.size(); ++i) {
            if (prompts[i].numeric_id != prompts[i - 1].numeric_id + 1) {
                throw std::invalid_argument("prompt IDs in each workload must be contiguous and ordered");
            }
        }

        const bool tree_mode = options.mode != "autoregressive";
        const bool adaptive_mode = options.mode == "pulsar-adaptive";
        std::unique_ptr<pulsar::runtime::AdaptiveTreeScheduler> adaptive_scheduler;
        if (adaptive_mode) {
            auto calibrator = pulsar::runtime::ConfidenceCalibrator::load_tsv(
                options.calibrator_path);
            auto costs = pulsar::runtime::load_action_costs_tsv(options.action_costs_path);
            adaptive_scheduler = std::make_unique<pulsar::runtime::AdaptiveTreeScheduler>(
                std::move(calibrator), std::move(costs));
        }
        std::size_t node_bound = 1;
        if (tree_mode) {
            node_bound = pulsar::runtime::candidate_tree_node_bound(
                options.tree_branching, options.tree_depth);
            if (options.batch_size > std::numeric_limits<std::size_t>::max() / node_bound) {
                throw std::length_error("tree batch sequence requirement overflows");
            }
        }
        const std::size_t max_sequences = options.batch_size * (tree_mode ? node_bound : 1);
        LlamaModelOptions model_options;
        model_options.model_path = options.target_path;
        model_options.max_context_tokens = options.context_size;
        model_options.max_sequences = max_sequences;
        model_options.batch_tokens = options.batch_tokens;
        model_options.threads = options.threads;
        const auto loading_started = std::chrono::steady_clock::now();
        LlamaModel target_model(model_options);
        std::unique_ptr<LlamaModel> draft_model;
        if (tree_mode) {
            LlamaModelOptions draft_options = model_options;
            draft_options.model_path = options.draft_path;
            draft_model = std::make_unique<LlamaModel>(draft_options);
        }
        const double model_load_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - loading_started).count();
        InstrumentedModel target(target_model);
        std::unique_ptr<InstrumentedModel> draft;
        if (draft_model != nullptr) draft = std::make_unique<InstrumentedModel>(*draft_model);

        GenerationOptions generation;
        generation.max_new_tokens = options.max_new_tokens;
        generation.max_context_tokens = options.context_size;
        generation.batch_size = options.batch_size;
        generation.sampling = options.sampling;

        // One excluded warm-up batch on the first fixed prompt. Timing files and
        // counters are cleared before any measured run.
        std::vector<Prompt> warmup_batch;
        for (const Prompt& prompt : warmup_prompts) {
            if (warmup_batch.size() == options.batch_size) break;
            warmup_batch.push_back(prompt);
        }
        if (warmup_batch.empty()) warmup_batch.push_back(prompts.front());
        target.clear_counters();
        if (draft != nullptr) draft->clear_counters();
        std::vector<std::string> warmup_texts;
        for (const Prompt& prompt : warmup_batch) warmup_texts.push_back(prompt.text);
        const std::uint64_t warmup_root = options.seeds.front();
        generation.seed = warmup_root;
        if (draft == nullptr) {
            static_cast<void>(pulsar::runtime::generate_batch(
                target, warmup_texts, generation, warmup_batch.front().numeric_id));
        } else {
            static_cast<void>(pulsar::runtime::generate_candidate_tree_batch(
                target, *draft, warmup_texts, generation, options.tree_branching,
                options.tree_depth, warmup_batch.front().numeric_id,
                adaptive_scheduler.get()));
        }
        if (adaptive_scheduler != nullptr) adaptive_scheduler->reset();
        target.clear_counters();
        if (draft != nullptr) draft->clear_counters();

        std::ofstream raw(options.raw_path, std::ios::binary | std::ios::trunc);
        std::ofstream outputs(options.outputs_path, std::ios::binary | std::ios::trunc);
        std::ofstream observation_file;
        if (tree_mode) {
            observation_file.open(options.observations_path, std::ios::binary | std::ios::trunc);
        }
        if (!raw || !outputs || (tree_mode && !observation_file)) {
            throw std::runtime_error("cannot open raw or output benchmark data file");
        }
        for (std::size_t trial = 0; trial < options.seeds.size(); ++trial) {
            const std::uint64_t seed_root = options.seeds[trial];
            std::size_t group = 0;
            for (std::size_t begin = 0; begin < prompts.size(); begin += options.batch_size, ++group) {
                const std::vector<Prompt> batch(
                    prompts.begin() + static_cast<std::ptrdiff_t>(begin),
                    prompts.begin() + static_cast<std::ptrdiff_t>(begin + options.batch_size));
                run_one_batch(options, generation, target, draft.get(), batch,
                              seed_root, trial, group,
                              target_model.kv_cache_bytes_per_token_estimate(),
                              draft_model == nullptr ? 0 : draft_model->kv_cache_bytes_per_token_estimate(),
                              raw, outputs, adaptive_scheduler.get(),
                              tree_mode ? &observation_file : nullptr);
            }
        }
        std::cerr << "loaded_ms=" << std::fixed << std::setprecision(3) << model_load_ms
                  << " peak_rss_bytes=" << peak_rss_bytes() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "pulsar-phase4-bench: " << error.what() << '\n';
        return 2;
    }
}
