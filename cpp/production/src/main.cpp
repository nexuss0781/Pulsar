#include "llama_model.hpp"
#include "pulsar/speculative_generation.hpp"
#include "pulsar/tree_generation.hpp"

#include <cerrno>
#include <csignal>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using pulsar::runtime::GenerationOptions;
using pulsar::runtime::LlamaModel;
using pulsar::runtime::LlamaModelOptions;

volatile std::sig_atomic_t cancellation_signal = 0;

void record_cancellation_signal(int signal_number) noexcept {
    cancellation_signal = signal_number;
}

int cancellation_exit_status() noexcept {
    return cancellation_signal == 0 ? 0 : 128 + cancellation_signal;
}

struct CliOptions {
    std::string model_path;
    std::string draft_model_path;
    std::string input_path = "-";
    std::string output_path = "-";
    std::string prompt;
    bool has_prompt = false;
    std::size_t max_new_tokens = 128;
    std::size_t context_size = 4096;
    std::size_t batch_size = 1;
    std::size_t batch_tokens = 512;
    std::size_t speculative_tokens = 4;
    bool speculative_tokens_set = false;
    std::size_t tree_branching = 2;
    std::size_t tree_depth = 4;
    bool tree_branching_set = false;
    bool tree_depth_set = false;
    std::uint64_t seed = 1;
    int threads = 0;
    pulsar::runtime::SamplingConfig sampling{};
};

void usage(std::ostream& out) {
    out << "Pulsar native CPU text generation\n\n"
        << "Usage:\n"
        << "  pulsar-generate --model MODEL.gguf [options]\n\n"
        << "Input is one prompt per line from --input FILE (default stdin).\n"
        << "Output is one JSON object per prompt to --output FILE (default stdout).\n\n"
        << "Options:\n"
        << "  --model PATH             Required local GGUF model\n"
        << "  --draft-model PATH       Optional compatible draft GGUF; enables exact speculation\n"
        << "  --speculative-tokens N   Draft proposals per block (default 4; requires --draft-model)\n"
        << "  --tree-branching N       Opt into fixed candidate tree; retain top N children (default 2)\n"
        << "  --tree-depth N           Candidate verifier positions per pulse (default 4)\n"
        << "  --input PATH             Prompt lines; use - for stdin (default)\n"
        << "  --output PATH            JSONL output; use - for stdout (default)\n"
        << "  --prompt TEXT            Generate one prompt instead of reading input\n"
        << "  --max-new-tokens N       Completion limit (default 128)\n"
        << "  --context-size N         Maximum context tokens per sequence (default 4096)\n"
        << "  --batch-size N           Concurrent sequences per microbatch (default 1)\n"
        << "  --batch-tokens N         Prompt-prefill token batch capacity (default 512)\n"
        << "  --threads N              CPU threads; 0 uses llama.cpp defaults\n"
        << "  --seed N                 Base seed; each prompt has an independent stream\n"
        << "  --temperature X          0 is greedy; default 0.8\n"
        << "  --top-k N                0 disables top-k; default 40\n"
        << "  --top-p X                Nucleus probability in (0,1]; default 0.95\n"
        << "  --repeat-penalty X       >= 1; default 1.0 (disabled)\n"
        << "  --repeat-window N        Recent tokens to penalize; default 64\n"
        << "  SIGINT/SIGTERM           Best-effort cancellation between inference steps\n"
        << "  --help                   Show this help\n";
}

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
    const std::uint64_t value = parse_u64(text, name);
    if (value > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
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

CliOptions parse_args(int argc, char** argv, bool& show_help) {
    CliOptions options;
    show_help = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            show_help = true;
            return options;
        } else if (arg == "--model") {
            options.model_path = require_value(i, argc, argv, arg);
        } else if (arg == "--draft-model") {
            options.draft_model_path = require_value(i, argc, argv, arg);
        } else if (arg == "--speculative-tokens") {
            options.speculative_tokens = parse_size(require_value(i, argc, argv, arg), arg);
            options.speculative_tokens_set = true;
        } else if (arg == "--tree-branching") {
            options.tree_branching = parse_size(require_value(i, argc, argv, arg), arg);
            options.tree_branching_set = true;
        } else if (arg == "--tree-depth") {
            options.tree_depth = parse_size(require_value(i, argc, argv, arg), arg);
            options.tree_depth_set = true;
        } else if (arg == "--input") {
            options.input_path = require_value(i, argc, argv, arg);
        } else if (arg == "--output") {
            options.output_path = require_value(i, argc, argv, arg);
        } else if (arg == "--prompt") {
            options.prompt = require_value(i, argc, argv, arg);
            options.has_prompt = true;
        } else if (arg == "--max-new-tokens") {
            options.max_new_tokens = parse_size(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--context-size") {
            options.context_size = parse_size(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--batch-size") {
            options.batch_size = parse_size(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--batch-tokens") {
            options.batch_tokens = parse_size(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--threads") {
            options.threads = parse_int(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--seed") {
            options.seed = parse_u64(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--temperature") {
            options.sampling.temperature = parse_double(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--top-k") {
            options.sampling.top_k = parse_size(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--top-p") {
            options.sampling.top_p = parse_double(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--repeat-penalty") {
            options.sampling.repetition_penalty = parse_double(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--repeat-window") {
            options.sampling.repetition_window = parse_size(require_value(i, argc, argv, arg), arg);
        } else {
            throw std::invalid_argument("unknown option: " + arg);
        }
    }

    if (options.model_path.empty()) {
        throw std::invalid_argument("--model is required");
    }
    if (options.has_prompt && options.input_path != "-") {
        throw std::invalid_argument("--prompt cannot be combined with --input");
    }
    if (options.draft_model_path.empty() && options.speculative_tokens_set) {
        throw std::invalid_argument("--speculative-tokens requires --draft-model");
    }
    if (!options.draft_model_path.empty() && options.speculative_tokens == 0) {
        throw std::invalid_argument("--speculative-tokens must be positive");
    }
    const bool tree_mode = options.tree_branching_set || options.tree_depth_set;
    if (tree_mode && options.draft_model_path.empty()) {
        throw std::invalid_argument("--tree-branching/--tree-depth require --draft-model");
    }
    if (tree_mode && options.speculative_tokens_set) {
        throw std::invalid_argument("candidate-tree options cannot be combined with --speculative-tokens");
    }
    if (tree_mode && options.tree_depth == 0) {
        throw std::invalid_argument("--tree-depth must be positive");
    }
    if (options.batch_size == 0 || options.context_size == 0 || options.batch_tokens == 0) {
        throw std::invalid_argument("batch size, context size, and batch tokens must be positive");
    }
    pulsar::runtime::validate_sampling_config(options.sampling);
    return options;
}

std::string json_escape(const std::string& text) {
    std::ostringstream out;
    out << '"';
    for (std::size_t i = 0; i < text.size();) {
        const unsigned char ch = static_cast<unsigned char>(text[i]);
        switch (ch) {
            case '"': out << "\\\""; ++i; continue;
            case '\\': out << "\\\\"; ++i; continue;
            case '\b': out << "\\b"; ++i; continue;
            case '\f': out << "\\f"; ++i; continue;
            case '\n': out << "\\n"; ++i; continue;
            case '\r': out << "\\r"; ++i; continue;
            case '\t': out << "\\t"; ++i; continue;
            default: break;
        }
        if (ch < 0x20) {
            out << "\\u00" << std::hex << std::setw(2) << std::setfill('0')
                << static_cast<unsigned int>(ch) << std::dec;
            ++i;
            continue;
        }
        if (ch < 0x80) {
            out << static_cast<char>(ch);
            ++i;
            continue;
        }

        std::size_t width = 0;
        std::uint32_t codepoint = 0;
        if (ch >= 0xC2 && ch <= 0xDF) {
            width = 2;
            codepoint = ch & 0x1F;
        } else if (ch >= 0xE0 && ch <= 0xEF) {
            width = 3;
            codepoint = ch & 0x0F;
        } else if (ch >= 0xF0 && ch <= 0xF4) {
            width = 4;
            codepoint = ch & 0x07;
        }
        bool valid = width != 0 && i + width <= text.size();
        if (valid) {
            for (std::size_t j = 1; j < width; ++j) {
                const unsigned char continuation = static_cast<unsigned char>(text[i + j]);
                if ((continuation & 0xC0) != 0x80) {
                    valid = false;
                    break;
                }
                codepoint = (codepoint << 6) | (continuation & 0x3F);
            }
            if ((width == 3 && codepoint < 0x800) ||
                (width == 4 && codepoint < 0x10000) ||
                codepoint > 0x10FFFF ||
                (codepoint >= 0xD800 && codepoint <= 0xDFFF)) {
                valid = false;
            }
        }
        if (!valid) {
            out << "\\uFFFD";
            ++i;
            continue;
        }
        out.write(text.data() + i, static_cast<std::streamsize>(width));
        i += width;
    }
    out << '"';
    return out.str();
}

void write_result(std::ostream& out, std::size_t id, const char* mode,
                  const pulsar::runtime::GenerationResult& result) {
    const double tokens_per_second = result.elapsed_ms > 0.0
        ? static_cast<double>(result.generated_tokens) * 1000.0 / result.elapsed_ms
        : 0.0;
    const double acceptance_rate = result.draft_proposals > 0
        ? static_cast<double>(result.accepted_proposals) /
              static_cast<double>(result.draft_proposals)
        : 0.0;
    out << "{\"id\":" << id
        << ",\"mode\":" << json_escape(mode)
        << ",\"prompt_tokens\":" << result.prompt_tokens
        << ",\"completion_tokens\":" << result.generated_tokens
        << ",\"draft_proposals\":" << result.draft_proposals
        << ",\"accepted_proposals\":" << result.accepted_proposals
        << ",\"rejected_blocks\":" << result.rejected_blocks
        << ",\"proposal_acceptance_rate\":" << std::setprecision(4) << acceptance_rate
        << ",\"finish_reason\":" << json_escape(pulsar::runtime::to_string(result.finish_reason))
        << ",\"elapsed_ms\":" << std::fixed << std::setprecision(3) << result.elapsed_ms
        << ",\"tokens_per_second\":" << std::setprecision(3) << tokens_per_second
        << ",\"text\":" << json_escape(result.text) << "}\n";
}

void process_prompts(LlamaModel& model,
                     LlamaModel* draft_model,
                     const std::vector<std::string>& prompts,
                     const GenerationOptions& generation,
                     std::size_t speculative_tokens,
                     bool tree_mode,
                     std::size_t tree_branching,
                     std::size_t tree_depth,
                     std::uint64_t first_id,
                     std::ostream& output) {
    const auto results = draft_model == nullptr
        ? pulsar::runtime::generate_batch(model, prompts, generation, first_id)
        : (tree_mode
            ? pulsar::runtime::generate_candidate_tree_batch(
                  model, *draft_model, prompts, generation, tree_branching,
                  tree_depth, first_id)
            : pulsar::runtime::generate_speculative_batch(
                  model, *draft_model, prompts, generation, speculative_tokens, first_id));
    const char* mode = draft_model == nullptr ? "autoregressive"
        : (tree_mode ? "exact_candidate_tree" : "exact_speculative_chain");
    for (std::size_t i = 0; i < results.size(); ++i) {
        write_result(output, static_cast<std::size_t>(first_id) + i, mode, results[i]);
    }
    output.flush();
    if (!output) {
        throw std::runtime_error("failed while writing generation output");
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::signal(SIGINT, record_cancellation_signal);
    std::signal(SIGTERM, record_cancellation_signal);
    try {
        bool show_help = false;
        const CliOptions cli = parse_args(argc, argv, show_help);
        if (show_help) {
            usage(std::cout);
            return 0;
        }

        std::ofstream output_file;
        std::ostream* output = &std::cout;
        if (cli.output_path != "-") {
            output_file.open(cli.output_path, std::ios::binary | std::ios::trunc);
            if (!output_file) {
                throw std::runtime_error("cannot open output file: " + cli.output_path);
            }
            output = &output_file;
        }

        LlamaModelOptions backend_options;
        backend_options.model_path = cli.model_path;
        backend_options.max_context_tokens = cli.context_size;
        backend_options.max_sequences = cli.batch_size;
        backend_options.batch_tokens = cli.batch_tokens;
        const bool tree_mode = cli.tree_branching_set || cli.tree_depth_set;
        if (tree_mode) {
            const std::size_t node_bound = pulsar::runtime::candidate_tree_node_bound(
                cli.tree_branching, cli.tree_depth);
            if (node_bound > std::numeric_limits<std::size_t>::max() / cli.batch_size) {
                throw std::length_error("candidate-tree batch sequence capacity overflows this platform");
            }
            backend_options.max_sequences = cli.batch_size * node_bound;
        } else if (!cli.draft_model_path.empty()) {
            if (cli.speculative_tokens > std::numeric_limits<std::size_t>::max() / cli.batch_size) {
                throw std::length_error("speculative batch token capacity overflows this platform");
            }
            backend_options.batch_tokens = std::max(
                backend_options.batch_tokens, cli.batch_size * cli.speculative_tokens);
        }
        backend_options.threads = cli.threads;
        LlamaModel model(backend_options);

        std::unique_ptr<LlamaModel> draft_model;
        if (!cli.draft_model_path.empty()) {
            LlamaModelOptions draft_options = backend_options;
            draft_options.model_path = cli.draft_model_path;
            draft_model = std::make_unique<LlamaModel>(draft_options);
        }

        GenerationOptions generation;
        generation.max_new_tokens = cli.max_new_tokens;
        generation.max_context_tokens = cli.context_size;
        generation.batch_size = cli.batch_size;
        generation.seed = cli.seed;
        generation.sampling = cli.sampling;
        generation.cancellation_requested = [] { return cancellation_signal != 0; };

        if (cli.has_prompt) {
            process_prompts(model, draft_model.get(), {cli.prompt}, generation,
                            cli.speculative_tokens, tree_mode, cli.tree_branching,
                            cli.tree_depth, 0, *output);
            return cancellation_exit_status();
        }

        std::ifstream input_file;
        std::istream* input = &std::cin;
        if (cli.input_path != "-") {
            input_file.open(cli.input_path, std::ios::binary);
            if (!input_file) {
                throw std::runtime_error("cannot open input file: " + cli.input_path);
            }
            input = &input_file;
        }

        std::vector<std::string> prompts;
        prompts.reserve(cli.batch_size);
        std::string line;
        std::uint64_t request_id = 0;
        while (cancellation_signal == 0 && std::getline(*input, line)) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            prompts.push_back(line);
            if (prompts.size() == cli.batch_size) {
                process_prompts(model, draft_model.get(), prompts, generation,
                                cli.speculative_tokens, tree_mode, cli.tree_branching,
                                cli.tree_depth, request_id, *output);
                request_id += prompts.size();
                prompts.clear();
            }
        }
        if (input->bad() && cancellation_signal == 0) {
            throw std::runtime_error("failed while reading prompt input");
        }
        if (cancellation_signal != 0) {
            return cancellation_exit_status();
        }
        if (!prompts.empty()) {
            process_prompts(model, draft_model.get(), prompts, generation,
                            cli.speculative_tokens, tree_mode, cli.tree_branching,
                            cli.tree_depth, request_id, *output);
        }
        return cancellation_exit_status();
    } catch (const std::exception& error) {
        std::cerr << "pulsar-generate: " << error.what() << '\n';
        return 2;
    }
}
