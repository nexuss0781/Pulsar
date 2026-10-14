#include "pulsar_engine.hpp"
#include "pulsar_speculative_v2.hpp"
#include "pulsar_stage4_bench.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using pulsar::v3::CandidateTree;
using pulsar::v3::Context;
using pulsar::v3::Token;
using pulsar::v3::Weights;
using Nanoseconds = std::chrono::nanoseconds;

struct Prompt {
    std::string id;
    Context context;
};

struct RowTimer {
    std::uint64_t nanoseconds = 0;
    std::uint64_t calls = 0;
    std::uint64_t values = 0;
};

struct RunStats {
    std::uint64_t seed = 0;
    std::uint64_t total_wall_ns = 0;
    std::uint64_t cycle_ns_total = 0;
    std::uint64_t tree_build_ns = 0;
    std::uint64_t threshold_select_ns = 0;
    std::uint64_t sample_ns = 0;
    std::uint64_t accepted_tokens = 0;
    std::uint64_t proposed_tokens = 0;
    std::uint64_t emitted_tokens = 0;
    std::uint64_t cycles = 0;
    std::uint64_t selected_tree_nodes_total = 0;
    std::uint64_t max_selected_tree_nodes = 0;
    std::uint64_t max_full_tree_bytes = 0;
    std::uint64_t max_selected_tree_bytes = 0;
    std::uint64_t selected_tree_bytes_total = 0;
    std::uint64_t hash = 14695981039346656037ULL;
    double selected_depth_total = 0.0;
    RowTimer draft;
    RowTimer target;
    std::vector<std::uint64_t> cycle_latencies_ns;
    std::vector<std::uint64_t> nodes_by_depth_total;
    std::vector<long double> retained_mass_total;
    std::vector<std::uint64_t> retained_mass_count;
};

std::vector<int> as_int_context(const Context& context) {
    std::vector<int> result;
    result.reserve(context.size());
    for (Token token : context) result.push_back(static_cast<int>(token));
    return result;
}

Weights copy_model_row(const double* row, std::size_t vocabulary) {
    return Weights(row, row + vocabulary);
}

std::uint64_t elapsed_ns(Clock::time_point start, Clock::time_point end) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<Nanoseconds>(end - start).count());
}

std::uint64_t tree_payload_bytes(const CandidateTree& tree) {
    std::uint64_t bytes = tree.nodes.capacity() * sizeof(pulsar::v3::CandidateTreeNode);
    for (const auto& node : tree.nodes) {
        bytes += node.context.capacity() * sizeof(Token);
        bytes += node.q.capacity() * sizeof(double);
        bytes += node.children.capacity() * sizeof(std::size_t);
    }
    return bytes;
}

std::uint64_t model_storage_bytes(const pulsar::Model& model) {
    return (model.P.capacity() + model.Q.capacity() + model.w.capacity() +
            model.isoK.capacity() + model.isoV.capacity() + model.qEnt.capacity() +
            model.qTop.capacity()) * sizeof(double);
}

std::uint64_t make_seed(std::size_t prompt_index, std::size_t repetition) {
    return 0x50554C5341520001ULL +
           static_cast<std::uint64_t>(prompt_index) * 0x9E3779B97F4A7C15ULL +
           static_cast<std::uint64_t>(repetition) * 0xD1B54A32D192ED03ULL;
}

void hash_token(RunStats& stats, Token token) {
    stats.hash ^= static_cast<std::uint64_t>(token);
    stats.hash *= 1099511628211ULL;
}

std::uint64_t percentile(std::vector<std::uint64_t> values, double p) {
    if (values.empty()) return 0;
    std::sort(values.begin(), values.end());
    const auto index = static_cast<std::size_t>(
        std::ceil(p * static_cast<double>(values.size())) - 1.0);
    return values[std::min(index, values.size() - 1)];
}

std::string hex64(std::uint64_t value) {
    std::ostringstream out;
    out << std::hex << std::setw(16) << std::setfill('0') << value;
    return out.str();
}

template <class Values>
std::string join_values(const Values& values) {
    std::ostringstream out;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) out << ';';
        out << values[i];
    }
    return out.str();
}

RunStats run_ar(const pulsar::Model& model, const Prompt& prompt,
                std::size_t token_limit, std::uint64_t seed) {
    RunStats stats;
    stats.seed = seed;
    std::mt19937_64 rng(seed);
    Context context = prompt.context;
    stats.cycle_latencies_ns.reserve(token_limit);

    const auto run_start = Clock::now();
    while (stats.emitted_tokens < token_limit) {
        const auto cycle_start = Clock::now();
        auto target_row = [&model, &stats](const Context& ctx) {
            const auto row_start = Clock::now();
            const auto int_ctx = as_int_context(ctx);
            const auto row = copy_model_row(
                model.prow(model.ctx3(int_ctx)), static_cast<std::size_t>(model.p.V));
            stats.target.nanoseconds += elapsed_ns(row_start, Clock::now());
            ++stats.target.calls;
            stats.target.values += row.size();
            return row;
        };
        const auto emitted = pulsar::v2::sample_autoregressive(
            context, 1, target_row, rng);
        if (emitted.size() != 1) throw std::runtime_error("AR baseline failed to emit one token");
        hash_token(stats, emitted.front());
        context.push_back(emitted.front());
        ++stats.emitted_tokens;
        ++stats.cycles;
        const std::uint64_t cycle_ns = elapsed_ns(cycle_start, Clock::now());
        stats.cycle_ns_total += cycle_ns;
        stats.cycle_latencies_ns.push_back(cycle_ns);
    }
    stats.total_wall_ns = elapsed_ns(run_start, Clock::now());
    return stats;
}

RunStats run_tree(const pulsar::Model& model, const Prompt& prompt,
                  std::size_t token_limit, std::uint64_t seed,
                  std::size_t configured_depth, std::size_t branch_factor,
                  double minimum_level_mass) {
    RunStats stats;
    stats.seed = seed;
    std::mt19937_64 rng(seed);
    Context context = prompt.context;
    stats.cycle_latencies_ns.reserve(token_limit);

    const auto run_start = Clock::now();
    while (stats.emitted_tokens < token_limit) {
        const auto cycle_start = Clock::now();
        const std::size_t remaining = token_limit - stats.emitted_tokens;
        const std::size_t pulse_depth = std::min(configured_depth, remaining);

        auto draft_row = [&model, &stats](const Context& ctx) {
            const auto row_start = Clock::now();
            const auto int_ctx = as_int_context(ctx);
            const auto row = copy_model_row(
                model.qrow(model.ctx2(int_ctx)), static_cast<std::size_t>(model.p.V));
            stats.draft.nanoseconds += elapsed_ns(row_start, Clock::now());
            ++stats.draft.calls;
            stats.draft.values += row.size();
            return row;
        };
        const auto build_start = Clock::now();
        CandidateTree tree = pulsar::v3::build_candidate_tree(
            context, branch_factor, pulse_depth, draft_row);
        stats.tree_build_ns += elapsed_ns(build_start, Clock::now());
        const std::uint64_t full_bytes = tree_payload_bytes(tree);
        stats.max_full_tree_bytes = std::max(stats.max_full_tree_bytes, full_bytes);

        const auto selection_start = Clock::now();
        const auto selection = pulsar::stage4::select_by_minimum_level_mass(
            tree, minimum_level_mass);
        stats.threshold_select_ns += elapsed_ns(selection_start, Clock::now());
        stats.selected_depth_total += static_cast<double>(selection.selected_depth);
        for (std::size_t i = 0; i < selection.retained_mass_by_depth.size(); ++i) {
            if (stats.retained_mass_total.size() <= i) {
                stats.retained_mass_total.resize(i + 1, 0.0L);
                stats.retained_mass_count.resize(i + 1, 0);
            }
            stats.retained_mass_total[i] += selection.retained_mass_by_depth[i];
            ++stats.retained_mass_count[i];
        }

        const std::uint64_t selected_bytes = tree_payload_bytes(tree);
        stats.max_selected_tree_bytes =
            std::max(stats.max_selected_tree_bytes, selected_bytes);
        stats.selected_tree_bytes_total += selected_bytes;
        stats.selected_tree_nodes_total += tree.nodes.size();
        stats.max_selected_tree_nodes =
            std::max<std::uint64_t>(stats.max_selected_tree_nodes, tree.nodes.size());
        for (const auto& node : tree.nodes) {
            if (stats.nodes_by_depth_total.size() <= node.depth) {
                stats.nodes_by_depth_total.resize(node.depth + 1, 0);
            }
            ++stats.nodes_by_depth_total[node.depth];
        }

        auto target_row = [&model, &stats](const Context& ctx) {
            const auto row_start = Clock::now();
            const auto int_ctx = as_int_context(ctx);
            const auto row = copy_model_row(
                model.prow(model.ctx3(int_ctx)), static_cast<std::size_t>(model.p.V));
            stats.target.nanoseconds += elapsed_ns(row_start, Clock::now());
            ++stats.target.calls;
            stats.target.values += row.size();
            return row;
        };
        const auto sample_start = Clock::now();
        const auto result = pulsar::v3::sample_candidate_tree(
            tree, context, target_row, rng);
        stats.sample_ns += elapsed_ns(sample_start, Clock::now());
        if (result.emitted_tokens.empty() || result.emitted_tokens.size() > remaining) {
            throw std::runtime_error("tree pulse emitted an invalid number of tokens");
        }
        stats.accepted_tokens += result.accepted_count;
        stats.proposed_tokens += result.proposed_tokens.size();
        for (Token token : result.emitted_tokens) {
            hash_token(stats, token);
            context.push_back(token);
            ++stats.emitted_tokens;
        }
        ++stats.cycles;
        const std::uint64_t cycle_ns = elapsed_ns(cycle_start, Clock::now());
        stats.cycle_ns_total += cycle_ns;
        stats.cycle_latencies_ns.push_back(cycle_ns);
    }
    stats.total_wall_ns = elapsed_ns(run_start, Clock::now());
    return stats;
}

void write_header(std::ofstream& out) {
    out << "kind,depth,branch_factor,threshold,prompt_id,prompt_context,repetition,seed,batch_size,sampling_settings,"
           "token_limit,output_tokens,cycles,accepted_tokens,proposed_tokens,output_hash,"
           "total_wall_ns,cycle_ns_total,cycle_p50_ns,cycle_p90_ns,cycle_p95_ns,cycle_p99_ns,"
           "draft_rows,draft_values,draft_callback_ns,target_rows,target_values,target_callback_ns,"
           "tree_build_ns,threshold_select_ns,sample_ns,selected_tree_nodes_total,"
           "mean_selected_depth,max_selected_tree_nodes,max_full_tree_payload_bytes,"
           "max_selected_tree_payload_bytes,mean_selected_tree_payload_bytes,tree_nodes_by_depth,"
           "mean_retained_mass_by_depth,model_storage_bytes,kv_cache_bytes,kv_cache_status\n";
}

void write_row(std::ofstream& out, const std::string& kind,
               std::size_t configured_depth, std::size_t branch_factor,
               double threshold, const Prompt& prompt, std::size_t repetition,
               std::size_t token_limit, const RunStats& stats,
               std::uint64_t model_bytes) {
    out << kind << ',';
    if (kind == "candidate_tree") {
        out << configured_depth << ',' << branch_factor << ',' << std::fixed
            << std::setprecision(4) << threshold;
    } else {
        out << "NA,NA,NA";
    }
    out << ',' << prompt.id << ",\"";
    for (std::size_t i = 0; i < prompt.context.size(); ++i) {
        if (i) out << ' ';
        out << prompt.context[i];
    }
    out << "\"," << repetition << ',' << stats.seed << ",1,categorical_temp1," << token_limit << ','
        << stats.emitted_tokens << ',' << stats.cycles << ',';
    if (kind == "candidate_tree") out << stats.accepted_tokens;
    else out << "NA";
    out << ',' << stats.proposed_tokens << ',' << hex64(stats.hash) << ','
        << stats.total_wall_ns << ',' << stats.cycle_ns_total << ','
        << percentile(stats.cycle_latencies_ns, 0.50) << ','
        << percentile(stats.cycle_latencies_ns, 0.90) << ','
        << percentile(stats.cycle_latencies_ns, 0.95) << ','
        << percentile(stats.cycle_latencies_ns, 0.99) << ','
        << stats.draft.calls << ',' << stats.draft.values << ',' << stats.draft.nanoseconds << ','
        << stats.target.calls << ',' << stats.target.values << ',' << stats.target.nanoseconds << ','
        << stats.tree_build_ns << ',' << stats.threshold_select_ns << ',' << stats.sample_ns << ','
        << stats.selected_tree_nodes_total << ',';
    if (stats.cycles) out << std::setprecision(6) << stats.selected_depth_total / stats.cycles;
    else out << "NA";
    out << ',' << stats.max_selected_tree_nodes << ',' << stats.max_full_tree_bytes << ','
        << stats.max_selected_tree_bytes << ',';
    if (stats.cycles && kind == "candidate_tree") {
        out << static_cast<double>(stats.selected_tree_bytes_total) / stats.cycles;
    } else {
        out << "NA";
    }
    out << ",\"" << join_values(stats.nodes_by_depth_total) << "\",\"";
    for (std::size_t i = 0; i < stats.retained_mass_total.size(); ++i) {
        if (i) out << ';';
        const long double mean = stats.retained_mass_count[i]
            ? stats.retained_mass_total[i] / stats.retained_mass_count[i] : 0.0L;
        out << std::setprecision(8) << static_cast<double>(mean);
    }
    out << "\"," << model_bytes << ",,not_applicable_toy_markov\n";
}

struct Options {
    std::string model_path = "inputs/model.bin";
    std::string raw_path = "stage4_raw.csv";
    std::size_t tokens = 512;
    std::size_t repetitions = 3;
};

std::size_t parse_size(const char* value, const char* name) {
    const std::string text(value);
    std::size_t consumed = 0;
    const unsigned long long parsed = std::stoull(text, &consumed);
    if (consumed != text.size() || parsed == 0 ||
        parsed > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument(std::string(name) + " must be a positive integer");
    }
    return static_cast<std::size_t>(parsed);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--model" && i + 1 < argc) options.model_path = argv[++i];
        else if (arg == "--raw" && i + 1 < argc) options.raw_path = argv[++i];
        else if (arg == "--tokens" && i + 1 < argc) options.tokens = parse_size(argv[++i], "--tokens");
        else if (arg == "--repeats" && i + 1 < argc) options.repetitions = parse_size(argv[++i], "--repeats");
        else throw std::invalid_argument("usage: pulsar_stage4_bench [--model path] [--raw path] [--tokens N] [--repeats N]");
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        pulsar::Model model;
        if (!pulsar::loadModel(model, options.model_path.c_str())) {
            throw std::runtime_error("failed to load the existing toy model fixture");
        }
        const std::vector<Prompt> prompts{
            {"canonical_5_3_9", Context{5, 3, 9}},
            {"alternate_1_4_7", Context{1, 4, 7}},
            {"zero_0_0_0", Context{0, 0, 0}},
        };
        const std::array<std::size_t, 3> depths{1, 2, 4};
        const std::array<std::size_t, 3> branches{1, 2, 4};
        const std::array<double, 3> thresholds{0.0, 0.20, 0.40};
        const std::uint64_t model_bytes = model_storage_bytes(model);

        std::ofstream out(options.raw_path);
        if (!out) throw std::runtime_error("cannot create raw CSV: " + options.raw_path);
        write_header(out);
        out << std::setprecision(9);

        std::size_t row_count = 0;
        for (std::size_t prompt_index = 0; prompt_index < prompts.size(); ++prompt_index) {
            for (std::size_t repetition = 0; repetition < options.repetitions; ++repetition) {
                const std::uint64_t seed = make_seed(prompt_index, repetition);
                const RunStats ar = run_ar(model, prompts[prompt_index], options.tokens, seed);
                write_row(out, "baseline_ar", 0, 0, 0.0, prompts[prompt_index],
                          repetition, options.tokens, ar, model_bytes);
                ++row_count;

                for (std::size_t depth : depths) {
                    for (std::size_t branch : branches) {
                        for (double threshold : thresholds) {
                            const RunStats candidate = run_tree(
                                model, prompts[prompt_index], options.tokens, seed,
                                depth, branch, threshold);
                            write_row(out, "candidate_tree", depth, branch, threshold,
                                      prompts[prompt_index], repetition, options.tokens,
                                      candidate, model_bytes);
                            ++row_count;
                        }
                    }
                }
            }
        }
        out.close();
        std::cout << "Stage 4 measurements: " << row_count << " raw runs written to "
                  << options.raw_path << "\n"
                  << "Fixture: existing synthetic Markov model (V=" << model.p.V
                  << "), prompts=3, batch_size=1, tokens/run=" << options.tokens
                  << ", repeats=" << options.repetitions << "\n"
                  << "No transformer runtime or KV cache is present in this benchmark.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Stage 4 benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
