#include "pulsar_engine.hpp"
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
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using Context = pulsar::v3::Context;
using Token = pulsar::v3::Token;
using Weights = pulsar::v3::Weights;

struct Action {
    std::size_t depth = 1;
    std::size_t branch = 1;
    double threshold = 0.0;
};

struct Prompt {
    std::string id;
    Context context;
};
struct PulseTraceRow {
    std::size_t context_state = 0;
    std::size_t depth = 0;
    std::size_t branch = 0;
    double threshold = 0.0;
    std::uint64_t emitted_tokens = 0;
    std::uint64_t elapsed_ns = 0;
};
struct RunResult {
    std::uint64_t tokens = 0;
    std::uint64_t accepted = 0;
    std::uint64_t cycles = 0;
    std::uint64_t wall_ns = 0;
    std::array<std::uint64_t, 5> decisions{};
};

struct PulseResult {
    std::vector<Token> emitted;
    std::vector<int> verifier_labels;
    std::size_t accepted = 0;
    std::size_t used_depth = 0;
};

std::size_t parse_size(const std::string& value, const char* name) {
    std::size_t consumed = 0;
    const unsigned long long parsed = std::stoull(value, &consumed);
    if (consumed != value.size() || parsed > static_cast<unsigned long long>(SIZE_MAX)) {
        throw std::invalid_argument(std::string("invalid ") + name);
    }
    return static_cast<std::size_t>(parsed);
}

std::vector<std::string> split(const std::string& text, char separator) {
    std::vector<std::string> parts;
    std::size_t begin = 0;
    while (begin <= text.size()) {
        const std::size_t end = text.find(separator, begin);
        parts.push_back(text.substr(begin, end == std::string::npos ? end : end - begin));
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return parts;
}

std::vector<Action> parse_actions(const std::string& text) {
    std::vector<Action> result;
    for (const std::string& item : split(text, ';')) {
        const auto fields = split(item, ',');
        if (fields.size() != 3) throw std::invalid_argument("actions must be depth,branch,threshold triples");
        Action action{parse_size(fields[0], "action depth"),
                      parse_size(fields[1], "action branch"), std::stod(fields[2])};
        if (action.depth == 0 || action.branch > 1000000 ||
            !std::isfinite(action.threshold) || action.threshold < 0.0 || action.threshold > 1.0) {
            throw std::invalid_argument("invalid action");
        }
        result.push_back(action);
    }
    if (result.empty()) throw std::invalid_argument("at least one action is required");
    return result;
}

Context prompt_context(const std::string& id) {
    if (id == "canonical_5_3_9") return {5, 3, 9};
    if (id == "alternate_1_4_7") return {1, 4, 7};
    if (id == "zero_0_0_0") return {0, 0, 0};
    throw std::invalid_argument("unknown prompt id: " + id);
}

int context3_index(const Context& context, int vocabulary) {
    const std::size_t n = context.size();
    const int a = n >= 3 ? static_cast<int>(context[n - 3]) : 0;
    const int b = n >= 2 ? static_cast<int>(context[n - 2]) : 0;
    const int c = n >= 1 ? static_cast<int>(context[n - 1]) : 0;
    return (a * vocabulary + b) * vocabulary + c;
}

int context2_index(int context3, int vocabulary) {
    return context3 % (vocabulary * vocabulary);
}

using TopLists = std::vector<std::vector<Token>>;

TopLists build_top_lists(const pulsar::Model& model) {
    const int vocabulary = model.p.V;
    const std::size_t rows = static_cast<std::size_t>(vocabulary) * vocabulary;
    TopLists result(rows);
    for (std::size_t row = 0; row < rows; ++row) {
        const double* q = model.qrow(static_cast<int>(row));
        std::vector<Token> tokens(static_cast<std::size_t>(vocabulary));
        for (int token = 0; token < vocabulary; ++token) {
            tokens[static_cast<std::size_t>(token)] = static_cast<Token>(token);
        }
        std::stable_sort(tokens.begin(), tokens.end(), [q](Token left, Token right) {
            if (q[left] != q[right]) return q[left] > q[right];
            return left < right;
        });
        tokens.erase(std::remove_if(tokens.begin(), tokens.end(), [q](Token token) {
            return !(q[token] > 0.0);
        }), tokens.end());
        result[row] = std::move(tokens);
    }
    return result;
}

bool has_retained_child(const std::vector<Token>& sorted_tokens,
                        std::size_t branch, Token proposal) {
    const std::size_t retained_count = std::min(branch, sorted_tokens.size());
    const auto begin = sorted_tokens.begin();
    const auto end = begin + static_cast<std::ptrdiff_t>(retained_count);
    return std::find(begin, end, proposal) != end;
}

PulseResult run_lazy_pulse(const pulsar::Model& model, const TopLists& top,
                           Context& context, std::mt19937_64& rng,
                           const Action& action) {
    PulseResult result;
    const int vocabulary = model.p.V;
    const std::size_t requested_depth = action.depth;
    result.used_depth = requested_depth;
    result.emitted.reserve(requested_depth);
    int state = context3_index(context, vocabulary);
    for (std::size_t depth = 0; depth < requested_depth; ++depth) {
        const int q_index = context2_index(state, vocabulary);
        const pulsar::v1::DistributionView target(
            model.prow(state), static_cast<std::size_t>(vocabulary));
        const pulsar::v1::DistributionView proposal(
            model.qrow(q_index), static_cast<std::size_t>(vocabulary));
        const auto checked = pulsar::v1::sample_verified_token(target, proposal, rng);
        context.push_back(static_cast<int>(checked.emitted_token));
        result.emitted.push_back(checked.emitted_token);
        result.verifier_labels.push_back(checked.accepted ? 1 : 0);
        if (!checked.accepted) return result;
        ++result.accepted;
        if (depth + 1 == requested_depth ||
            !has_retained_child(top[static_cast<std::size_t>(q_index)],
                                action.branch, checked.proposal_token)) {
            return result;
        }
        state = q_index * vocabulary + static_cast<int>(checked.proposal_token);
    }
    return result;
}

PulseResult run_eager_threshold_pulse(const pulsar::Model& model,
                                      Context& context, std::mt19937_64& rng,
                                      const Action& action) {
    const int vocabulary = model.p.V;
    auto draft_row = [&model, vocabulary](const Context& ctx) {
        const int index = context3_index(ctx, vocabulary);
        return Weights(model.qrow(context2_index(index, vocabulary)),
                       model.qrow(context2_index(index, vocabulary)) + vocabulary);
    };
    auto target_row = [&model, vocabulary](const Context& ctx) {
        const int index = context3_index(ctx, vocabulary);
        return Weights(model.prow(index), model.prow(index) + vocabulary);
    };
    pulsar::v3::CandidateTree tree = pulsar::v3::build_candidate_tree(
        context, action.branch, action.depth, draft_row);
    const auto selection = pulsar::stage4::select_by_minimum_level_mass(
        tree, action.threshold);
    const auto sampled = pulsar::v3::sample_candidate_tree(
        tree, context, target_row, rng);
    PulseResult result;
    result.emitted = sampled.emitted_tokens;
    result.accepted = sampled.accepted_count;
    result.used_depth = selection.selected_depth;
    result.verifier_labels.assign(result.accepted, 1);
    if (sampled.rejected_position.has_value()) result.verifier_labels.push_back(0);
    for (Token token : result.emitted) context.push_back(static_cast<int>(token));
    return result;
}

RunResult run_stream(const pulsar::Model& model, const TopLists& top,
                     Context context, std::uint64_t seed,
                     std::size_t token_limit, const Action& fixed_action,
                     const std::vector<Action>& actions,
                     const std::vector<std::uint8_t>& action_lookup,
                     const std::string& online_policy,
                     std::vector<PulseTraceRow>* trace_rows = nullptr) {
    RunResult result;
    std::mt19937_64 rng(seed);
    const int vocabulary = model.p.V;
    const std::array<Action, 3> online_actions{{{1, 4, 0.0}, {2, 4, 0.0}, {4, 4, 0.0}}};
    double ema = 0.5;
    std::array<std::uint64_t, 3> bandit_counts{};
    std::array<double, 3> bandit_reward_sums{};
    std::uint64_t bandit_updates = 0;
    const auto v3_size = static_cast<std::size_t>(vocabulary) * vocabulary * vocabulary;
    if (!online_policy.empty() && online_policy != "ema" && online_policy != "ucb1") {
        throw std::invalid_argument("unknown online policy");
    }
    if (!online_policy.empty() && !action_lookup.empty()) {
        throw std::invalid_argument("online and compiled-map modes are mutually exclusive");
    }
    if (!action_lookup.empty() && action_lookup.size() != v3_size) {
        throw std::invalid_argument("compiled action lookup has the wrong length");
    }
    while (result.tokens < token_limit) {
        const std::size_t trace_state = trace_rows == nullptr ? 0U :
            static_cast<std::size_t>(context3_index(context, vocabulary));
        const auto cycle_start = Clock::now();
        Action action = fixed_action;
        std::size_t bandit_action_index = 0;
        if (!action_lookup.empty()) {
            const std::size_t state = static_cast<std::size_t>(context3_index(context, vocabulary));
            const std::uint8_t action_index = action_lookup[state];
            if (action_index >= actions.size()) {
                throw std::invalid_argument("compiled action lookup has an invalid action index");
            }
            action = actions[action_index];
        } else if (online_policy == "ema") {
            const std::size_t index = static_cast<std::size_t>(ema >= 0.60) +
                                      static_cast<std::size_t>(ema >= 0.80);
            action = online_actions[index];
        } else if (online_policy == "ucb1") {
            bool unobserved = false;
            for (std::size_t i = 0; i < bandit_counts.size(); ++i) {
                if (bandit_counts[i] == 0) {
                    bandit_action_index = i;
                    unobserved = true;
                    break;
                }
            }
            if (!unobserved) {
                double best_score = -std::numeric_limits<double>::infinity();
                for (std::size_t i = 0; i < bandit_counts.size(); ++i) {
                    const double mean = bandit_reward_sums[i] /
                                        static_cast<double>(bandit_counts[i]);
                    const double bonus = std::sqrt(
                        2.0 * std::log(static_cast<double>(std::max<std::uint64_t>(bandit_updates, 1))) /
                        static_cast<double>(bandit_counts[i]));
                    if (mean + bonus > best_score) {
                        best_score = mean + bonus;
                        bandit_action_index = i;
                    }
                }
            }
            action = online_actions[bandit_action_index];
        }
        action.depth = std::min(action.depth,
                                token_limit - static_cast<std::size_t>(result.tokens));
        if (action.depth == 0) throw std::logic_error("zero-depth pulse is not allowed");
        const Action configured_action = action;
        PulseResult pulse = action.threshold == 0.0
            ? run_lazy_pulse(model, top, context, rng, action)
            : run_eager_threshold_pulse(model, context, rng, action);
        if (pulse.emitted.empty() || pulse.emitted.size() > token_limit - result.tokens) {
            throw std::runtime_error("Stage 5 pulse emitted an invalid token count");
        }
        result.tokens += pulse.emitted.size();
        result.accepted += pulse.accepted;
        ++result.cycles;
        if (pulse.used_depth < result.decisions.size()) ++result.decisions[pulse.used_depth];
        const std::uint64_t elapsed = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - cycle_start).count());
        result.wall_ns += elapsed;
        if (trace_rows != nullptr) {
            trace_rows->push_back({trace_state, configured_action.depth,
                                   configured_action.branch, configured_action.threshold,
                                   static_cast<std::uint64_t>(pulse.emitted.size()), elapsed});
        }
        if (online_policy == "ema") {
            for (int label : pulse.verifier_labels) {
                ema = 0.8 * ema + 0.2 * static_cast<double>(label);
            }
        } else if (online_policy == "ucb1") {
            const double reward = std::min(
                static_cast<double>(pulse.emitted.size()) * 1e9 /
                    static_cast<double>(std::max<std::uint64_t>(elapsed, 1)) / 100000.0,
                1.0);
            ++bandit_counts[bandit_action_index];
            bandit_reward_sums[bandit_action_index] += reward;
            ++bandit_updates;
        }
    }
    return result;
}

PulseResult eager_zero_threshold_reference(const pulsar::Model& model,
                                           Context context, std::mt19937_64& rng,
                                           const Action& action) {
    const int vocabulary = model.p.V;
    auto draft_row = [&model, vocabulary](const Context& ctx) {
        const int index = context3_index(ctx, vocabulary);
        const double* q = model.qrow(context2_index(index, vocabulary));
        return Weights(q, q + vocabulary);
    };
    auto target_row = [&model, vocabulary](const Context& ctx) {
        const int index = context3_index(ctx, vocabulary);
        const double* p = model.prow(index);
        return Weights(p, p + vocabulary);
    };
    auto tree = pulsar::v3::build_candidate_tree(context, action.branch,
                                                  action.depth, draft_row);
    pulsar::stage4::select_by_minimum_level_mass(tree, 0.0);
    const auto sampled = pulsar::v3::sample_candidate_tree(tree, context,
                                                           target_row, rng);
    PulseResult result{sampled.emitted_tokens, {}, sampled.accepted_count, action.depth};
    result.verifier_labels.assign(result.accepted, 1);
    if (sampled.rejected_position.has_value()) result.verifier_labels.push_back(0);
    return result;
}

void run_self_test(const pulsar::Model& model, const TopLists& top) {
    const std::array<Context, 3> prompts{{Context{5, 3, 9}, Context{1, 4, 7}, Context{0, 0, 0}}};
    std::uint64_t checks = 0;
    for (const Context& prompt : prompts) {
        for (std::size_t depth : {1U, 2U, 4U}) {
            for (std::size_t branch : {1U, 2U, 4U}) {
                const Action action{depth, branch, 0.0};
                for (std::uint64_t seed = 1; seed <= 64; ++seed) {
                    std::mt19937_64 lazy_rng(seed), eager_rng(seed);
                    Context lazy_context = prompt;
                    const PulseResult lazy = run_lazy_pulse(model, top, lazy_context,
                                                           lazy_rng, action);
                    const PulseResult eager = eager_zero_threshold_reference(
                        model, prompt, eager_rng, action);
                    if (lazy.emitted != eager.emitted || lazy.accepted != eager.accepted ||
                        lazy.used_depth != eager.used_depth) {
                        throw std::runtime_error("lazy/eager Stage 3 output parity failed");
                    }
                    ++checks;
                }
            }
        }
    }
    std::cout << "Stage 5 native exact-path parity: " << checks << "/" << checks
              << " PASS\n";
}

struct Options {
    std::string model_path;
    std::string prompt_id;
    std::string lookup_path;
    std::string actions_text;
    std::size_t token_limit = 0;
    std::uint64_t seed = 0;
    Action fixed_action{};
    bool fixed_set = false;
    bool self_test = false;
    std::string online_policy;
    std::string trace_path;
};

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument("missing value after " + arg);
            return argv[++i];
        };
        if (arg == "--model") options.model_path = value();
        else if (arg == "--prompt") options.prompt_id = value();
        else if (arg == "--tokens") options.token_limit = parse_size(value(), "token limit");
        else if (arg == "--seed") options.seed = static_cast<std::uint64_t>(std::stoull(value()));
        else if (arg == "--lookup") options.lookup_path = value();
        else if (arg == "--actions") options.actions_text = value();
        else if (arg == "--online") options.online_policy = value();
        else if (arg == "--trace") options.trace_path = value();
        else if (arg == "--fixed-depth") { options.fixed_action.depth = parse_size(value(), "fixed depth"); options.fixed_set = true; }
        else if (arg == "--fixed-branch") { options.fixed_action.branch = parse_size(value(), "fixed branch"); options.fixed_set = true; }
        else if (arg == "--fixed-threshold") { options.fixed_action.threshold = std::stod(value()); options.fixed_set = true; }
        else if (arg == "--self-test") options.self_test = true;
        else throw std::invalid_argument("unknown option: " + arg);
    }
    if (options.model_path.empty()) throw std::invalid_argument("--model is required");
    if (options.self_test) return options;
    if (options.token_limit == 0 || options.prompt_id.empty()) {
        throw std::invalid_argument("--tokens and --prompt are required");
    }
    if (!options.online_policy.empty()) {
        if ((options.online_policy != "ema" && options.online_policy != "ucb1") ||
            !options.lookup_path.empty() || options.fixed_set) {
            throw std::invalid_argument("online mode requires --online ema|ucb1 without other policy flags");
        }
    } else if (!options.lookup_path.empty()) {
        if (options.actions_text.empty() || options.fixed_set) {
            throw std::invalid_argument("compiled-map mode requires --actions and no --fixed-* options");
        }
    } else if (!options.fixed_set) {
        throw std::invalid_argument("provide either --fixed-* or --lookup");
    }
    return options;
}

std::vector<std::uint8_t> read_lookup(const std::string& path, std::size_t expected_size) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open action lookup: " + path);
    std::vector<std::uint8_t> result(expected_size);
    input.read(reinterpret_cast<char*>(result.data()), static_cast<std::streamsize>(result.size()));
    if (input.gcount() != static_cast<std::streamsize>(result.size()) || input.peek() != EOF) {
        throw std::runtime_error("action lookup has unexpected length");
    }
    return result;
}

void write_trace(const std::string& path, const std::vector<PulseTraceRow>& rows) {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("cannot open pulse trace: " + path);
    output << "context3,depth,branch,threshold,emitted_tokens,elapsed_ns\n";
    output << std::setprecision(17);
    for (const PulseTraceRow& row : rows) {
        output << row.context_state << ',' << row.depth << ',' << row.branch << ','
               << row.threshold << ',' << row.emitted_tokens << ',' << row.elapsed_ns << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        pulsar::Model model;
        if (!pulsar::loadModel(model, options.model_path.c_str())) {
            throw std::runtime_error("failed to load Stage 5 model fixture");
        }
        const TopLists top = build_top_lists(model);
        if (options.self_test) {
            run_self_test(model, top);
            return 0;
        }
        std::vector<Action> actions;
        std::vector<std::uint8_t> lookup;
        if (!options.lookup_path.empty()) {
            actions = parse_actions(options.actions_text);
            const std::size_t size = static_cast<std::size_t>(model.p.V) * model.p.V * model.p.V;
            lookup = read_lookup(options.lookup_path, size);
        }
        std::vector<PulseTraceRow> trace_rows;
        const RunResult result = run_stream(model, top, prompt_context(options.prompt_id),
                                            options.seed, options.token_limit,
                                            options.fixed_action, actions, lookup,
                                            options.online_policy,
                                            options.trace_path.empty() ? nullptr : &trace_rows);
        if (!options.trace_path.empty()) write_trace(options.trace_path, trace_rows);
        std::cout << result.tokens << ',' << result.accepted << ',' << result.cycles << ','
                  << result.wall_ns << ',' << result.decisions[1] << ','
                  << result.decisions[2] << ',' << result.decisions[4] << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Stage 5 native runner failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
