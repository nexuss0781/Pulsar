#include "pulsar_stage4_bench.hpp"

#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void test_threshold_trims_only_whole_levels() {
    using pulsar::v3::Context;
    const auto make_tree = [] {
        return pulsar::v3::build_candidate_tree(
            Context{0}, 1, 4,
            [](const Context&) { return pulsar::v3::Weights{0.7, 0.2, 0.1}; });
    };

    auto tree = make_tree();
    const auto selected = pulsar::stage4::select_by_minimum_level_mass(tree, 0.5);
    require(selected.selected_depth == 2 && tree.max_depth == 2,
            "0.5 threshold must keep root plus the first retained level");
    require(tree.nodes.size() == 2 && tree.nodes.front().children.size() == 1,
            "threshold pruning must leave a valid complete level");
    require(selected.retained_mass_by_depth.size() == 3 &&
                std::abs(selected.retained_mass_by_depth[1] - 0.7) < 1e-12,
            "retained mass must be the normalized q path mass");

    auto lower = make_tree();
    const auto lower_selected =
        pulsar::stage4::select_by_minimum_level_mass(lower, 0.4);
    require(lower_selected.selected_depth == 3 && lower.nodes.size() == 3,
            "0.4 threshold must retain the 0.7 and 0.49 mass levels");

    std::mt19937_64 rng(1);
    const auto sampled = pulsar::v3::sample_candidate_tree(
        tree, Context{0}, [](const Context&) {
            return pulsar::v3::Weights{0.7, 0.2, 0.1};
        }, rng);
    require(!sampled.emitted_tokens.empty() && sampled.emitted_tokens.size() <= 2,
            "threshold-selected tree must remain compatible with the Stage 3 sampler");
}

void test_zero_threshold_and_invalid_threshold() {
    using pulsar::v3::Context;
    auto tree = pulsar::v3::build_candidate_tree(
        Context{1}, 2, 3,
        [](const Context&) { return pulsar::v3::Weights{0.6, 0.3, 0.1}; });
    const auto selected = pulsar::stage4::select_by_minimum_level_mass(tree, 0.0);
    require(selected.selected_depth == 3 && tree.max_depth == 3,
            "zero threshold must preserve configured depth");

    bool rejected = false;
    try {
        auto invalid = pulsar::v3::build_candidate_tree(
            Context{}, 1, 1,
            [](const Context&) { return pulsar::v3::Weights{1.0}; });
        (void)pulsar::stage4::select_by_minimum_level_mass(invalid, 1.1);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "threshold above one must be rejected");
}

}  // namespace

int main() {
    try {
        test_threshold_trims_only_whole_levels();
        test_zero_threshold_and_invalid_threshold();
        std::cout << "Stage 4 threshold selector tests [PASS]\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Stage 4 threshold selector tests [FAIL]: " << error.what() << '\n';
        return 1;
    }
}
