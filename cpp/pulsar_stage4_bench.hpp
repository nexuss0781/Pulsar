// Stage 4 benchmark-only fixed threshold policy. Stage 3 sampling contracts remain unchanged.
#pragma once

#include "pulsar_tree_v3.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pulsar::stage4 {

struct ThresholdSelection {
    // max_depth counts verifier positions, including the root position.
    std::size_t selected_depth = 1;
    // Entry 0 is the root mass (1); later entries are retained path mass by level.
    std::vector<double> retained_mass_by_depth;
};

// The threshold is a minimum total q-probability mass at a whole tree level.
// If adding the next level would leave less than this much mass in retained
// prefixes, selection stops before that level. This is a benchmark policy, not
// the trainable pulse rule or a global-optimality claim.
inline ThresholdSelection select_by_minimum_level_mass(
    pulsar::v3::CandidateTree& tree, double minimum_level_mass) {
    if (!std::isfinite(minimum_level_mass) || minimum_level_mass < 0.0 ||
        minimum_level_mass > 1.0) {
        throw std::invalid_argument("minimum level mass must be finite and in [0, 1]");
    }
    if (tree.nodes.empty() || tree.max_depth == 0) {
        throw std::invalid_argument("threshold selection requires a non-empty tree");
    }

    ThresholdSelection result;
    result.selected_depth = 1;
    result.retained_mass_by_depth.push_back(1.0);
    std::vector<long double> reach(tree.nodes.size(), 0.0L);
    reach.front() = 1.0L;

    for (std::size_t depth = 1; depth < tree.max_depth; ++depth) {
        long double level_mass = 0.0L;
        for (std::size_t parent_index = 0; parent_index < tree.nodes.size();
             ++parent_index) {
            const auto& parent = tree.nodes[parent_index];
            if (parent.depth + 1 != depth || parent.children.empty()) {
                continue;
            }
            const long double row_mass = pulsar::v1::detail::total_mass(
                pulsar::v1::DistributionView(parent.q));
            for (std::size_t child_index : parent.children) {
                if (child_index >= tree.nodes.size()) {
                    throw std::invalid_argument("tree has an invalid child index");
                }
                const auto& child = tree.nodes[child_index];
                if (child.depth != depth || child.incoming_token >= parent.q.size()) {
                    throw std::invalid_argument("tree child metadata is inconsistent");
                }
                const long double child_mass =
                    reach[parent_index] *
                    static_cast<long double>(parent.q[child.incoming_token]) / row_mass;
                reach[child_index] = child_mass;
                level_mass += child_mass;
            }
        }
        result.retained_mass_by_depth.push_back(static_cast<double>(level_mass));
        if (level_mass < static_cast<long double>(minimum_level_mass)) {
            result.selected_depth = depth;
            break;
        }
        result.selected_depth = depth + 1;
    }

    if (result.selected_depth < tree.max_depth) {
        std::size_t retained_count = 0;
        while (retained_count < tree.nodes.size() &&
               tree.nodes[retained_count].depth < result.selected_depth) {
            ++retained_count;
        }
        std::vector<pulsar::v3::CandidateTreeNode> retained;
        retained.reserve(retained_count);
        for (std::size_t i = 0; i < retained_count; ++i) {
            auto node = std::move(tree.nodes[i]);
            std::vector<std::size_t> retained_children;
            retained_children.reserve(node.children.size());
            for (std::size_t child : node.children) {
                if (child < retained_count) {
                    retained_children.push_back(child);
                }
            }
            node.children = std::move(retained_children);
            retained.push_back(std::move(node));
        }
        tree.nodes = std::move(retained);
        tree.max_depth = result.selected_depth;
    }
    return result;
}

}  // namespace pulsar::stage4
