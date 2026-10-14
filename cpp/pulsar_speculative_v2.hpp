// Pulsar Stage 2: exact speculative-chain decoding, layered on the frozen v1
// categorical verifier. Draft proposals use q, and the verifier therefore uses
// that same q as its effective proposal law (the p/q rejection rule).
#pragma once

#include "pulsar_core_v1.hpp"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pulsar::v2 {

using Token = v1::Token;
using Weights = v1::Weights;
using Context = std::vector<Token>;

struct SpeculativeBlockResult {
    // All gamma proposals are drafted before verification starts.
    std::vector<Token> proposed_tokens;
    // Accepted proposals, followed by a residual replacement on rejection;
    // if all proposals pass, an optional target bonus token may follow.
    std::vector<Token> emitted_tokens;
    std::size_t accepted_count = 0;
    std::optional<std::size_t> rejected_position;
    bool bonus_emitted = false;
};

// Verify a proposal that has already been drawn from q. The q law is used both
// as the proposal distribution and as the denominator in the acceptance test.
// This is intentionally distinct from v1::sample_verified_token's more general
// API, where the caller may specify an effective pi different from draft q.
template <class URBG>
inline v1::VerifiedToken verify_proposed_token(v1::DistributionView target,
                                               v1::DistributionView q,
                                               Token proposal,
                                               URBG& rng) {
    if (target.size != q.size) {
        throw std::invalid_argument("target and draft vocabulary sizes differ");
    }
    const long double target_mass = v1::detail::total_mass(target);
    const long double q_mass = v1::detail::total_mass(q);
    if (proposal >= q.size) {
        throw std::invalid_argument("draft proposal token is outside the vocabulary");
    }
    const long double q_x = static_cast<long double>(q.data[proposal]) / q_mass;
    if (!(q_x > 0.0L)) {
        throw std::invalid_argument("draft proposal has zero probability under q");
    }
    const long double p_x = static_cast<long double>(target.data[proposal]) / target_mass;
    const long double accept_probability = std::min(1.0L, p_x / q_x);

    // Match the v1 verifier's random-number convention: draw a uniform even
    // when acceptance is certain, and draw a categorical residual only on reject.
    std::uniform_real_distribution<long double> uniform(0.0L, 1.0L);
    if (uniform(rng) < accept_probability) {
        return {proposal, proposal, true, static_cast<double>(accept_probability)};
    }

    const Weights residual = v1::residual_distribution(target, q);
    const Token emitted = v1::detail::draw(v1::DistributionView(residual), rng);
    return {proposal, emitted, false, static_cast<double>(accept_probability)};
}

// Verify an already-drafted block. Rows i are conditional on the context before
// proposal i, and proposals[i] is assumed to have been drawn from q_rows[i].
// Every row is validated before sampling. On the first rejection, append one
// residual replacement and stop; a bonus row is used only after all gamma pass.
template <class URBG>
inline SpeculativeBlockResult verify_speculative_block(
    const std::vector<Token>& proposals,
    const std::vector<Weights>& q_rows,
    const std::vector<Weights>& target_rows,
    const std::optional<Weights>& cached_bonus_target,
    URBG& rng) {
    if (proposals.size() != q_rows.size() || proposals.size() != target_rows.size()) {
        throw std::invalid_argument("proposal and row counts must match");
    }

    SpeculativeBlockResult result;
    result.proposed_tokens = proposals;
    if (proposals.empty()) {
        if (cached_bonus_target) {
            (void)v1::detail::total_mass(v1::DistributionView(*cached_bonus_target));
            result.emitted_tokens.push_back(
                v1::detail::draw(v1::DistributionView(*cached_bonus_target), rng));
            result.bonus_emitted = true;
        }
        return result;
    }

    const std::size_t vocabulary_size = q_rows.front().size();
    if (vocabulary_size == 0) {
        throw std::invalid_argument("draft rows must have non-empty vocabularies");
    }
    for (std::size_t i = 0; i < proposals.size(); ++i) {
        if (q_rows[i].size() != vocabulary_size || target_rows[i].size() != vocabulary_size) {
            throw std::invalid_argument("all speculative rows must share one vocabulary");
        }
        (void)v1::detail::total_mass(v1::DistributionView(q_rows[i]));
        (void)v1::detail::total_mass(v1::DistributionView(target_rows[i]));
        if (proposals[i] >= vocabulary_size || q_rows[i][proposals[i]] <= 0.0) {
            throw std::invalid_argument("proposal must have positive support under its draft row");
        }
    }

    for (std::size_t i = 0; i < proposals.size(); ++i) {
        const v1::VerifiedToken checked = verify_proposed_token(
            v1::DistributionView(target_rows[i]), v1::DistributionView(q_rows[i]),
            proposals[i], rng);
        if (!checked.accepted) {
            result.rejected_position = i;
            result.emitted_tokens.push_back(checked.emitted_token);
            return result;
        }
        result.emitted_tokens.push_back(checked.emitted_token);
        ++result.accepted_count;
    }

    if (cached_bonus_target) {
        if (cached_bonus_target->size() != vocabulary_size) {
            throw std::invalid_argument("bonus row vocabulary size differs from block rows");
        }
        (void)v1::detail::total_mass(v1::DistributionView(*cached_bonus_target));
        result.emitted_tokens.push_back(
            v1::detail::draw(v1::DistributionView(*cached_bonus_target), rng));
        result.bonus_emitted = true;
    }
    return result;
}

// Context-aware block driver. draft_row(context) returns q(.|context), and
// target_row(context) returns p(.|context). All gamma q proposals are drawn
// sequentially first; only then are the corresponding target rows obtained and
// verified left-to-right. cached_bonus_row(final_context) must be a cache-only
// lookup: return nullopt unless the target row is already available.
template <class DraftRowFn, class TargetRowFn, class CachedBonusRowFn, class URBG>
inline SpeculativeBlockResult sample_speculative_block(
    const Context& initial_context,
    std::size_t gamma,
    DraftRowFn&& draft_row,
    TargetRowFn&& target_row,
    CachedBonusRowFn&& cached_bonus_row,
    URBG& rng) {
    std::vector<Token> proposals;
    std::vector<Weights> q_rows;
    std::vector<Context> proposal_contexts;
    proposals.reserve(gamma);
    q_rows.reserve(gamma);
    proposal_contexts.reserve(gamma);

    Context context = initial_context;
    std::size_t vocabulary_size = 0;
    for (std::size_t i = 0; i < gamma; ++i) {
        Weights q = draft_row(context);
        (void)v1::detail::total_mass(v1::DistributionView(q));
        if (i == 0) {
            vocabulary_size = q.size();
        } else if (q.size() != vocabulary_size) {
            throw std::invalid_argument("draft rows must share one vocabulary");
        }
        proposal_contexts.push_back(context);
        proposals.push_back(v1::detail::draw(v1::DistributionView(q), rng));
        q_rows.push_back(std::move(q));
        context.push_back(proposals.back());
    }

    std::vector<Weights> target_rows;
    target_rows.reserve(gamma);
    for (const Context& proposal_context : proposal_contexts) {
        Weights p = target_row(proposal_context);
        (void)v1::detail::total_mass(v1::DistributionView(p));
        target_rows.push_back(std::move(p));
    }

    SpeculativeBlockResult result = verify_speculative_block(
        proposals, q_rows, target_rows, std::nullopt, rng);
    if (!result.rejected_position) {
        // Consult the cache only after the whole block passes; this callback is
        // not permission to force one more target-model evaluation. For gamma
        // zero, the initial context is the final context.
        std::optional<Weights> bonus = cached_bonus_row(context);
        if (bonus) {
            if (gamma > 0 && bonus->size() != vocabulary_size) {
                throw std::invalid_argument(
                    "bonus row vocabulary size differs from block rows");
            }
            (void)v1::detail::total_mass(v1::DistributionView(*bonus));
            result.emitted_tokens.push_back(
                v1::detail::draw(v1::DistributionView(*bonus), rng));
            result.bonus_emitted = true;
        }
    }
    return result;
}

// Plain autoregressive reference sampler: exactly one target categorical draw
// per requested token, with the same context/token IDs and fixed length limit.
template <class TargetRowFn, class URBG>
inline std::vector<Token> sample_autoregressive(const Context& initial_context,
                                               std::size_t max_new_tokens,
                                               TargetRowFn&& target_row,
                                               URBG& rng) {
    Context context = initial_context;
    std::vector<Token> result;
    result.reserve(max_new_tokens);
    for (std::size_t i = 0; i < max_new_tokens; ++i) {
        Weights p = target_row(context);
        (void)v1::detail::total_mass(v1::DistributionView(p));
        const Token token = v1::detail::draw(v1::DistributionView(p), rng);
        result.push_back(token);
        context.push_back(token);
    }
    return result;
}

}  // namespace pulsar::v2
