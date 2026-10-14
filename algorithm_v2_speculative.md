# Pulsar Stage 2 — speculative-chain contract

This document extends the frozen v1 contract in [`algorithm_v1.md`](algorithm_v1.md) without changing it. The implementation is an isolated C++17 layer in `cpp/pulsar_speculative_v2.hpp`.

## Laws and proposal generation

At context `c`, `p(·|c)` is the same processed target next-token law used by the plain autoregressive reference. A block of length `γ` is drafted sequentially: `yᵢ ~ q(·|cᵢ)`, where `cᵢ` is the initial context followed by the preceding draft proposals. Draft rows may be context-dependent. All `γ` proposals are drafted before verification begins; later proposals are still drawn even if an earlier candidate will subsequently be rejected.

For this speculative chain, the verifier's effective proposal law is the exact `q` row that generated the candidate. This is the `π=q` case of the frozen v1 verifier, so candidate `yᵢ` is accepted with probability `min(1, p(yᵢ|cᵢ)/q(yᵢ|cᵢ))`. On rejection, emit a token from the normalized residual `max(p(·|cᵢ)-q(·|cᵢ),0)` and stop the block. The residual is only sampled on rejection. Do not apply a distinct collapse processor to the verifier without explicitly using that same effective law for both proposal generation and verification.

## Left-to-right commit and bonus

Target rows `p(·|cᵢ)` for the drafted proposal contexts are obtained after drafting and verified left-to-right. Every accepted proposal is emitted unchanged. At the first rejection, emit its residual replacement and stop; do not verify or emit later candidates. If every proposal is accepted, an optional cached target row at the final context `c₍γ₎` may supply one bonus token. The bonus row is an explicit optional input/cache lookup; the block API does not require an extra target-model evaluation. If it is unavailable, the block emits only the `γ` accepted proposals.

The decoder's output limit is in the same token IDs/tokenization as the AR reference. A caller must not append a bonus past the remaining output-token budget. On rejection, the block emits exactly the accepted prefix plus its replacement; generation may then continue in a later block. The plain AR reference makes one target categorical draw per requested token and uses the same fixed output-token limit.

## Numerical and API assumptions

Rows are finite, non-negative categorical weights with positive total mass and a common vocabulary, matching v1. `std::discrete_distribution` and the v1 uniform acceptance convention are reused where practical. The block driver callbacks return owning `Weights`; `draft_row(context)` gives `q`, `target_row(context)` gives `p`, and `cached_bonus_row(final_context)` must return `nullopt` unless that row is already available. The low-level block verifier accepts scripted proposal tokens and aligned `q`/`p` rows for deterministic tests.

## Stage 2 exit criteria

1. Unit tests force the first rejection at every position in a block and check residual replacement plus stop behavior.
2. An all-accepted block emits a bonus only when a cached target row is supplied; rejection never emits a bonus.
3. Conditional next-token laws are checked by context against `p` and against a plain AR reference.
4. Sequence-level comparison uses separate independent seeds for the speculative and AR runs and compares conditional transition counts, rather than relying on unigram counts from a shared correlated random stream.
