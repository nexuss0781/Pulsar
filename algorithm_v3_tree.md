# Pulsar Stage 3 — bounded candidate tree and fall-through

Stage 3 adds an isolated C++17 tree sampler in `cpp/pulsar_tree_v3.hpp`. It builds on the frozen v1 verifier and leaves the Stage 2 speculative-chain layer unchanged.

## Tree construction

At each tree node with context `c`, the builder obtains and stores the **complete** draft row `q(·|c)`. At non-terminal positions it retains up to a fixed branch factor `B` of positive-probability `q` children, ordered by descending weight with token-ID tie breaking. Zero-weight entries are not retained as draft children. Child contexts are exactly the parent context followed by the child's token.

`max_depth` is the fixed maximum number of verifier positions in a pulse, including its root position. Thus depth 1 samples once at the root; no pulse can emit more than `max_depth` tokens. A `B=0` tree is valid and simply has no continuation children. Rows must be finite, non-negative categorical weights with positive total mass, and all rows share one vocabulary.

## Sampling and fall-through

The full stored `q` row is available at every node. By default it is also the effective proposal `π`. A caller may instead provide an explicit context-dependent `π` row; it may differ from `q`. In either case the sampler draws from `π` and passes that same law to the v1 exact verifier with target row `p`.

- On rejection, emit one draw from the verifier residual `max(p−π, 0)` and stop the pulse.
- On acceptance, emit the sampled token unchanged. Descend only if that exact token has a retained child.
- If an accepted token is not represented by a retained child, it is still emitted unchanged and continuation stops. The sampler never renormalizes `π` over retained children, filters its support, or substitutes a `q` child.
- At the configured depth limit, stop after the accepted token. Stage 3 does not add a bonus token.

Tree children are a bounded continuation budget, not the token eligibility set. The emitted next-token law at every visited context remains `p` by the v1 identity, regardless of `B`, depth, or the relationship between `q` and `π`.

## Context and integrity checks

The initial context must exactly match the tree root. Before descent, the sampler checks the current node's stored context/depth, recomputes the retained positive-`q` child set, and checks each traversed child's index, parent, incoming token, depth, and appended context. A stale context or malformed/non-child link raises `std::invalid_argument`; it cannot silently continue along a different path.

## Stage 3 exit criteria

1. Fixed `B` and depth bound generated rows and emitted tokens; full `q` rows remain stored at every node.
2. Deterministic fixtures check `π(x)a(x) + Zr(x) = p(x)` for the effective proposal, including zero support and equality cases.
3. An accepted token outside the retained tree is emitted unchanged, then stops; rejection emits the residual and stops.
4. Stale roots and stale child contexts are rejected before continuation.
5. Independent-seed conditional-transition tests compare tree mode with `p` and a plain autoregressive reference across contexts, using `π != q` and a pruned `B=1` tree.

Reproduce the C++ suite with:

```bash
cd cpp && make clean && make CXXFLAGS='-O2 -std=c++17 -Wall -Wextra -Wpedantic -Werror' test
```
