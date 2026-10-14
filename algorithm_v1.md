# Pulsar v1 — frozen mathematical contract

This file is the normative contract for the v1 core. The older research notes and optional modules are not part of this contract.

## Core objects

For each context `c` and finite vocabulary `V`, all laws below are categorical distributions over `V`:

- **`p(x | c)` — target law.** This is the actual next-token law to preserve, after every target-side processor that defines user-visible output (including temperature, truncation, penalties, or other sampling transforms). It is not necessarily the unprocessed model softmax.
- **`q(x | c)` — draft proposal law.** This is the draft distribution used to build candidate continuations and their path weights. It does not define the verifier's acceptance test unless it is also the effective collapse law.
- **`π(x | c)` — effective collapse law.** This is the actual distribution from which the collapse policy draws the token presented to the verifier, after all collapse-side sampling processors. The verifier must sample from and evaluate this exact same `π`. With no collapse-side processor, `π = q`; otherwise `π` may differ from `q`.

Implementations may store non-negative weights proportional to these laws, but normalize them consistently before sampling and ratio/residual calculations. `p`, `q`, and `π` must have the same vocabulary and positive total mass.

## Exact one-token verifier

At a context `c`:

1. Draw `x ~ π(· | c)`.
2. Accept and emit `x` with probability `a(x) = min(1, p(x | c) / π(x | c))` (for sampled `x`, `π(x | c) > 0`).
3. On rejection, emit `y ~ r(· | c)`, where
   `r(x | c) = max(p(x | c) − π(x | c), 0) / Z` and
   `Z = Σ_x max(p(x | c) − π(x | c), 0) = TV(p, π)`.
4. Stop this pulse after a rejection.

The accepted mass at each token is `min(p(x), π(x))`; the residual supplies exactly the remainder, so the emitted law is `p`. If `Z = 0`, then `p = π` and rejection has probability zero; the residual sampler is undefined and must not be used as a substitute distribution.

### C++ API and numerical assumptions

The header-only C++17 API is `sample_verified_token(target, effective_pi, rng)`, returning the sampled proposal token, emitted token, acceptance flag, and acceptance probability (the last is diagnostic and rounded to `double`). `DistributionView` accepts non-owning arrays of weights: both arrays must have the same non-zero vocabulary length and finite, non-negative weights with positive total mass. Weights need not sum to one; each vector is normalized internally. Zero-weight entries are allowed, including tokens with `p(x) > 0` and `π(x) = 0`; such tokens are supplied by the residual branch and cannot be proposed. Any standard C++ UniformRandomBitGenerator can be passed by reference. No language-model interface is part of this component.

The residual API reports `Z = Σ max(p−π, 0)` and returns the normalized positive residual only when `Z > 0`; requesting a residual at `Z = 0` throws `std::logic_error`. The verifier itself does not request that residual unless a proposal is rejected, and equal laws therefore never enter the undefined branch. No epsilon cutoff or target-distribution fallback is used: positive differences represented by the working floating-point arithmetic remain in the residual. Input weights are `double`; normalization and acceptance calculations use `long double`, while the normalized residual sampler uses `double` weights, so distinctions below the platform's representable precision cannot be promised. The exactness claim is the mathematical law; finite-precision implementations are checked with deterministic algebra identities and separate statistical tests.

For every token, the deterministic check evaluates the requested identity `π(x)·a(x) + Z·r(x) = p(x)`, where `a(x) = min(1, p(x)/π(x))` for `π(x) > 0`; for `π(x) = 0`, the proposal-side accepted mass is zero and `Z·r(x)` supplies `p(x)`. Tests include identical laws, zero proposal support, concentrated and nearly disjoint laws, small positive residuals, 2,000 reproducible randomized finite laws, and each transition row of a three-state Markov fixture. Monte Carlo output histograms are secondary checks with a stated simultaneous 99% Hoeffding confidence envelope; they do not replace the pointwise algebra checks.

## Candidate tree and path notation

A candidate tree is only a **precomputation/continuation budget**. Its children may be selected using `q`, but tree membership never changes the distribution `π` used for a collapse draw or by the verifier. An accepted sampled token with a retained child may continue along that precomputed branch. If it has no retained child, emit it unchanged and stop extending this pulse; do not resample, renormalize over retained children, or silently replace it with a draft candidate. A rejection emits the residual token and stops the pulse.

For a path `u = (x₁,…,xₖ)`, write

`path_weight(u | c) = ∏ᵢ q(xᵢ | cᵢ)`.

This product is a **path probability/weight**, not an amplitude. The v1 implementation uses probability weights only; it makes no Born-rule or quantum-state claim.

## Explicitly deferred from v1

Rotor mixing/readout, LIF timing, Gumbel/Born sampling, threshold bandits or learned threshold controllers, confidence calibration, and performance/latency optimization are optional follow-on modules. They are not required for v1 exactness and must not alter the core definitions above.

## v1 exit criteria

The v1 core is complete when all of the following hold:

1. This contract exists and unambiguously distinguishes `p`, `q`, and `π`.
2. A C++17 core verifier samples and verifies using the same explicit `π`, and computes rejection from `max(p−π, 0)` without epsilon-based substitution.
3. A deterministic-seed end-to-end test with `π != q` agrees empirically with `p` within a stated Monte Carlo tolerance.
4. That test confirms an accepted token outside the retained candidate set is emitted unchanged and ends the pulse; retained-child availability does not renormalize or change the proposal.
5. The equality case `p = π` never rejects and never invokes a zero-mass residual.
6. The existing C++ acceptance suite still passes.
