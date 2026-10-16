# Phase 4 evidence subset

Run: `20261015T202100Z`. This directory contains the frozen report and analysis summaries, normalized freeze/run metadata, split inputs, calibrator and tuning artifacts, all compact raw batch records and generated outputs, and the six calibration verifier-observation streams. `MANIFEST.sha256` lists relative repository paths and SHA-256 digests for every included file except the manifest itself.

This is a selected reproducibility subset, not a complete run archive. It omits execution logs (883,392,872 bytes in the source run), model binaries, and the evaluation/tuning verifier-observation streams. Consequently, the recorded report and derived evaluation metrics are preserved, but full per-token evaluation telemetry cannot be independently replayed from this commit.

For portability, host-local absolute path strings in the copied freeze/run/analysis/calibration/tuning metadata were normalized to repository-relative labels; recorded source/model digests and non-path experiment values were retained. The published `cpp/production/benchmark/PHASE4_PROTOCOL.md` also replaces host-local paths with portable descriptions. Its original experiment-time SHA-256 is recorded in the report; the committed portable copy has its own digest in `MANIFEST.sha256`.

The frozen report text and measurements are unchanged; Markdown hard-break trailing spaces were normalized so the repository whitespace check passes.
