# Phase 5 Validation Protocol

## Frozen implementation inputs

- Backend: [`UPSTREAM.md`](UPSTREAM.md), pinned llama.cpp commit `cd0a5b6a180dff0048e5fd993098b21c0b2c91b8`.
- Native build settings: CPU-only; `GGML_NATIVE=OFF`; `GGML_CPU_ALL_VARIANTS=OFF`; all configured GPU backends disabled. The inspected x86_64 build uses `-msse4.2 -mf16c -mfma -mbmi2 -mavx -mavx2`; AVX512 and AVX-VNNI are off, so it is **not** a generic x86-64 baseline binary. The AArch64 cross-build explicitly used `-march=armv8-a` and disabled CPU variants; it is QEMU/emulated test coverage, not native hardware evidence.
- Test coverage: deterministic C++ runtime suite; AddressSanitizer/UndefinedBehaviorSanitizer build; AArch64 cross-build and QEMU user-mode execution; local real-GGUF CLI smoke tests; staged installation and independent `find_package(PulsarRuntime)` consumer.
- Model fixtures used for local checks (not committed): [`tensorblock/tinyllama-15M-stories-GGUF`](https://huggingface.co/tensorblock/tinyllama-15M-stories-GGUF) Q2_K and Q3_K_M (Llama family, target/draft), and [`afrideva/Tinystories-gpt-0.1-3m-GGUF`](https://huggingface.co/afrideva/Tinystories-gpt-0.1-3m-GGUF) Q2_K (GPT-2 family). The GPT-2 architecture and 2048 context are reported by the [Hugging Face metadata API](https://huggingface.co/api/models/afrideva/Tinystories-gpt-0.1-3m-GGUF); the model card is [here](https://huggingface.co/afrideva/Tinystories-gpt-0.1-3m-GGUF/raw/main/README.md). Its fetched card/API does not declare a license, so the fixture is not redistributed. Actual local GPT-2 test file: 7,754,464 bytes, SHA-256 `0cfb3ef83cd05fba1daa40a8a0207edb7f4e9f9992bcf43b0bc6e679d6e8e58b`. Model files stay outside this repository.

## Real-GGUF smoke harness

From the repository root, after building `pulsar-generate` against the pinned llama.cpp source:

```sh
python3 cpp/production/tests/phase5_cli_smoke.py \
  --binary /tmp/pulsar-phase4-build/pulsar-generate \
  --llama-model /tmp/pulsar-smoke.gguf \
  --llama-draft /tmp/pulsar-smoke-draft.gguf \
  --gpt2-model /tmp/pulsar-smoke-gpt2-q2.gguf
```

The harness verifies CLI argument bounds, missing and truncated models, autoregressive execution on both model families, target/draft chain and tree modes, a long GPT-2 prompt, 16 requests across four microbatches, output-file JSONL, four deterministic warm repeats, SIGINT cancellation, and a subprocess-only 384 MiB address-space pressure case. It records diagnostic output but makes no latency or performance claim. `--skip-cancellation` and `--skip-resource-pressure` are available only to isolate a platform-specific test issue and must be recorded when used.

## Architecture distinction

The host architecture is recorded from `uname -m`. x86_64 builds executed on this computer are native x86_64 validation. An AArch64 cross-build with `aarch64-linux-gnu-g++` executed under `qemu-aarch64` is emulation and must be reported as such. Native ARM64 hardware validation remains open unless separately performed on an actual AArch64 host with matching OS and tests.

## Integrity and reproducibility

The harness writes only to a temporary directory. Test model weights are not added to the source tree. Record model path basename, file size, SHA-256, backend revision, host/kernel/compiler, build flags, exact test command, exit status, stdout/stderr summaries and any skipped checks in the Phase 5 report. Preserve Phase 3 and Phase 4 reports/raw evidence unchanged; Phase 5 evidence belongs under `paper/results/phase5/`.
