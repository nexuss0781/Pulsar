# Production runtime dependency record

The production backend uses the llama.cpp public C API and is pinned to the following exact upstream source revision:

- Project: [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
- Pinned commit: [`cd0a5b6a180dff0048e5fd993098b21c0b2c91b8`](https://github.com/ggml-org/llama.cpp/tree/cd0a5b6a180dff0048e5fd993098b21c0b2c91b8)
- Public C API source reviewed: [include/llama.h](https://github.com/ggml-org/llama.cpp/blob/cd0a5b6a180dff0048e5fd993098b21c0b2c91b8/include/llama.h)
- Upstream build overview: [README.md](https://github.com/ggml-org/llama.cpp/blob/cd0a5b6a180dff0048e5fd993098b21c0b2c91b8/README.md)

The adapter uses the public model/context initialization, vocabulary/tokenization, `llama_batch`/`llama_decode`, output-logit, sequence-memory clearing/removal, and end-of-generation APIs. Model loading fixes GPU layers to zero; context offload flags and CMake GPU backends are disabled. `GGML_NATIVE` is disabled in the production CMake integration so the build does not encode the builder's CPU instruction set.

The commit is pinned for reproducibility; upgrading llama.cpp requires rebuilding and rerunning runtime tests and real-GGUF smoke/rollback checks.
