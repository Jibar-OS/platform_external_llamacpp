# llama.cpp for AOSP

llama.cpp packaged for the Android Open Source Project build system.
Used by the AAOSP LLM System Service for local inference.

## Structure

```
platform_external_llamacpp/
├── Android.bp          # Soong build rules
├── jni/
│   └── llm_jni.cpp     # JNI bridge (symlinked from frameworks/base)
├── scripts/
│   ├── sync_upstream.sh    # Pull latest llama.cpp release
│   └── download_model.sh  # Download Qwen 2.5 GGUF models
├── src/                # llama.cpp upstream source (not checked in)
│   ├── include/
│   ├── ggml/
│   ├── src/
│   └── common/
└── README.md
```

## Setup

### 1. Sync llama.cpp source

```bash
cd external/llama.cpp
./scripts/sync_upstream.sh
```

This downloads the latest llama.cpp release and extracts it to `src/`.

### 2. Download a model

```bash
# Default: Qwen 2.5 3B (recommended for 8GB+ devices)
./scripts/download_model.sh

# Or specify a tier:
./scripts/download_model.sh --tier high    # Qwen 2.5 7B (12GB+ devices)
./scripts/download_model.sh --tier low     # Qwen 2.5 1.5B (4-8GB devices)
./scripts/download_model.sh --tier minimal # Qwen 2.5 0.5B (<4GB devices)
```

Models are downloaded to `$ANDROID_PRODUCT_OUT/data/local/llm/`.

### 3. Build

```bash
# Build the static library + JNI shared library
m libllama libllm_jni
```

### 4. Install model on device

```bash
adb push $ANDROID_PRODUCT_OUT/data/local/llm/*.gguf /data/local/llm/
```

## Model Selection

The LLM System Service auto-selects the model based on device RAM:

| RAM | Model | Size | Context |
|-----|-------|------|---------|
| 12GB+ | Qwen 2.5 7B Q4_K_M | ~4.4 GB | 8192 |
| 8GB+ | Qwen 2.5 3B Q4_K_M | ~2.0 GB | 4096 |
| 4-8GB | Qwen 2.5 1.5B Q4_K_M | ~1.1 GB | 2048 |
| <4GB | Qwen 2.5 0.5B Q8_0 | ~0.5 GB | 1024 |

Override via system properties:
```bash
adb shell setprop persist.llm.model.path /data/local/llm/custom-model.gguf
adb shell setprop persist.llm.context_size 4096
adb shell setprop persist.llm.gpu_layers 0
adb shell setprop persist.llm.n_threads 4
```

## Updating llama.cpp

```bash
# Edit LLAMA_CPP_VERSION in sync_upstream.sh, then:
./scripts/sync_upstream.sh

# Rebuild
m libllama libllm_jni
```

## License

- This packaging: Apache 2.0 (same as AOSP)
- llama.cpp: MIT License
- Qwen 2.5 models: Apache 2.0
