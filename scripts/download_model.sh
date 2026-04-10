#!/bin/bash
# Copyright (C) 2024 The AAOSP Project
# Licensed under the Apache License, Version 2.0
#
# Downloads Qwen 2.5 GGUF models for AAOSP
#
# Verified filenames against HuggingFace as of 2025:
#   0.5B: qwen2.5-0.5b-instruct-q8_0.gguf       (676 MB, single file)
#   1.5B: qwen2.5-1.5b-instruct-q4_k_m.gguf      (1.12 GB, single file)
#   3B:   qwen2.5-3b-instruct-q4_k_m.gguf         (2.1 GB, single file)
#   7B:   qwen2.5-7b-instruct-q4_k_m-*.gguf       (split into 2 parts)

set -euo pipefail

TIER="${1:-mid}"

# Handle --tier flag
if [ "$TIER" = "--tier" ] && [ -n "${2:-}" ]; then
    TIER="$2"
fi

HF_BASE="https://huggingface.co/Qwen"

case "$TIER" in
    high)
        MODEL_REPO="Qwen2.5-7B-Instruct-GGUF"
        # 7B Q4_K_M is split into 2 parts on HuggingFace
        MODEL_FILES=(
            "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
            "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf"
        )
        echo "=== Downloading Qwen 2.5 7B (Q4_K_M) — for 12GB+ devices ==="
        echo "    Note: 2 split files, llama.cpp loads them automatically"
        ;;
    mid)
        MODEL_REPO="Qwen2.5-3B-Instruct-GGUF"
        MODEL_FILES=("qwen2.5-3b-instruct-q4_k_m.gguf")
        echo "=== Downloading Qwen 2.5 3B (Q4_K_M) — recommended default ==="
        ;;
    low)
        MODEL_REPO="Qwen2.5-1.5B-Instruct-GGUF"
        MODEL_FILES=("qwen2.5-1.5b-instruct-q4_k_m.gguf")
        echo "=== Downloading Qwen 2.5 1.5B (Q4_K_M) — for 4-8GB devices ==="
        ;;
    minimal)
        MODEL_REPO="Qwen2.5-0.5B-Instruct-GGUF"
        MODEL_FILES=("qwen2.5-0.5b-instruct-q8_0.gguf")
        echo "=== Downloading Qwen 2.5 0.5B (Q8_0) — for <4GB devices ==="
        ;;
    *)
        echo "Usage: $0 [high|mid|low|minimal]"
        echo ""
        echo "  high    - Qwen 2.5 7B Q4_K_M   (~4.4 GB, 2 split files) for 12GB+ RAM"
        echo "  mid     - Qwen 2.5 3B Q4_K_M   (~2.1 GB) for 8GB+ RAM (default)"
        echo "  low     - Qwen 2.5 1.5B Q4_K_M (~1.1 GB) for 4-8GB RAM"
        echo "  minimal - Qwen 2.5 0.5B Q8_0   (~676 MB) for <4GB RAM"
        exit 1
        ;;
esac

# Determine output directory
if [ -n "${ANDROID_PRODUCT_OUT:-}" ]; then
    OUT_DIR="$ANDROID_PRODUCT_OUT/data/local/llm"
else
    OUT_DIR="$(pwd)/models"
fi

mkdir -p "$OUT_DIR"

# Download each file
ALL_EXIST=true
for MODEL_FILE in "${MODEL_FILES[@]}"; do
    if [ ! -f "$OUT_DIR/$MODEL_FILE" ]; then
        ALL_EXIST=false
        break
    fi
done

if [ "$ALL_EXIST" = true ]; then
    echo "Model already exists in $OUT_DIR"
    echo "Delete the files first if you want to re-download."
    exit 0
fi

for MODEL_FILE in "${MODEL_FILES[@]}"; do
    OUT_PATH="$OUT_DIR/$MODEL_FILE"

    if [ -f "$OUT_PATH" ]; then
        echo "Already exists: $MODEL_FILE"
        continue
    fi

    DOWNLOAD_URL="$HF_BASE/$MODEL_REPO/resolve/main/$MODEL_FILE"
    echo "Downloading: $MODEL_FILE"
    echo "URL: $DOWNLOAD_URL"
    echo ""

    curl -L --progress-bar -o "$OUT_PATH" "$DOWNLOAD_URL"

    echo "  Size: $(du -h "$OUT_PATH" | cut -f1)"
    echo ""
done

echo "=== Download complete ==="
echo "Model files in: $OUT_DIR"
ls -lh "$OUT_DIR"/*.gguf 2>/dev/null || true
echo ""
echo "To push to device:"
echo "  adb shell mkdir -p /data/local/llm"
echo "  adb push $OUT_DIR/*.gguf /data/local/llm/"
echo "  adb shell restorecon -R /data/local/llm/"
