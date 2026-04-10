#!/bin/bash
# Copyright (C) 2024 The AAOSP Project
# Licensed under the Apache License, Version 2.0
#
# Downloads Qwen 2.5 GGUF models for AAOSP

set -euo pipefail

TIER="${1:-mid}"  # high, mid, low, minimal

# Strip -- prefix if present (supports --tier high or just high)
TIER="${TIER#--tier}"
TIER="${TIER# }"
# Handle "--tier high" as two args
if [ "$TIER" = "--tier" ] && [ -n "${2:-}" ]; then
    TIER="$2"
fi

HF_BASE="https://huggingface.co/Qwen"

case "$TIER" in
    high)
        MODEL_REPO="Qwen2.5-7B-Instruct-GGUF"
        MODEL_FILE="qwen2.5-7b-instruct-q4_k_m.gguf"
        echo "=== Downloading Qwen 2.5 7B (Q4_K_M) — for 12GB+ devices ==="
        ;;
    mid)
        MODEL_REPO="Qwen2.5-3B-Instruct-GGUF"
        MODEL_FILE="qwen2.5-3b-instruct-q4_k_m.gguf"
        echo "=== Downloading Qwen 2.5 3B (Q4_K_M) — recommended default ==="
        ;;
    low)
        MODEL_REPO="Qwen2.5-1.5B-Instruct-GGUF"
        MODEL_FILE="qwen2.5-1.5b-instruct-q4_k_m.gguf"
        echo "=== Downloading Qwen 2.5 1.5B (Q4_K_M) — for 4-8GB devices ==="
        ;;
    minimal)
        MODEL_REPO="Qwen2.5-0.5B-Instruct-GGUF"
        MODEL_FILE="qwen2.5-0.5b-instruct-q8_0.gguf"
        echo "=== Downloading Qwen 2.5 0.5B (Q8_0) — for <4GB devices ==="
        ;;
    *)
        echo "Usage: $0 [high|mid|low|minimal]"
        echo ""
        echo "  high    - Qwen 2.5 7B Q4_K_M   (~4.4 GB) for 12GB+ RAM"
        echo "  mid     - Qwen 2.5 3B Q4_K_M   (~2.0 GB) for 8GB+ RAM (default)"
        echo "  low     - Qwen 2.5 1.5B Q4_K_M (~1.1 GB) for 4-8GB RAM"
        echo "  minimal - Qwen 2.5 0.5B Q8_0   (~0.5 GB) for <4GB RAM"
        exit 1
        ;;
esac

DOWNLOAD_URL="$HF_BASE/$MODEL_REPO/resolve/main/$MODEL_FILE"

# Determine output directory
if [ -n "${ANDROID_PRODUCT_OUT:-}" ]; then
    OUT_DIR="$ANDROID_PRODUCT_OUT/data/local/llm"
else
    OUT_DIR="$(pwd)/models"
fi

mkdir -p "$OUT_DIR"
OUT_PATH="$OUT_DIR/$MODEL_FILE"

if [ -f "$OUT_PATH" ]; then
    echo "Model already exists: $OUT_PATH"
    echo "Delete it first if you want to re-download."
    exit 0
fi

echo "Downloading to: $OUT_PATH"
echo "URL: $DOWNLOAD_URL"
echo ""

# Download with progress
curl -L --progress-bar -o "$OUT_PATH" "$DOWNLOAD_URL"

echo ""
echo "=== Download complete ==="
echo "Model: $OUT_PATH"
echo "Size:  $(du -h "$OUT_PATH" | cut -f1)"
echo ""

if [ -n "${ANDROID_PRODUCT_OUT:-}" ]; then
    echo "To push to device:"
    echo "  adb shell mkdir -p /data/local/llm"
    echo "  adb push $OUT_PATH /data/local/llm/"
else
    echo "To push to device:"
    echo "  adb shell mkdir -p /data/local/llm"
    echo "  adb push $OUT_PATH /data/local/llm/"
fi
