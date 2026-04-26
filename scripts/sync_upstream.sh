#!/bin/bash
# Copyright (C) 2024 The AAOSP Project
# Licensed under the Apache License, Version 2.0
#
# Syncs llama.cpp upstream source into src/

set -euo pipefail

LLAMA_CPP_REPO="https://github.com/ggml-org/llama.cpp"  # org moved ggerganov -> ggml-org in 2025

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$ROOT_DIR/src"

# Single source of truth for the upstream version pin. Same file is
# consumed by Android.bp's ggml-version-header genrule, so bumping
# version.txt updates both the compiled-in GGML_VERSION/GGML_COMMIT
# macros AND what this script downloads.
LLAMA_CPP_VERSION="$(cat "$ROOT_DIR/version.txt" | tr -d '[:space:]')"

echo "=== Syncing llama.cpp $LLAMA_CPP_VERSION ==="

# Clean old source
if [ -d "$SRC_DIR" ]; then
    echo "Removing old source..."
    rm -rf "$SRC_DIR"
fi

# Download and extract
TARBALL_URL="$LLAMA_CPP_REPO/archive/refs/tags/$LLAMA_CPP_VERSION.tar.gz"
TEMP_DIR=$(mktemp -d)
echo "Downloading $TARBALL_URL..."
curl -sL "$TARBALL_URL" | tar xz -C "$TEMP_DIR"

# Move to src/
mv "$TEMP_DIR/llama.cpp-${LLAMA_CPP_VERSION#b}"* "$SRC_DIR" 2>/dev/null || \
mv "$TEMP_DIR/llama.cpp-$LLAMA_CPP_VERSION"* "$SRC_DIR" 2>/dev/null || \
mv "$TEMP_DIR"/llama.cpp-* "$SRC_DIR"

rm -rf "$TEMP_DIR"

echo "=== llama.cpp $LLAMA_CPP_VERSION synced to $SRC_DIR ==="
echo ""
echo "Source files:"
echo "  include/: $(ls "$SRC_DIR/include/" 2>/dev/null | wc -l) files"
echo "  ggml/:    $(find "$SRC_DIR/ggml/" -name '*.c' -o -name '*.cpp' 2>/dev/null | wc -l) source files"
echo "  src/:     $(find "$SRC_DIR/src/" -name '*.cpp' 2>/dev/null | wc -l) source files"
echo "  common/:  $(find "$SRC_DIR/common/" -name '*.cpp' 2>/dev/null | wc -l) source files"
echo ""
echo "Next: m libllama libllm_jni"
