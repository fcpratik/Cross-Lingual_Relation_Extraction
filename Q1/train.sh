#!/bin/bash
set -e
OUTPUT_DIR="${1:-output}"
mkdir -p "$OUTPUT_DIR"
cd "$(dirname "$0")"
export TOKENIZERS_PARALLELISM=false
python train_classifier.py "$OUTPUT_DIR" ..
