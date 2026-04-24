#!/bin/bash
set -e
LANG_CODE="$1"
TEST_FILE="$2"
OUTPUT_DIR="$3"
if [[ -z "$LANG_CODE" || -z "$TEST_FILE" || -z "$OUTPUT_DIR" ]]; then
    echo "Usage: ./infer.sh <lang> <test_file> <output_dir>"
    exit 1
fi
cd "$(dirname "$0")"
export TOKENIZERS_PARALLELISM=false
python icl_inference.py "$LANG_CODE" "$TEST_FILE" "$OUTPUT_DIR"
