#!/bin/bash
# Task 2: Inference
# Usage: ./infer.sh <lang_code> <test_file_path> <output_dir>
LANG=${1:?"Usage: ./infer.sh <en/hi/kn/or/tcy> <test_file> <output_dir>"}
TEST_FILE=${2:?"Provide test file path"}
OUTPUT_DIR=${3:-"output"}
echo "============================================"
echo "Task 2: Inference"
echo "Language: $LANG | Test: $TEST_FILE"
echo "============================================"
python infer_generative.py "$LANG" "$TEST_FILE" "$OUTPUT_DIR"