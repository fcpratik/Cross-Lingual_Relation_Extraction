#!/bin/bash
# Task 1: Inference
# Usage: ./infer.sh <lang_code> <test_file_path> <output_dir>

LANG=${1:?"Usage: ./infer.sh <en/hi/kn> <test_file_path> <output_dir>"}
TEST_FILE=${2:?"Please provide test file path"}
OUTPUT_DIR=${3:-"output"}

echo "============================================"
echo "Task 1: Inference"
echo "Language:   $LANG"
echo "Test file:  $TEST_FILE"
echo "Output dir: $OUTPUT_DIR"
echo "============================================"

python infer_classifier.py "$LANG" "$TEST_FILE" "$OUTPUT_DIR"