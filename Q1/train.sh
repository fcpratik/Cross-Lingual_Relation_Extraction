#!/bin/bash
# Task 1: Train the classification-head RE model
# Usage: ./train.sh <output_dir>
# Run from inside Q1/ directory
# Expects ../en_sft_dataset/ and ../sft_dataset/ to exist

OUTPUT_DIR=${1:-"output"}

# Root dir is parent of Q1/
ROOT_DIR=".."

echo "============================================"
echo "Task 1: Training Classification Head RE"
echo "Output dir: $OUTPUT_DIR"
echo "Root dir:   $ROOT_DIR"
echo "============================================"

python train_classifier.py "$OUTPUT_DIR" "$ROOT_DIR"