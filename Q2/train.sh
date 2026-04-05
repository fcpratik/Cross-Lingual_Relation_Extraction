#!/bin/bash
# Task 2: Train autoregressive generation RE model
# Usage: ./train.sh <output_dir>
OUTPUT_DIR=${1:-"output"}
ROOT_DIR=".."
echo "============================================"
echo "Task 2: Training Generative RE"
echo "Output dir: $OUTPUT_DIR"
echo "============================================"
python train_generative.py "$OUTPUT_DIR" "$ROOT_DIR"