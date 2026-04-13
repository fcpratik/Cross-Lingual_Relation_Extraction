#!/bin/bash
# Usage: ./infer.sh <lang> <test_file> <output_dir>
python "$(dirname "$0")/infer_classifier.py" "$1" "$2" "$3"
