#!/bin/bash
python "$(dirname "$0")/train_classifier.py" "${1:-output}" "${2:-..}"
