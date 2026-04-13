#!/bin/bash
python "$(dirname "$0")/train_generative.py" "${1:-output}" "${2:-..}"