#!/bin/bash
# file name: gpuloops_run.sh

# Type Checking
mypy gpuspec

# Auto-Formatting
autopep8 -iraa gpuspec/
autopep8 -iraa tests/

# Testing
python -m pytest -s tests/gpuloops
