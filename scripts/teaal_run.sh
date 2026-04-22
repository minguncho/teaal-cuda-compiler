#!/bin/bash
# file name: teaal_run.sh

# Type Checking
mypy teaal

# Auto-Formatting
autopep8 -iraa teaal/
autopep8 -iraa tests/

# Testing
python -m pytest -s tests/teaal
