#!/bin/bash
# Run all tests
set -e
export PYTHONPATH=src
pytest tests
