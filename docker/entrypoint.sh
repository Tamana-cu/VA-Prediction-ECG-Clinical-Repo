#!/bin/bash
set -e


# Optional: run tests, then train model
echo "Running tests before starting training..."
pytest -q --disable-warnings || true


python src/train.py --config experiments/baseline_cnn_config.yaml
