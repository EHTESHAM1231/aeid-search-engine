#!/usr/bin/env bash
# Render build script
set -o errexit

pip install --upgrade pip
pip install -r requirements.txt

# Optional: cleanlab for label noise detection (app works without it)
pip install "cleanlab>=2.5.0" || echo "cleanlab install failed — skipping (optional dependency)"
