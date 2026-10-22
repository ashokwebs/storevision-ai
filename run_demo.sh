#!/usr/bin/env bash
# ==============================================================================
# run_demo.sh - Quick One-Command Demo for OpenCV AI Competition 2026 (AWS)
# ==============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

echo "========================================================"
echo "  StoreVision AI: Physical Retail Perception (OpenCV 5 + AWS)"
echo "========================================================"

if [ -d ".venv" ]; then
  . .venv/bin/activate
fi

export PYTHONPATH="$HERE"

echo "[1/2] Running automated OpenCV unit & regression tests..."
pytest -q tests/
echo "✔ All tests passed!"

echo ""
echo "[2/2] Running physical storefront audit on fixture..."
python -m storevision.cli audit --input fixtures/synthetic_storefront.jpg --output-dir out/

echo "========================================================"
echo "  ✔ Demo Complete! Inspect rectified signage in: out/"
echo "========================================================"
