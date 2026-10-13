#!/usr/bin/env bash
# Run the analysis scripts in the order used for the article.
set -euo pipefail
cd "$(dirname "$0")"
export CPTAC_DATA_DIR="${CPTAC_DATA_DIR:-./data/cptac}"
export ANALYSIS_OUT="${ANALYSIS_OUT:-./output}"
mkdir -p "$ANALYSIS_OUT/results"
for s in scripts/0{1..9}_*.py scripts/10_*.py; do
  echo "=== ${s} ==="
  python3 "${s}"
done
echo "Finished. Tables are in ${ANALYSIS_OUT}/results/"
