#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASES_FILE="${CASES_FILE:-${ROOT_DIR}/cases.toml}"
FIXTURES_DIR="${FIXTURES_DIR:-${ROOT_DIR}/fixtures}"
TAMER_OUT="${TAMER_OUT:-${ROOT_DIR}/results_tamer.csv}"
MULTIPERS_OUT="${MULTIPERS_OUT:-${ROOT_DIR}/results_multipers.csv}"
COMPARE_OUT="${COMPARE_OUT:-${ROOT_DIR}/comparison.csv}"
SUMMARY_OUT="${SUMMARY_OUT:-${ROOT_DIR}/comparison_summary.csv}"

SCALE="${SCALE:-1.0}"
PROFILE="${PROFILE:-desktop}"
REPS="${REPS:-}"
REGIME="${REGIME:-all}"

python "${ROOT_DIR}/generate_fixtures.py" \
  --cases "${CASES_FILE}" \
  --out_dir "${FIXTURES_DIR}" \
  --scale "${SCALE}" \
  --force

tamer_args=(
  --manifest="${FIXTURES_DIR}/manifest.toml"
  --out="${TAMER_OUT}"
  --profile="${PROFILE}"
  --regime="${REGIME}"
)
if [[ -n "${REPS}" ]]; then
  tamer_args+=(--reps="${REPS}")
fi
julia --project=. "${ROOT_DIR}/run_tamer.jl" "${tamer_args[@]}"

multipers_args=(
  --manifest "${FIXTURES_DIR}/manifest.toml"
  --out "${MULTIPERS_OUT}"
  --profile "${PROFILE}"
  --regime "${REGIME}"
)
if [[ -n "${REPS}" ]]; then
  multipers_args+=(--reps "${REPS}")
fi
python "${ROOT_DIR}/run_multipers.py" "${multipers_args[@]}"

python "${ROOT_DIR}/compare.py" \
  --tamer "${TAMER_OUT}" \
  --multipers "${MULTIPERS_OUT}" \
  --out "${COMPARE_OUT}" \
  --summary_out "${SUMMARY_OUT}"

echo ""
echo "Done."
echo "  tamer:      ${TAMER_OUT}"
echo "  multipers:  ${MULTIPERS_OUT}"
echo "  comparison: ${COMPARE_OUT}"
echo "  summary:    ${SUMMARY_OUT}"
