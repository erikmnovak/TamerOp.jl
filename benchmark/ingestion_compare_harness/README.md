# Ingestion Comparison Harness (tamer-op vs multipers)

This harness provides a reproducible, case-based comparison for point-cloud and image ingestion.

## What it benchmarks

Primary regimes:

- `claim_matching`
  - Radius-threshold parity regime with shared per-case radius from fixture manifest.
  - tamer-op: Rips + `sparsify=:radius`, `radius=claim_radius`, `collapse=:none`, `output_stage=:simplex_tree`.
  - multipers: `gudhi.RipsComplex(..., max_edge_length=claim_radius)` wrapped as `multipers.simplex_tree_multi.SimplexTreeMulti`.

- `normalized_parity`
  - Shared-threshold parity regime for direct algorithm comparison.
  - fixture generation writes `parity_radius` (radius-k policy) into the manifest.
  - tamer-op: Rips + `radius=parity_radius`, `sparsify=:none`, `collapse=:none`, `output_stage=:simplex_tree`.
  - multipers: `gudhi.RipsComplex(..., max_edge_length=parity_radius)` wrapped as `SimplexTreeMulti`.
  - fallback path (no `parity_radius` in manifest): `PointCloud2FilteredComplex(..., complex="rips", output_type="simplextree", num_collapses=0)`.

- `landmark_parity`
  - Shared deterministic landmark subset + shared radius threshold on that subset.
  - tamer-op: `kind=:landmark_rips` with manifest landmarks + `radius=landmark_radius`.
  - multipers: GUDHI Rips on the exact same landmark subset with `max_edge_length=landmark_radius`, wrapped as `SimplexTreeMulti`.
  - This is strict apples-to-apples for landmark-only Rips construction.

- `degree_rips_parity`
  - Shared radius-threshold parity for degree-Rips bifiltration.
  - tamer-op: `kind=:degree_rips`, `radius=degree_radius`, `output_stage=:simplex_tree`.
  - multipers: `filtrations.DegreeRips(points, threshold_radius=degree_radius)`.

- `rips_codensity_parity`
  - Shared radius + density-policy parity for codensity-style Rips bifiltration.
  - tamer-op: `kind=:rips_density`, `radius=codensity_radius`, `density_k=codensity_k`.
  - multipers: `filtrations.RipsCodensity(points, dtm_mass=codensity_dtm_mass, threshold_radius=codensity_radius)`.
  - Note: codensity definition differs across implementations; treat as directional parity.

- `rips_lowerstar_parity`
  - Shared radius + shared vertex function parity (`coord1`).
  - tamer-op: `kind=:function_rips`, `vertex_function=(p,_)->p[1]`, `radius=lowerstar_radius`.
  - multipers: `filtrations.RipsLowerstar(points=..., function=coord1, threshold_radius=lowerstar_radius)`.

- `delaunay_lowerstar_parity`
  - Shared Delaunay lower-star parity with the same vertex function (`coord1`).
  - tamer-op: `kind=:function_delaunay`, `vertex_function=(p,_)->p[1]`.
  - multipers: `filtrations.DelaunayLowerstar(points, function=coord1)`.
  - Some multipers builds may not include `function_delaunay`; those cases are skipped in `run_multipers.py`.

- `alpha_function_delaunay` (alpha parity; legacy regime name)
  - Non-Rips alpha-complex parity comparison for geometric filtrations in 2D.
  - tamer-op: `kind=:alpha`, `output_stage=:simplex_tree`.
  - multipers: `PointCloud2FilteredComplex(complex="alpha", output_type="simplextree")`.

- `core_delaunay_parity`
  - Non-Rips core-Delaunay parity comparison in 2D.
  - tamer-op: `kind=:core_delaunay`, `max_dim=...`, `output_stage=:simplex_tree`.
  - multipers: `filtrations.CoreDelaunay(...)`, then `prune_above_dimension(max_dim)`.

- `cubical_parity`
  - Image cubical filtration parity comparison.
  - tamer-op: `kind=:cubical`, `output_stage=:graded_complex` (cubical cells).
  - multipers: `filtrations.Cubical(image[...,None])`.

Outputs include cold and warm timing summaries and per-case slowdown factors.

## Files

- `cases.toml`: case matrix (sizes, dimensions, seeds, regimes).
- `generate_fixtures.py`: generates deterministic CSV point-cloud/image fixtures and manifest.
- `run_tamer.jl`: runs tamer-op ingestion on the fixture manifest.
- `run_multipers.py`: runs multipers ingestion on the same fixtures.
- `compare.py`: joins result CSVs and computes slowdown tables.
- `run_all.sh`: one-shot runner.

## Usage

From repo root:

```bash
# Quick sanity pass (scaled down fixtures)
SCALE=0.01 PROFILE=probe REGIME=all bash benchmark/ingestion_compare_harness/run_all.sh

# Desktop-safe default profile (recommended local default)
SCALE=1.0 PROFILE=desktop REGIME=all bash benchmark/ingestion_compare_harness/run_all.sh

# Balanced profile (more repetitions)
SCALE=1.0 PROFILE=balanced REGIME=all bash benchmark/ingestion_compare_harness/run_all.sh

# Heavier stress profile
SCALE=1.0 PROFILE=stress REGIME=all bash benchmark/ingestion_compare_harness/run_all.sh

# Explicitly override profile reps
SCALE=1.0 PROFILE=desktop REPS=6 REGIME=all bash benchmark/ingestion_compare_harness/run_all.sh

# Run an expanded apples-to-apples matrix (different n / ambient_dim / max_dim)
CASES_FILE=benchmark/ingestion_compare_harness/cases_matrix.toml \
FIXTURES_DIR=benchmark/ingestion_compare_harness/fixtures_matrix \
TAMER_OUT=benchmark/ingestion_compare_harness/results_tamer_matrix.csv \
MULTIPERS_OUT=benchmark/ingestion_compare_harness/results_multipers_matrix.csv \
COMPARE_OUT=benchmark/ingestion_compare_harness/comparison_matrix.csv \
SUMMARY_OUT=benchmark/ingestion_compare_harness/comparison_summary_matrix.csv \
SCALE=1.0 PROFILE=desktop REGIME=all \
bash benchmark/ingestion_compare_harness/run_all.sh

# Run non-Rips matrix (alpha/core_delaunay/cubical)
CASES_FILE=benchmark/ingestion_compare_harness/cases_nonrips.toml \
FIXTURES_DIR=benchmark/ingestion_compare_harness/fixtures_nonrips \
TAMER_OUT=benchmark/ingestion_compare_harness/results_tamer_nonrips.csv \
MULTIPERS_OUT=benchmark/ingestion_compare_harness/results_multipers_nonrips.csv \
COMPARE_OUT=benchmark/ingestion_compare_harness/comparison_nonrips.csv \
SUMMARY_OUT=benchmark/ingestion_compare_harness/comparison_summary_nonrips.csv \
SCALE=1.0 PROFILE=desktop REGIME=all \
bash benchmark/ingestion_compare_harness/run_all.sh

# Run landmark-subset parity matrix
CASES_FILE=benchmark/ingestion_compare_harness/cases_matrix.toml \
FIXTURES_DIR=benchmark/ingestion_compare_harness/fixtures_matrix \
TAMER_OUT=benchmark/ingestion_compare_harness/results_tamer_matrix.csv \
MULTIPERS_OUT=benchmark/ingestion_compare_harness/results_multipers_matrix.csv \
COMPARE_OUT=benchmark/ingestion_compare_harness/comparison_matrix.csv \
SUMMARY_OUT=benchmark/ingestion_compare_harness/comparison_summary_matrix.csv \
SCALE=1.0 PROFILE=desktop REGIME=landmark_parity \
bash benchmark/ingestion_compare_harness/run_all.sh
```

Or run steps manually:

```bash
python benchmark/ingestion_compare_harness/generate_fixtures.py --scale 1.0 --force
julia --project=. benchmark/ingestion_compare_harness/run_tamer.jl --manifest=benchmark/ingestion_compare_harness/fixtures/manifest.toml --profile=desktop
python benchmark/ingestion_compare_harness/run_multipers.py --manifest benchmark/ingestion_compare_harness/fixtures/manifest.toml --profile desktop
python benchmark/ingestion_compare_harness/compare.py
```

Profiles:

- `desktop` (default): `reps=4`, trims memory between cases/reps
- `balanced`: `reps=5`, trims memory between cases
- `stress`: `reps=9`
- `probe`: `reps=3`, trims memory between cases

Both runners accept explicit `--reps` to override the profile default.

## Robust Cold-Start Number (single metric)

For a single robust cold-start metric (less sensitive to case order/JIT effects),
run fresh-process repeats per case and aggregate medians:

```bash
python benchmark/ingestion_compare_harness/cold_robust.py \
  --manifest benchmark/ingestion_compare_harness/fixtures_50k/manifest.toml \
  --regime claim_matching \
  --restarts 3 \
  --profile desktop

# Run robust cold on the expanded matrix
python benchmark/ingestion_compare_harness/cold_robust.py \
  --manifest benchmark/ingestion_compare_harness/fixtures_matrix/manifest.toml \
  --regime all \
  --restarts 3 \
  --profile desktop \
  --raw_out benchmark/ingestion_compare_harness/cold_robust_raw_matrix.csv \
  --summary_out benchmark/ingestion_compare_harness/cold_robust_summary_matrix.csv
```

Outputs:
- `cold_robust_raw.csv`: all per-repeat cold measurements
- `cold_robust_summary.csv`: per-case median cold times + robust geomean slowdown

To rank where tamer is slower/faster:

```bash
python benchmark/ingestion_compare_harness/where_slower.py \
  --comparison benchmark/ingestion_compare_harness/comparison_matrix.csv \
  --manifest benchmark/ingestion_compare_harness/fixtures_matrix/manifest.toml \
  --out benchmark/ingestion_compare_harness/where_slower_matrix.csv
```

## Result files

Generated in this directory:

- `results_tamer.csv`
- `results_multipers.csv`
- `comparison.csv`
- `comparison_summary.csv`

`comparison_summary.csv` reports geometric-mean slowdown:

- `warm_geomean_slowdown_tamer_over_multipers`
- `cold_geomean_slowdown_tamer_over_multipers`

Values > 1.0 mean tamer-op is slower than multipers.

## Notes

- Use the same thread settings for both tools when comparing (e.g. single-thread vs all-core).
- `run_multipers.py` requires `multipers` installed in the active Python environment.
- `claim_matching` now aligns construction policy knobs via shared `claim_radius` and no collapse on both tools.
- `alpha_function_delaunay` now runs alpha-vs-alpha parity (name kept for fixture compatibility).
