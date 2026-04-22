# Thesis Macro Harness

This harness is the thesis-scale, `tamer-op`-only benchmark matrix.

It is separate from `benchmark/ingestion_compare_harness/`, which remains the smaller cross-tool parity suite.

## Components
- `generate_catalog.jl` builds the concrete job catalog.
- `generate_fixtures.jl` materializes persisted raw/presentation fixtures.
- `run_tamer_macro.jl` runs the `tamer-op` benchmark jobs and records light summaries plus digests.
- `summarize_results.py` groups result rows and emits CSV/Markdown/LaTeX summaries.
- `common.jl` owns shared size ladders, catalog rules, fixture generation, filtration specs, runner plumbing, and light output summaries.

## Benchmark contract
- One concrete job per `(source_case, filtration_family, invariant, degree)`.
- The main suite is `tamer-op` only.
- Large runs do not serialize full canonical outputs.
- Rows record:
  - end-to-end encode-plus-invariant timing,
  - allocations and GC time for the cold pass,
  - lightweight output summaries and stable digests,
  - backend/path labels such as `exact_h0_lazy` or `workflow_fallback`.

## Profiles
Catalog size ladders are profile-driven.

- `thesis`
  - intended for saved thesis runs.
- `smoke`
  - tiny deterministic fixtures for harness testing.

Runner repetition policy is separate and uses:
- `probe`
- `desktop`
- `balanced`
- `stress`
- `smoke`

## Typical usage
Generate the thesis manifest:

```bash
julia --project=. benchmark/thesis_macro_harness/generate_catalog.jl --profile=thesis --out=benchmark/thesis_macro_harness/manifest.toml
```

Generate fixtures:

```bash
julia --project=. benchmark/thesis_macro_harness/generate_fixtures.jl --manifest=benchmark/thesis_macro_harness/manifest.toml
```

Run a focused smoke subset:

```bash
julia --project=. benchmark/thesis_macro_harness/run_tamer_macro.jl --manifest=benchmark/thesis_macro_harness/manifest.toml --out=benchmark/thesis_macro_harness/results_smoke.csv --profile=probe --families=rips,graph_lower_star,lower_star --invariants=euler_signed_measure,rank_signed_measure --degrees=na,H0 --limit=8
```

Summarize:

```bash
python benchmark/thesis_macro_harness/summarize_results.py --results benchmark/thesis_macro_harness/results_smoke.csv --summary_out benchmark/thesis_macro_harness/results_smoke_summary.csv --markdown_out benchmark/thesis_macro_harness/results_smoke_summary.md --latex_out benchmark/thesis_macro_harness/results_smoke_summary.tex --errors_out benchmark/thesis_macro_harness/results_smoke_errors.csv
```

## Filters
`generate_fixtures.jl` supports:
- `--source_case_ids=a,b,c`
- `--source_kinds=point_cloud,graph,image,pl_fringe,flange`
- `--limit=N`
- `--force=true`

`run_tamer_macro.jl` supports:
- `--job_ids=a,b,c`
- `--source_case_ids=a,b,c`
- `--families=...`
- `--invariants=...`
- `--degrees=na,H0,H1,H2`
- `--source_kinds=...`
- `--limit=N`
- `--profile=probe|desktop|balanced|stress|smoke`
- `--fail_fast=true`

## Notes
- Presentation fixtures are synthetic and persisted via the existing JSON save/load surfaces.
- Family-specific ceilings are intentional. Not every family is expected to reach `50k`.
- Unsupported or failing jobs are recorded as `status=error` rows rather than aborting the full run by default.
