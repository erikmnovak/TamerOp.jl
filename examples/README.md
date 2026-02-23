# Onboarding Examples

This directory is an ordered, didactic onboarding suite for `PosetModules`.

Each script is a runnable Julia file and includes:
- a theme header,
- explicit pipeline stages,
- deterministic seeds,
- guardrails for combinatorial blowup,
- concrete output artifacts under `examples/_outputs/<example_name>/`.

## Run order

1. `00_quickstart_module_to_features.jl`
2. `01_graded_complex_by_hand.jl`
3. `02_point_cloud_bifiltration_rips_density.jl`
4. `03_image_bifiltration_distance_intensity.jl`
5. `04_graph_bifiltration_lowerstar_clique.jl`
6. `05_fibered_slices_landscapes_and_distances.jl`
7. `06_batch_experiment_spec_and_exports.jl`
8. `07_serialization_reproducible_runs.jl`
9. `90_advanced_indicator_resolutions_and_ext.jl` (optional advanced appendix)

## Running examples

From repo root:

```bash
julia --project examples/00_quickstart_module_to_features.jl
```

or run all in order:

```bash
for f in \
  examples/00_quickstart_module_to_features.jl \
  examples/01_graded_complex_by_hand.jl \
  examples/02_point_cloud_bifiltration_rips_density.jl \
  examples/03_image_bifiltration_distance_intensity.jl \
  examples/04_graph_bifiltration_lowerstar_clique.jl \
  examples/05_fibered_slices_landscapes_and_distances.jl \
  examples/06_batch_experiment_spec_and_exports.jl \
  examples/07_serialization_reproducible_runs.jl \
  examples/90_advanced_indicator_resolutions_and_ext.jl
do
  julia --project "$f"
done
```

## Notes

- Optional feature IO outputs (Arrow/Parquet/NPZ/CSV) depend on loaded extensions.
  Every example also writes manual CSV fallbacks so outputs are always inspectable.
- Data-ingestion API is canonical:
  - use `encode(data, filtration_or_spec; ...)`
  - use `stage=...` to request intermediate objects
    (`:simplex_tree`, `:graded_complex`, `:cochain`, `:module`, `:fringe`, `:flange`, `:encoding_result`)
  - default is `stage=:auto`, which follows `ConstructionOptions.output_stage`
- All examples use the canonical Workflow cache contract:
  - `cache=:auto` for per-call automatic caching
  - `cache=SessionCache()` for explicit cross-call reuse
- PL mode uses strict canonical values only: `:fast` / `:verified`.
