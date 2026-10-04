# Ingestion options and their mathematical effects

A filtration assigns cells the parameter values at which they appear, with
boundaries present whenever their cells are present. Turning it into a finite
module requires choosing a parameter grid, coefficient field, and homological
degree. These choices determine what is represented; other options only change
how the same finite data are stored. The
[finite-encoding introduction](finite_encodings.md) explains how the resulting
spaces and maps are related to the original parameters.

`PipelineOptions` controls how a built filtration becomes an encoded module.
The distinction between changing the module and changing its storage matters
when comparing results. For example, these two calls produce the same vector
spaces and maps, with structured and dense representations of the finite grid
order:

```julia
using TamerOp
using TamerOp.CoreModules: F3
points = PointCloud([[0.0], [1.0]])
filtration = RipsFiltration(max_dim=1)
structured = encode(points, filtration;
    pipeline=PipelineOptions(poset_kind=:signature, field=F3()))
dense = encode(points, filtration;
    pipeline=PipelineOptions(poset_kind=:dense, field=F3()))
```

The structured product of chains is the default. Dense storage explicitly
materializes the same relation, with the same vertex labels; it can require
quadratic memory in the number of grid vertices. `:regions` belongs to other
encoding families and is not an ingestion grid representation.

| Control | Executed behavior |
|---|---|
| `orientation` | Chooses sublevel (`+1`) or superlevel (`-1`) coordinates. Superlevel coordinates are negated before constructing the increasing grid order. Some geometric families require a fixed orientation. |
| `eps` | Rounds every grade to its nearest positive step, shared across coordinates or specified by a tuple. This changes the filtration and also applies to raw simplex-tree and graded-complex outputs. |
| `axes_policy=:encoding` | Uses supplied axes when present, otherwise the constructed grade axes. Quantization preserves explicitly supplied axes. |
| `axes_policy=:as_given` | Requires and uses explicitly supplied filtration axes. |
| `axes_policy=:coarsen` | Selects at most `max_axis_len` values from each axis, evenly spaced in index. A positive `max_axis_len` is required; that control is rejected under other policies. |
| `axis_kind` | Validates integer-typed axes (`:zn`) or noninteger-typed real axes (`:rn`). It does not convert coordinates. |
| `poset_kind` | Chooses structured (`:signature`) or dense (`:dense`) storage for every encoded stage, including lazy complexes and dimension summaries. |
| `field` | Sets the coefficient field. Supply an actual field object, such as `QQField()`, `F2()`, `Fp(5)`, or `RealField()`. An explicit `encode(...; field=...)` keyword takes precedence. The default is QQ. |

A supplied or coarsened grid that omits a cell's critical grade uses the
**floor placement** contract: the grade is placed at the preceding
grid coordinate. It is not restriction of the original persistence module to
those grid points. The result provenance reports `:floor_snapped` when this
happens. To retain the original filtration exactly, include every critical
grade. Quantization and grid coarsening are separate operations.

Raw `stage=:simplex_tree` and `stage=:graded_complex` outputs have no finite
poset or coefficient-field data. Their plans can still retain those controls
for a later encoded stage. `run_ingestion(plan)` uses the stage stored by
`plan_ingestion`; `run_ingestion(plan; stage=...)` overrides it, and
`encode(plan)` explicitly requests the final encoding result.

`ConstructionOptions.output_stage` accepts every executable stage, including
`:encoded_complex` and `:cohomology_dims`. Unsupported sparsification controls
are rejected for the actual dataset/filtration pair. Budgets are checked on
prebuilt complexes and graph/cubical materializers as well as point-cloud
construction. A memory budget bounds the documented construction storage
estimate; it is not a bound on Julia's entire process memory.

Session caches distinguish poset representations, coefficient fields and
constructed chain data. Reusing a session therefore does not change an
explicit representation or field request. Pipeline JSON schema version 3 stores field objects
with their prime or numerical tolerances, so replaying it preserves the same
choice. The strict loader requires schema version 3; use the writer to create
a compatible pipeline file.

The Julia options constructor requires integer orientation signs. At the JSON
boundary, exactly integral numbers normalize to integers: `1.0` and `1` both
represent the sign `+1`. Nonintegral numbers, zero, signs other than `+1`/`-1`,
and Booleans are rejected.

For `load_data`, table parsing controls (`header`, `delimiter`,
`comment_prefix`, `missing_policy`) apply to CSV/TSV/text. `cols` selects
point-cloud coordinates, while `u_col`, `v_col` and `weight_col` select graph
columns. Nondefault controls that a chosen format or dataset kind does not
use are rejected. Owned JSON and Ripser inputs use their canonical parsers;
configure construction when subsequently calling `encode`.
