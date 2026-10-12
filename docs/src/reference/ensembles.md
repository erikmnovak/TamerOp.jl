# Ensembles of persistence results

Use `persistence_ensemble` to align repeated scalar observations and
`summarize_ensemble` to prepare their descriptive or mean-inference summaries.
The [usage guide](../../tutorials/uncertainty.ipynb) explains the choices; the
[implementation account](../../implementation/ensembles.md) gives the algorithms.

## Inputs and alignment

```julia
e = persistence_ensemble([0. 2.; 2. 4.]; axes=([0.,3.],),
    sample_ids=["a","b"], statistic="Dimension at a fixed parameter",
    protocol="Two supplied computations", population="These examples")
s = summarize_ensemble(e)
ensemble_values(s)
```

Supplied array dimensions are `(replicates, length.(axes)...)`. Axes must be
finite and strictly increasing, including after conversion for drawing.
IDs are distinct strings. Values may be finite real numbers or `missing`;
`NaN`, infinity and Boolean values are rejected. No alignment, interpolation,
normalization or missing-value imputation is performed. Original values are
retained; statistical outputs are `Float64`.

The ordinary-diagram overload requires `grid`, `sample_ids`, `protocol` and
`population`. Choose `statistic=:landscape` (default, positive `layer`) or
`:betti`, with `dim=0` by default. Sublevel counts use `[birth,death)`;
superlevel counts use `(death,birth]` at original coordinates. Diagrams must
share field and order. `essential=:error` rejects essential intervals;
`:drop` explicitly omits them; `:include` is available for Betti counts only.

A vector of compatible `ScalarField` objects may instead provide a common
Cartesian domain. Select `layer` by identity or index. Axis coordinates, labels,
units and ordered layer identities must agree; field/degree/normalization remain
caller declarations. Scalar-field `NaN` masks become ensemble `missing` entries.
No ambient-landscape theorem is inferred from a field or slice stack.

## Summaries and statistical meaning

| Option/result | Meaning |
| --- | --- |
| `band=:quantile`, `quantiles=(0.1,0.9)` | Linear empirical quantiles of available replicates; descriptive. |
| `band=:range` | Observed extrema; descriptive. |
| `threshold=0` | `frequency` counts values strictly greater than the threshold. |
| `mean`, `median`, `std`, `counts` | Column mean, median, sample standard deviation and available denominator. |
| `band=:hoeffding`, `bounds=(lo,hi)` | Conditional finite-sample population-mean intervals under IID rows and known almost-sure bounds. Bounds are scalars or probe-shaped arrays. |
| `band=:bootstrap` | Approximate nonstudentized whole-row bootstrap mean bands; no finite-sample coverage guarantee. |
| `scope=:simultaneous` | Joint over all supplied probes; default for inference. |
| `scope=:pointwise` | Separate nominal mean coverage for each probe, not joint coverage. |
| `level=0.95` | Requested inference level; unrelated to empirical quantile spread. |
| `bootstrap_samples=1000`, `seed=0` | Private seeded row resampling; at least 100 resamples. |

Inference requires `sampling=:iid`, at least two complete rows, and probes fixed
independently of the observations. The software checks declarations and observed
bounds, not population independence. Both modes cover only the supplied probes.
Descriptive summaries permit missing values and retain each denominator; an
empty probe yields `NaN`, and sample standard deviation needs two observations.

`ensemble_values(s)` returns copied arrays with the probe-grid shape.
`ensemble_summary(e)`/`ensemble_summary(s)` and `describe` give the retained
context and coverage contract. `Featurizers.sample_ids(e)` and
`Featurizers.feature_values(e)` return copies; `feature_axes(e)` returns the
original coordinate vectors. Treat object storage as read-only.
`check_ensemble(x; throw=false)` validates retained shape/value contracts and
returns `Advanced.EnsembleValidationSummary`, with `valid` and `issues` fields.
Use `throw=true` to raise on malformed storage. Validation cannot establish
sampling independence or the provenance of supplied observations.

Budgets reject without truncation: `max_cells=1_000_000` at construction,
`max_work=10_000_000` for diagram interval/probe evaluation, and
`max_work=50_000_000` for summary/bootstrap arithmetic.

## Figures and linked selection

`visualize(s)` uses prepared values; it never resamples. Figures support one or
two coordinate axes. For higher-dimensional input, supply an explicitly selected
section before visualization. `sample` accepts an ID or one-based index;
`probe` is a one-based Cartesian linear index (first coordinate fastest).
`view` is one of `:mean`, `:lower`, `:upper`, `:std`, `:frequency`, `:counts`.
The default mean curve includes its band. `diagram_samples=(1,2)` requests up
to twelve retained ordinary-diagram facets. `max_cells=100_000` bounds the
supplied observation table used by a figure.

`samples` optionally contains one NamedTuple per replicate, with prepared static
`source` and/or `invariant` specifications and optional `metadata`. The selected
replicate displays those panels. Their relationship to observations is the
caller's declaration. No callback or distance computation is invoked on hover.

```julia
session = inspection_session(s)
TamerOp.Advanced.select_inspection!(session; sample="b", probe=2)
snapshot = TamerOp.Advanced.inspection_snapshot(session)
TamerOp.Advanced.close_inspection!(session)
```

Live WGL controls require Julia and permit replicate, probe and view changes.
Invalid selections preserve the previous state. Independent linked viewers
clean up their own listeners; closing the session releases its inputs and leaves
a static snapshot. Static Cairo exports and WGL HTML scenes do not provide live
statistical recomputation. See the [support matrix](../../visualization_support.md)
for the separate native, browser and notebook frontend contracts.

Statistical arithmetic is numerical. Known bounds are converted outward to
Float64, with their original declarations retained; a small outward endpoint
guard prevents a positive interval from disappearing through display rounding.
This is not an exact-arithmetic certificate for the floating reductions.
