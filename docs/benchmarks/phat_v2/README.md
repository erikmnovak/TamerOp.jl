# PHAT v2 result data

These files support the [current PHAT comparison](../phat.md), completed
2026-10-04 for `phat-v2-2026-10-04`. Its single scored population contains
48 medium/large inputs, with four families, four variants per family and three
sizes. All requests completed correctly. No pilot or earlier-study measurements
enter these results.

| File | Contents |
| --- | --- |
| `results.json` | Scope, family and phase aggregates, verification counts, elapsed stages, host observations and memory |
| `observations.json` | All 144 case/phase rows, including five paired process medians and available allocation/retention counters |
| `timings.csv` | Compact version of all 144 rows |
| `provenance.json` | Machine, versions, source/input/adapter identities and raw-evidence hashes |
| `family_ratios.svg` | Individual combined-phase ratios and family aggregates with intervals |
| `graph_scaling.svg`, `simplicial_scaling.svg`, `cubical_scaling.svg`, `algebraic_scaling.svg` | Three sizes of each of the sixteen variants, with both tools shown |
| `performance_profile.svg` | Fraction of the fixed cases completed within a given factor of the faster tool |
| `SHA256SUMS` | Digests of these files, excluding itself |

## PHAT v2 units and row meanings

Times are seconds; memory and allocations are bytes. `phase` is `construction`,
`query` or directly timed `combined`. `cells`, `nonzeros` and `bars` describe the
complete input and output. `size_level` is the position in a variant's ladder,
not a shared geometric dimension; `parameter` records its generator setting.
`mathematical_check` states the independent evidence available for that input.

Each tool's case time is the median of five process medians, each based on three
accepted fresh measurements after two warmups. `tools` retains those medians and
available counters. `phat_over_tamerop` is the geometric mean of the five paired
ratios; it can differ from dividing the displayed medians. Intervals are 95% t
intervals on five process-block log ratios, with four degrees of freedom. Family
and whole-suite intervals aggregate within paired blocks. All families, variants
and sizes have equal declared weight at their respective level.

The frozen practical classification uses `win` above 1.10, `loss` below 1/1.10
and `near_tie` between, with ratios always PHAT/TamerOp. Near ties are described
as similar performance in the report; these labels do not establish statistical
equivalence or imply that a full interval lies in one region.

Julia allocation and GC counters describe the call. Retained input/answer sizes
are separate roots. Process RSS includes runtime and untimed work and is sampled,
so brief peaks can be missed. PHAT allocation and retained-object counters are
null, not zero. Resource observations cover a shared desktop; system swap activity
is not attributed to a particular tool.

## PHAT v2 reuse and integrity

Run `sha256sum -c SHA256SUMS` in this directory to check file integrity.
This does not rerun the mathematical validators. Public per-pass summaries allow
the statistics and figures to be reconstructed, but do not recover every raw
sample. Exact fixtures, source/build recipes, validators and complete raw records
remain locally archived. These downloads are not an executable reproduction
bundle. Private paths and host names are omitted from public provenance.
