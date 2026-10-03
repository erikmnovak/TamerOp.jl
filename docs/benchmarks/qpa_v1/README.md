# QPA v1 result data

These files support the [QPA comparison](../qpa.md), completed 2026-10-03,
for candidate `qpa-v1-529681519b5d1b40`. They report the frozen 220-request
study, including its 20 unresolved QPA tensor comparisons. They do not contain
the full executable benchmark suite.

## Files and units

| File | Contents |
| --- | --- |
| `results.json` | Reviewed aggregate results, counts, versions, and stated scope |
| `timings.csv` | 660 rows: 220 cases × three measured phases; one row summarizes five passes |
| `observations.json` | The same 660 case/phase observations with per-pass medians, sample counts, and additional metadata |
| `provenance.json` | Selected run-time machine metadata, source identity, dependency versions, and archive hashes |
| `performance_profile.svg`, `input_sizes.svg` | Figures from the final analysis |
| `SHA256SUMS` | SHA-256 digests of this data directory, excluding the checksum file itself |

Times are **seconds**; allocations and exported-answer sizes are **bytes**.
Process RSS in the summary is **MiB** (2²⁰ bytes). Runtime allocation counters
are not retained memory, and Julia's counter does not include native FLINT
allocations. Exported JSON byte counts measure an output file's size, not the
memory occupied by the mathematical answer.

## Row meanings

`case` is the fixed request identifier. `workflow` names Q1–Q10 as described in
the report; `variant` distinguishes requests such as projective/injective
resolution and pushout/pullback. `area` is one of the five weighted mathematical
areas. `field` is QQ, F2, F3, or F101. `split` records development or the reserved
evaluation subset; that subset was evaluated after the source freeze and is
no longer unseen.

The `phase` values are:

- `native_construction`: build native inputs from the prepared category and
  decoded input.
- `complete_query`: produce the complete requested answer from native inputs.
- `construction_and_query`: directly time both stages together; the primary
  comparison.

In `observations.json`, each tool records five `pass_medians_seconds` and five
`sample_counts`. A complete pass median uses three independently reset samples.
`median_seconds` is the median of those five process medians, also exposed as
`TamerOp_seconds` or `QPA_seconds` in the CSV. Allocation columns summarize the
accepted per-timer counters; they are not allocation totals for a whole pass.

`ratio` is the geometric mean of the five paired QPA/TamerOp ratios. Consequently
it need not equal the quotient of the two displayed median times. `interval95`
(CSV: `ci_low`, `ci_high`) is the approximate paired-block t interval of the
log ratio, with four degrees of freedom. `pass_range` and `pass_ratios` retain
variation that the point estimate does not show.

`weight` is the case's exact rational share of the **original** 220-case suite;
it repeats across phases and must not be counted three times. For an aggregate,
select a phase, use mathematically matched complete pairs, then normalize the
retained weights within that subset. The primary subset has 200 cases and
original weight 9/10. `classification` uses a 10% practical margin: `win` above
1.10, `loss` below 1/1.10, and `near_tie` between them. A point classification
does not imply that its interval or every pass has the same classification.

JSON `null` and empty CSV fields mean unavailable, **not zero**. The 20 tensor
cases remain present. Their QPA query and combined timings are missing. Native
construction can have a time even when the requested final answer failed;
`mathematically_matched` and tool-level `verified` flags prevent treating that
as a completed comparison. The public phase aggregates consistently use the
same 200 matched requests.

`input_stalk_sum` sums dimensions of the supplied vector spaces at poset
vertices, counting each supplied module record once for that request.
`poset_vertices` counts vertices. `coefficient_max_bits` describes the largest
numerator/denominator bit length in the supplied module structure matrices;
it is not a bound on intermediate arithmetic. Together these describe inputs,
not a single universal complexity measure.

## Integrity and reuse

From this directory on a system with `sha256sum`, run:

```sh
sha256sum -c SHA256SUMS
```

This checks the downloaded files, not the mathematical computations. The source
archive and dependency hashes identify the private retained evidence without
claiming that the archive is available for download here. No absolute local
paths, host names, or unrelated project data are included.

The five medians per case are sufficient to recompute the reported speed
aggregates and intervals using the procedure in the comparison page. They
cannot recover the individual raw samples or independently rerun the answer
validators. A future portable reproduction bundle must supply those materials
and be identified separately from this results publication.
