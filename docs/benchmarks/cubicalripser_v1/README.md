# CubicalRipser v1 result data

See the [report](../cubicalripser.md) for the mathematical request, timing boundary,
qualification failures and interpretation. This directory contains all 48 cases,
not just the largest sizes shown in the report.

- [results.csv](results.csv): one row per request, seconds, paired ratios, input/output sizes.
- [analysis.json](analysis.json): process medians, ranges, allocations, retained storage, RSS and startup.
- [samples.json](samples.json): every accepted measured query and compilation/reset counters.
- [formal_manifest.json](formal_manifest.json): exact shapes, seeds, orientations, generator and input hashes.
- [TASK_FREEZE.json](TASK_FREEZE.json): specifications fixed before reserved input generation.
- [verification.json](verification.json): complete barcode agreement over all accepted calls.
- [barcodes.json](barcodes.json): every verified interval and multiplicity, one canonical multiset per case.
- [calibration.json](calibration.json): all sizing calls, including the larger binary limitation and timeout.
- [MACHINE.json](MACHINE.json): CPU, OS, kernel, compiler and frequency policy.
- [PUBLIC_CALL_REVIEW.json](PUBLIC_CALL_REVIEW.json): supported public calls and fresh-state inspection.
- [qualification.json](qualification.json): independent controls, including failures outside the timed scope.
- [candidate_hashes.json](candidate_hashes.json): frozen source file hashes.
- [binding.json](binding.json) and [build.json](build.json): machine, harness and unchanged competitor build.

Ratios are CubicalRipser time divided by TamerOp time. The result table uses the
geometric mean of three paired process ratios. Timing ranges are observed process
ranges, not confidence intervals. RSS, allocation traffic and retained-object
sizes have different meanings; consult the report before comparing them.

The compressed input paths refer to the preserved local archive. This download
contains definitions and hashes, not those arrays, environments or a complete
runnable reproduction bundle. No single end-to-end cold-start ratio is claimed.

In `barcodes.json`, a row is `[degree, birth, death, multiplicity]`; `null` death
means an essential interval. Superlevel grades use the negated scalar function.
The counts retain the full interval multiset, including duplicates.
