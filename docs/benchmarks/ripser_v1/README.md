# Ripser.py v1 benchmark data

Four paired passes, three fresh computations per case/tool/pass; 31 development
requests and six separately reported reserved evaluation cases. See the
[public report](../ripser.md) for timing boundaries and interpretation.

`summary.csv` contains median process times (seconds), allocations (bytes),
distinct retained-output estimates and RSS, sizes and output counts.
`summary.json` additionally retains all process medians and paired ratios.
Case ratios are geometric means of four paired process ratios; family summaries
weight cases equally, and the development aggregate weights six families equally.
Evaluation cases are not pooled into those aggregates. Ranges are across four
passes, not confidence intervals. `accepted_samples.json` retains raw timing
and compilation/reset evidence; `verification.json` describes independent checks.

The manifest identifies inputs by hashes; their files are in the local archive,
not this data directory. Frozen source/harness hashes identify the unreleased
development snapshot. Public data can reproduce summaries and figures, but the
full computational reproduction bundle remains future work.

Run `sha256sum -c SHA256SUMS` from this directory to check the public artifacts.
