# C++ Ripser v1 measurements

Read the [comparison report](../ripser_cpp.md) for the scope and interpretation.

- [Summary table](summary.csv) and [structured summaries](summary.json).
- [All observations](observations.json), including discarded warmups and rejected compilation samples.
- [Frozen cases and protocol](manifest.json), [input definitions](fixtures.json), [machine and build](machine_and_build.json), and [verification](verification.json).
- [TamerOp source hashes](candidate_hashes.json), [harness hashes](harness_hashes.json), and [checkpoint-recovery record](execution_recovery.json).
- [Sparse-cutoff limitation](sparse_threshold_limitation.json) and its [four-vertex input](sparse_threshold_control.txt).
- [Runtime/protocol revisions](runtime_configuration_revision.json), [excluded default-heap observations](initial_default_heap_observations.json), and [excluded shared-session heap-hint observations](shared_heap_hint_observations.json).
- [Checksums](SHA256SUMS).

`ratio` means C++ Ripser time divided by TamerOp time. Times are seconds;
memory and allocation values are bytes. Displayed times are medians of process
medians; ratios are geometric means of paired-process ratios. Bounds are
descriptive minimum/maximum process ratios, not confidence intervals.
The primary timer includes input parsing, native construction, complete
computation and barcode output. Julia construction and query columns describe
parts of that timer. Process RSS, allocation traffic and retained result size
measure different quantities. C++ RSS is sampled per invocation; Julia RSS
is the per-case process high-water mark, including runtime, compilation,
warmup and previous repetitions of that case. Each case starts a fresh Julia
process in each pass; it does not include other benchmark cases.

The local harness and original logs remain under `benchmark/` and `audit/`.
These public measurements are not a self-contained runnable reproduction
bundle. The previous Ripser.py archive remains separate and unchanged.
