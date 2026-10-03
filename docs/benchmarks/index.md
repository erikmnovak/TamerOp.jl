# Benchmark results

A finite encoding keeps a module's vector spaces and the maps between them.
Once we have that finite object, we can ask more than how many classes exist:
we can compare modules, construct new ones, and compute their homological
algebra. These benchmarks measure the work needed to answer such questions.

Each comparison starts with a mathematical request that both programs support.
The programs may represent the input differently and use different algorithms;
what must agree is the requested answer. We report computation time, completion
and correctness checks, and memory separately.

## Available comparisons

| Comparison | Mathematical scope | Latest study | Result |
| --- | --- | --- | --- |
| [TamerOp and QPA](qpa.md) | Maps, constructions, resolutions, diagrams, complexes, and derived operations on finite-poset modules | 2026-10-03; fixed suite of 220 requests | TamerOp was faster on all 200 requests completed correctly by both tools; weighted geometric mean speed ratio **135.70×** for construction plus query. QPA's 20 tensor-evaluation failures have no speed ratio. |

The QPA result measures **compiled code computing fresh mathematical results**.
Package loading and compilation are outside the timer, and results from earlier
queries are discarded. The report gives the exact timing boundary, tested
development snapshot, machine, input sizes, and all incomplete comparisons.
It is not an estimate of how long a newly launched Julia session takes to
produce its first answer.

QPA is the first completed comparison in this section. Other programs will get
their own reports when their studies are complete. The
[comparison roadmap](../benchmark_suites.md#planned-competitor-comparisons)
lists planned studies; a place on that list is not a measured result.

## Reading the results

A speed ratio is the other program's elapsed time divided by TamerOp's time for
the same request. A ratio of 2 means that TamerOp took half as long. An aggregate
ratio summarizes the declared suite; individual requests can behave quite
differently, so each report also provides results by task and downloadable
measurements. Missing or incorrect answers never become infinite speedups.

Computing with retained maps is one part of TamerOp's pipeline. A comparison
of finite algebra does not also measure constructing a geometric encoding,
reading a point cloud, calculating a matching distance, or drawing a figure.
Each later study will identify the part of the pipeline it covers and the
question its measurements answer.

## How this section will grow

Each report will state its mathematical inputs and outputs, tested versions,
machine and resource limits, compilation and reuse policy, correctness checks,
timing results, memory measurements, failures, and available evidence. Completed
studies retain their dates and source identities; later measurements must be
identified as a new study or revision.

For the measurement protocol, see the [benchmarking manual](../benchmarking.md).
For choosing a bounded suite and interpreting its aggregate, see
[Designing a comparison that can finish](../benchmark_suites.md). The results
pages and their figures and data live together under `docs/benchmarks/`, so
they can be read on GitHub and incorporated into the documentation website
without the author's local benchmark directories.
