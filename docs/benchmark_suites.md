# Designing a comparison that can finish

A finite encoding is useful because it lets a researcher ask several kinds of
questions about the same module. Our comparisons should establish how quickly
TamerOp answers a representative, **finite list** of those questions. A study
also helps us find weaknesses worth fixing. Those aims need an endpoint: a
slower case is a result, not an instruction to keep expanding the experiment.

Read this guide with the [benchmarking manual](benchmarking.md). It governs
suite design and completion; the manual governs timing, resets, correctness,
and evidence. The numerical defaults below are TamerOp project decisions,
informed by the literature. They are not academic certification thresholds.

The [planned competitor comparisons](#planned-competitor-comparisons) below
record the libraries we intend to compare, their mathematical overlap, and the
dated effort estimates alongside the current completion status.

## Choose the claim before the suite

Speed is the primary objective, subject to correct answers and a declared
resource budget. The main comparison uses compiled code and uncached
mathematics. Memory is a secondary measure and an important diagnostic: a larger
footprint does not cancel a faster verified computation. An out-of-memory
failure is a failure to finish within the budget, however, and cannot be omitted.

Keep startup and retained-result queries separate from that computation claim.
A package may lead on computation and lose on first-answer latency. State the
regime once, prominently, and report both when measured; do not blend them into
a score whose weights have no intended workload behind them.

Write a proposed claim such as:

> TamerOp is generally faster than X for the finite-poset algebra workflows
> represented by suite S, once code is compiled.

The named suite defines the evidence. A short headline can be followed by its
balanced speedup, representative runtimes, tested versions, and a link to full
results. Counts of faster and similar cases provide secondary context.
It need not enumerate every exception. A whole workflow family with a consistent
loss belongs in the accompanying explanation, not only in a supplementary CSV.
“Better” without a criterion is less informative than “generally faster.”

Distinguish three levels of generalization:

| Evidence | Claim it can support |
| --- | --- |
| One task across varied examples | A workflow-specific performance advantage |
| A fixed suite spanning several relevant tasks and inputs | A broader empirical conclusion for that application area |
| Finite measurements on one or several machines | No proof of universal superiority, asymptotic dominance, or performance on every future input |

A curated suite is not a random sample of all research workloads. Timing
confidence intervals address execution variability conditional on the chosen
inputs. They do not give a probability that TamerOp wins on an arbitrary module.

## What established practice contributes

There is no test count or maximum matrix size in the following sources that
certifies every software comparison as sufficient. They supply tools for
defining and measuring an experiment:

- [SPEC's overview](https://www.spec.org/cpu2017/Docs/overview.html) uses defined
  suites, repeated runs, and geometric means of ratios. It also explains why a
  standard suite cannot replace a user's actual workload.
- [The GAP Benchmark Suite](https://arxiv.org/abs/1508.03619) specifies operations,
  input graphs, and evaluation methods together. This is a graph-processing
  benchmark, unrelated to the GAP computer algebra system used by QPA.
- [Kalibera and Jones](https://kar.kent.ac.uk/33611/) explain how variation at
  different levels of an experiment affects repetition and uncertainty. Many
  iterations in one process cannot replace independent process runs.
- [Hoefler and Belli](https://htor.inf.ethz.ch/publications/img/hoefler-scientific-benchmarking.pdf)
  discuss experimental design and statistically sound performance reporting.
- [Dolan and Moré](https://arxiv.org/abs/cs/0102001) introduce performance profiles
  for comparing solvers across a collection of problems.

Our practical interpretation is to freeze representative tasks, inputs, weights,
and budgets; measure uncertainty at the relevant level; publish all outcomes;
then close that suite version. Endless size increases do not solve the question
of representativeness.

## Use the same architecture for each competitor

Begin with a bounded review of documented public capabilities. Record the
natural overlap with TamerOp, choose the intended application area, and classify
the rest as deferred, unsupported, or semantically unresolved. A catalogue of
possible features is not a mandatory implementation backlog.

Choose roughly three to six workflow families. A family answers a recognizable
user question, such as constructing an encoding, computing an invariant,
analyzing a map, or obtaining a resolution. Several variants can belong to one
family without receiving extra headline weight. Group closely related output
modes before allocating the headline score: Hom bases, Ext dimensions and full
Hom/Ext queries can share one area, alongside module constructions, resolutions,
diagrams and complexes. Count task cases separately from mathematical areas.
A competitor with a genuinely narrow overlap may need fewer families.

Build the suite from these roles:

| Role | Purpose | Default boundary |
| --- | --- | --- |
| Mathematical controls | Verify known answers, empty cases, maps and conventions | A few per relevant family |
| Representative cases | Cover materially different structures and outputs | Two to four justified structural families |
| Scaling cases | Test sensitivity to size | Three fixed levels on selected families |
| Difficult cases | Retain known weaknesses, dense arithmetic or boundary cases | A declared small selection |
| Evaluation cases | Check behavior beyond examples used for tuning | About 15–20% of cases reserved before optimization |
| Startup and memory observations | Describe deployment costs and feasibility | Selected cases or existing compatible evidence |

These are design defaults, not a requirement to form a Cartesian product.
Use a small coverage matrix: exercise every primary task and major input class,
then select combinations that test plausible interactions, such as field and
module type. Explain omitted combinations. Prefer approximately 20–40 task
cases per family to an unbounded generator. An area with several necessary
request variants may use more cases while retaining the same headline weight.
State actual matrix and output sizes;
“large” should describe the measured range, not an invented universal scale.

Use genuine application inputs when available. If none are available, say that
the evidence is synthetic and justify the mathematical families through the
public APIs and intended use. Do not invent a research use or infer workload
frequency from the number of API functions.

Stop a size ladder at its declared final level or resource cap. A flat curve,
a loss, or no observed crossover is a valid result. Add another level only if
the original ladder failed a stated design purpose—for example, all levels
were accidentally identical—not merely because larger inputs exist.

## Freeze selection, then measure

A suite manifest must name every scored case, its workflow family, input recipe
or hash, requested output, primary timer, weight, and eligibility checks.
Separate existing development evidence from unmeasured evaluation cases.
Choosing a suite after seeing older results is not preregistration of those
results. Freeze new cases and scoring before their confirmatory measurements.

The pilot checks correctness, runtime feasibility, warmup and timing precision.
It can establish fixed repetition and batching counts. It must not select the
winner, remove unfavorable structures, or tune weights to the rankings.
Any pilot-driven change is recorded before the main manifest is frozen.

Default to five independent paired process passes, with three separately reset
samples per case and tool in each pass. A pair means both tools are measured in
one scheduled block, serially, with reversed or randomized order. A pilot may
justify a different fixed count. Declare it before confirmation; do not keep
sampling until an interval favors TamerOp. Unresolved timings at the budget
limit stay unresolved.

Use per-process medians. For uncertainty, preserve dependence among cases run
in the same process/block; do not treat all individual timer rows as independent.
One declared option is an approximate 95% t interval on the independent block
log ratios, with the pass range also reported. State the assumptions and small
sample limitations. A different uncertainty method is acceptable when justified
and frozen in the analysis plan. Such intervals quantify timing variation, not
the representativeness of the inputs or protection from systematic host load.

Existing compatible evidence earns credit. It need not be rerun because this
guide exists. A new aggregate requiring a common implementation or additional
replications may require a bounded confirmation run; preserve the historical
conclusion under its original protocol.

## Score speed without letting one easy family dominate

For completed correct pairs define the speed ratio as competitor time divided
by TamerOp time. Ratios above one favor TamerOp. Choose the primary timer before
measurement. When constructors place useful work on different sides of a query
boundary, directly measured construction-plus-query is a good primary choice.
Keep construction and query-only views alongside it, without counting the same
case as three votes.

Give workflow families equal weight unless a real workload distribution
justifies another choice. Within each family declare request-variant weights,
then field/input weights in advance. Several related query modes share that
family's weight even when they receive more cases. Replications, extra plot
points, and correctness-only controls do not increase that family's weight.
Report unweighted counts too.

The report needs:

- A practical interpretation led by TamerOp's demonstrated strengths, following
  the [presentation guidance](benchmarking.md#practical-significance-and-presentation).
- A geometric mean of ratios, with its exact eligible-case denominator.
- Results by workflow, field and size, including absolute times, meaningful
  savings, similar performance and the largest losses.
- Weighted win, near-tie, loss and unresolved shares, and correct completion
  rates, as supporting detail rather than a substitute for the effect sizes.
- A performance profile or equivalent distribution of ratios, plus the bounded
  scaling curves. A profile shows the fraction of cases a tool solves within a
  given factor of the faster tool; failures remain unsolved.
- Allocations and memory with their measurement scopes, outside the speed score.

A useful default practical margin is 10%: a ratio above 1.10 is a point-estimate
win, below 1/1.10 a loss, and between those bounds a near-tie. Report uncertainty
separately; an interval spanning these regions means the classification is
uncertain. Near-tie estimates do not establish statistical equivalence.
For timings near clock resolution, freeze a batch of independent, reset
computations in the pilot, or mark the measurement unresolved.

Timeouts and invalid answers do not have exact ratios. Retain them in the fixed
completion and coverage denominator. Report a speed-ratio bound only when an
operation-level timeout supports it. Never insert a whole-process timeout into
an operation ratio. Do not present the geometric mean of survivors as a score
for the full suite.

For the project phrase **“generally faster on this suite”**, the default
editorial gate is: at least 80% of the fixed weighted cases are practical wins,
at least four fifths of workflow families favor TamerOp, and the lower endpoint
of the aggregate timing interval exceeds 1.10. Require these conclusions to
survive the declared pass-range/weight sensitivity checks. This is our chosen
strength of evidence, not a literature-derived significance law. Failure to
pass it changes the wording, not the suite.

Missing or censored ratios require particular care. The default full-suite
gate applies when all primary ratios are valid. Otherwise report completion
and completed-pair performance separately, with a narrower conclusion unless
a predeclared conservative bound establishes the broader statement. Do not
promote incorrect answers to speed wins.

## Separate investigation from confirmation

Use development examples to diagnose TamerOp. Prioritize substantial elapsed-time
losses in common tasks, systematic losses across a family, or resource failures.
A fractional-millisecond loss may matter in a frequent batch; report the batch
that makes it important. Faster arithmetic matters more than explaining a
competitor's internals. Preserve our mathematical objects and intended outputs.

Prioritize meaningful elapsed-time gains on larger inputs. A small absolute
slowdown on a tiny fixture can be an acceptable tradeoff for substantial gains
on larger requests. Report its absolute cost and relative change, and check
any existing batch workload that could make the cost material. This guides
optimization decisions; it does not change frozen suite weights or remove losses.

There is no fixed limit on the number of TamerOp bottlenecks or optimization
cycles. Address every diagnosed major bottleneck where a feasible improvement
preserves correctness, required outputs and TamerOp's intended capabilities.
Freeze the candidate when the remaining diagnosed, feasibly removable costs
would yield only minor gains in the relevant complete workflows. Winning the
comparison, or fixing a predetermined number of problems, does not establish
that condition.

Judge a bottleneck by its measured or credibly estimated effect on a complete
workflow, including meaningful batches and repeated use. Do not dismiss a large
loss in one family because its suite-wide weight is small. Reprofile affected
workflows after substantial changes: removing one bottleneck can expose another.
Consider the combined opportunity in several small costs, rather than labeling
each minor while their total still offers a major improvement.

Keep a record of every diagnosed major bottleneck: affected cases, evidence,
estimated removable cost, attempted fix, correctness checks, before/after
measurements and remaining opportunity. If a major cost cannot feasibly be
reduced, document why—for example, unavoidable output work or a required
mathematical capability. Keep it labeled major; lack of time or exhausting a
measurement budget is not evidence that the cost is minor or unavoidable.
Use measured gains, profiles and justified bounds to support the stopping
judgment. This establishes a practical endpoint for diagnosed problems, not
proof that no undiscovered optimization exists.

The suite's tasks and size endpoints stay fixed. Bound individual diagnostic
runs and record their costs separately from the confirmation budget; a fixed
measurement budget does not impose a hidden limit on optimization cycles.
Once the stopping condition is met, freeze the candidate and evaluate reserved
cases. If evaluation exposes another feasible major improvement, resume the
optimization work and preserve that evaluation as evidence. Cases whose
performance has been viewed are no longer held out: use fresh evaluation cases
from the same declared families for the revised candidate, without expanding
the task or size scope or erasing earlier results.

Competitor implementations stay unchanged. A bounded documented-route check
and fair harness are required; competitor repair and redesign are prohibited.
An internal performance explanation or a win in every component is unnecessary.

## Decide when enough is enough

A suite version is complete when:

1. Every planned case has a recorded outcome and a correctness disposition.
2. All claimed timings satisfy the declared reset, compilation and output rules.
3. Fixed repetitions finish or reach their recorded budget; uncertain cases are labeled.
4. Reserved evaluation cases have been assessed once after the candidate freeze.
5. Tables, bounded curves, failure records and a scoped conclusion regenerate from evidence.
6. Reproduction instructions, source/input identities and an integrity manifest are checked.
7. Every diagnosed major bottleneck has been addressed where feasible, with evidence
   that remaining feasible gains are minor and any unavoidable major costs documented.

A completed study may establish leadership, mixed performance, or an unresolved
comparison. **Completion must not depend on winning.** A correctness defect
blocks the affected claim until corrected and reverified; it does not license
deleting the case. A budget-limited measurement report can close with a narrower
claim, but the optimization objective remains incomplete if diagnosed major,
feasible improvements are still outstanding.

Do not reopen a finished version because another size, field or operation is
possible. Reopen for a demonstrated validity problem; use a new version for a
meaningful implementation change, a new real workload or an explicitly enlarged
claim. Record which of those reasons applies. Move to the next competitor when
the scoped study is complete.

## Planned competitor comparisons

Roadmap recorded 2026-09-30. **All 21 libraries below are in the comparison
programme**; the status below distinguishes completed studies from planned work.
Each comparison should help us understand a part of the finite-encoding workflow: constructing a
module, retaining and analyzing its maps, or obtaining an invariant or figure.
Specialized tools also give useful comparisons for direct filtration and
barcode tasks that do not require constructing an encoding.

The table preserves the order of **estimated remaining work, from least to
most**, as of 2026-09-30, before QPA v1 and PHAT v2 were completed. This includes
adapters, independent correctness checks, warmup/reset verification, timing and
reporting for the intended overlap. It credits existing work rather than
estimating every study from scratch. It is a planning judgment, not a measured
effort ranking or a ranking of importance, speed, or total library size.

Positions within a few places of one another are approximate. Unknown TamerOp
optimization work is not predictable from the competitor's feature list. Each
study still needs its bounded charter; these ranks do not set case counts,
machine-hour budgets or deadlines. The C01–C21 identifiers stay fixed when the
effort order changes.

**Status (2026-10-05):** C01 (QPA v1), C18 (PHAT v2) and C14 (Ripser.py v1)
are complete at their fixed scopes. The [PHAT report](benchmarks/phat.md)
covers 48 medium/large inputs across sixteen structural variants. The
[Ripser.py report](benchmarks/ripser.md) covers 31 development requests and six
separately reported reserved evaluation cases, with four paired process passes.
The other comparisons remain planned under this guide. “Planned” does not erase earlier exploratory measurements: inventory
and credit compatible evidence before deciding what remains to run. The scope
column identifies candidate common requests, not a completed certification that
every listed operation is already comparable in both implementations.

| Effort rank | ID | Library | Intended comparison scope | What must be aligned before timing |
| ---: | --- | --- | --- | --- |
| 1 | C18 | [PHAT](https://bitbucket.org/phat-code/phat/) | Supplied filtered boundary matrices to complete ordinary barcodes over F₂. | Same matrix and ordering; match finite and essential intervals, tied grades, and all supplied homology degrees; charge native and output conversion. |
| 2 | C14 | [Ripser.py](https://ripser.scikit-tda.org/en/latest/) | Vietoris–Rips persistence from point clouds and dense/sparse distances, weighted vertices, landmark selection and supported cocycles. The selected comparator uses the Ripser engine through Ripser.py's public interface. | Same metric, threshold, field and homology degrees; keep barcode-only and cocycle-retaining requests separate. The intended final comparison includes supported representative outputs, matched by field, scale and class checks rather than literal coefficient equality. A barcode-only request does not require a full encoding. |
| 3 | C16 | [PersistenceDiagrams.jl](https://mtsch.github.io/PersistenceDiagrams.jl/stable/) | Diagram distances/matchings, landscapes, images, Betti curves and plotting. | Same diagrams, endpoint/essential-bar conventions, feature parameters and sampling grid. |
| 4 | C04 | [FlangePresentations.jl](https://gitlab.com/flenzen/flangepresentations.jl) | Construct flat-injective presentations of finite-dimensional persistence modules from free resolutions. | Match supported module classes and starting information; verify represented spaces and maps, and charge any required resolution construction. |
| 5 | C07 | [mpfree](https://alexanderrolle.github.io/) | Bifiltered chain complexes to minimal presentations of persistent homology. | Match homology degree, coefficient field, grading and minimality; validate the presented module. |
| 6 | C08 | [2pac](https://gitlab.com/flenzen/2pac) | Two-parameter persistent cohomology and supported presentation/resolution outputs. | Fix homology/cohomology conversion, grading, field and requested resolution range. |
| 7 | C15 | [Ripserer.jl](https://mtsch.github.io/Ripserer.jl/dev/) | Rips, cubical and alpha persistence, including supported representative cycles/cocycles. | Match filtration and representative type; exclude compilation for both Julia implementations. |
| 8 | C17 | [Dionysus](https://mrzv.org/software/dionysus2/tutorial/persistence.html) | Persistent homology/cohomology and representatives; zigzag or vineyard tasks where both tools support the same request. | Match filtration events, directions, field and output semantics; record any currently unsupported branch explicitly. |
| 9 | C05 | [RIVET](https://rivet.readthedocs.io/en/latest/preliminaries.html) | Two-parameter Hilbert functions, bigraded Betti data, fibered barcode construction and slice queries; visual exploration. | Same module, grades and slice conventions; separate construction from new slice queries and GUI assessment. |
| 10 | C06 | [Hera](https://github.com/anigmetov/hera) | Bottleneck/Wasserstein distances and two-parameter matching-distance approximation. | Match normalization, essential bars, parameter domain and error guarantee; distinguish approximate matching from TamerOp's finite-window exact request. |
| 11 | C19 | [rhomboidtiling](https://github.com/geoo89/rhomboidtiling) | Planar/3D rhomboid tilings, higher-order Delaunay mosaics and multicover bifiltrations. | Match radius/depth conventions, depth cap, degeneracy assumptions, grades and sliced/unsliced construction. |
| 12 | C01 | [QPA](https://gap-packages.github.io/qpa/) | Finite-poset Hom/Ext, products and induced maps, module and tensor constructions, resolutions, diagrams and complexes. | Preserve the existing 220-case suite, categories, outputs, fields, resets and closure rules. |
| 13 | C21 | [MPIntInv](https://github.com/Enhao-Liu/MPIntInv) | Interval-based invariants and multiplicities of finite-grid representations; supported filtration-to-invariant workflows. | Match interval family and compression convention, finite-grid category and signed multiplicities; identify external backend work. |
| 14 | C12 | [SageMath](https://doc.sagemath.org/html/en/reference/quivers/index.html) | Quiver representations and Hom spaces; [chain complexes and homology](https://doc.sagemath.org/html/en/reference/homology/sage/homology/chain_complex.html). | Check support for required relations: a quiver alone does not enforce poset commutativity. |
| 15 | C03 | [Persistence-Algebra](https://github.com/JanJend/Persistence-Algebra) | Graded presentations, minimization, kernels, Hom maps and supported free resolutions. | Match grading and presentation semantics; begin with its documented F2 coefficient support. |
| 16 | C20 | [AIDA](https://github.com/JanJend/AIDA) | Indecomposable decomposition and supported presentation/decomposition tasks. | Establish an actual common output first; an exact indecomposable decomposition is different from an interval approximation or signed invariant decomposition. |
| 17 | C13 | [GUDHI](https://gudhi.github.io/) | Filtered-complex construction and persistent homology from point clouds, images and supplied complexes. | Match filtration definition, threshold, field and barcode/representative output; charge input conversion. |
| 18 | C10 | [Macaulay2](https://macaulay2.com/doc/Macaulay2/share/doc/Macaulay2/Macaulay2Doc/html/___Module.html) | Graded module constructions, free resolutions, Hom, Ext, Tor and homology. | Match polynomial-module interpretation, grading and degree components; do not substitute finite-poset derived groups. |
| 19 | C11 | [OSCAR](https://hechtiderlachs.github.io/presentation_SIAM25.pdf) | Multigraded modules, complexes, Hom/tensor constructions and resolutions; Julia-to-Julia algebra. | Match category and requested terms/maps; materialize lazy outputs and exclude compilation in both tools' computation timings. |
| 20 | C09 | [homalg](https://homalg-project.github.io/homalg_project/Modules/doc/chap10_mj.html) | Hom, tensor, Ext, Tor, complexes and operations on maps. | Identify the same module category and functors, with supported coefficient rings/backends and explicit comparisons of maps. |
| 21 | C02 | [multipers](https://davidlapous.github.io/multipers/index.html) | Data-to-invariant workflows: multifiltrations, slices, Hilbert/rank/Euler signed measures, approximations, features and visualization. | Same filtration, grid choices, coefficient field, requested invariant and approximation accuracy. |

The rough effort bands are **1–4: lower**, **5–9: moderate**, **10–16: higher**,
and **17–21: largest**. Narrow supplied-matrix or barcode requests tend to have
simpler answer comparisons. Presentations and resolutions require stronger
checks that the same module and maps are represented. Geometry, exact versus
approximate distances, and broader algebra workflows add more conventions and
validation work.

QPA is ranked 12 for *remaining* effort: its existing harnesses and verified
algebra save substantial work. That ranking preceded completion: QPA v1 is now
closed with the 2026-10-03 confirmation below. Its fixed cases need not be
expanded merely because other inputs exist. multipers is ranked 21 because of the
breadth of the planned filtration-to-invariant and feature workflows. Existing
ingestion/invariant comparison scripts are a starting point; review their
versions, output contracts and reset behavior before counting them as completed
work under this manual. No old timing is discarded merely because the roadmap
has changed.

AIDA (16) and MPIntInv (13) have particularly uncertain positions. Their
documented decomposition and interval-invariant outputs first need to be matched
to actual TamerOp capabilities. These ranks allow for that qualification work;
they do not assume that TamerOp already implements all those outputs or authorize
new mathematical features solely to make a comparison possible. If no equivalent
request exists, record the capability difference and the supported subset rather
than treating missing functionality as an indefinitely large benchmark task.

### Ripser.py development scope, 2026-10-04

C14 begins with **24 fixed development cases**, using Ripser.py 0.6.15
compiled with its documented Robin Hood hashing option and `-O3`. That is a
verified supported optimized build; it is not a claim that every possible
compiler configuration has been ranked. Both algorithms remained unchanged
during the pilot. **The pilot is complete:** all 24 cases completed both paired
process passes, with 288 accepted timed queries and 48 matched output checks.
No case timed out or exceeded its memory limit. Superseded adapter-development
measurements are excluded from the corrected baseline.

The six groups are native Euclidean point clouds, supplied dense distances,
supplied sparse distances, weighted vertices, landmark selection with its
coverage information, and small mathematical controls. Each group has four
cases. The fields are F2, F3 and F101; requests mostly include H0 through H1
or H2, with one explicit H3 control. Barcode-only requests and requests that
retain cocycles remain distinct. The largest inputs include 1,024 dense
points, 4,096 sparse/weighted vertices, and 128 landmarks from 2,048 points.

The primary timer covers the complete public query with compiled code and
fresh mathematical state. Native-container construction is recorded separately;
construction-plus-query is their measured sum. Two serial process passes
reverse tool order and collect three accepted samples per case and tool.
Julia compilation-contaminated samples are retained but excluded. Every worker
has an 8 GiB RSS ceiling, a 180-second phase limit and a 420-second startup
allowance. Failures stay in the record.

Ripser.py converts filtration grades to Float32. Supplied matrices therefore
use exactly shared representable grades, while native-cloud requests form a
separate numerical lane with an explicit rounding tolerance. Complete interval
multisets, essential bars and landmark metadata are checked. Independent
boundary reduction and graph-homology formulas provide additional controls.
Selected cocycles are checked as classes, including common class-space checks
on small examples; coefficients need not agree. Large connected H2 examples
still need stronger independent class identification before a broad claim
about all representative outputs.

Separate TamerOp profiles identify substantial costs in dense coface/queue
processing, landmark distance validation, and H2 clique catalogs. These are
development observations, not the final public comparison. Address the measured
major costs and settle a finite final suite.
Reserve genuinely untouched evaluation inputs before optimizing; the observed
pilot cases cannot become held-out evidence. Do not expand into unrelated
ordinary-persistence variants merely to enlarge this comparison. Keep the
existing case and resource boundaries until a final charter explicitly
replaces them.

The October 5 development rerun completed the same 24 cases in both process
passes after optimizing complete-graph distance lookup, F2 queue storage,
triangle catalogs and dense distance validation. All 288 timed samples were
compilation-free, all 48 matched output checks passed, and TamerOp's 48 exports
matched the previous baseline exactly. Focused Julia verification passed
1,111,406 checks. The largest dense circle improved from 29.2 s to 17.7 s;
the 512-point cocycle request from 2.72 s to 1.55 s; and the largest landmark
request from 46.2 ms to 21.8 ms. Planar H2 allocation traffic fell from 178 MB
to 76.8 MB, while its runtime improvement was modest and variable. Ripser.py
remains faster on that planar request, landmarks and several small controls.
The six evaluation recipes reserved before these changes remain unmeasured;
this is still development evidence, not a final confirmatory suite.

A further staged development study on seven existing requests implemented
compact reduction columns, shared edge ordering, a certified terminal radius,
reconstructed apparent pairs and compact triangle catalogs. New reversed-order
before/after passes measured 1.57 s to 0.219 s and 8.46 s to 0.757 s on the
256- and 384-point planar H0-H2 requests over F101; the 2,048-landmark workflow
fell from 4.41 s to 1.80 s. The final cohort contains 96 compilation-free samples.
All 31 existing development inputs matched their saved complete barcodes and
landmark exports, with selected cocycle witnesses also preserved. Independent
signed-zero loop and sphere checks caught and corrected an apparent-pair
ordering defect during development. Later unchanged-Ripser checks passed on
all seven timed requests; the tiny weighted-cycle control still favors Ripser.
Host activity varied, so these remain development observations. The six
reserved evaluation recipes remain untouched, and the final comparative
suite is still to be settled.

The remaining 24 of those 31 development requests have now been retimed on
that same accepted source in two reversed-order paired passes. All 288 samples
were compilation-free, all 48 cross-tool checks passed, and all 48 TamerOp
exports matched the preceding verified outputs. The 512-point planar H0-H2
request takes 1.85 s versus Ripser.py's 8.35 s; the 1,024-point circle takes
12.22 s versus 20.21 s. Landmark results remain mixed: 4,096/3,072 takes
4.05 s versus 5.15 s, while 2,048/128 takes 14.89 ms versus 5.27 ms and
4,096/512 takes 141.77 ms versus 101.51 ms. These 24 requests and the seven
earlier timings remain separate development cohorts, with no pooled final score.

Current-path profiles on four existing workloads identify substantial remaining
costs in large-circle queue removal, landmark input checking/selection, H2
catalogs and coface processing. Several preparation costs also add up on the
many-landmark workflow. The review is complete, but optimization closure and a
candidate freeze are not justified yet. No package implementation changed in
this review. The large-component H2 representative gap and the six untouched
evaluation recipes remain separate pending work; no final comparison was run.

A subsequent four-target optimization retained early equal-grade coface search,
fused landmark finiteness checks and cheaper graph/clearing preparation.
Its 31-case source-snapshot cohort passed all output checks, but a Zoom meeting
ended just before the final baseline; those time ratios are retained as
confounded development observations. A separate confirmation then alternated
both sources on the same CPU core in two independent process pairs, covering
six existing cases with 72 compilation-free samples. The 1,024-point circle
changed from 15.7 s to 13.4 s; the 4,096-point / 2,048-landmark query
changed from 2.26 s to 1.55 s. These are matched TamerOp
before/after measurements, separate from all earlier Ripser comparisons.

The canonical focused tests passed 1,133,859 assertions; all saved barcodes,
landmark outputs and selected cocycle exports were preserved across the 31
inputs. Four-child and bottom-up heaps, custom radix sorting, native QuickSort
and colex enumeration with stable grade sorting did not justify replacement.
That round retained the existing heap and catalog sort, with major costs still
visible in profiles. The large-H2 representative gap, reserved evaluation inputs and
final comparison remain pending.

A later follow-up retained fused landmark selection and native-float validation,
certified-radius coface neighbor lists, and sorted F2 batches with reusable
buffers and in-place sorting. Actually pruned neighbor graphs retain the
compact F2 heap; odd-prime reduction remains unchanged. Applying batches
everywhere slowed the largest landmark request, so that broader routing was
rejected, along with the triangle-capacity experiment.

The selected implementation passed 132 compilation-free samples in two
independent, reversed-order source comparisons, plus all 31 saved development
output checks. The 1,024-point circle changed from 11.2 s to
5.41 s; the 4,096-point / 3,072-landmark query changed from
4.37 s to 4.03 s. Host activity varied: these are paired
development observations, separate from prior comparator timings. Large
landmark queries allocate about 7-8% more; circle queries allocate less. Canonical
focused Rips, A115 and A117 tests passed on the integration snapshot, preserving
concurrent project changes. The final comparison and reserved evaluation work
remain pending.

### QPA results and development record

Read the [public QPA benchmark report](benchmarks/qpa.md) for the final
results, machine specifications, figures, and downloadable data. The
[results index](benchmarks/index.md) collects the completed QPA and PHAT
comparisons. This section retains the development record.

**QPA v1 is complete as of 2026-10-03.** All 220 fixed requests were attempted
in five paired passes. TamerOp returned independently verified answers for all
220; QPA returned matching answers for 200. QPA's elementary-tensor evaluation
failed on the other 20 requests. Those cases remain in the suite with no speed
ratio. The paragraphs below retain the development history; the final results
appear at the end of this section.

As of 2026-09-30, **all 220 task adapters are implemented**. All 96 Q1–Q5
cases and all 80 Q6–Q7 cases have passed mathematical verification in both
tools. Q6–Q7 covers 20 pushouts, 20 pullbacks, 20 homology requests and 20
mapping cones over QQ, F₂, F₃ and F₁₀₁. Its checks compare actual canonical
maps, homology structure maps and signed cone differentials, with explicit
isomorphisms between the two tools' outputs. Reserved inputs were checked
without timing them. Sixteen development examples also passed two timing
protocol checks: 192 measured rows across the three phases, with zero
compilation in every accepted Julia sample. These are protocol checks, not
the full-suite performance campaign.

The bounded development pilot has now run on 52 cases: one for each of the
thirteen requests over four fields, selected by input size. It produced 100
verified native answers and 48 matched mathematical outputs. There are 46
comparisons with matching construction-plus-query timing passes; 38 have two
passes and eight have one. All 551 accepted timing rows passed reset checks,
and every accepted Julia row recorded zero compilation and recompilation.
The 90-minute cap left the second rational pass incomplete. Four QPA tensor
errors and two requests without matching timing passes remain explicit gaps.
No reserved cases were timed.

These are development diagnostics on a shared machine. That pilot used a
nonmonotonic GAP clock; the subsequent configured-runtime preflight described
below resolves this prerequisite for future confirmation without changing QPA.
Final verification was separately bounded and recorded after the timing cap.

Development profiling covers seventeen existing cases and all thirteen request
types. Its first three improvements are now implemented: shared exact products
for rational representatives, request-local reuse of Yoneda lifts, and shared
products in composition, induced maps and validation. `ExtAlgebra` uses the same
public table request. Focused correctness checks cover exact/numerical fields,
signs, target-resolution transport, structured storage and invalid inputs.

A before/after development study independently verified all nineteen selected
answers and 228 main timing rows. The rational cube Yoneda table improves by
23.8–27.3× and the rational grid table by 2.83–4.20×. Dense first use and several
small requests regress; the kernel-only follow-up does not explain those losses.
The final baseline hit its time cap during the last dense repetitions: all main
cases have two complete passes, and four dense homology cells have only one
complete paired pass. Keep those limitations and regressions explicit.

A subsequent timing-only follow-up uses symmetric workers and records 380
accepted samples, with exact answers, verified resets and zero compilation.
The rational cube gain remains 22.9–24.4×; the complete grid Hom/Ext request is
1.11–1.16× faster. The small F3 product adds 0.18–0.65 ms, an accepted tradeoff
for this small request under the stated priority for larger workloads.

Two size-16 dense cells remain slower in the standalone runs. Separate stage
profiles and interleaved rollback checks do not consistently attribute those
losses to the new multiplication. A final CPU-time diagnostic puts the scalar
cost near parity to 6.6% higher, while batch8 is at parity or better. This does
not erase the standalone losses or establish their hardware/runtime cause.
The bounded regression investigation retains the implementation without a
speculative gate; the recorded uncertainty remains for fixed-suite confirmation.
The next implementation replaces rational Hom's repeated sparse RREF updates
with integer echelon rows and back substitution, and reuses native prime-field
coordinate factors on the owning result. Two further process pairs verified
520 warm-uncached samples and ten independently checked, identical answers.
The large-coefficient rational Hom request improves by 1.39–2.08× with 64% fewer
allocated bytes; F101 Yoneda tables improve by 1.72–2.90× with 40% fewer allocated
bytes. The full F101 tensor request ranges from near parity to 1.62× faster.

Focused verification passed 7,863 maintained-runner assertions, including
strict products and checked prime coordinates. Another 240 diagnostic rows
separate prime first use, retained queries and result-owned storage.

Those gains do not erase the controls: a size-16 rational cohomology scalar
request adds 2.23–2.27 ms in both pairs, and several other results are mixed.
Keep the dense loss as an unresolved observation, without assuming its cause
or adding an unmeasured threshold. These are TamerOp before/after results on a
shared host. M3 remains open, and no new QPA comparison is claimed.

A subsequent reprofile revisits the original seventeen development cases across
all thirteen request variants. Two timing processes and two separate sampling
processes completed within a forty-minute worker budget. All 570 timing samples
passed reset and zero-compilation checks; all seventeen workflow answers passed
independent mathematical verification. The existing dense and prime-coordinate
controls remain included, with no reserved timings or suite expansion.

The strongest remaining leads are rational factorization, checked F2 coordinate
application and temporary arrays in exact particular solves. Dense rational
scalar setup takes 18.79–21.64 ms; full RREF used for pivot-only selection is a
concrete target. Most smaller-request profiles have few snapshots, so these
are optimization candidates rather than measured removable costs. Retained
32-column F101 queries cost 0.19–0.33 ms on the size-16 controls; that alone does
not justify reversing useful factor reuse. Shared-host variation, empty profiles
and earlier losses remain explicit. No new production change or QPA ranking is
claimed, and M3 remains open.

The next bounded implementation removes full RREF from rational pivot-only
selection, removes temporary arrays from exact particular solves, and shares a
typed packed F2 application routine. All 9,237 focused assertions pass. Two
process pairs provide 396 verified-reset, zero-compilation main samples, with
all seventeen answers independently checked; 1,320 separate kernel samples
also pass. The size-16 rational factor kernel improves by 1.38–1.39× with about
26% fewer allocated bytes. Complete F2 and F101 product requests show observed
ratios of 1.53–2.32× and 1.20–2.00×; dense homology batch8 improves by 1.24–1.25×.

The dense cohomology scalar control is 0.99 ms slower in one pair and near
parity in the other; several other timings are mixed. Shared-host drift is
visible even on lightly affected controls, so the exact workflow ratios are
not isolated causal estimates. Stable kernel gains and reduced allocation
traffic support retaining these simple changes without adding caches or
thresholds. The initial kernel-driver logging failure and corrected rerun are
retained separately within the declared budget. This does not close M3 or
replace fixed-suite confirmation.

The subsequent residual-cost review collects 42 profiles with at least 128
snapshots each, then tests existing exact arithmetic routes in disposable
processes. All seventeen workflows and ten dense controls pass independent
checks; 336 feasibility samples pass reset and zero-compilation checks. The
size-16 dense coordinate controls improve by 2.50–4.88× when native exact
factorization and multiplication are combined, with conversions charged and
identical factors and coordinates. These are experiments, not implemented
package improvements or QPA speed ratios. Shared-host variation remains visible.

That review identified selective dense rational factor/product routing as the
remaining major target. Tiny-map and column-space changes showed modest or
inconsistent benefits; sending every small dense RREF through Nemo generally
lost. No additional major removable prime-coordinate or sparse-Hom cost was
demonstrated. The combined dense opportunity kept M3 open at that stage.

Selective routing is now implemented, including eager coordinate construction
and later queries. Dense factors use Nemo when both the whole matrix and its
leading square block are sufficiently dense; simple identity-block embeddings
stay native. Dense matrix products charge all conversions each call. Existing
factor storage, sharing, selected bases, checked membership and explicit backend
choices are preserved. Sparse and structured storage retain their native paths.
No new cache is introduced.

The final focused run passes 10,565 assertions. Two process pairs produce 492
accepted, zero-compilation samples, with verified resets for fresh results and
separate retained-plan timings. Independent checks verify all seventeen workflow
answers and sixteen dense formulas. Complete size-16 controls improve by
2.07–4.06×; retained checked queries improve by 1.45–2.06×. Dense controls supply
cycle and boundary bases; their timers include result construction and the
requested answer, not deriving those bases from a complex.

The final 600 kernel samples support the selection rules; an initial 300-sample
pilot and its identity-block regression remain recorded separately. Size-4
complete controls add at most 0.026 ms in the observed pairs. Other workflows
are mixed, with a largest observed addition of 0.517 ms in one pair. Unchanged
prime-field timings also drift, so their apparent gains are not credited to
rational routing. Reachable storage is unchanged on small controls and smaller
on the larger controls. Julia allocation counters omit FLINT allocations, and
no process-memory improvement is claimed. These are development diagnostics on
a machine without exclusive reservation, not confirmation confidence intervals.

The acceptance review of that version found two further feasible improvements
on existing controls. Sixteen dense profiles and 384 accepted, zero-compilation
samples support them. Using the
selected dense product for the complete exact membership check improves
32-column requests by 2.21–2.38×. Trying the leading square inverse before
general row selection, with fallback when singular, improves scalar requests
by 1.59–1.71×. Their combined full-request gains are 1.58–3.59×. Factors and
answers agree exactly. These are disposable experiments, not implemented
improvements or QPA speed ratios. Small controls and unfavorable results remain
included. That review kept M3 open until both changes were implemented and
validated.

The two follow-up optimizations are now implemented and verified. Eligible
rational batches use the selected exact product for their complete membership
certificate; explicit Julia-only solves keep their chosen backend. Eligible
dense factors try the leading square inverse and retain general row selection
when it is singular. Exact rows, coordinates, checks and storage are preserved.

The focused runner passes 11,133 assertions. Independent checks verify seventeen
workflow answers and sixteen dense formulas; 492 accepted complete/retained
samples and 264 kernel samples have zero timed compilation and verified resets.
Complete size-16 requests improve by 1.35–3.91× in two process pairs. Separately
retained batch8 queries improve by 3.47–5.02×; retained scalar queries remain
approximately unchanged. Reachable mathematical storage is unchanged for every
dense and retained control. Julia allocations exclude native FLINT memory.

Tradeoffs remain visible. A singular-leading factor adds 0.261–0.842 ms before
fallback, and rejecting an invalid first entry in a wide RHS rises from about
23 μs to 1.15–1.41 ms because the dense certificate computes the full product.
Size-4 complete controls add at most 0.234 ms. Other workflows are mixed, with a
largest observed addition of 2.080 ms. QQ Hom and kernel/image/cokernel cases
lose in both pairs; this study does not establish the cause. These small absolute
losses remain in the evidence and do not by themselves require another campaign.

The two demonstrated dense opportunities are closed. Residual sampling records
conversion, native exact arithmetic and remaining narrow certificates; sampled
cost alone does not demonstrate a further removable bottleneck or its combined
benefit. The 2026-10-03 acceptance review closes M3 for the fixed suite and
freezes this tested candidate. It accepts the documented small absolute losses
in exchange for the larger dense gains. No diagnosed feasible major opportunity
remains unresolved in the reviewed evidence. This is an empirical stopping
decision; it does not establish that every possible future optimization is minor.

A candidate image-preparation attempt reached its remaining budget. Both timing
variants used existing usable images or source loading, warmed methods, and
rejected timed compilation; these results make no startup claim. A later tool
session interrupted one candidate worker. Its partial rows remain separate,
its full allowance was charged against the original budget, and a replacement
completed. The machine was not exclusively reserved; all raw variation and
unfavorable cases are retained. No final QPA or reserved-case timings were added.

The GAP timing prerequisite is now resolved with a disclosed configured build
of unchanged GAP 4.16.1 and QPA 1.37. Its existing monotonic clock passes two
fresh-process checks. Across all thirteen request variants in QQ and F3, 24
old/new request pairs return identical independently verified answers; two
retain the same declared tensor failures. The benchmark adapters reject the
old nonmonotonic clock before collecting timings. The configured executable,
source hashes, compiler flags, failed attempts and preflight evidence must be
retained with the environment record. These acceptance checks established
runtime and harness readiness; the final comparison follows below. The local candidate archive
pins the tested source, dependency/configuration files, all 220 tasks, adapter
sources and runtime identities. It includes three runner-compatible candidate
records and a restorable source snapshot. The 184 development and 36 reserved
cases, weights, endpoints, resource limits and scoring rules are unchanged.

#### Final confirmation, 2026-10-03

The fixed campaign is complete. Strict package-image preparation and a fresh
strict load passed before timing; the frozen source and configured monotonic
GAP clock were checked before and after the campaign. All 120 workers finished
within their limits. Independent exact checks verified 220 TamerOp answers and
200 matching QPA answers. All 19,200 accepted timing rows pass the reset checks,
and accepted Julia samples record zero compilation or recompilation. All 36
reserved evaluation cases were assessed after the freeze.

For the 200 requests completed correctly by both tools, TamerOp is faster by
more than the 10% practical margin in every pass. The original-weight geometric
mean of QPA/TamerOp construction-plus-query time is **135.70×**, with an
approximate paired-block 95% interval of **134.64–136.77×**. Uniform-case
weighting gives 129.08×. The smallest primary point ratio is 1.76×. These are
results for the completed pairs, covering 90% of the original fixed weight.

| Requested answer | Matched cases | Weighted QPA/TamerOp time |
| --- | ---: | ---: |
| Complete Hom basis | 12 | 3.60× |
| Ext¹ dimension | 12 | 292.93× |
| Complete Hom/Ext query | 12 | 308.10× |
| Kernel, image and cokernel | 20 | 78.40× |
| Projective/injective resolutions | 40 | 739.15× |
| Pushouts and pullbacks | 40 | 23.48× |
| Homology and mapping cones | 40 | 132.87× |
| Yoneda product tables | 12 | 601.71× |
| Induced Ext maps | 12 | 1387.50× |
| Balanced tensor with elementary evaluations | 0 of 20 | No ratio: QPA evaluation errors |

Construction and query are also measured separately: their completed-pair
geometric means are 3.94× and 201.62×. Construction alone is more variable;
eight cases favor QPA in at least one pass. These do not change the directly
measured complete-request results. The 32 matched reserved cases give 111.82×;
the other four reserved cases are among QPA's tensor failures.

The full-suite numeric speed criterion remains unmet solely because those 20
requests have no valid paired ratio. With the unchanged full weights, the
outcome is 90% practical wins and 10% unresolved comparisons. The study closes
with a verified advantage on all 200 completed pairs and a separate completion
advantage; failed requests are not infinite speedups.

This comparison starts from the declared prepared finite category and decoded
input. Category construction, scalar decoding, package loading and compilation
are outside the primary timer. It is a comparison of compiled computation with
uncached mathematics over QQ, F₂, F₃ and F₁₀₁, not startup latency or every
possible module. The five-pass interval describes variation on the fixed
synthetic suite and this shared host. It does not establish a hardware- or
population-wide guarantee.

Peak sampled process RSS was 1494.5 MiB for TamerOp and 138.7 MiB for QPA.
These include runtime and temporary storage; retained mathematical storage was
not remeasured, and Julia's allocation counter excludes native FLINT memory.
The new campaign and initial report used 9.46 hours of the remaining allowance,
within the original combined 15-hour cap. Raw outputs, failed tensor attempts,
plots, independent calculation checks, source/runtime identities and integrity
checks are archived locally. A portable public reproduction bundle remains a
separate release task. M4 and M5 are complete; QPA v1 is closed without changing
its tasks, size bounds, weights or competitor implementation.

### PHAT implementation and development pilot, 2026-10-03

The development and freeze entries record their status at the time. PHAT v2
has since [completed final confirmation](#phat-v2-final-confirmation-and-closure-2026-10-04);
the [result report](benchmarks/phat.md) gives its current conclusion.

The comparison asks for the complete ordinary barcode of a supplied
filtered chain complex over F₂. This exercises TamerOp's direct
[ordinary-persistence route](ordinary_persistence.md): it does not require
constructing a general finite encoding to answer a one-parameter question.

The fixed suite contains 48 cases across graphs, simplicial complexes, cubical
complexes and algebraically constructed chain complexes, with three sizes and
four variants per family. Ten cases are reserved for evaluation after tuning.
Eight additional small controls check known answers, including essential
classes, higher homology, equal-grade events and empty input.

The adapters use unmodified PHAT v1.7's default twist reduction and TamerOp's
public barcode API. An independent checker derives barcodes from the ranks of
homology maps and verifies every input differential. Native construction, the
complete query, and construction plus query are measured separately. Each
request starts from verified fresh mathematical state; Julia measurements with
compilation are excluded from the computation summary. All 56 inputs matched
the independent oracle in TamerOp and in PHAT's twist and standard reductions. The 12-case development
pilot completed two process passes per tool with 432 accepted timing samples.
PHAT was faster on every pilot case for construction plus the complete query
under the declared pilot conditions. Development profiling has since identified
grade-conversion dispatch, column-reduction allocation, and native sparse
conversion as optimization targets. Those three changes are now implemented and
verified. On 16 existing development cases, construction plus the complete query
improves by 1.56–2.20× under the original collection policy, allocating 37–95%
fewer Julia bytes. Batches of fresh computations with normal allocator reuse
improve by 1.65–3.14×; they do not reuse previous barcodes. All 56 barcode checks
and 112 sublevel/superlevel representative comparisons agree with the baseline.
The constructor and ingestion checks passed alongside the ordinary-persistence
suite; a final run on the measured package image passed 7,793 ordinary-persistence
and typed-constructor assertions.

Some query-only cases have small absolute regressions, retained in the evidence.
Unchanged PHAT had the lower combined median in that refreshed 16-case
single-request comparison. Its collection policy and the subsequent study's
policy are recorded separately.

A second development study implemented clearing, a reusable binary-column
workspace, filtration-position indexing, faster exact chain validation, and a
graph route based on component merging and cycle births. Representative-producing
requests retain their selected cycles and filling chains. The final ordinary
owner suite passed 13,076 assertions; all 56 independent-oracle barcodes and
112 retained-representative fingerprints matched the previous implementation.

This follow-up measured the same 16 development inputs and a fixed supplement
of twelve larger inputs, with two process passes and five accepted samples per
phase. Full collection occurs before phase warmup; natural collections within
requests are charged, with fresh mathematical state for every computation.
Construction plus query improved on all 28 cases: geometric mean gains are
2.17× on the original inputs and 6.60× on the larger inputs. The largest
simplicial computation fell from 38.48 s to 6.62 s. These are changes relative
to the already improved implementation, not the original development pilot.

Against unchanged PHAT under this protocol, larger graph cases favor TamerOp
by 3.22–6.09×. Larger simplicial cases are near parity to modestly faster
(1.05–1.24× by median); PHAT remains faster on the larger cubical and algebraic
cases and on 15 of the 16 original small cases. The larger simplicial/cubical
inputs have checked chain identities, cross-tool barcodes and known terminal
Betti numbers, but lack a full independent barcode oracle. These are development
observations, not a general performance claim. Julia's peak process RSS did not
improve; input and answer retained sizes are unchanged.

A focused follow-up on 2026-10-04 confirmed column-addition and integer-ordering
costs. Whole-word addition for sufficiently dense pivot columns, local pivot
search, and stable integer sorting improved complete requests on 27 of the 28
existing development cases. Geometric mean gains against a fresh baseline are
1.20× on the original cases and 1.41× on the larger cases; the largest simplicial
case fell from 7.29 s to 4.53 s. The one small regression adds 1.9 microseconds.
The same adapter is used before and after: construction experiments and a SIMD
validation hint were not adopted because gains were inconsistent. All 13,696
focused assertions, 56 original oracle barcodes, and 112 retained-representative
comparisons passed. PHAT was not retimed in this follow-up, so it supplies no
new contemporaneous competitor ratio.

A subsequent bottleneck review retained this measured candidate. Two residual
proposals—precomputing every saved-column word mask despite storage growth, and
removing redundant indexing checks inside the validated reduction loop—were
tested separately and together. They established no repeatable substantial gain
on the reviewed development cases, so neither was installed. All 56 original
oracle barcodes passed in each of four variants, and 192 accepted diagnostic
samples had zero compilation/recompilation. Other Julia activity arose during
these probes; their shared-host timings do not establish precise effect sizes
or a bound on undiscovered improvements. The large simplicial reduction remains
a major computational cost, not a proven unavoidable one.

A further cubical-focused study adopted bulk extraction of completed binary
columns and pooled storage for saved pivot columns. Two alternating before/after
processes covered all 28 existing development fixtures, with five accepted
samples for construction, query and complete work: 1,680 accepted measurements,
zero reported compilation/recompilation, verified fresh mathematical state and
matching output digests. Complete-request medians improved on 24/28 cases
(13/16 original and 11/12 larger), including all 21 cases using the changed
reducer. All four observed losses belong to the unchanged graph route and
remain in the record. Geometric mean ratios were 1.065× and 1.117× for the
original and larger groups.

On the largest changed cases, complete requests went from 51.3 to 47.2 ms
(cubical, 262,144 cells), 4.16 to 3.64 s (simplicial, 396,606 cells), and 2.36 to
1.98 ms (algebraic, 1,548 cells). These are shared-host development results;
variation in the unchanged graph control cautions against treating the modest
cubical ratio as an exact isolated-machine gain. Pooling raises temporary
allocation on those cases: about 55.5 to 62.0 MB for the largest cubical request,
for example. Retained input and answer sizes are unchanged. The selected source
passed 14,307 owner assertions, the 56 original oracle cases and 112 exact
retained-representative comparisons. PHAT was not retimed in this study.

Same-grade filtered compression was also implemented and verified, including
independent homology-map rank checks. Explicit quotient construction creates
substantial boundary fill on the largest simplicial case. Projecting boundaries
on demand avoids that large intermediate, but still loses to extraction plus
pooling on the large controls. The largest cubical complex shrinks from 262,144
to 17,850 cells, yet computing the projection costs more than the subsequent
reduction saves. None of the three tested compression variants was installed;
no inactive alternate path remains in the library.

At this development checkpoint, the reviewed optimization work was complete
at its scope. Final harness qualification still had to reconcile the original
worker's per-request collection with the later before-warmup collection policy,
record the actual routes, and fix the final
source and run configuration before a candidate freeze. The ten reserved cases
had no recorded performance results, and confirmation was outstanding.
No final 48-case PHAT performance claim had been established at that checkpoint.

### Turn each planned comparison into a finished study

For every entry, first identify the mathematical object both tools can actually
return. Retaining the same module can be checked through compatible maps even
when storage and algorithms differ. For an invariant-only request, verify the
invariant actually requested. Record capability differences separately from
matched timings; a missing common output remains an explicit scope decision.

Use the [suite charter](#reusable-suite-charter) to fix an affordable set of
workflows, inputs, size limits, fields, evaluation cases and completion criteria
for that library. **The QPA suite's 220 cases are not a default for the other
20 comparisons.** A narrow specialist can justify a much smaller suite. Preserve
TamerOp's intended APIs and finite-encoding purpose when choosing the overlap.

For each study, record its status, checked versions/date, charter, credited
earlier evidence, remaining tasks, final report and reproduction instructions.
Use the same progression: scope and verify, pilot, optimize TamerOp where
substantial gains are feasible, freeze, confirm, archive and close. Speed remains
the primary measure; memory and startup remain separate. Competitor source stays
unchanged, while optional profiling is permitted by the
[comparison policy](benchmarking.md#keep-competitor-implementations-unchanged).

Several libraries call one another. Pin and disclose those backends, charge
their work inside the declared request, and do not count wrappers around the
same algorithm as independent evidence for a general speed claim. Visual
quality and ease of use deserve separate assessments from computation time.
After closing a scoped comparison, move to the next planned library rather than
continually enlarging the completed suite.

## Reusable suite charter

Copy this into the competitor's local study, then attach the manual's detailed
study brief:

```text
Suite/version and intended empirical claim:
Why these tasks represent the intended application area:
Required workflow families and exact outputs:
Known overlap deferred from this version, with reasons:
Fixed input recipes, size endpoints, fields and difficult controls:
Development/evaluation split and prior exposure to results:
Primary timer and weighting, with sensitivity view:
Pilot limit, fixed main repetitions, time/RSS limits and total budget:
Mathematical oracles and result-equivalence checks:
Practical margin, aggregate method and permitted claim wording:
Diagnosed major bottlenecks, feasibility evidence and residual-gain stopping rule:
Diagnostic run limits and costs, separate from the confirmation budget:
Completed-evidence credit and remaining implementation milestones:
Closure checklist, reproduction package and next-version triggers:
```

The next bounded cubical experiment retained two further changes: component
merging for H0 inside higher-dimensional complexes, and a reversed graph
calculation for a top boundary whose rows have at most two odd incidences.
Eligibility comes from the actual boundary matrix, including periodic
identifications; essential top-dimensional classes and arbitrary-chain fallbacks
are preserved. Direct packed-column storage was tested but not selected: it
helped the small dense algebraic control while hurting the target cubical case.
A subsequent cleanup of the shared graph helper also lacked a repeatable gain.

The selected implementation passed 18,022 ordinary-persistence assertions,
56 original independent barcode oracles, and 112 unchanged representative
fingerprints. Its two-pass comparison contains 1,680 accepted samples on the
same 28 development cases. All seven cubical complete medians improve: about
1.25–1.58× on the original controls and 1.12–1.28× on the larger supplement.
The largest cube changes from 46.49 to 36.42 ms, with allocation falling from
61.99 to 56.32 MB. The largest simplicial median is essentially unchanged;
the largest graph is 7.4% slower in the main comparison, with mixed results in
a further diagnostic. Across all 28 cases, 16 improve and 12 do not. These are
shared-machine development results, not a uniform performance improvement or a
new PHAT comparison. The applied source matches the tested snapshot. No
reserved timings, PHAT rerun, final-suite freeze, or confirmation campaign was
performed by this follow-up; the final PHAT steps above were still outstanding
at that checkpoint.

A subsequent workspace follow-up tested deferred allocation, full degree-specific
indexing, and a simpler version that compacts only the active boundary's rows.
None was adopted. Deferred allocation saves about 10% of 2D allocated bytes but
does not establish a consistent larger-case speedup. Full degree indexing slows
the largest cube in both passes and adds about 1.5 MB of allocation. Row-only
indexing lowers its allocation by about 1.1 MB, with mixed elapsed-time results.
It improves the intermediate 13,824- and 64,000-cell cube medians by about 1.06×
and 1.20× and helps the algebraic control; these positive results remain recorded
alongside the largest-case losses and variation. They do not justify a new
default or a size threshold selected from favorable cases.

The follow-up contains 1,980 accepted compiled-code, fresh-state samples over
eleven existing development cases, with two independent passes per experiment
and exact-source controls. Every variant matches the 56 original independent
barcode oracles. Two current profiles place the largest cube's middle-degree
matrix reduction at about 15.4 ms, versus about 0.011 ms for empty workspace
allocation. These are diagnostic stage times, not whole-request speedups.
Exact-source controls show substantial variation, so the results retain the
shared-host qualification. Production source and owner tests are unchanged;
previous correctness evidence remains credited without a redundant owner rerun.
These workspace hypotheses were closed without claiming optimality. This
follow-up did not perform the subsequent harness qualification or confirmation.


### PHAT candidate freeze, 2026-10-04

This entry records the candidate freeze before the final comparison. It fixed
the code and conditions under which each tool would recover an ordinary barcode;
the freeze itself did not establish a final speed advantage.

The isolated candidate matches the source with 18,022 passing owner assertions
and 112 unchanged representative fingerprints. Final harness qualification
checks all 56 original inputs and all twelve larger inputs in TamerOp and both
PHAT reduction methods. Eleven harness tests and 22 native reset assertions
pass. A two-pass development-only smoke supplies 432 accepted measurements with
zero compilation or recompilation; none of the ten reserved cases was timed.
The full package suite was not rerun for this source-identical candidate.

The final harness records actual public backend provenance, verifies fresh
mathematical state with a linear traversal, collects before phase warmup and
charges natural collections during requests. Separate freeze records lock the
source, fixtures, unchanged PHAT v1.7 binary, Julia runtime, harness and complete
run configuration. Both pass launch validation without starting measurements.
The candidate is an archived source snapshot, not a new Git tag or release.

The frozen confirmation plan used five paired process passes for the original
48 scored cases and, separately, the existing twelve larger inputs. Native construction,
complete query, and construction plus query remain distinct measurements.
The plan retained losses and uncertainty without letting the larger supplement
alter the primary score. At freeze time, the final campaigns and public PHAT
result report were pending, and host availability still required a check before
timing. The closure entry below records the later PHAT v2 scope and results.

The local archive is `audit/2026-10-04/phat_freeze/`, with acceptance evidence and
exact launch commands. It preserves the earlier qualification attempt and the
final corrected harness separately. The larger geometric cases retain their
stated limitation: cross-tool barcode agreement and terminal Betti checks,
rather than a complete independent barcode oracle.


### PHAT v2 final confirmation and closure, 2026-10-04

C18 / PHAT v2 is **closed** for comparison candidate `phat-v2-2026-10-04`, using
the accepted implementation `phat-2026-10-04`. The expanded suite has 48 requests:
four equally weighted families, four structural variants each and three sizes.
All proposed endpoints passed the pilot and were retained. No implementation
tuning, favorable case selection or weight changes followed that pilot.

TamerOp **1.33× as fast in the balanced aggregate** for construction plus complete barcode; all 48 medium/large requests completed correctly in both tools.
The PHAT/TamerOp aggregate is **1.329×**, with a 95% paired-process interval
of **1.290–1.370**.
The [current report](benchmarks/phat.md) presents the actual times, family and
variant differences, scaling curves, uncertainty, machine and resource evidence.

Five paired process passes produced 4,320 accepted samples and
2,880 warmups. All resets and complete barcode comparisons passed;
accepted compilation/recompilation counters were zero. There were
0 rejected compilation-contaminated attempts. No worker failed or
hit a resource limit. All 56 qualification inputs agreed in TamerOp and PHAT's
twist and standard reductions. The 15 harness checks passed.

The feasibility pilot took 6.1 minutes and final confirmation
19.1 minutes, within the roughly two-hour timing allowance.
Generation, qualification and publication are separate costs. New endpoints were
chosen to cover larger inputs, not to consume the whole time budget.

Independent full-barcode evidence covers graph and algebraic inputs and the
small controls. Larger simplicial/cubical inputs have exact chain/filtration
checks, known terminal topology and complete cross-tool agreement, a narrower
form of evidence than a separate full barcode oracle.

As requested, v2 **replaces the earlier public PHAT results**. Previous raw
evidence is retained only in the local audit archive; the public page and data
present this new size distribution without pooling old samples. A changed
aggregate across different workloads is not itself an implementation speedup.
The new evidence is in `audit/2026-10-04/phat_v2/`, with the older accepted
source remaining in its sealed `phat_freeze/` archive. A portable executable
reproduction bundle remains separate publication work. Earlier development
and freeze notes above retain their original dates and status.


### Ripser.py v1 final confirmation and closure, 2026-10-05

The [public report](benchmarks/ripser.md) closes C14's fixed v1 comparison.
All **37 requests** completed in both tools across four paired process passes:
the 31 existing development requests and six recipes reserved before tuning.
All **888 accepted samples** satisfy the fresh-mathematics, reset,
output-consistency and zero-Julia-compilation checks. All **148 paired
mathematical checks** pass. The reserved cases are reported separately and
were not used for post-freeze optimization.

The final candidate retains the measured landmark validation/selection,
coface and selective F2 queue improvements; unsuccessful alternatives remain
in their development archives. Remaining whole-matrix validation has a real
quadratic input-checking cost, and workload-specific losses are retained.
This is a practical closure of the fixed comparison, not a proof that no
future optimization can exist. No library code was tuned during final evaluation.

Independent sparse coboundary solvability closes the previous nontriviality
gap for the sampled large connected H2 witnesses in both tools. Verification
still samples at most six longest positive-degree cocycles per degree; it does
not certify every returned cochain or compare all large-output bases.

Query time, native construction and their per-sample sum are separate. The
1,024-point circle takes 7.01 s versus 30.4 s;
the 512-point planar H0-H2 request takes 2.15 s versus
11.7 s. Family summaries, all individual losses, paired-pass
ranges, machine details and memory measurements are in the report and downloads.
No development timing is pooled with the final measurements. The earlier
paragraphs are historical development records superseded by this closure.

The local archive is `audit/2026-10-05/ripser_final_v1/`. Public data live under
`docs/benchmarks/ripser_v1/`; a portable computational reproduction bundle is
still a separate release task. Future feature additions do not reopen this
version: use a new scoped study when the compared request changes.
