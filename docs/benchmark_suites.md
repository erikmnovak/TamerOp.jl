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
current order of work.

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
balanced speedup, fraction of wins, tested versions, and a link to full results.
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

- Weighted win, near-tie, loss and unresolved shares, and correct completion rates.
- A geometric mean of ratios, with its exact eligible-case denominator.
- Results by workflow and field, including absolute times and the largest losses.
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

Roadmap recorded 2026-09-30. **All 21 libraries below are in the planned
comparison programme**, including the current QPA study. Each comparison should
help us understand a part of the finite-encoding workflow: constructing a
module, retaining and analyzing its maps, or obtaining an invariant or figure.
Specialized tools also give useful comparisons for direct filtration and
barcode tasks that do not require constructing an encoding.

Finish QPA v1 before starting the next comparison. The table is ordered by
**estimated remaining work, from least to most**, as of 2026-09-30. This includes
adapters, independent correctness checks, warmup/reset verification, timing and
reporting for the intended overlap. It credits existing work rather than
estimating every study from scratch. It is a planning judgment, not a measured
effort ranking or a ranking of importance, speed, or total library size.

Positions within a few places of one another are approximate. Unknown TamerOp
optimization work is not predictable from the competitor's feature list. Each
study still needs its bounded charter; these ranks do not set case counts,
machine-hour budgets or deadlines. The C01–C21 identifiers stay fixed when the
effort order changes.

**Status:** C01 is in progress. C02–C21 are planned bounded comparisons under
this guide. “Planned” does not erase earlier exploratory measurements: inventory
and credit compatible evidence before deciding what remains to run. The scope
column identifies candidate common requests, not a completed certification that
every listed operation is already comparable in both implementations.

| Effort rank | ID | Library | Intended comparison scope | What must be aligned before timing |
| ---: | --- | --- | --- | --- |
| 1 | C18 | [PHAT](https://github.com/blazs/phat) | Supplied filtered boundary matrices to persistence pairs over F2. | Same matrix and ordering; include required output conversion and do not require unrequested module data. |
| 2 | C14 | [Ripser](https://ripser.github.io/ripser/) | Vietoris–Rips persistence from metric data to ordinary barcodes. | Same metric, threshold, field and homology degrees; a barcode-only request does not require a full encoding. |
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
algebra save substantial work, but 80 adapters and the main confirmation campaign
remain. It stays the active priority. multipers is ranked 21 because of the
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

### QPA progress and remaining work

As of 2026-09-30, QPA is the active, substantially developed comparison, but
its final campaign is not complete. The fixed suite has adapters for **140 of
220 tasks**: Q1–Q5 and Q8–Q10. All **96 Q1–Q5 cases** have passed mathematical
verification in both tools. Six development QQ examples have also passed the
timing protocol check; that check is not a 96-case performance campaign.

The remaining **80 Q6–Q7 cases** cover pushouts, pullbacks, homology and mapping
cones. Implement and verify these, complete the bounded development pilot,
address diagnosed major feasible TamerOp bottlenecks, then freeze the candidate
for confirmation and reserved evaluation. Existing historical timing evidence
continues to count within its original scope. Known QPA tensor-evaluation
failures remain recorded outcomes, not infinite speedups or deleted cases.
The existing QPA allocation, size bounds, budgets and scoring remain unchanged.

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
