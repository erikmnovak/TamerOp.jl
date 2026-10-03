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
algebra save substantial work. All adapters are implemented and the bounded
development pilot has run. Optimization assessment and the main confirmation
campaign remain. QPA stays the active priority. multipers is ranked 21 because of the
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

The subsequent acceptance review is complete, but **M3 remains open**. Sixteen
profiles of the current dense paths and 384 accepted, zero-compilation samples
identify two further feasible improvements on existing controls. Using the
selected dense product for the complete exact membership check improves
32-column requests by 2.21–2.38×. Trying the leading square inverse before
general row selection, with fallback when singular, improves scalar requests
by 1.59–1.71×. Their combined full-request gains are 1.58–3.59×. Factors and
answers agree exactly. These are disposable experiments, not implemented
improvements or QPA speed ratios. Small controls and unfavorable results remain
included. Implement and validate the two changes together before acceptance.

The GAP timing prerequisite is now resolved with a disclosed configured build
of unchanged GAP 4.16.1 and QPA 1.37. Its existing monotonic clock passes two
fresh-process checks. Across all thirteen request variants in QQ and F3, 24
old/new request pairs return identical independently verified answers; two
retain the same declared tensor failures. The benchmark adapters reject the
old nonmonotonic clock before collecting timings. The configured executable,
source hashes, compiler flags, failed attempts and preflight evidence must be
retained with the eventual environment record. This is runtime and harness
verification; the final QPA comparison has not run. Candidate freeze remains
pending, and no fixtures, size endpoints, weights or reserved cases change.

The final performance campaign remains open. Implement and reassess diagnosed
major feasible TamerOp improvements within the fixed suite, then freeze the
candidate for confirmation and reserved evaluation. Neither adapter
qualification nor this pilot establishes a full-suite speed result. Existing historical timing evidence
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
