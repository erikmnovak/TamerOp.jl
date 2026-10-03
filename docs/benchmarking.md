# Benchmarking TamerOp

A finite encoding retains a module's vector spaces and maps so that we can
answer many questions about them. Measuring its performance starts with a
question: how much work is needed to build that description, answer a new
question, or return to something already computed?

Read this manual before planning, changing or reporting a benchmark. It governs
new optimization and cross-software comparisons. The [finite-encoding
introduction](finite_encodings.md) explains the central object; the
[testing guide](testing.md) explains correctness checks. This manual does not
retroactively certify existing benchmark scripts or results.

Before choosing further experiments, use [Designing a comparison that can
finish](benchmark_suites.md). Freeze a finite suite, its intended claim and
completion criteria. Existing adequate evidence earns credit; another possible
input size or operation does not automatically extend the current study.

The [comparison roadmap](benchmark_suites.md#planned-competitor-comparisons)
tracks QPA and the other libraries planned for comparison, with their intended
overlap and remaining scope checks. Consult it when choosing the next study.

**The primary measure of computational efficiency is compiled code performing
the requested mathematics without reusing results from an earlier query.**
Elapsed time is the primary performance objective, subject to correct answers
and declared resource limits. Construction, compilation, legitimate reuse and
memory also need measurements. Report memory separately: a larger footprint
does not cancel a speed advantage, but a resource failure prevents completion.
A fixture is a fully specified example input. A sample is a timed execution; an
independent process run starts the runtime again and contains its own samples.

Our aim is strong evidence for TamerOp's speed and breadth in its intended
workflows. Treat performance leadership as a hypothesis to test. Attribute wins
to the tested implementations, versions, outputs and conditions. A comparison
of selected programs cannot prove that Julia, or TamerOp, beats every existing
package. Preserve losses and coverage gaps: they identify what to improve next.

| Task | Start here |
| --- | --- |
| Set the scope and decide when to stop | [Design a bounded suite](benchmark_suites.md) |
| Plan the comparison | [Define the question and output](#define-the-question-and-output) |
| Choose what to compare first | [Prioritize complete mathematical workflows](#prioritize-complete-mathematical-workflows) |
| Measure actual computation | [Measure compiled code with uncached mathematics](#measure-compiled-code-with-uncached-mathematics) |
| Investigate first use | [Measure and improve compilation](#measure-and-improve-compilation) |
| Diagnose storage | [Measure memory separately](#measure-memory-separately) |
| Plan a reproducible release | [Preserve and release the evidence](#preserve-and-release-the-evidence) |
| Begin a study | [Study brief and completion checks](#study-brief-and-completion-checks) |

Completed comparisons are collected in [Benchmark results](benchmarks/index.md),
starting with [QPA](benchmarks/qpa.md). Those pages present the measured
outcomes; this manual explains how to obtain and interpret them. Public
results and figures do not by themselves constitute a runnable reproduction
bundle.

## Define the question and output

For example, fix a finite poset, field, and modules M and N with specified spaces
and structure maps. Ask for complete Hom maps and positive-degree Ext
representatives with coordinates through a stated degree. Asking only for the
dimensions is another, usually cheaper task. Define the starting object,
required answer, hypotheses and stopping point before measuring.

State whether parsing, field conversion, construction, validation,
materialization and output writing are timed. A lazy result may defer the work.
Complete every requested part inside the timer, including synchronization for
parallel execution. Preserve or consume the answer so the computation is
observable. Keep normal public checks unless measuring a separately labeled
trusted-input path.

Useful boundaries include raw data or a presentation to an encoding; an encoding
to an invariant or algebraic result; raw input to the completed answer; and an
existing result to an additional query. Use the measured end-to-end task as the
primary evidence for a workflow claim. Add component timings when they answer a
specific diagnostic question. A matrix-kernel improvement must reach the
relevant public entrypoint before it becomes a user-facing speedup.

## Prioritize complete mathematical workflows

Start with what a user wants to learn from the finite encoding. For example,
a user may need complete Hom maps together with Ext representatives and their
coordinates to compare two modules. The primary comparison measures the time to
produce that whole answer from equivalent starting information. A verified
advantage for this task stands on its own: identifying which internal stage
accounts for the gain is not a prerequisite for reporting it. Different
algorithms, representations and reuse within the computation can all contribute
to a legitimate workflow advantage.

Choose the requested outputs for their mathematical purpose before looking at
timings. Cover a variety of relevant workflows and input families, including
unfavorable cases. Do not bundle unrelated requests or adjust their frequency
to obtain a preferred ranking. For a session answering several questions from
one encoding, explain why those questions belong together and measure the
session directly, giving both tools equivalent reuse opportunities.

A complete task can also be a single Hom basis or the dimension of Ext in one
degree. Include such standalone requests when they represent intended use or
are needed to support the scope of a claim. Computing dimensions alone has a
different finishing point from computing representatives and coordinates. A
win on the combined workflow does not establish a win on each standalone task;
a standalone loss does not invalidate the combined result. Preserve and report
both with their respective scope.

TamerOp component measurements and profiles serve a specific investigation:
locating a bottleneck, explaining a regression, examining allocation or scaling,
or checking whether a surprising comparison charges equivalent work. Select the
relevant diagnostics for that question. An exhaustive breakdown and wins in every
component are not completion requirements for a valid workflow comparison.
Time spent broadening meaningful workflow and input coverage can provide more
useful comparative evidence than explaining an already verified advantage.

Keep the reset boundary at the start of the declared task. A combined Hom/Ext
query may reuse a projective cover or resolution across its stages and degrees.
Resetting between those stages would measure a different algorithm. A separately
benchmarked standalone request begins from its own verified reset and declared
inputs; any supplied result or preparation must be charged or identified as
retained state under the timing regimes below. Separately reset component times
must not be summed to stand in for a directly measured complete workflow.

## Compare answers while respecting different designs

TamerOp's finite encoding, classifier and retained maps remain central. Another
tool may reach the same answer through a different representation or algorithm.
That is a valid comparison when both start with equivalent information, finish
the agreed task, and pass the mathematical checks. Matching implementation
steps or reshaping TamerOp to resemble a competitor is unnecessary.

Maintain a versioned coverage table for each competitor:

| Record for each task | Purpose |
| --- | --- |
| Input class, category, field and hypotheses | Establish the mathematical overlap |
| Required answer and equivalence check | Define a verifiable finishing point |
| Native public calls, backend and options | Identify the supported implementations tested |
| Starting representation and conversions | Prevent free preparation on only one side |
| Status and version/date checked | Distinguish verified overlap, untested overlap, unknown support, unsupported tasks and different semantics |
| Input families, evidence and remaining work | Track coverage of the intersection |

Survey the relevant APIs before selecting favorable cases. Freeze a staged
plan for a representative subset of the intended overlap, including explicit
deferrals and a completion rule; record which cells were actually tested.
The intersection inventory is not an obligation to benchmark every operation.
One Ext family cannot establish leadership over all finite algebra. Separate
TamerOp-only capabilities from comparative timings; unsupported tasks have no
speedup ratio. Recheck support when competitor versions change.

Use complementary experiments where appropriate. In a shared-algebra experiment,
both tools receive the same finite diagram in native storage, with construction
charged separately. In a native-workflow experiment, each tool follows its
supported route from a common external input to the answer, with all necessary
conversions charged. Disclose any unavoidable extra outputs. If starting
information or answers cannot be aligned, explain the difference and report a
directional comparison rather than a matched speedup.

### Keep competitor implementations unchanged

**Repairing, redesigning or modifying competitor implementations is prohibited
in this benchmarking work.** Compare unmodified released software through its
supported interfaces. Do not patch competitor source or replace its methods.
Implementation optimization work belongs to TamerOp.

A competitor's internal algorithm, data structures, arithmetic costs and missed
reuse are part of its measured performance. Leave those implementation costs
intact. A large measured ratio warrants checking our experimental setup and
answers; it does not authorize fixing the competitor's internal bottlenecks.
Report an unexplained timing difference as an observation without inventing a
cause. Timing, resource measurement, output validation and cache-reset
verification remain part of a fair comparison.

Profiling a competitor is allowed as an optional diagnostic when it answers a
concrete question about observed behavior. It must not change the competitor's
implementation. Run profiling separately from the measurements used for speedup
claims, so instrumentation overhead does not enter the comparison. Explaining
the competitor's bottlenecks is not a prerequisite for reporting a verified
workflow advantage, and profiling findings do not authorize repairs or redesigns.

The benchmark harness remains our responsibility. Use an appropriate supported
public route for the agreed answer. Avoid adding unnecessary calls, asking for
extra degrees, or discarding useful intermediate results within one task.
For example, if the competitor's own routine reconstructs a factor internally,
that cost belongs to its implementation. If our harness needlessly calls a
constructor again when the documented workflow reuses its result, correct the
harness. Different legitimate internal steps do not need to be matched.

Before the main run, make a bounded check of the documented entrypoints,
relevant algorithm options and normal reuse for the chosen task. Record the
selected calls, options and starting state. No proof of a globally fastest
configuration or exhaustive search of alternative algorithms is required.
A default-route comparison is valid when labeled as such. If an established,
more appropriate supported route is known, include it before making a broader
claim about the competitor's practical performance; do not select an inferior
route to preserve a favorable ratio. A suspected misuse or output mismatch needs
resolution, whereas a verified slower implementation can remain slower.

Use current supported competitor releases and record exact versions. Identify
any TamerOp development commit separately. Give both tools comparable resource
budgets and access to options and accelerators supported by their recorded
versions. For competitors, configuration is limited to those existing supported
settings; it must not change the implementation. Keep default and configured
results separate. Any configuration selection uses development fixtures and is
evaluated on held-out fixtures. Report common resource budgets separately from
each tool's tested practical configuration.

## Name the timing regime

Every row must identify its regime and retained state. Bare “cold” and “warm”
labels are insufficient.

| Regime | Available state | Measured work |
| --- | --- | --- |
| Installation/precompilation | Declared environment/depot state | Dependency preparation and compilation; distinguish downloads |
| Fresh-process first answer (`strict-cold`) | Installed dependencies and declared package images | Launch, import, remaining compilation, setup and first answer |
| Compiled code, uncached mathematics (`warm-uncached`) | Compiled methods and explicitly declared input/category preparation | New computation without a prior query's mathematical results |
| New query with shared preparation | Named category, geometry or module preparation | Another question in a realistic session |
| Query on a retained result (`warm-cached`, with state specified) | Named existing result | Additional coordinates, inspection or another result operation |
| Retrieve a completed answer | The answer itself | Retrieval or lookup |

The uncached row is the primary algorithm comparison. Reuse is a real benefit of
retaining the encoding; measure it separately. Preserve ordinary reuse *within*
one query: disabling internal memoization while it runs changes the algorithm.

Uncached mathematics does not imply empty CPU or filesystem caches. A fresh
process does not imply a clean installation. Do not routinely flush system
caches or delete shared depots to manufacture a cold measurement. A stage-local
first query is not the whole fresh-process first answer.

## Measure compiled code with uncached mathematics

Warm methods for the field, storage types, backend and branches to be exercised,
then arrange for the measured query to recompute its mathematics. The same
numerical input can be used again if the reset is verified.

1. Inventory state that saves work: input/result fields, session caches,
   resolutions and syzygies, geometry plans, matrix factors, coordinate
   projections, and relevant global, task-local and dependency caches. Identify
   keys based on object identity versus equality, and objects retaining entries.
2. Warm the operation and measurement wrapper on a separate valid fixture,
   including required output materialization. Give each runtime its appropriate
   execution/backend warmup.
3. Discard old results and reset the relevant mathematical state. Reconstruct
   native inputs as needed while keeping compiled methods. New wrappers, deep
   copies, `cache=:auto`, and `GC.gc()` alone do not prove that prior factors
   or results are unavailable.
4. Verify reset postconditions through inventories, hit/miss counters or another
   justified inspection. Identify preparation outside the timer. All query-enabling
   mathematical preprocessing beyond the declared input/category contract must
   be charged in the primary total or identified as reuse, including source-only
   resolutions or geometry preparation.
5. Time one complete query, letting caches populate naturally during it. Record
   allocations, garbage collection and compilation diagnostics, and keep the
   answer observable.
6. Validate outside the timer, then reset before another sample. Validation may
   itself populate caches; document its placement and retained objects.

Native constructors may themselves prepare factors or plans. Record that work
and report construction-plus-query totals when tools place equivalent work on
different sides of this boundary. A query-only comparison is conditional on
its declared prepared inputs.

Use equivalent starting knowledge in the other tool, even if reset commands
differ. Reset is experimental preparation outside the computation timer; normal
session behavior needs a separate measurement. If a relevant cache cannot be
inspected or reset, label the remaining state and uncertainty instead of
certifying uncached computation. An alternative is a fresh process with verified
precompiled methods and no retained training results, followed by a compilation
check. Streaming unseen valid inputs is also useful, but still check for
content-keyed reuse. New random bases alone do not change the underlying module.

Julia 1.12's `@timed` reports `time`, `bytes`, `gctime`, `compile_time` and
`recompile_time`. Preserve these raw fields. Require zero recorded compilation
and recompilation in accepted primary samples. Save a row with compilation as
such, diagnose the missing warmup, and reset before rerunning. Never subtract
compilation from elapsed time to manufacture a warm measurement. Counters cover
a defined measured region, not all compilation since launch. Zero counters also
do not certify the absence of native-library initialization or another backend's first-use work; declare
such preparation and any background activity. See
[Julia's timing documentation](https://docs.julialang.org/en/v1.12/base/base/#Base.@timed).

This **protocol sketch** names study-specific operations, not TamerOp APIs:

```text
warm the methods and measurement wrapper on a separate fixture
for each planned fixture and independent sample:
    discard old outputs; reset and inspect mathematical caches
    prepare native inputs; record construction separately
    verify absence of previous query-enabling mathematical results
    measure one complete query; save counters and any failure
    independently verify the answer outside the query timer
```

BenchmarkTools repeats evaluations and performs warmup/tuning. For stateful
operations, reset in per-sample setup and use `evals=1`; setup runs once per
sample, not per evaluation. Interpolate external inputs or use functions to
avoid measuring global-variable access. Check implausibly small times for
eliminated work or reuse. A single-call driver is often clearer for these
experiments. See the
[BenchmarkTools manual](https://juliaci.github.io/BenchmarkTools.jl/stable/manual/).

Preserve samples rather than calling the minimum “raw runtime.” Include garbage
collection during the operation. Record any forced collection before samples
outside the timer and use a comparable policy in the other tool. Measure normal
long sessions separately when allocation pressure matters.

## Measure and improve compilation

Growing an ordinary matrix need not create a new Julia type; changing scalar
types or entering another backend may require more compiled code. Measure
compilation for a workload family. It is neither a fixed universal penalty nor
evidence that the resulting algorithm must be faster. The compiler processes
harness code too, so distinguish it from production methods.

Separate installation/precompilation, launch, import, input/category preparation,
native construction and first query. Also measure uninterrupted launch-to-answer:
adding isolated stage medians does not produce that observation. Include
single-field processes and mixed-field sessions, recording field order and
which earlier results remain.

Use installed, usable package caches for the ordinary installation baseline.
Source-loading fallback and `--compiled-modules=no` are separate diagnostics.
On Julia 1.12, `--compiled-modules=strict` fails when a required usable module
cache is unavailable. Record package-image settings and native-image acceptance;
the flag does not precompile every possible call. Record source path, runtime,
flags, extensions and environment. Check flags against the recorded runtime's
[command-line documentation](https://docs.julialang.org/en/v1.12/manual/command-line-interface/).

These **terminal commands** illustrate separate preparation and import checks
in a Julia 1.12 source checkout. They are not a full first-answer benchmark:

```sh
julia --startup-file=no --project=. -e 'import Pkg; Pkg.instantiate(); Pkg.precompile()'
julia --startup-file=no --project=. --threads=1 --compiled-modules=strict -e 'import TamerOp'
```

Measure each in its own process when studying those stages. An already prepared
depot cannot measure clean-install cost; record its initial state and use a
dedicated environment for installation experiments. Preserve the user's depot.

Use this optimization sequence:

1. **Trace missing compilation.** Use `--trace-compile=PATH` or a pinned compiler
   profiler in a separate diagnostic run. Separate production methods,
   dependencies and harness wrappers; tracing is not the final timing run.
2. **Extend bounded precompile workloads.** Exercise small representative public
   tasks, required heavy outputs and nonzero higher-degree branches where
   relevant. Select common fields deliberately. Discard training objects and
   mathematical caches before saving package state.
3. **Investigate needless specialization and inference work.** Preserve efficient
   arithmetic kernels. Share orchestration only when measurements show a benefit
   without slowing the computation. Avoid many variants for incidental values
   that confer no measured advantage.
4. **Measure loading contributions.** Examine dependency initialization,
   deserialization and extensions before redesigning loading. Preserve exact
   algebra, geometry and finite-encoding capabilities.
5. **Consider optional system images for stable workflows.** Compare separately
   against ordinary installation; record build cost, CPU target and embedded
   versions. Do not require them for routine package use.

PrecompileTools can save compiled workload code for later sessions. Measure the
tradeoff: precompilation duration, image size, import time/RSS, remaining first
use, and compiled-code execution. Do not fill images with the whole test suite
or benchmark answers. See
[PrecompileTools](https://julialang.github.io/PrecompileTools.jl/stable/) and
[Julia's latency guidance](https://docs.julialang.org/en/v1/manual/performance-tips/#Execution-latency,-package-loading-and-package-precompiling-time).
Custom system images embed versions that take precedence over the active
environment; rebuild and verify them after package changes. See
[PackageCompiler](https://julialang.github.io/PackageCompiler.jl/stable/sysimages.html).

## Measure memory separately

| Measure | Meaning and limits |
| --- | --- |
| Allocated bytes during a call | Allocation traffic, including objects later collected and possibly compiler work |
| Retained input/result/cache objects | Storage reachable from declared roots; document sharing and omissions |
| Runtime live heap after collection | The runtime collector's own accounting boundary |
| Process RSS (resident set size) and peak RSS | Resident footprint including runtime, code, native libraries and allocator behavior |
| Package/system image size | Prepared-code storage on disk, distinct from resident memory |

Use separate retention-inspection processes when the inspector's own compilation
would contaminate lifecycle measurements. Record loading, construction, first
query, subsequent work and release checkpoints. Name units, estimators and
collection placement. Preserve discrepancies between independent OS counters;
they need not agree byte for byte.

Measure a union of roots: results often retain their modules, so individual
sizes overlap. Explain treatment of weak references, tasks, shared buffers,
closures and external allocations. Inventory discovered caches and entry counts;
discovery does not prove every dependency cache was found. Empty containers
still occupy memory.

Release results, modules and inputs in documented stages. Observe rather than
assume that collection returns pages to the OS. RSS minus root estimates does
not measure generated-code size. Runtime allocation counters and partial object
walkers cannot establish exact cross-language storage ratios. Small fixtures
do not settle large-module storage scaling or establish the absence of leaks.

## Choose inputs that test the claim

Define families before selecting favorable results. Include hand-checkable
controls, difficult synthetic cases and available research datasets. Hold out
some families from optimization. Retain unfavorable cases as regression
benchmarks, particularly dense inputs and one-off computations.

| Area | Useful variations |
| --- | --- |
| Finite algebra | Poset size, width/depth, relations/covers; chains, branches, grids and Boolean families; stalk dimensions, sparsity and ranks |
| Module structure | Intervals, projective/injective controls, nonsplit extensions, presentations, unequal supports/dimensions, nonzero higher Ext |
| Coefficients | QQ; supported prime fields including odd characteristic; numerator and denominator sizes separately; numerical fields with stated accuracy |
| Geometry/encodings | Ambient dimension, regions/facets, slopes, thin supports, boundaries, supported degeneracies, bounded/unbounded domains |
| Ingestion | Point, graph and image size; density, complex dimension and simplex counts; sparsification and approximation choices |
| Queries/outputs | Single query, batches, single degree, whole complex, dimensions/representatives, uniform/skewed and boundary-heavy queries |
| Resources/reuse | Serial/threaded execution, independent construction, shared preparation, repeated labels and retained-result calls |

Construct valid fixtures and verify their relations independently. Random edge
matrices usually fail commuting-path relations. Dense changes of basis test
arithmetic difficulty but not a new module type. Reducing rational data modulo a
prime can change ranks or invalidate a basis change; validate each field's case.
Do not count an empty coordinate loop as a nontrivial algebraic speedup.

Use a fixed, affordable size ladder and a declared resource limit. Stop at
the planned final level or the resource cap; neither a loss nor the existence
of larger inputs automatically extends the ladder. Record
actual encoding size, relations, matrix nonzeros, output size and relevant
coefficient growth. Raw point count alone need not predict difficulty. Include
real target shapes, tiny and moderate-sparsity cases, and both sides of backend
or algorithm thresholds. For saved outputs, record file size as well as time.
Show scaling curves and where no crossover occurred; do not extrapolate a
guaranteed eventual win from small examples.

For a fixed family, let T(n) = C_T + W_T(n) and Q(n) = C_Q + W_Q(n) describe the
two tools. C is declared startup cost; W is computation at size n in the chosen
regime. Treat C as fixed only within the declared type/backend/workload regime.
TamerOp recovers extra startup where W_Q(n) - W_T(n) exceeds C_T - C_Q.
For multiple independent inputs, accumulate savings and any new compilation.
This is an accounting model, not a measured asymptotic result.

## Verify the mathematical answer

Correctness makes a result eligible for performance claims. Run independent
small-case oracles before expensive experiments, then verify every reported
case at its declared level. Keep expensive checking outside query timers while
retaining normal public input validation. Avoid checking a timed object first
if that prepares hidden computation state.

Define “same output.” Hom bases can differ in entries while spanning the same
space: check naturality, independence and completeness. For Ext representatives,
check cycles, boundaries, quotient dimensions and coordinates. Module-valued
outputs require maps and commuting diagrams alongside dimensions. State the exact
sample family for sampled invariants. Numerical answers need tolerances,
conditioning and precision; distinguish them from exact output.

Different valid resolution models can satisfy a request for an Ext basis in
native coordinates. Establish that both models compute the stated Ext group in
the same category, field, variance and degree, and verify representative validity,
independence and completeness in each model, with independent checks appropriate
to the claim. Matching dimensions alone is insufficient. An explicit map between
the two returned bases is not automatically required for this performance
comparison. State when such an identification has not been constructed.

If a claim identifies particular extension classes, induced maps or products
across models, provide comparison maps or another mathematically justified
identification sufficient to check that claim. Separate native checks alone do
not establish it. Add these operations to the timed workflow when the user's
mathematical question requires them. A study of native basis construction need
not expand into a study of products solely because the tools use different
representations. Correctness and answer equivalence remain requirements;
matching implementation details is unnecessary.

Fix category, field, variance, degree, parameter orientation, window, endpoint
rules, grading and truncation. Ext/Tor on different encoding posets need not
agree without hypotheses and comparison maps. For geometric pipelines, match
radius versus squared radius, density definitions, landmark sets, dimension
caps and grids where those choices define the common task. Different valid
triangulations need not have identical cell counts; check the requested
mathematical answer with an appropriate equivalence criterion.

Independent predicates, derivations or resolutions provide stronger evidence
than wrappers around the same code. Digests identify canonical outputs or files;
they do not prove correctness. A dimension match cannot verify a basis or map.

Mismatches, crashes, timeouts and unsupported operations are distinct outcomes.
Keep them in the coverage table. Incorrect answers have no valid speedup;
timeouts provide a resource-bound observation, not infinite runtime or an exact
ratio. Show completed verified-pair summaries together with excluded/unresolved
cases and their reasons.

## Run controlled experiments and retain variation

Record CPU, memory, OS, runtime/package versions, source commit and patch,
dependency locks, backends, compiler flags, tuning profiles and relevant
environment variables. Record Julia and BLAS threads separately. Pin source and
configuration throughout a run; changes need a new identity. Serialize heavy
profiling/timing jobs rather than benchmarking them against one another.

Confirm machine availability when needed and monitor aggregate activity and
resource pressure without inspecting unrelated projects. If other work must
continue, proceed with correctness/diagnostics and label times as shared-machine
observations. More samples do not remove systematic contention. Do not stop
unrelated work or change system-wide settings without authorization.

Pilot to estimate cost, check counters and choose the sampling budget. Before
main runs, record process/sample counts, seeds, ordering, timeouts and stop rules.
Establish a baseline before the candidate. Use identical inputs/flags across
variants and reverse or randomize order in later independent passes. Require
at least two independent passes for noisy kernels, preserving both raw outputs;
publication claims may need more. Do not treat many iterations in one process
as independent process replications. Same-script A/B runs can help, provided
compiled code and cache state do not leak between variants.

Report per-case median runtime ratios plus allocation and GC differences, spread
and sample counts. Use uncertainty estimates reflecting the independent unit.
Keep distributions or quantiles when long pauses matter. Preserve failed and
interrupted attempts and monitor gaps; resume into new artifacts. Do not silently
replace unfavorable rows or merge pilots into the final sample population.

Match ratios by fixture and regime. Do not divide one tool's best case by the
other's unrelated worst case. Declare aggregation weights and eligible cases
before reporting; give family results and coverage counts alongside any geometric
mean. Distinguish medians of measured batch totals from sums of individual
medians. Separate single-field and mixed-field observations.

## Use measurements to choose a TamerOp optimization

The implementation changes in this section apply to TamerOp only. Competitor
implementations remain unchanged; optional competitor profiling follows the
comparison policy above.
Address every diagnosed major TamerOp bottleneck where feasible; there is no
fixed count of fixes or optimization cycles. Reassess affected workflows until
remaining diagnosed feasible improvements offer only minor gains, accounting
for their combined effect. Document major costs that cannot feasibly be reduced.
Follow the [optimization stopping rule](benchmark_suites.md#separate-investigation-from-confirmation);
a timing budget or an aggregate speed win does not by itself complete that work.
A profile suggests a hypothesis; a controlled before/after experiment tests it.
Keep fixtures and required outputs fixed. Match each revision to its own usable
precompiled image for startup tests. Hold tuning constant unless tuning is the
intervention. Check the code actually called: benchmark imports and owner-module
bindings must follow current ownership, not a stale source bootstrap.

For implementation changes, check uncached work, one-off setup, batches,
repeated use, allocations and retained memory on affected families. For
compilation changes, add precompile duration, image size, import footprint and
first answer. Keep correctness and unfavorable controls. Verify the complete
public task after optimizing its kernel. Give long harnesses bounded sections
so a focused experiment does not require rerunning unrelated work.

Avoid choosing cache thresholds from one fixture. Separate representation
benefits from saved work by comparing shared construction with independent
reconstruction and giving both tools equivalent reuse opportunities. Charge
setup and retained storage. Preserve maps and classifiers when the requested
TamerOp object includes them. The same principle applies to a pipeline that
answers several distinct questions from one encoding: measure the full workload
and disclose the amortization, rather than repeatedly rebuilding it on one side.

## Preserve and release the evidence

Plan reproducibility from the first local run. The current repository policy
keeps benchmarks, audits and examples local. This public manual does not change
that policy or announce a benchmark release. A later authorized bundle or
companion repository should run without private directories or sibling projects.

Map the study's file layout to these roles:

```text
STUDY.md           question, coverage matrix, regimes and conclusions
environment/      exact dependency/runtime versions, build recipes and flags
fixtures/         generators, seeds, provenance and input hashes
adapters/         native tool calls and explicit conversions
validation/       independent oracles and output comparisons
run/              preparation, smoke, measurement and resume commands
raw/              commands, logs, samples, monitoring and completion records
reports/          regenerated tables, figures and machine-readable results
source/           pinned source references, patches or snapshots
licenses/         redistribution terms and third-party notices
MANIFEST.json      hashes and sizes tying code, inputs and outputs together
```

Each raw row needs run/process ID, tool/version, fixture/seed, field, regime,
phase, sample index, retained state, elapsed time, available allocation/GC/
compilation counters, correctness status and failure details. Use null with a
reason for unavailable counters, not zero. Preserve commands, exit codes,
timeouts, validation records, sampled order and explicit completion markers.
A successful exit alone is insufficient when an interpreter can recover from a
script error. Check expected row counts and final success records. Bind results
to source/configuration hashes and keep pilot/superseded data distinguishable.

Before release, provide exact installation/build commands and pinned downloads
or container-image digests for all tools. Lockfiles do not pin the CPU, OS,
native compiler or future artifact availability. Containers simplify setup but
do not guarantee identical timing. Offer small verification, bounded comparison
and full-study commands with expected resources. Separate environment setup,
precompilation, validation, measurement and report generation.

Document data provenance and licenses. Check permission to redistribute
competitor source/binaries and datasets; use pinned retrieval/build recipes
where inclusion is inappropriate. Publish the necessary study material without
credentials, private command paths or unrelated audits. Keep a local original
and document sanitization of the public copy.

Verify archive contents against a manifest and save its checksum. Have another
person or clean environment follow the procedure without developer-local paths.
Check mathematical outputs exactly where appropriate while recording timing
variation across hardware. Freeze a release, preferably with a persistent
archival identifier, and map paper claims to specific result rows. Corrections
produce a new version with a changelog; old evidence remains identifiable.
A local ZIP alone does not complete this release process.

## Study brief and completion checks

Copy this **planning template** into each new study. Unknowns are acceptable
while planning; keep them visible if still unresolved in the final claim.

```text
Suite/version, finite scope, deferred overlap and completion rule:
Question and intended TamerOp workflow:
Primary user tasks and reason for their selection or combination:
Targeted TamerOp component diagnostics and the question each addresses, if needed:
Versions/commits and source/configuration hashes:
Competitors, surveyed overlap and untested/unsupported cells:
Supported public routes/options checked; selected route and default/configured scope:
Input category, field, hypotheses, conventions and data provenance:
Requested output, mathematical equivalence and independent oracle:
Starting information, timed boundary, conversions and materialization:
Regimes and input/category preparation retained in each:
Cache inventory, reset procedure and verified absence of prior-result reuse:
Warmup coverage and compilation diagnostics:
Families, size ladder, seeds, held-out and unfavorable cases:
Resources, machine availability and monitoring:
Pilot budget; main process/sample counts, order, timeouts and stop rules:
Memory roots/estimators, omissions and result lifetimes:
Raw schema, provenance binding, completion checks and failures:
Statistics/weights, practical speed margin and planned plots:
Major-bottleneck dispositions, residual-gain stopping rule and diagnostic budgets:
Candidate freeze and evaluation policy:
Claim decision if results favor TamerOp, are mixed or remain unresolved:
Reproduction commands, archive plan and third-party licenses:
Permitted conclusions and unresolved limitations:
```

Before reporting, confirm verified outputs, supported timing-regime labels,
consistent charging of setup/construction, and demonstrated cache reset for
uncached rows. Check compilation diagnostics for accepted compiled-code samples.
Preserve failures, variation and machine conditions, and regenerate reports from
raw evidence. State limitations from coverage, scale, inspection or activity.

Close the suite when its fixed plan has recorded outcomes, checked claims and
reproducible evidence. Passing a superiority threshold is a possible conclusion,
not a completion requirement. Apply the [closure rules](benchmark_suites.md#decide-when-enough-is-enough)
before recommending more work or moving to another competitor.

Historical artifacts keep their original definitions. A row named “cold” may
mean the first query after code warmup: preserve its raw label and explain its
meaning in a new interpretation. Fresh objects cannot be relabeled as uncached
mathematics without evidence about shared factors. A later manual or successful
test does not retroactively validate an earlier timing claim.
