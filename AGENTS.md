# AGENTS.md

Guidance for coding agents working in `tamer-op`.

## Core intent
- Optimize for performance and clean architecture.
- Keep APIs ergonomic for new users.
- Primary user base is mathematicians (not software developers); optimize naming, defaults, and docs/comments for mathematical readability first.
- Prefer clarity over compatibility shims (project is pre-release).
- Keep `Workflow` thin orchestration; place subsystem logic in dedicated files/modules.

## Project-specific principles
- No legacy-compatibility layers unless explicitly requested.
- No alias-heavy APIs in performance-critical surfaces.
- Keep canonical mode/value contracts strict and explicit.
- Avoid hidden behavior switches; backend/mode choice should be inspectable.
- Do not keep "legacy"/"shim" naming in production paths (`legacy_*`, alias symbols, old keyword bridges).
- Favor signature cleanup over overload accretion: remove convenience arities that do not add real capability.
- Keep internal plumbing internal (`_name` + non-export) when it is not part of the intended user surface.
- Internal function policy (practical):
  - Keep internal functions only if they do at least one of:
    - remove duplication in real call paths,
    - isolate hot-path logic for performance,
    - enforce invariants/contracts centrally,
    - provide reusable plumbing used in multiple places.
  - Remove internal functions that are:
    - dead (unused),
    - one-line wrappers with no semantic boundary,
    - internal knobs not consumed by runtime behavior,
    - diagnostic-only helpers that do not support active tests or critical debugging.

## API policy
- Breaking API changes are acceptable when they improve clarity/performance.
- Still keep convenience defaults (users should not need to pass empty `Options()` everywhere).
- Prefer one canonical public path over multiple near-duplicate wrappers.
- Advanced workflows must still be easy to discover and use without deep software-engineering context.
- If adding new options/fields, thread them through all relevant call paths (do not silently drop).
- Prefer a single cache keyword surface in Workflow APIs; avoid multi-keyword cache controls.
- Avoid adding new public wrappers unless they provide clear UX or measurable performance benefit.
- Standardize UX across analogous ingestion families (point cloud / graph / image): same option names and semantics where possible.
- For fringe construction, keep strict canonical signatures (no legacy aliases):
  - `one_by_one_fringe(P, U, D, scalar; field=...)` (+ explicit mask variant),
  - `FringeModule{K}(P, U, D, phi; field=...)` with explicit `field`.
- Remove forwarding wrappers that only restate existing behavior and blur the encode vs pmodule boundary.

## Module organization policy
- Keep `src/PosetModules.jl` as include-order + API binding/export map only.
- `src/DataIngestion.jl` and `src/Featurizers.jl` are standalone sibling modules (`PosetModules.DataIngestion`, `PosetModules.Featurizers`), not nested under `Workflow`.
- Do not include `DataIngestion.jl`/`Featurizers.jl` from `Workflow.jl`; load them from `PosetModules.jl`.
- Keep shared workflow cache/session helper plumbing in `CoreModules` (not duplicated across Workflow/DataIngestion/Featurizers).
- Keep APISurface contracts authoritative; bind root/advanced symbols from `APISurface.jl` lists rather than ad-hoc `using`/`export` accretion.
- Split by ownership/cohesion, not by line count alone:
  - if a file is one coherent subsystem that is just too large, keep one owner module in `src/<Owner>.jl` and move private implementation fragments into `src/<owner_snake_case>/`,
  - if a file mixes unrelated concepts, split those concepts into sibling owner modules instead of hiding them behind one catch-all owner.
- Prefer semantically honest owners over convenience buckets:
  - `FieldLinAlg` should remain the owner module for field-aware linear algebra, with private fragments under `src/field_linalg/`,
  - `CoreModules` should contain only true low-level shared runtime, not unrelated data/options/results/stats concepts.
- When using the owner-module + private-folder pattern:
  - keep the top-level owner file thin (module docstring, imports, ordered `include`s, public ownership),
  - keep private fragments non-exporting and scoped to that owner only.
- Do not create sibling public modules merely to reduce file size when the code still belongs to one coherent owner.

## Performance policy
- Prioritize hot paths in:
  - `src/FieldLinAlg.jl`
  - `src/Invariants.jl`
  - `src/ZnEncoding.jl`
  - `src/PLBackend.jl`
  - `src/PLPolyhedra.jl`
  - `src/DerivedFunctors.jl`
  - `src/IndicatorResolutions.jl`
- Avoid `Any`, tuple-key dict churn, and repeated materialization in inner loops.
- Prefer typed arrays, linear-index caches, packed representations, and preallocated outputs.
- Prefer sparse-native operations; avoid densification unless explicitly justified.
- Use two-phase threaded design: sequential compile/build cache, threaded read-only compute.
- Threaded loops should write by deterministic index, not shared mutable dictionaries.
- In threaded kernels, avoid relying on `threadid()`-indexed mutable vectors for `push!` accumulation; prefer per-work-item shards plus deterministic reduction.
- Do not keep complexity that does not benchmark as a win; revert or simplify if gains are noise/negative.
- Treat cold and warm behavior separately; optimize both and report both.
- For new batched/vectorized kernels, keep a correctness-equivalent scalar fallback and gate activation by problem size/shape.
- Prefer conservative heuristic gates first; then tune thresholds with benchmark evidence (do not force fast paths globally).
- Keep fast-path switches inspectable with internal (non-exported) knobs so A/B checks are possible without invasive code edits.
- Validate performance across both cache and non-cache call paths; avoid optimizing one while regressing the other.
- Validate serial and threaded regimes separately; a threaded win can hide a serial regression (and vice versa).
- Wire performance gains through canonical high-level entrypoints so simple users benefit without changing call patterns.

## Field and backend expectations
- Ground field support includes `QQ`, `F2`, `F3`, `Fp(p)`, and `RealField`.
- Keep field-generic behavior through `CoreModules` field APIs and `FieldLinAlg`.
- Preserve operation-specific backend selection (rank/nullspace/solve may choose differently).
- Maintain autotune + threshold persistence behavior (`linalg_thresholds.toml`) when touching linalg heuristics.

## PL geometry mode contract
- Canonical PL mode is strict:
  - `:fast`
  - `:verified`
- Validate via `validate_pl_mode` in `CoreModules`.
- Do not reintroduce alias symbols (`:hybrid_fast`, `:hybrid_verified`, `:hybrid`, `:exact`) for mode selection.

## PL geometry cache contract
- Keep cache levels explicit and strict:
  - `:light` (locate/prefilter only),
  - `:geometry` (exact region geometry on-demand),
  - `:full` (precomputed heavy geometry/facets/adjacency support).
- Auto-cache paths should default to `:light` and promote by call intent; avoid eager full-cache work for one-off exact queries.
- Keep a one-off no-cache exact route for small/single-region queries when global cache build cost dominates.
- Ensure compiled/forwarded entrypoints preserve the same cache-level/intent behavior as direct owner-module calls.

## Data ingestion policy
- Primary ingestion implementation lives in `src/DataIngestion.jl`; featurizer API lives in `src/Featurizers.jl`.
- Keep typed filtration dispatch + `FiltrationSpec` conversion model:
  - typed filtrations for extensibility,
  - `FiltrationSpec` as canonical serialized boundary.
- For large point clouds, prefer sparse/edge-driven construction with explicit budgets:
  - `sparsify=:knn|:radius|:greedy_perm`,
  - strict `ConstructionBudget` checks (`max_edges`, `max_simplices`, memory).
- Prefer `nn_backend=:auto` for user-facing defaults:
  - route to NN extension when available,
  - deterministic fallback to bruteforce when unavailable.
- Keep hot sparse graph builders free of Dict churn in inner loops (use packed keys / typed buffers / deterministic finalize).
- Prefer `SimplexTreeMulti` and graded/lazy stages for scale-sensitive paths; avoid eager full `|P|` expansion unless explicitly requested by output stage.
- Maintain preflight guardrails (`estimate_ingestion`) and early combinatorial checks before enumeration.
- Preserve multipers/RIVET comparison harness under `benchmark/ingestion_compare_harness/` and keep regimes apples-to-apples.

## Serialization policy
- Keep internal JSON save paths performance-first by default:
  - default to compact writes (`pretty=false`),
  - keep pretty-printing as an explicit opt-in only.
- For structured posets, avoid dense relation emission by default:
  - use `include_leq=:auto`,
  - require explicit `include_leq=true` when dense `leq` materialization is actually needed.
- Keep strict load validation as the default contract; when needed, expose explicit trusted fast paths (e.g. `validate_masks=false`) for internally produced artifacts only.
- For large homogeneous datasets (point clouds, graphs, etc.), prefer columnar schemas + typed decoding on hot load paths; keep compatibility loaders for prior schemas.
- In external/adaptor parsers, avoid per-entry warning spam in hot loops; aggregate repeated warnings to one warning per file/payload when possible.

## Testing policy
- Do **not** run the full suite unless explicitly requested by the user.
- Run focused tests for touched modules first (targeted harnesses are fine).
- Add/adjust targeted tests with each behavioral change, especially for:
  - mode/backends contracts
  - threaded vs serial parity
  - cache/no-cache parity
  - field parity
  - stage parity (`:simplex_tree`, `:graded_complex`, full pipeline where relevant)
- User expectation: tests must validate **algorithmic correctness**, not just smoke/regression/contracts.
  - Prefer hand-computable oracle fixtures where feasible.
  - If answers are known a priori, assert exact expected outputs.
  - For floating-point/RealField paths, use explicit tolerance-based oracle checks.
- Add negative-contract tests for new API contracts (`@test_throws` on invalid modes/options/shapes).
- Keep optional-dependency tests extension-aware and deterministic (skip only when package is missing).
- For heuristic-gated fast paths, add explicit parity tests that force both gate states (`off` vs `on`) and compare outputs.
- For geometry/invariant outputs with non-semantic ordering differences, canonicalize before comparison (stable sort + explicit rounding/tolerances).
- Include both tiny and moderate-size oracle tests when adding gates, so the gate boundary itself is validated.
- For threaded cache/build/read kernels, include `nthreads()>1` stress/parity tests that can catch write-contention/concurrency violations.
- Keep tests resilient to API-surface curation: correctness tests should call owner modules directly, not rely on curated binding layers.
- Keep shared test aliases centralized in `test/runtests.jl` when possible, and avoid shadowing those aliases with local variables inside test blocks.
- For serialization/schema changes, add both:
  - strict-contract tests (default path),
  - compatibility/fast-path tests (legacy loader path and trusted fast path parity where applicable).
- For module-level kernel changes (`Modules.jl`), include all of:
  - negative API-contract tests (`@test_throws` for invalid indices/comparability/shape conflicts),
  - `ModuleOptions` behavior parity tests,
  - direct non-QQ oracle tests (`F2`, `F3`, `Fp`, `RealField`),
  - randomized differential tests vs a naive oracle (chain + branching posets),
  - threaded contention parity tests for batched calls when `nthreads()>1`,
  - backendized-storage oracle checks (e.g. `BackendMatrix`/Nemo path when available),
  - lightweight perf-regression guard tests for core hot kernels.
- Remove temporary harness files after use.
- Keep test files coherent by subsystem; avoid creating one-off detached test files when an existing subsystem file is the right home.
- Avoid one-off test patterns used once only (e.g., bespoke skip helpers) when plain conditional tests are clearer.

## Benchmark policy
- For performance work, update or add microbenchmarks under `benchmark/`.
- Keep benchmarks focused and reproducible (warmup + comparable cases).
- Report speed deltas and note fidelity/correctness tradeoffs, if any.
- If introducing a new fast path, add a benchmark block that isolates that path (e.g., tiny-query/short-chain cases), and compare against a meaningful baseline.
- For ingestion and backend work, report both:
  - subsystem microbench deltas,
  - workflow-level impact (`encode(...)` path) when applicable.
- Keep external comparisons (e.g. multipers) regime-matched; explicitly label directional vs apples-to-apples results.
- Benchmark scripts should resolve symbols from owner modules (e.g. `CoreModules`, `RegionGeometry`) rather than curated API bindings.
- Prefer built-in A/B probes in the same benchmark script (toggle-based before/after) for deterministic local comparisons.
- Report allocations alongside runtime; GC/allocation regressions are first-class performance regressions.
- For serialization benchmarks, report file-size deltas alongside runtime/allocations; schema and pretty/compact choices are performance features.
- State timing policy explicitly in benchmark outputs (strict-cold vs warm-uncached vs warm-cached) and do not mix them silently.
- Before performance runs, ensure no stale heavy Julia jobs are competing for CPU/RAM; process contention can masquerade as regressions or crashes.
- For large architectural refactors, validate the owner module directly first (parse/smoke load/focused benchmark) before trusting root-module timings from a noisy environment.
- For noisy kernels, run at least two independent benchmark passes and keep both raw outputs before calling wins/regressions.
- Prefer benchmark harnesses with sectional switches (e.g. run only encoding or only dataset blocks) so long runs remain operable in constrained/dev environments.

### Benchmark run checklist (operational)
1. Ensure a clean benchmark environment:
   - stop stale heavy Julia processes,
   - avoid background benchmark/test runs,
   - note machine/thread settings used.
2. Declare timing policy up front:
   - strict-cold, warm-uncached, or warm-cached,
   - use the same policy for all compared variants/tools.
3. Run the baseline first and persist raw outputs (CSV/text) with clear filenames.
4. Run the candidate with identical inputs/flags, then compute explicit deltas:
   - median runtime ratio,
   - allocation/GC delta.
5. Validate correctness parity after performance changes:
   - targeted oracle/parity tests,
   - gate off/on parity when heuristic gates are involved.
6. Check both cache and non-cache paths, and serial vs threaded paths when relevant.
7. Record conclusions with caveats:
   - where wins occur,
   - where regressions/noise remain,
   - next tuning target (if any).

## Code hygiene
- Keep `src/PosetModules.jl` as an API map (avoid dumping implementation helpers there).
- Place logic in the most specific module that owns it.
- Internal-only functions must always use a leading underscore and remain non-exported.
- Internal performance tuning controls (feature flags, thresholds, gates) should remain non-exported and documented near the owning kernel.
- When splitting an owner module across private include files, give the owner file a module docstring and give each fragment a short header comment/docstring stating what that file owns and its scope.
- Keep comments concise and technical; avoid stale docs.
- Keep `PosetModules.jl` comments synchronized with real structure (no stale references like `PublicAPI.jl`/`AdvancedAPI.jl` when those files do not exist).
- If a constant/config appears dead, remove it or wire it fully.
- Prefer explicit, canonical names over aliases in all new code.
- When tightening API contracts, update tests/examples/docs in the same sweep so user-facing guidance stays consistent.
- Keep submodule files free of `export` blocks; only top-level `PosetModules.jl` should export curated API symbols.

## Examples and docs policy
- Examples are onboarding assets: optimize for didactic clarity over showcasing every internal stage.
- Prefer one coherent story per file with clear section headers and minimal control-flow clutter.
- Use comments to explain *semantics* (what object is returned, what key kwargs mean, what mathematical object is being computed), not just mechanics.
- For beginner examples, explicitly annotate return-object semantics (e.g., what `EncodingResult` contains and what key kwargs like `degree` mean).
- Keep beginner examples on canonical public entrypoints; reserve internal/plumbing paths for advanced examples only.

## Extensions and optional deps
- Keep optional ecosystem integrations in `ext/` (Tables, Arrow, CSV, NPZ, Parquet2, Folds, ProgressLogging, etc.).
- Core API should provide clear actionable errors if an extension-backed format/backend is requested but unavailable.

## Workflow cache contract
- Top-level Workflow entrypoints should expose one cache contract:
  - `cache=:auto` for per-call automatic caching.
  - `cache=sc::SessionCache` for explicit cross-call reuse.
- Do not expose `session_cache` as a parallel public keyword in top-level Workflow entrypoints.
- Specialized caches (`ResolutionCache`, `HomSystemCache`, `SlicePlanCache`) should be derived internally from the session cache path, not required from simple users.
- Tests and examples should use `cache=sc` (not `session_cache=sc`) when demonstrating reuse.
- Reuse expectation should be preserved across calls like:
  - `enc = encode(...; cache=sc)`
  - `E = ext(enc1, enc2; cache=sc)` (and similarly `tor`, `resolve`, `invariant`, `slice_barcodes`, `mp_landscape`).

## Test guardrails for API structure
- Keep a regression test that enforces no `export` statements outside `src/PosetModules.jl` (currently in `test/runtests.jl`).

## When uncertain
- Choose the path that is:
  1. faster in hot workloads,
  2. cleaner/smaller in API surface,
  3. stricter in contracts,
  4. easier to benchmark and test.
