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
- Do not keep complexity that does not benchmark as a win; revert or simplify if gains are noise/negative.
- Treat cold and warm behavior separately; optimize both and report both.

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

## Code hygiene
- Keep `src/PosetModules.jl` as an API map (avoid dumping implementation helpers there).
- Place logic in the most specific module that owns it.
- Internal-only functions must always use a leading underscore and remain non-exported.
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
