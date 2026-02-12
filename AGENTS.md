# AGENTS.md

Guidance for coding agents working in `tamer-op`.

## Core intent
- Optimize for performance and clean architecture.
- Keep APIs ergonomic for new users.
- Prefer clarity over compatibility shims (project is pre-release).

## Project-specific principles
- No legacy-compatibility layers unless explicitly requested.
- No alias-heavy APIs in performance-critical surfaces.
- Keep canonical mode/value contracts strict and explicit.
- Avoid hidden behavior switches; backend/mode choice should be inspectable.

## API policy
- Breaking API changes are acceptable when they improve clarity/performance.
- Still keep convenience defaults (users should not need to pass empty `Options()` everywhere).
- Prefer one canonical public path over multiple near-duplicate wrappers.
- If adding new options/fields, thread them through all relevant call paths (do not silently drop).
- Prefer a single cache keyword surface in Workflow APIs; avoid multi-keyword cache controls.

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

## Testing policy
- Do **not** run the full suite unless explicitly requested by the user.
- Run focused tests for touched modules first (targeted harnesses are fine).
- Add/adjust targeted tests with each behavioral change, especially for:
  - mode/backends contracts
  - threaded vs serial parity
  - cache/no-cache parity
  - field parity
- Remove temporary harness files after use.

## Benchmark policy
- For performance work, update or add microbenchmarks under `benchmark/`.
- Keep benchmarks focused and reproducible (warmup + comparable cases).
- Report speed deltas and note fidelity/correctness tradeoffs, if any.

## Code hygiene
- Keep `src/PosetModules.jl` as an API map (avoid dumping implementation helpers there).
- Place logic in the most specific module that owns it.
- Keep comments concise and technical; avoid stale docs.
- If a constant/config appears dead, remove it or wire it fully.

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

## When uncertain
- Choose the path that is:
  1. faster in hot workloads,
  2. cleaner/smaller in API surface,
  3. stricter in contracts,
  4. easier to benchmark and test.
