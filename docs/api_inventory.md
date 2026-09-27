# Maintaining the API inventory and documentation backlog

Use the generated [runtime inventory](api_inventory.toml) together with the
authored [coverage manifest](api_coverage.toml). They answer different questions:
the inventory records what bindings actually exist; the manifest records where
we intend to explain them and what work remains. The
[first learning path](learning_path.md) determines the first reference-writing
pass. Completing the other families is not a prerequisite for publishing that
finished path.

## Reproduce the inventory

Run these commands from the package root in its instantiated environment:

```sh
julia --startup-file=no --project=. docs/build_scripts/api_inventory.jl
julia --startup-file=no --project=. docs/build_scripts/api_inventory.jl --check
julia --startup-file=no --project=. docs/build_scripts/check_api_coverage.jl
```

The first command updates the tracked snapshot. The second loads the package
again and fails if the snapshot differs. It compares the Julia version as well
as the generated data, so switching even the Julia patch version requires a
reviewed regeneration. The last command validates coverage assignments and
reports the backlog from TOML without loading TamerOp. It does not replace the
runtime freshness check.

The inventory uses normal `import TamerOp`; it does not load source fragments,
test helpers, local examples, or files under `audit/`. Source fingerprints cover
`Project.toml`, Julia source under `src/` and `ext/`, and the generator. No
timestamp, checkout location, or Git commit enters the snapshot. Optional
integrations are not deliberately activated; any extensions loaded transitively
are recorded. Extension-specific methods still need separate semantic review.

## What the initial reconciliation found

The initial snapshot was generated on Julia 1.12.1. Counts exclude module
self-bindings and describe bindings or object identities as labelled; they
are not counts of independent mathematical capabilities.

| Measurement | Count |
| --- | ---: |
| Declared `SIMPLE_API` names | 203 |
| Declared `ADVANCED_API` names, including the simple list | 1,031 |
| Actual root exports | 206 |
| Actual `Advanced` exports | 1,205 |
| Advanced binding-table names absent from `ADVANCED_API` | 138 |
| Declared advanced names undefined or not exported | 0 |
| Curated qualified owner bindings | 1,186 |
| Distinct intended public objects after alias grouping | 1,218 |

The root's three extra exports are `Advanced`, `SIMPLE_API`, and `ADVANCED_API`.
The `Advanced` export count includes 37 modules. The static union of declared
names and binding-table names is therefore not a runtime export count.

Ten binding-table rows point to a different runtime target object than the
listed owner object. Nine reflect the simple `Workflow` binding taking
precedence over a same-named advanced owner generic. Both objects remain in
the inventory: their capabilities must not disappear merely because the names
match. The remaining row is `contains`: the preexisting `Base.contains`
binding prevents the advanced table from installing/exporting
`PLPolyhedra.contains`. The qualified owner operation remains accounted for;
the unrelated, unexported target is retained as reconciliation evidence and
excluded from the intended-public denominator. This documentation preparation
records the discrepancy without changing the package API.

## Identity, scope, and limits

Each `objects` record has a stable qualified `canonical_binding`, its aliases,
runtime defining module, declaration owners, tiers, and per-binding metadata.
Functions, types, and modules are grouped by runtime object identity, not by
name. Scalar constants remain separate bindings even if their values happen
to compare equal. A canonical key is a documentation identity, not a claim
that all methods of a generic belong to one subsystem.

For example, `DataFileIO.artifact` is distinct from `Featurizers.artifact`.
Conversely, several bindings of a shared generic need one canonical reference
location plus explanations of materially different method families. The
inventory's docstring flag only records presence. It does not prove that all
methods, mathematical assumptions, errors, or workflows are explained.

Qualified owner APIs require an intent decision beyond exports. Curated owner
bindings are retained even when a facade selects another object. The manifest
also explicitly selects documented owner operations outside those tables,
including field constructors, file-inspection accessors, and external readers.
Its `qualified_apis` entries record the evidence and assigned family.

The remaining `owner_candidates` are a review queue, not automatically public
API and not automatically internal. The scan retains local non-underscore
functions/types/modules and documented or explicitly public bindings, including
aliases imported from an owner's own child modules. It omits unrelated-owner
and Base imports. This is a conservative discovery aid, not proof that every
intentionally qualified operation has already been identified. Further intent
review belongs in the backlog, not in an inflated completion percentage.

## How to advance coverage honestly

The twelve families follow the planned reference organization. Owner-based
routing supplies an initial home; explicit overrides route cross-cutting
workflows and options by mathematical task. These assignments are provisional
until method-family review. In particular, the broad derived-functor owner
still needs finer separation of resolutions, products, and spectral sequences.
Every intended curated object must nevertheless have an unambiguous assignment.

The manifest records existing guide paths, planned reference paths, example
sources, and separate reference/example/review statuses. A planned path is
allowed to be absent while its status is `backlog`. An existing guide or a
passing oracle does not make its entire API family documented. Empty example
lists are explicit gaps. The first-path list selects the bindings to explain
first, including the specific box-presentation `encode` method and the
method-dependent results of `dimensions`.

For each writing pass:

1. Select a reader question and a method family from the backlog. Check whether
   an existing guide already owns its contract before writing another account.
2. Explain the mathematical purpose, inputs, options, returned object, field and
   domain assumptions, meaningful errors, and costs. Link a verified example.
3. Add or refine a manifest assignment when a generic spans families; retain
   one canonical home and use cross-references for other contexts.
4. Record execution and editorial review separately. Only update completion
   status when the corresponding artifact and review exist.
5. Regenerate the inventory after API changes and run both checks. Review new
   exports, owner candidates, and binding collisions without deleting working
   APIs merely to simplify the count.

The current manifest intentionally marks reference authoring as backlog. Its
purpose is to make the remaining work concrete while the first teaching path
and documentation site are implemented.
