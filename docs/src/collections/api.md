# API reference and operation help

Use reference material when you know the operation you need and want its
accepted arguments, return object, mathematical scope or validation behavior.
For a connected workflow with interpreted results, begin with
[Using TamerOp](using.md).

## Find an operation's contract

Julia's help mode reads the method documentation shipped with the package.
Import TamerOp, then type `?` at the Julia prompt to enter help mode:

```julia
import TamerOp as OP
import TamerOp.Advanced as OA
```

```text
help?> OP.encode
help?> OP.encoding_module
help?> OA.structure_map
```

`OP` exposes the main workflows and result accessors. `OA` exposes detailed
constructors, queries and validators. Subsystem owners also provide qualified
APIs, for example `OP.FieldLinAlg.solve_fullcolumn`; look up the method you
intend to call rather than inferring its contract from another overload.
Help displays method signatures, mathematical meaning and applicable options.

## Operations in context

These links lead to explanatory guides. The named operations also have
method-specific help in Julia.

| Task | Operations to look up | Guide |
| --- | --- | --- |
| Retain a finite description | `OP.encode`, `OP.encoding_poset`, `OP.encoding_map` | [Finite encodings](../../finite_encodings.md) |
| Query spaces and maps | `OP.dimensions`, `OA.locate`, `OA.structure_map` | [Exploring spaces and maps](../../spaces_and_maps.md) |
| Request deferred computations | `OP.describe`, `OP.encoding_module` | [Inspection and explicit computation](../../lazy_inspection.md) |
| Compute an ordinary barcode | `OP.persistence_diagram`, `OP.cubical_persistence` | [Ordinary persistence](../../ordinary_persistence.md) |
| View or export a result | `OP.available_visuals`, `OA.visual_spec`, `OP.visualize`, `OP.save_visual` | [Choosing views and inspecting intervals](../../visualization.md) |
| Interpret algebraic scope | `OP.hom`, `OP.ext`, `OP.tor`, `OP.resolve` | [Mathematical categories](../../math_categories.md) |
| Choose coefficients and geometry | `OA.EncodingOptions`, `OP.PipelineOptions` | [Options and their effects](../../option_contracts.md) |

## Shared contracts

<!-- COLLECTION: api -->

The [API inventory guide](../../api_inventory.md) explains how the public
bindings and documentation coverage are maintained. It is contributor
material; use operation help and the relevant contracts above for a query's
calling conventions.
