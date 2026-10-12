# API reference and operation help

When you know which operation you need, its reference explains the arguments
it accepts, the object it returns and the conditions under which the result
has the stated meaning. It also describes how to check your inputs. To see
how several operations work together on an example, begin with
[Using TamerOp](using.md), where the guides follow and interpret the computation.

## Find an operation's contract

An operation's contract is the combination of its accepted inputs, guarantees
and requirements. Julia's help mode reads the method documentation shipped
with the package, so you can consult these details in the version you are
using. Import TamerOp, then type `?` at the Julia prompt to enter help mode:

```julia
import TamerOp as OP
import TamerOp.Advanced as OA
```

```text
help?> OP.encode
help?> OP.encoding_module
help?> OA.structure_map
```

`OP` gives you the main workflows and operations for inspecting results.
`OA` gives access to more detailed ways of constructing, querying and checking
objects. Some operations belong to a particular part of the library, as in
`OP.FieldLinAlg.solve_fullcolumn` for a linear algebra calculation. A Julia
function can have several methods for different kinds of arguments; consult
the method you intend to call, since its conditions may differ from those of
another method. Help displays these calling forms, their mathematical meaning
and the available options.

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
