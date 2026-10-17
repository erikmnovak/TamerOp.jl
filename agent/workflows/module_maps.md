# Compute with a module's spaces and maps

Use this workflow when you already have vector spaces and linear maps on a
finite partially ordered set. Such a set, or **poset**, specifies which labels
can be compared. A **module** assigns a space to each label and a linear map to
each ordered pair, with composites agreeing along different paths.

Run [module_maps.jl](../examples/module_maps.jl) in your TamerOp environment:

```julia
import TamerOp
include(joinpath(pkgdir(TamerOp), "agent", "examples", "module_maps.jl"))
result = run_module_maps_example()
```

The example is a square with rational coefficients:

```text
                    4: Q
                  /      \
            [1 0]          [0 1]
               /              \
          2: Q²              3: Q²
               \              /
                [1;0]    [0;1]
                   \      /
                    1: Q
```

All arrows point upward. Following either path from label `1` to label `4`
sends the number `1` to `1`, so both paths give the same map. The script checks
this equality and inspects what happens to each basis vector at label `2`:
`[1,0]` survives, while `[0,1]` maps to zero. The dimensions alone would not
tell us which vector survives.

## Construct and inspect the module

The script imports `TamerOp` as `OP` and `TamerOp.Advanced` as `OA`.
`OA.FinitePoset(order)` receives a Boolean matrix where `order[i,j]` means
`i <= j`, including the diagonal and transitive relations. `OA.PModule` receives
the dimensions and a matrix for every **cover relation**, an ordered pair with
no label strictly between its endpoints. Other maps are obtained by composition.

A map's rows correspond to its target space and its columns to its source.
For example, the map from `1` to `2` has shape `(2,1)`. The matrices use `OP.QQ`
entries and `OP.CoreModules.QQField()`, so calculations use exact rational
arithmetic.

`OA.check_module` checks dimensions and stored map shapes.
`OA.check_module_functoriality` checks that different paths give the same
composite. Both matter when supplying maps yourself. After those checks,
`OA.structure_map(module_; source=1, target=4)` returns the composite matrix,
and `OP.rank_map(module_, 1, 4)` returns its rank without requiring you to
interpret its entries.

## Adapt the computation

Replace the order, dimensions and cover matrices together. Keep returned
structure matrices unchanged; they can share storage with later queries.
If a map has the wrong shape, compare its row and column counts with the target
and source dimensions. If the path check fails, inspect the competing
composites before continuing with algebraic operations on the module.

Labels `2` and `3` are incomparable. There is no prescribed structure map
between them; inserting a zero matrix would change the mathematical question.
Likewise, these labels have no point cloud coordinates attached to them. To
interpret the module on an original parameter space, retain or construct the
parameter assignment as part of a finite encoding.

Continue with [spaces and maps](https://tamerop.com/guides/spaces_and_maps.html) and
[finite encodings](https://tamerop.com/finite_encodings.html), or return to the
[agent guide](../README.md).
