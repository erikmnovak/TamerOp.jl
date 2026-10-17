# From points to a finite encoding

Use this workflow when the input is a point cloud with a numerical value attached
to each point. We let two thresholds vary: the largest permitted edge length and
the largest permitted vertex value. This produces a **bifiltration**, a family of
spaces that grows as either threshold increases.

Run [data_to_encoding.jl](../examples/data_to_encoding.jl) in the Julia environment
containing TamerOp:

```julia
import TamerOp
include(joinpath(pkgdir(TamerOp), "agent", "examples", "data_to_encoding.jl"))
result = run_data_to_encoding_example()
```

The three points lie at `0`, `1` and `2` on a line. Their attached values are
`0`, `1` and `0`, so a value threshold of `0` includes the two endpoints, while a
threshold of `1` admits the middle point. The script constructs the family with
`FunctionRipsFiltration` and calls
`encode(...; degree=0, field=OP.CoreModules.F2())`, with `OP` naming TamerOp. Degree zero homology has
one independent vector for each connected component.

## Read the result

A **finite encoding** stores the spaces and maps using a finite collection of
labels, together with a map from parameter points to those labels. Keep the
returned `EncodingResult` so later queries can use the original parameters.
Here each parameter point is `(edge_length, vertex_value)`.

| Parameters | Included points and connections | Components |
| --- | --- | --- |
| `(0, 0)` | The two endpoints, with no edge | 2 |
| `(0, 1)` | All three points, with no edges | 3 |
| `(1, 0)` | The endpoints remain too far apart | 2 |
| `(1, 1)` | The middle point connects the endpoints | 1 |
| `(2, 0)` | The endpoints connect directly | 1 |

The script checks this table through `rank_map(encoding, point, point)`.
For equal source and target, the map is the identity and its rank is the
space's dimension. For different comparable points, the **rank** counts how
many independent classes remain in the image. The map from `(1, 0)` to
`(1, 1)` has rank `1`: the two components merge into one. Increasing edge
length to `(2, 0)` also gives rank `1`, through a different change to the input.

## Adapt the computation

Replace the point matrix and `vertex_values` together; there is one value for
each row of the matrix. The first threshold is the distance between points,
not half that distance. Both coordinates increase in this construction, and
the vertex condition is a sublevel condition, `value <= threshold`.

`max_dim=1` includes vertices and edges, which suffice for connected components.
To study holes with `degree=1`, include triangles with at least `max_dim=2` so
the calculation can see when a loop fills. The example uses arithmetic in F₂,
the field with two elements; choose coefficients for the intended calculation.

Use selected `rank_map` queries when you need only a few comparisons. Read
`describe(encoding)` for the stored construction and `dimensions(encoding)`
when you need dimensions throughout the finite description. Request
`encoding_module(encoding)` when you need the full module and its maps.

If a query fails, check its parameter order and whether the source is at most
the target in both coordinates. A finite label is an internal index; it is not
a parameter value. Inspect the installed `encode` and `rank_map` docstrings
before changing options or translating an example from another version.

Continue with [finite encodings](https://tamerop.com/finite_encodings.html) and
the [API reference](https://tamerop.com/collections/api.html), or return to the
[agent guide](../README.md).
