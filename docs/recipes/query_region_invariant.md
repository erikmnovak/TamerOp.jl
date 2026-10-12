# Measure what persists through a branching region

Start with a finite module and a connected, order-convex set of its labels.
The result is the number of independent classes that persist compatibly
through that whole region. This example needs TamerOp and rational matrices;
it does not need a filtration or graphics backend.

The two source lines below map into the same line in a common target plane.
Their common persistent image has dimension one.

```julia
import TamerOp as OP
A = OP.Advanced
QQ = Rational{BigInt}
P = A.FinitePoset(Bool[1 0 1; 0 1 1; 0 0 1])
M = A.PModule{QQ}(P, [1, 1, 2],
    Dict((1, 3) => reshape(QQ[1, 0], 2, 1),
         (2, 3) => reshape(QQ[1, 0], 2, 1));
    field=OP.CoreModules.QQField())
region = [1, 2, 3]
A.check_generalized_rank_region(M; vertices=region, throw=true)
rank = OP.generalized_rank(M; vertices=region)
@assert rank == 1
```

The labels concern this finite poset. The query does not silently construct
an ambient geometric region. `[1,2]` would be disconnected; on a longer chain,
omitting a label between two selected labels would violate convexity.

Request matrices only if you need to inspect the comparison. `limit_basis`
stores compatible choices in the direct sum of the selected stalks;
`colimit_projection` identifies vectors related by the maps. The comparison
uses one stalk to send a compatible choice to its common quotient class.

```julia
witness = OP.generalized_rank(M; vertices=region, witnesses=true)
@assert OP.generalized_rank(witness) == rank
comparison = A.comparison_map(witness)
@assert size(comparison) == (2, 1)
```

To see why this is a region question, replace the second map by the column
`QQ[0,1]`. Both arrows still have rank one, but their image lines intersect
only in zero, so the whole-region rank becomes zero.

The [usage guide](../tutorials/generalized_rank_queries.ipynb) adds signed
reconstruction, grid boundary barcodes and continuous worms. The
[implementation account](../implementation/generalized_rank.md) explains the
exact comparison; the [region-view reference](../src/reference/visual_sessions.md#query-generalized-rank-regions)
specifies linked selection and its budgets.
