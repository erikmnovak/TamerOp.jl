# Compute persistence relative to a boundary

Suppose a segment joins two endpoints. As an absolute space it has no loop.
Relative to its endpoints, however, its oriented edge is a cycle: its boundary
lies entirely in the part we have declared zero. We will compute that relative
class and recover its boundary through the connecting map.

The input is a finite filtered complex and a selection of its cells. This
recipe uses exact rational grades and coefficients in F₃ so that the two
endpoint signs remain distinguishable.

```julia
using TamerOp, SparseArrays
using TamerOp.CoreModules: F3

segment = GradedComplex(
    [[10, 20], [30]],
    [sparse(reshape([-1, 1], 2, 1))],
    [(0//1,), (0//1,), (1//1,)],
)
pair = filtered_pair(segment, [[1, 2], Int[]]; field=F3())
diagram = persistence_diagram(pair)
essential_births(diagram; dim=1)    # [1//1]
```

The inner vectors select **positions within each dimension**, not the labels
10, 20 and 30. Both vertices belong to the subcomplex A; the edge does not.
The edge enters K at 1, giving the relative interval [1, ∞). There is no
relative H₀ class because all vertices have been quotiented out.

To inspect the map behind this answer, choose a scale at which the edge exists:

```julia
A = TamerOp.Advanced
h = A.relative_homology(pair; scale=1//1)
A.dimensions(h)                       # [0, 1]
maps = A.relative_maps(h; dim=1)
maps.connecting                      # 2 × 1: [2; 1] over F₃
```

The connecting map sends the edge class to its boundary, `[20] - [10]`.
The coefficient 2 is −1 in F₃. Rows are the chosen basis of H₀(A); columns
are the chosen basis of H₁(K,A). These are scale-specific bases, not canonical
names for barcode intervals. Inspect the basis and its underlying cells when
working with a larger example:

```julia
A.basis(h; dim=1)
c = A.relative_complex(h)
A.relative_cells(c; dim=1)           # ambient position 1, cell ID 30
A.relative_boundary(c; dim=1)        # a 0 × 1 zero matrix
```

## Let the subcomplex change later

A cell may already be present in K before it enters A. For example, let the two
vertices enter A at 2 and 3, and the edge at 4:

```julia
changing = filtered_pair(segment, [[1, 2], [1]];
    grades=[[2//1, 3//1], [4//1]], field=F3())
d = persistence_diagram(changing)
finite_intervals(d; dim=0)          # [(0//1, 1//1), (0//1, 2//1)]
finite_intervals(d; dim=1)          # [(3//1, 4//1)]

before = A.relative_homology(changing; scale=3//1)
after = A.relative_homology(changing; scale=4//1)
A.relative_structure_map(before, after; dim=1)  # 0 × 1
```

When the second endpoint enters A, the edge becomes a relative cycle. When the
edge itself enters A, that class becomes zero. Deleting A's eventual cells
from the start would lose this interval.

An empty selection, `[Int[], Int[]]`, recovers absolute persistence. Selecting
all cells with their inherited grades gives zero relative homology. Supplying
an edge without its nonzero boundary terms, or letting it enter A before its
boundary, raises an error.

Use the [pair reference](../src/reference/persistence_variants.md) for exact
input and map conventions, and the [implementation account](../implementation/relative_persistence.md)
for the two reduction routes. Standard interval queries and diagram analysis
are described in the [ordinary-persistence guide](../ordinary_persistence.md).
