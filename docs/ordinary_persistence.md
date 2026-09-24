# Ordinary persistent homology

`persistence_diagram` computes the persistent homology of a finite
one-parameter chain complex. Its current reduction backend works over
`TamerOp.CoreModules.F2()` only. Integer boundary coefficients are read modulo
two; other coefficient fields and symbolic field aliases are rejected.

```julia
using TamerOp

# A square ring appears at 0 and is filled at 5.
values = zeros(Int, 3, 3)
values[2, 2] = 5
D = cubical_persistence(values)
finite_intervals(D; dim=1)       # [(0, 5)]
essential_births(D; dim=0)       # [0]
persistence_intervals(D; dim=0)  # [(0, Inf)]
describe(D)
provenance(D)
```

The result represents homological degrees `H_0`, `H_1`, and so on. Degree
arguments are named keywords. A nonnegative degree above the stored range has
no intervals. Negative degrees are invalid. The two exact-data accessors return
copies, so editing their output does not modify the diagram.

## Interval conventions

A sublevel filtration includes cells whose grade is at most the parameter.
Its finite pair `(birth, death)` represents `[birth, death)`: the class exists
at its birth and is absent at its death. Essential classes persist to `Inf`.

A superlevel filtration includes cells whose grade is at least the parameter,
with maps toward decreasing parameters. Its pair `(birth, death)` has
`birth > death` and represents `(death, birth]` in the original numerical
coordinates. Essential classes persist to `-Inf`.

Cells at equal grades are inserted together mathematically. Internal reduction
orders faces before cofaces to break ties, but its zero-length pairs represent
the zero persistence module and are omitted from the returned barcode. There
is no diagnostic option that changes this barcode convention.

```julia
upper = cubical_persistence(5 .- values; order=:superlevel)
finite_intervals(upper; dim=1)       # [(5, 0)]
persistence_intervals(upper; dim=0)  # [(5, -Inf)]
```

Finite endpoints and essential births retain the supplied grade type, including
integers, rationals, arbitrary precision numbers and exact real algebraic
values. Essential births are stored separately from finite bars.
`persistence_intervals` combines them with an explicit floating infinity marker
without converting any birth or finite death. Use `finite_intervals` and
`essential_births` when exact homogeneous element types matter.

## Cubical inputs and periodicity

`cubical_persistence(values; input=:top_cells)` interprets a two-dimensional
array as grades on squares. A face receives the minimum value of its incident
squares for sublevels, or their maximum for superlevels. This gives the union
of closed squares selected at each parameter.

With `input=:vertices`, entries grade vertices. Each cell receives the maximum
of its vertices for sublevels (the lower star), or their minimum for superlevels
(the upper star). Vertex input supports arrays of any positive dimension.
These input conventions describe different filtrations and must be matched
when comparing software.

`periodic=true` identifies both ends of every axis. A tuple of Booleans selects
axes individually. One periodic axis gives a circle factor, including an axis
with a single cell; two periodic axes give a torus. Nonperiodic axes retain their
boundary. Inputs must be nonempty and have finite values.

```julia
torus = cubical_persistence(fill(2//3, 1, 1); periodic=(true, true))
essential_births(torus; dim=0)   # [2//3]
essential_births(torus; dim=1)   # [2//3, 2//3]
essential_births(torus; dim=2)   # [2//3]
check_torus_persistence(torus; throw=true)
```

`check_torus_persistence` checks the diagram and its expected essential counts
`(1,2,1)`. Counts alone do not prove that an unknown source is homeomorphic to
a torus. It returns a report by default; `throw=true` rejects failures. Use
`check_h1=false` or `check_h2=false` only when explicitly checking fewer degrees.

The equivalent typed vertex workflow is
`persistence_diagram(ImageNd(values), CubicalFiltration(periodic=true))`.
It preserves construction budgets and an explicitly supplied ingestion cache.
Both vertex entrypoints build cubical topology through the public ingestion
builder using order ranks, then restore original grades exactly. Thus the
builder's floating grade storage never rounds the actual vertex values; upper
stars also avoid negating integers at the limits of their range.

## General filtered complexes

Pass a one-parameter `GradedComplex` directly to `persistence_diagram`, or pass
data and a typed filtration to construct the complex first:

```julia
D = persistence_diagram([0.0 2.0; 2.0 0.0], RipsFiltration(max_dim=1))
finite_intervals(D; dim=0)  # [(0.0, 2.0)]
```

A direct complex is checked before reduction: packed cell dimensions and sparse
storage must be valid, boundaries must have the correct shapes, double
boundaries must vanish modulo two, and every nonzero boundary coefficient must
respect the selected filtration order. Empty typed complexes are accepted.
This is an algebraic filtration contract over `F2`; even integer boundary
coefficients vanish over that field.

Changing `order` on an arbitrary ingestion request does not turn a lower-star
builder into an upper-star builder. The resulting grades must satisfy the
chosen order or the call fails. The dedicated cubical vertex route handles the
upper-star construction explicitly. Other builders keep their own geometry and
grade-precision contracts; exact reduction does not certify their geometry.

`provenance(D)` records the field, homological convention, filtration direction,
interval convention, grade type and reduction backend. Cubical routes also
record array shape, periodicity and input convention. A generic ingestion build
does not currently carry execution provenance, so its effective construction
and any substitution are reported as `:not_recorded`, rather than inferred from
the requested filtration name. A hand-built `PersistenceDiagram` likewise
reports its computation history as `:not_recorded` unless metadata was supplied;
constructing stored intervals does not claim that the reducer ran.

## Results and figures

`result_summary(D)` and `describe(D)` show the same cheap mathematical summary.
The advanced `field(D)` and `filtration_order(D)` accessors return the coefficient
field and order. `check_persistence_diagram(D)` validates a hand-built diagram;
wrap its report with `persistence_validation_summary` for compact notebook
presentation. It checks barcode storage, not an unrecorded source computation.

`visualize(D; kind=:persistence_diagram, dim=1)` and
`visualize(D; kind=:barcode, dim=1)` use the shared visualization interface.
Plotting makes explicit Float64 display copies while retaining exact intervals
in the visualization metadata. Values that overflow or distinct endpoints
that collapse at display precision are rejected. Essential bars have a separate
labelled infinity lane or continuation arrows, including `-Inf` for superlevels.
Rendering requires an explicitly loaded plotting extension; constructing and
inspecting a diagram does not require a plotting package.
