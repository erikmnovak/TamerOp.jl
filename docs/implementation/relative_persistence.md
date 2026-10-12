# Computing persistence of a filtered pair

A relative class is represented by a chain whose boundary lies in a designated
subcomplex. For a segment K and its endpoints A, the edge is such a chain:
its two ends vanish in C(K)/C(A), while the edge remains. This makes relative
homology useful when a boundary, reference region or already-explained part
of a complex should count as zero.

The computational difficulty is that A can grow independently of K. A cell may
enter K, contribute to relative homology, and later enter A. Quotient chain
spaces can therefore lose coordinates even though both input filtrations only
grow. The package handles this with two reduction routes, while providing the
actual quotient complexes and maps independently of either route.

The [recipe](../recipes/relative_pair.md) constructs a small pair. The
[reference](../src/reference/persistence_variants.md) fixes the full input,
field, scale and matrix conventions.

## Keep both entry times

`FilteredPair` stores a copy of K's one-parameter graded complex, the selected
ambient positions of A's cells, their A-entry grades, the field and the order.
Dimension-local positions identify cells even when user-supplied IDs repeat.
A cell's ambient grade and subcomplex grade are different facts.

For sublevels, each selected cell must satisfy

```math
\operatorname{grade}_K(\sigma)\leq\operatorname{grade}_A(\sigma).
```

Every nonzero boundary term of a selected cell must also be selected and have
an A-grade no later than that cell. Superlevels reverse these inequalities.
Boundary equations are checked over the requested prime field, including
signed coefficients. This establishes a filtered coordinate subcomplex; it
does not infer a geometric realization from labels.

At a scale t, ambient active cells form Kₜ. Selected cells whose A-grade has
been reached form Aₜ. The remaining active cells form a coordinate basis of
the quotient Qₜ. Its differential is the ambient differential with both source
and target restricted to those complementary coordinates. The omitted boundary
terms vanish modulo Aₜ. Closure of Aₜ ensures that this operation still gives
a differential with square zero.

## Decide whether one quotient complex suffices

If every selected cell inherits its ambient grade, it is zero in the quotient
as soon as it appears. Removing those cells once produces a graded chain
complex whose ordinary reduction computes the relative barcode. This route
also includes an empty A and the equal pair A = K.

A later A-entry changes the situation. Consider an edge present in K at 1,
whose endpoints enter A at 2 and 3 and whose edge enters A at 4. The edge
becomes a relative cycle at 3 and disappears at 4. Removing all eventual
A-cells would erase the interval [3,4).

For this case the package builds the filtered algebraic mapping cone of the
inclusion i : C(A) → C(K). In degree d its coordinates are

```math
\operatorname{Cone}(i)_d=C_d(K)\oplus C_{d-1}(A),
```

```math
D_d=
\begin{pmatrix}
\partial_K&i\\0&-\partial_A
\end{pmatrix}.
```

The first block receives K's grades. The suspended A block receives A's own
entry grades. The minus sign is essential away from characteristic two:
D² = 0 follows from ∂Ki = i∂A. Negated coefficients are formed as modular
residues so that even the smallest machine integer is handled without signed
integer overflow.

The chain map `(k,a) ↦ [k]` from the cone to the quotient is a
quasi-isomorphism at every scale. Its kernel is the cone of the identity on
Aₜ and is contractible. These projections commute with changing t, giving an
isomorphism of persistence modules, not just matching Betti numbers at isolated
scales. Standard exact column reduction can therefore compute the required
barcode from a filtration whose chain coordinates only enter, never disappear.

| Subcomplex entry | Reduction input | Chain coordinates retained |
| :--- | :--- | :--- |
| A inherits every selected K-grade | Graded quotient | Cells outside A |
| Any selected cell enters A later | Filtered cone of inclusion | K-cells and suspended A-cells |

Both inputs go through the shared ordinary persistence reducer. The returned
intervals describe the quotient homology in the original grading.


The quotient definition and connecting map are the standard chain-level
construction of relative homology; see [Allen Hatcher, *Algebraic Topology*,
Section 2.1](https://pi.math.cornell.edu/~hatcher/AT/ATch2.pdf).
[Blaser and Brun, *Relative Persistent Homology*](https://doi.org/10.1007/s00454-022-00421-9)
discuss the persistent quotient viewpoint and relative geometric constructions.
TamerOp's general pair route does not implement their specialized
Delaunay–Čech construction.

## Recover maps in actual quotient coordinates

The barcode calculation stays separate from explicit homology queries.
`relative_complex(pair; scale)` constructs Aₜ, Kₜ and Qₜ directly, retaining
the ambient identity of every surviving coordinate. A filtration map places an
old coordinate in its corresponding new row; in Q, that row can disappear
because the cell has entered A, in which case its image is zero.

`relative_homology` passes each complex's adjacent differentials to the shared
homology engine. The engine chooses cycle representatives and computes their
coordinates modulo boundaries. Given these bases, inclusion and quotient
projection are applied to source cycles and resolved in the target homology
basis. This is the same operation used for the induced filtration maps.

For the connecting map, lift a quotient cycle z to its complementary ambient
coordinates and compute its ambient boundary. The quotient-cycle equation
says this boundary is supported in A. Its coordinates in H_(d−1)(A) give
δ[z]. Changing the lift adds an A-boundary, so the answer is independent of
that choice. The sign convention is

```math
\delta[z]=[\partial_K\widetilde z].
```

On an oriented edge from v₀ to v₁, this returns `[v₁] - [v₀]`. In F₃ the
coefficients are 1 and 2, respectively. The matrices depend on the chosen
homology bases, but their maps, ranks, exactness and naturality do not.

## Storage, reuse and the boundary of the result

The inherited-grade route has one column per cell outside A. The cone has at
most the number of K-cells plus the number of selected A-cells, and adds only
the inclusion entries and shifted A-boundaries to the sparse representation.
Both routes use the established prime-field persistence reducer; no separate
persistent elimination engine or result cache is introduced.

A barcode request does not compute the three homology bases at every scale.
Explicit snapshots copy their source pair and store the selected sparse
boundaries. Homology queries can subsequently materialize dense cycles and
coordinate factors. Reuse a snapshot when inspecting several maps at one
scale; the retained homology engine reuses its factors. Results at different
scales carry separate chosen bases and must be related by their induced maps.

The diagram records original cell IDs, selected positions, both grade sets,
field, order and the reduction route. Its ordinary interval/analysis/JSON
interfaces remain usable. Cone representatives are intentionally not passed
off as source chains, and pair metadata alone cannot reconstruct omitted
boundary matrices or source geometry. The explicit quotient queries supply
chain representatives in ambient cell coordinates; they do not certify a
matching between these bases and individual barcode intervals.

Implementation: [`relative.jl`](../../src/ordinary_persistence/relative.jl)
and the shared [homology engine](../../src/ChainComplexes.jl).

[Bibliography](references.md)
