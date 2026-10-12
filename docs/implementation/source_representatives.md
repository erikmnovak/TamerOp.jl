# Retaining representatives and tracing them to source cells

A persistence interval says when a class exists. Explaining where it exists
requires more information: a chain in the computed cell basis, the relationship
between that basis and the source cells, and geometry for those cells. None of
these can be reconstructed from interval endpoints alone. A finite encoding
likewise retains the spaces and maps needed for its module queries, but does
not automatically retain a comparison with every discarded source-chain basis.

This account follows the retained correspondence from computation to source.
The [source-cell guide](../tutorials/source_cells.ipynb) explores the pictures;
the [reference](../src/reference/source_representatives.md) specifies the API.
Ordinary persistence is a direct consumer. A supplied multiparameter cellular
complex can use the same machinery at a parameter without passing through an
ordinary barcode.

## Three records answer three different questions

`FilteredComplex` owns the actual cells, integer boundaries, grade sets, source
identities and geometric vertices. Cubical identities are origins and axis
masks; simplicial identities are ordered source-vertex tuples. A cell's index
is dimension-local in a chain and global in the complex viewer. Conversion
uses dimension offsets, never a spatial nearest-neighbor query.

`SourceCorrespondence` retains a map $i:C\to S$ from computed chains to source
chains. Its optional retraction $p:S\to C$ serves cohomology. Both are sparse,
degree-preserving matrices over an exact rational or prime field. Ordinary
persistence selects a prime field; direct finite/multiparameter chain queries
can use rational coefficients as well. `SourceChain` records
one resulting chain or cochain at one parameter, with that field and its
source geometry. Boundary matrices define orientations; geometric projections
and residue labels do not redefine the basis.

![A retained chain map lifts a selected cycle or death filling to original cells. The source boundary and activity are checked before drawing. Cocycles use the transpose of a retained retraction.](../assets/implementation/source_representatives.svg)

The ordinary diagram stores a correspondence only when the caller supplies it
or requests `source=true`. `representatives=true` and `cocycles=true` are
independent controls. The ordinary interval-only route pays no source retention
cost. A source request uses explicit cells; it cannot also promise the memory
behavior of implicit Rips reduction.

## The map must respect boundaries and appearances

In degree $d$, the lift satisfies

$$
\partial^S_d i_d=i_{d-1}\partial^C_d.
$$

Consequently a computed cycle lifts to a source cycle, and a computed death
filling lifts to a filling of that same source cycle. These identities are
checked over the actual field. Signs that disappear over $\mathbb F_2$ still
matter over an odd prime field.

A nonzero matrix coefficient also requires filtration compatibility. Whenever
its domain cell is present, its destination cell must be present. For a
multicritical cell, each appearance grade is checked against the destination's
union of upper sets, with the declared orientation on each coordinate. A
comparison at a single convenient parameter would be insufficient.

A supplied lift need only be a filtered chain map. Supplying $p$ additionally
requires $pi=1$, establishing a split injection on chains and homology. This
alone does not assert that the source has no additional classes. Automatic
subcomplex contraction establishes the stronger equivalence by eliminating
every complementary cell.

## Recovering a retraction through a reduction

For a reduced subcomplex, exact source identities first establish its inclusion
in the original selected model. The signed boundaries must commute with this
inclusion. Sampling and sparsification remain part of that model: restoring
cells removed by collapse does not restore discarded points or arbitrary
missing graph edges.

The source complement is contracted by algebraic cancellation at equal grade
sets. If the incidence from an extra cell $b$ to an extra face $a$ is the
invertible scalar $u$, eliminate the pair and update remaining boundaries by
the Schur complement

$$
D' = D-D[:,b]u^{-1}D[a,:].
$$

The removed rows and columns are then discarded. The accumulated projection
onto remaining chains is updated by the same elementary contraction. Because
the target is a subcomplex, the cancellation fixes it pointwise. Equal grade
sets make the contraction filtration preserving, including incomparable
multicritical appearances. If complementary cells remain and no admissible
pair exists, construction stops with an error requiring explicit maps; it
does not infer a lift from a barcode match.

For example, remove edge $[1,3]$ and face $[1,2,3]$ from a filled triangle.
The retained path has edges $[1,2]$ and $[2,3]$. Its retraction sends the removed
edge to their sum. This equation is signed: with
$\partial[1,2,3]=[2,3]-[1,3]+[1,2]$, all endpoints agree over any supported
prime field. A cochain taking values $(1,0)$ on the retained edges pulls back
to $(1,1,0)$ on the three original edges, annihilating the triangle boundary.

The ingestion collapse detector follows filtration-domination of flag graphs;
see [Alonso, Kerber and Pritam](https://arxiv.org/abs/2211.05574). The retained
matrix contraction here is an additional package computation on the explicit
cell complexes. It is not a claim that the graph detector returns lifting maps.

## Cycles, fillings and cocycles use different checks

An ordinary selected cycle is checked at its requested scale, defaulting to
birth. A finite-death filling is checked at death. The source equation
$\partial f=z$ joins these records; the filling need not be active before
death. The representative remains a noncanonical reduction choice.

Cochains travel in the opposite direction. A computed cocycle $\alpha$ lifts
to source by $p^*\alpha=p^\mathsf T\alpha$, restricted to active source cells.
Using $i$ in its place would have the wrong variance. The returned cochain is
checked against every active source coface. Equal interval/member indices in
independent homology and cohomology reductions do not assert dual bases.

The checks use original boundaries and grades before coordinates are converted
to drawing values. A periodic edge can have zero boundary while its unwrapped
drawing has distinct endpoints: those endpoints are identified. Displaying
that cell across the fundamental-domain seam is honest; replacing it by a long
interior chord is not. Higher chains retain their algebraic records, while
the source overlay renderer accepts degrees zero through two.

## Components retain a forest, not a list at every merge

For H₀, source membership comes directly from the checked vertex-edge
incidences. `component_history` orders the source edges by exact entry parameter
and joins their endpoint components with union-find. The older birth remains
the component label; equal births use source vertex order. A separate binary
forest retains both children, the actual joining edge and its exact parameter.
Cycle-closing edges and zero-boundary periodic loops do not create merges.
Equal-time nodes retain a deterministic binary order while still sharing one
simultaneous filtration event.

A query cuts this forest after all events at its parameter. Counting cut nodes
does not enumerate their members; a membership query descends to the original
vertex leaves. Thus the retained forest is linear in the number of cells,
rather than storing a potentially quadratic collection of member lists.
Following a birth vertex beyond its bar's death returns its current descendant
component, with that distinction explicit in the result.

For an oriented affine slice, a cell's entry parameter is the minimum over its
appearance grades of the maximum coordinate entry time. This uses the union
semantics of multicritical grades and requires strictly positive progress in
every oriented coordinate. The forest belongs to this specified slice. A
checked source correspondence additionally compares original and computed H₀
interval multisets before returning original-source memberships; it does not
assign an arbitrary reduction representative to one of several equal bars.

## A bounded exact improvement of a representative

At a fixed parameter, all cycles homologous to $z$ have the form
$z+\partial_{d+1}f$. Choose independent columns of the active boundary matrix,
forming $B$. If its rank is $r$ over $\mathbb F_p$, its image has exactly $p^r$
elements. The optimizer checks that $z$ is not itself a boundary, checks the
candidate budget, and enumerates this image.

For nonnegative cell weights $w_j$, it minimizes

$$
\sum_{j:x_j\ne0} w_j \quad\text{over }x\in z+\operatorname{im}B.
$$

Integer and rational weights remain exact; floating weights are compared as
their represented binary rational values. The answer retains the original
chain and an explicit $f$ with $x-z=\partial f$. The class is therefore both
unchanged and nonzero. A scale change changes the admissible boundary image;
the guarantee applies only to the declared parameter and finite source model.
This search has exponential cost and deliberately has a hard candidate budget.
No rendering path invokes it.

## Costs, reuse and future optimization

Source retention stores cells, geometry and sparse maps in addition to any
reduction columns. Constructing an uncollapsed source can be substantially
larger than constructing the reduced model. Algebraic cancellation can introduce
fill; it is an opt-in explanatory path rather than the default ingestion path.
Repeated source queries can reuse the verified correspondence, but arrays must
remain unchanged. Selection snapshots retain their chosen member and never
compute missing representatives during ordinary plotting.

The owned persistence format stores sparse map triplets, source boundaries,
grades, identities and geometry. Reloading validates map equations and
recomputes automatic contraction certificates. Retained cocycles are checked at
every source appearance parameter inside their interval, not only at birth. This is a heavier operation
than loading endpoint-only diagrams. Unsupported user callback metadata is
rejected by the safe data codec rather than serialized as executable code.

Future optimization can improve sparse cancellation, retain contractions while
ingesting, or replace the bounded search with a proved algorithm for a narrower
problem. Those changes must preserve the source boundary checks, grade
compatibility and declared optimization objective. General module-class
tracking and cross-panel connectors are separate consumers of
these records; a manually authored annotation does not acquire their guarantees.

See the [implementation bibliography](references.md) for related algorithms.
