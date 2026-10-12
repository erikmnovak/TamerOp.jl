# From selected images to submodules and quotients

Once a finite encoding has supplied spaces and maps, we can ask what a chosen
vector generates. Looking at the nonzero stalks of the whole module cannot
answer this: a map can kill one vector while preserving another. The computation
must carry coordinates through the maps and then retain the resulting subspaces
with their induced maps.

This account follows that operation through the general image and quotient
engines. The [usage guide](../tutorials/generated_submodules.ipynb) explores the
answer on overlapping squares; [API contracts](../src/reference/modules.md)
specify shapes, fields and costs at the call boundary.

## Propagate columns before constructing a new module

Let $M$ be a module on a finite poset $P$, let $p\in P$, and let $B$ be a matrix
whose columns lie in $M(p)$. The requested data are

$$Y_q=M(p\leq q)B\quad(p\leq q).$$

There is no such map outside the upper ideal of $p$. The stored image is
`nothing` there. At an applicable vertex a zero matrix instead means that
all chosen vectors were annihilated. Its row count still records the target
space and its column count still records the supplied vectors.

`propagate` first validates the module's storage. It then traverses cover
successors starting at $p$. A vertex is visited once; the first discovered
path supplies its image. Every subsequent incoming route is compared with
that image on the selected columns, and disagreement raises an error. These
cover equations establish path independence for the selection over exact
fields. As in other categorical operations, the input must be a valid module;
this check does not certify unselected vectors throughout the ambient diagram. Multiplying a cover matrix by the
selected columns avoids forming an entire composite map when only a few
vectors are needed. The routine computes a rank at each visited vertex and
retains the columns, including dependent columns, for inspection.

For example, the map $[0\;1]$ kills $e_1$ but preserves $e_2$ and $e_1+e_2$.
All three questions use the same ambient module. Following the full map's
rank would obscure their different answers.

The generated spaces are

$$N(q)=\begin{cases}\operatorname{im}Y_q,&p\leq q,\\0,&\text{otherwise}.\end{cases}$$

They are closed under structure maps because
$M(q\leq r)Y_q=Y_r$. Any submodule containing the columns of $B$ must contain
each $Y_q$, which proves minimality. It does not prove that $N$ has a complement.
For the chain $k\xrightarrow{1}k$, selecting the second stalk gives
$0\to k$. A proposed complement $k\to0$ fails closure under the identity map.

## Use the existing image engine

The default propagation result stops at the images and ranks. On the explicit
`generated_submodule` call, TamerOp constructs a natural map from $s$ copies
of the principal projective at $p$, where $s$ is the number of supplied columns. This free module has that many coordinates
at every vertex above $p$, identity maps there, and zero stalks elsewhere.
The morphism into $M$ has components $Y_q$.

The image engine chooses a column-space basis $I_q$ of each component. It then
finds the induced cover matrix $N_{uv}$ by solving

$$I_vN_{uv}=M_{uv}I_u.$$

Because the columns of $I_v$ are independent, the coordinate matrix is unique.
The resulting component matrices $I_q$ form the inclusion $N\hookrightarrow M$.
This construction reuses `image_submodule`; it does not duplicate categorical
coordinate logic in the viewer or attach arbitrary identities between spaces
with equal dimensions. The same image engine serves images of general module
morphisms, while its kernel counterpart selects nullspace bases and recovers
induced maps by the analogous equation.

The rational coordinate mechanisms are explained in
[Exact rational coordinates](qq_coordinates.md). Shared field-aware rank and
coordinate routines also serve prime and numerical fields. `RealField` gives
tolerance-dependent ranks and equations; traversing a numerically commutative
diagram is not a proof of exact path independence. `check_propagation` checks
the source columns and every applicable cover equation, using the configured
absolute and relative tolerances.

## Remove the generated submodule

`quotient_with_projection` delegates to the general cokernel construction for
the inclusion. At each stalk it chooses a quotient projection $Q_q$ with
kernel $\operatorname{im}I_q$. The induced cover map is determined by

$$\overline M_{uv}Q_u=Q_vM_{uv}.$$

This gives a module and a natural projection, not merely a list of quotient
dimensions. The two useful certificates are $Q_qI_q=0$ and the displayed
naturality equation. Zero stalks and matrices with zero rows or columns are
part of these equations, not exceptional mathematical cases.

These local image, kernel and quotient mechanisms also underlie universal
constructions. For example, a pullback of maps to a common target is the
kernel of their difference on a direct sum; a pushout is the cokernel of the
corresponding signed map into a direct sum. The construction maps carry the
universal-property information that a dimension table would discard.

## Coordinates, source cycles and reuse

Under invertible coordinate changes $C_q$, the stored maps become
$C_qM_{pq}C_p^{-1}$ and the selected columns become $C_pB$. The new propagated
columns are $C_qY_q$. Ranks and nonzero support therefore agree, and the new
inclusion describes the same subobject under the given module isomorphism.
Changing the basis of the selected subspace by right multiplication also
preserves its generated subobject. Individual numeric matrices need not agree.

For a homology module, an optional source-cell query combines a propagated
column with a supplied target homology basis of retained cycles. It verifies
cycle equations and independence and completeness modulo active boundaries.
The identification of that ordered basis with the module's coordinates is
supplied explicitly, and recorded as such. See
[Source representatives](source_representatives.md) for the underlying chain
correspondences; coordinate agreement is never inferred from geometric proximity.

The propagation result retains the ambient module, selected columns and their
images; an encoding workflow also retains the classifier. With $s$ columns,
the image payload stores $s\sum_{q\geq p}\dim M(q)$ coefficients, in addition
to retained input objects. Cover multiplication costs depend on the ambient
matrix storage and dimensions. Validation examines storage and the selected cover equations,
and rank calculations can dominate for many columns. Rational coefficient
sizes also matter, beyond these entry counts.

Reading an image or summary makes copies without new algebra. Constructing a
submodule additionally allocates its image bases, free-source maps and induced
cover maps. Quotients allocate their own coordinates and maps. None is an
implicit consequence of rendering. The finite graph has an explicit vertex
budget, and matrix crops affect drawing only. Input objects are read-only while
a result is reused; mutation requires a fresh propagation. This keeps future
optimization accountable to the same columns, subspaces and commuting equations.

[Implementation bibliography](references.md)
