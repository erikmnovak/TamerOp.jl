# Tameness and scope: when finite descriptions exist

The [chapter on indicator presentations](indicator_presentations.md) described
the square module using two regions and a coefficient matrix. The
[chapter on finite encodings](finite_encodings.md) recovered the same module
from a finite poset, its spaces and maps, and a classifier on the parameter
plane. When can a module have descriptions like these?

The square suggests that we should look for regions on which the same
spaces and maps can be used throughout. To understand what this requires,
we will describe its regions more carefully and then compare it with a
module whose spaces all have dimension one. That second module keeps
changing its maps as we move along the real line, and those changes will
prevent any finite encoding.

Our finite constructions already supply the condition we need. The next
chapter,
[why finite computations stay tame](practical_tameness.md), explains that
practical argument and the role of abelian categories. Here we establish
exactly what the condition means and which descriptions it guarantees.

You need the preceding chapters' definitions of modules, finite encodings,
and fringe presentations. We use a fixed coefficient field $\mathbb{k}$;
take $\mathbb{k}=\mathbb{Q}$ in the examples. The general statements apply
to an arbitrary parameter poset $Q$.

## What should it mean for a module to be constant on regions?

Recall the square $S=[0,2]^2$ in $\mathbb{R}^2$. Its module $M$ has
$M(q)=\mathbb{k}$ inside $S$ and $M(q)=0$ outside. A structure map between
comparable points inside is the identity; every other structure map is
the uniquely determined zero map of the appropriate shape.

There are two kinds of spaces here. Partition the plane into

$$
R_1=S,\qquad R_0=\mathbb{R}^2\setminus S.
$$

Use $\mathbb{k}$ as the common space for $R_1$ and $0$ for $R_0$.
Once these spaces are identified with the stalks, the maps depend only on
which two regions contain their endpoints:

| Source region | Target region | Map, whenever the parameters are comparable |
| --- | --- | --- |
| $R_1$ | $R_1$ | Identity on $\mathbb{k}$ |
| $R_1$ | $R_0$ | Zero map $\mathbb{k}\to0$ |
| $R_0$ | $R_1$ | Zero map $0\to\mathbb{k}$ |
| $R_0$ | $R_0$ | Unique map $0\to0$, also its identity |

The table tells us how to recover each map once we know the endpoint
regions. A partition by stalk dimensions would leave those maps unspecified.

For a general module, choose a partition $Q=\bigsqcup_i R_i$, a vector
space $V_i$ for each region, and isomorphisms
$\alpha_q:V_i\to M(q)$ for $q\in R_i$. It is a **constant subdivision**
if, for every ordered pair $(R_i,R_j)$, the map

$$
\alpha_r^{-1}\,M(q\leq r)\,\alpha_q:V_i\longrightarrow V_j
$$

is independent of the chosen comparable pair $q\in R_i$, $r\in R_j$.
We require this even when $i=j$, including the comparison of a point with
itself. This is the compatibility condition in [Ezra Miller's Definition 2.6](https://arxiv.org/html/2008.00063#S2.SS2).

To read the equation, start with a vector in $V_i$ and follow it through
$M(q)$ and $M(r)$. The last isomorphism expresses the answer in the common
space $V_j$. We use the same identifications for every comparison, so the
answer must agree even when we choose different endpoints in the two
regions. Choosing a fresh pair of bases for each arrow would evade that
requirement. The cited definition calls this compatibility **no monodromy**.

For points in the same region, comparison with a self-map forces the
displayed map to be the identity on $V_i$. Thus any structure map between
comparable points of that region must be an isomorphism. For incomparable
points there is no structure map to require. Regions need not be connected,
convex, or bounded; the square's entire exterior is already one region
in this subdivision.

## The notion of tameness used here

A module is **pointwise finite-dimensional** if each stalk $M(q)$ has
finite dimension. Ezra Miller also calls this $Q$-finite; it does not
mean that the parameter poset $Q$ is finite.

In this chapter, **tame** means pointwise finite-dimensional and admitting
a constant subdivision with finitely many regions. Both requirements
belong to [Definition 2.11](https://arxiv.org/html/2008.00063#S2.SS2).

The square is tame because the two regions above work and their common
spaces have dimensions one and zero. This describes the module on all of
$\mathbb{R}^2$, including its unbounded exterior. The word “finite” counts
the regions and the dimensions of their spaces; it places no bound on the
parameter domain.

The two requirements control different things. Finite-dimensional stalks
limit the size of each individual space. A finite constant subdivision
limits how many descriptions of spaces and maps are needed across the
domain. Together they give a uniform bound on stalk dimensions: take the
largest of the finitely many dimensions $\dim V_i$. The counterexample
below will show that even a uniform bound of one does not suffice by itself.

## A constant subdivision is not yet an encoding

The square's two regions describe its constant data, but we cannot use
those same two pieces as the fibers of a map that preserves order. To see the
obstruction, follow the three comparable points

$$
(-1,1)\leq(1,1)\leq(3,1).
$$

The first and last points are exterior, while the middle point is inside.
If $z$ labels the exterior and $a$ labels the square, monotonicity would
force $z\leq a\leq z$. Antisymmetry then gives $z=a$, incompatible with
the different stalk dimensions.

The constant subdivision allowed us to record the map for each pair of
region names without ordering the names themselves. An encoding requires
that order, because its labels must support a module whose structure maps
recover the original ones. We therefore need to separate some parts of the
exterior.

For this square we already have a suitable refinement. Record

$$
u(x,y)=\mathbf{1}_{\{x\geq0,\ y\geq0\}},\qquad
c(x,y)=\mathbf{1}_{\{x>2\ \text{or}\ y>2\}}.
$$

Both bits increase with the parameters, so $\pi(x,y)=(u(x,y),c(x,y))$
is order-preserving into $\{0<1\}^2$. Its four fibers split the exterior
into three pieces and retain the square as the fiber labeled $(1,0)$.
Give that label the space $\mathbb{k}$ and the other three labels zero.
This is the encoding on four labels developed earlier.

| Description of the same square module | What is finite? |
| --- | --- |
| Constant subdivision | Two regions with compatible identifications of spaces and maps |
| Signature encoding | Four labels, their ordered module, and the classifier $\pi$ |
| Encoding by nine regions | Nine labels, their ordered module, and the band classifier |
| Fringe presentation | One upset, one downset, and the coefficient matrix $[1]$ |

Each count refers to a different part of the description, so comparing
two regions with four labels tells us nothing about the number of persistent
classes. We have constructed useful descriptions without claiming that any
of them is the smallest possible one.

## Three equivalent ways to have finite data

For a fixed field and a **pointwise finite-dimensional module over any
poset**, the following are equivalent:

1. It admits a finite constant subdivision.
2. It admits a finite encoding: $M\cong\pi^*N$ with $\pi:Q\to P$
   order-preserving, $P$ finite, and every $N(p)$ finite-dimensional.
3. It admits a finite fringe presentation: an image of a map from a finite
   sum of upset indicators to a finite sum of downset indicators, with
   one scalar on each component's intersection.

The first two conditions are related by
[Theorem 4.22](https://arxiv.org/html/2008.00063#S4.SS4);
the third is included in
[Ezra Miller's syzygy theorem, Theorem 6.12](https://arxiv.org/html/2008.00063#S6.SS2).
The pointwise finite-dimensional hypothesis is explicit here: a finite
number of constant regions alone places no dimension bound on their common
spaces.

These are existence statements about the same module. They do not identify
the regions in one description with the labels or summands in another.
Here is how the constructions fit together.

### From an encoding to a constant subdivision

Take the nonempty fibers $\pi^{-1}(p)$. Identify each stalk in that fiber
with $N(p)$ using the given module isomorphism. A map between fibers is
then $N(p\leq p')$, independent of which comparable original parameters
were chosen. There are only finitely many fibers, so they give a finite
constant subdivision.

### From a fringe presentation to an encoding

For each upset, record membership; for each downset, record membership in
its complement. This produces finitely many increasing bits. Their
patterns form a finite ordered set. At each pattern, select the active
rows and columns of the coefficient matrix and take its image. Between
comparable patterns, restrict the appropriate projection onto the remaining target coordinates
to the image.

The [chapter on indicator presentations](indicator_presentations.md#how-the-presentation-leads-to-a-finite-encoding)
explained why these spaces and maps recover the presented module. The
construction uses the fixed scalar coefficients as well as the regions;
membership patterns alone would lose the linear algebra.

### From a constant subdivision to an encoding

Here we use the finite encoding theorem to organize the compatible
regional data into a finite ordered model. The construction forms upsets
from the constant regions and uses membership in those upsets to obtain
labels that respect the original order.

The conclusion concerns an encoding of the module. Its fibers may differ
from the given pieces, so we should not assume that an arbitrary chosen
subdivision will survive the construction as a refinement. For our square,
we can see directly that the four fibers above do refine the original
two regions.

### Why a finite encoding also yields a fringe presentation

There is a finite construction using linear algebra that explains this direction.
Start with the finite module $N$ on $P$. At a label $x$, form

$$
F(x)=\bigoplus_{p\leq x}N(p),\qquad
E(x)=\bigoplus_{r\geq x}N(r).
$$

After choosing a basis for every $N(p)$, $F$ is a finite sum of principal
upset indicators and $E$ a finite sum of principal downset indicators.
“Principal” means the upset $\{x:p\leq x\}$ or the downset
$\{x:x\leq r\}$ generated by one label.

Define maps

$$
F(x)\xrightarrow{a_x}N(x)\xrightarrow{b_x}E(x),
$$

where $a_x((v_p)_p)=\sum_{p\leq x}N(p\leq x)v_p$, and
$b_x(v)=(N(x\leq r)v)_{r\geq x}$.
The summand indexed by $x$ makes $a_x$ surjective and $b_x$ injective:
in each case that coordinate uses the identity of $N(x)$. The maps commute
with structure maps, so the image of $b\,a$ is isomorphic to $N$ as a module.

For a source summand indexed by $p$ and a target summand indexed by $r$,
the composite at any $p\leq x\leq r$ is

$$
N(x\leq r)\,N(p\leq x)=N(p\leq r).
$$

Its matrix is independent of $x$. Thus every scalar component is constant
on its upset–downset intersection, as a fringe presentation requires.
If $p\not\leq r$, that intersection is empty and the component is zero.
Pulling the construction back along $\pi$ replaces these regions by their
preimages and gives a finite fringe presentation of $M$. This establishes
existence; it need not give the smallest presentation. If a pulled-back
intersection is empty, set its unused coefficient entries to zero.

## A counterexample with dimension one everywhere

For this example only, use the one-parameter domain $\mathbb{R}$. Keep
$\mathbb{k}=\mathbb{Q}$. Put

$$
V(t)=\mathbb{k}\quad\text{for every }t\in\mathbb{R},
$$

and, for $s\leq t$, define

$$
V(s\leq t)=
\begin{cases}
\operatorname{id}_{\mathbb{k}},&\lfloor s\rfloor=\lfloor t\rfloor,\\
0,&\lfloor s\rfloor\ne\lfloor t\rfloor.
\end{cases}
$$

Here $\lfloor t\rfloor$ is the integer $n$ satisfying $n\leq t<n+1$.
The map is the identity within one unit interval and zero whenever it
crosses an integer boundary. For example,

$$
V(\tfrac14\leq\tfrac34)=[1],\qquad
V(\tfrac34\leq1)=[0],\qquad
V(1\leq\tfrac32)=[1].
$$

We can check the module axioms directly. If $s\leq t\leq u$, their
floors are nondecreasing. When the endpoint floors agree, the middle one
must agree too, so all the maps are identities. When the endpoint floors
differ, at least one of the two successive maps is zero, making their
composite zero just like the direct map. A point always has the same floor
as itself, which gives the identity self-map.

Equivalently, writing $I_n=[n,n+1)$,

$$
V\cong\bigoplus_{n\in\mathbb{Z}}\mathbb{k}[I_n].
$$

Here $\mathbb{k}[I_n]$ is the indicator module of the half-open interval
$[n,n+1)$. Exactly one summand is present at each
parameter. At an integer, one summand ends and the next begins, so the
stalk is still $\mathbb{k}$.

![The dimension is one throughout the displayed interval. Half-open bars show successive unit intervals. The selected parameters one quarter and three quarters both have label zero and structure matrix [1]; three quarters and one have labels zero and one and structure matrix [0].](assets/tameness/constant_dimension_changing_maps.svg)

*The upper line records dimensions, while the bars below show which
summand supplies the space. At $t=1$ the first bar ends and the next begins,
so the stalk remains $\mathbb{k}$ even though the map from $3/4$ is zero.
Filled and open endpoints mean included and excluded, respectively. This
is a schematic of a finite window; the unit intervals continue in both
directions.*

### Why no finite constant subdivision can work

Consider the infinitely many parameters $t_n=n+\tfrac12$ for integers
$n\geq0$. In any partition into finitely many regions, two of them,
say $t_n<t_m$, must belong to the same region.

Their structure map is zero, because they lie in different unit intervals.
Changing bases cannot turn a zero map into an isomorphism. But two comparable
points in one constant region must have an isomorphism between their
stalks: their transported map must agree with the identity self-map.
This is a contradiction. Hence $V$ is not tame.

There is also a direct obstruction to a finite encoding. Two of the
$t_n$ would have to receive the same finite label $p$. Their map would
then be recovered from $N(p\leq p)=\operatorname{id}_{N(p)}$ and would
be an isomorphism, again contradicting the zero map. The equivalence
theorem also rules out a finite fringe presentation.

The dimension remains one everywhere, but a finite description would
have to identify two of these intervals in a way their zero map forbids.
The dimension plot hides exactly the distinction used in the proof.

### A short rule and a finite window are different claims

The floor formula describes $V$ in a few symbols. It even gives an
encoding through the poset $\mathbb{Z}$, with $\mathbb{k}$ at every integer
and zero maps between distinct integers. That encoding is infinite.
A short program or formula is a different notion of finite description
from the finite encodings considered here.

If we restrict the domain to a fixed bounded interval, only finitely many
unit intervals meet it. Restricting the floor classifier to that finite
set of integer labels gives a finite encoding of the restricted module.
The restriction is tame, while the module on all of $\mathbb{R}$ is not.
If we widen the window, more integer labels appear while the dimension
profile stays flat. Every bounded restriction still uses only finitely many
labels. The obstruction on the whole line comes from needing infinitely
many of them at once, as the preceding proof shows.

### Distinguish this from q-tameness

In one-parameter persistence, **q-tame** usually means that every map
$V(s\leq t)$ with $s<t$ has finite rank; see
[Frédéric Chazal, William Crawley-Boevey, and Vin de Silva, Section 1.4](https://arxiv.org/html/1405.5644#S1.SS4).
Our counterexample has ranks zero or one, so it is q-tame. It is not tame
in the sense of finite constant subdivisions used in this chapter. The two
terms impose different requirements.

## From an existence theorem to an implemented construction

The theorem concerns modules over arbitrary posets and allows abstract
upsets, downsets, and classifiers. To compute, we must also provide a
representation of those ingredients and methods for evaluating membership
and order. Finitely many sets need not come with finite geometric
descriptions or an algorithm for testing membership.

The square supplies both kinds of information: its finite presentation
has the coefficient $[1]$, and four explicit inequalities describe its
regions. The selected encoder can construct the finite model and classify
queries. To use the theorem computationally, we need this additional representation
of the regions. It does not supply a general procedure that accepts an
arbitrary module and decides whether it is tame.

Some concrete TamerOp entry routes are:

| Starting information | What the package works with |
| --- | --- |
| A finite poset, finite-dimensional spaces, and compatible maps | A finite module such as `TamerOp.Advanced.PModule`, inspected and validated directly. It is already a finite representation. |
| The square's box regions and coefficient matrix | `encode(upsets, downsets, coefficients, options)` produces an `EncodingResult` for the selected supported construction. |
| A supported polyhedral presentation | Presentation encoding using the applicable geometric backend, with that backend's generator and arithmetic requirements. |
| Typed data together with a supported filtration | Data ingestion through `encode`, requesting the completed encoding stage when an `EncodingResult` is wanted. |

In the first row, the identity map of the finite poset already gives an
encoding in the mathematical sense, and the software can operate on the
finite module directly. There is no need to turn that observation into an
`encode(module)` call.

For geometric presentations, consult the
[geometry contracts](math_categories.md#geometric-encoding-and-boundary-classes).
For data, the [ingestion guide](ingestion_options.md) explains filtration,
degree, grid, truncation, and choices of intermediate stages. These determine
which module is represented. A finite encoding recovers that chosen module;
it does not undo a change made when constructing the input.

The coefficient field and geometric arithmetic are separate choices.
Rational coefficients make the stated linear algebra exact without
automatically making every geometric input exact. The
[exact-grade guide](exact_grades.md) describes guarantees for its specific
construction. Likewise, neither the existence theorem nor a successful
example asserts that the returned encoding is minimal or inexpensive for
every supported input.

The full syzygy theorem also gives finite indicator resolutions, which
organize successive relations among simpler modules. To use a resolution
for a derived computation, we still need to know which category contains
its terms and maps. In particular, Ext and Tor over the finite base require
the [category and comparison hypotheses](math_categories.md) described
in their guide.

## Why do finite constructions meet this condition?

The square has now given us a concrete version of each description in
the theorem, and the module on the real line has shown why the maps have
to be part of the finiteness condition. We can next ask how much of this
reasoning needs to be repeated when we work with a new finite input or
perform another calculation on an encoded module. Continue with [why finite computations stay tame](practical_tameness.md)
for the argument using a finite complex, a calculation on the square, and the
closure theorems of Ezra Miller and Lukas Waas.
