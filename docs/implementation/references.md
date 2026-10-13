# Implementation bibliography

Academic and software references cited in the [implementation accounts](index.md).

## Nemo and Hecke

Claus Fieker, William Hart, Tommy Hofmann, and Fredrik Johansson.
**Nemo/Hecke: Computer Algebra and Number Theory Packages for the Julia
Programming Language.** *ISSAC 2017*, 157–164.
[DOI: 10.1145/3087604.3087611](https://doi.org/10.1145/3087604.3087611) ·
[Open manuscript](https://arxiv.org/abs/1705.06134) ·
[Matrix API](https://nemocas.github.io/Nemo.jl/stable/matrix/).

**Role:** directly used software and its architecture. TamerOp delegates
selected exact matrix operations to Nemo; this paper explains its combination
of Julia algorithms and specialized native libraries.
[Use in the coordinate account](qq_coordinates.md#where-native-exact-arithmetic-earns-its-conversion-cost).

## FLINT

The FLINT team. **FLINT: Fast Library for Number Theory.**
[Project and citation guidance](https://flintlib.org/citation.html) ·
[Rational matrices](https://flintlib.org/doc/fmpq_mat.html) ·
[Rational reconstruction](https://flintlib.org/doc/fmpq.html#modular-reduction-and-rational-reconstruction).

**Role:** exact-arithmetic implementation used through Nemo, and a precise
reference for reconstruction bounds. The online matrix manual describes
multiple solving algorithms; listing them is not a claim about which one
a particular TamerOp request invokes.
[Use in the coordinate account](qq_coordinates.md#where-native-exact-arithmetic-earns-its-conversion-cost).

## Rational reconstruction

Paul S. Wang, M. J. T. Guy, and J. H. Davenport.
**P-adic reconstruction of rational numbers.** *ACM SIGSAM Bulletin*
**16**(2), 2–3, 1982.
[DOI: 10.1145/1089292.1089293](https://doi.org/10.1145/1089292.1089293) ·
[Paper scan](https://www.cs.drexel.edu/~jjohnson/2012-13/fall/cs300/resources/p2-wang.pdf).

**Role:** mathematical background for a technique implemented locally:
reconstructing small rational numbers from modular residues. It supports the
uniqueness argument; the implementation additionally checks the original
matrix equation. No claim of direct code derivation is made.
[Use in the coordinate account](qq_coordinates.md#when-the-modular-route-is-selected).

## Dixon lifting

John D. Dixon. **Exact solution of linear equations using p-adic expansions.**
*Numerische Mathematik* **40**, 137–141, 1982.
[DOI: 10.1007/BF01459082](https://doi.org/10.1007/BF01459082).

**Role:** a related exact-solving method, and context for the backend's
available algorithms. Lifting through powers of one prime differs from
TamerOp's local independent-prime CRT route. This entry does not claim a
local Dixon implementation.
[Use in the coordinate account](qq_coordinates.md#when-the-modular-route-is-selected).

## Generalized rank

Woojin Kim and Facundo Mémoli. **Generalized Persistence Diagrams for
Persistence Modules over Posets.** *Journal of Applied and Computational
Topology* **5**, 533–581, 2021.
[DOI: 10.1007/s41468-021-00075-1](https://doi.org/10.1007/s41468-021-00075-1).

**Role:** the mathematical limit-to-colimit rank and generalized persistence
diagram framework. TamerOp constructs finite constraint and relation matrices
locally, and limits signed reconstruction to the caller's declared family.

## Boundary zigzags

Tamal K. Dey, Woojin Kim and Facundo Mémoli. **Computing Generalized Rank
Invariant for 2-Parameter Persistence Modules via Zigzag Persistence and Its
Applications.** *Discrete & Computational Geometry* **71**, 67–94, 2024.
[DOI: 10.1007/s00454-023-00584-z](https://doi.org/10.1007/s00454-023-00584-z).
[Accessible construction and theorem](https://arxiv.org/html/2111.15058v3#S3).

**Role:** the boundary-cap path and full-bar characterization for finite
connected convex grid intervals. TamerOp retains repeated path occurrences
and computes its barcode by bounded exact rank inversion; it does not reuse
the paper's specialized filtration-level zigzag reduction.

## Generalized-rank invariant landscapes

Cheng Xin, Soham Mukherjee, Shreyas N. Samaga, and Tamal K. Dey.
**GRIL: A 2-parameter Persistence Based Vectorization for Machine Learning.**
*Proceedings of Machine Learning Research* **221**, 2023.
[Paper and proceedings record](https://proceedings.mlr.press/v221/xin23a.html).

**Role:** the continuous worm and landscape definition. TamerOp's supported-grid
implementation contracts constant fibers and searches exact critical widths;
it does not reuse the paper's filtration-level zigzag implementation.

## Bigraded Betti numbers and presentation diagrams

The RIVET developers. **Mathematical preliminaries**, RIVET documentation.
[Invariant definitions](https://rivet.readthedocs.io/en/latest/preliminaries.html#invariants-of-a-bipersistence-module).

**Role:** mathematical and visualization context for minimal multigraded free
resolutions and their Betti numbers. The finite-poset resolution views use
TamerOp's own stored projective/injective terms and verification algorithms;
supplied grade coordinates alone do not identify those terms with an ambient
free resolution. No RIVET algorithm or code is used by the renderer.
[Use in the visual-specification account](visual_specs.md#resolve-storage-conventions-before-drawing-an-arrow).

## Signed barcodes and prominence

Magnus Bakke Botnan, Steffen Oppermann and Steve Oudot.
**Signed Barcodes for Multi-parameter Persistence via Rank Decompositions and
Rank-Exact Resolutions.** *Foundations of Computational Mathematics* **25**,
1815–1874, 2025.
[DOI: 10.1007/s10208-024-09672-9](https://doi.org/10.1007/s10208-024-09672-9).

**Role:** mathematical definitions and visualization conventions. Sections 6.1–6.2
explain signed rectangle diagonals, signed side-length prominence and smoothing
through images of shift maps. The [visual specification account](visual_specs.md#prepare-signed-contributions-before-drawing)
applies these to supplied signed rank data with explicit endpoint conventions.
This is not a dependency on external plotting software.

## Kan extensions and adjunctions

Emily Riehl. **Category Theory in Context.** Dover, 2016.
[Author's book page](https://math.jhu.edu/~eriehl/context/).

**Role:** mathematical background for limits, colimits, natural transformations,
adjunctions and Kan extensions. Chapter 6, including the pointwise colimit
formula in Theorem 6.2.1, supplies the categorical setting. TamerOp's finite
relation matrices, coordinate choices and verification are local implementations;
this is not a claim of borrowed software or an encoding-independence theorem.
[Use in the transport account](change_of_posets.md#kan-extensions-retain-chosen-limit-and-colimit-coordinates).

## Yoneda composition and differential graded products

The Stacks Project Authors. **Ext groups**, Section 13.27,
[Tag 06XP](https://stacks.math.columbia.edu/tag/06XP), and **Differential
graded algebras**, Section 22.3,
[Tag 061U](https://stacks.math.columbia.edu/tag/061U).

**Role:** mathematical context for Ext composition, its extension interpretation,
and differential/product compatibility. TamerOp's generator-block lifts,
coordinate tables, Tor descent checks and visual preparation are local
implementations, not a dependency on Stacks software. The
[product account](yoneda_products.md) states the concrete grading conventions
and the difference between chain compatibility and associativity on homology.

- [Makie coordinate spaces](https://docs.makie.org/stable/reference/generic/space.html)
  and [layout observables](https://docs.makie.org/stable/explanations/layouting.html):
  backend mechanisms used to attach composition annotations to data viewports
  and measured coefficient labels.
- Makie's [segment](https://docs.makie.org/stable/reference/plots/linesegments.html),
  [polygon](https://docs.makie.org/stable/reference/plots/poly.html), and
  [text](https://docs.makie.org/stable/reference/plots/text.html) recipes provide
  the batched drawing primitives used by the shared renderer. TamerOp owns the
  semantic layer preparation, painting-order safeguards and exact table layout.

## Rhomboid and multicover constructions

René Corbet, Michael Kerber, Michael Lesnick and Georg Osang.
**Computing the multicover bifiltration.**
[Open manuscript, version 3](https://arxiv.org/html/2103.07823v3).
The rhomboid model, integer-depth slicing and comparison with multicover
supply the mathematical model of the
[exact construction account](rhomboid_construction.md). The positive-depth
coordinate in its figures explicitly reflects the usual negative-depth axis.

Herbert Edelsbrunner and Georg Osang.
**A Simple Algorithm for Higher-order Delaunay Mosaics and Alpha Shapes.**
[Open manuscript](https://arxiv.org/abs/2011.03617).
Algorithm 1 supplies the subset-label recovery behind the incremental
weighted-Delaunay discovery route; TamerOp's rational hull rebuilding and
backend selection are separate implementation choices.

Andrew J. Blumberg and Michael Lesnick.
**Stability of 2-parameter persistent homology.**
[Open manuscript, version 3](https://arxiv.org/html/2010.09628v3).
Section 4 supplies the subdivision–Čech/multicover comparison. It supports the
labelled flag model for degenerate inputs, independently of the inspector's
noninjective schematic layout.

Ulrich Bauer, Michael Kerber, Fabian Roll and Alexander Rolle.
**A Unified View on the Functorial Nerve Theorem and its Variations.**
[Open manuscript, version 6](https://arxiv.org/pdf/2203.03571v6).
Theorem 5.9(1b), Corollary 5.16 and Proposition 5.19 justify the closed-cover
hypotheses used for exact birth contacts in the implementation account.

- Oliver Vipond. [Multiparameter Persistence Landscapes](https://www.jmlr.org/papers/v21/19-054.html). *JMLR* 21(61), 2020. Diagonal-landscape interpretation used in the ambient query adapter.

## Ensemble mean uncertainty

- Wassily Hoeffding, *Probability inequalities for sums of bounded random variables*,
  Journal of the American Statistical Association 58 (1963), 13–30.
  [Paper](https://www.cs.rpi.edu/academics/courses/spring06/random/hoefding.pdf).
  Supplies the bounded-mean inequality used with a finite union bound in
  [ensemble computations](ensembles.md).
- Inés García-Redondo, Anthea Monod and Qiquan Wang,
  *Confidence Bands for Multiparameter Persistence Landscapes*.
  [arXiv:2504.01113](https://arxiv.org/abs/2504.01113).
  Develops functional mean-landscape inference; the ensemble implementation's
  finite-probe bootstrap does not automatically inherit its continuous-domain theorem.

## Relative homology and filtered pairs

Allen Hatcher. **Algebraic Topology.** Cambridge University Press, 2002,
Section 2.1. [Author's chapter](https://pi.math.cornell.edu/~hatcher/AT/ATch2.pdf).

**Role:** quotient chain complexes and the natural long exact homology
sequence; the [relative-persistence account](relative_persistence.md) uses
the boundary-of-a-lift convention for its connecting map.

Nello Blaser and Morten Brun. **Relative Persistent Homology.**
*Discrete & Computational Geometry* **68**, 949–963, 2022.
[DOI: 10.1007/s00454-022-00421-9](https://doi.org/10.1007/s00454-022-00421-9).

**Role:** the persistent quotient viewpoint and context for relative geometric
constructions. The general filtered-pair implementation is local; it does not
implement the paper's specialized relative Delaunay–Čech construction.

## Interval summand multiplicities

Hideto Asashiba and Enhao Liu. **Interval Multiplicities of Persistence
Modules.** [arXiv:2411.11594v6](https://arxiv.org/abs/2411.11594v6), 2026.

**Role:** the finite-poset multiplicity problem and the degree-zero top/socle
critical-set criterion (Lemma 3.39). TamerOp implements that pruning criterion
with endpoint antichains. Its local multiplicity engine uses the Hom
composition pairing proved in the [algorithm account](interval_decomposition.md);
it does not implement this paper's specialized presentation formulas,
essential covers or filtration-level reductions.

## Parameter atlas figures

David Loiseaux and Hannah Schreiber. **multipers: Multiparameter Persistence
for Machine Learning.** *Journal of Open Source Software* **9**(103), 6773,
2024. [DOI: 10.21105/joss.06773](https://doi.org/10.21105/joss.06773).

**Role:** Figure 1 provides a visual precedent for keeping a grid of source
filtration states visible together in the [atlas inspector](inspection_sessions.md#reuse-a-declared-atlas).

Marc Fersztand and Jan Jendrysiak. **Computing the Skyscraper Invariant.**
*SoCG 2026*, LIPIcs **367**, 47:1–47:23.
[DOI: 10.4230/LIPIcs.SoCG.2026.47](https://doi.org/10.4230/LIPIcs.SoCG.2026.47).

**Role:** Figure 3 relates a sampled bifiltration to its homology diagram.
This motivates the atlas's linked source states and maps; the atlas does not
compute the Skyscraper invariant.
