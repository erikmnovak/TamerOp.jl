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
