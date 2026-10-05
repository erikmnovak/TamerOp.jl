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
