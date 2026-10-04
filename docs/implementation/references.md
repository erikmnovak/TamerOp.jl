# Implementation bibliography

This is the shared citation list for the implementation accounts. Each entry
states its connection to the code or argument and links to its use. It begins
with [exact rational coordinates](qq_coordinates.md); it is not yet a complete
bibliography of TamerOp. Related methods are distinguished from algorithms
implemented locally and software used directly.

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
a particular TamerOp request invokes. The web manual is a moving reference,
not the dependency version of a benchmark.
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
[Use in the coordinate account](qq_coordinates.md#a-separate-route-through-modular-arithmetic).

## Dixon lifting

John D. Dixon. **Exact solution of linear equations using p-adic expansions.**
*Numerische Mathematik* **40**, 137–141, 1982.
[DOI: 10.1007/BF01459082](https://doi.org/10.1007/BF01459082).

**Role:** a related exact-solving method, and context for the backend's
available algorithms. Lifting through powers of one prime differs from
TamerOp's local independent-prime CRT route. This entry does not claim a
local Dixon implementation.
[Use in the coordinate account](qq_coordinates.md#a-separate-route-through-modular-arithmetic).

## Rank-one inverse identities

William W. Hager. **Updating the Inverse of a Matrix.** *SIAM Review*
**31**(2), 221–239, 1989.
[DOI: 10.1137/1031049](https://doi.org/10.1137/1031049) ·
[Author's copy](https://people.clas.ufl.edu/hager/files/update-1.pdf).

**Role:** mathematical reference for the Sherman–Morrison rank-one identity
used to construct independent exact test answers. This is not a claim that
TamerOp's production factorization updates inverses with that formula.
[Use in the coordinate account](qq_coordinates.md#how-the-contract-is-tested).

## QPA

QPA developers. **QPA — Quivers and Path Algebras**, manual,
Chapter 7: *Homomorphisms of Right Modules over Path Algebras*.
[Official manual](https://gap-packages.github.io/qpa/doc/chap7.html) ·
[Project](https://gap-packages.github.io/qpa/).

**Role:** independent comparison software and a source for its matrix
conventions. QPA uses row-vector module conventions. It is neither a TamerOp
dependency nor an asserted source of the coordinate solver.
[Use in the coordinate account](qq_coordinates.md#what-the-benchmark-iterations-changed) ·
[TamerOp comparison](../benchmarks/qpa.md).
