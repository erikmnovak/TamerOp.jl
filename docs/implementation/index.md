# Implementation reference

These accounts explain how TamerOp turns a mathematical operation into a
concrete computation: the retained objects, algorithms, correctness checks,
and measurements behind its implementation choices. They are for readers
asking **“how did they achieve this?”** Familiarity with the mathematical
operation is useful; each account develops the additional notation it needs.

The [exact rational coordinates](qq_coordinates.md) account follows one
repeated operation from a small matrix example through row selection,
factor reuse, native exact arithmetic, and homology coordinates. Its benchmark
discussion includes the cases that became slower and the limits of the evidence.

The [implementation bibliography](references.md) collects the academic and
software references cited here, with links back to the decisions they illuminate.
It currently covers this first account, rather than claiming to inventory all
the library's intellectual dependencies.

This category is separate from the introductory reading routes and the
supporting usage guides. It can be consulted by topic without following a
prescribed sequence.
