# Implementation reference

These accounts explain how TamerOp turns a mathematical operation into a
concrete computation: the retained objects, algorithms, correctness conditions,
and reasons for its implementation choices. They are for readers
asking **“how did they achieve this?”** Familiarity with the mathematical
operation is useful; each account develops the additional notation it needs.

<!-- COLLECTION: implementation -->

[Axis-aligned box encoding](box_encoding.md) explains how the grid retains
closed points, faces and corners, carries their maps, and certifies exact
representatives without invoking general polyhedral feasibility.

This category is separate from the introductory reading routes and the
supporting usage guides. It can be consulted by topic without following a
prescribed sequence.

[Linked inspection sessions](inspection_sessions.md) explains selection
transactions, bounded reuse and the difference between live Julia queries and
captured browser states. [Serialization and mathematical provenance](serialization.md)
follows those states into portable files while keeping the mathematical source,
its coordinate bases and the selected drawing separate.

[Computing generalized rank and GRIL](generalized_rank.md) follows compatible
sections through a common quotient, then explains finite-family reconstruction,
boundary zigzag witnesses and exact continuous-worm widths.

[Exact rhomboid multicover construction](rhomboid_construction.md) follows
site labels through constrained radii and signed boundaries into finite
encodings. It explains lifted cell geometry, closed depth caps and the
separation between source sampling and exact selected-parameter algebra.

[Retaining representatives and tracing them to source cells](source_representatives.md)
explains filtered chain maps, equal-grade contractions, contravariant cocycle
lifting and the bounded minimum-support search. Read it when a source overlay
needs a mathematical correspondence between computed and original cell bases.

[Change of poset, refinement and Kan extensions](change_of_posets.md) explains
how order maps, actual linear maps and finite-source realization certificates
survive re-encoding and transport, including their optimization boundaries.

[From selected images to submodules and quotients](submodules_and_quotients.md)
follows selected coordinates through the image engine and induced maps, then
through the quotient projection. It explains why a generated submodule need
not be a direct summand.

[Yoneda products and validated Tor algebra](yoneda_products.md) follows
comparison lifts and quotient coordinates, then explains what supplied chain
products must satisfy before they define multiplication on Tor homology.

The [landscape sampling account](feature_transforms.md) explains stored line parameters,
ambient grid queries, missing samples and conditional interpolation bounds.

[Aligned ensembles and uncertainty computations](ensembles.md) follows repeated
observables through probe alignment, empirical reductions, bounded-mean and
whole-row bootstrap bands, then prepared visual selection.

[Relative persistence of filtered pairs](relative_persistence.md) explains when
one quotient complex suffices, when delayed subcomplex entry needs a filtered
cone, and how the actual quotient bases and connecting maps are recovered.

[Interval multiplicities and certified decomposition](interval_decomposition.md)
follows ambient Hom maps into a scalar pairing, then endpoint pruning and
simultaneous splitting maps. It explains when a residual certifies failure of
interval-decomposability and when a selected family leaves that question open.
