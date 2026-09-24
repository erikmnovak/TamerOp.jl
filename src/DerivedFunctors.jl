module DerivedFunctors

"""
DerivedFunctors: owner module for Hom/Ext/Tor, resolutions, functoriality,
spectral sequences, and workflow wrappers built on finite-poset modules.

Hom, Ext, resolutions, and Yoneda products are computed in `Rep_k(P)` for
the actual finite input poset `P`. Tor pairs a module on `P^op` with one on
`P` over the finite incidence algebra. Independence of a projective/injective
resolution model holds within that category; it does not establish invariance
under a change of encoding or identification with ambient R^n/Z^n Ext/Tor.
Inspect `provenance(result)` or the `provenance` field of its summary for the
base poset, coefficient field, computed degrees and model. See
`docs/math_categories.md` for comparison hypotheses and counterexamples.

This file stays thin on purpose: it owns include order and the public surface,
while implementation lives in `src/derived_functors/`.
"""

include("derived_functors/shared.jl")
include("derived_functors/utils.jl")
include("derived_functors/graded_spaces.jl")
include("derived_functors/hom_ext_engine.jl")
include("derived_functors/resolutions.jl")
include("derived_functors/ext_tor_spaces.jl")
include("derived_functors/functoriality.jl")
include("derived_functors/algebras.jl")
include("derived_functors/spectral_sequences.jl")
include("derived_functors/backends.jl")
include("derived_functors/public_api.jl")

end
