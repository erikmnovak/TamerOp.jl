"""Bounded planar candidate interval supports from matched diagonal slices."""
module IntervalApproximations

using ..CoreModules: QQ, QQField, PrimeField, coeff_type
using ..FiniteFringe: GridPoset, cover_edges, nvertices
using ..Modules: PModule, map_leq
using ..Interleavings: RectilinearModule, InterleavingBudget, ambient_module
using ..Results: EncodingResult, encoding_module, encoding_map
import ..EncodingCore: locate
import ..SliceInvariants
import ..FieldLinAlg
import ..ChainComplexes: describe
import ..Results: provenance

include("interval_approximations/construction.jl")
include("interval_approximations/queries.jl")

end
