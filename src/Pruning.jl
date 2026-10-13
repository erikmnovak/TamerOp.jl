"""Exact epsilon-pruning of globally encoded rectilinear persistence modules."""
module Pruning

using ..CoreModules: QQ, coeff_type, eye
using ..FiniteFringe: cover_edges, nvertices
using ..Modules: PModule, PMorphism, structure_map, check_morphism
using ..Interleavings: RectilinearModule, InterleavingBudget, ambient_module,
    parameter_shift, refine_parameters, interleaving_problem,
    interleaving_certificate, check_interleaving
using ..EncodingCore: encoding_axes, locate
using ..DerivedFunctors: Hom, basis
using ..AbelianCategories: submodule, submodule_object, inclusion_map,
    quotient_with_projection, check_submodule
using ..Results: EncodingResult
import ..Results: encoding_module, provenance
import ..ChainComplexes: describe
import ..FieldLinAlg

include("pruning/construction.jl")
include("pruning/certificates.jl")

end
