"""Exact minimal presentations of graph H0 on a specified finite poset."""
module H0Presentations
using SparseArrays, LinearAlgebra
using ..CoreModules: AbstractCoeffField, QQField, PrimeField, RealField, coeff_type
using ..FiniteFringe: AbstractPoset, ProductOfChainsPoset, GridPoset, nvertices, leq, cover_edges
using ..Modules: PModule, PMorphism
import ..AbelianCategories
import ..DerivedFunctors
import ..FieldLinAlg
using ..Options: ResolutionOptions
import ..ChainComplexes: describe
import ..IndicatorTypes: generator_labels, relation_labels, presentation_matrix
import ..IndicatorResolutions: presentation_module, presentation_map, presentation_summary
import ..Results: provenance

include("h0_presentations/forest.jl")
include("h0_presentations/graphs.jl")
include("h0_presentations/results.jl")
end
