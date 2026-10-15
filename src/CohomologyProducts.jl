"""Exact graded cohomology algebras and their contravariant cup-image persistence.

Supplied rings are checked algebraically. Ordered simplicial cochains provide
an Alexander--Whitney product; arbitrary cellular boundaries do not specify one.
"""
module CohomologyProducts

# Share the finite-complex capability on the curated cohomology entrypoint.
import ..ModuleComplexes: cohomology_module


using ..CoreModules: QQField, PrimeField, FpElem, F2, coeff_type, coerce, eye
using ..FiniteFringe: AbstractPoset, FinitePoset, ProductOfChainsPoset, nvertices, leq, cover_edges
using ..Modules: PModule, check_module, check_module_functoriality
import ..Modules: structure_map
import ..FieldLinAlg
import ..DataTypes
import ..ChainComplexes: describe, dimensions
import ..FiniteFringe: field
import ..Results: provenance
import ..ZigzagPersistence

include("cohomology_products/algebras.jl")
include("cohomology_products/simplicial.jl")
include("cohomology_products/diagrams.jl")
include("cohomology_products/sources.jl")
include("cohomology_products/persistence.jl")

end
