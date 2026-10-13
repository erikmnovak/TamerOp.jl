"""Exact graded cohomology algebras and their contravariant cup-image persistence.

Supplied rings are checked algebraically. Ordered simplicial cochains provide
an Alexander--Whitney product; arbitrary cellular boundaries do not specify one.
"""
module CohomologyProducts

using ..CoreModules: QQField, PrimeField, FpElem, F2, coeff_type, coerce, eye
using ..FiniteFringe: FinitePoset, nvertices, leq
using ..Modules: PModule, check_module
import ..Modules: structure_map
import ..FieldLinAlg
import ..DataTypes
import ..ChainComplexes: describe, dimensions
import ..FiniteFringe: field
import ..Results: provenance
import ..ZigzagPersistence

include("cohomology_products/algebras.jl")
include("cohomology_products/simplicial.jl")
include("cohomology_products/persistence.jl")

end
