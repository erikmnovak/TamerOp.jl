"""Exact barcode-row representations of finite two-parameter modules over F2.

The construction retains barcode bases and the full connecting linear maps.
It follows the graphcode viewpoint of Kerber--Russold, arXiv:2503.07368,
Sections 3--4. Graphcodes depend on the chosen bases; connected components
give summands, not necessarily indecomposable summands.
"""
module Graphcodes

using LinearAlgebra: I
using ..CoreModules: F2, coeff_type, coerce
using ..FiniteFringe: ProductOfChainsPoset, GridPoset, nvertices, leq, cover_edges
using ..Modules: PModule, PMorphism, map_leq, check_module_functoriality
import ..FieldLinAlg
import ..ZigzagPersistence
import ..Results
import ..ChainComplexes: describe
import ..FiniteFringe: field

include("graphcodes/core.jl")
include("graphcodes/construction.jl")
include("graphcodes/compression.jl")

end
