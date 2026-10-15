"""Linear persistence contours and exact generator-complexity curves for
finitely presented bivariate interval modules. General direct sums are not
handled by adding interval values.
"""
module PersistenceContours
using ..CoreModules: QQ, QQField, PrimeField, coeff_type
using ..FiniteFringe: ProductOfChainsPoset, cover_edges, nvertices
using ..Modules: PModule, check_module_functoriality
import ..ChainComplexes: describe
import ..Results: provenance
import ..FiniteFringe: field

include("persistence_contours/types.jl")
include("persistence_contours/intervals.jl")
include("persistence_contours/curves.jl")
end
