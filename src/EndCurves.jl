"""
Counts and birth/death curves of finite-support lattice modules over exact fields.
The curves decompose the cokernel and kernel of `xy: M[-1,-1] \u2192 M`, with zero
extension outside the declared grid. Definitions follow Br\u00fcstle\u2013Oudot\u2013Scoccola\u2013
Thomas, *Counts and end-curves in two-parameter persistence*, Definitions 1\u20132
and Theorem 3.4: https://arxiv.org/html/2505.13412v2.
"""
module EndCurves

using ..CoreModules: QQField, PrimeField, coeff_type
using ..FiniteFringe: ProductOfChainsPoset, GridPoset, nvertices, cover_edges
using ..Modules: PModule, PMorphism, map_leq, check_module_functoriality, check_morphism
import ..AbelianCategories
import ..IntervalDecompositions
import ..FieldLinAlg
import ..Results
import ..ChainComplexes: describe
import ..FiniteFringe: field

include("end_curves/types.jl")
include("end_curves/computation.jl")

end
