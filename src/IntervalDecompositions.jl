"""
    IntervalDecompositions

Exact interval-summand multiplicities in the category of modules on a specified
finite poset. A composition pairing computes one multiplicity; top/socle endpoint
supports bound the complete candidate search. Optional simultaneous splitting
maps identify the interval part and its residual. This is not a solver for
general indecomposable summands or an ambient, encoding-independent barcode.
"""
module IntervalDecompositions

using ..CoreModules: QQField, PrimeField, eye
using ..FiniteFringe: nvertices, leq, cover_edges
using ..Modules: PModule, PMorphism, check_module_functoriality, check_morphism,
                 zero_pmodule, id_morphism, direct_sum
using ..DerivedFunctors: Hom, basis
using ..AbelianCategories: kernel_with_inclusion
import ..FieldLinAlg
import ..ChainComplexes: describe
import ..DerivedFunctors: source_module
import ..Encoding: source_poset
import ..Invariants: interval_family
import ..Results: provenance

include("interval_decompositions/core.jl")
include("interval_decompositions/results.jl")

end
