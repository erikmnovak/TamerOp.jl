"""
Certified Krull-Schmidt decomposition on a specified finite poset. The native
solver splits primary factors of exact endomorphisms and certifies terminal
pieces by a nilpotent ideal with field quotient, or exhaustive prime-field
search. The optional Hecke backend certifies rational division quotients by
Schur indices. No incomplete search is returned as a decomposition.
Splittings depend on bases; no ambient or stability claim is made.
"""
module IndecomposableDecompositions

import Nemo
using ..CoreModules: QQ, QQField, PrimeField, coeff_type, eye
using ..FiniteFringe: nvertices, cover_edges
using ..Modules: PModule, PMorphism, id_morphism, check_module_functoriality, check_morphism
using ..DerivedFunctors: Hom
import ..FieldLinAlg
import ..IntervalDecompositions
import ..IntervalDecompositions: decomposition_maps
import ..ChainComplexes: describe
import ..DerivedFunctors: source_module
import ..Encoding: source_poset
import ..Results: provenance

include("indecomposable_decompositions/algebra.jl")
include("indecomposable_decompositions/solver.jl")
include("indecomposable_decompositions/results.jl")
end
