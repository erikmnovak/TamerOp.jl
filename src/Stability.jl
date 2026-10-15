"""Exact, budgeted Harder--Narasimhan filtrations and skyscraper queries on finite encodings."""
module Stability

import Nemo
import ..EncodingCore as EC
import ..RegionGeometry as RG
import ..PLBackend as PLB
import ..PLPolyhedra as PLP
import ..Results
using ..ZnEncoding: SignaturePoset, region_signature

using Random: MersenneTwister, rand
using ..CoreModules: QQ, QQField, PrimeField, coeff_type, eye
using ..FiniteFringe: nvertices, leq, cover_edges, GridPoset, ProductOfChainsPoset
using ..Modules: PModule, PMorphism, map_leq, check_module_functoriality, check_morphism
import ..FieldLinAlg as FL
import ..AbelianCategories as AC
import ..IndecomposableDecompositions as ID
import ..ChainComplexes: describe, dimensions
import ..FiniteFringe: field
import ..Results: provenance

include("stability/types.jl")
include("stability/filtrations.jl")
include("stability/matrix_spaces.jl")
include("stability/skyscraper.jl")
include("stability/geometry.jl")
include("stability/chambers.jl")
include("stability/splittings.jl")
include("stability/landscapes.jl")

end
