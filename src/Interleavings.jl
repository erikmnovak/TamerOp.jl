"""Exact ambient shifts, interleaving certificates, and budgeted morphism search."""
module Interleavings

using ..CoreModules: QQ, QQField, PrimeField, coeff_type, coerce
using ..FiniteFringe: GridPoset, nvertices, leq, cover_edges
using ..Modules: PModule, PMorphism, map_leq, check_module_functoriality, check_morphism
import ..AbelianCategories: ambient_module
using ..Encoding: EncodingMap
using ..ChangeOfPosets: pullback
using ..DerivedFunctors: Hom, basis
using ..EncodingCore: CompiledEncoding, GridEncodingMap, encoding_map, encoding_poset, axes_from_encoding, locate
using ..PLBackend: PLEncodingMapBoxes
using ..ZnEncoding: ZnEncodingMap
using ..Results: EncodingResult
import ..Results: encoding_module
import ..EncodingCore: encoding_axes, locate
import ..FieldLinAlg
import ..ChainComplexes: describe
import ..Modules: structure_map
import ..Results: provenance

include("interleavings/ambient.jl")
include("interleavings/certificates.jl")
include("interleavings/search.jl")

end
