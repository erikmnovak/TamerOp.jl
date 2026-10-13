# Finite type-A representations, directed chain maps, and simplex event streams.
module ZigzagPersistence

using LinearAlgebra: I
using SparseArrays
using ..CoreModules: QQField, PrimeField, F2, FpElem, coeff_type, coerce, eye
import ..FieldLinAlg
import ..ChainComplexes
import ..ChainComplexes: describe, dimensions
import ..FiniteFringe: field
import ..Results: provenance

include("zigzag_persistence/core.jl")
include("zigzag_persistence/reduction.jl")
include("zigzag_persistence/chains.jl")

end
