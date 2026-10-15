"""Time-indexed source filtrations and their exact extended zigzag modules.

Spatiotemporal landscapes use generalized rank on closed integer-radius squares
in the occurrence-by-scale grid, following Flammer and Hueper,
*Spatiotemporal persistence landscapes*, J. Appl. Comput. Topol. 10, 24 (2026),
https://doi.org/10.1007/s41468-026-00251-1, Sections 3--4. The finite-grid
boundary convention is explicit; occurrence steps are not elapsed time.
"""
module TemporalPersistence
using SparseArrays
using ..CoreModules: F2, QQField, PrimeField, coeff_type, coerce
import ..DataTypes
import ..DataIngestion
import ..Options
import ..ChainComplexes
import ..FieldLinAlg
import ..FiniteFringe
import ..Modules
import ..Invariants
import ..ZigzagPersistence
import ..ChainComplexes: describe
include("temporal_persistence/core.jl")
include("temporal_persistence/ingestion.jl")
include("temporal_persistence/landscapes.jl")
end
