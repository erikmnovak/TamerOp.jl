"""Exact, budgeted Euclidean sparse multicover geometry and subdivision.

The construction follows Alonso, arXiv:2411.06986v2, Definitions 2.2--3.5
and the squared-radius optimization in Section 5.3. Radius thresholds retain
their inclusion convention. Geometry uses algebraic numbers, not tolerances.
"""
module SparseMulticovers

using LinearAlgebra: det
using SparseArrays: sparse, spzeros, SparseMatrixCSC
using ..CoreModules: QQ, QQField
using ..ExactReals: AlgebraicReal
using ..DataTypes: PointCloud, point_matrix, MultiCriticalGradedComplex, GradedComplex
using ..Results: EncodingResult
import ..Results: provenance, encoding_module
import ..ChainComplexes: describe
import ..EncodingCore
import ..DataIngestion
import ..Workflow: encode

include("sparse_multicovers/geometry.jl")
include("sparse_multicovers/subdivision.jl")
include("sparse_multicovers/queries.jl")

end
