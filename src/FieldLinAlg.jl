module FieldLinAlg
"""
    FieldLinAlg

Owner module for coefficient-field-aware linear algebra in PosetModules.

Scope:
- backend routing and threshold state,
- exact and floating-point kernels,
- sparse elimination utilities,
- public rank/nullspace/solve APIs.

Implementation is split across private include files in `src/field_linalg/`.
"""

import Nemo
using LinearAlgebra
using SparseArrays
using Random
using TOML
using Dates
using Statistics

using ..CoreModules: AbstractCoeffField, QQField, PrimeField, RealField, FpElem,
                     BackendMatrix, coeff_type, eye, QQ,
                     _unwrap_backend_matrix, _backend_kind, _backend_payload,
                     _set_backend_payload!

include("field_linalg/thresholds.jl")
include("field_linalg/autotune.jl")
include("field_linalg/backend_routing.jl")
include("field_linalg/nonqq_engines.jl")
include("field_linalg/sparse_rref.jl")
include("field_linalg/qq_engine.jl")
include("field_linalg/public_api.jl")

end # module FieldLinAlg
