module FieldLinAlg
# Generic linear algebra interface over configurable coefficient fields.

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

# -----------------------------------------------------------------------------
# Backend selection + Nemo hooks
# -----------------------------------------------------------------------------

const _NEMO_ENABLED = Ref(true)
_have_nemo() = _NEMO_ENABLED[]

const NEMO_THRESHOLD = Ref(50_000)     # heuristic: use Nemo if m*n >= threshold
const QQ_NEMO_RANK_THRESHOLD_SQUARE = Ref(50_000)
const QQ_NEMO_RANK_THRESHOLD_TALL = Ref(50_000)
const QQ_NEMO_RANK_THRESHOLD_WIDE = Ref(50_000)
const QQ_NEMO_NULLSPACE_THRESHOLD_SQUARE = Ref(50_000)
const QQ_NEMO_NULLSPACE_THRESHOLD_TALL = Ref(50_000)
const QQ_NEMO_NULLSPACE_THRESHOLD_WIDE = Ref(50_000)
const QQ_NEMO_SOLVE_THRESHOLD_SQUARE = Ref(50_000)
const QQ_NEMO_SOLVE_THRESHOLD_TALL = Ref(50_000)
const QQ_NEMO_SOLVE_THRESHOLD_WIDE = Ref(50_000)
const QQ_NEMO_SPARSE_SOLVE_THRESHOLD_SQUARE_LOW = Ref(10_000)
const QQ_NEMO_SPARSE_SOLVE_THRESHOLD_SQUARE_MID = Ref(3_000)
const QQ_NEMO_SPARSE_SOLVE_THRESHOLD_SQUARE_HIGH = Ref(1_200)
const QQ_NEMO_SPARSE_SOLVE_THRESHOLD_TALL_LOW = Ref(10_000)
const QQ_NEMO_SPARSE_SOLVE_THRESHOLD_TALL_MID = Ref(3_000)
const QQ_NEMO_SPARSE_SOLVE_THRESHOLD_TALL_HIGH = Ref(1_200)
const QQ_NEMO_SPARSE_SOLVE_THRESHOLD_WIDE_LOW = Ref(10_000)
const QQ_NEMO_SPARSE_SOLVE_THRESHOLD_WIDE_MID = Ref(3_000)
const QQ_NEMO_SPARSE_SOLVE_THRESHOLD_WIDE_HIGH = Ref(1_200)
const QQ_NEMO_SPARSE_SOLVE_POLICY_SQUARE_LOW = Ref(1)   # 1 => use nnz >= threshold, -1 => use nnz <= threshold
const QQ_NEMO_SPARSE_SOLVE_POLICY_SQUARE_MID = Ref(1)
const QQ_NEMO_SPARSE_SOLVE_POLICY_SQUARE_HIGH = Ref(1)
const QQ_NEMO_SPARSE_SOLVE_POLICY_TALL_LOW = Ref(1)
const QQ_NEMO_SPARSE_SOLVE_POLICY_TALL_MID = Ref(1)
const QQ_NEMO_SPARSE_SOLVE_POLICY_TALL_HIGH = Ref(1)
const QQ_NEMO_SPARSE_SOLVE_POLICY_WIDE_LOW = Ref(1)
const QQ_NEMO_SPARSE_SOLVE_POLICY_WIDE_MID = Ref(1)
const QQ_NEMO_SPARSE_SOLVE_POLICY_WIDE_HIGH = Ref(1)
const QQ_MODULAR_NULLSPACE_THRESHOLD_SQUARE = Ref(120_000)
const QQ_MODULAR_NULLSPACE_THRESHOLD_TALL = Ref(120_000)
const QQ_MODULAR_NULLSPACE_THRESHOLD_WIDE = Ref(120_000)
const QQ_MODULAR_SOLVE_THRESHOLD_SQUARE = Ref(120_000)
const QQ_MODULAR_SOLVE_THRESHOLD_TALL = Ref(120_000)
const QQ_MODULAR_SOLVE_THRESHOLD_WIDE = Ref(120_000)
const FP_NEMO_RANK_THRESHOLD = Ref(40_000)
const FP_NEMO_NULLSPACE_THRESHOLD = Ref(60_000)
const FP_NEMO_SOLVE_THRESHOLD = Ref(60_000)
const FLOAT_NULLSPACE_SVD_THRESHOLD = Ref(180_000)
const FLOAT_SPARSE_SVDS_MIN_DIM = Ref(1_024)
const FLOAT_SPARSE_SVDS_MIN_NNZ = Ref(120_000)
const ZN_QQ_DIMAT_SUBMATRIX_WORK_THRESHOLD = Ref(48)
const MODULAR_NULLSPACE_THRESHOLD = Ref(120_000)
const MODULAR_SOLVE_THRESHOLD = Ref(120_000)
const MODULAR_MIN_PRIMES = Ref(2)
const MODULAR_MAX_PRIMES = Ref(6)
const RANKQQ_DIM_SMALL_THRESHOLD = Ref(20_000)
const ROWPOS_STAMP_MIN_ROWS = Ref(128)
const ROWPOS_STAMP_DENSITY_NUM = Ref(1) # use stamp when nr/m >= 1/5
const ROWPOS_STAMP_DENSITY_DEN = Ref(5)
const TINY_LINALG_MAX_DIM = Ref(4)
const LINALG_THRESHOLD_SCHEMA_VERSION = 1
const _LINALG_THRESHOLDS_INITIALIZED = Ref(false)

@inline zn_qq_dimat_submatrix_work_threshold() = Int(ZN_QQ_DIMAT_SUBMATRIX_WORK_THRESHOLD[])
@inline function _set_zn_qq_dimat_submatrix_work_threshold!(x::Integer)
    ZN_QQ_DIMAT_SUBMATRIX_WORK_THRESHOLD[] = max(1, Int(x))
    return nothing
end

abstract type MatrixBackendTrait end
struct JuliaMatrixBackend <: MatrixBackendTrait end
struct NemoMatrixBackend <: MatrixBackendTrait end

const _JULIA_BACKEND_TRAIT = JuliaMatrixBackend()
const _NEMO_BACKEND_TRAIT = NemoMatrixBackend()
const _SVDS_IMPL = Ref{Any}(nothing)

@inline function _matrix_backend_trait(field::AbstractCoeffField, A;
                                      op::Symbol=:rank, backend::Symbol=:auto)
    be = _choose_linalg_backend(field, A; op=op, backend=backend)
    return be == :nemo ? _NEMO_BACKEND_TRAIT : _JULIA_BACKEND_TRAIT
end

const _QQ_TO_NEMO_CONVERSIONS = Base.Threads.Atomic{Int}(0)
const _QQ_TO_NEMO_CACHE_HITS = Base.Threads.Atomic{Int}(0)
const _QQ_FROM_NEMO_CONVERSIONS = Base.Threads.Atomic{Int}(0)
const _FP_TO_NEMO_CONVERSIONS = Base.Threads.Atomic{Int}(0)
const _FP_TO_NEMO_CACHE_HITS = Base.Threads.Atomic{Int}(0)
const _FP_FROM_NEMO_CONVERSIONS = Base.Threads.Atomic{Int}(0)

@inline _bump_counter!(x::Base.Threads.Atomic{Int}) = Base.Threads.atomic_add!(x, 1)

function _reset_conversion_counters!()
    _QQ_TO_NEMO_CONVERSIONS[] = 0
    _QQ_TO_NEMO_CACHE_HITS[] = 0
    _QQ_FROM_NEMO_CONVERSIONS[] = 0
    _FP_TO_NEMO_CONVERSIONS[] = 0
    _FP_TO_NEMO_CACHE_HITS[] = 0
    _FP_FROM_NEMO_CONVERSIONS[] = 0
    return nothing
end

function _conversion_counters()
    return (
        qq_to_nemo=_QQ_TO_NEMO_CONVERSIONS[],
        qq_to_nemo_cache_hits=_QQ_TO_NEMO_CACHE_HITS[],
        qq_from_nemo=_QQ_FROM_NEMO_CONVERSIONS[],
        fp_to_nemo=_FP_TO_NEMO_CONVERSIONS[],
        fp_to_nemo_cache_hits=_FP_TO_NEMO_CACHE_HITS[],
        fp_from_nemo=_FP_FROM_NEMO_CONVERSIONS[],
    )
end

@inline _have_svds_backend() = _SVDS_IMPL[] !== nothing

function _set_svds_impl!(impl)
    _SVDS_IMPL[] = impl
    return nothing
end

function _nullspace_float_svds(F::RealField, A)
    impl = _SVDS_IMPL[]
    impl === nothing && return nothing
    return impl(F, A)
end

@inline _nemo_dense_compatible(A) = A isa StridedMatrix
@inline _nemo_dense_compatible(::BackendMatrix) = true
@inline _is_sparse_like(A) =
    A isa SparseMatrixCSC ||
    A isa Transpose{<:Any,<:SparseMatrixCSC} ||
    A isa Adjoint{<:Any,<:SparseMatrixCSC}
@inline _sparse_parent(A::SparseMatrixCSC) = A
@inline _sparse_parent(A::Transpose{<:Any,<:SparseMatrixCSC}) = parent(A)
@inline _sparse_parent(A::Adjoint{<:Any,<:SparseMatrixCSC}) = parent(A)
@inline _sparse_nnz(A) = nnz(_sparse_parent(A))
@inline _use_rowpos_stamp(m::Int, nr::Int) =
    nr >= ROWPOS_STAMP_MIN_ROWS[] && nr * ROWPOS_STAMP_DENSITY_DEN[] >= m * ROWPOS_STAMP_DENSITY_NUM[]
@inline _is_tiny_dim(x::Int) = x <= TINY_LINALG_MAX_DIM[]
@inline _is_tiny_matrix_dims(m::Int, n::Int) = _is_tiny_dim(m) && _is_tiny_dim(n)
@inline _is_tiny_matrix(A::AbstractMatrix) = _is_tiny_matrix_dims(size(A, 1), size(A, 2))
@inline function _qq_shape_bucket(m::Int, n::Int)::Symbol
    # Bucket by aspect ratio to keep backend routing stable across square/tall/wide regimes.
    if m * 5 >= n * 4 && n * 5 >= m * 4
        return :square
    elseif m > n
        return :tall
    else
        return :wide
    end
end

@inline function _qq_nemo_threshold(op::Symbol, shape::Symbol)::Int
    if op == :rank
        return shape == :square ? QQ_NEMO_RANK_THRESHOLD_SQUARE[] :
               shape == :tall ? QQ_NEMO_RANK_THRESHOLD_TALL[] :
               QQ_NEMO_RANK_THRESHOLD_WIDE[]
    elseif op == :nullspace
        return shape == :square ? QQ_NEMO_NULLSPACE_THRESHOLD_SQUARE[] :
               shape == :tall ? QQ_NEMO_NULLSPACE_THRESHOLD_TALL[] :
               QQ_NEMO_NULLSPACE_THRESHOLD_WIDE[]
    elseif op == :solve
        return shape == :square ? QQ_NEMO_SOLVE_THRESHOLD_SQUARE[] :
               shape == :tall ? QQ_NEMO_SOLVE_THRESHOLD_TALL[] :
               QQ_NEMO_SOLVE_THRESHOLD_WIDE[]
    end
    return NEMO_THRESHOLD[]
end

@inline function _qq_modular_threshold(op::Symbol, shape::Symbol)::Int
    if op == :nullspace
        return shape == :square ? QQ_MODULAR_NULLSPACE_THRESHOLD_SQUARE[] :
               shape == :tall ? QQ_MODULAR_NULLSPACE_THRESHOLD_TALL[] :
               QQ_MODULAR_NULLSPACE_THRESHOLD_WIDE[]
    elseif op == :solve
        return shape == :square ? QQ_MODULAR_SOLVE_THRESHOLD_SQUARE[] :
               shape == :tall ? QQ_MODULAR_SOLVE_THRESHOLD_TALL[] :
               QQ_MODULAR_SOLVE_THRESHOLD_WIDE[]
    end
    return max(MODULAR_SOLVE_THRESHOLD[], MODULAR_NULLSPACE_THRESHOLD[])
end

@inline function _qq_sparse_density_bucket(dens::Float64)::Symbol
    if dens <= 0.08
        return :low
    elseif dens <= 0.28
        return :mid
    end
    return :high
end

@inline _qq_sparse_density_bucket(A) = _qq_sparse_density_bucket((size(A, 1) * size(A, 2)) == 0 ? 0.0 : (_sparse_nnz(A) / (size(A, 1) * size(A, 2))))

@inline function _qq_sparse_solve_fill(A)::Float64
    m, n = size(A)
    nnzA = _sparse_nnz(A)
    if m > n
        free_slots = (m - n) * n
        free_slots <= 0 && return 1.0
        return clamp((nnzA - n) / free_slots, 0.0, 1.0)
    end
    work = m * n
    work == 0 && return 0.0
    return clamp(nnzA / work, 0.0, 1.0)
end

@inline function _qq_sparse_solve_threshold(shape::Symbol, bucket::Symbol)::Int
    if shape == :square
        return bucket == :low ? QQ_NEMO_SPARSE_SOLVE_THRESHOLD_SQUARE_LOW[] :
               bucket == :mid ? QQ_NEMO_SPARSE_SOLVE_THRESHOLD_SQUARE_MID[] :
               QQ_NEMO_SPARSE_SOLVE_THRESHOLD_SQUARE_HIGH[]
    elseif shape == :tall
        return bucket == :low ? QQ_NEMO_SPARSE_SOLVE_THRESHOLD_TALL_LOW[] :
               bucket == :mid ? QQ_NEMO_SPARSE_SOLVE_THRESHOLD_TALL_MID[] :
               QQ_NEMO_SPARSE_SOLVE_THRESHOLD_TALL_HIGH[]
    end
    return bucket == :low ? QQ_NEMO_SPARSE_SOLVE_THRESHOLD_WIDE_LOW[] :
           bucket == :mid ? QQ_NEMO_SPARSE_SOLVE_THRESHOLD_WIDE_MID[] :
           QQ_NEMO_SPARSE_SOLVE_THRESHOLD_WIDE_HIGH[]
end

@inline function _qq_sparse_solve_use_ge(shape::Symbol, bucket::Symbol)::Bool
    if shape == :square
        return (bucket == :low ? QQ_NEMO_SPARSE_SOLVE_POLICY_SQUARE_LOW[] :
                bucket == :mid ? QQ_NEMO_SPARSE_SOLVE_POLICY_SQUARE_MID[] :
                QQ_NEMO_SPARSE_SOLVE_POLICY_SQUARE_HIGH[]) > 0
    elseif shape == :tall
        return (bucket == :low ? QQ_NEMO_SPARSE_SOLVE_POLICY_TALL_LOW[] :
                bucket == :mid ? QQ_NEMO_SPARSE_SOLVE_POLICY_TALL_MID[] :
                QQ_NEMO_SPARSE_SOLVE_POLICY_TALL_HIGH[]) > 0
    end
    return (bucket == :low ? QQ_NEMO_SPARSE_SOLVE_POLICY_WIDE_LOW[] :
            bucket == :mid ? QQ_NEMO_SPARSE_SOLVE_POLICY_WIDE_MID[] :
            QQ_NEMO_SPARSE_SOLVE_POLICY_WIDE_HIGH[]) > 0
end
@inline function _is_tiny_solve(B, Y)
    rhs = Y isa AbstractVector ? 1 : size(Y, 2)
    return _is_tiny_matrix_dims(size(B, 1), size(B, 2)) && _is_tiny_dim(rhs)
end
@inline function _is_tiny_mul(A::AbstractMatrix, B::AbstractMatrix)
    m, k = size(A)
    k2, n = size(B)
    return k == k2 && _is_tiny_matrix_dims(m, k) && _is_tiny_matrix_dims(k2, n)
end

function _matmul_tiny(A::AbstractMatrix{TA}, B::AbstractMatrix{TB}) where {TA,TB}
    m, k = size(A)
    k2, n = size(B)
    k == k2 || throw(DimensionMismatch("A and B inner dimensions must match"))
    T = promote_type(TA, TB)
    C = zeros(T, m, n)
    @inbounds for j in 1:n
        for t in 1:k
            bt = B[t, j]
            iszero(bt) && continue
            for i in 1:m
                C[i, j] += A[i, t] * bt
            end
        end
    end
    return C
end

@inline function _matmul(A::AbstractMatrix, B::AbstractMatrix)
    if _is_tiny_mul(A, B)
        return _matmul_tiny(A, B)
    end
    return A * B
end

struct _RowLocatorDict
    rowpos::Dict{Int,Int}
end

struct _RowLocatorStamp
    pos::Vector{Int}
    marks::Vector{UInt32}
    tag::UInt32
end

function _build_row_locator(rows::AbstractVector{Int}, m::Int)
    nr = length(rows)
    if _use_rowpos_stamp(m, nr)
        pos = zeros(Int, m)
        marks = zeros(UInt32, m)
        tag = UInt32(1)
        @inbounds for (i, r) in enumerate(rows)
            pos[r] = i
            marks[r] = tag
        end
        return _RowLocatorStamp(pos, marks, tag)
    end
    rowpos = Dict{Int,Int}()
    sizehint!(rowpos, nr)
    @inbounds for (i, r) in enumerate(rows)
        rowpos[r] = i
    end
    return _RowLocatorDict(rowpos)
end

@inline _row_lookup(loc::_RowLocatorDict, r::Int) = get(loc.rowpos, r, 0)
@inline _row_lookup(loc::_RowLocatorStamp, r::Int) = @inbounds (loc.marks[r] == loc.tag ? loc.pos[r] : 0)

@inline _linalg_repo_root() = normpath(joinpath(@__DIR__, ".."))

function _linalg_thresholds_path(; root::AbstractString=_linalg_repo_root())
    return joinpath(root, "linalg_thresholds.toml")
end

function _current_linalg_fingerprint()
    return Dict(
        "schema_version" => LINALG_THRESHOLD_SCHEMA_VERSION,
        "cpu_name" => String(Sys.CPU_NAME),
        "cpu_threads" => Int(Sys.CPU_THREADS),
        "julia_version" => string(VERSION),
        "word_size" => Int(Sys.WORD_SIZE),
        "os" => string(Sys.KERNEL),
        "blas_threads" => Int(LinearAlgebra.BLAS.get_num_threads()),
    )
end

function _current_linalg_thresholds()
    return Dict(
        "nemo_threshold" => Int(NEMO_THRESHOLD[]),
        "qq_nemo_rank_threshold_square" => Int(QQ_NEMO_RANK_THRESHOLD_SQUARE[]),
        "qq_nemo_rank_threshold_tall" => Int(QQ_NEMO_RANK_THRESHOLD_TALL[]),
        "qq_nemo_rank_threshold_wide" => Int(QQ_NEMO_RANK_THRESHOLD_WIDE[]),
        "qq_nemo_nullspace_threshold_square" => Int(QQ_NEMO_NULLSPACE_THRESHOLD_SQUARE[]),
        "qq_nemo_nullspace_threshold_tall" => Int(QQ_NEMO_NULLSPACE_THRESHOLD_TALL[]),
        "qq_nemo_nullspace_threshold_wide" => Int(QQ_NEMO_NULLSPACE_THRESHOLD_WIDE[]),
        "qq_nemo_solve_threshold_square" => Int(QQ_NEMO_SOLVE_THRESHOLD_SQUARE[]),
        "qq_nemo_solve_threshold_tall" => Int(QQ_NEMO_SOLVE_THRESHOLD_TALL[]),
        "qq_nemo_solve_threshold_wide" => Int(QQ_NEMO_SOLVE_THRESHOLD_WIDE[]),
        "qq_nemo_sparse_solve_threshold_square_low" => Int(QQ_NEMO_SPARSE_SOLVE_THRESHOLD_SQUARE_LOW[]),
        "qq_nemo_sparse_solve_threshold_square_mid" => Int(QQ_NEMO_SPARSE_SOLVE_THRESHOLD_SQUARE_MID[]),
        "qq_nemo_sparse_solve_threshold_square_high" => Int(QQ_NEMO_SPARSE_SOLVE_THRESHOLD_SQUARE_HIGH[]),
        "qq_nemo_sparse_solve_threshold_tall_low" => Int(QQ_NEMO_SPARSE_SOLVE_THRESHOLD_TALL_LOW[]),
        "qq_nemo_sparse_solve_threshold_tall_mid" => Int(QQ_NEMO_SPARSE_SOLVE_THRESHOLD_TALL_MID[]),
        "qq_nemo_sparse_solve_threshold_tall_high" => Int(QQ_NEMO_SPARSE_SOLVE_THRESHOLD_TALL_HIGH[]),
        "qq_nemo_sparse_solve_threshold_wide_low" => Int(QQ_NEMO_SPARSE_SOLVE_THRESHOLD_WIDE_LOW[]),
        "qq_nemo_sparse_solve_threshold_wide_mid" => Int(QQ_NEMO_SPARSE_SOLVE_THRESHOLD_WIDE_MID[]),
        "qq_nemo_sparse_solve_threshold_wide_high" => Int(QQ_NEMO_SPARSE_SOLVE_THRESHOLD_WIDE_HIGH[]),
        "qq_nemo_sparse_solve_policy_square_low" => Int(QQ_NEMO_SPARSE_SOLVE_POLICY_SQUARE_LOW[]),
        "qq_nemo_sparse_solve_policy_square_mid" => Int(QQ_NEMO_SPARSE_SOLVE_POLICY_SQUARE_MID[]),
        "qq_nemo_sparse_solve_policy_square_high" => Int(QQ_NEMO_SPARSE_SOLVE_POLICY_SQUARE_HIGH[]),
        "qq_nemo_sparse_solve_policy_tall_low" => Int(QQ_NEMO_SPARSE_SOLVE_POLICY_TALL_LOW[]),
        "qq_nemo_sparse_solve_policy_tall_mid" => Int(QQ_NEMO_SPARSE_SOLVE_POLICY_TALL_MID[]),
        "qq_nemo_sparse_solve_policy_tall_high" => Int(QQ_NEMO_SPARSE_SOLVE_POLICY_TALL_HIGH[]),
        "qq_nemo_sparse_solve_policy_wide_low" => Int(QQ_NEMO_SPARSE_SOLVE_POLICY_WIDE_LOW[]),
        "qq_nemo_sparse_solve_policy_wide_mid" => Int(QQ_NEMO_SPARSE_SOLVE_POLICY_WIDE_MID[]),
        "qq_nemo_sparse_solve_policy_wide_high" => Int(QQ_NEMO_SPARSE_SOLVE_POLICY_WIDE_HIGH[]),
        "qq_modular_nullspace_threshold_square" => Int(QQ_MODULAR_NULLSPACE_THRESHOLD_SQUARE[]),
        "qq_modular_nullspace_threshold_tall" => Int(QQ_MODULAR_NULLSPACE_THRESHOLD_TALL[]),
        "qq_modular_nullspace_threshold_wide" => Int(QQ_MODULAR_NULLSPACE_THRESHOLD_WIDE[]),
        "qq_modular_solve_threshold_square" => Int(QQ_MODULAR_SOLVE_THRESHOLD_SQUARE[]),
        "qq_modular_solve_threshold_tall" => Int(QQ_MODULAR_SOLVE_THRESHOLD_TALL[]),
        "qq_modular_solve_threshold_wide" => Int(QQ_MODULAR_SOLVE_THRESHOLD_WIDE[]),
        "fp_nemo_rank_threshold" => Int(FP_NEMO_RANK_THRESHOLD[]),
        "fp_nemo_nullspace_threshold" => Int(FP_NEMO_NULLSPACE_THRESHOLD[]),
        "fp_nemo_solve_threshold" => Int(FP_NEMO_SOLVE_THRESHOLD[]),
        "modular_nullspace_threshold" => Int(MODULAR_NULLSPACE_THRESHOLD[]),
        "modular_solve_threshold" => Int(MODULAR_SOLVE_THRESHOLD[]),
        "modular_min_primes" => Int(MODULAR_MIN_PRIMES[]),
        "modular_max_primes" => Int(MODULAR_MAX_PRIMES[]),
        "rankqq_dim_small_threshold" => Int(RANKQQ_DIM_SMALL_THRESHOLD[]),
        "float_nullspace_svd_threshold" => Int(FLOAT_NULLSPACE_SVD_THRESHOLD[]),
        "float_sparse_svds_min_dim" => Int(FLOAT_SPARSE_SVDS_MIN_DIM[]),
        "float_sparse_svds_min_nnz" => Int(FLOAT_SPARSE_SVDS_MIN_NNZ[]),
        "zn_qq_dimat_submatrix_work_threshold" => Int(ZN_QQ_DIMAT_SUBMATRIX_WORK_THRESHOLD[]),
    )
end

function _apply_linalg_thresholds!(vals)::Bool
    try
        qq_nemo_fallback = Int(get(vals, "nemo_threshold", NEMO_THRESHOLD[]))
        NEMO_THRESHOLD[] = qq_nemo_fallback
        QQ_NEMO_RANK_THRESHOLD_SQUARE[] = Int(get(vals, "qq_nemo_rank_threshold_square", qq_nemo_fallback))
        QQ_NEMO_RANK_THRESHOLD_TALL[] = Int(get(vals, "qq_nemo_rank_threshold_tall", qq_nemo_fallback))
        QQ_NEMO_RANK_THRESHOLD_WIDE[] = Int(get(vals, "qq_nemo_rank_threshold_wide", qq_nemo_fallback))
        QQ_NEMO_NULLSPACE_THRESHOLD_SQUARE[] = Int(get(vals, "qq_nemo_nullspace_threshold_square", qq_nemo_fallback))
        QQ_NEMO_NULLSPACE_THRESHOLD_TALL[] = Int(get(vals, "qq_nemo_nullspace_threshold_tall", qq_nemo_fallback))
        QQ_NEMO_NULLSPACE_THRESHOLD_WIDE[] = Int(get(vals, "qq_nemo_nullspace_threshold_wide", qq_nemo_fallback))
        QQ_NEMO_SOLVE_THRESHOLD_SQUARE[] = Int(get(vals, "qq_nemo_solve_threshold_square", qq_nemo_fallback))
        QQ_NEMO_SOLVE_THRESHOLD_TALL[] = Int(get(vals, "qq_nemo_solve_threshold_tall", qq_nemo_fallback))
        QQ_NEMO_SOLVE_THRESHOLD_WIDE[] = Int(get(vals, "qq_nemo_solve_threshold_wide", qq_nemo_fallback))
        qq_sparse_solve_square_fallback = QQ_NEMO_SOLVE_THRESHOLD_SQUARE[]
        qq_sparse_solve_tall_fallback = QQ_NEMO_SOLVE_THRESHOLD_TALL[]
        qq_sparse_solve_wide_fallback = QQ_NEMO_SOLVE_THRESHOLD_WIDE[]
        QQ_NEMO_SPARSE_SOLVE_THRESHOLD_SQUARE_LOW[] = Int(get(vals, "qq_nemo_sparse_solve_threshold_square_low", qq_sparse_solve_square_fallback))
        QQ_NEMO_SPARSE_SOLVE_THRESHOLD_SQUARE_MID[] = Int(get(vals, "qq_nemo_sparse_solve_threshold_square_mid", qq_sparse_solve_square_fallback))
        QQ_NEMO_SPARSE_SOLVE_THRESHOLD_SQUARE_HIGH[] = Int(get(vals, "qq_nemo_sparse_solve_threshold_square_high", qq_sparse_solve_square_fallback))
        QQ_NEMO_SPARSE_SOLVE_THRESHOLD_TALL_LOW[] = Int(get(vals, "qq_nemo_sparse_solve_threshold_tall_low", qq_sparse_solve_tall_fallback))
        QQ_NEMO_SPARSE_SOLVE_THRESHOLD_TALL_MID[] = Int(get(vals, "qq_nemo_sparse_solve_threshold_tall_mid", qq_sparse_solve_tall_fallback))
        QQ_NEMO_SPARSE_SOLVE_THRESHOLD_TALL_HIGH[] = Int(get(vals, "qq_nemo_sparse_solve_threshold_tall_high", qq_sparse_solve_tall_fallback))
        QQ_NEMO_SPARSE_SOLVE_THRESHOLD_WIDE_LOW[] = Int(get(vals, "qq_nemo_sparse_solve_threshold_wide_low", qq_sparse_solve_wide_fallback))
        QQ_NEMO_SPARSE_SOLVE_THRESHOLD_WIDE_MID[] = Int(get(vals, "qq_nemo_sparse_solve_threshold_wide_mid", qq_sparse_solve_wide_fallback))
        QQ_NEMO_SPARSE_SOLVE_THRESHOLD_WIDE_HIGH[] = Int(get(vals, "qq_nemo_sparse_solve_threshold_wide_high", qq_sparse_solve_wide_fallback))
        QQ_NEMO_SPARSE_SOLVE_POLICY_SQUARE_LOW[] = Int(get(vals, "qq_nemo_sparse_solve_policy_square_low", QQ_NEMO_SPARSE_SOLVE_POLICY_SQUARE_LOW[]))
        QQ_NEMO_SPARSE_SOLVE_POLICY_SQUARE_MID[] = Int(get(vals, "qq_nemo_sparse_solve_policy_square_mid", QQ_NEMO_SPARSE_SOLVE_POLICY_SQUARE_MID[]))
        QQ_NEMO_SPARSE_SOLVE_POLICY_SQUARE_HIGH[] = Int(get(vals, "qq_nemo_sparse_solve_policy_square_high", QQ_NEMO_SPARSE_SOLVE_POLICY_SQUARE_HIGH[]))
        QQ_NEMO_SPARSE_SOLVE_POLICY_TALL_LOW[] = Int(get(vals, "qq_nemo_sparse_solve_policy_tall_low", QQ_NEMO_SPARSE_SOLVE_POLICY_TALL_LOW[]))
        QQ_NEMO_SPARSE_SOLVE_POLICY_TALL_MID[] = Int(get(vals, "qq_nemo_sparse_solve_policy_tall_mid", QQ_NEMO_SPARSE_SOLVE_POLICY_TALL_MID[]))
        QQ_NEMO_SPARSE_SOLVE_POLICY_TALL_HIGH[] = Int(get(vals, "qq_nemo_sparse_solve_policy_tall_high", QQ_NEMO_SPARSE_SOLVE_POLICY_TALL_HIGH[]))
        QQ_NEMO_SPARSE_SOLVE_POLICY_WIDE_LOW[] = Int(get(vals, "qq_nemo_sparse_solve_policy_wide_low", QQ_NEMO_SPARSE_SOLVE_POLICY_WIDE_LOW[]))
        QQ_NEMO_SPARSE_SOLVE_POLICY_WIDE_MID[] = Int(get(vals, "qq_nemo_sparse_solve_policy_wide_mid", QQ_NEMO_SPARSE_SOLVE_POLICY_WIDE_MID[]))
        QQ_NEMO_SPARSE_SOLVE_POLICY_WIDE_HIGH[] = Int(get(vals, "qq_nemo_sparse_solve_policy_wide_high", QQ_NEMO_SPARSE_SOLVE_POLICY_WIDE_HIGH[]))

        qq_mod_ns_fallback = Int(get(vals, "modular_nullspace_threshold", MODULAR_NULLSPACE_THRESHOLD[]))
        qq_mod_solve_fallback = Int(get(vals, "modular_solve_threshold", MODULAR_SOLVE_THRESHOLD[]))
        QQ_MODULAR_NULLSPACE_THRESHOLD_SQUARE[] = Int(get(vals, "qq_modular_nullspace_threshold_square", qq_mod_ns_fallback))
        QQ_MODULAR_NULLSPACE_THRESHOLD_TALL[] = Int(get(vals, "qq_modular_nullspace_threshold_tall", qq_mod_ns_fallback))
        QQ_MODULAR_NULLSPACE_THRESHOLD_WIDE[] = Int(get(vals, "qq_modular_nullspace_threshold_wide", qq_mod_ns_fallback))
        QQ_MODULAR_SOLVE_THRESHOLD_SQUARE[] = Int(get(vals, "qq_modular_solve_threshold_square", qq_mod_solve_fallback))
        QQ_MODULAR_SOLVE_THRESHOLD_TALL[] = Int(get(vals, "qq_modular_solve_threshold_tall", qq_mod_solve_fallback))
        QQ_MODULAR_SOLVE_THRESHOLD_WIDE[] = Int(get(vals, "qq_modular_solve_threshold_wide", qq_mod_solve_fallback))

        FP_NEMO_RANK_THRESHOLD[] = Int(get(vals, "fp_nemo_rank_threshold", FP_NEMO_RANK_THRESHOLD[]))
        FP_NEMO_NULLSPACE_THRESHOLD[] = Int(get(vals, "fp_nemo_nullspace_threshold", FP_NEMO_NULLSPACE_THRESHOLD[]))
        FP_NEMO_SOLVE_THRESHOLD[] = Int(get(vals, "fp_nemo_solve_threshold", FP_NEMO_SOLVE_THRESHOLD[]))
        MODULAR_NULLSPACE_THRESHOLD[] = Int(get(vals, "modular_nullspace_threshold", MODULAR_NULLSPACE_THRESHOLD[]))
        MODULAR_SOLVE_THRESHOLD[] = Int(get(vals, "modular_solve_threshold", MODULAR_SOLVE_THRESHOLD[]))
        MODULAR_MIN_PRIMES[] = Int(get(vals, "modular_min_primes", MODULAR_MIN_PRIMES[]))
        MODULAR_MAX_PRIMES[] = Int(get(vals, "modular_max_primes", MODULAR_MAX_PRIMES[]))
        RANKQQ_DIM_SMALL_THRESHOLD[] = Int(get(vals, "rankqq_dim_small_threshold", RANKQQ_DIM_SMALL_THRESHOLD[]))
        FLOAT_NULLSPACE_SVD_THRESHOLD[] = Int(get(vals, "float_nullspace_svd_threshold", FLOAT_NULLSPACE_SVD_THRESHOLD[]))
        FLOAT_SPARSE_SVDS_MIN_DIM[] = Int(get(vals, "float_sparse_svds_min_dim", FLOAT_SPARSE_SVDS_MIN_DIM[]))
        FLOAT_SPARSE_SVDS_MIN_NNZ[] = Int(get(vals, "float_sparse_svds_min_nnz", FLOAT_SPARSE_SVDS_MIN_NNZ[]))
        _set_zn_qq_dimat_submatrix_work_threshold!(
            Int(get(vals, "zn_qq_dimat_submatrix_work_threshold",
                    ZN_QQ_DIMAT_SUBMATRIX_WORK_THRESHOLD[])))
        MODULAR_MIN_PRIMES[] = max(1, MODULAR_MIN_PRIMES[])
        MODULAR_MAX_PRIMES[] = max(MODULAR_MIN_PRIMES[], MODULAR_MAX_PRIMES[])
        RANKQQ_DIM_SMALL_THRESHOLD[] = max(1, RANKQQ_DIM_SMALL_THRESHOLD[])
        QQ_NEMO_RANK_THRESHOLD_SQUARE[] = max(1, QQ_NEMO_RANK_THRESHOLD_SQUARE[])
        QQ_NEMO_RANK_THRESHOLD_TALL[] = max(1, QQ_NEMO_RANK_THRESHOLD_TALL[])
        QQ_NEMO_RANK_THRESHOLD_WIDE[] = max(1, QQ_NEMO_RANK_THRESHOLD_WIDE[])
        QQ_NEMO_NULLSPACE_THRESHOLD_SQUARE[] = max(1, QQ_NEMO_NULLSPACE_THRESHOLD_SQUARE[])
        QQ_NEMO_NULLSPACE_THRESHOLD_TALL[] = max(1, QQ_NEMO_NULLSPACE_THRESHOLD_TALL[])
        QQ_NEMO_NULLSPACE_THRESHOLD_WIDE[] = max(1, QQ_NEMO_NULLSPACE_THRESHOLD_WIDE[])
        QQ_NEMO_SOLVE_THRESHOLD_SQUARE[] = max(1, QQ_NEMO_SOLVE_THRESHOLD_SQUARE[])
        QQ_NEMO_SOLVE_THRESHOLD_TALL[] = max(1, QQ_NEMO_SOLVE_THRESHOLD_TALL[])
        QQ_NEMO_SOLVE_THRESHOLD_WIDE[] = max(1, QQ_NEMO_SOLVE_THRESHOLD_WIDE[])
        QQ_NEMO_SPARSE_SOLVE_THRESHOLD_SQUARE_LOW[] = max(1, QQ_NEMO_SPARSE_SOLVE_THRESHOLD_SQUARE_LOW[])
        QQ_NEMO_SPARSE_SOLVE_THRESHOLD_SQUARE_MID[] = max(1, QQ_NEMO_SPARSE_SOLVE_THRESHOLD_SQUARE_MID[])
        QQ_NEMO_SPARSE_SOLVE_THRESHOLD_SQUARE_HIGH[] = max(1, QQ_NEMO_SPARSE_SOLVE_THRESHOLD_SQUARE_HIGH[])
        QQ_NEMO_SPARSE_SOLVE_THRESHOLD_TALL_LOW[] = max(1, QQ_NEMO_SPARSE_SOLVE_THRESHOLD_TALL_LOW[])
        QQ_NEMO_SPARSE_SOLVE_THRESHOLD_TALL_MID[] = max(1, QQ_NEMO_SPARSE_SOLVE_THRESHOLD_TALL_MID[])
        QQ_NEMO_SPARSE_SOLVE_THRESHOLD_TALL_HIGH[] = max(1, QQ_NEMO_SPARSE_SOLVE_THRESHOLD_TALL_HIGH[])
        QQ_NEMO_SPARSE_SOLVE_THRESHOLD_WIDE_LOW[] = max(1, QQ_NEMO_SPARSE_SOLVE_THRESHOLD_WIDE_LOW[])
        QQ_NEMO_SPARSE_SOLVE_THRESHOLD_WIDE_MID[] = max(1, QQ_NEMO_SPARSE_SOLVE_THRESHOLD_WIDE_MID[])
        QQ_NEMO_SPARSE_SOLVE_THRESHOLD_WIDE_HIGH[] = max(1, QQ_NEMO_SPARSE_SOLVE_THRESHOLD_WIDE_HIGH[])
        QQ_NEMO_SPARSE_SOLVE_POLICY_SQUARE_LOW[] = QQ_NEMO_SPARSE_SOLVE_POLICY_SQUARE_LOW[] >= 0 ? 1 : -1
        QQ_NEMO_SPARSE_SOLVE_POLICY_SQUARE_MID[] = QQ_NEMO_SPARSE_SOLVE_POLICY_SQUARE_MID[] >= 0 ? 1 : -1
        QQ_NEMO_SPARSE_SOLVE_POLICY_SQUARE_HIGH[] = QQ_NEMO_SPARSE_SOLVE_POLICY_SQUARE_HIGH[] >= 0 ? 1 : -1
        QQ_NEMO_SPARSE_SOLVE_POLICY_TALL_LOW[] = QQ_NEMO_SPARSE_SOLVE_POLICY_TALL_LOW[] >= 0 ? 1 : -1
        QQ_NEMO_SPARSE_SOLVE_POLICY_TALL_MID[] = QQ_NEMO_SPARSE_SOLVE_POLICY_TALL_MID[] >= 0 ? 1 : -1
        QQ_NEMO_SPARSE_SOLVE_POLICY_TALL_HIGH[] = QQ_NEMO_SPARSE_SOLVE_POLICY_TALL_HIGH[] >= 0 ? 1 : -1
        QQ_NEMO_SPARSE_SOLVE_POLICY_WIDE_LOW[] = QQ_NEMO_SPARSE_SOLVE_POLICY_WIDE_LOW[] >= 0 ? 1 : -1
        QQ_NEMO_SPARSE_SOLVE_POLICY_WIDE_MID[] = QQ_NEMO_SPARSE_SOLVE_POLICY_WIDE_MID[] >= 0 ? 1 : -1
        QQ_NEMO_SPARSE_SOLVE_POLICY_WIDE_HIGH[] = QQ_NEMO_SPARSE_SOLVE_POLICY_WIDE_HIGH[] >= 0 ? 1 : -1
        QQ_MODULAR_NULLSPACE_THRESHOLD_SQUARE[] = max(1, QQ_MODULAR_NULLSPACE_THRESHOLD_SQUARE[])
        QQ_MODULAR_NULLSPACE_THRESHOLD_TALL[] = max(1, QQ_MODULAR_NULLSPACE_THRESHOLD_TALL[])
        QQ_MODULAR_NULLSPACE_THRESHOLD_WIDE[] = max(1, QQ_MODULAR_NULLSPACE_THRESHOLD_WIDE[])
        QQ_MODULAR_SOLVE_THRESHOLD_SQUARE[] = max(1, QQ_MODULAR_SOLVE_THRESHOLD_SQUARE[])
        QQ_MODULAR_SOLVE_THRESHOLD_TALL[] = max(1, QQ_MODULAR_SOLVE_THRESHOLD_TALL[])
        QQ_MODULAR_SOLVE_THRESHOLD_WIDE[] = max(1, QQ_MODULAR_SOLVE_THRESHOLD_WIDE[])
    catch
        return false
    end
    return true
end

function _save_linalg_thresholds!(; path::AbstractString=_linalg_thresholds_path())
    doc = Dict(
        "meta" => Dict(
            "created_utc" => Dates.format(Dates.now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ"),
            "schema_version" => LINALG_THRESHOLD_SCHEMA_VERSION,
        ),
        "fingerprint" => _current_linalg_fingerprint(),
        "thresholds" => _current_linalg_thresholds(),
    )
    mkpath(dirname(path))
    open(path, "w") do io
        TOML.print(io, doc)
    end
    return path
end

function _fingerprints_match(a, b)::Bool
    keys = ("schema_version", "cpu_name", "cpu_threads", "julia_version", "word_size", "os", "blas_threads")
    for k in keys
        haskey(a, k) || return false
        haskey(b, k) || return false
        if string(a[k]) != string(b[k])
            return false
        end
    end
    return true
end

function _load_linalg_thresholds!(; path::AbstractString=_linalg_thresholds_path(), warn_on_mismatch::Bool=true)::Bool
    if !isfile(path)
        return false
    end
    doc = try
        TOML.parsefile(path)
    catch err
        @warn "FieldLinAlg: failed to parse thresholds file; using defaults." path exception=(err, catch_backtrace())
        return false
    end
    haskey(doc, "fingerprint") || return false
    haskey(doc, "thresholds") || return false

    current_fp = _current_linalg_fingerprint()
    stored_fp = doc["fingerprint"]
    if !_fingerprints_match(stored_fp, current_fp)
        if warn_on_mismatch
            @warn "FieldLinAlg: threshold fingerprint mismatch; using defaults. Run `FieldLinAlg.autotune_linalg_thresholds!()` to regenerate." path
        end
        return false
    end

    ok = _apply_linalg_thresholds!(doc["thresholds"])
    if !ok
        @warn "FieldLinAlg: malformed thresholds in file; using defaults." path
        return false
    end
    return true
end

function _bench_elapsed(f; reps::Int=2)
    f() # warmup
    tbest = Inf
    for _ in 1:reps
        GC.gc()
        t = @elapsed f()
        if t < tbest
            tbest = t
        end
    end
    return tbest
end

function _bench_elapsed_median(f; reps::Int=3)
    reps <= 1 && return _bench_elapsed(f; reps=1)
    f() # warmup
    ts = Vector{Float64}(undef, reps)
    for i in 1:reps
        GC.gc()
        ts[i] = @elapsed f()
    end
    sort!(ts)
    return ts[cld(reps, 2)]
end

@inline _autotune_nemo_shapes(profile::Symbol=:full) =
    profile == :startup ?
    ((96, 96), (128, 128), (96, 192), (192, 96), (128, 320), (320, 128)) :
    ((96, 96), (128, 128), (160, 160), (192, 192),
     (96, 192), (192, 96), (128, 320), (320, 128), (128, 400), (400, 128))

@inline _autotune_fp_rank_shapes(profile::Symbol=:full) =
    profile == :startup ?
    ((64, 64), (96, 96), (64, 128), (128, 64)) :
    ((64, 64), (96, 96), (128, 128), (64, 128), (128, 64), (96, 192), (192, 96))

@inline _autotune_fp_nullspace_shapes(profile::Symbol=:full) =
    profile == :startup ?
    ((32, 32), (48, 48), (32, 64), (64, 32)) :
    ((32, 32), (48, 48), (64, 64), (32, 64), (64, 32), (48, 96), (96, 48))

@inline _autotune_fp_solve_shapes(profile::Symbol=:full) =
    profile == :startup ?
    ((72, 48), (96, 64), (120, 80), (96, 48), (144, 72)) :
    ((72, 48), (96, 64), (120, 80), (144, 96), (96, 48), (144, 72), (192, 96))

@inline _autotune_float_dense_shapes(profile::Symbol=:full) =
    profile == :startup ?
    ((96, 96), (128, 128), (96, 192), (192, 96)) :
    ((96, 96), (128, 128), (160, 160), (96, 192), (192, 96), (128, 256), (256, 128))

@inline _autotune_modular_nullspace_shapes(profile::Symbol=:full) =
    profile == :startup ?
    ((56, 56), (64, 96), (96, 64), (72, 72)) :
    ((56, 56), (72, 72), (88, 88), (104, 104), (64, 96), (96, 64), (80, 120), (120, 80))

@inline _autotune_modular_solve_shapes(profile::Symbol=:full) =
    profile == :startup ?
    ((60, 40), (84, 56), (108, 72), (128, 88)) :
    ((60, 40), (84, 56), (108, 72), (132, 88), (96, 48), (128, 64), (160, 80))

@inline _autotune_rankqq_dim_shapes(profile::Symbol=:full) =
    profile == :startup ?
    ((64, 64), (96, 96), (64, 128), (128, 64)) :
    ((64, 64), (96, 96), (128, 128), (160, 160), (64, 128), (128, 64), (96, 192), (192, 96))

@inline _autotune_float_sparse_dims() = (512, 1024)
@inline _autotune_float_sparse_densities() = (0.004, 0.01)
@inline _autotune_zn_dimat_probe_ncuts(profile::Symbol=:full) =
    profile == :startup ? (2, 4, 6) : (2, 4, 6, 8)
@inline _autotune_zn_dimat_threshold_candidates(profile::Symbol=:full) =
    profile == :startup ? (24, 32, 48, 64, 96) : (16, 24, 32, 48, 64, 96, 128, 192)
@inline _autotune_zn_dimat_queries_per_probe(profile::Symbol=:full) =
    profile == :startup ? 800 : 1_500
@inline _autotune_zn_dimat_reps(profile::Symbol=:full) = profile == :startup ? 1 : 2
@inline function _autotune_zn_dimat_steps(profile::Symbol)
    return 2 * length(_autotune_zn_dimat_probe_ncuts(profile)) *
           length(_autotune_zn_dimat_threshold_candidates(profile))
end

@inline _noop_progress_step(::AbstractString) = nothing

mutable struct _AutotuneProgress
    total::Int
    done::Int
    enabled::Bool
    last_pct::Int
end

@inline function _progress_bar(done::Int, total::Int; width::Int=24)
    denom = max(total, 1)
    frac = clamp(done / denom, 0.0, 1.0)
    filled = clamp(round(Int, width * frac), 0, width)
    return "[" * repeat("=", filled) * repeat(".", width - filled) * "]"
end

function _autotune_progress_init(total::Int; enabled::Bool)
    return _AutotuneProgress(max(total, 1), 0, enabled, -1)
end

function _autotune_progress_step!(st::_AutotuneProgress, label::AbstractString)
    st.done = min(st.total, st.done + 1)
    st.enabled || return nothing
    pct = clamp(floor(Int, 100 * st.done / st.total), 0, 99)
    if pct != st.last_pct || !isempty(label)
        st.last_pct = pct
        @info "FieldLinAlg autotune progress" percent=pct bar=_progress_bar(st.done, st.total) step=label
    end
    return nothing
end

function _autotune_progress_finish!(st::_AutotuneProgress)
    st.done = st.total
    st.enabled || return nothing
    @info "FieldLinAlg autotune progress" percent=100 bar=_progress_bar(st.done, st.total) step="complete"
    return nothing
end

function _autotune_modular_steps(profile::Symbol)
    nmax = max(0, min(length(DEFAULT_MODULAR_PRIMES), 8) - 1) # 2:min(...)
    nmin = min(4, max(1, MODULAR_MAX_PRIMES[]))
    n_ns = length(_autotune_modular_nullspace_shapes(profile))
    n_solve = length(_autotune_modular_solve_shapes(profile))
    n_rank = length(_autotune_rankqq_dim_shapes(profile))
    # rank-prime sweep + min-prime sweep + nullspace crossover + solve crossover + rank crossover
    return 3 * nmax + 2 * nmin + n_ns + n_solve + n_rank
end

function _autotune_total_steps(profile::Symbol)
    steps = 0
    if _have_nemo()
        steps += length(_autotune_fp_rank_shapes(profile)) +
                 length(_autotune_fp_nullspace_shapes(profile)) +
                 length(_autotune_fp_solve_shapes(profile))
    end
    steps += length(_autotune_float_dense_shapes(profile))
    if _have_svds_backend()
        steps += length(_autotune_float_sparse_dims()) * length(_autotune_float_sparse_densities())
    end
    if profile == :full
        steps += _autotune_modular_steps(profile)
        steps += _autotune_zn_dimat_steps(profile)
    end
    steps += _autotune_qq_routing_steps(profile)
    return max(steps, 1)
end

function _fp_dense_rand(field::PrimeField, m::Int, n::Int; rng=Random.default_rng())
    p = field.p
    p > 3 || error("_fp_dense_rand only for p > 3")
    A = Matrix{FpElem{p}}(undef, m, n)
    @inbounds for i in 1:m, j in 1:n
        A[i, j] = FpElem{p}(rand(rng, 0:(p - 1)))
    end
    return A
end

function _qq_dense_rand(m::Int, n::Int; rng=Random.default_rng())
    A = Matrix{QQ}(undef, m, n)
    @inbounds for i in 1:m, j in 1:n
        A[i, j] = QQ(rand(rng, -3:3))
    end
    return A
end

function _qq_dense_singular_rand(n::Int; rng=Random.default_rng())
    A = _qq_dense_rand(n, n; rng=rng)
    n >= 2 || return A
    @inbounds A[:, n] = A[:, 1]
    return A
end

function _qq_dense_fullcolumn_rand(m::Int, n::Int; rng=Random.default_rng())
    B = _qq_dense_rand(m, n; rng=rng)
    d = min(m, n)
    @inbounds for i in 1:d
        B[i, i] = QQ(1)
    end
    return B
end

@inline function _qq_nonzero_rand(rng::Random.AbstractRNG)
    v = rand(rng, -3:3)
    while v == 0
        v = rand(rng, -3:3)
    end
    return QQ(v)
end

function _qq_sparse_rand(m::Int, n::Int, density::Float64; rng=Random.default_rng())
    dens = clamp(density, 1e-6, 1.0)
    nnz_target = max(1, round(Int, m * n * dens))
    I = Vector{Int}(undef, nnz_target)
    J = Vector{Int}(undef, nnz_target)
    V = Vector{QQ}(undef, nnz_target)
    @inbounds for t in 1:nnz_target
        I[t] = rand(rng, 1:m)
        J[t] = rand(rng, 1:n)
        V[t] = _qq_nonzero_rand(rng)
    end
    return sparse(I, J, V, m, n)
end

function _qq_sparse_fullcolumn_rand(m::Int, n::Int, density::Float64; rng=Random.default_rng())
    m >= n || error("_qq_sparse_fullcolumn_rand: requires m >= n")
    dens = clamp(density, 1e-6, 1.0)
    I = Int[]
    J = Int[]
    V = QQ[]
    sizehint!(I, n + round(Int, (m - n) * n * dens))
    sizehint!(J, n + round(Int, (m - n) * n * dens))
    sizehint!(V, n + round(Int, (m - n) * n * dens))
    @inbounds for i in 1:n
        push!(I, i)
        push!(J, i)
        push!(V, QQ(1))
    end
    @inbounds for i in (n + 1):m, j in 1:n
        rand(rng) < dens || continue
        push!(I, i)
        push!(J, j)
        push!(V, _qq_nonzero_rand(rng))
    end
    return sparse(I, J, V, m, n)
end

function _float_sparse_singular_rand(n::Int, density::Float64; rng=Random.default_rng())
    A = sprand(rng, Float64, n, n, density)
    if n >= 2
        A[:, n] = A[:, 1]
    end
    if nnz(A) == 0
        A[1, 1] = 1.0
        if n >= 2
            A[1, n] = 1.0
        end
    end
    return A
end

function _pick_crossover_threshold(works::Vector{Int}, tj::Vector{Float64}, tf::Vector{Float64}, default::Int;
                                   speedup_ratio::Float64=0.9)
    ord = sortperm(works)
    @inbounds for oi in ord
        if tf[oi] < speedup_ratio * tj[oi]
            return works[oi]
        end
    end
    return default
end

@inline function _find_crossover_pos(works::Vector{Int}, tj::Vector{Float64}, tf::Vector{Float64};
                                     speedup_ratio::Float64=0.9)
    ord = sortperm(works)
    @inbounds for (pos, oi) in enumerate(ord)
        if tf[oi] < speedup_ratio * tj[oi]
            return pos, ord
        end
    end
    return nothing, ord
end

function _refine_crossover_boundary!(works::Vector{Int},
                                     tj::Vector{Float64},
                                     tf::Vector{Float64},
                                     bench_pair::Function;
                                     speedup_ratio::Float64=0.9,
                                     extra_reps::Int=4,
                                     window::Int=1)
    pos, ord = _find_crossover_pos(works, tj, tf; speedup_ratio=speedup_ratio)
    pos === nothing && return nothing
    lo = max(1, pos - window)
    hi = min(length(ord), pos + window)
    @inbounds for q in lo:hi
        i = ord[q]
        tj_i, tf_i = bench_pair(i, extra_reps)
        tj[i] = tj_i
        tf[i] = tf_i
    end
    return nothing
end

@inline _threshold_regret(pred::Float64, best::Float64, margin::Float64) =
    max(0.0, pred / max(best, eps(Float64)) - (1.0 + margin))

function _smooth_by_work(works::Vector{Int}, vals::Vector{Float64}; radius::Int=1)
    n = length(works)
    ord = sortperm(works)
    out = similar(vals)
    @inbounds for p in 1:n
        lo = max(1, p - radius)
        hi = min(n, p + radius)
        localv = Vector{Float64}(undef, hi - lo + 1)
        t = 1
        for q in lo:hi
            localv[t] = vals[ord[q]]
            t += 1
        end
        out[ord[p]] = median(localv)
    end
    return out
end

function _nearest_indices(works::Vector{Int}, target::Int, k::Int)
    n = length(works)
    n == 0 && return Int[]
    ord = sortperm(abs.(works .- target))
    return ord[1:min(k, n)]
end

function _fit_single_threshold(works::Vector{Int}, t_exact::Vector{Float64}, t_other::Vector{Float64};
                               margin::Float64=0.15)
    cands = sort(unique(vcat(works, maximum(works) + 1)))
    best_thr = cands[end]
    best_loss = Inf
    best_mismatch = typemax(Int)
    @inline label(i) = (t_other[i] <= (1.0 - margin) * t_exact[i]) ? :other : :exact
    @inbounds for thr in cands
        loss = 0.0
        mismatch = 0
        for i in eachindex(works)
            pred_label = works[i] >= thr ? :other : :exact
            mismatch += pred_label == label(i) ? 0 : 1
            pred = pred_label == :other ? t_other[i] : t_exact[i]
            best = min(t_exact[i], t_other[i])
            loss += _threshold_regret(pred, best, margin)
        end
        if mismatch < best_mismatch || (mismatch == best_mismatch && loss < best_loss - 1e-12)
            best_loss = loss
            best_mismatch = mismatch
            best_thr = thr
        end
    end
    return best_thr, best_loss, best_mismatch
end

function _fit_double_threshold(works::Vector{Int},
                               t_exact::Vector{Float64},
                               t_mod::Vector{Float64},
                               t_nemo::Vector{Float64};
                               margin::Float64=0.15)
    cands = sort(unique(vcat(works, maximum(works) + 1)))
    best_mod = cands[end]
    best_nemo = cands[end]
    best_loss = Inf
    best_mismatch = typemax(Int)
    @inline function label(i)
        if t_nemo[i] <= (1.0 - margin) * min(t_exact[i], t_mod[i])
            return :nemo
        elseif t_mod[i] <= (1.0 - margin) * t_exact[i]
            return :modular
        else
            return :exact
        end
    end
    @inbounds for thr_mod in cands
        for thr_nemo in cands
            thr_nemo < thr_mod && continue
            loss = 0.0
            mismatch = 0
            for i in eachindex(works)
                pred_label = works[i] >= thr_nemo ? :nemo : (works[i] >= thr_mod ? :modular : :exact)
                mismatch += pred_label == label(i) ? 0 : 1
                pred = pred_label == :nemo ? t_nemo[i] : (pred_label == :modular ? t_mod[i] : t_exact[i])
                best = min(t_exact[i], min(t_mod[i], t_nemo[i]))
                loss += _threshold_regret(pred, best, margin)
            end
            if mismatch < best_mismatch || (mismatch == best_mismatch && loss < best_loss - 1e-12)
                best_loss = loss
                best_mismatch = mismatch
                best_mod = thr_mod
                best_nemo = thr_nemo
            end
        end
    end
    return best_mod, best_nemo, best_loss, best_mismatch
end

function _eval_single_threshold(works::Vector{Int}, t_exact::Vector{Float64}, t_other::Vector{Float64},
                                thr::Int; margin::Float64=0.15)
    loss = 0.0
    mismatch = 0
    @inline label(i) = (t_other[i] <= (1.0 - margin) * t_exact[i]) ? :other : :exact
    @inbounds for i in eachindex(works)
        pred_label = works[i] >= thr ? :other : :exact
        mismatch += pred_label == label(i) ? 0 : 1
        pred = pred_label == :other ? t_other[i] : t_exact[i]
        best = min(t_exact[i], t_other[i])
        loss += _threshold_regret(pred, best, margin)
    end
    return loss, mismatch
end

function _fit_single_threshold_le(works::Vector{Int}, t_exact::Vector{Float64}, t_other::Vector{Float64};
                                  margin::Float64=0.15)
    cands = sort(unique(vcat(works, minimum(works) - 1)))
    best_thr = cands[1]
    best_loss = Inf
    best_mismatch = typemax(Int)
    @inline label(i) = (t_other[i] <= (1.0 - margin) * t_exact[i]) ? :other : :exact
    @inbounds for thr in cands
        loss = 0.0
        mismatch = 0
        for i in eachindex(works)
            pred_label = works[i] <= thr ? :other : :exact
            mismatch += pred_label == label(i) ? 0 : 1
            pred = pred_label == :other ? t_other[i] : t_exact[i]
            best = min(t_exact[i], t_other[i])
            loss += _threshold_regret(pred, best, margin)
        end
        if mismatch < best_mismatch || (mismatch == best_mismatch && loss < best_loss - 1e-12)
            best_loss = loss
            best_mismatch = mismatch
            best_thr = thr
        end
    end
    return best_thr, best_loss, best_mismatch
end

function _eval_single_threshold_le(works::Vector{Int}, t_exact::Vector{Float64}, t_other::Vector{Float64},
                                   thr::Int; margin::Float64=0.15)
    loss = 0.0
    mismatch = 0
    @inline label(i) = (t_other[i] <= (1.0 - margin) * t_exact[i]) ? :other : :exact
    @inbounds for i in eachindex(works)
        pred_label = works[i] <= thr ? :other : :exact
        mismatch += pred_label == label(i) ? 0 : 1
        pred = pred_label == :other ? t_other[i] : t_exact[i]
        best = min(t_exact[i], t_other[i])
        loss += _threshold_regret(pred, best, margin)
    end
    return loss, mismatch
end

function _eval_double_threshold(works::Vector{Int}, t_exact::Vector{Float64}, t_mod::Vector{Float64}, t_nemo::Vector{Float64},
                                thr_mod::Int, thr_nemo::Int; margin::Float64=0.15)
    loss = 0.0
    mismatch = 0
    @inline function label(i)
        if t_nemo[i] <= (1.0 - margin) * min(t_exact[i], t_mod[i])
            return :nemo
        elseif t_mod[i] <= (1.0 - margin) * t_exact[i]
            return :modular
        else
            return :exact
        end
    end
    @inbounds for i in eachindex(works)
        pred_label = works[i] >= thr_nemo ? :nemo : (works[i] >= thr_mod ? :modular : :exact)
        mismatch += pred_label == label(i) ? 0 : 1
        pred = pred_label == :nemo ? t_nemo[i] : (pred_label == :modular ? t_mod[i] : t_exact[i])
        best = min(t_exact[i], min(t_mod[i], t_nemo[i]))
        loss += _threshold_regret(pred, best, margin)
    end
    return loss, mismatch
end

@inline function _accept_threshold_update(new_loss::Float64, new_mismatch::Int,
                                          old_loss::Float64, old_mismatch::Int)
    if !isfinite(old_loss)
        return isfinite(new_loss)
    end
    if new_loss <= old_loss * 0.99
        return true
    end
    if new_loss <= old_loss * 1.01 && new_mismatch < old_mismatch
        return true
    end
    return false
end

function _qq_probe_shapes(op::Symbol, shape::Symbol, profile::Symbol, holdout::Bool)
    if op == :rank
        if shape == :square
            return holdout ?
                (profile == :startup ? ((28, 28), (40, 40), (72, 72), (104, 104)) :
                                       ((24, 24), (36, 36), (56, 56), (72, 72), (88, 88), (104, 104), (120, 120))) :
                (profile == :startup ? ((24, 24), (40, 40), (64, 64), (96, 96), (128, 128)) :
                                       ((20, 20), (28, 28), (48, 48), (64, 64), (80, 80), (96, 96), (112, 112), (128, 128)))
        elseif shape == :tall
            return holdout ?
                (profile == :startup ? ((40, 20), (72, 36), (112, 56), (152, 76)) :
                                       ((36, 18), (52, 26), (104, 52), (120, 60), (136, 68), (152, 76), (176, 88))) :
                (profile == :startup ? ((32, 16), (64, 32), (96, 48), (128, 64), (160, 80)) :
                                       ((28, 14), (44, 22), (88, 44), (112, 56), (128, 64), (144, 72), (160, 80), (192, 96)))
        else
            return holdout ?
                (profile == :startup ? ((20, 40), (36, 72), (56, 112), (76, 152)) :
                                       ((18, 36), (26, 52), (52, 104), (60, 120), (68, 136), (76, 152), (88, 176))) :
                (profile == :startup ? ((16, 32), (32, 64), (48, 96), (64, 128), (80, 160)) :
                                       ((14, 28), (22, 44), (44, 88), (56, 112), (64, 128), (72, 144), (80, 160), (96, 192)))
        end
    elseif op == :nullspace
        if shape == :square
            return holdout ?
                (profile == :startup ? ((24, 24), (40, 40), (56, 56), (88, 88)) :
                                       ((20, 20), (32, 32), (48, 48), (64, 64), (80, 80), (96, 96), (112, 112))) :
                (profile == :startup ? ((20, 20), (32, 32), (48, 48), (72, 72), (96, 96)) :
                                       ((16, 16), (24, 24), (40, 40), (56, 56), (72, 72), (88, 88), (104, 104)))
        elseif shape == :tall
            return holdout ?
                (profile == :startup ? ((48, 24), (72, 36), (112, 56), (144, 72)) :
                                       ((40, 20), (56, 28), (96, 48), (112, 56), (128, 64), (144, 72), (160, 80))) :
                (profile == :startup ? ((40, 20), (64, 32), (96, 48), (128, 64), (160, 80)) :
                                       ((32, 16), (48, 24), (80, 40), (96, 48), (112, 56), (128, 64), (144, 72)))
        else
            return holdout ?
                (profile == :startup ? ((24, 48), (36, 72), (56, 112), (72, 144)) :
                                       ((20, 40), (28, 56), (48, 96), (56, 112), (64, 128), (72, 144), (80, 160))) :
                (profile == :startup ? ((20, 40), (32, 64), (48, 96), (64, 128), (80, 160)) :
                                       ((16, 32), (24, 48), (40, 80), (48, 96), (56, 112), (64, 128), (72, 144)))
        end
    elseif op == :solve
        if shape == :square
            return holdout ?
                (profile == :startup ? ((32, 32), (48, 48), (64, 64), (88, 88)) :
                                       ((28, 28), (40, 40), (56, 56), (72, 72), (88, 88), (104, 104), (120, 120))) :
                (profile == :startup ? ((24, 24), (40, 40), (56, 56), (80, 80), (104, 104)) :
                                       ((20, 20), (32, 32), (48, 48), (64, 64), (80, 80), (96, 96), (112, 112)))
        elseif shape == :tall
            return holdout ?
                (profile == :startup ? ((42, 28), (72, 48), (96, 64), (132, 88)) :
                                       ((36, 24), (54, 36), (84, 56), (108, 72), (132, 88), (156, 104), (180, 120))) :
                (profile == :startup ? ((36, 24), (60, 40), (84, 56), (108, 72), (132, 88)) :
                                       ((30, 20), (48, 32), (72, 48), (96, 64), (120, 80), (144, 96), (168, 112)))
        else
            # Full-column solve naturally uses m>=n; use square probes for wide bucket fallback.
            return _qq_probe_shapes(op, :square, profile, holdout)
        end
    end
    return Tuple{Int,Int}[]
end

@inline _qq_sparse_probe_densities(profile::Symbol) =
    profile == :startup ? (0.2,) : (0.2, 0.05)

function _qq_case(op::Symbol, m::Int, n::Int, rng::Random.AbstractRNG;
                  sparse::Bool=false, density::Float64=0.2)
    if op == :rank
        return sparse ? _qq_sparse_rand(m, n, density; rng=rng) : _qq_dense_rand(m, n; rng=rng)
    elseif op == :nullspace
        A = sparse ? _qq_sparse_rand(m, n, density; rng=rng) : _qq_dense_rand(m, n; rng=rng)
        min(m, n) >= 2 && (A[:, end] = A[:, 1])
        return A
    elseif op == :solve
        m >= n || error("solve probe requires m >= n")
        B = sparse ? _qq_sparse_fullcolumn_rand(m, n, density; rng=rng) : _qq_dense_fullcolumn_rand(m, n; rng=rng)
        X = _qq_dense_rand(n, 2; rng=rng)
        return (B, B * X)
    end
    error("unsupported QQ probe op: $(op)")
end

function _bench_qq_probe(op::Symbol, probe, backend::Symbol; reps::Int=1)
    F = QQField()
    try
        if op == :rank
            A = probe::AbstractMatrix{QQ}
            exact_backend = _is_sparse_like(A) ? :julia_sparse : :julia_exact
            if backend == :julia_exact
                return _bench_elapsed(() -> rank(F, A; backend=exact_backend); reps=reps)
            elseif backend == :nemo
                _have_nemo() || return Inf
                return _bench_elapsed(() -> rank(F, A; backend=:nemo); reps=reps)
            elseif backend == :modular
                return Inf
            end
        elseif op == :nullspace
            A = probe::AbstractMatrix{QQ}
            exact_backend = _is_sparse_like(A) ? :julia_sparse : :julia_exact
            if backend == :julia_exact
                return _bench_elapsed(() -> nullspace(F, A; backend=exact_backend); reps=reps)
            elseif backend == :nemo
                _have_nemo() || return Inf
                return _bench_elapsed(() -> nullspace(F, A; backend=:nemo); reps=reps)
            elseif backend == :modular
                _is_sparse_like(A) && return Inf
                return _bench_elapsed(() -> nullspace(F, A; backend=:modular); reps=reps)
            end
        elseif op == :solve
            B, Y = probe::Tuple{AbstractMatrix{QQ},Matrix{QQ}}
            exact_backend = _is_sparse_like(B) ? :julia_sparse : :julia_exact
            if backend == :julia_exact
                return _bench_elapsed(() -> solve_fullcolumn(F, B, Y; backend=exact_backend, check_rhs=false, cache=true); reps=reps)
            elseif backend == :nemo
                _have_nemo() || return Inf
                return _bench_elapsed(() -> solve_fullcolumn(F, B, Y; backend=:nemo, check_rhs=false, cache=true); reps=reps)
            elseif backend == :modular
                _is_sparse_like(B) && return Inf
                return _bench_elapsed(() -> solve_fullcolumn(F, B, Y; backend=:modular, check_rhs=false, cache=true); reps=reps)
            end
        end
    catch
        return Inf
    end
    return Inf
end

function _set_qq_threshold!(op::Symbol, shape::Symbol, backend::Symbol, value::Int)
    v = max(1, Int(value))
    if backend == :nemo && op == :rank
        if shape == :square
            QQ_NEMO_RANK_THRESHOLD_SQUARE[] = v
        elseif shape == :tall
            QQ_NEMO_RANK_THRESHOLD_TALL[] = v
        else
            QQ_NEMO_RANK_THRESHOLD_WIDE[] = v
        end
    elseif backend == :nemo && op == :nullspace
        if shape == :square
            QQ_NEMO_NULLSPACE_THRESHOLD_SQUARE[] = v
        elseif shape == :tall
            QQ_NEMO_NULLSPACE_THRESHOLD_TALL[] = v
        else
            QQ_NEMO_NULLSPACE_THRESHOLD_WIDE[] = v
        end
    elseif backend == :nemo && op == :solve
        if shape == :square
            QQ_NEMO_SOLVE_THRESHOLD_SQUARE[] = v
        elseif shape == :tall
            QQ_NEMO_SOLVE_THRESHOLD_TALL[] = v
        else
            QQ_NEMO_SOLVE_THRESHOLD_WIDE[] = v
        end
    elseif backend == :modular && op == :nullspace
        if shape == :square
            QQ_MODULAR_NULLSPACE_THRESHOLD_SQUARE[] = v
        elseif shape == :tall
            QQ_MODULAR_NULLSPACE_THRESHOLD_TALL[] = v
        else
            QQ_MODULAR_NULLSPACE_THRESHOLD_WIDE[] = v
        end
    elseif backend == :modular && op == :solve
        if shape == :square
            QQ_MODULAR_SOLVE_THRESHOLD_SQUARE[] = v
        elseif shape == :tall
            QQ_MODULAR_SOLVE_THRESHOLD_TALL[] = v
        else
            QQ_MODULAR_SOLVE_THRESHOLD_WIDE[] = v
        end
    end
    return nothing
end

@inline function _get_qq_threshold(op::Symbol, shape::Symbol, backend::Symbol)
    return backend == :nemo ? _qq_nemo_threshold(op, shape) : _qq_modular_threshold(op, shape)
end

function _qq_probe_cases(op::Symbol, shape::Symbol, profile::Symbol, holdout::Bool)
    shapes = collect(_qq_probe_shapes(op, shape, profile, holdout))
    if holdout && profile == :startup && length(shapes) > 2
        shapes = [shapes[1], shapes[end]]
    end
    cases = NamedTuple{(:m, :n, :sparse, :density),Tuple{Int, Int, Bool, Float64}}[]
    dens_train = _qq_sparse_probe_densities(profile)
    dens_hold = (dens_train[1],)
    sparse_dens = holdout ? dens_hold : dens_train
    sparse_work_min = profile == :startup ? 900 : 600
    for (m, n) in shapes
        push!(cases, (m=m, n=n, sparse=false, density=1.0))
        if op == :rank && m * n >= sparse_work_min
            for dens in sparse_dens
                push!(cases, (m=m, n=n, sparse=true, density=dens))
            end
        end
    end
    return cases
end

function _autotune_qq_shape_thresholds!(op::Symbol, shape::Symbol, profile::Symbol, rng::Random.AbstractRNG,
                                        progress_step::Function=_noop_progress_step)
    train_cases = _qq_probe_cases(op, shape, profile, false)
    hold_cases = _qq_probe_cases(op, shape, profile, true)
    isempty(train_cases) && return

    ntrain = length(train_cases)
    train_work = Vector{Int}(undef, ntrain)
    train_probe = Vector{Any}(undef, ntrain)
    t_exact = Vector{Float64}(undef, ntrain)
    t_mod = fill(Inf, ntrain)
    t_nemo = fill(Inf, ntrain)
    for (i, case) in enumerate(train_cases)
        m = case.m
        n = case.n
        probe = _qq_case(op, m, n, rng; sparse=case.sparse, density=case.density)
        train_probe[i] = probe
        train_work[i] = m * n
        t_exact[i] = _bench_qq_probe(op, probe, :julia_exact; reps=1)
        if op != :rank
            t_mod[i] = _bench_qq_probe(op, probe, :modular; reps=1)
        end
        t_nemo[i] = _bench_qq_probe(op, probe, :nemo; reps=1)
        progress_step("qq-$(op)-$(shape) train $(i)/$(ntrain)")
    end

    s_exact = _smooth_by_work(train_work, t_exact)
    s_mod = _smooth_by_work(train_work, t_mod)
    s_nemo = _smooth_by_work(train_work, t_nemo)
    margin = 0.15
    nnear = profile == :full ? 4 : 2
    reps_ref = profile == :full ? 4 : 2
    if op == :rank
        thr_nemo, _, _ = _fit_single_threshold(train_work, s_exact, s_nemo; margin=margin)
        for idx in _nearest_indices(train_work, thr_nemo, nnear)
            p = train_probe[idx]
            t_exact[idx] = _bench_qq_probe(op, p, :julia_exact; reps=reps_ref)
            t_nemo[idx] = _bench_qq_probe(op, p, :nemo; reps=reps_ref)
        end
        s_exact = _smooth_by_work(train_work, t_exact)
        s_nemo = _smooth_by_work(train_work, t_nemo)
        thr_nemo, _, _ = _fit_single_threshold(train_work, s_exact, s_nemo; margin=margin)

        # Holdout guard: keep incumbent if new threshold is worse.
        nh = length(hold_cases)
        h_work = Vector{Int}(undef, nh)
        h_e = Vector{Float64}(undef, nh)
        h_n = Vector{Float64}(undef, nh)
        for (i, case) in enumerate(hold_cases)
            m = case.m
            n = case.n
            p = _qq_case(op, m, n, rng; sparse=case.sparse, density=case.density)
            h_work[i] = m * n
            h_e[i] = _bench_qq_probe(op, p, :julia_exact; reps=1)
            h_n[i] = _bench_qq_probe(op, p, :nemo; reps=1)
            progress_step("qq-$(op)-$(shape) holdout $(i)/$(nh)")
        end
        old_thr = _get_qq_threshold(op, shape, :nemo)
        old_loss, old_mismatch = _eval_single_threshold(h_work, h_e, h_n, old_thr; margin=margin)
        new_loss, new_mismatch = _eval_single_threshold(h_work, h_e, h_n, thr_nemo; margin=margin)
        accept = _accept_threshold_update(new_loss, new_mismatch, old_loss, old_mismatch) ||
                 (new_mismatch == old_mismatch && new_loss <= old_loss * 1.01 && thr_nemo < old_thr)
        _set_qq_threshold!(op, shape, :nemo, accept ? thr_nemo : old_thr)
    else
        thr_mod, thr_nemo, _, _ = _fit_double_threshold(train_work, s_exact, s_mod, s_nemo; margin=margin)
        for idx in union(_nearest_indices(train_work, thr_mod, nnear), _nearest_indices(train_work, thr_nemo, nnear))
            p = train_probe[idx]
            t_exact[idx] = _bench_qq_probe(op, p, :julia_exact; reps=reps_ref)
            t_mod[idx] = _bench_qq_probe(op, p, :modular; reps=reps_ref)
            t_nemo[idx] = _bench_qq_probe(op, p, :nemo; reps=reps_ref)
        end
        s_exact = _smooth_by_work(train_work, t_exact)
        s_mod = _smooth_by_work(train_work, t_mod)
        s_nemo = _smooth_by_work(train_work, t_nemo)
        thr_mod, thr_nemo, _, _ = _fit_double_threshold(train_work, s_exact, s_mod, s_nemo; margin=margin)

        nh = length(hold_cases)
        h_work = Vector{Int}(undef, nh)
        h_e = Vector{Float64}(undef, nh)
        h_m = Vector{Float64}(undef, nh)
        h_n = Vector{Float64}(undef, nh)
        for (i, case) in enumerate(hold_cases)
            m = case.m
            n = case.n
            p = _qq_case(op, m, n, rng; sparse=case.sparse, density=case.density)
            h_work[i] = m * n
            h_e[i] = _bench_qq_probe(op, p, :julia_exact; reps=1)
            h_m[i] = _bench_qq_probe(op, p, :modular; reps=1)
            h_n[i] = _bench_qq_probe(op, p, :nemo; reps=1)
            progress_step("qq-$(op)-$(shape) holdout $(i)/$(nh)")
        end
        old_mod = _get_qq_threshold(op, shape, :modular)
        old_nemo = _get_qq_threshold(op, shape, :nemo)
        old_loss, old_mismatch = _eval_double_threshold(h_work, h_e, h_m, h_n, old_mod, old_nemo; margin=margin)
        new_loss, new_mismatch = _eval_double_threshold(h_work, h_e, h_m, h_n, thr_mod, thr_nemo; margin=margin)
        accept = _accept_threshold_update(new_loss, new_mismatch, old_loss, old_mismatch) ||
                 (new_mismatch == old_mismatch && new_loss <= old_loss * 1.01 &&
                  thr_mod <= old_mod && thr_nemo <= old_nemo &&
                  (thr_mod < old_mod || thr_nemo < old_nemo))
        if accept
            _set_qq_threshold!(op, shape, :modular, thr_mod)
            _set_qq_threshold!(op, shape, :nemo, thr_nemo)
        else
            _set_qq_threshold!(op, shape, :modular, old_mod)
            _set_qq_threshold!(op, shape, :nemo, old_nemo)
        end
    end
    return nothing
end

@inline function _set_qq_sparse_solve_threshold!(shape::Symbol, bucket::Symbol, value::Int)
    v = max(1, Int(value))
    if shape == :square
        if bucket == :low
            QQ_NEMO_SPARSE_SOLVE_THRESHOLD_SQUARE_LOW[] = v
        elseif bucket == :mid
            QQ_NEMO_SPARSE_SOLVE_THRESHOLD_SQUARE_MID[] = v
        else
            QQ_NEMO_SPARSE_SOLVE_THRESHOLD_SQUARE_HIGH[] = v
        end
    elseif shape == :tall
        if bucket == :low
            QQ_NEMO_SPARSE_SOLVE_THRESHOLD_TALL_LOW[] = v
        elseif bucket == :mid
            QQ_NEMO_SPARSE_SOLVE_THRESHOLD_TALL_MID[] = v
        else
            QQ_NEMO_SPARSE_SOLVE_THRESHOLD_TALL_HIGH[] = v
        end
    else
        if bucket == :low
            QQ_NEMO_SPARSE_SOLVE_THRESHOLD_WIDE_LOW[] = v
        elseif bucket == :mid
            QQ_NEMO_SPARSE_SOLVE_THRESHOLD_WIDE_MID[] = v
        else
            QQ_NEMO_SPARSE_SOLVE_THRESHOLD_WIDE_HIGH[] = v
        end
    end
    return nothing
end

@inline _get_qq_sparse_solve_threshold(shape::Symbol, bucket::Symbol) = _qq_sparse_solve_threshold(shape, bucket)

@inline function _set_qq_sparse_solve_policy!(shape::Symbol, bucket::Symbol, use_ge::Bool)
    v = use_ge ? 1 : -1
    if shape == :square
        if bucket == :low
            QQ_NEMO_SPARSE_SOLVE_POLICY_SQUARE_LOW[] = v
        elseif bucket == :mid
            QQ_NEMO_SPARSE_SOLVE_POLICY_SQUARE_MID[] = v
        else
            QQ_NEMO_SPARSE_SOLVE_POLICY_SQUARE_HIGH[] = v
        end
    elseif shape == :tall
        if bucket == :low
            QQ_NEMO_SPARSE_SOLVE_POLICY_TALL_LOW[] = v
        elseif bucket == :mid
            QQ_NEMO_SPARSE_SOLVE_POLICY_TALL_MID[] = v
        else
            QQ_NEMO_SPARSE_SOLVE_POLICY_TALL_HIGH[] = v
        end
    else
        if bucket == :low
            QQ_NEMO_SPARSE_SOLVE_POLICY_WIDE_LOW[] = v
        elseif bucket == :mid
            QQ_NEMO_SPARSE_SOLVE_POLICY_WIDE_MID[] = v
        else
            QQ_NEMO_SPARSE_SOLVE_POLICY_WIDE_HIGH[] = v
        end
    end
    return nothing
end

@inline _get_qq_sparse_solve_policy(shape::Symbol, bucket::Symbol) = _qq_sparse_solve_use_ge(shape, bucket)

@inline function _qq_sparse_solve_shapes(shape::Symbol, profile::Symbol, holdout::Bool)
    if shape == :square
        return holdout ?
            (profile == :startup ? ((48, 48), (72, 72), (88, 88)) :
                                   ((40, 40), (56, 56), (72, 72), (88, 88), (104, 104))) :
            (profile == :startup ? ((40, 40), (56, 56), (64, 64), (80, 80)) :
                                   ((32, 32), (48, 48), (64, 64), (80, 80), (96, 96)))
    elseif shape == :tall
        return holdout ?
            (profile == :startup ? ((72, 48), (90, 60), (96, 72), (132, 88)) :
                                   ((60, 40), (84, 56), (96, 72), (108, 72), (132, 88), (156, 104))) :
            (profile == :startup ? ((60, 40), (84, 56), (96, 72), (108, 72), (120, 80)) :
                                   ((48, 32), (72, 48), (96, 64), (108, 72), (120, 80), (144, 96)))
    end
    # Full-column solve is naturally square/tall; wide fallback mirrors square.
    return _qq_sparse_solve_shapes(:square, profile, holdout)
end

@inline function _qq_sparse_solve_probe_cases(shape::Symbol, bucket::Symbol, profile::Symbol, holdout::Bool)
    dens = bucket == :low ? 0.05 : (bucket == :mid ? 0.2 : 0.45)
    shapes = collect(_qq_sparse_solve_shapes(shape, profile, holdout))
    return [(m=s[1], n=s[2], density=dens) for s in shapes]
end

function _autotune_qq_sparse_solve_thresholds!(shape::Symbol, bucket::Symbol, profile::Symbol,
                                               rng::Random.AbstractRNG, progress_step::Function=_noop_progress_step)
    train_cases = _qq_sparse_solve_probe_cases(shape, bucket, profile, false)
    hold_cases = _qq_sparse_solve_probe_cases(shape, bucket, profile, true)
    isempty(train_cases) && return

    ntrain = length(train_cases)
    train_nnz = Vector{Int}(undef, ntrain)
    train_probe = Vector{Any}(undef, ntrain)
    t_exact = Vector{Float64}(undef, ntrain)
    t_nemo = Vector{Float64}(undef, ntrain)
    for (i, case) in enumerate(train_cases)
        B = _qq_sparse_fullcolumn_rand(case.m, case.n, case.density; rng=rng)
        X = _qq_dense_rand(case.n, 4; rng=rng)
        Y = B * X
        probe = (B, Y)
        train_probe[i] = probe
        train_nnz[i] = _sparse_nnz(B)
        t_exact[i] = _bench_qq_probe(:solve, probe, :julia_exact; reps=1)
        t_nemo[i] = _bench_qq_probe(:solve, probe, :nemo; reps=1)
        progress_step("qq-solve-sparse-$(shape)-$(bucket) train $(i)/$(ntrain)")
    end

    s_exact = _smooth_by_work(train_nnz, t_exact)
    s_nemo = _smooth_by_work(train_nnz, t_nemo)
    margin = 0.05
    nnear = profile == :full ? 4 : 2
    reps_ref = profile == :full ? 4 : 2
    thr_ge, _, _ = _fit_single_threshold(train_nnz, s_exact, s_nemo; margin=margin)
    thr_le, _, _ = _fit_single_threshold_le(train_nnz, s_exact, s_nemo; margin=margin)
    for idx in union(_nearest_indices(train_nnz, thr_ge, nnear), _nearest_indices(train_nnz, thr_le, nnear))
        p = train_probe[idx]
        t_exact[idx] = _bench_qq_probe(:solve, p, :julia_exact; reps=reps_ref)
        t_nemo[idx] = _bench_qq_probe(:solve, p, :nemo; reps=reps_ref)
    end
    s_exact = _smooth_by_work(train_nnz, t_exact)
    s_nemo = _smooth_by_work(train_nnz, t_nemo)
    thr_ge, _, _ = _fit_single_threshold(train_nnz, s_exact, s_nemo; margin=margin)
    thr_le, _, _ = _fit_single_threshold_le(train_nnz, s_exact, s_nemo; margin=margin)

    nh = length(hold_cases)
    h_nnz = Vector{Int}(undef, nh)
    h_e = Vector{Float64}(undef, nh)
    h_n = Vector{Float64}(undef, nh)
    for (i, case) in enumerate(hold_cases)
        B = _qq_sparse_fullcolumn_rand(case.m, case.n, case.density; rng=rng)
        X = _qq_dense_rand(case.n, 4; rng=rng)
        Y = B * X
        probe = (B, Y)
        h_nnz[i] = _sparse_nnz(B)
        h_e[i] = _bench_qq_probe(:solve, probe, :julia_exact; reps=1)
        h_n[i] = _bench_qq_probe(:solve, probe, :nemo; reps=1)
        progress_step("qq-solve-sparse-$(shape)-$(bucket) holdout $(i)/$(nh)")
    end

    ge_loss, ge_mismatch = _eval_single_threshold(h_nnz, h_e, h_n, thr_ge; margin=margin)
    le_loss, le_mismatch = _eval_single_threshold_le(h_nnz, h_e, h_n, thr_le; margin=margin)
    cand_use_ge = ge_loss < le_loss - 1e-9 || (abs(ge_loss - le_loss) <= 1e-9 && ge_mismatch <= le_mismatch)
    cand_thr = cand_use_ge ? thr_ge : thr_le
    cand_loss = cand_use_ge ? ge_loss : le_loss
    cand_mismatch = cand_use_ge ? ge_mismatch : le_mismatch

    old_thr = _get_qq_sparse_solve_threshold(shape, bucket)
    old_use_ge = _get_qq_sparse_solve_policy(shape, bucket)
    if old_use_ge
        old_loss, old_mismatch = _eval_single_threshold(h_nnz, h_e, h_n, old_thr; margin=margin)
        accept = _accept_threshold_update(cand_loss, cand_mismatch, old_loss, old_mismatch) ||
                 (cand_mismatch == old_mismatch && cand_loss <= old_loss * 1.01 &&
                  ((cand_use_ge && cand_thr < old_thr) || (!cand_use_ge)))
        if accept
            _set_qq_sparse_solve_threshold!(shape, bucket, cand_thr)
            _set_qq_sparse_solve_policy!(shape, bucket, cand_use_ge)
        end
    else
        old_loss, old_mismatch = _eval_single_threshold_le(h_nnz, h_e, h_n, old_thr; margin=margin)
        accept = _accept_threshold_update(cand_loss, cand_mismatch, old_loss, old_mismatch) ||
                 (cand_mismatch == old_mismatch && cand_loss <= old_loss * 1.01 &&
                  ((!cand_use_ge && cand_thr > old_thr) || cand_use_ge))
        if accept
            _set_qq_sparse_solve_threshold!(shape, bucket, cand_thr)
            _set_qq_sparse_solve_policy!(shape, bucket, cand_use_ge)
        end
    end
    return nothing
end

function _autotune_qq_sparse_solve_steps(profile::Symbol)
    steps = 0
    for shape in (:square, :tall), bucket in (:low, :mid, :high)
        steps += length(_qq_sparse_solve_probe_cases(shape, bucket, profile, false))
        steps += length(_qq_sparse_solve_probe_cases(shape, bucket, profile, true))
    end
    return steps
end

function _autotune_qq_routing_thresholds!(profile::Symbol=:full, progress_step::Function=_noop_progress_step)
    rng = Random.MersenneTwister(0x51515151)
    for shape in (:square, :tall, :wide)
        _autotune_qq_shape_thresholds!(:rank, shape, profile, rng, progress_step)
    end
    for shape in (:square, :tall, :wide)
        _autotune_qq_shape_thresholds!(:nullspace, shape, profile, rng, progress_step)
    end
    for shape in (:square, :tall, :wide)
        _autotune_qq_shape_thresholds!(:solve, shape, profile, rng, progress_step)
    end
    for shape in (:square, :tall), bucket in (:low, :mid, :high)
        _autotune_qq_sparse_solve_thresholds!(shape, bucket, profile, rng, progress_step)
    end
    # Mirror wide thresholds to square defaults for compatibility in rare wide-solve calls.
    for bucket in (:low, :mid, :high)
        _set_qq_sparse_solve_threshold!(:wide, bucket, _get_qq_sparse_solve_threshold(:square, bucket))
        _set_qq_sparse_solve_policy!(:wide, bucket, _get_qq_sparse_solve_policy(:square, bucket))
    end

    # Keep legacy scalar thresholds synchronized for compatibility/reporting.
    NEMO_THRESHOLD[] = Int(round(median(Float64[
        QQ_NEMO_RANK_THRESHOLD_SQUARE[],
        QQ_NEMO_RANK_THRESHOLD_TALL[],
        QQ_NEMO_RANK_THRESHOLD_WIDE[],
    ])))
    MODULAR_NULLSPACE_THRESHOLD[] = Int(round(median(Float64[
        QQ_MODULAR_NULLSPACE_THRESHOLD_SQUARE[],
        QQ_MODULAR_NULLSPACE_THRESHOLD_TALL[],
        QQ_MODULAR_NULLSPACE_THRESHOLD_WIDE[],
    ])))
    MODULAR_SOLVE_THRESHOLD[] = Int(round(median(Float64[
        QQ_MODULAR_SOLVE_THRESHOLD_SQUARE[],
        QQ_MODULAR_SOLVE_THRESHOLD_TALL[],
        QQ_MODULAR_SOLVE_THRESHOLD_WIDE[],
    ])))
    return nothing
end

function _autotune_qq_routing_steps(profile::Symbol)
    steps = 0
    for op in (:rank, :nullspace, :solve), shape in (:square, :tall, :wide)
        steps += length(_qq_probe_cases(op, shape, profile, false))
        steps += length(_qq_probe_cases(op, shape, profile, true))
    end
    steps += _autotune_qq_sparse_solve_steps(profile)
    return steps
end

function _autotune_fp_thresholds!(profile::Symbol=:full, progress_step::Function=_noop_progress_step)
    if !_have_nemo()
        return
    end
    F = PrimeField(5)
    rng = Random.MersenneTwister(0x54414d4552)
    rank_shapes = _autotune_fp_rank_shapes(profile)
    ns_shapes = _autotune_fp_nullspace_shapes(profile)
    solve_shapes = _autotune_fp_solve_shapes(profile)

    wr = Int[]
    trj = Float64[]
    trn = Float64[]
    rank_cases = Matrix{FpElem{5}}[]
    for (k, (m, n)) in enumerate(rank_shapes)
        A = _fp_dense_rand(F, m, n; rng=rng)
        push!(rank_cases, A)
        push!(wr, m * n)
        push!(trj, _bench_elapsed(() -> _rank_fp(A); reps=2))
        push!(trn, _bench_elapsed(() -> _nemo_rank(F, A); reps=2))
        progress_step("fp-rank threshold probe $(k)/$(length(rank_shapes))")
    end
    _refine_crossover_boundary!(wr, trj, trn,
        (i, reps) -> (_bench_elapsed(() -> _rank_fp(rank_cases[i]); reps=reps),
                      _bench_elapsed(() -> _nemo_rank(F, rank_cases[i]); reps=reps));
        extra_reps=(profile == :full ? 4 : 3),
        window=(profile == :full ? 2 : 1))
    FP_NEMO_RANK_THRESHOLD[] = _pick_crossover_threshold(wr, trj, trn, FP_NEMO_RANK_THRESHOLD[])

    wn = Int[]
    tnj = Float64[]
    tnn = Float64[]
    ns_cases = Matrix{FpElem{5}}[]
    for (k, (m, n)) in enumerate(ns_shapes)
        A = _fp_dense_rand(F, m, n; rng=rng)
        push!(ns_cases, A)
        push!(wn, m * n)
        push!(tnj, _bench_elapsed(() -> _nullspace_fp(A); reps=1))
        push!(tnn, _bench_elapsed(() -> _nemo_nullspace(F, A); reps=1))
        progress_step("fp-nullspace threshold probe $(k)/$(length(ns_shapes))")
    end
    _refine_crossover_boundary!(wn, tnj, tnn,
        (i, reps) -> (_bench_elapsed(() -> _nullspace_fp(ns_cases[i]); reps=reps),
                      _bench_elapsed(() -> _nemo_nullspace(F, ns_cases[i]); reps=reps));
        extra_reps=(profile == :full ? 3 : 2),
        window=(profile == :full ? 2 : 1))
    FP_NEMO_NULLSPACE_THRESHOLD[] = _pick_crossover_threshold(wn, tnj, tnn, FP_NEMO_NULLSPACE_THRESHOLD[])

    ws = Int[]
    tsj = Float64[]
    tsn = Float64[]
    solve_cases = Tuple{Matrix{FpElem{5}},Matrix{FpElem{5}}}[]
    for (k, (m, n)) in enumerate(solve_shapes)
        B = _fp_dense_rand(F, m, n; rng=rng)
        for i in 1:n
            B[i, i] = FpElem{5}(1)
        end
        X = _fp_dense_rand(F, n, 2; rng=rng)
        Y = B * X
        push!(solve_cases, (B, Y))
        push!(ws, m * n)
        push!(tsj, _bench_elapsed(() -> _solve_fullcolumn_fp(B, Y; check_rhs=false); reps=1))
        push!(tsn, _bench_elapsed(() -> _solve_fullcolumn_nemo_fp(F, B, Y; check_rhs=false, cache=false); reps=1))
        progress_step("fp-solve threshold probe $(k)/$(length(solve_shapes))")
    end
    _refine_crossover_boundary!(ws, tsj, tsn,
        (i, reps) -> begin
            B, Y = solve_cases[i]
            return (_bench_elapsed(() -> _solve_fullcolumn_fp(B, Y; check_rhs=false); reps=reps),
                    _bench_elapsed(() -> _solve_fullcolumn_nemo_fp(F, B, Y; check_rhs=false, cache=false); reps=reps))
        end;
        extra_reps=(profile == :full ? 3 : 2),
        window=(profile == :full ? 2 : 1))
    FP_NEMO_SOLVE_THRESHOLD[] = _pick_crossover_threshold(ws, tsj, tsn, FP_NEMO_SOLVE_THRESHOLD[])
end

function _autotune_float_thresholds!(profile::Symbol=:full, progress_step::Function=_noop_progress_step)
    F = RealField(Float64; rtol=1e-10, atol=1e-12)
    rng = Random.MersenneTwister(0x464c4f4154)
    shapes = _autotune_float_dense_shapes(profile)
    works = Int[]
    tqr = Float64[]
    tsvd = Float64[]
    cases = Matrix{Float64}[]
    for (k, (m, n)) in enumerate(shapes)
        A = rand(rng, Float64, m, n)
        push!(cases, A)
        push!(works, m * n)
        push!(tqr, _bench_elapsed(() -> _nullspace_float_qr_dense(F, A); reps=1))
        push!(tsvd, _bench_elapsed(() -> _nullspace_float_svd(F, A); reps=1))
        progress_step("float-dense nullspace threshold probe $(k)/$(length(shapes))")
    end
    _refine_crossover_boundary!(works, tqr, tsvd,
        (i, reps) -> (_bench_elapsed(() -> _nullspace_float_qr_dense(F, cases[i]); reps=reps),
                      _bench_elapsed(() -> _nullspace_float_svd(F, cases[i]); reps=reps));
        extra_reps=(profile == :full ? 4 : 3),
        window=(profile == :full ? 2 : 1))
    FLOAT_NULLSPACE_SVD_THRESHOLD[] = _pick_crossover_threshold(works, tqr, tsvd, FLOAT_NULLSPACE_SVD_THRESHOLD[])
end

function _autotune_float_sparse_svds_thresholds!(progress_step::Function=_noop_progress_step)
    _have_svds_backend() || return
    F = RealField(Float64; rtol=1e-10, atol=1e-12)
    rng = Random.MersenneTwister(0x53564453)
    dims = _autotune_float_sparse_dims()
    densities = _autotune_float_sparse_densities()

    winning_dims = Int[]
    winning_nnz = Int[]
    k = 0

    for n in dims
        for d in densities
            A = _float_sparse_singular_rand(n, d; rng=rng)
            tq = _bench_elapsed(() -> _nullspace_from_qr_sparse_float(F, A); reps=1)
            ts = _bench_elapsed(() -> begin
                Z = _nullspace_float_svds(F, A)
                Z === nothing || return Z
                return _nullspace_from_qr_sparse_float(F, A)
            end; reps=1)
            if ts < 0.9 * tq
                push!(winning_dims, n)
                push!(winning_nnz, nnz(A))
            end
            k += 1
            progress_step("float-sparse svds gate probe $(k)/$(length(dims) * length(densities))")
        end
    end

    isempty(winning_dims) && return
    FLOAT_SPARSE_SVDS_MIN_DIM[] = minimum(winning_dims)
    FLOAT_SPARSE_SVDS_MIN_NNZ[] = minimum(winning_nnz)
end

function _autotune_modular_thresholds!(profile::Symbol=:full, progress_step::Function=_noop_progress_step)
    rng = Random.MersenneTwister(0x4d4f44554c) # "MODUL"

    # Tune modular prime budget for _rankQQ_dim using exact-rank parity.
    rank_mats = [_qq_dense_rand(n, n; rng=rng) for n in (80, 112, 144)]
    rank_exact = [_rankQQ(A) for A in rank_mats]
    max_candidates = 2:min(length(DEFAULT_MODULAR_PRIMES), 8)
    best_max = MODULAR_MAX_PRIMES[]
    best_t = Inf
    for maxp in max_candidates
        ok = true
        ttot = 0.0
        for (k, (A, rex)) in enumerate(zip(rank_mats, rank_exact))
            rr = _rankQQ_dim(A; backend=:modular,
                            max_primes=maxp,
                            primes=DEFAULT_MODULAR_PRIMES,
                            small_threshold=1)
            ttot += _bench_elapsed(() -> _rankQQ_dim(A; backend=:modular,
                                                    max_primes=maxp,
                                                    primes=DEFAULT_MODULAR_PRIMES,
                                                    small_threshold=1); reps=1)
            progress_step("modular-rank max_primes=$(maxp) probe $(k)/$(length(rank_mats))")
            if rr != rex
                ok = false
                break
            end
        end
        if ok && ttot < best_t
            best_t = ttot
            best_max = maxp
        end
    end
    MODULAR_MAX_PRIMES[] = max(1, best_max)

    # Tune min_primes for reconstruction-based modular paths.
    ns_probe = _qq_dense_singular_rand(72; rng=rng)
    solve_B = _qq_dense_fullcolumn_rand(96, 64; rng=rng)
    solve_X = _qq_dense_rand(64, 2; rng=rng)
    solve_Y = solve_B * solve_X

    min_candidates = 1:min(4, MODULAR_MAX_PRIMES[])
    best_min = MODULAR_MIN_PRIMES[]
    best_min_t = Inf
    for minp in min_candidates
        ttot = 0.0
        N = _nullspace_modularQQ(ns_probe;
                                 primes=DEFAULT_MODULAR_PRIMES,
                                 min_primes=minp,
                                 max_primes=MODULAR_MAX_PRIMES[])
        Xh = _solve_fullcolumn_modularQQ(solve_B, solve_Y;
                                         primes=DEFAULT_MODULAR_PRIMES,
                                         min_primes=minp,
                                         max_primes=MODULAR_MAX_PRIMES[],
                                         check_rhs=false)
        ttot += _bench_elapsed(() -> _nullspace_modularQQ(ns_probe;
                                                           primes=DEFAULT_MODULAR_PRIMES,
                                                           min_primes=minp,
                                                           max_primes=MODULAR_MAX_PRIMES[]); reps=1)
        progress_step("modular min_primes=$(minp) nullspace probe")
        ttot += _bench_elapsed(() -> _solve_fullcolumn_modularQQ(solve_B, solve_Y;
                                                                  primes=DEFAULT_MODULAR_PRIMES,
                                                                  min_primes=minp,
                                                                  max_primes=MODULAR_MAX_PRIMES[],
                                                                  check_rhs=false); reps=1)
        progress_step("modular min_primes=$(minp) solve probe")
        ok = (N !== nothing && _verify_nullspaceQQ(ns_probe, N) &&
              Xh !== nothing && _verify_solveQQ(solve_B, Xh, solve_Y))
        if ok && ttot < best_min_t
            best_min_t = ttot
            best_min = minp
        end
    end
    MODULAR_MIN_PRIMES[] = min(best_min, MODULAR_MAX_PRIMES[])

    # Tune crossover threshold for QQ modular nullspace and solve.
    wn = Int[]
    tne = Float64[]
    tnm = Float64[]
    ns_shapes = _autotune_modular_nullspace_shapes(profile)
    ns_cases = Matrix{QQ}[]
    for (k, (m, n)) in enumerate(ns_shapes)
        A = _qq_dense_rand(m, n; rng=rng)
        min(m, n) >= 2 && (A[:, end] = A[:, 1])
        push!(ns_cases, A)
        push!(wn, m * n)
        push!(tne, _bench_elapsed(() -> _nullspaceQQ(A); reps=1))
        push!(tnm, _bench_elapsed(() -> begin
            N = _nullspace_modularQQ(A;
                                     primes=DEFAULT_MODULAR_PRIMES,
                                     min_primes=MODULAR_MIN_PRIMES[],
                max_primes=MODULAR_MAX_PRIMES[])
            N === nothing ? _nullspaceQQ(A) : N
        end; reps=1))
        progress_step("modular-nullspace crossover probe $(k)/$(length(ns_shapes))")
    end
    _refine_crossover_boundary!(wn, tne, tnm,
        (i, reps) -> begin
            A = ns_cases[i]
            return (_bench_elapsed(() -> _nullspaceQQ(A); reps=reps),
                    _bench_elapsed(() -> begin
                        N = _nullspace_modularQQ(A;
                                                 primes=DEFAULT_MODULAR_PRIMES,
                                                 min_primes=MODULAR_MIN_PRIMES[],
                                                 max_primes=MODULAR_MAX_PRIMES[])
                        N === nothing ? _nullspaceQQ(A) : N
                    end; reps=reps))
        end;
        extra_reps=4,
        window=2)
    MODULAR_NULLSPACE_THRESHOLD[] = _pick_crossover_threshold(wn, tne, tnm, MODULAR_NULLSPACE_THRESHOLD[])

    ws = Int[]
    tse = Float64[]
    tsm = Float64[]
    solve_shapes = _autotune_modular_solve_shapes(profile)
    solve_cases = Tuple{Matrix{QQ},Matrix{QQ}}[]
    for (k, (m, n)) in enumerate(solve_shapes)
        B = _qq_dense_fullcolumn_rand(m, n; rng=rng)
        X = _qq_dense_rand(size(B, 2), 2; rng=rng)
        Y = B * X
        push!(solve_cases, (B, Y))
        push!(ws, m * n)
        push!(tse, _bench_elapsed(() -> _solve_fullcolumnQQ(B, Y; check_rhs=false); reps=1))
        push!(tsm, _bench_elapsed(() -> begin
            Xh = _solve_fullcolumn_modularQQ(B, Y;
                                             primes=DEFAULT_MODULAR_PRIMES,
                                             min_primes=MODULAR_MIN_PRIMES[],
                                             max_primes=MODULAR_MAX_PRIMES[],
                                             check_rhs=false)
            Xh === nothing ? _solve_fullcolumnQQ(B, Y; check_rhs=false) : Xh
        end; reps=1))
        progress_step("modular-solve crossover probe $(k)/$(length(solve_shapes))")
    end
    _refine_crossover_boundary!(ws, tse, tsm,
        (i, reps) -> begin
            B, Y = solve_cases[i]
            return (_bench_elapsed(() -> _solve_fullcolumnQQ(B, Y; check_rhs=false); reps=reps),
                    _bench_elapsed(() -> begin
                        Xh = _solve_fullcolumn_modularQQ(B, Y;
                                                         primes=DEFAULT_MODULAR_PRIMES,
                                                         min_primes=MODULAR_MIN_PRIMES[],
                                                         max_primes=MODULAR_MAX_PRIMES[],
                                                         check_rhs=false)
                        Xh === nothing ? _solve_fullcolumnQQ(B, Y; check_rhs=false) : Xh
                    end; reps=reps))
        end;
        extra_reps=4,
        window=2)
    MODULAR_SOLVE_THRESHOLD[] = _pick_crossover_threshold(ws, tse, tsm, MODULAR_SOLVE_THRESHOLD[])

    # Tune _rankQQ_dim exact/modular crossover.
    wr = Int[]
    tre = Float64[]
    trm = Float64[]
    rank_shapes = _autotune_rankqq_dim_shapes(profile)
    rank_cases = Matrix{QQ}[]
    for (k, (m, n)) in enumerate(rank_shapes)
        A = _qq_dense_rand(m, n; rng=rng)
        push!(rank_cases, A)
        push!(wr, m * n)
        push!(tre, _bench_elapsed(() -> _rankQQ(A); reps=1))
        push!(trm, _bench_elapsed(() -> _rankQQ_dim(A; backend=:modular,
                                                   max_primes=MODULAR_MAX_PRIMES[],
                                                   primes=DEFAULT_MODULAR_PRIMES,
                                                   small_threshold=1); reps=1))
        progress_step("_rankQQ_dim crossover probe $(k)/$(length(rank_shapes))")
    end
    _refine_crossover_boundary!(wr, tre, trm,
        (i, reps) -> begin
            A = rank_cases[i]
            return (_bench_elapsed(() -> _rankQQ(A); reps=reps),
                    _bench_elapsed(() -> _rankQQ_dim(A; backend=:modular,
                                                    max_primes=MODULAR_MAX_PRIMES[],
                                                    primes=DEFAULT_MODULAR_PRIMES,
                                                    small_threshold=1); reps=reps))
        end;
        extra_reps=4,
        window=2)
    RANKQQ_DIM_SMALL_THRESHOLD[] = _pick_crossover_threshold(wr, tre, trm, RANKQQ_DIM_SMALL_THRESHOLD[])
end

function _zn_rand_phi(cm, field, nrows::Int, ncols::Int, rng::Random.AbstractRNG)
    K = cm.coeff_type(field)
    Phi = Matrix{K}(undef, nrows, ncols)
    @inbounds for i in 1:nrows, j in 1:ncols
        Phi[i, j] = cm.coerce(field, rand(rng, -2:2))
    end
    return Phi
end

function _zn_dimat_fixture_friendly(fz, cm, field; ncuts::Int, rng::Random.AbstractRNG)
    n = 2
    tau = fz.face(n, [false, true])
    thresholds = collect(-ncuts:ncuts)
    FlatT = typeof(fz.IndFlat(tau, (0, 0); id=:F0))
    InjT = typeof(fz.IndInj(tau, (1, 0); id=:E0))
    flats = Vector{FlatT}(undef, length(thresholds))
    injectives = Vector{InjT}(undef, length(thresholds))
    @inbounds for (i, t) in enumerate(thresholds)
        flats[i] = fz.IndFlat(tau, (t, 0); id=Symbol(:F, i))
        injectives[i] = fz.IndInj(tau, (t + 1, 0); id=Symbol(:E, i))
    end
    Phi = _zn_rand_phi(cm, field, length(injectives), length(flats), rng)
    FG = fz.Flange{cm.coeff_type(field)}(n, flats, injectives, Phi; field=field)
    return FG, (-2 * ncuts, -2 * ncuts), (2 * ncuts, 2 * ncuts)
end

function _zn_dimat_fixture_adversarial(fz, cm, field; ncuts::Int, rng::Random.AbstractRNG)
    n = 2
    m = 2 * ncuts + 4
    tau = fz.face(n, [false, false])
    FlatT = typeof(fz.IndFlat(tau, (0, 0); id=:F0))
    InjT = typeof(fz.IndInj(tau, (0, 0); id=:E0))
    flats = Vector{FlatT}(undef, m)
    injectives = Vector{InjT}(undef, m)
    lo = -2 * ncuts
    hi = 2 * ncuts
    @inbounds for i in 1:m
        flats[i] = fz.IndFlat(tau, (rand(rng, lo:hi), rand(rng, lo:hi)); id=Symbol(:F, i))
        injectives[i] = fz.IndInj(tau, (rand(rng, lo:hi), rand(rng, lo:hi)); id=Symbol(:E, i))
    end
    Phi = _zn_rand_phi(cm, field, m, m, rng)
    FG = fz.Flange{cm.coeff_type(field)}(n, flats, injectives, Phi; field=field)
    return FG, (lo, lo), (hi, hi)
end

function _zn_sample_box_points(a::NTuple{N,Int}, b::NTuple{N,Int}, nqueries::Int,
                               rng::Random.AbstractRNG) where {N}
    out = Vector{NTuple{N,Int}}(undef, nqueries)
    @inbounds for i in 1:nqueries
        out[i] = ntuple(k -> rand(rng, a[k]:b[k]), N)
    end
    return out
end

function _autotune_zn_dimat_threshold!(profile::Symbol=:full, progress_step::Function=_noop_progress_step)
    pm = parentmodule(@__MODULE__)
    if !isdefined(pm, :FlangeZn) || !isdefined(pm, :CoreModules)
        return nothing
    end
    fz = getfield(pm, :FlangeZn)
    cm = getfield(pm, :CoreModules)
    field = cm.QQField()
    nqueries = _autotune_zn_dimat_queries_per_probe(profile)
    reps = _autotune_zn_dimat_reps(profile)
    ncutses = _autotune_zn_dimat_probe_ncuts(profile)
    candidates = _autotune_zn_dimat_threshold_candidates(profile)
    rng = Random.MersenneTwister(0x5a4e44494d4154) # "ZNDIMAT"

    probes = Tuple{Any,Vector{NTuple{2,Int}},String}[]
    for ncuts in ncutses
        FGf, af, bf = _zn_dimat_fixture_friendly(fz, cm, field; ncuts=ncuts, rng=rng)
        Qf = _zn_sample_box_points(af, bf, nqueries, rng)
        push!(probes, (FGf, Qf, "friendly ncuts=$(ncuts)"))

        FGa, aa, ba = _zn_dimat_fixture_adversarial(fz, cm, field; ncuts=ncuts, rng=rng)
        Qa = _zn_sample_box_points(aa, ba, nqueries, rng)
        push!(probes, (FGa, Qa, "adversarial ncuts=$(ncuts)"))
    end

    # Warm up compilation for both dim_at branches on one representative probe.
    if !isempty(probes)
        FG0, Q0, _ = probes[1]
        for i in 1:min(length(Q0), 64)
            fz.dim_at(FG0, Q0[i])
        end
    end

    best_thr = Int(ZN_QQ_DIMAT_SUBMATRIX_WORK_THRESHOLD[])
    best_time = Inf
    old_thr = best_thr
    cand_times = Dict{Int,Float64}()
    for thr in candidates
        _set_zn_qq_dimat_submatrix_work_threshold!(thr)
        ttot = 0.0
        for (FG, Q, label) in probes
            ttot += _bench_elapsed_median(() -> begin
                s = 0
                @inbounds for g in Q
                    s += fz.dim_at(FG, g)
                end
                s
            end; reps=reps)
            progress_step("zn-dim_at threshold=$(thr) $(label)")
        end
        cand_times[thr] = ttot
        if ttot < best_time
            best_time = ttot
            best_thr = thr
        end
    end

    # Keep incumbent unless the candidate win is material to avoid noise-driven jumps.
    old_time = get(cand_times, old_thr, Inf)
    if best_thr != old_thr && !(best_time <= 0.95 * old_time)
        best_thr = old_thr
    end

    _set_zn_qq_dimat_submatrix_work_threshold!(best_thr)
    return nothing
end

function autotune_linalg_thresholds!(; path::AbstractString=_linalg_thresholds_path(),
                                     save::Bool=true,
                                     quiet::Bool=false,
                                     profile::Symbol=:full)
    profile in (:full, :startup) || error("autotune_linalg_thresholds!: profile must be :full or :startup.")
    progress = _autotune_progress_init(_autotune_total_steps(profile); enabled=!quiet)
    step = label -> _autotune_progress_step!(progress, label)
    old = _current_linalg_thresholds()
    try
        profile == :full && _autotune_modular_thresholds!(profile, step)
        _autotune_fp_thresholds!(profile, step)
        _autotune_float_thresholds!(profile, step)
        _autotune_float_sparse_svds_thresholds!(step)
        _autotune_qq_routing_thresholds!(profile, step)
        profile == :full && _autotune_zn_dimat_threshold!(profile, step)
    catch err
        _apply_linalg_thresholds!(old)
        rethrow(err)
    end
    _autotune_progress_finish!(progress)
    if save
        _save_linalg_thresholds!(; path=path)
    end
    quiet || @info "FieldLinAlg: autotuned thresholds." path profile thresholds=_current_linalg_thresholds()
    return _current_linalg_thresholds()
end

function _choose_linalg_backend(field::AbstractCoeffField, A; op::Symbol=:rank, backend::Symbol=:auto)
    backend != :auto && return backend
    if field isa QQField
        m, n = size(A)
        shape = _qq_shape_bucket(m, n)
        work = m * n
        if _is_sparse_like(A)
            dens = work == 0 ? 0.0 : (_sparse_nnz(A) / work)
            dens_bucket = _qq_sparse_density_bucket(_qq_sparse_solve_fill(A))
            if op == :nullspace
                if dens < 0.10
                    return :julia_sparse
                end
                if _have_nemo() && work >= _qq_nemo_threshold(:nullspace, shape)
                    return :nemo
                end
                return :julia_sparse
            end
            if op == :solve
                solve_thr = _qq_sparse_solve_threshold(shape, dens_bucket)
                use_ge = _qq_sparse_solve_use_ge(shape, dens_bucket)
                nnzA = _sparse_nnz(A)
                choose_nemo = use_ge ? (nnzA >= solve_thr) : (nnzA <= solve_thr)
                if _have_nemo() && choose_nemo
                    return :nemo
                end
                return :julia_sparse
            end
            if dens < 0.10 && work <= 2_000
                return :julia_sparse
            end
            if _have_nemo() && work >= _qq_nemo_threshold(:rank, shape)
                return :nemo
            end
            return :julia_sparse
        end
        if op == :nullspace
            if _have_nemo() && work >= _qq_nemo_threshold(:nullspace, shape)
                return :nemo
            end
            if work >= _qq_modular_threshold(:nullspace, shape)
                return :modular
            end
        end
        if op == :solve
            if _have_nemo() && work >= _qq_nemo_threshold(:solve, shape)
                return :nemo
            end
            if work >= _qq_modular_threshold(:solve, shape)
                return :modular
            end
        end
        if _have_nemo() && work >= _qq_nemo_threshold(:rank, shape)
            return :nemo
        end
        return :julia_exact
    end
    if field isa PrimeField && field.p == 2
        return :f2_bit
    end
    if field isa PrimeField && field.p == 3
        return :f3_table
    end
    if field isa PrimeField && field.p > 3
        if _is_sparse_like(A)
            return :fp_sparse
        end
        if _have_nemo() && _nemo_dense_compatible(A)
            work = size(A, 1) * size(A, 2)
            if op == :rank && work >= FP_NEMO_RANK_THRESHOLD[]
                return :nemo
            elseif op == :nullspace && work >= FP_NEMO_NULLSPACE_THRESHOLD[]
                return :nemo
            elseif op == :solve && work >= FP_NEMO_SOLVE_THRESHOLD[]
                return :nemo
            end
        end
        return :julia_exact
    end
    if field isa RealField
        if _is_sparse_like(A)
            m, n = size(A)
            if op == :nullspace &&
               _have_svds_backend() &&
               min(m, n) >= FLOAT_SPARSE_SVDS_MIN_DIM[] &&
               _sparse_nnz(A) >= FLOAT_SPARSE_SVDS_MIN_NNZ[]
                return :float_sparse_svds
            end
            return :float_sparse_qr
        end
        if op == :nullspace && size(A, 1) * size(A, 2) >= FLOAT_NULLSPACE_SVD_THRESHOLD[]
            return :float_dense_svd
        end
        return :float_dense_qr
    end
    return :julia_exact
end

# Conversions between Matrix{QQ} and Nemo matrices.
function _to_fmpq_mat(A::AbstractMatrix{QQ})
    _bump_counter!(_QQ_TO_NEMO_CONVERSIONS)
    return Nemo.matrix(Nemo.QQ, A)
end

function _to_fmpq_mat(A::BackendMatrix{QQ})
    if _backend_kind(A) != :nemo
        return _to_fmpq_mat(_unwrap_backend_matrix(A))
    end
    payload = _backend_payload(A)
    if payload !== nothing
        _bump_counter!(_QQ_TO_NEMO_CACHE_HITS)
        return payload
    end
    M = _to_fmpq_mat(_unwrap_backend_matrix(A))
    _set_backend_payload!(A, M)
    return M
end

function _to_fmpq_mat(A::Transpose{QQ,<:BackendMatrix{QQ}})
    return transpose(_to_fmpq_mat(parent(A)))
end

function _to_fmpq_mat(A::Adjoint{QQ,<:BackendMatrix{QQ}})
    return transpose(_to_fmpq_mat(parent(A)))
end

# Conversions between Matrix{FpElem{p}} and Nemo matrices over GF(p).
function _to_nemo_fp_mat(A::AbstractMatrix{FpElem{p}}) where {p}
    _bump_counter!(_FP_TO_NEMO_CONVERSIONS)
    Fp = Nemo.GF(p)
    m, n = size(A)
    M = Matrix{Int}(undef, m, n)
    @inbounds for i in 1:m, j in 1:n
        M[i, j] = A[i, j].val
    end
    return Nemo.matrix(Fp, M)
end

function _to_nemo_fp_mat(A::BackendMatrix{FpElem{p}}) where {p}
    if _backend_kind(A) != :nemo
        return _to_nemo_fp_mat(_unwrap_backend_matrix(A))
    end
    payload = _backend_payload(A)
    if payload isa Pair
        pp = first(payload)
        if pp == p
            _bump_counter!(_FP_TO_NEMO_CACHE_HITS)
            return last(payload)
        end
    end
    M = _to_nemo_fp_mat(_unwrap_backend_matrix(A))
    _set_backend_payload!(A, p => M)
    return M
end

function _to_nemo_fp_mat(A::Transpose{FpElem{p},<:BackendMatrix{FpElem{p}}}) where {p}
    return transpose(_to_nemo_fp_mat(parent(A)))
end

function _to_nemo_fp_mat(A::Adjoint{FpElem{p},<:BackendMatrix{FpElem{p}}}) where {p}
    return transpose(_to_nemo_fp_mat(parent(A)))
end

function _from_nemo_fp_mat(M, ::Val{p}) where {p}
    _bump_counter!(_FP_FROM_NEMO_CONVERSIONS)
    m, n = size(M)
    A = Matrix{FpElem{p}}(undef, m, n)
    @inbounds for i in 1:m, j in 1:n
        x = M[i, j]
        A[i, j] = FpElem{p}(Int(Nemo.lift(Nemo.ZZ, x)))
    end
    return A
end

function _from_fmpq_mat(M)
    _bump_counter!(_QQ_FROM_NEMO_CONVERSIONS)
    m, n = size(M)
    A = Matrix{QQ}(undef, m, n)
    @inbounds for i in 1:m, j in 1:n
        x = M[i, j]
        A[i, j] = QQ(BigInt(Nemo.numerator(x)), BigInt(Nemo.denominator(x)))
    end
    return A
end

# Nemo-backed implementations for QQ.
function _nemo_rref_qq_mat(M; pivots::Bool=true)
    _, R = Nemo.rref(M)
    Rq = _from_fmpq_mat(R)

    pivs = Int[]
    m, n = size(Rq)
    @inbounds for i in 1:m
        for j in 1:n
            if Rq[i, j] != 0
                push!(pivs, j)
                break
            end
        end
    end

    return pivots ? (Rq, Tuple(pivs)) : Rq
end

_nemo_rank_qq_mat(M) = Nemo.rank(M)

function _nemo_pivots_mat(M)
    _, R = Nemo.rref(M)
    pivs = Int[]
    m, n = size(R)
    @inbounds for i in 1:m
        for j in 1:n
            if !iszero(R[i, j])
                push!(pivs, j)
                break
            end
        end
    end
    return Tuple(pivs)
end

function _nemo_nullspace_qq_mat(M)
    _, N = Nemo.nullspace(M)
    return _from_fmpq_mat(N)
end

function _nemo_rref(::QQField, A::AbstractMatrix{QQ}; pivots::Bool=true)
    return _nemo_rref_qq_mat(_to_fmpq_mat(A); pivots=pivots)
end

_nemo_rank(::QQField, A::AbstractMatrix{QQ}) = _nemo_rank_qq_mat(_to_fmpq_mat(A))

function _nemo_nullspace(::QQField, A::AbstractMatrix{QQ})
    return _nemo_nullspace_qq_mat(_to_fmpq_mat(A))
end

# Nemo-backed implementations for Fp (p > 3).
function _nemo_rref_fp_mat(F::PrimeField, M; pivots::Bool=true)
    p = F.p
    p > 3 || error("nemo_rref: only for p > 3")
    _, R = Nemo.rref(M)
    Rf = _from_nemo_fp_mat(R, Val(p))

    pivs = Int[]
    m, n = size(Rf)
    @inbounds for i in 1:m
        for j in 1:n
            if Rf[i, j].val != 0
                push!(pivs, j)
                break
            end
        end
    end
    return pivots ? (Rf, Tuple(pivs)) : Rf
end

function _nemo_rref(F::PrimeField, A::AbstractMatrix{FpElem{p}}; pivots::Bool=true) where {p}
    p > 3 || error("nemo_rref: only for p > 3")
    F.p == p || error("nemo_rref: field mismatch")
    return _nemo_rref_fp_mat(F, _to_nemo_fp_mat(A); pivots=pivots)
end

function _nemo_rank_fp_mat(F::PrimeField, M)
    p = F.p
    p > 3 || error("nemo_rank: only for p > 3")
    return Nemo.rank(M)
end

_nemo_rank(F::PrimeField, A::AbstractMatrix{FpElem{p}}) where {p} =
    (p > 3 ? (F.p == p ? _nemo_rank_fp_mat(F, _to_nemo_fp_mat(A)) : error("nemo_rank: field mismatch")) :
     error("nemo_rank: only for p > 3"))

function _nemo_nullspace_fp_mat(F::PrimeField, M)
    p = F.p
    p > 3 || error("nemo_nullspace: only for p > 3")
    _, N = Nemo.nullspace(M)
    return _from_nemo_fp_mat(N, Val(p))
end

function _nemo_nullspace(F::PrimeField, A::AbstractMatrix{FpElem{p}}) where {p}
    p > 3 || error("nemo_nullspace: only for p > 3")
    F.p == p || error("nemo_nullspace: field mismatch")
    M = _to_nemo_fp_mat(A)
    return _nemo_nullspace_fp_mat(F, M)
end

@inline _nemo_matrix(::NemoMatrixBackend, ::QQField, A::AbstractMatrix{QQ}) = _to_fmpq_mat(A)
@inline function _nemo_matrix(::NemoMatrixBackend, F::PrimeField, A::AbstractMatrix{FpElem{p}}) where {p}
    p > 3 || error("Nemo backend matrix conversion is only defined for p > 3")
    F.p == p || error("Field mismatch for Nemo backend matrix conversion")
    return _to_nemo_fp_mat(A)
end

# -----------------------------------------------------------------------------
# F2 full-column solve cache
# -----------------------------------------------------------------------------

struct F2FullColumnFactor
    rows::Vector{Int}
    invB::Vector{Vector{UInt64}}  # packed n x n inverse
    n::Int
end

const _F2_FULLCOLUMN_FACTOR_CACHE = WeakKeyDict{Any,Any}()

function _clear_f2_fullcolumn_cache!()
    empty!(_F2_FULLCOLUMN_FACTOR_CACHE)
    return nothing
end

# -----------------------------------------------------------------------------
# F2 bitpacked engine
# -----------------------------------------------------------------------------

@inline _f2_blocks(ncols::Int) = (ncols + 63) >>> 6

@inline function _f2_getbit(row::Vector{UInt64}, col::Int)
    blk = (col - 1) >>> 6
    bit = (col - 1) & 63
    return (row[blk + 1] >>> bit) & UInt64(1)
end

# -----------------------------------------------------------------------------
# F3 table-based engine
# -----------------------------------------------------------------------------

const _F3_ADD = UInt8[
    0 1 2;
    1 2 0;
    2 0 1
]
const _F3_SUB = UInt8[
    0 2 1;
    1 0 2;
    2 1 0
]
const _F3_MUL = UInt8[
    0 0 0;
    0 1 2;
    0 2 1
]
const _F3_INV = UInt8[0, 1, 2]  # inv(0) unused; inv(1)=1, inv(2)=2

@inline _f3_sub(a::UInt8, b::UInt8) = _F3_SUB[a + 1, b + 1]
@inline _f3_mul(a::UInt8, b::UInt8) = _F3_MUL[a + 1, b + 1]
@inline _f3_inv(a::UInt8) = _F3_INV[a + 1]
@inline _f3_neg(a::UInt8) = _F3_SUB[1, a + 1]

struct F3FullColumnFactor
    rows::Vector{Int}
    invB::Matrix{FpElem{3}}
end

const _F3_FULLCOLUMN_FACTOR_CACHE = WeakKeyDict{Any,Any}()

function _clear_f3_fullcolumn_cache!()
    empty!(_F3_FULLCOLUMN_FACTOR_CACHE)
    return nothing
end

function _f3_uint(A::AbstractMatrix{FpElem{3}})
    m, n = size(A)
    M = Matrix{UInt8}(undef, m, n)
    @inbounds for i in 1:m, j in 1:n
        M[i, j] = UInt8(A[i, j].val)
    end
    return M
end

function _f3_uint(A::SparseMatrixCSC{FpElem{3},Int})
    m, n = size(A)
    M = fill(UInt8(0), m, n)
    @inbounds for col in 1:n
        for ptr in A.colptr[col]:(A.colptr[col + 1] - 1)
            r = A.rowval[ptr]
            v = A.nzval[ptr].val
            if v != 0
                M[r, col] = UInt8(v)
            end
        end
    end
    return M
end

function _f3_from_uint(M::Matrix{UInt8})
    m, n = size(M)
    A = Matrix{FpElem{3}}(undef, m, n)
    @inbounds for i in 1:m, j in 1:n
        A[i, j] = FpElem{3}(M[i, j])
    end
    return A
end

function _rref_f3(A::AbstractMatrix{FpElem{3}}; pivots::Bool=true)
    M = _f3_uint(A)
    m, n = size(M)
    pivs = Int[]
    row = 1

    for col in 1:n
        row > m && break
        pivrow = 0
        @inbounds for r in row:m
            if M[r, col] != 0x00
                pivrow = r
                break
            end
        end
        pivrow == 0 && continue

        if pivrow != row
            M[row, :], M[pivrow, :] = M[pivrow, :], M[row, :]
        end

        piv = M[row, col]
        invp = _f3_inv(piv)
        @inbounds for j in 1:n
            M[row, j] = _f3_mul(M[row, j], invp)
        end

        @inbounds for r in 1:m
            r == row && continue
            fac = M[r, col]
            fac == 0x00 && continue
            for j in 1:n
                M[r, j] = _f3_sub(M[r, j], _f3_mul(fac, M[row, j]))
            end
        end

        push!(pivs, col)
        row += 1
    end

    R = _f3_from_uint(M)
    return pivots ? (R, Tuple(pivs)) : R
end

function _rref_f3(A::SparseMatrixCSC{FpElem{3},Int}; pivots::Bool=true)
    m, n = size(A)
    R = _SparseRREF{FpElem{3}}(n)
    rows = _sparse_rows(A)
    for i in 1:m
        _sparse_rref_push_homogeneous!(R, rows[i])
    end

    # materialize dense RREF with pivot rows first, zero rows below
    rank = _rref_rank(R)
    Md = zeros(FpElem{3}, m, n)
    for i in 1:rank
        prow = R.pivot_rows[i]
        @inbounds for t in eachindex(prow.idx)
            Md[i, prow.idx[t]] = prow.val[t]
        end
    end
    pivs = copy(R.pivot_cols)
    return pivots ? (Md, Tuple(pivs)) : Md
end

function _rank_f3(A::AbstractMatrix{FpElem{3}})
    _, pivs = _rref_f3(A; pivots=true)
    return length(pivs)
end

function _rank_f3(A::SparseMatrixCSC{FpElem{3},Int})
    _, pivs = _rref_f3(A; pivots=true)
    return length(pivs)
end

_rank_f3(A::Transpose{FpElem{3},<:SparseMatrixCSC{FpElem{3},Int}}) = _rank_f3(sparse(A))
_rank_f3(A::Adjoint{FpElem{3},<:SparseMatrixCSC{FpElem{3},Int}})  = _rank_f3(sparse(A))

function _nullspace_f3(A::AbstractMatrix{FpElem{3}})
    R, pivs = _rref_f3(A; pivots=true)
    m, n = size(R)

    free = Int[]
    piv_i = 1
    npiv = length(pivs)
    @inbounds for j in 1:n
        if piv_i <= npiv && pivs[piv_i] == j
            piv_i += 1
        else
            push!(free, j)
        end
    end

    nfree = length(free)
    Z = zeros(FpElem{3}, n, nfree)
    nfree == 0 && return Z

    @inbounds for (k, jf) in enumerate(free)
        Z[jf, k] = FpElem{3}(1)
        for (i, jp) in enumerate(pivs)
            v = R[i, jf].val
            if v != 0
                Z[jp, k] = FpElem{3}(_f3_neg(UInt8(v)))
            end
        end
    end
    return Z
end

function _nullspace_f3(A::SparseMatrixCSC{FpElem{3},Int})
    m, n = size(A)
    R = _SparseRREF{FpElem{3}}(n)
    rows = _sparse_rows(A)
    for i in 1:m
        _sparse_rref_push_homogeneous!(R, rows[i])
    end
    return _nullspace_from_pivots(R, n)
end

_nullspace_f3(A::Transpose{FpElem{3},<:SparseMatrixCSC{FpElem{3},Int}}) = _nullspace_f3(sparse(A))
_nullspace_f3(A::Adjoint{FpElem{3},<:SparseMatrixCSC{FpElem{3},Int}})  = _nullspace_f3(sparse(A))

function _solve_fullcolumn_f3(B::AbstractMatrix{FpElem{3}},
                              Y::AbstractVecOrMat{FpElem{3}};
                              check_rhs::Bool=true,
                              cache::Bool=true,
                              factor::Union{Nothing,F3FullColumnFactor}=nothing)
    want_vec = false
    Ymat = Y
    if Y isa AbstractVector
        want_vec = true
        Ymat = reshape(Y, :, 1)
    end

    m, n = size(B)
    size(Ymat, 1) == m || throw(DimensionMismatch("B and Y must have same row count"))

    fact = factor
    if fact === nothing && cache && Base.ismutabletype(typeof(B))
        fact = get(_F3_FULLCOLUMN_FACTOR_CACHE, B, nothing)
    end

    if fact === nothing
        pivs = _pivot_cols_f3(transpose(B))
        if length(pivs) != n
            error("solve_fullcolumn_f3: expected full column rank, got rank $(length(pivs)) < $n")
        end
        Bsub = Matrix{FpElem{3}}(B[pivs, :])
        invB = _f3_inverse(Bsub)
        fact = F3FullColumnFactor(pivs, invB)
        if cache && Base.ismutabletype(typeof(B))
            _F3_FULLCOLUMN_FACTOR_CACHE[B] = fact
        end
    end

    rhs = size(Ymat, 2)
    X = fact.invB * Ymat[fact.rows, :]

    if check_rhs
        if B * X != Ymat
            error("solve_fullcolumn_f3: RHS check failed")
        end
    end

    return want_vec ? vec(X) : X
end

function _solve_fullcolumn_f3(B::SparseMatrixCSC{FpElem{3},Int},
                              Y::AbstractVecOrMat{FpElem{3}};
                              check_rhs::Bool=true,
                              cache::Bool=true,
                              factor::Union{Nothing,F3FullColumnFactor}=nothing)
    want_vec = false
    Ymat = Y
    if Y isa AbstractVector
        want_vec = true
        Ymat = reshape(Y, :, 1)
    end

    m, n = size(B)
    size(Ymat, 1) == m || throw(DimensionMismatch("B and Y must have same row count"))

    fact = factor
    if fact === nothing && cache && Base.ismutabletype(typeof(B))
        fact = get(_F3_FULLCOLUMN_FACTOR_CACHE, B, nothing)
    end

    if fact === nothing
        Bt = sparse(transpose(B))
        pivs = _pivot_cols_f3(Bt)
        if length(pivs) != n
            error("solve_fullcolumn_f3: expected full column rank, got rank $(length(pivs)) < $n")
        end
        Bsub = Matrix{FpElem{3}}(B[pivs, :])
        invB = _f3_inverse(Bsub)
        fact = F3FullColumnFactor(pivs, invB)
        if cache && Base.ismutabletype(typeof(B))
            _F3_FULLCOLUMN_FACTOR_CACHE[B] = fact
        end
    end

    X = fact.invB * Ymat[fact.rows, :]

    if check_rhs
        if B * X != Ymat
            error("solve_fullcolumn_f3: RHS check failed")
        end
    end

    return want_vec ? vec(X) : X
end

function _rank_restricted_f3(A::SparseMatrixCSC{FpElem{3},Int},
                             rows::AbstractVector{Int},
                             cols::AbstractVector{Int};
                             check::Bool=false)
    nr = length(rows)
    nc = length(cols)
    if nr == 0 || nc == 0
        return 0
    end

    if check
        m, n = size(A)
        for r in rows
            @assert 1 <= r <= m
        end
        for c in cols
            @assert 1 <= c <= n
        end
    end

    m, _ = size(A)
    M = zeros(UInt8, nr, nc)
    loc = _build_row_locator(rows, m)
    @inbounds for (jloc, col) in enumerate(cols)
        for ptr in A.colptr[col]:(A.colptr[col + 1] - 1)
            r = A.rowval[ptr]
            i = _row_lookup(loc, r)
            if i != 0
                v = A.nzval[ptr].val
                if v != 0
                    M[i, jloc] = UInt8(v)
                end
            end
        end
    end
    return _rank_f3(_f3_from_uint(M))
end

function _pivot_cols_f3(A::AbstractMatrix{FpElem{3}})
    _, pivs = _rref_f3(A; pivots=true)
    return collect(pivs)
end

function _pivot_cols_f3(A::SparseMatrixCSC{FpElem{3},Int})
    m, n = size(A)
    R = _SparseRREF{FpElem{3}}(n)
    rows = _sparse_rows(A)
    for i in 1:m
        _sparse_rref_push_homogeneous!(R, rows[i])
    end
    return copy(R.pivot_cols)
end

function _f3_inverse(B::Matrix{FpElem{3}})
    n = size(B, 1)
    size(B, 2) == n || error("_f3_inverse: square matrix required")

    M = _f3_uint(B)
    Aug = Matrix{UInt8}(undef, n, 2 * n)
    @inbounds for i in 1:n
        for j in 1:n
            Aug[i, j] = M[i, j]
        end
        for j in 1:n
            Aug[i, n + j] = (i == j) ? UInt8(1) : UInt8(0)
        end
    end

    row = 1
    for col in 1:n
        row > n && break
        pivrow = 0
        @inbounds for r in row:n
            if Aug[r, col] != 0x00
                pivrow = r
                break
            end
        end
        pivrow == 0 && continue
        if pivrow != row
            Aug[row, :], Aug[pivrow, :] = Aug[pivrow, :], Aug[row, :]
        end

        piv = Aug[row, col]
        invp = _f3_inv(piv)
        @inbounds for j in 1:2*n
            Aug[row, j] = _f3_mul(Aug[row, j], invp)
        end

        @inbounds for r in 1:n
            r == row && continue
            fac = Aug[r, col]
            fac == 0x00 && continue
            for j in 1:2*n
                Aug[r, j] = _f3_sub(Aug[r, j], _f3_mul(fac, Aug[row, j]))
            end
        end

        row += 1
    end

    if row <= n
        error("_f3_inverse: matrix not invertible")
    end

    invB = Matrix{FpElem{3}}(undef, n, n)
    @inbounds for i in 1:n, j in 1:n
        invB[i, j] = FpElem{3}(Aug[i, n + j])
    end
    return invB
end

# -----------------------------------------------------------------------------
# Real / floating-point engine
# -----------------------------------------------------------------------------

function _float_tol(F::RealField, A)
    return F.atol + F.rtol * opnorm(A, 1)
end

function _rank_float(F::RealField, A::StridedMatrix{<:Real})
    R = qr(A, Val(true)).R
    d = diag(R)
    tol = _float_tol(F, A)
    return count(x -> abs(x) > tol, d)
end

function _rank_float(F::RealField, A::AbstractMatrix{<:Real})
    return _rank_float(F, Matrix{Float64}(A))
end

function _rank_float_svd(F::RealField, A)
    s = svdvals(Matrix(A))
    tol = _float_tol(F, A)
    return count(>(tol), s)
end

function _rank_float(F::RealField, A::SparseMatrixCSC)
    m, n = size(A)
    if m == 0 || n == 0
        return 0
    end
    R = qr(A).R
    d = diag(R)
    tol = _float_tol(F, A)
    return count(x -> abs(x) > tol, d)
end

function _rank_float(F::RealField, A::Transpose{<:Real,<:SparseMatrixCSC})
    return _rank_float(F, parent(A))
end

function _rank_float(F::RealField, A::Adjoint{<:Real,<:SparseMatrixCSC})
    return _rank_float(F, parent(A))
end

function _nullspace_from_qr_sparse_float(F::RealField, A)
    n = size(A, 2)
    n == 0 && return zeros(Float64, 0, 0)

    Fq = qr(A)
    R = Fq.R
    d = diag(R)
    tol = _float_tol(F, A)
    r = count(x -> abs(x) > tol, d)
    nfree = n - r
    nfree <= 0 && return zeros(Float64, n, 0)

    Zperm = zeros(Float64, n, nfree)
    if r > 0
        R11 = Matrix(view(R, 1:r, 1:r))
        R12 = Matrix(view(R, 1:r, (r + 1):n))
        Zperm[1:r, :] .= -(R11 \ R12)
    end
    @inbounds for j in 1:nfree
        Zperm[r + j, j] = 1.0
    end

    Z = zeros(Float64, n, nfree)
    pcol = Fq.pcol
    @inbounds for j in 1:n
        Z[pcol[j], :] .= Zperm[j, :]
    end
    return Z
end

function _nullspace_float_qr_dense(F::RealField, A::AbstractMatrix{<:Real})
    n = size(A, 2)
    n == 0 && return zeros(Float64, 0, 0)

    Fq = qr(A, Val(true))
    R = Fq.R
    d = diag(R)
    tol = _float_tol(F, A)
    r = count(x -> abs(x) > tol, d)
    nfree = n - r
    nfree <= 0 && return zeros(Float64, n, 0)

    Zperm = zeros(Float64, n, nfree)
    if r > 0
        R11 = Matrix(view(R, 1:r, 1:r))
        R12 = Matrix(view(R, 1:r, (r + 1):n))
        Zperm[1:r, :] .= -(R11 \ R12)
    end
    @inbounds for j in 1:nfree
        Zperm[r + j, j] = 1.0
    end

    Z = zeros(Float64, n, nfree)
    piv = Vector{Int}(Fq.p[1:n])
    @inbounds for j in 1:n
        Z[piv[j], :] .= Zperm[j, :]
    end
    return Z
end

function _nullspace_float_svd(F::RealField, A)
    S = svd(Matrix(A); full=true)
    r = _rank_float_svd(F, A)
    n = size(A, 2)
    r >= n && return zeros(eltype(S.Vt), n, 0)
    return Matrix(S.Vt[(r + 1):end, :])'
end

function _nullspace_float(F::RealField, A::AbstractMatrix{<:Real})
    return _nullspace_float_qr_dense(F, A)
end

function _nullspace_float(F::RealField, A::SparseMatrixCSC)
    return _nullspace_from_qr_sparse_float(F, A)
end

function _nullspace_float(F::RealField, A::Transpose{<:Real,<:SparseMatrixCSC})
    return _nullspace_from_qr_sparse_float(F, A)
end

function _nullspace_float(F::RealField, A::Adjoint{<:Real,<:SparseMatrixCSC})
    return _nullspace_from_qr_sparse_float(F, A)
end

function _rref_float(F::RealField, A; pivots::Bool=true)
    M = Matrix(A)
    Q, R, piv = qr(M, Val(true))
    r = _rank_float(F, A)
    pivs = Vector{Int}(piv[1:r])
    return pivots ? (M, Tuple(pivs)) : M
end

function _rref_float(F::RealField, A::SparseMatrixCSC; pivots::Bool=true)
    Fq = qr(A)
    r = _rank_float(F, A)
    pivs = Vector{Int}(Fq.pcol[1:r])
    return pivots ? (copy(A), Tuple(pivs)) : copy(A)
end

function _colspace_float(F::RealField, A)
    Q, R, piv = qr(Matrix(A), Val(true))
    r = _rank_float(F, A)
    cols = piv[1:r]
    return Matrix(A)[:, cols]
end

function _colspace_float(F::RealField, A::Transpose{<:Real,<:SparseMatrixCSC})
    Fq = qr(A)
    r = _rank_float(F, A)
    cols = Fq.pcol[1:r]
    return A[:, cols]
end

function _colspace_float(F::RealField, A::Adjoint{<:Real,<:SparseMatrixCSC})
    Fq = qr(A)
    r = _rank_float(F, A)
    cols = Fq.pcol[1:r]
    return A[:, cols]
end

function _colspace_float(F::RealField, A::SparseMatrixCSC)
    Fq = qr(A)
    r = _rank_float(F, A)
    cols = Fq.pcol[1:r]
    return A[:, cols]
end

function _solve_fullcolumn_float(F::RealField, B, Y; check_rhs::Bool=true)
    Ymat = Y isa AbstractVector ? reshape(Y, :, 1) : Matrix(Y)
    Fq = qr(Matrix(B), Val(true))
    X = Fq \ Ymat
    if check_rhs
        R = Matrix(B) * X - Ymat
        tol = _float_tol(F, B)
        norm(R) <= tol || error("solve_fullcolumn_float: RHS residual too large")
    end
    return (Y isa AbstractVector) ? vec(X) : X
end

const _FLOAT_SPARSE_FACTOR_CACHE = Dict{NTuple{5,UInt},Any}()
const FLOAT_SPARSE_FACTOR_CACHE_MAX = Ref(128)

@inline function _float_sparse_cache_key(B::SparseMatrixCSC)
    return (
        objectid(B.colptr),
        objectid(B.rowval),
        objectid(B.nzval),
        UInt(size(B, 1)),
        UInt(size(B, 2)),
    )
end

function _solve_fullcolumn_float(F::RealField, B::SparseMatrixCSC, Y;
                                 check_rhs::Bool=true, cache::Bool=true, factor=nothing)
    Ymat = Y isa AbstractVector ? reshape(Y, :, 1) : Matrix(Y)
    fac = factor
    if fac === nothing && cache
        fac = get(_FLOAT_SPARSE_FACTOR_CACHE, _float_sparse_cache_key(B), nothing)
    end
    if fac === nothing
        fac = qr(B)
        if cache
            if length(_FLOAT_SPARSE_FACTOR_CACHE) >= FLOAT_SPARSE_FACTOR_CACHE_MAX[]
                empty!(_FLOAT_SPARSE_FACTOR_CACHE)
            end
            _FLOAT_SPARSE_FACTOR_CACHE[_float_sparse_cache_key(B)] = fac
        end
    end
    X = fac \ Ymat
    if check_rhs
        R = B * X - Ymat
        tol = _float_tol(F, B)
        norm(R) <= tol || error("solve_fullcolumn_float: RHS residual too large")
    end
    return (Y isa AbstractVector) ? vec(X) : X
end
# -----------------------------------------------------------------------------
# Generic Fp (p > 3) exact engine (fallback when Nemo is unavailable)
# -----------------------------------------------------------------------------

function _rref_fp(A::AbstractMatrix{FpElem{p}}; pivots::Bool=true) where {p}
    M = Matrix{FpElem{p}}(A)
    m, n = size(M)
    pivs = Int[]
    row = 1

    for col in 1:n
        row > m && break
        pivrow = 0
        @inbounds for r in row:m
            if M[r, col].val != 0
                pivrow = r
                break
            end
        end
        pivrow == 0 && continue

        if pivrow != row
            M[row, :], M[pivrow, :] = M[pivrow, :], M[row, :]
        end

        piv = M[row, col]
        invp = inv(piv)
        @inbounds for j in 1:n
            M[row, j] *= invp
        end

        @inbounds for r in 1:m
            r == row && continue
            fac = M[r, col]
            fac.val == 0 && continue
            for j in 1:n
                M[r, j] -= fac * M[row, j]
            end
        end

        push!(pivs, col)
        row += 1
    end

    return pivots ? (M, Tuple(pivs)) : M
end

function _rank_fp(A::AbstractMatrix{FpElem{p}}) where {p}
    _, pivs = _rref_fp(A; pivots=true)
    return length(pivs)
end

function _rank_fp(A::SparseMatrixCSC{FpElem{p},Int}) where {p}
    m, n = size(A)
    if m == 0 || n == 0
        return 0
    end
    rows = _sparse_rows(A)
    R = _SparseRREF{FpElem{p}}(n)
    maxrank = min(m, n)
    @inbounds for i in 1:m
        _sparse_rref_push_homogeneous!(R, rows[i])
        if _rref_rank(R) == maxrank
            break
        end
    end
    return _rref_rank(R)
end

_rank_fp(A::Transpose{FpElem{p},<:SparseMatrixCSC{FpElem{p},Int}}) where {p} = _rank_fp(parent(A))
_rank_fp(A::Adjoint{FpElem{p},<:SparseMatrixCSC{FpElem{p},Int}}) where {p} = _rank_fp(parent(A))

function _nullspace_fp(A::AbstractMatrix{FpElem{p}}) where {p}
    R, pivs = _rref_fp(A; pivots=true)
    m, n = size(R)

    free = Int[]
    piv_i = 1
    npiv = length(pivs)
    @inbounds for j in 1:n
        if piv_i <= npiv && pivs[piv_i] == j
            piv_i += 1
        else
            push!(free, j)
        end
    end

    nfree = length(free)
    Z = zeros(FpElem{p}, n, nfree)
    nfree == 0 && return Z

    @inbounds for (k, jf) in enumerate(free)
        Z[jf, k] = FpElem{p}(1)
        for (i, jp) in enumerate(pivs)
            v = R[i, jf]
            if v.val != 0
                Z[jp, k] = -v
            end
        end
    end
    return Z
end

function _nullspace_fp(A::SparseMatrixCSC{FpElem{p},Int}) where {p}
    m, n = size(A)
    rows = _sparse_rows(A)
    R = _SparseRREF{FpElem{p}}(n)
    maxrank = min(m, n)
    @inbounds for i in 1:m
        _sparse_rref_push_homogeneous!(R, rows[i])
        if _rref_rank(R) == maxrank
            break
        end
    end
    return _nullspace_from_pivots(R, n)
end

function _nullspace_fp_from_transposed_parent(A::SparseMatrixCSC{FpElem{p},Int}) where {p}
    # For transpose(A), rows are exactly the original sparse columns of A.
    m, n = size(A)
    R = _SparseRREF{FpElem{p}}(m)
    maxrank = min(n, m)
    for col in 1:n
        rng = A.colptr[col]:(A.colptr[col + 1] - 1)
        idx = Int[]
        val = FpElem{p}[]
        sizehint!(idx, length(rng))
        sizehint!(val, length(rng))
        @inbounds for ptr in rng
            v = A.nzval[ptr]
            if !iszero(v)
                push!(idx, A.rowval[ptr])
                push!(val, v)
            end
        end
        _sparse_rref_push_homogeneous!(R, SparseRow{FpElem{p}}(idx, val))
        if _rref_rank(R) == maxrank
            break
        end
    end
    return _nullspace_from_pivots(R, m)
end

_nullspace_fp(A::Transpose{FpElem{p},<:SparseMatrixCSC{FpElem{p},Int}}) where {p} = _nullspace_fp_from_transposed_parent(parent(A))
_nullspace_fp(A::Adjoint{FpElem{p},<:SparseMatrixCSC{FpElem{p},Int}})  where {p} = _nullspace_fp_from_transposed_parent(parent(A))

function _solve_fullcolumn_fp(B::AbstractMatrix{FpElem{p}},
                              Y::AbstractVecOrMat{FpElem{p}};
                              check_rhs::Bool=true) where {p}
    want_vec = false
    Ymat = Y
    if Y isa AbstractVector
        want_vec = true
        Ymat = reshape(Y, :, 1)
    end

    m, n = size(B)
    size(Ymat, 1) == m || throw(DimensionMismatch("B and Y must have same row count"))

    Aug = hcat(B, Ymat)
    R, pivs = _rref_fp(Aug; pivots=true)
    if length(pivs) != n
        error("solve_fullcolumn_fp: expected full column rank, got rank $(length(pivs)) < $n")
    end

    rhs = size(Ymat, 2)
    X = zeros(FpElem{p}, n, rhs)
    @inbounds for (row, pcol) in enumerate(pivs)
        X[pcol, :] = R[row, n+1:n+rhs]
    end

    if check_rhs
        if B * X != Ymat
            error("solve_fullcolumn_fp: RHS check failed")
        end
    end

    return want_vec ? vec(X) : X
end

function _solve_fullcolumn_fp(B::SparseMatrixCSC{FpElem{p},Int},
                              Y::AbstractVecOrMat{FpElem{p}};
                              check_rhs::Bool=true) where {p}
    want_vec = false
    Ymat = Y
    if Y isa AbstractVector
        want_vec = true
        Ymat = reshape(Y, :, 1)
    end

    m, n = size(B)
    size(Ymat, 1) == m || throw(DimensionMismatch("B and Y must have same row count"))

    rhs = size(Ymat, 2)
    RA = _SparseRREFAugmented{FpElem{p}}(n, rhs)
    rows = _sparse_rows(B)

    @inbounds for i in 1:m
        row_rhs = Vector{FpElem{p}}(undef, rhs)
        for j in 1:rhs
            row_rhs[j] = Ymat[i, j]
        end
        status = _sparse_rref_push_augmented!(RA, rows[i], row_rhs)
        if status === :inconsistent
            error("solve_fullcolumn_fp: RHS is not in column space of B")
        end
    end

    pivs = RA.rref.pivot_cols
    if length(pivs) != n
        error("solve_fullcolumn_fp: expected full column rank, got rank $(length(pivs)) < $n")
    end
    @inbounds for pcol in pivs
        if pcol > n
            error("solve_fullcolumn_fp: RHS is not in column space of B")
        end
    end

    X = zeros(FpElem{p}, n, rhs)
    @inbounds for (row, pcol) in enumerate(pivs)
        X[pcol, :] .= RA.pivot_rhs[row]
    end

    if check_rhs
        if B * X != Ymat
            error("solve_fullcolumn_fp: RHS check failed")
        end
    end

    return want_vec ? vec(X) : X
end

@inline function _f2_setbit!(row::Vector{UInt64}, col::Int)
    blk = (col - 1) >>> 6
    bit = (col - 1) & 63
    row[blk + 1] |= (UInt64(1) << bit)
end

function _pack_f2(A::AbstractMatrix{FpElem{2}})
    m, n = size(A)
    nb = _f2_blocks(n)
    rows = [fill(UInt64(0), nb) for _ in 1:m]
    @inbounds for i in 1:m
        row = rows[i]
        for j in 1:n
            if A[i, j].val != 0
                _f2_setbit!(row, j)
            end
        end
    end
    return rows, n
end

function _pack_f2(A::SparseMatrixCSC{FpElem{2},Int})
    m, n = size(A)
    nb = _f2_blocks(n)
    rows = [fill(UInt64(0), nb) for _ in 1:m]
    @inbounds for col in 1:n
        for ptr in A.colptr[col]:(A.colptr[col + 1] - 1)
            r = A.rowval[ptr]
            if A.nzval[ptr].val != 0
                _f2_setbit!(rows[r], col)
            end
        end
    end
    return rows, n
end

function _unpack_f2(rows::Vector{Vector{UInt64}}, m::Int, n::Int)
    A = Matrix{FpElem{2}}(undef, m, n)
    z = FpElem{2}(0)
    o = FpElem{2}(1)
    @inbounds for i in 1:m
        row = rows[i]
        for j in 1:n
            A[i, j] = (_f2_getbit(row, j) == 1) ? o : z
        end
    end
    return A
end

function _f2_row_xor!(a::Vector{UInt64}, b::Vector{UInt64})
    @inbounds for k in eachindex(a)
        a[k] = xor(a[k], b[k])
    end
end

function _f2_inverse_packed(B::AbstractMatrix{FpElem{2}})
    n = size(B, 1)
    size(B, 2) == n || error("_f2_inverse_packed: square matrix required")

    nb = _f2_blocks(2 * n)
    rows = [fill(UInt64(0), nb) for _ in 1:n]
    @inbounds for i in 1:n
        row = rows[i]
        for j in 1:n
            if B[i, j].val != 0
                _f2_setbit!(row, j)
            end
        end
        _f2_setbit!(row, n + i)
    end

    pivs = _gauss_jordan_f2!(rows, 2 * n; pivot_cols=n)
    if length(pivs) != n
        error("_f2_inverse_packed: matrix not invertible")
    end

    inv_rows = [fill(UInt64(0), _f2_blocks(n)) for _ in 1:n]
    @inbounds for i in 1:n
        row = rows[i]
        for j in 1:n
            if _f2_getbit(row, n + j) == 1
                _f2_setbit!(inv_rows[i], j)
            end
        end
    end
    return inv_rows
end

function _gauss_jordan_f2!(rows::Vector{Vector{UInt64}}, ncols::Int; pivot_cols::Int=ncols)
    m = length(rows)
    pivs = Int[]
    r = 1
    nb = _f2_blocks(ncols)
    for col in 1:pivot_cols
        r > m && break
        blk = (col - 1) >>> 6
        bit = (col - 1) & 63
        mask = UInt64(1) << bit
        piv = 0
        @inbounds for i in r:m
            if (rows[i][blk + 1] & mask) != 0
                piv = i
                break
            end
        end
        piv == 0 && continue

        if piv != r
            rows[r], rows[piv] = rows[piv], rows[r]
        end

        # Eliminate from all other rows to get RREF.
        @inbounds for i in 1:m
            if i != r && (rows[i][blk + 1] & mask) != 0
                _f2_row_xor!(rows[i], rows[r])
            end
        end

        push!(pivs, col)
        r += 1
    end
    return pivs
end

function _rank_f2(A::AbstractMatrix{FpElem{2}})
    rows, n = _pack_f2(A)
    pivs = _gauss_jordan_f2!(rows, n)
    return length(pivs)
end

function _rank_f2(A::SparseMatrixCSC{FpElem{2},Int})
    rows, n = _pack_f2(A)
    pivs = _gauss_jordan_f2!(rows, n)
    return length(pivs)
end

_rank_f2(A::Transpose{FpElem{2},<:SparseMatrixCSC{FpElem{2},Int}}) = _rank_f2(sparse(A))
_rank_f2(A::Adjoint{FpElem{2},<:SparseMatrixCSC{FpElem{2},Int}})  = _rank_f2(sparse(A))

function _rref_f2(A::AbstractMatrix{FpElem{2}}; pivots::Bool=true)
    rows, n = _pack_f2(A)
    pivs = _gauss_jordan_f2!(rows, n)
    R = _unpack_f2(rows, size(A, 1), n)
    return pivots ? (R, Tuple(pivs)) : R
end

function _rref_f2(A::SparseMatrixCSC{FpElem{2},Int}; pivots::Bool=true)
    rows, n = _pack_f2(A)
    pivs = _gauss_jordan_f2!(rows, n)
    R = _unpack_f2(rows, size(A, 1), n)
    return pivots ? (R, Tuple(pivs)) : R
end

function _pivot_cols_f2(A::AbstractMatrix{FpElem{2}})
    rows, n = _pack_f2(A)
    return _gauss_jordan_f2!(rows, n)
end

function _pivot_cols_f2(A::SparseMatrixCSC{FpElem{2},Int})
    rows, n = _pack_f2(A)
    return _gauss_jordan_f2!(rows, n)
end

function _nullspace_f2(A::AbstractMatrix{FpElem{2}})
    m, n = size(A)
    rows, _ = _pack_f2(A)
    pivs = _gauss_jordan_f2!(rows, n)

    free = Int[]
    piv_i = 1
    npiv = length(pivs)
    @inbounds for j in 1:n
        if piv_i <= npiv && pivs[piv_i] == j
            piv_i += 1
        else
            push!(free, j)
        end
    end

    nfree = length(free)
    Z = Matrix{FpElem{2}}(undef, n, nfree)
    z = FpElem{2}(0)
    o = FpElem{2}(1)
    @inbounds for j in 1:n
        for k in 1:nfree
            Z[j, k] = z
        end
    end
    nfree == 0 && return Z

    # Free variable basis vectors.
    @inbounds for (k, jf) in enumerate(free)
        Z[jf, k] = o
    end

    # For each pivot row, set pivot coordinate from free coordinates.
    @inbounds for (row, pcol) in enumerate(pivs)
        prow = rows[row]
        for (k, jf) in enumerate(free)
            if _f2_getbit(prow, jf) == 1
                Z[pcol, k] = o
            end
        end
    end
    return Z
end

function _nullspace_f2(A::SparseMatrixCSC{FpElem{2},Int})
    m, n = size(A)
    rows, _ = _pack_f2(A)
    pivs = _gauss_jordan_f2!(rows, n)

    free = Int[]
    piv_i = 1
    npiv = length(pivs)
    @inbounds for j in 1:n
        if piv_i <= npiv && pivs[piv_i] == j
            piv_i += 1
        else
            push!(free, j)
        end
    end

    nfree = length(free)
    Z = Matrix{FpElem{2}}(undef, n, nfree)
    z = FpElem{2}(0)
    o = FpElem{2}(1)
    @inbounds for j in 1:n
        for k in 1:nfree
            Z[j, k] = z
        end
    end
    nfree == 0 && return Z

    @inbounds for (k, jf) in enumerate(free)
        Z[jf, k] = o
    end

    @inbounds for (row, pcol) in enumerate(pivs)
        prow = rows[row]
        for (k, jf) in enumerate(free)
            if _f2_getbit(prow, jf) == 1
                Z[pcol, k] = o
            end
        end
    end
    return Z
end

_nullspace_f2(A::Transpose{FpElem{2},<:SparseMatrixCSC{FpElem{2},Int}}) = _nullspace_f2(sparse(A))
_nullspace_f2(A::Adjoint{FpElem{2},<:SparseMatrixCSC{FpElem{2},Int}})  = _nullspace_f2(sparse(A))

function _rank_restricted_f2(A::SparseMatrixCSC{FpElem{2},Int},
                             rows::AbstractVector{Int},
                             cols::AbstractVector{Int};
                             check::Bool=false)
    nr = length(rows)
    nc = length(cols)
    if nr == 0 || nc == 0
        return 0
    end

    if check
        m, n = size(A)
        for r in rows
            @assert 1 <= r <= m
        end
        for c in cols
            @assert 1 <= c <= n
        end
    end

    m, _ = size(A)
    loc = _build_row_locator(rows, m)

    nb = _f2_blocks(nc)
    subrows = [fill(UInt64(0), nb) for _ in 1:nr]
    @inbounds for (jloc, col) in enumerate(cols)
        for ptr in A.colptr[col]:(A.colptr[col + 1] - 1)
            r = A.rowval[ptr]
            i = _row_lookup(loc, r)
            if i != 0 && A.nzval[ptr].val != 0
                _f2_setbit!(subrows[i], jloc)
            end
        end
    end

    pivs = _gauss_jordan_f2!(subrows, nc)
    return length(pivs)
end

function _solve_fullcolumn_f2(B::AbstractMatrix{FpElem{2}},
                              Y::AbstractVecOrMat{FpElem{2}};
                              check_rhs::Bool=true,
                              cache::Bool=true,
                              factor::Union{Nothing,F2FullColumnFactor}=nothing)
    want_vec = false
    Ymat = Y
    if Y isa AbstractVector
        want_vec = true
        Ymat = reshape(Y, :, 1)
    end

    m, n = size(B)
    size(Ymat, 1) == m || throw(DimensionMismatch("B and Y must have same row count"))
    rhs = size(Ymat, 2)

    # Try cached factor first (if allowed).
    fact = factor
    if fact === nothing && cache && Base.ismutabletype(typeof(B))
        fact = get(_F2_FULLCOLUMN_FACTOR_CACHE, B, nothing)
    end

    if fact === nothing
        # Build factor: choose pivot rows from B^T and invert the minor.
        pivs = _pivot_cols_f2(transpose(B))
        if length(pivs) != n
            error("solve_fullcolumn_f2: expected full column rank, got rank $(length(pivs)) < $n")
        end
        Bsub = Matrix{FpElem{2}}(B[pivs, :])
        invB = _f2_inverse_packed(Bsub)
        fact = F2FullColumnFactor(pivs, invB, n)
        if cache && Base.ismutabletype(typeof(B))
            _F2_FULLCOLUMN_FACTOR_CACHE[B] = fact
        end
    end

    X = Matrix{FpElem{2}}(undef, n, rhs)
    z = FpElem{2}(0)
    o = FpElem{2}(1)
    @inbounds for i in 1:n, j in 1:rhs
        X[i, j] = z
    end

    # Compute X = invB * Y[rows, :]
    rows = fact.rows
    invB = fact.invB
    for j in 1:rhs
        # Build packed vector of Y[rows, j].
        v = fill(UInt64(0), _f2_blocks(n))
        @inbounds for i in 1:n
            if Ymat[rows[i], j].val != 0
                _f2_setbit!(v, i)
            end
        end

        @inbounds for i in 1:n
            acc = UInt64(0)
            for k in eachindex(invB[i])
                acc = xor(acc, invB[i][k] & v[k])
            end
            # parity of acc
            acc = xor(acc, acc >>> 32)
            acc = xor(acc, acc >>> 16)
            acc = xor(acc, acc >>> 8)
            acc = xor(acc, acc >>> 4)
            acc = xor(acc, acc >>> 2)
            acc = xor(acc, acc >>> 1)
            if acc & UInt64(1) != 0
                X[i, j] = o
            end
        end
    end

    if check_rhs
        # Verify B * X == Y (over F2).
        for i in 1:m
            for j in 1:rhs
                acc = 0
                @inbounds for k in 1:n
                    acc = xor(acc, (B[i, k].val & X[k, j].val))
                end
                acc == Ymat[i, j].val || error("solve_fullcolumn_f2: RHS check failed")
            end
        end
    end

    return want_vec ? vec(X) : X
end

function _solve_fullcolumn_f2(B::SparseMatrixCSC{FpElem{2},Int},
                              Y::AbstractVecOrMat{FpElem{2}};
                              check_rhs::Bool=true,
                              cache::Bool=true,
                              factor::Union{Nothing,F2FullColumnFactor}=nothing)
    want_vec = false
    Ymat = Y
    if Y isa AbstractVector
        want_vec = true
        Ymat = reshape(Y, :, 1)
    end

    m, n = size(B)
    size(Ymat, 1) == m || throw(DimensionMismatch("B and Y must have same row count"))
    rhs = size(Ymat, 2)

    fact = factor
    if fact === nothing && cache && Base.ismutabletype(typeof(B))
        fact = get(_F2_FULLCOLUMN_FACTOR_CACHE, B, nothing)
    end

    if fact === nothing
        Bt = sparse(transpose(B))
        pivs = _pivot_cols_f2(Bt)
        if length(pivs) != n
            error("solve_fullcolumn_f2: expected full column rank, got rank $(length(pivs)) < $n")
        end
        Bsub = Matrix{FpElem{2}}(B[pivs, :])
        invB = _f2_inverse_packed(Bsub)
        fact = F2FullColumnFactor(pivs, invB, n)
        if cache && Base.ismutabletype(typeof(B))
            _F2_FULLCOLUMN_FACTOR_CACHE[B] = fact
        end
    end

    X = Matrix{FpElem{2}}(undef, n, rhs)
    z = FpElem{2}(0)
    o = FpElem{2}(1)
    @inbounds for i in 1:n, j in 1:rhs
        X[i, j] = z
    end

    rows = fact.rows
    invB = fact.invB
    for j in 1:rhs
        v = fill(UInt64(0), _f2_blocks(n))
        @inbounds for i in 1:n
            if Ymat[rows[i], j].val != 0
                _f2_setbit!(v, i)
            end
        end

        @inbounds for i in 1:n
            acc = UInt64(0)
            for k in eachindex(invB[i])
                acc = xor(acc, invB[i][k] & v[k])
            end
            acc = xor(acc, acc >>> 32)
            acc = xor(acc, acc >>> 16)
            acc = xor(acc, acc >>> 8)
            acc = xor(acc, acc >>> 4)
            acc = xor(acc, acc >>> 2)
            acc = xor(acc, acc >>> 1)
            if acc & UInt64(1) != 0
                X[i, j] = o
            end
        end
    end

    if check_rhs
        for i in 1:m
            for j in 1:rhs
                acc = 0
                @inbounds for k in 1:n
                    acc = xor(acc, (B[i, k].val & X[k, j].val))
                end
                acc == Ymat[i, j].val || error("solve_fullcolumn_f2: RHS check failed")
            end
        end
    end

    return want_vec ? vec(X) : X
end

# -----------------------------------------------------------------------------
# Sparse streaming RREF (generic exact field)
# -----------------------------------------------------------------------------

mutable struct SparseRow{K}
    idx::Vector{Int}
    val::Vector{K}
end

SparseRow{K}() where {K} = SparseRow{K}(Int[], K[])

Base.isempty(r::SparseRow) = isempty(r.idx)
Base.length(r::SparseRow) = length(r.idx)
Base.copy(r::SparseRow{K}) where {K} = SparseRow{K}(copy(r.idx), copy(r.val))

mutable struct _SparseRowAccumulator{K}
    marks::Vector{UInt32}
    pos::Vector{Int}
    cols::Vector{Int}
    vals::Vector{K}
    tag::UInt32
end

function _SparseRowAccumulator{K}(nvars::Int) where {K}
    return _SparseRowAccumulator{K}(zeros(UInt32, nvars), zeros(Int, nvars), Int[], K[], UInt32(1))
end

function _reset_sparse_row_accumulator!(acc::_SparseRowAccumulator)
    empty!(acc.cols)
    empty!(acc.vals)
    tag = acc.tag + UInt32(1)
    if tag == 0
        fill!(acc.marks, 0)
        tag = UInt32(1)
    end
    acc.tag = tag
    return acc
end

@inline function _push_sparse_row_entry!(acc::_SparseRowAccumulator{K}, col::Int, v::K) where {K}
    iszero(v) && return acc
    @inbounds if acc.marks[col] != acc.tag
        acc.marks[col] = acc.tag
        pos = length(acc.cols) + 1
        acc.pos[col] = pos
        push!(acc.cols, col)
        push!(acc.vals, v)
    else
        pos = acc.pos[col]
        acc.vals[pos] += v
    end
    return acc
end

function _sort_parallel_insertion!(cols::Vector{Int}, vals)
    n = length(cols)
    n <= 1 && return
    @inbounds for i in 2:n
        c = cols[i]
        v = vals[i]
        j = i - 1
        while j >= 1 && cols[j] > c
            cols[j + 1] = cols[j]
            vals[j + 1] = vals[j]
            j -= 1
        end
        cols[j + 1] = c
        vals[j + 1] = v
    end
end

function _materialize_sparse_row!(row::SparseRow{K}, acc::_SparseRowAccumulator{K}) where {K}
    cols = acc.cols
    vals = acc.vals
    _sort_parallel_insertion!(cols, vals)

    n = length(cols)
    resize!(row.idx, n)
    resize!(row.val, n)

    w = 0
    @inbounds for i in 1:n
        v = vals[i]
        if !iszero(v)
            w += 1
            row.idx[w] = cols[i]
            row.val[w] = v
        end
    end

    resize!(row.idx, w)
    resize!(row.val, w)
    return row
end

@inline function _row_coeff(row::SparseRow{K}, j::Int) where {K}
    n = length(row.idx)
    n == 0 && return zero(K)
    if j < row.idx[1] || j > row.idx[end]
        return zero(K)
    end
    k = searchsortedfirst(row.idx, j)
    return (k <= n && row.idx[k] == j) ? row.val[k] : zero(K)
end

function _row_axpy!(
    row::SparseRow{K},
    a::K,
    other::SparseRow{K},
    tmp_idx::Vector{Int},
    tmp_val::Vector{K},
) where {K}
    iszero(a) && return tmp_idx, tmp_val

    empty!(tmp_idx)
    empty!(tmp_val)
    sizehint!(tmp_idx, length(row.idx) + length(other.idx))
    sizehint!(tmp_val, length(row.idx) + length(other.idx))

    i = 1
    j = 1
    m = length(row.idx)
    n = length(other.idx)

    @inbounds while i <= m || j <= n
        if j > n || (i <= m && row.idx[i] < other.idx[j])
            push!(tmp_idx, row.idx[i])
            push!(tmp_val, row.val[i])
            i += 1
        elseif i > m || other.idx[j] < row.idx[i]
            v = a * other.val[j]
            if !iszero(v)
                push!(tmp_idx, other.idx[j])
                push!(tmp_val, v)
            end
            j += 1
        else
            v = row.val[i] + a * other.val[j]
            if !iszero(v)
                push!(tmp_idx, row.idx[i])
                push!(tmp_val, v)
            end
            i += 1
            j += 1
        end
    end

    old_idx = row.idx
    old_val = row.val
    row.idx = tmp_idx
    row.val = tmp_val
    tmp_idx = old_idx
    tmp_val = old_val
    empty!(tmp_idx)
    empty!(tmp_val)
    return tmp_idx, tmp_val
end

function _row_axpy_modp!(
    row::SparseRow{Int},
    a::Int,
    other::SparseRow{Int},
    p::Int,
    tmp_idx::Vector{Int},
    tmp_val::Vector{Int},
)
    a = mod(a, p)
    a == 0 && return tmp_idx, tmp_val

    empty!(tmp_idx)
    empty!(tmp_val)
    sizehint!(tmp_idx, length(row.idx) + length(other.idx))
    sizehint!(tmp_val, length(row.idx) + length(other.idx))

    i = 1
    j = 1
    m = length(row.idx)
    n = length(other.idx)

    @inbounds while i <= m || j <= n
        if j > n || (i <= m && row.idx[i] < other.idx[j])
            v = row.val[i]
            if v != 0
                push!(tmp_idx, row.idx[i])
                push!(tmp_val, v)
            end
            i += 1
        elseif i > m || other.idx[j] < row.idx[i]
            v = mod(a * other.val[j], p)
            if v != 0
                push!(tmp_idx, other.idx[j])
                push!(tmp_val, v)
            end
            j += 1
        else
            v = mod(row.val[i] + a * other.val[j], p)
            if v != 0
                push!(tmp_idx, row.idx[i])
                push!(tmp_val, v)
            end
            i += 1
            j += 1
        end
    end

    old_idx = row.idx
    old_val = row.val
    row.idx = tmp_idx
    row.val = tmp_val
    tmp_idx = old_idx
    tmp_val = old_val
    empty!(tmp_idx)
    empty!(tmp_val)
    return tmp_idx, tmp_val
end

mutable struct _SparseRREF{K}
    nvars::Int
    pivot_pos::Vector{Int}
    pivot_cols::Vector{Int}
    pivot_rows::Vector{SparseRow{K}}
    scratch_cols::Vector{Int}
    scratch_coeffs::Vector{K}
    tmp_idx::Vector{Int}
    tmp_val::Vector{K}
end

function _SparseRREF{K}(nvars::Int) where {K}
    return _SparseRREF{K}(
        nvars,
        zeros(Int, nvars),
        Int[],
        SparseRow{K}[],
        Int[],
        K[],
        Int[],
        K[],
    )
end

@inline _rref_rank(R::_SparseRREF) = length(R.pivot_cols)

function _sparse_rref_push_homogeneous!(R::_SparseRREF{K}, row::SparseRow{K})::Bool where {K}
    isempty(row) && return false

    empty!(R.scratch_cols)
    empty!(R.scratch_coeffs)

    @inbounds for t in eachindex(row.idx)
        j = row.idx[t]
        pos = R.pivot_pos[j]
        if pos != 0
            push!(R.scratch_cols, j)
            push!(R.scratch_coeffs, row.val[t])
        end
    end

    @inbounds for k in eachindex(R.scratch_cols)
        j = R.scratch_cols[k]
        c = R.scratch_coeffs[k]
        if !iszero(c)
            prow = R.pivot_rows[R.pivot_pos[j]]
            R.tmp_idx, R.tmp_val = _row_axpy!(row, -c, prow, R.tmp_idx, R.tmp_val)
        end
    end

    isempty(row) && return false

    p = row.idx[1]
    @assert R.pivot_pos[p] == 0
    a = row.val[1]
    inv_a = inv(a)
    @inbounds for t in eachindex(row.val)
        row.val[t] *= inv_a
    end

    for pos in 1:length(R.pivot_rows)
        prow = R.pivot_rows[pos]
        c = _row_coeff(prow, p)
        if !iszero(c)
            R.tmp_idx, R.tmp_val = _row_axpy!(prow, -c, row, R.tmp_idx, R.tmp_val)
        end
    end

    push!(R.pivot_cols, p)
    push!(R.pivot_rows, copy(row))
    R.pivot_pos[p] = length(R.pivot_rows)
    return true
end

mutable struct _SparseRREFAugmented{K}
    rref::_SparseRREF{K}
    nrhs::Int
    pivot_rhs::Vector{Vector{K}}
end

function _SparseRREFAugmented{K}(nvars::Int, nrhs::Int) where {K}
    return _SparseRREFAugmented{K}(_SparseRREF{K}(nvars), nrhs, Vector{Vector{K}}())
end

function _sparse_rref_push_augmented!(
    R::_SparseRREFAugmented{K},
    row::SparseRow{K},
    rhs::Vector{K},
)::Symbol where {K}
    RR = R.rref
    @assert length(rhs) == R.nrhs

    isempty(row) && return all(iszero, rhs) ? :dependent : :inconsistent

    empty!(RR.scratch_cols)
    empty!(RR.scratch_coeffs)

    @inbounds for t in eachindex(row.idx)
        j = row.idx[t]
        pos = RR.pivot_pos[j]
        if pos != 0
            push!(RR.scratch_cols, j)
            push!(RR.scratch_coeffs, row.val[t])
        end
    end

    @inbounds for k in eachindex(RR.scratch_cols)
        j = RR.scratch_cols[k]
        c = RR.scratch_coeffs[k]
        if !iszero(c)
            pos = RR.pivot_pos[j]
            prow = RR.pivot_rows[pos]
            prhs = R.pivot_rhs[pos]

            RR.tmp_idx, RR.tmp_val = _row_axpy!(row, -c, prow, RR.tmp_idx, RR.tmp_val)
            @inbounds for t in 1:R.nrhs
                rhs[t] -= c * prhs[t]
            end
        end
    end

    if isempty(row)
        return all(iszero, rhs) ? :dependent : :inconsistent
    end

    p = row.idx[1]
    @assert RR.pivot_pos[p] == 0
    a = row.val[1]
    inv_a = inv(a)
    @inbounds for t in eachindex(row.val)
        row.val[t] *= inv_a
    end
    @inbounds for t in 1:R.nrhs
        rhs[t] *= inv_a
    end

    for pos in 1:length(RR.pivot_rows)
        prow = RR.pivot_rows[pos]
        c = _row_coeff(prow, p)
        if !iszero(c)
            RR.tmp_idx, RR.tmp_val = _row_axpy!(prow, -c, row, RR.tmp_idx, RR.tmp_val)
            prhs = R.pivot_rhs[pos]
            @inbounds for t in 1:R.nrhs
                prhs[t] -= c * rhs[t]
            end
        end
    end

    push!(RR.pivot_cols, p)
    push!(RR.pivot_rows, copy(row))
    push!(R.pivot_rhs, copy(rhs))
    RR.pivot_pos[p] = length(RR.pivot_rows)
    return :pivot
end

function _nullspace_from_pivots(R::_SparseRREF{K}, nvars::Int) where {K}
    @assert nvars == R.nvars

    free_cols = Int[]
    sizehint!(free_cols, nvars - _rref_rank(R))
    for j in 1:nvars
        if R.pivot_pos[j] == 0
            push!(free_cols, j)
        end
    end

    nfree = length(free_cols)
    Z = zeros(K, nvars, nfree)
    if nfree == 0
        return Z
    end

    free_pos = zeros(Int, nvars)
    for (k, j) in enumerate(free_cols)
        free_pos[j] = k
        Z[j, k] = one(K)
    end

    for pos in 1:length(R.pivot_cols)
        p = R.pivot_cols[pos]
        prow = R.pivot_rows[pos]
        @inbounds for t in eachindex(prow.idx)
            j = prow.idx[t]
            k = free_pos[j]
            if k != 0
                Z[p, k] = -prow.val[t]
            end
        end
    end
    return Z
end

function _sparse_rows(A::SparseMatrixCSC{K,Int}) where {K}
    m, n = size(A)

    row_nnz = zeros(Int, m)
    @inbounds for col in 1:n
        for ptr in A.colptr[col]:(A.colptr[col + 1] - 1)
            row_nnz[A.rowval[ptr]] += 1
        end
    end

    rows_idx = [Int[] for _ in 1:m]
    rows_val = [K[] for _ in 1:m]
    for i in 1:m
        sizehint!(rows_idx[i], row_nnz[i])
        sizehint!(rows_val[i], row_nnz[i])
    end

    @inbounds for col in 1:n
        for ptr in A.colptr[col]:(A.colptr[col + 1] - 1)
            r = A.rowval[ptr]
            v = A.nzval[ptr]
            if !iszero(v)
                push!(rows_idx[r], col)
                push!(rows_val[r], v)
            end
        end
    end

    rows = Vector{SparseRow{K}}(undef, m)
    for i in 1:m
        rows[i] = SparseRow{K}(rows_idx[i], rows_val[i])
    end
    return rows
end

# -----------------------------------------------------------------------------
# QQ engine (exact rational arithmetic)
# -----------------------------------------------------------------------------

# Exact linear algebra over QQ (pure Julia). Nemo acceleration is selected via
# _choose_linalg_backend and dispatched through _nemo_* methods in this module.
# Keep this hook for optional backend matrix wrappers.
@inline _to_backend_matrix(A) = A

function _rref_backend(A::AbstractMatrix{QQ}; pivots::Bool=true)
    M = Matrix{QQ}(A)
    m, n = size(M)
    pivs = Int[]
    row = 1

    for col in 1:n
        pivrow = 0
        @inbounds for r in row:m
            if M[r, col] != 0
                pivrow = r
                break
            end
        end
        pivrow == 0 && continue

        push!(pivs, col)

        if pivrow != row
            @inbounds for j in 1:n
                M[row, j], M[pivrow, j] = M[pivrow, j], M[row, j]
            end
        end

        pivval = M[row, col]
        @inbounds for j in 1:n
            M[row, j] /= pivval
        end

        @inbounds for r in 1:m
            r == row && continue
            fac = M[r, col]
            fac == 0 && continue
            for j in 1:n
                M[r, j] -= fac * M[row, j]
            end
        end

        row += 1
        row > m && break
    end

    return pivots ? (M, Tuple(pivs)) : M
end

"""
    _rrefQQ(A::AbstractMatrix{QQ}; pivots::Bool=true)

Return the row-reduced echelon form of `A`. If `pivots=true`, also return the
pivot column indices as a tuple.
"""
function _rrefQQ(A::AbstractMatrix{QQ}; pivots::Bool=true)
    return _rref_backend(_to_backend_matrix(A); pivots=pivots)
end

_rank_backend(A::AbstractMatrix{QQ}) = length(last(_rrefQQ(A; pivots=true)))

_rankQQ(A::AbstractMatrix{QQ}) = _rank_backend(_to_backend_matrix(A))
_rankQQ(A::AbstractMatrix{Float64}) = _rankQQ(Matrix{QQ}(A))

function _nullspace_backend(A::AbstractMatrix{QQ})
    R, pivs = _rrefQQ(A; pivots=true)
    m, n = size(R)

    # pivs is increasing; compute free cols without allocating a Set.
    free = Int[]
    piv_i = 1
    npiv = length(pivs)
    @inbounds for j in 1:n
        if piv_i <= npiv && pivs[piv_i] == j
            piv_i += 1
        else
            push!(free, j)
        end
    end

    isempty(free) && return zeros(QQ, n, 0)

    B = zeros(QQ, n, length(free))
    @inbounds for (k, jf) in enumerate(free)
        B[jf, k] = 1
        for (i, jp) in enumerate(pivs)
            B[jp, k] = -R[i, jf]
        end
    end
    return B
end

_nullspaceQQ(A::AbstractMatrix{QQ}) = _nullspace_backend(_to_backend_matrix(A))

function _colspaceQQ(A::AbstractMatrix{QQ})
    cols = collect(_pivot_columnsQQ(A))
    return A[:, cols]
end


# ---------------------------------------------------------------
# Reusable solvers for full-column-rank systems B * X = Y
# ---------------------------------------------------------------

"""
    FullColumnFactor{K}

Reusable data for solving `B * X = Y` when `B` has full column rank.

We select row indices `rows` such that `B[rows, :]` is an invertible n x n minor
(where n = size(B,2)), and precompute its inverse `invB`. Then for any RHS `Y`
in the column space of `B`, the unique solution is

    X = invB * Y[rows, :]

Notes
-----
* The factor does NOT store B, so it is safe to cache in a WeakKeyDict keyed by B.
* If B is mutated after factorization, the factor becomes invalid and must be rebuilt.
"""
struct FullColumnFactor{K}
    rows::Vector{Int}
    invB::Matrix{K}
end

struct NemoFullColumnFactorQQ
    rows::Vector{Int}
    invB::Any
end

struct NemoFullColumnFactorFp{p}
    rows::Vector{Int}
    invB::Any
end

# Cache of factors keyed by the left matrix B.
# IMPORTANT: values do not reference the key, so WeakKeyDict works correctly.
const _FULLCOLUMN_FACTOR_CACHE = WeakKeyDict{Any,Any}()
const _NEMO_FULLCOLUMN_FACTOR_CACHE_QQ = WeakKeyDict{Any,Any}()
const _NEMO_FULLCOLUMN_FACTOR_CACHE_FP = WeakKeyDict{Any,Any}()

@inline _can_weak_cache_key(x) = Base.ismutabletype(typeof(x))

# ---------------------------------------------------------------
# Pivot-column detection for QQ matrices (dense or sparse)
# ---------------------------------------------------------------

# Dense pivot columns via _rrefQQ
function _pivot_columnsQQ(A::AbstractMatrix{QQ})
    _, pivs = _rrefQQ(A)
    return pivs
end

# Sparse pivot columns via sparse RREF streaming
# Sparse pivot columns via sparse RREF streaming
function _pivot_columnsQQ(A::SparseMatrixCSC{QQ,Int})
    m, n = size(A)
    if m == 0 || n == 0
        return Int[]
    end
    R = _SparseRREF{QQ}(n)
    rows = _sparse_rows(A)
    maxrank = min(m, n)
    for i in 1:m
        _sparse_rref_push_homogeneous!(R, rows[i])
        if length(R.pivot_cols) == maxrank
            break
        end
    end
    return Tuple(sort!(copy(R.pivot_cols)))
end


# Avoid densification for transpose/adjoint sparse wrappers
_pivot_columnsQQ(A::Transpose{QQ,<:SparseMatrixCSC{QQ,Int}}) = _pivot_columnsQQ(sparse(A))
_pivot_columnsQQ(A::Adjoint{QQ,<:SparseMatrixCSC{QQ,Int}})  = _pivot_columnsQQ(sparse(A))

"""
    _factor_fullcolumnQQ(B::AbstractMatrix{<:QQ}) -> FullColumnFactor{QQ}

Build reusable factorization data for solving `B * X = Y` when B has full column rank.

Most users should call `_solve_fullcolumnQQ(B,Y)`; caching is automatic for mutable matrices.
"""
function _factor_fullcolumnQQ(B::AbstractMatrix{<:QQ})::FullColumnFactor{QQ}
    m, n = size(B)
    n == 0 && return FullColumnFactor{QQ}(Int[], Matrix{QQ}(I, 0, 0))

    rows = collect(_pivot_columnsQQ(transpose(B)))
    if length(rows) != n
        error("_solve_fullcolumnQQ: expected full column rank, got rank $(length(rows)) < $n")
    end

    Bsub = Matrix{QQ}(B[rows, :])
    invB = _solve_fullcolumn_rrefQQ(Bsub, Matrix{QQ}(I, n, n))
    return FullColumnFactor{QQ}(rows, invB)
end

# Ground-truth RREF solver, always exact, no caching
function _solve_fullcolumn_rrefQQ(B::AbstractMatrix{<:QQ}, Y::AbstractVecOrMat{<:QQ})
    want_vec = false
    Ymat = Y
    if Y isa AbstractVector
        want_vec = true
        Ymat = reshape(Y, :, 1)
    end

    m, n = size(B)
    size(Ymat, 1) == m || throw(DimensionMismatch("B and Y must have same row count"))

    if n == 0
        return want_vec ? QQ[] : zeros(QQ, 0, size(Ymat, 2))
    end

    By = hcat(Matrix{QQ}(B), Matrix{QQ}(Ymat))
    R, pivs = _rrefQQ(By)

    for j in 1:n
        (j in pivs) || error("expected full column rank; missing pivot in column $j")
    end
    for pj in pivs
        pj > n && error("right-hand side is not in column space of B")
    end

    X = R[1:n, n+1:end]
    return want_vec ? vec(X) : X
end

# Fast solve using factor data
function _solve_fullcolumn_factorQQ(B::AbstractMatrix{<:QQ}, fac::FullColumnFactor{QQ},
                                    Y::AbstractVecOrMat{<:QQ};
                                    check_rhs::Bool=true)
    want_vec = false
    Ymat = Y
    if Y isa AbstractVector
        want_vec = true
        Ymat = reshape(Y, :, 1)
    end

    m, n = size(B)
    size(Ymat, 1) == m || throw(DimensionMismatch("B and Y must have same row count"))
    length(fac.rows) == n || error("stale FullColumnFactor: wrong row set length")

    Ysub = Matrix{QQ}(Ymat[fac.rows, :])
    X = fac.invB * Ysub

    if check_rhs
        # Verify B*X == Y
        if B isa Matrix{QQ}
            k = size(X, 2)
            @inbounds for j in 1:k
                for i in 1:m
                    s = zero(QQ)
                    for t in 1:n
                        s += B[i, t] * X[t, j]
                    end
                    s == Ymat[i, j] || error("right-hand side is not in column space of B")
                end
            end
        else
            Matrix{QQ}(B * X) == Matrix{QQ}(Ymat) || error("right-hand side is not in column space of B")
        end
    end

    return want_vec ? vec(X) : X
end

"""
    _solve_fullcolumnQQ(B, Y; cache=true, factor=nothing, check_rhs=true)

Solve `B*X = Y` over QQ under the assumption that B has full column rank.

If cache=true, a reusable factor is used; for mutable B (e.g. Matrix) it is cached
so repeated solves do not redo elimination.
"""
function _solve_fullcolumnQQ(B::AbstractMatrix{<:QQ}, Y::AbstractVecOrMat{<:QQ};
                            cache::Bool=true,
                            factor::Union{Nothing,FullColumnFactor{QQ}}=nothing,
                            check_rhs::Bool=true)
    if cache
        fac = factor
        if fac === nothing
            if _can_weak_cache_key(B)
                fac = get!(_FULLCOLUMN_FACTOR_CACHE, B) do
                    _factor_fullcolumnQQ(B)
                end
            else
                fac = _factor_fullcolumnQQ(B)
            end
        end
        return _solve_fullcolumn_factorQQ(B, fac, Y; check_rhs=check_rhs)
    end

    return _solve_fullcolumn_rrefQQ(B, Y)
end

"""
    _clear_fullcolumn_cache!()

Drop cached FullColumnFactor objects (useful in long sessions).
"""
function _clear_fullcolumn_cache!()
    empty!(_FULLCOLUMN_FACTOR_CACHE)
    empty!(_NEMO_FULLCOLUMN_FACTOR_CACHE_QQ)
    empty!(_NEMO_FULLCOLUMN_FACTOR_CACHE_FP)
    empty!(_FLOAT_SPARSE_FACTOR_CACHE)
    return nothing
end

function _factor_fullcolumn_nemoQQ(B::AbstractMatrix{<:QQ})::NemoFullColumnFactorQQ
    m, n = size(B)
    n == 0 && return NemoFullColumnFactorQQ(Int[], Nemo.matrix(Nemo.QQ, Matrix{QQ}(I, 0, 0)))

    _, rows_tup = _nemo_rref(QQField(), transpose(B); pivots=true)
    rows = collect(rows_tup)
    if length(rows) != n
        error("solve_fullcolumn_nemoQQ: expected full column rank, got rank $(length(rows)) < $n")
    end

    Bsub = Matrix{QQ}(B[rows, :])
    invB = inv(_to_fmpq_mat(Bsub))
    return NemoFullColumnFactorQQ(rows, invB)
end

function _solve_fullcolumn_nemoQQ(B::AbstractMatrix{<:QQ}, Y::AbstractVecOrMat{<:QQ};
                                  check_rhs::Bool=true, cache::Bool=true,
                                  factor::Union{Nothing,FullColumnFactor{QQ},NemoFullColumnFactorQQ}=nothing)
    if factor isa FullColumnFactor{QQ}
        return _solve_fullcolumn_factorQQ(B, factor, Y; check_rhs=check_rhs)
    end

    want_vec = false
    Ymat = Y
    if Y isa AbstractVector
        want_vec = true
        Ymat = reshape(Y, :, 1)
    end

    m, n = size(B)
    size(Ymat, 1) == m || throw(DimensionMismatch("B and Y must have same row count"))

    fac = factor
    if fac === nothing
        if cache && _can_weak_cache_key(B)
            fac = get!(_NEMO_FULLCOLUMN_FACTOR_CACHE_QQ, B) do
                _factor_fullcolumn_nemoQQ(B)
            end
        else
            fac = _factor_fullcolumn_nemoQQ(B)
        end
    end

    Ysub = Matrix{QQ}(Ymat[fac.rows, :])
    Xn = fac.invB * _to_fmpq_mat(Ysub)
    X = _from_fmpq_mat(Xn)

    if check_rhs && !_verify_solveQQ(B, X, Ymat)
        error("solve_fullcolumn_nemoQQ: RHS check failed")
    end

    return want_vec ? vec(X) : X
end

function _factor_fullcolumn_nemo_fp(F::PrimeField, B::AbstractMatrix{FpElem{p}}) where {p}
    p > 3 || error("solve_fullcolumn_nemo_fp: only for p > 3")
    F.p == p || error("solve_fullcolumn_nemo_fp: field mismatch")

    m, n = size(B)
    n == 0 && return NemoFullColumnFactorFp{p}(Int[], Nemo.matrix(Nemo.GF(p), zeros(Int, 0, 0)))

    _, rows_tup = _nemo_rref(F, transpose(B); pivots=true)
    rows = collect(rows_tup)
    if length(rows) != n
        error("solve_fullcolumn_nemo_fp: expected full column rank, got rank $(length(rows)) < $n")
    end

    Bsub = Matrix{FpElem{p}}(B[rows, :])
    invB = inv(_to_nemo_fp_mat(Bsub))
    return NemoFullColumnFactorFp{p}(rows, invB)
end

function _solve_fullcolumn_nemo_fp(F::PrimeField, B::AbstractMatrix{FpElem{p}},
                                   Y::AbstractVecOrMat{FpElem{p}};
                                   check_rhs::Bool=true, cache::Bool=true,
                                   factor=nothing) where {p}
    p > 3 || error("solve_fullcolumn_nemo_fp: only for p > 3")
    F.p == p || error("solve_fullcolumn_nemo_fp: field mismatch")

    want_vec = false
    Ymat = Y
    if Y isa AbstractVector
        want_vec = true
        Ymat = reshape(Y, :, 1)
    end

    m, n = size(B)
    size(Ymat, 1) == m || throw(DimensionMismatch("B and Y must have same row count"))

    fac = factor
    if fac !== nothing && !(fac isa NemoFullColumnFactorFp{p})
        error("solve_fullcolumn_nemo_fp: incompatible factor type $(typeof(fac))")
    end
    if fac === nothing
        if cache && _can_weak_cache_key(B)
            fac = get!(_NEMO_FULLCOLUMN_FACTOR_CACHE_FP, B) do
                _factor_fullcolumn_nemo_fp(F, B)
            end
        else
            fac = _factor_fullcolumn_nemo_fp(F, B)
        end
    end

    Ysub = Matrix{FpElem{p}}(Ymat[fac.rows, :])
    Xn = fac.invB * _to_nemo_fp_mat(Ysub)
    X = _from_nemo_fp_mat(Xn, Val(p))

    if check_rhs && (B * X != Ymat)
        error("solve_fullcolumn_nemo_fp: RHS check failed")
    end

    return want_vec ? vec(X) : X
end


# ---------------------------------------------------------------------------
# Sparse elimination utilities for exact computations without densifying matrices.
#
# Motivation:
# Many "naturality systems" in this code base (e.g. Hom(M,N) and various lifting
# constraints) produce huge linear systems that are *extremely sparse* but were
# previously assembled into dense QQ matrices.  That is catastrophic for both
# memory and time.
#
# The routines below allow us to stream constraint rows one-at-a-time into an
# exact reduced-row-echelon (RREF) basis of the row space, represented as:
#
#   pivots[pivot_col] = Dict{Int,QQ}  (a pivot row with pivot entry 1 at pivot_col)
#
# Rows are sparse Dicts keyed by column index.  This makes memory proportional
# to the number of *nonzero* coefficients encountered, not to nvars*neqs.
#
# These helpers are intentionally internal (underscored) but documented because
# they are used across multiple modules (DerivedFunctors, ChainComplexes, etc.).
#
# The earlier implementation used nested Dict{Int, Dict{Int,QQ}} structures to
# represent sparse rows.  This is convenient but very allocation-heavy and
# cache-unfriendly.  The code below replaces that representation with:
#
#   * SparseRow{K}: sorted (col, val) vectors for each row.
#   * _SparseRREF{K}: a streaming RREF builder using merge-based sparse axpy.
#
# This layer is internal but sits on several hot paths (Hom computations,
# derived-functor lifting, and sparse linear solves).  Keeping it allocation-
# light has an outsized impact on overall library performance.
# ---------------------------------------------------------------------------



function _rankQQ(A::SparseMatrixCSC{QQ,Int})
    m, n = size(A)
    if m == 0 || n == 0
        return 0
    end

    R = _SparseRREF{QQ}(n)
    rows = _sparse_rows(A)

    maxrank = min(m, n)
    for i in 1:m
        _sparse_rref_push_homogeneous!(R, rows[i])
        if _rref_rank(R) == maxrank
            break
        end
    end
    return _rref_rank(R)
end

@inline function _modp_qQ(q::QQ, p::Int)::Int
    q == 0 && return 0
    den = mod(denominator(q), p)
    den == 0 && throw(DomainError(q, "denominator not invertible modulo $p"))
    num = mod(numerator(q), p)
    v = num * invmod(den, p)
    v %= p
    v < 0 && (v += p)
    return Int(v)
end

function _rref_modp_dense(A::AbstractMatrix{QQ}, p::Int)
    m, n = size(A)
    M = Matrix{Int}(undef, m, n)
    @inbounds for j in 1:n
        for i in 1:m
            M[i, j] = _modp_qQ(A[i, j], p)
        end
    end

    pivs = Int[]
    row = 1
    @inbounds for col in 1:n
        piv = row
        while piv <= m && M[piv, col] == 0
            piv += 1
        end
        piv > m && continue

        if piv != row
            for j in col:n
                M[row, j], M[piv, j] = M[piv, j], M[row, j]
            end
        end

        invp = invmod(M[row, col], p)
        for j in col:n
            v = M[row, j] * invp
            v %= p
            v < 0 && (v += p)
            M[row, j] = v
        end

        for i in 1:m
            i == row && continue
            f = M[i, col]
            f == 0 && continue
            for j in col:n
                v = M[i, j] - f * M[row, j]
                v %= p
                v < 0 && (v += p)
                M[i, j] = v
            end
        end

        push!(pivs, col)
        row += 1
        row > m && break
    end

    return M, pivs
end

function _nullspace_modp_from_rref(R::Matrix{Int}, pivs, n::Int, p::Int)
    npiv = length(pivs)
    free = Int[]
    piv_i = 1
    @inbounds for j in 1:n
        if piv_i <= npiv && pivs[piv_i] == j
            piv_i += 1
        else
            push!(free, j)
        end
    end
    isempty(free) && return Matrix{Int}(undef, n, 0)

    B = zeros(Int, n, length(free))
    @inbounds for (k, jf) in enumerate(free)
        B[jf, k] = 1
        for (i, jp) in enumerate(pivs)
            v = -R[i, jf]
            v %= p
            v < 0 && (v += p)
            B[jp, k] = v
        end
    end
    return B
end

function _solve_fullcolumn_modp(B::AbstractMatrix{QQ}, Y::AbstractVecOrMat{QQ}, p::Int)
    want_vec = false
    Ymat = Y
    if Y isa AbstractVector
        want_vec = true
        Ymat = reshape(Y, :, 1)
    end

    m, n = size(B)
    size(Ymat, 1) == m || throw(DimensionMismatch("B and Y must have same row count"))

    By = hcat(B, Ymat)
    R, pivs = _rref_modp_dense(By, p)
    for j in 1:n
        (j in pivs) || error("expected full column rank; missing pivot in column $j (mod $p)")
    end
    for pj in pivs
        pj > n && error("right-hand side is not in column space of B (mod $p)")
    end

    X = R[1:n, n+1:end]
    return want_vec ? vec(X) : X
end

function _nullspaceQQ(A::SparseMatrixCSC{QQ,Int})
    m, n = size(A)
    R = _SparseRREF{QQ}(n)
    rows = _sparse_rows(A)
    for i in 1:m
        _sparse_rref_push_homogeneous!(R, rows[i])
    end
    return _nullspace_from_pivots(R, n)
end

# Avoid densification for transpose/adjoint sparse wrappers
_nullspaceQQ(A::Transpose{QQ,<:SparseMatrixCSC{QQ,Int}}) = _nullspaceQQ(sparse(A))
_nullspaceQQ(A::Adjoint{QQ,<:SparseMatrixCSC{QQ,Int}})  = _nullspaceQQ(sparse(A))
_rankQQ(A::Transpose{QQ,<:SparseMatrixCSC{QQ,Int}}) = _rankQQ(sparse(A))
_rankQQ(A::Adjoint{QQ,<:SparseMatrixCSC{QQ,Int}})  = _rankQQ(sparse(A))

function _rankQQ_restricted(
    A::SparseMatrixCSC{QQ,Int},
    rows::AbstractVector{Int},
    cols::AbstractVector{Int};
    check::Bool=false,
)::Int
    nr = length(rows)
    nc = length(cols)
    if nr == 0 || nc == 0
        return 0
    end

    if check
        m, n = size(A)
        for r in rows
            @assert 1 <= r <= m
        end
        for c in cols
            @assert 1 <= c <= n
        end
    end

    m, _ = size(A)
    loc = _build_row_locator(rows, m)

    row_idx = [Int[] for _ in 1:nr]
    row_val = [QQ[] for _ in 1:nr]

    # Iterate over selected columns once. We remap columns to local indices
    # 1..nc so the RREF builder does not need to allocate pivot_pos of length
    # size(A,2) when only a small subset is used.
    @inbounds for (jloc, col) in enumerate(cols)
        for ptr in A.colptr[col]:(A.colptr[col + 1] - 1)
            r = A.rowval[ptr]
            i = _row_lookup(loc, r)
            if i != 0
                v = A.nzval[ptr]
                if !iszero(v)
                    push!(row_idx[i], jloc)
                    push!(row_val[i], v)
                end
            end
        end
    end

    R = _SparseRREF{QQ}(nc)
    maxrank = min(nr, nc)
    for i in 1:nr
        _sparse_rref_push_homogeneous!(R, SparseRow{QQ}(row_idx[i], row_val[i]))
        if _rref_rank(R) == maxrank
            break
        end
    end
    return _rref_rank(R)
end

function _rank_restricted_sparse_generic(
    A::SparseMatrixCSC{K,Int},
    rows::AbstractVector{Int},
    cols::AbstractVector{Int};
    check::Bool=false,
)::Int where {K}
    nr = length(rows)
    nc = length(cols)
    if nr == 0 || nc == 0
        return 0
    end

    if check
        m, n = size(A)
        for r in rows
            @assert 1 <= r <= m
        end
        for c in cols
            @assert 1 <= c <= n
        end
    end

    m, _ = size(A)
    loc = _build_row_locator(rows, m)

    row_idx = [Int[] for _ in 1:nr]
    row_val = [K[] for _ in 1:nr]

    @inbounds for (jloc, col) in enumerate(cols)
        for ptr in A.colptr[col]:(A.colptr[col + 1] - 1)
            r = A.rowval[ptr]
            i = _row_lookup(loc, r)
            if i != 0
                v = A.nzval[ptr]
                if !iszero(v)
                    push!(row_idx[i], jloc)
                    push!(row_val[i], v)
                end
            end
        end
    end

    R = _SparseRREF{K}(nc)
    maxrank = min(nr, nc)
    for i in 1:nr
        _sparse_rref_push_homogeneous!(R, SparseRow{K}(row_idx[i], row_val[i]))
        if _rref_rank(R) == maxrank
            break
        end
    end
    return _rref_rank(R)
end

function _sparse_extract_restricted(
    A::SparseMatrixCSC{T,Int},
    rows::AbstractVector{Int},
    cols::AbstractVector{Int};
    check::Bool=false,
) where {T}
    nr = length(rows)
    nc = length(cols)
    if nr == 0 || nc == 0
        return spzeros(T, nr, nc)
    end

    if check
        m, n = size(A)
        for r in rows
            @assert 1 <= r <= m
        end
        for c in cols
            @assert 1 <= c <= n
        end
    end

    m, _ = size(A)
    loc = _build_row_locator(rows, m)

    I = Int[]
    J = Int[]
    V = T[]
    sizehint!(I, nnz(A))
    sizehint!(J, nnz(A))
    sizehint!(V, nnz(A))

    @inbounds for (jloc, col) in enumerate(cols)
        for ptr in A.colptr[col]:(A.colptr[col + 1] - 1)
            r = A.rowval[ptr]
            i = _row_lookup(loc, r)
            if i != 0
                v = A.nzval[ptr]
                if !iszero(v)
                    push!(I, i)
                    push!(J, jloc)
                    push!(V, v)
                end
            end
        end
    end

    return sparse(I, J, V, nr, nc)
end

function _rank_restricted_float_sparse(
    F::RealField,
    A::SparseMatrixCSC,
    rows::AbstractVector{Int},
    cols::AbstractVector{Int};
    check::Bool=false,
)
    nr = length(rows)
    nc = length(cols)
    if nr == 0 || nc == 0
        return 0
    end
    S = _sparse_extract_restricted(A, rows, cols; check=check)
    R = qr(S).R
    tol = _float_tol(F, S)
    return count(x -> abs(x) > tol, diag(R))
end


"""
    _rank_modp_sparse(A::SparseMatrixCSC{QQ}, p::Int) -> Int

Compute the rank of a sparse QQ-matrix after reducing coefficients modulo `p`.

This is intended as a *fast certificate/diagnostic* tool:
- For matrices with integer coefficients, _rank_modp_sparse(A,p) <= _rankQQ(A).
- If _rank_modp_sparse(A,p) == ncols(A), then A has full column rank over QQ.

Notes
- Choose `p` as a reasonably large prime (e.g. around 1e6) to reduce the chance
  of accidental rank drops.
- If any denominator in A is divisible by p, reduction is undefined and this
  function throws DomainError.
"""
function _rank_modp_sparse(A::SparseMatrixCSC{QQ,Int}, p::Int)::Int
    m, n = size(A)
    if m == 0 || n == 0
        return 0
    end

    # Build sparse rows over Int mod p.
    row_nnz = zeros(Int, m)
    @inbounds for col in 1:n
        for ptr in A.colptr[col]:(A.colptr[col + 1] - 1)
            row_nnz[A.rowval[ptr]] += 1
        end
    end

    rows_idx = [Int[] for _ in 1:m]
    rows_val = [Int[] for _ in 1:m]
    @inbounds for i in 1:m
        sizehint!(rows_idx[i], row_nnz[i])
        sizehint!(rows_val[i], row_nnz[i])
    end

    @inbounds for col in 1:n
        for ptr in A.colptr[col]:(A.colptr[col + 1] - 1)
            r = A.rowval[ptr]
            v = A.nzval[ptr]
            if v != 0
                a = _modp_qQ(v, p)
                if a != 0
                    push!(rows_idx[r], col)
                    push!(rows_val[r], a)
                end
            end
        end
    end

    rows = Vector{SparseRow{Int}}(undef, m)
    @inbounds for i in 1:m
        rows[i] = SparseRow{Int}(rows_idx[i], rows_val[i])
    end

    pivot_pos = zeros(Int, n)
    pivot_cols = Int[]
    pivot_rows = SparseRow{Int}[]
    scratch_cols = Int[]
    scratch_coeffs = Int[]
    tmp_idx = Int[]
    tmp_val = Int[]
    maxrank = min(m, n)

    for i in 1:m
        row = rows[i]
        if isempty(row)
            continue
        end

        # Eliminate existing pivots (collect first to avoid mutating while iterating row).
        empty!(scratch_cols)
        empty!(scratch_coeffs)
        @inbounds for t in eachindex(row.idx)
            j = row.idx[t]
            pos = pivot_pos[j]
            if pos != 0
                push!(scratch_cols, j)
                push!(scratch_coeffs, row.val[t])
            end
        end
        @inbounds for k in eachindex(scratch_cols)
            j = scratch_cols[k]
            c = scratch_coeffs[k]
            if c != 0
                prow = pivot_rows[pivot_pos[j]]
                tmp_idx, tmp_val = _row_axpy_modp!(row, -c, prow, p, tmp_idx, tmp_val)
            end
        end
        if isempty(row)
            continue
        end

        piv = row.idx[1]
        a = row.val[1]
        if a == 0
            # Guard against malformed rows.
            continue
        end

        inv_a = invmod(a, p)
        @inbounds for t in eachindex(row.val)
            row.val[t] = mod(row.val[t] * inv_a, p)
        end

        # Eliminate new pivot from existing pivot rows.
        for pos in 1:length(pivot_rows)
            prow = pivot_rows[pos]
            c = _row_coeff(prow, piv)
            if c != 0
                tmp_idx, tmp_val = _row_axpy_modp!(prow, -c, row, p, tmp_idx, tmp_val)
            end
        end

        push!(pivot_cols, piv)
        push!(pivot_rows, copy(row))
        pivot_pos[piv] = length(pivot_rows)

        if length(pivot_rows) == maxrank
            break
        end
    end

    return length(pivot_rows)
end

# ---------------------------------------------------------------
# Fast rank computations for dimension-only queries
# ---------------------------------------------------------------

const DEFAULT_MODULAR_PRIMES = Int[
    998244353, 1000000007, 1000000009, 1004535809,
    1045430273, 1053818881, 1224736769, 1300234241,
]

"""
    _rank_modp_dense(A::AbstractMatrix{QQ}, p::Int) -> Int

Rank over F_p for dense matrices by Gaussian elimination.
Used by _rankQQ_dim for fast dimension computations.
"""
function _rank_modp_dense(A::AbstractMatrix{QQ}, p::Int)::Int
    m, n = size(A)
    (m == 0 || n == 0) && return 0

    M = Matrix{Int}(undef, m, n)

    @inbounds for j in 1:n
        for i in 1:m
            q = A[i, j]
            if q == 0
                M[i, j] = 0
            else
                num = Int(mod(numerator(q), p))
                den = Int(mod(denominator(q), p))
                v = num * invmod(den, p)
                v %= p
                v < 0 && (v += p)
                M[i, j] = v
            end
        end
    end

    r = 0
    row = 1
    @inbounds for col in 1:n
        piv = row
        while piv <= m && M[piv, col] == 0
            piv += 1
        end
        piv > m && continue

        if piv != row
            for j in col:n
                M[row, j], M[piv, j] = M[piv, j], M[row, j]
            end
        end

        invp = invmod(M[row, col], p)
        for j in col:n
            v = M[row, j] * invp
            v %= p
            v < 0 && (v += p)
            M[row, j] = v
        end

        for i in (row+1):m
            f = M[i, col]
            f == 0 && continue
            for j in col:n
                v = M[i, j] - f * M[row, j]
                v %= p
                v < 0 && (v += p)
                M[i, j] = v
            end
        end

        r += 1
        row += 1
        row > m && break
    end

    return r
end

_rank_modp(A::AbstractMatrix{QQ}, p::Int) = _rank_modp_dense(A, p)
_rank_modp(A::SparseMatrixCSC{QQ,Int}, p::Int) = _rank_modp_sparse(A, p)
_rank_modp(A::Transpose{QQ,<:SparseMatrixCSC{QQ,Int}}, p::Int) = _rank_modp_sparse(sparse(A), p)
_rank_modp(A::Adjoint{QQ,<:SparseMatrixCSC{QQ,Int}}, p::Int)  = _rank_modp_sparse(sparse(A), p)

# ---------------------------------------------------------------
# Modular nullspace / solve with rational reconstruction (QQ only)
# ---------------------------------------------------------------

function _crt_combine!(A::Matrix{BigInt}, modulus::BigInt, B::Matrix{Int}, p::Int)
    mp = Int(mod(modulus, p))
    invm = invmod(mp, p)
    m = modulus
    @inbounds for j in 1:size(A, 2)
        for i in 1:size(A, 1)
            a = A[i, j]
            b = B[i, j]
            r = Int(mod(a, p))
            t = (b - r) % p
            t < 0 && (t += p)
            t = (t * invm) % p
            A[i, j] = a + m * BigInt(t)
        end
    end
    return A
end

function _rat_reconstruct(r::BigInt, m::BigInt)
    r = mod(r, m)
    a0 = m
    a1 = r
    b0 = BigInt(0)
    b1 = BigInt(1)
    bound = isqrt(div(m, 2))

    while abs(a1) > bound
        q = div(a0, a1)
        a0, a1 = a1, a0 - q * a1
        b0, b1 = b1, b0 - q * b1
    end

    b1 == 0 && return nothing
    abs(b1) > bound && return nothing
    gcd(a1, b1) == 1 || return nothing
    b1 < 0 && (a1 = -a1; b1 = -b1)
    mod(b1 * r - a1, m) == 0 || return nothing
    return (a1, b1)
end

function _reconstruct_matrix_QQ(A::Matrix{BigInt}, mod::BigInt)
    m, n = size(A)
    out = Matrix{QQ}(undef, m, n)
    @inbounds for j in 1:n
        for i in 1:m
            rr = _rat_reconstruct(A[i, j], mod)
            rr === nothing && return nothing
            num, den = rr
            out[i, j] = QQ(num, den)
        end
    end
    return out
end

function _verify_nullspaceQQ(A::AbstractMatrix{QQ}, N::AbstractMatrix{QQ})::Bool
    size(N, 2) == 0 && return true
    Matrix{QQ}(A * N) == zeros(QQ, size(A, 1), size(N, 2))
end

function _verify_solveQQ(B::AbstractMatrix{QQ}, X::AbstractVecOrMat{QQ}, Y::AbstractVecOrMat{QQ})::Bool
    Matrix{QQ}(B * X) == Matrix{QQ}(Y)
end

function _nullspace_modularQQ(A::AbstractMatrix{QQ};
                              primes::Vector{Int}=DEFAULT_MODULAR_PRIMES,
                              min_primes::Int=MODULAR_MIN_PRIMES[],
                              max_primes::Int=MODULAR_MAX_PRIMES[])
    m, n = size(A)
    n == 0 && return zeros(QQ, 0, 0)

    pivs_ref = nothing
    basis_mod = Matrix{BigInt}(undef, 0, 0)
    mod = BigInt(1)
    used = 0

    for p in primes
        R, pivs = try
            _rref_modp_dense(A, p)
        catch
            continue
        end

        if pivs_ref === nothing
            pivs_ref = pivs
        elseif pivs != pivs_ref
            continue
        end
        used += 1

        basis_p = _nullspace_modp_from_rref(R, pivs, n, p)
        if size(basis_p, 2) == 0
            return zeros(QQ, n, 0)
        end

        if mod == 1
            basis_mod = Matrix{BigInt}(undef, size(basis_p, 1), size(basis_p, 2))
            @inbounds for j in 1:size(basis_p, 2)
                for i in 1:size(basis_p, 1)
                    basis_mod[i, j] = BigInt(basis_p[i, j])
                end
            end
            mod = BigInt(p)
        else
            _crt_combine!(basis_mod, mod, basis_p, p)
            mod *= p
        end

        if used >= min_primes
            N = _reconstruct_matrix_QQ(basis_mod, mod)
            if N !== nothing && _verify_nullspaceQQ(A, N)
                return N
            end
        end
        used >= max_primes && break
    end

    return nothing
end

function _solve_fullcolumn_modularQQ(B::AbstractMatrix{QQ}, Y::AbstractVecOrMat{QQ};
                                     primes::Vector{Int}=DEFAULT_MODULAR_PRIMES,
                                     min_primes::Int=MODULAR_MIN_PRIMES[],
                                     max_primes::Int=MODULAR_MAX_PRIMES[],
                                     check_rhs::Bool=true)
    want_vec = false
    Ymat = Y
    if Y isa AbstractVector
        want_vec = true
        Ymat = reshape(Y, :, 1)
    end

    n = size(B, 2)
    mod = BigInt(1)
    sol_mod = Matrix{BigInt}(undef, 0, 0)
    used = 0

    for p in primes
        Xp = try
            _solve_fullcolumn_modp(B, Ymat, p)
        catch
            continue
        end
        used += 1

        if mod == 1
            sol_mod = Matrix{BigInt}(undef, size(Xp, 1), size(Xp, 2))
            @inbounds for j in 1:size(Xp, 2)
                for i in 1:size(Xp, 1)
                    sol_mod[i, j] = BigInt(Xp[i, j])
                end
            end
            mod = BigInt(p)
        else
            _crt_combine!(sol_mod, mod, Xp, p)
            mod *= p
        end

        if used >= min_primes
            X = _reconstruct_matrix_QQ(sol_mod, mod)
            if X !== nothing
                if !check_rhs || _verify_solveQQ(B, X, Ymat)
                    return want_vec ? vec(X) : X
                end
            end
        end
        used >= max_primes && break
    end

    return nothing
end

"""
    _rankQQ_dim(A; backend=:auto, max_primes=4, primes=DEFAULT_MODULAR_PRIMES,
                 small_threshold=RANKQQ_DIM_SMALL_THRESHOLD[]) -> Int

Fast rank intended for dimension-only queries:
- backend=:exact  uses exact _rankQQ
- backend=:modular uses ranks mod several primes
- backend=:auto uses exact for small matrices, modular otherwise
"""
function _rankQQ_dim(A::AbstractMatrix{QQ};
                    backend::Symbol=:auto,
                    max_primes::Int=4,
                    primes::Vector{Int}=DEFAULT_MODULAR_PRIMES,
                    small_threshold::Int=RANKQQ_DIM_SMALL_THRESHOLD[])::Int
    m, n = size(A)
    (m == 0 || n == 0) && return 0

    backend == :exact && return _rankQQ(A)

    if backend == :auto
        (m*n <= small_threshold) && return _rankQQ(A)
        backend = :modular
    end

    backend != :modular && error("_rankQQ_dim: unsupported backend $(backend)")

    r = 0
    used = 0
    for p in primes
        used += 1
        try
            r = max(r, _rank_modp(A, p))
        catch
            continue
        end
        r == min(m,n) && break
        used >= max_primes && break
    end
    return r
end

function _rank_tiny_exact(A::AbstractMatrix{K}) where {K}
    m, n = size(A)
    if m == 0 || n == 0
        return 0
    end
    M = Matrix{K}(A)
    r = 0
    row = 1
    @inbounds for col in 1:n
        row > m && break
        piv = 0
        for rr in row:m
            if !iszero(M[rr, col])
                piv = rr
                break
            end
        end
        piv == 0 && continue
        if piv != row
            M[row, :], M[piv, :] = M[piv, :], M[row, :]
        end
        invp = inv(M[row, col])
        for j in col:n
            M[row, j] *= invp
        end
        for rr in (row + 1):m
            fac = M[rr, col]
            iszero(fac) && continue
            for j in col:n
                M[rr, j] -= fac * M[row, j]
            end
        end
        r += 1
        row += 1
    end
    return r
end

function _rank_tiny_float(F::RealField, A::AbstractMatrix)
    m, n = size(A)
    if m == 0 || n == 0
        return 0
    end
    M = Matrix{Float64}(A)
    tol = _float_tol(F, M)
    r = 0
    row = 1
    @inbounds for col in 1:n
        row > m && break
        piv = 0
        best = tol
        for rr in row:m
            v = abs(M[rr, col])
            if v > best
                best = v
                piv = rr
            end
        end
        piv == 0 && continue
        if piv != row
            M[row, :], M[piv, :] = M[piv, :], M[row, :]
        end
        pivval = M[row, col]
        for j in col:n
            M[row, j] /= pivval
        end
        for rr in (row + 1):m
            fac = M[rr, col]
            abs(fac) <= tol && continue
            for j in col:n
                M[rr, j] -= fac * M[row, j]
            end
        end
        r += 1
        row += 1
    end
    return r
end

@inline _rank_tiny(field::RealField, A::AbstractMatrix) = _rank_tiny_float(field, A)
@inline _rank_tiny(::AbstractCoeffField, A::AbstractMatrix) = _rank_tiny_exact(A)

@inline function _rhs_ok(field::RealField, BX::AbstractMatrix, Y::AbstractMatrix)
    return norm(BX - Y) <= _float_tol(field, BX)
end
@inline _rhs_ok(::AbstractCoeffField, BX::AbstractMatrix, Y::AbstractMatrix) = (BX == Y)

function _inverse_tiny_exact(B::Matrix{K}) where {K}
    n = size(B, 1)
    size(B, 2) == n || error("_inverse_tiny_exact: expected square matrix")
    Aug = zeros(K, n, 2 * n)
    @inbounds for i in 1:n
        for j in 1:n
            Aug[i, j] = B[i, j]
        end
        Aug[i, n + i] = one(K)
    end
    @inbounds for col in 1:n
        piv = 0
        for rr in col:n
            if !iszero(Aug[rr, col])
                piv = rr
                break
            end
        end
        piv == 0 && error("_inverse_tiny_exact: matrix not invertible")
        if piv != col
            Aug[col, :], Aug[piv, :] = Aug[piv, :], Aug[col, :]
        end
        invp = inv(Aug[col, col])
        for j in col:(2 * n)
            Aug[col, j] *= invp
        end
        for rr in 1:n
            rr == col && continue
            fac = Aug[rr, col]
            iszero(fac) && continue
            for j in col:(2 * n)
                Aug[rr, j] -= fac * Aug[col, j]
            end
        end
    end
    return Matrix(Aug[:, (n + 1):(2 * n)])
end

function _inverse_tiny_float(F::RealField, B::Matrix{Float64})
    n = size(B, 1)
    size(B, 2) == n || error("_inverse_tiny_float: expected square matrix")
    tol = _float_tol(F, B)
    Aug = zeros(Float64, n, 2 * n)
    @inbounds for i in 1:n
        for j in 1:n
            Aug[i, j] = B[i, j]
        end
        Aug[i, n + i] = 1.0
    end
    @inbounds for col in 1:n
        piv = 0
        best = tol
        for rr in col:n
            v = abs(Aug[rr, col])
            if v > best
                best = v
                piv = rr
            end
        end
        piv == 0 && error("_inverse_tiny_float: matrix not invertible")
        if piv != col
            Aug[col, :], Aug[piv, :] = Aug[piv, :], Aug[col, :]
        end
        invp = 1.0 / Aug[col, col]
        for j in col:(2 * n)
            Aug[col, j] *= invp
        end
        for rr in 1:n
            rr == col && continue
            fac = Aug[rr, col]
            abs(fac) <= tol && continue
            for j in col:(2 * n)
                Aug[rr, j] -= fac * Aug[col, j]
            end
        end
    end
    return Matrix(Aug[:, (n + 1):(2 * n)])
end

@inline _inverse_tiny(field::RealField, B::AbstractMatrix) = _inverse_tiny_float(field, Matrix{Float64}(B))
@inline _inverse_tiny(::AbstractCoeffField, B::AbstractMatrix{K}) where {K} = _inverse_tiny_exact(Matrix{K}(B))

function _tiny_row_combinations(m::Int, n::Int)
    if n == 0
        return [Int[]]
    elseif n == 1
        return [[i] for i in 1:m]
    elseif n == 2
        out = Vector{Vector{Int}}()
        for i in 1:(m - 1), j in (i + 1):m
            push!(out, [i, j])
        end
        return out
    elseif n == 3
        out = Vector{Vector{Int}}()
        for i in 1:(m - 2), j in (i + 1):(m - 1), k in (j + 1):m
            push!(out, [i, j, k])
        end
        return out
    elseif n == 4
        out = Vector{Vector{Int}}()
        for i in 1:(m - 3), j in (i + 1):(m - 2), k in (j + 1):(m - 1), l in (k + 1):m
            push!(out, [i, j, k, l])
        end
        return out
    end
    return Vector{Vector{Int}}()
end

function _select_fullrank_rows_tiny(field::AbstractCoeffField, B::AbstractMatrix)
    m, n = size(B)
    m < n && return Int[]
    for rows in _tiny_row_combinations(m, n)
        if _rank_tiny(field, view(B, rows, :)) == n
            return rows
        end
    end
    return Int[]
end

function _solve_fullcolumn_tiny(field::AbstractCoeffField, B, Y; check_rhs::Bool=true)
    want_vec = Y isa AbstractVector
    Ymat = want_vec ? reshape(Y, :, 1) : Matrix(Y)
    m, n = size(B)
    size(Ymat, 1) == m || throw(DimensionMismatch("B and Y must have same row count"))
    if n == 0
        if check_rhs
            z = zero(eltype(Ymat))
            @inbounds for y in Ymat
                y == z || error("solve_fullcolumn: right-hand side is not in column space of B")
            end
        end
        return want_vec ? eltype(Ymat)[] : zeros(eltype(Ymat), 0, size(Ymat, 2))
    end
    Bdense = Matrix(B)
    rows = _select_fullrank_rows_tiny(field, Bdense)
    isempty(rows) && error("solve_fullcolumn: expected full column rank in tiny solve path")
    invB = _inverse_tiny(field, view(Bdense, rows, :))
    X = _matmul(invB, Matrix(Ymat[rows, :]))
    if check_rhs
        BX = _matmul(Bdense, X)
        _rhs_ok(field, BX, Matrix(Ymat)) || error("solve_fullcolumn: RHS check failed in tiny solve path")
    end
    return want_vec ? vec(X) : X
end


# -----------------------------------------------------------------------------
# Public API (field-generic dispatch)
# -----------------------------------------------------------------------------

function rref(field::AbstractCoeffField, A; pivots::Bool=true, backend::Symbol=:auto)
    if field isa QQField
        trait = _matrix_backend_trait(field, A; op=:rref, backend=backend)
        if trait isa NemoMatrixBackend
            return _nemo_rref_qq_mat(_nemo_matrix(trait, field, A); pivots=pivots)
        end
        return _rrefQQ(A; pivots=pivots)
    end
    if field isa PrimeField && field.p == 2
        return _rref_f2(A; pivots=pivots)
    end
    if field isa PrimeField && field.p == 3
        return _rref_f3(A; pivots=pivots)
    end
    if field isa PrimeField && field.p > 3
        trait = _matrix_backend_trait(field, A; op=:rref, backend=backend)
        if trait isa NemoMatrixBackend
            return _nemo_rref_fp_mat(field, _nemo_matrix(trait, field, A); pivots=pivots)
        end
        return _rref_fp(A; pivots=pivots)
    end
    if field isa RealField
        _ = _choose_linalg_backend(field, A; op=:rref, backend=backend)
        return _rref_float(field, A; pivots=pivots)
    end
    error("FieldLinAlg.rref: unsupported field $(typeof(field))")
end

function rank(field::AbstractCoeffField, A; backend::Symbol=:auto)
    if backend == :auto && _is_tiny_matrix(A)
        return _rank_tiny(field, A)
    end
    if field isa QQField
        trait = _matrix_backend_trait(field, A; op=:rank, backend=backend)
        if trait isa NemoMatrixBackend
            return _nemo_rank_qq_mat(_nemo_matrix(trait, field, A))
        end
        return _rankQQ(A)
    end
    if field isa PrimeField && field.p == 2
        return _rank_f2(A)
    end
    if field isa PrimeField && field.p == 3
        return _rank_f3(A)
    end
    if field isa PrimeField && field.p > 3
        trait = _matrix_backend_trait(field, A; op=:rank, backend=backend)
        if trait isa NemoMatrixBackend
            return _nemo_rank_fp_mat(field, _nemo_matrix(trait, field, A))
        end
        return _rank_fp(A)
    end
    if field isa RealField
        be = _choose_linalg_backend(field, A; op=:rank, backend=backend)
        if be == :float_dense_svd
            return _rank_float_svd(field, A)
        end
        return _rank_float(field, A)
    end
    error("FieldLinAlg.rank: unsupported field $(typeof(field))")
end

function nullspace(field::AbstractCoeffField, A; backend::Symbol=:auto)
    if field isa QQField
        trait = _matrix_backend_trait(field, A; op=:nullspace, backend=backend)
        if trait isa NemoMatrixBackend
            return _nemo_nullspace_qq_mat(_nemo_matrix(trait, field, A))
        end
        be = _choose_linalg_backend(field, A; op=:nullspace, backend=backend)
        if be == :modular
            if !(A isa SparseMatrixCSC)
                N = _nullspace_modularQQ(A)
                N === nothing || return N
            end
        end
        return _nullspaceQQ(A)
    end
    if field isa PrimeField && field.p == 2
        return _nullspace_f2(A)
    end
    if field isa PrimeField && field.p == 3
        return _nullspace_f3(A)
    end
    if field isa PrimeField && field.p > 3
        trait = _matrix_backend_trait(field, A; op=:nullspace, backend=backend)
        if trait isa NemoMatrixBackend
            return _nemo_nullspace_fp_mat(field, _nemo_matrix(trait, field, A))
        end
        return _nullspace_fp(A)
    end
    if field isa RealField
        be = _choose_linalg_backend(field, A; op=:nullspace, backend=backend)
        if be == :float_dense_svd
            return _nullspace_float_svd(field, A)
        end
        if be == :float_sparse_svds
            Z = _nullspace_float_svds(field, A)
            Z === nothing || return Z
        end
        return _nullspace_float(field, A)
    end
    error("FieldLinAlg.nullspace: unsupported field $(typeof(field))")
end

function colspace(field::AbstractCoeffField, A; backend::Symbol=:auto)
    if field isa QQField
        trait = _matrix_backend_trait(field, A; op=:colspace, backend=backend)
        if trait isa NemoMatrixBackend
            pivs = _nemo_pivots_mat(_nemo_matrix(trait, field, A))
            return A[:, collect(pivs)]
        end
        return _colspaceQQ(A)
    end
    if field isa PrimeField && field.p == 2
        pivs = _pivot_cols_f2(A)
        return A[:, pivs]
    end
    if field isa PrimeField && field.p == 3
        _, pivs = _rref_f3(A; pivots=true)
        return A[:, collect(pivs)]
    end
    if field isa PrimeField && field.p > 3
        trait = _matrix_backend_trait(field, A; op=:colspace, backend=backend)
        if trait isa NemoMatrixBackend
            pivs = _nemo_pivots_mat(_nemo_matrix(trait, field, A))
            return A[:, collect(pivs)]
        end
        _, pivs = _rref_fp(A; pivots=true)
        return A[:, collect(pivs)]
    end
    if field isa RealField
        _ = _choose_linalg_backend(field, A; op=:colspace, backend=backend)
        return _colspace_float(field, A)
    end
    error("FieldLinAlg.colspace: unsupported field $(typeof(field))")
end

function solve_fullcolumn(field::AbstractCoeffField, B, Y;
                          check_rhs::Bool=true, backend::Symbol=:auto,
                          cache::Bool=true, factor=nothing)
    if backend == :auto && _is_tiny_solve(B, Y) && !(field isa RealField)
        return _solve_fullcolumn_tiny(field, B, Y; check_rhs=check_rhs)
    end
    if field isa QQField
        be = _choose_linalg_backend(field, B; op=:solve, backend=backend)
        if be == :nemo
            return _solve_fullcolumn_nemoQQ(B, Y; check_rhs=check_rhs, cache=cache, factor=factor)
        end
        if be == :modular
            if !(B isa SparseMatrixCSC)
                X = _solve_fullcolumn_modularQQ(B, Y; check_rhs=check_rhs)
                X === nothing || return X
            end
        end
        return _solve_fullcolumnQQ(B, Y; check_rhs=check_rhs)
    end
    if field isa PrimeField && field.p == 2
        return _solve_fullcolumn_f2(B, Y; check_rhs=check_rhs, cache=cache, factor=factor)
    end
    if field isa PrimeField && field.p == 3
        return _solve_fullcolumn_f3(B, Y; check_rhs=check_rhs, cache=cache, factor=factor)
    end
    if field isa PrimeField && field.p > 3
        be = _choose_linalg_backend(field, B; op=:solve, backend=backend)
        if be == :nemo
            return _solve_fullcolumn_nemo_fp(field, B, Y; check_rhs=check_rhs, cache=cache, factor=factor)
        end
        return _solve_fullcolumn_fp(B, Y; check_rhs=check_rhs)
    end
    if field isa RealField
        be = _choose_linalg_backend(field, B; op=:solve, backend=backend)
        if be == :float_sparse_qr
            Bs = B isa SparseMatrixCSC ? B : sparse(B)
            return _solve_fullcolumn_float(field, Bs, Y; check_rhs=check_rhs, cache=cache, factor=factor)
        end
        return _solve_fullcolumn_float(field, B, Y; check_rhs=check_rhs)
    end
    error("FieldLinAlg.solve_fullcolumn: unsupported field $(typeof(field))")
end

function rank_dim(field::AbstractCoeffField, A; backend::Symbol=:auto, kwargs...)
    if field isa QQField
        return _rankQQ_dim(A; backend=backend, kwargs...)
    end
    if field isa PrimeField && field.p == 2
        return _rank_f2(A)
    end
    if field isa PrimeField && field.p == 3
        return _rank_f3(A)
    end
    if field isa PrimeField && field.p > 3
        be = _choose_linalg_backend(field, A; op=:rank, backend=backend)
        if be == :nemo
            return _nemo_rank(field, A)
        end
        return _rank_fp(A)
    end
    if field isa RealField
        be = _choose_linalg_backend(field, A; op=:rank, backend=backend)
        if be == :float_dense_svd
            return _rank_float_svd(field, A)
        end
        return _rank_float(field, A)
    end
    return rank(field, A; backend=backend)
end

function rank_restricted(field::AbstractCoeffField, A::SparseMatrixCSC,
                         rows::AbstractVector{Int}, cols::AbstractVector{Int};
                         backend::Symbol=:auto, kwargs...)
    if _is_tiny_matrix_dims(length(rows), length(cols)) &&
       !(field isa RealField) &&
       !(field isa PrimeField && field.p > 3)
        check = haskey(kwargs, :check) ? kwargs[:check] : false
        S = Matrix(_sparse_extract_restricted(A, rows, cols; check=check))
        return rank(field, S; backend=backend)
    end
    if field isa QQField
        return _rankQQ_restricted(A, rows, cols; kwargs...)
    end
    if field isa PrimeField && field.p == 2
        return _rank_restricted_f2(A, rows, cols; kwargs...)
    end
    if field isa PrimeField && field.p == 3
        return _rank_restricted_f3(A, rows, cols; kwargs...)
    end
    if field isa PrimeField && field.p > 3
        return _rank_restricted_sparse_generic(A, rows, cols; kwargs...)
    end
    if field isa RealField
        return _rank_restricted_float_sparse(field, A, rows, cols; kwargs...)
    end
    error("FieldLinAlg.rank_restricted: unsupported field $(typeof(field))")
end

function rank_restricted(field::AbstractCoeffField, A::AbstractMatrix,
                         rows::AbstractVector{Int}, cols::AbstractVector{Int};
                         backend::Symbol=:auto, kwargs...)
    A isa SparseMatrixCSC && return rank_restricted(field, A, rows, cols; backend=backend, kwargs...)
    nr = length(rows)
    nc = length(cols)
    if nr == 0 || nc == 0
        return 0
    end
    return rank(field, view(A, rows, cols); backend=backend, kwargs...)
end

@inline function _restricted_rhs_view(Y::AbstractVector, rows::AbstractVector{Int})
    return view(Y, rows)
end

@inline function _restricted_rhs_view(Y::AbstractMatrix, rows::AbstractVector{Int})
    return view(Y, rows, :)
end

@inline function _drop_check_kw(kwargs::NamedTuple)
    return (; (k => v for (k, v) in pairs(kwargs) if k != :check)...)
end
@inline _drop_check_kw(kwargs::Base.Pairs) = _drop_check_kw(values(kwargs))

function nullspace_restricted(field::AbstractCoeffField, A::SparseMatrixCSC,
                              rows::AbstractVector{Int}, cols::AbstractVector{Int};
                              backend::Symbol=:auto, kwargs...)
    nr = length(rows)
    nc = length(cols)
    K = coeff_type(field)
    if nc == 0
        return zeros(K, 0, 0)
    end
    if nr == 0
        return eye(field, nc)
    end
    check = haskey(kwargs, :check) ? kwargs[:check] : false
    S = _sparse_extract_restricted(A, rows, cols; check=check)
    return nullspace(field, S; backend=backend)
end

function nullspace_restricted(field::AbstractCoeffField, A::AbstractMatrix,
                              rows::AbstractVector{Int}, cols::AbstractVector{Int};
                              backend::Symbol=:auto, kwargs...)
    A isa SparseMatrixCSC && return nullspace_restricted(field, A, rows, cols; backend=backend, kwargs...)
    nr = length(rows)
    nc = length(cols)
    K = coeff_type(field)
    if nc == 0
        return zeros(K, 0, 0)
    end
    if nr == 0
        return eye(field, nc)
    end
    return nullspace(field, view(A, rows, cols); backend=backend)
end

function solve_fullcolumn_restricted(field::AbstractCoeffField, B::SparseMatrixCSC,
                                     rows::AbstractVector{Int}, cols::AbstractVector{Int},
                                     Y::AbstractVecOrMat;
                                     check_rhs::Bool=true,
                                     backend::Symbol=:auto,
                                     kwargs...)
    nr = length(rows)
    nc = length(cols)
    if nc == 0
        rhs = Y isa AbstractVector ? 1 : size(Y, 2)
        K = coeff_type(field)
        Z = zeros(K, 0, rhs)
        return Y isa AbstractVector ? vec(Z) : Z
    end
    nr == 0 && error("solve_fullcolumn_restricted: cannot solve with zero selected rows and nonzero columns")
    check = haskey(kwargs, :check) ? kwargs[:check] : false
    kws = _drop_check_kw(kwargs)
    Bsub = _sparse_extract_restricted(B, rows, cols; check=check)
    Ysub = _restricted_rhs_view(Y, rows)
    return solve_fullcolumn(field, Bsub, Ysub; check_rhs=check_rhs, backend=backend, kws...)
end

function solve_fullcolumn_restricted(field::AbstractCoeffField, B::AbstractMatrix,
                                     rows::AbstractVector{Int}, cols::AbstractVector{Int},
                                     Y::AbstractVecOrMat;
                                     check_rhs::Bool=true,
                                     backend::Symbol=:auto,
                                     kwargs...)
    B isa SparseMatrixCSC && return solve_fullcolumn_restricted(field, B, rows, cols, Y;
                                                                check_rhs=check_rhs,
                                                                backend=backend, kwargs...)
    nr = length(rows)
    nc = length(cols)
    if nc == 0
        rhs = Y isa AbstractVector ? 1 : size(Y, 2)
        K = coeff_type(field)
        Z = zeros(K, 0, rhs)
        return Y isa AbstractVector ? vec(Z) : Z
    end
    nr == 0 && error("solve_fullcolumn_restricted: cannot solve with zero selected rows and nonzero columns")
    Bsub = view(B, rows, cols)
    Ysub = _restricted_rhs_view(Y, rows)
    kws = _drop_check_kw(kwargs)
    return solve_fullcolumn(field, Bsub, Ysub; check_rhs=check_rhs, backend=backend, kws...)
end

function __init__()
    _LINALG_THRESHOLDS_INITIALIZED[] && return
    path = _linalg_thresholds_path()
    loaded = _load_linalg_thresholds!(; path=path, warn_on_mismatch=true)
    if !loaded && !isfile(path)
        try
            autotune_linalg_thresholds!(; path=path, save=true, quiet=true, profile=:startup)
        catch err
            @warn "FieldLinAlg: startup autotune failed; using defaults." path exception=(err, catch_backtrace())
        end
    end
    _LINALG_THRESHOLDS_INITIALIZED[] = true
    return nothing
end

end # module FieldLinAlg
