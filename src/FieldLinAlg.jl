module FieldLinAlg
# Generic linear algebra interface over configurable coefficient fields.

using LinearAlgebra
using SparseArrays

using ..CoreModules: AbstractCoeffField, QQField, PrimeField, RealField, FpElem, coeff_type, QQ

export rref, rank, nullspace, colspace, solve_fullcolumn, rank_dim, rank_restricted,
       rankQQ, rankQQ_dim, nullityQQ_dim, nullspaceQQ, colspaceQQ, rrefQQ,
       solve_fullcolumnQQ, FullColumnFactor, factor_fullcolumnQQ,
       clear_fullcolumn_cache!, rank_modp_sparse, rank_modp_dense, rankQQ_restricted,
       have_nemo

# -----------------------------------------------------------------------------
# Backend selection + Nemo hooks
# -----------------------------------------------------------------------------

const _NEMO_ENABLED = Ref(false)
have_nemo() = _NEMO_ENABLED[]

const NEMO_THRESHOLD = Ref(50_000)     # heuristic: use Nemo if m*n >= threshold
const SPARSE_DENSITY_THRESHOLD = Ref(0.15)
const MODULAR_NULLSPACE_THRESHOLD = Ref(120_000)
const MODULAR_SOLVE_THRESHOLD = Ref(120_000)
const MODULAR_MIN_PRIMES = Ref(2)
const MODULAR_MAX_PRIMES = Ref(6)

function choose_linalg_backend(field::AbstractCoeffField, A; op::Symbol=:rank, backend::Symbol=:auto)
    backend != :auto && return backend
    if field isa QQField
        if A isa SparseMatrixCSC
            return :julia_sparse
        end
        m, n = size(A)
        if op == :nullspace && m * n >= MODULAR_NULLSPACE_THRESHOLD[]
            return :modular
        end
        if op == :solve && m * n >= MODULAR_SOLVE_THRESHOLD[]
            return :modular
        end
        if have_nemo() && m * n >= NEMO_THRESHOLD[]
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
        if have_nemo() && A isa StridedMatrix
            return :nemo
        end
        return :julia_exact
    end
    if field isa RealField
        return :float
    end
    return :julia_exact
end

_nemo_unavailable() = error("FieldLinAlg: Nemo backend requested but Nemo is not available")
_nemo_rref(::QQField, A; pivots::Bool=true) = _nemo_unavailable()
_nemo_rank(::QQField, A) = _nemo_unavailable()
_nemo_nullspace(::QQField, A) = _nemo_unavailable()
_nemo_rref(::PrimeField, A; pivots::Bool=true) = _nemo_unavailable()
_nemo_rank(::PrimeField, A) = _nemo_unavailable()
_nemo_nullspace(::PrimeField, A) = _nemo_unavailable()

# -----------------------------------------------------------------------------
# F2 full-column solve cache
# -----------------------------------------------------------------------------

struct F2FullColumnFactor
    rows::Vector{Int}
    invB::Vector{Vector{UInt64}}  # packed n x n inverse
    n::Int
end

const _F2_FULLCOLUMN_FACTOR_CACHE = WeakKeyDict{Any,Any}()

function clear_f2_fullcolumn_cache!()
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

@inline _f3_add(a::UInt8, b::UInt8) = _F3_ADD[a + 1, b + 1]
@inline _f3_sub(a::UInt8, b::UInt8) = _F3_SUB[a + 1, b + 1]
@inline _f3_mul(a::UInt8, b::UInt8) = _F3_MUL[a + 1, b + 1]
@inline _f3_inv(a::UInt8) = _F3_INV[a + 1]
@inline _f3_neg(a::UInt8) = _F3_SUB[1, a + 1]

struct F3FullColumnFactor
    rows::Vector{Int}
    invB::Matrix{FpElem{3}}
end

const _F3_FULLCOLUMN_FACTOR_CACHE = WeakKeyDict{Any,Any}()

function clear_f3_fullcolumn_cache!()
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
    return pivots ? (R, pivs) : R
end

function _rref_f3(A::SparseMatrixCSC{FpElem{3},Int}; pivots::Bool=true)
    m, n = size(A)
    R = SparseRREF{FpElem{3}}(n)
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
    return pivots ? (Md, pivs) : Md
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
    R = SparseRREF{FpElem{3}}(n)
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

    M = zeros(UInt8, nr, nc)
    rowpos = Dict{Int,Int}()
    sizehint!(rowpos, nr)
    for (i, r) in enumerate(rows)
        rowpos[r] = i
    end
    @inbounds for (jloc, col) in enumerate(cols)
        for ptr in A.colptr[col]:(A.colptr[col + 1] - 1)
            r = A.rowval[ptr]
            i = get(rowpos, r, 0)
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
    return pivs
end

function _pivot_cols_f3(A::SparseMatrixCSC{FpElem{3},Int})
    m, n = size(A)
    R = SparseRREF{FpElem{3}}(n)
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

function _rank_float(F::RealField, A)
    s = svdvals(Matrix(A))
    tol = _float_tol(F, A)
    return count(>(tol), s)
end

function _nullspace_float(F::RealField, A)
    S = svd(Matrix(A))
    tol = _float_tol(F, A)
    idx = findall(<=(tol), S.S)
    if isempty(idx)
        return zeros(eltype(S.Vt), size(A, 2), 0)
    end
    return Matrix(S.Vt[idx, :])'
end

function _rref_float(F::RealField, A; pivots::Bool=true)
    M = Matrix(A)
    Q, R, piv = qr(M, Val(true))
    r = _rank_float(F, A)
    pivs = Vector{Int}(piv[1:r])
    return pivots ? (M, pivs) : M
end

function _colspace_float(F::RealField, A)
    Q, R, piv = qr(Matrix(A), Val(true))
    r = _rank_float(F, A)
    cols = piv[1:r]
    return Matrix(A)[:, cols]
end

function _solve_fullcolumn_float(F::RealField, B, Y; check_rhs::Bool=true)
    Ymat = Y isa AbstractVector ? reshape(Y, :, 1) : Matrix(Y)
    X = Matrix(B) \ Ymat
    if check_rhs
        R = Matrix(B) * X - Ymat
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

    return pivots ? (M, pivs) : M
end

function _rank_fp(A::AbstractMatrix{FpElem{p}}) where {p}
    _, pivs = _rref_fp(A; pivots=true)
    return length(pivs)
end

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

_nullspace_fp(A::Transpose{FpElem{p},<:SparseMatrixCSC{FpElem{p},Int}}) where {p} = _nullspace_fp(sparse(A))
_nullspace_fp(A::Adjoint{FpElem{p},<:SparseMatrixCSC{FpElem{p},Int}})  where {p} = _nullspace_fp(sparse(A))

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

function _f2_pack_rows(mat::AbstractMatrix{FpElem{2}})
    m, n = size(mat)
    nb = _f2_blocks(n)
    rows = [fill(UInt64(0), nb) for _ in 1:m]
    @inbounds for i in 1:m
        row = rows[i]
        for j in 1:n
            if mat[i, j].val != 0
                _f2_setbit!(row, j)
            end
        end
    end
    return rows, n
end

function _f2_unpack_rows(rows::Vector{Vector{UInt64}}, m::Int, n::Int)
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
    return pivots ? (R, pivs) : R
end

function _rref_f2(A::SparseMatrixCSC{FpElem{2},Int}; pivots::Bool=true)
    rows, n = _pack_f2(A)
    pivs = _gauss_jordan_f2!(rows, n)
    R = _unpack_f2(rows, size(A, 1), n)
    return pivots ? (R, pivs) : R
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

    rowpos = Dict{Int,Int}()
    sizehint!(rowpos, nr)
    for (i, r) in enumerate(rows)
        rowpos[r] = i
    end

    nb = _f2_blocks(nc)
    subrows = [fill(UInt64(0), nb) for _ in 1:nr]
    @inbounds for (jloc, col) in enumerate(cols)
        for ptr in A.colptr[col]:(A.colptr[col + 1] - 1)
            r = A.rowval[ptr]
            i = get(rowpos, r, 0)
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

function normalize_sparse_row!(cols::Vector{Int}, vals::Vector{K}) where {K}
    @assert length(cols) == length(vals)
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

    w = 0
    i = 1
    @inbounds while i <= n
        c = cols[i]
        s = vals[i]
        i += 1
        while i <= n && cols[i] == c
            s += vals[i]
            i += 1
        end
        if !iszero(s)
            w += 1
            cols[w] = c
            vals[w] = s
        end
    end
    resize!(cols, w)
    resize!(vals, w)
    return
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

mutable struct SparseRREF{K}
    nvars::Int
    pivot_pos::Vector{Int}
    pivot_cols::Vector{Int}
    pivot_rows::Vector{SparseRow{K}}
    scratch_cols::Vector{Int}
    scratch_coeffs::Vector{K}
    tmp_idx::Vector{Int}
    tmp_val::Vector{K}
end

function SparseRREF{K}(nvars::Int) where {K}
    return SparseRREF{K}(
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

@inline _rref_rank(R::SparseRREF) = length(R.pivot_cols)

function _sparse_rref_push_homogeneous!(R::SparseRREF{K}, row::SparseRow{K})::Bool where {K}
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

mutable struct SparseRREFAugmented{K}
    rref::SparseRREF{K}
    nrhs::Int
    pivot_rhs::Vector{Vector{K}}
end

function SparseRREFAugmented{K}(nvars::Int, nrhs::Int) where {K}
    return SparseRREFAugmented{K}(SparseRREF{K}(nvars), nrhs, Vector{Vector{K}}())
end

function _sparse_rref_push_augmented!(
    R::SparseRREFAugmented{K},
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

function _nullspace_from_pivots(R::SparseRREF{K}, nvars::Int) where {K}
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

# Exact linear algebra over QQ (pure Julia). Nemo acceleration is handled
# by FieldLinAlg when requested.  ASCII-only version.


# Nemo integration is provided via FieldLinAlg (optional).

# Hook: an extension may overload this to convert A to a backend-specific
# matrix representation (e.g. Nemo.fmpq_mat). The fallback is identity.
@inline _to_backend_matrix(A) = A

# Hooked backends:
# - fallback methods live on AbstractMatrix{QQ}
# - Nemo extension will add methods on Nemo.fmpq_mat

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

    return pivots ? (M, collect(pivs)) : M
end

"""
    rrefQQ(A::AbstractMatrix{QQ}; pivots::Bool=true)

Return the row-reduced echelon form of `A`. If `pivots=true`, also return the
pivot column indices as a `Vector{Int}`.
"""
function rrefQQ(A::AbstractMatrix{QQ}; pivots::Bool=true)
    return _rref_backend(_to_backend_matrix(A); pivots=pivots)
end

_rank_backend(A::AbstractMatrix{QQ}) = length(last(rrefQQ(A; pivots=true)))

rankQQ(A::AbstractMatrix{QQ}) = _rank_backend(_to_backend_matrix(A))
rankQQ(A::AbstractMatrix{Float64}) = rankQQ(Matrix{QQ}(A))

function _nullspace_backend(A::AbstractMatrix{QQ})
    R, pivs = rrefQQ(A; pivots=true)
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

nullspaceQQ(A::AbstractMatrix{QQ}) = _nullspace_backend(_to_backend_matrix(A))

function colspaceQQ(A::AbstractMatrix{QQ})
    pivs = pivot_columnsQQ(A)
    return A[:, pivs]
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

# Cache of factors keyed by the left matrix B.
# IMPORTANT: values do not reference the key, so WeakKeyDict works correctly.
const _FULLCOLUMN_FACTOR_CACHE = WeakKeyDict{Any,Any}()

@inline _can_weak_cache_key(x) = Base.ismutabletype(typeof(x))

# ---------------------------------------------------------------
# Pivot-column detection for QQ matrices (dense or sparse)
# ---------------------------------------------------------------

# Dense pivot columns via rrefQQ
function pivot_columnsQQ(A::AbstractMatrix{QQ})::Vector{Int}
    _, pivs = rrefQQ(A)
    return collect(pivs)
end

# Sparse pivot columns via sparse RREF streaming
# Sparse pivot columns via sparse RREF streaming
function pivot_columnsQQ(A::SparseMatrixCSC{QQ,Int})::Vector{Int}
    m, n = size(A)
    if m == 0 || n == 0
        return Int[]
    end
    R = SparseRREF{QQ}(n)
    rows = _sparse_rows(A)
    maxrank = min(m, n)
    for i in 1:m
        _sparse_rref_push_homogeneous!(R, rows[i])
        if length(R.pivot_cols) == maxrank
            break
        end
    end
    return sort!(copy(R.pivot_cols))
end


# Avoid densification for transpose/adjoint sparse wrappers
pivot_columnsQQ(A::Transpose{QQ,<:SparseMatrixCSC{QQ,Int}}) = pivot_columnsQQ(sparse(A))
pivot_columnsQQ(A::Adjoint{QQ,<:SparseMatrixCSC{QQ,Int}})  = pivot_columnsQQ(sparse(A))

"""
    factor_fullcolumnQQ(B::AbstractMatrix{<:QQ}) -> FullColumnFactor{QQ}

Build reusable factorization data for solving `B * X = Y` when B has full column rank.

Most users should call `solve_fullcolumnQQ(B,Y)`; caching is automatic for mutable matrices.
"""
function factor_fullcolumnQQ(B::AbstractMatrix{<:QQ})::FullColumnFactor{QQ}
    m, n = size(B)
    n == 0 && return FullColumnFactor{QQ}(Int[], Matrix{QQ}(I, 0, 0))

    rows = pivot_columnsQQ(transpose(B))
    if length(rows) != n
        error("solve_fullcolumnQQ: expected full column rank, got rank $(length(rows)) < $n")
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
    R, pivs = rrefQQ(By)

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
    solve_fullcolumnQQ(B, Y; cache=true, factor=nothing, check_rhs=true)

Solve `B*X = Y` over QQ under the assumption that B has full column rank.

If cache=true, a reusable factor is used; for mutable B (e.g. Matrix) it is cached
so repeated solves do not redo elimination.
"""
function solve_fullcolumnQQ(B::AbstractMatrix{<:QQ}, Y::AbstractVecOrMat{<:QQ};
                            cache::Bool=true,
                            factor::Union{Nothing,FullColumnFactor{QQ}}=nothing,
                            check_rhs::Bool=true)
    if cache
        fac = factor
        if fac === nothing
            if _can_weak_cache_key(B)
                fac = get!(_FULLCOLUMN_FACTOR_CACHE, B) do
                    factor_fullcolumnQQ(B)
                end
            else
                fac = factor_fullcolumnQQ(B)
            end
        end
        return _solve_fullcolumn_factorQQ(B, fac, Y; check_rhs=check_rhs)
    end

    return _solve_fullcolumn_rrefQQ(B, Y)
end

"""
    clear_fullcolumn_cache!()

Drop cached FullColumnFactor objects (useful in long sessions).
"""
function clear_fullcolumn_cache!()
    empty!(_FULLCOLUMN_FACTOR_CACHE)
    return nothing
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
#   * SparseRREF{K}: a streaming RREF builder using merge-based sparse axpy.
#
# This layer is internal but sits on several hot paths (Hom computations,
# derived-functor lifting, and sparse linear solves).  Keeping it allocation-
# light has an outsized impact on overall library performance.
# ---------------------------------------------------------------------------



function rankQQ(A::SparseMatrixCSC{QQ,Int})
    m, n = size(A)
    if m == 0 || n == 0
        return 0
    end

    R = SparseRREF{QQ}(n)
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

function _nullspace_modp_from_rref(R::Matrix{Int}, pivs::Vector{Int}, n::Int, p::Int)
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

function nullspaceQQ(A::SparseMatrixCSC{QQ,Int})
    m, n = size(A)
    R = SparseRREF{QQ}(n)
    rows = _sparse_rows(A)
    for i in 1:m
        _sparse_rref_push_homogeneous!(R, rows[i])
    end
    return _nullspace_from_pivots(R, n)
end

# Avoid densification for transpose/adjoint sparse wrappers
nullspaceQQ(A::Transpose{QQ,<:SparseMatrixCSC{QQ,Int}}) = nullspaceQQ(sparse(A))
nullspaceQQ(A::Adjoint{QQ,<:SparseMatrixCSC{QQ,Int}})  = nullspaceQQ(sparse(A))
rankQQ(A::Transpose{QQ,<:SparseMatrixCSC{QQ,Int}}) = rankQQ(sparse(A))
rankQQ(A::Adjoint{QQ,<:SparseMatrixCSC{QQ,Int}})  = rankQQ(sparse(A))

function rankQQ_restricted(
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

    # Map global row indices to local 1..nr indices (sparse dictionary: avoids O(m) memory).
    rowpos = Dict{Int,Int}()
    sizehint!(rowpos, nr)
    for (i, r) in enumerate(rows)
        rowpos[r] = i
    end

    row_idx = [Int[] for _ in 1:nr]
    row_val = [QQ[] for _ in 1:nr]

    # Iterate over selected columns once. We remap columns to local indices
    # 1..nc so the RREF builder does not need to allocate pivot_pos of length
    # size(A,2) when only a small subset is used.
    @inbounds for (jloc, col) in enumerate(cols)
        for ptr in A.colptr[col]:(A.colptr[col + 1] - 1)
            r = A.rowval[ptr]
            i = get(rowpos, r, 0)
            if i != 0
                v = A.nzval[ptr]
                if !iszero(v)
                    push!(row_idx[i], jloc)
                    push!(row_val[i], v)
                end
            end
        end
    end

    R = SparseRREF{QQ}(nc)
    maxrank = min(nr, nc)
    for i in 1:nr
        _sparse_rref_push_homogeneous!(R, SparseRow{QQ}(row_idx[i], row_val[i]))
        if _rref_rank(R) == maxrank
            break
        end
    end
    return _rref_rank(R)
end


"""
    rank_modp_sparse(A::SparseMatrixCSC{QQ}, p::Int) -> Int

Compute the rank of a sparse QQ-matrix after reducing coefficients modulo `p`.

This is intended as a *fast certificate/diagnostic* tool:
- For matrices with integer coefficients, rank_modp_sparse(A,p) <= rankQQ(A).
- If rank_modp_sparse(A,p) == ncols(A), then A has full column rank over QQ.

Notes
- Choose `p` as a reasonably large prime (e.g. around 1e6) to reduce the chance
  of accidental rank drops.
- If any denominator in A is divisible by p, reduction is undefined and this
  function throws DomainError.
"""
function rank_modp_sparse(A::SparseMatrixCSC{QQ,Int}, p::Int)::Int
    m, n = size(A)
    if m == 0 || n == 0
        return 0
    end

    # Build per-row dictionaries over Int mod p.
    row_nnz = zeros(Int, m)
    @inbounds for col in 1:n
        for ptr in A.colptr[col]:(A.colptr[col + 1] - 1)
            row_nnz[A.rowval[ptr]] += 1
        end
    end

    rows = [Dict{Int,Int}() for _ in 1:m]
    @inbounds for i in 1:m
        sizehint!(rows[i], row_nnz[i])
    end

    @inbounds for col in 1:n
        for ptr in A.colptr[col]:(A.colptr[col + 1] - 1)
            r = A.rowval[ptr]
            v = A.nzval[ptr]
            if v != 0
                a = _modp_qQ(v, p)
                if a != 0
                    rows[r][col] = a
                end
            end
        end
    end

    # Incremental elimination over GF(p) using the same "pivot dict" idea.
    pivots = Dict{Int,Dict{Int,Int}}()
    scratch = Int[]

    for i in 1:m
        row = rows[i]
        if isempty(row)
            continue
        end

        # eliminate existing pivots
        empty!(scratch)
        for (j, _) in row
            if haskey(pivots, j)
                push!(scratch, j)
            end
        end
        for j in scratch
            if haskey(row, j)
                c = row[j] % p
                if c != 0
                    prow = pivots[j]
                    for (k, vk) in prow
                        newv = mod(get(row, k, 0) - c * vk, p)
                        if newv == 0
                            if haskey(row, k)
                                delete!(row, k)
                            end
                        else
                            row[k] = newv
                        end
                    end
                end
            end
        end
        if isempty(row)
            continue
        end

        piv = minimum(keys(row))
        a = row[piv] % p
        if a == 0
            # Should not happen if we drop zeros properly, but guard anyway.
            delete!(row, piv)
            continue
        end

        inv_a = invmod(a, p)
        for (k, vk) in row
            row[k] = mod(vk * inv_a, p)
        end

        # eliminate new pivot from existing pivot rows
        for (q, prow) in pivots
            if q == piv
                continue
            end
            if haskey(prow, piv)
                c = prow[piv] % p
                if c != 0
                    for (k, vk) in row
                        newv = mod(get(prow, k, 0) - c * vk, p)
                        if newv == 0
                            if haskey(prow, k)
                                delete!(prow, k)
                            end
                        else
                            prow[k] = newv
                        end
                    end
                end
            end
        end

        pivots[piv] = copy(row)

        if length(pivots) == min(m, n)
            break
        end
    end

    return length(pivots)
end

# ---------------------------------------------------------------
# Fast rank computations for dimension-only queries
# ---------------------------------------------------------------

const DEFAULT_MODULAR_PRIMES = Int[
    998244353, 1000000007, 1000000009, 1004535809,
    1045430273, 1053818881, 1224736769, 1300234241,
]

"""
    rank_modp_dense(A::AbstractMatrix{QQ}, p::Int) -> Int

Rank over F_p for dense matrices by Gaussian elimination.
Used by rankQQ_dim for fast dimension computations.
"""
function rank_modp_dense(A::AbstractMatrix{QQ}, p::Int)::Int
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

rank_modp(A::AbstractMatrix{QQ}, p::Int) = rank_modp_dense(A, p)
rank_modp(A::SparseMatrixCSC{QQ,Int}, p::Int) = rank_modp_sparse(A, p)
rank_modp(A::Transpose{QQ,<:SparseMatrixCSC{QQ,Int}}, p::Int) = rank_modp_sparse(sparse(A), p)
rank_modp(A::Adjoint{QQ,<:SparseMatrixCSC{QQ,Int}}, p::Int)  = rank_modp_sparse(sparse(A), p)

# ---------------------------------------------------------------
# Modular nullspace / solve with rational reconstruction (QQ only)
# ---------------------------------------------------------------

function _crt_combine!(A::Matrix{BigInt}, mod::BigInt, B::Matrix{Int}, p::Int)
    mp = Int(mod(mod, p))
    invm = invmod(mp, p)
    m = mod
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
    bound = isqrt(m ÷ 2)

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
    rankQQ_dim(A; backend=:auto, max_primes=4, primes=DEFAULT_MODULAR_PRIMES,
                 small_threshold=20_000) -> Int

Fast rank intended for dimension-only queries:
- backend=:exact  uses exact rankQQ
- backend=:modular uses ranks mod several primes
- backend=:auto uses exact for small matrices, modular otherwise
"""
function rankQQ_dim(A::AbstractMatrix{QQ};
                    backend::Symbol=:auto,
                    max_primes::Int=4,
                    primes::Vector{Int}=DEFAULT_MODULAR_PRIMES,
                    small_threshold::Int=20_000)::Int
    m, n = size(A)
    (m == 0 || n == 0) && return 0

    backend == :exact && return rankQQ(A)

    if backend == :auto
        (m*n <= small_threshold) && return rankQQ(A)
        backend = :modular
    end

    backend != :modular && error("rankQQ_dim: unsupported backend $(backend)")

    r = 0
    used = 0
    for p in primes
        used += 1
        try
            r = max(r, rank_modp(A, p))
        catch
            continue
        end
        r == min(m,n) && break
        used >= max_primes && break
    end
    return r
end

nullityQQ_dim(A::AbstractMatrix{QQ}; kwargs...) = size(A,2) - rankQQ_dim(A; kwargs...)


# -----------------------------------------------------------------------------
# Public API (field-generic dispatch)
# -----------------------------------------------------------------------------

function rref(field::AbstractCoeffField, A; pivots::Bool=true, backend::Symbol=:auto)
    if field isa QQField
        be = choose_linalg_backend(field, A; op=:rref, backend=backend)
        if be == :nemo
            return _nemo_rref(field, A; pivots=pivots)
        end
        return rrefQQ(A; pivots=pivots)
    end
    if field isa PrimeField && field.p == 2
        return _rref_f2(A; pivots=pivots)
    end
    if field isa PrimeField && field.p == 3
        return _rref_f3(A; pivots=pivots)
    end
    if field isa PrimeField && field.p > 3
        be = choose_linalg_backend(field, A; op=:rref, backend=backend)
        if be == :nemo
            return _nemo_rref(field, A; pivots=pivots)
        end
        return _rref_fp(A; pivots=pivots)
    end
    if field isa RealField
        return _rref_float(field, A; pivots=pivots)
    end
    error("FieldLinAlg.rref: unsupported field $(typeof(field))")
end

function rank(field::AbstractCoeffField, A; backend::Symbol=:auto)
    if field isa QQField
        be = choose_linalg_backend(field, A; op=:rank, backend=backend)
        if be == :nemo
            return _nemo_rank(field, A)
        end
        return rankQQ(A)
    end
    if field isa PrimeField && field.p == 2
        return _rank_f2(A)
    end
    if field isa PrimeField && field.p == 3
        return _rank_f3(A)
    end
    if field isa PrimeField && field.p > 3
        be = choose_linalg_backend(field, A; op=:rank, backend=backend)
        if be == :nemo
            return _nemo_rank(field, A)
        end
        return _rank_fp(A)
    end
    if field isa RealField
        return _rank_float(field, A)
    end
    error("FieldLinAlg.rank: unsupported field $(typeof(field))")
end

function nullspace(field::AbstractCoeffField, A; backend::Symbol=:auto)
    if field isa QQField
        be = choose_linalg_backend(field, A; op=:nullspace, backend=backend)
        if be == :nemo
            return _nemo_nullspace(field, A)
        end
        if be == :modular
            if !(A isa SparseMatrixCSC)
                N = _nullspace_modularQQ(A)
                N === nothing || return N
            end
        end
        return nullspaceQQ(A)
    end
    if field isa PrimeField && field.p == 2
        return _nullspace_f2(A)
    end
    if field isa PrimeField && field.p == 3
        return _nullspace_f3(A)
    end
    if field isa PrimeField && field.p > 3
        be = choose_linalg_backend(field, A; op=:nullspace, backend=backend)
        if be == :nemo
            return _nemo_nullspace(field, A)
        end
        return _nullspace_fp(A)
    end
    if field isa RealField
        return _nullspace_float(field, A)
    end
    error("FieldLinAlg.nullspace: unsupported field $(typeof(field))")
end

function colspace(field::AbstractCoeffField, A; backend::Symbol=:auto)
    if field isa QQField
        be = choose_linalg_backend(field, A; op=:colspace, backend=backend)
        if be == :nemo
            _, pivs = _nemo_rref(field, A; pivots=true)
            return A[:, pivs]
        end
        return colspaceQQ(A)
    end
    if field isa PrimeField && field.p == 2
        pivs = _pivot_cols_f2(A)
        return A[:, pivs]
    end
    if field isa PrimeField && field.p == 3
        _, pivs = _rref_f3(A; pivots=true)
        return A[:, pivs]
    end
    if field isa PrimeField && field.p > 3
        be = choose_linalg_backend(field, A; op=:colspace, backend=backend)
        if be == :nemo
            _, pivs = _nemo_rref(field, A; pivots=true)
            return A[:, pivs]
        end
        _, pivs = _rref_fp(A; pivots=true)
        return A[:, pivs]
    end
    if field isa RealField
        return _colspace_float(field, A)
    end
    error("FieldLinAlg.colspace: unsupported field $(typeof(field))")
end

function solve_fullcolumn(field::AbstractCoeffField, B, Y;
                          check_rhs::Bool=true, backend::Symbol=:auto,
                          cache::Bool=true, factor=nothing)
    if field isa QQField
        be = choose_linalg_backend(field, B; op=:solve, backend=backend)
        if be == :modular
            if !(B isa SparseMatrixCSC)
                X = _solve_fullcolumn_modularQQ(B, Y; check_rhs=check_rhs)
                X === nothing || return X
            end
        end
        return solve_fullcolumnQQ(B, Y; check_rhs=check_rhs)
    end
    if field isa PrimeField && field.p == 2
        return _solve_fullcolumn_f2(B, Y; check_rhs=check_rhs, cache=cache, factor=factor)
    end
    if field isa PrimeField && field.p == 3
        return _solve_fullcolumn_f3(B, Y; check_rhs=check_rhs, cache=cache, factor=factor)
    end
    if field isa PrimeField && field.p > 3
        be = choose_linalg_backend(field, B; op=:solve, backend=backend)
        if be == :nemo
            # Use rref to solve: nemo backend returns dense rref.
            return _solve_fullcolumn_fp(B, Y; check_rhs=check_rhs)
        end
        return _solve_fullcolumn_fp(B, Y; check_rhs=check_rhs)
    end
    if field isa RealField
        return _solve_fullcolumn_float(field, B, Y; check_rhs=check_rhs)
    end
    error("FieldLinAlg.solve_fullcolumn: unsupported field $(typeof(field))")
end

function rank_dim(field::AbstractCoeffField, A; backend::Symbol=:auto, kwargs...)
    if field isa QQField
        return rankQQ_dim(A; backend=backend, kwargs...)
    end
    if field isa PrimeField && field.p == 2
        return _rank_f2(A)
    end
    if field isa PrimeField && field.p == 3
        return _rank_f3(A)
    end
    if field isa PrimeField && field.p > 3
        be = choose_linalg_backend(field, A; op=:rank, backend=backend)
        if be == :nemo
            return _nemo_rank(field, A)
        end
        return _rank_fp(A)
    end
    if field isa RealField
        return _rank_float(field, A)
    end
    return rank(field, A; backend=backend)
end

function rank_restricted(field::AbstractCoeffField, A::SparseMatrixCSC,
                         rows::AbstractVector{Int}, cols::AbstractVector{Int};
                         backend::Symbol=:auto, kwargs...)
    if field isa QQField
        return rankQQ_restricted(A, rows, cols; kwargs...)
    end
    if field isa PrimeField && field.p == 2
        return _rank_restricted_f2(A, rows, cols; kwargs...)
    end
    if field isa PrimeField && field.p == 3
        return _rank_restricted_f3(A, rows, cols; kwargs...)
    end
    if field isa PrimeField && field.p > 3
        return _rank_fp(Matrix(A[rows, cols]))
    end
    error("FieldLinAlg.rank_restricted: unsupported field $(typeof(field))")
end

end # module FieldLinAlg
