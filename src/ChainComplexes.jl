module ChainComplexes

using LinearAlgebra
using SparseArrays
import Base.Threads

using ..CoreModules: AbstractCoeffField, RealField, QQField, QQ, coeff_type, field_from_eltype
using ..FieldLinAlg: SparseRow, _SparseRREFAugmented, _sparse_rref_push_augmented!
using ..FieldLinAlg

# ----------------------------
# Small, reliable linear solvers
# ----------------------------

# Solve A * X = B over a field, returning one particular solution with all free vars set to 0.
# For exact fields: uses RREF and throws if inconsistent.
# For RealField: uses least-squares and checks residual.
@inline _field_of(A::AbstractMatrix) = field_from_eltype(eltype(A))

function solve_particular(field::AbstractCoeffField, A::AbstractMatrix, B::AbstractMatrix)
    A0 = Matrix(A)
    B0 = Matrix(B)
    m, n = size(A0)
    @assert size(B0, 1) == m

    if field isa RealField
        X = A0 \ B0
        R = A0 * X - B0
        maxabs = isempty(R) ? 0.0 : maximum(abs, R)
        tol = field.atol + field.rtol * (isempty(A0) ? 0.0 : opnorm(A0, 1))
        maxabs <= tol || error("solve_particular: inconsistent system (residual=$maxabs, tol=$tol)")
        return X
    end

    Aug = hcat(A0, B0)
    R, pivs_all = FieldLinAlg.rref(field, Aug)
    rhs = size(B0, 2)

    for i in 1:m
        if all(R[i, 1:n] .== 0)
            if any(R[i, n+1:n+rhs] .!= 0)
                error("solve_particular: inconsistent system")
            end
        end
    end

    pivs = Int[]
    for p in pivs_all
        p <= n && push!(pivs, p)
    end

    X = zeros(eltype(A0), n, rhs)
    for (row, pcol) in enumerate(pivs)
        X[pcol, :] = R[row, n+1:n+rhs]
    end
    return X
end

# Internal: build transpose(A) as a materialized CSC matrix in O(nnz(A)).
# We do this explicitly because on newer Julia versions, transpose(::SparseMatrixCSC)
# may return a lazy Transpose wrapper that does not expose CSC internals (colptr/rowval/nzval).
function _transpose_csc(A::SparseMatrixCSC{T,Int}) where {T}
    m, n = size(A)

    colptrA = A.colptr
    rowvalA = A.rowval
    nzvalA = A.nzval

    # Count nnz in each row of A (these become column nnz counts in A').
    rowcounts = zeros(Int, m)
    @inbounds for j in 1:n
        for ptr in colptrA[j]:(colptrA[j + 1] - 1)
            rowcounts[rowvalA[ptr]] += 1
        end
    end

    # Build colptr for A' (which has m columns).
    colptrT = Vector{Int}(undef, m + 1)
    colptrT[1] = 1
    @inbounds for i in 1:m
        colptrT[i + 1] = colptrT[i] + rowcounts[i]
    end

    nnzT = colptrT[end] - 1
    rowvalT = Vector{Int}(undef, nnzT)
    nzvalT = Vector{T}(undef, nnzT)

    # Fill columns of A' by scanning columns of A in increasing order.
    # This guarantees row indices inside each column of A' are sorted.
    next = copy(colptrT)
    @inbounds for j in 1:n
        for ptr in colptrA[j]:(colptrA[j + 1] - 1)
            i = rowvalA[ptr]         # row in A = column in A'
            pos = next[i]
            rowvalT[pos] = j         # row in A' = column in A
            nzvalT[pos] = nzvalA[ptr]
            next[i] = pos + 1
        end
    end

    return SparseMatrixCSC{T,Int}(n, m, colptrT, rowvalT, nzvalT)
end


"""
    solve_particular(A::SparseMatrixCSC{K,Int}, B::AbstractMatrix{K}) -> Union{Matrix{K},Nothing}

Solve `A * X = B` over a field and return a particular solution `X` (free variables set to zero).
Return `nothing` if the system is inconsistent.

Performance notes:
- Does not materialize `A` as dense.
- Iterates rows of `A` via `transpose(A)` (so rows become CSC columns) and streams the resulting
  sparse equations into a sparse RREF builder (`FieldLinAlg._SparseRREFAugmented`).
"""
function solve_particular(A::SparseMatrixCSC{K,Int}, B::AbstractMatrix{K}) where {K}
    field = _field_of(B)
    if field isa RealField
        return solve_particular(field, Matrix(A), B)
    end
    m, n = size(A)
    mB, nrhs = size(B)
    @assert mB == m

    # Iterate rows of A efficiently: row i of A is column i of A' in CSC format.
    # Materialize A' explicitly (see _transpose_csc) to avoid Transpose wrappers.
    At = _transpose_csc(A)
    colptr = At.colptr
    rowval = At.rowval
    nzval = At.nzval

    R = _SparseRREFAugmented{K}(n, nrhs)
    row = SparseRow{K}()
    rhs = Vector{K}(undef, nrhs)

    for i in 1:m
        # Build the i-th equation row in sparse form (sorted indices, no duplicates).
        lo = colptr[i]
        hi = colptr[i + 1] - 1
        nnz_i = hi - lo + 1

        resize!(row.idx, nnz_i)
        resize!(row.val, nnz_i)

        k = 0
        @inbounds for ptr in lo:hi
            a = nzval[ptr]
            iszero(a) && continue
            k += 1
            row.idx[k] = rowval[ptr]   # column index in A
            row.val[k] = a
        end
        resize!(row.idx, k)
        resize!(row.val, k)

        # Copy RHS row (the elimination routine mutates rhs in-place).
        @inbounds for j in 1:nrhs
            rhs[j] = B[i, j]
        end

        status = _sparse_rref_push_augmented!(R, row, rhs)
        status === :inconsistent && return nothing
    end

    # Particular solution: free variables = 0, pivot variables = pivot RHS.
    X = zeros(K, n, nrhs)
    @inbounds for pos in 1:length(R.rref.pivot_cols)
        pcol = R.rref.pivot_cols[pos]
        rhsrow = R.pivot_rhs[pos]
        for j in 1:nrhs
            X[pcol, j] = rhsrow[j]
        end
    end
    return X
end
# Extend columns of C (k x r) to an invertible (k x k) matrix by adding standard basis vectors.
# This uses one elimination pass on transpose(Cbasis) to identify row pivots and then
# selects complement standard basis vectors from nonpivot rows.
function extend_to_basis(C::Matrix{K}) where {K}
    k = size(C, 1)
    if k == 0
        return zeros(K, 0, 0)
    end
    field = field_from_eltype(K)

    Cbasis = if size(C, 2) == 0
        zeros(K, k, 0)
    else
        FieldLinAlg.colspace(field, C)
    end
    r = size(Cbasis, 2)

    if r == k
        return Cbasis
    end
    if r == 0
        I = zeros(K, k, k)
        @inbounds for i in 1:k
            I[i, i] = one(K)
        end
        return I
    end

    # Pivot columns in transpose(Cbasis) correspond to a maximal set of rows of Cbasis
    # whose restriction gives an invertible r x r minor.
    _, pivs = FieldLinAlg.rref(field, Matrix(transpose(Cbasis)); pivots=true)
    pivot_mask = falses(k)
    @inbounds for p in pivs
        if 1 <= p <= k
            pivot_mask[p] = true
        end
    end

    comp_rows = Int[]
    @inbounds for i in 1:k
        if !pivot_mask[i]
            push!(comp_rows, i)
        end
    end

    Q = zeros(K, k, length(comp_rows))
    @inbounds for (j, irow) in enumerate(comp_rows)
        Q[irow, j] = one(K)
    end
    B = hcat(Cbasis, Q)

    if size(B, 2) != k
        error("extend_to_basis: could not extend to a full basis")
    end
    return B
end

# ----------------------------
# Cochain complexes
# ----------------------------

struct CochainComplex{K,A}
    tmin::Int
    tmax::Int
    dims::Vector{Int}                         # dims[t - tmin + 1] = dim C^t
    d::Vector{SparseMatrixCSC{K, Int}}        # d[idx] : C^t -> C^{t+1}
    labels::Vector{Vector{Int}}               # typed compute labels, per degree
    annotations::Union{Nothing,Vector{Vector{A}}}  # optional heterogeneous boundary metadata
end

# Max cohomological degree stored in a cochain complex.
maxdeg_of_complex(C::CochainComplex) = C.tmax

function degree_index(C::CochainComplex, t::Int)
    if t < C.tmin || t > C.tmax
        error("degree out of range")
    end
    return t - C.tmin + 1
end

"""
    CochainComplex{K}(tmin, tmax, dims, d; labels=nothing)

Construct a bounded cochain complex C with degrees tmin..tmax.

- dims[i] = dim C^{tmin + i - 1}.
- d[i] is d^{tmin + i - 1} : C^{tmin + i - 1} -> C^{tmin + i}.

`labels` is optional per-degree metadata: a vector of length length(dims),
where labels[i] is a vector of basis labels for C^{tmin + i - 1}. Core compute
paths keep typed integer labels; heterogeneous user metadata is preserved in
`annotations` and stays out of hot loops.

If labels is omitted, each degree stores an empty label list.
"""
function CochainComplex{K}(tmin::Int,
                           tmax::Int,
                           dims::Vector{Int},
                           d::Vector{SparseMatrixCSC{K,Int}};
                           labels=nothing) where {K}
    if tmax < tmin
        error("CochainComplex: require tmax >= tmin.")
    end
    if length(dims) != tmax - tmin + 1
        error("CochainComplex: dims must have length tmax - tmin + 1.")
    end
    if length(d) != tmax - tmin
        error("CochainComplex: d must have length tmax - tmin.")
    end
    for i in 1:length(d)
        src = dims[i]
        tgt = dims[i+1]
        if size(d[i], 1) != tgt || size(d[i], 2) != src
            error("CochainComplex: differential size mismatch at index $(i).")
        end
    end

    labs = Vector{Vector{Int}}(undef, length(dims))
    anns = nothing
    annT = Any
    if labels === nothing
        for i in 1:length(dims)
            labs[i] = Int[]
        end
    else
        if length(labels) != length(dims)
            error("CochainComplex: labels must have the same length as dims.")
        end
        # Fast typed boundary: integer labels remain in the compute path.
        all_int = true
        for i in 1:length(dims)
            li = labels[i]
            for x in li
                if !(x isa Int)
                    all_int = false
                    break
                end
            end
            all_int || break
        end
        if all_int
            for i in 1:length(dims)
                labs[i] = Int.(labels[i])
            end
        else
            accT = Union{}
            for i in 1:length(dims)
                li = labels[i]
                for x in li
                    accT = typejoin(accT, typeof(x))
                end
            end
            annT = accT === Union{} ? Any : accT
            anns = Vector{Vector{annT}}(undef, length(dims))
            for i in 1:length(dims)
                anns[i] = Vector{annT}(labels[i])
                # Keep compute-time metadata concrete and dimension-aligned.
                labs[i] = collect(1:dims[i])
            end
        end
    end

    return CochainComplex{K,annT}(tmin, tmax, dims, d, labs, anns)
end


# Holds cycle, boundary, and quotient data in a concrete basis.
struct CohomologyData{K}
    t::Int
    dimC::Int
    dimZ::Int
    dimB::Int
    dimH::Int
    K::Matrix{K}          # cycle basis in C^t
    B::Matrix{K}          # boundary basis in C^t
    Cx::Matrix{K}         # boundary subspace basis in K-coordinates
    Q::Matrix{K}          # complement basis in K-coordinates
    Bfull::Matrix{K}      # [Cx Q], square dimZ x dimZ, invertible
    Hrep::Matrix{K}       # cocycle representatives: K * Q
    Kfactor::Base.RefValue{Union{Nothing,FieldLinAlg.FullColumnFactor{QQ}}}
    Bfull_factor::Base.RefValue{Union{Nothing,FieldLinAlg.FullColumnFactor{QQ}}}
end

struct _DiffSummary{K}
    rank::Int
    ker::Matrix{K}
    img::Matrix{K}
end

@inline _empty_mat(::Type{K}, m::Int, n::Int) where {K} = zeros(K, m, n)
@inline _eye_mat(::Type{K}, n::Int) where {K} = Matrix{K}(I, n, n)
@inline _concrete_mat(A::Matrix{K}) where {K} = A
@inline _concrete_mat(A::AbstractMatrix{K}) where {K} = Matrix{K}(A)
@inline _use_batched_coordinate_solves(::Type{K}, nsrc::Int, ntgt::Int) where {K} =
    nsrc > 1 && ntgt > 0 && !(field_from_eltype(K) isa QQField)
@inline _use_exact_batched_coordinate_solves(::Type{K}, nsrc::Int, ntgt::Int) where {K} =
    nsrc > 1 && ntgt > 0 && (field_from_eltype(K) isa QQField)
@inline _use_long_exact_precompute(::Type{K}, X::CochainComplex{K}) where {K} =
    !(field_from_eltype(K) isa QQField)
@inline _use_page_workspace(::Type{K}) where {K} = !(field_from_eltype(K) isa QQField)
@inline _use_precolspace_den(::Type{K}) where {K} = !(field_from_eltype(K) isa QQField)

function _zero_cohomology_data(::Type{K}, t::Int) where {K}
    Z0 = _empty_mat(K, 0, 0)
    return CohomologyData{K}(t, 0, 0, 0, 0, Z0, Z0, Z0, Z0, Z0, Z0, Ref{Union{Nothing,FieldLinAlg.FullColumnFactor{QQ}}}(nothing), Ref{Union{Nothing,FieldLinAlg.FullColumnFactor{QQ}}}(nothing))
end

@inline _fullcolumn_factor_ref() = Ref{Union{Nothing,FieldLinAlg.FullColumnFactor{QQ}}}(nothing)

@inline function _fullcolumn_factor!(field::AbstractCoeffField, B, ref::Base.RefValue{Union{Nothing,FieldLinAlg.FullColumnFactor{QQ}}})
    field isa QQField || return nothing
    factor = ref[]
    if factor === nothing && size(B, 2) > 0
        factor = FieldLinAlg._factor_fullcolumnQQ(B)
        ref[] = factor
    end
    return factor
end

function _diff_summary(field::AbstractCoeffField, d::AbstractMatrix{K}) where {K}
    m, n = size(d)

    ker = if n == 0
        _empty_mat(K, 0, 0)
    elseif m == 0
        _eye_mat(K, n)
    else
        FieldLinAlg.nullspace(field, d)
    end

    img = if m == 0 || n == 0
        _empty_mat(K, m, 0)
    else
        FieldLinAlg.colspace(field, d)
    end

    return _DiffSummary{K}(size(img, 2), ker, img)
end

function _diff_summaries(C::CochainComplex{K}) where {K}
    field = field_from_eltype(K)
    out = Vector{_DiffSummary{K}}(undef, length(C.d))
    for i in eachindex(out)
        out[i] = _diff_summary(field, C.d[i])
    end
    return out
end

function _cohomology_data_from_bases(::Type{K},
                                     t::Int,
                                     dimCt::Int,
                                     Zin::AbstractMatrix{K},
                                     Bin::AbstractMatrix{K}) where {K}
    Z = _concrete_mat(Zin)
    B = _concrete_mat(Bin)
    dimZ = size(Z, 2)
    dimB = size(B, 2)
    field = field_from_eltype(K)

    if dimCt == 0
        return _zero_cohomology_data(K, t)
    end

    if dimB == 0
        Bcoords = _empty_mat(K, dimZ, 0)
        Bfull = _eye_mat(K, dimZ)
        Hrep = Z
        return CohomologyData{K}(t, dimCt, dimZ, 0, dimZ, Z, B, Bcoords, Bfull, Bfull, Hrep,
                                 _fullcolumn_factor_ref(),
                                 _fullcolumn_factor_ref())
    end

    X = _solve_fullcolumn_cached(field, Z, B)
    Cx = size(X, 2) == 0 ? _empty_mat(K, dimZ, 0) : FieldLinAlg.colspace(field, X)

    rB = size(Cx, 2)
    if rB == dimZ
        return CohomologyData{K}(t, dimCt, dimZ, rB, 0, Z, B, Cx, _empty_mat(K, dimZ, 0), Cx, _empty_mat(K, dimCt, 0),
                                 _fullcolumn_factor_ref(),
                                 _fullcolumn_factor_ref())
    end

    Bfull = extend_to_basis(Cx)
    Q = Bfull[:, rB+1:end]
    Hrep = Z * Q
    dimH = size(Q, 2)

    return CohomologyData{K}(t, dimCt, dimZ, rB, dimH, Z, B, Cx, Q, Bfull, Hrep,
                             _fullcolumn_factor_ref(),
                             _fullcolumn_factor_ref())
end

function _cohomology_data_from_diffs(::Type{K},
                                     t::Int,
                                     dimCt::Int,
                                     d_prev::AbstractMatrix{K},
                                     d_curr::AbstractMatrix{K}) where {K}
    field = field_from_eltype(K)

    if field isa QQField
        Z = if dimCt == 0
            _empty_mat(K, 0, 0)
        elseif size(d_curr, 1) == 0
            _eye_mat(K, dimCt)
        else
            FieldLinAlg.nullspace(field, d_curr)
        end

        B = if dimCt == 0
            _empty_mat(K, 0, 0)
        elseif size(d_prev, 2) == 0
            _empty_mat(K, dimCt, 0)
        else
            FieldLinAlg.colspace(field, d_prev)
        end

        dimZ = size(Z, 2)
        dimB = size(B, 2)

        if dimCt == 0
            return _zero_cohomology_data(K, t)
        end

        if dimB == 0
            Bcoords = _empty_mat(K, dimZ, 0)
            Bfull = _eye_mat(K, dimZ)
            return CohomologyData{K}(t, dimCt, dimZ, 0, dimZ, Z, B, Bcoords, Bfull, Bfull, Z,
                                     _fullcolumn_factor_ref(),
                                     _fullcolumn_factor_ref())
        end

        X = _solve_fullcolumn_cached(field, Z, B)
        Cx = size(X, 2) == 0 ? _empty_mat(K, dimZ, 0) : FieldLinAlg.colspace(field, X)

        rB = size(Cx, 2)
        if rB == dimZ
            return CohomologyData{K}(t, dimCt, dimZ, rB, 0, Z, B, Cx, _empty_mat(K, dimZ, 0), Cx, _empty_mat(K, dimCt, 0),
                                     _fullcolumn_factor_ref(),
                                     _fullcolumn_factor_ref())
        end
        Bfull = extend_to_basis(Cx)
        Q = Bfull[:, rB+1:end]
        Hrep = Z * Q
        dimH = size(Q, 2)
        return CohomologyData{K}(t, dimCt, dimZ, rB, dimH, Z, B, Cx, Q, Bfull, Hrep,
                                 _fullcolumn_factor_ref(),
                                 _fullcolumn_factor_ref())
    end

    Z = if dimCt == 0
        zeros(K, 0, 0)
    elseif size(d_curr, 1) == 0
        I = zeros(K, dimCt, dimCt)
        for i in 1:dimCt
            I[i, i] = one(K)
        end
        I
    else
        FieldLinAlg.nullspace(field, d_curr)
    end

    B = if dimCt == 0
        zeros(K, 0, 0)
    elseif size(d_prev, 2) == 0
        zeros(K, dimCt, 0)
    else
        FieldLinAlg.colspace(field, d_prev)
    end

    return _cohomology_data_from_bases(K, t, dimCt, Z, B)
end

# Compute cohomology data at degree t:
# Z^t = ker(d^t), B^t = im(d^{t-1}), H^t = Z^t / B^t
function _cohomology_data(C::CochainComplex{K},
                          idx::Int,
                          summaries::Union{Nothing,AbstractVector{_DiffSummary{K}}}=nothing) where {K}
    t = C.tmin + idx - 1
    dimCt = C.dims[idx]

    if summaries === nothing
        d_prev = (idx == 1) ? _empty_mat(K, dimCt, 0) : C.d[idx-1]
        d_curr = (idx > length(C.d)) ? _empty_mat(K, 0, dimCt) : C.d[idx]
        return _cohomology_data_from_diffs(K, t, dimCt, d_prev, d_curr)
    end

    Z = idx > length(summaries) ? _eye_mat(K, dimCt) : summaries[idx].ker
    B = idx == 1 ? _empty_mat(K, dimCt, 0) : summaries[idx - 1].img
    return _cohomology_data_from_bases(K, t, dimCt, Z, B)
end

function cohomology_data(C::CochainComplex{K}, t::Int) where {K}
    return _cohomology_data(C, degree_index(C, t))
end

"""
    cohomology_data(C::CochainComplex{K}) -> Vector{CohomologyData{K}}

Return cohomology data in every degree t in C.tmin:C.tmax.

The output is ordered by increasing degree:
the entry at index i corresponds to t = C.tmin + i - 1.

This whole-complex overload is required by higher-level routines (notably
`spectral_sequence`) that need all cohomology groups (and chosen bases) at once.

If you only need a single degree, use `cohomology_data(C, t)` instead.
"""
function cohomology_data(C::CochainComplex{K}) where {K}
    out = Vector{CohomologyData{K}}(undef, C.tmax - C.tmin + 1)
    summaries = _diff_summaries(C)
    if Threads.nthreads() > 1 && length(out) >= 2
        Threads.@threads for i in eachindex(out)
            out[i] = _cohomology_data(C, i, summaries)
        end
    else
        for i in eachindex(out)
            out[i] = _cohomology_data(C, i, summaries)
        end
    end
    return out
end

"""
    cohomology_dims(C::CochainComplex{K}; backend=:auto, max_primes=4, small_threshold=20_000) -> Vector{Int}

Return the cohomology dimensions for all degrees.

Uses:
    dim H^t = dim C^t - rank(d^{t+1}) - rank(d^t)

Calls FieldLinAlg.rank_dim for fast modular/exact hybrid computation.
"""
function cohomology_dims(C::CochainComplex{K};
                         backend::Symbol=:auto,
                         max_primes::Int=4,
                         small_threshold::Int=20_000)::Vector{Int} where {K}
    ndeg = C.tmax - C.tmin + 1
    out = Vector{Int}(undef, ndeg)
    field = field_from_eltype(K)
    ranks = Vector{Int}(undef, max(0, ndeg - 1))

    for i in eachindex(ranks)
        ranks[i] = FieldLinAlg.rank_dim(field, C.d[i];
                                        backend=backend,
                                        max_primes=max_primes,
                                        small_threshold=small_threshold)
    end

    for i in 1:ndeg
        r_in = i > 1 ? ranks[i - 1] : 0
        r_out = i <= ndeg - 1 ? ranks[i] : 0
        out[i] = C.dims[i] - r_in - r_out
    end
    return out
end


"""
    homology_dims(C::CochainComplex{K}) -> Vector{Int}

Alias for `cohomology_dims` on cochain complexes.

Rationale: in derived-category style code one often speaks informally about
"the homology of a complex" even when the grading is cohomological.
This alias is purely a convenience and does not change any grading conventions.
"""
homology_dims(C::CochainComplex{K}) where {K} = cohomology_dims(C)


# Reduce a cocycle z in C^t to coordinates in H^t, using precomputed cohomology data.
#
# Important: even when dimH == 0, we MUST still validate that z is a cocycle
# (i.e. z lies in Z^t = ker(d^t)). Therefore we do NOT early-return on dimH==0
# without first enforcing z in im(K).
function cohomology_coordinates(H::CohomologyData{K}, z::AbstractMatrix{K}) where {K}
    # Allow a 1 times dimC row matrix to be treated as a vector input.
    if size(z, 1) != H.dimC
        if size(z, 1) == 1 && size(z, 2) == H.dimC
            return cohomology_coordinates(H, vec(Matrix{K}(z)))
        end
        error("cohomology_coordinates: wrong ambient dimension; got size $(size(z)), expected $(H.dimC) times k")
    end

    # If Z^t = 0, then the only cocycle is 0.
    if H.dimZ == 0
        if !all(iszero, z)
            error("cohomology_coordinates: input is not a cocycle (Z^t = 0)")
        end
        return zeros(K, 0, size(z, 2))
    end

    # Enforce cocycle condition and compute Z-coordinates:
    # z in Z^t  iff  z in im(K), and then z = K * alpha for unique alpha.
    field = field_from_eltype(K)
    alpha = _solve_fullcolumn_cached(field, H.K, Matrix{K}(z), H.Kfactor)

    # If H^t = 0, every cocycle represents the zero class, but we already validated z in Z^t.
    if H.dimH == 0
        return zeros(K, 0, size(z, 2))
    end

    # Decompose alpha in the basis Bfull = [boundary-subspace | complement].
    # The last dimH entries are the cohomology coordinates.
    gamma = _solve_fullcolumn_cached(field, H.Bfull, alpha, H.Bfull_factor)
    rB = H.dimB
    return gamma[rB+1:end, :]                       # dimH times k
end

# Vector overload: treat a vector cocycle as a single RHS column.
function cohomology_coordinates(H::CohomologyData{K}, z::AbstractVector{K}) where {K}
    return cohomology_coordinates(H, reshape(Vector{K}(z), :, 1))
end

"""
    cohomology_representative(H::CohomologyData{K}, coords::AbstractVector{K})

Given coordinates in H^t (with respect to the basis encoded by `H.Hrep`), return
a cocycle representative in C^t.
"""
function cohomology_representative(H::CohomologyData{K}, coords::AbstractVector{K}) where {K}
    length(coords) == H.dimH || throw(DimensionMismatch(
        "cohomology_representative: expected coords of length $(H.dimH), got $(length(coords))"))
    if H.dimH == 0
        return zeros(K, H.dimC)
    end
    return H.Hrep * coords
end

"""
    cohomology_representative(H::CohomologyData{K}, coords::AbstractMatrix{K})

Column-wise version: each column of `coords` is a coordinate vector in H^t.
Returns a matrix whose columns are cocycle representatives in C^t.
"""
function cohomology_representative(H::CohomologyData{K}, coords::AbstractMatrix{K}) where {K}
    size(coords, 1) == H.dimH || throw(DimensionMismatch(
        "cohomology_representative: expected coords with $(H.dimH) rows, got $(size(coords, 1))"))
    if H.dimH == 0
        return zeros(K, H.dimC, size(coords, 2))
    end
    return H.Hrep * coords
end


# Given a linear map f: C^t -> D^t and cohomology data for both sides, compute induced map on H^t.
function induced_map_on_cohomology(src::CohomologyData{K}, tgt::CohomologyData{K}, f::AbstractMatrix{K}) where {K}
    if src.dimH == 0 || tgt.dimH == 0
        return zeros(K, tgt.dimH, src.dimH)
    end
    if _use_batched_coordinate_solves(K, src.dimH, tgt.dimH) ||
       _use_exact_batched_coordinate_solves(K, src.dimH, tgt.dimH)
        return cohomology_coordinates(tgt, f * src.Hrep)
    end
    M = zeros(K, tgt.dimH, src.dimH)
    for j in 1:src.dimH
        y = f * src.Hrep[:, j]
        coords = cohomology_coordinates(tgt, y)
        M[:, j] = coords[:, 1]
    end
    return M
end

# ----------------------------
# Homology (chain complexes) for Tor, etc.
# ----------------------------

struct HomologyData{K}
    s::Int
    dimC::Int
    dimZ::Int
    dimB::Int
    dimH::Int
    Z::Matrix{K}          # cycles in C_s
    B::Matrix{K}          # boundaries in C_s
    Cx::Matrix{K}         # boundaries in Z-coordinates
    Q::Matrix{K}          # complement in Z-coordinates
    Bfull::Matrix{K}
    Hrep::Matrix{K}       # cycle representatives in C_s
    Zfactor::Base.RefValue{Union{Nothing,FieldLinAlg.FullColumnFactor{QQ}}}
    Bfull_factor::Base.RefValue{Union{Nothing,FieldLinAlg.FullColumnFactor{QQ}}}
end

function _zero_homology_data(::Type{K}, s::Int) where {K}
    Z0 = _empty_mat(K, 0, 0)
    return HomologyData{K}(s, 0, 0, 0, 0, Z0, Z0, Z0, Z0, Z0, Z0, _fullcolumn_factor_ref(), _fullcolumn_factor_ref())
end

# Induced map on homology in a fixed degree.
# src, tgt are HomologyData objects for that degree, and f is the chain map matrix in that degree.
function induced_map_on_homology(src::HomologyData{K}, tgt::HomologyData{K}, f::AbstractMatrix{K}) where {K}
    if src.dimH == 0 || tgt.dimH == 0
        return zeros(K, tgt.dimH, src.dimH)
    end
    if _use_batched_coordinate_solves(K, src.dimH, tgt.dimH) ||
       _use_exact_batched_coordinate_solves(K, src.dimH, tgt.dimH)
        return homology_coordinates(tgt, f * src.Hrep)
    end
    H = zeros(K, tgt.dimH, src.dimH)
    for i in 1:src.dimH
        y = f * src.Hrep[:, i]
        H[:, i] .= vec(homology_coordinates(tgt, y))
    end
    return H
end

# Homology at degree s uses:
# cycles = ker(bd_s : C_s -> C_{s-1})
# boundaries = im(bd_{s+1} : C_{s+1} -> C_s)
function homology_data(bd_next::AbstractMatrix{K}, bd_curr::AbstractMatrix{K}, s::Int) where {K}
    bdN = bd_next
    bdC = bd_curr

    dimCs = size(bdC, 2)
    field = field_from_eltype(K)

    if field isa QQField
        Z = if dimCs == 0
            _empty_mat(K, 0, 0)
        elseif size(bdC, 1) == 0
            _eye_mat(K, dimCs)
        else
            FieldLinAlg.nullspace(field, bdC)
        end

        B = if dimCs == 0
            _empty_mat(K, 0, 0)
        elseif size(bdN, 2) == 0
            _empty_mat(K, dimCs, 0)
        else
            FieldLinAlg.colspace(field, bdN)
        end

        dimZ = size(Z, 2)
        dimB = size(B, 2)

        if dimCs == 0
            return _zero_homology_data(K, s)
        end

        if dimB == 0
            I = _eye_mat(K, dimZ)
            return HomologyData{K}(s, dimCs, dimZ, 0, dimZ, Z, B, _empty_mat(K, dimZ, 0), I, I, Z,
                                   _fullcolumn_factor_ref(),
                                   _fullcolumn_factor_ref())
        end

        X = _solve_fullcolumn_cached(field, Z, B)
        Cx = size(X, 2) == 0 ? _empty_mat(K, dimZ, 0) : FieldLinAlg.colspace(field, X)

        rB = size(Cx, 2)
        if rB == dimZ
            return HomologyData{K}(s, dimCs, dimZ, rB, 0, Z, B, Cx, _empty_mat(K, dimZ, 0), Cx, _empty_mat(K, dimCs, 0),
                                   _fullcolumn_factor_ref(),
                                   _fullcolumn_factor_ref())
        end
        Bfull = extend_to_basis(Cx)
        Q = Bfull[:, rB+1:end]
        Hrep = Z * Q
        dimH = size(Q, 2)

        return HomologyData{K}(s, dimCs, dimZ, rB, dimH, Z, B, Cx, Q, Bfull, Hrep,
                               _fullcolumn_factor_ref(),
                               _fullcolumn_factor_ref())
    end

    Z = if dimCs == 0
        zeros(K, 0, 0)
    elseif size(bdC, 1) == 0
        I = zeros(K, dimCs, dimCs)
        for i in 1:dimCs
            I[i, i] = one(K)
        end
        I
    else
        FieldLinAlg.nullspace(field, bdC)
    end

    B = if dimCs == 0
        zeros(K, 0, 0)
    elseif size(bdN, 2) == 0
        zeros(K, dimCs, 0)
    else
        FieldLinAlg.colspace(field, bdN)
    end

    dimZ = size(Z, 2)
    dimB = size(B, 2)

    if dimCs == 0
        return _zero_homology_data(K, s)
    end

    if dimB == 0
        I = _eye_mat(K, dimZ)
        return HomologyData{K}(s, dimCs, dimZ, 0, dimZ, Z, B, zeros(K, dimZ, 0), I, I, Z,
                               _fullcolumn_factor_ref(),
                               _fullcolumn_factor_ref())
    end

    X = if dimB == 0
        zeros(K, dimZ, 0)
    else
        _solve_fullcolumn_cached(field, Z, B)
    end

    Cx = if size(X, 2) == 0
        zeros(K, dimZ, 0)
    else
        FieldLinAlg.colspace(field, X)
    end

    rB = size(Cx, 2)
    if rB == dimZ
        return HomologyData{K}(s, dimCs, dimZ, rB, 0, Z, B, Cx, zeros(K, dimZ, 0), Cx, zeros(K, dimCs, 0),
                               _fullcolumn_factor_ref(),
                               _fullcolumn_factor_ref())
    end
    Bfull = extend_to_basis(Cx)
    Q = Bfull[:, rB+1:end]
    Hrep = Z * Q
    dimH = size(Q, 2)

    return HomologyData{K}(s, dimCs, dimZ, rB, dimH, Z, B, Cx, Q, Bfull, Hrep,
                           _fullcolumn_factor_ref(),
                           _fullcolumn_factor_ref())
end

# Reduce a cycle z in C_s to coordinates in H_s, using precomputed homology data.
#
# Important: even when dimH == 0, we MUST still validate that z is a cycle
# (i.e. z lies in Z_s = ker(bd_s)). Therefore we do NOT early-return on dimH==0.
function homology_coordinates(data::HomologyData{K}, z::AbstractVector{K}) where {K}
    return homology_coordinates(data, reshape(Vector{K}(z), :, 1))
end

# Convenience overload: allow matrix batches with ambient dimension in rows,
# and also tolerate a single row-vector input.
function homology_coordinates(data::HomologyData{K}, z::AbstractMatrix{K}) where {K}
    if size(z, 1) != data.dimC
        if size(z, 1) == 1 && size(z, 2) == data.dimC
            return homology_coordinates(data, reshape(vec(Matrix{K}(z)), :, 1))
        end
        error("homology_coordinates: wrong ambient dimension; got size $(size(z)), expected $(data.dimC) x k")
    end

    field = field_from_eltype(K)
    z0 = Matrix{K}(z)
    alpha = _solve_fullcolumn_cached(field, data.Z, z0, data.Zfactor)
    gamma = _solve_fullcolumn_cached(field, data.Bfull, alpha, data.Bfull_factor)
    rB = data.dimB
    return gamma[rB+1:end, :]
end



# ---------------------------------------------------------------------------
# Additional chain-level and derived-category infrastructure
# ---------------------------------------------------------------------------

# Safe accessors: treat degrees outside the stored range as zero objects.
function _dim_at(C::CochainComplex, t::Int)
    if t < C.tmin || t > C.tmax
        return 0
    end
    return C.dims[t - C.tmin + 1]
end

function _labels_at(C::CochainComplex, t::Int)
    if t < C.tmin || t > C.tmax
        return Int[]
    end
    return C.labels[t - C.tmin + 1]
end

function _diff_at(C::CochainComplex{K}, t::Int) where {K}
    # d^t : C^t -> C^{t+1}
    if t < C.tmin || t >= C.tmax
        return spzeros(K, _dim_at(C, t+1), _dim_at(C, t))
    end
    return C.d[t - C.tmin + 1]
end

"""
    extend_range(C, tmin, tmax) -> CochainComplex

Return a complex isomorphic to `C`, but regarded as living on the larger degree range
`tmin..tmax`, padding missing degrees with zero vector spaces and zero differentials.
"""
function extend_range(C::CochainComplex{K}, tmin::Int, tmax::Int) where {K}
    if tmax < tmin
        error("extend_range: require tmax >= tmin.")
    end
    dims_new = [ _dim_at(C, t) for t in tmin:tmax ]
    d_new = SparseMatrixCSC{K,Int}[]
    for t in tmin:(tmax-1)
        push!(d_new, _diff_at(C, t))
    end
    labels_new = [ Vector{Int}(_labels_at(C, t)) for t in tmin:tmax ]
    return CochainComplex{K}(tmin, tmax, dims_new, d_new; labels=labels_new)
end

# Sign convention: (C[k])^t = C^{t+k}, d_{C[k]} = (-1)^k d_C.
function shift(C::CochainComplex{K}, k::Int) where {K}
    tmin = C.tmin - k
    tmax = C.tmax - k
    dims = [ _dim_at(C, t+k) for t in tmin:tmax ]
    d = SparseMatrixCSC{K,Int}[]
    for t in tmin:(tmax-1)
        dt = _diff_at(C, t+k)
        if isodd(k)
            dt = -dt
        end
        push!(d, dt)
    end
    labels = [ Vector{Int}(_labels_at(C, t+k)) for t in tmin:tmax ]
    return CochainComplex{K}(tmin, tmax, dims, d; labels=labels)
end

"""
A cochain map f : C -> D, stored on a chosen degree interval f.tmin..f.tmax.

For degrees outside this interval, `f` is treated as the zero map.
This makes mapping cones and triangles robust under differing degree ranges.
"""
struct CochainMap{K}
    C::CochainComplex{K}
    D::CochainComplex{K}
    tmin::Int
    tmax::Int
    maps::Vector{SparseMatrixCSC{K,Int}}  # maps[t - tmin + 1] : C^t -> D^t
end

function _map_at(f::CochainMap{K}, t::Int) where {K}
    if t < f.tmin || t > f.tmax
        return spzeros(K, _dim_at(f.D, t), _dim_at(f.C, t))
    end
    return f.maps[t - f.tmin + 1]
end

function is_cochain_map(f::CochainMap{K})::Bool where {K}
    for t in f.tmin:f.tmax
        # Compare as sparse, do not densify:
        lhs = _diff_at(f.D, t) * _map_at(f, t)
        rhs = _map_at(f, t + 1) * _diff_at(f.C, t)
        diff = lhs - rhs
        dropzeros!(diff)
        if nnz(diff) != 0
            return false
        end
    end
    return true
end

"""
    CochainMap(C, D, maps; tmin=nothing, tmax=nothing, check=true)

Construct a cochain map C -> D.

If tmin/tmax are omitted, they default to min/max of the degree ranges of C and D.
The vector `maps` must have length tmax - tmin + 1 and contain maps C^t -> D^t.
"""
function CochainMap(C::CochainComplex{K},
                    D::CochainComplex{K},
                    maps::Vector{<:AbstractMatrix{K}};
                    tmin::Union{Nothing,Int}=nothing,
                    tmax::Union{Nothing,Int}=nothing,
                    check::Bool=true) where {K}
    tmin2 = tmin === nothing ? min(C.tmin, D.tmin) : tmin
    tmax2 = tmax === nothing ? max(C.tmax, D.tmax) : tmax
    if length(maps) != tmax2 - tmin2 + 1
        error("CochainMap: maps must have length tmax - tmin + 1.")
    end

    smaps = SparseMatrixCSC{K,Int}[]
    for (i, M) in enumerate(maps)
        t = tmin2 + i - 1
        Mi = sparse(M)
        if size(Mi, 1) != _dim_at(D, t) || size(Mi, 2) != _dim_at(C, t)
            error("CochainMap: map size mismatch at degree $t.")
        end
        push!(smaps, Mi)
    end

    f = CochainMap{K}(C, D, tmin2, tmax2, smaps)
    if check && !is_cochain_map(f)
        error("CochainMap: maps do not satisfy d_D f = f d_C.")
    end
    return f
end

"""
    mapping_cone(f) -> CochainComplex

Return the mapping cone Cone(f) of a cochain map f : C -> D.

Convention:
Cone(f)^t = D^t oplus C^{t+1},
d_Cone^t = [ d_D^t   f^{t+1} ]
           [   0   -d_C^{t+1} ].

The degree range of the cone is the smallest interval supporting all required terms:
tmin = min(D.tmin, C.tmin - 1), tmax = max(D.tmax, C.tmax - 1).
"""
function mapping_cone(f::CochainMap{K}) where {K}
    C, D = f.C, f.D
    tmin = min(D.tmin, C.tmin - 1)
    tmax = max(D.tmax, C.tmax - 1)

    dims = Int[]
    labels = Vector{Vector{Int}}()
    for t in tmin:tmax
        dimD = _dim_at(D, t)
        dimC1 = _dim_at(C, t+1)
        push!(dims, dimD + dimC1)

        labs = Int[]
        append!(labs, _labels_at(D, t))
        append!(labs, _labels_at(C, t+1))
        push!(labels, labs)
    end

    d = SparseMatrixCSC{K,Int}[]
    for t in tmin:(tmax-1)
        dimD_t  = _dim_at(D, t)
        dimD_t1 = _dim_at(D, t+1)
        dimC_t1 = _dim_at(C, t+1)
        dimC_t2 = _dim_at(C, t+2)

        dD = _diff_at(D, t)
        dC = _diff_at(C, t+1)
        f_tp1 = _map_at(f, t+1)  # C^{t+1} -> D^{t+1}

        Z = spzeros(K, dimC_t2, dimD_t)
        d_cone_t = [dD f_tp1; Z -dC]
        push!(d, d_cone_t)
    end

    return CochainComplex{K}(tmin, tmax, dims, d; labels=labels)
end

struct DistinguishedTriangle{K}
    C::CochainComplex{K}
    D::CochainComplex{K}
    cone::CochainComplex{K}
    Cshift::CochainComplex{K}
    f::CochainMap{K}
    i::CochainMap{K}
    p::CochainMap{K}
end

"""
    mapping_cone_triangle(f) -> DistinguishedTriangle

Return the standard distinguished triangle
C --f--> D --i--> Cone(f) --p--> C[1].
"""
function mapping_cone_triangle(f::CochainMap{K}) where {K}
    C, D = f.C, f.D
    cone = mapping_cone(f)
    Cshift = shift(C, 1)

    # i : D -> cone (inclusion into the first summand)
    tmin_i = min(D.tmin, cone.tmin)
    tmax_i = max(D.tmax, cone.tmax)
    imaps = SparseMatrixCSC{K,Int}[]
    for t in tmin_i:tmax_i
        dimD = _dim_at(D, t)
        dimCone = _dim_at(cone, t)
        I_D = spdiagm(0 => fill(one(K), dimD))
        Z = spzeros(K, dimCone - dimD, dimD)
        push!(imaps, [I_D; Z])
    end
    i = CochainMap(D, cone, imaps; tmin=tmin_i, tmax=tmax_i, check=true)

    # p : cone -> Cshift (projection onto the second summand)
    tmin_p = min(cone.tmin, Cshift.tmin)
    tmax_p = max(cone.tmax, Cshift.tmax)
    pmaps = SparseMatrixCSC{K,Int}[]
    for t in tmin_p:tmax_p
        dimD = _dim_at(D, t)
        dimC1 = _dim_at(C, t+1)
        Z = spzeros(K, dimC1, dimD)
        I_C = spdiagm(0 => fill(one(K), dimC1))
        push!(pmaps, [Z I_C])
    end
    p = CochainMap(cone, Cshift, pmaps; tmin=tmin_p, tmax=tmax_p, check=true)

    return DistinguishedTriangle{K}(C, D, cone, Cshift, f, i, p)
end

struct LongExactSequence{K}
    triangle::DistinguishedTriangle{K}
    tmin::Int
    tmax::Int
    HC::Vector{CohomologyData{K}}
    HD::Vector{CohomologyData{K}}
    Hcone::Vector{CohomologyData{K}}
    HCshift::Vector{CohomologyData{K}}
    fH::Vector{Matrix{K}}      # H^t(C) -> H^t(D)
    iH::Vector{Matrix{K}}      # H^t(D) -> H^t(Cone)
    pH::Vector{Matrix{K}}      # H^t(Cone) -> H^t(C[1])
    delta::Vector{Matrix{K}}  # H^t(Cone) -> H^{t+1}(C)
end

"""
    long_exact_sequence(tri) -> LongExactSequence

Given a distinguished triangle C -> D -> Cone -> C[1], compute the induced maps on cohomology
and package the long exact sequence data, including the connecting morphisms
delta : H^t(Cone) -> H^{t+1}(C).

The returned object stores all degrees on the global range covering C, D, Cone, and C[1],
treating missing degrees as zero.
"""
function long_exact_sequence(tri::DistinguishedTriangle{K}) where {K}
    C, D, cone, Cshift = tri.C, tri.D, tri.cone, tri.Cshift
    tmin = min(C.tmin, D.tmin, cone.tmin, Cshift.tmin)
    tmax = max(C.tmax, D.tmax, cone.tmax, Cshift.tmax)
    ndeg = tmax - tmin + 1

    function _cohomology_range(X::CochainComplex{K}) where {K}
        out = Vector{CohomologyData{K}}(undef, ndeg)
        if _use_long_exact_precompute(K, X)
            local_data = cohomology_data(X)
            for t in tmin:tmax
                idx = t - tmin + 1
                if X.tmin <= t <= X.tmax
                    out[idx] = local_data[t - X.tmin + 1]
                else
                    out[idx] = _zero_cohomology_data(K, t)
                end
            end
        else
            for t in tmin:tmax
                idx = t - tmin + 1
                if X.tmin <= t <= X.tmax
                    out[idx] = cohomology_data(X, t)
                else
                    out[idx] = _zero_cohomology_data(K, t)
                end
            end
        end
        return out
    end

    HC = _cohomology_range(C)
    HD = _cohomology_range(D)
    Hcone = _cohomology_range(cone)
    HCshift = Vector{CohomologyData{K}}(undef, ndeg)
    for (k, t) in enumerate(tmin:tmax)
        if t + 1 <= tmax
            Hn = HC[k + 1]
            HCshift[k] = CohomologyData{K}(t,
                                           Hn.dimC,
                                           Hn.dimZ,
                                           Hn.dimB,
                                           Hn.dimH,
                                           Hn.K,
                                           Hn.B,
                                           Hn.Cx,
                                           Hn.Q,
                                           Hn.Bfull,
                                           Hn.Hrep,
                                           Hn.Kfactor,
                                           Hn.Bfull_factor)
        else
            HCshift[k] = _zero_cohomology_data(K, t)
        end
    end

    fH = Vector{Matrix{K}}(undef, ndeg)
    iH = Vector{Matrix{K}}(undef, ndeg)
    pH = Vector{Matrix{K}}(undef, ndeg)
    delta = Vector{Matrix{K}}(undef, ndeg)
    for (k, t) in enumerate(tmin:tmax)
        fH[k] = induced_map_on_cohomology(HC[k], HD[k], _map_at(tri.f, t))
        iH[k] = induced_map_on_cohomology(HD[k], Hcone[k], _map_at(tri.i, t))
        if t + 1 > tmax
            pH[k] = induced_map_on_cohomology(Hcone[k], HCshift[k], _map_at(tri.p, t))
            delta[k] = zeros(K, 0, Hcone[k].dimH)
            continue
        end
        p_map = _map_at(tri.p, t)
        pH[k] = induced_map_on_cohomology(Hcone[k], HC[k + 1], p_map)
        delta[k] = pH[k]
    end

    return LongExactSequence{K}(tri, tmin, tmax, HC, HD, Hcone, HCshift, fH, iH, pH, delta)
end

# -------------------------------------------------------------------------------------
# Spectral sequences from (finite) double complexes
# -------------------------------------------------------------------------------------

"""
    DoubleComplex{K}

A finite first-quadrant style cochain bicomplex with bidegrees (a,b):

- Objects:  C^{a,b}  for amin <= a <= amax and bmin <= b <= bmax
- Vertical differential:   dv^{a,b} : C^{a,b} -> C^{a,b+1}
- Horizontal differential: dh^{a,b} : C^{a,b} -> C^{a+1,b}

We assume:
- dv circ dv = 0 and dh circ dh = 0 (within range),
- dv circ dh + dh circ dv = 0 (anti-commutation), as is standard for a bicomplex.

Storage convention:
- `dims[aidx,bidx] = dim(C^{a,b})` where aidx = a-amin+1 and bidx = b-bmin+1.
- `dv[aidx,bidx]` is the matrix for dv^{a,b} (or a correctly-sized zero matrix at the boundary).
- `dh[aidx,bidx]` is the matrix for dh^{a,b} (or a correctly-sized zero matrix at the boundary).
"""
struct DoubleComplex{K}
    amin::Int
    amax::Int
    bmin::Int
    bmax::Int
    dims::Matrix{Int}
    dv::Array{SparseMatrixCSC{K,Int},2}
    dh::Array{SparseMatrixCSC{K,Int},2}
end

# Max total degree stored in a bounded double complex.
maxdeg_of_complex(DC::DoubleComplex) = DC.amax + DC.bmax

# Internal helpers: safe access to dv^{a,b} and dh^{a,b} blocks.
#
# These are convenience functions for debugging / small hand-built examples.
# They return the stored sparse block if (a,b) is in range, otherwise a 0x0
# sparse matrix.
function _dv_at(DC::DoubleComplex{K}, a::Int, b::Int) where {K}
    if a < DC.amin || a > DC.amax || b < DC.bmin || b > DC.bmax
        return spzeros(K, 0, 0)
    end
    return DC.dv[a - DC.amin + 1, b - DC.bmin + 1]
end

function _dh_at(DC::DoubleComplex{K}, a::Int, b::Int) where {K}
    if a < DC.amin || a > DC.amax || b < DC.bmin || b > DC.bmax
        return spzeros(K, 0, 0)
    end
    return DC.dh[a - DC.amin + 1, b - DC.bmin + 1]
end


"""
    total_complex(DC) -> CochainComplex{K}

Form the total cochain complex Tot(DC) with grading t = a+b and differential d = dv + dh.
(We assume dh already carries whatever sign convention was used to enforce anti-commutation.)

This produces a *concrete* cochain complex suitable for cohomology computations.
"""
function total_complex(DC::DoubleComplex{K}) where {K}
    Tot, _ = _total_complex_with_blocks(DC)
    return Tot
end


# -----------------------------------------------------------------------------
# Spectral sequences from double complexes
# -----------------------------------------------------------------------------

# NOTE: Everything in this section is intended to be ASCII-only and uses exact
# field linear algebra (FieldLinAlg.jl) throughout.

# -----------------------------------------------------------------------------
# Linear algebra helper: explicit subquotients (for E_r terms)
# -----------------------------------------------------------------------------

"""
    SubquotientData{K}

An explicit subquotient space `Z / B` inside a fixed ambient vector space `V`.

The ambient space is identified with `K^N` for `N = ambient_dim`, and the data
is stored using explicit bases:

  * numerator basis `Zbasis` (matrix with columns spanning Z)
  * denominator basis `Bbasis` (matrix with columns spanning B, with B subset Z)
  * a chosen complement basis, giving representatives of a basis of `Z/B`

This is designed so that we can:
  * read off `dim(Z/B)`,
  * get explicit representatives of basis elements,
  * compute induced maps by acting on representatives and taking coordinates.
"""
struct SubquotientData{K}
    ambient_dim::Int
    dimZ::Int
    dimB::Int
    dimH::Int

    Zbasis::Matrix{K}   # ambient_dim x dimZ
    Bbasis::Matrix{K}   # ambient_dim x dimB

    Bcoords::Matrix{K}  # dimZ x dimB
    Bfull::Matrix{K}    # dimZ x dimZ (invertible extension of Bcoords)
    Hcoords::Matrix{K}  # dimZ x dimH
    Hrep::Matrix{K}     # ambient_dim x dimH
    Zsolve_rows::UnitRange{Int}
    Zsolve_basis::Matrix{K}
    Zsolve_factor::Base.RefValue{Union{Nothing,FieldLinAlg.FullColumnFactor{QQ}}}
    Bfull_factor::Base.RefValue{Union{Nothing,FieldLinAlg.FullColumnFactor{QQ}}}
end

"""
    subquotient_data(Zbasis, Bgens) -> SubquotientData{K}

Build explicit data for a subquotient `Z/B`.

Inputs:
  * `Zbasis`: ambient_dim x dimZ, columns form a basis of Z.
  * `Bgens`:  ambient_dim x m, columns generate B, with B subset span(Zbasis).

All arithmetic is exact for exact fields.
"""
@inline function _solve_fullcolumn_cached(field::AbstractCoeffField, B, Y; check_rhs::Bool=true)
    if field isa QQField
        return FieldLinAlg._solve_fullcolumnQQ(B, Y; check_rhs=check_rhs, cache=true)
    end
    return FieldLinAlg.solve_fullcolumn(field, B, Y; check_rhs=check_rhs)
end

@inline function _solve_fullcolumn_cached(field::QQField, B, Y, factor::FieldLinAlg.FullColumnFactor{QQ}; check_rhs::Bool=true)
    return FieldLinAlg._solve_fullcolumn_factorQQ(B, factor, Y; check_rhs=check_rhs)
end

@inline function _solve_fullcolumn_cached(field::AbstractCoeffField,
                                          B,
                                          Y,
                                          factor_ref::Base.RefValue{Union{Nothing,FieldLinAlg.FullColumnFactor{QQ}}};
                                          check_rhs::Bool=true)
    if field isa QQField
        factor = _fullcolumn_factor!(field, B, factor_ref)
        return factor === nothing ?
            _solve_fullcolumn_cached(field, B, Y; check_rhs=check_rhs) :
            _solve_fullcolumn_cached(field, B, Y, factor; check_rhs=check_rhs)
    end
    return _solve_fullcolumn_cached(field, B, Y; check_rhs=check_rhs)
end

function _subquotient_data_from_coords(Zbasis::AbstractMatrix{K},
                                       Bcoords_in_Z::AbstractMatrix{K};
                                       Zsolve_rows::UnitRange{Int}=1:size(Zbasis, 1),
                                       Zsolve_basis::AbstractMatrix{K}=Zbasis) where {K}
    field = field_from_eltype(K)
    Zmat = _concrete_mat(Zbasis)
    ambient_dim = size(Zmat, 1)
    dimZ = size(Zmat, 2)
    Zsolve = _concrete_mat(Zsolve_basis)

    if dimZ == 0
        Z0 = _empty_mat(K, ambient_dim, 0)
        return SubquotientData{K}(ambient_dim, 0, 0, 0,
                                  Z0, Z0, zeros(K, 0, 0), zeros(K, 0, 0),
                                  zeros(K, 0, 0), Z0, Zsolve_rows, Z0, _fullcolumn_factor_ref(), _fullcolumn_factor_ref())
    end

    if size(Bcoords_in_Z, 2) == 0
        Bcoords = _empty_mat(K, dimZ, 0)
    else
        Bcoords = FieldLinAlg.colspace(field, _concrete_mat(Bcoords_in_Z))
    end
    dimB = size(Bcoords, 2)
    Bbasis = Zmat * Bcoords

    Bfull = extend_to_basis(Bcoords)
    Hcoords = Bfull[:, (dimB+1):dimZ]
    dimH = size(Hcoords, 2)
    Hrep = Zmat * Hcoords

    return SubquotientData{K}(ambient_dim, dimZ, dimB, dimH,
                              Zmat, Bbasis, Bcoords, Bfull, Hcoords, Hrep,
                              Zsolve_rows, Zsolve,
                              _fullcolumn_factor_ref(),
                              _fullcolumn_factor_ref())
end

function subquotient_data(Zbasis::AbstractMatrix{K}, Bgens::AbstractMatrix{K}) where {K}
    field = field_from_eltype(K)
    Zmat = _concrete_mat(Zbasis)
    dimZ = size(Zmat, 2)
    Bcoords = if size(Bgens, 2) == 0 || dimZ == 0
        _empty_mat(K, dimZ, 0)
    else
        _solve_fullcolumn_cached(field, Zmat, Bgens)
    end
    return _subquotient_data_from_coords(Zmat, Bcoords; Zsolve_rows=1:size(Zmat, 1), Zsolve_basis=Zmat)
end

function _subquotient_coordinates_rows(SQ::SubquotientData{K}, zrows::AbstractMatrix{K}) where {K}
    size(zrows, 1) == size(SQ.Zsolve_basis, 1) ||
        error("subquotient_coordinates: restricted dimension mismatch")
    if SQ.dimH == 0
        return zeros(K, 0, size(zrows, 2))
    end
    field = field_from_eltype(K)
    alpha = _solve_fullcolumn_cached(field, SQ.Zsolve_basis, Matrix{K}(zrows), SQ.Zsolve_factor)
    gamma = _solve_fullcolumn_cached(field, SQ.Bfull, alpha, SQ.Bfull_factor)
    return gamma[(SQ.dimB+1):SQ.dimZ, :]
end

"""
    subquotient_coordinates(SQ, z) -> Matrix{K}

Given `SQ = Z/B` and column vector(s) `z` in Z, return coordinates in the
chosen basis of `Z/B`.
"""
function subquotient_coordinates(SQ::SubquotientData{K}, z::Matrix{K}) where {K}
    if size(z, 1) != SQ.ambient_dim
        error("subquotient_coordinates: ambient dimension mismatch")
    end
    return _subquotient_coordinates_rows(SQ, @view(z[SQ.Zsolve_rows, :]))
end

subquotient_coordinates(SQ::SubquotientData{K}, z::Vector{K}) where {K} =
    subquotient_coordinates(SQ, reshape(z, :, 1))

function _subquotient_pushforward_coords(src::SubquotientData{K},
                                         tgt::SubquotientData{K},
                                         f::AbstractMatrix{K}) where {K}
    if src.dimH == 0 || tgt.dimH == 0
        return zeros(K, tgt.dimH, src.dimH)
    end
    src_rows = src.Zsolve_rows
    tgt_rows = tgt.Zsolve_rows
    src_block = @view src.Hrep[src_rows, :]
    rhs_full = (@view f[:, src_rows]) * src_block
    return _subquotient_coordinates_rows(tgt, @view(rhs_full[tgt_rows, :]))
end

# -----------------------------------------------------------------------------
# Internal cached page and splitting types
# -----------------------------------------------------------------------------

struct SSPageData{K}
    r::Int
    dims::Matrix{Int}
    spaces::Array{SubquotientData{K},2}
end

struct SSSplitData{K}
    t::Int
    B::Matrix{K}
    Binv::Matrix{K}
    ranges::Dict{Tuple{Int,Int},UnitRange{Int}}
end

mutable struct _SSPageWorkspace{K}
    den_buf::Matrix{K}
end

_SSPageWorkspace(::Type{K}) where {K} = _SSPageWorkspace{K}(_empty_mat(K, 0, 0))

function _ss_den_buf!(ws::_SSPageWorkspace{K},
                      B::AbstractMatrix{K},
                      Zint::AbstractMatrix{K}) where {K}
    m = size(B, 1)
    nb = size(B, 2)
    nz = size(Zint, 2)
    n = nb + nz
    if size(ws.den_buf, 1) != m || size(ws.den_buf, 2) != n
        ws.den_buf = Matrix{K}(undef, m, n)
    end
    if nb > 0
        @views ws.den_buf[1:m, 1:nb] .= B
    end
    if nz > 0
        @views ws.den_buf[1:m, nb+1:n] .= Zint
    end
    return ws.den_buf
end

"""
    SpectralSequence{K}

v2 spectral sequence object.

Key features:
  * page(ss,r) for all r>=1 and :inf/:infty
  * differential(ss,r) for all r>=1
  * explicit subquotient models for each term
  * explicit filtration and edge maps
  * collapse/convergence utilities

Indexing note:
`page(ss,r)` and `term(ss,r,(a,b))` use the double-complex bidegree `(a,b)`
with `a` in the first axis and `b` in the second axis. If you need the
filtration/total-degree view, use `ss_key(first,a,b)` or `ss_key(ss,a,b)` to
map `(a,b)` to `(p,t)` where `p` is the filtration index and `t = a + b`.
"""
mutable struct _LazyValue{T}
    value::Union{Nothing,T}
end

struct SpectralSequence{K}
    DC::DoubleComplex{K}
    first::Symbol

    E1_dims::Matrix{Int}
    d1::Array{SparseMatrixCSC{K,Int},2}
    E2_dims::Matrix{Int}
    Einf_dims::Matrix{Int}
    Htot_dims::Vector{Int}

    Tot::CochainComplex{K}
    Htot::Vector{CohomologyData{K}}

    pmin::Int
    pmax::Int
    rmax_possible::Int

    blocks::Vector{Vector{NTuple{4,Int}}}
    fp_ranges::Matrix{UnitRange{Int}}
    filt_img_dims::Matrix{Int}
    filt_img::_LazyValue{Array{Matrix{K},2}}
    filt_img_cols::Vector{_LazyValue{Vector{Matrix{K}}}}
    Einf_spaces::_LazyValue{Array{SubquotientData{K},2}}

    page_cache::Dict{Int,SSPageData{K}}
    diff_cache::Dict{Int,Array{SparseMatrixCSC{K,Int},2}}
    split_cache::Dict{Int,SSSplitData{K}}
end

"""
    SpectralPage

User-facing wrapper for E_r dimension tables.
Supports:
  * matrix indexing P[i,j]
  * bidegree indexing P[(a,b)]
"""
struct SpectralPage{K} <: AbstractMatrix{Int}
    ss::SpectralSequence{K}
    r::Union{Int,Symbol}
    dims::Matrix{Int}
end

Base.IndexStyle(::Type{<:SpectralPage}) = IndexCartesian()
Base.size(P::SpectralPage) = size(P.dims)
Base.getindex(P::SpectralPage, i::Int, j::Int) = P.dims[i,j]

function Base.getindex(P::SpectralPage, ab::Tuple{Int,Int})
    ss = P.ss
    a, b = ab
    if a < ss.DC.amin || a > ss.DC.amax || b < ss.DC.bmin || b > ss.DC.bmax
        return 0
    end
    return P.dims[a - ss.DC.amin + 1, b - ss.DC.bmin + 1]
end

"""
    SpectralTermsPage

A page wrapper like `SpectralPage`, but storing the actual `SubquotientData`
models of the terms, not just dimensions.

Indexing conventions:
- `P[i,j]` uses matrix indices.
- `P[(a,b)]` uses bidegree indices.

This is intended as the "give me E2 as objects" entry point.
"""
struct SpectralTermsPage{K} <: AbstractMatrix{SubquotientData{K}}
    ss::SpectralSequence{K}
    r::Union{Int,Symbol}
    terms::Matrix{SubquotientData{K}}
end

const _spectral_exact_diff_mode = Ref{Symbol}(:coords)
const _spectral_exact_filtimg_cache_mode = Ref{Symbol}(:auto)
const _spectral_exact_filtimg_basis_mode = Ref{Symbol}(:auto)

@inline function _use_exact_diff_coords()
    mode = _spectral_exact_diff_mode[]
    mode === :coords && return true
    mode === :ambient && return false
    error("_spectral_exact_diff_mode must be :coords or :ambient")
end

Base.size(P::SpectralTermsPage) = size(P.terms)
Base.getindex(P::SpectralTermsPage, i::Int, j::Int) = P.terms[i, j]

function Base.getindex(P::SpectralTermsPage{K}, ab::Tuple{Int,Int}) where {K}
    ss = P.ss
    a, b = ab
    if a < ss.DC.amin || a > ss.DC.amax || b < ss.DC.bmin || b > ss.DC.bmax
        # Outside the defined rectangle, terms are zero.
        return _ss_zero_subquotient(K, 0)
    end
    return P.terms[a - ss.DC.amin + 1, b - ss.DC.bmin + 1]
end


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

function _ss_check_first(first::Symbol)
    if !(first == :vertical || first == :horizontal)
        error("spectral_sequence: first must be :vertical or :horizontal")
    end
end

_ss_p(first::Symbol, a::Int, b::Int) = (first == :vertical ? a : b)

function _ss_dr_target(first::Symbol, r::Int, a::Int, b::Int)
    if first == :vertical
        return (a + r, b - r + 1)
    else
        return (a - r + 1, b + r)
    end
end

function _ss_blocks(DC::DoubleComplex{K}) where {K}
    tmin = DC.amin + DC.bmin
    tmax = DC.amax + DC.bmax
    blocks = Vector{Vector{NTuple{4,Int}}}(undef, tmax - tmin + 1)
    for t in tmin:tmax
        off = 1
        info = NTuple{4,Int}[]
        for a in DC.amin:DC.amax
            b = t - a
            if DC.bmin <= b <= DC.bmax
                dim = DC.dims[a - DC.amin + 1, b - DC.bmin + 1]
                if dim > 0
                    push!(info, (a, b, off, dim))
                end
                off += dim
            end
        end
        blocks[t - tmin + 1] = info
    end
    return blocks
end

@inline function _ss_blocks_dim(blocks_t::Vector{NTuple{4,Int}})
    isempty(blocks_t) && return 0
    blk = blocks_t[end]
    return blk[3] + blk[4] - 1
end

function _ss_totdim(Tot::CochainComplex{K}, t::Int) where {K}
    if t < Tot.tmin || t > Tot.tmax
        return 0
    end
    return Tot.dims[t - Tot.tmin + 1]
end

"""
    ss_key(first, a, b) -> (p, t)

Return the filtration/total-degree key for bidegree `(a,b)` under the
chosen filtration `first` (`:vertical` or `:horizontal`). The total degree
is `t = a + b`.
"""
ss_key(first::Symbol, a::Int, b::Int) = (_ss_p(first, a, b), a + b)

"""
    ss_key(ss, a, b) -> (p, t)

Convenience wrapper using `ss.first`.
"""
ss_key(ss::SpectralSequence, a::Int, b::Int) = ss_key(ss.first, a, b)

function _ss_blocks_at(blocks::Vector{Vector{NTuple{4,Int}}}, tmin::Int, t::Int)
    if t < tmin || t > tmin + length(blocks) - 1
        return NTuple{4,Int}[]
    end
    return blocks[t - tmin + 1]
end

function _ss_Fp_range(first::Symbol,
                      blocks_t::Vector{NTuple{4,Int}},
                      dim_t::Int,
                      t::Int,
                      p::Int)
    if dim_t == 0
        return 1:0
    end

    if first == :vertical
        for blk in blocks_t
            a = blk[1]
            start = blk[3]
            if a >= p
                return start:dim_t
            end
        end
        return 1:0
    else
        athresh = t - p
        endpos = 0
        for blk in blocks_t
            a = blk[1]
            start = blk[3]
            dim = blk[4]
            if a <= athresh
                endpos = start + dim - 1
            else
                break
            end
        end
        if endpos == 0
            return 1:0
        end
        return 1:endpos
    end
end

function _ss_low_range(first::Symbol,
                       blocks_t::Vector{NTuple{4,Int}},
                       dim_t::Int,
                       t::Int,
                       p::Int)
    F = _ss_Fp_range(first, blocks_t, dim_t, t, p)
    if isempty(F)
        return 1:dim_t
    end
    if length(F) == dim_t
        return 1:0
    end
    if first == :vertical
        return 1:(Base.first(F) - 1)
    else
        return (Base.last(F) + 1):dim_t
    end
end

function _ss_inclusion_matrix(::Type{K}, dim::Int, r::UnitRange{Int}) where {K}
    k = length(r)
    if k == 0
        return spzeros(K, dim, 0)
    end
    return sparse(collect(r), collect(1:k), fill(one(K), k), dim, k)
end

@inline _ss_range_sig(r::UnitRange{Int}) = isempty(r) ? (1, 0) : (first(r), last(r))

@inline function _ss_filtered_signature(r_tm1::UnitRange{Int},
                                        r_t::UnitRange{Int},
                                        r_tp1::UnitRange{Int})
    s0, e0 = _ss_range_sig(r_tm1)
    s1, e1 = _ss_range_sig(r_t)
    s2, e2 = _ss_range_sig(r_tp1)
    return (s0, e0, s1, e1, s2, e2)
end

function _ss_filtered_cohomology_data(Tot::CochainComplex{K},
                                      fp_ranges::AbstractMatrix{UnitRange{Int}},
                                      ip::Int,
                                      tidx::Int) where {K}
    t = Tot.tmin + tidx - 1
    tlen = size(fp_ranges, 2)
    r_t = fp_ranges[ip, tidx]
    dimCt = length(r_t)

    if tidx == 1
        d_prev = zeros(K, dimCt, 0)
    else
        r_tm1 = fp_ranges[ip, tidx - 1]
        d_prev = Tot.d[tidx - 1][r_t, r_tm1]
    end

    if tidx == tlen
        d_curr = zeros(K, 0, dimCt)
    else
        r_tp1 = fp_ranges[ip, tidx + 1]
        d_curr = Tot.d[tidx][r_tp1, r_t]
    end

    return _cohomology_data_from_diffs(K, t, dimCt, d_prev, d_curr)
end

function _ss_E2_dims_from_d1(DC::DoubleComplex{K},
                             first::Symbol,
                             E1_dims::Matrix{Int},
                             d1::Array{SparseMatrixCSC{K,Int},2}) where {K}
    field = field_from_eltype(K)
    Alen, Blen = size(E1_dims)
    dranks = zeros(Int, Alen, Blen)
    dims = zeros(Int, Alen, Blen)

    for ai in 1:Alen, bi in 1:Blen
        D = d1[ai, bi]
        dranks[ai, bi] = nnz(D) == 0 ? 0 : FieldLinAlg.rank(field, D)
    end

    for ai in 1:Alen, bi in 1:Blen
        incoming = 0
        src_ai, src_bi = if first == :vertical
            (ai - 1, bi)
        else
            (ai, bi - 1)
        end
        if 1 <= src_ai <= Alen && 1 <= src_bi <= Blen
            incoming = dranks[src_ai, src_bi]
        end
        dims[ai, bi] = max(0, E1_dims[ai, bi] - dranks[ai, bi] - incoming)
    end

    return dims
end

# -----------------------------------------------------------------------------
# Page construction: explicit Z_r, B_r, and quotient models
# -----------------------------------------------------------------------------

function _ss_Z_basis(Tot::CochainComplex{K},
                     blocks::Vector{Vector{NTuple{4,Int}}},
                     first::Symbol,
                     t::Int,
                     p::Int,
                     r::Int) where {K}
    dim_t = _ss_totdim(Tot, t)
    if dim_t == 0
        return zeros(K, 0, 0)
    end

    tmin = Tot.tmin
    blocks_t = _ss_blocks_at(blocks, tmin, t)
    high = _ss_Fp_range(first, blocks_t, dim_t, t, p)
    dim_tp1 = _ss_totdim(Tot, t + 1)
    blocks_tp1 = _ss_blocks_at(blocks, tmin, t + 1)
    low_tp1 = _ss_low_range(first, blocks_tp1, dim_tp1, t + 1, p + r)
    return _ss_Z_basis(Tot, t, high, low_tp1)
end

function _ss_Z_basis(Tot::CochainComplex{K},
                     t::Int,
                     high::UnitRange{Int},
                     low_tp1::UnitRange{Int}) where {K}
    Z, _ = _ss_Z_basis_with_coords(Tot, t, high, low_tp1)
    return Z
end

function _ss_Z_basis_with_coords(Tot::CochainComplex{K},
                                 t::Int,
                                 high::UnitRange{Int},
                                 low_tp1::UnitRange{Int}) where {K}
    field = field_from_eltype(K)
    dim_t = _ss_totdim(Tot, t)
    if length(high) == 0
        return zeros(K, dim_t, 0), zeros(K, 0, 0)
    end

    D = _diff_at(Tot, t)
    A = @view D[low_tp1, high]
    Zcoords = FieldLinAlg.nullspace(field, A)

    Z = zeros(K, dim_t, size(Zcoords, 2))
    Z[high, :] = Zcoords
    return Z, Zcoords
end

function _ss_Z_intersection_Fp1(Zbasis::Matrix{K},
                                first::Symbol,
                                blocks_t::Vector{NTuple{4,Int}},
                                dim_t::Int,
                                t::Int,
                                p1::Int) where {K}
    low_p1 = _ss_low_range(first, blocks_t, dim_t, t, p1)
    return _ss_Z_intersection_Fp1(Zbasis, low_p1)
end

function _ss_Z_intersection_Fp1(Zbasis::Matrix{K},
                                low_p1::UnitRange{Int}) where {K}
    Zint_coords = _ss_Z_intersection_Fp1_coords(Zbasis, low_p1)
    if size(Zbasis, 2) == 0
        return zeros(K, size(Zbasis, 1), 0)
    end
    Zint = Zbasis * Zint_coords
    return FieldLinAlg.colspace(field_from_eltype(K), Zint)
end

function _ss_Z_intersection_Fp1_coords(Zbasis::Matrix{K},
                                       low_p1::UnitRange{Int}) where {K}
    field = field_from_eltype(K)
    dimZ = size(Zbasis, 2)
    if dimZ == 0
        return zeros(K, 0, 0)
    end

    A = @view Zbasis[low_p1, :]
    return FieldLinAlg.nullspace(field, A)
end

function _ss_B_basis(Tot::CochainComplex{K},
                     blocks::Vector{Vector{NTuple{4,Int}}},
                     first::Symbol,
                     t::Int,
                     p::Int,
                     r::Int) where {K}
    dim_t = _ss_totdim(Tot, t)
    if dim_t == 0
        return zeros(K, 0, 0)
    end

    tmin = Tot.tmin
    blocks_t = _ss_blocks_at(blocks, tmin, t)

    dim_tm1 = _ss_totdim(Tot, t - 1)
    blocks_tm1 = _ss_blocks_at(blocks, tmin, t - 1)
    dom_range = _ss_Fp_range(first, blocks_tm1, dim_tm1, t - 1, p - r + 1)
    low_p = _ss_low_range(first, blocks_t, dim_t, t, p)
    return _ss_B_basis(Tot, t, dim_t, dom_range, low_p)
end

function _ss_B_basis(Tot::CochainComplex{K},
                     t::Int,
                     dim_t::Int,
                     dom_range::UnitRange{Int},
                     low_p::UnitRange{Int}) where {K}
    field = field_from_eltype(K)
    if length(dom_range) == 0
        return zeros(K, dim_t, 0)
    end

    D = _diff_at(Tot, t - 1)
    Dsub = @view D[:, dom_range]

    A = @view Dsub[low_p, :]
    Kmat = FieldLinAlg.nullspace(field, A)

    Bd = Dsub * Kmat
    return FieldLinAlg.colspace(field, Bd)
end

function _ss_compute_page_term(DC::DoubleComplex{K},
                               Tot::CochainComplex{K},
                               blocks::Vector{Vector{NTuple{4,Int}}},
                               first::Symbol,
                               r::Int,
                               a::Int,
                               b::Int,
                               ws::_SSPageWorkspace{K}) where {K}
    if field_from_eltype(K) isa QQField
        return _ss_compute_page_term_exact(DC, Tot, blocks, first, r, a, b, ws)
    end
    return _ss_compute_page_term_ambient(DC, Tot, blocks, first, r, a, b, ws)
end

function _ss_compute_page_term_ambient(DC::DoubleComplex{K},
                                       Tot::CochainComplex{K},
                                       blocks::Vector{Vector{NTuple{4,Int}}},
                                       first::Symbol,
                                       r::Int,
                                       a::Int,
                                       b::Int,
                                       ws::_SSPageWorkspace{K}) where {K}
    field = field_from_eltype(K)
    p = _ss_p(first, a, b)
    t = a + b

    dim_t = _ss_totdim(Tot, t)
    blocks_t = _ss_blocks_at(blocks, Tot.tmin, t)
    high = _ss_Fp_range(first, blocks_t, dim_t, t, p)
    dim_tp1 = _ss_totdim(Tot, t + 1)
    blocks_tp1 = _ss_blocks_at(blocks, Tot.tmin, t + 1)
    low_tp1 = _ss_low_range(first, blocks_tp1, dim_tp1, t + 1, p + r)
    low_p1 = _ss_low_range(first, blocks_t, dim_t, t, p + 1)
    dim_tm1 = _ss_totdim(Tot, t - 1)
    blocks_tm1 = _ss_blocks_at(blocks, Tot.tmin, t - 1)
    dom_range = _ss_Fp_range(first, blocks_tm1, dim_tm1, t - 1, p - r + 1)
    low_p = _ss_low_range(first, blocks_t, dim_t, t, p)

    Z = _ss_Z_basis(Tot, t, high, low_tp1)
    B = _ss_B_basis(Tot, t, dim_t, dom_range, low_p)
    Zint = _ss_Z_intersection_Fp1(Z, low_p1)
    Den = if _use_page_workspace(K)
        FieldLinAlg.colspace(field, _ss_den_buf!(ws, B, Zint))
    elseif _use_precolspace_den(K)
        FieldLinAlg.colspace(field, hcat(B, Zint))
    else
        hcat(B, Zint)
    end
    return subquotient_data(Z, Den)
end

function _ss_compute_page_term_exact(DC::DoubleComplex{K},
                                     Tot::CochainComplex{K},
                                     blocks::Vector{Vector{NTuple{4,Int}}},
                                     first::Symbol,
                                     r::Int,
                                     a::Int,
                                     b::Int,
                                     ws::_SSPageWorkspace{K}) where {K}
    field = field_from_eltype(K)
    p = _ss_p(first, a, b)
    t = a + b

    dim_t = _ss_totdim(Tot, t)
    blocks_t = _ss_blocks_at(blocks, Tot.tmin, t)
    high = _ss_Fp_range(first, blocks_t, dim_t, t, p)
    dim_tp1 = _ss_totdim(Tot, t + 1)
    blocks_tp1 = _ss_blocks_at(blocks, Tot.tmin, t + 1)
    low_tp1 = _ss_low_range(first, blocks_tp1, dim_tp1, t + 1, p + r)
    low_p1 = _ss_low_range(first, blocks_t, dim_t, t, p + 1)
    dim_tm1 = _ss_totdim(Tot, t - 1)
    blocks_tm1 = _ss_blocks_at(blocks, Tot.tmin, t - 1)
    dom_range = _ss_Fp_range(first, blocks_tm1, dim_tm1, t - 1, p - r + 1)
    low_p = _ss_low_range(first, blocks_t, dim_t, t, p)

    Z, Zcoords = _ss_Z_basis_with_coords(Tot, t, high, low_tp1)
    B = _ss_B_basis(Tot, t, dim_t, dom_range, low_p)
    Zint_coords = _ss_Z_intersection_Fp1_coords(Z, low_p1)
    Zint = size(Zint_coords, 2) == 0 ? zeros(K, dim_t, 0) : Z * Zint_coords
    Den = if size(B, 2) == 0
        Zint
    elseif size(Zint, 2) == 0
        B
    else
        FieldLinAlg.colspace(field, _ss_den_buf!(ws, B, Zint))
    end
    Dencoords = if size(Den, 2) == 0
        zeros(K, size(Zcoords, 2), 0)
    else
        _solve_fullcolumn_cached(field, Zcoords, @view Den[high, :])
    end
    return _subquotient_data_from_coords(Z, Dencoords; Zsolve_rows=high, Zsolve_basis=Zcoords)
end

function _ss_compute_page_spaces(DC::DoubleComplex{K},
                                 Tot::CochainComplex{K},
                                 blocks::Vector{Vector{NTuple{4,Int}}},
                                 first::Symbol,
                                 r::Int;
                                 compute_dims::Bool=true,
                                 known_dims::Union{Nothing,AbstractMatrix{Int}}=nothing) where {K}
    Alen = DC.amax - DC.amin + 1
    Blen = DC.bmax - DC.bmin + 1
    dims = compute_dims ? (known_dims === nothing ? zeros(Int, Alen, Blen) : Matrix{Int}(known_dims)) : nothing
    spaces = Array{SubquotientData{K},2}(undef, Alen, Blen)
    ws = _SSPageWorkspace(K)

    for ai in 1:Alen, bi in 1:Blen
        a = DC.amin + ai - 1
        b = DC.bmin + bi - 1
        dim_hint = known_dims === nothing ? -1 : known_dims[ai, bi]
        SQ = if dim_hint == 0
            _ss_zero_subquotient(K, _ss_totdim(Tot, a + b))
        else
            _ss_compute_page_term(DC, Tot, blocks, first, r, a, b, ws)
        end
        spaces[ai, bi] = SQ
        compute_dims && known_dims === nothing && (dims[ai, bi] = SQ.dimH)
    end

    return dims, spaces
end

function _ss_compute_page_data(DC::DoubleComplex{K},
                               Tot::CochainComplex{K},
                               blocks::Vector{Vector{NTuple{4,Int}}},
                               first::Symbol,
                               r::Int) where {K}
    dims, spaces = _ss_compute_page_spaces(DC, Tot, blocks, first, r; compute_dims=true)
    return SSPageData{K}(r, dims, spaces)
end

function _ss_compute_page2_data(ss::SpectralSequence{K}) where {K}
    dims, spaces = _ss_compute_page_spaces(ss.DC, ss.Tot, ss.blocks, ss.first, 2;
                                           compute_dims=true,
                                           known_dims=ss.E2_dims)
    return SSPageData{K}(2, dims, spaces)
end

function _ss_compute_page_data_ambient(DC::DoubleComplex{K},
                                       Tot::CochainComplex{K},
                                       blocks::Vector{Vector{NTuple{4,Int}}},
                                       first::Symbol,
                                       r::Int) where {K}
    Alen = DC.amax - DC.amin + 1
    Blen = DC.bmax - DC.bmin + 1
    dims = zeros(Int, Alen, Blen)
    spaces = Array{SubquotientData{K},2}(undef, Alen, Blen)
    ws = _SSPageWorkspace(K)
    for ai in 1:Alen, bi in 1:Blen
        a = DC.amin + ai - 1
        b = DC.bmin + bi - 1
        SQ = _ss_compute_page_term_ambient(DC, Tot, blocks, first, r, a, b, ws)
        spaces[ai, bi] = SQ
        dims[ai, bi] = SQ.dimH
    end
    return SSPageData{K}(r, dims, spaces)
end

function _ss_compute_differential_spaces(DC::DoubleComplex{K},
                                         Tot::CochainComplex{K},
                                         first::Symbol,
                                         spaces::AbstractMatrix{SubquotientData{K}},
                                         r::Int) where {K}
    Alen = DC.amax - DC.amin + 1
    Blen = DC.bmax - DC.bmin + 1
    dr = Array{SparseMatrixCSC{K,Int},2}(undef, Alen, Blen)
    Itrip = Int[]
    Jtrip = Int[]
    Vtrip = K[]

    for ai in 1:Alen, bi in 1:Blen
        a = DC.amin + ai - 1
        b = DC.bmin + bi - 1
        src = spaces[ai, bi]
        dim_src = src.dimH

        a2, b2 = _ss_dr_target(first, r, a, b)

        if a2 < DC.amin || a2 > DC.amax || b2 < DC.bmin || b2 > DC.bmax
            dr[ai, bi] = spzeros(K, 0, dim_src)
            continue
        end

        tgt = spaces[a2 - DC.amin + 1, b2 - DC.bmin + 1]
        dim_tgt = tgt.dimH

        if dim_src == 0 || dim_tgt == 0
            dr[ai, bi] = spzeros(K, dim_tgt, dim_src)
            continue
        end

        tdeg = a + b
        d = _diff_at(Tot, tdeg)
        src_sq = src
        tgt_sq = tgt

        empty!(Itrip)
        empty!(Jtrip)
        empty!(Vtrip)

        if field_from_eltype(K) isa QQField && _use_exact_diff_coords()
            coords = _subquotient_pushforward_coords(src_sq, tgt_sq, d)
            @inbounds for j in 1:dim_src
                for i in 1:dim_tgt
                    aij = coords[i, j]
                    iszero(aij) && continue
                    push!(Itrip, i)
                    push!(Jtrip, j)
                    push!(Vtrip, aij)
                end
            end
        elseif _use_batched_coordinate_solves(K, dim_src, dim_tgt)
            coords = subquotient_coordinates(tgt_sq, d * src_sq.Hrep)
            @inbounds for j in 1:dim_src
                for i in 1:dim_tgt
                    aij = coords[i, j]
                    iszero(aij) && continue
                    push!(Itrip, i)
                    push!(Jtrip, j)
                    push!(Vtrip, aij)
                end
            end
        else
            for j in 1:dim_src
                vec = d * src_sq.Hrep[:, j]
                coords = subquotient_coordinates(tgt_sq, vec)[:, 1]
                @inbounds for i in 1:dim_tgt
                    aij = coords[i]
                    iszero(aij) && continue
                    push!(Itrip, i)
                    push!(Jtrip, j)
                    push!(Vtrip, aij)
                end
            end
        end

        dr[ai, bi] = sparse(Itrip, Jtrip, Vtrip, dim_tgt, dim_src)
    end

    return dr
end

function _ss_compute_differential(DC::DoubleComplex{K},
                                  Tot::CochainComplex{K},
                                  first::Symbol,
                                  pagedata::SSPageData{K},
                                  r::Int) where {K}
    return _ss_compute_differential_spaces(DC, Tot, first, pagedata.spaces, r)
end

function _total_complex_with_blocks(DC::DoubleComplex{K}) where {K}
    amin, amax = DC.amin, DC.amax
    bmin, bmax = DC.bmin, DC.bmax

    tmin = amin + bmin
    tmax = amax + bmax
    blocks = _ss_blocks(DC)
    na = amax - amin + 1
    start_tables = Vector{Vector{Int}}(undef, length(blocks))
    for tidx in eachindex(blocks)
        tab = zeros(Int, na)
        @inbounds for blk in blocks[tidx]
            tab[blk[1] - amin + 1] = blk[3]
        end
        start_tables[tidx] = tab
    end

    dims_tot = Int[_ss_blocks_dim(blocks_t) for blocks_t in blocks]
    d_tot = Vector{SparseMatrixCSC{K,Int}}(undef, max(0, length(dims_tot) - 1))

    for tidx in 1:length(d_tot)
        blocks_t = blocks[tidx]
        starts_tp1 = start_tables[tidx + 1]
        dom_dim = dims_tot[tidx]
        cod_dim = dims_tot[tidx + 1]

        I = Int[]
        J = Int[]
        V = K[]

        nnz_cap = 0
        @inbounds for blk in blocks_t
            a = blk[1]
            b = blk[2]
            aidx = a - amin + 1
            bidx = b - bmin + 1
            b < bmax && (nnz_cap += nnz(DC.dv[aidx, bidx]))
            a < amax && (nnz_cap += nnz(DC.dh[aidx, bidx]))
        end
        sizehint!(I, nnz_cap)
        sizehint!(J, nnz_cap)
        sizehint!(V, nnz_cap)

        @inbounds for blk in blocks_t
            a = blk[1]
            b = blk[2]
            dom0 = blk[3]
            aidx = a - amin + 1
            bidx = b - bmin + 1

            if b < bmax
                cod0 = starts_tp1[a - amin + 1]
                if cod0 != 0
                    B = DC.dv[aidx, bidx]
                    @inbounds for col in 1:size(B, 2)
                        j = dom0 + col - 1
                        for ptr in B.colptr[col]:(B.colptr[col + 1] - 1)
                            push!(I, cod0 + B.rowval[ptr] - 1)
                            push!(J, j)
                            push!(V, B.nzval[ptr])
                        end
                    end
                end
            end

            if a < amax
                cod0 = starts_tp1[a - amin + 2]
                if cod0 != 0
                    B = DC.dh[aidx, bidx]
                    @inbounds for col in 1:size(B, 2)
                        j = dom0 + col - 1
                        for ptr in B.colptr[col]:(B.colptr[col + 1] - 1)
                            push!(I, cod0 + B.rowval[ptr] - 1)
                            push!(J, j)
                            push!(V, B.nzval[ptr])
                        end
                    end
                end
            end
        end

        d_tot[tidx] = sparse(I, J, V, cod_dim, dom_dim)
    end

    return CochainComplex{K}(tmin, tmax, dims_tot, d_tot), blocks
end

function _ss_inclusion_coords(tgt::CohomologyData{K},
                              src::CohomologyData{K},
                              ambient_dim::Int,
                              r::UnitRange{Int}) where {K}
    if src.dimH == 0 || tgt.dimH == 0
        return zeros(K, tgt.dimH, src.dimH)
    end
    Y = zeros(K, ambient_dim, src.dimH)
    @views Y[r, :] .= src.Hrep
    return cohomology_coordinates(tgt, Y)
end

const _spectral_exact_horizontal_filtimg_mode = Ref{Symbol}(:auto)
const _spectral_exact_vertical_filtimg_mode = Ref{Symbol}(:auto)

@inline function _exact_horizontal_filtimg_auto(::Type{K},
                                                first::Symbol,
                                                Htot_dims::AbstractVector{Int},
                                                fp_ranges::AbstractMatrix{UnitRange{Int}}) where {K}
    if first != :horizontal || !(field_from_eltype(K) isa QQField)
        return false
    end
    plen, tlen = size(fp_ranges)
    total_hdim = sum(Htot_dims)
    repeated = any(_ss_exact_filtimg_cache_plan(fp_ranges; prefer_cache=false))
    return repeated && (plen * tlen >= 32 || total_hdim >= 72)
end

@inline function _use_exact_horizontal_filtimg(::Type{K},
                                               first::Symbol,
                                               Htot_dims::AbstractVector{Int},
                                               fp_ranges::AbstractMatrix{UnitRange{Int}}) where {K}
    mode = _spectral_exact_horizontal_filtimg_mode[]
    if mode === :optimized
        return false
    elseif mode === :legacy
        return first == :horizontal && (field_from_eltype(K) isa QQField)
    elseif mode === :auto
        return _exact_horizontal_filtimg_auto(K, first, Htot_dims, fp_ranges)
    end
    error("_spectral_exact_horizontal_filtimg_mode must be :auto, :optimized, or :legacy")
end

@inline function _use_exact_vertical_filtimg_batched(::Type{K}, first::Symbol) where {K}
    if first != :vertical || !(field_from_eltype(K) isa QQField)
        return false
    end
    mode = _spectral_exact_vertical_filtimg_mode[]
    mode === :batched && return true
    mode === :direct && return false
    mode === :auto && return false
    error("_spectral_exact_vertical_filtimg_mode must be :auto, :batched, or :direct")
end

@inline function _use_exact_filtimg_cache(repeats::Bool)
    mode = _spectral_exact_filtimg_cache_mode[]
    mode === :on && return true
    mode === :off && return false
    mode === :auto && return repeats
    error("_spectral_exact_filtimg_cache_mode must be :auto, :on, or :off")
end

function _ss_fp_ranges(Tot::CochainComplex{K},
                       blocks::Vector{Vector{NTuple{4,Int}}},
                       first::Symbol,
                       pmin::Int,
                       pmax::Int) where {K}
    tmin = Tot.tmin
    tmax = Tot.tmax
    tlen = tmax - tmin + 1
    plen = pmax - pmin + 1
    fp_ranges = Matrix{UnitRange{Int}}(undef, plen, tlen)
    for ip in 1:plen
        p = pmin + ip - 1
        for tidx in 1:tlen
            t = tmin + tidx - 1
            blocks_t = blocks[tidx]
            dim_t = _ss_blocks_dim(blocks_t)
            fp_ranges[ip, tidx] = _ss_Fp_range(first, blocks_t, dim_t, t, p)
        end
    end
    return fp_ranges
end

function _ss_exact_filtimg_cache_plan(fp_ranges::AbstractMatrix{UnitRange{Int}}; prefer_cache::Bool=false)
    plen, tlen = size(fp_ranges)
    prefer_cache && return fill(true, tlen)
    use_cache = falses(tlen)
    empty_range = 1:0
    for tidx in 1:tlen
        seen = Set{NTuple{6,Int}}()
        for ip in 1:plen
            r_t = fp_ranges[ip, tidx]
            r_tm1 = tidx == 1 ? empty_range : fp_ranges[ip, tidx - 1]
            r_tp1 = tidx == tlen ? empty_range : fp_ranges[ip, tidx + 1]
            sig = _ss_filtered_signature(r_tm1, r_t, r_tp1)
            if sig in seen
                use_cache[tidx] = _use_exact_filtimg_cache(true)
                break
            end
            push!(seen, sig)
        end
        use_cache[tidx] || (use_cache[tidx] = _use_exact_filtimg_cache(false))
    end
    return use_cache
end

@inline function _exact_columnwise_filtimg_auto(ss::SpectralSequence{K}, tidx::Int) where {K}
    if !(field_from_eltype(K) isa QQField)
        return false
    end
    plen = size(ss.fp_ranges, 1)
    hdim = ss.Htot_dims[tidx]
    hdim == 0 && return false
    total_img = sum(@view ss.filt_img_dims[:, tidx])
    max_img = maximum(@view ss.filt_img_dims[:, tidx])
    work = plen * hdim
    return work >= 64 || total_img >= 64 || max_img >= 16
end

@inline function _use_exact_columnwise_filtimg_basis(ss::SpectralSequence{K}, tidx::Int) where {K}
    if !(field_from_eltype(K) isa QQField)
        return false
    end
    mode = _spectral_exact_filtimg_basis_mode[]
    mode === :columnwise && return true
    mode === :full && return false
    mode === :auto && return true
    error("_spectral_exact_filtimg_basis_mode must be :auto, :columnwise, or :full")
end

@inline function _use_exact_columnwise_einf_basis(ss::SpectralSequence{K}, tidx::Int) where {K}
    if !(field_from_eltype(K) isa QQField)
        return false
    end
    mode = _spectral_exact_filtimg_basis_mode[]
    mode === :columnwise && return true
    mode === :full && return false
    mode === :auto && return _exact_columnwise_filtimg_auto(ss, tidx)
    error("_spectral_exact_filtimg_basis_mode must be :auto, :columnwise, or :full")
end

@inline function _ss_filtimg_dim(field::AbstractCoeffField,
                                 Htot_t::CohomologyData{K},
                                 dimHtot_t::Int,
                                 Ht::CohomologyData{K},
                                 ambient_dim::Int,
                                 r_t::UnitRange{Int}) where {K}
    if dimHtot_t == 0 || Ht.dimH == 0
        return 0
    end
    return FieldLinAlg.rank(field, _ss_inclusion_coords(Htot_t, Ht, ambient_dim, r_t))
end

function _ss_build_filt_img_exact_dims(Tot::CochainComplex{K},
                                       Htot::Vector{CohomologyData{K}},
                                       Htot_dims::Vector{Int},
                                       fp_ranges::AbstractMatrix{UnitRange{Int}};
                                       prefer_cache::Bool=false) where {K}
    field = field_from_eltype(K)
    tmin = Tot.tmin
    tmax = Tot.tmax
    tlen = tmax - tmin + 1
    plen = size(fp_ranges, 1)
    dims = zeros(Int, plen, tlen)
    hf_cache = [Dict{NTuple{6,Int},CohomologyData{K}}() for _ in 1:tlen]
    imgdim_cache = [Dict{NTuple{6,Int},Int}() for _ in 1:tlen]
    use_cache = _ss_exact_filtimg_cache_plan(fp_ranges; prefer_cache=prefer_cache)
    empty_range = 1:0

    for ip in 1:plen
        sigs = Vector{NTuple{6,Int}}(undef, tlen)
        for tidx in 1:tlen
            r_t = fp_ranges[ip, tidx]
            r_tm1 = tidx == 1 ? empty_range : fp_ranges[ip, tidx - 1]
            r_tp1 = tidx == tlen ? empty_range : fp_ranges[ip, tidx + 1]
            sig = _ss_filtered_signature(r_tm1, r_t, r_tp1)
            sigs[tidx] = sig
        end

        for tidx in 1:tlen
            sig = sigs[tidx]
            Ht = if use_cache[tidx]
                get!(hf_cache[tidx], sig) do
                    _ss_filtered_cohomology_data(Tot, fp_ranges, ip, tidx)
                end
            else
                _ss_filtered_cohomology_data(Tot, fp_ranges, ip, tidx)
            end
            if use_cache[tidx]
                dims[ip, tidx] = get!(imgdim_cache[tidx], sig) do
                    _ss_filtimg_dim(field, Htot[tidx], Htot_dims[tidx], Ht, Tot.dims[tidx], fp_ranges[ip, tidx])
                end
            else
                dims[ip, tidx] = _ss_filtimg_dim(field, Htot[tidx], Htot_dims[tidx], Ht, Tot.dims[tidx], fp_ranges[ip, tidx])
            end
        end
    end

    return dims
end

function _ss_filtered_complex(Tot::CochainComplex{K},
                              fp_ranges::AbstractMatrix{UnitRange{Int}},
                              ip::Int) where {K}
    tlen = size(fp_ranges, 2)
    dims = Vector{Int}(undef, tlen)
    d = Vector{SparseMatrixCSC{K,Int}}(undef, max(0, tlen - 1))
    for tidx in 1:tlen
        r_t = fp_ranges[ip, tidx]
        dims[tidx] = length(r_t)
        if tidx < tlen
            r_tp1 = fp_ranges[ip, tidx + 1]
            d[tidx] = Tot.d[tidx][r_tp1, r_t]
        end
    end
    return CochainComplex{K}(Tot.tmin, Tot.tmax, dims, d)
end

function _ss_build_filt_img_exact_dims_batched(Tot::CochainComplex{K},
                                               Htot::Vector{CohomologyData{K}},
                                               Htot_dims::Vector{Int},
                                               fp_ranges::AbstractMatrix{UnitRange{Int}}) where {K}
    field = field_from_eltype(K)
    tlen = Tot.tmax - Tot.tmin + 1
    plen = size(fp_ranges, 1)
    dims = zeros(Int, plen, tlen)
    for ip in 1:plen
        HFp = cohomology_data(_ss_filtered_complex(Tot, fp_ranges, ip))
        for tidx in 1:tlen
            dims[ip, tidx] = _ss_filtimg_dim(field, Htot[tidx], Htot_dims[tidx], HFp[tidx], Tot.dims[tidx], fp_ranges[ip, tidx])
        end
    end
    return dims
end

function _ss_build_filt_img_exact_bases(Tot::CochainComplex{K},
                                        Htot::Vector{CohomologyData{K}},
                                        Htot_dims::Vector{Int},
                                        fp_ranges::AbstractMatrix{UnitRange{Int}};
                                        prefer_cache::Bool=false) where {K}
    field = field_from_eltype(K)
    tmin = Tot.tmin
    tmax = Tot.tmax
    tlen = tmax - tmin + 1
    plen = size(fp_ranges, 1)
    filt_img = Array{Matrix{K},2}(undef, plen, tlen)
    hf_cache = [Dict{NTuple{6,Int},CohomologyData{K}}() for _ in 1:tlen]
    img_cache = [Dict{NTuple{6,Int},Matrix{K}}() for _ in 1:tlen]
    use_cache = _ss_exact_filtimg_cache_plan(fp_ranges; prefer_cache=prefer_cache)
    empty_range = 1:0

    for ip in 1:plen
        sigs = Vector{NTuple{6,Int}}(undef, tlen)
        for tidx in 1:tlen
            r_t = fp_ranges[ip, tidx]
            r_tm1 = tidx == 1 ? empty_range : fp_ranges[ip, tidx - 1]
            r_tp1 = tidx == tlen ? empty_range : fp_ranges[ip, tidx + 1]
            sig = _ss_filtered_signature(r_tm1, r_t, r_tp1)
            sigs[tidx] = sig
        end

        for tidx in 1:tlen
            sig = sigs[tidx]
            Ht = if use_cache[tidx]
                get!(hf_cache[tidx], sig) do
                    _ss_filtered_cohomology_data(Tot, fp_ranges, ip, tidx)
                end
            else
                _ss_filtered_cohomology_data(Tot, fp_ranges, ip, tidx)
            end
            if use_cache[tidx]
                filt_img[ip, tidx] = get!(img_cache[tidx], sig) do
                    if Htot_dims[tidx] == 0 || Ht.dimH == 0
                        zeros(K, Htot_dims[tidx], 0)
                    else
                        FieldLinAlg.colspace(field, _ss_inclusion_coords(Htot[tidx], Ht, Tot.dims[tidx], fp_ranges[ip, tidx]))
                    end
                end
            else
                filt_img[ip, tidx] =
                    if Htot_dims[tidx] == 0 || Ht.dimH == 0
                        zeros(K, Htot_dims[tidx], 0)
                    else
                        FieldLinAlg.colspace(field, _ss_inclusion_coords(Htot[tidx], Ht, Tot.dims[tidx], fp_ranges[ip, tidx]))
                    end
            end
        end
    end

    return filt_img
end

function _ss_build_filt_img_exact_bases_batched(Tot::CochainComplex{K},
                                                Htot::Vector{CohomologyData{K}},
                                                Htot_dims::Vector{Int},
                                                fp_ranges::AbstractMatrix{UnitRange{Int}}) where {K}
    field = field_from_eltype(K)
    tlen = Tot.tmax - Tot.tmin + 1
    plen = size(fp_ranges, 1)
    filt_img = Array{Matrix{K},2}(undef, plen, tlen)
    for ip in 1:plen
        HFp = cohomology_data(_ss_filtered_complex(Tot, fp_ranges, ip))
        for tidx in 1:tlen
            filt_img[ip, tidx] =
                if Htot_dims[tidx] == 0 || HFp[tidx].dimH == 0
                    zeros(K, Htot_dims[tidx], 0)
                else
                    FieldLinAlg.colspace(field, _ss_inclusion_coords(Htot[tidx], HFp[tidx], Tot.dims[tidx], fp_ranges[ip, tidx]))
                end
        end
    end
    return filt_img
end

function _ss_build_filt_img_exact_bases_t(Tot::CochainComplex{K},
                                          Htot::Vector{CohomologyData{K}},
                                          Htot_dims::Vector{Int},
                                          fp_ranges::AbstractMatrix{UnitRange{Int}},
                                          tidx::Int;
                                          prefer_cache::Bool=false) where {K}
    field = field_from_eltype(K)
    plen = size(fp_ranges, 1)
    bases = Vector{Matrix{K}}(undef, plen)
    hf_cache = Dict{NTuple{6,Int},CohomologyData{K}}()
    img_cache = Dict{NTuple{6,Int},Matrix{K}}()
    empty_range = 1:0
    repeated = false
    if !prefer_cache
        seen = Set{NTuple{6,Int}}()
        for ip in 1:plen
            r_t = fp_ranges[ip, tidx]
            r_tm1 = tidx == 1 ? empty_range : fp_ranges[ip, tidx - 1]
            r_tp1 = tidx == size(fp_ranges, 2) ? empty_range : fp_ranges[ip, tidx + 1]
            sig = _ss_filtered_signature(r_tm1, r_t, r_tp1)
            if sig in seen
                repeated = true
                break
            end
            push!(seen, sig)
        end
    end
    use_cache = _use_exact_filtimg_cache(prefer_cache || repeated)

    for ip in 1:plen
        r_t = fp_ranges[ip, tidx]
        r_tm1 = tidx == 1 ? empty_range : fp_ranges[ip, tidx - 1]
        r_tp1 = tidx == size(fp_ranges, 2) ? empty_range : fp_ranges[ip, tidx + 1]
        sig = _ss_filtered_signature(r_tm1, r_t, r_tp1)
        Ht = if use_cache
            get!(hf_cache, sig) do
                _ss_filtered_cohomology_data(Tot, fp_ranges, ip, tidx)
            end
        else
            _ss_filtered_cohomology_data(Tot, fp_ranges, ip, tidx)
        end
        if use_cache
            bases[ip] = get!(img_cache, sig) do
                if Htot_dims[tidx] == 0 || Ht.dimH == 0
                    zeros(K, Htot_dims[tidx], 0)
                else
                    FieldLinAlg.colspace(field, _ss_inclusion_coords(Htot[tidx], Ht, Tot.dims[tidx], r_t))
                end
            end
        else
            bases[ip] =
                if Htot_dims[tidx] == 0 || Ht.dimH == 0
                    zeros(K, Htot_dims[tidx], 0)
                else
                    FieldLinAlg.colspace(field, _ss_inclusion_coords(Htot[tidx], Ht, Tot.dims[tidx], r_t))
                end
        end
    end

    return bases
end

# -----------------------------------------------------------------------------
# Public constructor
# -----------------------------------------------------------------------------

function spectral_sequence(DC::DoubleComplex{K}; first::Symbol = :vertical) where {K}
    _ss_check_first(first)

    Tot, blocks = _total_complex_with_blocks(DC)
    Htot = cohomology_data(Tot)
    Htot_dims = [H.dimH for H in Htot]
    field = field_from_eltype(K)

    if first == :vertical
        pmin = DC.amin
        pmax = DC.amax + 1
        rmax_possible = (DC.amax - DC.amin) + 1
    else
        pmin = DC.bmin
        pmax = DC.bmax + 1
        rmax_possible = (DC.bmax - DC.bmin) + 1
    end

    tmin = Tot.tmin
    tmax = Tot.tmax
    tlen = tmax - tmin + 1
    plen = pmax - pmin + 1
    fp_ranges = _ss_fp_ranges(Tot, blocks, first, pmin, pmax)
    use_lazy_exact_filtimg = field isa QQField && (first == :vertical || _use_exact_horizontal_filtimg(K, first, Htot_dims, fp_ranges))
    prefer_exact_filtimg_cache = field isa QQField && first == :horizontal
    use_batched_vertical_filtimg = _use_exact_vertical_filtimg_batched(K, first)

    filt_img_dims, filt_img = if use_lazy_exact_filtimg
        ((use_batched_vertical_filtimg ?
          _ss_build_filt_img_exact_dims_batched(Tot, Htot, Htot_dims, fp_ranges) :
          _ss_build_filt_img_exact_dims(Tot, Htot, Htot_dims, fp_ranges; prefer_cache=prefer_exact_filtimg_cache)),
         _LazyValue{Array{Matrix{K},2}}(nothing))
    else
        tmp = Array{Matrix{K},2}(undef, plen, tlen)

        hf_cache = [Dict{NTuple{6,Int},CohomologyData{K}}() for _ in 1:tlen]
        img_cache = [Dict{NTuple{6,Int},Matrix{K}}() for _ in 1:tlen]
        empty_range = 1:0

        for ip in 1:plen
            for tidx in 1:tlen
                r_t = fp_ranges[ip, tidx]
                r_tm1 = tidx == 1 ? empty_range : fp_ranges[ip, tidx - 1]
                r_tp1 = tidx == tlen ? empty_range : fp_ranges[ip, tidx + 1]
                sig = _ss_filtered_signature(r_tm1, r_t, r_tp1)

                Ht = if haskey(hf_cache[tidx], sig)
                    hf_cache[tidx][sig]
                else
                    Hloc = _ss_filtered_cohomology_data(Tot, fp_ranges, ip, tidx)
                    hf_cache[tidx][sig] = Hloc
                    Hloc
                end

                if haskey(img_cache[tidx], sig)
                    tmp[ip, tidx] = img_cache[tidx][sig]
                else
                    M = if Htot_dims[tidx] == 0 || Ht.dimH == 0
                        zeros(K, Htot_dims[tidx], 0)
                    else
                        FieldLinAlg.colspace(field, _ss_inclusion_coords(Htot[tidx], Ht, Tot.dims[tidx], r_t))
                    end
                    img_cache[tidx][sig] = M
                    tmp[ip, tidx] = M
                end
            end
        end
        dims = zeros(Int, plen, tlen)
        for ip in 1:plen, tidx in 1:tlen
            dims[ip, tidx] = size(tmp[ip, tidx], 2)
        end
        (dims, _LazyValue{Array{Matrix{K},2}}(tmp))
    end

    Alen = DC.amax - DC.amin + 1
    Blen = DC.bmax - DC.bmin + 1
    Einf_dims = zeros(Int, Alen, Blen)

    for ai in 1:Alen, bi in 1:Blen
        a = DC.amin + ai - 1
        b = DC.bmin + bi - 1
        p = _ss_p(first, a, b)
        t = a + b

        tidx = t - tmin + 1
        ip = p - pmin + 1
        ip1 = ip + 1
        Einf_dims[ai, bi] = max(0, filt_img_dims[ip, tidx] - filt_img_dims[ip1, tidx])
    end

    page_cache = Dict{Int,SSPageData{K}}()
    diff_cache = Dict{Int,Array{SparseMatrixCSC{K,Int},2}}()
    split_cache = Dict{Int,SSSplitData{K}}()
    E1 = _ss_compute_page_data(DC, Tot, blocks, first, 1)
    d1 = _ss_compute_differential(DC, Tot, first, E1, 1)
    E2_dims = _ss_E2_dims_from_d1(DC, first, E1.dims, d1)

    page_cache[1] = E1
    diff_cache[1] = d1

    return SpectralSequence{K}(DC, first,
                               E1.dims, d1, E2_dims,
                               Einf_dims, Htot_dims,
                               Tot, Htot,
                               pmin, pmax, rmax_possible,
                               blocks, fp_ranges, filt_img_dims, filt_img,
                               [_LazyValue{Vector{Matrix{K}}}(nothing) for _ in 1:tlen],
                               _LazyValue{Array{SubquotientData{K},2}}(nothing),
                               page_cache, diff_cache, split_cache)
end

function _ss_filt_img_bases!(ss::SpectralSequence{K}) where {K}
    bases = ss.filt_img.value
    bases !== nothing && return bases
    prefer_cache = field_from_eltype(K) isa QQField && ss.first == :horizontal
    use_batched_vertical_filtimg = _use_exact_vertical_filtimg_batched(K, ss.first)
    bases = use_batched_vertical_filtimg ?
        _ss_build_filt_img_exact_bases_batched(ss.Tot, ss.Htot, ss.Htot_dims, ss.fp_ranges) :
        _ss_build_filt_img_exact_bases(ss.Tot, ss.Htot, ss.Htot_dims, ss.fp_ranges; prefer_cache=prefer_cache)
    ss.filt_img.value = bases
    for tidx in 1:size(bases, 2)
        ss.filt_img_cols[tidx].value = collect(@view bases[:, tidx])
    end
    return bases
end

function _ss_filt_img_basis_col!(ss::SpectralSequence{K}, tidx::Int) where {K}
    bases = ss.filt_img.value
    if bases !== nothing
        col = ss.filt_img_cols[tidx].value
        if col === nothing
            col = collect(@view bases[:, tidx])
            ss.filt_img_cols[tidx].value = col
        end
        return col
    end
    col = ss.filt_img_cols[tidx].value
    col !== nothing && return col
    prefer_cache = field_from_eltype(K) isa QQField && ss.first == :horizontal
    col = _ss_build_filt_img_exact_bases_t(ss.Tot, ss.Htot, ss.Htot_dims, ss.fp_ranges, tidx; prefer_cache=prefer_cache)
    ss.filt_img_cols[tidx].value = col
    return col
end

function _ss_build_einf_spaces(ss::SpectralSequence{K}) where {K}
    Alen = ss.DC.amax - ss.DC.amin + 1
    Blen = ss.DC.bmax - ss.DC.bmin + 1
    spaces = Array{SubquotientData{K},2}(undef, Alen, Blen)
    tcols = Vector{Union{Nothing,Vector{Matrix{K}}}}(undef, length(ss.Htot_dims))
    fill!(tcols, nothing)
    use_col = BitVector(undef, length(ss.Htot_dims))
    need_full = false
    for tidx in eachindex(use_col)
        use_col[tidx] = _use_exact_columnwise_einf_basis(ss, tidx)
        need_full |= !use_col[tidx]
    end
    full_bases = need_full ? _ss_filt_img_bases!(ss) : nothing

    for ai in 1:Alen, bi in 1:Blen
        a = ss.DC.amin + ai - 1
        b = ss.DC.bmin + bi - 1
        p = _ss_p(ss.first, a, b)
        t = a + b
        tidx = t - ss.Tot.tmin + 1
        dimH = ss.Htot_dims[tidx]

        if dimH == 0
            spaces[ai, bi] = _ss_zero_subquotient(K, 0)
            continue
        end

        if ss.Einf_dims[ai, bi] == 0
            spaces[ai, bi] = _ss_zero_subquotient(K, dimH)
            continue
        end

        ip = p - ss.pmin + 1
        ip1 = ip + 1
        col = tcols[tidx]
        if col === nothing
            col = use_col[tidx] ?
                _ss_filt_img_basis_col!(ss, tidx) :
                collect(@view full_bases[:, tidx])
            tcols[tidx] = col
        end
        spaces[ai, bi] = subquotient_data(col[ip], col[ip1])
    end

    return spaces
end

function _ss_einf_spaces!(ss::SpectralSequence{K}) where {K}
    spaces = ss.Einf_spaces.value
    spaces !== nothing && return spaces

    spaces = _ss_build_einf_spaces(ss)

    ss.Einf_spaces.value = spaces
    return spaces
end

# -----------------------------------------------------------------------------
# Public API: pages, terms, differentials
# -----------------------------------------------------------------------------

function page(ss::SpectralSequence{K}, r::Int) where {K}
    if r < 1
        error("page(ss,r): r must be >= 1")
    end
    if r >= ss.rmax_possible
        return SpectralPage{K}(ss, :inf, ss.Einf_dims)
    end
    if r == 2
        return SpectralPage{K}(ss, 2, ss.E2_dims)
    end
    if !haskey(ss.page_cache, r)
        ss.page_cache[r] = _ss_compute_page_data(ss.DC, ss.Tot, ss.blocks, ss.first, r)
    end
    return SpectralPage{K}(ss, r, ss.page_cache[r].dims)
end

function page(ss::SpectralSequence{K}, r::Symbol) where {K}
    if r == :inf || r == :infty
        return SpectralPage{K}(ss, :inf, ss.Einf_dims)
    end
    error("page(ss,r): unsupported symbol")
end

function term(ss::SpectralSequence{K}, r::Int, ab::Tuple{Int,Int}) where {K}
    a, b = ab
    if r >= ss.rmax_possible
        return term(ss, :inf, ab)
    end
    if r == 2 && !haskey(ss.page_cache, 2)
        ss.page_cache[2] = _ss_compute_page2_data(ss)
    end
    if !haskey(ss.page_cache, r)
        ss.page_cache[r] = _ss_compute_page_data(ss.DC, ss.Tot, ss.blocks, ss.first, r)
    end
    return ss.page_cache[r].spaces[a - ss.DC.amin + 1, b - ss.DC.bmin + 1]
end

function term(ss::SpectralSequence{K}, r::Symbol, ab::Tuple{Int,Int}) where {K}
    a, b = ab
    spaces = _ss_einf_spaces!(ss)
    return spaces[a - ss.DC.amin + 1, b - ss.DC.bmin + 1]
end

function _ss_compute_page_and_differential(DC::DoubleComplex{K},
                                           Tot::CochainComplex{K},
                                           blocks::Vector{Vector{NTuple{4,Int}}},
                                           first::Symbol,
                                           r::Int,
                                           pagedata::Union{Nothing,SSPageData{K}}=nothing) where {K}
    pagedata === nothing && (pagedata = _ss_compute_page_data(DC, Tot, blocks, first, r))
    return pagedata, _ss_compute_differential(DC, Tot, first, pagedata, r)
end

function differential(ss::SpectralSequence{K}, r::Int) where {K}
    if r < 1
        error("differential(ss,r): r must be >= 1")
    end
    if r >= ss.rmax_possible
        Alen = ss.DC.amax - ss.DC.amin + 1
        Blen = ss.DC.bmax - ss.DC.bmin + 1
        out = Array{SparseMatrixCSC{K,Int},2}(undef, Alen, Blen)
        for ai in 1:Alen, bi in 1:Blen
            out[ai, bi] = spzeros(K, 0, ss.Einf_dims[ai, bi])
        end
        return out
    end
    if !haskey(ss.diff_cache, r)
        if haskey(ss.page_cache, r)
            ss.diff_cache[r] = _ss_compute_differential(ss.DC, ss.Tot, ss.first, ss.page_cache[r], r)
        elseif r == 2
            pagedata = _ss_compute_page2_data(ss)
            _, ss.diff_cache[r] = _ss_compute_page_and_differential(ss.DC, ss.Tot, ss.blocks, ss.first, r, pagedata)
        elseif field_from_eltype(K) isa QQField && r >= 2
            _, ss.diff_cache[r] = _ss_compute_page_and_differential(ss.DC, ss.Tot, ss.blocks, ss.first, r)
        else
            ss.page_cache[r] = _ss_compute_page_data(ss.DC, ss.Tot, ss.blocks, ss.first, r)
            ss.diff_cache[r] = _ss_compute_differential(ss.DC, ss.Tot, ss.first, ss.page_cache[r], r)
        end
    end
    return ss.diff_cache[r]
end

function differential(ss::SpectralSequence{K}, r::Int, ab::Tuple{Int,Int}) where {K}
    a, b = ab
    dr = differential(ss, r)
    return dr[a - ss.DC.amin + 1, b - ss.DC.bmin + 1]
end

# -----------------------------------------------------------------------------
# Convenience helpers for mathematician-friendly access
# -----------------------------------------------------------------------------

"""
    E_r(ss, r) -> SpectralPage

Alias for `page(ss, r)`. This matches the common notation `E_r`.
"""
E_r(ss::SpectralSequence{K}, r::Union{Int,Symbol}) where {K} = page(ss, r)

"""
    page_terms(ss, r) -> Array{SubquotientData{K},2}

Return explicit `SubquotientData` models for all bidegrees on page `r`.

- For `r::Int < ss.rmax_possible`, this returns the cached/constructed page data.
- For `r >= ss.rmax_possible` and for `r == :inf/:infty`, this returns the `E_infty`
  terms (the graded pieces of the induced filtration on total cohomology).

This is useful when you want to iterate over all (a,b) and not just access
dimensions via `page(ss,r)`.
"""
function page_terms(ss::SpectralSequence{K}, r::Int) where {K}
    if r < 1
        error("page_terms(ss,r): r must be >= 1")
    end
    if r >= ss.rmax_possible
        return _ss_einf_spaces!(ss)
    end
    if r == 2 && !haskey(ss.page_cache, 2)
        ss.page_cache[2] = _ss_compute_page2_data(ss)
    end
    if !haskey(ss.page_cache, r)
        ss.page_cache[r] = _ss_compute_page_data(ss.DC, ss.Tot, ss.blocks, ss.first, r)
    end
    return ss.page_cache[r].spaces
end

function page_terms(ss::SpectralSequence{K}, r::Symbol) where {K}
    if r == :inf || r == :infty
        return _ss_einf_spaces!(ss)
    end
    error("page_terms(ss,r): unsupported symbol")
end

# Internal helper used by page_terms_dict/page_dims_dict and higher-level wrappers.
function _ss_page_data(ss::SpectralSequence{K}, r::Union{Int,Symbol}) where {K}
    if r isa Int
        if r < 1
            error("_ss_page_data: r must be >= 1")
        end
        if r >= ss.rmax_possible
            return SSPageData{K}(r, ss.Einf_dims, _ss_einf_spaces!(ss))
        end
        if r == 2 && !haskey(ss.page_cache, 2)
            ss.page_cache[2] = _ss_compute_page2_data(ss)
        end
        if !haskey(ss.page_cache, r)
            ss.page_cache[r] = _ss_compute_page_data(ss.DC, ss.Tot, ss.blocks, ss.first, r)
        end
        return ss.page_cache[r]
    end
    if r == :inf || r == :infty
        return SSPageData{K}(ss.rmax_possible, ss.Einf_dims, _ss_einf_spaces!(ss))
    end
    error("_ss_page_data: unsupported symbol")
end

"""
    dr_target(ss, r, (a,b)) -> Union{Tuple{Int,Int},Nothing}

Return the bidegree target of the r-th differential `d_r` starting at `(a,b)`.

Conventions:
- If `ss.first == :vertical`, then `d_r : E_r^{a,b} -> E_r^{a+r, b-r+1}`.
- If `ss.first == :horizontal`, then `d_r : E_r^{a,b} -> E_r^{a-r+1, b+r}`.

Returns `nothing` if the target lies outside the bidegree window of `ss.DC`.
"""
function dr_target(ss::SpectralSequence{K}, r::Int, ab::Tuple{Int,Int}) where {K}
    a, b = ab
    a2, b2 = _ss_dr_target(ss.first, r, a, b)
    if a2 < ss.DC.amin || a2 > ss.DC.amax || b2 < ss.DC.bmin || b2 > ss.DC.bmax
        return nothing
    end
    return (a2, b2)
end

dr_target(ss::SpectralSequence{K}, r::Int, a::Int, b::Int) where {K} = dr_target(ss, r, (a, b))

"""
    dr_source(ss, r, (a,b)) -> Union{Tuple{Int,Int},Nothing}

Return the bidegree source of the r-th differential `d_r` that lands in `(a,b)`.

This is the inverse of `dr_target` on bidegrees:
- If `ss.first == :vertical`, then sources are at `(a-r, b+r-1)`.
- If `ss.first == :horizontal`, then sources are at `(a+r-1, b-r)`.

Returns `nothing` if the source lies outside the bidegree window of `ss.DC`.
"""
function dr_source(ss::SpectralSequence{K}, r::Int, ab::Tuple{Int,Int}) where {K}
    a, b = ab
    a0, b0 = if ss.first == :vertical
        (a - r, b + r - 1)
    else
        (a + r - 1, b - r)
    end
    if a0 < ss.DC.amin || a0 > ss.DC.amax || b0 < ss.DC.bmin || b0 > ss.DC.bmax
        return nothing
    end
    return (a0, b0)
end

dr_source(ss::SpectralSequence{K}, r::Int, a::Int, b::Int) where {K} = dr_source(ss, r, (a, b))

# Allow `ss[r]` and `ss[r,(a,b)]` for quick interactive exploration.
Base.getindex(ss::SpectralSequence{K}, r::Union{Int,Symbol}) where {K} = page(ss, r)
Base.getindex(ss::SpectralSequence{K}, r::Union{Int,Symbol}, ab::Tuple{Int,Int}) where {K} = page(ss, r)[ab]

# A small convenience: `differential(ss, :inf)` returns the zero differential.
function differential(ss::SpectralSequence{K}, r::Symbol) where {K}
    if r == :inf || r == :infty
        return differential(ss, ss.rmax_possible)
    end
    error("differential(ss,r): unsupported symbol")
end

# -----------------------------------------------------------------------------
# Multiplicative structures (optional)
# -----------------------------------------------------------------------------

"""
    product_matrix(ss, r, (a,b), (c,d), mul) -> Matrix{K}

Given a multiplication `mul` on the total complex `Tot`, compute the induced bilinear
product on the E_r page as a matrix.

The induced product is

    E_r^{a,b} (x) E_r^{c,d}  ->  E_r^{a+c, b+d}.

Inputs:
- `r` must be an integer page number.
- `mul` must be a function with signature

      mul(t1::Int, x::Vector{K}, t2::Int, y::Vector{K}) -> Vector{K}

  where `x` is a cochain in Tot^{t1} (in the cochain basis used internally by `ss.Tot`)
  and `y` is a cochain in Tot^{t2}. The output must be a cochain in Tot^{t1+t2}.

The output is a matrix of size (dim_target x (dim1*dim2)), representing the bilinear map
on the chosen bases. Column ordering is lexicographic with the first input varying
fastest: columns correspond to pairs (i,j) with i in 1:dim1 and j in 1:dim2.

Important:
- This function does not (and cannot) verify that `mul` is compatible with the
  filtration or differential. If it is not, an error may be thrown when projecting
  the product back to E_r^{a+c,b+d}.
"""
function product_matrix(
    ss::SpectralSequence{K},
    r::Int,
    ab1::Tuple{Int,Int},
    ab2::Tuple{Int,Int},
    mul::Function
) where {K}
    if r < 1
        error("product_matrix(ss,r,...): r must be >= 1")
    end

    a1, b1 = ab1
    a2, b2 = ab2
    tgt = (a1 + a2, b1 + b2)

    SQ1 = term(ss, r, ab1)
    SQ2 = term(ss, r, ab2)
    SQt = term(ss, r, tgt)

    m = SQ1.dimH
    n = SQ2.dimH
    k = SQt.dimH

    M = zeros(K, k, m * n)
    if m == 0 || n == 0 || k == 0
        return M
    end

    t1 = a1 + b1
    t2 = a2 + b2
    expected_len = SQt.ambient_dim

    col = 1
    for j in 1:n
        y = SQ2.Hrep[:, j]
        for i in 1:m
            x = SQ1.Hrep[:, i]
            z = mul(t1, x, t2, y)
            zv = Vector{K}(z)
            if length(zv) != expected_len
                error("product_matrix: mul returned length " * string(length(zv)) *
                      " but expected " * string(expected_len))
            end
            coords = subquotient_coordinates(SQt, zv)
            M[:, col] = coords
            col += 1
        end
    end
    return M
end

"""
    product_coords(ss, r, (a,b), x, (c,d), y, mul) -> Matrix{K}

Multiply two elements on the E_r page, given in coordinates on the chosen bases.

Inputs:
- `x` is a column vector of length dim(E_r^{a,b})
- `y` is a column vector of length dim(E_r^{c,d})

The output is a column vector (as a 1-column Matrix{K}) of length dim(E_r^{a+c,b+d}).
"""
function product_coords(
    ss::SpectralSequence{K},
    r::Int,
    ab1::Tuple{Int,Int},
    xcoords::Vector{K},
    ab2::Tuple{Int,Int},
    ycoords::Vector{K},
    mul::Function
) where {K}
    if r < 1
        error("product_coords(ss,r,...): r must be >= 1")
    end

    a1, b1 = ab1
    a2, b2 = ab2
    tgt = (a1 + a2, b1 + b2)

    SQ1 = term(ss, r, ab1)
    SQ2 = term(ss, r, ab2)
    SQt = term(ss, r, tgt)

    if length(xcoords) != SQ1.dimH
        error("product_coords: xcoords has wrong length")
    end
    if length(ycoords) != SQ2.dimH
        error("product_coords: ycoords has wrong length")
    end
    if SQt.dimH == 0
        return zeros(K, 0, 1)
    end

    xrep = SQ1.Hrep * reshape(xcoords, :, 1)
    yrep = SQ2.Hrep * reshape(ycoords, :, 1)

    t1 = a1 + b1
    t2 = a2 + b2
    z = mul(t1, vec(xrep), t2, vec(yrep))
    zv = Vector{K}(z)
    if length(zv) != SQt.ambient_dim
        error("product_coords: mul returned wrong length")
    end
    return subquotient_coordinates(SQt, zv)
end


# -----------------------------------------------------------------------------
# Filtration and edge maps
# -----------------------------------------------------------------------------

function filtration_dims(ss::SpectralSequence{K}, t::Int) where {K}
    if t < ss.Tot.tmin || t > ss.Tot.tmax
        return Dict{Int,Int}()
    end
    tidx = t - ss.Tot.tmin + 1
    out = Dict{Int,Int}()
    for p in ss.pmin:ss.pmax
        ip = p - ss.pmin + 1
        out[p] = ss.filt_img_dims[ip, tidx]
    end
    return out
end

function filtration_dims(ss::SpectralSequence{K}, p::Int, t::Int) where {K}
    if t < ss.Tot.tmin || t > ss.Tot.tmax
        return 0
    end
    tidx = t - ss.Tot.tmin + 1
    if p <= ss.pmin
        return ss.filt_img_dims[1, tidx]
    elseif p >= ss.pmax
        return 0
    else
        return ss.filt_img_dims[p - ss.pmin + 1, tidx]
    end
end

function filtration_basis(ss::SpectralSequence{K}, p::Int, t::Int) where {K}
    if t < ss.Tot.tmin || t > ss.Tot.tmax
        return zeros(K, 0, 0)
    end
    tidx = t - ss.Tot.tmin + 1
    use_col = _use_exact_columnwise_filtimg_basis(ss, tidx)
    if p <= ss.pmin
        return use_col && ss.filt_img.value === nothing ?
            _ss_filt_img_basis_col!(ss, tidx)[1] :
            _ss_filt_img_bases!(ss)[1, tidx]
    elseif p >= ss.pmax
        return zeros(K, ss.Htot_dims[tidx], 0)
    else
        ip = p - ss.pmin + 1
        return use_col && ss.filt_img.value === nothing ?
            _ss_filt_img_basis_col!(ss, tidx)[ip] :
            _ss_filt_img_bases!(ss)[ip, tidx]
    end
end

# -----------------------------------------------------------------------------
# E_infty edge maps and explicit splittings (vector-space case)
# -----------------------------------------------------------------------------

# Internal helper: a canonical "zero" SubquotientData with a prescribed ambient dimension.
function _ss_zero_subquotient(::Type{K}, ambient_dim::Int) where {K}
    Z0 = zeros(K, ambient_dim, 0)
    return SubquotientData{K}(ambient_dim, 0, 0, 0,
                              Z0, Z0, zeros(K, 0, 0), zeros(K, 0, 0),
                              zeros(K, 0, 0), Z0, 1:ambient_dim, Z0, _fullcolumn_factor_ref(), _fullcolumn_factor_ref())
end

"""
    filtration_subquotient(ss, p, t) -> SubquotientData{K}

Return the graded piece `gr^p H^t = F^p H^t / F^{p+1} H^t` as an explicit subquotient.

This is, by definition, the `E_infty` term on total degree `t` at filtration index `p`.
We return a `SubquotientData` whose ambient space is the chosen coordinate space
for `H^t(Tot)` (so `ambient_dim == dim H^t`).

Notes:
- When `t` is out of range, an empty (0-dim) object is returned.
- When the corresponding bidegree lies outside the double complex window, the
  graded piece is zero and we return a 0-dimensional quotient in ambient_dim = dim H^t.
"""
function filtration_subquotient(ss::SpectralSequence{K}, p::Int, t::Int) where {K}
    if t < ss.Tot.tmin || t > ss.Tot.tmax
        return _ss_zero_subquotient(K, 0)
    end
    tidx = t - ss.Tot.tmin + 1
    dimH = ss.Htot_dims[tidx]
    if dimH == 0
        return _ss_zero_subquotient(K, 0)
    end

    a, b = if ss.first == :vertical
        (p, t - p)
    else
        (t - p, p)
    end

    if a < ss.DC.amin || a > ss.DC.amax || b < ss.DC.bmin || b > ss.DC.bmax
        return _ss_zero_subquotient(K, dimH)
    end

    return term(ss, :inf, (a, b))
end

# Small helper for identity matrices over K.
function _ss_identity(::Type{K}, n::Int) where {K}
    M = zeros(K, n, n)
    for i in 1:n
        M[i, i] = one(K)
    end
    return M
end

# Build (and cache) an explicit splitting of H^t into a direct sum of its E_infty pieces.
function _ss_split_tot_cohomology!(ss::SpectralSequence{K}, t::Int) where {K}
    if haskey(ss.split_cache, t)
        return ss.split_cache[t]
    end

    if t < ss.Tot.tmin || t > ss.Tot.tmax
        sd = SSSplitData{K}(t, zeros(K, 0, 0), zeros(K, 0, 0),
                             Dict{Tuple{Int,Int},UnitRange{Int}}())
        ss.split_cache[t] = sd
        return sd
    end

    tidx = t - ss.Tot.tmin + 1
    dimH = ss.Htot_dims[tidx]
    if dimH == 0
        sd = SSSplitData{K}(t, zeros(K, 0, 0), zeros(K, 0, 0),
                             Dict{Tuple{Int,Int},UnitRange{Int}}())
        ss.split_cache[t] = sd
        return sd
    end

    # Collect the E_infty pieces along the diagonal a+b=t, together with their filtration index p.
    pieces = Vector{Tuple{Int,Tuple{Int,Int},Matrix{K}}}()
    for a in ss.DC.amin:ss.DC.amax
        b = t - a
        if b < ss.DC.bmin || b > ss.DC.bmax
            continue
        end
        p = _ss_p(ss.first, a, b)
        SQ = term(ss, :inf, (a, b))
        push!(pieces, (p, (a, b), SQ.Hrep))
    end
    sort!(pieces, by=x->x[1])

    # Concatenate the chosen bases of the graded pieces to form a basis B of H^t.
    blocks = Matrix{K}[]
    ranges = Dict{Tuple{Int,Int},UnitRange{Int}}()
    col = 0
    for (_, ab, Hrep) in pieces
        d = size(Hrep, 2)
        if d == 0
            continue
        end
        push!(blocks, Hrep)
        ranges[ab] = (col + 1):(col + d)
        col += d
    end

    if col != dimH
        error("_ss_split_tot_cohomology!: graded pieces do not sum to dim H^$(t)")
    end

    B = hcat(blocks...)
    # Invert B exactly via a full-column solve against the identity.
    field = field_from_eltype(K)
    Binv = FieldLinAlg.solve_fullcolumn(field, B, _ss_identity(K, dimH))

    sd = SSSplitData{K}(t, B, Binv, ranges)
    ss.split_cache[t] = sd
    return sd
end

"""
    split_total_cohomology(ss, t) -> NamedTuple

Return a noncanonical splitting of `H^t(Tot)` into the direct sum of its `E_infty`
pieces along the diagonal `a+b=t`.

The return value is a NamedTuple `(B, Binv, ranges)`:

- `B`    : dimH x dimH, columns form a basis of `H^t` adapted to the filtration.
           It is obtained by concatenating the chosen bases of each `E_infty^{a,b}`.
- `Binv` : inverse of `B`.
- `ranges[(a,b)]` : the coordinate range in the `B`-basis corresponding to the
                    summand `E_infty^{a,b}`.

Because we work over a field, the filtration always splits as vector spaces.
This function chooses one explicit splitting by linear algebra and caches it.

This is useful for "edge projections" and for writing elements of `H^t` as sums
of `E_infty` components.
"""
function split_total_cohomology(ss::SpectralSequence{K}, t::Int) where {K}
    sd = _ss_split_tot_cohomology!(ss, t)
    return (B=sd.B, Binv=sd.Binv, ranges=sd.ranges)
end

# ---------------------------------------------------------------------------
# Spectral sequence workflow helpers (term-level pages, filtrations, extensions)
# ---------------------------------------------------------------------------

"""
    E_r_terms(ss, r) -> SpectralTermsPage

Return the `E_r` page as explicit `SubquotientData` objects (not just dimensions).

This is a thin wrapper around `page_terms(ss, r)` with bidegree indexing via
`P[(a,b)]`.

See also: `E_r`, `page_terms`, `term`.
"""
function E_r_terms(ss::SpectralSequence{K}, r::Union{Int,Symbol}) where {K}
    return SpectralTermsPage(ss, r, page_terms(ss, r))
end

"""
    E2_terms(ss) -> SpectralTermsPage

Convenience wrapper for the E2 page as subquotient objects.
"""
E2_terms(ss::SpectralSequence{K}) where {K} = E_r_terms(ss, 2)

"""
    page_terms_dict(ss, r; nonzero_only=true)
        -> Dict{Tuple{Int,Int},SubquotientData{K}}

Return a dictionary keyed by bidegree `(a,b)` whose values are the term models
on the `E_r` page.

Speed note:
- If `nonzero_only=true` (default), iteration is restricted to `ss.support`.
  This avoids scanning the full rectangular bounding box.
"""
function page_terms_dict(ss::SpectralSequence{K}, r::Union{Int,Symbol}; nonzero_only::Bool=true) where {K}
    pd = _ss_page_data(ss, r)
    out = Dict{Tuple{Int,Int},SubquotientData{K}}()
    if nonzero_only
        if hasproperty(ss, :support)
            for (a, b) in ss.support
                sq = pd.spaces[a - ss.DC.amin + 1, b - ss.DC.bmin + 1]
                if sq.dimH != 0
                    out[(a,b)] = sq
                end
            end
        else
            for a in ss.DC.amin:ss.DC.amax
                for b in ss.DC.bmin:ss.DC.bmax
                    out[(a,b)] = pd.spaces[a - ss.DC.amin + 1, b - ss.DC.bmin + 1]
                end
            end
        end
    else
        for a in ss.DC.amin:ss.DC.amax
            for b in ss.DC.bmin:ss.DC.bmax
                out[(a,b)] = pd.spaces[a - ss.DC.amin + 1, b - ss.DC.bmin + 1]
            end
        end
    end
    return out
end

"""
    page_dims_dict(ss, r; nonzero_only=true) -> Dict{Tuple{Int,Int},Int}

Return a dictionary keyed by bidegree `(a,b)` whose values are the dimensions
on the `E_r` page.

This complements `page_terms_dict` and mirrors `page(ss,r)` but in a map form.
"""
function page_dims_dict(ss::SpectralSequence{K}, r::Union{Int,Symbol}; nonzero_only::Bool=true) where {K}
    dims = if r isa Int
        page(ss, r).dims
    elseif r == :inf || r == :infty
        ss.Einf_dims
    else
        error("page_dims_dict: unsupported symbol")
    end
    out = Dict{Tuple{Int,Int},Int}()
    if nonzero_only
        if hasproperty(ss, :support)
            for (a, b) in ss.support
                d = dims[a - ss.DC.amin + 1, b - ss.DC.bmin + 1]
                if d != 0
                    out[(a,b)] = d
                end
            end
        else
            for a in ss.DC.amin:ss.DC.amax
                for b in ss.DC.bmin:ss.DC.bmax
                    out[(a,b)] = dims[a - ss.DC.amin + 1, b - ss.DC.bmin + 1]
                end
            end
        end
    else
        for a in ss.DC.amin:ss.DC.amax
            for b in ss.DC.bmin:ss.DC.bmax
                out[(a,b)] = dims[a - ss.DC.amin + 1, b - ss.DC.bmin + 1]
            end
        end
    end
    return out
end

"""
    page_dict(ss, r; nonzero_only=true) -> Dict{Tuple{Int,Int},Int}

Alias for `page_dims_dict`; returns E_r dimensions keyed by bidegree `(a,b)`.
"""
page_dict(ss::SpectralSequence{K}, r::Union{Int,Symbol}; nonzero_only::Bool=true) where {K} =
    page_dims_dict(ss, r; nonzero_only=nonzero_only)

"""
    diagonal_criterion(ss; r=:inf) -> Bool
    diagonal_criterion(ss, t; r=:inf) -> Bool

Dimension-only diagonal check for convergence.

For a fixed total degree `t`, the criterion is:
    sum_{a+b=t} dim(E_r^{a,b}) == dim(H^t(Tot))

If `t` is omitted, checks all total degrees in the range of the total complex.

This is necessary but not sufficient for convergence; it is a very useful
workflow/diagnostic check (and a good regression test).
"""
function diagonal_criterion(ss::SpectralSequence{K}, t::Int; r::Union{Int,Symbol}=:inf) where {K}
    if t < ss.Tot.tmin || t > ss.Tot.tmax
        return true
    end
    pg = page(ss, r)
    s = 0
    for a in ss.DC.amin:ss.DC.amax
        b = t - a
        if b < ss.DC.bmin || b > ss.DC.bmax
            continue
        end
        s += pg[(a,b)]
    end
    hdim = ss.Htot_dims[t - ss.Tot.tmin + 1]
    return s == hdim
end

function diagonal_criterion(ss::SpectralSequence{K}; r::Union{Int,Symbol}=:inf) where {K}
    for t in ss.Tot.tmin:ss.Tot.tmax
        if !diagonal_criterion(ss, t; r=r)
            return false
        end
    end
    return true
end

"""
    FiltrationData

A packaged view of the induced filtration on `H^t(Tot)` coming from `E_infty`.

Fields:
- `t`    : total cohomology degree
- `pmin` : minimal filtration index in the spectral sequence
- `pmax` : maximal filtration index (one past the last "graded piece" index)
- `dims[p]`  : dim(F^p H^t)
- `bases[p]` : a basis matrix for F^p H^t embedded in H^t (columns in H^t)
- `graded[p]`: subquotient model for gr^p H^t = F^p/F^{p+1} as `SubquotientData`
"""
struct FiltrationData{K}
    t::Int
    pmin::Int
    pmax::Int
    dims::Dict{Int,Int}
    bases::Dict{Int,Matrix{K}}
    graded::Dict{Int,SubquotientData{K}}
end

"""
    filtration_data(ss, t) -> FiltrationData

Build explicit filtration data for the total cohomology group H^t(Tot):
- dimensions of all filtration steps
- explicit bases (as columns) for each filtration step
- explicit subquotient models for the graded pieces (E_infty terms)
"""
function filtration_data(ss::SpectralSequence{K}, t::Int) where {K}
    dims = Dict{Int,Int}()
    bases = Dict{Int,Matrix{K}}()
    graded = Dict{Int,SubquotientData{K}}()

    # Filtration steps include extremes:
    for p in ss.pmin:ss.pmax
        dims[p] = filtration_dims(ss, p, t)
        bases[p] = filtration_basis(ss, p, t)
    end

    # Graded pieces exist for p in [pmin, pmax-1]:
    for p in ss.pmin:(ss.pmax - 1)
        graded[p] = filtration_subquotient(ss, p, t)
    end

    return FiltrationData(t, ss.pmin, ss.pmax, dims, bases, graded)
end

"""
    filtration_data(ss) -> Dict{Int,FiltrationData}

Compute filtration data for every total degree `t` in the range of `ss`.
"""
function filtration_data(ss::SpectralSequence{K}) where {K}
    out = Dict{Int,FiltrationData}()
    for t in ss.Tot.tmin:ss.Tot.tmax
        out[t] = filtration_data(ss, t)
    end
    return out
end

"""
    collapse_data(ss; r=collapse_page(ss)) -> NamedTuple

Machine-friendly convergence summary.

Returns a named tuple with:
- `collapse_r`  : smallest r where page dimensions stabilize to E_infty dimensions
- `diagonal_ok` : diagonal criterion check on that page
- `filtrations` : explicit filtrations for all total degrees (as `FiltrationData`)
"""
function collapse_data(ss::SpectralSequence{K}; r::Int=collapse_page(ss)) where {K}
    return (
        collapse_r = r,
        diagonal_ok = diagonal_criterion(ss; r=r),
        filtrations = filtration_data(ss),
    )
end

"""
    ExtensionProblem

A packaged view of the "extension problem" on a fixed total degree t.

Over a field, extensions always split, but in many workflows one still wants:
- the graded pieces (E_infty terms on the diagonal a+b=t)
- an explicit splitting of H^t as a direct sum of those pieces

Fields:
- `t`      : total degree
- `pieces` : vector of named tuples (a, b, p, term)
- `B`      : splitting matrix (maps graded-basis coords -> H^t coords)
- `Binv`   : inverse of B
- `ranges` : column ranges in B corresponding to each diagonal piece
"""
struct ExtensionProblem{K}
    t::Int
    pieces::Vector{NamedTuple{(:a,:b,:p,:term),Tuple{Int,Int,Int,SubquotientData{K}}}}
    B::Matrix{K}
    Binv::Matrix{K}
    ranges::Dict{Tuple{Int,Int},UnitRange{Int}}
end

"""
    extension_problem(ss, t) -> ExtensionProblem{K}

Return the E_infty diagonal pieces with total degree t and an explicit splitting
of H^t(Tot) as a direct sum of those pieces.

This is the "helper to extract extension problems" for typical spectral sequence
workflows.

Implementation details:
- Diagonal pieces are taken from E_infty (r = :inf).
- Splitting is obtained from `split_total_cohomology(ss, t)` (cached internally).
"""
function extension_problem(ss::SpectralSequence{K}, t::Int) where {K}
    pd = _ss_page_data(ss, :inf)

    pieces = NamedTuple{(:a,:b,:p,:term),Tuple{Int,Int,Int,SubquotientData{K}}}[]
    if hasproperty(ss, :support)
        for (a, b) in ss.support
            if a + b != t
                continue
            end
            sq = pd.spaces[a - ss.DC.amin + 1, b - ss.DC.bmin + 1]
            if sq.dimH == 0
                continue
            end
            p = (ss.first == :vertical) ? a : b
            push!(pieces, (a=a, b=b, p=p, term=sq))
        end
    else
        for a in ss.DC.amin:ss.DC.amax
            for b in ss.DC.bmin:ss.DC.bmax
                a + b == t || continue
                sq = pd.spaces[a - ss.DC.amin + 1, b - ss.DC.bmin + 1]
                sq.dimH == 0 && continue
                p = (ss.first == :vertical) ? a : b
                push!(pieces, (a=a, b=b, p=p, term=sq))
            end
        end
    end

    spl = split_total_cohomology(ss, t)
    return ExtensionProblem(t, pieces, spl.B, spl.Binv, spl.ranges)
end

"""
    extension_problem(ss) -> Dict{Int,ExtensionProblem{K}}

Compute extension problem packages for all total degrees t.
"""
function extension_problem(ss::SpectralSequence{K}) where {K}
    out = Dict{Int,ExtensionProblem{K}}()
    for t in ss.Tot.tmin:ss.Tot.tmax
        out[t] = extension_problem(ss, t)
    end
    return out
end


"""
    edge_inclusion(ss, (a,b)) -> Matrix{K}

Return the inclusion map

    E_infty^{a,b} -> H^{a+b}(Tot)

expressed in the chosen basis of `H^{a+b}(Tot)`.

The output is a matrix of size (dimH x dimEinf), whose columns are coordinate
vectors in `H^{a+b}(Tot)` representing a basis of the graded piece.
"""
edge_inclusion(ss::SpectralSequence{K}, ab::Tuple{Int,Int}) where {K} = term(ss, :inf, ab).Hrep

"""
    edge_projection(ss, (a,b)) -> Matrix{K}

Return a (noncanonical) projection

    H^{a+b}(Tot) -> E_infty^{a,b}

expressed in the chosen basis of `H^{a+b}(Tot)` and the basis of `E_infty^{a,b}`
used by `edge_inclusion`.

This projection is defined by the explicit splitting returned by
`split_total_cohomology(ss, a+b)`.
"""
function edge_projection(ss::SpectralSequence{K}, ab::Tuple{Int,Int}) where {K}
    a, b = ab
    t = a + b
    sd = _ss_split_tot_cohomology!(ss, t)

    dimH = size(sd.Binv, 1)
    if dimH == 0
        return zeros(K, 0, 0)
    end

    if !haskey(sd.ranges, ab)
        return zeros(K, 0, dimH)
    end

    rng = sd.ranges[ab]
    return sd.Binv[rng, :]
end


function collapse_page(ss::SpectralSequence{K}) where {K}
    for r in 1:ss.rmax_possible
        if page(ss, r).dims == ss.Einf_dims
            return r
        end
    end
    return ss.rmax_possible
end

function convergence_report(ss::SpectralSequence{K}; verbose::Bool=false) where {K}
    io = IOBuffer()
    println(io, "SpectralSequence report")
    println(io, "  first = ", ss.first)
    println(io, "  a in [", ss.DC.amin, ", ", ss.DC.amax, "]")
    println(io, "  b in [", ss.DC.bmin, ", ", ss.DC.bmax, "]")
    println(io, "  rmax_possible = ", ss.rmax_possible)
    println(io, "  collapse_page (dims) = ", collapse_page(ss))

    # Diagonal bookkeeping: for each total degree t,
    #   sum_{a+b=t} dim(E_infty^{a,b})  should equal  dim(H^t(Tot)).
    ok = true
    for t in ss.Tot.tmin:ss.Tot.tmax
        s = 0
        for a in ss.DC.amin:ss.DC.amax
            b = t - a
            if b < ss.DC.bmin || b > ss.DC.bmax
                continue
            end
            s += ss.Einf_dims[a - ss.DC.amin + 1, b - ss.DC.bmin + 1]
        end
        dimH = ss.Htot_dims[t - ss.Tot.tmin + 1]
        if s != dimH
            ok = false
        end
    end
    println(io, "  diagonal check (sum dim E_inf on a+b=t equals dim H^t): ", ok ? "OK" : "FAILED")

    if verbose
        println(io, "  per-total-degree summary:")
        for t in ss.Tot.tmin:ss.Tot.tmax
            tidx = t - ss.Tot.tmin + 1
            dimH = ss.Htot_dims[tidx]
            println(io, "    t = ", t, "  dim H^t = ", dimH)
            if dimH == 0
                continue
            end

            pieces = String[]
            for a in ss.DC.amin:ss.DC.amax
                b = t - a
                if b < ss.DC.bmin || b > ss.DC.bmax
                    continue
                end
                d = ss.Einf_dims[a - ss.DC.amin + 1, b - ss.DC.bmin + 1]
                if d > 0
                    push!(pieces, "(" * string(a) * "," * string(b) * "):" * string(d))
                end
            end
            println(io, "      E_inf pieces: ", isempty(pieces) ? "(none)" : join(pieces, ", "))

            fd = filtration_dims(ss, t)
            fparts = String[]
            for p in ss.pmin:ss.pmax
                if haskey(fd, p)
                    push!(fparts, "F^" * string(p) * "=" * string(fd[p]))
                end
            end
            println(io, "      filtration dims: ", isempty(fparts) ? "(none)" : join(fparts, ", "))
        end
    end

    return String(take!(io))
end

# -----------------------------------------------------------------------------
# Pretty-printing (ASCII)
# -----------------------------------------------------------------------------

function Base.show(io::IO, ss::SpectralSequence{K}) where {K}
    print(io,
          "SpectralSequence(first=",
          ss.first,
          ", a=",
          ss.DC.amin,
          ":",
          ss.DC.amax,
          ", b=",
          ss.DC.bmin,
          ":",
          ss.DC.bmax,
          ")")
end

function Base.show(io::IO, ::MIME"text/plain", ss::SpectralSequence{K}) where {K}
    println(io, "SpectralSequence")
    println(io, "  first = ", ss.first)
    println(io, "  a in [", ss.DC.amin, ", ", ss.DC.amax, "]")
    println(io, "  b in [", ss.DC.bmin, ", ", ss.DC.bmax, "]")
    println(io, "  rmax_possible = ", ss.rmax_possible)
    println(io, "  collapse_page (dims) = ", collapse_page(ss))
    println(io, "  total cohomology dims = ", ss.Htot_dims)
end

function Base.show(io::IO, P::SpectralPage{K}) where {K}
    rstr = (P.r == :inf ? "inf" : string(P.r))
    print(io, "SpectralPage(E_", rstr, " dimensions)")
end

function Base.show(io::IO, ::MIME"text/plain", P::SpectralPage{K}) where {K}
    ss = P.ss
    rstr = (P.r == :inf ? "inf" : string(P.r))
    println(io, "E_", rstr, " page (dimensions)")
    println(io, "  first = ", ss.first)

    arange = ss.DC.amin:ss.DC.amax
    brange = ss.DC.bmin:ss.DC.bmax

    # Choose a uniform cell width large enough for all dims on the page.
    w = 1
    for x in P.dims
        w = max(w, length(string(x)))
    end
    w = max(w, 2)

    # Header
    print(io, rpad("a\\b", 6))
    for b in brange
        print(io, " ", lpad(string(b), w))
    end
    println(io)

    # Rows
    for (i, a) in enumerate(arange)
        print(io, lpad(string(a), 5))
        for (j, _) in enumerate(brange)
            print(io, " ", lpad(string(P.dims[i, j]), w))
        end
        println(io)
    end
end

function Base.show(io::IO, SQ::SubquotientData{K}) where {K}
    print(io, "SubquotientData(dimH=", SQ.dimH, ")")
end

function Base.show(io::IO, ::MIME"text/plain", SQ::SubquotientData{K}) where {K}
    println(io, "SubquotientData (explicit Z/B model)")
    println(io, "  ambient_dim = ", SQ.ambient_dim)
    println(io, "  dimZ = ", SQ.dimZ)
    println(io, "  dimB = ", SQ.dimB)
    println(io, "  dimH = ", SQ.dimH)
end


total_cohomology_dims(ss::SpectralSequence{K}) where {K} = ss.Htot_dims



end
