# -----------------------------------------------------------------------------
# field_linalg/public_api.jl
#
# Scope:
#   Public FieldLinAlg entrypoints plus restricted helpers and module
#   initialization for threshold loading/autotune.
# -----------------------------------------------------------------------------

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
        be = _choose_solve_backend(field, B; backend=backend, factor=factor)
        if be == :nemo
            return _solve_fullcolumn_nemoQQ(B, Y; check_rhs=check_rhs, cache=cache, factor=factor)
        end
        if be == :modular
            if !(B isa SparseMatrixCSC)
                X = _solve_fullcolumn_modularQQ(B, Y; check_rhs=check_rhs)
                X === nothing || return X
            end
        end
        jfactor = factor isa FullColumnFactor{QQ} ? factor : nothing
        return _solve_fullcolumnQQ(B, Y; check_rhs=check_rhs, cache=cache, factor=jfactor)
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

@inline function _active_words_nonempty(words::AbstractVector{UInt64})::Bool
    @inbounds for w in words
        w == 0 || return true
    end
    return false
end

function _materialize_restricted_dense_from_words(A::AbstractMatrix{T},
                                                  row_words::AbstractVector{UInt64},
                                                  col_words::AbstractVector{UInt64},
                                                  nr::Int,
                                                  nc::Int,
                                                  nrows::Int,
                                                  ncols::Int;
                                                  check::Bool=false) where {T}
    if nr == 0 || nc == 0
        return Matrix{T}(undef, nr, nc)
    end
    B = Matrix{T}(undef, nr, nc)
    jloc = 0
    @inbounds for wj in eachindex(col_words)
        w = col_words[wj]
        while w != 0
            tz = trailing_zeros(w)
            col = ((wj - 1) << 6) + tz + 1
            if col <= ncols
                check && @assert 1 <= col <= size(A, 2)
                jloc += 1
                iloc = 0
                for wi in eachindex(row_words)
                    rw = row_words[wi]
                    while rw != 0
                        rtz = trailing_zeros(rw)
                        row = ((wi - 1) << 6) + rtz + 1
                        if row <= nrows
                            check && @assert 1 <= row <= size(A, 1)
                            iloc += 1
                            B[iloc, jloc] = A[row, col]
                        end
                        rw &= rw - 1
                    end
                end
            end
            w &= w - 1
        end
    end
    return B
end

function _decode_words_to_indices(words::AbstractVector{UInt64}, nmax::Int)
    out = Int[]
    sizehint!(out, nmax)
    @inbounds for wd in eachindex(words)
        w = words[wd]
        while w != 0
            tz = trailing_zeros(w)
            idx = ((wd - 1) << 6) + tz + 1
            idx <= nmax && push!(out, idx)
            w &= w - 1
        end
    end
    return out
end

function rank_restricted_words(field::AbstractCoeffField,
                               A::AbstractMatrix,
                               row_words::AbstractVector{UInt64},
                               col_words::AbstractVector{UInt64},
                               nr::Int,
                               nc::Int;
                               nrows::Int=size(A, 1),
                               ncols::Int=size(A, 2),
                               backend::Symbol=:auto,
                               kwargs...)
    if nr == 0 || nc == 0 || !_active_words_nonempty(row_words) || !_active_words_nonempty(col_words)
        return 0
    end
    if A isa SparseMatrixCSC
        rows = _decode_words_to_indices(row_words, nrows)
        cols = _decode_words_to_indices(col_words, ncols)
        return rank_restricted(field, A, rows, cols; backend=backend, kwargs...)
    end
    check = haskey(kwargs, :check) ? kwargs[:check] : false
    B = _materialize_restricted_dense_from_words(A, row_words, col_words, nr, nc, nrows, ncols; check=check)
    dense_kwargs = _drop_check_kw(kwargs)
    return rank(field, B; backend=backend, dense_kwargs...)
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
