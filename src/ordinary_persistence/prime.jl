# Exact prime-field validation and sparse ordinary column reduction. F2 keeps
# its binary kernels; this implementation shares their public result contract.

function _double_boundary_zero_prime(A, B, ::Type{K}) where {K}
    accumulator = zeros(K, size(A, 1))
    for col in axes(B, 2)
        nonzero = 0
        @inbounds for j in nzrange(B, col)
            b = K(B.nzval[j])
            iszero(b) && continue
            for i in nzrange(A, B.rowval[j])
                row = A.rowval[i]
                old = accumulator[row]
                value = old + K(A.nzval[i]) * b
                nonzero += Int(!iszero(value)) - Int(!iszero(old))
                accumulator[row] = value
            end
        end
        nonzero == 0 || return false
        # A successful column leaves every accumulator entry zero.
    end
    return true
end

# out = a + scale*b; all supports are distinct and sorted by filtration rank.
# Keep the output scratch separate so completed pivot columns stay immutable.
function _prime_add_scaled!(out::Vector{Pair{Int,K}}, a, b, scale::K, rank) where {K}
    empty!(out)
    sizehint!(out, length(a) + length(b); shrink=false)
    i, j = 1, 1
    @inbounds while i <= length(a) && j <= length(b)
        ai, bj = first(a[i]), first(b[j])
        if ai == bj
            value = last(a[i]) + scale * last(b[j])
            iszero(value) || push!(out, ai => value)
            i += 1; j += 1
        elseif rank[ai] < rank[bj]
            push!(out, a[i]); i += 1
        else
            push!(out, bj => scale * last(b[j])); j += 1
        end
    end
    @inbounds while i <= length(a)
        push!(out, a[i]); i += 1
    end
    @inbounds while j <= length(b)
        push!(out, first(b[j]) => scale * last(b[j])); j += 1
    end
    return out
end

function _prime_scale!(column, scale)
    @inbounds for i in eachindex(column)
        column[i] = first(column[i]) => scale * last(column[i])
    end
    return column
end

function _representative_chain(G::GradedComplex{N,T}, offsets, dim,
                               column::Vector{Pair{Int,K}}) where {N,T,K}
    # Public indices follow source-cell order, independently of filtration order.
    ordered = sort(column; by=first)
    indices = first.(ordered)
    return _PersistenceChain{T}(indices .- (offsets[dim + 1] - 1),
        getfield(G, :cell_ids)[indices], T[G.grades[i][1] for i in indices],
        Int[last(entry).val for entry in ordered])
end

function _reduce_prime!(intervals, essential, finite_reps, essential_reps,
                        G::GradedComplex{N,T}, dims, offsets, values, perm, rank,
                        ::Type{K}, ::Val{Keep}) where {N,T,K,Keep}
    total = length(values)
    reduced = Vector{Vector{Pair{Int,K}}}(undef, total)
    has_reduced = falses(total)
    positive, alive = falses(total), falses(total)
    changes = Keep ? Vector{Vector{Pair{Int,K}}}(undef, total) : nothing
    col, scratch = Pair{Int,K}[], Pair{Int,K}[]
    change = Keep ? Pair{Int,K}[] : nothing
    change_scratch = Keep ? Pair{Int,K}[] : nothing
    for cell in perm
        empty!(col)
        dim = dims[cell]
        if dim > 0
            boundary = G.boundaries[dim]
            local_col = cell - offsets[dim + 1] + 1
            @inbounds for ptr in nzrange(boundary, local_col)
                coefficient = K(boundary.nzval[ptr])
                iszero(coefficient) && continue
                row = offsets[dim] + boundary.rowval[ptr] - 1
                push!(col, row => coefficient)
            end
            sort!(col; by=entry -> rank[first(entry)])
        end
        if Keep
            empty!(change)
            push!(change, cell => one(K))
        end
        while !isempty(col)
            pivot = first(last(col))
            has_reduced[pivot] || break
            # Stored pivot columns have unit leading coefficient.
            scale = -last(last(col))
            _prime_add_scaled!(scratch, col, reduced[pivot], scale, rank)
            col, scratch = scratch, col
            if Keep
                _prime_add_scaled!(change_scratch, change, changes[pivot], scale, rank)
                change, change_scratch = change_scratch, change
            end
        end
        if isempty(col)
            positive[cell] = true
            alive[cell] = true
            Keep && (changes[cell] = copy(change))
        else
            pivot = first(last(col))
            positive[pivot] || error("ordinary persistence internal inconsistency: pivot did not birth a class.")
            scale = inv(last(last(col)))
            _prime_scale!(col, scale)
            has_reduced[pivot] = true
            reduced[pivot] = copy(col)
            alive[pivot] = false
            if Keep
                # Maintain boundary(change) == col through normalization too.
                _prime_scale!(change, scale)
                changes[pivot] = copy(change)
                if values[pivot] != values[cell]
                    push!(finite_reps[dim], _PersistenceRepresentative{T}(
                        values[pivot], values[cell],
                        _representative_chain(G, offsets, dim - 1, col),
                        _representative_chain(G, offsets, dim, change)))
                end
            end
            values[pivot] == values[cell] || push!(intervals[dim], (values[pivot], values[cell]))
        end
    end
    for cell in 1:total
        alive[cell] || continue
        dim = dims[cell]
        push!(essential[dim + 1], values[cell])
        Keep && push!(essential_reps[dim + 1], _PersistenceRepresentative{T}(
            values[cell], nothing, _representative_chain(G, offsets, dim, changes[cell]), nothing))
    end
    return nothing
end
