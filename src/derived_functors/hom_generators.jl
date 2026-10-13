# Generator-coordinate Hom on finite posets. This applies Hom(-,N) to a
# presentation of M, as in Jendrysiak, arXiv:2604.10381, Section 4 (Algorithm 7).
# The target stalks/maps are already given. Local elimination constructs the
# source presentation implicitly, retaining redundant relations at merges.

struct _HomGeneratorPlan{K}
    order::Vector{Int}
    selected::Vector{Vector{Tuple{Int,Int}}} # incoming slot/column, or (0, new column)
    inverses::Vector{Matrix{K}}
    offsets::Vector{Int}
end

# A small kernel is insufficient if constructing its coordinates costs more
# than streaming naturality. Require substantial compression, and reject tiny
# or generator-heavy requests before doing any elimination. This is only a
# structural bound: the actual generator count is checked after preparation.
function _hom_generator_candidate(M::PModule, N::PModule, nvars::Int)
    nvars >= max(128, 4length(M.dims)) || return false
    lower = 0
    for v in eachindex(M.dims)
        d = M.dims[v]
        bound = 0
        if N.dims[v] != 0
            for A in M.edge_maps.maps_from_pred[v]
                iszero(A) || (bound += min(d, size(A, 2)))
                bound >= d && break
            end
            lower += max(0, d-bound) * N.dims[v]
            4lower > nvars && return false
        end
    end
    return true
end

# Coordinate inclusions occur frequently in encodings and projective modules.
# Stop once these columns span the stalk; later incoming maps cannot change
# the ordered pivot basis. General columns retain the field-aware RREF route.
function _hom_pivot_inverse(field, A::Matrix{K}, incoming::Int) where {K}
    d = size(A, 1)
    seen = falses(d)
    pivots, rows = Int[], Int[]
    for j in axes(A, 2)
        length(pivots) == d && break
        row = 0
        for i in 1:d
            iszero(A[i,j]) && continue
            if row != 0
                R, piv = FieldLinAlg.rref(field, A)
                return Matrix(R[:, incoming+1:incoming+d]), piv
            end
            row = i
        end
        if row != 0 && !seen[row]
            seen[row] = true
            push!(pivots, j); push!(rows, row)
        end
    end
    inverse = zeros(K, d, d)
    for (j, r) in enumerate(rows)
        c = A[r,pivots[j]]
        inverse[j,r] = isone(c) ? one(K) : inv(c)
    end
    return inverse, pivots
end

function _hom_generator_plan(M::PModule{K}, N::PModule{K}) where {K}
    n = length(M.dims)
    store = M.edge_maps
    selected = Vector{Vector{Tuple{Int,Int}}}(undef, n)
    inverses = Vector{Matrix{K}}(undef, n)
    offsets = zeros(Int, n + 1)
    for v in 1:n
        d = M.dims[v]
        cols = sum((M.dims[u] for u in store.preds[v]); init=0)
        A = zeros(K, d, cols + d)
        labels = Tuple{Int,Int}[]
        pos = 0
        for (e, u) in enumerate(store.preds[v])
            du = M.dims[u]
            A[:, pos+1:pos+du] = store.maps_from_pred[v][e]
            append!(labels, ((e, j) for j in 1:du))
            pos += du
        end
        for j in 1:d
            A[j, cols+j] = one(K)
            push!(labels, (0, j))
        end
        inverse, pivots = _hom_pivot_inverse(M.field, A, cols)
        selected[v] = [labels[i] for i in pivots]
        inverses[v] = inverse
        offsets[v+1] = offsets[v] + count(p -> p[1] == 0, selected[v]) * N.dims[v]
    end
    # Vertex labels need not be a linear extension of the poset.
    indegree = length.(store.preds)
    order = findall(iszero, indegree)
    cursor = 1
    while cursor <= length(order)
        for v in store.succs[order[cursor]]
            indegree[v] -= 1
            indegree[v] == 0 && push!(order, v)
        end
        cursor += 1
    end
    length(order) == n || error("Hom: cover graph is not acyclic")
    return _HomGeneratorPlan{K}(order, selected, inverses, offsets)
end

function _hom_generator_basis(M::PModule{K}, N::PModule{K}, offs,
                              plan::_HomGeneratorPlan{K}) where {K}
    nf = plan.offsets[end]
    nf == 0 && return zeros(K, offs[end], 0)
    # Each column records the coefficients of one entry of F_v as a linear
    # function of generator images. Transposed storage makes equations CSC columns.
    lifts = Vector{SparseMatrixCSC{K,Int}}(undef, length(M.dims))
    R = K === QQ ? FieldLinAlg._SparseQQEchelon(nf) : _SparseRREF{K}(nf)
    row = SparseRow{K}()
    for v in plan.order
        dv, nv = M.dims[v], N.dims[v]
        rows, cols, vals = Int[], Int[], K[]
        free = plan.offsets[v]
        for (j, (e, k)) in enumerate(plan.selected[v])
            if e == 0
                for i in 1:nv
                    push!(rows, free+i); push!(cols, (j-1)*nv+i); push!(vals, one(K))
                end
                free += nv
            else
                u = M.edge_maps.preds[v][e]
                nu = N.dims[u]
                E = lifts[u]
                B = N.edge_maps.maps_from_pred[v][e]
                for b in 1:nu, i in 1:nv
                    c = B[i,b]
                    iszero(c) && continue
                    for t in nzrange(E, (k-1)*nu+b)
                        push!(rows, E.rowval[t]); push!(cols, (j-1)*nv+i)
                        push!(vals, isone(c) ? E.nzval[t] : c * E.nzval[t])
                    end
                end
            end
        end
        identityN = sparse(I, nv, nv)
        E = sparse(rows, cols, vals, nf, dv*nv)
        if plan.inverses[v] != I
            E = E * kron(sparse(plan.inverses[v]), identityN)
        end
        dropzeros!(E)
        lifts[v] = E
        for (e, u) in enumerate(M.edge_maps.preds[v])
            du = M.dims[u]
            (du == 0 || nv == 0) && continue
            # Selected incoming columns define the lift, so their naturality
            # equations hold identically. Only the remaining columns are relations.
            remaining = trues(du)
            for (slot, j) in plan.selected[v]
                slot == e && (remaining[j] = false)
            end
            js = findall(remaining)
            isempty(js) && continue
            A = M.edge_maps.maps_from_pred[v][e]
            B = N.edge_maps.maps_from_pred[v][e]
            nu = N.dims[u]
            entries = [(j-1)*nu+i for j in js for i in 1:nu]
            equations = lifts[u][:,entries] * kron(sparse(I, length(js), length(js)), sparse(transpose(B))) -
                        E * kron(sparse(A[:,js]), identityN)
            dropzeros!(equations)
            for j in axes(equations, 2)
                empty!(row.idx); empty!(row.val)
                for t in nzrange(equations, j)
                    push!(row.idx, equations.rowval[t]); push!(row.val, equations.nzval[t])
                end
                isempty(row) || FieldLinAlg._sparse_kernel_push!(R, row)
                length(R.pivot_cols) == nf && return zeros(K, offs[end], 0)
            end
        end
    end
    kernel = sparse(_nullspace_from_pivots(R, nf))
    result = zeros(K, offs[end], size(kernel, 2))
    for v in eachindex(M.dims)
        result[offs[v]+1:offs[v+1], :] = transpose(lifts[v]) * kernel
    end
    return result
end

# On a rooted forest with injective source edges, the source is projective:
# no relation equations remain. Transport source sections and target maps
# separately, then form their outer products once at each vertex. This avoids
# repeatedly transporting a tensor of Hom coefficients in rational arithmetic.
function _hom_projective_forest(M::PModule{QQ}, plan)
    maximum(M.dims; init=0) >= 3 || return false
    for v in eachindex(M.dims)
        predecessors = M.edge_maps.preds[v]
        length(predecessors) <= 1 || return false
        isempty(predecessors) && continue
        count(p -> p[1] != 0, plan.selected[v]) == M.dims[only(predecessors)] || return false
    end
    return true
end
_hom_projective_forest(M::PModule, plan) = false

function _hom_forest_basis(M::PModule{K}, N::PModule{K}, offs,
                           plan::_HomGeneratorPlan{K}) where {K}
    nf = plan.offsets[end]
    nf == 0 && return zeros(K, offs[end], 0)
    n = length(M.dims)
    births, qoffsets = Int[], Int[]
    generator_starts = zeros(Int, n)
    for v in 1:n
        generator_starts[v] = length(births)
        for (e, _) in plan.selected[v]
            e == 0 || continue
            local_index = length(births) - generator_starts[v]
            push!(qoffsets, plan.offsets[v] + local_index * N.dims[v])
            push!(births, v)
        end
    end
    ng = length(births)
    # sections[v] expresses stalk coordinates in the free generators. The
    # target transport is shared by all generators born at the same vertex.
    sections = Vector{SparseMatrixCSC{K,Int}}(undef, n)
    transports = Vector{Dict{Int,SparseMatrixCSC{K,Int}}}(undef, n)
    result = zeros(K, offs[end], nf)
    for v in plan.order
        dv, nv = M.dims[v], N.dims[v]
        rows, cols, vals = Int[], Int[], K[]
        maps = Dict{Int,SparseMatrixCSC{K,Int}}()
        newgen = generator_starts[v]
        for (j, (e, k)) in enumerate(plan.selected[v])
            if e == 0
                newgen += 1
                push!(rows, newgen); push!(cols, j); push!(vals, one(K))
                if !haskey(maps, v)
                    maps[v] = sparse(one(K) * I, nv, nv)
                end
            else
                u = M.edge_maps.preds[v][e]
                S = sections[u]
                for t in nzrange(S, k)
                    g = S.rowval[t]
                    root = births[g]
                    push!(rows, g); push!(cols, j); push!(vals, S.nzval[t])
                    if !haskey(maps, root)
                        maps[root] = sparse(N.edge_maps.maps_from_pred[v][e]) * transports[u][root]
                    end
                end
            end
        end
        S = sparse(rows, cols, vals, ng, dv)
        plan.inverses[v] == I || (S = S * sparse(plan.inverses[v]))
        dropzeros!(S)
        sections[v], transports[v] = S, maps
        # Each outer product is one generator-image contribution to F_v.
        # Source projectivity makes these independent without a kernel solve.
        for j in 1:dv, t in nzrange(S, j)
            g, c = S.rowval[t], S.nzval[t]
            T = maps[births[g]]
            for b in axes(T, 2), a in nzrange(T, b)
                z = T.nzval[a]
                result[offs[v] + (j-1)*nv + T.rowval[a], qoffsets[g]+b] =
                    isone(c) ? z : isone(z) ? c : c*z
            end
        end
    end
    return result
end
