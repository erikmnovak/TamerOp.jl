"""
Internal certified reductions of graded flag complexes.

An edge is removed only when a fixed vertex dominates its link at every grade
and the same domination holds in the requested dimension-truncated complex.
The latter condition preserves top-dimensional homology as well as the lower
degrees. All grades supplied here are already oriented coordinatewise upward.
"""
module SimplicialReduction

# Packed undirected keys avoid tuple allocation in the neighbor and clique scans.
# UInt128 accommodates two positive Int indices without an artificial vertex cap.
@inline function _edge_key(u::Int, v::Int)
    a, b = minmax(u, v)
    return (UInt128(a) << 64) | UInt128(b)
end

@inline function _grade_leq(a::NTuple{N,T}, b::NTuple{N,T}) where {N,T}
    @inbounds for k in 1:N
        a[k] <= b[k] || return false
    end
    return true
end

@inline function _grade_leq_join(a::NTuple{N,T}, b::NTuple{N,T},
                                c::NTuple{N,T}, d::NTuple{N,T}) where {N,T}
    @inbounds for k in 1:N
        a[k] <= max(b[k], c[k], d[k]) || return false
    end
    return true
end

# Exact clique decision on the current link, without per-recursion allocations.
# The caller supplies a reusable selected-vertex buffer of length at least target.
function _link_has_clique(vertices::Vector{Int}, excluded::Int, target::Int,
                          edge_index::Dict{UInt128,Int}, active::BitVector,
                          selected::Vector{Int}, first::Int=1, depth::Int=0)
    @inbounds for j in first:length(vertices)
        # This bound deliberately counts the excluded vertex, if still ahead;
        # it can only weaken pruning, never reject an existing clique.
        length(vertices) - j + 1 + depth >= target || return false
        v = vertices[j]
        v == excluded && continue
        adjacent = true
        for k in 1:depth
            idx = get(edge_index, _edge_key(v, selected[k]), 0)
            if idx == 0 || !active[idx]
                adjacent = false
                break
            end
        end
        adjacent || continue
        depth + 1 == target && return true
        selected[depth + 1] = v
        _link_has_clique(vertices, excluded, target, edge_index, active,
                         selected, j + 1, depth + 1) && return true
    end
    return false
end

"""
    _collapse_dominated_edges(edges, grades, n, max_dim) -> Vector{Int}

Return the original indices of retained undirected edges, in original order.
Grades must have the same positive arity, contain no NaN, and already use the
coordinatewise increasing convention. The caller retains all vertices and their
original grades, and supplies edge grades compatible with vertex births. Higher
simplex grades are the coordinatewise joins of their edge grades.

Each deletion is a strong filtration-domination: for edge `uv` and apex `w`,
`g(uw), g(vw) <= g(uv)`; for every other current common neighbor `x`, the edge
`wx` exists and `g(wx) <= join(g(uv), g(ux), g(vx))`. See Alonso, Kerber and
Pritam, *Filtration-Domination in Bifiltered Graphs*, Section 4 and Theorem 3.1,
<https://arxiv.org/abs/2211.05574>. The same comparisons and inclusion argument
apply in any number of parameters.

Dimension truncation needs a further certificate. In a `d`-skeleton, a clique
of `d-1` common neighbors excluding the apex would give a maximal `d`-simplex
containing `uv` which cannot be extended by `w`. We therefore reject such a
deletion. Otherwise every simplex containing `uv` but not `w` pairs with its
extension by `w`, within dimension `d` and at the same grade. Removing these
pairs in decreasing dimension gives elementary filtered collapses. Consequently
the inclusion of the reduced complex induces natural isomorphisms in EVERY
homology degree, over any field, including the top degree `d`.

The guard checks the final current graph, hence also every filtration grade.
Witnesses and cofaces always use current retained edges. A single deterministic
pass is made in decreasing lexicographic grade order; maximal reduction is not
claimed. For `max_dim <= 1` no edges are removed, since graph cycles must remain.
"""
function _collapse_dominated_edges(edges::Vector{NTuple{2,Int}},
                                    grades::Vector{<:NTuple{N,T}},
                                    n::Int, max_dim::Int) where {N,T<:Real}
    n >= 0 || throw(ArgumentError("edge collapse requires n >= 0."))
    max_dim >= 0 || throw(ArgumentError("edge collapse requires max_dim >= 0."))
    N > 0 || throw(ArgumentError("edge collapse requires a positive grade arity."))
    length(edges) == length(grades) ||
        throw(ArgumentError("edge collapse requires one grade per edge."))

    m = length(edges)
    edge_index = Dict{UInt128,Int}()
    sizehint!(edge_index, m)
    counts = zeros(Int, n)
    @inbounds for idx in eachindex(edges)
        u, v = edges[idx]
        1 <= u <= n && 1 <= v <= n && u != v ||
            throw(ArgumentError("edge collapse requires distinct endpoints in 1:n; got $(edges[idx])."))
        key = _edge_key(u, v)
        haskey(edge_index, key) &&
            throw(ArgumentError("edge collapse requires unique undirected edges; repeated $(edges[idx])."))
        edge_index[key] = idx
        for value in grades[idx]
            isnan(value) && throw(ArgumentError("edge collapse grades must not contain NaN."))
        end
        counts[u] += 1
        counts[v] += 1
    end
    (max_dim <= 1 || m == 0) && return collect(eachindex(edges))

    # Compressed adjacency stores original edge indices; deletions only flip the
    # active mask. This keeps references stable and makes witness reuse explicit.
    max_degree = isempty(counts) ? 0 : maximum(counts)
    offsets = Vector{Int}(undef, n + 1)
    offsets[1] = 1
    @inbounds for v in 1:n
        offsets[v + 1] = offsets[v] + counts[v]
        counts[v] = offsets[v]
    end
    adjacency = Vector{Int}(undef, 2m)
    @inbounds for idx in eachindex(edges)
        u, v = edges[idx]
        adjacency[counts[u]] = idx
        adjacency[counts[v]] = idx
        counts[u] += 1
        counts[v] += 1
    end
    active = trues(m)
    neighbors = Int[]
    incident_u = Int[]
    incident_v = Int[]
    sizehint!(neighbors, max_degree)
    sizehint!(incident_u, max_degree)
    sizehint!(incident_v, max_degree)
    selected = Vector{Int}(undef, min(max_dim - 1, n))
    order = sortperm(eachindex(edges);
                     by=i -> (grades[i], min(edges[i]...), max(edges[i]...)),
                     rev=true)

    @inbounds for idx in order
        u, v = edges[idx]
        if offsets[u + 1] - offsets[u] > offsets[v + 1] - offsets[v]
            u, v = v, u
        end
        empty!(neighbors)
        empty!(incident_u)
        empty!(incident_v)
        for p in offsets[u]:(offsets[u + 1] - 1)
            iu = adjacency[p]
            active[iu] || continue
            a, b = edges[iu]
            x = a == u ? b : a
            x == v && continue
            iv = get(edge_index, _edge_key(v, x), 0)
            (iv != 0 && active[iv]) || continue
            push!(neighbors, x)
            push!(incident_u, iu)
            push!(incident_v, iv)
        end
        isempty(neighbors) && continue

        # In a 2-skeleton an edge with two or more common neighbors cannot have
        # a cone link: the triangles are maximal and have different third vertices.
        max_dim == 2 && length(neighbors) > 1 && continue
        for j in eachindex(neighbors)
            w = neighbors[j]
            _grade_leq(grades[incident_u[j]], grades[idx]) || continue
            _grade_leq(grades[incident_v[j]], grades[idx]) || continue
            dominates = true
            for k in eachindex(neighbors)
                k == j && continue
                iw = get(edge_index, _edge_key(w, neighbors[k]), 0)
                if iw == 0 || !active[iw] ||
                   !_grade_leq_join(grades[iw], grades[idx],
                                    grades[incident_u[k]], grades[incident_v[k]])
                    dominates = false
                    break
                end
            end
            dominates || continue

            if length(neighbors) >= max_dim &&
               _link_has_clique(neighbors, w, max_dim - 1,
                                edge_index, active, selected)
                # Any strong apex is universal in this final link. A forbidden
                # clique for one therefore implies one for every other apex.
                break
            end
            active[idx] = false
            break
        end
    end
    return findall(active)
end

# Enumerate retained cliques by intersecting sorted forward adjacency lists.
# Only actual cliques are materialized, and budgets are checked before insertion.
function _flag_simplices(edges::Vector{NTuple{2,Int}}, n::Int, max_dim::Int;
                          max_simplices=nothing, memory_budget_bytes=nothing)
    n >= 0 && max_dim >= 0 || throw(ArgumentError("flag expansion requires nonnegative n and max_dim."))
    simplices = [Vector{Vector{Int}}() for _ in 0:max_dim]
    counts = zeros(Int, max_dim + 1)
    total = Ref(0)
    function append_simplex!(vertices)
        k = length(vertices)
        next_count = counts[k] + 1
        max_simplices === nothing || total[] < max_simplices ||
            throw(ArgumentError("flag expansion exceeds construction max_simplices=$max_simplices."))
        if memory_budget_bytes !== nothing
            previous = k > 1 ? counts[k - 1] : 0
            following = k <= max_dim ? counts[k + 1] : 0
            # Match the ingestion contract: peak dense boundary footprint.
            Int128(8) * next_count * max(previous, following) <= memory_budget_bytes ||
                throw(ArgumentError("flag expansion exceeds construction memory_budget_bytes=$memory_budget_bytes (estimated dense boundary footprint)."))
        end
        push!(simplices[k], copy(vertices))
        counts[k] = next_count
        total[] += 1
    end
    for v in 1:n
        append_simplex!(Int[v])
    end
    max_dim == 0 && return simplices
    forward = [Int[] for _ in 1:n]
    for (u, v) in edges
        a, b = minmax(u, v)
        1 <= a < b <= n || throw(ArgumentError("flag edges require distinct endpoints in 1:n."))
        push!(forward[a], b)
    end
    for neighbors in forward
        sort!(neighbors)
        allunique(neighbors) || throw(ArgumentError("flag edges must be unique."))
    end
    # Preserve sorted simplex order within dimensions for deterministic boundaries.
    function extend!(prefix, candidates)
        for (j, v) in enumerate(candidates)
            push!(prefix, v)
            append_simplex!(prefix)
            if length(prefix) <= max_dim
                next = Int[]
                neighbors = forward[v]
                p = j + 1
                q = 1
                while p <= length(candidates) && q <= length(neighbors)
                    a, b = candidates[p], neighbors[q]
                    if a == b
                        push!(next, a)
                        p += 1
                        q += 1
                    elseif a < b
                        p += 1
                    else
                        q += 1
                    end
                end
                isempty(next) || extend!(prefix, next)
            end
            pop!(prefix)
        end
    end
    for v in 1:n
        extend!(Int[v], forward[v])
    end
    return simplices
end

# A flag simplex is born at the coordinatewise join of its fixed vertex/edge
# grades. Callers must not recompute graph-derived values after deleting edges.
@inline _grade_join(a::NTuple{N,T}, b::NTuple{N,T}) where {N,T} =
    ntuple(k -> max(a[k], b[k]), N)

function _flag_grades(simplices, edges::Vector{NTuple{2,Int}},
                       vertex_grades::Vector{NTuple{N,T}},
                       edge_grades::Vector{NTuple{N,T}}) where {N,T<:Real}
    length(edges) == length(edge_grades) || throw(ArgumentError("one grade is required per edge."))
    edge_index = Dict{UInt128,Int}(_edge_key(u, v) => i for (i, (u, v)) in enumerate(edges))
    grades = NTuple{N,T}[]
    sizehint!(grades, sum(length, simplices))
    for cells in simplices, simplex in cells
        g = vertex_grades[first(simplex)]
        for v in simplex
            g = _grade_join(g, vertex_grades[v])
        end
        for i in 1:length(simplex), j in (i + 1):length(simplex)
            h = edge_grades[edge_index[_edge_key(simplex[i], simplex[j])]]
            g = _grade_join(g, h)
        end
        push!(grades, g)
    end
    return grades
end

end # module SimplicialReduction
