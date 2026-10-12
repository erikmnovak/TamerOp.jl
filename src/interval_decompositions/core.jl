"""
    IntervalDecompositionBudget(; max_vertices=512, max_dimension=2048,
        max_matrix_entries=8_000_000, max_antichains=10_000,
        max_candidates=100_000)

Bounds on the finite base, total stalk dimension, logical entries in each
algebra workspace, endpoint-antichain enumeration (per endpoint set), and
endpoint pairs or supplied intervals. Bounds apply before the bounded work.
Exceeding a bound throws `ArgumentError`; no partial result is returned.
Matrix bounds include dense Hom bases even when the equations are sparse.
Candidate enumeration can be exponential; use a supplied family for large bases.
"""
struct IntervalDecompositionBudget
    max_vertices::Int
    max_dimension::Int
    max_matrix_entries::Int
    max_antichains::Int
    max_candidates::Int
    function IntervalDecompositionBudget(; max_vertices=512, max_dimension=2048,
            max_matrix_entries=8_000_000, max_antichains=10_000,
            max_candidates=100_000)
        values = (max_vertices, max_dimension, max_matrix_entries,
                  max_antichains, max_candidates)
        all(x -> x isa Integer && !(x isa Bool) && 0 < x <= typemax(Int), values) ||
            throw(ArgumentError("interval-decomposition budgets must be positive machine integers"))
        new(Int.(values)...)
    end
end

function _bound(value, limit, name)
    value <= limit || throw(ArgumentError("interval-decomposition $name budget exceeded ($value > $limit)"))
    return nothing
end

function _context(M::PModule, budget)
    M.field isa Union{QQField,PrimeField} ||
        throw(ArgumentError("interval summands require exact QQField or PrimeField coefficients"))
    n = nvertices(M.Q)
    _bound(n, budget.max_vertices, "vertices")
    _bound(sum(big, M.dims; init=big(0)), budget.max_dimension, "dimension")
    _bound(big(n)^2, budget.max_matrix_entries, "order matrix")
    _bound(maximum(big, M.dims; init=big(0))^2,
           budget.max_matrix_entries, "path-validation matrix")
    # Covers must define a functor, including on merge shapes larger than squares.
    check_module_functoriality(M; throw=true)
    order = BitMatrix([leq(M.Q, u, v) for u in 1:n, v in 1:n])
    return order
end

function _labels(order, vertices)
    vertices isa Union{AbstractVector,AbstractSet,Tuple} ||
        throw(ArgumentError("vertices must be a finite collection of integer labels"))
    n = size(order, 1)
    isempty(vertices) && throw(ArgumentError("an interval must be nonempty"))
    length(vertices) <= n || throw(ArgumentError("an interval cannot contain more distinct labels than its finite base"))
    all(v -> v isa Integer && !(v isa Bool) && 1 <= v <= n, vertices) ||
        throw(ArgumentError("interval labels must be integers in 1:$n"))
    labels = sort!(Int[v for v in vertices])
    length(unique(labels)) == length(labels) || throw(ArgumentError("interval labels must be distinct"))
    _is_interval(order, labels) || throw(ArgumentError("vertices must be connected and order-convex in the finite base"))
    return labels
end

function _is_interval(order, labels)
    isempty(labels) && return false
    selected = falses(size(order, 1))
    selected[labels] .= true
    for z in eachindex(selected)
        selected[z] && continue
        any(a -> order[a,z], labels) && any(b -> order[z,b], labels) && return false
    end
    seen = falses(length(selected))
    seen[first(labels)] = true
    queue = [first(labels)]
    for a in queue, b in labels
        if !seen[b] && (order[a,b] || order[b,a])
            seen[b] = true
            push!(queue, b)
        end
    end
    return all(seen[labels])
end

function _endpoints(order, labels)
    minima = [v for v in labels if !any(u -> u != v && order[u,v], labels)]
    maxima = [v for v in labels if !any(u -> u != v && order[v,u], labels)]
    return minima, maxima
end

function _top_socle(M::PModule{K}, budget) where {K}
    top, socle = copy(M.dims), copy(M.dims)
    for v in eachindex(M.dims)
        incoming = [M.edge_maps[u,v] for u in M.edge_maps.preds[v]]
        outgoing = [M.edge_maps[v,w] for w in M.edge_maps.succs[v]]
        if !isempty(incoming)
            _bound(big(M.dims[v]) * sum(A -> big(size(A,2)), incoming),
                   budget.max_matrix_entries, "top matrix")
            top[v] -= FieldLinAlg.rank(M.field, reduce(hcat, incoming))
        end
        if !isempty(outgoing)
            _bound(big(M.dims[v]) * sum(A -> big(size(A,1)), outgoing),
                   budget.max_matrix_entries, "socle matrix")
            socle[v] -= FieldLinAlg.rank(M.field, reduce(vcat, outgoing))
        end
    end
    return top, socle
end

function _antichains(order, vertices, budget)
    result = Vector{Int}[]
    current = Int[]
    function visit(start)
        for j in start:length(vertices)
            v = vertices[j]
            any(u -> order[u,v] || order[v,u], current) && continue
            _bound(length(result)+1, budget.max_antichains, "endpoint antichains")
            push!(current, v)
            push!(result, copy(current))
            visit(j+1)
            pop!(current)
        end
    end
    visit(1)
    return result
end

function _candidates(M, order, top, socle, budget)
    lower = _antichains(order, findall(>(0), top), budget)
    upper = _antichains(order, findall(>(0), socle), budget)
    _bound(big(length(lower))*length(upper), budget.max_candidates, "endpoint pairs")
    family = Vector{Int}[]
    for a in lower, b in upper
        labels = [v for v in eachindex(M.dims)
                  if any(u -> order[u,v], a) && any(w -> order[v,w], b)]
        isempty(labels) && continue
        all(v -> M.dims[v] > 0, labels) || continue
        _endpoints(order, labels) == (a,b) || continue
        _is_interval(order, labels) || continue
        push!(family, labels)
    end
    sort!(family; by=Tuple)
    return family, length(lower)*length(upper)
end

"""
    interval_candidates(M; budget=IntervalDecompositionBudget())

Connected convex finite-label supports that pass the stalk and degree-zero
top/socle tests. Every possible interval summand is included; surviving a test
does not establish positive multiplicity. Enumeration uses antichains of the
top/socle supports, not all subsets of the entire base. See `interval_multiplicity`.
"""
function interval_candidates(M::PModule; budget::IntervalDecompositionBudget=IntervalDecompositionBudget())
    order = _context(M, budget)
    top, socle = _top_socle(M, budget)
    return first(_candidates(M, order, top, socle, budget))
end

function _interval_module(M::PModule{K}, labels, multiplicity=1) where {K}
    dims = zeros(Int, length(M.dims))
    dims[labels] .= multiplicity
    maps = Dict{Tuple{Int,Int},Matrix{K}}()
    for (u,v) in cover_edges(M.Q)
        maps[(u,v)] = dims[u] > 0 && dims[v] > 0 ?
            Matrix{K}(eye(M.field, multiplicity)) : zeros(K, dims[v], dims[u])
    end
    return PModule{K}(M.Q, dims, maps; field=M.field)
end

function _pairing(M::PModule{K}, labels, budget) where {K}
    d = sum(v -> big(M.dims[v]), labels)
    _bound(d*d, budget.max_matrix_entries, "Hom basis/pairing")
    I = _interval_module(M, labels)
    # End(k_I)=k for a nonempty connected interval, so one component determines
    # each composite. Public Hom bases retain the ambient boundary equations.
    incoming, outgoing = basis(Hom(I, M)), basis(Hom(M, I))
    q = first(labels)
    A = isempty(incoming) ? zeros(K, M.dims[q], 0) : reduce(hcat, [f.comps[q] for f in incoming])
    B = isempty(outgoing) ? zeros(K, 0, M.dims[q]) : reduce(vcat, [g.comps[q] for g in outgoing])
    return B*A, incoming, outgoing
end

function _upper_bound(M, labels, order, top, socle)
    minima, maxima = _endpoints(order, labels)
    return min(minimum(M.dims[labels]), minimum(top[minima]), minimum(socle[maxima]))
end

"""
    interval_multiplicity(M; vertices, budget=IntervalDecompositionBudget()) -> Int

Number of copies of the thin interval module on `vertices` in the Krull-Schmidt
decomposition of M on its finite base. The labels must be nonempty, connected
and order-convex. Exact QQ and prime fields only; the complete module is checked
for path independence. Maps crossing the interval boundary are included.

This is an actual summand multiplicity, not the generalized rank of a restriction.
For splitting maps use `interval_decomposition(M; family=[vertices], witnesses=true)`.
"""
function interval_multiplicity(M::PModule; vertices,
        budget::IntervalDecompositionBudget=IntervalDecompositionBudget())
    order = _context(M, budget)
    labels = _labels(order, vertices)
    top, socle = _top_socle(M, budget)
    _upper_bound(M, labels, order, top, socle) == 0 && return 0
    C, _, _ = _pairing(M, labels, budget)
    return FieldLinAlg.rank(M.field, C)
end

function _split(M::PModule{K}, labels, C, incoming, outgoing) where {K}
    _, pivot_cols = FieldLinAlg.rref(M.field, C)
    cols = collect(pivot_cols)
    _, pivot_rows = FieldLinAlg.rref(M.field, Matrix(transpose(C[:,cols])))
    rows = collect(pivot_rows)
    r = length(cols)
    Dinv = FieldLinAlg.solve_fullcolumn(M.field, C[rows,cols], eye(M.field,r))
    S = _interval_module(M, labels, r)
    inc = Matrix{K}[S.dims[v] == 0 ? zeros(K,M.dims[v],0) :
                   reduce(hcat, [incoming[j].comps[v] for j in cols]) for v in eachindex(M.dims)]
    proj = Matrix{K}[S.dims[v] == 0 ? zeros(K,0,M.dims[v]) :
                    Dinv * reduce(vcat, [outgoing[j].comps[v] for j in rows]) for v in eachindex(M.dims)]
    i, p = PMorphism{K}(S,M,inc), PMorphism{K}(M,S,proj)
    R, j = kernel_with_inclusion(p)
    q = PMorphism{K}(M,R,Matrix{K}[
        FieldLinAlg.solve_fullcolumn(M.field, j.comps[v],
            eye(M.field,M.dims[v])-inc[v]*proj[v]) for v in eachindex(M.dims)])
    return (; summand=S, inclusion=i, projection=p, residual=R,
              residual_inclusion=j, residual_projection=q)
end

function _declared_family(order, family, budget)
    family isa Union{AbstractVector,Tuple} ||
        throw(ArgumentError("family must be :all or a finite vector/tuple of interval label collections"))
    _bound(length(family), budget.max_candidates, "supplied intervals")
    labels = Vector{Int}[_labels(order,I) for I in family]
    length(Set(Tuple.(labels))) == length(labels) || throw(ArgumentError("family contains duplicate intervals"))
    sort!(labels; by=Tuple)
    return labels
end

"""
    interval_decomposition(M; family=:all, witnesses=false,
        budget=IntervalDecompositionBudget()) -> IntervalDecompositionResult

Compute actual multiplicities for a declared interval family or every possible
interval summand after safe top/socle pruning. Only positive multiplicities are
stored. `:all` certifies that the residual has no interval summands; it need not
be indecomposable. A nonzero residual from a supplied family leaves the global
interval-decomposability question undecided (`missing`).

The default retains counts and residual dimensions. `witnesses=true` additionally
constructs a simultaneous splitting `M = S + R` with natural inclusion/projection
maps, orthogonal summand/residual blocks, and full reconstruction. Bases/splittings
are not canonical. Resource exhaustion throws; it never means nondecomposability.
"""
function interval_decomposition(M::PModule{K}; family=:all, witnesses::Bool=false,
        budget::IntervalDecompositionBudget=IntervalDecompositionBudget()) where {K}
    order = _context(M, budget)
    top, socle = _top_socle(M, budget)
    exhaustive = family === :all
    candidates, scanned = exhaustive ? _candidates(M,order,top,socle,budget) :
        ( _declared_family(order,family,budget), length(family) )
    # Bound retained simultaneous maps and complement solves before extraction.
    witnesses && _bound(sum(d -> big(d)^2, M.dims; init=big(0)), budget.max_matrix_entries, "splitting maps")
    supports, multiplicities = Vector{Int}[], Int[]
    remainder = copy(M.dims)
    residual = M
    U = witnesses ? id_morphism(M) : nothing
    V = witnesses ? id_morphism(M) : nothing
    inclusions, projections = Vector{Matrix{K}}[], Vector{Matrix{K}}[]
    terms = PModule{K}[]
    tested = 0
    for labels in candidates
        _upper_bound(M,labels,order,top,socle) == 0 && continue
        tested += 1
        C, incoming, outgoing = _pairing(witnesses ? residual : M, labels, budget)
        r = FieldLinAlg.rank(M.field,C)
        r == 0 && continue
        push!(supports,copy(labels)); push!(multiplicities,r)
        remainder[labels] .-= r
        if witnesses
            w = _split(residual,labels,C,incoming,outgoing)
            push!(terms,w.summand)
            push!(inclusions,Matrix{K}[U.comps[v]*w.inclusion.comps[v] for v in eachindex(M.dims)])
            push!(projections,Matrix{K}[w.projection.comps[v]*V.comps[v] for v in eachindex(M.dims)])
            U = PMorphism{K}(w.residual,M,Matrix{K}[U.comps[v]*w.residual_inclusion.comps[v] for v in eachindex(M.dims)])
            V = PMorphism{K}(M,w.residual,Matrix{K}[w.residual_projection.comps[v]*V.comps[v] for v in eachindex(M.dims)])
            residual = w.residual
        end
    end
    maps = nothing
    if witnesses
        S = isempty(terms) ? zero_pmodule(M.Q; field=M.field) : reduce(direct_sum,terms)
        inc = Matrix{K}[isempty(inclusions) ? zeros(K,M.dims[v],0) :
                       reduce(hcat,[a[v] for a in inclusions]) for v in eachindex(M.dims)]
        proj = Matrix{K}[isempty(projections) ? zeros(K,0,M.dims[v]) :
                        reduce(vcat,[a[v] for a in projections]) for v in eachindex(M.dims)]
        maps = (; summand=S, inclusion=PMorphism{K}(S,M,inc), projection=PMorphism{K}(M,S,proj),
                  residual, residual_inclusion=U, residual_projection=V)
    end
    return IntervalDecompositionResult(M,supports,multiplicities,remainder,exhaustive,
        exhaustive ? nothing : candidates,(;scanned,candidates=length(candidates),tested),maps)
end
