"""
    GeneralizedRankBudget(; max_vertices=512, max_dimension=2048,
        max_matrix_entries=8_000_000, max_order_checks=2_000_000,
        max_queries=10_000, max_events=100_000)

Explicit per-query algebra/order bounds and per-call query/event bounds for
generalized rank, declared interval families, and GRIL. Matrix bounds count
logical entries, including potentially dense elimination work, not just nonzeros.
Exceeding a bound throws before that allocation/query; no partial summary is returned.
"""
struct GeneralizedRankBudget
    max_vertices::Int
    max_dimension::Int
    max_matrix_entries::Int
    max_order_checks::Int
    max_queries::Int
    max_events::Int
    function GeneralizedRankBudget(; max_vertices=512, max_dimension=2048,
            max_matrix_entries=8_000_000, max_order_checks=2_000_000,
            max_queries=10_000, max_events=100_000)
        limits = (max_vertices, max_dimension, max_matrix_entries,
                  max_order_checks, max_queries, max_events)
        all(x -> x isa Integer && !(x isa Bool) && 0 < x <= typemax(Int), limits) ||
            throw(ArgumentError("generalized-rank budgets must be positive machine integers"))
        new(Int.(limits)...)
    end
end

"""Typed query validation report with `.valid`, `.issues`, and the interpretation checked."""
struct GeneralizedRankValidationSummary{R} <: _InvariantValidationWrapper
    report::R
end
Base.show(io::IO, r::GeneralizedRankValidationSummary) =
    print(io, "GeneralizedRankValidationSummary(valid=", r.valid, ", issues=", r.issues, ")")
describe(r::GeneralizedRankValidationSummary) = r.report

@inline function _gr_bound(value::Integer, limit::Int, name)
    value <= limit || throw(ArgumentError("generalized-rank $name budget exceeded ($value > $limit)"))
    return nothing
end

function _gr_field(M::PModule)
    M.field isa Union{QQField,PrimeField} ||
        throw(ArgumentError("generalized rank requires an exact QQField or PrimeField; RealField is not supported"))
    return M.field
end

"""
    check_generalized_rank_region(M; vertices, budget=GeneralizedRankBudget(), throw=false)

Validate a nonempty, connected, order-convex selection of finite-poset labels.
This checks the query domain, not functoriality of a hand-built module. Labels
refer to `M.Q`; they are not samples of an unspecified ambient region.
"""
function check_generalized_rank_region(M::PModule; vertices,
        budget::GeneralizedRankBudget=GeneralizedRankBudget(), throw::Bool=false)
    issues = String[]
    labels = Int[]
    n = nvertices(M.Q)
    if !(vertices isa Union{AbstractVector,AbstractSet,Tuple})
        push!(issues, "vertices must be a finite collection of integer labels")
    elseif isempty(vertices)
        push!(issues, "the selected region must be nonempty")
    elseif length(vertices) > budget.max_vertices
        push!(issues, "selected vertex budget exceeded")
    elseif any(v -> !(v isa Integer) || v isa Bool || !(1 <= v <= n), vertices)
        push!(issues, "vertices must be integer labels between 1 and $n")
    else
        labels = sort!(Int[v for v in vertices])
        length(unique(labels)) == length(labels) || push!(issues, "vertices must be distinct")
    end
    if isempty(issues)
        k = length(labels)
        # Induced order, convexity against the ambient finite base, and covers.
        cost = big(k)^3 + 2 * big(n) * k
        if cost > budget.max_order_checks
            push!(issues, "selected-region order-check budget exceeded")
        else
            selected = Set(labels)
            for z in 1:n
                z in selected && continue
                if any(a -> leq(M.Q, a, z), labels) && any(b -> leq(M.Q, z, b), labels)
                    push!(issues, "region is not order-convex: omitted label $z lies between selected labels")
                    break
                end
            end
            seen = falses(k)
            seen[1] = true
            queue = [1]
            for a in queue
                for b in 1:k
                    if !seen[b] && (leq(M.Q, labels[a], labels[b]) || leq(M.Q, labels[b], labels[a]))
                        seen[b] = true
                        push!(queue, b)
                    end
                end
            end
            all(seen) || push!(issues, "region must be connected in its undirected comparability graph")
        end
    end
    result = GeneralizedRankValidationSummary((valid=isempty(issues), issues=issues,
                                               vertices=labels, interpretation=:finite_poset))
    throw && !result.valid && Base.throw(ArgumentError(join(issues, "; ")))
    return result
end

"""
    GeneralizedRankResult

Opt-in witnesses for the canonical limit-to-colimit map on a selected finite
region. `limit_basis` uses the direct sum of stalks in `selected_vertices` order;
`colimit_projection` maps that direct sum onto quotient coordinates.
Witness bases are coordinate choices, not canonical bases.
"""
struct GeneralizedRankResult{K,P,F}
    Q::P
    field::F
    vertices::Vector{Int}
    dims::Vector{Int}
    rank::Int
    limit::Matrix{K}
    quotient::Matrix{K}
    comparison::Matrix{K}
end
source_poset(r::GeneralizedRankResult) = r.Q
"""Return a copy of the finite labels in the witness direct-sum order."""
selected_vertices(r::GeneralizedRankResult) = copy(r.vertices)
"""Return a copy of the compatible-section basis, in selected-stalk direct-sum coordinates."""
limit_basis(r::GeneralizedRankResult) = copy(r.limit)
"""Return the full-row-rank quotient-coordinate map from the selected stalk direct sum."""
colimit_projection(r::GeneralizedRankResult) = copy(r.quotient)
"""Return a copy of the canonical limit-to-colimit map in the retained witness bases."""
comparison_map(r::GeneralizedRankResult) = copy(r.comparison)
generalized_rank(r::GeneralizedRankResult) = r.rank
"""Summarize a generalized-rank witness or declared-family signed result without elimination."""
generalized_rank_summary(r::GeneralizedRankResult) =
    (kind=:generalized_rank, rank=r.rank, nvertices=length(r.vertices),
     total_dimension=sum(r.dims), limit_dimension=size(r.limit, 2),
     colimit_dimension=size(r.quotient, 1), interpretation=:finite_poset, field=r.field)
describe(r::GeneralizedRankResult) = generalized_rank_summary(r)
Base.show(io::IO, r::GeneralizedRankResult) =
    print(io, "GeneralizedRankResult(rank=", r.rank, ", nvertices=", length(r.vertices), ")")

# The same presentation applies to a finite poset and to the finite diagram
# obtained by contracting connected constant fibers of a continuous worm.
function _gr_presentations(M::PModule{K}, labels, edges, budget) where {K}
    dims = M.dims[labels]
    d = sum(big, dims)
    _gr_bound(d, budget.max_dimension, "stalk dimension")
    nr = sum((big(dims[b]) for (a,b) in edges); init=big(0))
    nc = sum((big(dims[a]) for (a,b) in edges); init=big(0))
    # Includes bases/projection/comparison upper bounds, even for sparse inputs.
    _gr_bound(d * (nr + nc) + 3*d*d, budget.max_matrix_entries, "matrix entries")
    offsets = cumsum([0; dims])
    ci = Int[]; cj = Int[]; cv = K[]
    ri = Int[]; rj = Int[]; rv = K[]
    row = 0; col = 0
    for (a,b) in edges
        A = map_leq(M, labels[a], labels[b])
        for j in 1:dims[a], i in 1:dims[b]
            v = A[i,j]
            iszero(v) && continue
            push!(ci,row+i); push!(cj,offsets[a]+j); push!(cv,v)
            push!(ri,offsets[b]+i); push!(rj,col+j); push!(rv,-v)
        end
        for i in 1:dims[b]
            push!(ci,row+i); push!(cj,offsets[b]+i); push!(cv,-one(K))
        end
        for j in 1:dims[a]
            push!(ri,offsets[a]+j); push!(rj,col+j); push!(rv,one(K))
        end
        row += dims[b]; col += dims[a]
    end
    return sparse(ci,cj,cv,Int(nr),Int(d)), sparse(ri,rj,rv,Int(d),Int(nc)), offsets
end

function _gr_diagram(M::PModule, labels, edges, budget; witnesses=false)
    # A zero stalk forces the canonical comparison to be zero on a connected diagram.
    !witnesses && any(v -> iszero(M.dims[v]), labels) && return 0
    C, R, offsets = _gr_presentations(M, labels, edges, budget)
    L = Matrix(FieldLinAlg.nullspace(M.field, C))
    Q = Matrix(transpose(FieldLinAlg.nullspace(M.field, transpose(R))))
    # ONE anchor, never the sum over vertices (that sum fails in finite characteristic).
    anchor = offsets[1]+1:offsets[2]
    comparison = Q[:,anchor] * L[anchor,:]
    rank = FieldLinAlg.rank_dim(M.field, comparison)
    witnesses || return rank
    return GeneralizedRankResult(M.Q, M.field, copy(labels), M.dims[labels], rank, L, Q, comparison)
end

"""
    generalized_rank(M; vertices, witnesses=false, budget=GeneralizedRankBudget())

Exact rank of the canonical map `limit(M|I) -> colimit(M|I)` on the connected,
order-convex finite subposet `I = vertices`. Returns an integer by default;
`witnesses=true` retains a `GeneralizedRankResult`. Supports QQ and prime fields.
On a chain interval this equals the rank of its endpoint map. Branching domains
can distinguish modules with the same stalk dimensions and pairwise ranks.
No enumeration of all intervals is performed. An encoding wrapper uses its
finite labels; ambient geometric queries require a separate justified restriction.
"""
function generalized_rank(M::PModule; vertices, witnesses::Bool=false,
        budget::GeneralizedRankBudget=GeneralizedRankBudget())
    _gr_field(M)
    labels = check_generalized_rank_region(M; vertices=vertices, budget=budget, throw=true).vertices
    k = length(labels)
    order = [leq(M.Q,a,b) for a in labels, b in labels]
    edges = Tuple{Int,Int}[]
    for a in 1:k, b in 1:k
        a == b && continue
        order[a,b] || continue
        any(c -> c != a && c != b && order[a,c] && order[c,b], 1:k) && continue
        push!(edges, (a,b))
    end
    return _gr_diagram(M, labels, edges, budget; witnesses=witnesses)
end

"""
    IntervalRankSummary

Signed Mobius coefficients on a caller-declared finite family of connected,
convex subposets, ordered by inclusion. The compression contract is `:total`:
each query uses the whole restricted diagram. Reconstruction is guaranteed only
on the declared family. Negative coefficients are allowed and are not direct-sum
multiplicities or a claim of interval decomposability.
"""
struct IntervalRankSummary{P}
    Q::P
    family::Vector{Vector{Int}}
    ranks::Vector{Int}
    weights::Vector{BigInt}
end
source_poset(s::IntervalRankSummary) = s.Q
"""Return independent copies of the declared interval-family label sets, in input order."""
interval_family(s::IntervalRankSummary) = deepcopy(s.family)
"""Return the signed BigInt coefficients, in the order of `interval_family(summary)`."""
interval_coefficients(s::IntervalRankSummary) = copy(s.weights)
nentries(s::IntervalRankSummary) = length(s.family)
generalized_rank_summary(s::IntervalRankSummary) =
    (kind=:interval_rank_summary, nentries=nentries(s), nnonzero=count(!iszero,s.weights),
     compression=:total, reconstruction=:declared_family, interpretation=:finite_poset)
describe(s::IntervalRankSummary) = generalized_rank_summary(s)
Base.show(io::IO, s::IntervalRankSummary) =
    print(io, "IntervalRankSummary(nentries=", nentries(s), ", compression=:total)")

"""
    interval_rank_summary(M; family, compression=:total, budget=GeneralizedRankBudget())

Compute generalized ranks and signed Mobius coefficients on an explicit finite
interval family. No exhaustive family is generated. `:total` retains every map
needed by the restricted diagram; other compression systems need separate
theorems and are rejected. Coefficients use `BigInt` to avoid Mobius overflow.
"""
function interval_rank_summary(M::PModule; family, compression::Symbol=:total,
        budget::GeneralizedRankBudget=GeneralizedRankBudget())
    _gr_field(M)
    compression === :total || throw(ArgumentError("only the :total diagram compression is supported"))
    family isa Union{AbstractVector,Tuple} || throw(ArgumentError("family must be a finite vector or tuple of regions"))
    _gr_bound(length(family), budget.max_queries, "family queries")
    _gr_bound(big(length(family))^2, budget.max_order_checks, "family comparisons")
    regions = Vector{Int}[check_generalized_rank_region(M; vertices=I, budget=budget, throw=true).vertices for I in family]
    length(unique(regions)) == length(regions) || throw(ArgumentError("interval family must not contain duplicate regions"))
    ranks = Int[generalized_rank(M; vertices=I, budget=budget) for I in regions]
    weights = BigInt.(ranks)
    order = sortperm(length.(regions); rev=true)
    for (position,i) in enumerate(order)
        for j in @view order[1:position-1]
            issubset(regions[i],regions[j]) && (weights[i] -= weights[j])
        end
    end
    return IntervalRankSummary(M.Q, regions, ranks, weights)
end

"""
    reconstruct_rank(summary::IntervalRankSummary; vertices)

Reconstruct a declared query by summing signed coefficients of containing
family members. Reject undeclared queries rather than extrapolating a full
generalized-rank invariant from an incomplete family.
"""
function reconstruct_rank(s::IntervalRankSummary; vertices)
    vertices isa Union{AbstractVector,AbstractSet,Tuple} && !isempty(vertices) &&
        all(v -> v isa Integer && !(v isa Bool) && 1 <= v <= nvertices(s.Q), vertices) ||
        throw(ArgumentError("vertices must be a nonempty collection of finite-poset integer labels"))
    I = sort!(Int[v for v in vertices])
    allunique(I) || throw(ArgumentError("vertices must be distinct"))
    I in s.family || throw(ArgumentError("reconstruction is certified only for regions in the declared interval family"))
    return sum((s.weights[j] for j in eachindex(s.family) if issubset(I,s.family[j])); init=big(0))
end
