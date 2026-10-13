# Public result access, provenance and exact reconstruction validation.

"""
    IndecomposableDecompositionResult

A certified decomposition into individual indecomposable finite-poset modules.
Use `summand_dimensions`, `indecomposable_decomposition_summary` and
`is_indecomposable` for cheap queries. With `witnesses=true`,
`indecomposable_summands` and `decomposition_maps` expose the actual modules and
their natural inclusions/projections. The source and retained maps are references
and must be kept read-only; `check_indecomposable_decomposition` rechecks them.
"""
struct IndecomposableDecompositionResult{M,W}
    source::M
    dimensions::Vector{Vector{Int}}
    certificates::Vector{_Certificate}
    route::Symbol
    backend::Symbol
    statistics::NamedTuple{(:search_steps,:endomorphisms_tested,:primary_splits),Tuple{Int,Int,Int}}
    maps::W
end

source_module(r::IndecomposableDecompositionResult) = r.source
source_poset(r::IndecomposableDecompositionResult) = r.source.Q

"""Return one copied dimension vector per indecomposable, including repeated summands."""
summand_dimensions(r::IndecomposableDecompositionResult) = [copy(d) for d in r.dimensions]

"""Return retained summands; requires `witnesses=true`. Keep the modules read-only."""
indecomposable_summands(r::IndecomposableDecompositionResult) = copy(decomposition_maps(r).summands)

"""
    decomposition_maps(r::IndecomposableDecompositionResult)

Return `(summands, inclusions, projections)`. Every summand has its own natural
inclusion into the source and projection back. All cross composites vanish;
the projections split their inclusions and the resulting idempotents sum to the
source identity. Requires `witnesses=true`. Retained objects are read-only.
"""
function decomposition_maps(r::IndecomposableDecompositionResult)
    r.maps === nothing && throw(ArgumentError("Request witnesses=true to retain decomposition maps."))
    return r.maps
end

"""Test whether the certified result has exactly one nonzero summand; zero is not indecomposable."""
is_indecomposable(r::IndecomposableDecompositionResult) = length(r.dimensions) == 1
is_indecomposable(M::PModule;kwargs...) = is_indecomposable(indecomposable_decomposition(M;kwargs...))

"""
    indecomposable_decomposition_summary(result)

Report individual dimensions, proof kinds, the actual computational route and
search costs without materializing maps. Repeated dimension vectors are not an
isomorphism classification. For example, `describe(indecomposable_decomposition(M))`
is the cheap entry point; request witnesses only to inspect the splitting.
Certificate `residue_dimension` and `center_degree` are dimensions over the
original coefficient field; `schur_index` is the degree of the division algebra
over its center. Unknown invariants are `nothing`, not inferred from sampling.
"""
function indecomposable_decomposition_summary(r::IndecomposableDecompositionResult)
    return (;kind=:indecomposable_decomposition,scope=:finite_poset,complete=true,
        vertices=nvertices(r.source.Q),source_dimensions=copy(r.source.dims),
        summands=length(r.dimensions),summand_dimensions=summand_dimensions(r),
        certificates=_certificate_summary.(r.certificates),route=r.route,backend=r.backend,
        witnesses=r.maps !== nothing,statistics=r.statistics)
end
describe(r::IndecomposableDecompositionResult) = indecomposable_decomposition_summary(r)
function Base.show(io::IO,r::IndecomposableDecompositionResult)
    print(io,"IndecomposableDecompositionResult(",length(r.dimensions)," summands, route=",r.route,
        ", backend=",r.backend,", witnesses=",r.maps !== nothing,")")
end
provenance(r::IndecomposableDecompositionResult) = (;scope=:finite_poset,
    ground_field=r.source.field,route=r.route,backend=r.backend,ambient_identification=:not_asserted,
    stability=:not_asserted,bases=:noncanonical)

"""Exact validation outcome; inspect `.valid`, `.issues` and `.splitting_checked`."""
struct IndecomposableDecompositionValidationSummary
    valid::Bool
    issues::Vector{String}
    splitting_checked::Bool
end
Base.:(==)(a::IndecomposableDecompositionValidationSummary,b::IndecomposableDecompositionValidationSummary) =
    (a.valid,a.issues,a.splitting_checked) == (b.valid,b.issues,b.splitting_checked)
Base.isequal(a::IndecomposableDecompositionValidationSummary,b::IndecomposableDecompositionValidationSummary) =
    isequal((a.valid,a.issues,a.splitting_checked),(b.valid,b.issues,b.splitting_checked))
Base.hash(r::IndecomposableDecompositionValidationSummary,h::UInt) = hash((r.valid,r.issues,r.splitting_checked),h)
indecomposable_decomposition_summary(r::IndecomposableDecompositionValidationSummary) =
    (;valid=r.valid,issues=copy(r.issues),splitting_checked=r.splitting_checked)
describe(r::IndecomposableDecompositionValidationSummary) = indecomposable_decomposition_summary(r)
Base.show(io::IO,r::IndecomposableDecompositionValidationSummary) =
    print(io,"IndecomposableDecompositionValidationSummary(valid=",r.valid,", issues=",r.issues,")")

function _check_splitting(r,budget)
    M,w = r.source,r.maps
    n = length(r.dimensions)
    length(w.summands) == length(w.inclusions) == length(w.projections) == n ||
        throw(ArgumentError("Expected one module, inclusion and projection per summand."))
    K = coeff_type(M.field)
    total = Matrix{K}[zeros(K,d,d) for d in M.dims]
    state = _SearchState(0,0,0)
    for k in 1:n
        N,i,p = w.summands[k],w.inclusions[k],w.projections[k]
        N.Q === M.Q && N.field == M.field && N.dims == r.dimensions[k] ||
            throw(ArgumentError("Summand base, field or dimensions disagree with the result."))
        _context(N,budget)
        any(!iszero,N.dims) || throw(ArgumentError("Zero is not an indecomposable summand."))
        i.dom === N && i.cod === M && p.dom === M && p.cod === N ||
            throw(ArgumentError("Splitting map domains/codomains disagree with the summands."))
        check_morphism(i;throw=true);check_morphism(p;throw=true)
        _classify(N,budget,state,r.backend) isa _Certificate || throw(ArgumentError("A retained summand is decomposable."))
        for v in eachindex(M.dims)
            p.comps[v]*i.comps[v] == eye(M.field,N.dims[v]) ||
                throw(ArgumentError("A projection does not split its inclusion."))
            total[v] .+= i.comps[v]*p.comps[v]
            for j in 1:k-1
                all(iszero,p.comps[v]*w.inclusions[j].comps[v]) &&
                    all(iszero,w.projections[j].comps[v]*i.comps[v]) ||
                    throw(ArgumentError("Distinct summands are not orthogonal."))
            end
        end
    end
    all(total[v] == eye(M.field,M.dims[v]) for v in eachindex(M.dims)) ||
        throw(ArgumentError("The summand idempotents do not reconstruct the source identity."))
    return nothing
end

"""
    check_indecomposable_decomposition(result; throw=false, budget=...)

Recompute the exact decomposition/proof summaries and validate any retained
modules, natural maps, orthogonality and full reconstruction. This is an expensive
validator. Malformed results or insufficient checking budgets produce an invalid
typed summary; `throw=true` instead raises `ArgumentError`.
"""
function check_indecomposable_decomposition(r::IndecomposableDecompositionResult;
        throw::Bool=false,budget::IndecomposableDecompositionBudget=IndecomposableDecompositionBudget())
    issues = String[]
    checked = false
    try
        r.route in (:endomorphism_algebra,:bipath_intervals) ||
            Base.throw(ArgumentError("Unknown decomposition route."))
        _check_backend(r.backend,r.source.field)
        r.route == :bipath_intervals && r.backend != :native &&
            Base.throw(ArgumentError("Bipath extraction uses backend=:native."))
        expected = r.route == :bipath_intervals ? bipath_decomposition(r.source;budget) :
            indecomposable_decomposition(r.source;budget,backend=r.backend)
        r.dimensions == expected.dimensions || Base.throw(ArgumentError("Summand dimensions disagree with exact recomputation."))
        _certificate_summary.(r.certificates) == _certificate_summary.(expected.certificates) ||
            Base.throw(ArgumentError("Indecomposability certificates disagree with recomputation."))
        r.statistics == expected.statistics || Base.throw(ArgumentError("Search statistics disagree with recomputation."))
        if r.maps !== nothing
            _check_splitting(r,budget)
            checked = true
        end
    catch e
        e isa InterruptException && rethrow()
        push!(issues,sprint(showerror,e))
    end
    report = IndecomposableDecompositionValidationSummary(isempty(issues),issues,checked)
    throw && !report.valid && Base.throw(ArgumentError(join(issues,"; ")))
    return report
end
