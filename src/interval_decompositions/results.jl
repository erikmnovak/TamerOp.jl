"""
    IntervalDecompositionResult

Actual interval-summand multiplicities on a fixed finite base, residual dimensions,
search scope and optional simultaneous splitting. Construct with
`interval_decomposition`; inspect with `interval_decomposition_summary`,
`interval_family`, `interval_multiplicities`, `residual_dimensions`,
`is_interval_decomposable`, and `decomposition_maps`.
"""
struct IntervalDecompositionResult{M,W}
    source::M
    supports::Vector{Vector{Int}}
    multiplicities::Vector{Int}
    residual_dims::Vector{Int}
    exhaustive::Bool
    requested::Union{Nothing,Vector{Vector{Int}}}
    statistics::NamedTuple{(:scanned,:candidates,:tested),Tuple{Int,Int,Int}}
    maps::W
end

"""Return the positive-multiplicity interval supports, as copies of finite label sets."""
interval_family(r::IntervalDecompositionResult) = deepcopy(r.supports)
"""Return nonnegative actual summand counts in `interval_family(result)` order."""
interval_multiplicities(r::IntervalDecompositionResult) = copy(r.multiplicities)
"""Return stalk dimensions of the complement of the reported interval summands."""
residual_dimensions(r::IntervalDecompositionResult) = copy(r.residual_dims)
source_module(r::IntervalDecompositionResult) = r.source
source_poset(r::IntervalDecompositionResult) = r.source.Q

"""
    is_interval_decomposable(result)
    is_interval_decomposable(M; kwargs...)

Return `true` for zero residual, `false` for nonzero residual after complete
candidate search, or `missing` for nonzero residual with only a supplied family.
The module overload computes `interval_decomposition` with the same keywords.
Budget exhaustion throws instead of returning any of these answers.
"""
is_interval_decomposable(r::IntervalDecompositionResult) =
    all(iszero,r.residual_dims) ? true : r.exhaustive ? false : missing
is_interval_decomposable(M::PModule; kwargs...) = is_interval_decomposable(interval_decomposition(M;kwargs...))

"""
    decomposition_maps(result)

Return `(summand, inclusion, projection, residual, residual_inclusion,
residual_projection)`. Requires `witnesses=true` at construction. The summand's
basis blocks follow `interval_family`/`interval_multiplicities` order, omitting
blocks absent at a vertex. Both pairs split, their mixed composites vanish,
and their idempotents sum to the identity on `source_module(result)`.
The residual has no further copy of any queried interval.
"""
function decomposition_maps(r::IntervalDecompositionResult)
    r.maps === nothing && throw(ArgumentError("splitting maps were not retained; use witnesses=true"))
    return r.maps
end

"""
    interval_decomposition_summary(result)

Cheap summary of the finite category, actual counts, scope, residual and retained
witnesses. For example, `r = interval_decomposition(M; family=[[1]])` then
`interval_decomposition_summary(r)` distinguishes an undecided global question
from a certified positive or negative answer.
"""
interval_decomposition_summary(r::IntervalDecompositionResult) = (
    kind=:interval_decomposition, category=:finite_poset_representations,
    field=r.source.field, vertices=nvertices(r.source.Q),
    interval_types=length(r.supports), interval_copies=sum(r.multiplicities),
    residual_dimension=sum(r.residual_dims), exhaustive=r.exhaustive,
    interval_decomposable=is_interval_decomposable(r), witnesses=r.maps !== nothing,
    search=r.statistics)
describe(r::IntervalDecompositionResult) = interval_decomposition_summary(r)
Base.show(io::IO,r::IntervalDecompositionResult) = print(io,
    "IntervalDecompositionResult(interval_copies=",sum(r.multiplicities),
    ", residual_dimension=",sum(r.residual_dims),", exhaustive=",r.exhaustive,
    ", witnesses=",r.maps !== nothing,")")
Base.show(io::IO,::MIME"text/plain",r::IntervalDecompositionResult) = show(io,r)
provenance(r::IntervalDecompositionResult) = (
    category=:finite_poset_representations, base_poset=source_poset(r),
    field=r.source.field, model=:hom_composition_pairing,
    candidate_pruning=:top_socle_degree_zero, exhaustive=r.exhaustive,
    ambient_identification=:not_asserted)

"""Validation report with `valid`, `issues`, `multiplicities_checked`, and `splitting_checked`."""
struct IntervalDecompositionValidationSummary{R}
    report::R
end
Base.getproperty(r::IntervalDecompositionValidationSummary,s::Symbol) =
    s === :report ? getfield(r,:report) : getproperty(getfield(r,:report),s)
Base.propertynames(r::IntervalDecompositionValidationSummary,private::Bool=false) =
    private ? (:report,propertynames(r.report)...) : propertynames(r.report)
Base.:(==)(a::IntervalDecompositionValidationSummary,b::IntervalDecompositionValidationSummary) = a.report == b.report
Base.isequal(a::IntervalDecompositionValidationSummary,b::IntervalDecompositionValidationSummary) = isequal(a.report,b.report)
Base.hash(r::IntervalDecompositionValidationSummary,h::UInt) = hash(r.report,h)
describe(r::IntervalDecompositionValidationSummary) = r.report
interval_decomposition_summary(r::IntervalDecompositionValidationSummary) = r.report
Base.show(io::IO,r::IntervalDecompositionValidationSummary) = print(io,
    "IntervalDecompositionValidationSummary(valid=",r.valid,", issues=",r.issues,")")

function _check_maps(r)
    w = r.maps
    M,S,R = r.source,w.summand,w.residual
    R.dims == r.residual_dims || error("residual dimensions disagree with the retained module")
    expected = zeros(Int,length(M.dims))
    for (I,m) in zip(r.supports,r.multiplicities); expected[I] .+= m; end
    S.dims == expected || error("summand dimensions disagree with interval multiplicities")
    # Dimensions alone do not identify S: check its prescribed interval blocks.
    target = isempty(r.supports) ? zero_pmodule(M.Q;field=M.field) :
        reduce(direct_sum,[_interval_module(M,I,m) for (I,m) in zip(r.supports,r.multiplicities)])
    S.Q === M.Q && S.field == M.field || error("summand base/field mismatch")
    check_module_functoriality(S;throw=true)
    check_module_functoriality(R;throw=true)
    for (u,v) in cover_edges(M.Q)
        S.edge_maps[u,v] == target.edge_maps[u,v] || error("summand does not have the declared interval blocks")
    end
    for (f,dom,cod) in ((w.inclusion,S,M),(w.projection,M,S),
                        (w.residual_inclusion,R,M),(w.residual_projection,M,R))
        f.dom === dom && f.cod === cod || error("splitting map has the wrong domain or codomain")
        check_morphism(f;throw=true)
    end
    for v in eachindex(M.dims)
        i,p,j,q = w.inclusion.comps[v],w.projection.comps[v],
                  w.residual_inclusion.comps[v],w.residual_projection.comps[v]
        p*i == eye(M.field,S.dims[v]) || error("projection * inclusion is not identity")
        q*j == eye(M.field,R.dims[v]) || error("residual projection * inclusion is not identity")
        all(iszero,p*j) && all(iszero,q*i) || error("mixed splitting composites do not vanish")
        i*p+j*q == eye(M.field,M.dims[v]) || error("splitting does not reconstruct the source identity")
    end
end

"""
    check_interval_decomposition(result; budget=IntervalDecompositionBudget(), throw=false)

Recompute multiplicities and search scope against the current source module,
check residual dimensions, and verify all naturality/splitting/reconstruction
identities when maps are retained. This is an exact, potentially expensive
validation, not the cheap summary. Malformed result data become issues;
`throw=true` raises `ArgumentError` for an invalid report (including exhausted
validation budgets). Retained modules/maps are by reference and must not be
mutated during queries.
"""
function check_interval_decomposition(r::IntervalDecompositionResult;
        budget::IntervalDecompositionBudget=IntervalDecompositionBudget(),throw::Bool=false)
    issues = String[]
    multiplicities_checked = splitting_checked = false
    try
        r.exhaustive == (r.requested === nothing) || error("search scope and requested family disagree")
        expected = interval_decomposition(r.source;
            family=r.exhaustive ? :all : r.requested,budget)
        r.supports == expected.supports || error("positive interval supports disagree with recomputation")
        r.multiplicities == expected.multiplicities || error("multiplicities disagree with recomputation")
        r.residual_dims == expected.residual_dims || error("residual dimensions disagree with recomputation")
        r.statistics == expected.statistics || error("search statistics disagree with recomputation")
        multiplicities_checked = true
        if r.maps !== nothing
            _check_maps(r)
            splitting_checked = true
        end
    catch err
        err isa InterruptException && rethrow()
        push!(issues,sprint(showerror,err))
    end
    result = IntervalDecompositionValidationSummary((;valid=isempty(issues),issues,
        multiplicities_checked,splitting_checked))
    throw && !result.valid && Base.throw(ArgumentError(join(issues,"; ")))
    return result
end
