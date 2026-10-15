"""Integer-radius generalized-rank landscape on a finite occurrence/scale grid.
`values[i,j,k]` uses the layer number `layers[k]`. A true `censored[i,j,k]`
means the reported value is a lower bound at the observed grid boundary.
Arrays returned by `temporal_landscape_values` are copies.
"""
struct SpatiotemporalLandscape
    values::Array{Int,3}
    censored::BitArray{3}
    layers::Vector{Int}
    times::Vector{Rational{BigInt}}
    scales::Vector{Rational{BigInt}}
    boundary::Symbol
    queries::Int
end
temporal_landscape_values(l::SpatiotemporalLandscape)=(;values=copy(l.values),censored=copy(l.censored),layers=copy(l.layers))
describe(l::SpatiotemporalLandscape)=(;kind=:spatiotemporal_landscape,shape=size(l.values),layers=Tuple(l.layers),boundary=l.boundary,
    metric=:max_norm_on_integer_indices,radius=:nonnegative_integer,queries=l.queries,censored=count(l.censored))
Base.show(io::IO,l::SpatiotemporalLandscape)=print(io,"SpatiotemporalLandscape(",size(l.values),", ",l.boundary," boundary)")

"""
    spatiotemporal_landscape(result; layers=1:3, boundary=:censored,
                            budget=GeneralizedRankBudget())

At grid index `(i,j)`, layer `k` is the largest integer radius for which the
canonical limit-to-colimit map on the closed square has rank at least `k`.
The zero value also represents an empty admissible-radius set. The metric uses
occurrence-index and scale-index steps, not physical time and radius units.
Repeated time labels remain separate occurrences.

`boundary=:censored` reports a lower bound when a growing square reaches the
edge while its rank is still at least k. `boundary=:zero` declares the module
zero outside the finite convex grid; the next larger square then has rank zero
and every value is exact for that extension. This does not assert a zero
continuation of unobserved data. See Flammer--Hueper (2026), Sections 3--4,
https://doi.org/10.1007/s41468-026-00251-1.
"""
function spatiotemporal_landscape(r::TemporalPersistenceResult;layers=1:3,boundary=:censored,
        budget::Invariants.GeneralizedRankBudget=Invariants.GeneralizedRankBudget())
    boundary in (:censored,:zero) || throw(ArgumentError("Choose boundary=:censored or :zero."))
    ks=Int[_index(k,typemax(Int),"landscape layer") for k in layers]
    !isempty(ks) && issorted(ks) && allunique(ks) || throw(ArgumentError("Landscape layers must be nonempty, distinct and increasing."))
    n,m=length(r.times),length(r.scales)
    _bound(big(n)*m*length(ks),r.budget.max_matrix_entries,"landscape values")
    values=zeros(Int,n,m,length(ks));censored=falses(size(values))
    cache=Dict{NTuple{4,Int},Int}();limit=min(budget.max_queries,r.budget.max_landscape_queries)
    for i in 1:n,j in 1:m
        available=min(i-1,n-i,j-1,m-j)
        previous=typemax(Int)
        for radius in 0:available
            region=(i-radius,i+radius,j-radius,j+radius)
            rank_=get!(cache,region) do
                _bound(length(cache)+1,limit,"landscape rank queries")
                vertices=[_temporal_vertex(r,t,s) for s in region[3]:region[4] for t in region[1]:region[2]]
                Invariants.generalized_rank(r.module_object;vertices,budget)
            end
            rank_<=previous || error("Generalized rank increased on nested temporal squares.")
            previous=rank_
            for (layer,k) in enumerate(ks)
                if rank_>=k
                    values[i,j,layer]=radius
                    radius==available && boundary===:censored && (censored[i,j,layer]=true)
                end
            end
            rank_<first(ks) && break
        end
    end
    SpatiotemporalLandscape(values,censored,ks,copy(r.times),copy(r.scales),boundary,length(cache))
end
