# HN filtrations commute with direct sums. Verify actual splitting maps, then
# merge equal slopes; indecomposability is unnecessary for this inference.
function _checked_splitting(d,b)
    M=d.source;_input(M,b);w=ID.decomposition_maps(d);n=length(M.dims);F=M.field;K=coeff_type(F)
    count=length(w.summands)
    length(w.inclusions)==length(w.projections)==count || throw(ArgumentError("splitting map counts disagree"))
    total=[zeros(K,v,v) for v in M.dims]
    for i in 1:count
        N=w.summands[i];inc=w.inclusions[i];proj=w.projections[i]
        N.Q===M.Q && N.field==F || throw(ArgumentError("splitting uses a different finite category or coefficient field"))
        inc.dom===N && inc.cod===M && proj.dom===M && proj.cod===N || throw(ArgumentError("splitting map endpoints disagree"))
        _input(N,b)
        check_morphism(inc).valid && check_morphism(proj).valid || throw(ArgumentError("splitting maps are not natural morphisms"))
        for v in 1:n
            proj.comps[v]*inc.comps[v]==eye(F,N.dims[v]) || throw(ArgumentError("summand projection does not split inclusion"))
            total[v]+=inc.comps[v]*proj.comps[v]
        end
        for j in 1:i-1,v in 1:n
            iszero(proj.comps[v]*w.inclusions[j].comps[v]) &&
                iszero(w.projections[j].comps[v]*inc.comps[v]) || throw(ArgumentError("cross-summand composites are nonzero"))
        end
    end
    all(total[v]==eye(F,M.dims[v]) for v in 1:n) || throw(ArgumentError("summand projectors do not sum to identity"))
    M,w
end
function _merged_dimensions(M,results,slopes,b;witnesses=false)
    dims=Vector{Int}[];previous=zeros(Int,length(M.dims));retained=big(0)
    for s in slopes
        current=zeros(Int,length(M.dims))
        for r in results
            k=findlast(>=(s),r.slopes);k===nothing && continue
            current+=r.term_dimensions[k]
        end
        retained+=_filtration_entries(M,current,previous;witnesses)
        _bound(retained,b.max_entries,"retained filtration entries")
        push!(dims,current);previous=current
    end
    dims
end
function _merged_terms(M::PModule{K},split,results,slopes) where K
    terms=PModule{K}[];inclusions=PMorphism{K}[]
    for s in slopes
        columns=[zeros(K,d,0) for d in M.dims]
        for (i,r) in enumerate(results)
            k=findlast(>=(s),r.slopes);k===nothing && continue
            inc=r.witnesses.inclusions[k]
            for v in eachindex(columns)
                columns[v]=hcat(columns[v],split.inclusions[i].comps[v]*inc.comps[v])
            end
        end
        T,j=_subobject(M,columns);push!(terms,T);push!(inclusions,j)
    end
    terms,inclusions
end
function _factor_maps(M::PModule{K},terms,inclusions) where K
    factors=PModule{K}[];projections=PMorphism{K}[]
    previous=[zeros(K,d,0) for d in M.dims]
    for (T,j) in zip(terms,inclusions)
        B=[Matrix{K}(FL.solve_fullcolumn(M.field,j.comps[v],previous[v])) for v in eachindex(M.dims)]
        S,i=_subobject(T,B);D,q=AC.quotient_with_projection(i)
        push!(factors,D);push!(projections,q);previous=j.comps
    end
    factors,projections
end
function _split_results(f,split,b)
    results=[];states=0;subspaces=0;matrix_trials=0;largest_blowup=0;largest_matrix=0;extension_trials=0;largest_extension=1
    for N in split.summands
        remaining=HNBudget(max_vertices=b.max_vertices,max_entries=b.max_entries,
            max_chambers=b.max_chambers,max_states=b.max_states-states,
            max_subspaces=b.max_subspaces-subspaces,max_trials=b.max_trials-matrix_trials,
            max_blowup=b.max_blowup,max_extension_degree=b.max_extension_degree)
        r=f(N,remaining);push!(results,r)
        states+=r.statistics.states;subspaces+=r.statistics.subspaces
        matrix_trials+=r.statistics.matrix_trials
        extension_trials+=r.statistics.extension_trials
        largest_extension=max(largest_extension,r.statistics.largest_extension)
        largest_blowup=max(largest_blowup,r.statistics.largest_blowup)
        largest_matrix=max(largest_matrix,r.statistics.largest_matrix)
    end
    results,(;states,subspaces,matrix_trials,largest_blowup,largest_matrix,extension_trials,largest_extension)
end
"""Reuse a retained, checked direct-sum splitting for HN computation.
Every factor is computed in its own exact field; equal slopes are merged.
Smaller summands can reduce matrix work and make the cheap anchor routes
available even when the original module has thick stalks.
"""
function hn_filtration(d::ID.IndecomposableDecompositionResult,z::StabilityWeights;
        witnesses::Bool=false,budget::HNBudget=HNBudget())
    source,split=_checked_splitting(d,budget);M=_snapshot(source)
    length(z.mass)==length(M.dims) || throw(DimensionMismatch("one stability weight is required per vertex"))
    results,stats=_split_results((N,b)->hn_filtration(N,z;witnesses,budget=b),split,budget)
    slopes=sort!(unique(reduce(vcat,[r.slopes for r in results];init=QQ[]));rev=true)
    dims=_merged_dimensions(M,results,slopes,budget;witnesses)
    maps=if witnesses
        terms,inclusions=_merged_terms(M,split,results,slopes)
        factors,factor_projections=_factor_maps(M,terms,inclusions)
        (;terms,inclusions,factors,factor_projections)
    else
        nothing
    end
    HNFiltration(M,copy(z),slopes,dims,
        maps,:supplied_splitting,stats)
end
"""Compute an anchored skyscraper filtration from a retained direct-sum
decomposition, checking the reconstruction maps before reusing its factors.
"""
function skyscraper(d::ID.IndecomposableDecompositionResult;anchor,
        weights=ones(Int,length(d.source.dims)),witnesses::Bool=false,budget::HNBudget=HNBudget())
    source,split=_checked_splitting(d,budget);a=_vertex(source,anchor);M=_snapshot(source)
    weights=_rational.(collect(weights))
    length(weights)==length(M.dims) && all(>(0),weights) || throw(ArgumentError("supply one positive mass weight per vertex"))
    results,stats=_split_results((N,b)->skyscraper(N;anchor=a,weights,witnesses,budget=b),split,budget)
    slopes=sort!(unique(reduce(vcat,[r.slopes for r in results];init=QQ[]));rev=true)
    dims=_merged_dimensions(M,results,slopes,budget;witnesses)
    K=coeff_type(M.field);bases=Matrix{K}[]
    _bound(sum(big(M.dims[a])*d[a] for d in dims;init=big(0))+
        sum(_filtration_entries(M,d,i==1 ? zeros(Int,length(d)) : dims[i-1];witnesses) for (i,d) in enumerate(dims);init=big(0)),
        budget.max_entries,"retained filtration entries")
    for s in slopes
        columns=zeros(K,M.dims[a],0)
        for (i,r) in enumerate(results)
            k=findlast(>=(s),r.slopes);k===nothing && continue
            columns=hcat(columns,split.inclusions[i].comps[a]*r.source_bases[k])
        end
        push!(bases,columns)
    end
    maps=if witnesses
        terms,inclusions=_merged_terms(M,split,results,slopes)
        factors,factor_projections=_factor_maps(M,terms,inclusions)
        (;terms,inclusions,factors,factor_projections)
    else
        nothing
    end
    SkyscraperResult(M,a,copy(weights),slopes,bases,dims,maps,
        (;category=:finite_poset,measure=:declared_stalk_weights,anchor=a,route=:supplied_splitting),stats)
end
