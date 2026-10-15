# For rank invariants the diagonal box controls the landscape: every smaller
# coordinatewise interval factors through it. Critical cuts give exact suprema,
# including open endpoints and isolated singleton strata.
"""
    jordan_landscape(f; at, layer=1, power=0, budget=JordanBudget())

Return the exact supremal diagonal radius with rank at least `layer` in degree
`power`. Real radii return a rational supremum, even when it is not attained;
integer-domain radii return an integer maximum. An unbounded value is Inf.
No sampling window or interpolation is used.
"""
function jordan_landscape(f::JordanFiltration;at,layer=1,power=0,budget::JordanBudget=JordanBudget())
    x=_at(f,at);p=_power(f,power)
    layer isa Integer && !(layer isa Bool) && layer>=1 || throw(ArgumentError("layer must be a positive integer"))
    qualifies(t)=jordan_filtered_rank(f;source=x.-t,target=x.+t,power=p,budget)>=layer
    qualifies(QQ(0)) || return QQ(0)
    cuts=encoding_axes(f;budget)
    events=sort!(unique([QQ(0);[abs(c-x[i]) for i in eachindex(cuts) for c in cuts[i]]]))
    _bound(length(events),budget.max_events,"landscape events")
    if f.source.domain===:integer
        hi=BigInt(last(events))+1
        qualifies(QQ(hi)) && return Inf
        lo=big(0)
        while hi-lo>1
            mid=div(lo+hi,2)
            if qualifies(QQ(mid));lo=mid;else;hi=mid;end
        end
        return QQ(lo)
    end
    qualifies(last(events)+1) && return Inf
    # The first failing open interval determines the supremum. A rank present
    # only at an event changes attainment, not the value of the supremum.
    lo=1;hi=length(events)
    while lo<hi
        mid=div(lo+hi,2);t=(events[mid]+events[mid+1])/2
        if qualifies(t);lo=mid+1;else;hi=mid;end
    end
    events[lo]
end
function _comparison(f,g)
    f.source.domain==g.source.domain && length(f.source.cuts)==length(g.source.cuts) ||
        throw(ArgumentError("Jordan comparison domains must agree"))
    f.slices==g.slices || throw(ArgumentError("Compare the same ordered geometric slice family"))
    field(f)==field(g) || throw(ArgumentError("Jordan comparison coefficient fields must agree"))
end
function _epsilon(e,domain)
    e isa Union{Integer,Rational,AbstractFloat} && !(e isa Bool) && isfinite(e) && e>=0 ||
        throw(ArgumentError("epsilon must be a finite nonnegative scalar"))
    t=QQ(e)
    domain===:integer && !isinteger(t) && throw(ArgumentError("Integer erosion requires an integral epsilon"))
    t
end
"""
    jordan_erosion_test(f, g; epsilon, budget=JordanBudget())

Decide both erosion inequalities for all parameter pairs and all Jordan degrees.
The common refinement includes original cuts and their positive/negative epsilon
translations. It retains singleton and exterior strata, so this is a global
finite decision, not a check on a sample. Resource exhaustion throws.
"""
function jordan_erosion_test(f::JordanFiltration,g::JordanFiltration;epsilon,budget::JordanBudget=JordanBudget())
    _comparison(f,g);e=_epsilon(epsilon,f.source.domain)
    a=encoding_axes(f;budget);b=encoding_axes(g;budget)
    cuts=ntuple(i->sort!(unique([a[i];b[i];a[i].+e;a[i].-e;b[i].+e;b[i].-e])),length(a))
    _bound(sum(length,cuts),budget.max_events,"erosion cuts")
    Q,points=_parameter_grid(cuts,f.source.domain,budget)
    _bound(prod(div(big(m)*(m+1),2) for m in Q.sizes),budget.max_pairs,"erosion parameter pairs")
    for u in eachindex(points),v in eachindex(points)
        leq(Q,u,v) || continue
        x=points[u];y=points[v]
        fr=jordan_filtered_rank(f;source=x,target=y,budget)
        gr=jordan_filtered_rank(g;source=x,target=y,budget)
        fe=iszero(e) ? fr : jordan_filtered_rank(f;source=x.-e,target=y.+e,budget)
        ge=iszero(e) ? gr : jordan_filtered_rank(g;source=x.-e,target=y.+e,budget)
        all(fe .<= gr) && all(ge .<= fr) || return false
    end
    true
end
"""
    jordan_erosion_distance(f, g; budget=JordanBudget())

Exact global erosion distance for the full Jordan-filtered rank invariant.
Return a rational infimum or Inf. On real parameters the infimum need not be
attained; `jordan_erosion_test(...; epsilon=d)` checks attainment explicitly.
Integer parameters return the least admissible integer epsilon.

The decision changes only when translated cuts cross: epsilon is a difference
of two cuts or half such a difference. Open epsilon intervals are checked as
well as their boundaries. Infinite tails of the input modules remain present.
"""
function jordan_erosion_distance(f::JordanFiltration,g::JordanFiltration;budget::JordanBudget=JordanBudget())
    _comparison(f,g);a=encoding_axes(f;budget);b=encoding_axes(g;budget)
    cuts=ntuple(i->sort!(unique([a[i];b[i]])),length(a))
    _bound(sum(big(length(c))^2 for c in cuts;init=big(0)),budget.max_events,"erosion event pairs")
    events=Set{QQ}([QQ(0)])
    for c in cuts,x in c,y in c
        d=abs(x-y)
        for e in (d,d/2)
            if f.source.domain===:integer
                push!(events,QQ(ceil(BigInt,e)),QQ(floor(BigInt,e)+1))
            else
                push!(events,e)
            end
        end
    end
    _bound(length(events),budget.max_events,"erosion events")
    ts=sort!(collect(events));n=length(ts)
    probe(i)=f.source.domain===:integer ? ts[i] : i==n ? ts[i]+1 : (ts[i]+ts[i+1])/2
    test(i)=jordan_erosion_test(f,g;epsilon=probe(i),budget)
    test(n) || return Inf
    lo=1;hi=n
    while lo<hi
        mid=div(lo+hi,2)
        if test(mid);hi=mid;else;lo=mid+1;end
    end
    ts[lo]
end

function _stratum_index(cuts,domain,x)
    label=1;stride=1
    for i in eachindex(cuts)
        c=cuts[i];j=searchsortedlast(c,x[i])
        k=domain===:integer ? j+1 : 2j+1-(j>0 && c[j]==x[i])
        label+=(k-1)*stride
        stride*=domain===:integer ? length(c)+1 : 2length(c)+1
    end
    label
end
function _lift_components(f,g,at,delta,p,components,cuts,budget)
    x=_at(f,at);y=x.+delta
    a=_image_data(f,x,p,budget);b=_image_data(g,y,p,budget);K=coeff_type(field(f))
    maps=Matrix{K}[]
    _bound(sum(big(a.data.dims[i])*b.data.dims[i] for i in eachindex(a.data.dims)),budget.max_entries,"lifted transport entries")
    for slice in f.slices
        push!(maps,_diagonal([components[_stratum_index(cuts,f.source.domain,x.+z)] for z in slice],K))
    end
    _image_map(f,a,b,maps,budget)
end
"""
    jordan_interleaving(certificate; slices, power, budget=JordanBudget())

Transport a checked ambient interleaving to the selected Jordan image degree.
The returned certificate contains actual induced maps and is checked by the
independent interleaving equations. Both sides use the same geometric pattern.
For a diagonal epsilon this supplies the interleaving bound behind erosion and
landscape stability (Chindris, Kang and Kline, arXiv:2510.22116, Theorem 28).
Bare finite-poset certificates have no ambient translation and are rejected.
"""
function jordan_interleaving(c::InterleavingCertificate;slices,power,budget::JordanBudget=JordanBudget())
    check_interleaving(c;throw=true,budget=_ambient_budget(budget))
    meta=provenance(c)
    meta.kind===:ambient_interleaving || throw(ArgumentError("Jordan stability requires an ambient interleaving"))
    f=jordan_filtration(c.problem.left;slices,budget);g=jordan_filtration(c.problem.right;slices,budget)
    p=_power(f,power)
    A=jordan_module(f;power=p,budget);B=jordan_module(g;power=p,budget)
    problem=interleaving_problem(ambient_module(A),ambient_module(B),meta.translation;budget=_ambient_budget(budget))
    cuts=provenance(problem).cuts;_,points=_parameter_grid(cuts,f.source.domain,budget)
    original=interleaving_maps(c);K=coeff_type(field(f))
    forward=Matrix{K}[];reverse=Matrix{K}[]
    for x in points
        push!(forward,_lift_components(f,g,x,meta.translation,p,original.forward.comps,meta.cuts,budget))
        push!(reverse,_lift_components(g,f,x,meta.translation,p,original.reverse.comps,meta.cuts,budget))
    end
    interleaving_certificate(problem,forward,reverse;budget=_ambient_budget(budget))
end
