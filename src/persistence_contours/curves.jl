# Exact breakpoints come from changes in generator dominance or death. A
# sampled request retains exact values without silently interpolating a curve.
function _event(p,q,v)
    t=zero(QQ)
    for a in 1:2
        if v[a]==0
            p[a]<=q[a] || return nothing
        else
            t=max(t,(p[a]-q[a])/v[a])
        end
    end
    t
end
function _events(I,c,b)
    n=length(I.generators);corners=vcat(I.generators,I.relations)
    _limit(big(n)*length(corners),b.max_work,"event pairs")
    events=Set{QQ}([0])
    for p in corners,q in I.generators
        t=_event(p,q,c.direction);t===nothing && continue
        push!(events,t);_limit(length(events),b.max_events,"distinct events")
    end
    sort!(collect(events))
end
"""Compute the exact right-continuous stable-rank curve for a declared linear
contour and one bivariate interval. Without `scales`, all breakpoints are
certified. Supplied scales produce exact samples only. General direct sums
cannot be evaluated by adding these curves.
"""
function stable_rank(I::BivariateInterval;contour::LinearContour=LinearContour((1,1)),
        scales=nothing,witnesses::Bool=false,budget::ContourBudget=ContourBudget())
    I=BivariateInterval(I.generators,I.relations;field=I.field,budget)
    ts=scales===nothing ? _events(I,contour,budget) : _coordinate.(collect(scales))
    !isempty(ts) && first(ts)>=0 && issorted(ts) && allunique(ts) || throw(ArgumentError("scales must be nonempty, strictly increasing and nonnegative"))
    _limit(length(ts),budget.max_events,"query scales")
    _limit(big(length(ts))*(3length(I.generators)+length(I.relations)),budget.max_work,"curve sweep work")
    times=QQ[];values=Int[];proofs=witnesses ? ShiftDimensionResult[] : nothing
    for t in ts
        r=_shift_sweep(I,t.*contour.direction,budget,witnesses)
        if scales!==nothing || isempty(values) || r.value!=last(values)
            push!(times,t);push!(values,r.value);witnesses && push!(proofs,r)
        end
    end
    StableRankCurve(I,contour,times,values,proofs,scales===nothing ? :all_scales : :samples,
        (;events_evaluated=length(ts),generators=length(I.generators),killing_corners=length(I.relations)))
end
function stable_rank(::AbstractVector;kwargs...)
    throw(ArgumentError("stable rank is not additive: a list of intervals requires a general module solver"))
end
stable_rank(M::PModule;axes,budget::ContourBudget=ContourBudget(),kwargs...)=
    stable_rank(BivariateInterval(M;axes,budget);budget,kwargs...)
shift_dimension(M::PModule;axes,budget::ContourBudget=ContourBudget(),kwargs...)=
    shift_dimension(BivariateInterval(M;axes,budget);budget,kwargs...)
function check_stable_rank(r::StableRankCurve;budget::ContourBudget=ContourBudget(),throw=false)
    _validation(;throw) do
        r.scope in (:all_scales,:samples) || error("unknown curve scope")
        expected=stable_rank(r.interval;contour=r.contour,scales=r.scope===:all_scales ? nothing : r.scales,budget)
        expected.scales==r.scales && expected.values==r.values || error("incorrect curve values or breakpoints")
        if r.witnesses!==nothing
            length(r.witnesses)==length(r.values) || error("witness count mismatch")
            for (i,w) in enumerate(r.witnesses)
                w.interval.generators==r.interval.generators && w.interval.relations==r.interval.relations &&
                    w.interval.field==r.interval.field && w.shift==r.scales[i].*r.contour.direction && w.value==r.values[i] || error("curve witness describes a different query")
                check_shift_dimension(w;budget,throw=true)
            end
        end
        true
    end
end
