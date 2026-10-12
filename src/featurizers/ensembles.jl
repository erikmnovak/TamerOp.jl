# Aligned repeated numerical invariants and finite-grid mean inference.

"""
    PersistenceEnsemble

An owned snapshot of repeated scalar invariants on the same finite Cartesian
probe grid, constructed by `persistence_ensemble`. Rows are complete replicates;
columns follow Julia's Cartesian (first coordinate fastest) order. Original
values/coordinates are retained; summaries use Float64. No module correspondence
or statistical independence is inferred. Use `sample_ids`, `feature_values`,
`ensemble_summary`, and `summarize_ensemble`; treat retained arrays as read-only.
"""
struct PersistenceEnsemble{T,A,C,R}
    values::T
    axes::A
    ids::Vector{String}
    context::C
    results::R
end
_ensemble_text(s,name)=s isa AbstractString && !isempty(strip(s)) ? String(s) :
    throw(ArgumentError("$name must be a nonempty description."))
function _ensemble_axes(axes;max_cells=1_000_000)
    axes isa Tuple && !isempty(axes) || throw(ArgumentError("axes must be a nonempty tuple of coordinate vectors."))
    all(a->a isa AbstractVector || a isa Tuple,axes) || throw(ArgumentError("Each axis must be a finite coordinate vector or tuple."))
    prod(BigInt(length(a)) for a in axes)<=max_cells || throw(ArgumentError("Probe grid exceeds max_cells."))
    ax=map(collect,axes)
    for a in ax
        !isempty(a) && all(x->x isa Real && !(x isa Bool) && isfinite(x) && isfinite(Float64(x)),a) ||
            throw(ArgumentError("Coordinates must be nonempty finite real vectors."))
        all(>(0),diff(a)) && all(>(0),diff(Float64.(a))) ||
            throw(ArgumentError("Coordinates must increase strictly, including after display conversion; rescale colliding coordinates explicitly."))
    end
    ax
end

"""
    persistence_ensemble(values; axes, sample_ids, statistic, protocol, population,
        sampling=:descriptive, units="value", axis_labels=nothing,
        degree=nothing, field="not declared", layer=nothing,
        normalization="none", essential="not applicable", max_cells=1_000_000)

Align supplied scalar values with shape `(replicates, length.(axes)...)`.
`sample_ids` are unique strings. `statistic`, `protocol`, and `population` explain
what is measured, how repeats were generated, and what they represent. Axes,
field, degree, layer, units and normalization must already agree across rows;
this call does not interpolate, normalize or identify classes. Missing values
are allowed for descriptive available-case summaries, never confidence bands.
Finite values must fit Float64; original values survive in the snapshot.

`sampling=:iid` is the caller's assertion that rows are independent identically
distributed observations from the declared population. Repeated perturbations
of one subject are not independent subjects. Default `:descriptive` makes no
population inference. Budgets reject before copying; nothing is truncated.

    persistence_ensemble(diagrams; grid, sample_ids, statistic=:landscape,
        dim=0, layer=1, essential=:error, kwargs...)

Evaluate ordinary diagrams on a common original-parameter grid. `:landscape`
uses finite-interval tents (the selected positive layer); `:betti` counts active
intervals with sublevel `[b,d)` or superlevel `(d,b]` conventions. Essential
intervals require an explicit `:drop`, or `:include` for Betti counts. Diagrams
must share coefficient field and filtration order. No matching is inferred.
Work is bounded by `max_work` (default 10 million interval/probe operations).
"""
function persistence_ensemble(values::AbstractArray;axes,sample_ids,statistic,protocol,population,
        sampling=:descriptive,units="value",axis_labels=nothing,degree=nothing,field="not declared",
        layer=nothing,normalization="none",essential="not applicable",max_cells=1_000_000)
    max_cells isa Int && max_cells>0 || throw(ArgumentError("max_cells must be a positive Int."))
    length(values)<=max_cells || throw(ArgumentError("Ensemble exceeds max_cells; subset explicitly."))
    ax=_ensemble_axes(axes;max_cells);n=size(values,1)
    n>0 && size(values)==(n,length.(ax)...) || throw(ArgumentError("Expected shape (replicates, length.(axes)...)."))
    sample_ids isa Union{AbstractVector,Tuple} && length(sample_ids)==n || throw(ArgumentError("One sample ID is required per row."))
    ids=[_ensemble_text(id,:sample_ids) for id in sample_ids]
    length(unique(ids))==n || throw(ArgumentError("Sample IDs must be unique."))
    sampling in (:descriptive,:iid) || throw(ArgumentError("sampling must be :descriptive or :iid."))
    all(x->ismissing(x) || x isa Real && !(x isa Bool) && isfinite(x) && isfinite(Float64(x)),values) ||
        throw(ArgumentError("Values must be finite real numbers or missing; infinities/NaN are not missing-data markers."))
    degree===nothing || degree isa Int && degree>=0 || throw(ArgumentError("degree must be nonnegative or nothing."))
    layer===nothing || layer isa Int && layer>0 || throw(ArgumentError("layer must be positive or nothing."))
    labels=axis_labels===nothing ? ["Parameter $i" for i in eachindex(ax)] : collect(axis_labels)
    length(labels)==length(ax) || throw(ArgumentError("One label per coordinate axis is required."))
    labels=[_ensemble_text(s,:axis_labels) for s in labels]
    length(unique(labels))==length(labels) || throw(ArgumentError("Axis labels must be distinct."))
    context=(;statistic=_ensemble_text(statistic,:statistic),protocol=_ensemble_text(protocol,:protocol),
        population=_ensemble_text(population,:population),sampling,units=_ensemble_text(units,:units),axis_labels=Tuple(labels),
        degree,field=_ensemble_text(field,:field),layer,normalization=_ensemble_text(normalization,:normalization),
        essential=_ensemble_text(essential,:essential))
    PersistenceEnsemble(reshape(copy(values),n,:),ax,ids,context,nothing)
end
function persistence_ensemble(diagrams::AbstractVector{<:OrdinaryPersistence.PersistenceDiagram};
        grid,sample_ids,statistic=:landscape,dim=0,layer=1,essential=:error,max_work=10_000_000,max_cells=1_000_000,kwargs...)
    !isempty(diagrams) || throw(ArgumentError("Supply at least one diagram."))
    statistic in (:landscape,:betti) || throw(ArgumentError("statistic must be :landscape or :betti."))
    dim isa Int && dim>=0 && layer isa Int && layer>0 || throw(ArgumentError("dim must be nonnegative and layer positive."))
    essential in (:error,:drop,:include) || throw(ArgumentError("essential must be :error, :drop or :include."))
    essential===:include && statistic!==:betti && throw(ArgumentError("Essential inclusion is supported for Betti counts only."))
    max_work isa Int && max_work>0 && max_cells isa Int && max_cells>0 || throw(ArgumentError("Budgets must be positive Ints."))
    ax=_ensemble_axes((grid,);max_cells);ts=only(ax);firstdiag=first(diagrams)
    all(d->OrdinaryPersistence.field(d)==OrdinaryPersistence.field(firstdiag) &&
        OrdinaryPersistence.filtration_order(d)==OrdinaryPersistence.filtration_order(firstdiag),diagrams) ||
        throw(ArgumentError("Ensemble diagrams must have identical fields and filtration orders."))
    n=length(diagrams);p=length(ts)
    BigInt(n)*p<=max_cells || throw(ArgumentError("Ensemble exceeds max_cells."))
    finite=[OrdinaryPersistence.finite_intervals(d;dim) for d in diagrams]
    infinite=[OrdinaryPersistence.essential_births(d;dim) for d in diagrams]
    essential===:error && any(!isempty,infinite) && throw(ArgumentError("Essential intervals require an explicit policy."))
    BigInt(p)*sum(length(finite[i])+length(infinite[i]) for i in 1:n)<=max_work || throw(ArgumentError("Diagram evaluation exceeds max_work."))
    order=OrdinaryPersistence.filtration_order(firstdiag);Y=zeros(n,p)
    for i in 1:n,(j,t) in enumerate(ts)
        if statistic===:betti
            Y[i,j]=count(bd->order===:sublevel ? bd[1]<=t<bd[2] : bd[2]<t<=bd[1],finite[i])+
                (essential===:include ? count(b->order===:sublevel ? b<=t : t<=b,infinite[i]) : 0)
        else
            heights=[max(0.,Float64(order===:sublevel ? min(t-b,d-t) : min(b-t,t-d))) for (b,d) in finite[i]]
            length(heights)>=layer && (Y[i,j]=partialsort!(heights,layer;rev=true))
        end
    end
    e=persistence_ensemble(Y;axes=ax,sample_ids,statistic=statistic===:betti ? "Ordinary Betti count" : "Ordinary persistence landscape",
        degree=dim,layer=statistic===:betti ? nothing : layer,field=string(OrdinaryPersistence.field(firstdiag)),
        essential="$essential; counts per replicate $(length.(infinite)); $order",units=statistic===:betti ? "dimension" : "parameter",max_cells,kwargs...)
    PersistenceEnsemble(e.values,e.axes,e.ids,e.context,deepcopy(diagrams))
end
sample_ids(e::PersistenceEnsemble)=copy(e.ids)
feature_values(e::PersistenceEnsemble)=copy(e.values)
feature_axes(e::PersistenceEnsemble)=deepcopy(e.axes)
ensemble_summary(e::PersistenceEnsemble)=(;kind=:persistence_ensemble,samples=length(e.ids),probes=size(e.values,2),
    shape=length.(e.axes),context=e.context,missing=count(ismissing,e.values))
describe(e::PersistenceEnsemble)=ensemble_summary(e)
Base.show(io::IO,e::PersistenceEnsemble)=print(io,"PersistenceEnsemble(",length(e.ids)," replicates, ",size(e.values,2)," probes; ",e.context.sampling,")")

"""
    EnsembleSummary

Prepared mean, median, sample standard deviation, empirical band, support
frequency and optional population-mean band. Made by `summarize_ensemble`.
`ensemble_values(summary)` returns copied arrays in Cartesian grid shape;
`ensemble_summary(summary)` describes assumptions and coverage. Summaries retain
their owned ensemble snapshot; selection and rendering perform no resampling.
"""
struct EnsembleSummary{E,D,C}
    ensemble::E
    data::D
    contract::C
end

"""
    summarize_ensemble(ensemble; band=:quantile, quantiles=(0.1,0.9),
        threshold=0, level=0.95, scope=:simultaneous, bounds=nothing,
        bootstrap_samples=1000, seed=0, max_work=50_000_000)

Compute unweighted summaries on the declared probes. `:quantile` uses linear
empirical quantiles; `:range` uses extrema. These are descriptive spread across
available observations, not confidence. `frequency` is the proportion strictly
above `threshold`; `counts` records denominators. Empty probes give NaN; sample
standard deviation is NaN below two observations.

For `:hoeffding`, require complete IID rows and known almost-sure population
`bounds=(lo,hi)` (scalars or probe-shaped arrays), declared independently of the
observed extrema. Two-sided mean half-widths are `(hi-lo)*sqrt(log(2m/alpha)/(2n))`,
with `m=number of probes` for simultaneous coverage, or `m=1` for pointwise.
The interval is intersected with known bounds. Coverage is conditional on the
caller's sampling/bounding assertions, only at these fixed probes.

`:bootstrap` resamples whole IID rows with replacement using a private seeded
RNG. Symmetric mean bands use empirical quantiles of absolute mean deviations,
or their maximum over probes for simultaneous bands. Coverage is approximate
(asymptotic finite-dimensional bootstrap), never a finite-sample certificate.
No studentization, subject clustering, adaptive probes, or missing-data inference
is supplied. `bootstrap_samples>=100`; work/storage are checked before sampling.
Neither mode supplies confidence for an entire module or between grid nodes.
"""
function summarize_ensemble(e::PersistenceEnsemble;band=:quantile,quantiles=(0.1,0.9),threshold=0,
        level=0.95,scope=:simultaneous,bounds=nothing,bootstrap_samples=1000,seed=0,max_work=50_000_000)
    band in (:quantile,:range,:hoeffding,:bootstrap) || throw(ArgumentError("Unknown band."))
    scope in (:simultaneous,:pointwise) || throw(ArgumentError("scope must be :simultaneous or :pointwise."))
    threshold isa Real && !(threshold isa Bool) && isfinite(threshold) || throw(ArgumentError("threshold must be finite."))
    quantiles isa Union{AbstractVector,Tuple} && length(quantiles)==2 && all(x->x isa Real && isfinite(x),quantiles) && 0<=quantiles[1]<=quantiles[2]<=1 || throw(ArgumentError("quantiles must be ordered probabilities."))
    level isa Real && !(level isa Bool) && 0<level<1 || throw(ArgumentError("level must be strictly between zero and one."))
    max_work isa Int && max_work>0 || throw(ArgumentError("max_work must be a positive Int."))
    n,p=size(e.values);BigInt(n)*p<=max_work || throw(ArgumentError("Summary exceeds max_work."))
    inference=band in (:hoeffding,:bootstrap)
    inference && !(e.context.sampling===:iid && n>=2 && !any(ismissing,e.values)) &&
        throw(ArgumentError("Mean inference requires at least two complete IID replicates on fixed probes."))
    band===:hoeffding || bounds===nothing || throw(ArgumentError("bounds apply only to :hoeffding."))
    avg=fill(NaN,p);med=copy(avg);sd=copy(avg);lo=copy(avg);hi=copy(avg);freq=copy(avg);counts=zeros(Int,p)
    for j in 1:p
        col=Float64[x for x in view(e.values,:,j) if !ismissing(x)];counts[j]=length(col)
        isempty(col) && continue
        constant=all(==(first(col)),col)
        avg[j]=constant ? first(col) : Statistics.mean(col);med[j]=Statistics.median(col)
        length(col)>1 && (sd[j]=constant ? 0. : Statistics.std(col))
        freq[j]=count(x->!ismissing(x) && x>threshold,view(e.values,:,j))/length(col)
        lo[j],hi[j]=band===:range ? extrema(col) : Tuple(Statistics.quantile(col,collect(quantiles)))
    end
    bound_data=nothing
    if band===:hoeffding
        bounds isa Tuple && length(bounds)==2 || throw(ArgumentError("Supply known population bounds=(lo,hi), not observed extrema."))
        expand(x)=x isa Real ? fill(x,p) : x isa AbstractArray && size(x)==length.(e.axes) ? vec(copy(x)) :
            throw(ArgumentError("Each bound must be a scalar or have probe-grid shape."))
        raw_a,raw_b=expand(bounds[1]),expand(bounds[2])
        all(x->x isa Real && !(x isa Bool) && isfinite(x) && isfinite(Float64(x)),vcat(raw_a,raw_b)) && all(raw_a.<=raw_b) ||
            throw(ArgumentError("Population bounds must be finite and ordered."))
        all(raw_a[j]<=e.values[i,j]<=raw_b[j] for j in 1:p for i in 1:n) || throw(ArgumentError("An observation violates the declared population bounds."))
        # Conversion may widen a known bound, but must never shrink it. In
        # particular, distinct exact bounds may round to the same Float64.
        a=[Float64(x)>x ? prevfloat(Float64(x)) : Float64(x) for x in raw_a]
        b=[Float64(x)<x ? nextfloat(Float64(x)) : Float64(x) for x in raw_b]
        widths=(b.-a).*sqrt(log(2*(scope===:simultaneous ? p : 1)/(1-level))/(2n))
        for j in 1:p
            iszero(widths[j]) && a[j]<b[j] && (widths[j]=nextfloat(0.))
        end
        lo=[max(a[j],iszero(widths[j]) ? avg[j] : prevfloat(avg[j]-widths[j])) for j in 1:p]
        hi=[min(b[j],iszero(widths[j]) ? avg[j] : nextfloat(avg[j]+widths[j])) for j in 1:p]
        bound_data=(lower=a,upper=b,declared=deepcopy(bounds))
    elseif band===:bootstrap
        bootstrap_samples isa Int && bootstrap_samples>=100 || throw(ArgumentError("bootstrap_samples must be an Int >=100."))
        seed isa Int || throw(ArgumentError("seed must be an Int."))
        BigInt(bootstrap_samples)*n*p<=max_work || throw(ArgumentError("Bootstrap exceeds max_work."))
        Y=Float64.(e.values);rng=Random.Xoshiro(seed)
        deviations=zeros(bootstrap_samples,scope===:simultaneous ? 1 : p)
        for b in 1:bootstrap_samples
            rows=Random.rand(rng,1:n,n)
            for j in 1:p
                delta=abs(sum((Y[i,j]-avg[j])/n for i in rows))
                if scope===:simultaneous;deviations[b,1]=max(deviations[b,1],delta)
                else;deviations[b,j]=delta;end
            end
        end
        widths=[sort!(collect(view(deviations,:,j)))[ceil(Int,level*bootstrap_samples)] for j in axes(deviations,2)]
        scope===:simultaneous && (widths=fill(only(widths),p))
        lo=avg.-widths;hi=avg.+widths
    end
    all(x->isnan(x)||isfinite(x),vcat(avg,med,sd,lo,hi,freq)) || throw(ArgumentError("Summary overflows Float64; rescale input values explicitly."))
    contract=(;band,quantiles=Tuple(quantiles),threshold,level=inference ? Float64(level) : nothing,
        scope=inference ? scope : :descriptive,coverage=band===:hoeffding ? :conditional_finite_sample : band===:bootstrap ? :approximate : :none,
        domain=:supplied_probes_only,bounds=bound_data,bootstrap_samples=band===:bootstrap ? bootstrap_samples : 0,
        seed=band===:bootstrap ? seed : nothing,missing_policy=inference ? :reject : :available_case)
    EnsembleSummary(e,(;mean=avg,median=med,std=sd,lower=lo,upper=hi,frequency=freq,counts),contract)
end
ensemble_values(s::EnsembleSummary)=map(x->reshape(copy(x),length.(s.ensemble.axes)),s.data)
ensemble_summary(s::EnsembleSummary)=(;kind=:ensemble_summary,ensemble=ensemble_summary(s.ensemble),contract=deepcopy(s.contract))
describe(s::EnsembleSummary)=ensemble_summary(s)
Base.show(io::IO,s::EnsembleSummary)=print(io,"EnsembleSummary(",s.contract.band,", ",s.contract.scope,", ",size(s.ensemble.values,2)," probes)")

"""
    EnsembleValidationSummary

Typed result of `check_ensemble`, with `valid` and human-readable `issues`.
`describe` and `ensemble_summary` return a copied structured report.
"""
struct EnsembleValidationSummary
    kind::Symbol
    valid::Bool
    issues::Vector{String}
end
ensemble_summary(s::EnsembleValidationSummary)=(;s.kind,s.valid,issues=copy(s.issues))
describe(s::EnsembleValidationSummary)=ensemble_summary(s)
Base.show(io::IO,s::EnsembleValidationSummary)=print(io,"EnsembleValidationSummary(valid=",s.valid,", issues=",length(s.issues),")")

"""
    check_ensemble(x; throw=false)

Validate a retained ensemble or summary, returning `EnsembleValidationSummary`.
Malformed storage is reported through `valid=false` and `issues`; `throw=true`
raises an `ArgumentError` instead. This validates storage contracts, not sampling
independence or the mathematical provenance of supplied observations.
"""
function check_ensemble(e::Union{PersistenceEnsemble,EnsembleSummary};throw=false)
    issues=String[]
    try
        x=e isa EnsembleSummary ? e.ensemble : e
        persistence_ensemble(reshape(x.values,(length(x.ids),length.(x.axes)...));axes=x.axes,sample_ids=x.ids,x.context...,max_cells=length(x.values))
        if e isa EnsembleSummary
            p=size(x.values,2)
            all(v->length(v)==p,values(e.data)) || error("Summary shape disagrees with probes.")
            all(j->isnan(e.data.lower[j]) || e.data.lower[j]<=e.data.upper[j],1:p) || error("Band bounds are unordered.")
        end
    catch err
        err isa InterruptException && rethrow();push!(issues,sprint(showerror,err))
    end
    valid=isempty(issues)
    throw && !valid && Base.throw(ArgumentError(join(issues,"; ")))
    EnsembleValidationSummary(:ensemble_validation,valid,issues)
end
