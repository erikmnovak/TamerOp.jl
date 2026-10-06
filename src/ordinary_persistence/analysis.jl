# Degree-aware adapters; all matching and feature mathematics stays in SliceInvariants.

_analysis_coordinate(::Type{E}, x::Real) where {E} = E(x)
# Julia's rational conversion/rationalize can round BigFloat intermediates at
# ambient precision. Read its integer significand and binary exponent directly,
# as Base does for exact numerical comparison, without global precision changes.
function _analysis_coordinate(::Type{E}, x::BigFloat) where {E}
    significand, exponent, sign = Base.decompose(x)
    value = exponent >= 0 ? ((sign*significand)<<exponent)//big(1) :
                            (sign*significand)//(big(1)<<(-exponent))
    return E(value)
end

"""
    analysis_barcode(diagram; dim, essential=:error, essential_cap=nothing, origin=0, scale=1)

Return an independent interval vector for analysis in degree `dim`, retaining
multiplicity by repetition and stored finite-member order, then essential-member
order. Analysis time is `(grade-origin)/scale` for sublevels and
`(origin-grade)/scale` for superlevels; `scale` must be positive. Numerical feature
grids use these increasing coordinates. Integer, rational and floating inputs
are represented by exact rationals; algebraic inputs remain algebraic.

Essential-bar policies: `:error` rejects them, `:drop` explicitly omits them,
`:keep` gives death `Inf` for distance calculations, and `:cap` replaces only
essential deaths by `essential_cap` in ORIGINAL grade units. The cap must lie
strictly after every essential birth in filtration order. It does not truncate
finite intervals. No representatives or provenance are attached to this vector;
retain the diagram for those. Distances default to `:keep`; finite features
default to `:error` and require dropping or capping essential bars explicitly.
"""
function analysis_barcode(diag::PersistenceDiagram; dim::Integer,
        essential::Symbol=:error, essential_cap=nothing, origin::Real=0, scale::Real=1)
    check_persistence_diagram(diag; throw=true)
    essential in (:error,:drop,:keep,:cap) || throw(ArgumentError("essential must be :error, :drop, :keep, or :cap"))
    isfinite(origin) && isfinite(scale) && scale > 0 || throw(ArgumentError("origin must be finite and scale finite and positive"))
    (essential === :cap) == (essential_cap !== nothing) ||
        throw(ArgumentError("essential_cap is required exactly when essential=:cap"))
    essential_cap === nothing || (essential_cap isa Real && isfinite(essential_cap)) ||
        throw(ArgumentError("essential_cap must be a finite real grade"))
    finite = finite_intervals(diag; dim=dim)
    births = essential_births(diag; dim=dim)
    essential === :error && !isempty(births) && throw(ArgumentError("degree $dim has essential bars; choose essential=:drop, :cap, or :keep explicitly"))
    E = eltype(diag.essential_by_dim) <: AbstractVector{AlgebraicReal} ||
        origin isa AlgebraicReal || scale isa AlgebraicReal || essential_cap isa AlgebraicReal ? AlgebraicReal : QQ
    s = diag.order === :sublevel ? 1 : -1
    origin_value = _analysis_coordinate(E,origin)
    scale_value = _analysis_coordinate(E,scale)
    transform(x) = s*(_analysis_coordinate(E,x)-origin_value)/scale_value
    T = essential === :keep && !isempty(births) ? Union{E,Float64} : E
    bars = Tuple{T,T}[(transform(b),transform(d)) for (b,d) in finite]
    if essential === :cap
        cap = transform(essential_cap)
        all(b -> transform(b) < cap, births) || throw(ArgumentError("essential_cap must be strictly after all essential births in filtration order"))
        append!(bars,[(transform(b),cap) for b in births])
    elseif essential === :keep
        append!(bars,[(transform(b),Inf) for b in births])
    end
    return bars
end

function _analysis_pair(a,b,dim,essential,essential_cap,origin,scale)
    a.order === b.order || throw(ArgumentError("diagram comparison requires the same filtration order"))
    return analysis_barcode(a;dim,essential,essential_cap,origin,scale),
           analysis_barcode(b;dim,essential,essential_cap,origin,scale)
end

"""
    bottleneck_distance(a::PersistenceDiagram, b::PersistenceDiagram; dim, essential=:keep, ...)

Compare the same homological degree in increasing filtration coordinates.
Supports the coordinate and essential policies of `analysis_barcode`; both
diagrams must have the same order. Reuses the exact-cost matching engine; only
the returned scalar is converted to Float64. Field labels remain on the inputs;
this metric compares their interval multisets, not their retained representatives.
"""
function bottleneck_distance(a::PersistenceDiagram,b::PersistenceDiagram; dim::Integer,
        essential::Symbol=:keep,essential_cap=nothing,origin::Real=0,scale::Real=1,backend::Symbol=:auto)
    A,B = _analysis_pair(a,b,dim,essential,essential_cap,origin,scale)
    return bottleneck_distance(A,B;backend)
end

"""
    bottleneck_matching(a::PersistenceDiagram, b::PersistenceDiagram; dim, essential=:keep, ...)

Return the shared engine's matching and transformed point vectors. Indices refer
to each diagram's finite members followed by included essential members; repeated
intervals retain distinct indices. Zero denotes the diagonal, not a source-cell
correspondence. Coordinate and essential policies match `analysis_barcode`.
"""
function bottleneck_matching(a::PersistenceDiagram,b::PersistenceDiagram; dim::Integer,
        essential::Symbol=:keep,essential_cap=nothing,origin::Real=0,scale::Real=1,backend::Symbol=:auto)
    A,B = _analysis_pair(a,b,dim,essential,essential_cap,origin,scale)
    return bottleneck_matching(A,B;backend)
end

"""
    wasserstein_distance(a::PersistenceDiagram, b::PersistenceDiagram; dim, p=2, q=Inf, ...)

Numerical p-Wasserstein distance with ground norm q in (1,2,Inf). Essential bars
are matched by sorted births, cannot match the diagonal, and unequal essential
counts give Inf. Finite bars use the existing assignment engine. Endpoint
differences are formed before Float64 conversion; extreme scales that overflow
or underflow costs are rejected with a request to rescale. `p=Inf` means the
bottleneck metric and requires `q=Inf`. Defaults to keeping essential bars.
"""
function wasserstein_distance(a::PersistenceDiagram,b::PersistenceDiagram; dim::Integer,
        essential::Symbol=:keep,essential_cap=nothing,origin::Real=0,scale::Real=1,
        p::Real=2,q::Real=Inf,backend::Symbol=:auto)
    p >= 1 && !isnan(p) || throw(ArgumentError("p must be >= 1"))
    q in (1,2,Inf) || throw(ArgumentError("q must be 1, 2, or Inf"))
    backend in (:auto,:hungarian,:auction) || throw(ArgumentError("invalid Wasserstein backend"))
    A,B = _analysis_pair(a,b,dim,essential,essential_cap,origin,scale)
    return wasserstein_distance(A,B;p,q,backend)
end

function _analysis_finite(diag,dim,essential,essential_cap,origin,scale)
    bars = analysis_barcode(diag;dim,essential,essential_cap,origin,scale)
    all(x -> isfinite(x[2]),bars) || throw(ArgumentError("this feature requires finite bars; use essential=:drop or :cap"))
    return bars
end
function _analysis_counts(bars)
    counts = Dict{eltype(bars),Int}()
    for bar in bars; counts[bar] = get(counts,bar,0)+1; end
    return counts
end
function _analysis_grid(grid,name)
    values = collect(grid)
    !isempty(values) && all(x -> x isa Real && isfinite(x),values) || throw(ArgumentError("$name must be a nonempty finite real grid"))
    floats = Float64.(values)
    all(isfinite,floats) && all(diff(floats) .> 0) ||
        throw(ArgumentError("$name must remain strictly increasing and finite in Float64; rescale exact coordinates if needed"))
    return floats
end
function _analysis_tgrid(bars,tgrid,nsteps)
    if tgrid !== nothing
        tg = _analysis_grid(tgrid,"tgrid")
        length(tg) >= 2 || throw(ArgumentError("tgrid requires at least two points"))
        return tg
    end
    nsteps >= 2 || throw(ArgumentError("nsteps must be at least two"))
    isempty(bars) && return collect(range(0.0,1.0;length=nsteps))
    a = minimum(first,bars); b = maximum(last,bars)
    return _analysis_grid(range(Float64(a),Float64(b);length=nsteps),"automatic tgrid")
end

"""
    persistence_landscape(diag::PersistenceDiagram; dim, essential=:error, kmax=5, tgrid=nothing, nsteps=401, ...)

Sample the degree's landscape through the shared barcode algorithm. Coordinates
and essential policy follow `analysis_barcode`; grids and returned values use
Float64. Explicit grids must be strictly increasing with at least two points.
An automatic grid that collapses exact endpoints is rejected; use origin/scale.
"""
function persistence_landscape(diag::PersistenceDiagram;dim::Integer,essential::Symbol=:error,
        essential_cap=nothing,origin::Real=0,scale::Real=1,kmax::Int=5,tgrid=nothing,nsteps::Int=401)
    bars = _analysis_finite(diag,dim,essential,essential_cap,origin,scale)
    tg = _analysis_tgrid(bars,tgrid,nsteps)
    result = persistence_landscape(bars;kmax,tgrid=tg,nsteps)
    all(isfinite,result.values) || throw(ArgumentError("landscape values exceed Float64 range; rescale"))
    return result
end

"""
    persistence_image(diag::PersistenceDiagram; dim, essential=:error, ...)

Numerical Gaussian values at supplied grid centers (not pixel integrals), using
the shared image engine. Multiplicities are accumulated exactly before numeric
evaluation. `xgrid` and `ygrid` use analysis coordinates; other keywords are those
of the barcode method. Essential bars require explicit dropping or capping.
"""
function persistence_image(diag::PersistenceDiagram;dim::Integer,essential::Symbol=:error,
        essential_cap=nothing,origin::Real=0,scale::Real=1,xgrid=0:0.1:1,ygrid=0:0.1:1,
        sigma::Real=0.1,coords::Symbol=:birth_persistence,weighting=:persistence,p::Real=1,
        normalize::Symbol=:none,differentiable::Bool=false,threads::Bool=Threads.nthreads()>1)
    isfinite(sigma) && sigma > 0 || throw(ArgumentError("sigma must be finite and positive"))
    coords in (:birth_persistence,:birth_death,:midlife_persistence) || throw(ArgumentError("invalid image coordinates"))
    normalize in (:none,:l1,:l2,:max) || throw(ArgumentError("invalid image normalization"))
    bars = _analysis_finite(diag,dim,essential,essential_cap,origin,scale)
    result = persistence_image(_analysis_counts(bars);xgrid=_analysis_grid(xgrid,"xgrid"),
        ygrid=_analysis_grid(ygrid,"ygrid"),sigma,coords,weighting,p,normalize,differentiable,threads)
    all(isfinite,result.values) || throw(ArgumentError("image values exceed Float64 range; rescale"))
    return result
end

"""Degree-aware finite-bar silhouette on a strictly increasing numerical `tgrid`; see `analysis_barcode` for policies and coordinates."""
function persistence_silhouette(diag::PersistenceDiagram;dim::Integer,tgrid,
        essential::Symbol=:error,essential_cap=nothing,origin::Real=0,scale::Real=1,kwargs...)
    bars = _analysis_finite(diag,dim,essential,essential_cap,origin,scale)
    tg = _analysis_tgrid(bars,tgrid,2)
    result = persistence_silhouette(_analysis_counts(bars);tgrid=tg,kwargs...)
    all(isfinite,result) || throw(ArgumentError("silhouette values exceed Float64 range; rescale"))
    return result
end

"""Degree-aware persistent entropy, preserving repeated bars; essential and coordinate policies follow `analysis_barcode`."""
function barcode_entropy(diag::PersistenceDiagram;dim::Integer,essential::Symbol=:error,
        essential_cap=nothing,origin::Real=0,scale::Real=1,base::Real=exp(1),kwargs...)
    isfinite(base) && base > 0 && base != 1 || throw(ArgumentError("entropy base must be positive, finite and different from one"))
    bars = _analysis_finite(diag,dim,essential,essential_cap,origin,scale)
    value = barcode_entropy(_analysis_counts(bars);base,kwargs...)
    isfinite(value) || throw(ArgumentError("entropy computation exceeds Float64 range; rescale"))
    return value
end

"""Degree-aware numerical barcode statistics; use `persistence_diagram_summary` for counts/provenance without essential-bar removal."""
function barcode_summary(diag::PersistenceDiagram;dim::Integer,essential::Symbol=:error,
        essential_cap=nothing,origin::Real=0,scale::Real=1,normalize_entropy::Bool=true)
    bars = _analysis_finite(diag,dim,essential,essential_cap,origin,scale)
    result = barcode_summary(_analysis_counts(bars);normalize_entropy)
    all(isfinite,values(result)) || throw(ArgumentError("barcode statistics exceed Float64 range; rescale"))
    return result
end
