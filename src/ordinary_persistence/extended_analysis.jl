# Explicit scalar projections preserve the typed source intervals and provenance.

"""
    extended_analysis(diagram; dim, kind, operation=:barcode, origin=0, scale=1, kwargs...)

Analyze one declared interval type in original scalar units. `kind` must be a
single type, never `:all`. Relative and extended-negative components reverse
scalar orientation; this is recorded in `coordinate_sign`. `operation` is
`:barcode`, `:landscape`, `:image`, `:silhouette`, `:entropy`, or `:summary`;
remaining keywords go to the corresponding ordinary analysis engine.

Returns `(value, intervals, context)`, retaining exact decorated source members,
field, orientation, coordinate transform and provenance. Cross-phase equal-value
intervals remain in `intervals` but have zero scalar persistence and contribute
zero to these scalar features. This projection is not the two-copy order metric.
"""
function extended_analysis(d::ExtendedPersistenceDiagram;dim::Integer,kind::Symbol,
        operation::Symbol=:barcode,origin::Real=0,scale::Real=1,kwargs...)
    check_extended_persistence(d;throw=true)
    kind in _EXTENDED_KINDS || throw(ArgumentError("analysis requires one explicit extended interval type"))
    operation in (:barcode,:landscape,:image,:silhouette,:entropy,:summary) || throw(ArgumentError("unknown extended analysis operation"))
    isfinite(origin) && isfinite(scale) && scale>0 || throw(ArgumentError("origin must be finite and scale positive and finite"))
    rows=extended_intervals(d;dim,kind)
    sign=(filtration_order(d)===:sublevel ? 1 : -1)*(kind in (:relative,:extended_negative) ? -1 : 1)
    E=eltype(d.filtration.values)===AlgebraicReal || origin isa AlgebraicReal || scale isa AlgebraicReal ? AlgebraicReal : QQ
    project(x)=sign*(_analysis_coordinate(E,x)-_analysis_coordinate(E,origin))/_analysis_coordinate(E,scale)
    bars=Tuple{E,E}[(project(r.birth),project(r.death)) for r in rows]
    all(bd->bd[1]<=bd[2],bars) || error("extended component orientation mismatch")
    context=(;homology=:extended,dimension=Int(dim),interval_type=kind,field=field(d),order=filtration_order(d),
        coordinate_sign=sign,origin,scale,projection=:scalar_endpoints,zero_scalar_members=count(r->r.birth==r.death,rows),
        provenance=provenance(d))
    # The ordinary engine sees only nonempty projected bars; it is never handed
    # an extended interval disguised as an essential or ordinary source class.
    projected=PersistenceDiagram([[bd for bd in bars if bd[1]<bd[2]]],[E[]];field=field(d),meta=context)
    value=if operation===:barcode
        isempty(kwargs) || throw(ArgumentError("barcode projection accepts no additional keywords"))
        bars
    elseif operation===:landscape
        persistence_landscape(projected;dim=0,kwargs...)
    elseif operation===:image
        persistence_image(projected;dim=0,kwargs...)
    elseif operation===:silhouette
        persistence_silhouette(projected;dim=0,kwargs...)
    elseif operation===:entropy
        barcode_entropy(projected;dim=0,kwargs...)
    else
        barcode_summary(projected;dim=0,kwargs...)
    end
    return (;value,intervals=rows,context)
end

"""
    extended_distance(a, b; dim, metric=:bottleneck, kind=:all, p=2, q=Inf, backend=:auto)

Compare scalar endpoint multisets without matching different interval types.
Returns the value, per-type costs, decorated input members and provenance.
This explicitly uses diagonal deletion in EACH scalar component (including
cross-phase components); it is not an interleaving distance on the two-copy
index. Equal-valued cross-phase members have zero diagonal cost. Bottleneck
combines component costs by maximum; Wasserstein combines their p-th powers.
The inputs must have the same parameter orientation and coefficient field.
"""
function extended_distance(a::ExtendedPersistenceDiagram,b::ExtendedPersistenceDiagram;
        dim::Integer,metric::Symbol=:bottleneck,kind::Symbol=:all,p::Real=2,q::Real=Inf,backend::Symbol=:auto)
    metric in (:bottleneck,:wasserstein) || throw(ArgumentError("metric must be :bottleneck or :wasserstein"))
    kind in (_EXTENDED_KINDS...,:all) || throw(ArgumentError("unknown extended interval type"))
    filtration_order(a)===filtration_order(b) && field(a)==field(b) || throw(ArgumentError("extended comparison requires the same order and field"))
    p>=1 && !isnan(p) && q in (1,2,Inf) || throw(ArgumentError("invalid Wasserstein exponents"))
    metric===:wasserstein && isinf(p) && q!=Inf && throw(ArgumentError("infinite p requires q=Inf"))
    ks=kind===:all ? _EXTENDED_KINDS : (kind,)
    components=map(ks) do k
        x=extended_analysis(a;dim,kind=k);y=extended_analysis(b;dim,kind=k)
        value=metric===:bottleneck ? bottleneck_distance(x.value,y.value;backend) : wasserstein_distance(x.value,y.value;p,q,backend)
        (;kind=k,value,left=x.intervals,right=y.intervals)
    end
    costs=getproperty.(components,:value)
    value=metric===:bottleneck || isinf(p) ? maximum(costs;init=0.0) :
        (any(isinf,costs) ? Inf : let m=maximum(costs;init=0.0);iszero(m) ? 0.0 : m*sum((c/m)^p for c in costs)^(1/p);end)
    return (;value,components,context=(;homology=:extended,metric,dimension=Int(dim),kind,p,q,
        matching=:within_interval_type,diagonal=:scalar_component,projection=:scalar_endpoints,
        left=provenance(a),right=provenance(b)))
end

"""Copy extended intervals AND the source pair into a strict portable contract.
The integer pair preserves construction maps; exact original grades and typed
decorations are separate. Restoring recomputes the barcode to validate it."""
function extended_persistence_data(d::ExtendedPersistenceDiagram{T}) where {T}
    check_extended_persistence(d;throw=true)
    f=d.filtration;G=f.pair.ambient
    return (;grade_type=T,values=copy(f.values),order=f.order,field=field(d),meta=deepcopy(f.meta),
        cells=[collect(cs) for cs in G.cells_by_dim],grades=copy(G.grades),
        relative_grades=deepcopy(f.pair.grades),
        boundaries=[(;size=size(B),entries=findnz(B)) for B in G.boundaries],
        intervals=[[describe(r) for r in rows] for rows in d.intervals])
end

"""Restore and verify `extended_persistence_data`, including a fresh reduction
of the retained pair. Serialized interval labels alone do not certify a result."""
function extended_persistence_from_data(data)
    _diagram_data_keys(data,(:grade_type,:values,:order,:field,:meta,:cells,:grades,:relative_grades,:boundaries,:intervals))
    T=data.grade_type
    T isa Type && T<:Real && T!==Bool && isconcretetype(T) && all(v->v isa T,data.values) ||
        throw(ArgumentError("invalid extended scalar type or values"))
    B=SparseMatrixCSC{Int,Int}[sparse(b.entries...,b.size...) for b in data.boundaries]
    G=GradedComplex(data.cells,B,data.grades)
    pair=filtered_pair(G,[collect(1:length(cs)) for cs in G.cells_by_dim];grades=data.relative_grades,field=data.field)
    f=ExtendedFiltration(pair,T[v for v in data.values],data.order,data.meta)
    d=extended_persistence(f)
    [[describe(r) for r in rows] for rows in d.intervals]==data.intervals || throw(ArgumentError("saved extended intervals disagree with the source pair"))
    return d
end
