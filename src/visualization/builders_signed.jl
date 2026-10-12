# Signed observables: exact contribution selection precedes display conversion.
# Rectangle endpoint pairs and cumulative point atoms have different inversions.
const _SignedViewInput = Union{RectSignedBarcode{2},PointSignedMeasure{2},PointSignedMeasure{4},Invariants.IntervalRankSummary}

_visual_request_keywords(::_SignedViewInput,kind::Symbol) = kind in (:signed_cancellation,:signed_prominence,:signed_diagonals) ?
    (:query,:observable,:boundary,:direct,:term,:max_terms,:smoothing,:window,:units,:background) : ()
_visual_request_cost(::_SignedViewInput,::Symbol) =
    (;work=:scan_retained_signed_terms,direct_query=:optional_one_per_query,all_pairs=false,timing=:not_measured)
available_visuals(::PointSignedMeasure{4}) = (:signed_cancellation,:signed_diagonals,:signed_prominence)
available_visuals(::Invariants.IntervalRankSummary) = (:signed_cancellation,)

_signed_integer(x,name;lower=0) = x isa Integer && !(x isa Bool) && x >= lower ? Int(x) :
    throw(ArgumentError("$name must be an integer >= $lower."))
_signed_exact(x::Integer) = big(x)//big(1)
_signed_exact(x::Rational) = Rational{BigInt}(x)
_signed_exact(x::AbstractFloat) = isfinite(x) ? Rational{BigInt}(x) : x
_signed_exact(x::Real) = x
function _signed_pair(x,name;finite=true)
    (x isa Tuple || x isa AbstractVector) && length(x)==2 &&
        all(v->v isa Real && !(v isa Bool) && !isnan(v) && (!finite || isfinite(v)),x) ||
        throw(ArgumentError("$name must contain two $(finite ? "finite " : "")real coordinates."))
    return Tuple(_signed_exact(v) for v in x)
end
_signed_leq(a,b) = all(a[k]<=b[k] for k in 1:2)
_signed_role(w) = _VisualRole(w>0 ? :positive : w<0 ? :negative : :zero)
_signed_label(x) = x isa Rational && denominator(x)==1 ? string(numerator(x)) : string(x)
_signed_pair_label(p) = "("*join(_signed_label.(p),", ")*")"

function _signed_payload(obj::RectSignedBarcode{2};observable=nothing,boundary=nothing)
    SignedMeasures.check_rect_signed_barcode(obj;throw=true)
    observable in (nothing,:rank) || throw(ArgumentError("A rectangle barcode reconstructs pair ranks."))
    boundary in (nothing,:closed) || throw(ArgumentError("RectSignedBarcode has inclusive grid endpoints (:closed)."))
    records=[(;id=i,lower=_signed_pair(r.lo,"lower"),upper=_signed_pair(r.hi,"upper"),weight=big(w))
        for (i,(r,w)) in enumerate(zip(rectangles(obj),weights(obj)))]
    return (;family=:rectangles,observable=:rank,boundary=:closed,records,axes=axes(obj),domain=:declared_grid)
end
function _signed_payload(obj::PointSignedMeasure{N};observable=nothing,boundary=nothing) where {N}
    SignedMeasures.check_point_signed_measure(obj;throw=true)
    all(w->w isa Integer && !(w isa Bool),weights(obj)) ||
        throw(ArgumentError("Signed observable inspection requires integer coefficients."))
    if N==4
        observable in (nothing,:rank) || throw(ArgumentError("Four-coordinate endpoint pairs reconstruct pair ranks."))
        boundary in (nothing,:halfopen,:closed) || throw(ArgumentError("boundary must be :halfopen or :closed."))
        bc=boundary===nothing ? :halfopen : boundary
        records=map(enumerate(points(obj))) do (i,p)
            lo,hi=_signed_pair(p[1:2],"birth"),_signed_pair(p[3:4],"death";finite=false)
            all(k->bc===:halfopen ? lo[k]<hi[k] : lo[k]<=hi[k],1:2) ||
                throw(ArgumentError("Each endpoint pair must define a nonempty rectangle."))
            (;id=i,lower=lo,upper=hi,weight=big(weights(obj)[i]))
        end
        return (;family=:rectangles,observable=:rank,boundary=bc,records,axes=axes(obj)[1:2],domain=:supplied_endpoint_measure)
    end
    observable in (nothing,:hilbert,:euler,:grid_function) || throw(ArgumentError("Point atoms need observable=:hilbert, :euler or :grid_function."))
    boundary in (nothing,:lower_orthant) || throw(ArgumentError("Point atoms use inclusive lower-orthant accumulation."))
    records=[(;id=i,point=_signed_pair(p,"atom"),weight=big(weights(obj)[i])) for (i,p) in enumerate(points(obj))]
    return (;family=:points,observable=observable===nothing ? :grid_function : observable,
        boundary=:lower_orthant,records,axes=axes(obj),domain=:declared_grid)
end
function _signed_payload(obj::Invariants.IntervalRankSummary;observable=nothing,boundary=nothing)
    observable in (nothing,:generalized_rank) || throw(ArgumentError("Interval families reconstruct generalized rank."))
    boundary in (nothing,:containment) || throw(ArgumentError("Interval families use finite-label containment."))
    family=Invariants.interval_family(obj);ws=Invariants.interval_coefficients(obj)
    length(family)==length(ws)==length(obj.ranks) && allunique(family) || throw(ArgumentError("Malformed interval summary."))
    for I in family
        !isempty(I) && issorted(I) && allunique(I) && all(v->1<=v<=nvertices(obj.Q),I) || throw(ArgumentError("Invalid interval labels."))
    end
    records=[(;id=i,vertices=I,weight=ws[i]) for (i,I) in enumerate(family)]
    return (;family=:intervals,observable=:generalized_rank,boundary=:containment,records,axes=nothing,domain=:declared_family)
end

function _signed_options(obj;query=nothing,observable=nothing,boundary=nothing,direct=nothing,
        term=nothing,max_terms=40,smoothing=(0,0),window=nothing,units=("parameter 1","parameter 2"),background=nothing)
    data=_signed_payload(obj;observable,boundary)
    budget=_signed_integer(max_terms,:max_terms;lower=1)
    selected=term===nothing ? nothing : _signed_integer(term,:term)
    selected==0 && (selected=nothing)
    selected===nothing || selected<=length(data.records) || throw(ArgumentError("term is outside the retained signed terms."))
    eps=_signed_pair(smoothing,"smoothing")
    all(>=(0),eps) || throw(ArgumentError("smoothing must be nonnegative."))
    data.family===:rectangles || all(iszero,eps) || throw(ArgumentError("Smoothing is defined here only for rectangle rank measures."))
    units isa Tuple && length(units)==2 && all(u->u isa AbstractString && !isempty(u),units) ||
        throw(ArgumentError("units must name the two parameter units."))
    data.family===:intervals && window!==nothing && throw(ArgumentError("Finite-label intervals have no parameter-plane display window."))
    win=window===nothing ? nothing : begin
        (window isa Tuple || window isa AbstractVector) && length(window)==2 || throw(ArgumentError("window must be (lower,upper)."))
        lo,hi=_signed_pair(window[1],"window lower"),_signed_pair(window[2],"window upper")
        all(lo[k]<hi[k] && isfinite(hi[k]-lo[k]) for k in 1:2) || throw(ArgumentError("The display window must have positive width."))
        (lo,hi)
    end
    q=if query===nothing
        nothing
    elseif data.family===:rectangles
        query isa NamedTuple && length(query)==2 && all(k->haskey(query,k),(:p,:q)) || throw(ArgumentError("A rank query is (p=(x,y), q=(x,y))."))
        p,q=_signed_pair(query.p,"query.p"),_signed_pair(query.q,"query.q")
        _signed_leq(p,q) || throw(ArgumentError("Rank queries require p <= q coordinatewise."))
        effective=ntuple(k->q[k]+eps[k],2)
        if data.domain===:declared_grid
            all(k->p[k] in data.axes[k] && effective[k] in data.axes[k],1:2) ||
                throw(ArgumentError("Inclusive-grid reconstruction requires p and q+smoothing on the declared axes."))
        end
        (;p,q,effective_q=effective)
    elseif data.family===:points
        point=_signed_pair(query,"query")
        all(k->point[k] in data.axes[k],1:2) || throw(ArgumentError("Cumulative queries must lie on the declared axes."))
        point
    else
        query isa NamedTuple && keys(query)==(:vertices,) || throw(ArgumentError("A family query is (vertices=[...],)."))
        Invariants.reconstruct_rank(obj;vertices=query.vertices)
        (;vertices=sort!(collect(Int,query.vertices)))
    end
    direct===nothing || q!==nothing || throw(ArgumentError("A direct comparison needs an explicit query."))
    if background!==nothing
        data.family===:intervals && throw(ArgumentError("A finite-label family has no parameter-plane background."))
        background isa NamedTuple && length(background)==5 && all(k->haskey(background,k),(:axes,:values,:observable,:source,:units)) ||
            throw(ArgumentError("background needs (axes, values, observable, source, units)."))
        background.observable in (:hilbert,:euler,:rank_diagonal) || throw(ArgumentError("Name a Hilbert, Euler or diagonal-rank background."))
        background.source isa AbstractString && !isempty(background.source) || throw(ArgumentError("Name the background's source."))
        background.units==units || throw(ArgumentError("Background and signed glyphs must use the same units."))
        background.axes==data.axes || throw(ArgumentError("Background axes must agree exactly with the declared parameter grid."))
        background.values isa AbstractMatrix && size(background.values)==(length(data.axes[2]),length(data.axes[1])) ||
            throw(ArgumentError("Background rows follow y, columns follow x."))
        all(a->!isempty(a),background.axes) || throw(ArgumentError("A sampled background needs nonempty axes."))
        length(background.values)<=10000 || throw(ArgumentError("Supply at most 10000 background samples."))
        all(x->x isa Real && isfinite(x) && isfinite(Float64(x)),background.values) || throw(ArgumentError("Background values must be finite."))
    end
    return (;data,query=q,direct,selected,budget,smoothing=eps,window=win,units,background)
end
function _append_visual_request_issues!(issues::Vector{String},obj::_SignedViewInput,kind::Symbol;kwargs...)
    kind in (:signed_cancellation,:signed_prominence,:signed_diagonals) && _signed_options(obj;kwargs...)
    return issues
end

function _signed_contributions(options)
    d,q,eps=options.data,options.query,options.smoothing
    records=map(d.records) do r
        if d.family===:rectangles
            hi=ntuple(k->r.upper[k]-eps[k],2)
            survives=all(k->d.boundary===:halfopen ? r.lower[k]<hi[k] : r.lower[k]<=hi[k],1:2)
            contributes=q!==nothing && survives && _signed_leq(r.lower,q.p) &&
                all(k->d.boundary===:halfopen ? q.q[k]<hi[k] : q.q[k]<=hi[k],1:2)
            merge(r,(;smoothed_upper=hi,survives,contributes,contribution=contributes ? r.weight : big(0)))
        else
            contributes=q!==nothing && (d.family===:points ? _signed_leq(r.point,q) : issubset(q.vertices,r.vertices))
            merge(r,(;survives=true,contributes,contribution=contributes ? r.weight : big(0)))
        end
    end
    positive=sum((r.contribution for r in records if r.contribution>0);init=big(0))
    negative=sum((r.contribution for r in records if r.contribution<0);init=big(0))
    reconstructed=q===nothing ? nothing : positive+negative
    direct_value=if options.direct===nothing
        nothing
    elseif d.family===:rectangles
        options.direct(q.p,q.effective_q)
    elseif d.family===:points
        options.direct(q)
    else
        options.direct(copy(q.vertices))
    end
    direct_value===nothing || (direct_value isa Integer && !(direct_value isa Bool)) ||
        throw(ArgumentError("The direct observable must return an integer."))
    options.direct!==nothing && direct_value===nothing && throw(ArgumentError("The direct observable returned nothing."))
    return (;records,positive,negative,reconstructed,direct_value,
        agreement=direct_value===nothing ? :not_checked : reconstructed==direct_value ? :equal : :mismatch)
end

function _signed_display_window(o,records)
    o.window===nothing || return o.window
    coords=Tuple[]
    for r in records
        if o.data.family===:points
            push!(coords,r.point)
        else
            push!(coords,r.lower)
            push!(coords,ntuple(k->isfinite(r.smoothed_upper[k]) ? r.smoothed_upper[k] : r.lower[k],2))
        end
    end
    if o.query!==nothing
        o.data.family===:points ? push!(coords,o.query) : append!(coords,[o.query.p,o.query.q])
    end
    isempty(coords) && append!(coords,[(0,0),(1,1)])
    lo=ntuple(k->minimum(p[k] for p in coords),2)
    hi=ntuple(k->maximum(p[k] for p in coords),2)
    span=ntuple(k->max(hi[k]-lo[k],1),2)
    return (ntuple(k->lo[k]-span[k]/10,2),ntuple(k->hi[k]+span[k]/5,2))
end
function _signed_draw(x)
    y=Float64(x)
    isfinite(y) || throw(ArgumentError("Coordinates or weights exceed floating display range; rescale the drawing units."))
    return y
end
_signed_xy(p)=(_signed_draw(p[1]),_signed_draw(p[2]))
function _signed_legend()
    _default_legend(visible=true,entries=(;positive=(;label="positive (+)",color=_VisualRole(:positive),style=:marker),
        negative=(;label="negative (-)",color=_VisualRole(:negative),style=:marker)))
end

function _signed_geometry(o,c,ids;diagonals=false)
    d=o.data;win=_signed_display_window(o,c.records)
    lo,hi=_signed_xy(win[1]),_signed_xy(win[2])
    all(lo[k]<hi[k] && isfinite(hi[k]-lo[k]) for k in 1:2) || throw(ArgumentError("Display coordinates collide; rescale the parameter units."))
    layers=AbstractVisualizationLayer[];targets=NamedTuple[];label_positions=NTuple{2,Float64}[]
    if o.background!==nothing
        b=o.background
        samples=[(_signed_draw(x),_signed_draw(y)) for x in b.axes[1] for y in b.axes[2]]
        push!(layers,PointLayer(samples,_VisualRole(:muted),0.5,23.))
        push!(layers,PointLayer(samples,Float64.(vec(b.values)),0.65,19.,:grays))
    end
    for id in ids
        r=c.records[id];r.survives || continue
        color=_signed_role(r.weight);alpha=o.query===nothing || r.contributes ? 0.9 : 0.25
        width=o.selected==id ? 4. : 1.7
        if d.family===:points
            p=_signed_xy(r.point)
            all(lo[k]<=p[k]<=hi[k] for k in 1:2) || continue
            push!(layers,PointLayer([p],color,alpha,o.selected==id ? 20. : 13.))
            push!(layers,TextLayer(["#$id: $(_signed_label(r.weight))"],[(p[1],p[2]+0.035*(hi[2]-lo[2]))],color,11.))
            push!(targets,(;id,point=p))
        else
            any(k->r.lower[k]>win[2][k] || r.smoothed_upper[k]<win[1][k],1:2) && continue
            a=_signed_xy(ntuple(k->clamp(r.lower[k],win[1][k],win[2][k]),2))
            b=_signed_xy(ntuple(k->clamp(r.smoothed_upper[k],win[1][k],win[2][k]),2))
            if diagonals
                # Endpoint glyphs deliberately connect clipped endpoints. Their
                # meaning is a signed rectangle, never a module structure map.
                push!(layers,SegmentLayer([(a...,b...)],color,alpha,width,r.weight<0 ? :dash : :solid))
            else
                push!(layers,RectLayer([(a...,b...)],color,color,0.10,0.))
                push!(layers,SegmentLayer([(a[1],a[2],b[1],a[2]),(a[1],a[2],a[1],b[2])],color,alpha,width))
                push!(layers,SegmentLayer([(a[1],b[2],b[1],b[2]),(b[1],a[2],b[1],b[2])],color,alpha,width,
                    d.boundary===:halfopen ? :dash : :solid))
            end
            mid=((a[1]+b[1])/2,(a[2]+b[2])/2)
            push!(layers,PointLayer([a],color,alpha,8.))
            # Move only the annotation when coincident/nearby centers would
            # hide a coefficient. The target and rectangle remain unchanged.
            pos=(min(mid[1],hi[1]-0.20*(hi[1]-lo[1])),mid[2])
            step=0.07*(hi[2]-lo[2])
            for offset in (0,1,-1,2,-2,3,-3,4,-4)
                candidate=(pos[1],clamp(mid[2]+offset*step,lo[2]+step,hi[2]-step))
                if all(p->abs(p[1]-candidate[1])>0.22*(hi[1]-lo[1]) || abs(p[2]-candidate[2])>=0.9step,label_positions)
                    pos=candidate;break
                end
            end
            push!(label_positions,pos)
            pos==mid || push!(layers,SegmentLayer([(mid...,pos...)],color,0.4,0.7))
            push!(layers,TextLayer(["#$id: $(_signed_label(r.weight))"],[pos],color,11.))
            push!(targets,(;id,point=mid))
            any(k->r.lower[k]<win[1][k],1:2) &&
                push!(layers,TextLayer(["clipped"],[a],color,10.))
            for k in 1:2
                if !isfinite(r.smoothed_upper[k]) || r.smoothed_upper[k]>win[2][k]
                    note=isfinite(r.smoothed_upper[k]) ? "clipped" : (k==1 ? "x: +inf" : "y: +inf")
                    pos=(min(b[1],hi[1]-0.22*(hi[1]-lo[1])),min(b[2],hi[2]-0.06*(hi[2]-lo[2])))
                    push!(layers,TextLayer([note],[pos],color,10.))
                    break
                end
            end
        end
    end
    if o.query!==nothing
        ps=d.family===:points ? [o.query] : [o.query.p,o.query.q]
        for (i,p) in enumerate(ps)
            all(win[1][k]<=p[k]<=win[2][k] for k in 1:2) || continue
            xy=_signed_xy(p)
            push!(layers,PointLayer([xy],_VisualRole(:selected),1.,14.))
            push!(layers,TextLayer([d.family===:points ? "query" : i==1 ? "p" : "q"],[(xy[1],xy[2]-0.04*(hi[2]-lo[2]))],_VisualRole(:foreground),12.))
        end
    end
    isempty(ids) && push!(layers,TextLayer(["No signed terms"],[((lo[1]+hi[1])/2,(lo[2]+hi[2])/2)],_VisualRole(:muted),15.))
    title=d.family===:points ? "$(d.observable) point atoms" : diagonals ? "Signed rectangle diagonals" : "Signed rectangles"
    subtitle=d.family===:points ? "Sum atoms in the query's lower orthant." : diagonals ?
        "Both endpoints retained; segments represent rectangles, not maps." :
        d.boundary===:halfopen ? "Closed lower edges; dashed open upper edges." : "Inclusive endpoints on the declared grid."
    return VisualizationSpec(:signed_geometry;title,subtitle,layers,legend=_signed_legend(),
        axes=_default_axes_2d(xlabel=o.units[1],ylabel=o.units[2],xlimits=(lo[1],hi[1]),ylimits=(lo[2],hi[2])),
        metadata=(;pick_targets=targets,window=win,background=o.background,display_clipping=true,
            point_colorbar_label=o.background===nothing ? "" : "$(o.background.observable) samples"))
end

function _signed_prominence(o,c,ids)
    finite_points=Tuple[]
    for r in c.records
        r.survives || continue
        push!(finite_points,ntuple(k->isfinite(r.smoothed_upper[k]) ?
            Int(sign(r.weight))*(r.smoothed_upper[k]-r.lower[k]) : 0,2))
    end
    bound=max(maximum((abs(x) for p in finite_points for x in p);init=0),1)
    cap=_signed_draw(bound*6/5)
    layers=AbstractVisualizationLayer[SegmentLayer([(-1.4cap,0.,1.8cap,0.),(0.,-1.4cap,0.,1.8cap)],_VisualRole(:muted),0.7,1.,:dash)]
    targets=NamedTuple[];vectors=NamedTuple[]
    for id in ids
        r=c.records[id];r.survives || continue
        v=ntuple(k->iszero(r.weight) ? 0 : Int(sign(r.weight))*(r.smoothed_upper[k]-r.lower[k]),2)
        p=ntuple(k->isfinite(v[k]) ? _signed_draw(v[k]) : Float64(sign(v[k]))*cap,2)
        push!(vectors,(;id,vector=v,multiplicity=abs(r.weight)))
        push!(layers,PointLayer([p],_signed_role(r.weight),o.query===nothing || r.contributes ? 1. : 0.3,o.selected==id ? 20. : 13.))
        push!(layers,TextLayer(["#$id x$(abs(r.weight))"*(all(isfinite,v) ? "" : " (inf)")],[(p[1],p[2]+0.06cap)],_signed_role(r.weight),11.))
        push!(targets,(;id,point=p))
    end
    isempty(vectors) && push!(layers,TextLayer(["No surviving displayed terms"],[(0.,0.3cap)],_VisualRole(:muted),14.))
    return VisualizationSpec(:signed_prominence;title="Signed prominence",subtitle="Sign times side lengths; axes are the zero-width locus. Infinity is labelled.",layers,
        axes=_default_axes_2d(xlabel="signed width ($(o.units[1]))",ylabel="signed width ($(o.units[2]))",xlimits=(-1.4cap,1.8cap),ylimits=(-1.4cap,1.8cap)),
        legend=_signed_legend(),metadata=(;vectors,pick_targets=targets))
end

function _signed_family_panel(o,c,ids)
    labels=collect(1:min(nvertices(o.data.poset),40))
    entries=[string(v in c.records[id].vertices ? 1 : 0) for id in ids,v in labels]
    layer=MatrixLayer(entries,["#$id ($(c.records[id].weight))" for id in ids],string.(labels))
    return VisualizationSpec(:signed_family;title="Declared interval supports",subtitle="Rows are signed family terms; 1 means the finite label belongs. First $(length(labels)) labels.",
        layers=AbstractVisualizationLayer[layer],metadata=(;panel_style=:matrix,matrix_corner="term / finite label",matrix_row_heading="Signed terms",
            empty_matrix_reason="the declared interval family is empty",
            matrix_size=(length(c.records),nvertices(o.data.poset)),displayed_rows=ids,displayed_columns=labels))
end

function _signed_arithmetic(o,c,ids)
    o.query===nothing && return _inspection_text_panel("Choose a query",["Select a comparable pair, grid point, or declared interval.","All coefficients remain available; drawing budgets never truncate a sum."])
    layers=AbstractVisualizationLayer[];targets=NamedTuple[]
    bound=max(maximum((abs(c.records[i].contribution) for i in ids);init=big(0)),big(1))
    width=_signed_draw(bound)*1.4
    for (row,id) in enumerate(ids)
        r=c.records[id];x=_signed_draw(r.contribution);y=Float64(row)
        push!(layers,SegmentLayer([(0.,y,x,y)],_signed_role(r.weight),1.,o.selected==id ? 5. : 3.))
        push!(layers,PointLayer([(x,y)],_signed_role(r.weight),1.,9.))
        push!(layers,TextLayer([_signed_label(r.contribution)],[(x,y+0.22)],_VisualRole(:foreground),11.))
        push!(targets,(;id,point=(x,y)))
    end
    title="$(c.positive) + ($(c.negative)) = $(c.reconstructed)"
    subtitle=c.direct_value===nothing ? "Sum over ALL terms; direct observable not supplied." :
        "Direct $(o.data.observable) = $(c.direct_value); $(c.agreement===:equal ? "equal" : "MISMATCH")."
    return VisualizationSpec(:signed_arithmetic;title,subtitle,layers,legend=_signed_legend(),
        axes=_default_axes_2d(xlabel="signed contribution",ylabel="term ID",xlimits=(-width,width),ylimits=(0.,max(1.,length(ids)+1.)),
            yticks=(Float64.(eachindex(ids)),string.(ids)),aspect=:auto),metadata=(;pick_targets=targets))
end

function _signed_spec(obj,kind;kwargs...)
    o=_signed_options(obj;kwargs...)
    obj isa Invariants.IntervalRankSummary && (o=merge(o,(;data=merge(o.data,(;poset=obj.Q)))))
    c=_signed_contributions(o)
    ids=collect(1:min(length(c.records),o.budget))
    o.selected===nothing || o.selected in ids || (ids[end]=o.selected)
    panels=VisualizationSpec[]
    if o.data.family===:intervals
        push!(panels,_signed_family_panel(o,c,ids))
    else
        push!(panels,_signed_geometry(o,c,ids;diagonals=kind===:signed_diagonals))
        o.data.family===:rectangles && push!(panels,_signed_prominence(o,c,ids))
        kind===:signed_prominence && reverse!(panels)
    end
    push!(panels,_signed_arithmetic(o,c,ids))
    d=o.data
    finite_coordinates=if d.family===:rectangles
        [x for r in c.records for x in (r.lower...,r.smoothed_upper...) if isfinite(x)]
    elseif d.family===:points
        [x for r in c.records for x in r.point]
    else
        Int[]
    end
    display_collisions=length(unique(finite_coordinates))-length(unique(_signed_draw.(finite_coordinates)))
    lines=["Observable: $(d.observable); boundary: $(d.boundary).", "Reconstruction domain: $(d.domain).",
        "$(length(ids)) / $(length(c.records)) terms displayed; every term enters the selected sum.",
        "$(count(r->!r.survives,c.records)) terms vanish at smoothing $(_signed_pair_label(o.smoothing)).",
        "Term IDs refer to this input, not matched bars across different modules.",
        "Display windows clip glyphs only; infinite endpoints and exact sums are retained."]
    o.background===nothing || push!(lines,"Background: $(o.background.observable) samples of $(o.background.source); matching axes and units.")
    display_collisions==0 || push!(lines,"Distinct exact coordinates coincide in the drawing; consult the selected exact endpoints.")
    if d.family===:rectangles
        append!(lines,["Rule: lower <= p and q $(d.boundary===:closed ? "<=" : "<") upper - smoothing.",
            "Smoothing compares the original rank at (p, q + smoothing).",
            d.domain===:declared_grid ? "Grid queries only; a continuous module extension is not inferred." :
            "The supplied endpoint measure defines the scope; no global source certificate is inferred."])
    elseif d.family===:points
        push!(lines,"Rule: sum weights of atoms <= query, coordinatewise.")
    else
        push!(lines,"Rule: sum coefficients of declared intervals containing the query.")
    end
    o.selected===nothing || begin
        r=c.records[o.selected]
        push!(lines,"Selected #$(r.id): weight $(r.weight); contribution $(o.query===nothing ? "no query" : r.contribution).")
        if d.family===:rectangles
            append!(lines,["Lower $(_signed_pair_label(r.lower)); upper $(_signed_pair_label(r.upper)).",
                "Smoothed upper $(_signed_pair_label(r.smoothed_upper)); survives: $(r.survives)."])
        elseif d.family===:points
            push!(lines,"Atom $(_signed_pair_label(r.point)).")
        else
            push!(lines,"Finite labels: $(r.vertices).")
        end
    end
    push!(lines,"Signed reconstruction is not a decomposition of the module.")
    push!(panels,_inspection_text_panel("Read the signed sum",lines))
    return VisualizationSpec(kind;title="Signed contributions and cancellation",subtitle="$(d.observable): retain the query, signs and reconstruction rule together.",panels,
        metadata=(;observable=d.observable,boundary=d.boundary,domain=d.domain,query=o.query,smoothing=o.smoothing,
            records=c.records,positive=c.positive,negative=c.negative,reconstructed=c.reconstructed,direct_value=c.direct_value,
            agreement=c.agreement,display_collisions,selected_term=o.selected,displayed_term_ids=ids,units=o.units,
            panel_columns=2,figure_size=(1450,d.family===:rectangles ? 1200 : 1050),input_ids=:local_only,scope=:supplied_signed_observable))
end
_visual_spec(obj::Union{PointSignedMeasure{4},Invariants.IntervalRankSummary},kind::Symbol;kwargs...)=_signed_spec(obj,kind;kwargs...)
