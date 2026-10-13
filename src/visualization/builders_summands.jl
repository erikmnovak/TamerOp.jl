# Checked direct sums and their drawings. No decomposition algorithm lives here.
"""
    SummandCollection

A finite module, supplied summand modules and their checked inclusions and
projections, plus a checked residual. Construct with `summand_collection`.
`describe` reports the splitting and multiplicities; `check_summand_collection`
rechecks the equations. Modules and maps are retained by reference and must be
kept read-only. A supplied general piece is not asserted to be indecomposable.
"""
struct SummandCollection{M,R,I,P}
    source::M
    pieces::Vector{Modules.PModule}
    inclusions::Vector{Modules.PMorphism}
    projections::Vector{Modules.PMorphism}
    residual::R
    residual_inclusion::I
    residual_projection::P
    labels::Vector{String}
    multiplicities::Vector{Int}
    interval_supports::Vector{Vector{Int}}
    grades::Vector{NamedTuple}
    budget::IntervalDecompositions.IntervalDecompositionBudget
end

"""Exact validation of a supplied splitting; use `.valid`, `.issues`, and `describe`."""
struct SummandValidationSummary
    valid::Bool
    issues::Vector{String}
end
Base.:(==)(a::SummandValidationSummary,b::SummandValidationSummary)=a.valid==b.valid && a.issues==b.issues
Base.isequal(a::SummandValidationSummary,b::SummandValidationSummary)=isequal((a.valid,a.issues),(b.valid,b.issues))
Base.hash(r::SummandValidationSummary,h::UInt)=hash((r.valid,r.issues),h)
describe(r::SummandValidationSummary)=(;valid=r.valid,issues=copy(r.issues))
Base.show(io::IO,r::SummandValidationSummary)=print(io,"SummandValidationSummary(valid=",r.valid,", issues=",r.issues,")")

function _summand_preflight(M,pieces,budget;candidate_count=length(pieces))
    budget isa IntervalDecompositions.IntervalDecompositionBudget || throw(ArgumentError("budget must be IntervalDecompositionBudget."))
    M.field isa Union{CoreModules.QQField,CoreModules.PrimeField} || throw(ArgumentError("A certified split requires exact rational or prime-field coefficients."))
    n=nvertices(M.Q)
    n<=budget.max_vertices && BigInt(n)^2<=budget.max_matrix_entries || throw(ArgumentError("Finite base exceeds the summand validation budget."))
    all(N->N.Q===M.Q && N.field==M.field,pieces) || throw(ArgumentError("Every piece must have the identical finite base and field."))
    all(N->sum(BigInt.(N.dims))<=budget.max_dimension,[M;pieces]) || throw(ArgumentError("Stalk dimensions exceed the summand validation budget."))
    sum(BigInt(d)^2 for d in M.dims;init=big(0))<=budget.max_matrix_entries || throw(ArgumentError("Splitting matrices exceed max_matrix_entries."))
    candidate_count<=budget.max_candidates && BigInt(length(pieces))^2<=budget.max_matrix_entries || throw(ArgumentError("Too many supplied summands."))
    nothing
end

"""
    summand_collection(M, pieces; inclusions, projections, residual=nothing,
        residual_inclusion=nothing, residual_projection=nothing, labels=nothing,
        grades=nothing, budget=Advanced.IntervalDecompositionBudget())
    summand_collection(result::IntervalDecompositionResult; budget=...)

Check actual maps exhibiting `M = pieces[1] + ... + pieces[end] + residual`.
For every vertex, all projection/inclusion composites are the appropriate
identity or zero, and the idempotents sum to the identity of M. Cover naturality
and module functoriality are checked too. Omitting the residual means the zero
module, so the pieces must reconstruct all of M. Numerical fields are rejected.

The A46 overload needs `witnesses=true` and retains one block per interval type,
with its actual multiplicity. Supplied general pieces have multiplicity one;
no indecomposability or isomorphism classification is inferred. Optional
`grades` contains one `(generators=[(x,y),...], relations=[(x,y),...])` record per
piece: these are user-supplied annotations, not a verified presentation.

Use `visualize(collection)` for a static gallery or `summand_session(collection)`
for linked exploration. Session selection never reruns decomposition.
"""
function summand_collection(M::Modules.PModule,pieces;inclusions,projections,
        residual=nothing,residual_inclusion=nothing,residual_projection=nothing,
        labels=nothing,grades=nothing,budget=IntervalDecompositions.IntervalDecompositionBudget())
    ps=Modules.PModule[p for p in pieces]
    _summand_preflight(M,ps,budget)
    length(inclusions)==length(projections)==length(ps) || throw(ArgumentError("Supply one inclusion and projection per piece."))
    K=CoreModules.coeff_type(M.field)
    R=residual===nothing ? Modules.zero_pmodule(M.Q;field=M.field) : residual
    if residual===nothing
        residual_inclusion===nothing && residual_projection===nothing || throw(ArgumentError("Residual maps require a residual module."))
        ri=Modules.PMorphism{K}(R,M,[zeros(K,d,0) for d in M.dims])
        rp=Modules.PMorphism{K}(M,R,[zeros(K,0,d) for d in M.dims])
    else
        residual_inclusion isa Modules.PMorphism && residual_projection isa Modules.PMorphism || throw(ArgumentError("Supply both residual maps."))
        ri,rp=residual_inclusion,residual_projection
    end
    names=labels===nothing ? ["S$i" for i in eachindex(ps)] : String.(labels)
    length(names)==length(ps) && all(!isempty,names) && length(unique(names))==length(names) || throw(ArgumentError("labels must be distinct nonempty names, one per piece."))
    gs=grades===nothing ? NamedTuple[(;generators=(),relations=()) for _ in ps] : NamedTuple[
        (;generators=Tuple(_visual_point(p) for p in g.generators),relations=Tuple(_visual_point(p) for p in g.relations)) for g in grades]
    length(gs)==length(ps) || throw(ArgumentError("Supply one grades record per piece."))
    all(g->length(g.generators)+length(g.relations)<=budget.max_candidates,gs) || throw(ArgumentError("Too many supplied grade markers."))
    c=SummandCollection(M,ps,Modules.PMorphism[i for i in inclusions],Modules.PMorphism[p for p in projections],R,ri,rp,
        names,ones(Int,length(ps)),[Int[] for _ in ps],gs,budget)
    check_summand_collection(c;throw=true)
    c
end

"""
    summand_collection(result::IndecomposableDecompositionResult;
        budget=Advanced.IndecomposableDecompositionBudget())

Validate a certified decomposition with retained witnesses, then pass its actual
individual summands and splitting maps to the existing gallery/inspector. The
collection reports checked reconstruction; each input result separately retains
its indecomposability certificates. No support approximation is inferred.
"""
function summand_collection(r::IndecomposableDecompositions.IndecomposableDecompositionResult;
        budget=IndecomposableDecompositions.IndecomposableDecompositionBudget())
    w = IndecomposableDecompositions.decomposition_maps(r)
    IndecomposableDecompositions.check_indecomposable_decomposition(r;budget,throw=true)
    ib = IntervalDecompositions.IntervalDecompositionBudget(max_vertices=budget.max_vertices,
        max_dimension=budget.max_dimension,max_matrix_entries=budget.max_matrix_entries,
        max_candidates=budget.max_dimension)
    return summand_collection(IndecomposableDecompositions.source_module(r),w.summands;
        inclusions=w.inclusions,projections=w.projections,budget=ib)
end

function summand_collection(r::IntervalDecompositions.IntervalDecompositionResult;
        budget=IntervalDecompositions.IntervalDecompositionBudget())
    w=IntervalDecompositions.decomposition_maps(r)
    IntervalDecompositions.check_interval_decomposition(r;budget,throw=true)
    M=IntervalDecompositions.source_module(r);K=CoreModules.coeff_type(M.field)
    supports=Invariants.interval_family(r);ms=IntervalDecompositions.interval_multiplicities(r)
    offsets=zeros(Int,length(M.dims));ps=Modules.PModule[];ins=Modules.PMorphism[];outs=Modules.PMorphism[]
    for (support,m) in zip(supports,ms)
        dims=[q in support ? m : 0 for q in eachindex(M.dims)]
        ranges=[offsets[q]+1:offsets[q]+dims[q] for q in eachindex(dims)]
        S=Modules.PModule{K}(M.Q,dims,Dict(e=>Matrix{K}(w.summand.edge_maps[e[1],e[2]][ranges[e[2]],ranges[e[1]]]) for e in FiniteFringe.cover_edges(M.Q));field=M.field)
        push!(ps,S)
        push!(ins,Modules.PMorphism{K}(S,M,[Matrix{K}(w.inclusion.comps[q][:,ranges[q]]) for q in eachindex(dims)]))
        push!(outs,Modules.PMorphism{K}(M,S,[Matrix{K}(w.projection.comps[q][ranges[q],:]) for q in eachindex(dims)]))
        offsets .+= dims
    end
    c=SummandCollection(M,ps,ins,outs,w.residual,w.residual_inclusion,w.residual_projection,
        ["I$i" for i in eachindex(ps)],ms,supports,NamedTuple[(;generators=(),relations=()) for _ in ps],budget)
    check_summand_collection(c;throw=true)
    c
end

"""
    check_summand_collection(collection; throw=false)

Recheck all functoriality, naturality, retraction, orthogonality and reconstruction
equations in the retained coordinates. Invalid hand-built storage becomes a
`SummandValidationSummary`; `throw=true` raises `ArgumentError` on failure.
"""
function check_summand_collection(c::SummandCollection;throw=false)
    issues=String[]
    try
        M=c.source;ps=[c.pieces;c.residual];ins=[c.inclusions;c.residual_inclusion];outs=[c.projections;c.residual_projection]
        _summand_preflight(M,ps,c.budget;candidate_count=length(c.pieces))
        length(ps)==length(ins)==length(outs) || error("Splitting arrays have different lengths")
        length(c.labels)==length(c.pieces)==length(c.multiplicities)==length(c.interval_supports)==length(c.grades) || error("Summand metadata lengths disagree")
        all(>(0),c.multiplicities) || error("Multiplicities must be positive")
        Modules.check_module_functoriality(M;throw=true)
        for (S,i,p) in zip(ps,ins,outs)
            Modules.check_module_functoriality(S;throw=true)
            i.dom===S && i.cod===M && p.dom===M && p.cod===S || error("Inclusion/projection endpoints disagree")
            Modules.check_morphism(i;throw=true);Modules.check_morphism(p;throw=true)
        end
        for q in eachindex(M.dims)
            total=zeros(CoreModules.coeff_type(M.field),M.dims[q],M.dims[q])
            for a in eachindex(ps)
                total+=ins[a].comps[q]*outs[a].comps[q]
                for b in eachindex(ps)
                    prod=outs[a].comps[q]*ins[b].comps[q]
                    (a==b ? prod==CoreModules.eye(M.field,ps[a].dims[q]) : all(iszero,prod)) || error("Retraction or mixed zero equation fails at vertex $q")
                end
            end
            total==CoreModules.eye(M.field,M.dims[q]) || error("Idempotents do not reconstruct the identity at vertex $q")
        end
        for (j,support) in enumerate(c.interval_supports)
            isempty(support) && continue
            S=c.pieces[j];m=c.multiplicities[j]
            Invariants.check_generalized_rank_region(S;vertices=support,throw=true)
            S.dims==[q in support ? m : 0 for q in eachindex(S.dims)] || error("Interval multiplicity dimensions disagree")
            for (u,v) in FiniteFringe.cover_edges(S.Q)
                expected=u in support && v in support ? CoreModules.eye(S.field,m) : zeros(CoreModules.coeff_type(S.field),S.dims[v],S.dims[u])
                S.edge_maps[u,v]==expected || error("Declared interval coordinates are not identity blocks")
            end
        end
    catch err
        err isa InterruptException && rethrow();push!(issues,sprint(showerror,err))
    end
    r=SummandValidationSummary(isempty(issues),issues)
    throw && !r.valid && Base.throw(ArgumentError(join(issues,"; ")))
    r
end
"""
    summand_summary(collection)

Describe the checked finite splitting: field, block/copy counts, local labels,
interval multiplicities and residual dimensions. This reads retained metadata;
use `check_summand_collection` to recheck maps after any external mutation.
For example, `c = summand_collection(interval_decomposition(M; witnesses=true))`
followed by `summand_summary(c)` precedes a gallery or live `summand_session(c)`.
"""
summand_summary(c::SummandCollection)=(;kind=:checked_summands,field=c.source.field,
    category=:finite_poset_representations,blocks=length(c.pieces),copies=sum(c.multiplicities),
    labels=copy(c.labels),multiplicities=copy(c.multiplicities),residual_dimensions=copy(c.residual.dims),
    certification=:checked_splitting,indecomposability=:only_declared_interval_copies,
    geometry=:not_attached,identity=:local_to_this_split)
describe(c::SummandCollection)=summand_summary(c)
Base.show(io::IO,c::SummandCollection)=print(io,"SummandCollection(blocks=",length(c.pieces),", copies=",sum(c.multiplicities),", residual_dimension=",sum(c.residual.dims),")")

"""
    SummandSupportDrawing(boxes; provenance, error_statement="No error bound supplied.")

User-supplied support pictures, one list of `(lower_corner, upper_corner)` boxes
per summand block. Boxes use `[lo1,hi1) x [lo2,hi2)`; overlaps denote a union.
They may form staircases. Their correspondence with the algebra is unverified,
even when the splitting itself is exact. Provenance and the supplied error
statement remain visible. Use an identical-base `encoding` instead for exact
classifier fibers. No hull is inferred from sampled tracks.
"""
struct SummandSupportDrawing
    boxes::Vector{Vector{Tuple{_VisualPoint,_VisualPoint}}}
    provenance::String
    error_statement::String
    function SummandSupportDrawing(boxes;provenance,error_statement="No error bound supplied.")
        provenance isa AbstractString && !isempty(strip(provenance)) || throw(ArgumentError("Supply nonempty provenance."))
        error_statement isa AbstractString && !isempty(strip(error_statement)) || throw(ArgumentError("Supply a nonempty error statement."))
        sum(length,boxes;init=0)<=10_000 || throw(ArgumentError("At most 10,000 supplied support boxes are allowed."))
        data=[Tuple{_VisualPoint,_VisualPoint}[_visual_box_2d(b) for b in group] for group in boxes]
        new(data,String(provenance),String(error_statement))
    end
end
describe(d::SummandSupportDrawing)=(;kind=:supplied_support_drawing,blocks=length(d.boxes),boxes=sum(length,d.boxes;init=0),
    geometry=:unverified,provenance=d.provenance,error_statement=d.error_statement,boundaries=:lower_closed_upper_open)
Base.show(io::IO,d::SummandSupportDrawing)=print(io,"SummandSupportDrawing(blocks=",length(d.boxes),", geometry=unverified)")

const _SummandInput=Union{SummandCollection,IntervalDecompositions.IntervalDecompositionResult,
    IndecomposableDecompositions.IndecomposableDecompositionResult}
const _SUMMAND_OPTIONS=(:selected,:focus,:vertex,:pair,:layout,:sort_by,:min_value,:type_filter,:limit,
    :scale,:shared_scale,:hom,:hom_pair,:basis_index,:matrix_limit)
available_visuals(::_SummandInput)=(:summand_gallery,)
_visual_request_keywords(::_SummandInput,::Symbol)=(_SUMMAND_OPTIONS...,:encoding,:box,:drawing,:max_vertices,:max_summands,:max_hom_pairs,:max_hom_entries)

function _summand_prepare(c;encoding=nothing,box=nothing,drawing=nothing,max_vertices=120,max_summands=128,
        max_hom_pairs=64,max_hom_entries=1_000_000)
    for (x,name) in ((max_vertices,:max_vertices),(max_summands,:max_summands),(max_hom_pairs,:max_hom_pairs),(max_hom_entries,:max_hom_entries))
        _signed_integer(x,name;lower=1)
    end
    nvertices(c.source.Q)<=max_vertices && length(c.pieces)<=max_summands || throw(ArgumentError("Summand drawing exceeds max_vertices or max_summands."))
    check_summand_collection(c;throw=true)
    encoding!==nothing && drawing!==nothing && throw(ArgumentError("Choose exact encoding geometry or a supplied drawing."))
    geometry=nothing;pi=nothing
    if encoding!==nothing
        encoding isa EncodingResult && Results.encoding_poset(encoding)===c.source.Q || throw(ArgumentError("encoding must have the identical finite base of the split."))
        _inspection_has_geometry(encoding) || throw(ArgumentError("This classifier has no planar support geometry."))
        pi=_inspection_classifier(encoding_map(encoding))
        # Bound rectangular arrangements before constructing their subdivisions.
        if hasproperty(pi,:coords)
            prod(big(2length(a)+3) for a in pi.coords)<=100_000 || throw(ArgumentError("Classifier geometry exceeds the cell budget."))
        end
        geometry=_region_geometry_2d(pi;box)
        length(geometry.components)<=100_000 || throw(ArgumentError("Classifier geometry exceeds the cell budget."))
    elseif drawing!==nothing
        drawing isa SummandSupportDrawing && length(drawing.boxes)==length(c.pieces) || throw(ArgumentError("drawing must have one support list per summand block."))
        box===nothing || throw(ArgumentError("Supplied drawings use their complete box extent; box applies to classifier geometry."))
    else
        box===nothing || throw(ArgumentError("box requires an encoding."))
    end
    stats=[(;id=j,label=c.labels[j],multiplicity=c.multiplicities[j],
        support_vertices=count(>(0),S.dims),total_dimension=sum(S.dims),max_dimension=maximum(S.dims;init=0),
        interval=!isempty(c.interval_supports[j]),thin=maximum(S.dims;init=0)<=1,
        generators=length(c.grades[j].generators),relations=length(c.grades[j].relations)) for (j,S) in enumerate(c.pieces)]
    (;geometry,pi,drawing,hasse=_hasse_spec(c.source.Q;dims=c.source.dims),stats,
        max_hom_pairs=Int(max_hom_pairs),max_hom_entries=Int(max_hom_entries))
end

function _summand_options(c,prepared;selected=nothing,focus=nothing,vertex=nothing,pair=nothing,
        layout=:facets,sort_by=:index,min_value=0,type_filter=:all,limit=8,scale=:linear,shared_scale=true,
        hom=false,hom_pair=nothing,basis_index=1,matrix_limit=(8,8))
    n=length(c.pieces);N=length(c.source.dims)
    layout in (:facets,:overlay,:outline,:isolate) || throw(ArgumentError("layout must be :facets, :overlay, :outline or :isolate."))
    layout in (:overlay,:outline) && prepared.geometry===nothing && prepared.drawing===nothing && throw(ArgumentError("Overlay and outline need encoding geometry or supplied support boxes."))
    sort_by in (:index,:multiplicity,:support_vertices,:total_dimension,:max_dimension,:generators,:relations) || throw(ArgumentError("Unknown ranking statistic."))
    type_filter in (:all,:interval,:thin,:thick) || throw(ArgumentError("type_filter must be :all, :interval, :thin or :thick."))
    _signed_integer(min_value,:min_value);lim=_signed_integer(limit,:limit;lower=1)
    lim<=16 || throw(ArgumentError("At most 16 summand blocks can be displayed at once."))
    scale in (:linear,:log) || throw(ArgumentError("scale must be :linear or :log."))
    shared_scale isa Bool && hom isa Bool || throw(ArgumentError("shared_scale and hom must be Bool."))
    ids=selected===nothing ? collect(1:n) : Int[_signed_integer(x,:selected;lower=1) for x in selected]
    length(ids)==length(unique(ids)) && all(j->1<=j<=n,ids) || throw(ArgumentError("selected must contain distinct existing block IDs."))
    filter!(j->(type_filter===:all || type_filter===:interval && prepared.stats[j].interval ||
        type_filter===:thin && prepared.stats[j].thin || type_filter===:thick && !prepared.stats[j].thin) &&
        (sort_by===:index ? j : getproperty(prepared.stats[j],sort_by))>=min_value,ids)
    sort!(ids;by=j->sort_by===:index ? (j,j) : (-getproperty(prepared.stats[j],sort_by),j))
    resize!(ids,min(lim,length(ids)))
    f=focus===nothing ? (isempty(ids) ? 0 : first(ids)) : _signed_integer(focus,:focus)
    (f==0 || f in ids) || throw(ArgumentError("focus must be a displayed ID or 0 for no detail."))
    v=vertex===nothing ? (N==0 ? 0 : 1) : _signed_integer(vertex,:vertex)
    (N==0 ? v==0 : 1<=v<=N) || throw(ArgumentError("vertex is outside the finite base."))
    pair===nothing || (pair isa Union{Tuple,AbstractVector} && length(pair)==2 && all(q->q isa Integer && !(q isa Bool) && 1<=q<=N,pair)) || throw(ArgumentError("pair must contain two finite labels."))
    hom_pair===nothing || (hom_pair isa Union{Tuple,AbstractVector} && length(hom_pair)==2 && all(j->j isa Integer && !(j isa Bool),hom_pair)) || throw(ArgumentError("hom_pair must contain two integer block IDs."))
    hp=hom_pair===nothing ? nothing : Tuple(Int.(hom_pair))
    hp===nothing || (hom && length(hp)==2 && all(j->j in ids,hp)) || throw(ArgumentError("hom_pair needs hom=true and two displayed block IDs."))
    bi=_signed_integer(basis_index,:basis_index;lower=1)
    length(matrix_limit)==2 || throw(ArgumentError("matrix_limit requires two positive counts."))
    ml=Tuple(_signed_integer(x,:matrix_limit;lower=1) for x in matrix_limit)
    hom && BigInt(length(ids))^2>prepared.max_hom_pairs && throw(ArgumentError("Selected Hom table exceeds max_hom_pairs."))
    (;selected=selected===nothing ? nothing : copy(selected),focus=f,vertex=v,pair=pair===nothing ? nothing : Tuple(Int.(pair)),
        layout,sort_by,min_value=Int(min_value),type_filter,limit=lim,scale,shared_scale,hom,hom_pair=hp,basis_index=bi,matrix_limit=ml,ids)
end

# Quantitative colors have fixed bins; zero is distinct under both transformations.
const _SUMMAND_DIM_COLORS=(:lightcyan,:paleturquoise,:skyblue,:deepskyblue,:dodgerblue,:royalblue,:mediumblue,:midnightblue)
_summand_dim_color(d,maximum_,scale)=d==0 ? _VisualRole(:background) :
    _SUMMAND_DIM_COLORS[clamp(1+floor(Int,7*(scale===:log ? log(d)/log(max(2,maximum_)) : d/max(1,maximum_))),1,8)]
function _summand_dimension_legend(maxdim,scale)
    entries=Pair{Symbol,Any}[Symbol("0")=>_VisualRole(:background)]
    for color in _SUMMAND_DIM_COLORS
        ds=[d for d in 1:maxdim if _summand_dim_color(d,maxdim,scale)===color]
        isempty(ds) && continue
        push!(entries,Symbol(first(ds)==last(ds) ? string(first(ds)) : "$(first(ds))-$(last(ds))")=>color)
    end
    _default_legend(visible=true,entries=(;entries...))
end
function _summand_finite_panel(prepared,dims,title,o,maxdim)
    positions=prepared.hasse.metadata.positions
    layers=AbstractVisualizationLayer[layer for layer in prepared.hasse.layers if !(layer isa Union{PointLayer,TextLayer})]
    for (q,point) in enumerate(positions)
        push!(layers,PointLayer([point],_VisualRole(q==o.vertex ? :foreground : :border),1.,q==o.vertex ? 28. : 21.))
        push!(layers,PointLayer([point],_summand_dim_color(dims[q],maxdim,o.scale),1.,q==o.vertex ? 24. : 18.))
    end
    push!(layers,TextLayer(["$q: $(dims[q])" for q in eachindex(dims)],[(p[1]+0.12,p[2]+0.12) for p in positions],_VisualRole(:foreground),11.))
    targets=[(;kind=:vertex,vertex=q,point=p,label="Label $q; dimension $(dims[q])") for (q,p) in enumerate(positions)]
    isempty(positions) && push!(layers,TextLayer(["Empty finite base"],[(0.,0.)],_VisualRole(:muted),12.))
    VisualizationSpec(:summand_dimensions;title,subtitle="Finite labels; schematic order\nLabel: dimension; white = zero",layers,
        axes=prepared.hasse.axes,legend=_summand_dimension_legend(maxdim,o.scale),metadata=(;dimensions=copy(dims),pick_targets=targets,max_dimension=maxdim,scale=o.scale))
end

function _summand_geometry_background(geometry;support=false)
    layers=AbstractVisualizationLayer[]
    geometry.has_unrepresented_area || return layers
    lo,hi=_drawing_point.(geometry.box)
    push!(layers,RectLayer([(lo...,hi...)],_VisualRole(:unrepresented),_VisualRole(:unrepresented),1.,0.))
    if support
        for c in geometry.components
            c.region_id>0 && c.dimension==2 || continue
            points=_drawing_point.(c.vertices)
            _visual_polygon_dimension(unique(points))==2 || continue
            push!(layers,PolygonLayer([points],_VisualRole(:background),_VisualRole(:background),1.,0.))
        end
    end
    layers
end

function _summand_geometry_panel(prepared,dims,title,o,maxdim;support=false,category=1,outline=false)
    geom=prepared.geometry;pi=prepared.pi
    value(q)=q==0 ? -1 : support ? Int(dims[q]>0) : dims[q]
    comps=NamedTuple[]
    for c in geom.components
        v=value(c.region_id)
        edges=BitVector(value(_visual_locate(pi,((c.vertices[i][1]+c.vertices[mod1(i+1,length(c.vertices))][1])/2,
            (c.vertices[i][2]+c.vertices[mod1(i+1,length(c.vertices))][2])/2)))==v for i in eachindex(c.edge_included))
        push!(comps,merge(c,(;region_id=v,vertex_included=BitVector(value(_visual_locate(pi,p))==v for p in c.vertices),edge_included=edges)))
    end
    colors=Dict{Int,Union{Symbol,_VisualRole}}(v=>(v<0 ? _VisualRole(:unrepresented) :
        support && v>0 ? _VisualRole(:categorical,category) : _summand_dim_color(v,maxdim,o.scale)) for v in unique(value(c.region_id) for c in geom.components))
    support && filter!(c->c.region_id>0,comps)
    g=merge(geom,(;components=comps,has_unrepresented_area=false))
    layers=_region_geometry_layers(g;colors)
    if !support
        layers=AbstractVisualizationLayer[l isa PolygonLayer ? PolygonLayer(l.polygons,l.fill_color,l.stroke_color,1.0,l.linewidth) : l for l in layers]
    end
    if outline
        filter!(l->!(l isa PolygonLayer),layers)
    end
    # General polyhedral classifiers need not enumerate their missing cells.
    # Paint their mask explicitly. Overlay pieces share one such background,
    # so a later piece cannot erase an earlier support fill.
    if !support || !(o.layout in (:overlay,:outline))
        layers=vcat(_summand_geometry_background(geom;support),layers)
    end
    targets=[(;kind=:vertex,vertex=c.region_id,point=_drawing_point((sum(first,c.vertices)/length(c.vertices),sum(last,c.vertices)/length(c.vertices))),
        label="Finite label $(c.region_id); dimension $(c.region_id==0 ? "unrepresented" : dims[c.region_id])") for c in geom.components if c.region_id>0 && c.dimension==2]
    collision=!isempty(geom.coordinate_collisions)||!isempty(geom.dimension_collapses)
    VisualizationSpec(:summand_support;title,
        subtitle="Classifier fibers; solid included, dashed excluded, dotted window\n"*(collision ? "Exact coordinates collide at drawing precision" : "White = zero; gray = unrepresented"),
        layers,axes=geom.axes,legend=support ? _default_legend() : _summand_dimension_legend(maxdim,o.scale),metadata=(;dimensions=copy(dims),geometry=geom,geometry_status=:exact_classifier_fibers,
            pick_targets=targets,max_dimension=maxdim,scale=o.scale))
end

function _summand_supplied_panel(d,j,title;outline=false)
    layers=AbstractVisualizationLayer[];allboxes=reduce(vcat,d.boxes;init=Tuple{_VisualPoint,_VisualPoint}[])
    for (a,b) in d.boxes[j]
        lo,hi=_drawing_point(a),_drawing_point(b);color=_VisualRole(:categorical,j)
        !outline && push!(layers,RectLayer([(lo...,hi...)],color,color,0.35,0.))
        push!(layers,SegmentLayer([(lo...,hi[1],lo[2]),(lo...,lo[1],hi[2])],color,1.,2.,:solid))
        push!(layers,SegmentLayer([(hi...,lo[1],hi[2]),(hi...,hi[1],lo[2])],color,1.,2.,:dash))
    end
    isempty(layers) && push!(layers,TextLayer(["No supplied boxes"],[(0.,0.)],_VisualRole(:muted),12.))
    box=isempty(allboxes) ? ((0,0),(1,1)) : (ntuple(k->minimum(b[1][k] for b in allboxes),2),ntuple(k->maximum(b[2][k] for b in allboxes),2))
    VisualizationSpec(:supplied_summand_support;title,subtitle="Supplied boxes; geometry unverified\nBox edges: lower included, upper excluded; overlaps form a union",
        layers,axes=_geometry_axes_2d(box),metadata=(;geometry_status=:unverified,provenance=d.provenance,error_statement=d.error_statement,boxes=deepcopy(d.boxes[j]),pick_targets=NamedTuple[]))
end

function _summand_hom(c,prepared,ids,cache)
    entries=Dict{Tuple{Int,Int},Int}()
    filter!(p->all(j->j in ids,p.first),cache)
    for a in ids,b in ids
        key=(a,b)
        if !haskey(cache,key)
            A,B=c.pieces[a],c.pieces[b]
            d=sum(BigInt(A.dims[q])*B.dims[q] for q in eachindex(A.dims);init=big(0))
            d^2<=prepared.max_hom_entries || throw(ArgumentError("Hom($a,$b) exceeds max_hom_entries."))
            cache[key]=DerivedFunctors.Hom(A,B)
        end
        entries[key]=DerivedFunctors.dim(cache[key])
    end
    entries
end

function _summand_snapshot(c,prepared,o;hom_cache=Dict{Tuple{Int,Int},Any}())
    selected=zeros(Int,length(c.source.dims))
    for j in o.ids;selected .+= c.pieces[j].dims;end
    omitted=c.source.dims-selected
    maxdim=maximum(c.source.dims;init=0)
    panel(dims,title)=prepared.geometry===nothing ? _summand_finite_panel(prepared,dims,title,o,o.shared_scale ? maxdim : maximum(dims;init=0)) :
        _summand_geometry_panel(prepared,dims,title,o,o.shared_scale ? maxdim : maximum(dims;init=0))
    panels=VisualizationSpec[panel(c.source.dims,"Full module M"),panel(selected,"Selected blocks"),panel(omitted,"Complement of the selection")]
    gallery=VisualizationSpec[]
    displayed=o.layout===:isolate ? (o.focus==0 ? Int[] : [o.focus]) : o.ids
    for j in displayed
        title="$(c.labels[j]) [$j], multiplicity $(c.multiplicities[j])"
        push!(gallery,prepared.drawing!==nothing ? _summand_supplied_panel(prepared.drawing,j,title;outline=o.layout===:outline) :
            prepared.geometry!==nothing ? _summand_geometry_panel(prepared,c.pieces[j].dims,title,o,o.shared_scale ? maxdim : maximum(c.pieces[j].dims;init=0);
                support=!isempty(c.interval_supports[j]) || o.layout in (:overlay,:outline),category=j,outline=o.layout===:outline) : panel(c.pieces[j].dims,title))
    end
    for (k,j) in enumerate(displayed)
        p=gallery[k]
        targets=[merge(t,(;block=j,label="Block $j; "*t.label)) for t in get(p.metadata,:pick_targets,NamedTuple[])]
        layers=copy(p.layers)
        if prepared.geometry!==nothing
            for (name,points) in (("g",c.grades[j].generators),("r",c.grades[j].relations))
                drawn=_drawing_point.(collect(points))
                isempty(drawn) && continue
                push!(layers,PointLayer(drawn,_VisualRole(:categorical,j),1.,12.))
                push!(layers,TextLayer(["$name$i (supplied)" for i in eachindex(drawn)],drawn,_VisualRole(:foreground),11.))
            end
        end
        gallery[k]=VisualizationSpec(p.kind;title=p.title,subtitle=p.subtitle,layers,axes=p.axes,legend=p.legend,metadata=merge(p.metadata,(;pick_targets=targets)))
    end
    if o.layout in (:overlay,:outline) && !isempty(gallery)
        prepared.geometry===nothing && prepared.drawing===nothing && throw(ArgumentError("Overlay and outline need encoding geometry or supplied support boxes."))
        layers=reduce(vcat,(p.layers for p in gallery);init=AbstractVisualizationLayer[])
        prepared.geometry===nothing || (layers=vcat(_summand_geometry_background(prepared.geometry;support=true),layers))
        entries=(; (Symbol("$(c.labels[j]) [$j]")=>_VisualRole(:categorical,j) for j in displayed)...)
        push!(panels,VisualizationSpec(:summand_overlay;title="Selected support overlay",subtitle="Each color identifies a block; overlap is not an extra summand",
            layers,axes=first(gallery).axes,legend=_default_legend(;visible=true,entries),
            metadata=(;ids=copy(displayed),geometry_status=prepared.drawing===nothing ? :exact_classifier_fibers : :unverified)))
    else
        append!(panels,gallery)
    end
    focus=o.focus;v=o.vertex
    if focus>0 && (o.layout in (:overlay,:outline) || prepared.drawing!==nothing)
        push!(panels,panel(c.pieces[focus].dims,"$(c.labels[focus]): stalk dimensions"))
    end
    if focus>0 && v>0
        for (A,title) in ((c.inclusions[focus].comps[v],"$(c.labels[focus]) inclusion at $v"),
                (c.projections[focus].comps[v],"$(c.labels[focus]) projection at $v"))
            push!(panels,_algebra_matrix_panel(A,title,c.source.field,o.matrix_limit))
        end
        if o.pair!==nothing
            u,w=o.pair
            if leq(c.source.Q,u,w)
                push!(panels,_algebra_matrix_panel(Modules.map_leq(c.pieces[focus],u,w),"$(c.labels[focus]): structure map $u to $w",c.source.field,o.matrix_limit))
            else
                push!(panels,_inspection_text_panel("No forward structure map",["The selected finite labels are not ordered from $u to $w."]))
            end
        end
    end
    homdata=o.hom ? _summand_hom(c,prepared,o.ids,hom_cache) : Dict{Tuple{Int,Int},Int}()
    hom_matrix=nothing;hom_component=nothing
    if o.hom
        hom_matrix=[homdata[a,b] for b in o.ids,a in o.ids]
        push!(panels,_presentation_matrix_panel(hom_matrix,"Ordinary Hom dimensions",c.labels[o.ids],c.labels[o.ids],o.matrix_limit;
            subtitle="Source columns, target rows; not Hom^alpha or Hasse adjacency"))
        if o.hom_pair!==nothing
            a,b=o.hom_pair;H=hom_cache[a,b];d=homdata[a,b]
            if d>0
                o.basis_index<=d || throw(ArgumentError("basis_index exceeds the selected Hom space."))
                f=DerivedFunctors.basis(H)[o.basis_index]
                if v>0
                    hom_component=copy(Modules.component(f,v))
                    push!(panels,_algebra_matrix_panel(hom_component,"Hom($a,$b), basis $(o.basis_index), vertex $v",c.source.field,o.matrix_limit))
                end
            else
                push!(panels,_inspection_text_panel("Zero Hom space",["There is no nonzero morphism from block $a to block $b."]))
            end
        end
    end
    status=prepared.drawing!==nothing ? "Geometry unverified: $(prepared.drawing.provenance). $(prepared.drawing.error_statement)" :
        prepared.geometry===nothing ? "Schematic finite order; no ambient support boundary asserted." : "Exact classifier fibers within the displayed window."
    lines=["All inclusions/projections passed naturality, retraction and reconstruction checks.",
        "$(length(o.ids)) of $(length(c.pieces)) blocks selected; $(sum(c.multiplicities[o.ids])) of $(sum(c.multiplicities)) copies.",
        "Complement = omitted blocks + retained residual; residual dimensions $(c.residual.dims).",
        "Ranking: $(o.sort_by); filter: $(o.type_filter); minimum $(o.min_value); display limit $(o.limit).",
        "$(o.shared_scale ? "Shared" : "Independent") $(o.scale) dimension colors; full maximum $maxdim; zero is white.",
        status,"IDs belong to this supplied splitting; no stable matching across modules is asserted."]
    if focus>0
        s=prepared.stats[focus];g=c.grades[focus]
        push!(lines,"Selected block $(s.id): $(s.support_vertices) support labels, total dimension $(s.total_dimension), maximum stalk $(s.max_dimension).")
        (!isempty(g.generators)||!isempty(g.relations)) && push!(lines,"Supplied grades (presentation unverified): generators $(g.generators); relations $(g.relations).")
    end
    push!(panels,_inspection_text_panel("Reading the summands",lines))
    positions=Tuple{UnitRange{Int},UnitRange{Int}}[]
    for j in 1:length(panels)-1
        row=cld(j,3);col=mod1(j,3);push!(positions,(row:row,col:col))
    end
    row=cld(length(panels)-1,3)+1;push!(positions,(row:row,1:3))
    weights=[fill(380.,row-1);max(260.,30length(lines))]
    VisualizationSpec(:summand_gallery;title="Summands of a finite module",
        subtitle="Certified splitting over $(_inspection_field_label(c.source.field)); geometry has its own status",panels,
        metadata=(;selection=o,selected_ids=copy(o.ids),selected_dimensions=selected,complement_dimensions=omitted,
            residual_dimensions=copy(c.residual.dims),statistics=deepcopy(prepared.stats),certification=:checked_splitting,
            geometry_status=prepared.drawing!==nothing ? :unverified : prepared.geometry===nothing ? :schematic : :exact_classifier_fibers,
            hom_dimensions=hom_matrix,hom_component,panel_positions=positions,panel_row_weights=weights,figure_size=(1450,sum(weights)+90)))
end

function _visual_spec(input::_SummandInput,::Symbol;kwargs...)
    c=input isa SummandCollection ? input : summand_collection(input)
    prepkeys=(:encoding,:box,:drawing,:max_vertices,:max_summands,:max_hom_pairs,:max_hom_entries)
    prepared=_summand_prepare(c;(; (k=>v for (k,v) in kwargs if k in prepkeys)...)...)
    o=_summand_options(c,prepared;(; (k=>v for (k,v) in kwargs if k in _SUMMAND_OPTIONS)...)...)
    _summand_snapshot(c,prepared,o)
end
_visual_request_cost(::_SummandInput,::Symbol)=(;work=:checked_direct_sum,geometry=:declared_classifier_or_supplied,
    hom=:explicit_bounded_opt_in,identity=:local_to_supplied_splitting,timing=:not_measured)

function _append_visual_request_issues!(issues::Vector{String},input::_SummandInput,::Symbol;kwargs...)
    input isa IntervalDecompositions.IntervalDecompositionResult &&
        !IntervalDecompositions.interval_decomposition_summary(input).witnesses && push!(issues,"The gallery needs witnesses=true in interval_decomposition.")
    input isa IndecomposableDecompositions.IndecomposableDecompositionResult &&
        !IndecomposableDecompositions.indecomposable_decomposition_summary(input).witnesses &&
        push!(issues,"The gallery needs witnesses=true in indecomposable_decomposition or bipath_decomposition.")
    for (key,allowed) in ((:layout,(:facets,:overlay,:outline,:isolate)),
            (:sort_by,(:index,:multiplicity,:support_vertices,:total_dimension,:max_dimension,:generators,:relations)),
            (:type_filter,(:all,:interval,:thin,:thick)),(:scale,(:linear,:log)))
        haskey(kwargs,key) && !(kwargs[key] in allowed) && push!(issues,"$key must be one of $allowed.")
    end
    for key in (:hom,:shared_scale)
        haskey(kwargs,key) && !(kwargs[key] isa Bool) && push!(issues,"$key must be Bool.")
    end
    for key in (:limit,:max_vertices,:max_summands,:max_hom_pairs,:max_hom_entries,:basis_index)
        haskey(kwargs,key) && _signed_integer(kwargs[key],key;lower=1)
    end
    get(kwargs,:encoding,nothing)!==nothing && get(kwargs,:drawing,nothing)!==nothing && push!(issues,"Choose an encoding or a supplied drawing, not both.")
    issues
end

source_module(c::SummandCollection)=c.source
function _summand_id(c,id)
    id===:residual && return 0
    id isa Integer && !(id isa Bool) && 1<=id<=length(c.pieces) || throw(ArgumentError("id must be a block ID or :residual."))
    Int(id)
end
"""
    summand_module(collection; id)

Return the retained module for a positive block ID, or `id=:residual` for the
original residual. `source_module(collection)` returns the full module. These
objects are read-only references; a display complement may also include omitted
blocks, unlike this original residual.
"""
function summand_module(c::SummandCollection;id)
    j=_summand_id(c,id);j==0 ? c.residual : c.pieces[j]
end
"""
    summand_maps(collection; id)

Return `(inclusion, projection)` for a block ID or `:residual`. The inclusion
goes into `source_module(collection)` and the projection comes back. The
retained morphisms support ordinary morphism and matrix inspection.
"""
function summand_maps(c::SummandCollection;id)
    j=_summand_id(c,id)
    j==0 ? (;inclusion=c.residual_inclusion,projection=c.residual_projection) :
        (;inclusion=c.inclusions[j],projection=c.projections[j])
end
