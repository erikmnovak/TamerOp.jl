# H0 on the actual cellular one-skeleton. Geometry never determines incidence.
"""
    ComponentHistory

A merge forest for one specified one-parameter filtration or oriented affine
slice of a `FilteredComplex`. Construct with `component_history`. The forest
retains original vertex/edge indices and exact event parameters, including
simultaneous and zero-length births. Elder ties use the source vertex index;
this is a reproducible choice, not a canonical identity of tied classes.

Use `component_count` for a cheap count, `components_at` or `component_members`
for requested memberships, and `component_intervals`/`component_merges` for
bars and events. Retained inputs/arrays must be treated as read-only.
"""
struct ComponentHistory{S,T,C,P}
    source::S
    slice::C
    births::Vector{T}
    edge_times::Vector{T}
    endpoints::Vector{Tuple{Int,Int}}
    parent::Vector{Int}
    children::Vector{Tuple{Int,Int}}
    times::Vector{T}
    elders::Vector{Int}
    merge_edges::Vector{Int}
    deaths::Vector{Union{Nothing,T}}
    absorbed_by::Vector{Int}
    roots::Vector{Int}
    provenance::P
end

function _component_scalar(x)
    x isa Real && !(x isa Bool) && isfinite(x) || throw(ArgumentError("Component parameters must be finite real numbers."))
    x isa AlgebraicReal && return x
    x isa Union{Integer,Rational,AbstractFloat} && return QQ(x)
    throw(ArgumentError("Use integer, rational, finite floating-point or supported algebraic parameters."))
end

function _component_slice(x,slice)
    N=DataIngestion.DataTypes.parameter_dim(DataIngestion.graded_complex(x))
    o=DataIngestion.grade_orientation(x)
    if slice===nothing
        N==1 || throw(ArgumentError("A multiparameter complex needs slice=(base=(...), direction=(...)). There is no universal merge tree."))
        return (;base=(QQ(0),),direction=(QQ(o[1]),),kind=:original,orientation=o)
    end
    slice isa NamedTuple && Set(keys(slice))==Set((:base,:direction)) ||
        throw(ArgumentError("slice must contain exactly base and direction."))
    all(v->v isa Union{Tuple,AbstractVector} && length(v)==N,(slice.base,slice.direction)) ||
        throw(ArgumentError("Slice base and direction must have $N coordinates."))
    base=Tuple(_component_scalar.(slice.base));direction=Tuple(_component_scalar.(slice.direction))
    all(i->o[i]*direction[i]>0,1:N) || throw(ArgumentError("Every slice direction must be strictly positive in the source's oriented order."))
    return (;base,direction,kind=:affine,orientation=o)
end

"""
    component_history(source::FilteredComplex; slice=nothing)
    component_history(source::SourceCorrespondence)
    component_history(diagram::PersistenceDiagram)

Compute H0 component births and merges from the retained source one-skeleton.
Each nonzero edge boundary must have two opposite unit coefficients; loops
with zero boundary do not merge components. Higher cells do not change H0.
The calculation is field independent for this cellular contract.

One-parameter sources use their original parameter (including superlevels).
For multiparameter sources supply `slice=(base=..., direction=...)`, with every
direction strictly positive in the declared oriented order. A cell with several
critical grades enters at the minimum of their slice-entry times. Slice queries
use its affine parameter t. No normalization or sampling is hidden.

Vertices precede edges at tied events; the older component survives, with ties
broken by source vertex index. Membership at an event includes all cells at that
event. Zero-length bars remain in the forest and are omitted from its ordinary
diagram. Source-correspondence inputs require a verified retraction and the
same graph-incidence contract on both complexes. Storage is linear in vertices/edges; memberships are materialized only
when requested. A diagram input must retain source correspondence. Its H0
interval multiset is checked against the original source; repeated equal bars
do not identify individual reduction-basis representatives.
"""
function component_history(x::DataIngestion.FilteredComplex;slice=nothing)
    DataIngestion.check_filtered_complex(x;throw=true)
    s=_component_slice(x,slice);G=DataIngestion.graded_complex(x)
    counts=DataIngestion.DataTypes.cell_counts(G);n=first(counts);m=length(counts)>1 ? counts[2] : 0
    # Normalize integers and binary floating inputs to exact rational values;
    # algebraic grades retain their ordered algebraic field.
    entry(i)=minimum(maximum((_component_scalar(g[k])-s.base[k])/s.direction[k] for k in eachindex(g))
        for g in DataIngestion.DataTypes.cell_grade_set(G,i))
    raw=[entry(i) for i in 1:n+m]
    T=isempty(raw) ? QQ : foldl(promote_type,(typeof(v) for v in raw);init=QQ)
    values=T.(raw);births=values[1:n];edge_times=values[n+1:end]
    endpoints=Tuple{Int,Int}[]
    if m>0
        B=DataIngestion.DataTypes.boundary_map(G;dim=1)
        for e in 1:m
            nz=[k for k in nzrange(B,e) if !iszero(nonzeros(B)[k])]
            if isempty(nz)
                push!(endpoints,(0,0));continue
            end
            length(nz)==2 && all(k->abs(nonzeros(B)[k])==1,nz) &&
                sum(nonzeros(B)[k] for k in nz)==0 ||
                throw(ArgumentError("Edge $e is not an oriented graph edge; H0 component membership requires opposite unit endpoints."))
            u,v=rowvals(B)[nz[1]],rowvals(B)[nz[2]]
            max(births[u],births[v])<=edge_times[e] || throw(ArgumentError("An edge precedes an endpoint on the selected slice."))
            push!(endpoints,(u,v))
        end
    end
    # Union-find identifies components; a separate persistent binary forest
    # records membership without storing a quadratic list at every merge.
    uf=collect(1:n);tree=collect(1:n)
    function root(i)
        while uf[i]!=i;uf[i]=uf[uf[i]];i=uf[i];end
        i
    end
    parent=zeros(Int,n);children=fill((0,0),n);times=copy(births);elders=collect(1:n);merge_edges=zeros(Int,n)
    deaths=Union{Nothing,T}[nothing for _ in 1:n];absorbed_by=zeros(Int,n)
    for e in sortperm(1:m;by=e->(edge_times[e],e))
        u,v=endpoints[e];u==0 && continue
        a,b=root(u),root(v);a==b && continue
        # The union representative is the declared elder, not union-find rank.
        (births[a],a)>(births[b],b) && ((a,b)=(b,a))
        left,right=tree[a],tree[b];node=length(times)+1
        push!(parent,0);push!(children,(left,right));push!(times,edge_times[e]);push!(elders,a);push!(merge_edges,e)
        parent[left]=node;parent[right]=node;deaths[b]=edge_times[e];absorbed_by[b]=a
        uf[b]=a;tree[a]=node
    end
    roots=[tree[i] for i in 1:n if uf[i]==i]
    p=(;kind=:component_history,source_model=x.provenance.model,construction=x.provenance,
        field_independent=true,tie_rule=:birth_then_vertex_index,event_boundary=:all_equal_events_included,
        slice_kind=s.kind,source_identity=:retained_source_cells,diagram_correspondence=:not_requested)
    return ComponentHistory(x,s,births,edge_times,endpoints,parent,children,times,elders,merge_edges,deaths,absorbed_by,roots,p)
end

function _component_recontext(h,p)
    ComponentHistory(h.source,h.slice,h.births,h.edge_times,h.endpoints,h.parent,h.children,
        h.times,h.elders,h.merge_edges,h.deaths,h.absorbed_by,h.roots,merge(h.provenance,p))
end
function component_history(c::DataIngestion.SourceCorrespondence)
    DataIngestion.check_source_correspondence(c;throw=true)
    c.retraction===nothing && throw(ArgumentError("A source correspondence needs a verified retraction for component comparison."))
    source=component_history(c.source);computed=component_history(c.computed)
    _component_barcode_key(source)==_component_barcode_key(computed) ||
        throw(ArgumentError("Computed and original source H0 intervals disagree."))
    return _component_recontext(source,(;source_identity=:verified_original_correspondence,
        diagram_correspondence=:endpoint_multisets,correspondence=c.certificate,correspondence_field=c.field))
end
function component_history(d::PersistenceDiagram)
    check_persistence_diagram(d;throw=true)
    d.retained_source===nothing && throw(ArgumentError("Component histories need retained source cells; compute the diagram with source=true or supply a FilteredComplex."))
    h=component_history(d.retained_source)
    expected=(sort([(r.birth,r.death) for r in component_intervals(h) if r.death!==nothing]),
        sort([r.birth for r in component_intervals(h) if r.death===nothing]))
    actual=(sort([(_component_scalar(b),_component_scalar(e)) for (b,e) in finite_intervals(d;dim=0)]),
        sort(_component_scalar.(essential_births(d;dim=0))))
    expected==actual || throw(ArgumentError("Diagram H0 intervals disagree with the original source merge history."))
    return _component_recontext(h,(;diagram_correspondence=:endpoint_multisets,diagram_field=d.field))
end

_component_barcode_key(h)=(sort([(r.birth,r.death) for r in component_intervals(h) if r.death!==nothing]),
    sort([r.birth for r in component_intervals(h) if r.death===nothing]))
_component_external(h,t)=h.slice.kind==:original ? h.slice.direction[1]*t : t
function _component_progress(h,scale)
    t=_component_scalar(scale)
    return h.slice.kind==:original ? h.slice.direction[1]*t : t
end
"""`component_parameter(h, scale)` gives the original parameter tuple at a query scale."""
function component_parameter(h::ComponentHistory,scale)
    t=_component_progress(h,scale)
    return map((b,d)->b+t*d,h.slice.base,h.slice.direction)
end
"""`component_history_summary(h)` reports retained sizes and conventions without enumerating memberships."""
component_history_summary(h::ComponentHistory)=(;kind=:component_history,vertices=length(h.births),edges=length(h.edge_times),
    merges=length(h.times)-length(h.births),essential=length(h.roots),
    ephemeral=count(i->h.deaths[i]!==nothing && h.deaths[i]==h.births[i],eachindex(h.births)),
    slice=h.slice,provenance=h.provenance)
describe(h::ComponentHistory)=component_history_summary(h)
provenance(h::ComponentHistory)=h.provenance
Base.show(io::IO,h::ComponentHistory)=print(io,"ComponentHistory(",length(h.births)," vertices, ",length(h.times)-length(h.births)," merges)")

"""`component_intervals(h; include_ephemeral=false)` returns bars labelled by their source birth vertex. Death is `nothing` for essential bars; superlevel endpoints use original units."""
function component_intervals(h::ComponentHistory;include_ephemeral=false)
    include_ephemeral isa Bool || throw(ArgumentError("include_ephemeral must be Bool."))
    [(;component=i,birth=_component_external(h,h.births[i]),
        death=h.deaths[i]===nothing ? nothing : _component_external(h,h.deaths[i]),
        ephemeral=h.deaths[i]!==nothing && h.births[i]==h.deaths[i],absorbed_by=h.absorbed_by[i],
        source=DataIngestion.cell_source(h.source,i)) for i in eachindex(h.births)
        if include_ephemeral || h.deaths[i]===nothing || h.births[i]!=h.deaths[i]]
end
"""`component_merges(h)` lists actual merge edges, both child nodes, surviving/absorbed birth vertices, and exact scales. Equal scales are simultaneous; their binary display order is a choice."""
function component_merges(h::ComponentHistory)
    n=length(h.births)
    [(;node=i,scale=_component_external(h,h.times[i]),edge=h.merge_edges[i],source_edge=DataIngestion.cell_source(h.source,n+h.merge_edges[i]),
        children=h.children[i],survivor=h.elders[i],absorbed=h.elders[last(h.children[i])]) for i in n+1:length(h.times)]
end
function _component_cut(h,t)
    nodes=Int[];stack=copy(h.roots);n=length(h.births)
    while !isempty(stack)
        node=pop!(stack)
        if h.times[node]<=t;push!(nodes,node)
        elseif node>n;append!(stack,h.children[node]);end
    end
    return sort!(nodes;by=i->h.elders[i])
end
function _component_vertices(h,node)
    ids=Int[];stack=[node];n=length(h.births)
    while !isempty(stack)
        i=pop!(stack)
        i<=n ? push!(ids,i) : append!(stack,h.children[i])
    end
    sort!(ids)
end
"""`component_count(h; scale)` counts actual components after all events at `scale`, without materializing memberships."""
component_count(h::ComponentHistory;scale)=length(_component_cut(h,_component_progress(h,scale)))
"""`components_at(h; scale)` materializes the partition of active source vertices, with persistent elder labels and original source identities."""
function components_at(h::ComponentHistory;scale)
    [(;component=h.elders[node],node,vertices=ids,sources=[DataIngestion.cell_source(h.source,i) for i in ids])
        for node in _component_cut(h,_component_progress(h,scale)) for ids in (_component_vertices(h,node),)]
end
"""
    component_members(h; scale, component)

Follow the source birth vertex labelled `component`. Return its current actual
component and original vertex IDs. `status=:merged` explicitly follows its
descendant after the selected bar dies; it does not assert that the bar survives.
Before birth the membership is empty. All events at the queried scale are included.
"""
function component_members(h::ComponentHistory;scale,component)
    component isa Int && 1<=component<=length(h.births) || throw(ArgumentError("component must be a source birth-vertex index."))
    t=_component_progress(h,scale);node=component
    born=h.births[node]<=t
    while born && h.parent[node]!=0 && h.times[h.parent[node]]<=t;node=h.parent[node];end
    ids=born ? _component_vertices(h,node) : Int[]
    status=!born ? :not_born : h.deaths[component]!==nothing && h.deaths[component]<=t ? :merged : :alive
    return (;component,status,node=born ? node : nothing,descendant=born ? h.elders[node] : nothing,
        vertices=ids,sources=[DataIngestion.cell_source(h.source,i) for i in ids])
end

"""`persistence_diagram(h::ComponentHistory; field=F2())` returns its H0 intervals. It is field independent under the checked graph-incidence contract; component labels are retained in metadata, not arbitrary reduction representatives."""
function persistence_diagram(h::ComponentHistory;field=F2())
    records=component_intervals(h);T=eltype(h.births)
    finite=Tuple{T,T}[(r.birth,r.death) for r in records if r.death!==nothing]
    essential=T[r.birth for r in records if r.death===nothing]
    order=h.slice.kind==:original && h.slice.direction[1]<0 ? :superlevel : :sublevel
    PersistenceDiagram([finite],[essential];field,order,meta=(;component_history=h.provenance,
        finite_components=[r.component for r in records if r.death!==nothing],
        essential_components=[r.component for r in records if r.death===nothing]))
end

"""Validation of a supplied merge history, including its source, slice and reconstructed forest. This explicit check rebuilds the history; ordinary count/membership queries do not."""
function check_component_history(h::ComponentHistory;throw=false)
    issues=String[]
    try
        slice=h.slice.kind==:original ? nothing : (;h.slice.base,h.slice.direction)
        expected=component_history(h.source;slice)
        h.slice==expected.slice || push!(issues,"Invalid retained slice conventions.")
        for key in (:births,:edge_times,:endpoints,:parent,:children,:times,:elders,:merge_edges,:deaths,:absorbed_by,:roots)
            getfield(h,key)==getfield(expected,key) || push!(issues,"Invalid retained $key.")
        end
    catch err;err isa InterruptException && rethrow();push!(issues,sprint(showerror,err));end
    valid=isempty(issues)
    throw && !valid && Base.throw(ArgumentError(join(issues,"\n")))
    return (;kind=:component_history_validation,valid,issues)
end
