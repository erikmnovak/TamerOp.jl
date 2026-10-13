# Extended persistence uses one filtered pair on two ordered copies of the
# parameter line. The inclusion cone is naturally equivalent to its quotients.
# Cohen-Steiner, Edelsbrunner, Harer, doi:10.1007/s10208-008-9027-z.

"""A vertex-defined ordinary-to-relative filtration. Use `extended_filtration`,
`extended_steps`, `extended_pair`, and `extended_persistence`. Scalar grades are
retained exactly; integer steps are only an order encoding, not a metric."""
struct ExtendedFiltration{T<:Real,M<:NamedTuple}
    pair::FilteredPair{Int}
    values::Vector{T}
    order::Symbol
    meta::M
end

"""An interval on two ordered parameter copies. `birth_phase` and `death_phase`
are `:ordinary` or `:relative`; birth is included and death excluded in that
order. `kind` is `:ordinary`, `:relative`, `:extended_positive`,
`:extended_negative`, or `:extended_zero`. The last three cross the junction;
equal scalar endpoints on different copies do NOT give an empty interval.
Positive/negative compare endpoints in the original filtration orientation.
`birth_step` and `death_step` retain their order even across the junction."""
struct ExtendedInterval{T<:Real}
    dimension::Int
    kind::Symbol
    birth::T
    death::T
    birth_phase::Symbol
    death_phase::Symbol
    birth_included::Bool
    death_included::Bool
    birth_step::Int
    death_step::Int
end

"""Typed extended persistence in every ambient chain degree. Access intervals
with `extended_intervals(result; dim, kind=:all)`; no essential bars are capped
or discarded. The retained filtration supplies actual quotient and homology maps."""
struct ExtendedPersistenceDiagram{T<:Real,F<:ExtendedFiltration{T}}
    filtration::F
    intervals::Vector{Vector{ExtendedInterval{T}}}
end

const _EXTENDED_KINDS = (:ordinary,:relative,:extended_positive,:extended_negative,:extended_zero)
field(x::ExtendedFiltration) = field(x.pair)
field(x::ExtendedPersistenceDiagram) = field(x.filtration)
filtration_order(x::ExtendedFiltration) = x.order
filtration_order(x::ExtendedPersistenceDiagram) = filtration_order(x.filtration)
provenance(x::ExtendedFiltration) = merge(deepcopy(x.meta), (;homology=:extended,
    order=x.order, field=field(x), parameter_copies=(:ordinary,:relative),
    ordinary_closure=:closed_subcomplex,relative_closure=:closed_subcomplex,
    construction=:filtered_inclusion_cone,metric=:original_scalar_values))
provenance(x::ExtendedPersistenceDiagram) = provenance(x.filtration)

function _extended_from_grades(G, lower::Vector{T}, upper::Vector{T}, F, order, meta) where {T<:Real}
    _normalize_order(order); _normalize_field(F)
    length(lower)==length(upper)==length(G.grades) || throw(ArgumentError("cell grade counts disagree"))
    all(isfinite,lower) && all(isfinite,upper) || throw(ArgumentError("extended grades must be finite"))
    values=sort!(unique!(map(v->iszero(v) ? zero(v) : v, [lower;upper]));rev=order===:superlevel)
    n=length(values)
    rank=Dict(v=>i for (i,v) in enumerate(values))
    ordinary=order===:sublevel ? upper : lower
    relative=order===:sublevel ? lower : upper
    K=GradedComplex([collect(cs) for cs in G.cells_by_dim],deepcopy(G.boundaries),
        NTuple{1,Int}[(rank[v],) for v in ordinary])
    offsets=getfield(K,:dim_offsets)
    ids=[collect(1:length(cs)) for cs in K.cells_by_dim]
    grades=[[2n+1-rank[relative[i]] for i in offsets[d]:offsets[d+1]-1] for d in eachindex(ids)]
    pair=filtered_pair(K,ids;grades,field=F)
    return ExtendedFiltration(pair,values,order,meta)
end

"""
    extended_filtration(G; vertex_values, cell_vertices, field=F2(), order=:sublevel)
    extended_filtration(simplices, vertex_values; field=F2(), order=:sublevel, max_cells=100_000)
    extended_filtration(values::AbstractArray; periodic=false, field=F2(), order=:sublevel,
                        construction=ConstructionOptions())

Construct K's lower-star filtration followed by H(K,K^t), where K^t is the
upper-star subcomplex. Superlevel order exchanges the two stars. Both stars
include cells at equality. Old grades on G are replaced by this scalar function.
`cell_vertices` gives vertex POSITIONS for every cell in dimension-major order;
each boundary cell's vertices must be a subset. This checks a based chain model,
not geometric realizability of a supplied cellular attaching map.

The simplicial form takes simplices or maximal faces, closes under faces, and
includes every supplied vertex (including isolated ones). It uses ascending
vertex orientations and a strict cell budget. The array form uses vertex-valued
cubes, with explicit periodic identifications; these are not top-cell values.
No geometry is inferred from boundary coefficients or coincident coordinates.
"""
function extended_filtration(G::GradedComplex;vertex_values::AbstractVector{T},
        cell_vertices::AbstractVector,field=F2(),order::Symbol=:sublevel) where {T<:Real}
    isconcretetype(T) && T!==Bool && all(isfinite,vertex_values) ||
        throw(ArgumentError("vertex values need a concrete finite real type other than Bool"))
    counts=length.(G.cells_by_dim); offsets=getfield(G,:dim_offsets)
    !isempty(counts) && counts[1]==length(vertex_values) || throw(ArgumentError("one scalar per degree-zero cell is required"))
    length(cell_vertices)==length(G.grades) || throw(ArgumentError("one vertex list per cell is required"))
    supports=Vector{Int}[]
    for (i,vs) in enumerate(cell_vertices)
        (vs isa AbstractVector || vs isa Tuple) && !isempty(vs) &&
            all(v->v isa Integer && !(v isa Bool) && 1<=v<=counts[1],vs) && allunique(vs) ||
            throw(ArgumentError("invalid vertex positions for cell $i"))
        i<=counts[1] && collect(vs)!=[i] && throw(ArgumentError("degree-zero cell $i must have vertex list [$i]"))
        push!(supports,sort!(Int.(collect(vs))))
    end
    for d in eachindex(G.boundaries)
        B=G.boundaries[d]
        size(B)==(counts[d],counts[d+1]) || throw(ArgumentError("boundary shape disagrees with cells"))
        for j in axes(B,2), p in nzrange(B,j)
            iszero(B.nzval[p]) && continue
            issubset(supports[offsets[d]+B.rowval[p]-1],supports[offsets[d+1]+j-1]) ||
                throw(ArgumentError("cell vertices do not contain the vertices of its boundary"))
        end
    end
    lo=T[minimum(vertex_values[vs]) for vs in supports]
    hi=T[maximum(vertex_values[vs]) for vs in supports]
    return _extended_from_grades(G,lo,hi,field,order,(;input=:vertex_defined_chain_complex,
        vertex_values=copy(vertex_values),cell_vertices=supports,source_cells=Tuple(Tuple(cs) for cs in G.cells_by_dim),
        geometry=:not_asserted))
end

function extended_filtration(simplices::AbstractVector,values::AbstractVector{T};
        field=F2(),order::Symbol=:sublevel,max_cells::Integer=100_000) where {T<:Real}
    !(max_cells isa Bool) && 0<max_cells<=typemax(Int) || throw(ArgumentError("max_cells must be positive"))
    length(values)<=max_cells || throw(ArgumentError("vertices exceed max_cells"))
    faces=Set{Tuple{Vararg{Int}}}((i,) for i in eachindex(values))
    queue=Tuple{Vararg{Int}}[]
    for s in simplices
        (s isa AbstractVector || s isa Tuple) && !isempty(s) && allunique(s) &&
            all(v->v isa Integer && !(v isa Bool) && 1<=v<=length(values),s) ||
            throw(ArgumentError("simplices must use distinct positions in the vertex-value vector"))
        # One simplex alone already requires this many cells in its closure.
        length(s)<8sizeof(Int)-1 && (big(1)<<length(s))-1<=max_cells ||
            throw(ArgumentError("simplex face closure exceeds max_cells"))
        push!(queue,Tuple(sort!(Int.(collect(s)))))
    end
    while !isempty(queue)
        s=pop!(queue); s in faces && continue
        length(faces)<max_cells || throw(ArgumentError("face closure exceeds max_cells"))
        push!(faces,s)
        length(s)>1 && append!(queue,[Tuple(s[k] for k in eachindex(s) if k!=j) for j in eachindex(s)])
    end
    ordered=sort!(collect(faces);by=s->(length(s),s)); nd=maximum(length,ordered;init=1)
    groups=[filter(s->length(s)==d,ordered) for d in 1:nd]
    B=SparseMatrixCSC{Int,Int}[]
    for d in 2:nd
        lookup=Dict(s=>i for (i,s) in enumerate(groups[d-1])); rows=Int[];cols=Int[];vals=Int[]
        for (j,s) in enumerate(groups[d]), k in eachindex(s)
            push!(rows,lookup[Tuple(s[q] for q in eachindex(s) if q!=k)])
            push!(cols,j);push!(vals,isodd(k) ? 1 : -1)
        end
        push!(B,sparse(rows,cols,vals,length(groups[d-1]),length(groups[d])))
    end
    G=GradedComplex([collect(1:length(g)) for g in groups],B,NTuple{1,Int}[(0,) for _ in ordered])
    f=extended_filtration(G;vertex_values=values,cell_vertices=ordered,field,order)
    return ExtendedFiltration(f.pair,f.values,f.order,merge(f.meta,(;input=:simplicial_vertex_function,
        geometry=:abstract_simplicial_complex)))
end

"""Use an existing simplex tree's abstract cells with an explicit new scalar
function, indexed by its vertex IDs. Multigrades on the tree are replaced;
extra entries in `vertex_values` create isolated vertices as in the list form."""
function extended_filtration(tree::DataTypes.SimplexTreeMulti;vertex_values::AbstractVector,kwargs...)
    DataTypes.check_simplex_tree_multi(tree;throw=true)
    simplices=[collect(DataTypes.simplex_vertices(tree,i)) for i in 1:DataTypes.simplex_count(tree)]
    return extended_filtration(simplices,vertex_values;kwargs...)
end

function extended_filtration(values::AbstractArray{T};periodic=false,field=F2(),
        order::Symbol=:sublevel,construction::ConstructionOptions=ConstructionOptions()) where {T<:Real}
    T!==Bool || throw(ArgumentError("vertex values must not be Bool"))
    _normalize_order(order);_normalize_field(field);_validate_cubical_values(values)
    per=_periodic_tuple(periodic,Val(ndims(values)))
    lower=_vertex_cubical_complex(values,per,:superlevel,construction,nothing)
    upper=_vertex_cubical_complex(values,per,:sublevel,construction,nothing)
    return _extended_from_grades(upper,T[g[1] for g in lower.grades],T[g[1] for g in upper.grades],field,order,
        (;input=:cubical_vertex_function,shape=size(values),periodic=per,vertex_values=copy(vec(values)),
            source_cells=Tuple(Tuple(cs) for cs in upper.cells_by_dim),geometry=:cubical_vertex_model))
end

"""Copy the checked integer-step pair. At steps 0 through 2n it represents
empty -> K_t -> K -> (K,K^t) -> (K,K). Use A118's `relative_complex`,
`relative_homology`, `relative_maps` and `relative_structure_map` for actual
quotients and induced matrices. `extended_steps` gives the original parameters."""
extended_pair(f::ExtendedFiltration)=deepcopy(f.pair)
extended_pair(d::ExtendedPersistenceDiagram)=extended_pair(d.filtration)

"""Return step, phase and exact scalar value for every event, including ties
as one event per parameter copy. Step zero is the initial empty state."""
function extended_steps(f::ExtendedFiltration)
    n=length(f.values)
    return [(;step=i,phase=i<=n ? :ordinary : :relative,value=f.values[i<=n ? i : 2n+1-i]) for i in 1:2n]
end
extended_steps(d::ExtendedPersistenceDiagram)=extended_steps(d.filtration)

function _extended_step(f::ExtendedFiltration,phase,scale)
    phase in (:ordinary,:relative) || throw(ArgumentError("phase must be :ordinary or :relative"))
    scale isa Real && !(scale isa Bool) && isfinite(scale) || throw(ArgumentError("scale must be finite and real"))
    n=length(f.values)
    ascending=(phase===:ordinary)==(f.order===:sublevel)
    v=phase===:ordinary ? f.values : reverse(f.values)
    return (phase===:ordinary ? 0 : n)+searchsortedlast(v,scale;rev=!ascending)
end

"""`extended_homology(f; phase, scale)` constructs actual quotient homology at
the original scalar parameter. Its `RelativeHomology` snapshot uses integer
step coordinates internally; `basis`, `relative_cells` and `relative_maps`
retain ambient dimension-local cell identities and the connecting map."""
function extended_homology(f::ExtendedFiltration;phase::Symbol,scale::Real)
    return relative_homology(f.pair;scale=_extended_step(f,phase,scale))
end
extended_homology(d::ExtendedPersistenceDiagram;kwargs...)=extended_homology(d.filtration;kwargs...)

"""`extended_structure_map(f; source=(phase,scale), target=(phase,scale), dim)`
returns the induced homology matrix, target rows and source columns, in the
bases of `extended_homology`. The source must precede the target on the two-copy
line. Set `chains=true` for the quotient-chain map instead."""
function extended_structure_map(f::ExtendedFiltration;source::Tuple,target::Tuple,dim::Integer,chains::Bool=false)
    length(source)==length(target)==2 || throw(ArgumentError("source and target must be (phase, scale)"))
    source_step=_extended_step(f,source...);target_step=_extended_step(f,target...)
    if source[1]===target[1]
        ascending=(source[1]===:ordinary)==(f.order===:sublevel)
        (ascending ? source[2]<=target[2] : source[2]>=target[2]) || throw(ArgumentError("parameters run against the phase order"))
    elseif source[1]===:relative && target[1]===:ordinary
        throw(ArgumentError("relative parameters follow every ordinary parameter"))
    end
    a=relative_complex(f.pair;scale=source_step)
    b=relative_complex(f.pair;scale=target_step)
    return chains ? relative_structure_map(a,b;dim) : relative_structure_map(relative_homology(a),relative_homology(b);dim)
end
extended_structure_map(d::ExtendedPersistenceDiagram;kwargs...)=extended_structure_map(d.filtration;kwargs...)

function _extended_interval(f::ExtendedFiltration{T},dim,b,d) where {T}
    n=length(f.values)
    birth=f.values[b<=n ? b : 2n+1-b]; death=f.values[d<=n ? d : 2n+1-d]
    kind=d<=n ? :ordinary : b>n ? :relative : birth==death ? :extended_zero :
        ((birth<death)==(f.order===:sublevel) ? :extended_positive : :extended_negative)
    return ExtendedInterval{T}(dim,kind,birth,death,b<=n ? :ordinary : :relative,
        d<=n ? :ordinary : :relative,true,false,b,d)
end

"""`extended_persistence(input, ...; kwargs...)` computes the entire extended
sequence by reducing one inclusion cone. Accepts an `ExtendedFiltration` or
the inputs of `extended_filtration`. Uses unreduced homology over a prime field.
Same-phase zero-length bars are omitted; cross-phase equal-valued bars remain.
All essential ordinary classes die by the final zero quotient. Original grades
are never rescaled or replaced by floating-point sentinel values."""
function extended_persistence(f::ExtendedFiltration{T}) where {T}
    check_extended_persistence(f;throw=true)
    reduced=persistence_diagram(f.pair)
    all(isempty,reduced.essential_by_dim) || error("extended filtration failed to terminate at zero")
    rows=[ExtendedInterval{T}[_extended_interval(f,d-1,b,e) for (b,e) in bars]
        for (d,bars) in enumerate(reduced.finite_by_dim)]
    return ExtendedPersistenceDiagram(deepcopy(f),rows)
end
extended_persistence(x,args...;kwargs...)=extended_persistence(extended_filtration(x,args...;kwargs...))

"""`extended_intervals(result; dim, kind=:all)` copies the typed interval
members in a degree; `kind=:extended` selects all three cross-phase types."""
function extended_intervals(d::ExtendedPersistenceDiagram{T};dim::Integer,kind::Symbol=:all) where {T}
    slot=_degree_slot(dim)
    kind in (_EXTENDED_KINDS...,:all,:extended) || throw(ArgumentError("unknown extended interval kind"))
    rows=slot<=length(d.intervals) ? d.intervals[slot] : ExtendedInterval{T}[]
    return [r for r in rows if kind===:all || r.kind===kind ||
        (kind===:extended && r.birth_phase===:ordinary && r.death_phase===:relative)]
end

function extended_persistence_summary(f::ExtendedFiltration)
    return (;kind=:extended_filtration,field=field(f),order=f.order,scalar_events=length(f.values),
        steps=2length(f.values),cells=Tuple(length.(f.pair.indices)),provenance=provenance(f))
end
function extended_persistence_summary(d::ExtendedPersistenceDiagram)
    return merge(extended_persistence_summary(d.filtration),(;kind=:extended_persistence,
        interval_counts=Tuple(length.(d.intervals)),types=NamedTuple{_EXTENDED_KINDS}(
            Tuple(sum(count(r->r.kind===k,rows) for rows in d.intervals) for k in _EXTENDED_KINDS))))
end
describe(x::Union{ExtendedFiltration,ExtendedPersistenceDiagram})=extended_persistence_summary(x)
result_summary(d::ExtendedPersistenceDiagram)=extended_persistence_summary(d)
Base.show(io::IO,f::ExtendedFiltration)=print(io,"ExtendedFiltration(",f.order,", ",field(f),", ",2length(f.values)," steps)")
Base.show(io::IO,d::ExtendedPersistenceDiagram)=print(io,"ExtendedPersistenceDiagram(",filtration_order(d),", ",field(d),", ",sum(length,d.intervals)," intervals)")
Base.show(io::IO,r::ExtendedInterval)=print(io,"ExtendedInterval(H",r.dimension,", ",r.kind,", [",r.birth_phase,":",r.birth,", ",r.death_phase,":",r.death,"))")
describe(r::ExtendedInterval)=(;dimension=r.dimension,kind=r.kind,birth=r.birth,death=r.death,
    birth_phase=r.birth_phase,death_phase=r.death_phase,birth_included=r.birth_included,
    death_included=r.death_included,birth_step=r.birth_step,death_step=r.death_step)
Base.:(==)(a::ExtendedInterval,b::ExtendedInterval)=describe(a)==describe(b)
Base.isequal(a::ExtendedInterval,b::ExtendedInterval)=isequal(describe(a),describe(b))
Base.hash(r::ExtendedInterval,h::UInt)=hash(describe(r),h)

"""Check the two-phase pair and typed interval storage. Set `recompute=true`
to independently compare stored endpoints with a fresh cone reduction."""
function check_extended_persistence(x::Union{ExtendedFiltration,ExtendedPersistenceDiagram};throw::Bool=false,recompute::Bool=false)
    issues=String[]
    try
        f=x isa ExtendedFiltration ? x : x.filtration
        _normalize_order(f.order);check_filtered_pair(f.pair;throw=true)
        n=length(f.values)
        all(isfinite,f.values) && all(i->f.order===:sublevel ? f.values[i]<f.values[i+1] : f.values[i]>f.values[i+1],1:n-1) ||
            Base.throw(ArgumentError("scalar events must be finite and strictly ordered"))
        f.pair.order===:sublevel || Base.throw(ArgumentError("step pair must increase"))
        for d in eachindex(f.pair.indices)
            f.pair.indices[d]==collect(1:length(f.pair.ambient.cells_by_dim[d])) || Base.throw(ArgumentError("final subcomplex must be all K"))
            all(g->n<g<=2n,f.pair.grades[d]) || Base.throw(ArgumentError("relative grades must be in the second phase"))
        end
        all(g->1<=g[1]<=n,f.pair.ambient.grades) || Base.throw(ArgumentError("ambient grades must be in the first phase"))
        if x isa ExtendedPersistenceDiagram
            length(x.intervals)==length(f.pair.indices) || Base.throw(ArgumentError("interval degree count disagrees"))
            for (d,rows) in enumerate(x.intervals), r in rows
                1<=r.birth_step<r.death_step<=2n || Base.throw(ArgumentError("invalid interval steps"))
                r==_extended_interval(f,d-1,r.birth_step,r.death_step) || Base.throw(ArgumentError("interval types or endpoint decorations disagree"))
            end
            recompute && x.intervals!=extended_persistence(f).intervals && Base.throw(ArgumentError("intervals disagree with the retained filtration"))
        end
    catch err
        err isa InterruptException && rethrow()
        push!(issues,sprint(showerror,err))
    end
    throw && !isempty(issues) && Base.throw(ArgumentError(join(issues,"; ")))
    return (;kind=:extended_persistence_validation,valid=isempty(issues),issues=Tuple(issues))
end
"""Readable typed report for `check_extended_persistence`."""
struct ExtendedPersistenceValidationSummary{R<:NamedTuple}
    report::R
end
ExtendedPersistenceValidationSummary(x::Union{ExtendedFiltration,ExtendedPersistenceDiagram};kwargs...)=
    ExtendedPersistenceValidationSummary(check_extended_persistence(x;kwargs...))
describe(s::ExtendedPersistenceValidationSummary)=s.report
Base.show(io::IO,s::ExtendedPersistenceValidationSummary)=print(io,"ExtendedPersistenceValidationSummary(valid=",s.report.valid,", issues=",s.report.issues,")")
Base.:(==)(a::ExtendedPersistenceValidationSummary,b::ExtendedPersistenceValidationSummary)=a.report==b.report
Base.isequal(a::ExtendedPersistenceValidationSummary,b::ExtendedPersistenceValidationSummary)=isequal(a.report,b.report)
Base.hash(s::ExtendedPersistenceValidationSummary,h::UInt)=hash(s.report,h)
