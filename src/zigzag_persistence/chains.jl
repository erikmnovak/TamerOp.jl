# Homological chain complexes and checked simplicial additions/deletions.
struct ZigzagComplex{K,F}
    dims::Vector{Int}
    boundaries::Vector{Matrix{K}}
    labels::Vector{Vector{Int}}
    field::F
end

function ZigzagComplex(dims,boundaries;field=F2(),labels=nothing,
                       budget::ZigzagBudget=ZigzagBudget())
    _exact_field(field)
    isempty(dims) && throw(ArgumentError("a chain complex must include degree zero"))
    _bound(length(dims),big(budget.max_cells)+1,"chain degree slots")
    dd=Int[_dimension(d) for d in dims]
    _bound(sum(big,dims),budget.max_cells,"cells")
    _bound(maximum(dd),budget.max_dimension,"chain dimension")
    labels===nothing && (labels=[collect(1:d) for d in dd])
    length(boundaries)==length(dd)-1 || throw(ArgumentError("one boundary is required between consecutive degrees"))
    length(labels)==length(dd) || throw(ArgumentError("one label vector is required per degree"))
    labs=Vector{Int}[]
    for d in eachindex(dd)
        length(labels[d])==dd[d] && allunique(labels[d]) &&
            all(x->x isa Integer && !(x isa Bool) && typemin(Int)<=x<=typemax(Int),labels[d]) ||
            throw(ArgumentError("chain labels must be distinct integer IDs in each degree"))
        push!(labs,Int.(labels[d]))
    end
    for d in eachindex(boundaries)
        boundaries[d] isa AbstractMatrix && size(boundaries[d])==(dd[d],dd[d+1]) ||
            throw(ArgumentError("boundary shape disagrees in degree $d"))
    end
    _bound(sum((big(dd[d])*dd[d+1] for d in eachindex(boundaries));init=big(0))+
        6big(maximum(dd))^2,budget.max_matrix_entries,"chain algebra entries")
    K=coeff_type(field); bs=Matrix{K}[_matrix(field,A) for A in boundaries]
    for d in 2:length(bs)
        all(iszero,bs[d-1]*bs[d]) || throw(ArgumentError("consecutive boundaries do not compose to zero in degree $d"))
    end
    ZigzagComplex{K,typeof(field)}(dd,bs,labs,field)
end
dimensions(C::ZigzagComplex)=copy(C.dims)
field(C::ZigzagComplex)=C.field
describe(C::ZigzagComplex)=(kind=:zigzag_chain_complex,dimensions=Tuple(C.dims),field=C.field,grading=:homological)
Base.show(io::IO,C::ZigzagComplex)=print(io,"ZigzagComplex(",Tuple(C.dims),", ",C.field,")")
function check_zigzag(C::ZigzagComplex;throw::Bool=false,budget::ZigzagBudget=ZigzagBudget())
    _validation(throw) do
        ZigzagComplex(C.dims,C.boundaries;field=C.field,labels=C.labels,budget)
    end
end
_dim(C::ZigzagComplex,d)=0<=d<length(C.dims) ? C.dims[d+1] : 0
function _boundary(C::ZigzagComplex{K},d) where {K}
    1<=d<length(C.dims) ? C.boundaries[d] : zeros(K,_dim(C,d-1),_dim(C,d))
end

struct _ChainSource{C,M,H,S}
    complexes::C
    maps::M
    homology::H
    degree::Int
    simplices::S
end

function zigzag_persistence(complexes::AbstractVector{<:ZigzagComplex},maps;
                            directions,degree::Integer=0,times=nothing,
                            representatives::Bool=false,budget::ZigzagBudget=ZigzagBudget())
    _chain_persistence(complexes,maps,directions,degree,times,representatives,budget,nothing)
end

function _chain_persistence(complexes,maps,directions,degree,times,reps,budget,simplices)
    degree=_dimension(degree); n=length(complexes)
    degree<typemax(Int) || throw(ArgumentError("homology degree must be smaller than typemax(Int)"))
    _bound(n,budget.max_vertices,"vertices")
    ts=_times(times===nothing ? (1:n) : times,n); dirs=_directions(directions,n)
    length(maps)==n-1 || throw(ArgumentError("one chain map is required per adjacent pair"))
    F=field(first(complexes)); K=coeff_type(F)
    all(C->field(C)==F,complexes) || throw(ArgumentError("all chain complexes must use the same field"))
    cs=ZigzagComplex{K,typeof(F)}[]
    entries=big(0); cells=big(0)
    for C in complexes
        cells+=sum(big,C.dims)
        _bound(cells,budget.max_cells,"total chain cells")
        entries+=sum((big(length(B)) for B in C.boundaries);init=big(0))
        _bound(entries,budget.max_matrix_entries,"retained chain entries")
        push!(cs,ZigzagComplex(C.dims,C.boundaries;field=F,labels=C.labels,budget))
    end
    ms=Vector{Matrix{K}}[]
    for e in eachindex(maps)
        s,t=dirs[e]===:forward ? (e,e+1) : (e+1,e)
        a,b=cs[s],cs[t]; degrees=max(length(a.dims),length(b.dims))
        length(maps[e])==degrees || throw(ArgumentError("chain map $e needs a component in every source/target degree"))
        for d in 0:degrees-1
            A=maps[e][d+1]
            A isa AbstractMatrix && size(A)==(_dim(b,d),_dim(a,d)) ||
                throw(ArgumentError("chain map $e has wrong direction or shape in degree $d"))
            entries+=big(length(A))
        end
        _bound(entries,budget.max_matrix_entries,"retained chain/map entries")
        components=Matrix{K}[_matrix(F,A) for A in maps[e]]
        for d in 1:degrees-1
            _boundary(b,d)*components[d+1]==components[d]*_boundary(a,d) ||
                throw(ArgumentError("chain map $e does not commute with the boundary in degree $d"))
        end
        push!(ms,components)
    end
    homology_entries=sum((6big(_dim(C,degree))^2 for C in cs);init=big(0))
    _bound(entries+homology_entries,budget.max_matrix_entries,"chain and homology entries")
    hs=[ChainComplexes.homology_data(_boundary(C,degree+1),_boundary(C,degree),degree;field=F) for C in cs]
    induced=Matrix{K}[]
    for e in eachindex(ms)
        s,t=dirs[e]===:forward ? (e,e+1) : (e+1,e)
        A=degree<length(ms[e]) ? ms[e][degree+1] : zeros(K,_dim(cs[t],degree),_dim(cs[s],degree))
        push!(induced,ChainComplexes.induced_map_on_homology(hs[s],hs[t],A))
    end
    z=ZigzagModule([H.dimH for H in hs],induced;directions=dirs,field=F,times=ts,budget)
    # Chain data and source identities are opt-in together with representatives.
    source=reps ? _ChainSource(cs,ms,hs,degree,simplices) : (degree=degree,)
    _decompose(z,reps,budget,source)
end

function zigzag_cycle(r::ZigzagResult;interval::Integer,vertex::Integer)
    r.source isa _ChainSource || throw(ArgumentError("compute a chain zigzag with representatives=true to request source cycles"))
    coords=zigzag_representative(r;interval,vertex)
    src=r.source; C=src.complexes[vertex]; d=src.degree
    coeffs=vec(ChainComplexes.homology_representative(src.homology[vertex],coords))
    labels=d<length(C.labels) ? copy(C.labels[d+1]) : Int[]
    cells=src.simplices===nothing ? nothing : deepcopy(src.simplices[vertex][d+1])
    (vertex=Int(vertex),degree=d,labels=labels,coefficients=coeffs,simplices=cells)
end

function zigzag_chain_witness(r::ZigzagResult{K};interval::Integer,edge::Integer) where {K}
    r.source isa _ChainSource || throw(ArgumentError("compute a chain zigzag with representatives=true to request chain witnesses"))
    edge isa Bool && throw(ArgumentError("edge must not be Bool"))
    1<=edge<=length(r.diagram.maps) || throw(ArgumentError("edge outside the zigzag"))
    s,t=r.diagram.directions[edge]===:forward ? (edge,edge+1) : (edge+1,edge)
    x=zigzag_cycle(r;interval,vertex=s)
    a,b=r.supports[interval]; src=r.source; d=src.degree; C=src.complexes[t]
    y=a<=t<=b ? zigzag_cycle(r;interval,vertex=t).coefficients : zeros(K,_dim(C,d))
    A=d<length(src.maps[edge]) ? src.maps[edge][d+1] : zeros(K,_dim(C,d),length(x.coefficients))
    difference=A*x.coefficients-y
    D=_boundary(C,d+1)
    # Obtain an actual filling from pivot columns, not just a zero quotient test.
    _,pivots=FieldLinAlg.rref(field(r),D)
    columns=Int[p for p in pivots]
    image=D[:,columns]
    coordinates=vec(FieldLinAlg.solve_fullcolumn(field(r),image,reshape(difference,:,1)))
    filling=zeros(K,size(D,2)); filling[columns]=coordinates
    D*filling==difference || error("zigzag chain witness failed its boundary equation")
    labels=d+1<length(C.labels) ? copy(C.labels[d+2]) : Int[]
    (source=s,target=t,degree=d,difference=difference,filling=filling,filling_labels=labels)
end

struct ZigzagEvent{T<:Real}
    operation::Symbol
    simplex::Vector{Int}
    time::T
end
function _simplex(vertices)
    !isempty(vertices) && all(v->v isa Integer && !(v isa Bool) && 1<=v<=typemax(Int),vertices) ||
        throw(ArgumentError("a simplex needs positive integer vertex IDs"))
    allunique(vertices) || throw(ArgumentError("a simplex cannot repeat vertices"))
    sort!(Int[v for v in vertices])
end
function ZigzagEvent(operation::Symbol,simplex;time::Real)
    operation in (:add,:delete) || throw(ArgumentError("simplex events must be :add or :delete"))
    isfinite(time) && !(time isa Bool) || throw(ArgumentError("event time must be finite"))
    ZigzagEvent(operation,_simplex(simplex),time)
end
describe(e::ZigzagEvent)=(operation=e.operation,simplex=copy(e.simplex),time=e.time)
Base.show(io::IO,e::ZigzagEvent)=print(io,"ZigzagEvent(",e.operation,", ",e.simplex,", time=",e.time,")")

_faces(s)=(Tuple(s[j] for j in eachindex(s) if j!=i) for i in eachindex(s))

function zigzag_persistence(events::AbstractVector{<:ZigzagEvent};initial=Vector{Int}[],
                            initial_time::Real=isempty(events) ? 0 : first(events).time,
                            degree::Integer=0,field=F2(),representatives::Bool=false,
                            budget::ZigzagBudget=ZigzagBudget())
    _exact_field(field); degree=_dimension(degree)
    _bound(length(events)+1,budget.max_vertices,"vertices")
    times=_times([initial_time;[e.time for e in events]],length(events)+1)
    active=Set{Tuple{Vararg{Int}}}()
    _bound(length(initial),budget.max_cells,"initial cells")
    for vertices in initial
        _bound(length(vertices),budget.max_cells,"simplex vertices")
        s=Tuple(_simplex(vertices))
        s in active && throw(ArgumentError("initial complex repeats a simplex"))
        push!(active,s)
    end
    for s in active
        length(s)==1 || all(f->f in active,_faces(s)) ||
            throw(ArgumentError("initial simplices must include every face"))
    end
    states=[sort!(collect(active);by=s->(length(s),s))]
    cumulative=big(length(active)); _bound(cumulative,budget.max_cells,"event-state cells")
    directions=Symbol[]
    for (e,event) in enumerate(events)
        _bound(length(event.simplex),budget.max_cells,"simplex vertices")
        s=Tuple(_simplex(event.simplex))
        if event.operation===:add
            s in active && throw(ArgumentError("event $e adds an already active simplex"))
            length(s)==1 || all(f->f in active,_faces(s)) ||
                throw(ArgumentError("event $e adds a simplex before its faces"))
            push!(active,s); push!(directions,:forward)
        elseif event.operation===:delete
            s in active || throw(ArgumentError("event $e deletes an inactive simplex"))
            any(t->length(t)>length(s) && all(v->v in t,s),active) &&
                throw(ArgumentError("event $e deletes a simplex with an active coface"))
            delete!(active,s); push!(directions,:backward)
        else
            throw(ArgumentError("event $e has an invalid operation"))
        end
        cumulative+=length(active); _bound(cumulative,budget.max_cells,"event-state cells")
        push!(states,sort!(collect(active);by=s->(length(s),s)))
    end
    # Use a common degree range, retaining every face needed for higher homology.
    maxdegree=max(0,maximum((length(s)-1 for state in states for s in state);init=0))
    grouped=[[[collect(s) for s in state if length(s)==d+1] for d in 0:maxdegree] for state in states]
    allcells=sort!(unique([s for state in states for s in state]);by=s->(length(s),s))
    ids=Dict(s=>i for (i,s) in enumerate(allcells)); K=coeff_type(field)
    cs=ZigzagComplex{K,typeof(field)}[]; entries=big(0)
    for groups in grouped
        dims=length.(groups); bs=Matrix{K}[]
        _bound(maximum(dims),budget.max_dimension,"chain dimension")
        for d in 1:maxdegree
            entries+=big(dims[d])*dims[d+1]
            _bound(entries,budget.max_matrix_entries,"event boundaries")
            B=zeros(K,dims[d],dims[d+1]); rows=Dict(Tuple(s)=>i for (i,s) in enumerate(groups[d]))
            for (j,s) in enumerate(groups[d+1]), (i,f) in enumerate(_faces(s))
                B[rows[f],j]=isodd(i) ? one(K) : -one(K)
            end
            push!(bs,B)
        end
        labels=[[ids[Tuple(s)] for s in group] for group in groups]
        push!(cs,ZigzagComplex(dims,bs;field,labels,budget))
    end
    maps=Vector{Matrix{K}}[]
    for e in eachindex(events)
        s,t=directions[e]===:forward ? (e,e+1) : (e+1,e)
        comps=Matrix{K}[]
        for d in eachindex(cs[s].dims)
            entries+=big(cs[t].dims[d])*cs[s].dims[d]
            _bound(entries,budget.max_matrix_entries,"event boundaries/maps")
            A=zeros(K,cs[t].dims[d],cs[s].dims[d]); rows=Dict(id=>i for (i,id) in enumerate(cs[t].labels[d]))
            for (j,id) in enumerate(cs[s].labels[d]); A[rows[id],j]=one(K); end
            push!(comps,A)
        end
        push!(maps,comps)
    end
    _chain_persistence(cs,maps,directions,degree,times,representatives,budget,grouped)
end
