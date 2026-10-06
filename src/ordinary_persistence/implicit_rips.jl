# Implicit Rips coboundary reduction for barcode requests through H2.
# Reverse filtration order, clearing and reduction-matrix reconstruction follow
# Bauer, Ripser (2021), Sections 3.3-3.5: https://arxiv.org/abs/1908.02518.
# Original/reduced coboundary matrices are never materialized. Internal
# cochains implement the reduction; public retention is an explicit opt-in.

struct _RipsGraph{Complete}
    neighbors::Vector{Vector{Int}}
    distances::Vector{Vector{Float64}}
    edges::Vector{NTuple{2,Int}}
    grades::Vector{Float64}
    binomial::Matrix{Int}
    births::Vector{Float64}
    cutoff::Float64
    # Usually aliases neighbors; a terminal radius supplies a smaller coface view.
    active_neighbors::Vector{Vector{Int}}
end

function _rips_graph(payload, dimension)
    n = payload.n
    kmax = min(dimension + 1, n)
    for k in 1:kmax
        binomial(big(n), k) <= typemax(Int) ||
            throw(ArgumentError("implicit Rips simplex indices overflow Int at dimension $(k-1); reduce the requested degree/point set or use the explicit method."))
    end
    choose = zeros(Int, n + 1, kmax + 1)
    choose[:, 1] .= 1
    for v in 1:n, k in 1:min(v,kmax)
        choose[v+1,k+1] = choose[v,k] + choose[v,k+1]
    end
    complete = length(payload.edges) == div(n * (n - 1), 2)
    if complete
        # Fill sorted rows directly, independent of the supplied edge order.
        # Complete rows omit only their diagonal, so no permutation is needed.
        neighbors = [Vector{Int}(undef,n-1) for _ in 1:n]
        distances = [Vector{Float64}(undef,n-1) for _ in 1:n]
        @inbounds for u in 1:n, i in 1:n-1
            neighbors[u][i] = i + (i >= u)
        end
        @inbounds for (i,(u,v)) in enumerate(payload.edges)
            distances[u][v - (v > u)] = payload.dists[i]
            distances[v][u - (u > v)] = payload.dists[i]
        end
    else
        neighbors = [Int[] for _ in 1:n]
        distances = [Float64[] for _ in 1:n]
        for (i,(u,v)) in enumerate(payload.edges)
            push!(neighbors[u],v); push!(distances[u],payload.dists[i])
            push!(neighbors[v],u); push!(distances[v],payload.dists[i])
        end
        for v in 1:n
            if !issorted(neighbors[v])
                perm = sortperm(neighbors[v])
                neighbors[v] = neighbors[v][perm]
                distances[v] = distances[v][perm]
            end
        end
    end
    return _RipsGraph{complete}(neighbors,distances,payload.edges,payload.dists,choose,get(payload,:births,zeros(n)),Inf,neighbors)
end

@inline function _rips_distance(g::_RipsGraph{false}, u::Int, v::Int)
    ns = g.neighbors[u]
    i = searchsortedfirst(ns,v)
    return i <= length(ns) && ns[i] == v ? g.distances[u][i] : Inf
end

@inline function _rips_distance(g::_RipsGraph{true}, u::Int, v::Int)
    u == v && return Inf
    @inbounds d = g.distances[u][v - (v > u)]
    return d <= g.cutoff ? d : Inf
end

@inline function _rips_index(g::_RipsGraph, vertices::NTuple{N,Int}) where {N}
    id = 1
    @inbounds for j in 1:N
        id += g.binomial[vertices[j],j+1]
    end
    return id
end

# Enumerate only the requested clique size, with no boundary construction and
# no table of its faces. Used for column catalogs and optional budget counting.
function _rips_visit_extensions(f, g, vertices::NTuple{M,Int}, grade, ::Val{N}) where {M,N}
    if M == N
        f(vertices,grade)
        return nothing
    end
    ns = g.neighbors[last(vertices)]
    for i in searchsortedlast(ns,last(vertices))+1:length(ns)
        v = ns[i]
        next_grade = grade
        for u in vertices
            next_grade = max(next_grade,_rips_distance(g,u,v))
        end
        isfinite(next_grade) || continue
        _rips_visit_extensions(f,g,(vertices...,v),next_grade,Val(N))
    end
    return nothing
end

function _rips_visit_cliques(f,g,::Val{N}) where {N}
    for v in eachindex(g.neighbors)
        _rips_visit_extensions(f,g,(v,),g.births[v],Val(N))
    end
    return nothing
end

# Enumerate triangles directly: edge lists already provide two of their grades.
function _rips_visit_cliques(f::F,g,::Val{3}) where {F}
    for u in eachindex(g.neighbors)
        nu=g.neighbors[u]
        @inbounds for j in searchsortedlast(nu,u)+1:length(nu)
            v=nu[j];uv=max(g.births[u],g.distances[u][j]);nv=g.neighbors[v]
            uv <= g.cutoff || continue
            for k in searchsortedlast(nv,v)+1:length(nv)
                g.distances[v][k] <= g.cutoff || continue
                w=nv[k];uw=_rips_distance(g,u,w)
                isfinite(uw) || continue
                f((u,v,w),max(uv,uw,g.distances[v][k]))
            end
        end
    end
    return nothing
end

function _rips_check_expansion_budget(g,payload,dimension)
    budget = DataIngestion._construction_budget(payload.spec)
    n,m = length(g.neighbors),length(g.edges)
    DataIngestion._construction_check_max_simplices!(n,0,payload.spec)
    budget.max_simplices === nothing && budget.memory_budget_bytes === nothing && return nothing
    counts = BigInt[big(n)]
    total = big(n)
    for dim in 1:dimension
        cap = budget.max_simplices === nothing ? nothing : big(budget.max_simplices)-total
        if budget.memory_budget_bytes !== nothing && last(counts) > 0
            memory_cap = div(big(budget.memory_budget_bytes),8last(counts))
            cap = cap === nothing ? memory_cap : min(cap,memory_cap)
        end
        count = 0
        if dim == 1
            count = m
        else
            _rips_visit_cliques(g,Val(dim+1)) do vertices,grade
                count += 1
                (cap === nothing || count <= cap) ||
                    throw(ArgumentError("Rips construction budget exceeded while counting dimension $dim; no cells were discarded."))
            end
        end
        (cap === nothing || count <= cap) ||
            throw(ArgumentError("Rips construction budget exceeded at dimension $dim; no cells were discarded."))
        total += count
        push!(counts,big(count))
    end
    DataIngestion._construction_check_counts!(counts,payload.spec)
    return counts
end

struct _RipsSimplex{N}
    vertices::NTuple{N,Int}
    index::Int
    grade::Float64
end

function _rips_record_catalog(g,::Val{N},cleared) where {N}
    columns = _RipsSimplex{N}[]
    # Complete graphs have a known catalog size before clearing. Avoid repeated
    # growth/copies of the simplex records; sparse graphs keep incremental growth.
    g isa _RipsGraph{true} && isinf(g.cutoff) && sizehint!(columns,
        max(0,binomial(length(g.neighbors),N)-length(cleared)))
    _rips_visit_cliques(g,Val(N)) do vertices,grade
        id = _rips_index(g,vertices)
        id in cleared || push!(columns,_RipsSimplex(vertices,id,grade))
    end
    sort!(columns; by=s -> (s.grade,s.index), rev=true)
    return columns
end

_rips_catalog(g,size::Val,cleared) = _rips_record_catalog(g,size,cleared)

# Recover increasing vertices from the combinatorial (colex) simplex index.
function _rips_vertices(g, index::Int, ::Val{N}) where {N}
    vertices = ntuple(_ -> 0,Val(N))
    rest = index - 1
    upper = length(g.neighbors)
    for k in N:-1:1
        lo,hi = k,upper
        while lo < hi
            mid = (lo + hi + 1) >> 1
            if g.binomial[mid,k+1] <= rest
                lo = mid
            else
                hi = mid - 1
            end
        end
        vertices = Base.setindex(vertices,lo,k)
        rest -= g.binomial[lo,k+1]
        upper = lo - 1
    end
    return vertices
end

# Facets obtained by deleting successive vertices have decreasing colex IDs.
# At nonzero equal grades, the first is the youngest facet. Signed zeros
# require the full catalog order. Some edges have no equal-grade facet.
function _rips_youngest_facet(g, vertices::NTuple{M,Int}, grade) where {M}
    youngest = nothing
    for removed in 1:M
        face = ntuple(i -> vertices[i < removed ? i : i+1],Val(M-1))
        # Match catalog construction exactly, including the sign of zero.
        birth = g.births[first(face)]
        for i in 1:M-1, j in i+1:M-1
            birth = max(birth,_rips_distance(g,face[i],face[j]))
        end
        birth == grade || continue
        candidate = (_RipsSimplex(face,_rips_index(g,face),birth),removed)
        # Nonzero equal Float64 grades have the same ordering bits, so the
        # first equal-grade facet is youngest. Signed zeros need the full
        # column order: +0.0 columns precede -0.0 columns in reverse reduction.
        iszero(grade) || return candidate
        if youngest === nothing || isless((first(youngest).grade,first(youngest).index),
                                           (birth,first(candidate).index))
            youngest = candidate
        end
    end
    return youngest
end

mutable struct _RipsPivots{K,G,C}
    # Positive indices are inline singleton columns; negative indices refer
    # to entries of combinations. No boxed vector is needed for a singleton.
    explicit::Dict{Int,Pair{Int,K}}
    combinations::Vector{Vector{Pair{Int,K}}}
    graph::G
    columns::C
    active::Int
    apparent::Bool
end

function _rips_apparent_reducer(p::_RipsPivots{K}, entry, coface) where {K}
    youngest = _rips_youngest_facet(p.graph,coface,entry.grade)
    youngest === nothing && return nothing
    face,removed = youngest
    # Locate the facet in the actual catalog: clearing can omit a simplex.
    lo,hi = 1,length(p.columns)
    key = (face.grade,face.index)
    while lo <= hi
        mid = (lo+hi) >> 1
        s = p.columns[mid]
        other = (s.grade,s.index)
        if isless(key,other)
            lo = mid+1
        elseif isless(other,key)
            hi = mid-1
        else
            mid < p.active || return nothing
            # Local incidence alone is insufficient: the coface must also be
            # the oldest equal-grade coface of this facet.
            anchor = first(face.vertices)
            for v in face.vertices
                length(p.graph.active_neighbors[v]) < length(p.graph.active_neighbors[anchor]) && (anchor=v)
            end
            for w in p.graph.active_neighbors[anchor]
                w in face.vertices && continue
                all(v -> _rips_distance(p.graph,v,w) <= face.grade,face.vertices) || continue
                pos = 1+count(<(w),face.vertices)
                oldest = ntuple(i -> i < pos ? face.vertices[i] : i == pos ? w : face.vertices[i-1],Val(length(coface)))
                oldest == coface || return nothing
                return mid => (isodd(removed) ? one(K) : -one(K))
            end
            return nothing
        end
    end
    return nothing
end

_rips_claimed(p::AbstractDict,entry,coface,s) = haskey(p,entry.index)
function _rips_claimed(p::_RipsPivots,entry,coface,s)
    youngest = _rips_youngest_facet(p.graph,coface,entry.grade)
    if youngest !== nothing && first(youngest).index == s.index
        # This is the first equal-grade coface, hence an apparent pair. Its
        # normalized singleton can be reconstructed if another column needs it.
        p.apparent = true
        return false
    end
    return haskey(p.explicit,entry.index) || _rips_apparent_reducer(p,entry,coface) !== nothing
end

function _rips_reducer(p::_RipsPivots,entry,::Val{N}) where {N}
    explicit = get(p.explicit,entry.index,nothing)
    if explicit !== nothing
        return first(explicit) > 0 ? explicit : p.combinations[-first(explicit)]
    end
    return _rips_apparent_reducer(p,entry,_rips_vertices(p.graph,entry.index,Val(N+1)))
end

# Pack the exact grade and three 21-bit vertices. Comparing vertex codes
# (largest vertex first) gives colex order without storing a separate index.
# Reading a column needs shifts and three binomial lookups, not unranking.
struct _RipsTriangleCatalog{G} <: AbstractVector{_RipsSimplex{3}}
    keys::Vector{UInt128}
    graph::G
end
Base.size(c::_RipsTriangleCatalog) = size(c.keys)
Base.IndexStyle(::Type{<:_RipsTriangleCatalog}) = IndexLinear()
@inline function _rips_catalog_key(grade::Float64,index::Integer)
    bits = reinterpret(UInt64,grade)
    ordered = bits & (UInt64(1)<<63) == 0 ? xor(bits,UInt64(1)<<63) : ~bits
    return (UInt128(ordered)<<64) | UInt128(index)
end
@inline function Base.getindex(c::_RipsTriangleCatalog,i::Int)
    key = c.keys[i]
    ordered = UInt64(key >> 64)
    bits = ordered & (UInt64(1)<<63) == 0 ? ~ordered : xor(ordered,UInt64(1)<<63)
    code = UInt64(key & typemax(UInt64))
    mask = (UInt64(1)<<21)-1
    vertices = (Int(code & mask),Int((code >> 21) & mask),Int(code >> 42))
    return _RipsSimplex(vertices,_rips_index(c.graph,vertices),reinterpret(Float64,bits))
end
function _rips_catalog(g,::Val{3},cleared)
    # A representation capacity bound, not a performance threshold. Larger
    # vertex labels retain the ordinary record path used by other dimensions.
    length(g.neighbors) < (1<<21) || return _rips_record_catalog(g,Val(3),cleared)
    keys = UInt128[]
    g isa _RipsGraph{true} && isinf(g.cutoff) && sizehint!(keys,
        max(0,binomial(length(g.neighbors),3)-length(cleared)))
    _rips_visit_cliques(g,Val(3)) do vertices,grade
        id = _rips_index(g,vertices)
        if !(id in cleared)
            u,v,w = vertices
            code = UInt64(u) | (UInt64(v)<<21) | (UInt64(w)<<42)
            push!(keys,_rips_catalog_key(grade,code))
        end
    end
    sort!(keys;rev=true)
    return _RipsTriangleCatalog(keys,g)
end

struct _RipsTerm{K}
    index::Int
    grade::Float64
    coefficient::K
end

@inline _rips_term_less(a,b) = a.grade < b.grade || (a.grade == b.grade && a.index < b.index)

function _rips_heap_push!(heap::Vector{T},entry::T) where {T}
    push!(heap,entry)
    i = length(heap)
    @inbounds while i > 1
        parent = i >> 1
        _rips_term_less(entry,heap[parent]) || break
        heap[i] = heap[parent]
        i = parent
    end
    @inbounds heap[i] = entry
    return nothing
end

function _rips_heap_pop!(heap)
    entry, tail = first(heap), pop!(heap)
    isempty(heap) && return entry
    i = 1
    @inbounds while 2i <= length(heap)
        child = 2i
        child < length(heap) && _rips_term_less(heap[child+1],heap[child]) && (child += 1)
        _rips_term_less(heap[child],tail) || break
        heap[i] = heap[child]
        i = child
    end
    @inbounds heap[i] = tail
    return entry
end

function _rips_pop_pivot!(heap)
    while !isempty(heap)
        entry = _rips_heap_pop!(heap)
        c = entry.coefficient
        while !isempty(heap) && first(heap).index == entry.index
            c += _rips_heap_pop!(heap).coefficient
        end
        iszero(c) || return _RipsTerm(entry.index,entry.grade,c)
    end
    return nothing
end

# In characteristic two every nonzero queued coefficient is one. Keep just
# the simplex index and grade; duplicate removals compute their parity.
struct _RipsBinaryTerm
    index::Int
    grade::Float64
end
# Sort each newly generated batch once and cancel equal simplex IDs by parity.
# A heap of run heads merges the remaining terms in the exact pivot order;
# active runs stay immutable. Buffer capacity is reused between columns
# within this query; old queue terms are discarded at column resets.
struct _RipsRunHead
    term::_RipsBinaryTerm
    run::Int
end
@inline _rips_term_less(a::_RipsRunHead,b::_RipsRunHead) = _rips_term_less(a.term,b.term)
mutable struct _RipsBinaryQueue
    pending::Vector{_RipsBinaryTerm}
    # Reusable buffers; positions holds only the active prefix for this column.
    runs::Vector{Vector{_RipsBinaryTerm}}
    positions::Vector{Int}
    heads::Vector{_RipsRunHead}
    count::Int
    hint::Int
end
_RipsBinaryQueue(hint::Int) = _RipsBinaryQueue(sizehint!(_RipsBinaryTerm[],hint),Vector{_RipsBinaryTerm}[],Int[],_RipsRunHead[],0,hint)
Base.length(q::_RipsBinaryQueue) = q.count
Base.isempty(q::_RipsBinaryQueue) = q.count == 0
function Base.empty!(q::_RipsBinaryQueue)
    empty!(q.positions);empty!(q.heads);q.count=0
    isempty(q.runs) || (q.pending=q.runs[1])
    empty!(q.pending)
    return q
end
function _rips_heap_push!(q::_RipsBinaryQueue,entry::_RipsBinaryTerm)
    # Reserve a new slot only on first use; later columns reuse its capacity.
    if length(q.positions)==length(q.runs)
        sizehint!(q.pending,q.hint)
        push!(q.runs,q.pending)
    end
    push!(q.pending,entry);q.count+=1
    return nothing
end
function _rips_heap_push!(q::_RipsBinaryQueue,entry::_RipsTerm{FpElem{2}})
    iszero(entry.coefficient) || _rips_heap_push!(q,_RipsBinaryTerm(entry.index,entry.grade))
    return nothing
end
function _rips_compact_run!(run::Vector{_RipsBinaryTerm})
    output=0;i=1
    @inbounds while i<=length(run)
        entry=run[i];odd=true;i+=1
        while i<=length(run) && run[i].index==entry.index
            odd=!odd;i+=1
        end
        if odd;output+=1;run[output]=entry;end
    end
    resize!(run,output)
end
function _rips_flush!(q::_RipsBinaryQueue)
    isempty(q.pending) && return nothing
    run=q.pending;before=length(run)
    # Equal grade/index keys are identical F2 terms; stability is unnecessary.
    sort!(run;alg=Base.Sort.QuickSort,lt=_rips_term_less)
    _rips_compact_run!(run)
    q.count-=before-length(run)
    if isempty(run);return nothing;end
    push!(q.positions,1)
    _rips_heap_push!(q.heads,_RipsRunHead(first(run),length(q.positions)))
    next=length(q.positions)+1
    q.pending=next<=length(q.runs) ? empty!(q.runs[next]) : _RipsBinaryTerm[]
    return nothing
end
function _rips_heap_pop!(q::_RipsBinaryQueue)
    head=first(q.heads);id=head.run;position=q.positions[id]+1;q.count-=1
    if position>length(q.runs[id])
        _rips_heap_pop!(q.heads)
    else
        q.positions[id]=position
        tail=_RipsRunHead(q.runs[id][position],id)
        i=1
        @inbounds while 2i<=length(q.heads)
            child=2i
            child<length(q.heads) && _rips_term_less(q.heads[child+1],q.heads[child]) && (child+=1)
            _rips_term_less(q.heads[child],tail) || break
            q.heads[i]=q.heads[child];i=child
        end
        @inbounds q.heads[i]=tail
    end
    return head.term
end
function _rips_pop_pivot!(q::_RipsBinaryQueue)
    _rips_flush!(q)
    while !isempty(q)
        entry=_rips_heap_pop!(q);odd=true
        while !isempty(q) && first(q.heads).term.index==entry.index
            _rips_heap_pop!(q);odd=!odd
        end
        odd && return _RipsTerm(entry.index,entry.grade,one(FpElem{2}))
    end
    return nothing
end

function _rips_heap_push!(heap::Vector{_RipsBinaryTerm},entry::_RipsTerm{FpElem{2}})
    iszero(entry.coefficient) || _rips_heap_push!(heap,_RipsBinaryTerm(entry.index,entry.grade))
    return nothing
end
function _rips_pop_pivot!(heap::Vector{_RipsBinaryTerm})
    while !isempty(heap)
        entry=_rips_heap_pop!(heap)
        odd=true
        while !isempty(heap) && first(heap).index==entry.index
            _rips_heap_pop!(heap);odd=!odd
        end
        odd && return _RipsTerm(entry.index,entry.grade,one(FpElem{2}))
    end
    return nothing
end

# Use the compact term heap when terminal pruning shortens neighbor lists.
# Sorted runs are reserved for unpruned coface searches.
function _rips_reduction_queue(g,::Type{FpElem{2}})
    pruned=any(i -> length(g.active_neighbors[i]) < length(g.neighbors[i]),eachindex(g.neighbors))
    return pruned ? _RipsBinaryTerm[] : _RipsBinaryQueue(maximum(length,g.active_neighbors;init=0))
end
_rips_reduction_queue(g,::Type{K}) where {K} = _RipsTerm{K}[]

mutable struct _RipsReductionStats
    cofacet_visits::Int
    column_additions::Int
    emergent_pairs::Int
    peak_heap_terms::Int
    peak_reduction_terms::Int
    largest_column_catalog::Int
    apparent_pairs::Int
end
_RipsReductionStats() = _RipsReductionStats(0,0,0,0,0,0,0)

# All cofaces are generated from the selected graph. Inserting vertices in
# increasing order increases the colex index. Hence the first coface at the
# column's own grade is its earliest possible pivot. If unclaimed, reduction
# may stop immediately (a zero-length emergent pair).
function _rips_coboundary!(heap,g,s::_RipsSimplex{N},scale::K,pivots,stats;
                            shortcut::Bool=false) where {N,K}
    vertices = s.vertices
    anchor = first(vertices)
    for v in vertices
        length(g.active_neighbors[v]) < length(g.active_neighbors[anchor]) && (anchor=v)
    end
    if shortcut
        # Look for the first equal-grade coface before filling the queue:
        # an emergent pair discards every higher-grade coface. If this first
        # candidate is claimed, perform the full reduction as usual.
        for w in g.active_neighbors[anchor]
            w in vertices && continue
            grade = s.grade
            same_grade = true
            for v in vertices
                distance = _rips_distance(g,v,w)
                if distance > s.grade
                    same_grade = false
                    break
                end
                # Preserve signed-zero grades even though equality is numerical.
                grade = max(grade,distance)
            end
            same_grade || continue
            pos = 1 + count(v -> v < w,vertices)
            coface = ntuple(i -> i < pos ? vertices[i] : i == pos ? w : vertices[i-1],Val(N+1))
            entry = _RipsTerm(_rips_index(g,coface),grade,isodd(pos) ? scale : -scale)
            stats.cofacet_visits += 1
            if !_rips_claimed(pivots,entry,coface,s)
                empty!(heap)
                stats.emergent_pairs += 1
                return entry
            end
            break
        end
    end
    for w in g.active_neighbors[anchor]
        w in vertices && continue
        grade = s.grade
        for v in vertices
            grade = max(grade,_rips_distance(g,v,w))
        end
        isfinite(grade) || continue
        pos = 1 + count(v -> v < w,vertices)
        coface = ntuple(i -> i < pos ? vertices[i] : i == pos ? w : vertices[i-1],Val(N+1))
        coefficient = isodd(pos) ? scale : -scale
        entry = _RipsTerm(_rips_index(g,coface),grade,coefficient)
        stats.cofacet_visits += 1
        _rips_heap_push!(heap,entry)
        stats.peak_heap_terms = max(stats.peak_heap_terms,length(heap))
    end
    return nothing
end

function _rips_cocycle(columns,change,birth,death,source_indices)
    entries = sort!(collect(change);by=e -> columns[first(e)].index)
    cells = [columns[first(e)] for e in entries]
    indices = [s.index for s in cells]
    cochain = _PersistenceChain(indices,copy(indices),[s.grade for s in cells],
        Int[last(e).val for e in entries])
    vertices = [source_indices === nothing ? collect(s.vertices) :
        Int[source_indices[v] for v in s.vertices] for s in cells]
    return _PersistenceCocycle{Float64}(birth,death,cochain,vertices)
end

function _rips_reduce_degree!(finite,essential,g,columns,::Type{K},stats,
        retained=nothing,dim=0,source_indices=nothing; need_clearing::Bool=true) where {K}
    # Specialize the reduction on the chosen concrete queue outside its loop.
    return _rips_reduce_with_queue!(_rips_reduction_queue(g,K),finite,essential,
        g,columns,K,stats,retained,dim,source_indices;need_clearing)
end

function _rips_reduce_with_queue!(heap,finite,essential,g,columns,::Type{K},stats,
        retained,dim,source_indices;need_clearing::Bool) where {K}
    # A stored change vector v represents a reduced column delta(v), normalized
    # to have pivot coefficient one. Reconstruct delta(v) only when needed.
    pivots = _RipsPivots(Dict{Int,Pair{Int,K}}(),Vector{Pair{Int,K}}[],g,columns,0,false)
    cleared = need_clearing ? Set{Int}() : nothing
    change = Dict{Int,K}()
    stored_terms = 0
    for (j,s) in enumerate(columns)
        empty!(heap); empty!(change)
        expanded = false
        pivots.active = j
        pivots.apparent = false
        pivot = _rips_coboundary!(heap,g,s,one(K),pivots,stats;shortcut=true)
        if pivot === nothing
            pivot = _rips_pop_pivot!(heap)
        end
        apparent = pivots.apparent
        stored = pivot === nothing || apparent ? nothing : _rips_reducer(pivots,pivot,Val(length(s.vertices)))
        while stored !== nothing
            _rips_heap_push!(heap,pivot)
            scale = -pivot.coefficient
            if !expanded
                change[j] = one(K)
                expanded = true
            end
            terms = stored isa Pair ? (stored,) : stored
            for (i,c) in terms
                a = scale*c
                value = get(change,i,zero(K)) + a
                iszero(value) ? delete!(change,i) : (change[i]=value)
                _rips_coboundary!(heap,g,columns[i],a,pivots,stats)
            end
            stats.column_additions += 1
            pivot = _rips_pop_pivot!(heap)
            stored = pivot === nothing ? nothing : _rips_reducer(pivots,pivot,Val(length(s.vertices)))
        end
        if pivot === nothing
            push!(essential,s.grade)
            retained === nothing || push!(retained.essential[dim+1],
                _rips_cocycle(columns,expanded ? change : (j=>one(K),),s.grade,nothing,source_indices))
        else
            pivot.grade >= s.grade || error("implicit Rips pivot precedes its birth.")
            need_clearing && push!(cleared,pivot.index)
            if apparent
                stats.apparent_pairs += 1
                continue
            end
            normalization = inv(pivot.coefficient)
            v = expanded ? sort!([i => normalization*c for (i,c) in change];by=first) : j=>normalization
            if v isa Pair
                pivots.explicit[pivot.index] = v
            else
                push!(pivots.combinations,v)
                pivots.explicit[pivot.index] = -length(pivots.combinations)=>zero(K)
            end
            stored_terms += v isa Pair ? 1 : length(v)
            stats.peak_reduction_terms = max(stats.peak_reduction_terms,stored_terms)
            if s.grade != pivot.grade
                push!(finite,(s.grade,pivot.grade))
                retained === nothing || push!(retained.finite[dim+1],
                    _rips_cocycle(columns,v isa Pair ? (v,) : v,s.grade,pivot.grade,source_indices))
            end
        end
    end
    return cleared
end

@inline function _rips_find!(parents,v)
    while parents[v] != v
        parents[v] = parents[parents[v]]
        v = parents[v]
    end
    return v
end

function _rips_h0!(finite,essential,g,columns=_rips_catalog(g,Val(2),Set{Int}()))
    parents = collect(eachindex(g.neighbors))
    sizes = ones(Int,length(parents))
    cleared = Set{Int}()
    births = copy(g.births)
    for s in Iterators.reverse(columns)
        u,v = s.vertices
        a,b = _rips_find!(parents,u),_rips_find!(parents,v)
        a == b && continue
        sizes[a] < sizes[b] && ((a,b)=(b,a))
        parents[b] = a; sizes[a] += sizes[b]
        younger = max(births[a],births[b])
        births[a] = min(births[a],births[b])
        push!(cleared,s.index)
        younger == s.grade || push!(finite,(younger,s.grade))
    end
    append!(essential,(births[v] for v in eachindex(parents) if parents[v]==v))
    return cleared
end

function _implicit_rips_diagram(data,filtration,field,degree,requested,::Val{Keep}=Val(false); landmarks=nothing) where {Keep}
    max_dim = Int(get(DataIngestion.filtration_parameters(filtration),:max_dim,1))
    effective_dim = min(max_dim,requested === nothing ? degree : degree+1)
    payload = DataIngestion._rips_persistence_graph(data,filtration,effective_dim)
    if landmarks !== nothing
        payload = merge(payload,(source_indices=landmarks.indices,original_n=length(landmarks.nearest_distances)))
    end
    effective_dim = max(0,min(effective_dim,payload.n-1))
    g = _rips_graph(payload,effective_dim)
    budget_counts = _rips_check_expansion_budget(g,payload,effective_dim)
    # At the enclosing radius a universal vertex makes the flag complex a cone.
    # This preserves bounded barcode requests, not top-dimensional truncations.
    # Keep the full graph for cocycle retention and validate user budgets first.
    terminal = nothing
    if !Keep && requested !== nothing && degree < effective_dim &&
            DataIngestion.filtration_kind(filtration) !== :edge_weighted &&
            g isa _RipsGraph{true} && payload.n > 1
        radius, apex = findmin(maximum(row) for row in g.distances)
        terminal = (radius=radius,apex=apex,certificate=:universal_vertex_cone)
        # Preserve the original neighbor/distance row alignment for catalogs.
        # Coface searches can reuse a smaller view without testing every
        # excluded neighbor on each reduction step.
        active = [Int[v for v in g.neighbors[u] if _rips_distance(g,u,v) <= radius]
            for u in eachindex(g.neighbors)]
        g = _RipsGraph{true}(g.neighbors,g.distances,g.edges,g.grades,
            g.binomial,g.births,radius,active)
    end
    finite = [Tuple{Float64,Float64}[] for _ in 0:degree]
    essential = [Float64[] for _ in 0:degree]
    stats = _RipsReductionStats()
    retained = Keep ? _PersistenceCocycles([_PersistenceCocycle{Float64}[] for _ in 0:degree],
        [_PersistenceCocycle{Float64}[] for _ in 0:degree], :selected_vertex_colex_simplices) : nothing
    if Keep
        cleared = Set{Int}()
        for d in 0:min(degree,effective_dim)
            columns = _rips_catalog(g,Val(d+1),cleared)
            stats.largest_column_catalog = max(stats.largest_column_catalog,length(columns))
            if d == effective_dim
                for (j,s) in enumerate(columns)
                    push!(essential[d+1],s.grade)
                    push!(retained.essential[d+1],_rips_cocycle(columns,
                        [j=>one(coeff_type(field))],s.grade,nothing,payload.source_indices))
                end
            else
                cleared = _rips_reduce_degree!(finite[d+1],essential[d+1],g,columns,
                    coeff_type(field),stats,retained,d,payload.source_indices;
                    need_clearing=d < min(degree,effective_dim))
            end
        end
    elseif effective_dim == 0
        append!(essential[1],g.births)
    else
        edges = _rips_catalog(g,Val(2),Set{Int}())
        stats.largest_column_catalog = length(edges)
        cleared = _rips_h0!(finite[1],essential[1],g,edges)
        # Capture this set by value; later degrees replace `cleared`.
        let h0_cleared = cleared
            filter!(s -> !(s.index in h0_cleared),edges)
        end
        for d in 1:min(degree,effective_dim)
            columns = d == 1 ? edges : _rips_catalog(g,Val(d+1),cleared)
            stats.largest_column_catalog = max(stats.largest_column_catalog,length(columns))
            if d == effective_dim
                append!(essential[d+1],(s.grade for s in columns))
            else
                cleared = _rips_reduce_degree!(finite[d+1],essential[d+1],g,columns,coeff_type(field),stats;
                    need_clearing=d < min(degree,effective_dim))
            end
        end
    end
    if Keep
        _sort_cocycles!(finite,essential,retained,:sublevel)
    else
        foreach(sort!,finite); foreach(sort!,essential)
    end
    selection = DataIngestion.filtration_kind(filtration) === :edge_weighted ? :none_in_reduction :
                payload.source_indices !== nothing ? :landmark_restriction :
                payload.graph_selection === :knn ? :neighbor_graph : :none_in_reduction
    meta = (backend=:implicit_rips_cohomology,
        construction=(requested=DataIngestion.filtration_kind(filtration),effective=:rips_flag_graph,substitution=:none),
        grade_arithmetic=:Float64, approximation=selection, discretization=:none,
        chain_validation=:oriented_flag_complex,
        computation=(method=:implicit,homology_through=degree,
            simplex_dimension_used=effective_dim,clearing=true,emergent_pairs=true,
            boundary_matrix_materialized=false,reduced_coboundary_matrix_retained=false,
            cross_call_mathematical_cache=false,terminal_radius=terminal),
        input_selection=(original_vertices=payload.original_n,vertices=payload.n,
            source_indices=payload.source_indices,graph_selection=payload.graph_selection,
            input_edges=payload.input_edges,retained_edges=count(<=(g.cutoff),g.grades),collapse=payload.collapse),
        reduction_stats=(cofacet_visits=stats.cofacet_visits,column_additions=stats.column_additions,
            emergent_pairs=stats.emergent_pairs,apparent_pairs=stats.apparent_pairs,peak_heap_terms=stats.peak_heap_terms,
            peak_reduction_terms=stats.peak_reduction_terms,largest_column_catalog=stats.largest_column_catalog),
        budget_counts=budget_counts)
    return PersistenceDiagram(finite,essential,field,:sublevel,meta,nothing,retained)
end
