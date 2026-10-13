# Local vineyard repair and transactional edits of a retained reduction.
# R=DV transpositions: Cohen-Steiner, Edelsbrunner, Morozov (2006), and
# Morozov--Patel, Appendix A: https://www.mrzv.org/publications/2d-persistence-computation/manuscript/
# Only the columns affected by triangularity/pivot collisions are reduced.

mutable struct _PersistenceEdit{T,K}
    old_n::Int
    old_next::Int
    old_permutation::Vector{Int}
    old_grades::Union{Nothing,Vector{T}}
    boundary::Dict{Int,Vector{Pair{Int,K}}}
    reduced::Dict{Int,Vector{Pair{Int,K}}}
    change::Dict{Int,Vector{Pair{Int,K}}}
    additions::Int
    columns::Int
    swaps::Int
    old_retained::Bool
end
function _PersistenceEdit(s::PersistenceState{T,K},kind) where {T,K}
    _PersistenceEdit{T,K}(length(s.ids),s.next_id,
        kind===:append ? Int[] : copy(s.permutation),kind===:grades ? copy(s.grades) : nothing,
        Dict{Int,Vector{Pair{Int,K}}}(),Dict{Int,Vector{Pair{Int,K}}}(),
        Dict{Int,Vector{Pair{Int,K}}}(),0,0,0,s.retained)
end
function _incremental_replace!(s,tx,j,r,v)
    count=s.entries-length(s.reduced[j])-length(s.change[j])+length(r)+length(v)
    count<=s.budget.max_entries || throw(ArgumentError("update exceeds max_entries"))
    for (name,rows,new) in ((:reduced,s.reduced_rows,r),(:change,s.change_rows,v))
        columns=getfield(s,name);old=columns[j];old===new && continue
        j<=tx.old_n && get!(getfield(tx,name),j,old)
        for e in old;delete!(rows[first(e)],j);end
        for e in new;push!(rows[first(e)],j);end
        columns[j]=new
    end
    s.entries=count
end
function _incremental_add!(s::PersistenceState{T,K},tx,j,k,a::K) where {T,K}
    iszero(a) && return
    tx.additions+=1
    tx.additions<=s.budget.max_column_additions || throw(ArgumentError("update exceeds max_column_additions"))
    # Bound scratch before combining columns; each output support is a union.
    length(s.reduced[j])+length(s.reduced[k])+length(s.change[j])+length(s.change[k])<=s.budget.max_entries ||
        throw(ArgumentError("update scratch exceeds max_entries"))
    r=_prime_add_scaled!(Pair{Int,K}[],s.reduced[j],s.reduced[k],a,s.rank)
    v=_prime_add_scaled!(Pair{Int,K}[],s.change[j],s.change[k],a,s.rank)
    _incremental_replace!(s,tx,j,r,v)
end
function _incremental_rollback!(s,tx)
    for name in (:ids,:labels,:dims,:grades,:rank,:boundary)
        resize!(getfield(s,name),tx.old_n)
    end
    if tx.old_retained
        resize!(s.reduced,tx.old_n);resize!(s.change,tx.old_n)
    else
        _incremental_drop_reduction!(s)
    end
    for name in (:boundary,:reduced,:change),(j,col) in getfield(tx,name)
        (name===:boundary || tx.old_retained) && (getfield(s,name)[j]=col)
    end
    if isempty(tx.old_permutation) && tx.old_n>0
        resize!(s.permutation,tx.old_n) # append-only transaction
    else
        s.permutation=tx.old_permutation
    end
    tx.old_grades===nothing || (s.grades=tx.old_grades)
    s.next_id=tx.old_next
    _incremental_reindex!(s)
end

function _incremental_local_pairs(s,a,b)
    births=Int[]
    for j in (a,b)
        p=isempty(s.reduced[j]) ? j : _incremental_low(s.reduced[j])
        p in births || push!(births,p)
    end
    [_incremental_pair(s,p) for p in births]
end
function _incremental_continuation(before,after,cells)
    used=falses(length(after));result=Tuple{Int,Int}[]
    for p in before
        candidates=Int[j for (j,q) in enumerate(after) if !used[j] && q.dimension==p.dimension &&
            (p.death_cell!=0 && !(p.death_cell in cells) ? q.death_cell==p.death_cell :
             !(p.birth_cell in cells) ? q.birth_cell==p.birth_cell : q==p)]
        if isempty(candidates)
            candidates=Int[j for (j,q) in enumerate(after) if !used[j] && q.dimension==p.dimension &&
                p.death_cell==0 && q.death_cell==0]
        end
        length(candidates)==1 || error("vineyard repair did not determine a local continuation")
        j=only(candidates);used[j]=true;push!(result,(p.birth_cell,after[j].birth_cell))
    end
    all(used) || error("vineyard repair changed the number of local pairs")
    result
end

function _incremental_repair!(s,tx,initial)
    queue=unique(filter(!=(0),initial))
    while !isempty(queue)
        sort!(queue;by=j->s.rank[j]);j=popfirst!(queue);tx.columns+=1
        while !isempty(s.reduced[j])
            p=_incremental_low(s.reduced[j]);k=s.pivot[p]
            if k==0 || k==j
                s.pivot[p]=j;break
            elseif s.rank[k]>s.rank[j]
                s.pivot[p]=j;k in queue || push!(queue,k);break
            end
            a=-last(last(s.reduced[j]))/last(last(s.reduced[k]))
            _incremental_add!(s,tx,j,k,a)
        end
    end
end

function _incremental_transpose!(s,tx,position;time=nothing,tie=:algebraic_edit)
    i=position;a,b=s.permutation[i],s.permutation[i+1]
    any(e->first(e)==a,s.boundary[b]) && throw(ArgumentError("cannot transpose a cell with its boundary face"))
    tx.swaps+=1;tx.swaps<=s.budget.max_swaps || throw(ArgumentError("update exceeds max_swaps"))
    before=_incremental_local_pairs(s,a,b);cells=(s.ids[a],s.ids[b])
    affected=unique(filter(!=(0),[a,b,s.pivot[a],s.pivot[b]]))
    for j in affected
        p=_incremental_low(s.reduced[j]);p==0 || (s.pivot[p]=0)
    end
    # Remove the only V entry which could move below the diagonal.
    entry=_incremental_entry(s.change[b],a)
    iszero(entry) || _incremental_add!(s,tx,b,a,-entry/_incremental_entry(s.change[a],a))
    s.permutation[i],s.permutation[i+1]=b,a;s.rank[a],s.rank[b]=i+1,i
    # Stable row IDs need no relabelling. Only columns containing both rows
    # need their two adjacent sparse entries exchanged.
    for (name,rows) in ((:reduced,s.reduced_rows),(:change,s.change_rows))
        for j in intersect(rows[a],rows[b])
            col=copy(getfield(s,name)[j])
            x=findfirst(e->first(e)==a,col);y=findfirst(e->first(e)==b,col)
            col[x],col[y]=col[y],col[x]
            _incremental_replace!(s,tx,j,name===:reduced ? col : s.reduced[j],name===:change ? col : s.change[j])
        end
    end
    _incremental_repair!(s,tx,affected)
    after=_incremental_local_pairs(s,a,b)
    continuation=tie===:algebraic_edit ? Tuple{Int,Int}[] : _incremental_continuation(before,after,cells)
    VineyardEvent(cells,time,before,after,continuation,tie)
end

function _incremental_append!(s::PersistenceState{T,K},tx,record;strict) where {T,K}
    record isa NamedTuple || throw(ArgumentError("new cells must be named tuples"))
    all(k->k in (:dimension,:grade,:boundary,:id,:label),keys(record)) &&
        all(k->haskey(record,k),(:dimension,:grade,:boundary)) ||
        throw(ArgumentError("a cell needs dimension, grade and boundary; optional id and label"))
    n=length(s.ids);n<s.budget.max_cells || throw(ArgumentError("active cells exceed max_cells"))
    id=_incremental_integer(get(record,:id,s.next_id),:id)
    id>=s.next_id || throw(ArgumentError("new cell IDs must be at least $(s.next_id); IDs are never reused"))
    d=_incremental_integer(record.dimension,:dimension;lower=0)
    d<=s.budget.max_cells || throw(ArgumentError("cell dimension exceeds max_cells"))
    grade=_incremental_grade(T,record.grade)
    label=get(record,:label,id);label isa Integer && !(label isa Bool) && typemin(Int)<=label<=typemax(Int) ||
        throw(ArgumentError("label must fit Int"))
    if strict && !isempty(s.permutation)
        j=last(s.permutation)
        (_incremental_before(s,s.grades[j],grade) &&
            (s.grades[j]!=grade || (s.dims[j],s.ids[j])<(d,id))) ||
            throw(ArgumentError("append must follow the current filtration order; use insert for an earlier cell"))
    end
    accum=Dict{Int,K}()
    for e in record.boundary
        e isa Pair || throw(ArgumentError("boundary entries must be cell ID => integer coefficient"))
        first(e) isa Integer && !(first(e) isa Bool) || throw(ArgumentError("boundary IDs must be integers"))
        last(e) isa Integer && !(last(e) isa Bool) || throw(ArgumentError("boundary coefficients must be integers"))
        j=_incremental_slot(s,first(e));v=K(last(e))
        iszero(v) && continue
        s.dims[j]==d-1 || throw(ArgumentError("boundary must lower dimension by one"))
        _incremental_before(s,s.grades[j],grade) || throw(ArgumentError("cell enters before its boundary"))
        accum[j]=get(accum,j,zero(K))+v
        length(accum)<=s.budget.max_entries || throw(ArgumentError("boundary exceeds max_entries"))
    end
    col=Pair{Int,K}[j=>v for (j,v) in accum if !iszero(v)];sort!(col;by=e->s.rank[first(e)])
    dd=Dict{Int,K}()
    for e in col,f in s.boundary[first(e)]
        j=first(f);dd[j]=get(dd,j,zero(K))+last(e)*last(f)
        length(dd)<=s.budget.max_entries || throw(ArgumentError("double boundary scratch exceeds max_entries"))
    end
    all(iszero,values(dd)) || throw(ArgumentError("new cell has nonzero double boundary"))
    s.entries+(s.retained ? 2length(col)+1 : length(col))<=s.budget.max_entries || throw(ArgumentError("new cell exceeds max_entries"))
    push!(s.ids,id);push!(s.labels,Int(label));push!(s.dims,d);push!(s.grades,grade)
    s.slots[id]=n+1;s.next_id=id+1;push!(s.permutation,n+1);push!(s.rank,n+1)
    push!(s.boundary,col);push!(s.cofaces,Set{Int}())
    for e in col;push!(s.cofaces[first(e)],n+1);end
    s.entries+=length(col)
    if s.retained
        push!(s.reduced,copy(col));push!(s.change,Pair{Int,K}[(n+1)=>one(K)])
        push!(s.reduced_rows,Set{Int}());push!(s.change_rows,Set{Int}([n+1]));push!(s.pivot,0)
        for e in col;push!(s.reduced_rows[first(e)],n+1);end
        s.entries+=length(col)+1
    end
    if !strict
        position=findfirst(j->_incremental_less(s,n+1,j),s.permutation)
        if position!==nothing
            pop!(s.permutation);insert!(s.permutation,position,n+1)
            for i in position:length(s.permutation);s.rank[s.permutation[i]]=i;end
        end
    end
    # Existing columns never contain the new row. Their relative order and
    # triangularity survive insertion; only new/displaced pivots need repair.
    s.retained && _incremental_repair!(s,tx,[n+1])
    n+1
end

function _incremental_delete!(s,tx,j)
    isempty(s.cofaces[j]) || throw(ArgumentError("cell still has a coface"))
    affected=Int[]
    if s.retained
        affected=sort!(collect(setdiff(s.change_rows[j],Set([j])));by=k->s.rank[k])
        for k in Iterators.flatten(((j,),affected))
            p=_incremental_low(s.reduced[k]);p==0 || (s.pivot[p]=0)
        end
        # A maximal cell has zero boundary row. Clear its basis coordinate from
        # later columns before dropping it, then repair their reduced pivots.
        diagonal=_incremental_entry(s.change[j],j)
        for k in affected
            a=-_incremental_entry(s.change[k],j)/diagonal
            _incremental_add!(s,tx,k,j,a)
        end
    end
    get!(tx.boundary,j,s.boundary[j])
    for e in s.boundary[j];delete!(s.cofaces[first(e)],j);end
    s.entries-=length(s.boundary[j]);s.boundary[j]=empty(s.boundary[j])
    s.retained && _incremental_replace!(s,tx,j,empty(s.reduced[j]),empty(s.change[j]))
    position=s.rank[j];deleteat!(s.permutation,position);s.rank[j]=0
    for i in position:length(s.permutation);s.rank[s.permutation[i]]=i;end
    s.retained && _incremental_repair!(s,tx,affected)
end
function _incremental_compact!(s)
    n=length(s.ids);m=length(s.permutation);m==n && return
    # Fill deleted slots with surviving tail cells. Copy only columns containing
    # a moved row; unaffected sparse columns and incidence sets are retained.
    holes=Int[j for j in 1:m if s.rank[j]==0]
    tail=Int[j for j in m+1:n if s.rank[j]>0]
    for j in 1:n;s.rank[j]==0 && delete!(s.slots,s.ids[j]);end
    remap=Dict(zip(tail,holes))
    for (old,new) in zip(tail,holes)
        for (name,rows) in ((:boundary,s.cofaces),(:reduced,s.reduced_rows),(:change,s.change_rows))
            name===:boundary || s.retained || continue
            columns=getfield(s,name)
            for j in rows[old]
                columns[j]=[first(e)==old ? new=>last(e) : e for e in columns[j]]
            end
            rows[new]=rows[old]
            for e in columns[old]
                delete!(rows[first(e)],old);push!(rows[first(e)],new)
            end
            columns[new]=columns[old]
        end
        for name in (:ids,:labels,:dims,:grades);getfield(s,name)[new]=getfield(s,name)[old];end
        s.slots[s.ids[new]]=new
    end
    for name in (:ids,:labels,:dims,:grades,:rank,:boundary,:cofaces)
        resize!(getfield(s,name),m)
    end
    if s.retained
        for name in (:reduced,:change,:reduced_rows,:change_rows,:pivot)
            resize!(getfield(s,name),m)
        end
    end
    for i in eachindex(s.permutation)
        j=get(remap,s.permutation[i],s.permutation[i]);s.permutation[i]=j;s.rank[j]=i
    end
    fill!(s.pivot,0)
    for (j,c) in enumerate(s.reduced)
        p=_incremental_low(c);p==0 || (s.pivot[p]=j)
    end
    s
end

# A bounded kinetic heap. Stale neighbor events are discarded by adjacency and
# final-order checks. Exact crossing times avoid epsilon-dependent tie paths.
function _vineyard_push!(heap,item)
    push!(heap,item);i=length(heap)
    while i>1
        p=i>>1;heap[p]<=item && break
        heap[i]=heap[p];i=p
    end
    heap[i]=item
end
function _vineyard_pop!(heap)
    result=first(heap);lastitem=pop!(heap);isempty(heap) && return result
    i=1
    while 2i<=length(heap)
        j=2i;j<length(heap) && heap[j+1]<heap[j] && (j+=1)
        lastitem<=heap[j] && break
        heap[i]=heap[j];i=j
    end
    heap[i]=lastitem;result
end
function _vineyard_rational(x)
    x isa Union{Integer,Rational,AbstractFloat} ||
        throw(ArgumentError("linear vineyard updates need integer, rational or floating grades with exact rational conversion"))
    Rational{BigInt}(x)
end
function _incremental_target_grades(s::PersistenceState{T},input) where {T}
    target=copy(s.grades);seen=Set{Int}()
    for e in input
        e isa Pair || throw(ArgumentError("grades must give stable cell ID => value pairs"))
        j=_incremental_slot(s,first(e));j in seen && throw(ArgumentError("duplicate grade update"))
        push!(seen,j);target[j]=_incremental_grade(T,last(e))
    end
    for j in seen
        for e in s.boundary[j]
            _incremental_before(s,target[first(e)],target[j]) || throw(ArgumentError("updated grades violate the filtration"))
        end
        for k in s.cofaces[j]
            _incremental_before(s,target[j],target[k]) || throw(ArgumentError("updated grades violate the filtration"))
        end
    end
    final=sort(collect(eachindex(s.ids));lt=(a,b)->_incremental_less(s,a,b,target))
    target,final
end
function _incremental_regrade!(s,tx,target,final)
    finalrank=invperm(final)
    # Exact trajectories are needed only for cells participating in crossings.
    # Cache them once for large reorderings without converting every unchanged
    # grade to BigInt rationals for a local edit.
    start=Vector{Rational{BigInt}}(undef,length(s.ids));delta=similar(start)
    converted=falses(length(s.ids))
    function trajectory(j)
        if !converted[j]
            start[j]=_vineyard_rational(s.grades[j])
            delta[j]=_vineyard_rational(target[j])-start[j];converted[j]=true
        end
        start[j],delta[j]
    end
    heap=Tuple{Rational{BigInt},Int,Int}[]
    function enqueue(i)
        1<=i<length(s.permutation) || return
        a,b=s.permutation[i],s.permutation[i+1]
        finalrank[a]>finalrank[b] || return
        a0,da=trajectory(a);b0,db=trajectory(b);denominator=da-db
        t=iszero(denominator) ? zero(Rational{BigInt}) : (b0-a0)/denominator
        0<=t<=1 || error("invalid kinetic crossing time")
        _vineyard_push!(heap,(t,a,b))
    end
    for i in 1:length(s.permutation)-1;enqueue(i);end
    events=VineyardEvent[];previous=zero(Rational{BigInt})
    while !isempty(heap)
        t,a,b=_vineyard_pop!(heap);i=s.rank[a]
        i<length(s.permutation) && s.permutation[i+1]==b && finalrank[a]>finalrank[b] || continue
        t>=previous || error("kinetic crossings are out of order")
        push!(events,_incremental_transpose!(s,tx,i;time=t,tie=:isolated));previous=t
        enqueue(i-1);enqueue(i+1)
    end
    s.permutation==final || error("kinetic updates did not reach the requested order")
    for i in eachindex(events)
        e=events[i]
        same=(i>1 && events[i-1].time==e.time) || (i<length(events) && events[i+1].time==e.time)
        same && (events[i]=VineyardEvent(e.cells,e.time,e.before,e.after,e.continuation,:simultaneous))
    end
    s.grades=target
    events
end

function _incremental_witness_preflight(s,witnesses,extra=0)
    witnesses isa Bool || throw(ArgumentError("witnesses must be Bool"))
    witnesses && BigInt(length(s.ids)+extra)^2>s.budget.max_entries &&
        throw(ArgumentError("explicit witness maps exceed the squared-cell max_entries bound"))
end

"""
    update_persistence!(state; append=nothing, insert=nothing, delete=nothing,
                        grades=nothing, continuation=false, witnesses=false)

Commit exactly one kind of edit atomically, returning a `PersistenceUpdate`.
`append`/`insert` are lists of `(dimension, grade, boundary)` named tuples;
boundaries contain stable cell ID => integer coefficient pairs. Optional IDs
must increase monotonically; omitted IDs are assigned. Append requires a new
filtration suffix. Insert permits earlier grades. Structural edits repair
affected columns when a reduction is retained; otherwise only the source changes.
`delete` lists a coface-closed set of active IDs. Deletion compacts storage;
removed IDs cannot be reused. Grade changes give stable ID => value pairs.

By default grade changes update only the endpoint. An unchanged filtration
order reuses any retained reduction; a reordering discards it. Subsequent diagram
queries use the ordinary reducer without rebuilding retained R/D/V or recording
crossings. Endpoint updates do not claim a continuation between persistence pairs.

For grade changes, `continuation=true` follows the straight-line homotopy,
including exact crossing times and simultaneous ties. If no reduction is retained,
it is built once at the starting endpoint and that cost is reported. This opt-in
can be substantially slower for broad reorderings. It never falls back to
endpoint-only computation, even when a work budget is exhausted.

Unknown IDs, invalid boundaries/grades and exhausted budgets leave the state
unchanged. Inspect `persistence_update_summary` for the method and correspondence.
Representatives are available from `persistence_diagram(state; representatives=true)`.
Opt in to copied chain snapshots with `witnesses=true` for actual inclusion maps
at a queried scale. An edit sequence is not itself a zigzag barcode.
"""
function update_persistence!(s::PersistenceState{T,K};append=nothing,insert=nothing,
        delete=nothing,grades=nothing,continuation=false,witnesses=false) where {T,K}
    lock(s.lock) do
        _incremental_require_open(s)
        options=(;append,insert,delete,grades);kinds=[k for k in keys(options) if options[k]!==nothing]
        length(kinds)==1 || throw(ArgumentError("give exactly one of append, insert, delete or grades"))
        kind=only(kinds);input=options[kind]
        continuation isa Bool || throw(ArgumentError("continuation must be Bool"))
        continuation && kind!==:grades && throw(ArgumentError("continuation=true requires a grade update"))
        input isa Union{AbstractVector,Tuple,AbstractDict} || throw(ArgumentError("edit input must be a finite collection"))
        extra=kind in (:append,:insert) ? length(input) : 0
        length(s.ids)+extra<=s.budget.max_cells || throw(ArgumentError("active cells exceed max_cells"))
        _incremental_witness_preflight(s,witnesses,extra)
        target,final=kind===:grades ? _incremental_target_grades(s,input) : (T[],Int[])
        before=witnesses ? _incremental_snapshot(s) : nothing
        if kind===:grades && !continuation
            # Prepare every fallible allocation before committing the endpoint.
            rank=invperm(final);retained=s.retained && final==s.permutation
            after=witnesses ? _PersistenceSnapshot{T,K}(copy(s.ids),copy(s.dims),copy(target),deepcopy(s.boundary)) : nothing
            method=retained ? :reuse_order : :endpoint_regrade
            u=PersistenceUpdate{T,K}(kind,s.revision,s.revision+1,Int[],Int[],VineyardEvent[],
                Pair{Int,T}[],Pair{Int,T}[],before,after,s.field,s.order,s.budget,0,0,method,0,retained)
            retained || _incremental_drop_reduction!(s)
            s.grades=target;s.permutation=final;s.rank=rank;s.revision+=1
            return u
        end
        start=continuation ? Pair{Int,T}[s.ids[j]=>s.grades[j] for j in eachindex(s.ids)] : Pair{Int,T}[]
        tx=_PersistenceEdit(s,kind);added=Int[];removed=Int[];events=VineyardEvent[]
        builds=0
        method=continuation ? :vineyard : s.retained ? :retained_repair : :source_edit
        try
            if kind===:grades
                if !s.retained
                    tx.additions=_incremental_retain!(s);tx.columns=length(s.ids);builds=1
                end
                events=_incremental_regrade!(s,tx,target,final)
            elseif kind in (:append,:insert)
                for record in input
                    j=_incremental_append!(s,tx,record;strict=kind===:append);push!(added,s.ids[j])
                end
            else
                slots=Int[_incremental_slot(s,id) for id in input]
                allunique(slots) || throw(ArgumentError("duplicate deletion ID"))
                selected=Set(slots)
                all(j->issubset(s.cofaces[j],selected),slots) || throw(ArgumentError("deletion must include every coface of each removed cell"))
                sort!(slots;by=j->s.rank[j],rev=true)
                for j in slots
                    push!(removed,s.ids[j]);_incremental_delete!(s,tx,j)
                end
            end
        catch
            _incremental_rollback!(s,tx);rethrow()
        end
        kind===:delete && _incremental_compact!(s)
        after=witnesses ? _incremental_snapshot(s) : nothing
        finish=continuation ? Pair{Int,T}[s.ids[j]=>s.grades[j] for j in eachindex(s.ids)] : Pair{Int,T}[]
        old=s.revision;s.revision+=1
        PersistenceUpdate{T,K}(kind,old,s.revision,added,removed,events,start,finish,
            before,after,s.field,s.order,s.budget,tx.additions,tx.columns,method,builds,s.retained)
    end
end

"""Transpose two adjacent, equal-grade cells which are not a face/coface pair.
This explicitly requests a deterministic vineyard tie-order update, building a
retained reduction first if absent. Use
`update_persistence!(s; grades, continuation=true)` for actual grade crossings
and their rational event times. Positions are one-based."""
function transpose_persistence!(s::PersistenceState{T,K};position,witnesses=false) where {T,K}
    lock(s.lock) do
        _incremental_require_open(s);i=_incremental_integer(position,:position)
        i<length(s.permutation) || throw(ArgumentError("position needs a following cell"))
        a,b=s.permutation[i],s.permutation[i+1]
        s.grades[a]==s.grades[b] || throw(ArgumentError("explicit transposition requires equal grades; use a grade update to cross"))
        _incremental_witness_preflight(s,witnesses)
        before=witnesses ? _incremental_snapshot(s) : nothing;tx=_PersistenceEdit(s,:transpose)
        builds=0
        event=try
            if !s.retained
                tx.additions=_incremental_retain!(s);tx.columns=length(s.ids);builds=1
            end
            _incremental_transpose!(s,tx,i;tie=:explicit_tie)
        catch;_incremental_rollback!(s,tx);rethrow();end
        after=witnesses ? _incremental_snapshot(s) : nothing;old=s.revision;s.revision+=1
        PersistenceUpdate{T,K}(:transpose,old,s.revision,Int[],Int[],[event],Pair{Int,T}[],Pair{Int,T}[],
            before,after,s.field,s.order,s.budget,tx.additions,tx.columns,:vineyard,builds,s.retained)
    end
end
