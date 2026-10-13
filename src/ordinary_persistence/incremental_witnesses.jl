# Exact validation and optional scale-specific inclusion maps for edit records.

function _incremental_product(columns,column,::Type{K},budget) where {K}
    result=Dict{Int,K}()
    for e in column,f in columns[first(e)]
        j=first(f);v=get(result,j,zero(K))+last(e)*last(f)
        iszero(v) ? delete!(result,j) : (result[j]=v)
        length(result)<=budget.max_entries || throw(ArgumentError("validation scratch exceeds max_entries"))
    end
    result
end
function _incremental_validation(f;throw=false)
    issues=String[]
    try;f()
    catch err;err isa InterruptException && rethrow();push!(issues,sprint(showerror,err));end
    result=PersistenceStateValidationSummary(isempty(issues),issues)
    throw && !result.valid && Base.throw(ArgumentError(join(issues,"; ")))
    result
end

"""Recheck field arithmetic, filtration, D squared, row indexes and storage
budgets. When a reduction is retained, also audit R=D*V, triangularity and unique
pivots; otherwise require all reduction storage to be empty. This is a full audit,
not work repeated on every update. Malformed storage gives `.valid=false`."""
function check_persistence_state(s::PersistenceState{T,K};throw=false) where {T,K}
    lock(s.lock) do
        _incremental_validation(;throw) do
            _normalize_field(s.field);_normalize_order(s.order)
            K===coeff_type(s.field) || error("coefficient type and field disagree")
            n=length(s.ids)
            all(name->length(getfield(s,name))==n,(:labels,:dims,:grades,:slots,:permutation,:rank,
                :boundary,:cofaces)) || error("storage lengths disagree")
            all(name->length(getfield(s,name))==(s.retained ? n : 0),
                (:reduced,:change,:reduced_rows,:change_rows,:pivot)) || error("reduction storage disagrees with retained status")
            s.closed && (n!=0 || s.retained) && error("closed state retains cells or reduction")
            n<=s.budget.max_cells || error("active cells exceed max_cells")
            allunique(s.ids) && all(id->0<id<s.next_id,s.ids) || error("invalid stable IDs")
            all(d->0<=d<=s.budget.max_cells,s.dims) && all(isfinite,s.grades) || error("invalid dimensions or grades")
            sort(s.permutation)==collect(1:n) && s.rank==invperm(s.permutation) || error("invalid filtration permutation")
            s.slots==Dict(id=>i for (i,id) in enumerate(s.ids)) || error("stable ID index disagrees")
            all(i->_incremental_before(s,s.grades[s.permutation[i]],s.grades[s.permutation[i+1]]),1:n-1) || error("filtration grades are not ordered")
            count=sum(length,s.boundary;init=0)+sum(length,s.reduced;init=0)+sum(length,s.change;init=0)
            s.entries==count<=s.budget.max_entries || error("retained entry count or budget disagrees")
            pivot=zeros(Int,s.retained ? n : 0)
            for (name,columns) in ((:boundary,s.boundary),(:reduced,s.reduced),(:change,s.change)),j in eachindex(columns)
                c=columns[j];rows=first.(c)
                allunique(rows) && all(i->1<=i<=n,rows) && all(e->!iszero(last(e)),c) || error("invalid sparse column")
                all(i->s.dims[i]==s.dims[j]-(name===:change ? 0 : 1),rows) || error("column mixes chain degrees")
                all(i->name===:change ? s.rank[i]<=s.rank[j] : s.rank[i]<s.rank[j],rows) || error("column is not filtration triangular")
                name===:boundary || issorted(s.rank[rows]) || error("sparse entries are not rank sorted")
                if name===:change
                    !iszero(_incremental_entry(c,j)) || error("basis diagonal vanishes")
                    _incremental_product(s.boundary,c,K,s.budget)==Dict(s.reduced[j]) || error("R does not equal D*V")
                elseif name===:boundary
                    isempty(_incremental_product(s.boundary,c,K,s.budget)) || error("double boundary is nonzero")
                elseif !isempty(c)
                    p=first(last(c));pivot[p]==0 || error("duplicate reduced pivot")
                    isempty(s.reduced[p]) || error("pivot is not a positive cell")
                    pivot[p]=j
                end
            end
            pivot==s.pivot || error("pivot index disagrees")
            for (columns,rows) in ((s.boundary,s.cofaces),(s.reduced,s.reduced_rows),(s.change,s.change_rows))
                columns===s.boundary || s.retained || continue
                rows==_incremental_rows(columns,n) || error("row incidence index disagrees")
            end
        end
    end
end

"""Validate an update record's exact crossing times, local pairing bijections and
endpoint continuity. Optional copied chain snapshots are checked as complexes.
This verifies the record, not a unique homology-class identity at a tied crossing."""
function check_persistence_update(u::PersistenceUpdate{T,K};throw=false) where {T,K}
    _incremental_validation(;throw) do
        _normalize_field(u.field);_normalize_order(u.order)
        K===coeff_type(u.field) || error("update coefficient type and field disagree")
        u.kind in (:append,:insert,:delete,:grades,:transpose) || error("unknown edit kind")
        if u.method===:vineyard
            u.kind in (:grades,:transpose) && u.retained || error("invalid vineyard route")
            u.reduction_builds in (0,1) || error("invalid reduction preparation count")
        else
            u.reduction_builds==0 && isempty(u.events) || error("endpoint/structural update claims vineyard work")
            if u.kind===:grades
                u.method===(u.retained ? :reuse_order : :endpoint_regrade) || error("invalid endpoint route")
            else
                u.kind in (:append,:insert,:delete) &&
                    u.method===(u.retained ? :retained_repair : :source_edit) || error("invalid structural route")
            end
        end
        if u.method in (:source_edit,:endpoint_regrade,:reuse_order)
            u.column_additions==u.reduced_columns==0 || error("source-only edit claims reduction work")
        end
        u.before_revision>=0 && u.after_revision==u.before_revision+1 || error("nonconsecutive revisions")
        allunique(u.added) && allunique(u.removed) && isempty(intersect(u.added,u.removed)) || error("invalid created/deleted IDs")
        all(>(0),u.added) && all(>(0),u.removed) || error("cell IDs must be positive")
        u.kind in (:append,:insert) || isempty(u.added) || error("edit kind cannot create cells")
        u.kind===:delete || isempty(u.removed) || error("edit kind cannot remove cells")
        length(u.events)<=u.budget.max_swaps && 0<=u.column_additions<=u.budget.max_column_additions &&
            u.reduced_columns>=0 || error("invalid update work counts")
        (u.before===nothing)==(u.after===nothing) || error("missing witness snapshot")
        start=Dict(u.start_grades);stop=Dict(u.end_grades)
        length(start)==length(u.start_grades) && length(stop)==length(u.end_grades) || error("duplicate grade IDs")
        all(>(0),keys(start)) && all(>(0),keys(stop)) && all(isfinite,values(start)) && all(isfinite,values(stop)) || error("invalid grade records")
        grade_path=u.kind===:grades && u.method===:vineyard
        grade_path && keys(start)!=keys(stop) && error("grade path changes cell IDs")
        grade_path || (isempty(start) && isempty(stop)) || error("update without continuation claims a grade path")
        previous=zero(Rational{BigInt})
        for (index,e) in enumerate(u.events)
            length(unique(e.cells))==2 && all(>(0),e.cells) || error("invalid transposed cells")
            if u.kind in (:append,:insert,:delete)
                e.tie===:algebraic_edit && isempty(e.continuation) && e.time===nothing || error("structural edit claims vine correspondence")
                continue
            end
            before=Dict(p.birth_cell=>p for p in e.before);after=Dict(p.birth_cell=>p for p in e.after)
            length(before)==length(e.before) && length(after)==length(e.after) || error("duplicate local pair")
            all(p->p.dimension>=0 && p.birth_cell>0 && p.death_cell>=0 && p.birth_cell!=p.death_cell,
                Iterators.flatten((e.before,e.after))) || error("invalid local pair")
            length(e.continuation)==length(before)==length(after) || error("continuation is not a bijection")
            Set(first.(e.continuation))==Set(keys(before)) && Set(last.(e.continuation))==Set(keys(after)) || error("continuation misses pairs")
            all(pair->before[first(pair)].dimension==after[last(pair)].dimension,e.continuation) || error("continuation changes degree")
            u.kind===:transpose && (e.time!==nothing || e.tie!==:explicit_tie) && error("explicit tie has an event time")
            if u.kind===:grades
                t=e.time;t!==nothing && previous<=t<=1 || error("invalid ordered crossing time")
                simultaneous=(index>1 && u.events[index-1].time==t) ||
                    (index<length(u.events) && u.events[index+1].time==t)
                e.tie===(simultaneous ? :simultaneous : :isolated) || error("crossing tie label disagrees")
                previous=t
                value(id)=(1-t)*_vineyard_rational(start[id])+t*_vineyard_rational(stop[id])
                value(e.cells[1])==value(e.cells[2]) || error("transposition is not at a crossing")
                for (a,b) in e.continuation
                    p,q=before[a],after[b]
                    p.dimension==q.dimension && value(p.birth_cell)==value(q.birth_cell) || error("birth continuation is discontinuous")
                    if p.death_cell==0 || q.death_cell==0
                        p.death_cell==q.death_cell || error("essential status changed at a crossing")
                    else
                        value(p.death_cell)==value(q.death_cell) || error("death continuation is discontinuous")
                    end
                end
            end
        end
        for snapshot in (u.before,u.after)
            snapshot===nothing && continue
            n=length(snapshot.ids)
            allunique(snapshot.ids) && all(>(0),snapshot.ids) || error("invalid witness IDs")
            length(snapshot.dims)==length(snapshot.grades)==length(snapshot.boundary)==n || error("witness lengths disagree")
            n<=u.budget.max_cells && BigInt(n)^2<=u.budget.max_entries || error("witness exceeds budget")
            G,_=_incremental_complex(snapshot)
            _validate_complex(G,u.order,coeff_type(u.field))
        end
        if u.before!==nothing
            setdiff(Set(u.before.ids),Set(u.removed))==setdiff(Set(u.after.ids),Set(u.added)) || error("witness cell sets disagree with edit")
        end
    end
end

function _incremental_scale_data(snapshot,order,scale,dim,::Type{K}) where {K}
    active=[i for i in eachindex(snapshot.ids) if order===:sublevel ? snapshot.grades[i]<=scale : snapshot.grades[i]>=scale]
    ids(d)=[i for i in active if snapshot.dims[i]==d]
    function boundary(d)
        rows=ids(d-1);cols=ids(d);lookup=Dict(i=>j for (j,i) in enumerate(rows))
        ri=Int[];ci=Int[];vs=K[]
        for (j,col) in enumerate(cols),e in snapshot.boundary[col]
            i=get(lookup,first(e),0);i>0 || error("active source is not a subcomplex")
            push!(ri,i);push!(ci,j);push!(vs,last(e))
        end
        sparse(ri,ci,vs,length(rows),length(cols))
    end
    (;all_ids=snapshot.ids[active],cells=snapshot.ids[ids(dim)],lower_cells=snapshot.ids[ids(dim-1)],upper_cells=snapshot.ids[ids(dim+1)],
        down=boundary(dim),up=boundary(dim+1))
end

"""
    persistence_update_map(update; dim, scale, direction=:auto)

Return the actual chain inclusion and induced homology matrix at one finite scale
from an update computed with `witnesses=true`. Matrices have target rows and
source columns. Cycle bases, source/target stable IDs and both differentials are
returned for checking. Direction is `:before_to_after` or `:after_to_before`;
`:auto` chooses an available inclusion, preferring forward if both apply.
Reject incomparable active subcomplexes. A vineyard pairing is not this map,
and this query does not compute the zigzag intervals of an edit history.
"""
function persistence_update_map(u::PersistenceUpdate{T,K};dim::Integer,scale::Real,
        direction::Symbol=:auto) where {T,K}
    d=_degree_slot(dim)-1
    isfinite(scale) && !(scale isa Bool) || throw(ArgumentError("scale must be finite"))
    direction in (:auto,:before_to_after,:after_to_before) || throw(ArgumentError("invalid map direction"))
    u.before!==nothing && u.after!==nothing || throw(ArgumentError("request witnesses=true when making the update"))
    max(BigInt(length(u.before.ids))^2,BigInt(length(u.after.ids))^2)<=u.budget.max_entries ||
        throw(ArgumentError("homology map exceeds max_entries"))
    a=_incremental_scale_data(u.before,u.order,scale,d,K)
    b=_incremental_scale_data(u.after,u.order,scale,d,K)
    forward=issubset(a.all_ids,b.all_ids);backward=issubset(b.all_ids,a.all_ids)
    if direction===:auto
        direction=forward ? :before_to_after : backward ? :after_to_before :
            throw(ArgumentError("neither active subcomplex includes into the other at this scale"))
    end
    (direction===:before_to_after ? forward : backward) || throw(ArgumentError("requested direction is not a subcomplex inclusion"))
    source,target=direction===:before_to_after ? (a,b) : (b,a)
    chain_map=_relative_coordinate_map(K,source.cells,target.cells)
    lower_map=_relative_coordinate_map(K,source.lower_cells,target.lower_cells)
    upper_map=_relative_coordinate_map(K,source.upper_cells,target.upper_cells)
    target.down*chain_map==lower_map*source.down || throw(ArgumentError("snapshots do not define the claimed chain inclusion"))
    target.up*upper_map==chain_map*source.up || throw(ArgumentError("inclusion does not preserve incoming boundaries"))
    hs=ChainComplexes.homology_data(source.up,source.down,d;field=u.field)
    ht=ChainComplexes.homology_data(target.up,target.down,d;field=u.field)
    homology_map=ChainComplexes.induced_map_on_homology(hs,ht,chain_map)
    (;direction,dimension=d,scale,field=u.field,source_cells=copy(source.cells),target_cells=copy(target.cells),
        chain_map,homology_map,source_cycles=copy(basis(hs)),target_cycles=copy(basis(ht)),
        source_boundary=source.down,target_boundary=target.down,
        source_incoming=source.up,target_incoming=target.up)
end
