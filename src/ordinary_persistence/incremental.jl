# Retained sparse R = D*V reductions of based filtered chain complexes.
# Rows use compact storage slots; public cell IDs are monotone and never reused.

"""Bounds for stored cells, nonzeros across D/R/V, adjacent swaps and column
additions per retained update. These are logical bounds, not bytes. A fresh
`persistence_diagram` query uses the ordinary reducer's allocation policy."""
struct PersistenceUpdateBudget
    max_cells::Int
    max_entries::Int
    max_swaps::Int
    max_column_additions::Int
    function PersistenceUpdateBudget(;max_cells=100_000,max_entries=8_000_000,
            max_swaps=100_000,max_column_additions=1_000_000)
        xs=(max_cells,max_entries,max_swaps,max_column_additions)
        all(x->x isa Integer && !(x isa Bool) && 0<x<=typemax(Int),xs) ||
            throw(ArgumentError("incremental persistence budgets must be positive machine integers"))
        new(Int.(xs)...)
    end
end
describe(b::PersistenceUpdateBudget)=(;max_cells=b.max_cells,max_entries=b.max_entries,
    max_swaps=b.max_swaps,max_column_additions=b.max_column_additions)

"""Mutable filtered complex with an optional exact prime-field reduction.
Construct with `persistence_state`; opt in to `retain_reduction=true` for local
repair. By default diagram queries use the ordinary reducer. Grade continuation
requires `update_persistence!(s; grades, continuation=true)` explicitly.
query with `persistence_diagram`, `persistence_cells`, `persistence_pairing`,
and `persistence_column`. Mutate only through `update_persistence!` or
`transpose_persistence!`. No edit history is retained by the state."""
mutable struct PersistenceState{T<:Real,K}
    ids::Vector{Int}
    labels::Vector{Int}
    dims::Vector{Int}
    grades::Vector{T}
    slots::Dict{Int,Int}
    permutation::Vector{Int}
    rank::Vector{Int}
    boundary::Vector{Vector{Pair{Int,K}}}
    reduced::Vector{Vector{Pair{Int,K}}}
    change::Vector{Vector{Pair{Int,K}}}
    reduced_rows::Vector{Set{Int}}
    change_rows::Vector{Set{Int}}
    cofaces::Vector{Set{Int}}
    pivot::Vector{Int}
    entries::Int
    next_id::Int
    revision::Int
    field::PrimeField
    order::Symbol
    budget::PersistenceUpdateBudget
    retained::Bool
    closed::Bool
    lock::ReentrantLock
end

struct _PersistenceSnapshot{T,K}
    ids::Vector{Int}
    dims::Vector{Int}
    grades::Vector{T}
    boundary::Vector{Vector{Pair{Int,K}}}
end

"""A cell pairing, including zero-length pairs. IDs name cells in one state lineage;
`death_cell == 0` denotes an unpaired birth, never a source cell."""
struct PersistencePair
    dimension::Int
    birth_cell::Int
    death_cell::Int
end
Base.:(==)(a::PersistencePair,b::PersistencePair)=
    (a.dimension,a.birth_cell,a.death_cell)==(b.dimension,b.birth_cell,b.death_cell)
Base.isequal(a::PersistencePair,b::PersistencePair)=a==b
Base.hash(a::PersistencePair,h::UInt)=hash((a.dimension,a.birth_cell,a.death_cell),h)
describe(p::PersistencePair)=(;dimension=p.dimension,birth_cell=p.birth_cell,death_cell=p.death_cell)

"""One actual adjacent transposition. `continuation` relates before/after birth
IDs by unchanged far endpoints, not optimal diagram matching. Simultaneous
crossings have a declared deterministic ordering and need not identify unique
vines. Algebraic insertion/deletion moves carry no vine continuation."""
struct VineyardEvent
    cells::Tuple{Int,Int}
    time::Union{Nothing,Rational{BigInt}}
    before::Vector{PersistencePair}
    after::Vector{PersistencePair}
    continuation::Vector{Tuple{Int,Int}}
    tie::Symbol
end
describe(e::VineyardEvent)=(;cells=e.cells,time=e.time,before=copy(e.before),after=copy(e.after),
    continuation=copy(e.continuation),tie=e.tie)

"""One committed update; no mutable state or reduction columns are retained.
Optional `witnesses=true` stores independent before/after chain snapshots for
`persistence_update_map`. Only `continuation=true` grade updates record the
straight-line homotopy and its exact crossings. An endpoint-only update has no
vineyard correspondence, even if its event list and a constant path are both empty.
Use `persistence_update_summary` to distinguish these contracts."""
struct PersistenceUpdate{T,K}
    kind::Symbol
    before_revision::Int
    after_revision::Int
    added::Vector{Int}
    removed::Vector{Int}
    events::Vector{VineyardEvent}
    start_grades::Vector{Pair{Int,T}}
    end_grades::Vector{Pair{Int,T}}
    before::Union{Nothing,_PersistenceSnapshot{T,K}}
    after::Union{Nothing,_PersistenceSnapshot{T,K}}
    field::PrimeField
    order::Symbol
    budget::PersistenceUpdateBudget
    column_additions::Int
    reduced_columns::Int
    method::Symbol
    reduction_builds::Int
    retained::Bool
end
function persistence_update_summary(u::PersistenceUpdate)
    (;kind=u.kind,before_revision=u.before_revision,after_revision=u.after_revision,
        added=copy(u.added),removed=copy(u.removed),swaps=length(u.events),
        column_additions=u.column_additions,reduced_columns=u.reduced_columns,
        method=u.method,full_recomputations=u.reduction_builds,fallback=:none,witnesses=u.before!==nothing,
        next_diagram=u.retained ? :retained_reduction : :fresh_reduction,
        correspondence=u.method===:vineyard ? :vineyard_pairing :
            u.kind in (:append,:insert,:delete) ? :use_inclusion_maps : :none,
        simultaneous_events=count(e->e.tie in (:simultaneous,:explicit_tie),u.events))
end
describe(u::PersistenceUpdate)=persistence_update_summary(u)
Base.show(io::IO,u::PersistenceUpdate)=print(io,"PersistenceUpdate(",u.kind,", revision=",
    u.after_revision,", method=",u.method,", swaps=",length(u.events),")")

"""Result of a complete exact reduction or update-record validation."""
struct PersistenceStateValidationSummary
    valid::Bool
    issues::Vector{String}
end
Base.:(==)(a::PersistenceStateValidationSummary,b::PersistenceStateValidationSummary)=a.valid==b.valid && a.issues==b.issues
Base.isequal(a::PersistenceStateValidationSummary,b::PersistenceStateValidationSummary)=isequal((a.valid,a.issues),(b.valid,b.issues))
Base.hash(a::PersistenceStateValidationSummary,h::UInt)=hash((a.valid,a.issues),h)
describe(v::PersistenceStateValidationSummary)=(;valid=v.valid,issues=copy(v.issues))
Base.show(io::IO,v::PersistenceStateValidationSummary)=print(io,"PersistenceStateValidationSummary(valid=",v.valid,", issues=",v.issues,")")

_incremental_require_open(s)=s.closed && throw(ArgumentError("incremental persistence state is closed"))
function _incremental_integer(x,name;lower=1)
    x isa Integer && !(x isa Bool) && lower<=x<typemax(Int) ||
        throw(ArgumentError("$name must be an integer at least $lower and below typemax(Int)"))
    Int(x)
end
function _incremental_slot(s,id)
    _incremental_integer(id,:cell)
    i=get(s.slots,Int(id),0)
    i>0 || throw(ArgumentError("unknown or deleted cell ID $id"))
    i
end
function _incremental_grade(::Type{T},x) where {T}
    x isa Real && !(x isa Bool) && isfinite(x) || throw(ArgumentError("grades must be finite real values"))
    y=try convert(T,x) catch; throw(ArgumentError("grade is not representable in $T")); end
    y==x || throw(ArgumentError("grade conversion to $T loses precision"))
    y
end
function _incremental_less(s,a,b,grades=s.grades)
    x,y=grades[a],grades[b]
    x==y ? (s.dims[a],s.ids[a])<(s.dims[b],s.ids[b]) :
        (s.order===:sublevel ? x<y : x>y)
end
_incremental_before(s,a,b)=s.order===:sublevel ? a<=b : a>=b
_incremental_low(c)=isempty(c) ? 0 : first(last(c))
function _incremental_entry(c::Vector{Pair{Int,K}},row) where {K}
    for e in c
        first(e)==row && return last(e)
    end
    zero(K)
end

function _incremental_rows(columns,n)
    rows=[Set{Int}() for _ in 1:n]
    for (j,col) in enumerate(columns),e in col;push!(rows[first(e)],j);end
    rows
end
function _incremental_reindex!(s)
    n=length(s.ids)
    s.slots=Dict(id=>j for (j,id) in enumerate(s.ids))
    s.rank=invperm(s.permutation)
    s.reduced_rows=s.retained ? _incremental_rows(s.reduced,n) : Set{Int}[]
    s.change_rows=s.retained ? _incremental_rows(s.change,n) : Set{Int}[]
    s.cofaces=_incremental_rows(s.boundary,n)
    s.pivot=zeros(Int,s.retained ? n : 0)
    for (j,c) in enumerate(s.reduced)
        p=_incremental_low(c);p==0 || (s.pivot[p]=j)
    end
    s.entries=sum(length,s.boundary;init=0)+sum(length,s.reduced;init=0)+sum(length,s.change;init=0)
    s
end

"""
    persistence_state(G; field=F2(), order=:sublevel, retain_reduction=false,
                      budget=Advanced.PersistenceUpdateBudget())

Copy a finite one-parameter `GradedComplex` over a supported prime field. By
default only the source complex is retained, and `persistence_diagram(s)` uses
the optimized ordinary reducer without recording crossings or retaining `R` and `V`.
Use `retain_reduction=true` to build sparse `R=D*V` for repeated local edits;
this can help append/insertion workloads but is not a general speed guarantee.
Initial stable IDs are flattened source positions;
source labels are separate and may repeat. Ties use dimension then stable ID.
Grade updates are endpoint-only unless `continuation=true` is requested. A
reordering endpoint update discards the old reduction; later diagram queries
do not rebuild it. Requesting continuation builds it once when absent and
reports that preparation in the update summary. Insertions and subcomplex
deletions repair a reduction only when one is already retained. Boundary
replacement, cohomology updates and a zigzag barcode are outside this state contract.
"""
function persistence_state(G::GradedComplex{N,T};field=F2(),order::Symbol=:sublevel,
        retain_reduction=false,budget=PersistenceUpdateBudget()) where {N,T}
    _normalize_field(field);_normalize_order(order)
    budget isa PersistenceUpdateBudget || throw(ArgumentError("budget must be PersistenceUpdateBudget"))
    retain_reduction isa Bool || throw(ArgumentError("retain_reduction must be Bool"))
    n=length(G.grades);n<=budget.max_cells || throw(ArgumentError("active cells exceed max_cells"))
    sum(nnz,G.boundaries;init=0)+(retain_reduction ? n : 0)<=budget.max_entries || throw(ArgumentError("input exceeds max_entries"))
    K=coeff_type(field);offsets=_validate_complex(G,order,K)
    dims=_cell_dimensions(offsets);grades=T[g[1] for g in G.grades]
    cols=[Pair{Int,K}[] for _ in 1:n]
    for j in 1:n
        d=dims[j];d==0 && continue
        B=G.boundaries[d];local_j=j-offsets[d+1]+1
        for p in nzrange(B,local_j)
            c=K(B.nzval[p]);iszero(c) || push!(cols[j],(offsets[d]+B.rowval[p]-1)=>c)
        end
    end
    s=PersistenceState(collect(1:n),copy(getfield(G,:cell_ids)),dims,grades,Dict{Int,Int}(),
        collect(1:n),zeros(Int,n),cols,Vector{Pair{Int,K}}[],
        Vector{Pair{Int,K}}[],Set{Int}[],Set{Int}[],
        Set{Int}[],Int[],0,n+1,0,field,order,budget,false,false,ReentrantLock())
    sort!(s.permutation;lt=(a,b)->_incremental_less(s,a,b));s.rank=invperm(s.permutation)
    for c in s.boundary;sort!(c;by=e->s.rank[first(e)]);end
    _incremental_reindex!(s)
    retain_reduction && _incremental_retain!(s)
    s
end

# Build into private buffers, committing only after every budget check succeeds.
# No boundary columns are changed, so failed preparation leaves the source intact.
function _incremental_retain!(s::PersistenceState{T,K}) where {T,K}
    n=length(s.ids);budget=s.budget;entries=sum(length,s.boundary;init=0)
    entries+n<=budget.max_entries || throw(ArgumentError("reduction exceeds max_entries"))
    reduced=Vector{Vector{Pair{Int,K}}}(undef,n);change=similar(reduced);pivot=zeros(Int,n)
    additions=0
    for j in s.permutation
        r=copy(s.boundary[j]);v=Pair{Int,K}[j=>one(K)]
        sort!(r;by=e->s.rank[first(e)])
        while !isempty(r) && pivot[first(last(r))]!=0
            k=pivot[first(last(r))]
            a=-last(last(r))/last(last(reduced[k]))
            r=_prime_add_scaled!(Pair{Int,K}[],r,reduced[k],a,s.rank)
            v=_prime_add_scaled!(Pair{Int,K}[],v,change[k],a,s.rank)
            additions+=1
            additions<=budget.max_column_additions || throw(ArgumentError("initial reduction exceeds max_column_additions"))
            length(r)+length(v)+entries<=budget.max_entries || throw(ArgumentError("reduction exceeds max_entries"))
        end
        entries+=length(r)+length(v)
        entries<=budget.max_entries || throw(ArgumentError("reduction exceeds max_entries"))
        reduced[j]=r;change[j]=v
        isempty(r) || (pivot[first(last(r))]=j)
    end
    reduced_rows=_incremental_rows(reduced,n);change_rows=_incremental_rows(change,n)
    s.reduced=reduced;s.change=change;s.pivot=pivot
    s.reduced_rows=reduced_rows;s.change_rows=change_rows;s.entries=entries;s.retained=true
    additions
end
function _incremental_drop_reduction!(s)
    for name in (:reduced,:change,:reduced_rows,:change_rows,:pivot)
        setfield!(s,name,empty(getfield(s,name)))
    end
    s.retained=false;s.entries=sum(length,s.boundary;init=0)
end

function persistence_state_summary(s::PersistenceState)
    lock(s.lock) do
        (;kind=:incremental_persistence,field=s.field,order=s.order,revision=s.revision,
            cells=length(s.ids),stored_entries=s.entries,next_cell_id=s.next_id,
            closed=s.closed,history_length=0,reduction=s.retained ? :sparse_R_equals_DV : :none,
            diagram_method=s.retained ? :retained_reduction : :fresh_reduction,
            fallback=:none,source_identity=:monotone_cell_ids,budget=describe(s.budget))
    end
end
describe(s::PersistenceState)=persistence_state_summary(s)
Base.show(io::IO,s::PersistenceState)=print(io,"PersistenceState(cells=",length(s.ids),", revision=",s.revision,", retained=",s.retained,", field=",s.field,", closed=",s.closed,")")
field(s::PersistenceState)=s.field
filtration_order(s::PersistenceState)=s.order
provenance(s::PersistenceState)=persistence_state_summary(s)

"""Copy cell records in current filtration order. Boundary entries use stable IDs;
`label` preserves the original supplied source label, independently of identity."""
function persistence_cells(s::PersistenceState)
    lock(s.lock) do
        _incremental_require_open(s)
        [(;id=s.ids[j],label=s.labels[j],dimension=s.dims[j],grade=s.grades[j],
            boundary=[s.ids[first(e)]=>Int(last(e).val) for e in s.boundary[j]]) for j in s.permutation]
    end
end
function _incremental_pair(s,b)
    d=s.pivot[b]
    PersistencePair(s.dims[b],s.ids[b],d==0 ? 0 : s.ids[d])
end
"""Copy pairings from a retained reduction, including zero-length pairs by default.
Essential deaths use cell ID 0. IDs remain distinct when endpoints coincide.
Requires a retained reduction; endpoint-only states can query diagrams instead."""
function persistence_pairing(s::PersistenceState;include_zero::Bool=true)
    lock(s.lock) do
        _incremental_require_open(s)
        s.retained || throw(ArgumentError("pairings require a retained reduction; construct with retain_reduction=true or request continuation=true on a grade update"))
        [_incremental_pair(s,b) for b in s.permutation if isempty(s.reduced[b]) &&
            (include_zero || s.pivot[b]==0 || s.grades[b]!=s.grades[s.pivot[b]])]
    end
end
"""Copy a retained `:boundary`, `:reduced` or `:basis` column as stable ID => residue
pairs. Basis columns satisfy boundary(basis_j) = reduced_j; they are cycles only
when the reduced column is zero. Only `:boundary` is available without a
retained reduction."""
function persistence_column(s::PersistenceState;cell,kind::Symbol=:reduced)
    lock(s.lock) do
        _incremental_require_open(s);j=_incremental_slot(s,cell)
        kind in (:boundary,:reduced,:basis) || throw(ArgumentError("kind must be :boundary, :reduced or :basis"))
        kind===:boundary || s.retained || throw(ArgumentError("reduced and basis columns require a retained reduction"))
        c=kind===:boundary ? s.boundary[j] : kind===:reduced ? s.reduced[j] : s.change[j]
        [s.ids[first(e)]=>Int(last(e).val) for e in c]
    end
end

function _incremental_snapshot(s::PersistenceState{T,K}) where {T,K}
    _PersistenceSnapshot{T,K}(copy(s.ids),copy(s.dims),copy(s.grades),deepcopy(s.boundary))
end
function _incremental_complex(s)
    nd=maximum(s.dims;init=0)+1
    groups=[sort(findall(==(d),s.dims);by=i->s.ids[i]) for d in 0:nd-1]
    local_index=zeros(Int,length(s.ids))
    for group in groups,(i,j) in enumerate(group);local_index[j]=i;end
    Bs=SparseMatrixCSC{Int,Int}[]
    for d in 1:nd-1
        rows=Int[];cols=Int[];vs=Int[]
        for (j,col) in enumerate(groups[d+1]),e in s.boundary[col]
            push!(rows,local_index[first(e)]);push!(cols,j);push!(vs,Int(last(e).val))
        end
        push!(Bs,sparse(rows,cols,vs,length(groups[d]),length(groups[d+1])))
    end
    T=eltype(s.grades)
    G=GradedComplex([s.ids[g] for g in groups],Bs,Tuple{T}[(s.grades[j],) for group in groups for j in group])
    G,local_index
end

"""Return an independent diagram. Use retained pivots when available, otherwise
the optimized ordinary reducer. Fresh queries do not retain reduction columns
or cache diagrams. `representatives=true` supplies exact cycles and fillings in
stable-cell coordinates on either route, without requesting vineyard continuation.
Use `persistence_cells` for labels and dimensions; `provenance` reports the route."""
function persistence_diagram(s::PersistenceState{T,K};representatives::Bool=false) where {T,K}
    lock(s.lock) do
        _incremental_require_open(s)
        if !s.retained
            G,_=_incremental_complex(s)
            d=persistence_diagram(G;field=s.field,order=s.order,representatives)
            meta=merge(d.meta,(;source=:incremental_complex,revision=s.revision,
                source_identity=:monotone_cell_ids,update_query=:fresh_reduction))
            return PersistenceDiagram(d.finite_by_dim,d.essential_by_dim,s.field,s.order,meta,d.retained_representatives)
        end
        nd=maximum(s.dims;init=0)+1
        finite=[Tuple{T,T}[] for _ in 1:nd];essential=[T[] for _ in 1:nd]
        fr=representatives ? [_PersistenceRepresentative{T}[] for _ in 1:nd] : nothing
        er=representatives ? [_PersistenceRepresentative{T}[] for _ in 1:nd] : nothing
        locals=representatives ? zeros(Int,length(s.ids)) : Int[]
        if representatives
            for d in 0:nd-1
                group=sort(findall(==(d),s.dims);by=i->s.ids[i])
                for (i,j) in enumerate(group);locals[j]=i;end
            end
        end
        chain(c)=begin
            sorted=sort(c;by=e->s.ids[first(e)]);js=first.(sorted)
            _PersistenceChain{T}(locals[js],s.ids[js],s.grades[js],Int[last(e).val for e in sorted])
        end
        for b in s.permutation
            isempty(s.reduced[b]) || continue
            d=s.pivot[b];slot=s.dims[b]+1;birth=s.grades[b]
            if d==0
                push!(essential[slot],birth)
                representatives && push!(er[slot],_PersistenceRepresentative{T}(birth,nothing,chain(s.change[b]),nothing))
            elseif birth!=s.grades[d]
                death=s.grades[d];push!(finite[slot],(birth,death))
                representatives && push!(fr[slot],_PersistenceRepresentative{T}(birth,death,chain(s.reduced[d]),chain(s.change[d])))
            end
        end
        for (values,reps) in ((finite,fr),(essential,er)),d in 1:nd
            issorted(values[d];rev=s.order===:superlevel) && continue
            perm=sortperm(values[d];rev=s.order===:superlevel);values[d]=values[d][perm]
            representatives && (reps[d]=reps[d][perm])
        end
        meta=(;source=:incremental_complex,revision=s.revision,backend=:retained_sparse_reduction,update_query=:retained_reduction,
            grade_arithmetic=:exact_stored_values,approximation=:none_in_reduction,
            source_identity=:monotone_cell_ids,construction=(;requested=:graded_chain_complex,effective=:graded_chain_complex,substitution=:none))
        PersistenceDiagram(finite,essential,s.field,s.order,meta,
            representatives ? _PersistenceRepresentatives{T}(fr,er) : nothing)
    end
end

"""Release a state's reduction and source storage. Previously returned diagrams,
updates and optional witnesses remain independent and usable."""
function close_persistence!(s::PersistenceState)
    lock(s.lock) do
        for name in (:ids,:labels,:dims,:grades,:slots,:permutation,:rank,:boundary,:reduced,
                :change,:reduced_rows,:change_rows,:cofaces,:pivot)
            setfield!(s,name,empty(getfield(s,name)))
        end
        s.entries=0;s.retained=false;s.closed=true;s
    end
end
