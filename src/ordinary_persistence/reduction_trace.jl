# Opt-in reference column reduction. Ordinary production reducers never record a trace.

"""Bounds for an inspectable reference reduction. `max_matrix_entries` bounds
the entries in each dense matrix; the trace stores operations, not a matrix
copy for every step. These are logical storage limits, not byte estimates."""
struct ReductionBudget
    max_cells::Int
    max_matrix_entries::Int
    max_operations::Int
    function ReductionBudget(;max_cells=128,max_matrix_entries=16_384,max_operations=20_000)
        xs=(max_cells,max_matrix_entries,max_operations)
        all(x->x isa Integer && !(x isa Bool) && 0<x<=typemax(Int),xs) ||
            throw(ArgumentError("Reduction budgets must be positive machine integers."))
        new(Int.(xs)...)
    end
end
describe(b::ReductionBudget)=(;max_cells=b.max_cells,max_matrix_entries=b.max_matrix_entries,max_operations=b.max_operations)

struct _ReductionOperation{K}
    kind::Symbol
    column::Int
    source::Int
    coefficient::K
end

"""An exact, explicitly named reference column reduction over a prime field.
Create with `persistence_reduction`. Cells have a deterministic filtration order
(grade, then dimension, then original storage order); cell labels may repeat
across dimensions. Inspect with `reduction_step`, `reduction_cells`,
`reduction_pairs` and `reduction_witness`. This trace does not describe the
operations of the optimized ordinary, cubical, cohomology or Rips reducers.
Treat its internal arrays as read-only; public accessors return copies."""
struct PersistenceReduction{T,K,S}
    boundary::Matrix{K}
    cells::Vector{@NamedTuple{index::Int,source_index::Int,dimension::Int,local_index::Int,label::Int,grade::T}}
    operations::Vector{_ReductionOperation{K}}
    field::PrimeField
    order::Symbol
    source::S
    budget::ReductionBudget
end

"""Validation of a reference reduction, including its complete operation
sequence, chain identity and comparison with ordinary persistence."""
struct ReductionValidationSummary
    valid::Bool
    issues::Vector{String}
    operations_checked::Int
    ordinary_checked::Bool
end
Base.:(==)(a::ReductionValidationSummary,b::ReductionValidationSummary)=
    (a.valid,a.issues,a.operations_checked,a.ordinary_checked)==(b.valid,b.issues,b.operations_checked,b.ordinary_checked)
Base.isequal(a::ReductionValidationSummary,b::ReductionValidationSummary)=a==b
Base.hash(a::ReductionValidationSummary,h::UInt)=hash((a.valid,Tuple(a.issues),a.operations_checked,a.ordinary_checked),h)
Base.show(io::IO,r::ReductionValidationSummary)=print(io,"ReductionValidationSummary(valid=",r.valid,", operations=",r.operations_checked,")")
describe(r::ReductionValidationSummary)=(;r.valid,issues=copy(r.issues),r.operations_checked,r.ordinary_checked)

_reduction_pivot(R,j)=something(findlast(!iszero,view(R,:,j)),0)
function _reduction_size(n,b)
    n<=b.max_cells || throw(ArgumentError("Reference reduction cell budget exceeded ($n > $(b.max_cells))."))
    big(n)^2<=b.max_matrix_entries || throw(ArgumentError("Reference reduction matrix-entry budget exceeded."))
end
function _reduction_add!(R,V,j,k,a)
    for i in axes(R,1)
        R[i,j]+=a*R[i,k]
        V[i,j]+=a*V[i,k]
    end
end
function _reduction_order(G,order)
    cells=collect(eachindex(G.grades));dims=G.cell_dims
    sort!(cells;lt=(a,b)->G.grades[a]==G.grades[b] ?
        (dims[a]==dims[b] ? a<b : dims[a]<dims[b]) :
        (order===:sublevel ? G.grades[a][1]<G.grades[b][1] : G.grades[a][1]>G.grades[b][1]))
end

"""
    persistence_reduction(complex; field=F2(), order=:sublevel, budget=ReductionBudget())

Record a bounded reference reduction of a one-parameter `GradedComplex` or
`FilteredComplex`. Each column is started, simplified by additions of earlier
completed columns, then finalized. Replaying an operation applies the same
addition to `R` and `V`, preserving `R = D*V`. Geometry is retained only when
explicitly supplied by a `FilteredComplex`. No optimized reducer is instrumented.
"""
function persistence_reduction(G::GradedComplex{N,T};field=F2(),order=:sublevel,budget=ReductionBudget()) where {N,T}
    _normalize_field(field);_normalize_order(order)
    budget isa ReductionBudget || throw(ArgumentError("Use a ReductionBudget."))
    n=length(G.grades);_reduction_size(n,budget)
    K=coeff_type(field);offsets=_validate_complex(G,order,K)
    length(offsets)-1<=budget.max_cells || throw(ArgumentError("Reference reduction chain-degree budget exceeded."))
    all(allunique,G.cells_by_dim) || throw(ArgumentError("Source cell labels must be unique within each chain degree."))
    perm=_reduction_order(G,order);rank=invperm(perm)
    cells=[(;index=j,source_index=i,dimension=G.cell_dims[i],local_index=i-offsets[G.cell_dims[i]+1]+1,
        label=G.cell_ids[i],grade=G.grades[i][1]) for (j,i) in enumerate(perm)]
    # Supply an explicit element type for an empty complex.
    C=@NamedTuple{index::Int,source_index::Int,dimension::Int,local_index::Int,label::Int,grade::T}
    D=zeros(K,n,n)
    for (d,B) in enumerate(G.boundaries),j in axes(B,2),z in nzrange(B,j)
        D[rank[offsets[d]+rowvals(B)[z]-1],rank[offsets[d+1]+j-1]]=K(nonzeros(B)[z])
    end
    operations=_ReductionOperation{K}[];R=copy(D);V=Matrix{K}(I,n,n);owners=zeros(Int,n)
    function record(kind,j,k=0,a=zero(K))
        length(operations)<budget.max_operations || throw(ArgumentError("Reference reduction operation budget exceeded."))
        push!(operations,_ReductionOperation(kind,j,k,a))
    end
    for j in 1:n
        record(:start,j)
        while true
            p=_reduction_pivot(R,j);p==0 && break
            k=owners[p];k==0 && break
            a=-R[p,j]/R[p,k]
            record(:add,j,k,a);_reduction_add!(R,V,j,k,a)
        end
        p=_reduction_pivot(R,j);p==0 || (owners[p]=j)
        record(:finish,j)
    end
    PersistenceReduction(D,C[c for c in cells],operations,field,order,nothing,budget)
end
function persistence_reduction(x::DataIngestion.FilteredComplex;field=F2(),order=:sublevel,budget=ReductionBudget())
    budget isa ReductionBudget || throw(ArgumentError("Use a ReductionBudget."))
    _reduction_size(length(x.complex.grades),budget)
    DataIngestion.check_filtered_complex(x;throw=true)
    DataIngestion.grade_orientation(x)==(order===:sublevel ? 1 : -1,) ||
        throw(ArgumentError("Source orientation must agree with the reduction order."))
    r=persistence_reduction(DataIngestion.graded_complex(x);field,order,budget)
    PersistenceReduction(r.boundary,r.cells,r.operations,r.field,r.order,deepcopy(x),r.budget)
end

function reduction_summary(r::PersistenceReduction)
    (;kind=:persistence_reduction,algorithm=:reference_column_reduction,field=r.field,order=r.order,
      cells=length(r.cells),operations=length(r.operations),column_additions=count(o->o.kind===:add,r.operations),
      geometry=r.source!==nothing,tie_order=(:dimension,:source_storage),storage=:matrix_and_operations,
      budget=describe(r.budget))
end
describe(r::PersistenceReduction)=reduction_summary(r)
Base.show(io::IO,r::PersistenceReduction)=print(io,"PersistenceReduction(",length(r.cells)," cells, ",length(r.operations)," reference operations over F",r.field.p,")")
field(r::PersistenceReduction)=r.field
filtration_order(r::PersistenceReduction)=r.order
"""Cell identities in reduction order. `source_index` refers to the original
dimension-grouped complex; `index` is the matrix row/column, starting at one."""
reduction_cells(r::PersistenceReduction)=copy(r.cells)
"""The exact reference operations, with their prime-field coefficients."""
reduction_operations(r::PersistenceReduction)=[(;o.kind,o.column,o.source,o.coefficient) for o in r.operations]

function _reduction_step_index(r,step)
    step isa Integer && !(step isa Bool) && 0<=step<=length(r.operations) ||
        throw(ArgumentError("Reduction step must be in 0:$(length(r.operations))."))
    Int(step)
end
function _reduction_replay(r,step;validate=false)
    n=length(r.cells);K=eltype(r.boundary);R=copy(r.boundary);V=Matrix{K}(I,n,n)
    owners=zeros(Int,n);prefix=0;active=0
    for o in @view r.operations[1:step]
        j,k,a=o.column,o.source,o.coefficient
        if o.kind===:start
            active==0 && j==prefix+1 && j<=n && k==0 && iszero(a) ||
                throw(ArgumentError("Invalid start-column operation."))
            active=j
        elseif o.kind===:add
            active==j && 1<=k<=prefix && r.cells[k].dimension==r.cells[j].dimension ||
                throw(ArgumentError("Column addition must use an earlier completed column of the same dimension."))
            p=_reduction_pivot(R,j)
            p>0 && owners[p]==k && a==-R[p,j]/R[p,k] ||
                throw(ArgumentError("Column addition does not cancel the current pivot."))
            _reduction_add!(R,V,j,k,a)
        elseif o.kind===:finish
            active==j && k==0 && iszero(a) || throw(ArgumentError("Invalid finish-column operation."))
            p=_reduction_pivot(R,j)
            p==0 || (owners[p]==0 && iszero(view(R,:,p))) ||
                throw(ArgumentError("A finalized pivot is already paired or did not birth a class."))
            p==0 || (owners[p]=j)
            prefix=j;active=0
        else
            throw(ArgumentError("Unknown reference reduction operation $(o.kind)."))
        end
    end
    # Every accepted operation preserves the identity; check its accumulated
    # result once, avoiding a cubic matrix product after every scalar event.
    validate && R!=r.boundary*V && throw(ArgumentError("The reduction identity R = D*V failed."))
    (;R,V,owners,prefix,active,provisional_pivot=active==0 ? 0 : _reduction_pivot(R,active))
end
function _reduction_pairs(r,s)
    [(;dimension=r.cells[b].dimension,birth_cell=b,death_cell=s.owners[b],
        birth=r.cells[b].grade,death=s.owners[b]==0 ? nothing : r.cells[s.owners[b]].grade,
        zero_length=s.owners[b]!=0 && r.cells[b].grade==r.cells[s.owners[b]].grade)
        for b in 1:s.prefix if iszero(view(s.R,:,b))]
end
"""
    reduction_step(trace; step=0)

Reconstruct `D`, `R` and `V` at an operation boundary. `prefix` is the number of
fully processed columns; `active` and `provisional_pivot` describe unfinished
arithmetic. `pairs` belongs only to that prefix. An unpaired birth can die at a
later step, and a prefix ending inside a grade tie need not be a full sublevel.
"""
function reduction_step(r::PersistenceReduction;step=0)
    k=_reduction_step_index(r,step);s=_reduction_replay(r,k)
    n=length(r.cells);tied=0<s.prefix<n && r.cells[s.prefix].grade==r.cells[s.prefix+1].grade
    (;step=k,D=copy(r.boundary),R=s.R,V=s.V,prefix=s.prefix,active=s.active,
      provisional_pivot=s.provisional_pivot,pairs=_reduction_pairs(r,s),complete=k==length(r.operations),
      inside_grade_tie=tied,operation=k==0 ? nothing : _reduction_operation_at(r,k))
end
function _reduction_operation_at(r,k)
    o=r.operations[k];(;o.kind,o.column,o.source,o.coefficient)
end
"""Finalized pairs and currently unpaired births at a selected operation step.
Equal-grade pairs are retained and marked `zero_length`; ordinary persistence
omits their empty intervals. A zero death-cell index means unpaired in this prefix."""
reduction_pairs(r::PersistenceReduction;step=length(r.operations))=
    reduction_step(r;step).pairs

function _reduction_chain(r,v)
    indices=findall(!iszero,v)
    (;cells=copy(r.cells[indices]),coefficients=copy(v[indices]))
end
function _reduction_witness(r,s,birth)
    birth isa Integer && !(birth isa Bool) || throw(ArgumentError("Choose a birth cell from reduction_pairs."))
    pair=findfirst(p->p.birth_cell==birth,s.pairs)
    pair===nothing && throw(ArgumentError("Cell $birth is not a finalized birth in this prefix."))
    p=s.pairs[pair];d=p.death_cell
    cycle=d==0 ? copy(s.V[:,birth]) : copy(s.R[:,d])
    filling=d==0 ? nothing : copy(s.V[:,d])
    iszero(r.boundary*cycle) || error("Retained reduction cycle has a nonzero boundary.")
    filling===nothing || r.boundary*filling==cycle || error("Retained filling has the wrong boundary.")
    (;pair=p,cycle=_reduction_chain(r,cycle),filling=filling===nothing ? nothing : _reduction_chain(r,filling),
      cycle_vector=cycle,filling_vector=filling,field=r.field,order=r.order,prefix=s.prefix,
      choice=:reference_reduction,normalization=:as_reduced)
end
"""Return a checked cycle and, for a finalized finite pair, a filling whose
boundary is that cycle. Coefficients use reduction-order cells and are basis
dependent. This does not choose a shortest or geometrically preferred cycle."""
reduction_witness(r::PersistenceReduction;birth,step=length(r.operations))=
    _reduction_witness(r,reduction_step(r;step),birth)

function persistence_diagram(r::PersistenceReduction{T};step=length(r.operations)) where {T}
    s=reduction_step(r;step)
    nd=isempty(r.cells) ? 1 : maximum(c.dimension for c in r.cells)+1
    finite=[Tuple{T,T}[] for _ in 1:nd];essential=[T[] for _ in 1:nd]
    for p in s.pairs
        p.death===nothing ? push!(essential[p.dimension+1],p.birth) :
            p.zero_length || push!(finite[p.dimension+1],(p.birth,p.death))
    end
    foreach(x->sort!(x;rev=r.order===:superlevel),finite)
    foreach(x->sort!(x;rev=r.order===:superlevel),essential)
    PersistenceDiagram(finite,essential;field=r.field,order=r.order,
        meta=(;algorithm=:reference_column_reduction,prefix=s.prefix,inside_grade_tie=s.inside_grade_tie))
end

# Reconstruct the dimension-grouped complex from the declared ordered matrix.
function _reduction_complex(r)
    n=length(r.cells);nd=isempty(r.cells) ? 1 : maximum(c.dimension for c in r.cells)+1
    groups=[findall(c->c.dimension==d,r.cells) for d in 0:nd-1]
    cells=[[r.cells[i].label for i in g] for g in groups]
    boundaries=[sparse(Int[r.boundary[i,j].val for i in groups[d],j in groups[d+1]]) for d in 1:nd-1]
    grades=[(r.cells[i].grade,) for g in groups for i in g]
    GradedComplex(cells,boundaries,grades)
end

"""
    check_persistence_reduction(trace; throw=false)

Validate mutable storage, the field, filtration and tie order, `D\u00b2 = 0`, every
operation and `R = D*V`, and the final interval multiset against ordinary
persistence. Validation replays the bounded trace; it is not a constant-time query.
"""
function check_persistence_reduction(r::PersistenceReduction;throw=false)
    issues=String[];checked=0;ordinary=false
    try
        _normalize_order(r.order);_normalize_field(r.field)
        K=coeff_type(r.field);eltype(r.boundary)===K || error("Boundary coefficient field mismatch.")
        n=length(r.cells);_reduction_size(n,r.budget)
        length(r.operations)<=r.budget.max_operations || error("Operation budget exceeded.")
        size(r.boundary)==(n,n) || error("Boundary matrix shape mismatch.")
        sort([c.source_index for c in r.cells])==collect(1:n) || error("Source indices must form a permutation.")
        all(c->0<=c.dimension<r.budget.max_cells && c.local_index>=1 && isfinite(c.grade),r.cells) || error("Invalid cell dimension, local index or grade.")
        for (i,c) in enumerate(r.cells)
            c.index==i || error("Matrix cell indices are inconsistent.")
            if i>1
                a=r.cells[i-1]
                ordered=a.grade==c.grade ? (a.dimension,a.source_index)<(c.dimension,c.source_index) :
                    (r.order===:sublevel ? a.grade<c.grade : a.grade>c.grade)
                ordered || error("Filtration or deterministic tie order is invalid.")
            end
            for j in 1:n
                iszero(r.boundary[j,i]) && continue
                j<i && r.cells[j].dimension==c.dimension-1 || error("Boundary violates dimension or filtration order.")
            end
        end
        for d in unique([c.dimension for c in r.cells])
            group=sort(filter(c->c.dimension==d,r.cells);by=c->c.source_index)
            [c.local_index for c in group]==collect(1:length(group)) || error("Source local indices are inconsistent.")
            allunique([c.label for c in group]) || error("Cell labels repeat within one chain degree.")
        end
        iszero(r.boundary*r.boundary) || error("Boundary squared is nonzero in the selected field.")
        s=_reduction_replay(r,length(r.operations);validate=true);checked=length(r.operations)
        s.prefix==n && s.active==0 || error("The operation trace is incomplete.")
        if r.source!==nothing
            DataIngestion.check_filtered_complex(r.source;throw=true)
            other=persistence_reduction(r.source;field=r.field,order=r.order,budget=r.budget)
            other.cells==r.cells && other.boundary==r.boundary || error("Retained source and reduction disagree.")
        end
        G=_reduction_complex(r);a=persistence_diagram(G;field=r.field,order=r.order);b=persistence_diagram(r)
        for d in 0:DataTypes.max_dim(G)
            finite_intervals(a;dim=d)==finite_intervals(b;dim=d) && essential_births(a;dim=d)==essential_births(b;dim=d) ||
                error("Final pairs disagree with ordinary persistence.")
        end
        ordinary=true
    catch err
        err isa InterruptException && rethrow()
        push!(issues,sprint(showerror,err))
    end
    valid=isempty(issues)
    throw && !valid && Base.throw(ArgumentError(join(issues,"\n")))
    ReductionValidationSummary(valid,issues,checked,ordinary)
end
