# Portable in-memory result contract used by owned serialization, without
# coupling the earlier-loaded JSON owner to private representative types.

_diagram_chain_data(c::_PersistenceChain) =
    (indices=copy(c.indices),ids=copy(c.ids),grades=copy(c.grades),coefficients=copy(c.coefficients))

"""
    persistence_diagram_data(diagram)

Copy a diagram into a named-tuple data contract carrying the grade type, exact
intervals, field, order, metadata, and optional retained cycle/cochain records.
This is a heavier interchange operation; use ordinary semantic accessors for
queries. `persistence_diagram_from_data` checks and restores this contract.
"""
function persistence_diagram_data(d::PersistenceDiagram{T}) where {T}
    check_persistence_diagram(d;throw=true)
    reps = d.retained_representatives
    repdata(r) = (birth=r.birth,death=r.death,cycle=_diagram_chain_data(r.cycle),
        bounding_chain=r.bounding_chain === nothing ? nothing : _diagram_chain_data(r.bounding_chain))
    cocycles = d.retained_cocycles
    cocdata(r) = (birth=r.birth,death=r.death,cochain=_diagram_chain_data(r.cochain),vertices=deepcopy(r.vertices))
    return (grade_type=T,finite=deepcopy(d.finite_by_dim),essential=deepcopy(d.essential_by_dim),
        field=d.field,order=d.order,meta=deepcopy(d.meta),
        representatives=reps === nothing ? nothing :
            (finite=[[repdata(r) for r in rows] for rows in reps.finite],essential=[[repdata(r) for r in rows] for rows in reps.essential]),
        cocycles=cocycles === nothing ? nothing : (indexing=cocycles.indexing,
            finite=[[cocdata(r) for r in rows] for rows in cocycles.finite],essential=[[cocdata(r) for r in rows] for rows in cocycles.essential]))
end

function _diagram_data_keys(data,names)
    data isa NamedTuple && Set(keys(data)) == Set(names) || throw(ArgumentError("invalid persistence-diagram data fields; expected $names"))
end
function _diagram_chain_from_data(c,::Type{T}) where {T}
    _diagram_data_keys(c,(:indices,:ids,:grades,:coefficients))
    all(v -> v isa AbstractVector && all(x -> x isa Integer && !(x isa Bool),v),
        (c.indices,c.ids,c.coefficients)) || throw(ArgumentError("chain indices, IDs and coefficients must be integer vectors"))
    c.grades isa AbstractVector && all(x -> x isa T,c.grades) || throw(ArgumentError("chain grades disagree with grade_type"))
    return _PersistenceChain{T}(Int.(c.indices),Int.(c.ids),T[x for x in c.grades],Int.(c.coefficients))
end

"""
    persistence_diagram_from_data(data)

Restore the owned `persistence_diagram_data` contract with strict endpoint,
field, order and retained-storage checks. This checks stored structure and
filtration support; without the source differentials it does not independently
re-prove cycle/cocycle equations. Source geometry is never inferred from IDs.
"""
function persistence_diagram_from_data(data)
    _diagram_data_keys(data,(:grade_type,:finite,:essential,:field,:order,:meta,:representatives,:cocycles))
    T = data.grade_type
    T isa Type && T <: Real && isconcretetype(T) || throw(ArgumentError("grade_type must be a concrete real type"))
    data.finite isa AbstractVector && all(rows -> rows isa AbstractVector &&
        all(bd -> bd isa Tuple{T,T},rows),data.finite) || throw(ArgumentError("finite intervals disagree with grade_type"))
    data.essential isa AbstractVector && all(rows -> rows isa AbstractVector && all(x -> x isa T,rows),data.essential) ||
        throw(ArgumentError("essential births disagree with grade_type"))
    data.field isa PrimeField && data.order isa Symbol && data.meta isa NamedTuple || throw(ArgumentError("invalid field, order or metadata"))
    # Values already have type T. Calling T(x) would round a BigFloat to the
    # ambient precision, destroying the precision restored by the scalar codec.
    finite = Vector{Tuple{T,T}}[Tuple{T,T}[(b,d) for (b,d) in rows] for rows in data.finite]
    essential = Vector{T}[T[x for x in rows] for rows in data.essential]
    reps = if data.representatives === nothing
        nothing
    else
        _diagram_data_keys(data.representatives,(:finite,:essential))
        function restore_rep(r)
            _diagram_data_keys(r,(:birth,:death,:cycle,:bounding_chain))
            r.birth isa T && (r.death === nothing || r.death isa T) || throw(ArgumentError("representative grades disagree with grade_type"))
            _PersistenceRepresentative{T}(r.birth,r.death,
                _diagram_chain_from_data(r.cycle,T),r.bounding_chain === nothing ? nothing : _diagram_chain_from_data(r.bounding_chain,T))
        end
        _PersistenceRepresentatives{T}([_PersistenceRepresentative{T}[restore_rep(r) for r in rows] for rows in data.representatives.finite],
            [_PersistenceRepresentative{T}[restore_rep(r) for r in rows] for rows in data.representatives.essential])
    end
    coc = if data.cocycles === nothing
        nothing
    else
        _diagram_data_keys(data.cocycles,(:finite,:essential,:indexing))
        function restore_coc(r)
            _diagram_data_keys(r,(:birth,:death,:cochain,:vertices))
            r.birth isa T && (r.death === nothing || r.death isa T) || throw(ArgumentError("cocycle grades disagree with grade_type"))
            r.vertices === nothing || (r.vertices isa AbstractVector && all(v -> v isa AbstractVector &&
                all(i -> i isa Integer && !(i isa Bool),v),r.vertices)) || throw(ArgumentError("invalid cocycle vertices"))
            _PersistenceCocycle{T}(r.birth,r.death,
                _diagram_chain_from_data(r.cochain,T),r.vertices === nothing ? nothing : [Int.(v) for v in r.vertices])
        end
        _PersistenceCocycles{T}([_PersistenceCocycle{T}[restore_coc(r) for r in rows] for rows in data.cocycles.finite],
            [_PersistenceCocycle{T}[restore_coc(r) for r in rows] for rows in data.cocycles.essential],data.cocycles.indexing)
    end
    diag = PersistenceDiagram(finite,essential,_normalize_field(data.field),_normalize_order(data.order),deepcopy(data.meta),reps,coc)
    check_persistence_diagram(diag;throw=true)
    return diag
end
