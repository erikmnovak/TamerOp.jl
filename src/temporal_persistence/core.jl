"""Bounds on temporal ingestion, retained exact algebra, and landscape queries.
Matrix entries include dense homology work; no sampling replaces exhausted work.
"""
struct TemporalBudget
    max_occurrences::Int
    max_scales::Int
    max_points::Int
    max_cells::Int
    max_matrix_entries::Int
    max_order_entries::Int
    max_landscape_queries::Int
    function TemporalBudget(;max_occurrences=63,max_scales=32,max_points=256,
            max_cells=20_000,max_matrix_entries=4_000_000,max_order_entries=1_000_000,
            max_landscape_queries=10_000)
        xs=(max_occurrences,max_scales,max_points,max_cells,max_matrix_entries,max_order_entries,max_landscape_queries)
        all(x->x isa Integer && !(x isa Bool) && 0<x<=typemax(Int),xs) ||
            throw(ArgumentError("Temporal budgets must be positive integers fitting Int."))
        new(Int.(xs)...)
    end
end
_bound(x,limit,name)=x<=limit || throw(ArgumentError("Temporal $name budget exceeded ($x > $limit)."))
_index(x,n,name)=x isa Integer && !(x isa Bool) && 1<=x<=n ? Int(x) : throw(ArgumentError("$name must be an integer in 1:$n."))
_field(F)=F isa Union{QQField,PrimeField} ? F : throw(ArgumentError("Temporal persistence requires QQ or a prime field."))
function _axis(xs,name;strict=true)
    isempty(xs) && throw(ArgumentError("$name cannot be empty."))
    all(x->x isa Union{Integer,Rational,AbstractFloat} && !(x isa Bool) && isfinite(x),xs) ||
        throw(ArgumentError("$name must contain finite integer, rational or floating values."))
    # Binary floating values are retained exactly, without rational guessing.
    out=Rational{BigInt}.(xs)
    issorted(out) && (!strict || allunique(out)) || throw(ArgumentError("$name must be $(strict ? "strictly increasing" : "nondecreasing")."))
    out
end
_zbudget(b)=ZigzagPersistence.ZigzagBudget(max_vertices=b.max_occurrences,
    max_dimension=b.max_cells,max_cells=b.max_cells,max_matrix_entries=b.max_matrix_entries,
    max_representative_entries=b.max_matrix_entries)

"""A finite functor on a directed time path times an ordered scale grid.
Use `temporal_barcode`, `temporal_map`, `temporal_source`, `temporal_cycle` and
`spatiotemporal_landscape` to inspect it. Sources and returned mathematical
objects are read only. Repeated time labels remain distinct occurrences.
"""
struct TemporalPersistenceResult{K,F}
    module_object::Modules.PModule{K}
    times::Vector{Rational{BigInt}}
    scales::Vector{Rational{BigInt}}
    directions::Vector{Symbol}
    sources::Vector{DataIngestion.FilteredComplex}
    barcodes::Vector{ZigzagPersistence.ZigzagResult}
    homology::Union{Nothing,Matrix{Any}}
    active::Matrix{Vector{Int}}
    degree::Int
    field::F
    provenance::NamedTuple
    budget::TemporalBudget
end
_temporal_vertex(r,i,j)=i+(j-1)*length(r.times)
function temporal_summary(r::TemporalPersistenceResult)
    (;kind=:temporal_persistence,occurrences=length(r.times),scales=length(r.scales),degree=r.degree,
        field=r.field,representatives=r.homology!==nothing,directions=Tuple(r.directions),
        construction=r.provenance.construction,domain=:finite_occurrence_scale_grid,
        source_cells=sum(x->sum(DataTypes.cell_counts(x.complex)),r.sources),
        landscape_metric=:max_norm_on_integer_indices)
end
describe(r::TemporalPersistenceResult)=temporal_summary(r)
Base.show(io::IO,r::TemporalPersistenceResult)=print(io,"TemporalPersistenceResult(",length(r.times)," occurrences, ",length(r.scales)," scales, H",r.degree,", ",r.field,")")
"""Return copies of occurrence labels, spatial scales and directed time edges."""
temporal_axes(r::TemporalPersistenceResult)=(;times=copy(r.times),scales=copy(r.scales),directions=copy(r.directions))
"""The exact zigzag at one stored scale index. Endpoints refer to occurrences."""
temporal_barcode(r::TemporalPersistenceResult;scale_index)=r.barcodes[_index(scale_index,length(r.scales),"scale_index")]
"""The retained source filtration at an occurrence; bridge states are sources too."""
temporal_source(r::TemporalPersistenceResult;occurrence)=r.sources[_index(occurrence,length(r.times),"occurrence")]
"""Return the exact finite module. Its label is `occurrence + (scale_index-1)*n`."""
temporal_module(r::TemporalPersistenceResult)=r.module_object
"""Copy a structure map between comparable `(occurrence, scale_index)` pairs.
Backward arrows are queried in their actual direction; no inverse is invented.
"""
function temporal_map(r::TemporalPersistenceResult;source,target)
    length(source)==length(target)==2 || throw(ArgumentError("source and target need occurrence and scale indices."))
    v(x)=_temporal_vertex(r,_index(x[1],length(r.times),"occurrence"),_index(x[2],length(r.scales),"scale_index"))
    a,b=v(source),v(target)
    FiniteFringe.leq(r.module_object.Q,a,b) || throw(ArgumentError("The selected temporal vertices are incomparable or reversed."))
    copy(Modules.map_leq(r.module_object,a,b))
end
"""Lift one compatible interval representative to a checked source cycle.
Requires `representatives=true` during temporal construction. The interval basis
is a choice made by exact zigzag reduction, not a canonical naming of classes.
"""
function temporal_cycle(r::TemporalPersistenceResult;scale_index,interval,occurrence)
    j=_index(scale_index,length(r.scales),"scale_index");i=_index(occurrence,length(r.times),"occurrence")
    r.homology===nothing && throw(ArgumentError("Construct with representatives=true to retain source cycles."))
    bar=temporal_barcode(r;scale_index=j)
    v=ZigzagPersistence.zigzag_representative(bar;interval,vertex=i)
    K=coeff_type(r.field);x=r.sources[i];counts=DataTypes.cell_counts(x.complex)
    n=r.degree<length(counts) ? counts[r.degree+1] : 0
    coeffs=zeros(K,n);coeffs[r.active[i,j]]=vec(ChainComplexes.homology_representative(r.homology[i,j],v))
    c=DataIngestion.SourceChain(x,coeffs,r.degree,(r.scales[j],),r.field,:cycle,
        (;choice=:compatible_zigzag_interval,interval,occurrence=i,scale_index=j,canonical=false))
    DataIngestion.check_source_chain(c;throw=true);c
end

struct TemporalValidationSummary
    valid::Bool
    issues::Vector{String}
end
Base.show(io::IO,v::TemporalValidationSummary)=print(io,"TemporalValidationSummary(valid=",v.valid,", issues=",length(v.issues),")")
describe(v::TemporalValidationSummary)=(;valid=v.valid,issues=copy(v.issues))
"""Check the time/scale functor, each row decomposition and retained source cycles."""
function check_temporal(r::TemporalPersistenceResult;throw=false)
    issues=String[]
    try
        n,m=length(r.times),length(r.scales)
        _axis(r.times,"times";strict=false)==r.times || error("Time labels disagree.")
        _axis(r.scales,"scales")==r.scales || error("Scale labels disagree.")
        length(r.sources)==n && length(r.directions)==n-1 && length(r.barcodes)==m || error("Temporal array lengths disagree.")
        size(r.active)==(n,m) || error("Active source cell table has the wrong shape.")
        r.homology===nothing || size(r.homology)==(n,m) || error("Homology table has the wrong shape.")
        FiniteFringe.nvertices(r.module_object.Q)==n*m || error("Finite category has the wrong size.")
        _bound((big(n)*m)^2,r.budget.max_order_entries,"order entries")
        Modules.check_module(r.module_object;throw=true)
        order=_time_order(r.directions,n)
        for a in 1:n*m,b in 1:n*m
            FiniteFringe.leq(r.module_object.Q,a,b)==(order[mod1(a,n),mod1(b,n)] && cld(a,n)<=cld(b,n)) || error("Temporal category disagrees with its directed product.")
        end
        for j in 1:m-1,e in 1:n-1
            a,b=r.directions[e]===:forward ? (e,e+1) : (e+1,e)
            temporal_map(r;source=(b,j),target=(b,j+1))*temporal_map(r;source=(a,j),target=(b,j))==
                temporal_map(r;source=(a,j+1),target=(b,j+1))*temporal_map(r;source=(a,j),target=(a,j+1)) || error("A temporal square does not commute.")
        end
        for j in 1:m
            bar=r.barcodes[j];ZigzagPersistence.check_zigzag(bar;throw=true)
            bar.diagram.dims==r.module_object.dims[(1:n).+(j-1)*n] || error("Temporal barcode dimensions disagree.")
            bar.diagram.directions==r.directions && bar.diagram.times==r.times || error("Barcode arrow/time convention disagrees.")
            for e in 1:n-1
                a,b=r.directions[e]===:forward ? (e,e+1) : (e+1,e)
                bar.diagram.maps[e]==temporal_map(r;source=(a,j),target=(b,j)) || error("Temporal row map disagrees.")
            end
            if r.homology!==nothing
                for (id,(a,b)) in enumerate(bar.supports),i in a:b
                    temporal_cycle(r;scale_index=j,interval=id,occurrence=i)
                end
            end
        end
    catch err
        err isa InterruptException && rethrow();push!(issues,sprint(showerror,err))
    end
    throw && !isempty(issues) && Base.throw(ArgumentError(join(issues,"; ")))
    TemporalValidationSummary(isempty(issues),issues)
end
