"""
    persistence_diagram(complex::FilteredComplex; representatives=false, cocycles=false,
                        field=F2(), order=:sublevel)
    persistence_diagram(correspondence::SourceCorrespondence; representatives=false,
                        cocycles=false, order=:sublevel)

Compute intervals while retaining the explicitly supplied source cells and
verified correspondence. Cycles/cochains are still separate opt-ins. Queries
return checked source chains; ordinary barcode plots do not request them.
`source=true` on data/filtration and cubical entrypoints constructs this retained
data explicitly and uses the explicit-complex route. Without geometric data,
use a supplied `FilteredComplex`; coordinates are never invented from cell IDs.
"""
function persistence_diagram(x::DataIngestion.FilteredComplex;field=F2(),kwargs...)
    return persistence_diagram(DataIngestion.source_correspondence(x;field);kwargs...)
end
function persistence_diagram(c::DataIngestion.SourceCorrespondence;
        order=:sublevel,representatives=false,cocycles=false)
    DataIngestion.check_source_correspondence(c;throw=true)
    c.retraction===nothing && throw(ArgumentError("Ordinary source-class retention requires a verified splitting retraction; use lift_source_chain for a general chain map."))
    c.computed.orientation==(order==:sublevel ? 1 : -1,) ||
        throw(ArgumentError("The source correspondence must have the requested one-parameter orientation."))
    d=persistence_diagram(DataIngestion.graded_complex(c.computed);order,field=c.field,representatives,cocycles)
    meta=merge(d.meta,(;geometry=:retained_source_cells,source_correspondence=c.certificate,
        source_construction=c.source.provenance))
    return PersistenceDiagram(d.finite_by_dim,d.essential_by_dim,d.field,d.order,meta,
        d.retained_representatives,d.retained_cocycles,c)
end

function _source_record_vector(c,record)
    K=coeff_type(c.field)
    G=DataIngestion.graded_complex(c.computed)
    n=length(G.cells_by_dim[record.dimension+1])
    v=zeros(K,n)
    offset=sum(length(G.cells_by_dim[k]) for k in 1:record.dimension;init=0)
    for (i,a,id,g) in zip(record.cell_indices,record.coefficients,record.cell_ids,record.cell_grades)
        1<=i<=n || throw(ArgumentError("Retained chain indices disagree with source cells."))
        G.cells_by_dim[record.dimension+1][i]==id && only(G.grades[offset+i])==g ||
            throw(ArgumentError("Retained chain IDs or grades disagree with the computed source cells."))
        v[i]=K(a)
    end
    return v
end

function _check_source_storage!(issues,diag)
    c=diag.retained_source
    c===nothing && return
    try
        c isa DataIngestion.SourceCorrespondence || error("Invalid source correspondence storage.")
        c.field==diag.field || error("Source correspondence field disagrees with diagram.")
        c.computed.orientation==(diag.order==:sublevel ? 1 : -1,) || error("Source orientation disagrees with diagram.")
        DataIngestion.check_source_correspondence(c;throw=true)
        c.retraction===nothing && error("Ordinary source-class retention requires a verified splitting retraction.")
        for (kind,groups) in ((:finite,diag.finite_by_dim),(:essential,diag.essential_by_dim))
            for (slot,group) in enumerate(groups),index in eachindex(group)
                diag.retained_representatives===nothing || persistence_representative(diag;dim=slot-1,kind,index)
                if diag.retained_cocycles!==nothing
                    birth=kind==:finite ? first(group[index]) : group[index]
                    death=kind==:finite ? last(group[index]) : (diag.order==:sublevel ? Inf : -Inf)
                    window=get(diag.meta,:window,:unrestricted)
                    scales=unique(vcat([birth],[only(g) for g in DataIngestion.graded_complex(c.source).grades]))
                    for scale in scales
                        (diag.order==:sublevel ? birth<=scale<death : death<scale<=birth) || continue
                        window isa Tuple && !(window[1]<=scale<=window[2]) && continue
                        persistence_cocycle(diag;dim=slot-1,kind,index,scale)
                    end
                end
            end
        end
    catch err
        err isa InterruptException && rethrow()
        push!(issues,sprint(showerror,err))
    end
end

function _representative_scale(diag,birth,death,scale)
    t=scale===nothing ? birth : scale
    t isa Real && !(t isa Bool) && isfinite(t) || throw(ArgumentError("scale must be a finite real parameter."))
    (diag.order==:sublevel ? birth<=t<death : death<t<=birth) ||
        throw(ArgumentError("scale must lie in the selected interval (finite death excluded)."))
    window=get(diag.meta,:window,:unrestricted)
    window isa Tuple && !(window[1]<=t<=window[2]) && throw(ArgumentError("scale is outside the constructed filtration window."))
    return t
end

function _representative_source(diag,result,scale)
    birth,death=result.interval
    t=_representative_scale(diag,birth,death,scale)
    c=diag.retained_source
    c===nothing && return merge(result,(;scale=t,source_cycle=nothing,source_bounding_chain=nothing))
    context=(;choice=result.choice,interval=result.interval,interval_kind=result.kind,member=result.index)
    cycle=_ordinary_source_context(DataIngestion.lift_source_chain(c;dim=result.dimension,
        coefficients=_source_record_vector(c,result.cycle),parameter=t,kind=:cycle),context)
    filling=result.bounding_chain===nothing ? nothing : _ordinary_source_context(DataIngestion.lift_source_chain(c;
        dim=result.dimension+1,coefficients=_source_record_vector(c,result.bounding_chain),parameter=death,kind=:chain),context)
    if filling!==nothing
        B=DataIngestion.DataTypes.boundary_map(DataIngestion.graded_complex(c.source);dim=result.dimension+1)
        SparseMatrixCSC{coeff_type(diag.field),Int}(B)*filling.coefficients==cycle.coefficients ||
            throw(ArgumentError("Retained source filling does not bound the selected cycle."))
    end
    return merge(result,(;scale=t,source_geometry=:verified_source_cells,source_cycle=cycle,source_bounding_chain=filling))
end

function _cocycle_source(diag,result)
    c=diag.retained_source
    c===nothing && return merge(result,(;source_cochain=nothing))
    cochain=DataIngestion.lift_source_chain(c;dim=result.dimension,
        coefficients=_source_record_vector(c,result.cochain),parameter=result.scale,kind=:cocycle)
    cochain=_ordinary_source_context(cochain,(;choice=result.choice,interval=result.interval,
        interval_kind=result.kind,member=result.index,variance=:contravariant))
    return merge(result,(;source_cochain=cochain))
end

_ordinary_source_context(c,context)=DataIngestion.SourceChain(c.source,c.coefficients,c.dimension,
    c.parameter,c.field,c.kind,merge(c.metadata,context))

# Portable sparse integer-residue matrices and primitive source geometry. This
# keeps JSON independent of private chain and ingestion implementation types.
function _source_complex_data(x)
    G=DataIngestion.graded_complex(x)
    return (;cells=[collect(v) for v in G.cells_by_dim],grades=copy(G.grades),
        boundaries=[(;size=size(B),entries=findnz(B)) for B in G.boundaries],
        geometry=deepcopy(x.geometry),sources=Any[deepcopy(s) for s in x.sources],orientation=x.orientation,provenance=x.provenance)
end
function _source_correspondence_data(c)
    c===nothing && return nothing
    matrix(M)=(;size=size(M),entries=let (r,s,v)=findnz(M);(r,s,getfield.(v,:val));end)
    return (;computed=_source_complex_data(c.computed),source=_source_complex_data(c.source),
        lift=matrix.(c.lift),retraction=c.retraction===nothing ? nothing : matrix.(c.retraction),certificate=c.certificate)
end
function _source_correspondence_from_data(data,field)
    data===nothing && return nothing
    _diagram_data_keys(data,(:computed,:source,:lift,:retraction,:certificate))
    function complex(d)
        _diagram_data_keys(d,(:cells,:grades,:boundaries,:geometry,:sources,:orientation,:provenance))
        B=[sparse(b.entries...,b.size...) for b in d.boundaries]
        G=GradedComplex(d.cells,B,d.grades)
        return DataIngestion.filtered_complex(G;geometry=d.geometry,source_ids=d.sources,
            orientation=d.orientation,construction=d.provenance.effective,model=d.provenance.model,provenance=d.provenance)
    end
    matrices(ds)=[sparse(r.entries[1],r.entries[2],coeff_type(field).(r.entries[3]),r.size...) for r in ds]
    # Certificates are recomputed; untrusted serialized labels cannot confer
    # an equivalence guarantee on an arbitrary supplied chain map.
    a,b=complex(data.computed),complex(data.source)
    c=DataIngestion.source_correspondence(a,b;field,lift=matrices(data.lift),
        retraction=data.retraction===nothing ? nothing : matrices(data.retraction))
    if data.certificate==:equal_grade_contraction
        automatic=DataIngestion.source_correspondence(a,b;field)
        c.lift==automatic.lift && c.retraction==automatic.retraction ||
            throw(ArgumentError("Stored source contraction disagrees with its recomputed certificate."))
        return automatic
    end
    data.certificate==:supplied_filtered_chain_map || throw(ArgumentError("Unknown source certificate."))
    return c
end

function _ordinary_source_cubical(G,shape,periodic,order,input)
    N=length(shape)
    dims=input==:vertices ? shape : ntuple(i->shape[i]+Int(!periodic[i]),N)
    provenance=(;image_values=input,grade_arithmetic=:exact_input_values,
        source_field=input,source_field_dimension=input==:top_cells ? N : 0,
        orientation_convention=:alternating_low_face_positive)
    orientation=(order==:sublevel ? 1 : -1,)
    if N>=2
        return DataIngestion.filtered_complex(G;cubical_shape=dims,periodic,orientation,
            construction="cubical $(input)",provenance)
    end
    n=only(dims);ne=n-Int(!only(periodic))
    geometry=vcat([[(i,0)] for i in 1:n],[[(i,0),(i+1,0)] for i in 1:ne])
    sources=vcat([(;origin=(i,),mask=(0,)) for i in 1:n],[ (;origin=(i,),mask=(1,)) for i in 1:ne])
    return DataIngestion.filtered_complex(G;geometry,source_ids=sources,orientation,
        construction="cubical $(input)",provenance=merge(provenance,(;periodic,
            domain=((1,n+Int(only(periodic))),),spatial_dim=1,source_mapping=:cubical_origin_mask)))
end
