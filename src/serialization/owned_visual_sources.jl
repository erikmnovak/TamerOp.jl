# Source objects remain mathematical artifacts. The visual scene keeps only a
# content identity and, optionally, one of these bounded owned source records.

function _visual_source_vertex_budget(n,max_vertices)
    max_vertices isa Integer && !(max_vertices isa Bool) && max_vertices>0 ||
        throw(ArgumentError("max_vertices must be a positive integer"))
    n isa Integer && !(n isa Bool) && 0<=n<=max_vertices ||
        throw(ArgumentError("source poset exceeds max_vertices=$max_vertices"))
    return Int(n)
end

function _visual_source_check_poset_budget(obj,max_vertices,depth=0)
    depth<=64 || throw(ArgumentError("source poset nesting exceeds 64"))
    obj isa AbstractDict && haskey(obj,"kind") && haskey(obj,"n") ||
        throw(ArgumentError("source poset requires a kind and vertex count"))
    _visual_source_vertex_budget(obj["n"],max_vertices)
    if obj["kind"]=="ProductPoset"
        _visual_source_check_poset_budget(obj["left"],max_vertices,depth+1)
        _visual_source_check_poset_budget(obj["right"],max_vertices,depth+1)
        big(obj["left"]["n"])*obj["right"]["n"]==obj["n"] || throw(ArgumentError("product poset counts disagree"))
    elseif obj["kind"]=="ProductOfChainsPoset"
        sizes=obj["sizes"]
        sizes isa AbstractVector && all(v->v isa Integer && !(v isa Bool) && v>=0,sizes) &&
            prod(big.(sizes);init=big(1))==obj["n"] || throw(ArgumentError("chain product counts disagree"))
    elseif obj["kind"]=="GridPoset"
        # Exact coordinate rows use offsets; ordinary rows store the axes.
        coords=_coordinate_rows_from_obj(obj["coords"],get(obj,"exact_coords",nothing))
        prod(big.(length.(coords));init=big(1))==obj["n"] || throw(ArgumentError("grid counts disagree"))
    end
end

function _visual_module_source_obj(M;max_vertices)
    Base.@nospecialize M
    MD=getfield(parentmodule(@__MODULE__),:Modules)
    M isa MD.PModule || throw(ArgumentError("visual module sources require a finite PModule"))
    _visual_source_vertex_budget(nvertices(M.Q),max_vertices)
    MD.check_module_functoriality(M;throw=true)
    edges=sort!(collect(M.edge_maps);by=first)
    return Dict("kind"=>"CoordinatePModule","schema_version"=>1,
        "basis_convention"=>"stored_coordinates","poset"=>_poset_obj(M.Q),
        "field"=>visual_data_to_json(M.field),"dimensions"=>copy(M.dims),
        "edges"=>[Dict("source"=>u,"target"=>v,"matrix"=>visual_data_to_json(A)) for ((u,v),A) in edges])
end

function _visual_module_source_from_obj(obj;max_vertices,shared_poset=nothing)
    _visual_data_keys(obj,("kind","schema_version","basis_convention","poset","field","dimensions","edges"),"module source")
    obj["kind"]=="CoordinatePModule" && obj["schema_version"]===1 && obj["basis_convention"]=="stored_coordinates" ||
        throw(ArgumentError("unsupported coordinate module source schema"))
    _visual_source_check_poset_budget(obj["poset"],max_vertices)
    P=shared_poset===nothing ? _parse_poset_from_obj(obj["poset"]) : shared_poset
    _visual_data_canonical_json(_poset_obj(P))==_visual_data_canonical_json(obj["poset"]) ||
        throw(ArgumentError("source poset differs from its canonical owned representation"))
    field=visual_data_from_json(obj["field"])
    field isa AbstractCoeffField || throw(ArgumentError("source field is not a coefficient field"))
    dims=obj["dimensions"]
    dims isa AbstractVector && length(dims)==nvertices(P) &&
        all(v->v isa Integer && !(v isa Bool) && 0<=v<=typemax(Int),dims) || throw(ArgumentError("invalid source dimensions"))
    K=coeff_type(field); edges=Dict{Tuple{Int,Int},AbstractMatrix{K}}()
    obj["edges"] isa AbstractVector || throw(ArgumentError("module source edges must be an array"))
    previous=(0,0)
    for edge in obj["edges"]
        _visual_data_keys(edge,("source","target","matrix"),"module source edge")
        u,v=edge["source"],edge["target"]
        all(i->i isa Integer && !(i isa Bool) && 1<=i<=nvertices(P),(u,v)) || throw(ArgumentError("invalid module edge labels"))
        key=(Int(u),Int(v)); key>previous || throw(ArgumentError("module edges must be sorted and unique"));previous=key
        A=visual_data_from_json(edge["matrix"])
        A isa AbstractMatrix{K} || throw(ArgumentError("source matrix coefficient type disagrees with its field"))
        edges[key]=A
    end
    MD=getfield(parentmodule(@__MODULE__),:Modules)
    MD.check_module_data(P,Int.(dims),edges;throw=true)
    typed_edges=if isempty(edges)
        Dict{Tuple{Int,Int},Matrix{K}}()
    else
        T=typeof(first(values(edges)))
        all(A->typeof(A)===T,values(edges)) || throw(ArgumentError("module source matrices must use one declared storage representation"))
        Dict{Tuple{Int,Int},T}(edges)
    end
    M=MD.PModule{K}(P,Int.(dims),typed_edges;field)
    MD.check_module_functoriality(M;throw=true)
    return M
end

function _visual_owned_source_payload(savefn,object)
    return mktemp() do path,io
        close(io);savefn(path,object)
        JSON3.read(read(path,String))
    end
end

function _visual_owned_source_restore(loadfn,payload)
    return mktemp() do path,io
        write(io,_visual_data_canonical_json(payload));close(io)
        loadfn(path)
    end
end

"""
    Serialization.visual_source_to_json(object; max_vertices=10_000)

Save a supported source as canonical owned JSON text, independently of a view.
Finite modules retain their labelled poset, coefficient field and every cover
matrix in the original stalk bases. Encodings reuse the owned encoding format
and add that coordinate module, so reopening a selected vector does not silently
change bases. Ordinary diagrams reuse their full owned result format. Runtime
caches, callbacks and construction workspaces are excluded.
"""
function visual_source_to_json(object;max_vertices=10_000)
    Base.@nospecialize object
    _visual_source_vertex_budget(0,max_vertices)
    MD=getfield(parentmodule(@__MODULE__),:Modules)
    OP=_persistence_owner()
    payload=if object isa MD.PModule
        Dict("source_kind"=>"pmodule","module"=>_visual_module_source_obj(object;max_vertices))
    elseif object isa EncodingResult
        _visual_source_vertex_budget(nvertices(object.P),max_vertices)
        M=materialize_module(object.M)
        M.Q===object.P || throw(ArgumentError("encoding and coordinate module use different poset objects"))
        object.opts.field==M.field || throw(ArgumentError("encoding options disagree with the coordinate module field"))
        opts=NamedTuple{fieldnames(typeof(object.opts))}(Tuple(getfield(object.opts,k) for k in fieldnames(typeof(object.opts))))
        Dict("source_kind"=>"encoding","encoding"=>_visual_owned_source_payload(save_encoding_json,object),
            "module"=>_visual_module_source_obj(M;max_vertices),"options"=>visual_data_to_json(opts),
            "backend"=>String(object.backend))
    elseif object isa OP.PersistenceDiagram
        Dict("source_kind"=>"persistence_diagram","diagram"=>_visual_owned_source_payload(save_persistence_diagram_json,object))
    else
        throw(ArgumentError("no owned visual source format for $(typeof(object)); supported sources are PModule, EncodingResult and PersistenceDiagram"))
    end
    return _visual_data_canonical_json(Dict("kind"=>"VisualSource","schema_version"=>1,"payload"=>payload))
end

"""
    Serialization.visual_source_from_json(text; max_vertices=10_000)

Restore a supported mathematical source, checking its owned schema, field,
labelled poset, matrix shapes and cover-path independence. Encoding sources keep
the saved coordinate bases as well as the separately owned fringe/classifier.
"""
function visual_source_from_json(text::AbstractString;max_vertices=10_000)
    try
        _visual_source_vertex_budget(0,max_vertices)
        obj=JSON3.read(text)
        _visual_data_keys(obj,("kind","schema_version","payload"),"visual source")
        obj["kind"]=="VisualSource" && obj["schema_version"]===1 || throw(ArgumentError("unsupported visual source schema"))
        p=obj["payload"]
        p isa AbstractDict && haskey(p,"source_kind") || throw(ArgumentError("source payload lacks its kind"))
        if p["source_kind"]=="pmodule"
            _visual_data_keys(p,("source_kind","module"),"module source payload")
            return _visual_module_source_from_obj(p["module"];max_vertices)
        elseif p["source_kind"]=="encoding"
            _visual_data_keys(p,("source_kind","encoding","module","options","backend"),"encoding source payload")
            _visual_source_check_poset_budget(p["encoding"]["poset"],max_vertices)
            enc=_visual_owned_source_restore(load_encoding_json,p["encoding"])
            M=_visual_module_source_from_obj(p["module"];max_vertices,shared_poset=enc.P)
            field=provenance(enc).field
            M.field==field && M.dims==materialize_module(enc.M).dims || throw(ArgumentError("coordinate module disagrees with the owned encoding field or dimensions"))
            o=visual_data_from_json(p["options"])
            o isa NamedTuple && keys(o)==(:backend,:max_regions,:strict_eps,:poset_kind,:field) || throw(ArgumentError("invalid saved encoding options"))
            opts=EncodingOptions(;o...)
            opts.field==field || throw(ArgumentError("encoding options disagree with the source field"))
            p["backend"] isa AbstractString && !isempty(p["backend"]) || throw(ArgumentError("invalid encoding backend identity"))
            recorded=_encoding_provenance_from_obj(p["encoding"]["mathematical_provenance"],enc.pi)
            meta=(;provenance=recorded)
            if haskey(p["encoding"],"coordinate_semantics")
                sem=p["encoding"]["coordinate_semantics"]
                meta=merge(meta,(;coordinate_semantics=(;grade_scale=Symbol(sem["grade_scale"]),orientation=Tuple(Int.(sem["orientation"])),grade_arithmetic=Symbol(sem["grade_arithmetic"]))))
            end
            return EncodingResult(enc.P,M,enc.pi;H=enc.H,opts,backend=Symbol(p["backend"]),meta)
        elseif p["source_kind"]=="persistence_diagram"
            _visual_data_keys(p,("source_kind","diagram"),"persistence source payload")
            return _visual_owned_source_restore(load_persistence_diagram_json,p["diagram"])
        end
        throw(ArgumentError("unsupported source kind $(p["source_kind"])"))
    catch err
        err isa InterruptException && rethrow()
        err isa ArgumentError && rethrow()
        throw(ArgumentError("invalid visual source: $(sprint(showerror,err))"))
    end
end

"""SHA-256 content identity of the canonical owned source, independent of runtime caches."""
visual_source_identity(object;kwargs...)=bytes2hex(SHA.sha256(codeunits(visual_source_to_json(object;kwargs...))))
