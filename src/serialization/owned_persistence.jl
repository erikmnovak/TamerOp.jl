# Owned ordinary-persistence results. Safe tagged data, never eval or Julia
# object deserialization; exact scalars and supported provenance types survive.

const _PD_TYPES = Dict{String,Any}(string(T)=>T for T in
    (Any,Nothing,Missing,Bool,Int8,Int16,Int32,Int64,Int128,UInt8,UInt16,UInt32,UInt64,UInt128,
     BigInt,Float16,Float32,Float64,BigFloat,Rational,AlgebraicReal,String,Symbol,
     Tuple,NamedTuple,Array,Vector,Matrix,Dict,Pair,UnitRange,StepRange,AbstractFloat,Real))
for (name,T) in (("PointCloud",PointCloud),("ImageNd",ImageNd),("GraphData",GraphData),
                 ("EmbeddedPlanarGraph2D",EmbeddedPlanarGraph2D),("GradedComplex",GradedComplex),
                 ("MultiCriticalGradedComplex",MultiCriticalGradedComplex),("SparseMatrixCSC",SparseMatrixCSC))
    _PD_TYPES[name] = T
end

_persistence_owner() = getfield(parentmodule(@__MODULE__),:OrdinaryPersistence)
function _pd_keys(obj,expected)
    obj isa AbstractDict && Set(String.(keys(obj))) == Set(expected) ||
        throw(ArgumentError("invalid persistence JSON keys; expected $(collect(expected))"))
end
function _pd_type(T)
    T isa Union && return Dict("tag"=>"union","parameters"=>[_pd_encode(t) for t in Base.uniontypes(T)])
    for (name,base) in _PD_TYPES
        T === base && return Dict("tag"=>"type","name"=>name,"parameters"=>[])
    end
    if T isa DataType
        base = Base.typename(T).wrapper
        for (name,allowed) in _PD_TYPES
            base === allowed && return Dict("tag"=>"type","name"=>name,"parameters"=>[_pd_encode(p) for p in T.parameters])
        end
    end
    throw(ArgumentError("unsupported persistence metadata/grade type $T"))
end

function _pd_encode(x)
    x === nothing && return Dict("tag"=>"nothing")
    x === missing && return Dict("tag"=>"missing")
    x isa Bool && return Dict("tag"=>"bool","value"=>x)
    x isa Type && return _pd_type(x)
    x isa PrimeField && return Dict("tag"=>"field","p"=>x.p)
    x isa Symbol && return Dict("tag"=>"symbol","value"=>String(x))
    x isa AbstractString && return Dict("tag"=>"string","value"=>String(x))
    x isa Integer && return Dict("tag"=>"integer","type"=>_pd_type(typeof(x)),"value"=>string(x))
    if x isa Rational
        return Dict("tag"=>"rational","type"=>_pd_type(typeof(x)),"value"=>rational_to_string(QQ(x)))
    elseif x isa AlgebraicReal
        a = _algebraic_coordinate_obj(x)
        return Dict("tag"=>"algebraic","polynomial"=>a.polynomial,"real_root_index"=>a.real_root_index)
    elseif x isa AbstractFloat
        value = string(x)
        return Dict("tag"=>"float","type"=>_pd_type(typeof(x)),"value"=>value,"precision"=>precision(x))
    elseif x isa NamedTuple
        return Dict("tag"=>"named_tuple","names"=>String.(collect(keys(x))),"values"=>[_pd_encode(v) for v in values(x)])
    elseif x isa Tuple
        return Dict("tag"=>"tuple","values"=>[_pd_encode(v) for v in x])
    elseif x isa Pair
        return Dict("tag"=>"pair","values"=>[_pd_encode(first(x)),_pd_encode(last(x))])
    elseif x isa AbstractArray
        return Dict("tag"=>"array","element_type"=>_pd_type(eltype(x)),"size"=>collect(size(x)),"values"=>[_pd_encode(v) for v in x][:])
    elseif x isa AbstractDict
        return Dict("tag"=>"dict","key_type"=>_pd_type(keytype(x)),"value_type"=>_pd_type(valtype(x)),"entries"=>[_pd_encode(k=>v) for (k,v) in pairs(x)])
    end
    DI = getfield(parentmodule(@__MODULE__),:DataIngestion)
    if x isa DI.LandmarkSelection
        data = describe(x)
        return Dict("tag"=>"landmarks","summary"=>_pd_encode(data),
                    "distances"=>_pd_encode(data.distances_retained ? DI.landmark_distances(x) : nothing))
    end
    throw(ArgumentError("unsupported persistence metadata value $(typeof(x)); use supported named-tuple data instead"))
end

function _pd_decode(obj,depth=0)
    depth <= 128 || throw(ArgumentError("persistence metadata nesting exceeds 128 levels"))
    obj isa AbstractDict && haskey(obj,"tag") || throw(ArgumentError("persistence data requires a tagged object"))
    tag = String(obj["tag"])
    read(x) = _pd_decode(x,depth+1)
    if tag in ("nothing","missing")
        _pd_keys(obj,("tag",)); return tag == "nothing" ? nothing : missing
    elseif tag in ("bool","symbol","string")
        _pd_keys(obj,("tag","value"))
        tag == "bool" && (obj["value"] isa Bool || throw(ArgumentError("invalid Bool")); return obj["value"])
        obj["value"] isa AbstractString || throw(ArgumentError("invalid string/symbol"))
        return tag == "symbol" ? Symbol(obj["value"]) : String(obj["value"])
    elseif tag == "type"
        _pd_keys(obj,("tag","name","parameters"))
        haskey(_PD_TYPES,String(obj["name"])) || throw(ArgumentError("unsupported serialized type"))
        base = _PD_TYPES[String(obj["name"])]
        params = read.(obj["parameters"])
        return isempty(params) ? base : Core.apply_type(base,params...)
    elseif tag == "union"
        _pd_keys(obj,("tag","parameters")); return Union{read.(obj["parameters"])...}
    elseif tag == "field"
        _pd_keys(obj,("tag","p"))
        obj["p"] isa Integer && !(obj["p"] isa Bool) || throw(ArgumentError("field characteristic must be an integer"))
        return PrimeField(Int(obj["p"]))
    elseif tag in ("integer","rational","float")
        _pd_keys(obj,tag == "float" ? ("tag","type","value","precision") : ("tag","type","value"))
        T = read(obj["type"]); value = String(obj["value"])
        if tag == "integer"
            T <: Integer && T !== Bool || throw(ArgumentError("invalid integer type"))
            v = parse(T,value); string(v) == value || throw(ArgumentError("noncanonical integer")); return v
        elseif tag == "rational"
            T <: Rational || throw(ArgumentError("invalid rational type"))
            return T(_exact_rational_coordinate(value))
        end
        T <: AbstractFloat || throw(ArgumentError("invalid float type"))
        bits = obj["precision"]
        bits isa Integer && !(bits isa Bool) && 2 <= bits <= 1_000_000 || throw(ArgumentError("invalid floating precision"))
        if T === BigFloat
            v = BigFloat(value,RoundNearest;precision=Int(bits))
            string(v) == value || throw(ArgumentError("noncanonical BigFloat value"))
            return v
        end
        v = parse(T,value)
        precision(v) == bits && string(v) == value || throw(ArgumentError("noncanonical floating value"))
        return v
    elseif tag == "algebraic"
        _pd_keys(obj,("tag","polynomial","real_root_index"))
        return _algebraic_coordinate_from_obj(obj)
    elseif tag == "named_tuple"
        _pd_keys(obj,("tag","names","values"))
        names = Symbol.(obj["names"])
        allunique(names) && length(names)==length(obj["values"]) || throw(ArgumentError("invalid named-tuple fields"))
        return NamedTuple{Tuple(names)}(Tuple(read.(obj["values"])))
    elseif tag in ("tuple","pair")
        _pd_keys(obj,("tag","values")); vals = Tuple(read.(obj["values"]))
        tag == "tuple" && return vals
        length(vals)==2 || throw(ArgumentError("a pair needs two entries")); return vals[1]=>vals[2]
    elseif tag == "array"
        _pd_keys(obj,("tag","element_type","size","values"))
        dims = obj["size"]
        all(n -> n isa Integer && !(n isa Bool) && 0 <= n <= typemax(Int),dims) || throw(ArgumentError("invalid array dimensions"))
        prod(big.(dims);init=big(1)) == length(obj["values"]) || throw(ArgumentError("array shape disagrees with stored entries"))
        T = read(obj["element_type"])
        values = read.(obj["values"])
        all(x -> x isa T,values) || throw(ArgumentError("array values disagree with declared element type"))
        vals = T[values...]
        return reshape(vals,Tuple(Int.(dims)))
    elseif tag == "dict"
        _pd_keys(obj,("tag","key_type","value_type","entries"))
        K,V = read(obj["key_type"]),read(obj["value_type"])
        result = Dict{K,V}()
        for pair in read.(obj["entries"])
            pair isa Pair && !haskey(result,first(pair)) || throw(ArgumentError("invalid or duplicate dictionary entry"))
            first(pair) isa K && last(pair) isa V || throw(ArgumentError("dictionary entries disagree with declared types"))
            result[first(pair)] = last(pair)
        end
        return result
    elseif tag == "landmarks"
        _pd_keys(obj,("tag","summary","distances"))
        s = read(obj["summary"]); distances = read(obj["distances"])
        DI = getfield(parentmodule(@__MODULE__),:DataIngestion)
        result = DI.LandmarkSelection(s.indices,s.insertion_radii,s.nearest_distances,s.nearest_landmark_indices,distances,s.geometry)
        isequal(describe(result),s) || throw(ArgumentError("landmark metadata is inconsistent"))
        return result
    end
    throw(ArgumentError("unknown persistence data tag $tag"))
end

"""
    save_persistence_diagram_json(path, diagram; profile=:compact, pretty=nothing)

Save a versioned owned result, including exact grade values/types, coefficient
field, orientation, all degrees, provenance and available cycles/cocycles.
Supported metadata consists of scalar/array/tuple/dictionary data, source types
from a closed allowlist, and landmark selections. Unsupported objects fail before
writing; they are never silently converted to display strings. Finite Float64 values
round-trip bit-exactly, including signed zero; BigFloat retains its precision.
"""
function save_persistence_diagram_json(path::AbstractString,diagram;profile::Symbol=:compact,pretty::Union{Nothing,Bool}=nothing)
    compact = _resolve_owned_json_pretty(profile,pretty)
    OP = _persistence_owner()
    diagram isa OP.PersistenceDiagram || throw(ArgumentError("expected a PersistenceDiagram"))
    data = OP.persistence_diagram_data(diagram)
    payload = _pd_encode(data)
    obj = Dict("kind"=>"PersistenceDiagram","schema_version"=>1,
        "field_characteristic"=>data.field.p,"order"=>String(data.order),
        "finite_counts"=>length.(data.finite),"essential_counts"=>length.(data.essential),
        "representatives_available"=>data.representatives !== nothing,
        "cocycles_available"=>data.cocycles !== nothing,"data"=>payload)
    return _json_write(path,obj;pretty=compact)
end

"""
    load_persistence_diagram_json(path)

Strictly restore an owned diagram. Only allowlisted data types are reconstructed;
no source text is evaluated. Structural representative validation is performed,
but independent algebraic certification requires the original complex.
"""
function load_persistence_diagram_json(path::AbstractString)
    try
        obj = JSON3.read(read(path,String))
        _pd_keys(obj,("kind","schema_version","field_characteristic","order","finite_counts","essential_counts","representatives_available","cocycles_available","data"))
        obj["kind"] == "PersistenceDiagram" && obj["schema_version"] isa Integer &&
            !(obj["schema_version"] isa Bool) && obj["schema_version"] == 1 || throw(ArgumentError("unsupported persistence diagram schema"))
        obj["field_characteristic"] isa Integer && !(obj["field_characteristic"] isa Bool) || throw(ArgumentError("invalid field header"))
        for key in ("finite_counts","essential_counts")
            obj[key] isa AbstractVector && all(n -> n isa Integer && !(n isa Bool) && n >= 0,obj[key]) || throw(ArgumentError("invalid count header"))
        end
        result = _persistence_owner().persistence_diagram_from_data(_pd_decode(obj["data"]))
        summary = describe(result)
        summary.field.p == obj["field_characteristic"] && String(summary.order) == obj["order"] &&
            collect(summary.finite_counts) == obj["finite_counts"] && collect(summary.essential_counts) == obj["essential_counts"] &&
            summary.representatives_available === obj["representatives_available"] && summary.cocycles_available === obj["cocycles_available"] ||
            throw(ArgumentError("persistence header disagrees with its data"))
        return result
    catch err
        err isa InterruptException && rethrow()
        throw(ArgumentError("invalid persistence diagram JSON: $(sprint(showerror,err))"))
    end
end

"""Inspect the owned persistence artifact header without restoring its retained chains."""
persistence_diagram_json_summary(path::AbstractString) =
    _expect_artifact_kind(inspect_json(path), "PersistenceDiagram", :persistence_diagram_json_summary)
"""Validate an owned persistence result and return a serialization report; `throw=true` rejects invalid files."""
check_persistence_diagram_json(path::AbstractString;throw::Bool=false) =
    _check_owned_json(path,"PersistenceDiagram",()->load_persistence_diagram_json(path),:check_persistence_diagram_json;throw)
