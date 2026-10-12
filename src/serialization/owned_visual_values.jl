# Declarative values shared by saved views and plotted-data exports. Runtime
# modules, callbacks, classifiers and caches are deliberately not object codecs.

const _VISUAL_DATA_BASE_TYPES = Dict{String,Any}(string(T)=>T for T in
    (Any,Nothing,Missing,Bool,Char,Int8,Int16,Int32,Int64,Int128,
     UInt8,UInt16,UInt32,UInt64,UInt128,BigInt,Float16,Float32,Float64,
     BigFloat,Rational,AlgebraicReal,String,Symbol,Tuple,NamedTuple,Array,
     Dict,Pair,UnitRange,StepRange,LinRange,AbstractFloat,Real,Number))
const _VISUAL_DATA_RECORDS = (:HeatmapLayer,:RectLayer,:PolygonLayer,:Polygon3Layer,
    :SegmentLayer,:PolylineLayer,:PointLayer,:Point3Layer,:TextLayer,:MatrixLayer,
    :Segment3Layer,:BarcodeLayer,:VisualizationSpec,:VisualStyle,:_VisualRole,
    :VisualContext,:VisualAnchor,:VisualAnnotation)
const _VISUAL_DATA_TYPE_NAMES = (_VISUAL_DATA_RECORDS...,:AbstractVisualizationLayer)

_visual_value_owner() = getfield(parentmodule(@__MODULE__),:Visualization)

# JSON object keys and mathematical dictionary entries both have canonical
# ordering. JSON objects alone do not cover dictionary data carried as pairs.
function _visual_data_canonical_json(obj)
    Base.@nospecialize obj
    plain=JSON3.read(JSON3.write(obj))
    io=IOBuffer()
    function writevalue(x)
        Base.@nospecialize x
        if x isa AbstractDict
            names=String.(collect(keys(x)))
            allunique(names) || throw(ArgumentError("duplicate declarative JSON keys"))
            print(io,'{')
            for (i,k) in enumerate(sort!(names))
                i>1 && print(io,',')
                print(io,JSON3.write(k),':')
                value=x[k]
                if k=="entries" && get(x,"tag",nothing) in ("dict","dictionary")
                    value=sort!(collect(value);by=entry->_visual_data_canonical_json(entry["values"][1]))
                end
                writevalue(value)
            end
            print(io,'}')
        elseif x isa AbstractVector
            print(io,'[')
            for (i,v) in enumerate(x)
                i>1 && print(io,',');writevalue(v)
            end
            print(io,']')
        else
            print(io,JSON3.write(x))
        end
    end
    writevalue(plain)
    return String(take!(io))
end

function _visual_data_budget(max_depth,max_entries)
    max_depth isa Integer && !(max_depth isa Bool) && 1 <= max_depth <= 1024 ||
        throw(ArgumentError("max_depth must be an integer in 1:1024"))
    max_entries isa Integer && !(max_entries isa Bool) && 1 <= max_entries <= typemax(Int) ||
        throw(ArgumentError("max_entries must be a positive integer"))
    return (;max_depth=Int(max_depth),max_entries=Int(max_entries),entries=Ref(0))
end

function _visual_data_step!(budget,depth,path)
    depth <= budget.max_depth || throw(ArgumentError("$path exceeds the declarative data nesting budget"))
    budget.entries[] < budget.max_entries || throw(ArgumentError("$path exceeds the declarative data entry budget"))
    budget.entries[] += 1
end

function _visual_data_keys(obj,expected,path)
    obj isa AbstractDict || throw(ArgumentError("$path requires a JSON object"))
    names=String.(collect(keys(obj)))
    allunique(names) && Set(names)==Set(expected) ||
        throw(ArgumentError("$path has invalid keys; expected $(collect(expected))"))
end

function _visual_data_base(name::String)
    haskey(_VISUAL_DATA_BASE_TYPES,name) && return _VISUAL_DATA_BASE_TYPES[name]
    name=="FpElem" && return FpElem
    name=="SparseMatrixCSC" && return SparseMatrixCSC
    name in String.(_VISUAL_DATA_TYPE_NAMES) && return getfield(_visual_value_owner(),Symbol(name))
    throw(ArgumentError("unsupported declarative type $name"))
end

function _visual_data_type(T,encode,path)
    Base.@nospecialize T
    T isa Union && return Dict("tag"=>"union","parameters"=>[encode(t,"$path.union") for t in Base.uniontypes(T)])
    if T isa Core.TypeofVararg
        isdefined(T,:N) || throw(ArgumentError("$path cannot store an unbounded Vararg"))
        return Dict("tag"=>"vararg","type"=>encode(T.T,"$path.element"),"length"=>encode(T.N,"$path.length"))
    end
    for (name,base) in _VISUAL_DATA_BASE_TYPES
        T === base && return Dict("tag"=>"type","name"=>name,"parameters"=>[])
    end
    for name in ("FpElem","SparseMatrixCSC",String.(_VISUAL_DATA_TYPE_NAMES)...)
        T === _visual_data_base(name) && return Dict("tag"=>"type","name"=>name,"parameters"=>[])
    end
    if T isa UnionAll
        body=Base.unwrap_unionall(T)
        if body isa DataType && Base.typename(body).wrapper===NamedTuple
            names=body.parameters[1]
            names isa Tuple && all(n->n isa Symbol,names) && T===NamedTuple{names} &&
                return Dict("tag"=>"named_tuple_type","names"=>encode(names,"$path.names"))
        end
    end
    if T isa DataType
        base=Base.typename(T).wrapper
        names=(collect(keys(_VISUAL_DATA_BASE_TYPES))...,"FpElem","SparseMatrixCSC",String.(_VISUAL_DATA_TYPE_NAMES)...)
        for name in names
            base === _visual_data_base(name) || continue
            return Dict("tag"=>"type","name"=>name,
                "parameters"=>[encode(p,"$path.parameter[$i]") for (i,p) in enumerate(T.parameters)])
        end
    end
    throw(ArgumentError("$path has unsupported declarative type $T"))
end

function _visual_data_encode(x,budget,depth,path)
    Base.@nospecialize x
    _visual_data_step!(budget,depth,path)
    write(v,p) = _visual_data_encode(v,budget,depth+1,p)
    (x isa Type || x isa Core.TypeofVararg) && return _visual_data_type(x,write,path)
    if x===nothing || x===missing || x isa Union{Bool,Symbol,AbstractString,Integer,Rational,AlgebraicReal,AbstractFloat}
        return Dict("tag"=>"scalar","value"=>_pd_encode(x))
    elseif x isa Char
        return Dict("tag"=>"character","value"=>string(x))
    elseif x isa QQField
        return Dict("tag"=>"qq_field")
    elseif x isa PrimeField
        return Dict("tag"=>"prime_field","characteristic"=>x.p)
    elseif x isa RealField
        return Dict("tag"=>"real_field","rtol"=>write(x.rtol,"$path.rtol"),"atol"=>write(x.atol,"$path.atol"))
    elseif x isa FpElem
        return Dict("tag"=>"prime_scalar","characteristic"=>typeof(x).parameters[1],"value"=>x.val)
    elseif x isa NamedTuple
        return Dict("tag"=>"named_tuple","names"=>String.(collect(keys(x))),
            "values"=>[write(v,"$path.$k") for (k,v) in pairs(x)])
    elseif x isa Tuple
        return Dict("tag"=>"tuple","values"=>[write(v,"$path[$i]") for (i,v) in enumerate(x)])
    elseif x isa Pair
        return Dict("tag"=>"pair","values"=>[write(first(x),"$path.key"),write(last(x),"$path.value")])
    elseif x isa UnitRange
        return Dict("tag"=>"unit_range","first"=>write(first(x),"$path.first"),"last"=>write(last(x),"$path.last"))
    elseif x isa StepRange
        return Dict("tag"=>"step_range","first"=>write(first(x),"$path.first"),
            "step"=>write(step(x),"$path.step"),"last"=>write(last(x),"$path.last"))
    elseif x isa LinRange
        return Dict("tag"=>"linear_range","first"=>write(first(x),"$path.first"),
            "last"=>write(last(x),"$path.last"),"length"=>length(x))
    elseif x isa SparseMatrixCSC
        return Dict("tag"=>"sparse_matrix","size"=>collect(size(x)),
            "element_type"=>write(eltype(x),"$path.element_type"),
            "index_type"=>write(eltype(x.rowval),"$path.index_type"),
            "column_pointers"=>write(x.colptr,"$path.column_pointers"),
            "row_indices"=>write(x.rowval,"$path.row_indices"),"values"=>write(x.nzval,"$path.values"))
    elseif x isa AbstractArray
        return Dict("tag"=>"array","element_type"=>write(eltype(x),"$path.element_type"),
            "size"=>collect(size(x)),"values"=>[write(v,"$path[$i]") for (i,v) in enumerate(x)][:])
    elseif x isa AbstractDict
        entries=[write(k=>v,"$path[$(repr(k))]") for (k,v) in pairs(x)]
        sort!(entries;by=entry->_visual_data_canonical_json(entry["values"][1]))
        return Dict("tag"=>"dictionary","key_type"=>write(keytype(x),"$path.key_type"),
            "value_type"=>write(valtype(x),"$path.value_type"),
            "entries"=>entries)
    end
    V=_visual_value_owner()
    name=Symbol(nameof(typeof(x)))
    if name in _VISUAL_DATA_RECORDS && x isa getfield(V,name)
        data=NamedTuple{fieldnames(typeof(x))}(Tuple(getfield(x,k) for k in fieldnames(typeof(x))))
        return Dict("tag"=>"visual_record","name"=>String(name),"fields"=>write(data,path))
    end
    throw(ArgumentError("$path contains unsupported runtime value $(typeof(x)); retain declarative data or attach an owned mathematical artifact"))
end

function _visual_data_dimensions(dims,path)
    dims isa AbstractVector && length(dims)<=32 &&
        all(n->n isa Integer && !(n isa Bool) && 0<=n<=typemax(Int),dims) ||
        throw(ArgumentError("$path requires nonnegative integer array dimensions"))
    return Tuple(Int.(dims))
end

function _visual_data_scalar(obj,budget,path)
    obj isa AbstractDict && haskey(obj,"tag") && obj["tag"] isa AbstractString ||
        throw(ArgumentError("$path requires an exact scalar payload"))
    tag=String(obj["tag"])
    tag in ("nothing","missing","bool","symbol","string","integer","rational","algebraic","float") ||
        throw(ArgumentError("$path has a non-scalar payload"))
    expected = tag in ("nothing","missing") ? ("tag",) :
        tag in ("bool","symbol","string") ? ("tag","value") :
        tag=="algebraic" ? ("tag","polynomial","real_root_index") :
        tag=="float" ? ("tag","type","value","precision") : ("tag","type","value")
    _visual_data_keys(obj,expected,path)
    if tag in ("integer","rational","float")
        haskey(obj,"type") || throw(ArgumentError("$path scalar lacks its type"))
        t=obj["type"]
        _visual_data_keys(t,("tag","name","parameters"),"$path.type")
        t["tag"]=="type" && t["name"] isa AbstractString && t["parameters"] isa AbstractVector ||
            throw(ArgumentError("invalid scalar type"))
        if tag=="rational"
            t["name"]=="Rational" && length(t["parameters"])==1 || throw(ArgumentError("invalid rational scalar type"))
            t=only(t["parameters"])
            _visual_data_keys(t,("tag","name","parameters"),"$path.type.parameter")
        end
        t["tag"]=="type" && t["name"] isa AbstractString &&
            t["parameters"] isa AbstractVector && isempty(t["parameters"]) || throw(ArgumentError("invalid scalar primitive type"))
        T=get(_VISUAL_DATA_BASE_TYPES,String(t["name"]),nothing)
        T isa Type && (tag=="float" ? T<:AbstractFloat : T<:Integer && T!==Bool) ||
            throw(ArgumentError("invalid scalar primitive type"))
    elseif tag=="algebraic"
        polynomial=get(obj,"polynomial",nothing)
        polynomial isa AbstractVector && 2<=length(polynomial)<=budget.max_entries-budget.entries[] ||
            throw(ArgumentError("invalid algebraic polynomial or entry budget"))
        budget.entries[] += length(polynomial)
        all(v->v isa AbstractString,polynomial) || throw(ArgumentError("invalid algebraic coefficients"))
        index=get(obj,"real_root_index",nothing)
        index isa Integer && !(index isa Bool) && 1<=index<=length(polynomial)-1 ||
            throw(ArgumentError("invalid algebraic root index"))
    end
    return _pd_decode(obj)
end

function _visual_data_construct(name,fields,path)
    Base.@nospecialize fields
    Symbol(name) in _VISUAL_DATA_RECORDS || throw(ArgumentError("$path names an unsupported visual record $name"))
    fields isa NamedTuple || throw(ArgumentError("$path visual fields must be named data"))
    V=_visual_value_owner(); T=getfield(V,Symbol(name))
    # Field sets are schema, not executable constructor keyword suggestions.
    expected = name=="VisualStyle" ? (:palette,:fontsize,:font,:mono_font,:linewidth_scale,:markersize_scale,:gap,:padding,:colormap,:colors) :
        name=="VisualAnnotation" ? (:from,:to,:kind,:label,:curve,:evidence) : fieldnames(T)
    keys(fields)==expected || throw(ArgumentError("$path has invalid $name fields"))
    if name in ("VisualStyle","VisualContext")
        return T(;fields...)
    elseif name=="VisualAnchor"
        return T(fields.panel;space=fields.space,point=fields.point,target=fields.target)
    elseif name=="VisualAnnotation"
        return V._check_annotation(T(values(fields)...))
    end
    return T(values(fields)...)
end

function _visual_data_decode(obj,budget,depth,path)
    Base.@nospecialize obj
    _visual_data_step!(budget,depth,path)
    obj isa AbstractDict && haskey(obj,"tag") && obj["tag"] isa AbstractString ||
        throw(ArgumentError("$path requires tagged declarative data"))
    tag=String(obj["tag"])
    read(v,p)=_visual_data_decode(v,budget,depth+1,p)
    keys!(names)=_visual_data_keys(obj,("tag",names...),path)
    if tag=="scalar"
        keys!(("value",)); return _visual_data_scalar(obj["value"],budget,path)
    elseif tag=="character"
        keys!(("value",)); v=obj["value"]
        v isa AbstractString && length(v)==1 || throw(ArgumentError("$path requires one character"))
        return only(v)
    elseif tag=="type"
        keys!(("name","parameters")); obj["name"] isa AbstractString || throw(ArgumentError("invalid type name"))
        obj["parameters"] isa AbstractVector || throw(ArgumentError("invalid type parameters"))
        base=_visual_data_base(String(obj["name"]))
        params=[read(v,"$path.parameter[$i]") for (i,v) in enumerate(obj["parameters"])]
        return isempty(params) ? base : Core.apply_type(base,params...)
    elseif tag=="named_tuple_type"
        keys!(("names",)); names=read(obj["names"],"$path.names")
        names isa Tuple && all(n->n isa Symbol,names) && allunique(names) ||
            throw(ArgumentError("invalid partially specified named-tuple type"))
        return NamedTuple{names}
    elseif tag=="union"
        keys!(("parameters",)); ps=obj["parameters"]
        ps isa AbstractVector && length(ps)>=2 || throw(ArgumentError("$path requires at least two union types"))
        types=[read(v,"$path.union") for v in ps]
        all(v->v isa Type,types) && allunique(types) || throw(ArgumentError("invalid union types"))
        return Union{types...}
    elseif tag=="vararg"
        keys!(("type","length")); T=read(obj["type"],"$path.type"); n=read(obj["length"],"$path.length")
        T isa Type && n isa Integer && !(n isa Bool) && 0<=n<=budget.max_entries || throw(ArgumentError("invalid fixed Vararg"))
        return Vararg{T,n}
    elseif tag=="qq_field"
        keys!(()); return QQField()
    elseif tag in ("prime_field","prime_scalar")
        keys!(tag=="prime_field" ? ("characteristic",) : ("characteristic","value"))
        p=obj["characteristic"]
        p isa Integer && !(p isa Bool) || throw(ArgumentError("invalid field characteristic"))
        F=PrimeField(p)
        tag=="prime_field" && return F
        v=obj["value"]
        v isa Integer && !(v isa Bool) && 0<=v<p || throw(ArgumentError("invalid canonical prime-field scalar"))
        return coerce(F,v)
    elseif tag=="real_field"
        keys!(("rtol","atol")); r=read(obj["rtol"],"$path.rtol"); a=read(obj["atol"],"$path.atol")
        r isa AbstractFloat && typeof(r)===typeof(a) && all(v->isfinite(v)&&v>=0,(r,a)) ||
            throw(ArgumentError("invalid real-field tolerances"))
        return RealField(typeof(r);rtol=r,atol=a)
    elseif tag=="named_tuple"
        keys!(("names","values")); ns=obj["names"]; vs=obj["values"]
        ns isa AbstractVector && vs isa AbstractVector && length(ns)==length(vs) &&
            all(v->v isa AbstractString,ns) && allunique(ns) || throw(ArgumentError("invalid named fields"))
        return NamedTuple{Tuple(Symbol.(ns))}(Tuple(read(v,"$path.$k") for (k,v) in zip(ns,vs)))
    elseif tag in ("tuple","pair")
        keys!(("values",)); obj["values"] isa AbstractVector || throw(ArgumentError("invalid tuple values"))
        vs=Tuple(read(v,"$path[$i]") for (i,v) in enumerate(obj["values"]))
        tag=="tuple" && return vs
        length(vs)==2 || throw(ArgumentError("a pair requires two values")); return vs[1]=>vs[2]
    elseif tag in ("unit_range","step_range","linear_range")
        keys!(tag=="unit_range" ? ("first","last") : tag=="step_range" ? ("first","step","last") : ("first","last","length"))
        a=read(obj["first"],"$path.first"); b=read(obj["last"],"$path.last")
        tag=="unit_range" && return UnitRange(a,b)
        tag=="step_range" && return StepRange(a,read(obj["step"],"$path.step"),b)
        n=obj["length"]; n isa Integer && !(n isa Bool) && 0<=n<=budget.max_entries || throw(ArgumentError("invalid linear-range length"))
        return LinRange(a,b,Int(n))
    elseif tag=="array"
        keys!(("element_type","size","values")); dims=_visual_data_dimensions(obj["size"],path)
        vs=obj["values"]; vs isa AbstractVector && prod(big.(dims);init=big(1))==length(vs) &&
            length(vs)<=budget.max_entries-budget.entries[] || throw(ArgumentError("$path array shape or budget disagrees with its entries"))
        T=read(obj["element_type"],"$path.element_type"); T isa Type || throw(ArgumentError("invalid element type"))
        data=[read(v,"$path[$i]") for (i,v) in enumerate(vs)]
        all(v->v isa T,data) || throw(ArgumentError("$path values disagree with the declared element type"))
        return reshape(T[data...],dims)
    elseif tag=="sparse_matrix"
        keys!(("size","element_type","index_type","column_pointers","row_indices","values"))
        dims=_visual_data_dimensions(obj["size"],path); length(dims)==2 || throw(ArgumentError("sparse matrix requires two dimensions"))
        T=read(obj["element_type"],"$path.element_type"); I=read(obj["index_type"],"$path.index_type")
        T isa Type && I isa Type && I<:Integer && I!==Bool || throw(ArgumentError("invalid sparse scalar/index types"))
        cp=read(obj["column_pointers"],"$path.column_pointers"); rows=read(obj["row_indices"],"$path.row_indices"); vals=read(obj["values"],"$path.values")
        cp isa Vector{I} && rows isa Vector{I} && vals isa Vector{T} && length(cp)==dims[2]+1 &&
            !isempty(cp) && first(cp)==1 && last(cp)==length(vals)+1 && length(rows)==length(vals) && issorted(cp) &&
            all(r->1<=r<=dims[1],rows) || throw(ArgumentError("invalid sparse CSC structure"))
        for j in 1:dims[2]
            lo,hi=Int(cp[j]),Int(cp[j+1])-1
            all(k->rows[k]<rows[k+1],lo:(hi-1)) || throw(ArgumentError("sparse rows must be strictly increasing in each column"))
        end
        return SparseMatrixCSC{T,I}(dims...,cp,rows,vals)
    elseif tag=="dictionary"
        keys!(("key_type","value_type","entries")); K=read(obj["key_type"],"$path.key_type"); V=read(obj["value_type"],"$path.value_type")
        K isa Type && V isa Type && obj["entries"] isa AbstractVector || throw(ArgumentError("invalid dictionary schema"))
        result=Dict{K,V}()
        for (i,raw) in enumerate(obj["entries"])
            p=read(raw,"$path.entry[$i]")
            p isa Pair && first(p) isa K && last(p) isa V && !haskey(result,first(p)) || throw(ArgumentError("invalid or duplicate dictionary entry"))
            result[first(p)]=last(p)
        end
        return result
    elseif tag=="visual_record"
        keys!(("name","fields")); obj["name"] isa AbstractString || throw(ArgumentError("invalid visual record name"))
        return _visual_data_construct(String(obj["name"]),read(obj["fields"],path),path)
    end
    throw(ArgumentError("$path has unknown declarative tag $tag"))
end

"""
    Serialization.visual_data_to_json(value; max_depth=128, max_entries=10_000_000)

Encode a declarative visual value as JSON-compatible tagged data. Supported
values include exact scalars, coefficient fields, typed containers, sparse
matrices, drawing layers, specifications, styles and semantic annotations.
Runtime mathematical objects, callbacks and caches are rejected with their data
path; save those objects using their owned result format. Array shape, exact
coefficient values, scalar types and explicit display coordinates are separate.
"""
function visual_data_to_json(value;max_depth=128,max_entries=10_000_000)
    Base.@nospecialize value
    return _visual_data_encode(value,_visual_data_budget(max_depth,max_entries),0,"data")
end

"""
    Serialization.visual_data_from_json(data; max_depth=128, max_entries=10_000_000)

Decode the closed declarative schema produced by `visual_data_to_json`.
Unknown tags, fields and types, duplicate names, inconsistent array shapes and
invalid sparse storage are rejected. No names are evaluated as Julia code.
This reconstructs drawing data, not live computation or theorem certificates.
"""
function visual_data_from_json(data;max_depth=128,max_entries=10_000_000)
    try
        return _visual_data_decode(data,_visual_data_budget(max_depth,max_entries),0,"data")
    catch err
        err isa ArgumentError && rethrow()
        throw(ArgumentError("invalid declarative visual data: $(sprint(showerror,err))"))
    end
end

_visual_data_obj(x;path="data",max_depth=128,max_entries=10_000_000) =
    _visual_data_encode(x,_visual_data_budget(max_depth,max_entries),0,path)
_visual_data_from_obj(x;path="data",max_depth=128,max_entries=10_000_000) =
    _visual_data_decode(x,_visual_data_budget(max_depth,max_entries),0,path)
