# File: Serialization.jl

"""
PosetModules.Serialization

All JSON-facing I/O lives here.

Separation of concerns
----------------------
A) Internal formats (owned/stable):
   - `save_*_json` / `load_*_json`
   - Schemas are controlled by PosetModules. Loaders are intentionally strict.

B) External adapters (CAS ingestion):
   - `parse_*_json` / `*_from_*`
   - Best-effort parsers for JSON emitted by external CAS tools (Macaulay2, Singular, ...).
     These schemas are not owned by PosetModules and may change upstream.

C) Invariant caches (MPPI):
   - `save_mpp_*_json` / `load_mpp_*_json`
   - Convenience cache formats for expensive derived objects defined in `PosetModules.Invariants`.

File structure (keep in this order)
-----------------------------------
1) Shared helpers
2) A. Internal formats
3) B. External adapters
4) C. Invariant caches
5) D. Additional serializers/loaders

If you add new JSON formats, put them in the appropriate section and keep the
public API functions (`save_*`, `load_*`, `parse_*`) at the top of that section.
"""
module Serialization

using JSON3
using SparseArrays

import ..CoreModules
using ..CoreModules: QQ, AbstractCoeffField, QQField, RealField, PrimeField,
    coeff_type, coerce, FpElem, rational_to_string, string_to_rational
import ..FlangeZn: Face, IndFlat, IndInj, Flange, canonical_matrix
import ..FiniteFringe: AbstractPoset, FinitePoset, ProductOfChainsPoset, GridPoset, ProductPoset,
                       FringeModule, nvertices, leq_matrix
import ..ZnEncoding: SignaturePoset, PackedSignatureRows
using ..FiniteFringe
using ..Modules: PModule, _clear_cover_cache!
using ..DataTypes: PointCloud, ImageNd, GraphData, EmbeddedPlanarGraph2D, GradedComplex,
                   MultiCriticalGradedComplex, SimplexTreeMulti
using ..Options: FiltrationSpec, ConstructionBudget, ConstructionOptions, PipelineOptions, EncodingOptions
using ..EncodingCore: GridEncodingMap, CompiledEncoding
using ..Results: EncodingResult
import ..Results: materialize_module
import ..ZnEncoding
import ..PLPolyhedra
import ..PLBackend
import ..IndicatorResolutions: pmodule_from_fringe, fringe_presentation

# Schema versions for JSON formats
const PIPELINE_SCHEMA_VERSION = 2
const ENCODING_SCHEMA_VERSION = 1
const PLFRINGE_SCHEMA_VERSION = 1
const TAMER_FEATURE_SCHEMA_VERSION = v"0.2.0"

# =============================================================================
# 1) Shared helpers
# =============================================================================

"""
    feature_schema_header(; format=nothing) -> Dict{String,Any}

Canonical schema header for feature artifacts owned by PosetModules.
"""
function feature_schema_header(; format::Union{Nothing,Symbol}=nothing)
    hdr = Dict{String,Any}(
        "kind" => "features",
        "schema_version" => string(TAMER_FEATURE_SCHEMA_VERSION),
    )
    format === nothing || (hdr["format"] = String(format))
    return hdr
end

"""
    validate_feature_metadata_schema(meta; max_version=TAMER_FEATURE_SCHEMA_VERSION)

Validate a feature metadata object against the canonical feature schema header.
Returns `true` on success and throws on invalid/unsupported schema tags.
"""
function validate_feature_metadata_schema(meta; max_version::VersionNumber=TAMER_FEATURE_SCHEMA_VERSION)
    kind = haskey(meta, "kind") ? String(meta["kind"]) : ""
    kind == "features" || error("Feature metadata has unsupported kind: $(kind)")
    haskey(meta, "schema_version") || error("Feature metadata missing schema_version")
    ver = try
        VersionNumber(String(meta["schema_version"]))
    catch
        bad = haskey(meta, "schema_version") ? meta["schema_version"] : missing
        error("Feature metadata has invalid schema_version: $(bad)")
    end
    ver <= max_version || error("Unsupported feature metadata schema_version: $(ver)")
    return true
end

function _field_to_obj(field::AbstractCoeffField)
    if field isa QQField
        return Dict("kind" => "qq")
    elseif field isa RealField
        T = coeff_type(field)
        return Dict("kind" => "real",
                    "T" => string(T),
                    "rtol" => field.rtol,
                    "atol" => field.atol)
    elseif field isa PrimeField
        return Dict("kind" => "fp", "p" => field.p)
    end
    error("Unsupported coefficient field for JSON serialization: $(typeof(field))")
end

function _field_from_obj(obj)
    kind = lowercase(String(obj["kind"]))
    if kind == "qq"
        return QQField()
    elseif kind == "real"
        Tname = String(obj["T"])
        T = Tname == "Float64" ? Float64 :
            Tname == "Float32" ? Float32 :
            error("Unsupported real field type in JSON: $(Tname)")
        rtol = haskey(obj, "rtol") ? T(obj["rtol"]) : sqrt(eps(T))
        atol = haskey(obj, "atol") ? T(obj["atol"]) : zero(T)
        return RealField(T; rtol=rtol, atol=atol)
    elseif kind == "fp"
        p = Int(obj["p"])
        return PrimeField(p)
    end
    error("Unsupported coeff_field kind: $(kind)")
end

function _scalar_to_json(field::AbstractCoeffField, x)
    if field isa QQField
        return rational_to_string(QQ(x))
    elseif field isa RealField
        return Float64(x)
    elseif field isa PrimeField
        return Int(coerce(field, x).val)
    end
    error("Unsupported coefficient field for scalar serialization: $(typeof(field))")
end

function _scalar_from_json(field::AbstractCoeffField, val)
    if field isa QQField
        if val isa Integer
            return QQ(BigInt(val))
        end
        s = String(val)
        if occursin("/", s)
            return string_to_rational(s)
        end
        return QQ(parse(BigInt, strip(s)))
    elseif field isa RealField
        T = coeff_type(field)
        return val isa AbstractString ? T(parse(Float64, val)) : T(val)
    elseif field isa PrimeField
        return coerce(field, val isa AbstractString ? parse(Int, val) : Int(val))
    end
    error("Unsupported coefficient field for scalar parsing: $(typeof(field))")
end

@inline function _json_write(path::AbstractString, obj; pretty::Bool=true, indent::Int=2)
    open(path, "w") do io
        if pretty && indent > 0
            ac = JSON3.AlignmentContext(:Left, UInt8(clamp(indent, 0, 255)), UInt8(0), UInt8(0))
            JSON3.pretty(io, obj, ac; allow_inf=true)
        else
            JSON3.write(io, obj; allow_inf=true)
        end
    end
    return path
end

@inline _json_read(path::AbstractString) = open(JSON3.read, path)

@inline function _resolve_validation_mode(validation::Symbol)::Bool
    validation === :strict && return true
    validation === :trusted && return false
    error("validation must be :strict or :trusted. Use :strict for external/untrusted files and :trusted for PosetModules-produced files you trust.")
end

@inline function _resolve_encoding_output_mode(output::Symbol)::Symbol
    output === :fringe && return :fringe
    output === :fringe_with_pi && return :fringe_with_pi
    output === :encoding_result && return :encoding_result
    error("output must be one of :fringe, :fringe_with_pi, :encoding_result.")
end

@inline function _resolve_encoding_save_profile(profile::Symbol)
    profile === :compact && return (include_pi=true, include_leq=:auto, pretty=false)
    profile === :portable && return (include_pi=true, include_leq=true, pretty=false)
    profile === :debug && return (include_pi=true, include_leq=true, pretty=true)
    error("profile must be :compact, :portable, or :debug.")
end

"""
    inspect_json(path) -> NamedTuple

Quick metadata probe for PosetModules-owned JSON artifacts.
Returns a compact summary without reconstructing full domain objects.
"""
function inspect_json(path::AbstractString)
    obj = _json_read(path)
    kind = haskey(obj, "kind") ? String(obj["kind"]) : "unknown"
    schema_version = haskey(obj, "schema_version") ? Int(obj["schema_version"]) : nothing

    if kind == "FiniteEncodingFringe"
        poset = obj["poset"]
        coeff = obj["coeff_field"]
        return (
            kind = kind,
            schema_version = schema_version,
            field = haskey(coeff, "kind") ? String(coeff["kind"]) : "unknown",
            poset_kind = haskey(poset, "kind") ? String(poset["kind"]) : "unknown",
            nvertices = haskey(poset, "n") ? Int(poset["n"]) : missing,
            n_upsets = haskey(obj, "U") && haskey(obj["U"], "nrows") ? Int(obj["U"]["nrows"]) : missing,
            n_downsets = haskey(obj, "D") && haskey(obj["D"], "nrows") ? Int(obj["D"]["nrows"]) : missing,
            has_pi = haskey(obj, "pi"),
        )
    elseif kind == "FlangeZn"
        return (
            kind = kind,
            schema_version = schema_version,
            n = haskey(obj, "n") ? Int(obj["n"]) : missing,
            n_flats = haskey(obj, "flats") ? length(obj["flats"]) : missing,
            n_injectives = haskey(obj, "injectives") ? length(obj["injectives"]) : missing,
            has_phi = haskey(obj, "phi"),
        )
    elseif kind == "PLFringe"
        return (
            kind = kind,
            schema_version = schema_version,
            n = haskey(obj, "n") ? Int(obj["n"]) : missing,
            n_upsets = haskey(obj, "ups") ? length(obj["ups"]) : missing,
            n_downsets = haskey(obj, "downs") ? length(obj["downs"]) : missing,
            has_phi = haskey(obj, "phi"),
        )
    elseif kind == "PointCloud" || kind == "GraphData" || kind == "ImageNd" ||
           kind == "EmbeddedPlanarGraph2D" || kind == "GradedComplex" ||
           kind == "MultiCriticalGradedComplex" || kind == "SimplexTreeMulti"
        return (kind = kind, schema_version = schema_version)
    elseif haskey(obj, "dataset") && haskey(obj, "spec")
        dataset = obj["dataset"]
        return (
            kind = "PipelineJSON",
            schema_version = schema_version,
            dataset_kind = haskey(dataset, "kind") ? String(dataset["kind"]) : "unknown",
            has_pipeline_options = haskey(obj, "pipeline_options"),
            has_degree = haskey(obj, "degree"),
        )
    else
        return (kind = kind, schema_version = schema_version)
    end
end

Base.@kwdef mutable struct _PointCloudColumnarJSON
    kind::String = ""
    n::Int = 0
    d::Int = 0
    points_flat::Vector{Float64} = Float64[]
end
JSON3.StructTypes.StructType(::Type{_PointCloudColumnarJSON}) = JSON3.StructTypes.Mutable()

Base.@kwdef mutable struct _GraphDataColumnarJSON
    kind::String = ""
    n::Int = 0
    edges_u::Vector{Int} = Int[]
    edges_v::Vector{Int} = Int[]
    coords_dim::Union{Nothing,Int} = nothing
    coords_flat::Union{Nothing,Vector{Float64}} = nothing
    weights::Union{Nothing,Vector{Float64}} = nothing
end
JSON3.StructTypes.StructType(::Type{_GraphDataColumnarJSON}) = JSON3.StructTypes.Mutable()

@inline function _pointcloud_from_flat(n::Int, d::Int, flat::Vector{Float64})
    length(flat) == n * d || error("PointCloud points_flat length mismatch.")
    pts = Vector{Vector{Float64}}(undef, n)
    t = 1
    @inbounds for i in 1:n
        row = Vector{Float64}(undef, d)
        for j in 1:d
            row[j] = flat[t]
            t += 1
        end
        pts[i] = row
    end
    return PointCloud(pts)
end

@inline function _coords_from_flat(n::Int, d::Int, flat::Vector{Float64})
    d >= 0 || error("GraphData coords_dim must be nonnegative.")
    if d == 0
        return [Float64[] for _ in 1:n]
    end
    length(flat) == n * d || error("GraphData coords_flat length mismatch.")
    out = Vector{Vector{Float64}}(undef, n)
    t = 1
    @inbounds for i in 1:n
        row = Vector{Float64}(undef, d)
        for j in 1:d
            row[j] = flat[t]
            t += 1
        end
        out[i] = row
    end
    return out
end

@inline function _graph_from_columns(n::Int,
                                     edges_u::Vector{Int},
                                     edges_v::Vector{Int};
                                     coords_dim::Union{Nothing,Int}=nothing,
                                     coords_flat::Union{Nothing,Vector{Float64}}=nothing,
                                     weights::Union{Nothing,Vector{Float64}}=nothing)
    length(edges_u) == length(edges_v) || error("GraphData edge column lengths mismatch.")
    edges = Vector{Tuple{Int,Int}}(undef, length(edges_u))
    @inbounds for i in eachindex(edges_u)
        edges[i] = (edges_u[i], edges_v[i])
    end
    coords = if coords_dim === nothing || coords_flat === nothing
        nothing
    else
        _coords_from_flat(n, coords_dim, coords_flat)
    end
    return GraphData(n, edges; coords=coords, weights=weights, T=Float64)
end

@inline function _resolve_include_leq(P::AbstractPoset, include_leq::Union{Bool,Symbol})
    if include_leq === :auto
        return P isa FinitePoset
    end
    include_leq isa Bool || error("include_leq must be Bool or :auto.")
    return include_leq
end

# Typed encoding JSON schema (v1) for fast load paths.
abstract type _CoeffFieldJSON end
abstract type _PosetJSON end
abstract type _MaskJSON end
abstract type _PhiJSON end
abstract type _PiJSON end

Base.@kwdef mutable struct _QQFieldJSON <: _CoeffFieldJSON
    kind::String = "qq"
end

Base.@kwdef mutable struct _RealFieldJSON <: _CoeffFieldJSON
    kind::String = "real"
    T::String = "Float64"
    rtol::Union{Nothing,Float64} = nothing
    atol::Union{Nothing,Float64} = nothing
end

Base.@kwdef mutable struct _FpFieldJSON <: _CoeffFieldJSON
    kind::String = "fp"
    p::Int = 2
end

Base.@kwdef mutable struct _MaskPackedWordsJSON <: _MaskJSON
    kind::String = "packed_words_v1"
    nrows::Int = 0
    ncols::Int = 0
    words_per_row::Int = 0
    words::Vector{UInt64} = UInt64[]
end

Base.@kwdef mutable struct _FinitePosetJSON <: _PosetJSON
    kind::String = "FinitePoset"
    n::Int = 0
    leq::_MaskPackedWordsJSON = _MaskPackedWordsJSON()
end

Base.@kwdef mutable struct _ProductOfChainsPosetJSON <: _PosetJSON
    kind::String = "ProductOfChainsPoset"
    n::Int = 0
    sizes::Vector{Int} = Int[]
    leq::Union{Nothing,_MaskPackedWordsJSON} = nothing
end

Base.@kwdef mutable struct _GridPosetJSON <: _PosetJSON
    kind::String = "GridPoset"
    n::Int = 0
    coords::Vector{Vector{Float64}} = Vector{Vector{Float64}}()
    leq::Union{Nothing,_MaskPackedWordsJSON} = nothing
end

Base.@kwdef mutable struct _ProductPosetJSON <: _PosetJSON
    kind::String = "ProductPoset"
    n::Int = 0
    left::Union{Nothing,_PosetJSON} = nothing
    right::Union{Nothing,_PosetJSON} = nothing
    leq::Union{Nothing,_MaskPackedWordsJSON} = nothing
end

Base.@kwdef mutable struct _SignaturePosetJSON <: _PosetJSON
    kind::String = "SignaturePoset"
    n::Int = 0
    sig_y::_MaskPackedWordsJSON = _MaskPackedWordsJSON()
    sig_z::_MaskPackedWordsJSON = _MaskPackedWordsJSON()
    leq::Union{Nothing,_MaskPackedWordsJSON} = nothing
end

Base.@kwdef mutable struct _PhiQQChunksJSON <: _PhiJSON
    kind::String = "qq_chunks_v1"
    m::Int = 0
    k::Int = 0
    base::Int = 1_000_000_000
    num_sign::Vector{Int8} = Int8[]
    num_ptr::Vector{Int} = Int[]
    num_chunks::Vector{UInt32} = UInt32[]
    den_ptr::Vector{Int} = Int[]
    den_chunks::Vector{UInt32} = UInt32[]
end

Base.@kwdef mutable struct _PhiFpFlatJSON <: _PhiJSON
    kind::String = "fp_flat_v1"
    m::Int = 0
    k::Int = 0
    data::Vector{Int} = Int[]
end

Base.@kwdef mutable struct _PhiRealFlatJSON <: _PhiJSON
    kind::String = "real_flat_v1"
    m::Int = 0
    k::Int = 0
    data::Vector{Float64} = Float64[]
end

Base.@kwdef mutable struct _FaceGeneratorJSON
    b::Vector{Int} = Int[]
    tau::Vector{Int} = Int[]
end

Base.@kwdef mutable struct _GridEncodingMapJSON <: _PiJSON
    kind::String = "GridEncodingMap"
    coords::Vector{Vector{Float64}} = Vector{Vector{Float64}}()
    orientation::Vector{Int} = Int[]
end

Base.@kwdef mutable struct _ZnEncodingMapJSON <: _PiJSON
    kind::String = "ZnEncodingMap"
    n::Int = 0
    coords::Vector{Vector{Int}} = Vector{Vector{Int}}()
    sig_y::_MaskPackedWordsJSON = _MaskPackedWordsJSON()
    sig_z::_MaskPackedWordsJSON = _MaskPackedWordsJSON()
    reps::Vector{Vector{Int}} = Vector{Vector{Int}}()
    flats::Vector{_FaceGeneratorJSON} = _FaceGeneratorJSON[]
    injectives::Vector{_FaceGeneratorJSON} = _FaceGeneratorJSON[]
    cell_shape::Union{Nothing,Vector{Int}} = nothing
    cell_strides::Union{Nothing,Vector{Int}} = nothing
    cell_to_region::Union{Nothing,Vector{Int}} = nothing
end

Base.@kwdef mutable struct _PLEncodingMapBoxesJSON <: _PiJSON
    kind::String = "PLEncodingMapBoxes"
    n::Int = 0
    coords::Vector{Vector{Float64}} = Vector{Vector{Float64}}()
    sig_y::_MaskPackedWordsJSON = _MaskPackedWordsJSON()
    sig_z::_MaskPackedWordsJSON = _MaskPackedWordsJSON()
    reps::Vector{Vector{Float64}} = Vector{Vector{Float64}}()
    Ups::Vector{Vector{Float64}} = Vector{Vector{Float64}}()
    Downs::Vector{Vector{Float64}} = Vector{Vector{Float64}}()
    cell_shape::Vector{Int} = Int[]
    cell_strides::Vector{Int} = Int[]
    cell_to_region::Vector{Int} = Int[]
    coord_flags::Vector{Vector{UInt8}} = Vector{Vector{UInt8}}()
    axis_is_uniform::Vector{Bool} = Bool[]
    axis_step::Vector{Float64} = Float64[]
    axis_min::Vector{Float64} = Float64[]
end

Base.@kwdef mutable struct _FiniteEncodingFringeJSONV1
    kind::String = ""
    schema_version::Int = 0
    poset::_PosetJSON = _FinitePosetJSON()
    U::_MaskJSON = _MaskPackedWordsJSON()
    D::_MaskJSON = _MaskPackedWordsJSON()
    coeff_field::_CoeffFieldJSON = _QQFieldJSON()
    phi::_PhiJSON = _PhiQQChunksJSON()
    pi::Union{Nothing,_PiJSON} = nothing
end

JSON3.StructTypes.StructType(::Type{_CoeffFieldJSON}) = JSON3.StructTypes.AbstractType()
JSON3.StructTypes.subtypekey(::Type{_CoeffFieldJSON}) = :kind
JSON3.StructTypes.subtypes(::Type{_CoeffFieldJSON}) = (
    qq = _QQFieldJSON,
    real = _RealFieldJSON,
    fp = _FpFieldJSON,
)
JSON3.StructTypes.StructType(::Type{_QQFieldJSON}) = JSON3.StructTypes.Mutable()
JSON3.StructTypes.StructType(::Type{_RealFieldJSON}) = JSON3.StructTypes.Mutable()
JSON3.StructTypes.StructType(::Type{_FpFieldJSON}) = JSON3.StructTypes.Mutable()

JSON3.StructTypes.StructType(::Type{_PosetJSON}) = JSON3.StructTypes.AbstractType()
JSON3.StructTypes.subtypekey(::Type{_PosetJSON}) = :kind
JSON3.StructTypes.subtypes(::Type{_PosetJSON}) = (
    FinitePoset = _FinitePosetJSON,
    ProductOfChainsPoset = _ProductOfChainsPosetJSON,
    GridPoset = _GridPosetJSON,
    ProductPoset = _ProductPosetJSON,
    SignaturePoset = _SignaturePosetJSON,
)
JSON3.StructTypes.StructType(::Type{_FinitePosetJSON}) = JSON3.StructTypes.Mutable()
JSON3.StructTypes.StructType(::Type{_ProductOfChainsPosetJSON}) = JSON3.StructTypes.Mutable()
JSON3.StructTypes.StructType(::Type{_GridPosetJSON}) = JSON3.StructTypes.Mutable()
JSON3.StructTypes.StructType(::Type{_ProductPosetJSON}) = JSON3.StructTypes.Mutable()
JSON3.StructTypes.StructType(::Type{_SignaturePosetJSON}) = JSON3.StructTypes.Mutable()

JSON3.StructTypes.StructType(::Type{_MaskJSON}) = JSON3.StructTypes.AbstractType()
JSON3.StructTypes.subtypekey(::Type{_MaskJSON}) = :kind
JSON3.StructTypes.subtypes(::Type{_MaskJSON}) = (
    packed_words_v1 = _MaskPackedWordsJSON,
)
JSON3.StructTypes.StructType(::Type{_MaskPackedWordsJSON}) = JSON3.StructTypes.Mutable()

JSON3.StructTypes.StructType(::Type{_PhiJSON}) = JSON3.StructTypes.AbstractType()
JSON3.StructTypes.subtypekey(::Type{_PhiJSON}) = :kind
JSON3.StructTypes.subtypes(::Type{_PhiJSON}) = (
    qq_chunks_v1 = _PhiQQChunksJSON,
    fp_flat_v1 = _PhiFpFlatJSON,
    real_flat_v1 = _PhiRealFlatJSON,
)
JSON3.StructTypes.StructType(::Type{_PhiQQChunksJSON}) = JSON3.StructTypes.Mutable()
JSON3.StructTypes.StructType(::Type{_PhiFpFlatJSON}) = JSON3.StructTypes.Mutable()
JSON3.StructTypes.StructType(::Type{_PhiRealFlatJSON}) = JSON3.StructTypes.Mutable()

JSON3.StructTypes.StructType(::Type{_PiJSON}) = JSON3.StructTypes.AbstractType()
JSON3.StructTypes.subtypekey(::Type{_PiJSON}) = :kind
JSON3.StructTypes.subtypes(::Type{_PiJSON}) = (
    GridEncodingMap = _GridEncodingMapJSON,
    ZnEncodingMap = _ZnEncodingMapJSON,
    PLEncodingMapBoxes = _PLEncodingMapBoxesJSON,
)
JSON3.StructTypes.StructType(::Type{_FaceGeneratorJSON}) = JSON3.StructTypes.Mutable()
JSON3.StructTypes.StructType(::Type{_GridEncodingMapJSON}) = JSON3.StructTypes.Mutable()
JSON3.StructTypes.StructType(::Type{_ZnEncodingMapJSON}) = JSON3.StructTypes.Mutable()
JSON3.StructTypes.StructType(::Type{_PLEncodingMapBoxesJSON}) = JSON3.StructTypes.Mutable()
JSON3.StructTypes.StructType(::Type{_FiniteEncodingFringeJSONV1}) = JSON3.StructTypes.Mutable()

@inline _field_from_typed(obj::_QQFieldJSON) = QQField()
@inline function _field_from_typed(obj::_RealFieldJSON)
    Tname = obj.T
    T = Tname == "Float64" ? Float64 :
        Tname == "Float32" ? Float32 :
        error("Unsupported real field type in JSON: $(Tname)")
    rtol = obj.rtol === nothing ? sqrt(eps(T)) : T(obj.rtol)
    atol = obj.atol === nothing ? zero(T) : T(obj.atol)
    return RealField(T; rtol=rtol, atol=atol)
end
@inline _field_from_typed(obj::_FpFieldJSON) = PrimeField(obj.p)

@inline function _mask_lastword(ncols::Int)::UInt64
    rem = ncols & 63
    return rem == 0 ? typemax(UInt64) : (UInt64(1) << rem) - 1
end

@inline function _csv_escape(x)
    s = string(x)
    if occursin(',', s) || occursin('"', s)
        s = replace(s, '"' => "\"\"")
        return "\"" * s * "\""
    end
    return s
end

"""
    _write_feature_csv_wide(path, X, names, ids; ids_col=:id, include_ids=true)

Internal CSV fallback writer for wide feature tables.
Rows are samples and columns are features.
"""
function _write_feature_csv_wide(path::AbstractString,
                                 X::AbstractMatrix,
                                 names::AbstractVector,
                                 ids::AbstractVector{<:AbstractString};
                                 ids_col::Symbol=:id,
                                 include_ids::Bool=true)
    nfeat = size(X, 2)
    length(names) == nfeat || throw(ArgumentError("_write_feature_csv_wide: feature-name count mismatch"))
    length(ids) == size(X, 1) || throw(ArgumentError("_write_feature_csv_wide: id count mismatch"))
    open(path, "w") do io
        hdr = include_ids ? Any[ids_col; names] : Any[names...]
        println(io, join(_csv_escape.(hdr), ","))
        @inbounds for i in 1:size(X, 1)
            row = Vector{Any}(undef, nfeat + (include_ids ? 1 : 0))
            t = 1
            if include_ids
                row[t] = ids[i]
                t += 1
            end
            for j in 1:nfeat
                row[t] = X[i, j]
                t += 1
            end
            println(io, join(_csv_escape.(row), ","))
        end
    end
    return path
end

"""
    _write_feature_csv_long(path, X, names, ids; include_sample_index=true)

Internal CSV fallback writer for long feature tables.
"""
function _write_feature_csv_long(path::AbstractString,
                                 X::AbstractMatrix,
                                 names::AbstractVector,
                                 ids::AbstractVector{<:AbstractString};
                                 include_sample_index::Bool=true)
    nfeat = size(X, 2)
    length(names) == nfeat || throw(ArgumentError("_write_feature_csv_long: feature-name count mismatch"))
    length(ids) == size(X, 1) || throw(ArgumentError("_write_feature_csv_long: id count mismatch"))
    open(path, "w") do io
        if include_sample_index
            println(io, "id,feature,value,sample_index")
        else
            println(io, "id,feature,value")
        end
        @inbounds for i in 1:size(X, 1)
            idi = ids[i]
            for j in 1:nfeat
                if include_sample_index
                    vals = (idi, names[j], X[i, j], i)
                else
                    vals = (idi, names[j], X[i, j])
                end
                println(io, join(_csv_escape.(vals), ","))
            end
        end
    end
    return path
end

# =============================================================================
# A) Internal formats (owned/stable)
# =============================================================================

# -----------------------------------------------------------------------------
# A1) Flange (Z^n)  (FlangeZn.Flange)
# -----------------------------------------------------------------------------

"""
    save_flange_json(path, FG::FlangeZn.Flange)

Stable PosetModules-owned schema:

{
  "kind": "FlangeZn",
  "n": n,
  "flats":      [ {"b":[...], "tau":[i1,i2,...]}, ... ],
  "injectives": [ {"b":[...], "tau":[...]} , ... ],
  "coeff_field": { ... },
  "phi": [[ "num/den", ...], ...]   # rows = #injectives, cols = #flats
}

Notes
* `tau` is stored as a list of 1-based coordinate indices where the face is true.
* Scalars are encoded according to the coefficient field descriptor.
"""
function save_flange_json(path::AbstractString, FG::Flange; pretty::Bool=false)
    n = FG.n
    flats = [Dict("b" => collect(F.b), "tau" => findall(identity, F.tau.coords)) for F in FG.flats]
    injectives = [Dict("b" => collect(E.b), "tau" => findall(identity, E.tau.coords)) for E in FG.injectives]
    phi = [[_scalar_to_json(FG.field, FG.phi[i, j]) for j in 1:length(FG.flats)]
           for i in 1:length(FG.injectives)]
    obj = Dict("kind" => "FlangeZn",
               "n" => n,
               "flats" => flats,
               "injectives" => injectives,
               "coeff_field" => _field_to_obj(FG.field),
               "phi" => phi)
    return _json_write(path, obj; pretty=pretty)
end

# -----------------------------------------------------------------------------
# A0) Datasets + pipeline specs (Workflow)
# -----------------------------------------------------------------------------

function _obj_from_dataset(data)
    if data isa PointCloud
        pts = data.points
        npts = length(pts)
        d = npts == 0 ? 0 : length(pts[1])
        for i in 2:npts
            length(pts[i]) == d || error("PointCloud serialization expects uniform point dimension.")
        end
        T = npts == 0 ? Float64 : eltype(pts[1])
        flat = Vector{T}(undef, npts * d)
        t = 1
        @inbounds for i in 1:npts
            pi = pts[i]
            for j in 1:d
                flat[t] = pi[j]
                t += 1
            end
        end
        return Dict("kind" => "PointCloud",
                    "layout" => "columnar_v1",
                    "n" => npts,
                    "d" => d,
                    "points_flat" => flat)
    elseif data isa ImageNd
        return Dict("kind" => "ImageNd",
                    "size" => collect(size(data.data)),
                    "data" => collect(vec(data.data)))
    elseif data isa GraphData
        nedges = length(data.edges)
        edges_u = Vector{Int}(undef, nedges)
        edges_v = Vector{Int}(undef, nedges)
        @inbounds for eidx in 1:nedges
            u, v = data.edges[eidx]
            edges_u[eidx] = u
            edges_v[eidx] = v
        end
        coords_dim = nothing
        coords_flat = nothing
        if data.coords !== nothing
            coords = data.coords
            ncoords = length(coords)
            ncoords == data.n || error("GraphData coords length must equal n for columnar serialization.")
            d = ncoords == 0 ? 0 : length(coords[1])
            for i in 2:ncoords
                length(coords[i]) == d || error("GraphData coords must all have the same dimension.")
            end
            Tcoord = ncoords == 0 ? Float64 : eltype(coords[1])
            buf = Vector{Tcoord}(undef, ncoords * d)
            t = 1
            @inbounds for i in 1:ncoords
                ci = coords[i]
                for j in 1:d
                    buf[t] = ci[j]
                    t += 1
                end
            end
            coords_dim = d
            coords_flat = buf
        end
        return Dict("kind" => "GraphData",
                    "layout" => "columnar_v1",
                    "n" => data.n,
                    "edges_u" => edges_u,
                    "edges_v" => edges_v,
                    "coords_dim" => coords_dim,
                    "coords_flat" => coords_flat,
                    "weights" => data.weights === nothing ? nothing : collect(data.weights))
    elseif data isa EmbeddedPlanarGraph2D
        return Dict("kind" => "EmbeddedPlanarGraph2D",
                    "vertices" => [collect(v) for v in data.vertices],
                    "edges" => [collect(e) for e in data.edges],
                    "polylines" => data.polylines === nothing ? nothing : [[collect(p) for p in poly] for poly in data.polylines],
                    "bbox" => data.bbox === nothing ? nothing : collect(data.bbox))
    elseif data isa GradedComplex
        bnds = Any[]
        for B in data.boundaries
            Ii, Jj, Vv = findnz(B)
            push!(bnds, Dict(
                "m" => size(B, 1),
                "n" => size(B, 2),
                "I" => collect(Ii),
                "J" => collect(Jj),
                "V" => collect(Vv),
            ))
        end
        return Dict("kind" => "GradedComplex",
                    "cells_by_dim" => [collect(c) for c in data.cells_by_dim],
                    "boundaries" => bnds,
                    "grades" => [collect(g) for g in data.grades],
                    "cell_dims" => collect(data.cell_dims))
    elseif data isa MultiCriticalGradedComplex
        bnds = Any[]
        for B in data.boundaries
            Ii, Jj, Vv = findnz(B)
            push!(bnds, Dict(
                "m" => size(B, 1),
                "n" => size(B, 2),
                "I" => collect(Ii),
                "J" => collect(Jj),
                "V" => collect(Vv),
            ))
        end
        return Dict("kind" => "MultiCriticalGradedComplex",
                    "cells_by_dim" => [collect(c) for c in data.cells_by_dim],
                    "boundaries" => bnds,
                    "grades" => [[collect(g) for g in gs] for gs in data.grades],
                    "cell_dims" => collect(data.cell_dims))
    elseif data isa SimplexTreeMulti
        return Dict("kind" => "SimplexTreeMulti",
                    "simplex_offsets" => collect(data.simplex_offsets),
                    "simplex_vertices" => collect(data.simplex_vertices),
                    "simplex_dims" => collect(data.simplex_dims),
                    "dim_offsets" => collect(data.dim_offsets),
                    "grade_offsets" => collect(data.grade_offsets),
                    "grade_data" => [collect(g) for g in data.grade_data])
    else
        error("Unsupported dataset type for serialization.")
    end
end

function _dataset_from_obj(obj)
    kind = String(obj["kind"])
    if kind == "PointCloud"
        if haskey(obj, "points_flat")
            n = Int(obj["n"])
            d = Int(obj["d"])
            return _pointcloud_from_flat(n, d, Vector{Float64}(obj["points_flat"]))
        end
        # Legacy compatibility schema.
        pts = [Vector{Float64}(p) for p in obj["points"]]
        return PointCloud(pts)
    elseif kind == "ImageNd"
        sz = Vector{Int}(obj["size"])
        flat = Vector{Float64}(obj["data"])
        data = reshape(flat, Tuple(sz))
        return ImageNd(data)
    elseif kind == "GraphData"
        n = Int(obj["n"])
        weights = obj["weights"] === nothing ? nothing : Vector{Float64}(obj["weights"])
        if haskey(obj, "edges_u")
            coords_dim = haskey(obj, "coords_dim") && obj["coords_dim"] !== nothing ? Int(obj["coords_dim"]) : nothing
            coords_flat = haskey(obj, "coords_flat") && obj["coords_flat"] !== nothing ?
                Vector{Float64}(obj["coords_flat"]) : nothing
            return _graph_from_columns(n,
                                       Vector{Int}(obj["edges_u"]),
                                       Vector{Int}(obj["edges_v"]);
                                       coords_dim=coords_dim,
                                       coords_flat=coords_flat,
                                       weights=weights)
        end
        # Legacy compatibility schema.
        edges = [ (Int(e[1]), Int(e[2])) for e in obj["edges"] ]
        coords = obj["coords"] === nothing ? nothing : [Vector{Float64}(c) for c in obj["coords"]]
        return GraphData(n, edges; coords=coords, weights=weights, T=Float64)
    elseif kind == "EmbeddedPlanarGraph2D"
        verts = [Vector{Float64}(v) for v in obj["vertices"]]
        edges = [ (Int(e[1]), Int(e[2])) for e in obj["edges"] ]
        polylines = obj["polylines"] === nothing ? nothing :
            [[Vector{Float64}(p) for p in poly] for poly in obj["polylines"]]
        bbox = obj["bbox"] === nothing ? nothing : (Float64(obj["bbox"][1]),
                                                   Float64(obj["bbox"][2]),
                                                   Float64(obj["bbox"][3]),
                                                   Float64(obj["bbox"][4]))
        return EmbeddedPlanarGraph2D(verts, edges; polylines=polylines, bbox=bbox)
    elseif kind == "GradedComplex"
        cells = [Vector{Int}(c) for c in obj["cells_by_dim"]]
        boundaries = SparseMatrixCSC{Int,Int}[]
        for b in obj["boundaries"]
            m = Int(b["m"]); n = Int(b["n"])
            I = Vector{Int}(b["I"])
            J = Vector{Int}(b["J"])
            V = Vector{Int}(b["V"])
            push!(boundaries, sparse(I, J, V, m, n))
        end
        grades = [Vector{Float64}(g) for g in obj["grades"]]
        cell_dims = Vector{Int}(obj["cell_dims"])
        return GradedComplex(cells, boundaries, grades; cell_dims=cell_dims)
    elseif kind == "MultiCriticalGradedComplex"
        cells = [Vector{Int}(c) for c in obj["cells_by_dim"]]
        boundaries = SparseMatrixCSC{Int,Int}[]
        for b in obj["boundaries"]
            m = Int(b["m"]); n = Int(b["n"])
            I = Vector{Int}(b["I"])
            J = Vector{Int}(b["J"])
            V = Vector{Int}(b["V"])
            push!(boundaries, sparse(I, J, V, m, n))
        end
        grades = [[Vector{Float64}(g) for g in gs] for gs in obj["grades"]]
        cell_dims = Vector{Int}(obj["cell_dims"])
        return MultiCriticalGradedComplex(cells, boundaries, grades; cell_dims=cell_dims)
    elseif kind == "SimplexTreeMulti"
        simplex_offsets = Vector{Int}(obj["simplex_offsets"])
        simplex_vertices = Vector{Int}(obj["simplex_vertices"])
        simplex_dims = Vector{Int}(obj["simplex_dims"])
        dim_offsets = Vector{Int}(obj["dim_offsets"])
        grade_offsets = Vector{Int}(obj["grade_offsets"])
        raw_grades = obj["grade_data"]
        isempty(raw_grades) && error("SimplexTreeMulti JSON payload has empty grade_data.")
        N = length(raw_grades[1])
        grade_data = Vector{NTuple{N,Float64}}(undef, length(raw_grades))
        for i in eachindex(raw_grades)
            g = raw_grades[i]
            length(g) == N || error("SimplexTreeMulti JSON grade arity mismatch at index $i.")
            grade_data[i] = ntuple(k -> Float64(g[k]), N)
        end
        return SimplexTreeMulti(simplex_offsets, simplex_vertices, simplex_dims,
                                dim_offsets, grade_offsets, grade_data)
    else
        error("Unknown dataset kind: $kind")
    end
end

@inline function _construction_budget_obj(b::ConstructionBudget)
    return Dict(
        "max_simplices" => b.max_simplices,
        "max_edges" => b.max_edges,
        "memory_budget_bytes" => b.memory_budget_bytes,
    )
end

@inline function _construction_options_obj(c::ConstructionOptions)
    return Dict(
        "sparsify" => String(c.sparsify),
        "collapse" => String(c.collapse),
        "output_stage" => String(c.output_stage),
        "budget" => _construction_budget_obj(c.budget),
    )
end

function _spec_obj(spec::FiltrationSpec)
    params = Dict{String,Any}()
    for (k, v) in pairs(spec.params)
        if k == :construction
            if v isa ConstructionOptions
                params["construction"] = _construction_options_obj(v)
            elseif v isa ConstructionBudget
                params["construction"] = Dict("budget" => _construction_budget_obj(v))
            else
                params["construction"] = v
            end
        else
            params[String(k)] = v
        end
    end
    return Dict("kind" => String(spec.kind), "params" => params)
end

function _spec_from_obj(obj)
    kind = Symbol(String(obj["kind"]))
    params_obj = obj["params"]
    params = (; (Symbol(k) => params_obj[k] for k in keys(params_obj))...)
    return FiltrationSpec(; kind=kind, params...)
end

function _pipeline_options_from_spec(spec::FiltrationSpec)
    p = spec.params
    return PipelineOptions(;
        orientation = get(p, :orientation, nothing),
        axes_policy = Symbol(get(p, :axes_policy, :encoding)),
        axis_kind = get(p, :axis_kind, nothing),
        eps = get(p, :eps, nothing),
        poset_kind = Symbol(get(p, :poset_kind, :signature)),
        field = get(p, :field, nothing),
        max_axis_len = get(p, :max_axis_len, nothing),
    )
end

function _pipeline_options_from_any(spec::FiltrationSpec, x)
    if x === nothing
        return _pipeline_options_from_spec(spec)
    elseif x isa PipelineOptions
        return x
    elseif x isa NamedTuple
        return PipelineOptions(; x...)
    elseif x isa AbstractDict
        vals = (; (Symbol(k) => x[k] for k in keys(x))...)
        return PipelineOptions(; vals...)
    end
    throw(ArgumentError("pipeline_opts must be nothing, PipelineOptions, NamedTuple, or AbstractDict."))
end

function _pipeline_options_obj(opts::PipelineOptions)
    return Dict(
        "orientation" => opts.orientation,
        "axes_policy" => String(opts.axes_policy),
        "axis_kind" => opts.axis_kind,
        "eps" => opts.eps,
        "poset_kind" => String(opts.poset_kind),
        "field" => opts.field,
        "max_axis_len" => opts.max_axis_len,
    )
end

function _pipeline_options_from_obj(obj)::PipelineOptions
    orient_raw = get(obj, "orientation", nothing)
    orientation = if orient_raw isa AbstractVector
        ntuple(i -> Int(orient_raw[i]), length(orient_raw))
    else
        orient_raw
    end
    axis_kind_raw = get(obj, "axis_kind", nothing)
    axis_kind = axis_kind_raw isa AbstractString ? Symbol(axis_kind_raw) : axis_kind_raw
    field_raw = get(obj, "field", nothing)
    field = field_raw isa AbstractString ? Symbol(field_raw) : field_raw
    return PipelineOptions(;
        orientation = orientation,
        axes_policy = Symbol(get(obj, "axes_policy", "encoding")),
        axis_kind = axis_kind,
        eps = get(obj, "eps", nothing),
        poset_kind = Symbol(get(obj, "poset_kind", "signature")),
        field = field,
        max_axis_len = get(obj, "max_axis_len", nothing),
    )
end

"""
    save_dataset_json(path, data)

Serialize a dataset (PointCloud, ImageNd, GraphData, EmbeddedPlanarGraph2D, GradedComplex, MultiCriticalGradedComplex, SimplexTreeMulti).
"""
function save_dataset_json(path::AbstractString, data; pretty::Bool=false)
    return _json_write(path, _obj_from_dataset(data); pretty=pretty)
end

"""
    load_dataset_json(path)

Load a dataset serialized by `save_dataset_json`.
"""
function load_dataset_json(path::AbstractString)
    raw = read(path, String)
    kind_hdr = JSON3.read(raw, NamedTuple{(:kind,),Tuple{String}})
    kind = kind_hdr.kind
    if kind == "PointCloud" && occursin("\"points_flat\"", raw)
        obj = JSON3.read(raw, _PointCloudColumnarJSON)
        return _pointcloud_from_flat(obj.n, obj.d, obj.points_flat)
    elseif kind == "GraphData" && occursin("\"edges_u\"", raw) && occursin("\"edges_v\"", raw)
        obj = JSON3.read(raw, _GraphDataColumnarJSON)
        return _graph_from_columns(obj.n, obj.edges_u, obj.edges_v;
                                   coords_dim=obj.coords_dim,
                                   coords_flat=obj.coords_flat,
                                   weights=obj.weights)
    end
    return _dataset_from_obj(JSON3.read(raw))
end

"""
    save_pipeline_json(path, data, spec; degree=nothing, pipeline_opts=nothing)

Serialize a dataset + filtration spec and structured `PipelineOptions` in one JSON.
"""
function save_pipeline_json(path::AbstractString, data, spec::FiltrationSpec;
                            degree=nothing, pipeline_opts=nothing,
                            pretty::Bool=false)
    popts = _pipeline_options_from_any(spec, pipeline_opts)
    obj = Dict(
        "schema_version" => PIPELINE_SCHEMA_VERSION,
        "dataset" => _obj_from_dataset(data),
        "spec" => _spec_obj(spec),
        "degree" => degree,
        "pipeline_options" => _pipeline_options_obj(popts),
    )
    return _json_write(path, obj; pretty=pretty)
end

"""
    load_pipeline_json(path) -> (data, spec, degree, pipeline_opts)

Inverse of `save_pipeline_json`.
"""
function load_pipeline_json(path::AbstractString)
    obj = _json_read(path)
    version = haskey(obj, "schema_version") ? Int(obj["schema_version"]) : 0
    version == PIPELINE_SCHEMA_VERSION || error("Unsupported pipeline JSON schema_version: $(version). Expected $(PIPELINE_SCHEMA_VERSION).")
    data = _dataset_from_obj(obj["dataset"])
    spec = _spec_from_obj(obj["spec"])
    degree = haskey(obj, "degree") ? obj["degree"] : nothing
    haskey(obj, "pipeline_options") || error("pipeline_options field is required in pipeline JSON.")
    pipeline_opts = _pipeline_options_from_obj(obj["pipeline_options"])
    return data, spec, degree, pipeline_opts
end

# =============================================================================
# B0) Interop adapters: GUDHI / Ripserer / Eirene (JSON)
# =============================================================================

function _simplicial_boundary_from_lists(simplices::Vector{Vector{Int}},
                                         faces::Vector{Vector{Int}})
    face_index = Dict{Tuple{Vararg{Int}},Int}()
    for (i, f) in enumerate(faces)
        face_index[Tuple(f)] = i
    end
    I = Int[]
    J = Int[]
    V = Int[]
    for (j, s) in enumerate(simplices)
        k = length(s)
        for i in 1:k
            f = [s[t] for t in 1:k if t != i]
            row = face_index[Tuple(f)]
            push!(I, row)
            push!(J, j)
            push!(V, isodd(i) ? 1 : -1)
        end
    end
    return sparse(I, J, V, length(faces), length(simplices))
end

function _graded_complex_from_simplex_list(simplices::Vector{Vector{Int}}, grades_any::AbstractVector)
    length(simplices) == length(grades_any) ||
        error("simplices and grades length mismatch.")
    max_dim = maximum(length.(simplices)) - 1
    by_dim = [Vector{Vector{Int}}() for _ in 0:max_dim]
    grades = Vector{Vector{Float64}}()
    for (s, g) in zip(simplices, grades_any)
        d = length(s) - 1
        push!(by_dim[d+1], s)
        if g isa AbstractVector
            push!(grades, Vector{Float64}(g))
        else
            push!(grades, [Float64(g)])
        end
    end
    boundaries = SparseMatrixCSC{Int,Int}[]
    for d in 2:length(by_dim)
        push!(boundaries, _simplicial_boundary_from_lists(by_dim[d], by_dim[d-1]))
    end
    cells = [collect(1:length(by_dim[d])) for d in 1:length(by_dim)]
    return GradedComplex(cells, boundaries, grades)
end

"""
    load_gudhi_json(path) -> GradedComplex

Expected schema:
{
  "simplices": [[0],[1],[0,1],...],
  "filtration": [0.0, 0.0, 1.0, ...]   # or list-of-lists for multiparameter
}
"""
function load_gudhi_json(path::AbstractString)
    obj = _json_read(path)
    simplices = [Vector{Int}(s) for s in obj["simplices"]]
    grades = obj["filtration"]
    return _graded_complex_from_simplex_list(simplices, grades)
end

"""
    load_ripserer_json(path) -> GradedComplex

Expected schema:
{
  "simplices": [[0],[1],[0,1],...],
  "filtration": [0.0, 0.0, 1.0, ...]
}
"""
function load_ripserer_json(path::AbstractString)
    obj = _json_read(path)
    simplices = [Vector{Int}(s) for s in obj["simplices"]]
    grades = obj["filtration"]
    return _graded_complex_from_simplex_list(simplices, grades)
end

"""
    load_eirene_json(path) -> GradedComplex

Expected schema:
{
  "simplices": [[0],[1],[0,1],...],
  "filtration": [0.0, 0.0, 1.0, ...]
}
"""
function load_eirene_json(path::AbstractString)
    obj = _json_read(path)
    simplices = [Vector{Int}(s) for s in obj["simplices"]]
    grades = obj["filtration"]
    return _graded_complex_from_simplex_list(simplices, grades)
end

function _read_structured_lines(path::AbstractString)
    out = String[]
    open(path, "r") do io
        for raw in eachline(io)
            line = strip(raw)
            isempty(line) && continue
            startswith(line, "#") && continue
            startswith(line, "//") && continue
            push!(out, line)
        end
    end
    return out
end

function _take_rivet_flags(lines::Vector{String})
    flags = Dict{String,String}()
    i = 1
    while i <= length(lines) && startswith(lines[i], "--")
        parts = split(lines[i], r"\\s+"; limit=2)
        key = lowercase(replace(parts[1], "--" => ""))
        val = length(parts) == 2 ? strip(parts[2]) : "true"
        flags[key] = val
        i += 1
    end
    return flags, lines[i:end]
end

function _minimal_bigrades(grades::Vector{NTuple{2,Float64}})
    u = unique(grades)
    keep = trues(length(u))
    for i in eachindex(u)
        ai = u[i]
        for j in eachindex(u)
            i == j && continue
            aj = u[j]
            if aj[1] <= ai[1] && aj[2] <= ai[2] && (aj != ai)
                keep[i] = false
                break
            end
        end
    end
    out = u[keep]
    sort!(out)
    return out
end

function _parse_rivet_simplex_grade_line(line::AbstractString)
    parts = split(line, ';'; limit=2)
    length(parts) == 2 || error("RIVET bifiltration line must contain ';': $(line)")
    simplex = Int[parse(Int, t) for t in split(strip(parts[1]))]
    isempty(simplex) && error("RIVET bifiltration simplex cannot be empty.")
    sort!(unique!(simplex))

    gtoks = split(replace(strip(parts[2]), "," => " "))
    isempty(gtoks) && error("RIVET bifiltration line has no grades: $(line)")
    iseven(length(gtoks)) || error("RIVET bifiltration grade list must have even length: $(line)")
    grades = NTuple{2,Float64}[]
    for i in 1:2:length(gtoks)
        push!(grades, (parse(Float64, gtoks[i]), parse(Float64, gtoks[i+1])))
    end
    return simplex, _minimal_bigrades(grades)
end

function _normalize_simplex_indices!(simplices::Vector{Vector{Int}})
    minv = minimum(minimum(s) for s in simplices)
    if minv == 0
        for s in simplices
            for i in eachindex(s)
                s[i] += 1
            end
        end
    elseif minv < 1
        error("RIVET simplices must be 0-based or 1-based integer indices.")
    end
    return simplices
end

function _graded_complex_from_simplex_list_multicritical(simplices::Vector{Vector{Int}},
                                                         gradesets::Vector{Vector{NTuple{2,Float64}}})
    length(simplices) == length(gradesets) || error("simplices and grade sets length mismatch.")
    max_dim = maximum(length.(simplices)) - 1
    by_dim = [Vector{Vector{Int}}() for _ in 0:max_dim]
    g_by_dim = [Vector{Vector{NTuple{2,Float64}}}() for _ in 0:max_dim]
    for (s, gs) in zip(simplices, gradesets)
        d = length(s) - 1
        push!(by_dim[d+1], s)
        push!(g_by_dim[d+1], gs)
    end
    boundaries = SparseMatrixCSC{Int,Int}[]
    for d in 2:length(by_dim)
        push!(boundaries, _simplicial_boundary_from_lists(by_dim[d], by_dim[d-1]))
    end
    cells = [collect(1:length(by_dim[d])) for d in 1:length(by_dim)]
    flat_multi = Vector{Vector{NTuple{2,Float64}}}()
    for d in 1:length(g_by_dim)
        append!(flat_multi, g_by_dim[d])
    end
    if all(length(gs) == 1 for gs in flat_multi)
        flat = [gs[1] for gs in flat_multi]
        return GradedComplex(cells, boundaries, flat)
    end
    return MultiCriticalGradedComplex(cells, boundaries, flat_multi)
end

"""
    load_rivet_bifiltration(path) -> Union{GradedComplex, MultiCriticalGradedComplex}

Parse a RIVET bifiltration text file. Supports:
- modern `--datatype bifiltration` files with lines `simplex ; x y [x y ...]`
- legacy header-based files beginning with `bifiltration`.
"""
function load_rivet_bifiltration(path::AbstractString)
    raw = _read_structured_lines(path)
    isempty(raw) && error("RIVET bifiltration: empty file.")
    flags, lines = _take_rivet_flags(raw)
    if haskey(flags, "datatype")
        lowercase(flags["datatype"]) == "bifiltration" ||
            error("RIVET loader expected --datatype bifiltration, got $(flags["datatype"]).")
    end
    isempty(lines) && error("RIVET bifiltration: no payload lines.")

    payload = lines
    if lowercase(lines[1]) == "bifiltration"
        length(lines) >= 5 || error("RIVET legacy bifiltration header is incomplete.")
        lowercase(lines[2]) == "s" || error("RIVET legacy bifiltration: only simplicial ('s') format is supported.")
        ns = parse(Int, strip(lines[4]))
        payload = lines[5:end]
        length(payload) == ns || error("RIVET legacy bifiltration: expected $(ns) simplex lines, found $(length(payload)).")
    end

    simplices = Vector{Vector{Int}}(undef, length(payload))
    gradesets = Vector{Vector{NTuple{2,Float64}}}(undef, length(payload))
    for i in eachindex(payload)
        s, gs = _parse_rivet_simplex_grade_line(payload[i])
        simplices[i] = s
        gradesets[i] = gs
    end
    _normalize_simplex_indices!(simplices)
    return _graded_complex_from_simplex_list_multicritical(simplices, gradesets)
end

function _parse_rivet_firep_column(line::AbstractString)
    parts = split(line, ';'; limit=2)
    length(parts) == 2 || error("RIVET FIRep column line must contain ';': $(line)")
    gtok = split(strip(parts[1]))
    length(gtok) == 2 || error("RIVET FIRep column grade must have exactly two coordinates: $(line)")
    grade = (parse(Float64, gtok[1]), parse(Float64, gtok[2]))
    rhs = strip(parts[2])
    idxs = isempty(rhs) ? Int[] : Int[parse(Int, t) for t in split(rhs)]
    return grade, idxs
end

"""
    load_rivet_firep(path) -> GradedComplex

Parse a RIVET FIRep text file (`--datatype firep` or raw FIRep payload).
Builds a graded complex with dimensions 0,1,2 from the FIRep matrices.
"""
function load_rivet_firep(path::AbstractString)
    raw = _read_structured_lines(path)
    isempty(raw) && error("RIVET FIRep: empty file.")
    flags, lines = _take_rivet_flags(raw)
    if haskey(flags, "datatype")
        lowercase(flags["datatype"]) == "firep" ||
            error("RIVET loader expected --datatype firep, got $(flags["datatype"]).")
    end
    isempty(lines) && error("RIVET FIRep: missing payload.")

    hdr = split(lines[1])
    length(hdr) == 3 || error("RIVET FIRep header must be: t s r")
    t = parse(Int, hdr[1])  # C2 generators
    s = parse(Int, hdr[2])  # C1 generators
    r = parse(Int, hdr[3])  # C0 generators
    t >= 0 && s >= 0 && r >= 0 || error("RIVET FIRep counts must be nonnegative.")
    length(lines) == 1 + t + s || error("RIVET FIRep: expected $(1+t+s) payload lines, found $(length(lines)).")

    c2_grades = Vector{NTuple{2,Float64}}(undef, t)
    I2 = Int[]; J2 = Int[]
    for j in 1:t
        g, rows = _parse_rivet_firep_column(lines[1 + j])
        c2_grades[j] = g
        for i in rows
            push!(I2, i)
            push!(J2, j)
        end
    end
    if !isempty(I2)
        minimum(I2) == 0 && (I2 .= I2 .+ 1)
        minimum(I2) >= 1 || error("RIVET FIRep: invalid C2->C1 row index.")
        maximum(I2) <= s || error("RIVET FIRep: C2->C1 row index out of range.")
    end

    c1_grades = Vector{NTuple{2,Float64}}(undef, s)
    I1 = Int[]; J1 = Int[]
    for j in 1:s
        g, rows = _parse_rivet_firep_column(lines[1 + t + j])
        c1_grades[j] = g
        for i in rows
            push!(I1, i)
            push!(J1, j)
        end
    end
    if !isempty(I1)
        minimum(I1) == 0 && (I1 .= I1 .+ 1)
        minimum(I1) >= 1 || error("RIVET FIRep: invalid C1->C0 row index.")
        maximum(I1) <= r || error("RIVET FIRep: C1->C0 row index out of range.")
    end

    B1 = sparse(I1, J1, ones(Int, length(I1)), r, s)  # C1 -> C0
    B2 = sparse(I2, J2, ones(Int, length(I2)), s, t)  # C2 -> C1

    c0_grades = Vector{NTuple{2,Float64}}(undef, r)
    incident = [Int[] for _ in 1:r]
    Ii, Jj, _ = findnz(B1)
    @inbounds for k in eachindex(Ii)
        push!(incident[Ii[k]], Jj[k])
    end
    for i in 1:r
        if isempty(incident[i])
            c0_grades[i] = (0.0, 0.0)
        else
            xs = Float64[c1_grades[j][1] for j in incident[i]]
            ys = Float64[c1_grades[j][2] for j in incident[i]]
            c0_grades[i] = (minimum(xs), minimum(ys))
        end
    end

    cells = [collect(1:r), collect(1:s), collect(1:t)]
    grades = vcat(c0_grades, c1_grades, c2_grades)
    return GradedComplex(cells, [B1, B2], grades)
end

"""
    load_gudhi_txt(path) -> GradedComplex
    load_ripserer_txt(path) -> GradedComplex
    load_eirene_txt(path) -> GradedComplex

Parse a simplex filtration from a text file with one simplex per line.

Supported line formats (whitespace-separated):
1) "dim v1 v2 ... vk filtration"
2) "v1 v2 ... vk filtration"   (dimension inferred from count)

Blank lines and lines starting with '#' are ignored.
"""
function _load_simplex_filtration_txt(path::AbstractString)
    lines = String[]
    open(path, "r") do io
        for raw in eachline(io)
            line = strip(raw)
            isempty(line) && continue
            startswith(line, "#") && continue
            push!(lines, line)
        end
    end

    # Heuristic: if any line has exactly two tokens, treat the file as
    # "vertices + filtration" (no leading dimension token). Otherwise, use
    # the "dim v1 v2 ... filtration" format.
    has_dim_prefix = true
    for line in lines
        parts = split(line)
        length(parts) >= 2 || error("Invalid simplex line: '$line'")
        if length(parts) == 2
            has_dim_prefix = false
            break
        end
    end

    simplices = Vector{Vector{Int}}()
    grades = Vector{Float64}()
    for line in lines
        parts = split(line)
        if has_dim_prefix
            dim = parse(Int, parts[1])
            verts = [parse(Int, parts[i]) for i in 2:(dim+2)]
            filt = parse(Float64, parts[dim+3])
            push!(simplices, verts)
            push!(grades, filt)
        else
            verts = [parse(Int, parts[i]) for i in 1:(length(parts)-1)]
            filt = parse(Float64, parts[end])
            push!(simplices, verts)
            push!(grades, filt)
        end
    end

    return _graded_complex_from_simplex_list(simplices, grades)
end

load_gudhi_txt(path::AbstractString) = _load_simplex_filtration_txt(path)
load_ripserer_txt(path::AbstractString) = _load_simplex_filtration_txt(path)
load_eirene_txt(path::AbstractString) = _load_simplex_filtration_txt(path)

# -----------------------------------------------------------------------------
# B2) Interop adapters: boundary/reduced complexes and direct PModules
# -----------------------------------------------------------------------------

@inline _packed_signature_rows_type(nw::Int) = Core.apply_type(PackedSignatureRows, nw)

@inline function _pack_words_obj_from_matrix(words::Matrix{UInt64}, bitlen::Int)
    nw, nrows = size(words)
    ncols = bitlen
    flat = Vector{UInt64}(undef, nrows * nw)
    @inbounds for i in 1:nrows
        base = (i - 1) * nw
        for w in 1:nw
            flat[base + w] = words[w, i]
        end
        if ncols > 0
            flat[base + nw] &= _mask_lastword(ncols)
        end
    end
    return _MaskPackedWordsJSON(kind="packed_words_v1",
                                nrows=nrows,
                                ncols=ncols,
                                words_per_row=nw,
                                words=flat)
end

@inline function _pack_signature_rows_obj(rows::PackedSignatureRows)
    return _pack_words_obj_from_matrix(rows.words, rows.bitlen)
end

@inline function _pack_bitmatrix_obj(L::BitMatrix)
    return _pack_words_obj_from_matrix(_pack_bitmatrix_words(L), size(L, 2))
end

function _pack_bitmatrix_words(L::BitMatrix)
    nrows, ncols = size(L)
    nw = max(1, cld(max(ncols, 1), 64))
    words = zeros(UInt64, nw, nrows)
    @inbounds for i in 1:nrows
        for j in 1:ncols
            if L[i, j]
                w = ((j - 1) >>> 6) + 1
                words[w, i] |= (UInt64(1) << ((j - 1) & 63))
            end
        end
        if ncols > 0
            words[nw, i] &= _mask_lastword(ncols)
        end
    end
    return words
end

function _unpack_words_matrix(mask_obj::_MaskPackedWordsJSON, name::String;
                              nrows_expected::Union{Nothing,Int}=nothing,
                              ncols_expected::Union{Nothing,Int}=nothing)
    nrows = mask_obj.nrows
    ncols = mask_obj.ncols
    nw = mask_obj.words_per_row
    nrows_expected === nothing || (nrows == nrows_expected || error("$(name).nrows must equal $(nrows_expected)."))
    ncols_expected === nothing || (ncols == ncols_expected || error("$(name).ncols must equal $(ncols_expected)."))
    nw == max(1, cld(max(ncols, 1), 64)) || error("$(name).words_per_row is inconsistent with ncols.")
    flat = mask_obj.words
    length(flat) == nrows * nw || error("$(name).words length mismatch.")
    words = Matrix{UInt64}(undef, nw, nrows)
    @inbounds for i in 1:nrows
        base = (i - 1) * nw
        for w in 1:nw
            words[w, i] = flat[base + w]
        end
        if ncols > 0
            words[nw, i] &= _mask_lastword(ncols)
        end
    end
    return words, nrows, ncols
end

function _parse_signature_rows_packed(mask_obj::_MaskPackedWordsJSON, row_name::String)
    words, _, bitlen = _unpack_words_matrix(mask_obj, row_name)
    return _packed_signature_rows_type(size(words, 1))(words, bitlen)
end

function _parse_signature_rows_packed(rows_any, row_name::String)
    rows_any isa AbstractVector || error("$(row_name) must be a list-of-lists.")
    nrows = length(rows_any)
    bitlen = if nrows == 0
        0
    else
        first_row = rows_any[1]
        first_row isa AbstractVector || error("$(row_name) row 1 must be a list.")
        length(first_row)
    end
    nw = max(1, cld(max(bitlen, 1), 64))
    words = zeros(UInt64, nw, nrows)
    @inbounds for i in 1:nrows
        row = rows_any[i]
        row isa AbstractVector || error("$(row_name) row $(i) must be a list.")
        length(row) == bitlen || error("$(row_name) row $(i) length mismatch; expected $(bitlen).")
        for j in 1:bitlen
            x = row[j]
            x isa Bool || error("$(row_name) entries must be Bool (row $(i), col $(j)).")
            if x
                w = ((j - 1) >>> 6) + 1
                words[w, i] |= (UInt64(1) << ((j - 1) & 63))
            end
        end
    end
    return _packed_signature_rows_type(nw)(words, bitlen)
end

function _decode_bitmatrix(mask_obj::_MaskPackedWordsJSON, name::String,
                           nrows_expected::Int, ncols_expected::Int)
    words, nrows, ncols = _unpack_words_matrix(mask_obj, name;
                                               nrows_expected=nrows_expected,
                                               ncols_expected=ncols_expected)
    L = falses(nrows, ncols)
    @inbounds for i in 1:nrows
        for j in 1:ncols
            w = ((j - 1) >>> 6) + 1
            bit = UInt64(1) << ((j - 1) & 63)
            L[i, j] = (words[w, i] & bit) != 0
        end
    end
    return L
end

@inline function _is_packed_words_obj(obj)::Bool
    return !(obj isa AbstractVector) && haskey(obj, "kind") && String(obj["kind"]) == "packed_words_v1"
end

function _parse_poset_from_typed(poset_obj::_FinitePosetJSON)
    n = poset_obj.n
    leq = _decode_bitmatrix(poset_obj.leq, "FinitePoset.leq", n, n)
    P = FinitePoset(leq)
    _clear_cover_cache!(P)
    return P
end

function _parse_poset_from_typed(poset_obj::_ProductOfChainsPosetJSON)
    P = ProductOfChainsPoset(poset_obj.sizes)
    poset_obj.n == nvertices(P) || error("ProductOfChainsPoset.n mismatch.")
    _clear_cover_cache!(P)
    return P
end

function _parse_poset_from_typed(poset_obj::_GridPosetJSON)
    coords_any = poset_obj.coords
    coords = ntuple(i -> Vector{Float64}(coords_any[i]), length(coords_any))
    P = GridPoset(coords)
    poset_obj.n == nvertices(P) || error("GridPoset.n mismatch.")
    _clear_cover_cache!(P)
    return P
end

function _parse_poset_from_typed(poset_obj::_ProductPosetJSON)
    poset_obj.left === nothing && error("ProductPoset missing required key 'left'.")
    poset_obj.right === nothing && error("ProductPoset missing required key 'right'.")
    P1 = _parse_poset_from_typed(poset_obj.left)
    P2 = _parse_poset_from_typed(poset_obj.right)
    P = ProductPoset(P1, P2)
    poset_obj.n == nvertices(P) || error("ProductPoset.n mismatch.")
    _clear_cover_cache!(P)
    return P
end

function _parse_poset_from_typed(poset_obj::_SignaturePosetJSON)
    sig_y = _parse_signature_rows_packed(poset_obj.sig_y, "SignaturePoset.sig_y")
    sig_z = _parse_signature_rows_packed(poset_obj.sig_z, "SignaturePoset.sig_z")
    P = SignaturePoset(sig_y, sig_z)
    poset_obj.n == nvertices(P) || error("SignaturePoset.n mismatch.")
    _clear_cover_cache!(P)
    return P
end

function _parse_poset_from_typed(poset_obj::_PosetJSON)
    error("Unsupported typed poset payload: $(typeof(poset_obj))")
end

@inline function _pack_masks_obj(masks::AbstractVector{<:BitVector})
    nrows = length(masks)
    ncols = nrows == 0 ? 0 : length(masks[1])
    nw = max(1, cld(max(ncols, 1), 64))
    words = zeros(UInt64, nrows * nw)
    @inbounds for i in 1:nrows
        mask = masks[i]
        length(mask) == ncols || error("mask row length mismatch at row $(i).")
        base = (i - 1) * nw
        chunks = mask.chunks
        nchunks = length(chunks)
        for w in 1:min(nchunks, nw)
            words[base + w] = chunks[w]
        end
        if ncols > 0
            words[base + nw] &= _mask_lastword(ncols)
        end
    end
    return _MaskPackedWordsJSON(kind="packed_words_v1",
                                nrows=nrows,
                                ncols=ncols,
                                words_per_row=nw,
                                words=words)
end

function _decode_masks(mask_obj::_MaskPackedWordsJSON, name::String, ncols_expected::Int)
    words, nrows, ncols = _unpack_words_matrix(mask_obj, name; ncols_expected=ncols_expected)
    nw = size(words, 1)
    masks = Vector{BitVector}(undef, nrows)
    if ncols == 0
        @inbounds for i in 1:nrows
            masks[i] = BitVector(undef, 0)
        end
        return masks
    end
    lastmask = _mask_lastword(ncols)
    nchunks = cld(ncols, 64)
    @inbounds for i in 1:nrows
        mask = falses(ncols)
        for w in 1:nchunks
            mask.chunks[w] = words[w, i]
        end
        mask.chunks[nchunks] &= lastmask
        masks[i] = mask
    end
    return masks
end

@inline function _build_upsets(P::AbstractPoset,
                               masks::Vector{BitVector},
                               validate_masks::Bool)
    U = Vector{FiniteFringe.Upset}(undef, length(masks))
    @inbounds for t in eachindex(masks)
        mask = masks[t]
        if validate_masks
            Uc = FiniteFringe.upset_closure(P, mask)
            Uc.mask == mask || error("U[$(t)] is not an upset under strict validation. If this file was produced by PosetModules and you trust it, load with validation=:trusted.")
            U[t] = Uc
        else
            U[t] = FiniteFringe.Upset(P, mask)
        end
    end
    return U
end

@inline function _build_downsets(P::AbstractPoset,
                                 masks::Vector{BitVector},
                                 validate_masks::Bool)
    D = Vector{FiniteFringe.Downset}(undef, length(masks))
    @inbounds for t in eachindex(masks)
        mask = masks[t]
        if validate_masks
            Dc = FiniteFringe.downset_closure(P, mask)
            Dc.mask == mask || error("D[$(t)] is not a downset under strict validation. If this file was produced by PosetModules and you trust it, load with validation=:trusted.")
            D[t] = Dc
        else
            D[t] = FiniteFringe.Downset(P, mask)
        end
    end
    return D
end

@inline function _phi_dims(phi::_PhiJSON)
    return phi.m, phi.k
end

const _QQ_CHUNK_BASE = 1_000_000_000
const _QQ_CHUNK_BASE_BIG = BigInt(_QQ_CHUNK_BASE)

@inline function _bigint_to_chunks!(chunks::Vector{UInt32}, ptr::Vector{Int}, x::BigInt)
    y = abs(x)
    if y == 0
        push!(chunks, UInt32(0))
        push!(ptr, length(chunks) + 1)
        return
    end
    while y != 0
        y, r = divrem(y, _QQ_CHUNK_BASE)
        push!(chunks, UInt32(r))
    end
    push!(ptr, length(chunks) + 1)
end

@inline function _chunks_to_bigint(chunks::Vector{UInt32}, ptr::Vector{Int}, idx::Int, name::String)
    a = ptr[idx]
    b = ptr[idx + 1] - 1
    (a >= 1 && b >= a && b <= length(chunks)) || error("$(name) chunk pointer out of range at index $(idx).")
    x = BigInt(0)
    @inbounds for t in b:-1:a
        x *= _QQ_CHUNK_BASE_BIG
        x += Int(chunks[t])
    end
    return x
end

function _phi_obj(H::FringeModule)
    m, k = size(H.phi)
    if H.field isa QQField
        len = m * k
        num_sign = Vector{Int8}(undef, len)
        num_ptr = Int[1]
        den_ptr = Int[1]
        num_chunks = UInt32[]
        den_chunks = UInt32[]
        sizehint!(num_ptr, len + 1)
        sizehint!(den_ptr, len + 1)
        @inbounds for idx in 1:len
            q = QQ(H.phi[idx])
            num = numerator(q)
            den = denominator(q)
            num_sign[idx] = num > 0 ? Int8(1) : (num < 0 ? Int8(-1) : Int8(0))
            _bigint_to_chunks!(num_chunks, num_ptr, num)
            _bigint_to_chunks!(den_chunks, den_ptr, den)
        end
        return _PhiQQChunksJSON(kind="qq_chunks_v1",
                                m=m,
                                k=k,
                                base=_QQ_CHUNK_BASE,
                                num_sign=num_sign,
                                num_ptr=num_ptr,
                                num_chunks=num_chunks,
                                den_ptr=den_ptr,
                                den_chunks=den_chunks)
    elseif H.field isa PrimeField
        data = Vector{Int}(undef, m * k)
        @inbounds for idx in 1:(m * k)
            data[idx] = Int(coerce(H.field, H.phi[idx]).val)
        end
        return _PhiFpFlatJSON(kind="fp_flat_v1", m=m, k=k, data=data)
    elseif H.field isa RealField
        data = Vector{Float64}(undef, m * k)
        @inbounds for idx in 1:(m * k)
            data[idx] = Float64(H.phi[idx])
        end
        return _PhiRealFlatJSON(kind="real_flat_v1", m=m, k=k, data=data)
    end
    error("Unsupported coefficient field for phi serialization: $(typeof(H.field))")
end

function _decode_phi(phi_obj::_PhiQQChunksJSON,
                     saved_field::QQField,
                     target_field::QQField,
                     m_expected::Int,
                     k_expected::Int)
    m, k = _phi_dims(phi_obj)
    (m == m_expected && k == k_expected) || error("phi dimensions mismatch (expected $(m_expected)x$(k_expected), got $(m)x$(k)).")
    phi_obj.base == _QQ_CHUNK_BASE || error("qq_chunks_v1.base must be $(_QQ_CHUNK_BASE).")
    len = m * k
    length(phi_obj.num_sign) == len || error("qq_chunks_v1.num_sign length mismatch.")
    length(phi_obj.num_ptr) == len + 1 || error("qq_chunks_v1.num_ptr length mismatch.")
    length(phi_obj.den_ptr) == len + 1 || error("qq_chunks_v1.den_ptr length mismatch.")
    phi_obj.num_ptr[1] == 1 || error("qq_chunks_v1.num_ptr must start at 1.")
    phi_obj.den_ptr[1] == 1 || error("qq_chunks_v1.den_ptr must start at 1.")
    phi_obj.num_ptr[end] == length(phi_obj.num_chunks) + 1 || error("qq_chunks_v1.num_ptr terminator mismatch.")
    phi_obj.den_ptr[end] == length(phi_obj.den_chunks) + 1 || error("qq_chunks_v1.den_ptr terminator mismatch.")
    K = coeff_type(target_field)
    Phi = Matrix{K}(undef, m, k)
    @inbounds for idx in 1:len
        s = phi_obj.num_sign[idx]
        (s == -1 || s == 0 || s == 1) || error("qq_chunks_v1.num_sign must be in {-1,0,1}.")
        num = _chunks_to_bigint(phi_obj.num_chunks, phi_obj.num_ptr, idx, "qq_chunks_v1.num")
        den = _chunks_to_bigint(phi_obj.den_chunks, phi_obj.den_ptr, idx, "qq_chunks_v1.den")
        den == 0 && error("qq_chunks_v1.den must be nonzero.")
        if s < 0
            num = -num
        elseif s == 0
            num = BigInt(0)
        end
        Phi[idx] = QQ(num // den)
    end
    return Phi
end

function _decode_phi(phi_obj::_PhiQQChunksJSON,
                     saved_field::QQField,
                     target_field::AbstractCoeffField,
                     m_expected::Int,
                     k_expected::Int)
    m, k = _phi_dims(phi_obj)
    (m == m_expected && k == k_expected) || error("phi dimensions mismatch (expected $(m_expected)x$(k_expected), got $(m)x$(k)).")
    phi_obj.base == _QQ_CHUNK_BASE || error("qq_chunks_v1.base must be $(_QQ_CHUNK_BASE).")
    len = m * k
    length(phi_obj.num_sign) == len || error("qq_chunks_v1.num_sign length mismatch.")
    length(phi_obj.num_ptr) == len + 1 || error("qq_chunks_v1.num_ptr length mismatch.")
    length(phi_obj.den_ptr) == len + 1 || error("qq_chunks_v1.den_ptr length mismatch.")
    phi_obj.num_ptr[1] == 1 || error("qq_chunks_v1.num_ptr must start at 1.")
    phi_obj.den_ptr[1] == 1 || error("qq_chunks_v1.den_ptr must start at 1.")
    phi_obj.num_ptr[end] == length(phi_obj.num_chunks) + 1 || error("qq_chunks_v1.num_ptr terminator mismatch.")
    phi_obj.den_ptr[end] == length(phi_obj.den_chunks) + 1 || error("qq_chunks_v1.den_ptr terminator mismatch.")
    K = coeff_type(target_field)
    Phi = Matrix{K}(undef, m, k)
    @inbounds for idx in 1:len
        s = phi_obj.num_sign[idx]
        (s == -1 || s == 0 || s == 1) || error("qq_chunks_v1.num_sign must be in {-1,0,1}.")
        num = _chunks_to_bigint(phi_obj.num_chunks, phi_obj.num_ptr, idx, "qq_chunks_v1.num")
        den = _chunks_to_bigint(phi_obj.den_chunks, phi_obj.den_ptr, idx, "qq_chunks_v1.den")
        den == 0 && error("qq_chunks_v1.den must be nonzero.")
        if s < 0
            num = -num
        elseif s == 0
            num = BigInt(0)
        end
        Phi[idx] = coerce(target_field, QQ(num // den))
    end
    return Phi
end

function _decode_phi(phi_obj::_PhiFpFlatJSON,
                     saved_field::PrimeField,
                     target_field::PrimeField,
                     m_expected::Int,
                     k_expected::Int)
    m, k = _phi_dims(phi_obj)
    (m == m_expected && k == k_expected) || error("phi dimensions mismatch (expected $(m_expected)x$(k_expected), got $(m)x$(k)).")
    len = m * k
    length(phi_obj.data) == len || error("fp_flat_v1.data length mismatch.")
    K = coeff_type(target_field)
    Phi = Matrix{K}(undef, m, k)
    @inbounds for idx in 1:len
        Phi[idx] = coerce(target_field, phi_obj.data[idx])
    end
    return Phi
end

function _decode_phi(phi_obj::_PhiFpFlatJSON,
                     saved_field::PrimeField,
                     target_field::AbstractCoeffField,
                     m_expected::Int,
                     k_expected::Int)
    m, k = _phi_dims(phi_obj)
    (m == m_expected && k == k_expected) || error("phi dimensions mismatch (expected $(m_expected)x$(k_expected), got $(m)x$(k)).")
    len = m * k
    length(phi_obj.data) == len || error("fp_flat_v1.data length mismatch.")
    K = coeff_type(target_field)
    Phi = Matrix{K}(undef, m, k)
    @inbounds for idx in 1:len
        v = coerce(saved_field, phi_obj.data[idx])
        Phi[idx] = coerce(target_field, v)
    end
    return Phi
end

function _decode_phi(phi_obj::_PhiRealFlatJSON,
                     saved_field::RealField,
                     target_field::RealField,
                     m_expected::Int,
                     k_expected::Int)
    m, k = _phi_dims(phi_obj)
    (m == m_expected && k == k_expected) || error("phi dimensions mismatch (expected $(m_expected)x$(k_expected), got $(m)x$(k)).")
    len = m * k
    length(phi_obj.data) == len || error("real_flat_v1.data length mismatch.")
    K = coeff_type(target_field)
    Phi = Matrix{K}(undef, m, k)
    @inbounds for idx in 1:len
        Phi[idx] = K(phi_obj.data[idx])
    end
    return Phi
end

function _decode_phi(phi_obj::_PhiRealFlatJSON,
                     saved_field::RealField,
                     target_field::AbstractCoeffField,
                     m_expected::Int,
                     k_expected::Int)
    m, k = _phi_dims(phi_obj)
    (m == m_expected && k == k_expected) || error("phi dimensions mismatch (expected $(m_expected)x$(k_expected), got $(m)x$(k)).")
    len = m * k
    length(phi_obj.data) == len || error("real_flat_v1.data length mismatch.")
    K = coeff_type(target_field)
    Phi = Matrix{K}(undef, m, k)
    @inbounds for idx in 1:len
        Phi[idx] = coerce(target_field, phi_obj.data[idx])
    end
    return Phi
end

function _decode_phi(phi_obj::_PhiJSON,
                     saved_field::AbstractCoeffField,
                     target_field::AbstractCoeffField,
                     m_expected::Int,
                     k_expected::Int)
    error("Unsupported phi payload/field combination: $(typeof(phi_obj)) with saved_field=$(typeof(saved_field))")
end

function _parse_poset_from_obj(poset_obj)
    kind = haskey(poset_obj, "kind") ? String(poset_obj["kind"]) : "FinitePoset"
    if kind == "FinitePoset"
        haskey(poset_obj, "n") || error("poset missing required key 'n'.")
        haskey(poset_obj, "leq") || error("poset missing required key 'leq'.")
        n = Int(poset_obj["n"])
        leq_any = poset_obj["leq"]
        leq = if _is_packed_words_obj(leq_any)
            packed = _MaskPackedWordsJSON(kind=String(leq_any["kind"]),
                                          nrows=Int(leq_any["nrows"]),
                                          ncols=Int(leq_any["ncols"]),
                                          words_per_row=Int(leq_any["words_per_row"]),
                                          words=Vector{UInt64}(leq_any["words"]))
            _decode_bitmatrix(packed, "FinitePoset.leq", n, n)
        else
            leq_any isa AbstractVector || error("poset.leq must be a list-of-lists.")
            length(leq_any) == n || error("poset.leq must have n=$(n) rows")
            L = falses(n, n)
            for i in 1:n
                row = leq_any[i]
                row isa AbstractVector || error("poset.leq row $(i) must be a list.")
                length(row) == n || error("poset.leq row length mismatch (row $(i); expected n=$(n)).")
                for j in 1:n
                    x = row[j]
                    x isa Bool || error("poset.leq entries must be Bool (row $(i), col $(j)).")
                    L[i, j] = x
                end
            end
            L
        end
        P = FinitePoset(leq)
        _clear_cover_cache!(P)
        return P
    elseif kind == "ProductOfChainsPoset"
        haskey(poset_obj, "sizes") || error("ProductOfChainsPoset missing required key 'sizes'.")
        sizes = Vector{Int}(poset_obj["sizes"])
        P = ProductOfChainsPoset(sizes)
        _clear_cover_cache!(P)
        return P
    elseif kind == "GridPoset"
        haskey(poset_obj, "coords") || error("GridPoset missing required key 'coords'.")
        coords_any = poset_obj["coords"]
        coords_any isa AbstractVector || error("GridPoset.coords must be a list-of-lists.")
        coords = ntuple(i -> Vector{Float64}(coords_any[i]), length(coords_any))
        P = GridPoset(coords)
        _clear_cover_cache!(P)
        return P
    elseif kind == "ProductPoset"
        haskey(poset_obj, "left") || error("ProductPoset missing required key 'left'.")
        haskey(poset_obj, "right") || error("ProductPoset missing required key 'right'.")
        P1 = _parse_poset_from_obj(poset_obj["left"])
        P2 = _parse_poset_from_obj(poset_obj["right"])
        P = ProductPoset(P1, P2)
        _clear_cover_cache!(P)
        return P
    elseif kind == "SignaturePoset"
        haskey(poset_obj, "sig_y") || error("SignaturePoset missing required key 'sig_y'.")
        haskey(poset_obj, "sig_z") || error("SignaturePoset missing required key 'sig_z'.")
        sig_y_any = poset_obj["sig_y"]
        sig_z_any = poset_obj["sig_z"]
        sig_y = if _is_packed_words_obj(sig_y_any)
            _parse_signature_rows_packed(
                _MaskPackedWordsJSON(kind=String(sig_y_any["kind"]),
                                     nrows=Int(sig_y_any["nrows"]),
                                     ncols=Int(sig_y_any["ncols"]),
                                     words_per_row=Int(sig_y_any["words_per_row"]),
                                     words=Vector{UInt64}(sig_y_any["words"])),
                "SignaturePoset.sig_y")
        else
            _parse_signature_rows_packed(sig_y_any, "SignaturePoset.sig_y")
        end
        sig_z = if _is_packed_words_obj(sig_z_any)
            _parse_signature_rows_packed(
                _MaskPackedWordsJSON(kind=String(sig_z_any["kind"]),
                                     nrows=Int(sig_z_any["nrows"]),
                                     ncols=Int(sig_z_any["ncols"]),
                                     words_per_row=Int(sig_z_any["words_per_row"]),
                                     words=Vector{UInt64}(sig_z_any["words"])),
                "SignaturePoset.sig_z")
        else
            _parse_signature_rows_packed(sig_z_any, "SignaturePoset.sig_z")
        end
        P = SignaturePoset(sig_y, sig_z)
        _clear_cover_cache!(P)
        return P
    else
        error("Unsupported poset kind: $(kind)")
    end
end

function _poset_obj(P::AbstractPoset; include_leq::Union{Bool,Symbol}=:auto)
    include_leq_resolved = _resolve_include_leq(P, include_leq)
    if P isa FinitePoset
        include_leq_resolved || error("Cannot omit leq for FinitePoset serialization.")
        L = leq_matrix(P)
        return Dict("kind" => "FinitePoset",
                    "n" => nvertices(P),
                    "leq" => _pack_bitmatrix_obj(L))
    elseif P isa ProductOfChainsPoset
        obj = Dict("kind" => "ProductOfChainsPoset",
                   "n" => nvertices(P),
                   "sizes" => collect(P.sizes))
        if include_leq_resolved
            L = leq_matrix(P)
            obj["leq"] = _pack_bitmatrix_obj(L)
        end
        return obj
    elseif P isa GridPoset
        obj = Dict("kind" => "GridPoset",
                   "n" => nvertices(P),
                   "coords" => [collect(c) for c in P.coords])
        if include_leq_resolved
            L = leq_matrix(P)
            obj["leq"] = _pack_bitmatrix_obj(L)
        end
        return obj
    elseif P isa ProductPoset
        obj = Dict("kind" => "ProductPoset",
                   "n" => nvertices(P),
                   "left" => _poset_obj(P.P1; include_leq=(include_leq === :auto ? :auto : include_leq_resolved)),
                   "right" => _poset_obj(P.P2; include_leq=(include_leq === :auto ? :auto : include_leq_resolved)))
        if include_leq_resolved
            L = leq_matrix(P)
            obj["leq"] = _pack_bitmatrix_obj(L)
        end
        return obj
    elseif P isa SignaturePoset
        obj = Dict("kind" => "SignaturePoset",
                   "n" => nvertices(P),
                   "sig_y" => _pack_signature_rows_obj(P.sig_y),
                   "sig_z" => _pack_signature_rows_obj(P.sig_z))
        if include_leq_resolved
            L = leq_matrix(P)
            obj["leq"] = _pack_bitmatrix_obj(L)
        end
        return obj
    else
        L = leq_matrix(P)
        return Dict("kind" => "FinitePoset",
                    "n" => nvertices(P),
                    "leq" => _pack_bitmatrix_obj(L))
    end
end

"""
    load_boundary_complex_json(path) -> GradedComplex

Expected schema (external adapter):
{
  "cells_by_dim": [[1,2,...], [1,2,...], ...]  // or "counts_by_dim": [n0, n1, ...]
  "boundaries": [ {"m":..,"n":..,"I":[..],"J":[..],"V":[..]}, ... ],
  "grades": [ [..], [..], ... ],
  "cell_dims": [..]   // optional
}
"""
function load_boundary_complex_json(path::AbstractString)
    obj = _json_read(path)
    cells = if haskey(obj, "cells_by_dim")
        [Vector{Int}(c) for c in obj["cells_by_dim"]]
    elseif haskey(obj, "counts_by_dim")
        counts = Vector{Int}(obj["counts_by_dim"])
        out = Vector{Vector{Int}}(undef, length(counts))
        for d in 1:length(counts)
            out[d] = collect(1:counts[d])
        end
        out
    else
        error("boundary complex JSON missing 'cells_by_dim' or 'counts_by_dim'.")
    end
    boundaries = SparseMatrixCSC{Int,Int}[]
    for b in obj["boundaries"]
        m = Int(b["m"]); n = Int(b["n"])
        I = Vector{Int}(b["I"])
        J = Vector{Int}(b["J"])
        V = Vector{Int}(b["V"])
        push!(boundaries, sparse(I, J, V, m, n))
    end
    grades = [Vector{Float64}(g) for g in obj["grades"]]
    cell_dims = haskey(obj, "cell_dims") ? Vector{Int}(obj["cell_dims"]) : nothing
    return GradedComplex(cells, boundaries, grades; cell_dims=cell_dims)
end

"""
    load_reduced_complex_json(path) -> GradedComplex

Alias for `load_boundary_complex_json`, intended for reduced boundary matrices.
"""
load_reduced_complex_json(path::AbstractString) = load_boundary_complex_json(path)

"""
    load_pmodule_json(path; field=nothing) -> PModule

Expected schema:
{
  "poset": { "kind": "FinitePoset", "n": n, "leq": [[...]] },
  "dims": [d1, d2, ...],
  "edges": [ {"src": i, "dst": j, "mat": [[...]]}, ... ],
  "coeff_field": { ... }   // optional; defaults to QQ
}
"""
function load_pmodule_json(path::AbstractString; field::Union{Nothing,AbstractCoeffField}=nothing)
    obj = _json_read(path)
    haskey(obj, "poset") || error("pmodule JSON missing 'poset'.")
    P = _parse_poset_from_obj(obj["poset"])
    saved_field = haskey(obj, "coeff_field") ? _field_from_obj(obj["coeff_field"]) : QQField()
    target_field = field === nothing ? saved_field : field
    K = coeff_type(target_field)
    haskey(obj, "dims") || error("pmodule JSON missing 'dims'.")
    dims = Vector{Int}(obj["dims"])
    length(dims) == nvertices(P) || error("pmodule dims length mismatch with poset size.")
    haskey(obj, "edges") || error("pmodule JSON missing 'edges'.")
    edge_maps = Dict{Tuple{Int,Int},Matrix{K}}()
    for e in obj["edges"]
        src = Int(e["src"])
        dst = Int(e["dst"])
        mat_any = e["mat"]
        mat_any isa AbstractVector || error("pmodule edge mat must be a list-of-lists.")
        m = length(mat_any)
        n = m == 0 ? 0 : length(mat_any[1])
        M = zeros(K, m, n)
        for i in 1:m
            row = mat_any[i]
            length(row) == n || error("pmodule edge mat row length mismatch.")
            for j in 1:n
                val = _scalar_from_json(saved_field, row[j])
                M[i, j] = target_field === saved_field ? val : coerce(target_field, val)
            end
        end
        edge_maps[(src, dst)] = M
    end
    return PModule{K}(P, dims, edge_maps; field=target_field)
end

# -----------------------------------------------------------------------------
# B1) Interop adapters: Ripser/DIPHA distance matrix formats
# -----------------------------------------------------------------------------

function _read_numeric_rows(path::AbstractString)
    rows = Vector{Vector{Float64}}()
    open(path, "r") do io
        for raw in eachline(io)
            line = strip(raw)
            isempty(line) && continue
            startswith(line, "#") && continue
            line = replace(line, ',' => ' ')
            line = replace(line, ';' => ' ')
            parts = split(line)
            isempty(parts) && continue
            row = Float64[parse(Float64, p) for p in parts]
            push!(rows, row)
        end
    end
    return rows
end

function _matrix_from_rows(rows::Vector{Vector{Float64}})
    isempty(rows) && error("distance matrix: empty file.")

    if length(rows) >= 2 && length(rows[1]) == 1
        n_header = Int(round(rows[1][1]))
        if n_header >= 1 && length(rows) - 1 == n_header &&
           all(length(rows[i]) == n_header for i in 2:length(rows))
            dist = zeros(Float64, n_header, n_header)
            for i in 1:n_header
                dist[i, :] = rows[i + 1]
            end
            return dist
        end
    end

    same_len = all(length(r) == length(rows[1]) for r in rows)
    if same_len
        m = length(rows)
        k = length(rows[1])
        if m == k
            dist = zeros(Float64, m, m)
            for i in 1:m
                dist[i, :] = rows[i]
            end
            return dist
        elseif k == 1
            vals = [r[1] for r in rows]
            n = round(Int, sqrt(length(vals)))
            n * n == length(vals) || error("distance matrix: flat list length is not a perfect square.")
            dist = zeros(Float64, n, n)
            for i in 1:n, j in 1:n
                dist[i, j] = vals[(i - 1) * n + j]
            end
            return dist
        elseif m == 1
            vals = rows[1]
            n = round(Int, sqrt(length(vals)))
            n * n == length(vals) || error("distance matrix: flat list length is not a perfect square.")
            dist = zeros(Float64, n, n)
            for i in 1:n, j in 1:n
                dist[i, j] = vals[(i - 1) * n + j]
            end
            return dist
        end
    end

    vals = reduce(vcat, rows)
    n = round(Int, sqrt(length(vals)))
    n * n == length(vals) || error("distance matrix: could not infer square size.")
    dist = zeros(Float64, n, n)
    for i in 1:n, j in 1:n
        dist[i, j] = vals[(i - 1) * n + j]
    end
    return dist
end

function _infer_n_from_triangular_len(len::Int)
    n1 = Int(floor((sqrt(1 + 8 * len) - 1) / 2))
    if div(n1 * (n1 + 1), 2) == len
        return n1, true
    end
    n2 = Int(floor((1 + sqrt(1 + 8 * len)) / 2))
    if div(n2 * (n2 - 1), 2) == len
        return n2, false
    end
    return 0, false
end

function _triangular_from_rows(rows::Vector{Vector{Float64}}; upper::Bool)
    n = length(rows)
    if upper
        include_diag = if all(length(rows[i]) == n - i + 1 for i in 1:n)
            true
        elseif all(length(rows[i]) == n - i for i in 1:n)
            false
        else
            return nothing
        end
        dist = fill(Inf, n, n)
        for i in 1:n
            row = rows[i]
            if include_diag
                for k in 1:(n - i + 1)
                    j = i + k - 1
                    dist[i, j] = row[k]
                    dist[j, i] = row[k]
                end
            else
                for k in 1:(n - i)
                    j = i + k
                    dist[i, j] = row[k]
                    dist[j, i] = row[k]
                end
            end
            dist[i, i] = 0.0
        end
        return dist
    else
        include_diag = if all(length(rows[i]) == i for i in 1:n)
            true
        elseif all(length(rows[i]) == i - 1 for i in 1:n)
            false
        else
            return nothing
        end
        dist = fill(Inf, n, n)
        for i in 1:n
            row = rows[i]
            if include_diag
                for j in 1:i
                    dist[i, j] = row[j]
                    dist[j, i] = row[j]
                end
            else
                for j in 1:(i - 1)
                    dist[i, j] = row[j]
                    dist[j, i] = row[j]
                end
            end
            dist[i, i] = 0.0
        end
        return dist
    end
end

function _triangular_from_vals(vals::Vector{Float64}; upper::Bool)
    n, include_diag = _infer_n_from_triangular_len(length(vals))
    n > 0 || error("triangular distance list length is not valid.")
    dist = fill(Inf, n, n)
    idx = 1
    if upper
        for i in 1:n
            if include_diag
                for j in i:n
                    val = vals[idx]; idx += 1
                    dist[i, j] = val
                    dist[j, i] = val
                end
            else
                for j in (i + 1):n
                    val = vals[idx]; idx += 1
                    dist[i, j] = val
                    dist[j, i] = val
                end
            end
            dist[i, i] = 0.0
        end
    else
        for i in 1:n
            if include_diag
                for j in 1:i
                    val = vals[idx]; idx += 1
                    dist[i, j] = val
                    dist[j, i] = val
                end
            else
                for j in 1:(i - 1)
                    val = vals[idx]; idx += 1
                    dist[i, j] = val
                    dist[j, i] = val
                end
            end
            dist[i, i] = 0.0
        end
    end
    return dist
end

function _combinations(n::Int, k::Int)
    if k == 0
        return [Int[]]
    end
    out = Vector{Vector{Int}}()
    function rec(start::Int, acc::Vector{Int})
        if length(acc) == k
            push!(out, copy(acc))
            return
        end
        for i in start:(n - (k - length(acc)) + 1)
            push!(acc, i)
            rec(i + 1, acc)
            pop!(acc)
        end
    end
    rec(1, Int[])
    return out
end

@inline function _dm_budget_check_max_simplices!(total::Integer, budget::ConstructionBudget)
    ms = budget.max_simplices
    if ms !== nothing && total > ms
        error("distance matrix Rips: exceeded max_simplices=$(ms).")
    end
    return nothing
end

@inline function _dm_budget_check_max_edges!(edge_count, budget::ConstructionBudget)
    cap = budget.max_edges
    if cap !== nothing && big(edge_count) > big(cap)
        error("distance matrix Rips: exceeded max_edges=$(cap).")
    end
    return nothing
end

function _dm_edges_radius(dist::AbstractMatrix{<:Real}, radius::Float64)
    n = size(dist, 1)
    edges = Vector{Vector{Int}}()
    for i in 1:n, j in i+1:n
        if Float64(dist[i, j]) <= radius
            push!(edges, [i, j])
        end
    end
    return edges
end

function _dm_edges_knn(dist::AbstractMatrix{<:Real}, k::Int)
    n = size(dist, 1)
    0 < k < n || error("construction.sparsify=:knn requires 0 < knn < n.")
    e = Set{Tuple{Int,Int}}()
    for i in 1:n
        neigh = [(Float64(dist[i, j]), j) for j in 1:n if j != i]
        sort!(neigh, by=x -> x[1])
        tmax = min(k, length(neigh))
        for t in 1:tmax
            j = neigh[t][2]
            a, b = min(i, j), max(i, j)
            push!(e, (a, b))
        end
    end
    edges = [[ab[1], ab[2]] for ab in e]
    sort!(edges; by=s -> (s[1], s[2]))
    return edges
end

function _dm_edges_collapse_dominated(edges::Vector{Vector{Int}},
                                      dist::AbstractMatrix{<:Real};
                                      tol::Float64=1e-12)
    n = size(dist, 1)
    out = Vector{Vector{Int}}()
    for e in edges
        u, v = e[1], e[2]
        duv = Float64(dist[u, v])
        dominated = false
        for w in 1:n
            (w == u || w == v) && continue
            if max(Float64(dist[u, w]), Float64(dist[w, v])) <= duv + tol
                dominated = true
                break
            end
        end
        dominated || push!(out, e)
    end
    return out
end

function _dm_edges_collapse_acyclic(edges::Vector{Vector{Int}},
                                    dist::AbstractMatrix{<:Real})
    n = size(dist, 1)
    parent = collect(1:n)
    rank = zeros(Int, n)
    function findp(x)
        while parent[x] != x
            parent[x] = parent[parent[x]]
            x = parent[x]
        end
        return x
    end
    function unite(x, y)
        rx, ry = findp(x), findp(y)
        rx == ry && return false
        if rank[rx] < rank[ry]
            parent[rx] = ry
        elseif rank[rx] > rank[ry]
            parent[ry] = rx
        else
            parent[ry] = rx
            rank[rx] += 1
        end
        return true
    end
    idx = sortperm(1:length(edges); by=i -> Float64(dist[edges[i][1], edges[i][2]]))
    out = Vector{Vector{Int}}()
    for i in idx
        e = edges[i]
        unite(e[1], e[2]) && push!(out, e)
    end
    return out
end

function _dm_apply_collapse(edges::Vector{Vector{Int}},
                            dist::AbstractMatrix{<:Real},
                            collapse::Symbol)
    if collapse == :none
        return edges
    elseif collapse == :dominated_edges
        return _dm_edges_collapse_dominated(edges, dist)
    elseif collapse == :acyclic
        return _dm_edges_collapse_acyclic(edges, dist)
    end
    error("construction.collapse must be :none, :dominated_edges, or :acyclic.")
end

function _graded_complex_from_distance_matrix(dist::AbstractMatrix{<:Real};
                                              max_dim::Int=1,
                                              radius::Union{Nothing,Real}=nothing,
                                              knn::Union{Nothing,Int}=nothing,
                                              construction::ConstructionOptions=ConstructionOptions())
    size(dist, 1) == size(dist, 2) || error("distance matrix must be square.")
    max_dim >= 0 || error("max_dim must be >= 0.")
    n = size(dist, 1)
    n > 0 || error("distance matrix has size 0.")

    sparsify = construction.sparsify
    collapse = construction.collapse
    budget = construction.budget

    if sparsify == :greedy_perm
        error("construction.sparsify=:greedy_perm is not supported for distance-matrix ingestion.")
    end
    if sparsify != :none && max_dim > 1
        error("construction.sparsify=$(sparsify) currently supports max_dim <= 1 for distance-matrix ingestion.")
    end
    if collapse != :none && sparsify == :none
        error("construction.collapse requires construction.sparsify != :none for distance-matrix ingestion.")
    end
    if radius !== nothing && sparsify != :radius
        error("radius is only valid when construction.sparsify=:radius.")
    end
    if knn !== nothing && sparsify != :knn
        error("knn is only valid when construction.sparsify=:knn.")
    end

    simplices = Vector{Vector{Vector{Int}}}(undef, max_dim + 1)
    simplices[1] = [ [i] for i in 1:n ]
    total = length(simplices[1])

    if sparsify == :none
        if max_dim >= 1
            _dm_budget_check_max_edges!(binomial(big(n), big(2)), budget)
        end
        for k in 2:max_dim+1
            sims = Vector{Vector{Int}}()
            for comb in _combinations(n, k)
                push!(sims, comb)
            end
            simplices[k] = sims
            total += length(sims)
            _dm_budget_check_max_simplices!(total, budget)
        end
    else
        edges = if sparsify == :radius
            radius === nothing && error("construction.sparsify=:radius requires radius.")
            _dm_edges_radius(dist, Float64(radius))
        elseif sparsify == :knn
            knn === nothing && error("construction.sparsify=:knn requires knn.")
            _dm_edges_knn(dist, Int(knn))
        else
            error("construction.sparsify must be :none, :radius, or :knn for distance-matrix ingestion.")
        end
        edges = _dm_apply_collapse(edges, dist, collapse)
        _dm_budget_check_max_edges!(length(edges), budget)
        simplices = [simplices[1], edges]
        max_dim = 1
        total += length(edges)
        _dm_budget_check_max_simplices!(total, budget)
    end

    grades = Vector{Vector{Float64}}()
    for _ in simplices[1]
        push!(grades, [0.0])
    end
    for k in 2:max_dim+1
        for s in simplices[k]
            maxd = 0.0
            for i in 1:length(s)
                for j in (i+1):length(s)
                    d = Float64(dist[s[i], s[j]])
                    if d > maxd
                        maxd = d
                    end
                end
            end
            push!(grades, [maxd])
        end
    end

    boundaries = SparseMatrixCSC{Int,Int}[]
    for k in 2:max_dim+1
        Bk = _simplicial_boundary_from_lists(simplices[k], simplices[k-1])
        push!(boundaries, Bk)
    end
    cells = [collect(1:length(simplices[k])) for k in 1:length(simplices)]
    return GradedComplex(cells, boundaries, grades)
end

"""
    load_ripser_point_cloud(path) -> PointCloud

Parse a Ripser-style point cloud (whitespace-separated coordinates per line).
"""
function load_ripser_point_cloud(path::AbstractString)
    rows = _read_numeric_rows(path)
    isempty(rows) && error("point cloud file has no points.")
    return PointCloud([Vector{Float64}(r) for r in rows])
end

"""
    load_ripser_distance(path; max_dim=1, radius=nothing, knn=nothing,
                         construction=ConstructionOptions()) -> GradedComplex

Parse a full distance matrix (square) and build a 1-parameter Rips graded complex.
"""
function load_ripser_distance(path::AbstractString;
                              max_dim::Int=1,
                              radius::Union{Nothing,Real}=nothing,
                              knn::Union{Nothing,Int}=nothing,
                              construction::ConstructionOptions=ConstructionOptions())
    rows = _read_numeric_rows(path)
    dist = _matrix_from_rows(rows)
    return _graded_complex_from_distance_matrix(dist;
                                                max_dim=max_dim,
                                                radius=radius,
                                                knn=knn,
                                                construction=construction)
end

"""
    load_ripser_lower_distance(path; max_dim=1, radius=nothing, knn=nothing,
                               construction=ConstructionOptions()) -> GradedComplex

Parse a lower-triangular distance matrix (row-wise or flat list) and build a Rips complex.
"""
function load_ripser_lower_distance(path::AbstractString;
                                    max_dim::Int=1,
                                    radius::Union{Nothing,Real}=nothing,
                                    knn::Union{Nothing,Int}=nothing,
                                    construction::ConstructionOptions=ConstructionOptions())
    rows = _read_numeric_rows(path)
    dist = _triangular_from_rows(rows; upper=false)
    if dist === nothing
        dist = _triangular_from_vals(reduce(vcat, rows); upper=false)
    end
    return _graded_complex_from_distance_matrix(dist;
                                                max_dim=max_dim,
                                                radius=radius,
                                                knn=knn,
                                                construction=construction)
end

"""
    load_ripser_upper_distance(path; max_dim=1, radius=nothing, knn=nothing,
                               construction=ConstructionOptions()) -> GradedComplex

Parse an upper-triangular distance matrix (row-wise or flat list) and build a Rips complex.
"""
function load_ripser_upper_distance(path::AbstractString;
                                    max_dim::Int=1,
                                    radius::Union{Nothing,Real}=nothing,
                                    knn::Union{Nothing,Int}=nothing,
                                    construction::ConstructionOptions=ConstructionOptions())
    rows = _read_numeric_rows(path)
    dist = _triangular_from_rows(rows; upper=true)
    if dist === nothing
        dist = _triangular_from_vals(reduce(vcat, rows); upper=true)
    end
    return _graded_complex_from_distance_matrix(dist;
                                                max_dim=max_dim,
                                                radius=radius,
                                                knn=knn,
                                                construction=construction)
end

"""
    load_ripser_sparse_triplet(path; max_dim=1, radius=nothing, knn=nothing,
                               construction=ConstructionOptions()) -> GradedComplex

Parse a sparse triplet format: each nonzero entry as "i j d".
Indices can be 0-based or 1-based.
"""
function load_ripser_sparse_triplet(path::AbstractString;
                                    max_dim::Int=1,
                                    radius::Union{Nothing,Real}=nothing,
                                    knn::Union{Nothing,Int}=nothing,
                                    construction::ConstructionOptions=ConstructionOptions())
    rows = _read_numeric_rows(path)
    isempty(rows) && error("sparse triplet file is empty.")
    for row in rows
        length(row) == 3 || error("sparse triplet rows must have 3 entries.")
    end
    idxs = [Int(round(r[1])) for r in rows]
    jdxs = [Int(round(r[2])) for r in rows]
    base0 = any(i == 0 for i in idxs) || any(j == 0 for j in jdxs)
    if base0
        idxs .= idxs .+ 1
        jdxs .= jdxs .+ 1
    end
    n = max(maximum(idxs), maximum(jdxs))
    n > 0 || error("sparse triplet: could not infer matrix size.")
    dist = fill(Inf, n, n)
    for i in 1:n
        dist[i, i] = 0.0
    end
    for t in 1:length(rows)
        i = idxs[t]
        j = jdxs[t]
        d = Float64(rows[t][3])
        if d < dist[i, j]
            dist[i, j] = d
            dist[j, i] = d
        end
    end
    return _graded_complex_from_distance_matrix(dist;
                                                max_dim=max_dim,
                                                radius=radius,
                                                knn=knn,
                                                construction=construction)
end

"""
    load_ripser_binary_lower_distance(path; max_dim=1, radius=nothing, knn=nothing,
                                      construction=ConstructionOptions()) -> GradedComplex

Parse Ripser's binary lower-triangular distance matrix (Float64 values).
"""
function load_ripser_binary_lower_distance(path::AbstractString;
                                           max_dim::Int=1,
                                           radius::Union{Nothing,Real}=nothing,
                                           knn::Union{Nothing,Int}=nothing,
                                           construction::ConstructionOptions=ConstructionOptions())
    vals = Float64[]
    open(path, "r") do io
        while !eof(io)
            push!(vals, read(io, Float64))
        end
    end
    dist = _triangular_from_vals(vals; upper=false)
    return _graded_complex_from_distance_matrix(dist;
                                                max_dim=max_dim,
                                                radius=radius,
                                                knn=knn,
                                                construction=construction)
end

"""
    load_dipha_distance_matrix(path; max_dim=1, radius=nothing, knn=nothing,
                               construction=ConstructionOptions()) -> GradedComplex

Parse a DIPHA binary distance matrix and build a Rips graded complex.
"""
function load_dipha_distance_matrix(path::AbstractString;
                                    max_dim::Int=1,
                                    radius::Union{Nothing,Real}=nothing,
                                    knn::Union{Nothing,Int}=nothing,
                                    construction::ConstructionOptions=ConstructionOptions())
    open(path, "r") do io
        eof(io) && error("DIPHA distance matrix: empty file.")
        magic = read(io, Int64)
        magic == 8067171840 || error("DIPHA: invalid magic value.")
        _ = read(io, Int64) # file type id
        n = Int(read(io, Int64))
        n > 0 || error("DIPHA: invalid matrix size.")
        vals = Vector{Float64}(undef, n * n)
        read!(io, vals)
        dist = zeros(Float64, n, n)
        idx = 1
        for i in 1:n, j in 1:n
            dist[i, j] = vals[idx]
            idx += 1
        end
        return _graded_complex_from_distance_matrix(dist;
                                                    max_dim=max_dim,
                                                    radius=radius,
                                                    knn=knn,
                                                    construction=construction)
    end
end

"""
    load_ripser_lower_distance_streaming(path; radius, max_dim=1) -> GradedComplex

Streaming reader for lower-triangular distance matrices (text). Builds a 1-skeleton
Rips complex without loading the full matrix.
"""
function load_ripser_lower_distance_streaming(path::AbstractString; radius, max_dim::Int=1)
    max_dim == 1 || error("streaming lower distance currently supports max_dim=1 only.")
    radius === nothing && error("streaming lower distance requires radius.")
    edges = Vector{Vector{Int}}()
    grades = Vector{Vector{Float64}}()
    n = 0
    include_diag = nothing
    open(path, "r") do io
        for raw in eachline(io)
            line = strip(raw)
            isempty(line) && continue
            startswith(line, "#") && continue
            line = replace(line, ',' => ' ')
            parts = split(line)
            isempty(parts) && continue
            row = Float64[parse(Float64, p) for p in parts]
            n += 1
            if include_diag === nothing
                if length(row) == n
                    include_diag = true
                elseif length(row) == n - 1
                    include_diag = false
                else
                    error("streaming lower distance: row length mismatch at row $(n).")
                end
            else
                expected = include_diag ? n : n - 1
                length(row) == expected || error("streaming lower distance: row length mismatch at row $(n).")
            end
            for j in 1:length(row)
                if include_diag && j == n
                    continue
                end
                d = row[j]
                if d <= radius
                    push!(edges, [j, n])
                    push!(grades, [d])
                end
            end
        end
    end
    n > 0 || error("streaming lower distance: no rows found.")
    vertices = [ [i] for i in 1:n ]
    cells = [collect(1:length(vertices)), collect(1:length(edges))]
    all_grades = Vector{Vector{Float64}}(undef, length(vertices) + length(edges))
    for i in 1:length(vertices)
        all_grades[i] = [0.0]
    end
    for i in 1:length(edges)
        all_grades[length(vertices) + i] = grades[i]
    end
    B1 = _simplicial_boundary_from_lists(edges, vertices)
    return GradedComplex(cells, [B1], all_grades)
end

"Inverse of `save_flange_json`."
function load_flange_json(path::AbstractString; field::Union{Nothing,AbstractCoeffField}=nothing)
    obj = _json_read(path)
    @assert haskey(obj, "kind") && String(obj["kind"]) == "FlangeZn"
    n = Int(obj["n"])

    mkface(idxs) = Face(n, begin
        m = falses(n)
        for t in idxs
            m[Int(t)] = true
        end
        m
    end)

    flats = [IndFlat(mkface(Vector{Int}(f["tau"])), Vector{Int}(f["b"]); id=:F)
             for f in obj["flats"]]
    injectives = [IndInj(mkface(Vector{Int}(e["tau"])), Vector{Int}(e["b"]); id=:E)
                  for e in obj["injectives"]]

    saved_field = haskey(obj, "coeff_field") ? _field_from_obj(obj["coeff_field"]) : QQField()
    target_field = field === nothing ? saved_field : field
    K = coeff_type(target_field)
    m = length(injectives)
    k = length(flats)
    Phi = Matrix{K}(undef, m, k)
    for i in 1:m, j in 1:k
        s = _scalar_from_json(saved_field, obj["phi"][i][j])
        if target_field !== saved_field
            s = coerce(target_field, s)
        end
        Phi[i, j] = s
    end
    return Flange{K}(n, flats, injectives, Phi; field=target_field)
end

# -----------------------------------------------------------------------------
# A1b) PL fringe (R^n)  (PLPolyhedra.PLFringe)
# -----------------------------------------------------------------------------

@inline _qq_json(x::QQ) = _scalar_to_json(QQField(), x)
@inline _qq_parse(x) = _scalar_from_json(QQField(), x)

function _qq_vector_to_obj(v::AbstractVector{QQ})
    out = Vector{Any}(undef, length(v))
    @inbounds for i in eachindex(v)
        out[i] = _qq_json(v[i])
    end
    return out
end

function _qq_matrix_to_obj(A::AbstractMatrix{QQ})
    m, n = size(A)
    rows = Vector{Any}(undef, m)
    @inbounds for i in 1:m
        row = Vector{Any}(undef, n)
        for j in 1:n
            row[j] = _qq_json(A[i, j])
        end
        rows[i] = row
    end
    return rows
end

function _parse_qq_vector(v_any, label::String)
    v_any isa AbstractVector || error("$(label) must be a vector.")
    out = Vector{QQ}(undef, length(v_any))
    @inbounds for i in eachindex(v_any)
        out[i] = _qq_parse(v_any[i])
    end
    return out
end

function _parse_qq_matrix(A_any, label::String)
    A_any isa AbstractVector || error("$(label) must be a matrix encoded as row vectors.")
    m = length(A_any)
    if m == 0
        return Matrix{QQ}(undef, 0, 0)
    end
    row1 = A_any[1]
    row1 isa AbstractVector || error("$(label)[1] must be a vector.")
    n = length(row1)
    A = Matrix{QQ}(undef, m, n)
    @inbounds for i in 1:m
        row = A_any[i]
        row isa AbstractVector || error("$(label)[$i] must be a vector.")
        length(row) == n || error("$(label) row-length mismatch at row $(i).")
        for j in 1:n
            A[i, j] = _qq_parse(row[j])
        end
    end
    return A
end

function _strict_mask_from_json(mask_any, nrows::Int, label::String)
    if mask_any === nothing
        return falses(nrows)
    elseif mask_any isa AbstractVector{Bool}
        length(mask_any) == nrows || error("$(label) length mismatch.")
        return BitVector(mask_any)
    elseif mask_any isa AbstractVector
        bits = falses(nrows)
        for t in mask_any
            idx = Int(t)
            1 <= idx <= nrows || error("$(label) index $(idx) out of range 1:$(nrows).")
            bits[idx] = true
        end
        return bits
    end
    error("$(label) must be Bool vector or index list.")
end

function _pl_hpoly_to_obj(h::PLPolyhedra.HPoly)
    return Dict(
        "A" => _qq_matrix_to_obj(h.A),
        "b" => _qq_vector_to_obj(h.b),
        "strict_mask" => collect(h.strict_mask),
        "strict_eps" => _qq_json(h.strict_eps),
    )
end

function _pl_union_to_obj(U::PLPolyhedra.PolyUnion)
    return Dict(
        "n" => U.n,
        "parts" => [_pl_hpoly_to_obj(p) for p in U.parts],
    )
end

function _parse_pl_hpoly_obj(part_obj, n::Int, label::String)
    haskey(part_obj, "A") || error("$(label) missing A.")
    haskey(part_obj, "b") || error("$(label) missing b.")
    A = _parse_qq_matrix(part_obj["A"], "$(label).A")
    b = _parse_qq_vector(part_obj["b"], "$(label).b")
    size(A, 1) == length(b) || error("$(label): size(A,1) must match length(b).")
    size(A, 2) == n || error("$(label): A has wrong ambient dimension $(size(A,2)); expected $(n).")
    strict_mask = _strict_mask_from_json(get(part_obj, "strict_mask", nothing), size(A, 1), "$(label).strict_mask")
    strict_eps = haskey(part_obj, "strict_eps") ? _qq_parse(part_obj["strict_eps"]) : PLPolyhedra.STRICT_EPS_QQ
    return PLPolyhedra.HPoly(n, A, b, nothing, strict_mask, strict_eps)
end

function _parse_pl_union_obj(union_obj, n::Int, label::String)
    parts_any = if haskey(union_obj, "parts")
        union_obj["parts"]
    else
        Any[union_obj]
    end
    parts_any isa AbstractVector || error("$(label).parts must be a vector.")
    parts = Vector{PLPolyhedra.HPoly}(undef, length(parts_any))
    @inbounds for i in eachindex(parts_any)
        parts[i] = _parse_pl_hpoly_obj(parts_any[i], n, "$(label).parts[$(i)]")
    end
    return PLPolyhedra.PolyUnion(n, parts)
end

function _parse_pl_generators(obj, key::String, n::Int, which::Symbol)
    haskey(obj, key) || error("PLFringe JSON missing '$(key)'.")
    gens_any = obj[key]
    gens_any isa AbstractVector || error("PLFringe $(key) must be a vector.")
    if which === :up
        out = Vector{PLPolyhedra.PLUpset}(undef, length(gens_any))
        @inbounds for i in eachindex(gens_any)
            U = _parse_pl_union_obj(gens_any[i], n, "$(key)[$(i)]")
            out[i] = PLPolyhedra.PLUpset(U)
        end
        return out
    end
    out = Vector{PLPolyhedra.PLDownset}(undef, length(gens_any))
    @inbounds for i in eachindex(gens_any)
        D = _parse_pl_union_obj(gens_any[i], n, "$(key)[$(i)]")
        out[i] = PLPolyhedra.PLDownset(D)
    end
    return out
end

function _parse_pl_phi(obj, n_down::Int, n_up::Int)
    if !haskey(obj, "phi")
        n_down == 0 || n_up == 0 || error("PLFringe JSON missing 'phi'.")
        return zeros(QQ, n_down, n_up)
    end
    Phi = _parse_qq_matrix(obj["phi"], "phi")
    size(Phi, 1) == n_down || error("PLFringe phi has wrong number of rows $(size(Phi,1)); expected $(n_down).")
    size(Phi, 2) == n_up || error("PLFringe phi has wrong number of cols $(size(Phi,2)); expected $(n_up).")
    return Phi
end

function _parse_pl_fringe_obj(obj; strict_schema::Bool)
    if strict_schema
        haskey(obj, "kind") && String(obj["kind"]) == "PLFringe" ||
            error("load_pl_fringe_json: expected kind=\"PLFringe\".")
        version = haskey(obj, "schema_version") ? Int(obj["schema_version"]) : 0
        version == PLFRINGE_SCHEMA_VERSION ||
            error("Unsupported PLFringe JSON schema_version: $(version). Expected $(PLFRINGE_SCHEMA_VERSION).")
    end
    n = Int(obj["n"])
    n >= 0 || error("PLFringe n must be >= 0.")
    ups_key = if haskey(obj, "ups")
        "ups"
    elseif !strict_schema && haskey(obj, "upsets")
        "upsets"
    else
        "ups"
    end
    downs_key = if haskey(obj, "downs")
        "downs"
    elseif !strict_schema && haskey(obj, "downsets")
        "downsets"
    else
        "downs"
    end
    Ups = _parse_pl_generators(obj, ups_key, n, :up)
    Downs = _parse_pl_generators(obj, downs_key, n, :down)
    Phi = _parse_pl_phi(obj, length(Downs), length(Ups))
    return PLPolyhedra.PLFringe(n, Ups, Downs, Phi)
end

"""
    save_pl_fringe_json(path, F::PLPolyhedra.PLFringe; pretty=false)

Stable PosetModules-owned PL fringe schema for `PLPolyhedra.PLFringe`.
"""
function save_pl_fringe_json(path::AbstractString, F::PLPolyhedra.PLFringe; pretty::Bool=false)
    ups = [_pl_union_to_obj(U.U) for U in F.Ups]
    downs = [_pl_union_to_obj(D.D) for D in F.Downs]
    phi = _qq_matrix_to_obj(F.Phi)
    obj = Dict(
        "kind" => "PLFringe",
        "schema_version" => PLFRINGE_SCHEMA_VERSION,
        "n" => F.n,
        "coeff_field" => _field_to_obj(QQField()),
        "ups" => ups,
        "downs" => downs,
        "phi" => phi,
    )
    return _json_write(path, obj; pretty=pretty)
end

"""
    load_pl_fringe_json(path) -> PLPolyhedra.PLFringe

Inverse of `save_pl_fringe_json`.
"""
function load_pl_fringe_json(path::AbstractString)
    obj = _json_read(path)
    return _parse_pl_fringe_obj(obj; strict_schema=true)
end

# -----------------------------------------------------------------------------
# A2) Finite encodings (FiniteFringe + typed v1 schema)
# -----------------------------------------------------------------------------

@inline function _pi_to_obj(pi)
    if pi isa CompiledEncoding
        pi = pi.pi
    end
    if pi isa GridEncodingMap
        return _GridEncodingMapJSON(kind="GridEncodingMap",
                                    coords=[collect(ax) for ax in pi.coords],
                                    orientation=collect(pi.orientation))
    elseif pi isa ZnEncoding.ZnEncodingMap
        return _ZnEncodingMapJSON(
            kind="ZnEncodingMap",
            n=pi.n,
            coords=[collect(ax) for ax in pi.coords],
            sig_y=_pack_signature_rows_obj(pi.sig_y),
            sig_z=_pack_signature_rows_obj(pi.sig_z),
            reps=[collect(r) for r in pi.reps],
            flats=[_FaceGeneratorJSON(collect(f.b), findall(identity, f.tau.coords)) for f in pi.flats],
            injectives=[_FaceGeneratorJSON(collect(e.b), findall(identity, e.tau.coords)) for e in pi.injectives],
            cell_shape=collect(pi.cell_shape),
            cell_strides=collect(pi.cell_strides),
            cell_to_region=pi.cell_to_region === nothing ? nothing : collect(pi.cell_to_region),
        )
    elseif pi isa PLBackend.PLEncodingMapBoxes
        return _PLEncodingMapBoxesJSON(
            kind="PLEncodingMapBoxes",
            n=pi.n,
            coords=[collect(ax) for ax in pi.coords],
            sig_y=_pack_masks_obj(pi.sig_y),
            sig_z=_pack_masks_obj(pi.sig_z),
            reps=[collect(r) for r in pi.reps],
            Ups=[collect(u.ell) for u in pi.Ups],
            Downs=[collect(d.u) for d in pi.Downs],
            cell_shape=collect(pi.cell_shape),
            cell_strides=collect(pi.cell_strides),
            cell_to_region=collect(pi.cell_to_region),
            coord_flags=[collect(f) for f in pi.coord_flags],
            axis_is_uniform=collect(pi.axis_is_uniform),
            axis_step=collect(pi.axis_step),
            axis_min=collect(pi.axis_min),
        )
    end
    error("Unsupported encoding map type for JSON serialization.")
end

@inline function _zn_sigkey(sig_y::PackedSignatureRows{MY},
                            sig_z::PackedSignatureRows{MZ},
                            t::Int) where {MY,MZ}
    ywords = ntuple(i -> sig_y.words[i, t], Val(MY))
    zwords = ntuple(i -> sig_z.words[i, t], Val(MZ))
    return ZnEncoding.SigKey{MY,MZ}(ywords, zwords)
end

function _pi_from_typed(P::AbstractPoset, obj::_GridEncodingMapJSON)
    n = length(obj.coords)
    coords = ntuple(i -> Vector{Float64}(obj.coords[i]), n)
    length(obj.orientation) == n || error("GridEncodingMap.orientation length mismatch.")
    orientation = ntuple(i -> Int(obj.orientation[i]), n)
    return GridEncodingMap(P, coords; orientation=orientation)
end

function _pi_from_typed(::AbstractPoset, obj::_ZnEncodingMapJSON)
    n = obj.n
    length(obj.coords) == n || error("ZnEncodingMap.coords length mismatch.")
    coords = ntuple(i -> Vector{Int}(obj.coords[i]), n)
    reps = [ntuple(i -> Int(r[i]), n) for r in obj.reps]
    mkface(idxs) = begin
        m = falses(n)
        for t in idxs
            m[Int(t)] = true
        end
        Face(n, m)
    end
    flats = [IndFlat(mkface(f.tau), Vector{Int}(f.b); id=:F) for f in obj.flats]
    injectives = [IndInj(mkface(e.tau), Vector{Int}(e.b); id=:E) for e in obj.injectives]
    sig_y = _parse_signature_rows_packed(obj.sig_y, "ZnEncodingMap.sig_y")
    sig_z = _parse_signature_rows_packed(obj.sig_z, "ZnEncodingMap.sig_z")
    length(sig_y) == length(reps) || error("ZnEncodingMap.sig_y region count mismatch.")
    length(sig_z) == length(reps) || error("ZnEncodingMap.sig_z region count mismatch.")
    sig_y.bitlen == length(flats) || error("ZnEncodingMap.sig_y bit length mismatch with flats.")
    sig_z.bitlen == length(injectives) || error("ZnEncodingMap.sig_z bit length mismatch with injectives.")
    MY = size(sig_y.words, 1)
    MZ = size(sig_z.words, 1)
    sig_to_region = Dict{ZnEncoding.SigKey{MY,MZ},Int}()
    @inbounds for t in 1:length(sig_y)
        sig_to_region[_zn_sigkey(sig_y, sig_z, t)] = t
    end
    if obj.cell_shape === nothing || obj.cell_strides === nothing
        return ZnEncoding.ZnEncodingMap(n, coords, sig_y, sig_z, reps, flats, injectives, sig_to_region)
    end
    cell_shape = Vector{Int}(obj.cell_shape)
    cell_strides = Vector{Int}(obj.cell_strides)
    length(cell_shape) == n || error("ZnEncodingMap.cell_shape length mismatch.")
    length(cell_strides) == n || error("ZnEncodingMap.cell_strides length mismatch.")
    cell_to_region = obj.cell_to_region === nothing ? nothing : Vector{Int}(obj.cell_to_region)
    return ZnEncoding.ZnEncodingMap(n, coords, sig_y, sig_z, reps, flats, injectives, sig_to_region,
                                    ntuple(i -> cell_shape[i], n),
                                    ntuple(i -> cell_strides[i], n),
                                    cell_to_region)
end

function _pi_from_typed(::AbstractPoset, obj::_PLEncodingMapBoxesJSON)
    n = obj.n
    length(obj.coords) == n || error("PLEncodingMapBoxes.coords length mismatch.")
    coords = ntuple(i -> Vector{Float64}(obj.coords[i]), n)
    reps = [ntuple(i -> Float64(r[i]), n) for r in obj.reps]
    Ups = [PLBackend.BoxUpset(Vector{Float64}(u)) for u in obj.Ups]
    Downs = [PLBackend.BoxDownset(Vector{Float64}(d)) for d in obj.Downs]
    sig_y = _decode_masks(obj.sig_y, "PLEncodingMapBoxes.sig_y", length(Ups))
    sig_z = _decode_masks(obj.sig_z, "PLEncodingMapBoxes.sig_z", length(Downs))
    length(sig_y) == length(reps) || error("PLEncodingMapBoxes.sig_y region count mismatch.")
    length(sig_z) == length(reps) || error("PLEncodingMapBoxes.sig_z region count mismatch.")
    MY = max(1, cld(max(length(Ups), 1), 64))
    MZ = max(1, cld(max(length(Downs), 1), 64))
    sig_to_region = Dict{PLBackend.SigKey{MY,MZ},Int}()
    @inbounds for t in eachindex(sig_y)
        ywords = PLBackend._pack_bitvector_words(sig_y[t], Val(MY))
        zwords = PLBackend._pack_bitvector_words(sig_z[t], Val(MZ))
        sig_to_region[PLBackend.SigKey{MY,MZ}(ywords, zwords)] = t
    end
    return PLBackend.PLEncodingMapBoxes{n,MY,MZ}(
        n, coords, sig_y, sig_z, reps, Ups, Downs, sig_to_region,
        Vector{Int}(obj.cell_shape),
        Vector{Int}(obj.cell_strides),
        Vector{Int}(obj.cell_to_region),
        [Vector{UInt8}(f) for f in obj.coord_flags],
        BitVector(obj.axis_is_uniform),
        Vector{Float64}(obj.axis_step),
        Vector{Float64}(obj.axis_min))
end

function _pi_from_typed(::AbstractPoset, obj::_PiJSON)
    error("Unsupported encoding map kind: $(typeof(obj))")
end

function _encoding_obj(H::FringeModule{K};
                       pi=nothing,
                       include_leq::Union{Bool,Symbol}=:auto) where {K}
    P = H.P

    U_masks = _pack_masks_obj([U.mask for U in H.U])
    D_masks = _pack_masks_obj([D.mask for D in H.D])
    phi = _phi_obj(H)

    obj = Dict(
        "kind" => "FiniteEncodingFringe",
        "schema_version" => ENCODING_SCHEMA_VERSION,
        "poset" => _poset_obj(P; include_leq=include_leq),
        "U" => U_masks,
        "D" => D_masks,
        "coeff_field" => _field_to_obj(H.field),
        "phi" => phi,
    )
    if pi !== nothing
        obj["pi"] = _pi_to_obj(pi)
    end
    return obj
end

function save_encoding_json(path::AbstractString, H::FringeModule{K};
                            profile::Symbol=:compact,
                            include_leq::Union{Nothing,Bool,Symbol}=nothing,
                            pretty::Union{Nothing,Bool}=nothing) where {K}
    defaults = _resolve_encoding_save_profile(profile)
    include_leq_resolved = include_leq === nothing ? defaults.include_leq : include_leq
    pretty_resolved = pretty === nothing ? defaults.pretty : pretty
    return _json_write(path, _encoding_obj(H; include_leq=include_leq_resolved); pretty=pretty_resolved)
end

function save_encoding_json(path::AbstractString, P::AbstractPoset, H::FringeModule{K}, pi;
                            profile::Symbol=:compact,
                            include_leq::Union{Nothing,Bool,Symbol}=nothing,
                            pretty::Union{Nothing,Bool}=nothing) where {K}
    defaults = _resolve_encoding_save_profile(profile)
    include_leq_resolved = include_leq === nothing ? defaults.include_leq : include_leq
    pretty_resolved = pretty === nothing ? defaults.pretty : pretty
    P === H.P || error("save_encoding_json: P does not match H.P.")
    return _json_write(path, _encoding_obj(H; pi=pi, include_leq=include_leq_resolved); pretty=pretty_resolved)
end

"""
    save_encoding_json(path, enc::EncodingResult; profile=:compact, include_pi=nothing, include_leq=nothing, pretty=nothing)

Convenience serialization entrypoint for workflow users.
"""
function save_encoding_json(path::AbstractString, enc::EncodingResult;
                            profile::Symbol=:compact,
                            include_pi::Union{Nothing,Bool}=nothing,
                            include_leq::Union{Nothing,Bool,Symbol}=nothing,
                            pretty::Union{Nothing,Bool}=nothing)
    defaults = _resolve_encoding_save_profile(profile)
    include_pi_resolved = include_pi === nothing ? defaults.include_pi : include_pi
    include_leq_resolved = include_leq === nothing ? defaults.include_leq : include_leq
    pretty_resolved = pretty === nothing ? defaults.pretty : pretty
    H = enc.H
    H === nothing && (H = fringe_presentation(materialize_module(enc.M)))
    H isa FringeModule || error("save_encoding_json: EncodingResult.H must be a FringeModule.")
    if include_pi_resolved
        return save_encoding_json(path, enc.P, H, enc.pi; include_leq=include_leq_resolved, pretty=pretty_resolved)
    end
    return save_encoding_json(path, H; include_leq=include_leq_resolved, pretty=pretty_resolved)
end

# Load the schema emitted by save_encoding_json.
#
# This loader is intentionally strict: it expects the schema emitted by
# save_encoding_json (missing required keys => error).
function _load_encoding_json_v1(raw::AbstractString;
                                output::Symbol=:encoding_result,
                                field::Union{Nothing,AbstractCoeffField}=nothing,
                                validation::Symbol=:strict)
    outmode = _resolve_encoding_output_mode(output)
    validate_masks = _resolve_validation_mode(validation)
    obj = JSON3.read(raw, _FiniteEncodingFringeJSONV1)
    obj.kind == "FiniteEncodingFringe" || error("Unsupported encoding JSON kind: $(obj.kind)")
    obj.schema_version == ENCODING_SCHEMA_VERSION ||
        error("Unsupported encoding JSON schema_version: $(obj.schema_version)")

    P = _parse_poset_from_typed(obj.poset)
    n = nvertices(P)
    Umasks = _decode_masks(obj.U, "U", n)
    Dmasks = _decode_masks(obj.D, "D", n)
    U = _build_upsets(P, Umasks, validate_masks)
    D = _build_downsets(P, Dmasks, validate_masks)

    saved_field = _field_from_typed(obj.coeff_field)
    target_field = field === nothing ? saved_field : field
    K = coeff_type(target_field)
    m = length(D)
    k = length(U)
    Phi = _decode_phi(obj.phi, saved_field, target_field, m, k)

    H = FiniteFringe.FringeModule{K}(P, U, D, Phi; field=target_field)
    if outmode === :fringe
        return H
    elseif outmode === :fringe_with_pi
        obj.pi === nothing && error("load_encoding_json: output=:fringe_with_pi requires a stored pi payload. Re-save with save_encoding_json(...; include_pi=true) or load with output=:fringe.")
        return H, _pi_from_typed(P, obj.pi)
    elseif outmode === :encoding_result
        obj.pi === nothing && error("load_encoding_json: output=:encoding_result requires a stored pi payload. Re-save with save_encoding_json(...; include_pi=true) or load with output=:fringe.")
        pi = _pi_from_typed(P, obj.pi)
        M = pmodule_from_fringe(H)
        return EncodingResult(P, M, pi;
                              H=H,
                              presentation=nothing,
                              opts=EncodingOptions(),
                              backend=:serialization,
                              meta=(; source=:load_encoding_json, schema_version=obj.schema_version))
    end
    error("unreachable output mode: $(outmode)")
end

"""
    load_encoding_json(path; output=:encoding_result, field=nothing, validation=:strict)

Load an encoding artifact written by `save_encoding_json`.

Keywords
--------
- `output=:fringe | :fringe_with_pi | :encoding_result`
- `field`: optional coefficient-field override
- `validation=:strict | :trusted`
"""
function load_encoding_json(path::AbstractString;
                            output::Symbol=:encoding_result,
                            field::Union{Nothing,AbstractCoeffField}=nothing,
                            validation::Symbol=:strict)
    raw = read(path, String)
    return _load_encoding_json_v1(raw; output=output, field=field, validation=validation)
end

# =============================================================================
# B) External adapters (CAS ingestion)
# =============================================================================

@inline function _external_get_any(obj, keys::Tuple{Vararg{String}})
    for k in keys
        if haskey(obj, k)
            return obj[k]
        end
    end
    return nothing
end

function _external_parse_mask_rows(mask_any, name::String, ncols::Int)
    if mask_any isa AbstractDict
        if _is_packed_words_obj(mask_any)
            packed = _MaskPackedWordsJSON(kind=String(mask_any["kind"]),
                                          nrows=Int(mask_any["nrows"]),
                                          ncols=Int(mask_any["ncols"]),
                                          words_per_row=Int(mask_any["words_per_row"]),
                                          words=Vector{UInt64}(mask_any["words"]))
            return _decode_masks(packed, name, ncols)
        end
        rows_any = _external_get_any(mask_any, ("rows", "masks", "data"))
        rows_any === nothing && error("$(name): unsupported mask object; expected packed_words_v1 or rows/masks/data.")
        return _external_parse_mask_rows(rows_any, name, ncols)
    end

    mask_any isa AbstractVector || error("$(name) must be a vector of masks.")
    rows = Vector{BitVector}(undef, length(mask_any))
    @inbounds for i in eachindex(mask_any)
        row_any = mask_any[i]
        row_any isa AbstractVector || error("$(name)[$(i)] must be a vector.")
        if length(row_any) == ncols && all(x -> x isa Bool, row_any)
            rows[i] = BitVector(Bool[x for x in row_any])
        else
            mask = falses(ncols)
            for t in row_any
                idx = Int(t)
                1 <= idx <= ncols || error("$(name)[$(i)] contains out-of-range index $(idx) for ncols=$(ncols).")
                mask[idx] = true
            end
            rows[i] = mask
        end
    end
    return rows
end

function _external_parse_field(obj, field_override::Union{Nothing,AbstractCoeffField})
    saved_field = if haskey(obj, "coeff_field")
        _field_from_obj(obj["coeff_field"])
    elseif haskey(obj, "field")
        f = obj["field"]
        if f isa AbstractDict
            _field_from_obj(f)
        elseif f isa AbstractString
            sf = lowercase(String(f))
            if sf == "qq"
                QQField()
            elseif sf == "f2"
                PrimeField(2)
            elseif sf == "f3"
                PrimeField(3)
            else
                QQField()
            end
        else
            QQField()
        end
    else
        QQField()
    end
    target_field = field_override === nothing ? saved_field : field_override
    return saved_field, target_field
end

function _external_parse_phi(phi_any,
                             saved_field::AbstractCoeffField,
                             target_field::AbstractCoeffField,
                             m_expected::Int,
                             k_expected::Int)
    if phi_any isa AbstractDict && haskey(phi_any, "kind")
        kind = String(phi_any["kind"])
        if kind == "qq_chunks_v1"
            phi_obj = _PhiQQChunksJSON(kind=kind,
                                       m=Int(phi_any["m"]),
                                       k=Int(phi_any["k"]),
                                       base=Int(phi_any["base"]),
                                       num_sign=Vector{Int8}(Int8.(phi_any["num_sign"])),
                                       num_ptr=Vector{Int}(phi_any["num_ptr"]),
                                       num_chunks=Vector{UInt32}(UInt32.(phi_any["num_chunks"])),
                                       den_ptr=Vector{Int}(phi_any["den_ptr"]),
                                       den_chunks=Vector{UInt32}(UInt32.(phi_any["den_chunks"])))
            return _decode_phi(phi_obj, saved_field, target_field, m_expected, k_expected)
        elseif kind == "fp_flat_v1"
            phi_obj = _PhiFpFlatJSON(kind=kind,
                                     m=Int(phi_any["m"]),
                                     k=Int(phi_any["k"]),
                                     p=Int(phi_any["p"]),
                                     data=Vector{Int}(phi_any["data"]))
            return _decode_phi(phi_obj, saved_field, target_field, m_expected, k_expected)
        elseif kind == "real_flat_v1"
            phi_obj = _PhiRealFlatJSON(kind=kind,
                                       m=Int(phi_any["m"]),
                                       k=Int(phi_any["k"]),
                                       T=String(phi_any["T"]),
                                       data=Vector{Float64}(phi_any["data"]))
            return _decode_phi(phi_obj, saved_field, target_field, m_expected, k_expected)
        end
        error("Unsupported external phi kind: $(kind)")
    end

    phi_any isa AbstractVector || error("phi must be a matrix encoded as row vectors.")
    length(phi_any) == m_expected || error("phi row count mismatch (expected $(m_expected), got $(length(phi_any))).")
    K = coeff_type(target_field)
    Phi = Matrix{K}(undef, m_expected, k_expected)
    @inbounds for i in 1:m_expected
        row = phi_any[i]
        row isa AbstractVector || error("phi[$(i)] must be a row vector.")
        length(row) == k_expected || error("phi column count mismatch at row $(i) (expected $(k_expected), got $(length(row))).")
        for j in 1:k_expected
            s = _scalar_from_json(saved_field, row[j])
            Phi[i, j] = coerce(target_field, s)
        end
    end
    return Phi
end

"""
    parse_finite_fringe_json(json_src; field=nothing, validation=:strict) -> FringeModule

Best-effort parser for external finite-fringe JSON.

Supported top-level forms:
- canonical encoding form with `poset`, `U`/`D`, `phi`
- external aliases with `poset`, `upsets`/`downsets`, `phi`
- minimal dense-poset form with top-level `n`, `leq`, plus masks and `phi`
"""
function parse_finite_fringe_json(json_src;
                                  field::Union{Nothing,AbstractCoeffField}=nothing,
                                  validation::Symbol=:strict)
    obj = JSON3.read(json_src)
    validate_masks = _resolve_validation_mode(validation)

    poset_any = _external_get_any(obj, ("poset",))
    if poset_any === nothing
        haskey(obj, "n") || error("parse_finite_fringe_json: missing poset (expected `poset` or top-level `n`+`leq`).")
        haskey(obj, "leq") || error("parse_finite_fringe_json: missing top-level `leq` for dense-poset fallback.")
        poset_any = Dict("kind" => "FinitePoset", "n" => obj["n"], "leq" => obj["leq"])
    end
    P = _parse_poset_from_obj(poset_any)
    n = nvertices(P)

    U_any = _external_get_any(obj, ("U", "upsets", "ups"))
    D_any = _external_get_any(obj, ("D", "downsets", "downs"))
    U_any === nothing && error("parse_finite_fringe_json: missing upset masks (`U` or `upsets`).")
    D_any === nothing && error("parse_finite_fringe_json: missing downset masks (`D` or `downsets`).")

    Umasks = _external_parse_mask_rows(U_any, "U", n)
    Dmasks = _external_parse_mask_rows(D_any, "D", n)
    U = _build_upsets(P, Umasks, validate_masks)
    D = _build_downsets(P, Dmasks, validate_masks)

    haskey(obj, "phi") || error("parse_finite_fringe_json: missing `phi` matrix.")
    saved_field, target_field = _external_parse_field(obj, field)
    Phi = _external_parse_phi(obj["phi"], saved_field, target_field, length(D), length(U))
    K = coeff_type(target_field)
    return FiniteFringe.FringeModule{K}(P, U, D, Phi; field=target_field)
end

"""
    parse_pl_fringe_json(json_src) -> PLPolyhedra.PLFringe

Best-effort parser for external PL fringe JSON.

Accepted top-level keys:
- `n` (required)
- `ups` or `upsets` (required): vector of upset generators
- `downs` or `downsets` (required): vector of downset generators
- `phi` (required unless one axis is empty)

Each generator can be encoded either as:
- `{ "parts": [ { "A": [...], "b": [...], ... }, ... ] }`, or
- a single-part shorthand `{ "A": [...], "b": [...], ... }`.
"""
function parse_pl_fringe_json(json_src)
    obj = JSON3.read(json_src)
    return _parse_pl_fringe_obj(obj; strict_schema=false)
end

"""
    finite_fringe_from_m2(cmd::Cmd; jsonpath=nothing, field=nothing, validation=:strict)
        -> FringeModule

Run a CAS command that prints (or writes) finite-fringe JSON accepted by
`parse_finite_fringe_json`, then parse it.
"""
function finite_fringe_from_m2(cmd::Cmd; jsonpath::Union{Nothing,String}=nothing,
                               field::Union{Nothing,AbstractCoeffField}=nothing,
                               validation::Symbol=:strict)
    if jsonpath === nothing
        io = read(cmd, String)
        return parse_finite_fringe_json(io; field=field, validation=validation)
    end
    run(cmd)
    open(jsonpath, "r") do io
        return parse_finite_fringe_json(io; field=field, validation=validation)
    end
end

"""
JSON schema expected from an external CAS (Macaulay2, Singular, ...):

{
  "n": 3,                                   // ambient dimension
  "coeff_field": { "kind": "qq" },          // optional; defaults to QQ
  "flats": [
     {"b":[0,0,0], "tau":[true,false,false], "id":"F1"},
     {"b":[2,1,0], "tau":[false,false,true], "id":"F2"}
  ],
  "injectives": [
     {"b":[1,3,5], "tau":[true,false,false], "id":"E1"},
     {"b":[4,4,0], "tau":[false,true,false], "id":"E2"}
  ],
  // Optional: monomial matrix rows=#injectives, cols=#flats
  "phi": [[1,0],
          [0,1]]
}

Notes:
* `tau` denotes a face of N^n. We accept either a Bool vector or a list of indices.
* Scalars in `phi` are interpreted in QQ (exact rationals).
* If `phi` is omitted, we fall back to `canonical_matrix(flats, injectives)`.
"""
function parse_flange_json(json_src; field::Union{Nothing,AbstractCoeffField}=nothing)
    obj = JSON3.read(json_src)
    n = Int(obj["n"])
    saved_field = if haskey(obj, "coeff_field")
        _field_from_obj(obj["coeff_field"])
    elseif haskey(obj, "field")
        String(obj["field"]) == "QQ" ? QQField() : QQField()
    else
        QQField()
    end
    target_field = field === nothing ? saved_field : field
    K = coeff_type(target_field)

    function _mkface(n::Int, tau_any)
        if tau_any isa AbstractVector{Bool}
            return Face(n, BitVector(tau_any))
        end
        bits = falses(n)
        for t in tau_any
            bits[Int(t)] = true
        end
        return Face(n, bits)
    end

    flats = IndFlat{n}[]
    for f in obj["flats"]
        b = Vector{Int}(f["b"])
        tau = _mkface(n, f["tau"])
        id = Symbol(String(get(f, "id", "F")))
        push!(flats, IndFlat(tau, b; id=id))
    end

    injectives = IndInj{n}[]
    for e in obj["injectives"]
        b = Vector{Int}(e["b"])
        tau = _mkface(n, e["tau"])
        id = Symbol(String(get(e, "id", "E")))
        push!(injectives, IndInj(tau, b; id=id))
    end

    Phi = if haskey(obj, "phi")
        A = obj["phi"]
        m = length(injectives)
        ncol = length(flats)
        M = zeros(K, m, ncol)
        nonintegral_numeric_entries = 0
        @assert length(A) == m "phi: wrong number of rows"
        for i in 1:m
            row = A[i]
            @assert length(row) == ncol "phi: wrong number of cols"
            for j in 1:ncol
                val = row[j]
                if val isa String
                    M[i, j] = _scalar_from_json(saved_field, val)
                elseif val isa Integer
                    M[i, j] = _scalar_from_json(saved_field, val)
                else
                    nonintegral_numeric_entries += 1
                    M[i, j] = _scalar_from_json(saved_field, val)
                end
                if target_field !== saved_field
                    M[i, j] = coerce(target_field, M[i, j])
                end
            end
        end
        if nonintegral_numeric_entries > 0
            @warn "phi has $(nonintegral_numeric_entries) non-integer numeric entries; prefer exact strings \"num/den\" for exactness"
        end
        M
    else
        canonical_matrix(flats, injectives; field=target_field)
    end

    return Flange{K}(n, flats, injectives, Phi; field=target_field)
end

"""
    flange_from_m2(cmd::Cmd; jsonpath=nothing) -> Flange{QQ}

Run a CAS command that prints (or writes) the JSON described in `parse_flange_json`,
then parse it into a `Flange{QQ}`.
"""
function flange_from_m2(cmd::Cmd; jsonpath::Union{Nothing,String}=nothing,
                        field::Union{Nothing,AbstractCoeffField}=nothing)
    if jsonpath === nothing
        io = read(cmd, String)
        return parse_flange_json(io; field=field)
    end
    run(cmd)
    open(jsonpath, "r") do io
        return parse_flange_json(io; field=field)
    end
end

# =============================================================================
# C) Invariant caches (MPPI)
# =============================================================================

# MPPI types live in `PosetModules.Invariants`. We intentionally do NOT import
# them here to avoid include-order constraints. Instead, we fetch the module
# lazily when the MPPI JSON functions are called.

@inline function _invariants_module()
    PM = parentmodule(@__MODULE__)
    isdefined(PM, :Invariants) || error("MPPI JSON: PosetModules.Invariants is not loaded.")
    return getfield(PM, :Invariants)
end

function _mpp_floatvec2(x)::Vector{Float64}
    length(x) == 2 || error("MPPI JSON: expected a length-2 vector")
    return Float64[Float64(x[1]), Float64(x[2])]
end

function _mpp_decomposition_to_dict(decomp)
    lines = Vector{Any}(undef, length(decomp.lines))
    for (i, ls) in enumerate(decomp.lines)
        lines[i] = Dict(
            "dir" => ls.dir,
            "off" => ls.off,
            "x0" => ls.x0,
            "omega" => ls.omega,
        )
    end

    summands = Vector{Any}(undef, length(decomp.summands))
    for k in 1:length(decomp.summands)
        segs = decomp.summands[k]
        arr = Vector{Any}(undef, length(segs))
        for j in 1:length(segs)
            (p, q, om) = segs[j]
            arr[j] = Dict("p" => p, "q" => q, "omega" => om)
        end
        summands[k] = arr
    end

    lo, hi = decomp.box

    return Dict(
        "kind" => "MPPDecomposition",
        "version" => 1,
        "lines" => lines,
        "summands" => summands,
        "weights" => decomp.weights,
        "box" => Dict("lo" => lo, "hi" => hi),
    )
end

function _mpp_decomposition_from_dict(obj)
    if !haskey(obj, "kind") || String(obj["kind"]) != "MPPDecomposition"
        error("MPPI JSON: expected kind == 'MPPDecomposition'")
    end

    Inv = _invariants_module()
    LineSpec = getfield(Inv, :MPPLineSpec)
    Decomp = getfield(Inv, :MPPDecomposition)

    lines_obj = obj["lines"]
    lines = Vector{LineSpec}(undef, length(lines_obj))
    for (i, l) in enumerate(lines_obj)
        dir = _mpp_floatvec2(l["dir"])
        off = Float64(l["off"])
        x0 = _mpp_floatvec2(l["x0"])
        omega = Float64(l["omega"])
        lines[i] = LineSpec(dir, off, x0, omega)
    end

    summands_obj = obj["summands"]
    summands = Vector{Vector{Tuple{Vector{Float64},Vector{Float64},Float64}}}(undef, length(summands_obj))
    for (k, s) in enumerate(summands_obj)
        segs = Vector{Tuple{Vector{Float64},Vector{Float64},Float64}}(undef, length(s))
        for (j, seg) in enumerate(s)
            p = _mpp_floatvec2(seg["p"])
            q = _mpp_floatvec2(seg["q"])
            om = Float64(seg["omega"])
            segs[j] = (p, q, om)
        end
        summands[k] = segs
    end

    weights_obj = obj["weights"]
    weights = Float64[Float64(w) for w in weights_obj]

    box_obj = obj["box"]
    lo = _mpp_floatvec2(box_obj["lo"])
    hi = _mpp_floatvec2(box_obj["hi"])

    return Decomp(lines, summands, weights, (lo, hi))
end

function _mpp_image_to_dict(img; include_decomp::Bool=true)
    ny, nx = size(img.img)
    mat = Vector{Any}(undef, ny)
    for i in 1:ny
        mat[i] = [img.img[i, j] for j in 1:nx]
    end

    d = Dict(
        "kind" => "MPPImage",
        "version" => 1,
        "sigma" => img.sigma,
        "xgrid" => img.xgrid,
        "ygrid" => img.ygrid,
        "img" => mat,
    )

    if include_decomp
        d["decomp"] = _mpp_decomposition_to_dict(img.decomp)
    end

    return d
end

function _mpp_image_from_dict(obj)
    if !haskey(obj, "kind") || String(obj["kind"]) != "MPPImage"
        error("MPPI JSON: expected kind == 'MPPImage'")
    end

    Inv = _invariants_module()
    Image = getfield(Inv, :MPPImage)

    sig = Float64(obj["sigma"])
    xgrid = Float64[Float64(x) for x in obj["xgrid"]]
    ygrid = Float64[Float64(y) for y in obj["ygrid"]]

    rows = obj["img"]
    length(rows) == length(ygrid) || error("MPPI JSON: img row count does not match ygrid")
    imgmat = zeros(Float64, length(ygrid), length(xgrid))
    for i in 1:length(ygrid)
        row = rows[i]
        length(row) == length(xgrid) || error("MPPI JSON: img column count does not match xgrid")
        for j in 1:length(xgrid)
            imgmat[i, j] = Float64(row[j])
        end
    end

    haskey(obj, "decomp") || error("MPPI JSON: missing field 'decomp' (cannot reconstruct MPPImage without it)")
    decomp = _mpp_decomposition_from_dict(obj["decomp"])

    return Image(xgrid, ygrid, imgmat, sig, decomp)
end

"""
    save_mpp_decomposition_json(path, decomp)

Save an `MPPDecomposition` to a JSON file.

This is a good cache point: the decomposition contains the slice tracks and weights,
but not the full image grid. After loading, evaluate images via `mpp_image(decomp; ...)`.

Returns `path`.
"""
function save_mpp_decomposition_json(path::AbstractString, decomp)
    obj = _mpp_decomposition_to_dict(decomp)
    return _json_write(path, obj)
end

"""
    load_mpp_decomposition_json(path)

Load an `MPPDecomposition` written by `save_mpp_decomposition_json`.
"""
function load_mpp_decomposition_json(path::AbstractString)
    obj = _json_read(path)
    return _mpp_decomposition_from_dict(obj)
end

"""
    save_mpp_image_json(path, img; include_decomp=true)

Save an `MPPImage` (including its decomposition by default) to a JSON file.

Returns `path`.
"""
function save_mpp_image_json(path::AbstractString, img; include_decomp::Bool=true)
    obj = _mpp_image_to_dict(img; include_decomp=include_decomp)
    return _json_write(path, obj)
end

"""
    load_mpp_image_json(path)

Load an `MPPImage` written by `save_mpp_image_json`.

Note: `load_mpp_image_json` requires that the JSON contains a `"decomp"` field.
"""
function load_mpp_image_json(path::AbstractString)
    obj = _json_read(path)
    return _mpp_image_from_dict(obj)
end

end # module Serialization
