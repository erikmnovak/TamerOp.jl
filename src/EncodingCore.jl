# =============================================================================
# EncodingCore.jl
#
# Shared encoding-map abstractions and compiled/grid encoding helpers used by
# Zn, PL, workflow, and invariant layers.
# =============================================================================
module EncodingCore

using ..CoreModules: EncodingCache, SessionCache, _encoding_cache!, _WORKFLOW_ENCODING_CACHE_KEY

"""
    AbstractPLikeEncodingMap

Abstract supertype for encoding-map objects used throughout the finite-encoding
and PL backends.
"""
abstract type AbstractPLikeEncodingMap end

const PLikeEncodingMap = AbstractPLikeEncodingMap

"""
    CompiledEncoding(P, pi; axes=nothing, reps=nothing, meta=NamedTuple())

Primary wrapper for encoding maps. Stores the finite encoding poset `P` and
cached metadata used by hot query paths.
"""
struct CompiledEncoding{PiType,PType,AxesType,RepsType,MetaType} <: AbstractPLikeEncodingMap
    P::PType
    pi::PiType
    axes::AxesType
    reps::RepsType
    meta::MetaType
end

function compile_encoding(P, pi; axes=nothing, reps=nothing, meta=NamedTuple())
    axes_val = axes === nothing ? _axes_or_nothing(pi) : axes
    reps_val = reps === nothing ? _reps_or_nothing(pi) : reps
    return CompiledEncoding(P, pi, axes_val, reps_val, meta)
end

compile_encoding(::Any, ::Nothing; kwargs...) = nothing

function Base.getproperty(enc::CompiledEncoding, name::Symbol)
    if name === :P || name === :pi || name === :axes || name === :reps || name === :meta
        return getfield(enc, name)
    end
    return getproperty(getfield(enc, :pi), name)
end

function Base.propertynames(enc::CompiledEncoding, private::Bool=false)
    return (fieldnames(typeof(enc))..., propertynames(enc.pi, private)...)
end

"""
    locate(pi, x)

Generic classifier hook for finite encodings.
"""
function locate end
function locate_many! end
function locate_many end

const _ENCODINGCORE_CACHED_LOCATE_STYLE = Ref(true)
const _LOCATE_STYLE_LOCK = Base.ReentrantLock()
const _LOCATE_STYLE_CACHE = Dict{Tuple{DataType,Bool,Bool,Bool},UInt8}()

const _LOCATE_STYLE_PLAIN = UInt8(0)
const _LOCATE_STYLE_STRICT_ONLY = UInt8(1)
const _LOCATE_STYLE_STRICT_CLOSURE = UInt8(2)

@inline _geometry_fingerprint(pi) = UInt(objectid(pi))
@inline _geometry_fingerprint(enc::CompiledEncoding) = _geometry_fingerprint(enc.pi)
@inline _locate_style_owner(pi) = typeof(pi)
@inline _locate_style_owner(enc::CompiledEncoding) = _locate_style_owner(enc.pi)

@inline function _locate_style_tag(style::UInt8)
    style == _LOCATE_STYLE_STRICT_CLOSURE && return Val(:strict_closure)
    style == _LOCATE_STYLE_STRICT_ONLY && return Val(:strict_only)
    return Val(:plain)
end

function _probe_locate_style(pi, x0::AbstractVector{<:Real}; strict::Bool, closure::Bool)
    try
        locate(pi, x0; strict=strict, closure=closure)
        return _LOCATE_STYLE_STRICT_CLOSURE
    catch err
        err isa MethodError || rethrow()
    end
    try
        locate(pi, x0; strict=strict)
        return _LOCATE_STYLE_STRICT_ONLY
    catch err
        err isa MethodError || rethrow()
    end
    locate(pi, x0)
    return _LOCATE_STYLE_PLAIN
end

function _probe_locate_many_style(pi, x0::AbstractVector{<:Real}; strict::Bool, closure::Bool)
    probe = Matrix{Float64}(undef, length(x0), 1)
    @inbounds for i in eachindex(x0)
        probe[i, 1] = float(x0[i])
    end
    dest = Vector{Int}(undef, 1)
    try
        locate_many!(dest, pi, probe; strict=strict, closure=closure)
        return _LOCATE_STYLE_STRICT_CLOSURE
    catch err
        err isa MethodError || rethrow()
    end
    try
        locate_many!(dest, pi, probe; strict=strict)
        return _LOCATE_STYLE_STRICT_ONLY
    catch err
        err isa MethodError || rethrow()
    end
    locate_many!(dest, pi, probe)
    return _LOCATE_STYLE_PLAIN
end

function _locate_call_style(pi, x0::AbstractVector{<:Real}; strict::Bool, closure::Bool, batched::Bool=false)
    if !_ENCODINGCORE_CACHED_LOCATE_STYLE[]
        style = batched ?
            _probe_locate_many_style(pi, x0; strict=strict, closure=closure) :
            _probe_locate_style(pi, x0; strict=strict, closure=closure)
        return _locate_style_tag(style)
    end
    key = (_locate_style_owner(pi), strict, closure, batched)
    Base.lock(_LOCATE_STYLE_LOCK)
    try
        cached = get(_LOCATE_STYLE_CACHE, key, typemin(UInt8))
        if cached != typemin(UInt8)
            return _locate_style_tag(cached)
        end
    finally
        Base.unlock(_LOCATE_STYLE_LOCK)
    end
    style = batched ?
        _probe_locate_many_style(pi, x0; strict=strict, closure=closure) :
        _probe_locate_style(pi, x0; strict=strict, closure=closure)
    Base.lock(_LOCATE_STYLE_LOCK)
    try
        _LOCATE_STYLE_CACHE[key] = style
    finally
        Base.unlock(_LOCATE_STYLE_LOCK)
    end
    return _locate_style_tag(style)
end

function _clear_locate_style_cache!()
    Base.lock(_LOCATE_STYLE_LOCK)
    try
        empty!(_LOCATE_STYLE_CACHE)
    finally
        Base.unlock(_LOCATE_STYLE_LOCK)
    end
    return nothing
end

function locate(pi::AbstractPLikeEncodingMap, x::NTuple{N,<:Real}) where {N}
    throw(MethodError(locate, (pi, x)))
end

function locate(pi::AbstractPLikeEncodingMap, x::NTuple{N,<:Real}; kwargs...) where {N}
    throw(ArgumentError("locate($(typeof(pi)), NTuple; kwargs...) is not implemented. Define an explicit tuple locate method for this encoding map type."))
end

locate(enc::CompiledEncoding, x) = locate(enc.pi, x)
locate(enc::CompiledEncoding, x::NTuple{N,<:Real}) where {N} = locate(enc.pi, x)
locate(enc::CompiledEncoding, x::AbstractVector) = locate(enc.pi, x)
locate(enc::CompiledEncoding, x; kwargs...) = locate(enc.pi, x; kwargs...)
locate(enc::CompiledEncoding, x::NTuple{N,<:Real}; kwargs...) where {N} = locate(enc.pi, x; kwargs...)
locate(enc::CompiledEncoding, x::AbstractVector; kwargs...) = locate(enc.pi, x; kwargs...)

function locate_many!(dest::AbstractVector{<:Integer}, pi::AbstractPLikeEncodingMap,
                      X::AbstractMatrix{<:Real}; kwargs...)
    length(dest) == size(X, 2) || error("locate_many!: destination length mismatch")
    @inbounds for j in 1:size(X, 2)
        dest[j] = locate(pi, view(X, :, j); kwargs...)
    end
    return dest
end

function locate_many(pi::AbstractPLikeEncodingMap, X::AbstractMatrix{<:Real}; kwargs...)
    out = Vector{Int}(undef, size(X, 2))
    return locate_many!(out, pi, X; kwargs...)
end

locate_many!(dest::AbstractVector{<:Integer}, enc::CompiledEncoding, X::AbstractMatrix{<:Real}; kwargs...) =
    locate_many!(dest, enc.pi, X; kwargs...)
locate_many(enc::CompiledEncoding, X::AbstractMatrix{<:Real}; kwargs...) =
    locate_many(enc.pi, X; kwargs...)

"""
    dimension(pi) -> Int

Ambient parameter-space dimension of an encoding map.
"""
function dimension end

dimension(enc::CompiledEncoding) = dimension(enc.pi)

"""
    representatives(pi)

Return a finite collection of representative points for the regions of `pi`.
"""
function representatives end

@inline representatives(::AbstractPLikeEncodingMap) = nothing
@inline _reps_or_nothing(::Any) = nothing
@inline _reps_or_nothing(pi::AbstractPLikeEncodingMap) = representatives(pi)

function representatives(enc::CompiledEncoding)
    enc.reps === nothing ? representatives(enc.pi) : enc.reps
end

"""
    axes_from_encoding(pi)

Return coordinate axes derived from the encoding when available.
"""
function axes_from_encoding end

@inline axes_from_encoding(::AbstractPLikeEncodingMap) = nothing
@inline _axes_or_nothing(::Any) = nothing
@inline _axes_or_nothing(pi::AbstractPLikeEncodingMap) = axes_from_encoding(pi)

function axes_from_encoding(enc::CompiledEncoding)
    enc.axes === nothing ? axes_from_encoding(enc.pi) : enc.axes
end

@inline function _compile_encoding_cached(P, pi, session_cache::Union{Nothing,SessionCache})
    if session_cache === nothing
        ec = EncodingCache()
        return compile_encoding(P, pi; meta=ec)
    end
    ec = _encoding_cache!(session_cache, _WORKFLOW_ENCODING_CACHE_KEY)
    return compile_encoding(P, pi; meta=ec)
end

"""
    GridEncodingMap(P, coords; orientation=ntuple(_->1, N))

Axis-aligned grid encoding map for a product-style grid of coordinates.
"""
struct GridEncodingMap{N,T,P} <: AbstractPLikeEncodingMap
    P::P
    coords::NTuple{N,Vector{T}}
    orientation::NTuple{N,Int}
    sizes::NTuple{N,Int}
    strides::NTuple{N,Int}
end

function _grid_strides(sizes::NTuple{N,Int}) where {N}
    strides = Vector{Int}(undef, N)
    strides[1] = 1
    for i in 2:N
        strides[i] = strides[i - 1] * sizes[i - 1]
    end
    return ntuple(i -> strides[i], N)
end

function GridEncodingMap(P, coords::NTuple{N,Vector{T}};
                         orientation::NTuple{N,Int}=ntuple(_ -> 1, N)) where {N,T}
    sizes = ntuple(i -> length(coords[i]), N)
    for i in 1:N
        o = orientation[i]
        (o == 1 || o == -1) || error("GridEncodingMap: orientation[$i] must be +1 or -1.")
    end
    return GridEncodingMap{N,T,typeof(P)}(P, coords, orientation, sizes, _grid_strides(sizes))
end

dimension(pi::GridEncodingMap{N}) where {N} = N
axes_from_encoding(pi::GridEncodingMap) = pi.coords

function locate(pi::GridEncodingMap{N,T}, x::AbstractVector{<:Real}) where {N,T}
    length(x) == N || error("GridEncodingMap.locate: expected vector length $(N), got $(length(x)).")
    lin = 1
    for i in 1:N
        xi = pi.orientation[i] == 1 ? x[i] : -x[i]
        if xi == 0
            xi = zero(xi)
        end
        idx = searchsortedlast(pi.coords[i], xi)
        if idx < 1
            return 0
        end
        lin += (idx - 1) * pi.strides[i]
    end
    return lin
end

function locate(pi::GridEncodingMap{N,T}, x::NTuple{N,<:Real}) where {N,T}
    lin = 1
    @inbounds for i in 1:N
        xi = pi.orientation[i] == 1 ? x[i] : -x[i]
        if xi == 0
            xi = zero(xi)
        end
        idx = searchsortedlast(pi.coords[i], xi)
        if idx < 1
            return 0
        end
        lin += (idx - 1) * pi.strides[i]
    end
    return lin
end

function locate_many!(dest::AbstractVector{<:Integer}, pi::GridEncodingMap{N,T},
                      X::AbstractMatrix{<:Real}; kwargs...) where {N,T}
    size(X, 1) == N || error("locate_many!: expected X with $(N) rows, got $(size(X, 1))")
    length(dest) == size(X, 2) || error("locate_many!: destination length mismatch")
    _ = kwargs
    @inbounds for j in 1:size(X, 2)
        lin = 1
        for i in 1:N
            xi = pi.orientation[i] == 1 ? X[i, j] : -X[i, j]
            if xi == 0
                xi = zero(xi)
            end
            idx = searchsortedlast(pi.coords[i], xi)
            if idx < 1
                lin = 0
                break
            end
            lin += (idx - 1) * pi.strides[i]
        end
        dest[j] = lin
    end
    return dest
end

function representatives(pi::GridEncodingMap{N,T}) where {N,T}
    total = prod(pi.sizes)
    reps = Vector{NTuple{N,T}}(undef, total)
    idxs = ones(Int, N)
    for lin in 1:total
        reps[lin] = ntuple(i -> pi.coords[i][idxs[i]], N)
        for i in 1:N
            idxs[i] += 1
            if idxs[i] <= pi.sizes[i]
                break
            else
                idxs[i] = 1
            end
        end
    end
    return reps
end

end # module EncodingCore
