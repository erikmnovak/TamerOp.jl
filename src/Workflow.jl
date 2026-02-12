# =============================================================================
# Workflow.jl
#
# User-facing workflow orchestration plus lightweight ingestion types.
#
# PosetModules.jl is meant to be an "API map" file (includes + re-exports).
# This file holds the glue code that makes the public workflow smooth and
# predictable, and now also hosts lightweight ingestion types plus a small
# grid encoding map (so users can build pipelines without extra modules):
#
#   presentation  ->  encode(...)  ->  EncodingResult
#                                ->  resolve(...) -> ResolutionResult
#                                ->  invariants(...) -> InvariantResult
#                                ->  ext(...) / tor(...) etc.
#
# Implementation details remain in the submodules:
#   - ZnEncoding.jl (Z^n encoding)
#   - PLBackend.jl / PLPolyhedra.jl (R^n encoding backends)
#   - DerivedFunctors.jl (Ext/Tor + resolutions)
#   - Invariants.jl
#
# File structure:
#   Section 1: Workflow orchestration + public entrypoints
#   Section 2: Lightweight ingestion types (PointCloud, FiltrationSpec, ...)
#   Section 3: GridEncodingMap helpers
#   Section 4: Internal helpers (data ingestion pipeline)
# =============================================================================

module Workflow

using LinearAlgebra  # UniformScaling I / transpose
using SparseArrays
using JSON3
using Dates

# Keep this file self-contained: import the handful of names it mentions
# in type annotations or uses unqualified.
using ..CoreModules: QQ, QQField, AbstractCoeffField, coeff_type, coerce,
                    EncodingOptions, ResolutionOptions, DerivedFunctorOptions, InvariantOptions,
                    ResolutionCache, SessionCache, EncodingCache,
                    encoding_cache!, session_resolution_cache, session_hom_cache, set_session_hom_cache!,
                    session_slice_plan_cache, set_session_slice_plan_cache!,
                    EncodingResult, ResolutionResult, InvariantResult,
                    change_field, AbstractPLikeEncodingMap,
                    CompiledEncoding, compile_encoding,
                    PointCloud, ImageNd, GraphData, EmbeddedPlanarGraph2D,
                    GradedComplex, FiltrationSpec, GridEncodingMap, grid_index
import ..CoreModules: encode_from_data, ingest,
                     locate, dimension, representatives, axes_from_encoding, _grid_strides,
                     GridEncodingMap
import ..Serialization
import ..Serialization: TAMER_FEATURE_SCHEMA_VERSION, feature_schema_header, validate_feature_metadata_schema

import ..IndicatorResolutions
import ..ZnEncoding
using ..IndicatorResolutions: pmodule_from_fringe
using ..FlangeZn: Flange
using ..PLPolyhedra
using ..PLBackend: BoxUpset, BoxDownset, encode_fringe_boxes
using ..Encoding: build_uptight_encoding_from_fringe,
                 pushforward_fringe_along_encoding,
                 PostcomposedEncodingMap
using ..Modules: PModule, PMorphism, cover_edges, dim_at
using ..FiniteFringe: AbstractPoset, FinitePoset, GridPoset, ProductOfChainsPoset, nvertices
using ..DerivedFunctors
using ..Invariants
import ..Modules
import ..ModuleComplexes


# -----------------------------------------------------------------------------
# Section 3: GridEncodingMap helpers
# -----------------------------------------------------------------------------

"""
    poset_from_axes(axes; orientation=ntuple(_->1, N), kind=:grid) -> AbstractPoset

Build the product-of-chains poset on a grid defined by `axes`.
`axes` is an N-tuple of sorted coordinate vectors. `orientation[i]` is +1 for
sublevel-style order (increasing) and -1 for superlevel-style order (decreasing).

Notes
-----
- If all `orientation[i] == 1` and `kind=:grid`, a structured `ProductOfChainsPoset`
  is returned to avoid materializing the transitive closure.
- Otherwise, we fall back to a dense `FinitePoset`.
"""
function poset_from_axes(axes::NTuple{N,Vector{T}};
                         orientation::NTuple{N,Int}=ntuple(_ -> 1, N),
                         kind::Symbol = :grid) where {N,T}
    sizes = ntuple(i -> length(axes[i]), N)
    total = 1
    for i in 1:N
        total *= sizes[i]
    end

    if kind == :grid
        if all(o -> o == 1, orientation)
            return ProductOfChainsPoset(sizes)
        end
        kind = :dense
    end
    if kind == :dense
        # Enumerate all multi-indices in mixed radix order.
        idxs = Vector{NTuple{N,Int}}(undef, total)
        cur = ones(Int, N)
        for lin in 1:total
            idxs[lin] = ntuple(i -> cur[i], N)
            for i in 1:N
                cur[i] += 1
                if cur[i] <= sizes[i]
                    break
                else
                    cur[i] = 1
                end
            end
        end

        leq = falses(total, total)
        for i in 1:total
            ai = idxs[i]
            for j in 1:total
                bj = idxs[j]
                ok = true
                for k in 1:N
                    if orientation[k] == 1
                        if ai[k] > bj[k]
                            ok = false
                            break
                        end
                    else
                        if ai[k] < bj[k]
                            ok = false
                            break
                        end
                    end
                end
                leq[i, j] = ok
            end
        end

        return FinitePoset(leq; check=false)
    else
        error("poset_from_axes: kind must be :grid or :dense")
    end
end

function GridEncodingMap(P::AbstractPoset, coords::NTuple{N,Vector{T}};
                         orientation::NTuple{N,Int}=ntuple(_ -> 1, N)) where {N,T}
    sizes = ntuple(i -> length(coords[i]), N)
    total = 1
    for i in 1:N
        total *= sizes[i]
    end
    total == nvertices(P) || error("GridEncodingMap: grid size $(total) does not match nvertices(P)=$(nvertices(P)).")
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

function representatives(pi::GridEncodingMap{N,T}) where {N,T}
    # Cartesian product of coordinate axes (grid points).
    reps = Vector{NTuple{N,T}}(undef, nvertices(pi.P))
    idxs = ones(Int, N)
    for lin in 1:nvertices(pi.P)
        reps[lin] = ntuple(i -> pi.coords[i][idxs[i]], N)
        # advance mixed radix counter
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
using ..ModuleComplexes: ModuleCochainComplex, cohomology_module

using ..IndicatorResolutions: projective_cover
using ..FiniteFringe: AbstractPoset, FringeModule, Upset, Downset, principal_upset, principal_downset,
                     leq, nvertices, poset_equal_opposite
using ..FlangeZn: Face, IndFlat, IndInj, Flange

const _WORKFLOW_ENCODING_CACHE_KEY = typemax(UInt)

@inline function _workflow_encoding_cache(session_cache::Union{Nothing,SessionCache})
    session_cache === nothing && return nothing
    return encoding_cache!(session_cache, _WORKFLOW_ENCODING_CACHE_KEY)
end

@inline function _resolve_workflow_session_cache(cache)
    if cache === :auto
        return SessionCache()
    elseif cache === nothing
        return nothing
    elseif cache isa SessionCache
        return cache
    end
    throw(ArgumentError("cache must be :auto, nothing, or SessionCache"))
end

@inline function _resolve_workflow_specialized_cache(cache, ::Type{T}) where {T}
    # Public workflow contract stays simple: only :auto|nothing|SessionCache.
    # Specialized caches are derived from the resolved SessionCache internally.
    return nothing, _resolve_workflow_session_cache(cache)
end

@inline function _compile_encoding_cached(P, pi, session_cache::Union{Nothing,SessionCache})
    if session_cache === nothing
        # Keep a per-encoding cache even without a SessionCache so repeated
        # geometry/invariant queries on one EncodingResult can reuse artifacts.
        ec = EncodingCache()
        return compile_encoding(P, pi; meta=(encoding_cache=ec,))
    end
    ec = encoding_cache!(session_cache, P)
    return compile_encoding(P, pi; meta=(encoding_cache=ec))
end

@inline function _encoding_with_session_cache(enc::EncodingResult,
                                              session_cache::Union{Nothing,SessionCache})
    session_cache === nothing && return enc
    raw_pi = enc.pi isa CompiledEncoding ? enc.pi.pi : enc.pi
    pi2 = _compile_encoding_cached(enc.P, raw_pi, session_cache)
    return EncodingResult(enc.P, enc.M, pi2;
                          H=enc.H,
                          presentation=enc.presentation,
                          opts=enc.opts,
                          backend=enc.backend,
                          meta=enc.meta)
end

@inline function _resolution_cache_from_session(cache::Union{Nothing,ResolutionCache},
                                                session_cache::Union{Nothing,SessionCache},
                                                M=nothing)
    cache !== nothing && return cache
    session_cache === nothing && return nothing
    return M === nothing ? session_resolution_cache(session_cache) :
                           session_resolution_cache(session_cache, M)
end

@inline function _hom_cache_from_session(cache::Union{Nothing,DerivedFunctors.HomSystemCache},
                                         session_cache::Union{Nothing,SessionCache},
                                         ::Type{K}) where {K}
    cache !== nothing && return cache
    session_cache === nothing && return nothing
    hs = session_hom_cache(session_cache)
    if !(hs isa DerivedFunctors.HomSystemCache{DerivedFunctors.HomSpace{K},SparseMatrixCSC{K,Int},SparseMatrixCSC{K,Int}})
        hs = DerivedFunctors.HomSystemCache{K}()
        set_session_hom_cache!(session_cache, hs)
    end
    return hs
end

@inline function _slice_plan_cache_from_session(cache::Union{Nothing,Invariants.SlicePlanCache},
                                                session_cache::Union{Nothing,SessionCache})
    cache !== nothing && return cache
    session_cache === nothing && return nothing
    sp = session_slice_plan_cache(session_cache)
    if !(sp isa Invariants.SlicePlanCache)
        sp = Invariants.SlicePlanCache()
        set_session_slice_plan_cache!(session_cache, sp)
    end
    return sp
end

# -----------------------------------------------------------------------------
# Backend selection for R^n encoding (PLBackend vs PLPolyhedra)

# Workflow entrypoints designed to:
# - accept input presentations,
# - call lower-level encoding / resolution / derived-functor / invariant routines,
# - return small results objects with provenance:
#
#   presentation -> encode(...)    -> EncodingResult
#               -> resolve(...)   -> ResolutionResult
#               -> invariant(...) -> InvariantResult
#
# Derived-functor entrypoints (`hom`, `ext`, `tor`) return graded objects
# queried via the GradedSpaces interface (degree_range, dim, basis, coordinates, representative).
#
# A future goal is that these are essentially the only names exported by default.

"""
    has_polyhedra_backend()::Bool

Return true if the optional Polyhedra-based backend is available at runtime.
"""
has_polyhedra_backend()::Bool = PLPolyhedra.HAVE_POLY

has_polyhedra_backend()::Bool = PLPolyhedra.HAVE_POLY

"""
    available_pl_backends()::Vector{Symbol}

Report which R^n encoding backends are available in this session.
Always includes :pl_backend (because PLBackend is always loaded).
Includes :pl if Polyhedra/CDDLib are available.
"""
function available_pl_backends()::Vector{Symbol}
    out = Symbol[:pl_backend]
    if has_polyhedra_backend()
        push!(out, :pl)
    end
    return out
end

# Internal: normalize user backend selectors for R^n
# Canonical symbols:
#   :pl_backend  (PLBackend, axis-aligned box partition)
#   :pl          (PLPolyhedra / Polyhedra backend)
function _normalize_pl_backend(b::Symbol)::Symbol
    b == :auto && return :auto
    b == :pl && return :pl
    (b == :pl_backend || b == :pl_backend_boxes || b == :boxes || b == :axis) && return :pl_backend
    error("Unknown R^n encoding backend symbol: $(b). Try :auto, :pl_backend, or :pl.")
end

# Internal: infer ambient dimension from box generators.
function _infer_box_dim(Ups::Vector{BoxUpset}, Downs::Vector{BoxDownset})
    if !isempty(Ups)
        return length(Ups[1].ell)
    elseif !isempty(Downs)
        return length(Downs[1].u)
    else
        error("Cannot infer dimension: both Ups and Downs are empty.")
    end
end

"""
    supports_pl_backend(Ups::Vector{BoxUpset}, Downs::Vector{BoxDownset}; opts=EncodingOptions())::Bool

Return true iff PLBackend can encode this box presentation under the given options.
This checks:
- axis-aligned boxes with finite coordinates,
- region count does not exceed opts.max_regions if set.
"""
function supports_pl_backend(Ups::Vector{BoxUpset}, Downs::Vector{BoxDownset};
                             opts::EncodingOptions=EncodingOptions())::Bool
    opts = opts
    max_regions = opts.max_regions
    # PLBackend can only handle finite, axis-aligned boxes (encoded as BoxUpset/BoxDownset).
    # If any generator has +/-Inf bounds, or other invalid bounds, reject here.
    for U in Ups
        for a in U.ell
            isfinite(a) || return false
        end
    end
    for D in Downs
        for b in D.u
            isfinite(b) || return false
        end
    end

    # If max_regions is set, do a cheap upper bound check on the grid size.
    # PLBackend partitions via coordinate grid from generator endpoints.
    if max_regions !== nothing
        n = _infer_box_dim(Ups, Downs)
        coords = [QQ[] for _ in 1:n]
        for U in Ups
            for i in 1:n
                push!(coords[i], U.ell[i])
            end
        end
        for D in Downs
            for i in 1:n
                push!(coords[i], D.u[i])
            end
        end
        # unique, sorted
        for i in 1:n
            coords[i] = sort!(unique!(coords[i]))
        end
        # number of cells in the induced grid:
        # product over dimensions of (k_i - 1).
        cells = 1
        for i in 1:n
            ki = length(coords[i])
            if ki < 2
                return false
            end
            cells *= (ki - 1)
            if cells > max_regions
                return false
            end
        end
    end

    return true
end

"""
    supports_pl_backend(F::PLPolyhedra.PLFringe; opts=EncodingOptions())::Bool

Return true iff the PLFringe can be converted to a box presentation and encoded by PLBackend
under the given options.
"""
function supports_pl_backend(F::PLPolyhedra.PLFringe; opts::EncodingOptions=EncodingOptions())::Bool
    opts = opts
    try
        Ups, Downs = boxes_from_pl_fringe(F)
        return supports_pl_backend(Ups, Downs; opts=opts)
    catch
        return false
    end
end

# -----------------------------------------------------------------------------
# Conversions between PLFringe and box generators for PLBackend

"""
    boxes_from_pl_fringe(F::PLPolyhedra.PLFringe) -> (Ups::Vector{BoxUpset}, Downs::Vector{BoxDownset})

Convert an axis-aligned PLFringe (as used by PLPolyhedra) into box generators for PLBackend.

Throws if the fringe contains non-axis-aligned generators or is otherwise incompatible.
"""
function boxes_from_pl_fringe(F::PLPolyhedra.PLFringe)
    Ups = Vector{BoxUpset}(undef, length(F.Ups))
    for i in eachindex(F.Ups)
        U = F.Ups[i]
        if length(U.A) != 2 * F.n
            error("PLFringe upset is not axis-aligned (A has wrong size).")
        end
        lo = Vector{QQ}(undef, F.n)
        hi = Vector{QQ}(undef, F.n)
        for j in 1:F.n
            lo[j] = U.b[j]
            hi[j] = U.b[F.n + j]
        end
        Ups[i] = BoxUpset(lo, hi)
    end

    Downs = Vector{BoxDownset}(undef, length(F.Downs))
    for i in eachindex(F.Downs)
        D = F.Downs[i]
        if length(D.A) != 2 * F.n
            error("PLFringe downset is not axis-aligned (A has wrong size).")
        end
        lo = Vector{QQ}(undef, F.n)
        hi = Vector{QQ}(undef, F.n)
        for j in 1:F.n
            lo[j] = D.b[j]
            hi[j] = D.b[F.n + j]
        end
        Downs[i] = BoxDownset(lo, hi)
    end

    return Ups, Downs
end

"""
    pl_fringe_from_boxes(Ups::Vector{BoxUpset}, Downs::Vector{BoxDownset}) -> PLPolyhedra.PLFringe

Convert a box presentation into a PLFringe (axis-aligned) for use with PLPolyhedra.
"""
function pl_fringe_from_boxes(Ups::Vector{BoxUpset}, Downs::Vector{BoxDownset})
    n = _infer_box_dim(Ups, Downs)

    PUps = Vector{PLPolyhedra.PLUpset}(undef, length(Ups))
    for i in eachindex(Ups)
        ell = Ups[i].ell
        A = -Matrix{QQ}(I, n, n)
        b = -ell
        PUps[i] = PLPolyhedra.PLUpset(A, b)
    end

    PDowns = Vector{PLPolyhedra.PLDownset}(undef, length(Downs))
    for i in eachindex(Downs)
        u = Downs[i].u
        A = Matrix{QQ}(I, n, n)
        b = u
        PDowns[i] = PLPolyhedra.PLDownset(A, b)
    end

    return PLPolyhedra.PLFringe(n, PUps, PDowns)
end

# -----------------------------------------------------------------------------
# Backend chooser and encoding wrappers for R^n

"""
    choose_pl_backend(...; opts=EncodingOptions())::Symbol

Choose which R^n encoding backend to use.
Returns :pl_backend or :pl.

Rules:
- If opts.backend is explicitly set (e.g. :pl_backend or :pl), try to honor it.
- If opts.backend=:auto, prefer :pl_backend when supported; otherwise fall back to :pl.
- If :pl is requested but Polyhedra backend is unavailable, throw.
"""
function choose_pl_backend(Ups::Vector{BoxUpset}, Downs::Vector{BoxDownset};
                           opts::EncodingOptions=EncodingOptions())::Symbol
    opts = opts
    b = _normalize_pl_backend(opts.backend)
    if b == :pl_backend
        supports_pl_backend(Ups, Downs; opts=opts) || error("PLBackend requested, but this input is not supported by PLBackend.")
        return :pl_backend
    elseif b == :pl
        has_polyhedra_backend() || error("PLPolyhedra backend requested, but Polyhedra/CDDLib is not available.")
        return :pl
    elseif b == :auto
        if supports_pl_backend(Ups, Downs; opts=opts)
            return :pl_backend
        else
            has_polyhedra_backend() || error("PLBackend not applicable and Polyhedra backend is unavailable.")
            return :pl
        end
    else
        error("Internal error: unexpected normalized backend $(b).")
    end
end

function choose_pl_backend(F::PLPolyhedra.PLFringe; opts::EncodingOptions=EncodingOptions())::Symbol
    opts = opts
    b = _normalize_pl_backend(opts.backend)
    if b == :pl_backend
        supports_pl_backend(F; opts=opts) || error("PLBackend requested, but this PLFringe is not supported by PLBackend.")
        return :pl_backend
    elseif b == :pl
        has_polyhedra_backend() || error("PLPolyhedra backend requested, but Polyhedra/CDDLib is not available.")
        return :pl
    elseif b == :auto
        if supports_pl_backend(F; opts=opts)
            return :pl_backend
        else
            has_polyhedra_backend() || error("PLBackend not applicable and Polyhedra backend is unavailable.")
            return :pl
        end
    else
        error("Internal error: unexpected normalized backend $(b).")
    end
end

"""
    encode_from_fringe(...)

Unified R^n encoder:
- If :pl_backend is chosen, uses PLBackend.encode_fringe_boxes(...)
- If :pl is chosen, uses PLPolyhedra.encode_from_PL_fringe(...)
"""
function encode_from_fringe(Ups::Vector{BoxUpset}, Downs::Vector{BoxDownset}, Phi::AbstractMatrix,
                            opts::EncodingOptions=EncodingOptions())
    opts = opts
    b = choose_pl_backend(Ups, Downs; opts=opts)
    Phi_QQ = Matrix{QQ}(Phi)
    if b == :pl_backend
        P, H, pi = encode_fringe_boxes(Ups, Downs, Phi_QQ, opts; poset_kind=opts.poset_kind)
    else
        F = pl_fringe_from_boxes(Ups, Downs)
        P, H, pi = PLPolyhedra.encode_from_PL_fringe(F, opts; poset_kind=opts.poset_kind)
    end
    H = (opts.field == QQField()) ? H : change_field(H, opts.field)
    return P, H, pi
end

function encode_from_fringe(F::PLPolyhedra.PLFringe, opts::EncodingOptions=EncodingOptions())
    opts = opts
    b = choose_pl_backend(F; opts=opts)
    if b == :pl_backend
        Ups, Downs = boxes_from_pl_fringe(F)
        Phi = reshape(ones(QQ, length(Downs) * length(Ups)), length(Downs), length(Ups))
        P, H, pi = encode_fringe_boxes(Ups, Downs, Phi, opts; poset_kind=opts.poset_kind)
    else
        P, H, pi = PLPolyhedra.encode_from_PL_fringe(F, opts; poset_kind=opts.poset_kind)
    end
    H = (opts.field == QQField()) ? H : change_field(H, opts.field)
    return P, H, pi
end

"""
    encode_pmodule_from_fringe(...)

Return (M, pi) where M is the finite-poset PModule obtained from the encoded fringe module.
"""
function encode_pmodule_from_fringe(Ups::Vector{BoxUpset}, Downs::Vector{BoxDownset}, Phi::AbstractMatrix,
                                    opts::EncodingOptions=EncodingOptions())
    opts = opts
    P, H, pi = encode_from_fringe(Ups, Downs, Phi, opts)
    return pmodule_from_fringe(H), pi
end

function encode_pmodule_from_fringe(F::PLPolyhedra.PLFringe, opts::EncodingOptions=EncodingOptions())
    opts = opts
    P, H, pi = encode_from_fringe(F, opts)
    return pmodule_from_fringe(H), pi
end

encode_from_fringe(Ups::Vector{BoxUpset}, Downs::Vector{BoxDownset}, Phi::AbstractMatrix;
                   opts::EncodingOptions=EncodingOptions()) =
    encode_from_fringe(Ups, Downs, Phi, opts)

encode_from_fringe(F::PLPolyhedra.PLFringe;
                   opts::EncodingOptions=EncodingOptions()) =
    encode_from_fringe(F, opts)

encode_pmodule_from_fringe(Ups::Vector{BoxUpset}, Downs::Vector{BoxDownset}, Phi::AbstractMatrix;
                           opts::EncodingOptions=EncodingOptions()) =
    encode_pmodule_from_fringe(Ups, Downs, Phi, opts)

encode_pmodule_from_fringe(F::PLPolyhedra.PLFringe;
                           opts::EncodingOptions=EncodingOptions()) =
    encode_pmodule_from_fringe(F, opts)

@inline function _return_encoding(res::EncodingResult, output::Symbol)
    if output === :result
        return res
    elseif output === :raw
        return (res.P, res.H, res.pi)
    end
    error("output must be :result or :raw (got $output)")
end

# -----------------------------------------------------------------------------
# Workflow entrypoints (narrative API)

"""
    encode(x; backend=:auto, max_regions=nothing, strict_eps=nothing, poset_kind=:signature,
           output=:result) -> EncodingResult
    encode(x1, x2, ...; backend=:auto, ...) -> Vector{EncodingResult}

High-level entrypoint that turns a presentation object into a finite encoding poset model.

- Z^n inputs (Flange{K}) use the Zn encoder (backend=:auto or :zn).
- R^n inputs (PLFringe, or box generators) use:
    * PLBackend when possible (axis-aligned, finite bounds, and not too many regions),
    * otherwise PLPolyhedra (if Polyhedra/CDDLib are available).

The returned EncodingResult stores:
- P : the finite encoding poset
- H : the pushed-down fringe module on P (optional but populated here)
- M : the finite-poset module pmodule_from_fringe(H)
- pi: the classifier map from the original domain to P
- presentation / opts / backend / meta : provenance

Set `output=:raw` to return `(P, H, pi)` instead of an EncodingResult.

Notes
-----
* For multiple PL fringes, only the PLPolyhedra backend currently supports common-encoding.
  If you need common-encoding and you asked for PLBackend explicitly, we throw an error.
"""
# -----------------------------------------------------------------------------
# Section 1: Workflow orchestration + public entrypoints
# -----------------------------------------------------------------------------

function encode(x; backend::Symbol=:auto, max_regions=nothing, strict_eps=nothing,
                poset_kind::Symbol=:signature, field::AbstractCoeffField=QQField(),
                output::Symbol=:result,
                cache=:auto)
    session_cache = _resolve_workflow_session_cache(cache)
    enc = EncodingOptions(; backend=backend, max_regions=max_regions,
                          strict_eps=strict_eps, poset_kind=poset_kind, field=field)
    return encode(x, enc; output=output, cache=session_cache)
end

# -----------------------
# Z^n presentations

function encode(FG::Flange{K}, enc::EncodingOptions;
                output::Symbol=:result,
                cache=:auto) where {K}
    session_cache = _resolve_workflow_session_cache(cache)
    if enc.backend != :auto && enc.backend != :zn
        error("encode(Flange): EncodingOptions.backend must be :auto or :zn (got $(enc.backend))")
    end
    FG2 = (FG.field == enc.field) ? FG : change_field(FG, enc.field)
    P, H, pi = ZnEncoding.encode_from_flange(FG2, enc)
    M = pmodule_from_fringe(H)
    pi2 = _compile_encoding_cached(P, pi, session_cache)
    res = EncodingResult(P, M, pi2; H=H, presentation=FG2, opts=enc, backend=:zn, meta=(;))
    return _return_encoding(res, output)
end

function encode(FG::Flange{K};
                backend::Symbol=:auto,
                max_regions=nothing,
                strict_eps=nothing,
                poset_kind::Symbol=:signature,
                field::Union{AbstractCoeffField,Nothing}=nothing,
                output::Symbol=:result,
                cache=:auto) where {K}
    session_cache = _resolve_workflow_session_cache(cache)
    field === nothing && (field = FG.field)
    enc2 = EncodingOptions(; backend=backend, max_regions=max_regions,
                           strict_eps=strict_eps, poset_kind=poset_kind, field=field)
    return encode(FG, enc2; output=output, cache=session_cache)
end

function encode(FGs::Union{AbstractVector{<:Flange}, Tuple{Vararg{Flange}}},
                enc::EncodingOptions;
                output::Symbol=:result,
                cache=:auto)
    session_cache = _resolve_workflow_session_cache(cache)
    if enc.backend != :auto && enc.backend != :zn
        error("encode(Vector{Flange}): EncodingOptions.backend must be :auto or :zn (got $(enc.backend))")
    end
    K2 = coeff_type(enc.field)
    FGs2 = Vector{Flange{K2}}(undef, length(FGs))
    @inbounds for i in eachindex(FGs)
        FGi = FGs[i]
        FGs2[i] = (FGi.field == enc.field) ? FGi : change_field(FGi, enc.field)
    end
    P, Hs, pi = ZnEncoding.encode_from_flanges(FGs2, enc)
    out = Vector{EncodingResult}(undef, length(FGs))
    pi2 = _compile_encoding_cached(P, pi, session_cache)
    for i in eachindex(FGs2)
        H = Hs[i]
        out[i] = EncodingResult(P, pmodule_from_fringe(H), pi2;
                                H=H, presentation=FGs2[i], opts=enc, backend=:zn, meta=(;))
    end
    return output === :result ? out : [_return_encoding(enc_i, output) for enc_i in out]
end

function encode(FGs::Union{AbstractVector{<:Flange}, Tuple{Vararg{Flange}}};
                backend::Symbol=:auto,
                max_regions=nothing,
                strict_eps=nothing,
                poset_kind::Symbol=:signature,
                field::Union{AbstractCoeffField,Nothing}=nothing,
                output::Symbol=:result,
                cache=:auto)
    session_cache = _resolve_workflow_session_cache(cache)
    field === nothing && (field = FGs[1].field)
    enc2 = EncodingOptions(; backend=backend, max_regions=max_regions,
                           strict_eps=strict_eps, poset_kind=poset_kind, field=field)
    return encode(FGs, enc2; output=output, cache=session_cache)
end

encode(FG1::Flange, FG2::Flange; kwargs...) =
    encode((FG1, FG2); kwargs...)

encode(FG1::Flange, FG2::Flange, FG3::Flange; kwargs...) =
    encode((FG1, FG2, FG3); kwargs...)

# -----------------------
# R^n presentations

function encode(F::PLPolyhedra.PLFringe, enc::EncodingOptions=EncodingOptions();
                output::Symbol=:result,
                cache=:auto)
    session_cache = _resolve_workflow_session_cache(cache)
    P, H, pi = encode_from_fringe(F, enc)
    M = pmodule_from_fringe(H)
    b = choose_pl_backend(F; opts=enc)
    pi2 = _compile_encoding_cached(P, pi, session_cache)
    res = EncodingResult(P, M, pi2; H=H, presentation=F, opts=enc, backend=b, meta=(;))
    return _return_encoding(res, output)
end

function encode(Ups::Vector{BoxUpset}, Downs::Vector{BoxDownset}, Phi::AbstractMatrix,
                enc::EncodingOptions=EncodingOptions();
                output::Symbol=:result,
                cache=:auto)
    session_cache = _resolve_workflow_session_cache(cache)
    P, H, pi = encode_from_fringe(Ups, Downs, Phi, enc)
    M = pmodule_from_fringe(H)
    b = choose_pl_backend(Ups, Downs; opts=enc)
    pi2 = _compile_encoding_cached(P, pi, session_cache)
    res = EncodingResult(P, M, pi2;
                         H=H, presentation=(Ups=Ups, Downs=Downs, Phi=Phi),
                         opts=enc, backend=b, meta=(;))
    return _return_encoding(res, output)
end

# Convenience overloads for common BoxFringe encodings
encode(Ups::Vector{BoxUpset}, Downs::Vector{BoxDownset}, enc::EncodingOptions=EncodingOptions();
       output::Symbol=:result,
       cache=:auto) =
    encode(Ups, Downs,
           reshape(ones(QQ, length(Downs) * length(Ups)), length(Downs), length(Ups)),
           enc; output=output, cache=cache)

function encode(Ups::Vector{BoxUpset}, Downs::Vector{BoxDownset};
                backend::Symbol=:auto, max_regions=nothing, strict_eps=nothing,
                poset_kind::Symbol=:signature, field::AbstractCoeffField=QQField(),
                output::Symbol=:result,
                cache=:auto)
    enc = EncodingOptions(; backend=backend, max_regions=max_regions,
                          strict_eps=strict_eps, poset_kind=poset_kind, field=field)
    return encode(Ups, Downs, enc; output=output, cache=cache)
end

function encode(Ups::Vector{BoxUpset}, Downs::Vector{BoxDownset}, Phi_vec::AbstractVector,
                enc::EncodingOptions=EncodingOptions();
                output::Symbol=:result,
                cache=:auto)
    Phi = reshape(Phi_vec, length(Downs), length(Ups))
    return encode(Ups, Downs, Phi, enc; output=output, cache=cache)
end

function encode(Fs::AbstractVector{<:PLPolyhedra.PLFringe}, enc::EncodingOptions;
                output::Symbol=:result,
                cache=:auto)
    session_cache = _resolve_workflow_session_cache(cache)
    # Today, only the PLPolyhedra backend supports common-encoding multiple PL fringes.
    if _normalize_pl_backend(enc.backend) == :pl_backend
        error("encode(Vector{PLFringe}): common encoding for PLBackend is not implemented; use backend=:pl or backend=:auto.")
    end
    if enc.backend != :auto && enc.backend != :pl
        error("encode(Vector{PLFringe}): EncodingOptions.backend must be :auto or :pl (got $(enc.backend))")
    end
    P, Hs, pi = PLPolyhedra.encode_from_PL_fringes(Fs, enc; poset_kind=enc.poset_kind)
    out = Vector{EncodingResult}(undef, length(Fs))
    pi2 = _compile_encoding_cached(P, pi, session_cache)
    for i in eachindex(Fs)
        H = Hs[i]
        out[i] = EncodingResult(P, pmodule_from_fringe(H), pi2;
                                H=H, presentation=Fs[i], opts=enc, backend=:pl, meta=(;))
    end
    return output === :result ? out : [_return_encoding(enc_i, output) for enc_i in out]
end

encode(Fs::AbstractVector{<:PLPolyhedra.PLFringe};
       enc::EncodingOptions=EncodingOptions(),
       output::Symbol=:result,
       cache=:auto) =
    encode(Fs, enc;
           output=output,
           cache=cache)

encode(F1::PLPolyhedra.PLFringe, F2::PLPolyhedra.PLFringe; kwargs...) =
    encode(PLPolyhedra.PLFringe[F1, F2]; kwargs...)

encode(F1::PLPolyhedra.PLFringe, F2::PLPolyhedra.PLFringe, F3::PLPolyhedra.PLFringe; kwargs...) =
    encode(PLPolyhedra.PLFringe[F1, F2, F3]; kwargs...)

# -----------------------------------------------------------------------------
# Section 4: Internal helpers (data ingestion pipeline)
# -----------------------------------------------------------------------------

@inline function _tuple_leq(a::NTuple{N,Int}, b::NTuple{N,Int},
                            orientation::NTuple{N,Int}) where {N}
    @inbounds for i in 1:N
        if orientation[i] == 1
            if a[i] > b[i]
                return false
            end
        else
            if a[i] < b[i]
                return false
            end
        end
    end
    return true
end

function _grid_tuples(sizes::NTuple{N,Int}) where {N}
    total = 1
    for i in 1:N
        total *= sizes[i]
    end
    out = Vector{NTuple{N,Int}}(undef, total)
    cur = ones(Int, N)
    for lin in 1:total
        out[lin] = ntuple(i -> cur[i], N)
        for i in 1:N
            cur[i] += 1
            if cur[i] <= sizes[i]
                break
            else
                cur[i] = 1
            end
        end
    end
    return out
end

function _axes_from_grades(grades::Vector{<:NTuple{N,T}}, n::Int;
                           orientation::Union{Nothing,NTuple{N,Int}}=nothing) where {N,T}
    n == N || error("grade length $N does not match n=$n")
    orient = orientation === nothing ? ntuple(_ -> 1, N) : orientation
    for i in 1:N
        (orient[i] == 1 || orient[i] == -1) || error("orientation[$i] must be +1 or -1.")
    end

    RT = promote_type(T, Int)
    axes = [RT[] for _ in 1:n]
    for g in grades
        length(g) == n || error("grade has length $(length(g)) but expected $n")
        for i in 1:n
            val = orient[i] == 1 ? RT(g[i]) : -RT(g[i])
            push!(axes[i], val)
        end
    end
    for i in 1:n
        axes[i] = sort!(unique!(axes[i]))
    end
    return ntuple(i -> axes[i], n)
end

function _validate_axes_sorted(axes)
    for (i, ax) in enumerate(axes)
        if !issorted(ax)
            error("encode_from_data: axis $i must be sorted ascending.")
        end
    end
    return nothing
end

function _validate_axes_kind(axes; axis_kind=nothing)
    # axis_kind may be :zn / :rn to force integer vs real axes.
    is_int_typed_axis(ax) = all(x -> x isa Integer, ax)
    is_int_valued_axis(ax) = all(x -> (x isa Integer) || (x isa Real && isinteger(x)), ax)

    if axis_kind === nothing
        return nothing
    end

    axis_kind in (:zn, :rn) || error("encode_from_data: axis_kind must be :zn or :rn.")
    if axis_kind == :zn
        for (i, ax) in enumerate(axes)
            if !is_int_typed_axis(ax)
                error("encode_from_data: axis $i must be integer-typed for axis_kind=:zn.")
            end
        end
    else
        for (i, ax) in enumerate(axes)
            if is_int_typed_axis(ax)
                error("encode_from_data: axis $i is integer-typed but axis_kind=:rn.")
            end
        end
    end
    return nothing
end

function _axes_key(axes)
    return tuple((Tuple(ax) for ax in axes)...)
end

function _coarsen_axis(ax::Vector{T}, max_len::Int) where {T}
    n = length(ax)
    n <= max_len && return ax
    idxs = round.(Int, range(1, n; length=max_len))
    return ax[idxs]
end

function _coarsen_axes(axes::NTuple{N,Vector{T}}, max_len::Int) where {N,T}
    return ntuple(i -> _coarsen_axis(axes[i], max_len), N)
end

function _quantize_grades(G::GradedComplex, eps)
    N = length(G.grades[1])
    eps_vec = if eps isa Number
        ntuple(_ -> Float64(eps), N)
    elseif eps isa Tuple
        length(eps) == N || error("encode_from_data: eps length mismatch.")
        ntuple(i -> Float64(eps[i]), N)
    else
        error("encode_from_data: eps must be a number or tuple.")
    end
    grades = Vector{NTuple{N,Float64}}(undef, length(G.grades))
    for i in eachindex(G.grades)
        g = G.grades[i]
        grades[i] = ntuple(j -> round(Float64(g[j]) / eps_vec[j]) * eps_vec[j], N)
    end
    return GradedComplex(G.cells_by_dim, G.boundaries, grades; cell_dims=G.cell_dims)
end

function _poset_from_axes_cached(axes, orientation;
                                 cache::Union{Nothing,EncodingCache}=nothing)
    key = (_axes_key(axes), orientation)
    if cache === nothing
        return poset_from_axes(axes; orientation=orientation, kind=:grid)
    end
    Base.lock(cache.lock)
    try
        P = get(cache.posets, key, nothing)
        if P === nothing
            P = poset_from_axes(axes; orientation=orientation, kind=:grid)
            cache.posets[key] = P
        end
        return P
    finally
        Base.unlock(cache.lock)
    end
end

function _poset_vertex_coords(axes::NTuple{N,Vector{T}}) where {N,T}
    sizes = ntuple(i -> length(axes[i]), N)
    idxs = _grid_tuples(sizes)
    coords = Vector{Vector{Int}}(undef, length(idxs))
    for i in eachindex(idxs)
        coords[i] = [Int(axes[j][idxs[i][j]]) for j in 1:N]
    end
    return coords
end

function _grades_by_dim(G::GradedComplex)
    counts = [length(c) for c in G.cells_by_dim]
    total = sum(counts)
    length(G.grades) == total ||
        error("GradedComplex.grades length $(length(G.grades)) does not match total cells $(total).")
    out = Vector{Vector{typeof(G.grades[1])}}(undef, length(counts))
    idx = 1
    for d in 1:length(counts)
        out[d] = Vector{typeof(G.grades[1])}(undef, counts[d])
        for j in 1:counts[d]
            out[d][j] = G.grades[idx]
            idx += 1
        end
    end
    return out
end

function _birth_indices(grades::Vector{<:NTuple{N,<:Real}},
                        axes::NTuple{N,Vector{<:Real}},
                        orientation::NTuple{N,Int}) where {N}
    births = Vector{NTuple{N,Int}}(undef, length(grades))
    for i in eachindex(grades)
        g = grades[i]
        length(g) == N || error("grade length $(length(g)) does not match N=$(N)")
        births[i] = ntuple(k -> begin
            val = orientation[k] == 1 ? g[k] : -g[k]
            idx = searchsortedlast(axes[k], val)
            idx >= 1 || error("grade component $(g[k]) falls below axis minimum for axis $k")
            idx
        end, N)
    end
    return births
end

function _active_lists(births::Vector{NTuple{N,Int}},
                       vertex_idxs::Vector{NTuple{N,Int}},
                       orientation::NTuple{N,Int}) where {N}
    active = Vector{Vector{Int}}(undef, length(vertex_idxs))
    for i in eachindex(vertex_idxs)
        lst = Int[]
        vi = vertex_idxs[i]
        for c in eachindex(births)
            if _tuple_leq(births[c], vi, orientation)
                push!(lst, c)
            end
        end
        active[i] = lst
    end
    return active
end

function _pos_maps(active::Vector{Vector{Int}})
    maps = Vector{Dict{Int,Int}}(undef, length(active))
    for i in eachindex(active)
        d = Dict{Int,Int}()
        for (j, c) in enumerate(active[i])
            d[c] = j
        end
        maps[i] = d
    end
    return maps
end

function _pmodule_from_active_lists(P::AbstractPoset,
                                    active::Vector{Vector{Int}};
                                    pos_maps::Union{Nothing,Vector{Dict{Int,Int}}}=nothing,
                                    field::AbstractCoeffField=QQField())
    K = coeff_type(field)
    dims = [length(lst) for lst in active]
    edge_maps = Dict{Tuple{Int,Int}, Matrix{K}}()
    for (u, v) in cover_edges(P)
        Mu = active[u]
        Mv = active[v]
        pos_v = if pos_maps === nothing
            d = Dict{Int,Int}()
            for (i, c) in enumerate(Mv)
                d[c] = i
            end
            d
        else
            pos_maps[v]
        end
        A = zeros(K, length(Mv), length(Mu))
        for (j, c) in enumerate(Mu)
            i = get(pos_v, c, 0)
            if i != 0
                A[i, j] = one(K)
            end
        end
        edge_maps[(u, v)] = A
    end
    return PModule{K}(P, dims, edge_maps; field=field)
end

function cochain_complex_from_graded_complex(G::GradedComplex,
                                             P::AbstractPoset,
                                             axes::NTuple{N,Vector{T}};
                                             orientation::NTuple{N,Int}=ntuple(_ -> 1, N),
                                             field::AbstractCoeffField=QQField()) where {N,T}
    sizes = ntuple(i -> length(axes[i]), N)
    vertex_idxs = _grid_tuples(sizes)

    grades_by_dim = _grades_by_dim(G)
    births_by_dim = Vector{Vector{NTuple{N,Int}}}(undef, length(grades_by_dim))
    active_by_dim = Vector{Vector{Vector{Int}}}(undef, length(grades_by_dim))
    pos_by_dim = Vector{Vector{Dict{Int,Int}}}(undef, length(grades_by_dim))
    for d in 1:length(grades_by_dim)
        births_by_dim[d] = _birth_indices(grades_by_dim[d], axes, orientation)
        active_by_dim[d] = _active_lists(births_by_dim[d], vertex_idxs, orientation)
        pos_by_dim[d] = _pos_maps(active_by_dim[d])
    end

    K = coeff_type(field)
    terms = Vector{PModule{K}}(undef, length(grades_by_dim))
    for d in 1:length(grades_by_dim)
        terms[d] = _pmodule_from_active_lists(P, active_by_dim[d]; pos_maps=pos_by_dim[d], field=field)
    end

    diffs = PMorphism{K}[]
    expected = length(grades_by_dim) - 1
    boundaries = G.boundaries
    if length(boundaries) < expected
        for k in (length(boundaries) + 1):expected
            rows = length(grades_by_dim[k])
            cols = length(grades_by_dim[k + 1])
            push!(boundaries, spzeros(Int, rows, cols))
        end
    elseif length(boundaries) > expected
        error("GradedComplex.boundaries length mismatch (expected $(expected)).")
    end

    for k in 1:length(boundaries)
        B = boundaries[k]  # boundary C_{k+1} -> C_k
        Ii, Jj, Vv_int = findnz(B)
        Vv = Vector{K}(undef, length(Vv_int))
        @inbounds for t in eachindex(Vv_int)
            Vv[t] = coerce(field, Vv_int[t])
        end
        comps = Vector{Matrix{K}}(undef, nvertices(P))
        for i in 1:nvertices(P)
            Lk = active_by_dim[k][i]
            Lk1 = active_by_dim[k+1][i]
            rowmap = pos_by_dim[k][i]
            colmap = pos_by_dim[k+1][i]
            M = zeros(K, length(Lk1), length(Lk))
            @inbounds for t in eachindex(Ii)
                r = get(rowmap, Ii[t], 0)
                c = get(colmap, Jj[t], 0)
                if r != 0 && c != 0
                    M[c, r] = Vv[t]
                end
            end
            comps[i] = M
        end
        push!(diffs, PMorphism{K}(terms[k], terms[k+1], comps))
    end

    return ModuleCochainComplex(terms, diffs; tmin=0, check=true)
end

"""
    fringe_presentation(M::PModule{K}) -> FringeModule{K}

Construct a canonical fringe presentation whose image recovers `M`.
"""
function fringe_presentation(M::PModule{K}) where {K}
    P = M.Q
    field = M.field
    F0, pi0, gens_at_F0 = projective_cover(M)
    E0, iota0, gens_at_E0 = IndicatorResolutions._injective_hull(M)

    # Map F0 -> E0 whose image is M.
    comps = Vector{Matrix{K}}(undef, nvertices(P))
    for i in 1:nvertices(P)
        comps[i] = iota0.comps[i] * pi0.comps[i]
    end
    f = PMorphism{K}(F0, E0, comps)

    U = IndicatorResolutions._principal_upsets_from_gens(P, gens_at_F0)
    D = IndicatorResolutions._principal_downsets_from_gens(P, gens_at_E0)

    L0 = IndicatorResolutions._local_index_list_up(P, gens_at_F0)
    L1 = IndicatorResolutions._local_index_list_down(P, gens_at_E0)

    globalU = Tuple{Int,Int}[]
    for p in 1:nvertices(P)
        append!(globalU, gens_at_F0[p])
    end
    globalD = Tuple{Int,Int}[]
    for u in 1:nvertices(P)
        append!(globalD, gens_at_E0[u])
    end

    function local_pos(L, g::Tuple{Int,Int}, i::Int)
        for (j, gg) in enumerate(L[i])
            if gg == g
                return j
            end
        end
        return 0
    end

    phi = spzeros(K, length(D), length(U))
    for (lambda, (plambda, jlambda)) in enumerate(globalU)
        for (theta, (ptheta, jtheta)) in enumerate(globalD)
            if leq(P, plambda, ptheta)
                i = plambda
                col = local_pos(L0, (plambda, jlambda), i)
                row = local_pos(L1, (ptheta, jtheta), i)
                if col > 0 && row > 0
                    val = f.comps[i][row, col]
                    if val != 0
                        phi[theta, lambda] = val
                    end
                end
            end
        end
    end

    return FringeModule{K}(P, U, D, phi)
end

"""
    flange_presentation(M::PModule{K}, pi::GridEncodingMap) -> Flange{K}

Construct a Z^n flange presentation for an integer-valued grid encoding.
"""
function flange_presentation(M::PModule{K}, pi::GridEncodingMap) where {K}
    P = M.Q
    field = M.field
    n = length(pi.coords)
    for ax in pi.coords
        all(x -> x isa Integer || (x isa Real && isinteger(x)), ax) ||
            error("flange_presentation: axes must be integer-valued.")
    end
    F0, pi0, gens_at_F0 = projective_cover(M)
    E0, iota0, gens_at_E0 = IndicatorResolutions._injective_hull(M)

    comps = Vector{Matrix{K}}(undef, nvertices(P))
    for i in 1:nvertices(P)
        comps[i] = iota0.comps[i] * pi0.comps[i]
    end
    f = PMorphism{K}(F0, E0, comps)

    tau = Face(n, falses(n))
    coords = _poset_vertex_coords(pi.coords)

    flats = IndFlat{n}[]
    for p in 1:nvertices(P)
        for _ in gens_at_F0[p]
            push!(flats, IndFlat(tau, coords[p]; id=:F))
        end
    end

    injectives = IndInj{n}[]
    for u in 1:nvertices(P)
        for _ in gens_at_E0[u]
            push!(injectives, IndInj(tau, coords[u]; id=:E))
        end
    end

    L0 = IndicatorResolutions._local_index_list_up(P, gens_at_F0)
    L1 = IndicatorResolutions._local_index_list_down(P, gens_at_E0)

    globalU = Tuple{Int,Int}[]
    for p in 1:nvertices(P)
        append!(globalU, gens_at_F0[p])
    end
    globalD = Tuple{Int,Int}[]
    for u in 1:nvertices(P)
        append!(globalD, gens_at_E0[u])
    end

    function local_pos(L, g::Tuple{Int,Int}, i::Int)
        for (j, gg) in enumerate(L[i])
            if gg == g
                return j
            end
        end
        return 0
    end

    Phi = spzeros(K, length(injectives), length(flats))
    for (lambda, (plambda, jlambda)) in enumerate(globalU)
        for (theta, (ptheta, jtheta)) in enumerate(globalD)
            if leq(P, plambda, ptheta)
                i = plambda
                col = local_pos(L0, (plambda, jlambda), i)
                row = local_pos(L1, (ptheta, jtheta), i)
                if col > 0 && row > 0
                    val = f.comps[i][row, col]
                    if val != 0
                        Phi[theta, lambda] = val
                    end
                end
            end
        end
    end

    return Flange{K}(n, flats, injectives, Matrix{K}(Phi); field=field)
end

function _simplicial_boundary(simplices::Vector{Vector{Int}}, faces::Vector{Vector{Int}})
    K = isempty(simplices) ? length(faces[1]) + 1 : length(simplices[1])
    face_index = Dict{Tuple{Vararg{Int}},Int}()
    for (i, f) in enumerate(faces)
        face_index[Tuple(f)] = i
    end
    I = Int[]
    J = Int[]
    V = Int[]
    for (j, s) in enumerate(simplices)
        for i in 1:K
            f = [s[t] for t in 1:K if t != i]
            row = face_index[Tuple(f)]
            push!(I, row)
            push!(J, j)
            push!(V, isodd(i) ? 1 : -1)
        end
    end
    return sparse(I, J, V, length(faces), length(simplices))
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

function _filter_params(params::NamedTuple, drop::Vector{Symbol})
    return (; (k => v for (k, v) in pairs(params) if !(k in drop))...)
end

@inline function _euclidean_distance(p::AbstractVector{<:Real}, q::AbstractVector{<:Real})
    length(p) == length(q) || error("PointCloud: point dimension mismatch.")
    s = 0.0
    @inbounds for i in 1:length(p)
        d = Float64(p[i]) - Float64(q[i])
        s += d * d
    end
    return sqrt(s)
end

function _knn_distances(points::AbstractVector{<:AbstractVector{<:Real}}, k::Int)
    n = length(points)
    dists = Vector{Float64}(undef, n)
    for i in 1:n
        ds = Float64[]
        for j in 1:n
            if i == j
                continue
            end
            push!(ds, _euclidean_distance(points[i], points[j]))
        end
        sort!(ds)
        k <= length(ds) || error("kNN k=$k exceeds number of neighbors.")
        dists[i] = ds[k]
    end
    return dists
end

function _graded_complex_from_point_cloud(data::PointCloud, spec::FiltrationSpec)
    max_dim = get(spec.params, :max_dim, 1)
    kind = spec.kind
    points = data.points
    n = length(points)
    if n == 0
        error("PointCloud has no points.")
    end
    # pairwise distances
    dist = zeros(Float64, n, n)
    for i in 1:n
        for j in (i+1):n
            d = _euclidean_distance(points[i], points[j])
            dist[i, j] = d
            dist[j, i] = d
        end
    end
    # simplices by dimension
    simplices = Vector{Vector{Vector{Int}}}(undef, max_dim + 1)
    simplices[1] = [ [i] for i in 1:n ]
    total = length(simplices[1])
    max_simplices = get(spec.params, :max_simplices, nothing)

    if get(spec.params, :sparse_rips, false)
        radius = get(spec.params, :radius, nothing)
        radius === nothing && error("sparse_rips requires radius.")
        # 1-skeleton only in sparse mode for speed.
        sims = Vector{Vector{Int}}()
        for i in 1:n, j in i+1:n
            if dist[i, j] <= radius
                push!(sims, [i, j])
            end
        end
        simplices = [simplices[1], sims]
        max_dim = 1
        total += length(sims)
        if max_simplices !== nothing && total > max_simplices
            error("PointCloud Rips: exceeded max_simplices=$(max_simplices).")
        end
    elseif get(spec.params, :approx_rips, false)
        radius = get(spec.params, :radius, nothing)
        radius === nothing && error("approx_rips requires radius.")
        max_edges = get(spec.params, :max_edges, nothing)
        max_degree = get(spec.params, :max_degree, nothing)
        sample_frac = get(spec.params, :sample_frac, nothing)
        sims = Vector{Vector{Int}}()
        deg = zeros(Int, n)
        for i in 1:n
            for j in i+1:n
                if dist[i, j] <= radius
                    if max_degree !== nothing && (deg[i] >= max_degree || deg[j] >= max_degree)
                        continue
                    end
                    if sample_frac !== nothing && rand() > sample_frac
                        continue
                    end
                    push!(sims, [i, j])
                    deg[i] += 1
                    deg[j] += 1
                    if max_edges !== nothing && length(sims) >= max_edges
                        break
                    end
                end
            end
            if max_edges !== nothing && length(sims) >= max_edges
                break
            end
        end
        simplices = [simplices[1], sims]
        max_dim = 1
        total += length(sims)
        if max_simplices !== nothing && total > max_simplices
            error("PointCloud Rips: exceeded max_simplices=$(max_simplices).")
        end
    else
        for k in 2:max_dim+1
            if max_simplices !== nothing
                comb_count = binomial(big(n), big(k))
                if big(total) + comb_count > big(max_simplices)
                    error("PointCloud Rips: would exceed max_simplices=$(max_simplices) at simplex dimension $(k - 1).")
                end
            end
            sims = _combinations(n, k)
            simplices[k] = sims
            total += length(sims)
        end
    end

    if kind == :rips
        grades = Vector{NTuple{1,Float64}}()
        for s in simplices[1]
            push!(grades, (0.0,))
        end
        for k in 2:max_dim+1
            for s in simplices[k]
                maxd = 0.0
                for i in 1:length(s)
                    for j in (i+1):length(s)
                        d = dist[s[i], s[j]]
                        if d > maxd
                            maxd = d
                        end
                    end
                end
                push!(grades, (maxd,))
            end
        end
    elseif kind == :rips_density || kind == :rips_knn
        knn_k = get(spec.params, :density_k, 2)
        dens = _knn_distances(points, knn_k)
        grades = Vector{NTuple{2,Float64}}()
        for s in simplices[1]
            v = s[1]
            push!(grades, (0.0, dens[v]))
        end
        for k in 2:max_dim+1
            for s in simplices[k]
                maxd = 0.0
                maxden = 0.0
                for i in 1:length(s)
                    vi = s[i]
                    if dens[vi] > maxden
                        maxden = dens[vi]
                    end
                    for j in (i+1):length(s)
                        d = dist[s[i], s[j]]
                        if d > maxd
                            maxd = d
                        end
                    end
                end
                push!(grades, (maxd, maxden))
            end
        end
    elseif kind == :witness
        landmarks = get(spec.params, :landmarks, nothing)
        landmarks === nothing && error("witness requires landmarks.")
        length(landmarks) > 0 || error("witness: landmarks cannot be empty.")
        pts = [points[i] for i in landmarks]
        params2 = _filter_params(spec.params, [:landmarks, :sparse_rips, :radius])
        params2 = merge(params2, (max_dim = max_dim,))
        return _graded_complex_from_point_cloud(PointCloud(pts), FiltrationSpec(; kind=:rips, params2...))
    else
        error("Unsupported point cloud filtration kind: $(kind).")
    end

    boundaries = SparseMatrixCSC{Int,Int}[]
    for k in 2:max_dim+1
        Bk = _simplicial_boundary(simplices[k], simplices[k-1])
        push!(boundaries, Bk)
    end

    cells = [collect(1:length(simplices[k])) for k in 1:length(simplices)]
    G = GradedComplex(cells, boundaries, grades)
    N = length(grades[1])
    orientation = get(spec.params, :orientation, ntuple(_ -> 1, N))
    axes = get(spec.params, :axes, _axes_from_grades(grades, N; orientation=orientation))
    return G, axes, orientation
end

function _graded_complex_from_graph(data::GraphData, spec::FiltrationSpec)
    kind = spec.kind
    n = data.n
    edges = data.edges
    if kind == :graph_lower_star || kind == :clique_lower_star
        vertex_grades = get(spec.params, :vertex_grades, nothing)
        vertex_grades === nothing && error("graph filtration requires vertex_grades.")
        length(vertex_grades) == n || error("vertex_grades length mismatch.")
        N = length(vertex_grades[1])
        grades = Vector{NTuple{N,eltype(vertex_grades[1])}}()
        for g in vertex_grades
            length(g) == N || error("vertex_grades entries must have length N=$N")
            push!(grades, ntuple(i -> g[i], N))
        end

        if kind == :graph_lower_star
            simplices0 = [ [i] for i in 1:n ]
            simplices1 = [ [u, v] for (u, v) in edges ]
            for (u, v) in edges
                gu = vertex_grades[u]
                gv = vertex_grades[v]
                push!(grades, ntuple(i -> max(gu[i], gv[i]), N))
            end
            B1 = _simplicial_boundary(simplices1, simplices0)
            G = GradedComplex([collect(1:n), collect(1:length(edges))], [B1], grades)
            orientation = get(spec.params, :orientation, ntuple(_ -> 1, N))
            axes = get(spec.params, :axes, _axes_from_grades(grades, N; orientation=orientation))
            return G, axes, orientation
        else
            max_dim = get(spec.params, :max_dim, 2)
            adj = falses(n, n)
            for (u, v) in edges
                adj[u, v] = true
                adj[v, u] = true
            end
            simplices = Vector{Vector{Vector{Int}}}(undef, max_dim + 1)
            simplices[1] = [ [i] for i in 1:n ]
            for k in 2:max_dim+1
                sims = Vector{Vector{Int}}()
                for comb in _combinations(n, k)
                    ok = true
                    for i in 1:k
                        for j in (i+1):k
                            if !adj[comb[i], comb[j]]
                                ok = false
                                break
                            end
                        end
                        if !ok
                            break
                        end
                    end
                    if ok
                        push!(sims, comb)
                    end
                end
                simplices[k] = sims
            end
            for k in 2:max_dim+1
                for s in simplices[k]
                    push!(grades, ntuple(i -> maximum(vertex_grades[v][i] for v in s), N))
                end
            end
            boundaries = SparseMatrixCSC{Int,Int}[]
            for k in 2:max_dim+1
                push!(boundaries, _simplicial_boundary(simplices[k], simplices[k-1]))
            end
            cells = [collect(1:length(simplices[k])) for k in 1:length(simplices)]
            G = GradedComplex(cells, boundaries, grades)
            orientation = get(spec.params, :orientation, ntuple(_ -> 1, N))
            axes = get(spec.params, :axes, _axes_from_grades(grades, N; orientation=orientation))
            return G, axes, orientation
        end
    elseif kind == :edge_weighted
        weights = get(spec.params, :edge_weights, nothing)
        weights === nothing && error("edge_weighted requires edge_weights.")
        length(weights) == length(edges) || error("edge_weights length mismatch.")
        grades = Vector{NTuple{1,Float64}}()
        for _ in 1:n
            push!(grades, (0.0,))
        end
        for w in weights
            push!(grades, (Float64(w),))
        end
        simplices0 = [ [i] for i in 1:n ]
        simplices1 = [ [u, v] for (u, v) in edges ]
        B1 = _simplicial_boundary(simplices1, simplices0)
        G = GradedComplex([collect(1:n), collect(1:length(edges))], [B1], grades)
        orientation = get(spec.params, :orientation, (1,))
        axes = get(spec.params, :axes, _axes_from_grades(grades, 1; orientation=orientation))
        return G, axes, orientation
    else
        error("Unsupported graph filtration kind: $(kind).")
    end
end

function _distance_transform(mask::AbstractArray{Bool})
    dims = size(mask)
    N = length(dims)
    coords = CartesianIndices(mask)
    true_pts = [c for c in coords if mask[c]]
    out = zeros(Float64, dims)
    for c in coords
        if mask[c]
            out[c] = 0.0
            continue
        end
        best = Inf
        for t in true_pts
            s = 0.0
            for i in 1:N
                d = (c[i] - t[i])
                s += d * d
            end
            if s < best
                best = s
            end
        end
        out[c] = sqrt(best)
    end
    return out
end

function _cubical_structure(dims::NTuple{N,Int};
                            cache::Union{Nothing,EncodingCache}=nothing) where {N}
    key = Tuple(dims)
    if cache !== nothing
        Base.lock(cache.lock)
        try
            cached = get(cache.cubical, key, nothing)
            cached === nothing || return cached
        finally
            Base.unlock(cache.lock)
        end
    end
    cells_by_dim = Vector{Vector{Tuple{NTuple{N,Int},NTuple{N,Int}}}}(undef, N + 1)
    cell_index = Vector{Dict{Tuple{NTuple{N,Int},NTuple{N,Int}},Int}}(undef, N + 1)
    for k in 0:N
        cells_by_dim[k+1] = Tuple{NTuple{N,Int},NTuple{N,Int}}[]
        cell_index[k+1] = Dict{Tuple{NTuple{N,Int},NTuple{N,Int}},Int}()
    end

    for mask_vec in _combinations(N, 0)
        mask = ntuple(i -> 0, N)
        for coords in CartesianIndices(Tuple(dims))
            origin = ntuple(i -> coords[i], N)
            key2 = (origin, mask)
            push!(cells_by_dim[1], key2)
        end
    end
    for k in 1:N
        for mask_idxs in _combinations(N, k)
            mask = ntuple(i -> (i in mask_idxs ? 1 : 0), N)
            ranges = Vector{UnitRange{Int}}(undef, N)
            for i in 1:N
                ranges[i] = 1:(dims[i] - mask[i])
            end
            for coords in CartesianIndices(Tuple(ranges))
                origin = ntuple(i -> coords[i], N)
                key2 = (origin, mask)
                push!(cells_by_dim[k+1], key2)
            end
        end
    end

    for k in 0:N
        for (i, key2) in enumerate(cells_by_dim[k+1])
            cell_index[k+1][key2] = i
        end
    end

    boundaries = SparseMatrixCSC{Int,Int}[]
    for k in 1:N
        I = Int[]
        J = Int[]
        V = Int[]
        for (j, (origin, mask)) in enumerate(cells_by_dim[k+1])
            axes = [i for i in 1:N if mask[i] == 1]
            for (pos, axis) in enumerate(axes)
                mask_face = ntuple(i -> (i == axis ? 0 : mask[i]), N)
                origin_low = origin
                origin_high = ntuple(i -> (i == axis ? origin[i] + 1 : origin[i]), N)
                low_key = (origin_low, mask_face)
                high_key = (origin_high, mask_face)
                row_low = cell_index[k][low_key]
                row_high = cell_index[k][high_key]
                sign_low = isodd(pos) ? 1 : -1
                sign_high = -sign_low
                push!(I, row_low); push!(J, j); push!(V, sign_low)
                push!(I, row_high); push!(J, j); push!(V, sign_high)
            end
        end
        push!(boundaries, sparse(I, J, V, length(cells_by_dim[k]), length(cells_by_dim[k+1])))
    end

    cached = (cells_by_dim=cells_by_dim, cell_index=cell_index, boundaries=boundaries)
    if cache !== nothing
        Base.lock(cache.lock)
        try
            return get!(cache.cubical, key, cached)
        finally
            Base.unlock(cache.lock)
        end
    end
    return cached
end

function _graded_complex_from_image_channels(channels::Vector{<:AbstractArray},
                                             spec::FiltrationSpec;
                                             cache::Union{Nothing,EncodingCache}=nothing)
    img = channels[1]
    dims = size(img)
    for c in channels
        size(c) == dims || error("All channels must have the same size.")
    end
    N = length(dims)
    C = length(channels)

    cached = _cubical_structure(dims; cache=cache)
    cells_by_dim = cached.cells_by_dim
    cell_index = cached.cell_index

    grades = Vector{NTuple{C,Float64}}()
    for k in 0:N
        for (origin, mask) in cells_by_dim[k+1]
            maxv = fill(-Inf, C)
            ranges = Vector{UnitRange{Int}}(undef, N)
            for i in 1:N
                ranges[i] = origin[i]:(origin[i] + mask[i])
            end
            for coords in CartesianIndices(Tuple(ranges))
                for ch in 1:C
                    v = Float64(channels[ch][coords])
                    if v > maxv[ch]
                        maxv[ch] = v
                    end
                end
            end
            push!(grades, ntuple(i -> maxv[i], C))
        end
    end

    cells = [collect(1:length(cells_by_dim[k+1])) for k in 0:N]
    G = GradedComplex(cells, cached.boundaries, grades)
    orientation = get(spec.params, :orientation, ntuple(_ -> 1, C))
    axes = get(spec.params, :axes, _axes_from_grades(grades, C; orientation=orientation))
    return G, axes, orientation
end

function _graded_complex_from_image(data::ImageNd, spec::FiltrationSpec;
                                    cache::Union{Nothing,EncodingCache}=nothing)
    if haskey(spec.params, :channels)
        chans = spec.params[:channels]
        return _graded_complex_from_image_channels(chans, spec; cache=cache)
    elseif spec.kind == :image_distance_bifiltration
        mask = get(spec.params, :mask, nothing)
        mask === nothing && error("image_distance_bifiltration requires mask.")
        dist = _distance_transform(mask)
        chans = [data.data, dist]
        return _graded_complex_from_image_channels(chans, spec; cache=cache)
    else
        return _graded_complex_from_image_channels([data.data], spec; cache=cache)
    end
end

function _graded_complex_from_data(data, spec::FiltrationSpec;
                                   cache::Union{Nothing,EncodingCache}=nothing)
    if data isa GradedComplex
        N = length(data.grades[1])
        orientation = get(spec.params, :orientation, ntuple(_ -> 1, N))
        axes = get(spec.params, :axes, _axes_from_grades(data.grades, N; orientation=orientation))
        return data, axes, orientation
    elseif data isa PointCloud
        return _graded_complex_from_point_cloud(data, spec)
    elseif data isa ImageNd
        return _graded_complex_from_image(data, spec; cache=cache)
    elseif data isa GraphData
        return _graded_complex_from_graph(data, spec)
    elseif data isa EmbeddedPlanarGraph2D
        if spec.kind == :wing_vein_bifiltration
            grid = get(spec.params, :grid, (32, 32))
            bbox = get(spec.params, :bbox, data.bbox)
            bbox === nothing && error("wing_vein_bifiltration requires bbox.")
            xmin, xmax, ymin, ymax = bbox
            nx, ny = grid
            xs = range(xmin, xmax; length=nx)
            ys = range(ymin, ymax; length=ny)
            distV = zeros(Float64, nx, ny)
            distE = zeros(Float64, nx, ny)
            verts = data.vertices
            segments = Vector{Tuple{Vector{Float64},Vector{Float64}}}()
            if data.polylines === nothing
                for (u, v) in data.edges
                    push!(segments, (verts[u], verts[v]))
                end
            else
                for poly in data.polylines
                    for i in 1:(length(poly)-1)
                        push!(segments, (poly[i], poly[i+1]))
                    end
                end
            end
            for i in 1:nx, j in 1:ny
                x = xs[i]; y = ys[j]
                dv = Inf
                for v in verts
                    d = hypot(x - v[1], y - v[2])
                    if d < dv
                        dv = d
                    end
                end
                distV[i, j] = dv
                de = Inf
                for (a, b) in segments
                    ax, ay = a[1], a[2]
                    bx, by = b[1], b[2]
                    vx = bx - ax
                    vy = by - ay
                    wx = x - ax
                    wy = y - ay
                    t = (vx*wx + vy*wy) / (vx*vx + vy*vy)
                    if t < 0
                        px, py = ax, ay
                    elseif t > 1
                        px, py = bx, by
                    else
                        px, py = ax + t*vx, ay + t*vy
                    end
                    d = hypot(x - px, y - py)
                    if d < de
                        de = d
                    end
                end
                distE[i, j] = de
            end
            chans = [-distV, distE]
            img = ImageNd(chans[1])
            spec2 = FiltrationSpec(; kind=:image_distance_bifiltration, channels=chans,
                                    orientation=get(spec.params, :orientation, (-1, 1)))
            return _graded_complex_from_image(img, spec2; cache=cache)
        else
            # Treat as a graph with externally supplied vertex grades
            gspec = FiltrationSpec(; kind=spec.kind, spec.params...)
            gdata = GraphData(length(data.vertices), data.edges;
                              coords=data.vertices, weights=nothing, T=eltype(data.vertices[1]))
            return _graded_complex_from_graph(gdata, gspec)
        end
    else
        error("Unsupported dataset type for encode_from_data.")
    end
end

"""
    encode_from_data(data, spec; degree=0, output=:result) -> EncodingResult

Ingest a dataset + filtration spec into a persistence module and return
an EncodingResult with a fringe presentation. Use `output=:raw` to get
`(P, H, pi)` instead.
"""
function encode_from_data(data, spec::FiltrationSpec;
                          degree::Int=0,
                          output::Symbol=:result,
                          return_tuple::Bool=false,
                          emit::Symbol=:fringe,
                          field::AbstractCoeffField=QQField(),
                          cache=:auto)
    session_cache = _resolve_workflow_session_cache(cache)
    enc_cache = _workflow_encoding_cache(session_cache)
    G, axes, orientation = _graded_complex_from_data(data, spec; cache=enc_cache)
    if haskey(spec.params, :eps)
        G = _quantize_grades(G, spec.params[:eps])
        axes = _axes_from_grades(G.grades, length(G.grades[1]); orientation=orientation)
    end
    axes_policy = get(spec.params, :axes_policy, :encoding)
    if axes_policy == :as_given
        haskey(spec.params, :axes) || error("encode_from_data: axes_policy=:as_given requires axes.")
        axes = spec.params[:axes]
    elseif axes_policy == :coarsen
        max_len = get(spec.params, :max_axis_len, nothing)
        max_len === nothing && error("encode_from_data: axes_policy=:coarsen requires max_axis_len.")
        axes = _coarsen_axes(axes, Int(max_len))
    elseif axes_policy != :encoding
        error("encode_from_data: unknown axes_policy $(axes_policy).")
    end
    _validate_axes_sorted(axes)
    _validate_axes_kind(axes; axis_kind=get(spec.params, :axis_kind, nothing))
    if emit == :flange && get(spec.params, :axis_kind, nothing) != :zn
        error("encode_from_data: emit=:flange requires axis_kind=:zn (integer axes).")
    end
    P = _poset_from_axes_cached(axes, orientation; cache=enc_cache)
    pi = GridEncodingMap(P, axes; orientation=orientation)
    pi2 = _compile_encoding_cached(P, pi, session_cache)
    C = cochain_complex_from_graded_complex(G, P, axes; orientation=orientation, field=field)
    M = cohomology_module(C, degree)
    H = fringe_presentation(M)
    if return_tuple
        output = :raw
    end
    if emit == :flange
        FG = flange_presentation(M, pi)
        res = EncodingResult(P, M, pi2; H=H, presentation=(data=data, spec=spec),
                             opts=EncodingOptions(; backend=:data, field=field), backend=:data,
                             meta=(; flange=FG))
        return _return_encoding(res, output)
    elseif emit != :fringe
        error("encode_from_data: emit must be :fringe or :flange.")
    end
    res = EncodingResult(P, M, pi2; H=H, presentation=(data=data, spec=spec),
                         opts=EncodingOptions(; backend=:data, field=field), backend=:data, meta=(;))
    return _return_encoding(res, output)
end

ingest(data, spec::FiltrationSpec;
       degree::Int=0,
       field::AbstractCoeffField=QQField(),
       cache=:auto) =
    encode_from_data(data, spec;
                     degree=degree,
                     output=:raw,
                     field=field,
                     cache=cache)

# -----------------------------------------------------------------------------
# Coarsening / compression of finite encodings

"""
    coarsen(enc::EncodingResult; method=:uptight) -> EncodingResult

Coarsen/compress the finite encoding poset of an existing `EncodingResult`.

Currently supported:
- `method=:uptight`: build an uptight encoding from `enc.H`, push the fringe forward
  along the resulting finite encoding map, rebuild the pmodule, and postcompose the
  *ambient* classifier so `locate` still works on the new region poset.

This leaves `enc.backend` unchanged (the ambient encoding backend is still the same),
but replaces `(P, M, pi, H)` with their coarsened versions.
"""
function coarsen(enc::EncodingResult;
                 method::Symbol = :uptight,
                 cache=:auto)
    session_cache = _resolve_workflow_session_cache(cache)
    method === :uptight || error("coarsen: unsupported method=$method. Currently only :uptight is supported.")

    # 1) Compute coarsening map pi : (old P) -> (new P2)
    upt = build_uptight_encoding_from_fringe(enc.H)
    pi = upt.pi

    # 2) Push fringe module forward along pi
    H2 = pushforward_fringe_along_encoding(enc.H, pi)

    # 3) Rebuild pmodule on the coarsened region poset
    M2 = pmodule_from_fringe(H2)

    # 4) Postcompose old ambient classifier with the coarsening map
    pi2 = PostcomposedEncodingMap(enc.pi, pi)
    pi2c = _compile_encoding_cached(pi.P, pi2, session_cache)

    # 5) Preserve meta (Dict or NamedTuple), but record what happened
    meta2 = if enc.meta isa AbstractDict
        d = copy(enc.meta)
        d[:coarsen_method] = method
        d[:coarsen_n_before] = nvertices(enc.P)
        d[:coarsen_n_after] = nvertices(pi.P)
        d
    elseif enc.meta isa NamedTuple
        merge(enc.meta, (coarsen_method = method, coarsen_n_before = nvertices(enc.P), coarsen_n_after = nvertices(pi.P)))
    else
        Dict{Symbol,Any}(
            :orig_meta => enc.meta,
            :coarsen_method => method,
            :coarsen_n_before => nvertices(enc.P),
            :coarsen_n_after => nvertices(pi.P),
        )
    end

    return EncodingResult(pi.P, M2, pi2c;
        H = H2,
        presentation = enc.presentation,
        opts = enc.opts,
        backend = enc.backend,
        meta = meta2,
    )
end

# -----------------------------------------------------------------------------
# Accessors (avoid field spelunking in user code / docs)
# -----------------------------------------------------------------------------

"""
    poset(enc::EncodingResult)

Return the finite encoding poset used by `enc`.
"""
poset(enc::EncodingResult) = enc.P

"""
    pmodule(enc::EncodingResult)

Return the encoded module stored inside `enc`.
"""
pmodule(enc::EncodingResult) = enc.M

"""
    classifier(enc::EncodingResult)

Return the classifier / encoding map `pi` stored inside `enc`.
"""
classifier(enc::EncodingResult) = enc.pi

"""
    backend(enc::EncodingResult)

Return the backend symbol used during encoding.
"""
backend(enc::EncodingResult) = enc.backend

"""
    presentation(enc::EncodingResult)

Return the original presentation object used to create `enc`.
"""
presentation(enc::EncodingResult) = enc.presentation


# -----------------------------------------------------------------------------
# Homological algebra helpers for ResolutionResult
# -----------------------------------------------------------------------------

"""
    resolution(res::ResolutionResult)

Return the underlying resolution object stored inside `res`.
"""
resolution(res::ResolutionResult) = res.res

"""
    betti(res::ResolutionResult)

Return the Betti table stored inside `res` (may be `nothing`).
"""
betti(res::ResolutionResult) = res.betti

"""
    minimality_report(res::ResolutionResult)

Return the minimality report stored inside `res` (may be `nothing`).
"""
minimality_report(res::ResolutionResult) = res.minimality

# Forward minimality helpers for explicit resolution objects.
minimality_report(res::DerivedFunctors.Resolutions.ProjectiveResolution{K};
                  check_cover::Bool=true) where {K} =
    DerivedFunctors.minimality_report(res; check_cover=check_cover)
minimality_report(res::DerivedFunctors.Resolutions.InjectiveResolution{K};
                  check_hull::Bool=true) where {K} =
    DerivedFunctors.minimality_report(res; check_hull=check_hull)

"""
    is_minimal(res::ResolutionResult) -> Bool

Return whether the resolution stored in `res` was proven minimal by the chosen backend.
"""
is_minimal(res::ResolutionResult) = res.minimality !== nothing

is_minimal(res::DerivedFunctors.Resolutions.ProjectiveResolution{K};
           check_cover::Bool=true) where {K} =
    DerivedFunctors.is_minimal(res; check_cover=check_cover)
is_minimal(res::DerivedFunctors.Resolutions.InjectiveResolution{K};
           check_hull::Bool=true) where {K} =
    DerivedFunctors.is_minimal(res; check_hull=check_hull)

assert_minimal(res::DerivedFunctors.Resolutions.ProjectiveResolution{K};
               check_cover::Bool=true) where {K} =
    DerivedFunctors.assert_minimal(res; check_cover=check_cover)
assert_minimal(res::DerivedFunctors.Resolutions.InjectiveResolution{K};
               check_hull::Bool=true) where {K} =
    DerivedFunctors.assert_minimal(res; check_hull=check_hull)



# -----------------------------------------------------------------------------
# Derived functors and invariants from workflow objects

"""
    hom(A, B)

Compute Hom(A, B) for finite-poset modules.

Convenience overloads accept EncodingResult and unwrap the underlying PModule.
"""
function hom(A::EncodingResult, B::EncodingResult;
             cache=:auto)
    (A.P === B.P) || error("hom: encodings are on different posets; use encode(x, y; ...) to common-encode first.")
    K = coeff_type(A.M.field)
    cache_hom, session_cache = _resolve_workflow_specialized_cache(cache, DerivedFunctors.HomSystemCache)
    cache2 = _hom_cache_from_session(cache_hom, session_cache, K)
    return DerivedFunctors.Hom(A.M, B.M; cache=cache2)
end

function hom(A::EncodingResult, B::Modules.PModule{K};
             cache=:auto) where {K}
    (A.M.Q === B.Q) || error("hom: posets mismatch.")
    cache_hom, session_cache = _resolve_workflow_specialized_cache(cache, DerivedFunctors.HomSystemCache)
    cache2 = _hom_cache_from_session(cache_hom, session_cache, K)
    return DerivedFunctors.Hom(A.M, B; cache=cache2)
end

function hom(A::Modules.PModule{K}, B::EncodingResult;
             cache=:auto) where {K}
    (A.Q === B.M.Q) || error("hom: posets mismatch.")
    cache_hom, session_cache = _resolve_workflow_specialized_cache(cache, DerivedFunctors.HomSystemCache)
    cache2 = _hom_cache_from_session(cache_hom, session_cache, K)
    return DerivedFunctors.Hom(A, B.M; cache=cache2)
end

function hom(A::Modules.PModule{K}, B::Modules.PModule{K};
             cache=:auto) where {K}
    (A.Q === B.Q) || error("hom: posets mismatch.")
    cache_hom, session_cache = _resolve_workflow_specialized_cache(cache, DerivedFunctors.HomSystemCache)
    cache2 = _hom_cache_from_session(cache_hom, session_cache, K)
    return DerivedFunctors.Hom(A, B; cache=cache2)
end

"""
    tor(Rop, L; maxdeg=3, model=:auto, cache=:auto)

Compute Tor_t(Rop, L), where `Rop` is a right-module represented as a module on
the opposite poset P^op, and `L` is a left-module on P.

For EncodingResult inputs, the underlying posets must be opposite:
    poset_equal_opposite(L.P, Rop.P)
"""
function tor(Rop::EncodingResult, L::EncodingResult;
             maxdeg::Int=3, model::Symbol=:auto,
             cache=:auto)
    poset_equal_opposite(L.P, Rop.P) || error("tor: expected first argument on opposite poset of the second.")
    df = DerivedFunctorOptions(; maxdeg=maxdeg, model=model, canon=:none)
    model_eff = model == :auto ? :first : model
    cache_target = model_eff == :second ? L.M : Rop.M
    cache_res, session_cache = _resolve_workflow_specialized_cache(cache, ResolutionCache)
    cache2 = _resolution_cache_from_session(cache_res, session_cache, cache_target)
    return DerivedFunctors.Tor(Rop.M, L.M, df; cache=cache2)
end

function tor(Rop::Modules.PModule{K}, L::Modules.PModule{K};
             maxdeg::Int=3, model::Symbol=:auto,
             cache=:auto) where {K}
    poset_equal_opposite(L.Q, Rop.Q) || error("tor: expected first argument on opposite poset of the second.")
    df = DerivedFunctorOptions(; maxdeg=maxdeg, model=model, canon=:none)
    model_eff = model == :auto ? :first : model
    cache_target = model_eff == :second ? L : Rop
    cache_res, session_cache = _resolve_workflow_specialized_cache(cache, ResolutionCache)
    cache2 = _resolution_cache_from_session(cache_res, session_cache, cache_target)
    return DerivedFunctors.Tor(Rop, L, df; cache=cache2)
end

tor(Rop::EncodingResult, L::Modules.PModule{K}; kwargs...) where {K} = tor(Rop.M, L; kwargs...)
tor(Rop::Modules.PModule{K}, L::EncodingResult; kwargs...) where {K} = tor(Rop, L.M; kwargs...)


"""
    ext_algebra(enc; maxdeg=3, model=:auto)

Compute the Ext-algebra Ext^*(M, M) with Yoneda product, where M is the module stored in `enc`.
"""
function ext_algebra(enc::EncodingResult; maxdeg::Int=3, model::Symbol=:auto, canon::Symbol=:auto)
    df = DerivedFunctorOptions(; maxdeg=maxdeg, model=model, canon=canon)
    return DerivedFunctors.ExtAlgebra(enc.M, df)
end

"""
    ext(A::EncodingResult, B::EncodingResult; maxdeg=3, model=:auto, canon=:auto, cache=:auto)

Compute Ext^t(A, B) using the finite-poset modules stored in EncodingResult.

If `A` and `B` are not encoded on the same poset object, you must common-encode first:
    encs = encode(x, y; backend=...)
    E = ext(encs[1], encs[2])
"""
function ext(A::EncodingResult, B::EncodingResult;
             maxdeg::Int=3, model::Symbol=:auto, canon::Symbol=:auto,
             cache=:auto)
    (A.P === B.P) || error("ext: encodings are on different posets; use encode(x, y; ...) to common-encode first.")
    df = DerivedFunctorOptions(; maxdeg=maxdeg, model=model, canon=canon)
    model_eff = model == :auto ? :projective : model
    cache_target = model_eff == :injective ? B.M :
                   model_eff == :projective ? A.M : nothing
    cache_res, session_cache = _resolve_workflow_specialized_cache(cache, ResolutionCache)
    cache2 = _resolution_cache_from_session(cache_res, session_cache, cache_target)
    return DerivedFunctors.Ext(A.M, B.M, df; cache=cache2)
end

function ext(A::EncodingResult, B::Modules.PModule{K};
             maxdeg::Int=3, model::Symbol=:auto, canon::Symbol=:auto,
             cache=:auto) where {K}
    (A.M.Q === B.Q) || error("ext: module B lives on a different poset; common-encode first.")
    df = DerivedFunctorOptions(; maxdeg=maxdeg, model=model, canon=canon)
    model_eff = model == :auto ? :projective : model
    cache_target = model_eff == :injective ? B :
                   model_eff == :projective ? A.M : nothing
    cache_res, session_cache = _resolve_workflow_specialized_cache(cache, ResolutionCache)
    cache2 = _resolution_cache_from_session(cache_res, session_cache, cache_target)
    return DerivedFunctors.Ext(A.M, B, df; cache=cache2)
end

"""
    resolve(enc::EncodingResult; kind=:projective, opts=ResolutionOptions(),
            minimality=false, cache=:auto)

Compute a projective or injective resolution starting from an EncodingResult.
Returns a ResolutionResult that stores the resolution plus provenance.

- kind=:projective (default) computes a ProjectiveResolution and stores its Betti table.
- kind=:injective computes an InjectiveResolution and stores its Bass table in `meta`.

Minimality checks can be expensive; enable with `minimality=true`.

If `cache` is `:auto` (default), a temporary session cache is used for the call.
Pass a shared `SessionCache()` via `cache=...` to reuse resolutions across calls.
"""
function resolve(enc::EncodingResult;
                 kind::Symbol=:projective,
                 opts::ResolutionOptions=ResolutionOptions(),
                 minimality::Bool=false,
                 check_hull::Bool=true,
                 cache=:auto)
    opts = opts
    cache_res, session_cache = _resolve_workflow_specialized_cache(cache, ResolutionCache)
    cache2 = _resolution_cache_from_session(cache_res, session_cache, enc.M)
    kind_norm = kind in (:proj, :projective) ? :projective :
                kind in (:inj, :injective)   ? :injective  :
                error("resolve: kind must be :projective or :injective (got $(kind))")

    if kind_norm == :projective
        res = DerivedFunctors.projective_resolution(enc.M, opts; cache=cache2)
        b = DerivedFunctors.betti(res)
        mrep = minimality ? DerivedFunctors.minimality_report(res) : nothing
        return ResolutionResult(res; enc=enc, betti=b, minimality=mrep, opts=opts, meta=(kind=:projective,))
    else
        res = DerivedFunctors.injective_resolution(enc.M, opts; cache=cache2)
        bass = DerivedFunctors.bass(res)
        mrep = minimality ? DerivedFunctors.minimality_report(res; check_hull=check_hull) : nothing
        return ResolutionResult(res; enc=enc, betti=nothing, minimality=mrep, opts=opts,
                                meta=(kind=:injective, bass=bass))
    end
end

# -----------------------------------------------------------------------------
# Complex-level homological algebra
# -----------------------------------------------------------------------------

"""
    rhom(C, N; kwargs...)

Compute RHom(C, N) where C is a module cochain complex and N is a module.
"""
function rhom(C::ModuleComplexes.ModuleCochainComplex{K}, N::Modules.PModule{K};
              cache=:auto,
              kwargs...) where {K}
    cache_hom, session_cache = _resolve_workflow_specialized_cache(cache, DerivedFunctors.HomSystemCache)
    cache2 = _hom_cache_from_session(cache_hom, session_cache, K)
    return ModuleComplexes.RHom(C, N; cache=cache2, kwargs...)
end

rhom(C::ModuleComplexes.ModuleCochainComplex{K}, N::EncodingResult;
     cache=:auto,
     kwargs...) where {K} =
    rhom(C, N.M; cache=cache, kwargs...)


"""
    derived_tensor(Rop, C; kwargs...)

Compute derived tensor Rop \\otimes^L C, where Rop is on P^op and C is on P.
"""
derived_tensor(Rop::Modules.PModule{K}, C::ModuleComplexes.ModuleCochainComplex{K}; kwargs...) where {K} =
    ModuleComplexes.DerivedTensor(Rop, C; kwargs...)

derived_tensor(Rop::EncodingResult, C::ModuleComplexes.ModuleCochainComplex{K}; kwargs...) where {K} =
    derived_tensor(Rop.M, C; kwargs...)


"""
    hyperext(C, N; maxdeg=3, kwargs...)

Compute HyperExt^t(C, N) for a module cochain complex C and a module N.
"""
function hyperext(C::ModuleComplexes.ModuleCochainComplex{K}, N::Modules.PModule{K};
                  maxdeg::Int=3,
                  cache=:auto,
                  kwargs...) where {K}
    df = DerivedFunctorOptions(; maxdeg=maxdeg, model=:auto, canon=:none)
    cache_hom, session_cache = _resolve_workflow_specialized_cache(cache, DerivedFunctors.HomSystemCache)
    cache2 = _hom_cache_from_session(cache_hom, session_cache, K)
    return ModuleComplexes.hyperExt(C, N, df; cache=cache2, kwargs...)
end

hyperext(C::ModuleComplexes.ModuleCochainComplex{K}, N::EncodingResult;
         cache=:auto,
         kwargs...) where {K} =
    hyperext(C, N.M; cache=cache, kwargs...)


"""
    hypertor(Rop, C; maxdeg=3, kwargs...)

Compute HyperTor_t(Rop, C) for a right-module Rop (on P^op) and a module complex C (on P).
"""
function hypertor(Rop::Modules.PModule{K}, C::ModuleComplexes.ModuleCochainComplex{K};
                  maxdeg::Int=3, kwargs...) where {K}
    df = DerivedFunctorOptions(; maxdeg=maxdeg, model=:auto, canon=:none)
    return ModuleComplexes.hyperTor(Rop, C, df; kwargs...)
end

hypertor(Rop::EncodingResult, C::ModuleComplexes.ModuleCochainComplex{K}; kwargs...) where {K} =
    hypertor(Rop.M, C; kwargs...)



# Internal helper: call an invariant function with a few common signatures.
function _call_invariant(f, enc::EncodingResult, opts::InvariantOptions; kwargs...)
    M = enc.M
    pi = enc.pi

    if hasmethod(f, Tuple{typeof(M), typeof(pi), typeof(opts)})
        return f(M, pi, opts; kwargs...)
    elseif hasmethod(f, Tuple{typeof(M), typeof(pi)})
        return f(M, pi; kwargs...)
    elseif hasmethod(f, Tuple{typeof(M), typeof(opts)})
        return f(M, opts; kwargs...)
    elseif hasmethod(f, Tuple{typeof(M)})
        return f(M; kwargs...)
    else
        error("invariants: no method found for $(f) with signatures (M), (M,pi), (M,opts), or (M,pi,opts)")
    end
end

# Internal helper: call an invariant function with a few common signatures.
function _call_invariant(f, enc::EncodingResult, opts::InvariantOptions; kwargs...)
    if hasmethod(f, Tuple{typeof(enc.M), typeof(enc.pi), typeof(opts)})
        return f(enc.M, enc.pi, opts; kwargs...)
    elseif hasmethod(f, Tuple{typeof(enc.M), typeof(enc.pi)})
        return f(enc.M, enc.pi; kwargs...)
    elseif hasmethod(f, Tuple{typeof(enc.M), typeof(opts)})
        return f(enc.M, opts; kwargs...)
    elseif hasmethod(f, Tuple{typeof(enc.M)})
        return f(enc.M; kwargs...)
    else
        error("invariant: no method found for invariant function $(f). Expected f(M), f(M,pi), f(M,opts), or f(M,pi,opts).")
    end
end


"""
    invariant(enc::EncodingResult; which=:rank_invariant, opts=InvariantOptions(), kwargs...) -> InvariantResult

Compute a single invariant from the `Invariants` submodule and wrap it in an `InvariantResult`.

`which` may be:
- a Symbol naming a function in `Invariants` (e.g. `:rank_invariant`)
- a callable itself
"""
function invariant(enc::EncodingResult;
                   which=:rank_invariant,
                   opts::InvariantOptions=InvariantOptions(),
                   cache=:auto,
                   kwargs...)
    opts = opts
    session_cache = _resolve_workflow_session_cache(cache)
    enc2 = _encoding_with_session_cache(enc, session_cache)
    f = which isa Symbol ? getfield(Invariants, which) : which
    val = _call_invariant(f, enc2, opts; kwargs...)
    return InvariantResult(enc2, which, val; opts=opts, meta=NamedTuple())
end


"""
    invariants(enc::EncodingResult; which=..., opts=InvariantOptions(), kwargs...) -> Vector{InvariantResult}

Batch convenience wrapper around `invariant`.

`which` may be:
- a Symbol or callable (compute one)
- a vector of Symbols / callables (compute many)
"""
function invariants(enc::EncodingResult;
                    which=[:rank_invariant],
                    opts::InvariantOptions=InvariantOptions(),
                    cache=:auto,
                    kwargs...)
    opts = opts
    if which isa AbstractVector
        return [invariant(enc; which=w, opts=opts, cache=cache, kwargs...) for w in which]
    else
        return [invariant(enc; which=which, opts=opts, cache=cache, kwargs...)]
    end
end


# -----------------------------------------------------------------------------
# Curated invariant entrypoints (stable value-returning wrappers)
# -----------------------------------------------------------------------------

rank_invariant(M::PModule{K}, opts::InvariantOptions=InvariantOptions(); kwargs...) where {K} =
    Invariants.rank_invariant(M, opts; kwargs...)

rank_map(M::PModule{K}, pi, x, y, opts::InvariantOptions=InvariantOptions(); kwargs...) where {K} =
    Invariants.rank_map(M, pi, x, y, opts; kwargs...)
rank_map(M::PModule{K}, a::Int, b::Int; kwargs...) where {K} = Invariants.rank_map(M, a, b; kwargs...)

restricted_hilbert(M::PModule{K}) where {K} = Invariants.restricted_hilbert(M)

restricted_hilbert(M::PModule{K}, pi, x, opts::InvariantOptions=InvariantOptions(); kwargs...) where {K} =
    Invariants.restricted_hilbert(M, pi, x, opts; kwargs...)

function euler_surface(M::PModule{K}, pi, opts::InvariantOptions=InvariantOptions(); kwargs...) where {K}
    opts = opts
    if any(haskey(kwargs, k) for k in (:axes, :axes_policy, :max_axis_len, :box, :threads, :strict))
        opts = InvariantOptions(
            axes = get(kwargs, :axes, opts.axes),
            axes_policy = get(kwargs, :axes_policy, opts.axes_policy),
            max_axis_len = get(kwargs, :max_axis_len, opts.max_axis_len),
            box = get(kwargs, :box, opts.box),
            threads = get(kwargs, :threads, opts.threads),
            strict = get(kwargs, :strict, opts.strict),
            pl_mode = get(kwargs, :pl_mode, opts.pl_mode),
        )
        kw_nt = NamedTuple(kwargs)
        kwargs = Base.structdiff(kw_nt, (; axes=nothing, axes_policy=nothing, max_axis_len=nothing, box=nothing, threads=nothing, strict=nothing, pl_mode=nothing))
    end
    return Invariants.euler_surface(M, pi, opts; kwargs...)
end

euler_surface(C::ModuleCochainComplex{K}, pi, opts::InvariantOptions=InvariantOptions(); kwargs...) where {K} =
    Invariants.euler_surface(C, pi, opts; kwargs...)

slice_barcode(M::PModule{K}, chain::AbstractVector{Int}, opts::InvariantOptions=InvariantOptions(); kwargs...) where {K} =
    Invariants.slice_barcode(M, chain; kwargs...)

slice_barcodes(M::PModule{K}, chains::AbstractVector, opts::InvariantOptions=InvariantOptions(); kwargs...) where {K} =
    Invariants.slice_barcodes(M, chains; kwargs...)

function slice_barcodes(M::PModule{K}, plan::Invariants.CompiledSlicePlan, opts::InvariantOptions=InvariantOptions(); kwargs...) where {K}
    opts0 = opts
    kwargs_nt = NamedTuple(kwargs)
    for k in keys(kwargs_nt)
        (k == :threads || k == :packed) && continue
        throw(ArgumentError("slice_barcodes(plan): unsupported keyword :$k"))
    end
    return Invariants.slice_barcodes(M, plan;
        packed = get(kwargs_nt, :packed, false),
        threads = get(kwargs_nt, :threads, opts0.threads))
end

function slice_barcodes(M::PModule{K}, pi, opts::InvariantOptions=InvariantOptions();
                        cache=:auto,
                        kwargs...) where {K}
    opts0 = opts
    kwargs_nt = NamedTuple(kwargs)
    haskey(kwargs_nt, :box) &&
        throw(ArgumentError("slice_barcodes: pass box via opts::InvariantOptions, not keyword :box"))
    haskey(kwargs_nt, :strict) &&
        throw(ArgumentError("slice_barcodes: pass strict via opts::InvariantOptions, not keyword :strict"))
    kwargs2 = (; (k => v for (k, v) in pairs(kwargs_nt) if k != :threads)...)
    cache_slice, session_cache = _resolve_workflow_specialized_cache(cache, Invariants.SlicePlanCache)
    cache2 = _slice_plan_cache_from_session(cache_slice, session_cache)
    return Invariants.slice_barcodes(M, pi;
        opts = opts0,
        threads = get(kwargs_nt, :threads, opts0.threads),
        cache = cache2,
        kwargs2...)
end

rank_invariant(enc::EncodingResult; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    invariant(enc; which=:rank_invariant, opts=opts, kwargs...).value

restricted_hilbert(enc::EncodingResult; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    invariant(enc; which=:restricted_hilbert, opts=opts, kwargs...).value

euler_surface(enc::EncodingResult; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    invariant(enc; which=:euler_surface, opts=opts, kwargs...).value

slice_barcode(enc::EncodingResult; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    invariant(enc; which=:slice_barcode, opts=opts, kwargs...).value

function slice_barcodes(enc::EncodingResult;
                        opts::InvariantOptions=InvariantOptions(),
                        cache=:auto,
                        kwargs...)
    session_cache = _resolve_workflow_session_cache(cache)
    return slice_barcodes(enc.M, enc.pi, opts;
                          cache=session_cache,
                          kwargs...)
end

function mp_landscape(enc::EncodingResult;
                      opts::InvariantOptions=InvariantOptions(),
                      cache=:auto,
                      kwargs...)
    session_cache = _resolve_workflow_session_cache(cache)
    return mp_landscape(enc.M, enc.pi;
                        opts=opts,
                        cache=session_cache,
                        kwargs...)
end
mp_landscape(M::PModule{K}, slices::AbstractVector; kwargs...) where {K} =
    Invariants.mp_landscape(M, slices; kwargs...)
function mp_landscape(M::PModule{K}, pi;
                      opts::InvariantOptions=InvariantOptions(),
                      cache=:auto,
                      kwargs...) where {K}
    opt_keys = (:box, :strict, :threads, :axes, :axes_policy, :max_axis_len, :pl_mode)
    cache_slice, session_cache = _resolve_workflow_specialized_cache(cache, Invariants.SlicePlanCache)
    cache2 = _slice_plan_cache_from_session(cache_slice, session_cache)
    opts0 = opts
    if any(k -> haskey(kwargs, k), opt_keys)
        base = opts0
        opts = InvariantOptions(
            box = get(kwargs, :box, base.box),
            strict = get(kwargs, :strict, base.strict),
            threads = get(kwargs, :threads, base.threads),
            axes = get(kwargs, :axes, base.axes),
            axes_policy = get(kwargs, :axes_policy, base.axes_policy),
            max_axis_len = get(kwargs, :max_axis_len, base.max_axis_len),
            pl_mode = get(kwargs, :pl_mode, base.pl_mode),
        )
        kwargs_nt = NamedTuple(kwargs)
        kwargs2 = (; (k => v for (k, v) in pairs(kwargs_nt) if !(k in opt_keys))...)
        return Invariants.mp_landscape(M, pi, opts; cache=cache2, kwargs2...)
    end
    return Invariants.mp_landscape(M, pi, opts0; cache=cache2, kwargs...)
end

mpp_decomposition(enc::EncodingResult; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    invariant(enc; which=:mpp_decomposition, opts=opts, kwargs...).value

mpp_image(enc::EncodingResult; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    invariant(enc; which=:mpp_image, opts=opts, kwargs...).value

# -----------------------------------------------------------------------------
# Featurizer specs and dataset featurization
# -----------------------------------------------------------------------------

abstract type AbstractFeaturizerSpec end

"""
    AbstractInvariantCache

Typed cache protocol used by featurizers for `build_cache` / `transform(spec, cache)`.
"""
abstract type AbstractInvariantCache end

"""
    BatchOptions

Execution contract for dataset-level batch APIs.
- `threaded`: enable parallel execution.
- `backend`: `:serial`, `:threads`, or `:folds` (optional extension backend).
- `progress`: reserved hook for optional progress extensions.
- `deterministic`: use static partitioning with deterministic slot writes.
- `chunk_size`: chunk size for static chunking (`0` means auto/static partition).
"""
struct BatchOptions
    threaded::Bool
    backend::Symbol
    progress::Bool
    deterministic::Bool
    chunk_size::Int
end

function BatchOptions(; threaded::Bool=true,
                        backend::Symbol = (threaded ? :threads : :serial),
                        progress::Bool=false,
                        deterministic::Bool=true,
                        chunk_size::Int=0)
    backend in (:serial, :threads, :folds) ||
        throw(ArgumentError("BatchOptions.backend must be :serial, :threads, or :folds"))
    chunk_size >= 0 || throw(ArgumentError("BatchOptions.chunk_size must be >= 0"))
    return BatchOptions(threaded, backend, progress, deterministic, chunk_size)
end

"""
    ModuleInvariantCache

Cache wrapper for module-only featurizers (`RankGridSpec`).
"""
struct ModuleInvariantCache{K,F<:AbstractCoeffField,MatT<:AbstractMatrix{K}} <: AbstractInvariantCache
    M::PModule{K,F,MatT}
    opts::InvariantOptions
    threads::Bool
    level::Symbol
end

"""
    RestrictedHilbertInvariantCache

Dedicated cache object for restricted-Hilbert-driven computations on `(M, pi)`.
This is currently used by `EulerSurfaceSpec` transforms.
"""
struct RestrictedHilbertInvariantCache{K,F<:AbstractCoeffField,MatT<:AbstractMatrix{K},PiT} <: AbstractInvariantCache
    M::PModule{K,F,MatT}
    pi::PiT
    opts::InvariantOptions
    threads::Bool
    level::Symbol
    hilbert::Vector{Int}
end

"""
    EncodingInvariantCache

Cache wrapper for featurizers that require both `(M, pi)`.
It owns per-sample typed memo tables for slice plans and projected caches.
"""
mutable struct EncodingInvariantCache{K,F<:AbstractCoeffField,MatT<:AbstractMatrix{K},PiT} <: AbstractInvariantCache
    M::PModule{K,F,MatT}
    pi::PiT
    opts::InvariantOptions
    threads::Bool
    level::Symbol
    session_cache::Union{Nothing,SessionCache}
    slice_plan_cache::Invariants.SlicePlanCache
    slice_plans::Dict{UInt,Invariants.CompiledSlicePlan}
    projected_arrangements::Dict{UInt,Invariants.ProjectedArrangement}
    projected_module::Dict{UInt,Invariants.ProjectedBarcodeCache{K}}
    projected_refs::Dict{Tuple{UInt,UInt},Invariants.ProjectedBarcodeCache}
    fibered::Dict{UInt,Invariants.FiberedBarcodeCache2D{K}}
    rank_query::Union{Nothing,Invariants.RankQueryCache}
end

const _InvariantCacheHandle = Union{
    ModuleInvariantCache,
    RestrictedHilbertInvariantCache,
    EncodingInvariantCache,
}

"""
    LandscapeSpec

Typed spec for slice-based landscape vectorization.
"""
struct LandscapeSpec <: AbstractFeaturizerSpec
    directions::Vector{Vector{Float64}}
    offsets::Vector{Vector{Float64}}
    offset_weights::Union{Nothing,Vector{Float64},Matrix{Float64}}
    kmax::Int
    tgrid::Vector{Float64}
    aggregate::Symbol
    normalize_weights::Bool
    tmin::Union{Nothing,Float64}
    tmax::Union{Nothing,Float64}
    nsteps::Int
    strict::Bool
    drop_unknown::Bool
    dedup::Bool
    normalize_dirs::Symbol
    direction_weight::Symbol
    threads::Union{Nothing,Bool}
end

"""
    PersistenceImageSpec

Typed spec for slice-based persistence-image vectorization.
"""
struct PersistenceImageSpec <: AbstractFeaturizerSpec
    directions::Vector{Vector{Float64}}
    offsets::Vector{Vector{Float64}}
    offset_weights::Union{Nothing,Vector{Float64},Matrix{Float64}}
    xgrid::Vector{Float64}
    ygrid::Vector{Float64}
    sigma::Float64
    coords::Symbol
    weighting::Symbol
    p::Float64
    normalize::Symbol
    aggregate::Symbol
    normalize_weights::Bool
    tmin::Union{Nothing,Float64}
    tmax::Union{Nothing,Float64}
    nsteps::Int
    strict::Bool
    drop_unknown::Bool
    dedup::Bool
    normalize_dirs::Symbol
    direction_weight::Symbol
    threads::Union{Nothing,Bool}
end

"""
    EulerSurfaceSpec

Typed spec for Euler characteristic surface vectorization.
"""
struct EulerSurfaceSpec{A<:Union{Nothing,NTuple}} <: AbstractFeaturizerSpec
    axes::A
    axes_policy::Symbol
    max_axis_len::Int
    strict::Union{Nothing,Bool}
    threads::Union{Nothing,Bool}
end

"""
    RankGridSpec

Typed spec for flattening rank invariant on an n x n grid of region pairs.
"""
struct RankGridSpec <: AbstractFeaturizerSpec
    nvertices::Int
    store_zeros::Bool
    threads::Union{Nothing,Bool}
end

"""
    SlicedBarcodeSpec

Typed spec for slice-barcode derived features (summary/entropy).
"""
struct SlicedBarcodeSpec{S<:Tuple} <: AbstractFeaturizerSpec
    directions::Vector{Vector{Float64}}
    offsets::Vector{Vector{Float64}}
    offset_weights::Union{Nothing,Vector{Float64},Matrix{Float64}}
    featurizer::Symbol
    summary_fields::S
    summary_normalize_entropy::Bool
    entropy_normalize::Bool
    entropy_weighting::Symbol
    entropy_p::Float64
    aggregate::Symbol
    normalize_weights::Bool
    tmin::Union{Nothing,Float64}
    tmax::Union{Nothing,Float64}
    nsteps::Int
    strict::Bool
    drop_unknown::Bool
    dedup::Bool
    normalize_dirs::Symbol
    direction_weight::Symbol
    threads::Union{Nothing,Bool}
end

"""
    SignedBarcodeImageSpec

Typed spec for rectangle signed-barcode image vectorization.
"""
struct SignedBarcodeImageSpec{A<:Union{Nothing,NTuple}} <: AbstractFeaturizerSpec
    xs::Vector{Float64}
    ys::Vector{Float64}
    sigma::Float64
    mode::Symbol
    method::Symbol
    axes::A
    axes_policy::Symbol
    max_axis_len::Int
    strict::Union{Nothing,Bool}
    threads::Union{Nothing,Bool}
end

"""
    ProjectedDistancesSpec

Typed spec for projected-distance features against a fixed reference bank.
"""
struct ProjectedDistancesSpec{R<:AbstractVector,D<:Union{Nothing,Vector{Vector{Float64}}}} <: AbstractFeaturizerSpec
    references::R
    reference_names::Vector{Symbol}
    dist::Symbol
    p::Float64
    q::Float64
    agg::Symbol
    directions::D
    n_dirs::Int
    normalize::Symbol
    enforce_monotone::Symbol
    precompute::Bool
    threads::Union{Nothing,Bool}
end

"""
    CompositeSpec

Concatenate multiple featurizer specs in a deterministic order.
"""
struct CompositeSpec{S<:Tuple} <: AbstractFeaturizerSpec
    specs::S
    namespacing::Bool
end

"""
    FeatureSet

Canonical output for batch featurization.
`X` uses row-major sample layout: size(X) == (nsamples, nfeatures).
"""
struct FeatureSet{T,MT}
    X::Matrix{T}
    names::Vector{Symbol}
    ids::Vector{String}
    meta::MT
end

Base.size(F::FeatureSet) = size(F.X)
@inline nsamples(F::FeatureSet) = size(F.X, 1)
@inline nfeatures(F::FeatureSet) = size(F.X, 2)
@inline asrowmajor(F::FeatureSet) = F.X
@inline ascolmajor(F::FeatureSet) = permutedims(F.X)
@inline feature_axes(F::FeatureSet) = _meta_get(F.meta, :feature_axes, nothing)

"""
    FeatureSetWideTable(fs)
    FeatureSetLongTable(fs)

Table wrappers used by optional `Tables.jl` extension.
"""
struct FeatureSetWideTable{T}
    fs::T
end

struct FeatureSetLongTable{T}
    fs::T
end

const TAMER_EXPERIMENT_SCHEMA_VERSION = v"0.1.0"

"""
    ExperimentIOConfig

Output policy for `run_experiment`.

- `format` controls table mode (`:wide` / `:long`) for table-based writers.
- `formats` controls which artifact files are emitted per featurizer:
  `:arrow`, `:parquet`, `:npz`, `:csv_wide`, `:csv_long`.
"""
struct ExperimentIOConfig
    outdir::Union{Nothing,String}
    prefix::String
    format::Symbol
    formats::Vector{Symbol}
    write_metadata::Bool
    overwrite::Bool
end

function ExperimentIOConfig(; outdir::Union{Nothing,AbstractString}=nothing,
                              prefix::AbstractString="experiment",
                              format::Symbol=:wide,
                              formats::Union{Symbol,AbstractVector,Tuple}=(:arrow,),
                              write_metadata::Bool=true,
                              overwrite::Bool=true)
    format in (:wide, :long) || throw(ArgumentError("ExperimentIOConfig.format must be :wide or :long"))
    formats0 = Symbol[]
    raw = formats isa Symbol ? (formats,) : formats
    raw isa AbstractVector || raw isa Tuple ||
        throw(ArgumentError("ExperimentIOConfig.formats must be a Symbol, tuple, or vector"))
    for f in raw
        fs = Symbol(f)
        fs in (:arrow, :parquet, :npz, :csv_wide, :csv_long) ||
            throw(ArgumentError("ExperimentIOConfig.formats contains unsupported format $(fs)"))
        push!(formats0, fs)
    end
    # deterministic order + dedup for manifest reproducibility
    pref = Dict(:arrow=>1, :parquet=>2, :npz=>3, :csv_wide=>4, :csv_long=>5)
    formats1 = unique(formats0)
    sort!(formats1; by=f->pref[f])
    return ExperimentIOConfig(outdir === nothing ? nothing : String(outdir),
                              String(prefix),
                              format,
                              formats1,
                              write_metadata,
                              overwrite)
end

"""
    ExperimentSpec

Typed experiment plan: list of featurizers + shared execution/output settings.
"""
struct ExperimentSpec{S<:Tuple,F1,F2,C,MT}
    name::String
    featurizers::S
    opts::InvariantOptions
    batch::BatchOptions
    cache::C
    on_unsupported::Symbol
    idfun::F1
    labelfun::F2
    io::ExperimentIOConfig
    metadata::MT
end

@inline function _normalize_featurizers(specs)
    if specs isa AbstractFeaturizerSpec
        return (specs,)
    elseif specs isa Tuple
        all(s -> s isa AbstractFeaturizerSpec, specs) ||
            throw(ArgumentError("ExperimentSpec.featurizers tuple must contain only AbstractFeaturizerSpec items"))
        return specs
    elseif specs isa AbstractVector
        all(s -> s isa AbstractFeaturizerSpec, specs) ||
            throw(ArgumentError("ExperimentSpec.featurizers vector must contain only AbstractFeaturizerSpec items"))
        return tuple(specs...)
    end
    throw(ArgumentError("ExperimentSpec.featurizers must be a spec, tuple of specs, or vector of specs"))
end

function ExperimentSpec(featurizers;
                        name::AbstractString="experiment",
                        opts::InvariantOptions=InvariantOptions(),
                        batch::Union{BatchOptions,Nothing}=nothing,
                        cache=:auto,
                        on_unsupported::Symbol=:error,
                        idfun=nothing,
                        labelfun=nothing,
                        io::ExperimentIOConfig=ExperimentIOConfig(),
                        metadata=NamedTuple())
    specs = _normalize_featurizers(featurizers)
    isempty(specs) && throw(ArgumentError("ExperimentSpec requires at least one featurizer"))
    (on_unsupported in (:error, :skip, :missing)) ||
        throw(ArgumentError("ExperimentSpec.on_unsupported must be :error, :skip, or :missing"))
    (cache === :auto || cache === nothing || cache isa SessionCache) ||
        throw(ArgumentError("ExperimentSpec.cache must be :auto, nothing, or SessionCache"))
    opts0 = opts
    batch0 = batch === nothing ? BatchOptions() : batch
    return ExperimentSpec{typeof(specs),typeof(idfun),typeof(labelfun),typeof(cache),typeof(metadata)}(
        String(name),
        specs,
        opts0,
        batch0,
        cache,
        on_unsupported,
        idfun,
        labelfun,
        io,
        metadata,
    )
end

"""
    ExperimentArtifact

One featurizer run in an experiment.
"""
struct ExperimentArtifact{F<:AbstractFeaturizerSpec,FS}
    key::Symbol
    spec::F
    features::FS
    elapsed_seconds::Float64
    cache_stats::Dict{String,Any}
    feature_paths::Dict{Symbol,String}
    metadata_path::Union{Nothing,String}
end

"""
    ExperimentResult

Output of `run_experiment`.
"""
struct ExperimentResult{ES,AV,MT}
    spec::ES
    artifacts::AV
    total_elapsed_seconds::Float64
    run_dir::Union{Nothing,String}
    manifest_path::Union{Nothing,String}
    metadata::MT
end

"""
    LoadedExperimentArtifact

Artifact record returned by `load_experiment`.
"""
struct LoadedExperimentArtifact
    key::Symbol
    spec::Union{Nothing,AbstractFeaturizerSpec}
    opts::Union{Nothing,InvariantOptions}
    features::Union{Nothing,FeatureSet}
    metadata::Any
    elapsed_seconds::Float64
    cache_stats::Dict{String,Any}
    feature_paths::Dict{Symbol,String}
    metadata_path::Union{Nothing,String}
end

"""
    LoadedExperimentResult

Return type for `load_experiment`.
"""
struct LoadedExperimentResult
    manifest::Dict{String,Any}
    artifacts::Vector{LoadedExperimentArtifact}
    run_dir::Union{Nothing,String}
    manifest_path::String
    total_elapsed_seconds::Float64
end

@inline function _meta_get(meta::NamedTuple, key::Symbol, default=nothing)
    return haskey(meta, key) ? getfield(meta, key) : default
end
@inline function _meta_get(meta::AbstractDict, key::Symbol, default=nothing)
    return haskey(meta, key) ? meta[key] : default
end
@inline function _meta_get(meta, key::Symbol, default=nothing)
    return hasproperty(meta, key) ? getproperty(meta, key) : default
end

@inline _jsonable(x::Nothing) = nothing
@inline _jsonable(x::Bool) = x
@inline _jsonable(x::Integer) = x
@inline _jsonable(x::AbstractFloat) = x
@inline _jsonable(x::AbstractString) = String(x)
@inline _jsonable(x::Symbol) = String(x)
@inline _jsonable(x::VersionNumber) = string(x)
@inline _jsonable(x::DataType) = string(x)
@inline _jsonable(x::AbstractCoeffField) = string(x)

function _jsonable(x::NamedTuple)
    out = Dict{String,Any}()
    for (k, v) in pairs(x)
        out[String(k)] = _jsonable(v)
    end
    return out
end

function _jsonable(x::AbstractVector)
    return [_jsonable(v) for v in x]
end

function _jsonable(x::Tuple)
    return [_jsonable(v) for v in x]
end

function _jsonable(x::AbstractDict)
    out = Dict{String,Any}()
    for (k, v) in pairs(x)
        out[String(k)] = _jsonable(v)
    end
    return out
end

function _jsonable(x)
    T = typeof(x)
    if isstructtype(T)
        fields = Dict{String,Any}()
        for f in fieldnames(T)
            fields[String(f)] = _jsonable(getfield(x, f))
        end
        return Dict(
            "type" => string(nameof(T)),
            "module" => string(parentmodule(T)),
            "fields" => fields,
        )
    end
    return string(x)
end

@inline _is_simple_json_scalar(x) =
    x === nothing || x isa Bool || x isa Integer || x isa AbstractFloat || x isa AbstractString || x isa Symbol

function _reference_stub_json(x)
    if _is_simple_json_scalar(x)
        return _jsonable(x)
    end
    T = typeof(x)
    return Dict(
        "kind" => "reference_stub",
        "type" => string(nameof(T)),
        "module" => string(parentmodule(T)),
        "objectid" => string(UInt(objectid(x))),
    )
end

function _spec_jsonable(spec::ProjectedDistancesSpec)
    ref_ids = String[string(nm) for nm in spec.reference_names]
    return Dict(
        "type" => "ProjectedDistancesSpec",
        "module" => string(parentmodule(typeof(spec))),
        "fields" => Dict(
            "reference_ids" => ref_ids,
            "references" => [_reference_stub_json(r) for r in spec.references],
            "reference_names" => [String(nm) for nm in spec.reference_names],
            "dist" => String(spec.dist),
            "p" => spec.p,
            "q" => spec.q,
            "agg" => String(spec.agg),
            "directions" => spec.directions === nothing ? nothing : _jsonable(spec.directions),
            "n_dirs" => spec.n_dirs,
            "normalize" => String(spec.normalize),
            "enforce_monotone" => String(spec.enforce_monotone),
            "precompute" => spec.precompute,
            "threads" => spec.threads,
        ),
    )
end

function _spec_jsonable(spec::CompositeSpec)
    return Dict(
        "type" => "CompositeSpec",
        "module" => string(parentmodule(typeof(spec))),
        "fields" => Dict(
            "specs" => [_spec_jsonable(s) for s in spec.specs],
            "namespacing" => spec.namespacing,
        ),
    )
end

_spec_jsonable(spec::AbstractFeaturizerSpec) = _jsonable(spec)

function _pkg_version_or_unknown()
    try
        v = Base.pkgversion(parentmodule(@__MODULE__))
        return v === nothing ? "unknown" : string(v)
    catch
        return "unknown"
    end
end

function _git_commit_or_unknown()
    root = normpath(joinpath(@__DIR__, ".."))
    try
        return readchomp(`git -C $root rev-parse --short=12 HEAD`)
    catch
        return "unknown"
    end
end

@inline default_feature_metadata_path(path::AbstractString) = String(path) * ".meta.json"

function feature_metadata(fs::FeatureSet;
                          format::Symbol=:wide,
                          git_commit::Union{Nothing,AbstractString}=nothing)
    spec = _meta_get(fs.meta, :spec, nothing)
    opts = _meta_get(fs.meta, :opts, nothing)
    labels = _meta_get(fs.meta, :labels, nothing)
    unsupported_policy = _meta_get(fs.meta, :unsupported_policy, nothing)
    skipped_indices = _meta_get(fs.meta, :skipped_indices, Int[])
    cache_mode = _meta_get(fs.meta, :cache_mode, nothing)
    threaded = _meta_get(fs.meta, :threaded, nothing)
    gc = git_commit === nothing ? _git_commit_or_unknown() : String(git_commit)
    out = feature_schema_header(format=format)
    out["layout"] = "rows=samples, cols=features"
    out["n_samples"] = nsamples(fs)
    out["n_features"] = nfeatures(fs)
    out["numeric_type"] = string(eltype(fs.X))
    out["feature_names"] = [String(nm) for nm in fs.names]
    spec_axes = spec isa AbstractFeaturizerSpec ? feature_axes(spec) : _meta_get(fs.meta, :feature_axes, nothing)
    out["feature_axes"] = _jsonable(spec_axes)
    out["ids"] = copy(fs.ids)
    out["spec"] = spec isa AbstractFeaturizerSpec ? _spec_jsonable(spec) : _jsonable(spec)
    out["opts"] = _jsonable(opts)
    out["labels"] = _jsonable(labels)
    out["unsupported_policy"] = _jsonable(unsupported_policy)
    out["skipped_indices"] = _jsonable(skipped_indices)
    out["cache_mode"] = _jsonable(cache_mode)
    out["threaded"] = _jsonable(threaded)
    out["package_version"] = _pkg_version_or_unknown()
    out["git_commit"] = gc
    return out
end

function save_metadata_json(path::AbstractString, meta)
    open(path, "w") do io
        JSON3.write(io, _jsonable(meta); allow_inf=true, indent=2)
    end
    return path
end

@inline _obj_haskey(obj::AbstractDict, key::AbstractString) = haskey(obj, key)
@inline _obj_haskey(obj, key::AbstractString) = try
    haskey(obj, key)
catch
    false
end

@inline _obj_get(obj, key::AbstractString, default=nothing) =
    _obj_haskey(obj, key) ? obj[key] : default

@inline _as_symbol(x) = Symbol(String(x))
@inline _as_bool_or_nothing(x) = x === nothing ? nothing : Bool(x)
@inline _as_float_or_nothing(x) = x === nothing ? nothing : Float64(x)

@inline _float_vec(v) = Float64[Float64(x) for x in v]
@inline _vecvec_float(vv) = [Float64[Float64(x) for x in v] for v in vv]

function _axes_from_json(x)
    x === nothing && return nothing
    parts = [Float64[Float64(v) for v in ax] for ax in x]
    return tuple(parts...)
end

function _box_from_json(x)
    x === nothing && return nothing
    if x isa AbstractVector && length(x) == 2 &&
       x[1] isa AbstractVector && x[2] isa AbstractVector
        return (Float64[Float64(v) for v in x[1]], Float64[Float64(v) for v in x[2]])
    end
    return x
end

function _offset_weights_from_json(x)
    x === nothing && return nothing
    x isa AbstractVector || throw(ArgumentError("offset_weights metadata must be a vector/matrix-like JSON array or null"))
    if isempty(x)
        return Float64[]
    end
    if x[1] isa AbstractVector
        nr = length(x)
        nc = length(x[1])
        M = Matrix{Float64}(undef, nr, nc)
        @inbounds for i in 1:nr
            length(x[i]) == nc || throw(ArgumentError("offset_weights rows must have consistent lengths"))
            for j in 1:nc
                M[i, j] = Float64(x[i][j])
            end
        end
        return M
    end
    return Float64[Float64(v) for v in x]
end

function _to_plain(x)
    if x isa AbstractVector
        return Any[_to_plain(v) for v in x]
    elseif x isa AbstractDict
        out = Dict{String,Any}()
        for (k, v) in pairs(x)
            out[String(k)] = _to_plain(v)
        end
        return out
    else
        return x
    end
end

@inline function _spec_fields_dict(meta)
    if _obj_haskey(meta, "fields")
        return meta["fields"]
    end
    return meta
end

"""
    spec_from_metadata(meta_or_spec; resolve_ref=nothing, require_resolved_refs=false) -> AbstractFeaturizerSpec

Reconstruct a typed featurizer spec from metadata JSON payloads produced by
`feature_metadata`/`save_metadata_json`. Accepts either a full metadata object
containing `"spec"` or the spec payload itself.

For `ProjectedDistancesSpec`, pass `resolve_ref=id -> obj` to rehydrate
reference objects from persisted `reference_ids`.
"""
function spec_from_metadata(meta_or_spec;
                            resolve_ref::Union{Nothing,Function}=nothing,
                            require_resolved_refs::Bool=false)::AbstractFeaturizerSpec
    meta_or_spec isa AbstractFeaturizerSpec && return meta_or_spec
    node = _obj_haskey(meta_or_spec, "spec") ? meta_or_spec["spec"] : meta_or_spec
    (node === nothing || !_obj_haskey(node, "type")) &&
        throw(ArgumentError("spec_from_metadata expects a serialized spec payload with a `type` field"))
    Tname = String(node["type"])
    fields = _spec_fields_dict(node)

    if Tname == "LandscapeSpec"
        return LandscapeSpec(
            directions=_vecvec_float(fields["directions"]),
            offsets=_vecvec_float(fields["offsets"]),
            offset_weights=_offset_weights_from_json(_obj_get(fields, "offset_weights", nothing)),
            kmax=Int(fields["kmax"]),
            tgrid=_float_vec(fields["tgrid"]),
            aggregate=_as_symbol(fields["aggregate"]),
            normalize_weights=Bool(fields["normalize_weights"]),
            tmin=_as_float_or_nothing(_obj_get(fields, "tmin", nothing)),
            tmax=_as_float_or_nothing(_obj_get(fields, "tmax", nothing)),
            nsteps=Int(fields["nsteps"]),
            strict=Bool(fields["strict"]),
            drop_unknown=Bool(fields["drop_unknown"]),
            dedup=Bool(fields["dedup"]),
            normalize_dirs=_as_symbol(fields["normalize_dirs"]),
            direction_weight=_as_symbol(fields["direction_weight"]),
            threads=_as_bool_or_nothing(_obj_get(fields, "threads", nothing)),
        )
    elseif Tname == "PersistenceImageSpec"
        return PersistenceImageSpec(
            directions=_vecvec_float(fields["directions"]),
            offsets=_vecvec_float(fields["offsets"]),
            offset_weights=_offset_weights_from_json(_obj_get(fields, "offset_weights", nothing)),
            xgrid=_float_vec(fields["xgrid"]),
            ygrid=_float_vec(fields["ygrid"]),
            sigma=Float64(fields["sigma"]),
            coords=_as_symbol(fields["coords"]),
            weighting=_as_symbol(fields["weighting"]),
            p=Float64(fields["p"]),
            normalize=_as_symbol(fields["normalize"]),
            aggregate=_as_symbol(fields["aggregate"]),
            normalize_weights=Bool(fields["normalize_weights"]),
            tmin=_as_float_or_nothing(_obj_get(fields, "tmin", nothing)),
            tmax=_as_float_or_nothing(_obj_get(fields, "tmax", nothing)),
            nsteps=Int(fields["nsteps"]),
            strict=Bool(fields["strict"]),
            drop_unknown=Bool(fields["drop_unknown"]),
            dedup=Bool(fields["dedup"]),
            normalize_dirs=_as_symbol(fields["normalize_dirs"]),
            direction_weight=_as_symbol(fields["direction_weight"]),
            threads=_as_bool_or_nothing(_obj_get(fields, "threads", nothing)),
        )
    elseif Tname == "EulerSurfaceSpec"
        return EulerSurfaceSpec(
            axes=_axes_from_json(_obj_get(fields, "axes", nothing)),
            axes_policy=_as_symbol(fields["axes_policy"]),
            max_axis_len=Int(fields["max_axis_len"]),
            strict=_as_bool_or_nothing(_obj_get(fields, "strict", nothing)),
            threads=_as_bool_or_nothing(_obj_get(fields, "threads", nothing)),
        )
    elseif Tname == "RankGridSpec"
        return RankGridSpec(
            nvertices=Int(fields["nvertices"]),
            store_zeros=Bool(fields["store_zeros"]),
            threads=_as_bool_or_nothing(_obj_get(fields, "threads", nothing)),
        )
    elseif Tname == "SlicedBarcodeSpec"
        summary_fields = Tuple(_as_symbol(v) for v in fields["summary_fields"])
        return SlicedBarcodeSpec(
            directions=_vecvec_float(fields["directions"]),
            offsets=_vecvec_float(fields["offsets"]),
            offset_weights=_offset_weights_from_json(_obj_get(fields, "offset_weights", nothing)),
            featurizer=_as_symbol(fields["featurizer"]),
            summary_fields=summary_fields,
            summary_normalize_entropy=Bool(fields["summary_normalize_entropy"]),
            entropy_normalize=Bool(fields["entropy_normalize"]),
            entropy_weighting=_as_symbol(fields["entropy_weighting"]),
            entropy_p=Float64(fields["entropy_p"]),
            aggregate=_as_symbol(fields["aggregate"]),
            normalize_weights=Bool(fields["normalize_weights"]),
            tmin=_as_float_or_nothing(_obj_get(fields, "tmin", nothing)),
            tmax=_as_float_or_nothing(_obj_get(fields, "tmax", nothing)),
            nsteps=Int(fields["nsteps"]),
            strict=Bool(fields["strict"]),
            drop_unknown=Bool(fields["drop_unknown"]),
            dedup=Bool(fields["dedup"]),
            normalize_dirs=_as_symbol(fields["normalize_dirs"]),
            direction_weight=_as_symbol(fields["direction_weight"]),
            threads=_as_bool_or_nothing(_obj_get(fields, "threads", nothing)),
        )
    elseif Tname == "SignedBarcodeImageSpec"
        return SignedBarcodeImageSpec(
            xs=_float_vec(fields["xs"]),
            ys=_float_vec(fields["ys"]),
            sigma=Float64(fields["sigma"]),
            mode=_as_symbol(fields["mode"]),
            method=_as_symbol(fields["method"]),
            axes=_axes_from_json(_obj_get(fields, "axes", nothing)),
            axes_policy=_as_symbol(fields["axes_policy"]),
            max_axis_len=Int(fields["max_axis_len"]),
            strict=_as_bool_or_nothing(_obj_get(fields, "strict", nothing)),
            threads=_as_bool_or_nothing(_obj_get(fields, "threads", nothing)),
        )
    elseif Tname == "ProjectedDistancesSpec"
        refs_raw = _obj_get(fields, "references", Any[])
        names = Symbol[_as_symbol(nm) for nm in fields["reference_names"]]
        ref_ids = String[string(x) for x in _obj_get(fields, "reference_ids", String[string(nm) for nm in names])]
        refs = if resolve_ref === nothing
            Any[_to_plain(r) for r in refs_raw]
        else
            out = Vector{Any}(undef, length(ref_ids))
            @inbounds for i in eachindex(ref_ids)
                resolved = resolve_ref(ref_ids[i])
                if resolved === nothing
                    require_resolved_refs &&
                        throw(ArgumentError("spec_from_metadata: unresolved projected reference id $(ref_ids[i])"))
                    if i <= length(refs_raw)
                        out[i] = _to_plain(refs_raw[i])
                    else
                        out[i] = Dict("kind" => "missing_reference", "id" => ref_ids[i])
                    end
                else
                    out[i] = resolved
                end
            end
            out
        end
        dirs = _obj_get(fields, "directions", nothing)
        return ProjectedDistancesSpec(
            refs;
            reference_names=names,
            dist=_as_symbol(fields["dist"]),
            p=Float64(fields["p"]),
            q=Float64(fields["q"]),
            agg=_as_symbol(fields["agg"]),
            directions=dirs === nothing ? nothing : _vecvec_float(dirs),
            n_dirs=Int(fields["n_dirs"]),
            normalize=_as_symbol(fields["normalize"]),
            enforce_monotone=_as_symbol(fields["enforce_monotone"]),
            precompute=Bool(fields["precompute"]),
            threads=_as_bool_or_nothing(_obj_get(fields, "threads", nothing)),
        )
    elseif Tname == "CompositeSpec"
        specs_raw = _obj_get(fields, "specs", Any[])
        sub_specs = map(s -> spec_from_metadata(s;
                                                resolve_ref=resolve_ref,
                                                require_resolved_refs=require_resolved_refs),
                        specs_raw)
        return CompositeSpec(tuple(sub_specs...); namespacing=Bool(_obj_get(fields, "namespacing", true)))
    else
        throw(ArgumentError("Unsupported featurizer spec type in metadata: $(Tname)"))
    end
end

"""
    load_spec_with_resolver(meta_or_spec, resolve_ref; require_all=true) -> AbstractFeaturizerSpec

Typed spec round-trip with explicit projected-reference rehydration callback.
`resolve_ref` is called as `resolve_ref(reference_id::String)` for each projected
reference id encoded in metadata.
"""
function load_spec_with_resolver(meta_or_spec,
                                 resolve_ref::Function;
                                 require_all::Bool=true)::AbstractFeaturizerSpec
    return spec_from_metadata(meta_or_spec;
                              resolve_ref=resolve_ref,
                              require_resolved_refs=require_all)
end

"""
    invariant_options_from_metadata(meta_or_opts) -> InvariantOptions

Reconstruct a typed `InvariantOptions` object from metadata payloads. Accepts
either a full metadata object containing `"opts"` or the serialized options.
"""
function invariant_options_from_metadata(meta_or_opts)::InvariantOptions
    meta_or_opts isa InvariantOptions && return meta_or_opts
    node = _obj_haskey(meta_or_opts, "opts") ? meta_or_opts["opts"] : meta_or_opts
    node === nothing && return InvariantOptions()
    fields = _spec_fields_dict(node)
    return InvariantOptions(
        axes=_axes_from_json(_obj_get(fields, "axes", nothing)),
        axes_policy=_as_symbol(_obj_get(fields, "axes_policy", :encoding)),
        max_axis_len=Int(_obj_get(fields, "max_axis_len", 256)),
        box=_box_from_json(_obj_get(fields, "box", nothing)),
        threads=_as_bool_or_nothing(_obj_get(fields, "threads", nothing)),
        strict=_as_bool_or_nothing(_obj_get(fields, "strict", nothing)),
        pl_mode=_as_symbol(_obj_get(fields, "pl_mode", :fast)),
    )
end

function load_metadata_json(path::AbstractString;
                            validate_feature_schema::Bool=false,
                            typed::Bool=false,
                            resolve_ref::Union{Nothing,Function}=nothing,
                            require_resolved_refs::Bool=false)
    obj = open(JSON3.read, path)
    if validate_feature_schema
        validate_feature_metadata_schema(obj)
    end
    if !typed
        return obj
    end
    spec = _obj_haskey(obj, "spec") ? spec_from_metadata(obj["spec"];
                                                          resolve_ref=resolve_ref,
                                                          require_resolved_refs=require_resolved_refs) : nothing
    opts = _obj_haskey(obj, "opts") ? invariant_options_from_metadata(obj["opts"]) : InvariantOptions()
    return (raw=obj, spec=spec, opts=opts)
end

function _featureset_from_wide_columntable(cols; ids_col::Symbol=:id, meta=NamedTuple())
    names_nt = propertynames(cols)
    ids_col in names_nt || throw(ArgumentError("wide feature table must include id column :$(ids_col)"))
    ids_raw = getproperty(cols, ids_col)
    ids = String[string(x) for x in ids_raw]
    ns = length(ids)
    feat_syms = [nm for nm in names_nt if nm != ids_col]
    nf = length(feat_syms)
    nf == 0 && return FeatureSet(zeros(Float64, ns, 0), Symbol[], ids, meta)

    colvecs = Vector{AbstractVector}(undef, nf)
    T = Union{}
    @inbounds for j in 1:nf
        vj = collect(getproperty(cols, feat_syms[j]))
        length(vj) == ns || throw(DimensionMismatch("column $(feat_syms[j]) has length $(length(vj)); expected $ns"))
        colvecs[j] = vj
        T = j == 1 ? eltype(vj) : promote_type(T, eltype(vj))
    end

    X = Matrix{T}(undef, ns, nf)
    @inbounds for j in 1:nf
        copyto!(view(X, :, j), colvecs[j])
    end
    return FeatureSet(X, Symbol.(feat_syms), ids, meta)
end

function _featureset_from_long_columntable(cols; meta=NamedTuple())
    names_nt = propertynames(cols)
    (:id in names_nt && :feature in names_nt && :value in names_nt) ||
        throw(ArgumentError("long feature table must include columns :id, :feature, :value"))
    id_col = collect(getproperty(cols, :id))
    feat_col = collect(getproperty(cols, :feature))
    val_col = collect(getproperty(cols, :value))
    n = length(id_col)
    (length(feat_col) == n && length(val_col) == n) ||
        throw(DimensionMismatch("long feature table columns must have equal lengths"))

    names = Symbol[]
    feat_pos = Dict{Symbol,Int}()
    if :sample_index in names_nt
        sidx_col = collect(getproperty(cols, :sample_index))
        length(sidx_col) == n || throw(DimensionMismatch("sample_index column must match long table length"))

        ns = 0
        @inbounds for s in sidx_col
            si = Int(s)
            si >= 1 || throw(ArgumentError("sample_index values must be >= 1"))
            ns = max(ns, si)
        end
        ids = ["" for _ in 1:ns]

        @inbounds for k in 1:n
            fk = Symbol(feat_col[k])
            if !haskey(feat_pos, fk)
                push!(names, fk)
                feat_pos[fk] = length(names)
            end
            i = Int(sidx_col[k])
            if isempty(ids[i])
                ids[i] = string(id_col[k])
            end
        end
        @inbounds for i in eachindex(ids)
            isempty(ids[i]) && (ids[i] = string(i))
        end

        nf = length(names)
        T = eltype(val_col)
        has_missing = Missing <: T
        X = has_missing ? Matrix{Union{Missing,Base.nonmissingtype(T)}}(missing, ns, nf) :
                          zeros(T, ns, nf)
        @inbounds for k in 1:n
            i = Int(sidx_col[k])
            j = feat_pos[Symbol(feat_col[k])]
            X[i, j] = val_col[k]
        end
        return FeatureSet(X, names, ids, meta)
    else
        ids = String[]
        id_pos = Dict{String,Int}()
        @inbounds for k in 1:n
            idk = string(id_col[k])
            if !haskey(id_pos, idk)
                push!(ids, idk)
                id_pos[idk] = length(ids)
            end
            fk = Symbol(feat_col[k])
            if !haskey(feat_pos, fk)
                push!(names, fk)
                feat_pos[fk] = length(names)
            end
        end

        ns = length(ids)
        nf = length(names)
        T = eltype(val_col)
        has_missing = Missing <: T
        X = has_missing ? Matrix{Union{Missing,Base.nonmissingtype(T)}}(missing, ns, nf) :
                          zeros(T, ns, nf)
        @inbounds for k in 1:n
            i = id_pos[string(id_col[k])]
            j = feat_pos[Symbol(feat_col[k])]
            X[i, j] = val_col[k]
        end
        return FeatureSet(X, names, ids, meta)
    end
end

function _featureset_from_columntable(cols;
                                      format::Symbol=:wide,
                                      ids_col::Symbol=:id,
                                      meta=NamedTuple())
    if format == :wide
        return _featureset_from_wide_columntable(cols; ids_col=ids_col, meta=meta)
    elseif format == :long
        return _featureset_from_long_columntable(cols; meta=meta)
    else
        throw(ArgumentError("_featureset_from_columntable: format must be :wide or :long"))
    end
end

function save_features_arrow(path, fs; kwargs...)
    throw(ArgumentError("save_features_arrow requires Arrow.jl extension (load Arrow and ensure TamerOpArrowExt is available)."))
end

function load_features_arrow(path; kwargs...)
    throw(ArgumentError("load_features_arrow requires Arrow.jl extension (load Arrow and ensure TamerOpArrowExt is available)."))
end

function save_features_parquet(path, fs; kwargs...)
    throw(ArgumentError("save_features_parquet requires Parquet2.jl extension (load Parquet2 and ensure TamerOpParquet2Ext is available)."))
end

function load_features_parquet(path; kwargs...)
    throw(ArgumentError("load_features_parquet requires Parquet2.jl extension (load Parquet2 and ensure TamerOpParquet2Ext is available)."))
end

function save_features_npz(path, fs; kwargs...)
    throw(ArgumentError("save_features_npz requires NPZ.jl extension (load NPZ and ensure TamerOpNPZExt is available)."))
end

function load_features_npz(path; kwargs...)
    throw(ArgumentError("load_features_npz requires NPZ.jl extension (load NPZ and ensure TamerOpNPZExt is available)."))
end

function save_features_csv(path, fs; kwargs...)
    throw(ArgumentError("save_features_csv requires CSV.jl extension (load CSV and ensure TamerOpCSVExt is available)."))
end

function load_features_csv(path; kwargs...)
    throw(ArgumentError("load_features_csv requires CSV.jl extension (load CSV and ensure TamerOpCSVExt is available)."))
end

@inline function _interop_format_from_path(path::AbstractString)
    ext = lowercase(splitext(String(path))[2])
    if ext == ".arrow"
        return :arrow
    elseif ext == ".parquet" || ext == ".pq"
        return :parquet
    elseif ext == ".npz"
        return :npz
    elseif ext == ".csv"
        return :csv
    end
    return nothing
end

@inline function _resolve_interop_format(path::AbstractString, format::Symbol)
    if format === :auto
        fmt = _interop_format_from_path(path)
        fmt === nothing && throw(ArgumentError("save/load_features with format=:auto requires path extension in (.arrow, .parquet, .npz, .csv)."))
        return fmt
    end
    return format
end

@inline function _normalize_feature_mode(mode::Symbol)
    (mode === :wide || mode === :long) && return mode
    throw(ArgumentError("feature mode must be :wide or :long (got $(mode))"))
end

@inline function _normalize_feature_layout(layout::Symbol)
    (layout === :samples_by_features || layout === :features_by_samples) && return layout
    throw(ArgumentError("feature layout must be :samples_by_features or :features_by_samples (got $(layout))"))
end

"""
    save_features(path, fs::FeatureSet; format=:auto, layout=:samples_by_features,
                  mode=:wide, metadata=true, kwargs...)

Unified feature artifact writer. Dispatches to Arrow/Parquet/NPZ/CSV backend writers.
Actual backend implementations live in package extensions.
"""
function save_features(path::AbstractString,
                       fs::FeatureSet;
                       format::Symbol=:auto,
                       layout::Symbol=:samples_by_features,
                       mode::Symbol=:wide,
                       metadata::Bool=true,
                       kwargs...)
    fmt = _resolve_interop_format(path, format)
    mode0 = _normalize_feature_mode(mode)
    layout0 = _normalize_feature_layout(layout)

    if (fmt === :arrow || fmt === :parquet) && layout0 !== :samples_by_features
        throw(ArgumentError("layout=$(layout0) is not supported for $(fmt); use :samples_by_features"))
    end

    if fmt === :arrow
        return save_features_arrow(path, fs; format=mode0, include_metadata=metadata, kwargs...)
    elseif fmt === :parquet
        return save_features_parquet(path, fs; format=mode0, include_metadata=metadata, kwargs...)
    elseif fmt === :npz
        return save_features_npz(path, fs; format=mode0, layout=layout0, include_metadata=metadata, kwargs...)
    elseif fmt === :csv
        return save_features_csv(path, fs; format=mode0, layout=layout0, include_metadata=metadata, kwargs...)
    end
    throw(ArgumentError("Unsupported feature format: $(fmt). Supported: :arrow, :parquet, :npz, :csv"))
end

"""
    load_features(path; format=:auto, layout=:auto, mode=:wide, kwargs...)

Unified feature artifact loader. Dispatches to Arrow/Parquet/NPZ/CSV backend loaders.
Actual backend implementations live in package extensions.
"""
function load_features(path::AbstractString;
                       format::Symbol=:auto,
                       layout::Symbol=:auto,
                       mode::Symbol=:wide,
                       kwargs...)
    fmt = _resolve_interop_format(path, format)
    mode0 = _normalize_feature_mode(mode)
    layout0 = layout === :auto ? nothing : _normalize_feature_layout(layout)

    if (fmt === :arrow || fmt === :parquet) &&
       !(layout0 === nothing || layout0 === :samples_by_features)
        throw(ArgumentError("layout=$(layout0) is not supported for $(fmt); use :samples_by_features"))
    end

    if fmt === :arrow
        return load_features_arrow(path; format=mode0, kwargs...)
    elseif fmt === :parquet
        return load_features_parquet(path; format=mode0, kwargs...)
    elseif fmt === :npz
        return load_features_npz(path; format=mode0, layout=layout0, kwargs...)
    elseif fmt === :csv
        return load_features_csv(path; format=mode0, layout=layout0, kwargs...)
    end
    throw(ArgumentError("Unsupported feature format: $(fmt). Supported: :arrow, :parquet, :npz, :csv"))
end

"""
    EulerSurfaceLongTable(values; axes=nothing, id="sample")

Long-form table wrapper for 2D Euler surfaces.
"""
struct EulerSurfaceLongTable{T<:Real}
    id::String
    x::Vector{Float64}
    y::Vector{Float64}
    values::Matrix{T}
end

"""
    PersistenceImageLongTable(pi; id="sample")

Long-form table wrapper for `PersistenceImage1D`.
"""
struct PersistenceImageLongTable
    id::String
    image::Invariants.PersistenceImage1D
end

"""
    MPLandscapeLongTable(L; id="sample")

Long-form table wrapper for `MPLandscape`.
"""
struct MPLandscapeLongTable{D,O}
    id::String
    landscape::Invariants.MPLandscape{D,O}
end

"""
    PointSignedMeasureLongTable(pm; id="sample")

Long-form table wrapper for signed point measures.
"""
struct PointSignedMeasureLongTable{N,T,W}
    id::String
    measure::Invariants.PointSignedMeasure{N,T,W}
end

@inline feature_table(fs::FeatureSet; format::Symbol=:wide) =
    format == :wide ? FeatureSetWideTable(fs) :
    format == :long ? FeatureSetLongTable(fs) :
    throw(ArgumentError("feature_table: format must be :wide or :long"))

function euler_surface_table(values::AbstractMatrix{T};
                             axes=nothing,
                             id::AbstractString="sample") where {T<:Real}
    if axes === nothing
        x = Float64[i for i in 1:size(values, 1)]
        y = Float64[j for j in 1:size(values, 2)]
    else
        length(axes) == 2 || throw(ArgumentError("euler_surface_table: axes must be a 2-tuple"))
        x = Float64[float(v) for v in axes[1]]
        y = Float64[float(v) for v in axes[2]]
        length(x) == size(values, 1) || throw(DimensionMismatch("x-axis length does not match first dimension"))
        length(y) == size(values, 2) || throw(DimensionMismatch("y-axis length does not match second dimension"))
    end
    return EulerSurfaceLongTable{T}(String(id), x, y, Matrix(values))
end

@inline persistence_image_table(pi::Invariants.PersistenceImage1D; id::AbstractString="sample") =
    PersistenceImageLongTable(String(id), pi)

@inline mp_landscape_table(L::Invariants.MPLandscape{D,O}; id::AbstractString="sample") where {D,O} =
    MPLandscapeLongTable{D,O}(String(id), L)

@inline point_signed_measure_table(pm::Invariants.PointSignedMeasure{N,T,W}; id::AbstractString="sample") where {N,T,W} =
    PointSignedMeasureLongTable{N,T,W}(String(id), pm)

@inline _default_threads_flag(flag::Union{Nothing,Bool}) =
    flag === nothing ? (Threads.nthreads() > 1) : flag

@inline function _resolve_spec_threads(spec_threads::Union{Nothing,Bool},
                                       opts::InvariantOptions,
                                       threaded_default::Bool)
    if spec_threads !== nothing
        return spec_threads
    end
    if opts.threads !== nothing
        return opts.threads
    end
    return threaded_default
end

@inline _float_vector(v::Real) = Float64[float(v)]
@inline _float_vector(v::AbstractVector) = Float64[float(x) for x in v]
@inline _float_vector(v::AbstractArray) = Float64[float(x) for x in vec(v)]

function _to_vecvec_float(xs)
    out = Vector{Vector{Float64}}(undef, length(xs))
    @inbounds for i in eachindex(xs)
        out[i] = Float64[float(x) for x in xs[i]]
    end
    return out
end

function _to_offset_weights(weights)
    weights === nothing && return nothing
    if weights isa AbstractVector
        return Float64[float(x) for x in weights]
    elseif weights isa AbstractMatrix
        return Float64[float(x) for x in weights]
    else
        throw(ArgumentError("offset_weights must be a vector, matrix, or nothing"))
    end
end

function LandscapeSpec(; directions,
                        offsets,
                        offset_weights=nothing,
                        kmax::Int=5,
                        tgrid,
                        aggregate::Symbol=:mean,
                        normalize_weights::Bool=true,
                        tmin=nothing,
                        tmax=nothing,
                        nsteps::Int=401,
                        strict::Bool=true,
                        drop_unknown::Bool=true,
                        dedup::Bool=true,
                        normalize_dirs::Symbol=:none,
                        direction_weight::Symbol=:none,
                        threads=nothing)
    return LandscapeSpec(
        _to_vecvec_float(directions),
        _to_vecvec_float(offsets),
        _to_offset_weights(offset_weights),
        kmax,
        Float64[float(x) for x in tgrid],
        aggregate,
        normalize_weights,
        tmin === nothing ? nothing : float(tmin),
        tmax === nothing ? nothing : float(tmax),
        nsteps,
        strict,
        drop_unknown,
        dedup,
        normalize_dirs,
        direction_weight,
        threads === nothing ? nothing : Bool(threads),
    )
end

function PersistenceImageSpec(; directions,
                               offsets,
                               offset_weights=nothing,
                               xgrid,
                               ygrid,
                               sigma::Real=1.0,
                               coords::Symbol=:birth_persistence,
                               weighting::Symbol=:persistence,
                               p::Real=1.0,
                               normalize::Symbol=:none,
                               aggregate::Symbol=:mean,
                               normalize_weights::Bool=true,
                               tmin=nothing,
                               tmax=nothing,
                               nsteps::Int=401,
                               strict::Bool=true,
                               drop_unknown::Bool=true,
                               dedup::Bool=true,
                               normalize_dirs::Symbol=:none,
                               direction_weight::Symbol=:none,
                               threads=nothing)
    return PersistenceImageSpec(
        _to_vecvec_float(directions),
        _to_vecvec_float(offsets),
        _to_offset_weights(offset_weights),
        Float64[float(x) for x in xgrid],
        Float64[float(y) for y in ygrid],
        float(sigma),
        coords,
        weighting,
        float(p),
        normalize,
        aggregate,
        normalize_weights,
        tmin === nothing ? nothing : float(tmin),
        tmax === nothing ? nothing : float(tmax),
        nsteps,
        strict,
        drop_unknown,
        dedup,
        normalize_dirs,
        direction_weight,
        threads === nothing ? nothing : Bool(threads),
    )
end

function EulerSurfaceSpec(; axes=nothing,
                           axes_policy::Symbol=:encoding,
                           max_axis_len::Int=256,
                           strict=nothing,
                           threads=nothing)
    axes2 = axes === nothing ? nothing :
        ntuple(i -> Float64[float(x) for x in axes[i]], length(axes))
    strict2 = strict === nothing ? nothing : Bool(strict)
    threads2 = threads === nothing ? nothing : Bool(threads)
    return EulerSurfaceSpec{typeof(axes2)}(axes2, axes_policy, max_axis_len, strict2, threads2)
end

RankGridSpec(; nvertices::Int, store_zeros::Bool=true, threads=nothing) =
    RankGridSpec(nvertices, store_zeros, threads === nothing ? nothing : Bool(threads))

function SlicedBarcodeSpec(; directions,
                            offsets,
                            offset_weights=nothing,
                            featurizer::Symbol=:summary,
                            summary_fields::Tuple=Invariants._DEFAULT_BARCODE_SUMMARY_FIELDS,
                            summary_normalize_entropy::Bool=true,
                            entropy_normalize::Bool=true,
                            entropy_weighting::Symbol=:persistence,
                            entropy_p::Real=1.0,
                            aggregate::Symbol=:mean,
                            normalize_weights::Bool=true,
                            tmin=nothing,
                            tmax=nothing,
                            nsteps::Int=401,
                            strict::Bool=true,
                            drop_unknown::Bool=true,
                            dedup::Bool=true,
                            normalize_dirs::Symbol=:none,
                            direction_weight::Symbol=:none,
                            threads=nothing)
    return SlicedBarcodeSpec(
        _to_vecvec_float(directions),
        _to_vecvec_float(offsets),
        _to_offset_weights(offset_weights),
        featurizer,
        Tuple(summary_fields),
        summary_normalize_entropy,
        entropy_normalize,
        entropy_weighting,
        float(entropy_p),
        aggregate,
        normalize_weights,
        tmin === nothing ? nothing : float(tmin),
        tmax === nothing ? nothing : float(tmax),
        nsteps,
        strict,
        drop_unknown,
        dedup,
        normalize_dirs,
        direction_weight,
        threads === nothing ? nothing : Bool(threads),
    )
end

function SignedBarcodeImageSpec(; xs,
                                 ys,
                                 sigma::Real=1.0,
                                 mode::Symbol=:center,
                                 method::Symbol=:bulk,
                                 axes=nothing,
                                 axes_policy::Symbol=:encoding,
                                 max_axis_len::Int=256,
                                 strict=nothing,
                                 threads=nothing)
    axes2 = axes === nothing ? nothing :
        ntuple(i -> Float64[float(x) for x in axes[i]], length(axes))
    strict2 = strict === nothing ? nothing : Bool(strict)
    threads2 = threads === nothing ? nothing : Bool(threads)
    return SignedBarcodeImageSpec{typeof(axes2)}(
        Float64[float(x) for x in xs],
        Float64[float(y) for y in ys],
        float(sigma),
        mode,
        method,
        axes2,
        axes_policy,
        max_axis_len,
        strict2,
        threads2,
    )
end

function ProjectedDistancesSpec(references;
                                reference_names=nothing,
                                dist::Symbol=:bottleneck,
                                p::Real=1.0,
                                q::Real=1.0,
                                agg::Symbol=:mean,
                                directions=nothing,
                                n_dirs::Int=32,
                                normalize::Symbol=:L1,
                                enforce_monotone::Symbol=:upper,
                                precompute::Bool=true,
                                threads=nothing)
    refs = collect(references)
    names = if reference_names === nothing
        [Symbol("ref_$(i)") for i in 1:length(refs)]
    else
        Symbol[Symbol(x) for x in reference_names]
    end
    length(names) == length(refs) || throw(ArgumentError("reference_names length must match references length"))
    dirs2 = directions === nothing ? nothing : _to_vecvec_float(directions)
    return ProjectedDistancesSpec{typeof(refs),typeof(dirs2)}(
        refs,
        names,
        dist,
        float(p),
        float(q),
        agg,
        dirs2,
        n_dirs,
        normalize,
        enforce_monotone,
        precompute,
        threads === nothing ? nothing : Bool(threads),
    )
end

CompositeSpec(specs::Tuple; namespacing::Bool=true) = CompositeSpec{typeof(specs)}(specs, namespacing)

function _slice_feature_names(prefix::String, per_slice::Int, nd::Int, no::Int, aggregate::Symbol)
    if aggregate == :stack
        out = Vector{Symbol}(undef, per_slice * nd * no)
        idx = 1
        for i in 1:nd, j in 1:no, k in 1:per_slice
            out[idx] = Symbol(prefix * "__d$(i)_o$(j)_f$(k)")
            idx += 1
        end
        return out
    else
        return [Symbol(prefix * "__f$(k)") for k in 1:per_slice]
    end
end

feature_axes(::AbstractFeaturizerSpec) = NamedTuple()

@inline _copy_vecvec(xs::Vector{Vector{Float64}}) = [copy(v) for v in xs]

function _axis_namedtuple(axes::NTuple{N,Vector{Float64}}, prefix::String="axis") where {N}
    names = ntuple(i -> Symbol(prefix * "_" * string(i)), N)
    vals = ntuple(i -> copy(axes[i]), N)
    return NamedTuple{names}(vals)
end

function feature_axes(spec::LandscapeSpec)
    base = (k=collect(1:spec.kmax), t=copy(spec.tgrid))
    if spec.aggregate == :stack
        return (direction=collect(1:length(spec.directions)),
                offset=collect(1:length(spec.offsets)),
                base...)
    end
    return (aggregate=spec.aggregate,
            base...,
            directions=_copy_vecvec(spec.directions),
            offsets=_copy_vecvec(spec.offsets))
end

feature_names(spec::LandscapeSpec) =
    _slice_feature_names("landscape", spec.kmax * length(spec.tgrid),
                         length(spec.directions), length(spec.offsets), spec.aggregate)

nfeatures(spec::LandscapeSpec) = length(feature_names(spec))

function feature_axes(spec::PersistenceImageSpec)
    base = (x=copy(spec.xgrid), y=copy(spec.ygrid))
    if spec.aggregate == :stack
        return (direction=collect(1:length(spec.directions)),
                offset=collect(1:length(spec.offsets)),
                base...)
    end
    return (aggregate=spec.aggregate,
            base...,
            directions=_copy_vecvec(spec.directions),
            offsets=_copy_vecvec(spec.offsets))
end

feature_names(spec::PersistenceImageSpec) =
    _slice_feature_names("persistence_image", length(spec.xgrid) * length(spec.ygrid),
                         length(spec.directions), length(spec.offsets), spec.aggregate)

nfeatures(spec::PersistenceImageSpec) = length(feature_names(spec))

function feature_axes(spec::EulerSurfaceSpec)
    spec.axes === nothing && return (axes_policy=spec.axes_policy, axes=nothing)
    return (_axis_namedtuple(spec.axes)..., axes_policy=spec.axes_policy)
end

function feature_names(spec::EulerSurfaceSpec)
    spec.axes === nothing && return Symbol[]
    nf = prod(length(ax) for ax in spec.axes)
    return [Symbol("euler__f$(i)") for i in 1:nf]
end

nfeatures(spec::EulerSurfaceSpec) = length(feature_names(spec))

feature_axes(spec::RankGridSpec) =
    (a=collect(1:spec.nvertices), b=collect(1:spec.nvertices), store_zeros=spec.store_zeros)

feature_names(spec::RankGridSpec) = [Symbol("rank__a$(a)_b$(b)") for a in 1:spec.nvertices for b in 1:spec.nvertices]
nfeatures(spec::RankGridSpec) = length(feature_names(spec))

@inline function _sliced_stat_axis(spec::SlicedBarcodeSpec)
    if spec.featurizer == :summary
        return Tuple(Symbol(s) for s in spec.summary_fields)
    elseif spec.featurizer == :entropy
        return (:entropy,)
    end
    return ()
end

function feature_axes(spec::SlicedBarcodeSpec)
    stats = _sliced_stat_axis(spec)
    if spec.aggregate == :stack
        return (direction=collect(1:length(spec.directions)),
                offset=collect(1:length(spec.offsets)),
                stat=stats)
    end
    return (aggregate=spec.aggregate,
            stat=stats,
            directions=_copy_vecvec(spec.directions),
            offsets=_copy_vecvec(spec.offsets))
end

function feature_names(spec::SlicedBarcodeSpec)
    base = if spec.featurizer == :summary
        length(spec.summary_fields)
    elseif spec.featurizer == :entropy
        1
    else
        throw(ArgumentError("SlicedBarcodeSpec supports featurizer=:summary or :entropy"))
    end
    return _slice_feature_names("sliced_barcode", base,
                                length(spec.directions), length(spec.offsets), spec.aggregate)
end

nfeatures(spec::SlicedBarcodeSpec) = length(feature_names(spec))

function feature_axes(spec::SignedBarcodeImageSpec)
    out = (x=copy(spec.xs), y=copy(spec.ys), mode=spec.mode, method=spec.method)
    spec.axes === nothing && return out
    return (out..., _axis_namedtuple(spec.axes, "source_axis")..., axes_policy=spec.axes_policy)
end

feature_names(spec::SignedBarcodeImageSpec) =
    [Symbol("signed_barcode_image__x$(i)_y$(j)") for i in 1:length(spec.xs) for j in 1:length(spec.ys)]

nfeatures(spec::SignedBarcodeImageSpec) = length(feature_names(spec))

function feature_axes(spec::ProjectedDistancesSpec)
    out = (reference=copy(spec.reference_names), dist=spec.dist, agg=spec.agg)
    if spec.directions === nothing
        return out
    end
    return (out..., direction=collect(1:length(spec.directions)))
end

feature_names(spec::ProjectedDistancesSpec) = copy(spec.reference_names)
nfeatures(spec::ProjectedDistancesSpec) = length(spec.reference_names)

function feature_axes(spec::CompositeSpec)
    comps = Vector{Any}(undef, length(spec.specs))
    start_idx = 1
    @inbounds for i in eachindex(spec.specs)
        sub = spec.specs[i]
        nsub = nfeatures(sub)
        stop_idx = start_idx + nsub - 1
        comps[i] = (
            name = spec.namespacing ? string(nameof(typeof(sub))) : string("component_", i),
            spec_type = string(nameof(typeof(sub))),
            start = start_idx,
            stop = stop_idx,
            axes = feature_axes(sub),
        )
        start_idx = stop_idx + 1
    end
    return (namespacing=spec.namespacing, components=comps)
end

function feature_names(spec::CompositeSpec)
    out = Symbol[]
    for (i, sub) in enumerate(spec.specs)
        names = feature_names(sub)
        if spec.namespacing
            prefix = string(nameof(typeof(sub)))
            append!(out, Symbol(prefix * "__" * String(n)) for n in names)
        else
            append!(out, names)
        end
    end
    return out
end

nfeatures(spec::CompositeSpec) = sum(nfeatures(s) for s in spec.specs)

@inline function _sample_module(obj)
    if obj isa EncodingResult
        return obj.M
    elseif obj isa PModule
        return obj
    elseif obj isa Tuple && length(obj) >= 1 && obj[1] isa PModule
        return obj[1]
    elseif obj isa NamedTuple && hasproperty(obj, :M) && getproperty(obj, :M) isa PModule
        return getproperty(obj, :M)
    end
    throw(ArgumentError("sample must provide a PModule (EncodingResult, PModule, or (M,pi))"))
end

@inline function _sample_module_pi(obj)
    if obj isa EncodingResult
        return obj.M, obj.pi
    elseif obj isa Tuple && length(obj) >= 2 && obj[1] isa PModule
        return obj[1], obj[2]
    elseif obj isa NamedTuple && hasproperty(obj, :M) && hasproperty(obj, :pi)
        M = getproperty(obj, :M)
        M isa PModule || throw(ArgumentError("NamedTuple sample field :M must be PModule"))
        return M, getproperty(obj, :pi)
    end
    throw(ArgumentError("sample must provide both module and encoding map (EncodingResult or (M,pi))"))
end

@inline _supports_module(obj) = (obj isa EncodingResult) || (obj isa PModule) ||
    (obj isa Tuple && length(obj) >= 1 && obj[1] isa PModule) ||
    (obj isa NamedTuple && hasproperty(obj, :M) && getproperty(obj, :M) isa PModule)

@inline _supports_module_pi(obj) = (obj isa EncodingResult) ||
    (obj isa Tuple && length(obj) >= 2 && obj[1] isa PModule) ||
    (obj isa NamedTuple && hasproperty(obj, :M) && hasproperty(obj, :pi) && getproperty(obj, :M) isa PModule)

"""
    supports(spec, sample) -> Bool

Return whether a featurizer spec supports a sample without throwing.
"""
supports(::LandscapeSpec, obj) = _supports_module_pi(obj)
supports(::PersistenceImageSpec, obj) = _supports_module_pi(obj)
supports(::EulerSurfaceSpec, obj) = _supports_module_pi(obj)
supports(::RankGridSpec, obj) = _supports_module(obj)
supports(::SlicedBarcodeSpec, obj) = _supports_module_pi(obj)

function supports(::SignedBarcodeImageSpec, obj)
    _supports_module_pi(obj) || return false
    _, pi = _sample_module_pi(obj)
    pi0 = pi isa CompiledEncoding ? pi.pi : pi
    return pi0 isa ZnEncoding.ZnEncodingMap
end

function supports(spec::ProjectedDistancesSpec, obj)
    _supports_module_pi(obj) || return false
    @inbounds for r in spec.references
        _supports_module_pi(r) || return false
    end
    return true
end

function supports(spec::CompositeSpec, obj)
    @inbounds for s in spec.specs
        supports(s, obj) || return false
    end
    return true
end

@inline _cache_sample(cache::ModuleInvariantCache) = cache.M
@inline _cache_sample(cache::RestrictedHilbertInvariantCache) = (cache.M, cache.pi)
@inline _cache_sample(cache::EncodingInvariantCache) = (cache.M, cache.pi)

@inline function _default_invariant_opts(opts::InvariantOptions)
    return opts.axes === nothing &&
           opts.axes_policy === :encoding &&
           opts.max_axis_len == 256 &&
           opts.box === nothing &&
           opts.threads === nothing &&
           opts.strict === nothing &&
           opts.pl_mode === :fast
end

@inline _cache_opts(cache::AbstractInvariantCache, opts::InvariantOptions) =
    (_default_invariant_opts(opts) ? cache.opts : opts)
@inline _cache_opts(cache::AbstractInvariantCache, ::Nothing) = cache.opts

@inline function _cache_threaded(cache::AbstractInvariantCache, threaded::Bool)
    return cache.threads ? threaded : false
end

@inline function _validate_cache_level(level::Symbol)
    level in (:auto, :none, :slice, :projected, :fibered, :all) ||
        throw(ArgumentError("cache level must be one of :auto, :none, :slice, :projected, :fibered, :all"))
    return level
end

@inline _unwrap_cache_pi(pi) = pi isa CompiledEncoding ? pi.pi : pi

@inline function _supports_fibered_cache(pi)
    pi0 = _unwrap_cache_pi(pi)
    return pi0 isa Invariants.PLikeEncodingMap &&
           hasproperty(pi0, :n) &&
           Int(getproperty(pi0, :n)) == 2
end

@inline function _effective_cache_level(level::Symbol, spec::AbstractFeaturizerSpec, pi=nothing)
    level == :auto || return level
    if spec isa ProjectedDistancesSpec
        return :projected
    elseif spec isa SignedBarcodeImageSpec
        return :all
    elseif spec isa Union{LandscapeSpec,PersistenceImageSpec,SlicedBarcodeSpec}
        # Keep semantics stable by default; fibered path is opt-in via level=:fibered/:all.
        return :slice
    elseif spec isa CompositeSpec
        return :all
    end
    return :none
end

function build_cache(M::PModule{K,F,MatT};
                     opts::InvariantOptions=InvariantOptions(),
                     level::Symbol=:auto,
                     threaded::Bool=true) where {K,F<:AbstractCoeffField,MatT<:AbstractMatrix{K}}
    _validate_cache_level(level)
    opts0 = opts
    return ModuleInvariantCache{K,F,MatT}(M, opts0, threaded, level)
end

function build_cache(M::PModule{K,F,MatT}, pi;
                     opts::InvariantOptions=InvariantOptions(),
                     level::Symbol=:auto,
                     threaded::Bool=true,
                     cache=:auto) where {K,F<:AbstractCoeffField,MatT<:AbstractMatrix{K}}
    _validate_cache_level(level)
    opts0 = opts
    session_cache = _resolve_workflow_session_cache(cache)
    spcache = _slice_plan_cache_from_session(nothing, session_cache)
    if spcache === nothing
        spcache = Invariants.SlicePlanCache()
    end
    return EncodingInvariantCache{K,F,MatT,typeof(pi)}(
        M,
        pi,
        opts0,
        threaded,
        level,
        session_cache,
        spcache,
        Dict{UInt,Invariants.CompiledSlicePlan}(),
        Dict{UInt,Invariants.ProjectedArrangement}(),
        Dict{UInt,Invariants.ProjectedBarcodeCache{K}}(),
        Dict{Tuple{UInt,UInt},Invariants.ProjectedBarcodeCache}(),
        Dict{UInt,Invariants.FiberedBarcodeCache2D{K}}(),
        nothing,
    )
end

function build_restricted_hilbert_cache(
    M::PModule{K,F,MatT},
    pi;
    opts::InvariantOptions=InvariantOptions(),
    level::Symbol=:auto,
    threaded::Bool=true,
) where {K,F<:AbstractCoeffField,MatT<:AbstractMatrix{K}}
    _validate_cache_level(level)
    opts0 = opts
    h = Invariants.restricted_hilbert(M)
    return RestrictedHilbertInvariantCache{K,F,MatT,typeof(pi)}(
        M, pi, opts0, threaded, level, h
    )
end

build_cache(enc::EncodingResult; kwargs...) = build_cache(enc.M, enc.pi; kwargs...)

function build_cache(obj;
                     opts::InvariantOptions=InvariantOptions(),
                     level::Symbol=:auto,
                     threaded::Bool=true,
                     cache=:auto)
    _validate_cache_level(level)
    if _supports_module_pi(obj)
        M, pi = _sample_module_pi(obj)
        return build_cache(M, pi; opts=opts, level=level, threaded=threaded, cache=cache)
    elseif _supports_module(obj)
        M = _sample_module(obj)
        return build_cache(M; opts=opts, level=level, threaded=threaded)
    end
    throw(ArgumentError("build_cache: sample must provide a PModule or (PModule, pi)"))
end

build_cache(M::PModule, pi, opts::InvariantOptions; kwargs...) =
    build_cache(M, pi; opts=opts, kwargs...)

build_cache(obj, opts::InvariantOptions; kwargs...) =
    build_cache(obj; opts=opts, kwargs...)

function build_cache(M::PModule{K,F,MatT},
                     spec::AbstractFeaturizerSpec;
                     opts::InvariantOptions=InvariantOptions(),
                     level::Symbol=:auto,
                     threaded::Bool=true,
                     cache=:auto) where {K,F<:AbstractCoeffField,MatT<:AbstractMatrix{K}}
    supports(spec, M) || throw(ArgumentError("build_cache: sample is unsupported for $(typeof(spec))"))
    c = build_cache(M; opts=opts, level=level, threaded=threaded)
    _prime_cache!(c, spec; opts=opts, threaded=threaded)
    return c
end

@inline function _slice_compile_kwargs(spec::LandscapeSpec, opts0::InvariantOptions, threads0::Bool)
    return (
        directions=spec.directions,
        offsets=spec.offsets,
        offset_weights=spec.offset_weights,
        normalize_weights=spec.normalize_weights,
        tmin=spec.tmin,
        tmax=spec.tmax,
        nsteps=spec.nsteps,
        drop_unknown=spec.drop_unknown,
        dedup=spec.dedup,
        normalize_dirs=spec.normalize_dirs,
        direction_weight=spec.direction_weight,
        threads=threads0,
    )
end

@inline function _slice_compile_kwargs(spec::PersistenceImageSpec, opts0::InvariantOptions, threads0::Bool)
    return (
        directions=spec.directions,
        offsets=spec.offsets,
        offset_weights=spec.offset_weights,
        normalize_weights=spec.normalize_weights,
        tmin=spec.tmin,
        tmax=spec.tmax,
        nsteps=spec.nsteps,
        drop_unknown=spec.drop_unknown,
        dedup=spec.dedup,
        normalize_dirs=spec.normalize_dirs,
        direction_weight=spec.direction_weight,
        threads=threads0,
    )
end

@inline function _slice_compile_kwargs(spec::SlicedBarcodeSpec, opts0::InvariantOptions, threads0::Bool)
    return (
        directions=spec.directions,
        offsets=spec.offsets,
        offset_weights=spec.offset_weights,
        normalize_weights=spec.normalize_weights,
        tmin=spec.tmin,
        tmax=spec.tmax,
        nsteps=spec.nsteps,
        drop_unknown=spec.drop_unknown,
        dedup=spec.dedup,
        normalize_dirs=spec.normalize_dirs,
        direction_weight=spec.direction_weight,
        threads=threads0,
    )
end

@inline function _slice_compile_opts(spec::Union{LandscapeSpec,PersistenceImageSpec,SlicedBarcodeSpec},
                                     opts0::InvariantOptions)
    strict0 = opts0.strict === nothing ? spec.strict : opts0.strict
    box0 = opts0.box
    if box0 === nothing && spec.tmin === nothing && spec.tmax === nothing
        box0 = :auto
    end
    return InvariantOptions(
        axes=opts0.axes,
        axes_policy=opts0.axes_policy,
        max_axis_len=opts0.max_axis_len,
        box=box0,
        threads=opts0.threads,
        strict=strict0,
        pl_mode=opts0.pl_mode,
    )
end

function _slice_plan_for!(cache::EncodingInvariantCache,
                          spec::Union{LandscapeSpec,PersistenceImageSpec,SlicedBarcodeSpec},
                          opts0::InvariantOptions,
                          threads0::Bool)
    opts_compile = _slice_compile_opts(spec, opts0)
    kws = _slice_compile_kwargs(spec, opts0, threads0)
    # Keep geometry-identical slice specs on the same compiled plan.
    key = UInt(hash((opts_compile.box, opts_compile.strict, kws)))
    return get!(cache.slice_plans, key) do
        Invariants.compile_slices(cache.pi, opts_compile; cache=cache.slice_plan_cache, kws...)
    end
end

function _fibered_cache_for!(cache::EncodingInvariantCache,
                             spec::Union{LandscapeSpec,PersistenceImageSpec,SlicedBarcodeSpec},
                             opts0::InvariantOptions,
                             threads0::Bool,
                             level::Symbol)
    pi0 = _unwrap_cache_pi(cache.pi)
    _supports_fibered_cache(pi0) ||
        throw(ArgumentError("fibered cache requested for non-2D/non-PL encoding"))
    precompute = level == :all ? :cells_barcodes : :none
    key = UInt(hash((opts0.box, opts0.strict, spec.normalize_dirs, precompute)))
    return get!(cache.fibered, key) do
        Invariants.fibered_barcode_cache_2d(
            cache.M,
            pi0,
            opts0;
            precompute=precompute,
            normalize_dirs=spec.normalize_dirs,
            threads=threads0,
        )
    end
end

@inline function _rank_query_cache_for!(cache::EncodingInvariantCache)
    pi0 = _unwrap_cache_pi(cache.pi)
    pi0 isa ZnEncoding.ZnEncodingMap || return nothing
    if cache.rank_query === nothing
        cache.rank_query = Invariants.RankQueryCache(pi0)
    end
    return cache.rank_query
end

function _projected_arrangement_for!(cache::EncodingInvariantCache,
                                     spec::ProjectedDistancesSpec,
                                     threads0::Bool)
    key = UInt(hash((spec.directions, spec.n_dirs, spec.normalize, spec.enforce_monotone)))
    return get!(cache.projected_arrangements, key) do
        enc_cache = _workflow_encoding_cache(cache.session_cache)
        if spec.directions === nothing
            Invariants.projected_arrangement(cache.pi;
                n_dirs=spec.n_dirs,
                normalize=spec.normalize,
                enforce_monotone=spec.enforce_monotone,
                cache=enc_cache,
                threads=threads0)
        else
            Invariants.projected_arrangement(cache.pi;
                dirs=spec.directions,
                normalize=spec.normalize,
                enforce_monotone=spec.enforce_monotone,
                cache=enc_cache,
                threads=threads0)
        end
    end
end

function _projected_cache_for!(cache::EncodingInvariantCache,
                               spec::ProjectedDistancesSpec,
                               threads0::Bool)
    arr_key = UInt(hash((spec.directions, spec.n_dirs, spec.normalize, spec.enforce_monotone)))
    arr = _projected_arrangement_for!(cache, spec, threads0)
    cM = get!(cache.projected_module, arr_key) do
        Invariants.projected_barcode_cache(cache.M, arr; precompute=spec.precompute)
    end
    if spec.precompute
        Invariants.projected_barcodes(cM; threads=threads0)
    end
    refs = Vector{Invariants.ProjectedBarcodeCache}(undef, length(spec.references))
    @inbounds for i in eachindex(spec.references)
        Mi, _ = _sample_module_pi(spec.references[i])
        rid = UInt(objectid(Mi))
        refs[i] = get!(cache.projected_refs, (arr_key, rid)) do
            Invariants.projected_barcode_cache(Mi, arr; precompute=spec.precompute)
        end
        if spec.precompute
            Invariants.projected_barcodes(refs[i]; threads=threads0)
        end
    end
    return cM, refs
end

@inline function _prime_cache!(cache::AbstractInvariantCache, spec::AbstractFeaturizerSpec;
                               opts::InvariantOptions=InvariantOptions(),
                               threaded::Bool=true)
    return cache
end

function _prime_cache!(cache::EncodingInvariantCache,
                       spec::Union{LandscapeSpec,PersistenceImageSpec,SlicedBarcodeSpec};
                       opts::InvariantOptions=InvariantOptions(),
                       threaded::Bool=true)
    opts0 = _cache_opts(cache, opts)
    threads0 = _resolve_spec_threads(spec.threads, opts0, _cache_threaded(cache, threaded))
    lvl = _effective_cache_level(cache.level, spec, cache.pi)
    if lvl == :fibered || lvl == :all
        _fibered_cache_for!(cache, spec, opts0, threads0, lvl)
    else
        _slice_plan_for!(cache, spec, opts0, threads0)
    end
    return cache
end

function _prime_cache!(cache::EncodingInvariantCache,
                       spec::ProjectedDistancesSpec;
                       opts::InvariantOptions=InvariantOptions(),
                       threaded::Bool=true)
    opts0 = _cache_opts(cache, opts)
    threads0 = _resolve_spec_threads(spec.threads, opts0, _cache_threaded(cache, threaded))
    lvl = _effective_cache_level(cache.level, spec, cache.pi)
    if lvl == :projected || lvl == :all
        _projected_cache_for!(cache, spec, threads0)
    end
    return cache
end

function _prime_cache!(cache::EncodingInvariantCache,
                       spec::SignedBarcodeImageSpec;
                       opts::InvariantOptions=InvariantOptions(),
                       threaded::Bool=true)
    opts0 = _cache_opts(cache, opts)
    threads0 = _resolve_spec_threads(spec.threads, opts0, _cache_threaded(cache, threaded))
    _ = threads0
    lvl = _effective_cache_level(cache.level, spec, cache.pi)
    if lvl == :all
        _rank_query_cache_for!(cache)
    end
    return cache
end

function _prime_cache!(cache::AbstractInvariantCache,
                       spec::CompositeSpec;
                       opts::InvariantOptions=InvariantOptions(),
                       threaded::Bool=true)
    @inbounds for sub in spec.specs
        _prime_cache!(cache, sub; opts=opts, threaded=threaded)
    end
    return cache
end

function build_cache(obj,
                     spec::AbstractFeaturizerSpec;
                     opts::InvariantOptions=InvariantOptions(),
                     level::Symbol=:auto,
                     threaded::Bool=true)
    supports(spec, obj) || throw(ArgumentError("build_cache: sample is unsupported for $(typeof(spec))"))
    if spec isa RankGridSpec
        return build_cache(_sample_module(obj);
            opts=opts,
            level=level,
            threaded=threaded)
    elseif spec isa EulerSurfaceSpec
        M, pi = _sample_module_pi(obj)
        return build_restricted_hilbert_cache(M, pi;
            opts=opts,
            level=level,
            threaded=threaded)
    end
    cache = build_cache(obj; opts=opts, level=level, threaded=threaded)
    _prime_cache!(cache, spec; opts=opts, threaded=threaded)
    return cache
end

function cache_stats(cache::ModuleInvariantCache)
    return (kind=:module, threads=cache.threads, level=cache.level)
end

function cache_stats(cache::RestrictedHilbertInvariantCache)
    return (kind=:restricted_hilbert, threads=cache.threads, level=cache.level, n=length(cache.hilbert))
end

function cache_stats(cache::EncodingInvariantCache)
    return (
        kind=:encoding,
        threads=cache.threads,
        level=cache.level,
        n_slice_plans=length(cache.slice_plans),
        n_projected_arrangements=length(cache.projected_arrangements),
        n_projected_module=length(cache.projected_module),
        n_projected_refs=length(cache.projected_refs),
        n_fibered=length(cache.fibered),
        has_rank_query=(cache.rank_query !== nothing),
    )
end

function transform(spec::AbstractFeaturizerSpec,
                   cache::AbstractInvariantCache;
                   opts::InvariantOptions=InvariantOptions(),
                   threaded::Bool=true)
    return transform(spec, _cache_sample(cache); opts=opts, threaded=_cache_threaded(cache, threaded))
end

function _slice_feature_from_barcode(
    bc,
    featurizer;
    kmax::Int=5,
    tgrid=nothing,
    sil_weighting=:persistence,
    sil_p::Real=1,
    sil_normalize::Bool=true,
    xgrid=nothing,
    ygrid=nothing,
    img_sigma::Real=1.0,
    img_coords::Symbol=:birth_persistence,
    img_weighting=:persistence,
    img_p::Real=1,
    img_normalize::Symbol=:none,
    entropy_normalize::Bool=true,
    entropy_weighting=:persistence,
    entropy_p::Real=1,
    summary_fields=Invariants._DEFAULT_BARCODE_SUMMARY_FIELDS,
    summary_normalize_entropy::Bool=true,
)
    if featurizer == :landscape
        pl = Invariants.persistence_landscape(bc; kmax=kmax, tgrid=tgrid)
        return Invariants._landscape_feature_vector(pl)
    elseif featurizer == :silhouette
        s = Invariants.persistence_silhouette(
            bc;
            tgrid=tgrid,
            weighting=sil_weighting,
            p=sil_p,
            normalize=sil_normalize,
        )
        return Float64[s...]
    elseif featurizer == :image
        PI = Invariants.persistence_image(
            bc;
            xgrid=xgrid,
            ygrid=ygrid,
            sigma=img_sigma,
            coords=img_coords,
            weighting=img_weighting,
            p=img_p,
            normalize=img_normalize,
        )
        return Invariants._image_feature_vector(PI)
    elseif featurizer == :entropy
        e = Invariants.barcode_entropy(
            bc;
            normalize=entropy_normalize,
            weighting=entropy_weighting,
            p=entropy_p,
        )
        return Float64[float(e)]
    elseif featurizer == :summary
        return Invariants._barcode_summary_vector(
            bc;
            fields=summary_fields,
            normalize_entropy=summary_normalize_entropy,
        )
    elseif featurizer isa Function
        return Invariants._as_feature_vector(featurizer(bc))
    end
    throw(ArgumentError("unsupported slice featurizer $(featurizer)"))
end

function _slice_features_from_packed_grid(
    bars::Invariants.PackedBarcodeGrid,
    W::AbstractMatrix{Float64};
    featurizer,
    aggregate::Symbol,
    threads::Bool,
    kmax::Int=5,
    tgrid=nothing,
    sil_weighting=:persistence,
    sil_p::Real=1,
    sil_normalize::Bool=true,
    xgrid=nothing,
    ygrid=nothing,
    img_sigma::Real=1.0,
    img_coords::Symbol=:birth_persistence,
    img_weighting=:persistence,
    img_p::Real=1,
    img_normalize::Symbol=:none,
    entropy_normalize::Bool=true,
    entropy_weighting=:persistence,
    entropy_p::Real=1,
    summary_fields=Invariants._DEFAULT_BARCODE_SUMMARY_FIELDS,
    summary_normalize_entropy::Bool=true,
)
    tg = tgrid
    if (featurizer == :landscape || featurizer == :silhouette) && tg === nothing
        tg = Invariants._default_tgrid_from_barcodes(bars; nsteps=401)
    end
    if tg !== nothing
        tg = Invariants._clean_tgrid(tg)
    end

    xg = xgrid
    yg = ygrid
    if featurizer == :image && (xg === nothing || yg === nothing)
        xg2, yg2 = Invariants._default_image_grids_from_barcodes(
            bars;
            img_xgrid=xg,
            img_ygrid=yg,
            img_birth_range=nothing,
            img_pers_range=nothing,
            img_nbirth=20,
            img_npers=20,
        )
        xg = xg2
        yg = yg2
    end

    nd, no = size(bars)
    feats = Array{Vector{Float64}}(undef, nd, no)
    if threads && Threads.nthreads() > 1
        Threads.@threads for idx in 1:(nd * no)
            i = div(idx - 1, no) + 1
            j = (idx - 1) % no + 1
            feats[i, j] = _slice_feature_from_barcode(
                bars[i, j],
                featurizer;
                kmax=kmax,
                tgrid=tg,
                sil_weighting=sil_weighting,
                sil_p=sil_p,
                sil_normalize=sil_normalize,
                xgrid=xg,
                ygrid=yg,
                img_sigma=img_sigma,
                img_coords=img_coords,
                img_weighting=img_weighting,
                img_p=img_p,
                img_normalize=img_normalize,
                entropy_normalize=entropy_normalize,
                entropy_weighting=entropy_weighting,
                entropy_p=entropy_p,
                summary_fields=summary_fields,
                summary_normalize_entropy=summary_normalize_entropy,
            )
        end
    else
        @inbounds for i in 1:nd, j in 1:no
            feats[i, j] = _slice_feature_from_barcode(
                bars[i, j],
                featurizer;
                kmax=kmax,
                tgrid=tg,
                sil_weighting=sil_weighting,
                sil_p=sil_p,
                sil_normalize=sil_normalize,
                xgrid=xg,
                ygrid=yg,
                img_sigma=img_sigma,
                img_coords=img_coords,
                img_weighting=img_weighting,
                img_p=img_p,
                img_normalize=img_normalize,
                entropy_normalize=entropy_normalize,
                entropy_weighting=entropy_weighting,
                entropy_p=entropy_p,
                summary_fields=summary_fields,
                summary_normalize_entropy=summary_normalize_entropy,
            )
        end
    end
    return Invariants._aggregate_feature_vectors(feats, W; aggregate=aggregate, unwrap_scalar=true)
end

function transform(spec::LandscapeSpec,
                   cache::EncodingInvariantCache;
                   opts::InvariantOptions=InvariantOptions(),
                   threaded::Bool=true)
    opts0 = _cache_opts(cache, opts)
    threads0 = _resolve_spec_threads(spec.threads, opts0, _cache_threaded(cache, threaded))
    strict0 = opts0.strict === nothing ? spec.strict : opts0.strict
    lvl = _effective_cache_level(cache.level, spec, cache.pi)
    vals = if lvl == :fibered || lvl == :all
        fcache = _fibered_cache_for!(cache, spec, opts0, threads0, lvl)
        data = Invariants.slice_barcodes(
            fcache;
            dirs=spec.directions,
            offsets=spec.offsets,
            values=:t,
            packed=true,
            direction_weight=spec.direction_weight,
            offset_weights=spec.offset_weights,
            normalize_weights=spec.normalize_weights,
            threads=threads0,
        )
        _slice_features_from_packed_grid(
            data.barcodes,
            data.weights;
            featurizer=:landscape,
            aggregate=spec.aggregate,
            threads=threads0,
            kmax=spec.kmax,
            tgrid=spec.tgrid,
        )
    else
        spcache = lvl == :none ? nothing : cache.slice_plan_cache
        if spcache !== nothing
            _slice_plan_for!(cache, spec, opts0, threads0)
        end
        Invariants.slice_features(
            cache.M, cache.pi;
            directions=spec.directions,
            offsets=spec.offsets,
            offset_weights=spec.offset_weights,
            tmin=spec.tmin,
            tmax=spec.tmax,
            nsteps=spec.nsteps,
            strict=strict0,
            box=opts0.box,
            drop_unknown=spec.drop_unknown,
            dedup=spec.dedup,
            normalize_dirs=spec.normalize_dirs,
            direction_weight=spec.direction_weight,
            featurizer=:landscape,
            aggregate=spec.aggregate,
            normalize_weights=spec.normalize_weights,
            kmax=spec.kmax,
            tgrid=spec.tgrid,
            threads=threads0,
            cache=spcache,
        )
    end
    return _float_vector(vals)
end

function transform(spec::PersistenceImageSpec,
                   cache::EncodingInvariantCache;
                   opts::InvariantOptions=InvariantOptions(),
                   threaded::Bool=true)
    opts0 = _cache_opts(cache, opts)
    threads0 = _resolve_spec_threads(spec.threads, opts0, _cache_threaded(cache, threaded))
    strict0 = opts0.strict === nothing ? spec.strict : opts0.strict
    lvl = _effective_cache_level(cache.level, spec, cache.pi)
    vals = if lvl == :fibered || lvl == :all
        fcache = _fibered_cache_for!(cache, spec, opts0, threads0, lvl)
        data = Invariants.slice_barcodes(
            fcache;
            dirs=spec.directions,
            offsets=spec.offsets,
            values=:t,
            packed=true,
            direction_weight=spec.direction_weight,
            offset_weights=spec.offset_weights,
            normalize_weights=spec.normalize_weights,
            threads=threads0,
        )
        _slice_features_from_packed_grid(
            data.barcodes,
            data.weights;
            featurizer=:image,
            aggregate=spec.aggregate,
            threads=threads0,
            xgrid=spec.xgrid,
            ygrid=spec.ygrid,
            img_sigma=spec.sigma,
            img_coords=spec.coords,
            img_weighting=spec.weighting,
            img_p=spec.p,
            img_normalize=spec.normalize,
        )
    else
        spcache = lvl == :none ? nothing : cache.slice_plan_cache
        if spcache !== nothing
            _slice_plan_for!(cache, spec, opts0, threads0)
        end
        Invariants.slice_features(
            cache.M, cache.pi;
            directions=spec.directions,
            offsets=spec.offsets,
            offset_weights=spec.offset_weights,
            tmin=spec.tmin,
            tmax=spec.tmax,
            nsteps=spec.nsteps,
            strict=strict0,
            box=opts0.box,
            drop_unknown=spec.drop_unknown,
            dedup=spec.dedup,
            normalize_dirs=spec.normalize_dirs,
            direction_weight=spec.direction_weight,
            featurizer=:image,
            aggregate=spec.aggregate,
            normalize_weights=spec.normalize_weights,
            img_xgrid=spec.xgrid,
            img_ygrid=spec.ygrid,
            img_sigma=spec.sigma,
            img_coords=spec.coords,
            img_weighting=spec.weighting,
            img_p=spec.p,
            img_normalize=spec.normalize,
            threads=threads0,
            cache=spcache,
        )
    end
    return _float_vector(vals)
end

function transform(spec::SlicedBarcodeSpec,
                   cache::EncodingInvariantCache;
                   opts::InvariantOptions=InvariantOptions(),
                   threaded::Bool=true)
    opts0 = _cache_opts(cache, opts)
    threads0 = _resolve_spec_threads(spec.threads, opts0, _cache_threaded(cache, threaded))
    strict0 = opts0.strict === nothing ? spec.strict : opts0.strict
    lvl = _effective_cache_level(cache.level, spec, cache.pi)
    vals = if lvl == :fibered || lvl == :all
        fcache = _fibered_cache_for!(cache, spec, opts0, threads0, lvl)
        data = Invariants.slice_barcodes(
            fcache;
            dirs=spec.directions,
            offsets=spec.offsets,
            values=:t,
            packed=true,
            direction_weight=spec.direction_weight,
            offset_weights=spec.offset_weights,
            normalize_weights=spec.normalize_weights,
            threads=threads0,
        )
        _slice_features_from_packed_grid(
            data.barcodes,
            data.weights;
            featurizer=spec.featurizer,
            aggregate=spec.aggregate,
            threads=threads0,
            summary_fields=spec.summary_fields,
            summary_normalize_entropy=spec.summary_normalize_entropy,
            entropy_normalize=spec.entropy_normalize,
            entropy_weighting=spec.entropy_weighting,
            entropy_p=spec.entropy_p,
        )
    else
        spcache = lvl == :none ? nothing : cache.slice_plan_cache
        if spcache !== nothing
            _slice_plan_for!(cache, spec, opts0, threads0)
        end
        Invariants.slice_features(
            cache.M, cache.pi;
            directions=spec.directions,
            offsets=spec.offsets,
            offset_weights=spec.offset_weights,
            tmin=spec.tmin,
            tmax=spec.tmax,
            nsteps=spec.nsteps,
            strict=strict0,
            box=opts0.box,
            drop_unknown=spec.drop_unknown,
            dedup=spec.dedup,
            normalize_dirs=spec.normalize_dirs,
            direction_weight=spec.direction_weight,
            featurizer=spec.featurizer,
            aggregate=spec.aggregate,
            normalize_weights=spec.normalize_weights,
            summary_fields=spec.summary_fields,
            summary_normalize_entropy=spec.summary_normalize_entropy,
            entropy_normalize=spec.entropy_normalize,
            entropy_weighting=spec.entropy_weighting,
            entropy_p=spec.entropy_p,
            threads=threads0,
            cache=spcache,
        )
    end
    return _float_vector(vals)
end

function transform(spec::ProjectedDistancesSpec,
                   cache::EncodingInvariantCache;
                   opts::InvariantOptions=InvariantOptions(),
                   threaded::Bool=true)
    opts0 = _cache_opts(cache, opts)
    threads0 = _resolve_spec_threads(spec.threads, opts0, _cache_threaded(cache, threaded))
    cM, refs = _projected_cache_for!(cache, spec, threads0)
    out = Vector{Float64}(undef, length(refs))
    @inbounds for i in eachindex(refs)
        out[i] = Invariants.projected_distance(cM, refs[i];
            dist=spec.dist,
            p=spec.p,
            q=spec.q,
            agg=spec.agg,
            threads=threads0)
    end
    return out
end

@inline function _eulersurface_vector(obj, pi, spec::EulerSurfaceSpec, opts0::InvariantOptions, threads0::Bool)
    axes0 = spec.axes === nothing ? opts0.axes : spec.axes
    strict0 = spec.strict === nothing ? opts0.strict : spec.strict
    opts2 = InvariantOptions(
        axes=axes0,
        axes_policy=spec.axes_policy,
        max_axis_len=spec.max_axis_len,
        box=opts0.box,
        threads=threads0,
        strict=strict0,
        pl_mode=opts0.pl_mode,
    )
    surf = Invariants.euler_surface(obj, pi, opts2)
    return _float_vector(surf)
end

@inline function _rankgrid_vector(M::PModule, spec::RankGridSpec, opts0::InvariantOptions)
    nP = nvertices(M.Q)
    spec.nvertices == nP || throw(ArgumentError("RankGridSpec.nvertices=$(spec.nvertices) does not match module poset size $nP"))
    tbl = Invariants.rank_invariant(M, opts0; store_zeros=spec.store_zeros)
    out = zeros(Float64, spec.nvertices * spec.nvertices)
    @inbounds for a in 1:spec.nvertices, b in 1:spec.nvertices
        out[(a - 1) * spec.nvertices + b] = Float64(get(tbl, (a, b), 0))
    end
    return out
end

function transform(spec::EulerSurfaceSpec,
                   cache::RestrictedHilbertInvariantCache;
                   opts::InvariantOptions=InvariantOptions(),
                   threaded::Bool=true)
    opts0 = _cache_opts(cache, opts)
    threads0 = _resolve_spec_threads(spec.threads, opts0, _cache_threaded(cache, threaded))
    return _eulersurface_vector(cache.hilbert, cache.pi, spec, opts0, threads0)
end

function transform(spec::EulerSurfaceSpec,
                   cache::EncodingInvariantCache;
                   opts::InvariantOptions=InvariantOptions(),
                   threaded::Bool=true)
    opts0 = _cache_opts(cache, opts)
    threads0 = _resolve_spec_threads(spec.threads, opts0, _cache_threaded(cache, threaded))
    return _eulersurface_vector(cache.M, cache.pi, spec, opts0, threads0)
end

function transform(spec::EulerSurfaceSpec,
                   cache::AbstractInvariantCache;
                   opts::InvariantOptions=InvariantOptions(),
                   threaded::Bool=true)
    return transform(spec, _cache_sample(cache); opts=opts, threaded=_cache_threaded(cache, threaded))
end

function transform(spec::RankGridSpec,
                   cache::AbstractInvariantCache;
                   opts::InvariantOptions=InvariantOptions(),
                   threaded::Bool=true)
    opts0 = _cache_opts(cache, opts)
    M = _sample_module(_cache_sample(cache))
    return _rankgrid_vector(M, spec, opts0)
end

function transform(spec::SignedBarcodeImageSpec,
                   cache::EncodingInvariantCache;
                   opts::InvariantOptions=InvariantOptions(),
                   threaded::Bool=true)
    opts0 = _cache_opts(cache, opts)
    pi0 = cache.pi isa CompiledEncoding ? cache.pi.pi : cache.pi
    pi0 isa ZnEncoding.ZnEncodingMap ||
        throw(ArgumentError("SignedBarcodeImageSpec currently requires a ZnEncodingMap-compatible encoding"))
    axes0 = spec.axes === nothing ? opts0.axes : spec.axes
    strict0 = spec.strict === nothing ? opts0.strict : spec.strict
    threads0 = _resolve_spec_threads(spec.threads, opts0, _cache_threaded(cache, threaded))
    opts2 = InvariantOptions(
        axes=axes0,
        axes_policy=spec.axes_policy,
        max_axis_len=spec.max_axis_len,
        box=opts0.box,
        threads=threads0,
        strict=strict0,
        pl_mode=opts0.pl_mode,
    )
    lvl = _effective_cache_level(cache.level, spec, cache.pi)
    rq_cache = lvl == :all ? _rank_query_cache_for!(cache) : nothing
    sb = Invariants.rectangle_signed_barcode(
        cache.M,
        pi0,
        opts2;
        method=spec.method,
        threads=threads0,
        rq_cache=rq_cache,
    )
    img = Invariants.rectangle_signed_barcode_image(sb;
        xs=spec.xs,
        ys=spec.ys,
        sigma=spec.sigma,
        mode=spec.mode,
        threads=threads0)
    return _float_vector(img)
end

function transform(spec::LandscapeSpec, obj; opts::InvariantOptions=InvariantOptions(), threaded::Bool=true)
    cache = build_cache(obj, spec; opts=opts, threaded=threaded)
    return transform(spec, cache; opts=opts, threaded=threaded)
end

function transform(spec::PersistenceImageSpec, obj; opts::InvariantOptions=InvariantOptions(), threaded::Bool=true)
    cache = build_cache(obj, spec; opts=opts, threaded=threaded)
    return transform(spec, cache; opts=opts, threaded=threaded)
end

function transform(spec::EulerSurfaceSpec, obj; opts::InvariantOptions=InvariantOptions(), threaded::Bool=true)
    cache = build_cache(obj, spec; opts=opts, threaded=threaded)
    return transform(spec, cache; opts=opts, threaded=threaded)
end

function transform(spec::RankGridSpec, obj; opts::InvariantOptions=InvariantOptions(), threaded::Bool=true)
    opts0 = opts
    M = _sample_module(obj)
    return _rankgrid_vector(M, spec, opts0)
end

function transform(spec::SlicedBarcodeSpec, obj; opts::InvariantOptions=InvariantOptions(), threaded::Bool=true)
    cache = build_cache(obj, spec; opts=opts, threaded=threaded)
    return transform(spec, cache; opts=opts, threaded=threaded)
end

function transform(spec::SignedBarcodeImageSpec, obj; opts::InvariantOptions=InvariantOptions(), threaded::Bool=true)
    cache = build_cache(obj, spec; opts=opts, threaded=threaded)
    return transform(spec, cache; opts=opts, threaded=threaded)
end

function transform(spec::ProjectedDistancesSpec, obj; opts::InvariantOptions=InvariantOptions(), threaded::Bool=true)
    cache = build_cache(obj, spec; opts=opts, threaded=threaded)
    return transform(spec, cache; opts=opts, threaded=threaded)
end

@inline _is_slice_family_spec(spec) = spec isa Union{LandscapeSpec,PersistenceImageSpec,SlicedBarcodeSpec}

function _slice_group_data_for_spec(
    cache::EncodingInvariantCache,
    spec::Union{LandscapeSpec,PersistenceImageSpec,SlicedBarcodeSpec},
    opts0::InvariantOptions,
    threads0::Bool,
)
    lvl = _effective_cache_level(cache.level, spec, cache.pi)
    if lvl == :fibered || lvl == :all
        fcache = _fibered_cache_for!(cache, spec, opts0, threads0, lvl)
        data = Invariants.slice_barcodes(
            fcache;
            dirs=spec.directions,
            offsets=spec.offsets,
            values=:t,
            packed=true,
            direction_weight=spec.direction_weight,
            offset_weights=spec.offset_weights,
            normalize_weights=spec.normalize_weights,
            threads=threads0,
        )
        return data.barcodes, data.weights
    end
    spcache = lvl == :none ? nothing : cache.slice_plan_cache
    if spcache !== nothing
        _slice_plan_for!(cache, spec, opts0, threads0)
    end
    opts_compile = _slice_compile_opts(spec, opts0)
    kws = _slice_compile_kwargs(spec, opts0, threads0)
    plan = Invariants.compile_slices(cache.pi, opts_compile; cache=spcache, kws...)
    task = Invariants.SliceBarcodesTask(; packed=true, threads=threads0)
    data = Invariants.run_invariants(plan, cache.M, task)
    return data.barcodes, data.weights
end

function _composite_slice_vector(
    spec::LandscapeSpec,
    bars::Invariants.PackedBarcodeGrid,
    W::AbstractMatrix{Float64},
    threads0::Bool,
)
    return _float_vector(_slice_features_from_packed_grid(
        bars, W;
        featurizer=:landscape,
        aggregate=spec.aggregate,
        threads=threads0,
        kmax=spec.kmax,
        tgrid=spec.tgrid,
    ))
end

function _composite_slice_vector(
    spec::PersistenceImageSpec,
    bars::Invariants.PackedBarcodeGrid,
    W::AbstractMatrix{Float64},
    threads0::Bool,
)
    return _float_vector(_slice_features_from_packed_grid(
        bars, W;
        featurizer=:image,
        aggregate=spec.aggregate,
        threads=threads0,
        xgrid=spec.xgrid,
        ygrid=spec.ygrid,
        img_sigma=spec.sigma,
        img_coords=spec.coords,
        img_weighting=spec.weighting,
        img_p=spec.p,
        img_normalize=spec.normalize,
    ))
end

function _composite_slice_vector(
    spec::SlicedBarcodeSpec,
    bars::Invariants.PackedBarcodeGrid,
    W::AbstractMatrix{Float64},
    threads0::Bool,
)
    return _float_vector(_slice_features_from_packed_grid(
        bars, W;
        featurizer=spec.featurizer,
        aggregate=spec.aggregate,
        threads=threads0,
        summary_fields=spec.summary_fields,
        summary_normalize_entropy=spec.summary_normalize_entropy,
        entropy_normalize=spec.entropy_normalize,
        entropy_weighting=spec.entropy_weighting,
        entropy_p=spec.entropy_p,
    ))
end

function _fill_composite_slice_parts!(
    parts::Vector{Vector{Float64}},
    filled::BitVector,
    spec::CompositeSpec,
    cache::EncodingInvariantCache,
    opts0::InvariantOptions,
    threaded0::Bool,
)
    groups = Dict{UInt,Vector{Int}}()
    for i in eachindex(spec.specs)
        sub = spec.specs[i]
        _is_slice_family_spec(sub) || continue
        threads_i = _resolve_spec_threads(getproperty(sub, :threads), opts0, threaded0)
        key = UInt(hash((_slice_compile_kwargs(sub, opts0, threads_i), _effective_cache_level(cache.level, sub, cache.pi))))
        push!(get!(groups, key) do
            Int[]
        end, i)
    end

    for idxs in values(groups)
        first_spec = spec.specs[first(idxs)]::Union{LandscapeSpec,PersistenceImageSpec,SlicedBarcodeSpec}
        threads_g = _resolve_spec_threads(getproperty(first_spec, :threads), opts0, threaded0)
        bars, W = _slice_group_data_for_spec(cache, first_spec, opts0, threads_g)
        for idx in idxs
            sub = spec.specs[idx]
            parts[idx] = _composite_slice_vector(sub, bars, W, threads_g)
            filled[idx] = true
        end
    end
    return nothing
end

@inline _is_projected_family_spec(spec) = spec isa ProjectedDistancesSpec

function _fill_composite_projected_parts!(
    parts::Vector{Vector{Float64}},
    filled::BitVector,
    spec::CompositeSpec,
    cache::EncodingInvariantCache,
    opts0::InvariantOptions,
    threaded0::Bool,
)
    groups = Dict{UInt,Vector{Int}}()
    for i in eachindex(spec.specs)
        sub = spec.specs[i]
        _is_projected_family_spec(sub) || continue
        threads_i = _resolve_spec_threads(sub.threads, opts0, threaded0)
        key = UInt(hash((sub.references, sub.directions, sub.n_dirs, sub.normalize, sub.enforce_monotone, sub.precompute, threads_i)))
        push!(get!(groups, key) do
            Int[]
        end, i)
    end

    for idxs in values(groups)
        first_spec = spec.specs[first(idxs)]::ProjectedDistancesSpec
        threads_g = _resolve_spec_threads(first_spec.threads, opts0, threaded0)
        cM, refs = _projected_cache_for!(cache, first_spec, threads_g)
        for idx in idxs
            sub = spec.specs[idx]::ProjectedDistancesSpec
            out = Vector{Float64}(undef, length(refs))
            @inbounds for i in eachindex(refs)
                out[i] = Invariants.projected_distance(cM, refs[i];
                    dist=sub.dist,
                    p=sub.p,
                    q=sub.q,
                    agg=sub.agg,
                    threads=threads_g)
            end
            parts[idx] = out
            filled[idx] = true
        end
    end
    return nothing
end

@inline _is_signed_barcode_family_spec(spec) = spec isa SignedBarcodeImageSpec

function _signed_barcode_for_spec(
    cache::EncodingInvariantCache,
    spec::SignedBarcodeImageSpec,
    opts0::InvariantOptions,
    threads0::Bool,
)
    pi0 = _unwrap_cache_pi(cache.pi)
    pi0 isa ZnEncoding.ZnEncodingMap ||
        throw(ArgumentError("SignedBarcodeImageSpec currently requires a ZnEncodingMap-compatible encoding"))
    axes0 = spec.axes === nothing ? opts0.axes : spec.axes
    strict0 = spec.strict === nothing ? opts0.strict : spec.strict
    opts2 = InvariantOptions(
        axes=axes0,
        axes_policy=spec.axes_policy,
        max_axis_len=spec.max_axis_len,
        box=opts0.box,
        threads=threads0,
        strict=strict0,
        pl_mode=opts0.pl_mode,
    )
    lvl = _effective_cache_level(cache.level, spec, cache.pi)
    rq_cache = lvl == :all ? _rank_query_cache_for!(cache) : nothing
    return Invariants.rectangle_signed_barcode(
        cache.M,
        pi0,
        opts2;
        method=spec.method,
        threads=threads0,
        rq_cache=rq_cache,
    )
end

function _fill_composite_signed_barcode_parts!(
    parts::Vector{Vector{Float64}},
    filled::BitVector,
    spec::CompositeSpec,
    cache::EncodingInvariantCache,
    opts0::InvariantOptions,
    threaded0::Bool,
)
    groups = Dict{UInt,Vector{Int}}()
    for i in eachindex(spec.specs)
        sub = spec.specs[i]
        _is_signed_barcode_family_spec(sub) || continue
        threads_i = _resolve_spec_threads(sub.threads, opts0, threaded0)
        axes_i = sub.axes === nothing ? opts0.axes : sub.axes
        strict_i = sub.strict === nothing ? opts0.strict : sub.strict
        key = UInt(hash((sub.method, axes_i, sub.axes_policy, sub.max_axis_len, strict_i, threads_i)))
        push!(get!(groups, key) do
            Int[]
        end, i)
    end

    for idxs in values(groups)
        first_spec = spec.specs[first(idxs)]::SignedBarcodeImageSpec
        threads_g = _resolve_spec_threads(first_spec.threads, opts0, threaded0)
        sb = _signed_barcode_for_spec(cache, first_spec, opts0, threads_g)
        for idx in idxs
            sub = spec.specs[idx]::SignedBarcodeImageSpec
            img = Invariants.rectangle_signed_barcode_image(sb;
                xs=sub.xs,
                ys=sub.ys,
                sigma=sub.sigma,
                mode=sub.mode,
                threads=threads_g)
            parts[idx] = _float_vector(img)
            filled[idx] = true
        end
    end
    return nothing
end

@inline _is_euler_family_spec(spec) = spec isa EulerSurfaceSpec

function _fill_composite_euler_parts!(
    parts::Vector{Vector{Float64}},
    filled::BitVector,
    spec::CompositeSpec,
    cache::EncodingInvariantCache,
    opts0::InvariantOptions,
    threaded0::Bool,
)
    groups = Dict{UInt,Vector{Int}}()
    for i in eachindex(spec.specs)
        sub = spec.specs[i]
        _is_euler_family_spec(sub) || continue
        threads_i = _resolve_spec_threads(sub.threads, opts0, threaded0)
        axes_i = sub.axes === nothing ? opts0.axes : sub.axes
        strict_i = sub.strict === nothing ? opts0.strict : sub.strict
        key = UInt(hash((axes_i, sub.axes_policy, sub.max_axis_len, opts0.box, strict_i, threads_i)))
        push!(get!(groups, key) do
            Int[]
        end, i)
    end

    isempty(groups) && return nothing
    h = Invariants.restricted_hilbert(cache.M)
    for idxs in values(groups)
        first_spec = spec.specs[first(idxs)]::EulerSurfaceSpec
        threads_g = _resolve_spec_threads(first_spec.threads, opts0, threaded0)
        vals = _eulersurface_vector(h, cache.pi, first_spec, opts0, threads_g)
        for idx in idxs
            parts[idx] = vals
            filled[idx] = true
        end
    end
    return nothing
end

function _fill_composite_euler_parts!(
    parts::Vector{Vector{Float64}},
    filled::BitVector,
    spec::CompositeSpec,
    cache::RestrictedHilbertInvariantCache,
    opts0::InvariantOptions,
    threaded0::Bool,
)
    groups = Dict{UInt,Vector{Int}}()
    for i in eachindex(spec.specs)
        sub = spec.specs[i]
        _is_euler_family_spec(sub) || continue
        threads_i = _resolve_spec_threads(sub.threads, opts0, threaded0)
        axes_i = sub.axes === nothing ? opts0.axes : sub.axes
        strict_i = sub.strict === nothing ? opts0.strict : sub.strict
        key = UInt(hash((axes_i, sub.axes_policy, sub.max_axis_len, opts0.box, strict_i, threads_i)))
        push!(get!(groups, key) do
            Int[]
        end, i)
    end

    for idxs in values(groups)
        first_spec = spec.specs[first(idxs)]::EulerSurfaceSpec
        threads_g = _resolve_spec_threads(first_spec.threads, opts0, threaded0)
        vals = _eulersurface_vector(cache.hilbert, cache.pi, first_spec, opts0, threads_g)
        for idx in idxs
            parts[idx] = vals
            filled[idx] = true
        end
    end
    return nothing
end

@inline _is_rank_family_spec(spec) = spec isa RankGridSpec

function _fill_composite_rank_parts!(
    parts::Vector{Vector{Float64}},
    filled::BitVector,
    spec::CompositeSpec,
    cache::AbstractInvariantCache,
    opts0::InvariantOptions,
)
    M = _sample_module(_cache_sample(cache))
    groups = Dict{UInt,Vector{Int}}()
    for i in eachindex(spec.specs)
        sub = spec.specs[i]
        _is_rank_family_spec(sub) || continue
        key = UInt(hash((sub.nvertices, sub.store_zeros)))
        push!(get!(groups, key) do
            Int[]
        end, i)
    end

    for idxs in values(groups)
        first_spec = spec.specs[first(idxs)]::RankGridSpec
        vals = _rankgrid_vector(M, first_spec, opts0)
        for idx in idxs
            parts[idx] = vals
            filled[idx] = true
        end
    end
    return nothing
end

function transform(spec::CompositeSpec, obj; opts::InvariantOptions=InvariantOptions(), threaded::Bool=true)
    cache = build_cache(obj, spec; opts=opts, threaded=threaded)
    return transform(spec, cache; opts=opts, threaded=threaded)
end

function transform(spec::CompositeSpec,
                   cache::AbstractInvariantCache;
                   opts::InvariantOptions=InvariantOptions(),
                   threaded::Bool=true)
    opts0 = _cache_opts(cache, opts)
    threaded0 = _cache_threaded(cache, threaded)
    _prime_cache!(cache, spec; opts=opts0, threaded=threaded0)
    parts = Vector{Vector{Float64}}(undef, length(spec.specs))
    filled = falses(length(spec.specs))

    if cache isa EncodingInvariantCache
        _fill_composite_slice_parts!(parts, filled, spec, cache, opts0, threaded0)
        _fill_composite_projected_parts!(parts, filled, spec, cache, opts0, threaded0)
        _fill_composite_signed_barcode_parts!(parts, filled, spec, cache, opts0, threaded0)
        _fill_composite_euler_parts!(parts, filled, spec, cache, opts0, threaded0)
    elseif cache isa RestrictedHilbertInvariantCache
        _fill_composite_euler_parts!(parts, filled, spec, cache, opts0, threaded0)
    end
    _fill_composite_rank_parts!(parts, filled, spec, cache, opts0)

    total = 0
    @inbounds for i in eachindex(spec.specs)
        if !filled[i]
            parts[i] = transform(spec.specs[i], cache; opts=opts0, threaded=threaded0)
        end
        v = parts[i]
        total += length(v)
    end
    out = Vector{Float64}(undef, total)
    pos = 1
    @inbounds for v in parts
        copyto!(out, pos, v, 1, length(v))
        pos += length(v)
    end
    return out
end

function transform!(dest::AbstractVector, spec::AbstractFeaturizerSpec, obj;
                    opts::InvariantOptions=InvariantOptions(),
                    threaded::Bool=true)
    vals = transform(spec, obj; opts=opts, threaded=threaded)
    length(dest) == length(vals) || throw(DimensionMismatch("destination length $(length(dest)) != feature length $(length(vals))"))
    copyto!(dest, vals)
    return dest
end

function _feature_names_or_default(spec::AbstractFeaturizerSpec, n::Int)
    names = feature_names(spec)
    isempty(names) && return [Symbol("f$(i)") for i in 1:n]
    length(names) == n || throw(DimensionMismatch("feature_names(spec) has length $(length(names)) but transform output has length $n"))
    return names
end

function _resolve_feature_session_cache(cache)
    if cache === :auto
        return SessionCache(), :session_auto
    elseif cache === nothing
        return nothing, :none
    elseif cache isa SessionCache
        return cache, :session_user
    else
        throw(ArgumentError("cache must be :auto, nothing, or SessionCache"))
    end
end

@inline function _batch_foreach_serial(n::Int, f)
    @inbounds for i in 1:n
        f(i)
    end
    return nothing
end

function _batch_foreach_threads(n::Int, f; chunk_size::Int=0, deterministic::Bool=true)
    n <= 1 && return _batch_foreach_serial(n, f)
    nt = Threads.nthreads()
    nt <= 1 && return _batch_foreach_serial(n, f)

    if deterministic
        if chunk_size > 0
            nchunks = cld(n, chunk_size)
            Threads.@threads for c in 1:nchunks
                lo = (c - 1) * chunk_size + 1
                hi = min(n, c * chunk_size)
                @inbounds for i in lo:hi
                    f(i)
                end
            end
            return nothing
        end

        nt_eff = min(nt, n)
        base = fld(n, nt_eff)
        remn = n - base * nt_eff
        Threads.@threads for t in 1:nt_eff
            extra = t <= remn ? 1 : 0
            start = (t - 1) * base + min(t - 1, remn) + 1
            stop = start + base + extra - 1
            @inbounds for i in start:stop
                f(i)
            end
        end
        return nothing
    end

    Threads.@threads for i in 1:n
        f(i)
    end
    return nothing
end

@inline function _batch_foreach(n::Int, opts::BatchOptions, f)
    if !opts.threaded || n <= 1 || opts.backend == :serial
        return _batch_foreach_serial(n, f)
    elseif opts.backend == :threads
        return _batch_foreach_threads(n, f; chunk_size=opts.chunk_size, deterministic=opts.deterministic)
    elseif opts.backend == :folds
        impl = _BATCH_IMPL[]
        if impl !== nothing
            return impl.foreach_indexed(n, f; chunk_size=opts.chunk_size, deterministic=opts.deterministic)
        end
        # Fallback keeps behavior usable when Folds extension is not loaded.
        return _batch_foreach_threads(n, f; chunk_size=opts.chunk_size, deterministic=opts.deterministic)
    end
    throw(ArgumentError("unsupported batch backend: $(opts.backend)"))
end

@inline _batch_foreach(f, n::Int, opts::BatchOptions) = _batch_foreach(n, opts, f)

const _BATCH_IMPL = Ref{Any}(nothing)
@inline _set_batch_impl!(impl) = (_BATCH_IMPL[] = impl; nothing)

const _PROGRESS_IMPL = Ref{Any}(nothing)
@inline _set_progress_impl!(impl) = (_PROGRESS_IMPL[] = impl; nothing)

const _KERNELFUNCTIONS_IMPL = Ref{Any}(nothing)
@inline _set_kernelfunctions_impl!(impl) = (_KERNELFUNCTIONS_IMPL[] = impl; nothing)

const _DISTANCES_IMPL = Ref{Any}(nothing)
@inline _set_distances_impl!(impl) = (_DISTANCES_IMPL[] = impl; nothing)

"""
    mp_landscape_kernel_object(; kwargs...)
    projected_kernel_object(; kwargs...)
    mpp_image_kernel_object(; kwargs...)
    point_signed_measure_kernel_object(; kwargs...)
    rectangle_signed_barcode_kernel_object(; kwargs...)

Construct KernelFunctions-compatible kernel wrapper objects (provided by the
KernelFunctions extension). These wrappers can be passed to
`KernelFunctions.kernelmatrix` for pairwise kernel evaluation on vectors of
compatible objects.
"""
@inline function mp_landscape_kernel_object(; kwargs...)
    impl = _KERNELFUNCTIONS_IMPL[]
    impl === nothing && throw(ArgumentError("mp_landscape_kernel_object requires KernelFunctions.jl extension (load KernelFunctions and ensure TamerOpKernelFunctionsExt is available)."))
    return impl.mp_landscape(; kwargs...)
end

@inline function projected_kernel_object(; kwargs...)
    impl = _KERNELFUNCTIONS_IMPL[]
    impl === nothing && throw(ArgumentError("projected_kernel_object requires KernelFunctions.jl extension (load KernelFunctions and ensure TamerOpKernelFunctionsExt is available)."))
    return impl.projected(; kwargs...)
end

@inline function mpp_image_kernel_object(; kwargs...)
    impl = _KERNELFUNCTIONS_IMPL[]
    impl === nothing && throw(ArgumentError("mpp_image_kernel_object requires KernelFunctions.jl extension (load KernelFunctions and ensure TamerOpKernelFunctionsExt is available)."))
    return impl.mpp_image(; kwargs...)
end

@inline function point_signed_measure_kernel_object(; kwargs...)
    impl = _KERNELFUNCTIONS_IMPL[]
    impl === nothing && throw(ArgumentError("point_signed_measure_kernel_object requires KernelFunctions.jl extension (load KernelFunctions and ensure TamerOpKernelFunctionsExt is available)."))
    return impl.point_signed_measure(; kwargs...)
end

@inline function rectangle_signed_barcode_kernel_object(; kwargs...)
    impl = _KERNELFUNCTIONS_IMPL[]
    impl === nothing && throw(ArgumentError("rectangle_signed_barcode_kernel_object requires KernelFunctions.jl extension (load KernelFunctions and ensure TamerOpKernelFunctionsExt is available)."))
    return impl.rectangle_signed_barcode(; kwargs...)
end

"""
    matching_distance_metric(; kwargs...)
    mp_landscape_distance_metric(; kwargs...)
    projected_distance_metric(; kwargs...)
    bottleneck_distance_metric(; kwargs...)
    wasserstein_distance_metric(; kwargs...)
    mpp_image_distance_metric(; kwargs...)

Construct Distances.jl `PreMetric` wrappers around PosetModules distances.
These can be used with `Distances.evaluate` and `Distances.pairwise`.
"""
@inline function matching_distance_metric(; kwargs...)
    impl = _DISTANCES_IMPL[]
    impl === nothing && throw(ArgumentError("matching_distance_metric requires Distances.jl extension (load Distances and ensure TamerOpDistancesExt is available)."))
    return impl.matching(; kwargs...)
end

@inline function mp_landscape_distance_metric(; kwargs...)
    impl = _DISTANCES_IMPL[]
    impl === nothing && throw(ArgumentError("mp_landscape_distance_metric requires Distances.jl extension (load Distances and ensure TamerOpDistancesExt is available)."))
    return impl.mp_landscape(; kwargs...)
end

@inline function projected_distance_metric(; kwargs...)
    impl = _DISTANCES_IMPL[]
    impl === nothing && throw(ArgumentError("projected_distance_metric requires Distances.jl extension (load Distances and ensure TamerOpDistancesExt is available)."))
    return impl.projected(; kwargs...)
end

@inline function bottleneck_distance_metric(; kwargs...)
    impl = _DISTANCES_IMPL[]
    impl === nothing && throw(ArgumentError("bottleneck_distance_metric requires Distances.jl extension (load Distances and ensure TamerOpDistancesExt is available)."))
    return impl.bottleneck(; kwargs...)
end

@inline function wasserstein_distance_metric(; kwargs...)
    impl = _DISTANCES_IMPL[]
    impl === nothing && throw(ArgumentError("wasserstein_distance_metric requires Distances.jl extension (load Distances and ensure TamerOpDistancesExt is available)."))
    return impl.wasserstein(; kwargs...)
end

@inline function mpp_image_distance_metric(; kwargs...)
    impl = _DISTANCES_IMPL[]
    impl === nothing && throw(ArgumentError("mpp_image_distance_metric requires Distances.jl extension (load Distances and ensure TamerOpDistancesExt is available)."))
    return impl.mpp_image(; kwargs...)
end

@inline function _progress_init(total::Int, label::AbstractString, enabled::Bool)
    enabled || return nothing
    impl = _PROGRESS_IMPL[]
    impl === nothing && return nothing
    return impl.init(total, label)
end

@inline function _progress_step!(state, delta::Int=1)
    impl = _PROGRESS_IMPL[]
    (state === nothing || impl === nothing) && return nothing
    return impl.step!(state, delta)
end

@inline function _progress_finish!(state)
    impl = _PROGRESS_IMPL[]
    (state === nothing || impl === nothing) && return nothing
    return impl.finish!(state)
end

struct _CompiledFeaturizePlan{FS<:AbstractFeaturizerSpec,AX,LAB}
    spec::FS
    opts::InvariantOptions
    on_unsupported::Symbol
    cache_mode::Symbol
    batch::BatchOptions
    feature_names::Vector{Symbol}
    feature_axes::AX
    ids::Vector{String}
    labels::LAB
    skipped_indices::Vector{Int}
    session_cache::Union{Nothing,SessionCache}
    sample_indices::Vector{Int}
    output_rows::Vector{Int}
    caches::Vector{_InvariantCacheHandle}
    first_slot::Int
    first_values::Vector{Float64}
    nrows::Int
    nfeatures::Int
    allow_missing::Bool
end

@inline function _feature_ids(samples::AbstractVector, idfun, idxs::AbstractVector{Int})
    if idfun === nothing
        return [string(i) for i in idxs]
    end
    return [string(idfun(samples[i])) for i in idxs]
end

@inline function _feature_labels(samples::AbstractVector, labelfun, idxs::AbstractVector{Int})
    labelfun === nothing && return nothing
    return [labelfun(samples[i]) for i in idxs]
end

function _compile_featurize(samples::AbstractVector,
                            spec::AbstractFeaturizerSpec,
                            opts0::InvariantOptions,
                            batch_opts::BatchOptions,
                            cache_mode::Symbol,
                            on_unsupported::Symbol,
                            session_cache::Union{Nothing,SessionCache},
                            idfun,
                            labelfun)
    ns = length(samples)
    axes_spec = feature_axes(spec)

    supported = BitVector(undef, ns)
    first_supported = 0
    @inbounds for i in 1:ns
        ok = supports(spec, samples[i])
        supported[i] = ok
        if ok && first_supported == 0
            first_supported = i
        end
    end

    if on_unsupported == :error
        bad = findfirst(x -> !x, supported)
        bad === nothing || throw(ArgumentError("sample at index $(bad) is unsupported for $(typeof(spec))"))
    end

    skipped = findall(x -> !x, supported)
    all_idxs = collect(1:ns)

    if first_supported == 0
        nf = nfeatures(spec)
        names = _feature_names_or_default(spec, nf)
        if on_unsupported == :skip
            labels = labelfun === nothing ? nothing : Any[]
            return _CompiledFeaturizePlan(
                spec, opts0, on_unsupported, cache_mode, batch_opts,
                names, axes_spec, String[], labels, skipped, session_cache,
                Int[], Int[], _InvariantCacheHandle[], 0, Float64[], 0, nf, false,
            )
        end
        return _CompiledFeaturizePlan(
            spec, opts0, on_unsupported, cache_mode, batch_opts,
            names, axes_spec, _feature_ids(samples, idfun, all_idxs),
            _feature_labels(samples, labelfun, all_idxs), skipped, session_cache,
            Int[], Int[], _InvariantCacheHandle[], 0, Float64[], ns, nf, true,
        )
    end

    sample_indices = findall(supported)
    output_rows = if on_unsupported == :skip
        collect(1:length(sample_indices))
    else
        copy(sample_indices)
    end
    ids = on_unsupported == :skip ?
        _feature_ids(samples, idfun, sample_indices) :
        _feature_ids(samples, idfun, all_idxs)
    labels = on_unsupported == :skip ?
        _feature_labels(samples, labelfun, sample_indices) :
        _feature_labels(samples, labelfun, all_idxs)

    caches = Vector{_InvariantCacheHandle}(undef, length(sample_indices))
    first_slot = 0
    @inbounds for slot in eachindex(sample_indices)
        sidx = sample_indices[slot]
        cache = build_cache(samples[sidx], spec;
                            opts=opts0,
                            threaded=batch_opts.threaded)
        caches[slot] = cache
        if sidx == first_supported
            first_slot = slot
        end
    end
    first_slot > 0 || error("_compile_featurize: internal error (missing first supported slot)")

    first_values = transform(spec, caches[first_slot]; opts=opts0, threaded=batch_opts.threaded)
    nf = length(first_values)
    names = _feature_names_or_default(spec, nf)
    nrows = on_unsupported == :skip ? length(sample_indices) : ns
    allow_missing = on_unsupported == :missing

    return _CompiledFeaturizePlan(
        spec, opts0, on_unsupported, cache_mode, batch_opts,
        names, axes_spec, ids, labels, skipped, session_cache,
        sample_indices, output_rows, caches, first_slot, first_values, nrows, nf, allow_missing,
    )
end

function _run_featurize!(X, compiled::_CompiledFeaturizePlan, batch_opts::BatchOptions)
    ns = length(compiled.sample_indices)
    ns == 0 && return X

    pstate = _progress_init(ns, "featurize", batch_opts.progress)
    row0 = compiled.output_rows[compiled.first_slot]
    @inbounds copyto!(view(X, row0, :), compiled.first_values)
    _progress_step!(pstate, 1)

    threaded_run = (
        batch_opts.threaded &&
        batch_opts.backend != :serial &&
        Threads.nthreads() > 1 &&
        ns > 1
    )

    _batch_foreach(ns, batch_opts) do slot
        slot == compiled.first_slot && return
        row = compiled.output_rows[slot]
        cache = compiled.caches[slot]
        transform!(view(X, row, :), compiled.spec, cache;
                   opts=compiled.opts,
                   threaded=batch_opts.threaded)
        if !threaded_run
            _progress_step!(pstate, 1)
        end
    end
    if threaded_run && ns > 1
        _progress_step!(pstate, ns - 1)
    end
    _progress_finish!(pstate)
    return X
end

"""
    featurize(samples, spec; opts=InvariantOptions(), idfun=nothing, labelfun=nothing,
              batch=BatchOptions(), cache=:auto, on_unsupported=:error) -> FeatureSet

Apply a typed featurizer spec to a batch of samples and return a matrix with
row-major sample layout (`nsamples x nfeatures`).
"""
function featurize(samples::AbstractVector,
                   spec::AbstractFeaturizerSpec;
                   opts::InvariantOptions=InvariantOptions(),
                   idfun=nothing,
                   labelfun=nothing,
                   batch::BatchOptions=BatchOptions(),
                   cache=:auto,
                   on_unsupported::Symbol=:error)
    opts0 = opts
    batch_opts = batch
    ns = length(samples)
    session_cache, cache_mode = _resolve_feature_session_cache(cache)
    (on_unsupported == :error || on_unsupported == :skip || on_unsupported == :missing) ||
        throw(ArgumentError("on_unsupported must be :error, :skip, or :missing"))

    if ns == 0
        nf = nfeatures(spec)
        X0 = zeros(Float64, 0, nf)
        names0 = _feature_names_or_default(spec, nf)
        meta0 = (spec=spec, opts=opts0, cache_mode=cache_mode, threaded=batch_opts.threaded, batch=batch_opts,
                 feature_axes=feature_axes(spec),
                 labels=nothing, unsupported_policy=on_unsupported,
                 skipped_indices=Int[])
        return FeatureSet(X0, names0, String[], meta0)
    end

    compiled = _compile_featurize(samples, spec, opts0, batch_opts, cache_mode,
                                  on_unsupported, session_cache, idfun, labelfun)
    X = if compiled.allow_missing
        Matrix{Union{Missing,Float64}}(missing, compiled.nrows, compiled.nfeatures)
    else
        Matrix{Float64}(undef, compiled.nrows, compiled.nfeatures)
    end
    _run_featurize!(X, compiled, batch_opts)

    meta = (spec=compiled.spec,
            opts=compiled.opts,
            cache_mode=compiled.cache_mode,
            threaded=batch_opts.threaded,
            batch=batch_opts,
            feature_axes=compiled.feature_axes,
            labels=compiled.labels,
            session_cache=compiled.session_cache,
            unsupported_policy=compiled.on_unsupported,
            skipped_indices=compiled.skipped_indices)
    return FeatureSet(X, compiled.feature_names, compiled.ids, meta)
end

"""
    batch_transform(samples, spec; kwargs...) -> FeatureSet

Explicit dataset-level batch transform entrypoint. Semantics are identical to
`featurize` and it returns the same `FeatureSet` payload.
"""
function batch_transform(samples::AbstractVector,
                         spec::AbstractFeaturizerSpec;
                         opts::InvariantOptions=InvariantOptions(),
                         idfun=nothing,
                         labelfun=nothing,
                         batch::BatchOptions=BatchOptions(),
                         cache=:auto,
                         on_unsupported::Symbol=:error)
    return featurize(samples, spec;
                     opts=opts,
                     idfun=idfun,
                     labelfun=labelfun,
                     batch=batch,
                     cache=cache,
                     on_unsupported=on_unsupported)
end

"""
    batch_transform!(X, samples, spec; kwargs...) -> FeatureSet

In-place dataset-level batch transform. `X` must have shape
`(nrows, nfeatures)` consistent with the chosen unsupported-sample policy.
Returns a `FeatureSet` wrapper around the provided matrix.
"""
function batch_transform!(X::Matrix,
                          samples::AbstractVector,
                          spec::AbstractFeaturizerSpec;
                          opts::InvariantOptions=InvariantOptions(),
                          idfun=nothing,
                          labelfun=nothing,
                          batch::BatchOptions=BatchOptions(),
                          cache=:auto,
                          on_unsupported::Symbol=:error)
    opts0 = opts
    batch_opts = batch
    session_cache, cache_mode = _resolve_feature_session_cache(cache)
    (on_unsupported == :error || on_unsupported == :skip || on_unsupported == :missing) ||
        throw(ArgumentError("on_unsupported must be :error, :skip, or :missing"))

    compiled = _compile_featurize(samples, spec, opts0, batch_opts, cache_mode,
                                  on_unsupported, session_cache, idfun, labelfun)
    size(X, 1) == compiled.nrows ||
        throw(DimensionMismatch("batch_transform!: destination has $(size(X,1)) rows, expected $(compiled.nrows)"))
    size(X, 2) == compiled.nfeatures ||
        throw(DimensionMismatch("batch_transform!: destination has $(size(X,2)) columns, expected $(compiled.nfeatures)"))

    if compiled.allow_missing
        has_missing_support = Missing <: eltype(X)
        if !has_missing_support && compiled.nrows != length(compiled.sample_indices)
            throw(ArgumentError("batch_transform!: destination element type $(eltype(X)) cannot represent missing values required by on_unsupported=:missing"))
        end
        has_missing_support && fill!(X, missing)
    end

    _run_featurize!(X, compiled, batch_opts)
    meta = (spec=compiled.spec,
            opts=compiled.opts,
            cache_mode=compiled.cache_mode,
            threaded=batch_opts.threaded,
            batch=batch_opts,
            feature_axes=compiled.feature_axes,
            labels=compiled.labels,
            session_cache=compiled.session_cache,
            unsupported_policy=compiled.on_unsupported,
            skipped_indices=compiled.skipped_indices)
    return FeatureSet(X, compiled.feature_names, compiled.ids, meta)
end

@inline function _resolution_cache_stats(rc::ResolutionCache)
    return Dict(
        "projective" => length(rc.projective),
        "injective" => length(rc.injective),
        "indicator" => length(rc.indicator),
        "projective_shards" => sum(length, rc.projective_shards),
        "injective_shards" => sum(length, rc.injective_shards),
        "indicator_shards" => sum(length, rc.indicator_shards),
    )
end

function _session_cache_stats(session::SessionCache)
    n_encoding = length(session.encoding)
    n_modules = length(session.modules)
    enc_posets = 0
    enc_cubical = 0
    enc_region_posets = 0
    mod_payload = 0
    mod_resolution_projective = 0
    mod_resolution_injective = 0
    mod_resolution_indicator = 0
    for ec in values(session.encoding)
        enc_posets += length(ec.posets)
        enc_cubical += length(ec.cubical)
        enc_region_posets += length(ec.region_posets)
    end
    for mc in values(session.modules)
        mod_payload += length(mc.payload)
        mod_resolution_projective += length(mc.resolution.projective)
        mod_resolution_injective += length(mc.resolution.injective)
        mod_resolution_indicator += length(mc.resolution.indicator)
    end
    out = Dict{String,Any}(
        "encoding_buckets" => n_encoding,
        "module_buckets" => n_modules,
        "encoding_posets" => enc_posets,
        "encoding_cubical" => enc_cubical,
        "encoding_region_posets" => enc_region_posets,
        "module_payload_entries" => mod_payload,
        "module_resolution_projective" => mod_resolution_projective,
        "module_resolution_injective" => mod_resolution_injective,
        "module_resolution_indicator" => mod_resolution_indicator,
        "session_resolution" => _resolution_cache_stats(session.resolution),
        "has_hom_system_cache" => session.hom_system !== nothing,
        "has_slice_plan_cache" => session.slice_plan !== nothing,
        "product_dense_entries" => length(session.product_dense),
        "product_object_entries" => length(session.product_obj),
    )
    return out
end

@inline function _experiment_cache_handle(cache)
    if cache === :auto
        return SessionCache(), :session_auto_shared
    elseif cache === nothing
        return nothing, :none
    elseif cache isa SessionCache
        return cache, :session_user
    end
    throw(ArgumentError("run_experiment: cache must be :auto, nothing, or SessionCache"))
end

@inline function _sanitize_filename(s::AbstractString)
    out = IOBuffer()
    for c in s
        if isletter(c) || isnumeric(c) || c == '_' || c == '-' || c == '.'
            print(out, c)
        else
            print(out, '_')
        end
    end
    str = String(take!(out))
    return isempty(str) ? "item" : str
end

@inline function _featurizer_key(spec::AbstractFeaturizerSpec, i::Int)
    tname = String(nameof(typeof(spec)))
    return Symbol("s", lpad(string(i), 2, '0'), "__", _sanitize_filename(lowercase(tname)))
end

@inline function _resolve_run_dir(io::ExperimentIOConfig, exp_name::String)
    do_write = !isempty(io.formats) || io.write_metadata
    do_write || return nothing
    base = io.outdir === nothing ? joinpath(pwd(), "experiment_outputs") : io.outdir
    run_tag = _sanitize_filename(io.prefix * "__" * exp_name)
    run_dir = joinpath(base, run_tag)
    if isdir(run_dir) && !io.overwrite
        throw(ArgumentError("run_experiment: output directory exists and overwrite=false: $(run_dir)"))
    end
    mkpath(run_dir)
    return run_dir
end

@inline function _run_one_experiment_featurizer(samples::AbstractVector,
                                                spec::AbstractFeaturizerSpec,
                                                opts::InvariantOptions,
                                                batch::BatchOptions,
                                                cache,
                                                on_unsupported::Symbol,
                                                idfun,
                                                labelfun)
    t0 = time_ns()
    fs = featurize(samples, spec;
                   opts=opts,
                   batch=batch,
                   cache=cache,
                   on_unsupported=on_unsupported,
                   idfun=idfun,
                   labelfun=labelfun)
    elapsed = (time_ns() - t0) / 1.0e9
    return fs, elapsed
end

"""
    run_experiment(exp::ExperimentSpec, samples) -> ExperimentResult

Run a featurization experiment over `samples` with one or more featurizers.

Behavior:
- returns in-memory `FeatureSet` outputs + timing/cache metadata,
- optionally writes Arrow/Parquet feature tables and JSON sidecars,
- writes a run manifest JSON when metadata writing is enabled.
"""
function run_experiment(exp::ExperimentSpec, samples::AbstractVector)
    run_dir = _resolve_run_dir(exp.io, exp.name)
    cache, cache_mode = _experiment_cache_handle(exp.cache)

    artifacts = Vector{ExperimentArtifact}(undef, length(exp.featurizers))
    t_total0 = time_ns()

    for i in eachindex(exp.featurizers)
        sp = exp.featurizers[i]
        key = _featurizer_key(sp, i)
        fs, elapsed = _run_one_experiment_featurizer(samples, sp, exp.opts, exp.batch, cache,
                                                     exp.on_unsupported, exp.idfun, exp.labelfun)
        cstats = cache isa SessionCache ? _session_cache_stats(cache) : Dict{String,Any}()
        feature_paths = Dict{Symbol,String}()
        metadata_path = nothing

        if run_dir !== nothing
            stem = _sanitize_filename(String(key))
            for fmt in exp.io.formats
                if fmt == :arrow
                    apath = joinpath(run_dir, stem * ".arrow")
                    save_features_arrow(apath, fs; format=exp.io.format, include_metadata=false)
                    feature_paths[:arrow] = apath
                elseif fmt == :parquet
                    ppath = joinpath(run_dir, stem * ".parquet")
                    save_features_parquet(ppath, fs; format=exp.io.format, include_metadata=false)
                    feature_paths[:parquet] = ppath
                elseif fmt == :npz
                    npath = joinpath(run_dir, stem * ".npz")
                    save_features_npz(npath, fs; format=exp.io.format, layout=:samples_by_features, include_metadata=false)
                    feature_paths[:npz] = npath
                elseif fmt == :csv_wide
                    cpath = joinpath(run_dir, stem * "__wide.csv")
                    save_features_csv(cpath, fs; format=:wide, layout=:samples_by_features, include_metadata=false)
                    feature_paths[:csv_wide] = cpath
                elseif fmt == :csv_long
                    cpath = joinpath(run_dir, stem * "__long.csv")
                    save_features_csv(cpath, fs; format=:long, layout=:samples_by_features, include_metadata=false)
                    feature_paths[:csv_long] = cpath
                else
                    throw(ArgumentError("run_experiment: unsupported output format $(fmt)"))
                end
            end
            if exp.io.write_metadata
                md = feature_metadata(fs; format=exp.io.format)
                md["kind"] = "experiment_feature"
                md["experiment_name"] = exp.name
                md["experiment_schema_version"] = string(TAMER_EXPERIMENT_SCHEMA_VERSION)
                md["featurizer_key"] = String(key)
                md["elapsed_seconds"] = elapsed
                md["cache_mode"] = String(cache_mode)
                md["cache_stats"] = _jsonable(cstats)
                md["batch"] = _jsonable(exp.batch)
                md["on_unsupported"] = String(exp.on_unsupported)
                md["artifact_formats"] = [String(f) for f in exp.io.formats]
                md["artifact_layout"] = "samples_by_features"
                md["feature_paths"] = _jsonable(feature_paths)
                metadata_path = joinpath(run_dir, stem * ".meta.json")
                save_metadata_json(metadata_path, md)
            end
        end

        artifacts[i] = ExperimentArtifact(key, sp, fs, elapsed, cstats, feature_paths, metadata_path)
    end

    total_elapsed = (time_ns() - t_total0) / 1.0e9
    manifest_path = nothing
    result_meta = Dict{String,Any}(
        "experiment_name" => exp.name,
        "experiment_schema_version" => string(TAMER_EXPERIMENT_SCHEMA_VERSION),
        "feature_schema_version" => string(TAMER_FEATURE_SCHEMA_VERSION),
        "n_featurizers" => length(exp.featurizers),
        "n_samples" => length(samples),
        "cache_mode" => String(cache_mode),
        "total_elapsed_seconds" => total_elapsed,
        "started_at_utc" => string(now(UTC)),
        "metadata" => _jsonable(exp.metadata),
    )
    if cache isa SessionCache
        result_meta["final_cache_stats"] = _jsonable(_session_cache_stats(cache))
    end

    if run_dir !== nothing && exp.io.write_metadata
        manifest = Dict{String,Any}(
            "kind" => "experiment_manifest",
            "experiment_name" => exp.name,
            "experiment_schema_version" => string(TAMER_EXPERIMENT_SCHEMA_VERSION),
            "feature_schema_version" => string(TAMER_FEATURE_SCHEMA_VERSION),
            "total_elapsed_seconds" => total_elapsed,
            "batch" => _jsonable(exp.batch),
            "cache_mode" => String(cache_mode),
            "on_unsupported" => String(exp.on_unsupported),
            "table_format" => String(exp.io.format),
            "artifact_formats" => [String(f) for f in exp.io.formats],
            "artifact_layout" => "samples_by_features",
            "run_dir" => run_dir,
            "artifacts" => [
                Dict(
                    "key" => String(a.key),
                    "featurizer_type" => string(nameof(typeof(a.spec))),
                    "elapsed_seconds" => a.elapsed_seconds,
                    "feature_paths" => _jsonable(a.feature_paths),
                    "metadata_path" => a.metadata_path,
                    "cache_stats" => _jsonable(a.cache_stats),
                    "n_features" => nfeatures(a.features),
                    "n_rows" => nsamples(a.features),
                ) for a in artifacts
            ],
            "metadata" => _jsonable(exp.metadata),
        )
        manifest_path = joinpath(run_dir, "manifest.json")
        save_metadata_json(manifest_path, manifest)
    end

    return ExperimentResult(exp, artifacts, total_elapsed, run_dir, manifest_path, result_meta)
end

function run_experiment(featurizers,
                        samples::AbstractVector;
                        name::AbstractString="experiment",
                        opts::InvariantOptions=InvariantOptions(),
                        batch::Union{BatchOptions,Nothing}=nothing,
                        cache=:auto,
                        on_unsupported::Symbol=:error,
                        idfun=nothing,
                        labelfun=nothing,
                        io::ExperimentIOConfig=ExperimentIOConfig(),
                        metadata=NamedTuple())
    exp = ExperimentSpec(featurizers;
                         name=name,
                         opts=opts,
                         batch=batch,
                         cache=cache,
                         on_unsupported=on_unsupported,
                         idfun=idfun,
                         labelfun=labelfun,
                         io=io,
                         metadata=metadata)
    return run_experiment(exp, samples)
end

@inline function _resolve_manifest_path(path::AbstractString)
    if isdir(path)
        mp = joinpath(path, "manifest.json")
        isfile(mp) || throw(ArgumentError("load_experiment: manifest.json not found in directory $(path)"))
        return mp
    end
    isfile(path) || throw(ArgumentError("load_experiment: path does not exist: $(path)"))
    return String(path)
end

@inline function _abspath_or_join(base::Union{Nothing,AbstractString}, path::AbstractString)
    if isabspath(path)
        return String(path)
    end
    base === nothing && return String(path)
    return joinpath(String(base), String(path))
end

@inline function _feature_paths_from_manifest(entry, run_dir::Union{Nothing,String})
    out = Dict{Symbol,String}()
    raw = _obj_get(entry, "feature_paths", nothing)
    raw === nothing && return out
    for (k, v) in pairs(raw)
        ks = Symbol(String(k))
        vs = String(v)
        out[ks] = _abspath_or_join(run_dir, vs)
    end
    return out
end

@inline function _pick_feature_path(paths::Dict{Symbol,String}, prefer::Symbol)
    if isempty(paths)
        return nothing, nothing
    end
    if prefer == :arrow
        return get(paths, :arrow, nothing), :arrow
    elseif prefer == :parquet
        return get(paths, :parquet, nothing), :parquet
    elseif prefer == :npz
        return get(paths, :npz, nothing), :npz
    elseif prefer == :csv_wide
        return get(paths, :csv_wide, nothing), :csv_wide
    elseif prefer == :csv_long
        return get(paths, :csv_long, nothing), :csv_long
    elseif prefer == :csv
        if haskey(paths, :csv_wide)
            return paths[:csv_wide], :csv_wide
        elseif haskey(paths, :csv_long)
            return paths[:csv_long], :csv_long
        end
        return nothing, nothing
    elseif prefer == :auto
        if haskey(paths, :arrow)
            return paths[:arrow], :arrow
        elseif haskey(paths, :parquet)
            return paths[:parquet], :parquet
        elseif haskey(paths, :npz)
            return paths[:npz], :npz
        elseif haskey(paths, :csv_wide)
            return paths[:csv_wide], :csv_wide
        elseif haskey(paths, :csv_long)
            return paths[:csv_long], :csv_long
        end
        return nothing, nothing
    end
    throw(ArgumentError("load_experiment: prefer must be :auto, :arrow, :parquet, :npz, :csv, :csv_wide, :csv_long, or :none"))
end

@inline function _load_feature_file(path::AbstractString, fmt::Symbol;
                                    table_format::Symbol=:wide,
                                    ids_col::Symbol=:id)
    if fmt == :arrow
        return load_features_arrow(path; format=table_format, ids_col=ids_col)
    elseif fmt == :parquet
        return load_features_parquet(path; format=table_format, ids_col=ids_col)
    elseif fmt == :npz
        return load_features_npz(path; format=table_format)
    elseif fmt == :csv_wide
        return load_features_csv(path; format=:wide, ids_col=ids_col)
    elseif fmt == :csv_long
        return load_features_csv(path; format=:long, ids_col=ids_col)
    end
    throw(ArgumentError("_load_feature_file: unsupported feature format $(fmt)"))
end

"""
    load_experiment(path; load_features=true, prefer=:auto, strict=true,
                    resolve_ref=nothing, require_resolved_refs=false) -> LoadedExperimentResult

Load an experiment run from a manifest path (or run directory containing `manifest.json`).

`load_features` controls whether feature tables are loaded from persisted files.
`prefer` chooses file format when multiple exist (`:auto`, `:arrow`, `:parquet`,
`:npz`, `:csv`, `:csv_wide`, `:csv_long`, `:none`).
"""
function load_experiment(path::AbstractString;
                         load_features::Bool=true,
                         prefer::Symbol=:auto,
                         strict::Bool=true,
                         resolve_ref::Union{Nothing,Function}=nothing,
                         require_resolved_refs::Bool=false)
    prefer in (:auto, :arrow, :parquet, :npz, :csv, :csv_wide, :csv_long, :none) ||
        throw(ArgumentError("load_experiment: prefer must be :auto, :arrow, :parquet, :npz, :csv, :csv_wide, :csv_long, or :none"))

    manifest_path = _resolve_manifest_path(path)
    manifest_raw = load_metadata_json(manifest_path; validate_feature_schema=false)
    manifest = _to_plain(manifest_raw)
    kind = String(_obj_get(manifest, "kind", ""))
    kind == "experiment_manifest" ||
        throw(ArgumentError("load_experiment: expected kind=experiment_manifest, got $(kind)"))

    run_dir = _obj_get(manifest, "run_dir", dirname(manifest_path))
    run_dir = run_dir === nothing ? nothing : String(run_dir)
    entries = _obj_get(manifest, "artifacts", Any[])
    artifacts = Vector{LoadedExperimentArtifact}(undef, length(entries))

    for i in eachindex(entries)
        e = entries[i]
        key = Symbol(String(_obj_get(e, "key", "artifact_$(i)")))
        elapsed = Float64(_obj_get(e, "elapsed_seconds", 0.0))
        cstats_raw = _obj_get(e, "cache_stats", Dict{String,Any}())
        cstats_plain = _to_plain(cstats_raw)
        cstats = cstats_plain isa AbstractDict ? Dict{String,Any}(String(k)=>v for (k,v) in pairs(cstats_plain)) : Dict{String,Any}()

        metadata_path = _obj_get(e, "metadata_path", nothing)
        metadata_path = metadata_path === nothing ? nothing : _abspath_or_join(run_dir, String(metadata_path))
        md = nothing
        spec = nothing
        opts = nothing
        if metadata_path !== nothing && isfile(metadata_path)
            md = load_metadata_json(metadata_path; validate_feature_schema=false)
            if _obj_haskey(md, "kind")
                md_kind = String(md["kind"])
                if !(md_kind == "features" || md_kind == "experiment_feature")
                    strict && throw(ArgumentError("load_experiment: unsupported artifact metadata kind $(md_kind) for $(String(key))"))
                end
            end
            try
                spec = _obj_haskey(md, "spec") ? spec_from_metadata(md["spec"];
                                                                    resolve_ref=resolve_ref,
                                                                    require_resolved_refs=require_resolved_refs) : nothing
            catch err
                strict && rethrow(err)
                spec = nothing
            end
            try
                opts = _obj_haskey(md, "opts") ? invariant_options_from_metadata(md["opts"]) : nothing
            catch err
                strict && rethrow(err)
                opts = nothing
            end
        elseif strict
            throw(ArgumentError("load_experiment: missing metadata sidecar for artifact $(String(key))"))
        end

        feature_paths = _feature_paths_from_manifest(e, run_dir)
        feat = nothing
        table_format = if md !== nothing && _obj_haskey(md, "format")
            Symbol(String(md["format"]))
        else
            :wide
        end
        if load_features && prefer != :none
            fpath, ffmt = _pick_feature_path(feature_paths, prefer)
            if fpath === nothing
                strict && throw(ArgumentError("load_experiment: no feature file path found for artifact $(String(key))"))
            elseif isfile(fpath)
                try
                    feat = _load_feature_file(fpath, ffmt; table_format=table_format)
                catch err
                    strict && rethrow(err)
                    feat = nothing
                end
            elseif strict
                throw(ArgumentError("load_experiment: feature file not found for artifact $(String(key)): $(fpath)"))
            end
        end

        artifacts[i] = LoadedExperimentArtifact(
            key,
            spec,
            opts,
            feat,
            md,
            elapsed,
            cstats,
            feature_paths,
            metadata_path,
        )
    end

    total_elapsed = Float64(_obj_get(manifest, "total_elapsed_seconds", 0.0))
    return LoadedExperimentResult(manifest, artifacts, run_dir, manifest_path, total_elapsed)
end


"""
    matching_distance(encA, encB; method=:auto, opts=InvariantOptions(), kwargs...)

Distance between two encoded modules, assuming a common encoding map `pi` (as produced by
`encode([A,B]; ...)` style common-encoding).

`method`:
- `:auto` or `:approx`    uses `Invariants.matching_distance_approx`
- `:exact_2d`             uses `Invariants.matching_distance_exact_2d`
"""
function matching_distance(encA::EncodingResult, encB::EncodingResult;
                           method::Symbol=:auto,
                           opts::InvariantOptions=InvariantOptions(),
                           cache=:auto,
                           kwargs...)
    opts = opts
    (encA.P === encB.P) || error("matching_distance: encodings are on different posets; common-encode first.")
    (encA.pi === encB.pi) || error("matching_distance: encodings do not share a common classifier map pi; common-encode first.")

    pi = encA.pi
    cache_slice, session_cache = _resolve_workflow_specialized_cache(cache, Invariants.SlicePlanCache)
    cache2 = _slice_plan_cache_from_session(cache_slice, session_cache)
    if method == :auto || method == :approx
        return Invariants.matching_distance_approx(encA.M, encB.M, pi, opts; cache=cache2, kwargs...)
    elseif method == :exact_2d
        return Invariants.matching_distance_exact_2d(encA.M, encB.M, pi, opts; kwargs...)
    else
        error("matching_distance: unknown method=$(method). Supported: :auto, :approx, :exact_2d")
    end
end

end # module Workflow
