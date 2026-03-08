"""
    RegionGeometry

Geometry-related extension points and generic geometry utilities for encoding maps.

This module intentionally lives outside `CoreModules` to keep `CoreModules` a small
project prelude (scalars, tiny utilities, and core interface hooks).

Design:
- Backends extend hook functions such as `region_weights`, `region_bbox`,
  `region_adjacency`, and `region_boundary_measure`.
- This module provides generic wrappers and derived quantities (diameter,
  aspect ratio, isoperimetric ratio, anisotropy scores, etc.) built on those hooks.

Note:
- The classifier hook `locate(pi, x)` lives in `EncodingCore` and is imported here
  for algorithms that need point membership queries.
"""
module RegionGeometry

using LinearAlgebra
using Random

import ..EncodingCore: locate, locate_many!, _geometry_fingerprint, _locate_call_style

# Internal, inspectable gates for A/B testing and conservative rollout.
const _REGION_BATCHED_LOCATE = Ref(true)
const _REGION_FAST_WRAPPERS = Ref(true)
const _REGION_BATCHED_LOCATE_MIN_PROPOSALS = Ref(128)
const _REGION_LOCATE_BATCH_SIZE = Ref(256)
const _REGION_DIRECT_VOLUME = Ref(true)
const _REGION_WORKSPACE_REUSE = Ref(true)
const _REGION_BLOCKED_PROJECTION = Ref(true)
const _REGION_BLOCKED_PROJECTION_MIN_ACCEPTED = Ref(8)
const _REGION_BLOCKED_COVARIANCE = Ref(true)
const _REGION_SAMPLED_SUMMARY_CACHE = Ref(true)
const _REGION_SHARDED_SAMPLE_CACHE = Ref(true)
const _REGION_FAST_CACHE_KEYS = Ref(true)
const _REGION_SAMPLE_CACHE_MAX = Ref(256)
const _REGION_CACHE_SHARDS = 16

# -----------------------------------------------------------------------------
"""
    region_weights(pi; box=nothing, kwargs...) -> AbstractVector

Optional hook for weighting invariants computed on a finite encoding.

Many invariants in multiparameter persistence reduce, after choosing a finite
encoding map `pi : Q -> P`, to data indexed by regions/vertices in a finite
poset `P`. When comparing two modules via the restricted Hilbert function
(dimension surface), it is often useful to weight each region by its "size"
(e.g. volume in R^n, or lattice-point count in Z^n) inside a bounding box.

This function is an *extension point* (like `locate`). Concrete encoding map
types may provide methods.

Common conventions:
- `region_weights(pi; box=(a,b))` returns a vector `w` with `w[t] >= 0`.
- `length(w)` should equal the number of encoded regions (i.e. `P.n`).
- If no meaningful weighting is available, a method may return all ones.
"""
function region_weights end

"""
    region_bbox(pi, r; box=nothing, kwargs...)

Geometric hook: return a bounding box for region `r` of an encoding map `pi`.

Intended meaning:
- For encodings of R^n: return an axis-aligned bounding box for the region in R^n.
- Many regions are naturally unbounded; in that case, you should either:
  * require a user-supplied finite `box=(a,b)` and compute the bbox of (region intersect box), or
  * return `-Inf` / `Inf` in unbounded directions.

Return convention:
- Return `nothing` if the region has empty intersection with the supplied `box`
  (or is otherwise empty for the intended semantics).
- Otherwise return `(a, b)` where `a[i] <= b[i]` are the lower/upper corners.

This is an extension point: concrete encoding map types may provide methods.
"""
function region_bbox end

"""
    region_volume(pi, r; box=nothing, kwargs...) -> Real

Geometric hook: return the volume/weight of a single region.

Concrete backends are encouraged to implement this directly when a single-region
query is materially cheaper than building the full vector returned by
`region_weights`. The default fallback still extracts the `r`-th entry from
`region_weights`.
"""
function region_volume(pi, r::Integer; box=nothing, closure::Bool=true, cache=nothing, kwargs...)
    box = _resolve_box(box, cache)
    if _REGION_DIRECT_VOLUME[]
        fast = _region_volume_fast(pi, r; box=box, closure=closure, cache=cache)
        fast === nothing || return float(fast)
    end
    return _region_volume_from_weights(pi, r; box=box, closure=closure, cache=cache, kwargs...)
end

"""
    region_diameter(pi, r; metric=:L2, box=nothing, method=:bbox, kwargs...) -> Real

Estimate the diameter of region `r` under the chosen metric.

Default implementation (`method == :bbox`):
Compute the diameter of the bounding box returned by
`region_bbox(pi, r; box=box, kwargs...)`.

This default is conservative: it is an upper bound on the true diameter inside
the box (if the region is non-convex, disconnected, etc.).

Metrics supported by the default method:
- `:L2`   (Euclidean)
- `:Linf` (max norm)
- `:L1`   (taxicab)

Concrete encodings may provide specialized methods, e.g. a vertex-based diameter
for convex polyhedral regions.
"""
function region_diameter(
    pi,
    r::Integer;
    metric::Symbol=:L2,
    box=nothing,
    method::Symbol=:bbox,
    kwargs...
)
    method == :bbox || error("region_diameter: method must be :bbox (or implement a specialized method)")

    bb = region_bbox(pi, r; box=box, kwargs...)
    bb === nothing && return 0.0

    a, b = bb
    length(a) == length(b) || error("region_diameter: region_bbox returned mismatched endpoints")
    n = length(a)

    if metric == :Linf
        d = 0.0
        for i in 1:n
            li = float(b[i]) - float(a[i])
            isfinite(li) || return Inf
            d = max(d, abs(li))
        end
        return d
    elseif metric == :L2
        acc = 0.0
        for i in 1:n
            li = float(b[i]) - float(a[i])
            isfinite(li) || return Inf
            acc += li * li
        end
        return sqrt(acc)
    elseif metric == :L1
        s = 0.0
        for i in 1:n
            li = float(b[i]) - float(a[i])
            isfinite(li) || return Inf
            s += abs(li)
        end
        return s
    else
        error("region_diameter: metric must be :L1, :L2, or :Linf")
    end
end

"""
    region_widths(pi, r; box=nothing, kwargs...) -> Union{Nothing,Vector{Float64}}

Side lengths of the axis-aligned bounding box of region `r`.

This is computed from `region_bbox(pi, r; box=box, kwargs...)` as `b - a`,
where `(a,b)` is the returned bounding box.

Returns `nothing` if `region_bbox` returns `nothing` (empty intersection).

Note:
- If you do not supply a finite `box`, some coordinates may be unbounded and
  the corresponding widths may be `Inf`.
"""
function region_widths(pi, r::Integer; box=nothing, kwargs...)
    bb = region_bbox(pi, r; box=box, kwargs...)
    bb === nothing && return nothing
    a, b = bb
    length(a) == length(b) || error("region_widths: region_bbox returned mismatched endpoints")
    n = length(a)
    w = Vector{Float64}(undef, n)
    for i in 1:n
        w[i] = float(b[i]) - float(a[i])
    end
    return w
end

"""
    region_centroid(pi, r; box=nothing, method=:bbox, kwargs...) -> Union{Nothing,Vector{Float64}}

A convenient representative point ("centroid") for region `r`.

By default (`method == :bbox`), this returns the midpoint of the bounding box
returned by `region_bbox(pi, r; box=box, kwargs...)`.

Returns `nothing` if:
- the region does not intersect the box, or
- the bounding box is unbounded in some coordinate (supply a finite `box`).

Encodings may provide more refined notions (e.g. true centroid, Monte Carlo
centroid) by adding specialized methods.
"""
function region_centroid(pi, r::Integer; box=nothing, method::Symbol=:bbox, kwargs...)
    method == :bbox || error("region_centroid: unsupported method $method (default is :bbox)")
    cache = _kwget(kwargs, :cache, nothing)
    closure = _kwget(kwargs, :closure, true)
    box = _resolve_box(box, cache)
    return _region_centroid_maybe_cached(pi, r; box=box, method=:bbox, closure=closure, cache=cache)
end

"""
    region_aspect_ratio(pi, r; box=nothing, kwargs...) -> Real

A simple anisotropy proxy for region `r` based on its axis-aligned bounding box.

Let `w = region_widths(pi, r; ...)`. This function returns:

    max(w) / min_positive(w)

where `min_positive(w)` is the smallest strictly positive width.

Conventions:
- Returns 0.0 if the region has empty intersection with the given box.
- Returns 1.0 if the region collapses to a point (all widths are 0).
- Returns Inf if the region is unbounded (some width is Inf) or degenerate
  (some widths are 0 while others are positive).
"""
function region_aspect_ratio(pi, r::Integer; box=nothing, kwargs...)
    w = region_widths(pi, r; box=box, kwargs...)
    w === nothing && return 0.0

    maxw = 0.0
    minw = Inf
    for wi in w
        aw = abs(float(wi))
        isfinite(aw) || return Inf
        maxw = max(maxw, aw)
        if aw > 0.0
            minw = min(minw, aw)
        end
    end

    if maxw == 0.0
        return 1.0
    elseif minw == Inf
        return Inf
    else
        return maxw / minw
    end
end

"""
    region_adjacency(pi; box, kwargs...) -> AbstractDict

Geometric hook: report how regions meet inside a finite bounding window.

Implementations should return a dictionary whose keys are unordered region pairs
`(r,s)` with `r < s`. The value is the estimated (n-1)-dimensional measure of
the interface between regions `r` and `s` inside `box=(a,b)`.

Notes:
- For n=2 this is (a notion of) boundary length.
- For n=3 this is boundary area.
- For n=1 this is a 0-dimensional measure (counting boundary points), so each
  interior boundary typically contributes 1.

This is currently implemented for axis-aligned box grids (`PLEncodingMapBoxes`).
"""
function region_adjacency end

"""
    region_facet_count(pi, r; kwargs...) -> Int

Geometric hook: return a facet/constraint count for region `r`.

For polyhedral backends, a reasonable default is the number of inequalities in
the stored H-representation. This is a complexity proxy (it may count redundant
inequalities and thus may exceed the true number of facets).
"""
function region_facet_count end

"""
    region_vertex_count(pi, r; box, kwargs...) -> Union{Nothing,Int}

Geometric hook: return the number of vertices of `region r` inside the window
`box=(a,b)`.

This may be expensive. Implementations are allowed to return `nothing` if vertex
enumeration is not attempted (e.g. too many constraints / combinations).
"""
function region_vertex_count end


"""
    region_boundary_measure(pi, r; box=nothing, strict=true)

Boundary measure of region `r` inside the window `box=(a,b)`.

This is the (n-1)-dimensional Hausdorff measure of the boundary of `R_r cap box`:
* n == 2: perimeter
* n == 3: surface area
* general n: hypersurface measure

Backends should implement this when feasible. The default method throws an error.
"""
function region_boundary_measure(pi, r; box=nothing, strict=true)
    error("region_boundary_measure not implemented for $(typeof(pi))")
end

"""
    region_boundary_measure_breakdown(pi, r; box=nothing, kwargs...) -> Vector{NamedTuple}

Diagnostic geometric hook: a per-facet decomposition of the boundary measure of
region `r` inside the window `box=(a,b)`.

Each entry is a NamedTuple and should contain at least:
- `measure::Float64`

Backends may optionally provide additional fields such as:
- `normal::Vector{Float64}`: an (outward) normal of the facet (not necessarily unit).
- `point::Vector{Float64}`: a representative point on the facet (e.g. barycenter).
- `neighbor::Union{Nothing,Int}`: the neighboring region across the facet, if detectable.
- `kind::Symbol`: e.g. `:internal` (between regions) or `:box` (window boundary).

The total boundary measure should satisfy, approximately:

    region_boundary_measure(pi, r; box=box) approx sum(e.measure for e in breakdown)

This is a diagnostic-level query and may not be implemented for all encodings.
"""
function region_boundary_measure_breakdown(pi, r; box=nothing, kwargs...)
    error("region_boundary_measure_breakdown not implemented for $(typeof(pi))")
end


# --- helper hooks ------------------------------------------------------------

# Owner backends may provide a concrete cached box and direct cached geometry
# answers. RegionGeometry keeps the default path generic and only uses these
# hooks when a backend opts in with a typed method.
@inline _cache_box(::Nothing) = nothing
@inline _cache_box(cache) = nothing
@inline _resolve_box(box, cache) = box === nothing ? _cache_box(cache) : box

@inline _region_bbox_fast(pi, r::Integer;
    box, strict::Bool=true, closure::Bool=true, cache=nothing) = nothing
@inline _region_centroid_fast(pi, r::Integer;
    box, method::Symbol=:bbox, closure::Bool=true, cache=nothing) = nothing
@inline _region_volume_fast(pi, r::Integer; box, closure::Bool=true, cache=nothing) = nothing
@inline _region_boundary_measure_fast(pi, r::Integer;
    box, strict::Bool=true, closure::Bool=true, cache=nothing) = nothing
@inline _region_circumradius_fast(pi, r::Integer;
    box, center=:bbox, metric::Symbol=:L2, method::Symbol=:bbox,
    strict::Bool=true, closure::Bool=true, cache=nothing) = nothing
@inline _region_minkowski_functionals_fast(pi, r::Integer;
    box, volume=nothing, boundary=nothing, mean_width_method::Symbol=:auto,
    mean_width_ndirs::Integer=256, mean_width_rng=Random.default_rng(),
    mean_width_directions=nothing, strict::Bool=true, closure::Bool=true,
    cache=nothing) = nothing

@inline _kwget(kwargs, key::Symbol, default) = get(kwargs, key, default)

@inline function _locate_dispatch(pi, x, ::Val{:strict_closure}; strict::Bool, closure::Bool)
    return locate(pi, x; strict=strict, closure=closure)
end

@inline function _locate_dispatch(pi, x, ::Val{:strict_only}; strict::Bool, closure::Bool)
    _ = closure
    return locate(pi, x; strict=strict)
end

@inline function _locate_dispatch(pi, x, ::Val{:plain}; strict::Bool, closure::Bool)
    _ = (strict, closure)
    return locate(pi, x)
end

function _resolve_locate_style(pi, x0::AbstractVector{<:Real}; strict::Bool, closure::Bool)
    return _locate_call_style(pi, x0; strict=strict, closure=closure, batched=false)
end

@inline function _locate_many_dispatch!(dest, pi, X, ::Val{:strict_closure}; strict::Bool, closure::Bool)
    return locate_many!(dest, pi, X; strict=strict, closure=closure)
end

@inline function _locate_many_dispatch!(dest, pi, X, ::Val{:strict_only}; strict::Bool, closure::Bool)
    _ = closure
    return locate_many!(dest, pi, X; strict=strict)
end

@inline function _locate_many_dispatch!(dest, pi, X, ::Val{:plain}; strict::Bool, closure::Bool)
    _ = (strict, closure)
    return locate_many!(dest, pi, X)
end

function _resolve_locate_many_style(pi, x0::AbstractVector{<:Real}; strict::Bool, closure::Bool)
    return _locate_call_style(pi, x0; strict=strict, closure=closure, batched=true)
end

@inline function _zero_moments!(sumx::AbstractVector{Float64}, sumxx::AbstractMatrix{Float64})
    fill!(sumx, 0.0)
    fill!(sumxx, 0.0)
    return nothing
end

@inline function _compute_block_moments!(ws::_RegionBatchWorkspace,
    X::AbstractMatrix{Float64}, ncols::Int)
    ncols <= 0 && return nothing
    n = size(X, 1)
    block_sum = ws.block_sum
    fill!(block_sum, 0.0)
    gram = ws.gram
    if !_REGION_BLOCKED_COVARIANCE[] || ncols <= 1
        fill!(gram, 0.0)
        @inbounds for j in 1:ncols
            for i in 1:n
                xij = X[i, j]
                block_sum[i] += xij
            end
            for c in 1:n, r in 1:n
                gram[r, c] += X[r, j] * X[c, j]
            end
        end
        return nothing
    end
    @inbounds for j in 1:ncols, i in 1:n
        block_sum[i] += X[i, j]
    end
    mul!(gram, view(X, 1:n, 1:ncols), transpose(view(X, 1:n, 1:ncols)))
    return nothing
end

@inline function _accumulate_block_moments!(sumx::AbstractVector{Float64},
    sumxx::AbstractMatrix{Float64}, ws::_RegionBatchWorkspace)
    @inbounds for i in eachindex(sumx)
        sumx[i] += ws.block_sum[i]
    end
    @inbounds for j in axes(sumxx, 2), i in axes(sumxx, 1)
        sumxx[i, j] += ws.gram[i, j]
    end
    return nothing
end

mutable struct _RegionBatchWorkspace
    n::Int
    batchsize::Int
    ndirs::Int
    mu::Vector{Float64}
    C::Matrix{Float64}
    mu_b::Vector{Float64}
    C_b::Matrix{Float64}
    block_sum::Vector{Float64}
    gram::Matrix{Float64}
    X::Matrix{Float64}
    locs::Vector{Int}
    x::Vector{Float64}
    accepted::Matrix{Float64}
    proj::Matrix{Float64}
    minproj::Vector{Float64}
    maxproj::Vector{Float64}
end

@inline function _RegionBatchWorkspace(n::Integer, batchsize::Integer, ndirs::Integer=0)
    return _RegionBatchWorkspace(
        Int(n),
        Int(batchsize),
        Int(ndirs),
        Vector{Float64}(undef, n),
        Matrix{Float64}(undef, n, n),
        Vector{Float64}(undef, n),
        Matrix{Float64}(undef, n, n),
        Vector{Float64}(undef, n),
        Matrix{Float64}(undef, n, n),
        Matrix{Float64}(undef, n, batchsize),
        Vector{Int}(undef, batchsize),
        Vector{Float64}(undef, n),
        Matrix{Float64}(undef, n, batchsize),
        Matrix{Float64}(undef, max(1, ndirs), batchsize),
        Vector{Float64}(undef, max(1, ndirs)),
        Vector{Float64}(undef, max(1, ndirs)),
    )
end

const _REGION_WORKSPACES = let n = max(1, Base.Threads.nthreads())
    [Dict{Tuple{Int,Int,Int},_RegionBatchWorkspace}() for _ in 1:n]
end

mutable struct _RegionSampleCache{T}
    locks::Vector{Base.ReentrantLock}
    shards::Vector{Dict{UInt64,T}}
end

@inline _region_cache_nshards() = Threads.nthreads() == 1 ? 1 : _REGION_CACHE_SHARDS

function _RegionSampleCache(::Type{T}) where {T}
    nshards = _region_cache_nshards()
    return _RegionSampleCache([Base.ReentrantLock() for _ in 1:nshards],
        [Dict{UInt64,T}() for _ in 1:nshards])
end

@inline function _region_cache_shard(cache::_RegionSampleCache, key::UInt64)
    nshards = (_REGION_SHARDED_SAMPLE_CACHE[] ? length(cache.shards) : 1)
    return Int(mod(key, UInt64(nshards))) + 1
end

@inline function _region_cache_shard_cap(cache::_RegionSampleCache)
    return max(1, cld(_REGION_SAMPLE_CACHE_MAX[], max(1, (_REGION_SHARDED_SAMPLE_CACHE[] ? length(cache.shards) : 1))))
end

function Base.length(cache::_RegionSampleCache)
    total = 0
    for shard in cache.shards
        total += length(shard)
    end
    return total
end

struct _PrincipalSummaryEntry
    mean::Vector{Float64}
    cov::Matrix{Float64}
    evals::Vector{Float64}
    evecs::Matrix{Float64}
    mean_stderr::Vector{Float64}
    evals_stderr::Vector{Float64}
    batch_evals::Vector{Vector{Float64}}
    batch_n_accepted::Vector{Int}
    n_accepted::Int
    n_proposed::Int
end

const _REGION_PRINCIPAL_SUMMARY_CACHE = _RegionSampleCache(_PrincipalSummaryEntry)
const _REGION_MEAN_WIDTH_CACHE = _RegionSampleCache(Float64)

function _clear_region_geometry_runtime_caches!()
    for cache in (_REGION_PRINCIPAL_SUMMARY_CACHE, _REGION_MEAN_WIDTH_CACHE)
        for i in eachindex(cache.shards)
            Base.lock(cache.locks[i])
            try
                empty!(cache.shards[i])
            finally
                Base.unlock(cache.locks[i])
            end
        end
    end
    for ws in _REGION_WORKSPACES
        empty!(ws)
    end
    return nothing
end

@inline function _region_workspace(n::Integer, batchsize::Integer, ndirs::Integer=0)
    if !_REGION_WORKSPACE_REUSE[]
        return _RegionBatchWorkspace(n, batchsize, ndirs)
    end
    tid = min(Base.Threads.threadid(), length(_REGION_WORKSPACES))
    key = (Int(n), Int(batchsize), Int(ndirs))
    return get!(_REGION_WORKSPACES[tid], key) do
        _RegionBatchWorkspace(n, batchsize, ndirs)
    end
end

@inline function _sample_cache_get(cache::_RegionSampleCache{T}, key::UInt64) where {T}
    shard = _region_cache_shard(cache, key)
    Base.lock(cache.locks[shard])
    try
        return get(cache.shards[shard], key, nothing)
    finally
        Base.unlock(cache.locks[shard])
    end
end

@inline function _sample_cache_insert!(cache::_RegionSampleCache{T}, key::UInt64, value::T) where {T}
    shard = _region_cache_shard(cache, key)
    Base.lock(cache.locks[shard])
    try
        bucket = cache.shards[shard]
        if length(bucket) >= _region_cache_shard_cap(cache)
            empty!(bucket)
        end
        bucket[key] = value
    finally
        Base.unlock(cache.locks[shard])
    end
    return value
end

@inline function _hash_float_sequence(xs)
    h = hash(length(xs))
    @inbounds for x in xs
        h = hash(float(x), h)
    end
    return h
end

@inline _box_cache_hash(box) = hash((_hash_float_sequence(box[1]), _hash_float_sequence(box[2])))

@inline _pi_geometry_cache_hash(pi) = hash(_geometry_fingerprint(pi))

function _rng_cache_hash_slow(rng)
    rc = try
        copy(rng)
    catch
        try
            deepcopy(rng)
        catch
            return nothing
        end
    end
    vals = ntuple(_ -> rand(rc, UInt64), 4)
    return hash((typeof(rng), vals))
end

function _rng_cache_hash(rng::Random.MersenneTwister)
    if !_REGION_FAST_CACHE_KEYS[]
        return _rng_cache_hash_slow(rng)
    end
    st = getfield(rng, :state)
    return hash((typeof(rng), getfield(rng, :seed), getfield(rng, :idxF), getfield(rng, :idxI),
        getfield(rng, :adv), getfield(rng, :adv_jump), getfield(rng, :adv_vals),
        getfield(rng, :adv_ints), hash(getfield(st, :val))))
end

function _rng_cache_hash(rng::Random.Xoshiro)
    if !_REGION_FAST_CACHE_KEYS[]
        return _rng_cache_hash_slow(rng)
    end
    return hash((typeof(rng), getfield(rng, :s0), getfield(rng, :s1),
        getfield(rng, :s2), getfield(rng, :s3), getfield(rng, :s4)))
end

_rng_cache_hash(rng) = _rng_cache_hash_slow(rng)

function _directions_cache_hash(directions, ndirs::Integer)
    directions === nothing && return hash((:random, ndirs))
    return hash((size(directions), _hash_float_sequence(directions)))
end

function _principal_summary_cache_key(pi, r::Integer;
    box, strict::Bool, closure::Bool, nsamples::Integer,
    max_proposals::Integer, return_info::Bool, nbatches::Integer, rng)
    rngh = _rng_cache_hash(rng)
    rngh === nothing && return nothing
    return UInt64(hash((_pi_geometry_cache_hash(pi), Int(r), _box_cache_hash(box),
        strict, closure, Int(nsamples), Int(max_proposals),
        return_info, Int(nbatches), rngh)))
end

function _mean_width_cache_key(pi, r::Integer;
    box, strict::Bool, closure::Bool, nsamples::Integer,
    max_proposals::Integer, ndirs::Integer, directions, rng)
    rngh = _rng_cache_hash(rng)
    rngh === nothing && return nothing
    dirh = _directions_cache_hash(directions, ndirs)
    return UInt64(hash((_pi_geometry_cache_hash(pi), Int(r), _box_cache_hash(box),
        strict, closure, Int(nsamples), Int(max_proposals), Int(ndirs), dirh, rngh)))
end


"""
    region_perimeter(pi, r; box, kwargs...) -> Float64

Convenience wrapper for 2D regions. This calls
`region_boundary_measure(pi, r; box=box, kwargs...)`.
"""
function region_perimeter(pi, r; box=nothing, cache=nothing, kwargs...)
    box = _resolve_box(box, cache)
    box === nothing && error("region_perimeter: please provide box=(a,b)")

    a, _ = box
    length(a) == 2 || error("region_perimeter: expected 2D box, got length(a)=$(length(a))")
    if cache === nothing
        return float(region_boundary_measure(pi, r; box=box, kwargs...))
    end
    return float(region_boundary_measure(pi, r; box=box, cache=cache, kwargs...))
end

"""
    region_surface_area(pi, r; box, kwargs...) -> Float64

Convenience wrapper for 3D and higher. This calls
`region_boundary_measure(pi, r; box=box, kwargs...)`.
"""
function region_surface_area(pi, r; box=nothing, cache=nothing, kwargs...)
    box = _resolve_box(box, cache)
    box === nothing && error("region_surface_area: please provide box=(a,b)")

    a, _ = box
    length(a) >= 3 || error("region_surface_area: expected 3D+ box, got length(a)=$(length(a))")
    if cache === nothing
        return float(region_boundary_measure(pi, r; box=box, kwargs...))
    end
    return float(region_boundary_measure(pi, r; box=box, cache=cache, kwargs...))
end

"""
    region_principal_directions(pi, r; box, nsamples=20_000, rng=Random.default_rng(),
        strict=true, closure=true, max_proposals=10*nsamples,
        return_info=false, nbatches=0)

Estimate the mean, covariance matrix, and principal directions for the region `r`
inside a *finite* `box = (ell, u)` by rejection sampling.

Return value (always):
- `mean`  : estimated mean of a uniform point in the region (within the box)
- `cov`   : estimated covariance matrix
- `evals` : eigenvalues of `cov`, sorted in descending order
- `evecs` : corresponding eigenvectors (columns), matching `evals`
- `n_accepted`, `n_proposed` : acceptance diagnostics

If `return_info=true`, additional uncertainty estimates are returned:
- `mean_stderr` : coordinatewise standard error for the mean estimate
- `evals_stderr`: standard error for the eigenvalue estimates, based on batching
- `batch_evals` : per-batch eigenvalue vectors (each sorted)
- `batch_n_accepted` : accepted samples per batch
- `nbatches`    : number of batches actually used

Batching behavior:
- If `return_info=true` and `nbatches == 0`, a default of 10 batches is used.
- Batch statistics are computed on *accepted* samples.

Speed notes:
- Uses preallocated vectors and in-place Welford updates to reduce allocations.
"""
function region_principal_directions(pi, r::Integer;
    box=nothing,
    nsamples::Int=20_000,
    rng::AbstractRNG=Random.default_rng(),
    strict::Bool=true,
    closure::Bool=true,
    max_proposals::Int=10*nsamples,
    return_info::Bool=false,
    nbatches::Int=0
)
    use_batched = _REGION_BATCHED_LOCATE[] &&
        max_proposals >= _REGION_BATCHED_LOCATE_MIN_PROPOSALS[]
    if use_batched
        return _region_principal_directions_batched(pi, r;
            box=box, nsamples=nsamples, rng=rng, strict=strict, closure=closure,
            max_proposals=max_proposals, return_info=return_info, nbatches=nbatches)
    end
    return _region_principal_directions_scalar(pi, r;
        box=box, nsamples=nsamples, rng=rng, strict=strict, closure=closure,
        max_proposals=max_proposals, return_info=return_info, nbatches=nbatches)
end

@inline function _batch_eval_from_moments(sumx::AbstractVector{Float64},
    sumxx::AbstractMatrix{Float64}, nacc::Int)
    n = length(sumx)
    if nacc <= 1
        return zeros(Float64, n)
    end
    invn = inv(float(nacc))
    cov = Matrix{Float64}(undef, n, n)
    @inbounds for j in 1:n, i in 1:n
        centered = sumxx[i, j] - (sumx[i] * sumx[j]) * invn
        cov[i, j] = centered / (nacc - 1)
    end
    E = eigen(Symmetric(cov))
    p = sortperm(E.values, rev=true)
    return E.values[p]
end

function _principal_directions_result(sumx::Vector{Float64}, sumxx::Matrix{Float64},
    batch_evals::Vector{Vector{Float64}}, batch_n::Vector{Int},
    nacc::Int, nprop::Int, return_info::Bool)
    n = length(sumx)
    if nacc <= 1
        mean = zeros(Float64, n)
        cov = zeros(Float64, n, n)
        evals = zeros(Float64, n)
        evecs = Matrix{Float64}(I, n, n)
        mean_stderr = fill(NaN, n)
        evals_stderr = fill(NaN, n)
    else
        invn = inv(float(nacc))
        mean = copy(sumx)
        rmul!(mean, invn)
        cov = Matrix{Float64}(undef, n, n)
        @inbounds for j in 1:n, i in 1:n
            centered = sumxx[i, j] - (sumx[i] * sumx[j]) * invn
            cov[i, j] = centered / (nacc - 1)
        end
        E = eigen(Symmetric(cov))
        p = sortperm(E.values, rev=true)
        evals = E.values[p]
        evecs = E.vectors[:, p]
        mean_stderr = sqrt.(diag(cov) ./ nacc)

        if length(batch_evals) >= 2
            k = length(batch_evals)
            evals_stderr = Vector{Float64}(undef, n)
            @inbounds for i in 1:n
                s = 0.0
                ss = 0.0
                for bvals in batch_evals
                    v = bvals[i]
                    s += v
                    ss += v * v
                end
                m = s / k
                var = (ss - k * m * m) / (k - 1)
                evals_stderr[i] = sqrt(var < 0.0 ? 0.0 : var) / sqrt(k)
            end
        else
            evals_stderr = fill(NaN, n)
        end
    end

    if !return_info
        return (mean=mean, cov=cov, evals=evals, evecs=evecs,
            n_accepted=nacc, n_proposed=nprop)
    end

    return (mean=mean, cov=cov, evals=evals, evecs=evecs,
        mean_stderr=mean_stderr, evals_stderr=evals_stderr,
        batch_evals=[copy(v) for v in batch_evals], batch_n_accepted=copy(batch_n),
        nbatches=length(batch_evals),
        n_accepted=nacc, n_proposed=nprop)
end

@inline _principal_cov_blocksize(nsamples::Integer) =
    _REGION_BLOCKED_COVARIANCE[] ? max(1, min(Int(nsamples), _REGION_LOCATE_BATCH_SIZE[])) : 1

@inline function _principal_flush_buffer!(sumx::AbstractVector{Float64}, sumxx::AbstractMatrix{Float64},
    sumx_b::AbstractVector{Float64}, sumxx_b::AbstractMatrix{Float64},
    accepted::AbstractMatrix{Float64}, nbuf::Int, ws::_RegionBatchWorkspace, want_batches::Int)
    nbuf <= 0 && return 0
    _compute_block_moments!(ws, accepted, nbuf)
    _accumulate_block_moments!(sumx, sumxx, ws)
    if want_batches > 0
        _accumulate_block_moments!(sumx_b, sumxx_b, ws)
    end
    return 0
end

@inline function _principal_finish_batch!(batch_evals::Vector{Vector{Float64}}, batch_n::Vector{Int},
    sumx_b::AbstractVector{Float64}, sumxx_b::AbstractMatrix{Float64}, nacc_b::Int)
    if nacc_b > 1
        push!(batch_evals, _batch_eval_from_moments(sumx_b, sumxx_b, nacc_b))
        push!(batch_n, nacc_b)
    end
    _zero_moments!(sumx_b, sumxx_b)
    return nothing
end

function _region_principal_directions_scalar(pi, r::Integer;
    box=nothing,
    nsamples::Int=20_000,
    rng::AbstractRNG=Random.default_rng(),
    strict::Bool=true,
    closure::Bool=true,
    max_proposals::Int=10 * nsamples,
    return_info::Bool=false,
    nbatches::Int=0
)
    box === nothing && error("A finite box is required for region_principal_directions.")
    a, b = box
    n = length(a)

    ws = _region_workspace(n, _principal_cov_blocksize(nsamples), 0)
    sumx = ws.mu
    sumxx = ws.C
    sumx_b = ws.mu_b
    sumxx_b = ws.C_b
    x = ws.x
    accepted = ws.accepted
    _zero_moments!(sumx, sumxx)
    _zero_moments!(sumx_b, sumxx_b)

    nacc = 0
    nprop = 0
    nbuf = 0
    nacc_b = 0

    want_batches = return_info ? (nbatches > 0 ? nbatches : 10) : 0
    batch_evals = Vector{Vector{Float64}}()
    batch_n = Int[]
    if want_batches > 0
        sizehint!(batch_evals, want_batches)
        sizehint!(batch_n, want_batches)
    end
    batch_target = want_batches > 0 ? max(2, Int(floor(nsamples / want_batches))) : 0

    x0 = Float64[(a[i] + b[i]) / 2 for i in 1:n]
    locate_style = _resolve_locate_style(pi, x0; strict=strict, closure=closure)

    while (nacc < nsamples) && (nprop < max_proposals)
        @inbounds for i in 1:n
            x[i] = rand(rng) * (b[i] - a[i]) + a[i]
        end
        nprop += 1

        if _locate_dispatch(pi, x, locate_style; strict=strict, closure=closure) == r
            nacc += 1
            nbuf += 1
            @inbounds for i in 1:n
                accepted[i, nbuf] = x[i]
            end
            if want_batches > 0
                nacc_b += 1
            end

            if nbuf >= size(accepted, 2) || (want_batches > 0 && nacc_b >= batch_target) || nacc >= nsamples
                nbuf = _principal_flush_buffer!(sumx, sumxx, sumx_b, sumxx_b, accepted, nbuf, ws, want_batches)
            end
            if want_batches > 0 && nacc_b >= batch_target
                _principal_finish_batch!(batch_evals, batch_n, sumx_b, sumxx_b, nacc_b)
                nacc_b = 0
            end
        end
    end

    nbuf = _principal_flush_buffer!(sumx, sumxx, sumx_b, sumxx_b, accepted, nbuf, ws, want_batches)
    if want_batches > 0 && nacc_b > 0
        _principal_finish_batch!(batch_evals, batch_n, sumx_b, sumxx_b, nacc_b)
    end

    return _principal_directions_result(sumx, sumxx, batch_evals, batch_n, nacc, nprop, return_info)
end

function _region_principal_directions_batched(pi, r::Integer;
    box=nothing,
    nsamples::Int=20_000,
    rng::AbstractRNG=Random.default_rng(),
    strict::Bool=true,
    closure::Bool=true,
    max_proposals::Int=10 * nsamples,
    return_info::Bool=false,
    nbatches::Int=0
)
    box === nothing && error("A finite box is required for region_principal_directions.")
    a, b = box
    n = length(a)

    ws = _region_workspace(n, max(1, _REGION_LOCATE_BATCH_SIZE[]), 0)
    sumx = ws.mu
    sumxx = ws.C
    sumx_b = ws.mu_b
    sumxx_b = ws.C_b
    accepted = ws.accepted
    _zero_moments!(sumx, sumxx)
    _zero_moments!(sumx_b, sumxx_b)

    nacc = 0
    nprop = 0
    nbuf = 0
    nacc_b = 0

    want_batches = return_info ? (nbatches > 0 ? nbatches : 10) : 0
    batch_evals = Vector{Vector{Float64}}()
    batch_n = Int[]
    if want_batches > 0
        sizehint!(batch_evals, want_batches)
        sizehint!(batch_n, want_batches)
    end
    batch_target = want_batches > 0 ? max(2, Int(floor(nsamples / want_batches))) : 0

    x0 = Float64[(a[i] + b[i]) / 2 for i in 1:n]
    locate_style = _resolve_locate_many_style(pi, x0; strict=strict, closure=closure)

    while (nacc < nsamples) && (nprop < max_proposals)
        nbatch = min(size(ws.X, 2), max_proposals - nprop)
        @inbounds for j in 1:nbatch, i in 1:n
            ws.X[i, j] = rand(rng) * (b[i] - a[i]) + a[i]
        end
        Xbatch = view(ws.X, :, 1:nbatch)
        locbatch = view(ws.locs, 1:nbatch)
        _locate_many_dispatch!(locbatch, pi, Xbatch, locate_style; strict=strict, closure=closure)
        nprop += nbatch

        @inbounds for j in 1:nbatch
            locbatch[j] == r || continue
            nacc += 1
            nbuf += 1
            for i in 1:n
                accepted[i, nbuf] = ws.X[i, j]
            end
            if want_batches > 0
                nacc_b += 1
            end

            if nbuf >= size(accepted, 2) || (want_batches > 0 && nacc_b >= batch_target) || nacc >= nsamples
                nbuf = _principal_flush_buffer!(sumx, sumxx, sumx_b, sumxx_b, accepted, nbuf, ws, want_batches)
            end
            if want_batches > 0 && nacc_b >= batch_target
                _principal_finish_batch!(batch_evals, batch_n, sumx_b, sumxx_b, nacc_b)
                nacc_b = 0
            end

            nacc >= nsamples && break
        end
    end

    nbuf = _principal_flush_buffer!(sumx, sumxx, sumx_b, sumxx_b, accepted, nbuf, ws, want_batches)
    if want_batches > 0 && nacc_b > 0
        _principal_finish_batch!(batch_evals, batch_n, sumx_b, sumxx_b, nacc_b)
    end

    return _principal_directions_result(sumx, sumxx, batch_evals, batch_n, nacc, nprop, return_info)
end



# ------------------------------------------------------------------------------
# Additional geometric descriptors for regions
#
# These are intended as "mathematician-friendly" higher-level quantities built
# on top of the primitive geometry routines (volume, boundary measure, bbox,
# principal directions).
#
# Most of these routines only make sense for bounded regions, so they generally
# require a finite `box=(a,b)` to intersect with.
# ------------------------------------------------------------------------------

# Internal helper: volume of the unit n-ball in Euclidean space (Float64).
# We avoid SpecialFunctions by using the recursion:
#   omega_0 = 1, omega_1 = 2, omega_n = (2*pi/n) * omega_{n-2}.
@inline function _unit_ball_volume(n::Integer)
    n < 0 && error("unit ball volume: dimension must be >= 0, got $n")
    n == 0 && return 1.0
    n == 1 && return 2.0

    if iseven(n)
        omega = 1.0
        k = 2
        while k <= n
            omega *= (2.0 * pi) / k
            k += 2
        end
        return omega
    else
        omega = 2.0
        k = 3
        while k <= n
            omega *= (2.0 * pi) / k
            k += 2
        end
        return omega
    end
end

@inline function _isoperimetric_constant(n::Integer)
    # Sharp Euclidean isoperimetric inequality:
    #   S >= c_n * V^((n-1)/n),  where c_n = n * omega_n^(1/n).
    n < 1 && error("isoperimetric constant: dimension must be >= 1, got $n")
    omega = _unit_ball_volume(n)
    return float(n) * omega^(1.0 / float(n))
end

# Internal helper: circumradius of an axis-aligned bounding box around a center.
@inline function _bbox_circumradius(lo::AbstractVector, hi::AbstractVector,
    c::AbstractVector, metric::Symbol)
    n = length(lo)
    length(hi) == n || error("bbox_circumradius: lo/hi length mismatch")
    length(c) == n || error("bbox_circumradius: center length mismatch")

    if metric === :L2
        s2 = 0.0
        @inbounds for i in 1:n
            di = max(abs(lo[i] - c[i]), abs(hi[i] - c[i]))
            s2 += di * di
        end
        return sqrt(s2)
    elseif metric === :Linf
        dmax = 0.0
        @inbounds for i in 1:n
            di = max(abs(lo[i] - c[i]), abs(hi[i] - c[i]))
            dmax = max(dmax, di)
        end
        return dmax
    elseif metric === :L1
        s = 0.0
        @inbounds for i in 1:n
            di = max(abs(lo[i] - c[i]), abs(hi[i] - c[i]))
            s += di
        end
        return s
    else
        error("bbox_circumradius: unknown metric=$metric (use :L2, :L1, or :Linf)")
    end
end

# A small helper to call region_volume / region_boundary_measure with optional
# closure/cache/strict keywords, while gracefully degrading when the backend
# does not accept these keywords.
function _region_volume_maybe_cached_slow(pi, r::Integer; box, closure::Bool=true, cache=nothing)
    return _region_volume_from_weights(pi, r; box=box, closure=closure, cache=cache)
end

function _region_volume_maybe_cached(pi, r::Integer; box, closure::Bool=true, cache=nothing)
    if _REGION_DIRECT_VOLUME[]
        fast = _region_volume_fast(pi, r; box=box, closure=closure, cache=cache)
        fast === nothing || return float(fast)
    end
    return _region_volume_maybe_cached_slow(pi, r; box=box, closure=closure, cache=cache)
end

function _region_weights_maybe_cached(pi; box=nothing, closure::Bool=true, cache=nothing, kwargs...)
    if cache !== nothing
        try
            return region_weights(pi; box=box, closure=closure, cache=cache, kwargs...)
        catch e
            if !(e isa MethodError)
                rethrow()
            end
        end
    end
    try
        return region_weights(pi; box=box, closure=closure, kwargs...)
    catch e
        if e isa MethodError
            return region_weights(pi; box=box, kwargs...)
        end
        rethrow()
    end
end

function _region_volume_from_weights(pi, r::Integer; box=nothing, closure::Bool=true, cache=nothing, kwargs...)
    w = _region_weights_maybe_cached(pi; box=box, closure=closure, cache=cache, kwargs...)
    (1 <= r <= length(w)) || error("region_volume: region index out of range")
    return float(w[Int(r)])
end

function _region_boundary_measure_maybe_cached_slow(pi, r::Integer; box, strict::Bool=true,
    closure::Bool=true, cache=nothing)
    if cache !== nothing
        try
            return float(region_boundary_measure(pi, r; box=box, strict=strict, closure=closure, cache=cache))
        catch e
            if !(e isa MethodError)
                rethrow()
            end
        end
    end

    try
        return float(region_boundary_measure(pi, r; box=box, strict=strict, closure=closure))
    catch e
        if e isa MethodError
            try
                return float(region_boundary_measure(pi, r; box=box, strict=strict))
            catch e2
                if e2 isa MethodError
                    return float(region_boundary_measure(pi, r; box=box))
                end
                rethrow()
            end
        end
        rethrow()
    end
end

function _region_boundary_measure_maybe_cached(pi, r::Integer; box, strict::Bool=true,
    closure::Bool=true, cache=nothing)
    if _REGION_FAST_WRAPPERS[]
        fast = _region_boundary_measure_fast(pi, r;
            box=box, strict=strict, closure=closure, cache=cache)
        fast === nothing || return float(fast)
    end
    return _region_boundary_measure_maybe_cached_slow(pi, r;
        box=box, strict=strict, closure=closure, cache=cache)
end

function _region_bbox_maybe_cached(pi, r::Integer; box, strict::Bool=true,
    closure::Bool=true, cache=nothing)
    if _REGION_FAST_WRAPPERS[]
        fast = _region_bbox_fast(pi, r; box=box, strict=strict, closure=closure, cache=cache)
        fast === nothing || return fast
    end
    if cache === nothing
        try
            return region_bbox(pi, r; box=box, strict=strict, closure=closure)
        catch e
            if e isa MethodError
                try
                    return region_bbox(pi, r; box=box, strict=strict)
                catch e2
                    if e2 isa MethodError
                        return region_bbox(pi, r; box=box)
                    end
                    rethrow()
                end
            end
            rethrow()
        end
    end
    try
        return region_bbox(pi, r; box=box, strict=strict, closure=closure, cache=cache)
    catch e
        if e isa MethodError
            try
                return region_bbox(pi, r; box=box, strict=strict, closure=closure)
            catch e2
                if e2 isa MethodError
                    try
                        return region_bbox(pi, r; box=box, strict=strict)
                    catch e3
                        if e3 isa MethodError
                            return region_bbox(pi, r; box=box)
                        end
                        rethrow()
                    end
                end
                rethrow()
            end
        end
        rethrow()
    end
end

@inline function _centroid_from_bbox(bb)
    bb === nothing && return nothing
    a, b = bb
    n = length(a)
    c = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        ai = float(a[i])
        bi = float(b[i])
        (isfinite(ai) && isfinite(bi)) || return nothing
        c[i] = (ai + bi) / 2.0
    end
    return c
end

function _region_centroid_maybe_cached(pi, r::Integer; box, method::Symbol=:bbox,
    closure::Bool=true, cache=nothing)
    if _REGION_FAST_WRAPPERS[]
        fast = _region_centroid_fast(pi, r;
            box=box, method=method, closure=closure, cache=cache)
        fast === nothing || return fast
    end
    if method === :bbox
        return _centroid_from_bbox(_region_bbox_maybe_cached(pi, r;
            box=box, strict=true, closure=closure, cache=cache))
    end

    if cache === nothing
        try
            return region_centroid(pi, r; box=box, method=method, closure=closure)
        catch e
            if e isa MethodError
                return region_centroid(pi, r; box=box, method=method)
            end
            rethrow()
        end
    end
    try
        return region_centroid(pi, r; box=box, method=method, closure=closure, cache=cache)
    catch e
        if e isa MethodError
            try
                return region_centroid(pi, r; box=box, method=method, closure=closure)
            catch e2
                if e2 isa MethodError
                    return region_centroid(pi, r; box=box, method=method)
                end
                rethrow()
            end
        end
        rethrow()
    end
end

@inline function _principal_summary_result(entry::_PrincipalSummaryEntry, return_info::Bool)
    if !return_info
        return (mean=copy(entry.mean), cov=copy(entry.cov), evals=copy(entry.evals),
            evecs=copy(entry.evecs), n_accepted=entry.n_accepted, n_proposed=entry.n_proposed)
    end
    return (mean=copy(entry.mean), cov=copy(entry.cov), evals=copy(entry.evals),
        evecs=copy(entry.evecs), mean_stderr=copy(entry.mean_stderr),
        evals_stderr=copy(entry.evals_stderr),
        batch_evals=[copy(v) for v in entry.batch_evals],
        batch_n_accepted=copy(entry.batch_n_accepted), nbatches=length(entry.batch_evals),
        n_accepted=entry.n_accepted, n_proposed=entry.n_proposed)
end

function _principal_summary_entry(mu, cov, evals, evecs, mean_stderr, evals_stderr,
    batch_evals, batch_n, nacc::Int, nprop::Int)
    return _PrincipalSummaryEntry(copy(mu), copy(cov), copy(evals), copy(evecs),
        copy(mean_stderr), copy(evals_stderr), [copy(v) for v in batch_evals], copy(batch_n),
        nacc, nprop)
end

function _principal_directions_compute(pi, r::Integer; box, nsamples::Integer,
    rng, strict::Bool=true, closure::Bool=true, max_proposals::Integer=10*nsamples,
    return_info::Bool=false, nbatches::Int=0)
    try
        return region_principal_directions(pi, r; box=box, nsamples=nsamples, rng=rng,
            strict=strict, closure=closure, max_proposals=max_proposals,
            return_info=return_info, nbatches=nbatches)
    catch e
        if e isa MethodError
            return region_principal_directions(pi, r; box=box, nsamples=nsamples, rng=rng,
                strict=strict, max_proposals=max_proposals,
                return_info=return_info, nbatches=nbatches)
        end
        rethrow()
    end
end

function _principal_summary_entry_maybe_closure(pi, r::Integer; box, nsamples::Integer,
    rng, strict::Bool=true, closure::Bool=true, max_proposals::Integer=10*nsamples,
    return_info::Bool=false, nbatches::Int=0)
    key = _REGION_SAMPLED_SUMMARY_CACHE[] ? _principal_summary_cache_key(pi, r;
        box=box, strict=strict, closure=closure, nsamples=nsamples,
        max_proposals=max_proposals, return_info=return_info, nbatches=nbatches, rng=rng) : nothing
    if key !== nothing
        cached = _sample_cache_get(_REGION_PRINCIPAL_SUMMARY_CACHE, key)
        cached === nothing || return cached
    end

    pd = _principal_directions_compute(pi, r; box=box, nsamples=nsamples, rng=rng,
        strict=strict, closure=closure, max_proposals=max_proposals,
        return_info=return_info, nbatches=nbatches)
    entry = _principal_summary_entry(pd.mean, pd.cov, pd.evals, pd.evecs,
        get(pd, :mean_stderr, fill(NaN, length(pd.mean))),
        get(pd, :evals_stderr, fill(NaN, length(pd.evals))),
        get(pd, :batch_evals, Vector{Vector{Float64}}()),
        get(pd, :batch_n_accepted, Int[]),
        pd.n_accepted, pd.n_proposed)

    if key !== nothing
        _sample_cache_insert!(_REGION_PRINCIPAL_SUMMARY_CACHE, key, entry)
    end
    return entry
end

function _principal_directions_maybe_closure(pi, r::Integer; box, nsamples::Integer,
    rng, strict::Bool=true, closure::Bool=true, max_proposals::Integer=10*nsamples,
    return_info::Bool=false, nbatches::Int=0)
    entry = _principal_summary_entry_maybe_closure(pi, r; box=box, nsamples=nsamples,
        rng=rng, strict=strict, closure=closure, max_proposals=max_proposals,
        return_info=return_info, nbatches=nbatches)
    return _principal_summary_result(entry, return_info)
end

function _mean_width_cached(pi, r::Integer; box, method::Symbol=:auto,
    ndirs::Integer=256, nsamples::Integer=4000, max_proposals::Integer=10*nsamples,
    rng=Random.default_rng(), directions=nothing,
    strict::Bool=true, closure::Bool=true, cache=nothing)
    return region_mean_width(pi, r; box=box, method=method, ndirs=ndirs,
        nsamples=nsamples, max_proposals=max_proposals, rng=rng, directions=directions,
        strict=strict, closure=closure, cache=cache)
end


"""
    region_chebyshev_ball(pi, r; box, metric=:L2, method=:auto, kwargs...) -> NamedTuple

Return a `(center, radius)` pair describing a large inscribed ball of region `r`,
intersected with `box=(a,b)`.

This is backend-dependent:
- Convex polyhedral backends can compute a true Chebyshev (largest inscribed) ball.
- Non-convex backends may return a lower bound (still a valid inscribed ball).

The ball is interpreted in the norm specified by `metric`:
- `:L2`   Euclidean ball
- `:Linf` axis-aligned cube (L_infinity ball)
- `:L1`   cross-polytope (L1 ball)

Backends that do not implement this should throw an error.
"""
function region_chebyshev_ball(pi, r::Integer; box=nothing, metric::Symbol=:L2,
    method::Symbol=:auto, kwargs...)
    error("region_chebyshev_ball not implemented for $(typeof(pi)).")
end

"""
    region_chebyshev_center(pi, r; kwargs...) -> Vector{Float64}

Return the center of `region_chebyshev_ball(pi,r; ...)`.
"""
function region_chebyshev_center(pi, r::Integer; kwargs...)
    return region_chebyshev_ball(pi, r; kwargs...).center
end

"""
    region_inradius(pi, r; kwargs...) -> Float64

Return the radius of `region_chebyshev_ball(pi,r; ...)`.
"""
function region_inradius(pi, r::Integer; kwargs...)
    return region_chebyshev_ball(pi, r; kwargs...).radius
end

"""
    region_circumradius(pi, r; box, center=:bbox, metric=:L2, method=:bbox, kwargs...) -> Float64

Return an outer radius for region `r` intersected with `box=(a,b)`.

This is the radius of a ball (in the specified norm) centered at `center` that
contains the set.

Default behavior is robust across backends: it uses only the region bounding box.

Arguments:
- `center` can be:
  - `:bbox`       (default) center of the region bounding box
  - `:centroid`   region centroid (exact for some backends, otherwise approximate)
  - `:chebyshev`  Chebyshev center (requires `region_chebyshev_ball`)
  - an explicit vector
- `method=:bbox` uses only the axis-aligned bounding box.

Backends may define more accurate methods (e.g. using vertices or cell corners).
"""
function region_circumradius(pi, r::Integer; box=nothing, center=:bbox,
    metric::Symbol=:L2, method::Symbol=:bbox, kwargs...)
    cache = _kwget(kwargs, :cache, nothing)
    closure = _kwget(kwargs, :closure, true)
    strict = _kwget(kwargs, :strict, true)
    box = _resolve_box(box, cache)
    box === nothing && error("region_circumradius: box=(a,b) is required")
    metric = Symbol(metric)
    method = Symbol(method)

    if _REGION_FAST_WRAPPERS[]
        fast = _region_circumradius_fast(pi, r;
            box=box, center=center, metric=metric, method=method,
            strict=strict, closure=closure, cache=cache)
        fast === nothing || return float(fast)
    end

    # Choose the center.
    c = nothing
    if center === :chebyshev
        c = region_chebyshev_center(pi, r; box=box, metric=metric, kwargs...)
    elseif center === :centroid
        c = _region_centroid_maybe_cached(pi, r; box=box, method=:bbox,
            closure=closure, cache=cache)
    elseif center === :bbox
        c = _centroid_from_bbox(_region_bbox_maybe_cached(pi, r; box=box,
            strict=strict, closure=closure, cache=cache))
    else
        c = center
    end

    if method === :bbox
        lo, hi = _region_bbox_maybe_cached(pi, r; box=box,
            strict=strict, closure=closure, cache=cache)
        return _bbox_circumradius(lo, hi, c, metric)
    else
        error("region_circumradius: method=$method not available for $(typeof(pi))")
    end
end

"""
    region_boundary_to_volume_ratio(pi, r; box, volume=nothing, boundary=nothing,
                                    strict=true, closure=true, cache=nothing) -> Float64

Return boundary_measure / volume for region `r` intersected with `box=(a,b)`.

This is not scale-invariant, but is often a useful "boundary-to-volume" statistic.
If volume is zero, returns `Inf` if boundary is positive, and `NaN` if both are zero.

If you already computed `volume` and/or `boundary`, pass them in to avoid recomputation.
"""
function region_boundary_to_volume_ratio(pi, r::Integer; box=nothing,
    volume=nothing, boundary=nothing,
    strict::Bool=true, closure::Bool=true, cache=nothing)
    box === nothing && error("region_boundary_to_volume_ratio: box=(a,b) is required")

    V = volume === nothing ? _region_volume_maybe_cached(pi, r; box=box, closure=closure, cache=cache) : float(volume)
    S = boundary === nothing ? _region_boundary_measure_maybe_cached(pi, r; box=box, strict=strict, closure=closure, cache=cache) : float(boundary)

    if V == 0.0
        return (S == 0.0) ? NaN : Inf
    end
    return S / V
end

"""
    region_isoperimetric_ratio(pi, r; box, volume=nothing, boundary=nothing,
                              kind=:quotient, strict=true, closure=true, cache=nothing) -> Float64

Compute an isoperimetric shape statistic for region `r` intersected with `box=(a,b)`.

Let `V` be the n-dimensional volume and `S` the (n-1)-dimensional boundary measure.
For `n >= 2`, the sharp Euclidean isoperimetric inequality states

    S >= c_n * V^((n-1)/n),   c_n = n * omega_n^(1/n),

where `omega_n` is the volume of the unit n-ball.

Supported `kind` values:
- `:quotient`           returns `c_n * V^((n-1)/n) / S` (1 for Euclidean balls; typically <= 1)
- `:ratio`              reciprocal of `:quotient`
- `:deficit`            `1 - :quotient`
- `:planar`             planar quotient `4*pi*V / S^2` (requires n==2)
- `:boundary_to_volume` returns `S / V` (not scale-invariant)

If `V <= 0` or `S <= 0`, returns `NaN`.
"""
function region_isoperimetric_ratio(pi, r::Integer; box=nothing,
    volume=nothing, boundary=nothing, kind::Symbol=:quotient,
    strict::Bool=true, closure::Bool=true, cache=nothing)
    box === nothing && error("region_isoperimetric_ratio: box=(a,b) is required")

    V = volume === nothing ? _region_volume_maybe_cached(pi, r; box=box, closure=closure, cache=cache) : float(volume)
    S = boundary === nothing ? _region_boundary_measure_maybe_cached(pi, r; box=box, strict=strict, closure=closure, cache=cache) : float(boundary)

    if kind === :boundary_to_volume
        return region_boundary_to_volume_ratio(pi, r; box=box, volume=V, boundary=S,
            strict=strict, closure=closure, cache=cache)
    end

    a, b = box
    n = length(a)
    n >= 2 || error("region_isoperimetric_ratio: requires ambient dimension n>=2, got n=$n")

    if V <= 0.0 || S <= 0.0
        return NaN
    end

    if kind === :planar
        n == 2 || error("region_isoperimetric_ratio(kind=:planar) requires n==2, got n=$n")
        return (4.0 * Base.MathConstants.pi * V) / (S * S)
    end

    cn = _isoperimetric_constant(n)
    q = cn * V^((float(n) - 1.0) / float(n)) / S

    if kind === :quotient
        return q
    elseif kind === :ratio
        return 1.0 / q
    elseif kind === :deficit
        return 1.0 - q
    else
        error("region_isoperimetric_ratio: unknown kind=$kind")
    end
end

# Random directions on S^{n-1}. Columns are unit vectors.
function _random_unit_directions(n::Integer, ndirs::Integer; rng=Random.default_rng())
    n >= 1 || error("random_unit_directions: n must be >= 1")
    ndirs >= 1 || error("random_unit_directions: ndirs must be >= 1")

    U = Matrix{Float64}(undef, n, ndirs)
    @inbounds for j in 1:ndirs
        s2 = 0.0
        for i in 1:n
            u = randn(rng)
            U[i, j] = u
            s2 += u * u
        end
        if s2 == 0.0
            fill!(view(U, :, j), 0.0)
            U[1, j] = 1.0
        else
            invs = inv(sqrt(s2))
            for i in 1:n
                U[i, j] *= invs
            end
        end
    end
    return U
end

@inline function _direction_matrix(directions, n::Integer, ndirs::Integer; rng=Random.default_rng())
    U = directions === nothing ? _random_unit_directions(n, ndirs; rng=rng) : Matrix{Float64}(directions)
    size(U, 1) == n || error("region_mean_width: directions must have size (n,ndirs)")
    return U
end

function _update_projection_extrema_scalar!(minproj::AbstractVector{Float64}, maxproj::AbstractVector{Float64},
    U::AbstractMatrix{<:Real}, X::AbstractMatrix{Float64}, col::Int)
    @inbounds for j in 1:size(U, 2)
        s = 0.0
        for i in 1:size(U, 1)
            s += U[i, j] * X[i, col]
        end
        if s < minproj[j]
            minproj[j] = s
        end
        if s > maxproj[j]
            maxproj[j] = s
        end
    end
    return nothing
end

function _update_projection_extrema_blocked!(minproj::AbstractVector{Float64}, maxproj::AbstractVector{Float64},
    U::AbstractMatrix{Float64}, accepted::AbstractMatrix{Float64}, naccepted::Int,
    ws::_RegionBatchWorkspace)
    if naccepted <= 0
        return nothing
    end
    if !_REGION_BLOCKED_PROJECTION[] || naccepted < _REGION_BLOCKED_PROJECTION_MIN_ACCEPTED[]
        @inbounds for j in 1:naccepted
            _update_projection_extrema_scalar!(minproj, maxproj, U, accepted, j)
        end
        return nothing
    end

    proj = view(ws.proj, 1:size(U, 2), 1:naccepted)
    mul!(proj, transpose(U), view(accepted, 1:size(U, 1), 1:naccepted))
    @inbounds for i in 1:size(U, 2)
        lo = minproj[i]
        hi = maxproj[i]
        for j in 1:naccepted
            s = proj[i, j]
            if s < lo
                lo = s
            end
            if s > hi
                hi = s
            end
        end
        minproj[i] = lo
        maxproj[i] = hi
    end
    return nothing
end

"""
    region_mean_width(pi, r; box, method=:auto, ndirs=256, nsamples=4000,
                      max_proposals=10*nsamples, rng=Random.default_rng(),
                      directions=nothing, strict=true, closure=true, cache=nothing) -> Float64

Estimate the mean width of region `r` intersected with `box=(a,b)`.

Mean width is the average of the width
`w(u) = sup_{x in K} <u,x> - inf_{x in K} <u,x>`
over unit directions `u` on the sphere.

Supported methods:
- `method=:cauchy` (only n==2, convex planar sets): mean_width = perimeter/pi
- `method=:mc`     Monte Carlo estimate using random directions and random points
- `method=:auto`   uses `:cauchy` when n==2, otherwise `:mc`

Backends may implement more accurate methods (e.g. using vertices or cell corners)
by defining a more specific method.
"""
function region_mean_width(pi, r::Integer; box=nothing, method::Symbol=:auto,
    ndirs::Integer=256, nsamples::Integer=4000, max_proposals::Integer=10*nsamples,
    rng=Random.default_rng(), directions=nothing,
    strict::Bool=true, closure::Bool=true, cache=nothing)
    method = Symbol(method)
    box === nothing && error("region_mean_width: box=(a,b) is required")
    a, _ = box
    n = length(a)

    if method === :auto
        method = (n == 2) ? :cauchy : :mc
    end

    if method === :cauchy
        n == 2 || error("region_mean_width(method=:cauchy) requires n==2, got n=$n")
        return region_perimeter(pi, r; box=box, strict=strict, cache=cache) / pi
    elseif method !== :mc
        error("region_mean_width: unknown method=$method (use :auto, :cauchy, :mc)")
    end

    key = _REGION_SAMPLED_SUMMARY_CACHE[] ? _mean_width_cache_key(pi, r;
        box=box, strict=strict, closure=closure, nsamples=nsamples,
        max_proposals=max_proposals, ndirs=ndirs, directions=directions, rng=rng) : nothing
    if key !== nothing
        cached = _sample_cache_get(_REGION_MEAN_WIDTH_CACHE, key)
        cached === nothing || return cached
    end

    use_batched = _REGION_BATCHED_LOCATE[] &&
        max_proposals >= _REGION_BATCHED_LOCATE_MIN_PROPOSALS[]
    mw = if use_batched
        _region_mean_width_batched(pi, r; box=box, ndirs=ndirs, nsamples=nsamples,
            max_proposals=max_proposals, rng=rng, directions=directions,
            strict=strict, closure=closure)
    else
        _region_mean_width_scalar(pi, r; box=box, ndirs=ndirs, nsamples=nsamples,
        max_proposals=max_proposals, rng=rng, directions=directions,
        strict=strict, closure=closure)
    end

    if key !== nothing
        _sample_cache_insert!(_REGION_MEAN_WIDTH_CACHE, key, float(mw))
    end
    return mw
end

function _region_mean_width_scalar(pi, r::Integer; box,
    ndirs::Integer=256, nsamples::Integer=4000, max_proposals::Integer=10*nsamples,
    rng=Random.default_rng(), directions=nothing,
    strict::Bool=true, closure::Bool=true)
    box === nothing && error("region_mean_width: box=(a,b) is required")
    a, b = box
    n = length(a)

    U = _direction_matrix(directions, n, ndirs; rng=rng)
    ws = _region_workspace(n, 1, size(U, 2))
    minproj = ws.minproj
    maxproj = ws.maxproj
    fill!(minproj, Inf)
    fill!(maxproj, -Inf)
    x = ws.x
    x0 = Float64[(a[i] + b[i]) / 2 for i in 1:n]
    locate_style = _resolve_locate_style(pi, x0; strict=strict, closure=closure)
    nacc = 0
    proposals = 0
    @inbounds while nacc < nsamples && proposals < max_proposals
        proposals += 1
        for i in 1:n
            x[i] = a[i] + rand(rng) * (b[i] - a[i])
        end
        q = _locate_dispatch(pi, x, locate_style; strict=strict, closure=closure)
        if q == r
            nacc += 1
            ws.accepted[:, 1] = x
            _update_projection_extrema_blocked!(minproj, maxproj, U, ws.accepted, 1, ws)
        elseif q == 0 && strict
            error("region_mean_width: encountered locate()==0; use strict=false or closure=true.")
        end
    end

    if nacc == 0
        return 0.0
    end

    wsum = 0.0
    @inbounds for j in 1:length(minproj)
        if isfinite(minproj[j])
            wsum += (maxproj[j] - minproj[j])
        end
    end
    return wsum / float(length(minproj))
end

function _region_mean_width_batched(pi, r::Integer; box,
    ndirs::Integer=256, nsamples::Integer=4000, max_proposals::Integer=10*nsamples,
    rng=Random.default_rng(), directions=nothing,
    strict::Bool=true, closure::Bool=true)
    box === nothing && error("region_mean_width: box=(a,b) is required")
    a, b = box
    n = length(a)

    U = _direction_matrix(directions, n, ndirs; rng=rng)
    ws = _region_workspace(n, max(1, _REGION_LOCATE_BATCH_SIZE[]), size(U, 2))
    minproj = ws.minproj
    maxproj = ws.maxproj
    fill!(minproj, Inf)
    fill!(maxproj, -Inf)
    x0 = Float64[(a[i] + b[i]) / 2 for i in 1:n]
    locate_style = _resolve_locate_many_style(pi, x0; strict=strict, closure=closure)

    nacc = 0
    proposals = 0
    @inbounds while nacc < nsamples && proposals < max_proposals
        nbatch = min(size(ws.X, 2), max_proposals - proposals)
        for j in 1:nbatch, i in 1:n
            ws.X[i, j] = a[i] + rand(rng) * (b[i] - a[i])
        end
        Xbatch = view(ws.X, :, 1:nbatch)
        locbatch = view(ws.locs, 1:nbatch)
        _locate_many_dispatch!(locbatch, pi, Xbatch, locate_style; strict=strict, closure=closure)
        proposals += nbatch

        naccepted_batch = 0
        for j in 1:nbatch
            q = locbatch[j]
            if q == r
                nacc += 1
                naccepted_batch += 1
                @inbounds for i in 1:n
                    ws.accepted[i, naccepted_batch] = ws.X[i, j]
                end
                nacc >= nsamples && break
            elseif q == 0 && strict
                error("region_mean_width: encountered locate()==0; use strict=false or closure=true.")
            end
        end
        _update_projection_extrema_blocked!(minproj, maxproj, U, ws.accepted, naccepted_batch, ws)
    end

    if nacc == 0
        return 0.0
    end

    wsum = 0.0
    @inbounds for j in eachindex(minproj)
        if isfinite(minproj[j])
            wsum += (maxproj[j] - minproj[j])
        end
    end
    return wsum / float(length(minproj))
end

"""
    region_minkowski_functionals(pi, r; box, volume=nothing, boundary=nothing,
                                 mean_width_method=:auto, mean_width_ndirs=256,
                                 mean_width_rng=Random.default_rng(), mean_width_directions=nothing,
                                 strict=true, closure=true, cache=nothing) -> NamedTuple

Compute a small bundle of Minkowski-type functionals for region `r` intersected with `box=(a,b)`.

Returned fields:
- `volume`           n-dimensional volume
- `boundary_measure` (n-1)-dimensional boundary measure
- `mean_width`       estimated mean width (see `region_mean_width`)

Pass `volume` and/or `boundary` if you already computed them.
"""
function region_minkowski_functionals(pi, r::Integer; box=nothing,
    volume=nothing, boundary=nothing,
    mean_width_method::Symbol=:auto,
    mean_width_ndirs::Integer=256,
    mean_width_rng=Random.default_rng(),
    mean_width_directions=nothing,
    strict::Bool=true, closure::Bool=true, cache=nothing)

    box === nothing && error("region_minkowski_functionals: box=(a,b) is required")

    if _REGION_FAST_WRAPPERS[]
        fast = _region_minkowski_functionals_fast(pi, r;
            box=box, volume=volume, boundary=boundary,
            mean_width_method=mean_width_method, mean_width_ndirs=mean_width_ndirs,
            mean_width_rng=mean_width_rng, mean_width_directions=mean_width_directions,
            strict=strict, closure=closure, cache=cache)
        fast === nothing || return fast
    end

    V = volume === nothing ? _region_volume_maybe_cached(pi, r; box=box, closure=closure, cache=cache) : float(volume)
    S = boundary === nothing ? _region_boundary_measure_maybe_cached(pi, r; box=box, strict=strict, closure=closure, cache=cache) : float(boundary)

    mw = float(_mean_width_cached(pi, r; box=box, method=mean_width_method,
        ndirs=mean_width_ndirs, rng=mean_width_rng,
        directions=mean_width_directions, strict=strict, closure=closure, cache=cache))

    return (volume=V, boundary_measure=S, mean_width=mw)
end

"""
    region_covariance_anisotropy(pi, r; box, kind=:ratio, epsilon=0.0,
                                 nsamples=20000, max_proposals=10*nsamples,
                                 rng=Random.default_rng(), strict=true, closure=true) -> Float64

Compute an anisotropy score from the covariance eigenvalues returned by
`region_principal_directions(pi,r; ...)`.

Let `lambda_max >= ... >= lambda_min` be covariance eigenvalues.
Supported `kind`:
- `:ratio`      lambda_max / max(lambda_min, epsilon)
- `:log_ratio`  log(lambda_max / max(lambda_min, epsilon))
- `:normalized` (lambda_max - lambda_min) / (lambda_max + lambda_min + epsilon)

If lambda_max == 0 (degenerate region), returns 1 for `:ratio`, and 0 otherwise.
"""
function region_covariance_anisotropy(pi, r::Integer; box=nothing,
    kind::Symbol=:ratio, epsilon::Real=0.0,
    nsamples::Integer=20000, max_proposals::Integer=10*nsamples,
    rng=Random.default_rng(), strict::Bool=true, closure::Bool=true)

    box === nothing && error("region_covariance_anisotropy: box=(a,b) is required")

    entry = _principal_summary_entry_maybe_closure(pi, r; box=box, nsamples=nsamples,
        rng=rng, strict=strict, closure=closure, max_proposals=max_proposals)

    evals = entry.evals
    isempty(evals) && return NaN
    lmax = float(evals[1])
    lmin = float(evals[end])
    eps = float(epsilon)

    if lmax == 0.0
        return (kind === :ratio) ? 1.0 : 0.0
    end

    if kind === :ratio
        return lmax / max(lmin, eps)
    elseif kind === :log_ratio
        return log(lmax / max(lmin, eps))
    elseif kind === :normalized
        return (lmax - lmin) / (lmax + lmin + eps)
    else
        error("region_covariance_anisotropy: unknown kind=$kind")
    end
end

"""
    region_covariance_eccentricity(pi, r; box, epsilon=0.0, kwargs...) -> Float64

An "eccentricity" score derived from covariance eigenvalues:

    ecc = sqrt(max(0, 1 - lambda_min / max(lambda_max, epsilon))).

Returns 0 for isotropic or point-like regions.
"""
function region_covariance_eccentricity(pi, r::Integer; box=nothing,
    epsilon::Real=0.0,
    nsamples::Integer=20000, max_proposals::Integer=10*nsamples,
    rng=Random.default_rng(), strict::Bool=true, closure::Bool=true)

    box === nothing && error("region_covariance_eccentricity: box=(a,b) is required")

    entry = _principal_summary_entry_maybe_closure(pi, r; box=box, nsamples=nsamples,
        rng=rng, strict=strict, closure=closure, max_proposals=max_proposals)

    evals = entry.evals
    isempty(evals) && return NaN
    lmax = float(evals[1])
    lmin = float(evals[end])
    eps = float(epsilon)

    if lmax == 0.0
        return 0.0
    end

    t = 1.0 - (lmin / max(lmax, eps))
    return sqrt(max(t, 0.0))
end

# -----------------------------------------------------------------------------
# Internal helpers for covariance-based anisotropy scores.
#
# These are intentionally small, pure numeric helpers. They are kept unexported.
# -----------------------------------------------------------------------------

@inline function _covariance_anisotropy_from_evals(evals::AbstractVector{<:Real};
    kind::Symbol = :ratio,
    epsilon::Real = 0.0
)
    isempty(evals) && return NaN
    # Ensure we use float arithmetic.
    lmax = float(evals[1])
    lmin = float(evals[end])
    eps = float(epsilon)

    if !(isfinite(lmax) && isfinite(lmin))
        return NaN
    end

    if lmax <= 0.0
        return kind == :log_ratio ? 0.0 : 1.0
    end

    denom = max(lmin, eps)
    denom > 0.0 || return kind == :log_ratio ? Inf : Inf

    if kind == :ratio
        return lmax / denom
    elseif kind == :log_ratio
        return log(lmax / denom)
    elseif kind == :normalized
        # Scale-free proxy in [0,1) when evals are nonnegative.
        return (lmax - denom) / (lmax + denom + eps)
    else
        error("_covariance_anisotropy_from_evals: unknown kind=$(kind); use :ratio, :log_ratio, or :normalized")
    end
end

@inline function _covariance_eccentricity_from_evals(evals::AbstractVector{<:Real};
    epsilon::Real = 0.0
)
    isempty(evals) && return NaN
    lmax = float(evals[1])
    lmin = float(evals[end])
    eps = float(epsilon)

    if !(isfinite(lmax) && isfinite(lmin))
        return NaN
    end

    if lmax <= 0.0
        return 0.0
    end

    denom = max(lmax, eps)
    denom > 0.0 || return 0.0
    t = 1.0 - lmin / denom
    return sqrt(max(0.0, t))
end


"""
    region_anisotropy_scores(pi, r; box, epsilon=0.0, nsamples=20_000,
        max_proposals=10*nsamples, rng=Random.default_rng(),
        strict=true, closure=true, return_info=false, nbatches=0)

Compute several scale-invariant anisotropy scores derived from the covariance
matrix of a uniform point in region `r` (restricted to the finite `box`).

Output fields:
- `ratio`       : lambda_max / lambda_min (>= 1)
- `log_ratio`   : log(ratio)
- `normalized`  : (lambda_max - lambda_min) / (lambda_max + lambda_min) in [0,1)
- `eccentricity`: sqrt(1 - lambda_min/lambda_max) in [0,1)

If `epsilon > 0`, eigenvalues smaller than `epsilon` are clamped.

If `return_info=true`, requests batched diagnostics from
`region_principal_directions` and adds:
- `ratio_stderr`, `log_ratio_stderr`, `normalized_stderr`, `eccentricity_stderr`
computed from batch-to-batch variability (standard error of the estimate),
and includes `pca=pd` with sampling diagnostics.

This answers: "how stable is this feature under sampling?"
"""
function region_anisotropy_scores(pi, r::Integer;
    box=nothing,
    epsilon::Real=0.0,
    nsamples::Int=20_000,
    max_proposals::Int=10*nsamples,
    rng::AbstractRNG=Random.default_rng(),
    strict::Bool=true,
    closure::Bool=true,
    return_info::Bool=false,
    nbatches::Int=0
)
    box === nothing && error("A finite box is required for region_anisotropy_scores.")

    if !return_info
        entry = _principal_summary_entry_maybe_closure(pi, r;
            box=box,
            nsamples=nsamples,
            max_proposals=max_proposals,
            rng=rng,
            strict=strict,
            closure=closure,
            return_info=false,
            nbatches=0)
        evals = entry.evals
        ratio = _covariance_anisotropy_from_evals(evals; kind=:ratio, epsilon=epsilon)
        log_ratio = _covariance_anisotropy_from_evals(evals; kind=:log_ratio, epsilon=epsilon)
        normalized = _covariance_anisotropy_from_evals(evals; kind=:normalized, epsilon=epsilon)
        ecc = _covariance_eccentricity_from_evals(evals; epsilon=epsilon)
        return (ratio=ratio, log_ratio=log_ratio, normalized=normalized, eccentricity=ecc)
    end

    pd = _principal_directions_maybe_closure(pi, r;
        box=box,
        nsamples=nsamples,
        max_proposals=max_proposals,
        rng=rng,
        strict=strict,
        closure=closure,
        return_info=true,
        nbatches=nbatches)

    evals = pd.evals

    ratio = _covariance_anisotropy_from_evals(evals; kind=:ratio, epsilon=epsilon)
    log_ratio = _covariance_anisotropy_from_evals(evals; kind=:log_ratio, epsilon=epsilon)
    normalized = _covariance_anisotropy_from_evals(evals; kind=:normalized, epsilon=epsilon)
    ecc = _covariance_eccentricity_from_evals(evals; epsilon=epsilon)

    ratio_se = NaN
    log_ratio_se = NaN
    normalized_se = NaN
    ecc_se = NaN

    if haskey(pd, :batch_evals) && (pd.batch_evals !== nothing) && (length(pd.batch_evals) >= 2)
        k = length(pd.batch_evals)

        sr = 0.0; ssr = 0.0
        slr = 0.0; sslr = 0.0
        sn = 0.0; ssn = 0.0
        se = 0.0; sse = 0.0

        for bevals in pd.batch_evals
            rb = _covariance_anisotropy_from_evals(bevals; kind=:ratio, epsilon=epsilon)
            lrb = _covariance_anisotropy_from_evals(bevals; kind=:log_ratio, epsilon=epsilon)
            nb = _covariance_anisotropy_from_evals(bevals; kind=:normalized, epsilon=epsilon)
            eb = _covariance_eccentricity_from_evals(bevals; epsilon=epsilon)

            sr += rb; ssr += rb*rb
            slr += lrb; sslr += lrb*lrb
            sn += nb; ssn += nb*nb
            se += eb; sse += eb*eb
        end

        mr = sr / k
        var_r = (ssr - k*mr*mr) / (k - 1)
        var_r = var_r < 0.0 ? 0.0 : var_r
        ratio_se = sqrt(var_r) / sqrt(k)

        mlr = slr / k
        var_lr = (sslr - k*mlr*mlr) / (k - 1)
        var_lr = var_lr < 0.0 ? 0.0 : var_lr
        log_ratio_se = sqrt(var_lr) / sqrt(k)

        mn = sn / k
        var_n = (ssn - k*mn*mn) / (k - 1)
        var_n = var_n < 0.0 ? 0.0 : var_n
        normalized_se = sqrt(var_n) / sqrt(k)

        me = se / k
        var_e = (sse - k*me*me) / (k - 1)
        var_e = var_e < 0.0 ? 0.0 : var_e
        ecc_se = sqrt(var_e) / sqrt(k)
    end

    return (ratio=ratio, ratio_stderr=ratio_se,
        log_ratio=log_ratio, log_ratio_stderr=log_ratio_se,
        normalized=normalized, normalized_stderr=normalized_se,
        eccentricity=ecc, eccentricity_stderr=ecc_se,
        pca=pd)
end

end # module
