module FlangeZn

using LinearAlgebra
using ..CoreModules: QQ, AbstractCoeffField, coeff_type, field_from_eltype, QQField, coerce
using ..CoreModules.CoeffFields: zeros
import ..FieldLinAlg
import ..CoreModules: change_field

# ---------------------------------------------------------------------------
# Face
# ---------------------------------------------------------------------------

"""
    Face(n, coords)
    Face(n, idxs)

A face of the coordinate orthant in Z^n.

Internally we store the face as a `BitVector` mask of length `n` in the field
`coords`, where `coords[i] == true` means coordinate `i` is part of the face.

Constructor forms:

* `Face(n, coords::BitVector)` or `Face(n, coords::Vector{Bool})`:
  treat `coords` as a boolean mask of length `n`.

* `Face(n, idxs::Vector{Int})`:
  treat `idxs` as a list of indices in `1:n` that are `true` in the mask.

For user-facing code, prefer `face(n, coords)` which is more permissive about
the input container type.
"""
struct Face
    n::Int
    coords::BitVector
end

# Convert a list of indices in 1:n to a BitVector mask.
function _indices_to_bitvector(n::Int, idxs::Vector{Int})::BitVector
    mask = falses(n)
    for i in idxs
        if i < 1 || i > n
            error("Face: index $(i) out of range 1:$(n)")
        end
        mask[i] = true
    end
    return mask
end

# idx list constructor
Face(n::Int, idxs::Vector{Int}) = Face(n, _indices_to_bitvector(n, idxs))

# boolean mask constructor (Vector{Bool})
function Face(n::Int, coords::Vector{Bool})
    if length(coords) != n
        error("Face: expected Bool mask of length $(n), got $(length(coords))")
    end
    return Face(n, BitVector(coords))
end

"""
    face(n, coords)

Convenience constructor for `Face`:

* `coords` may be a `BitVector`, a `Vector{Bool}`, a `Vector{Int}` (index list),
  or any iterable producing `Bool` or `Int`.

We also accept `coords = []` (empty) to mean the zero-dimensional face
(no fixed coordinates).
"""
function face(n::Integer, coords)
    nn = Int(n)

    if coords isa BitVector
        if length(coords) != nn
            error("Face: expected BitVector of length $(nn), got $(length(coords))")
        end
        return Face(nn, coords)
    elseif coords isa Vector{Bool}
        return Face(nn, coords)
    elseif coords isa Vector{Int}
        # Interpret as an index list in 1:nn.
        if length(coords) == 0
            return Face(nn, falses(nn))
        end
        return Face(nn, coords)
    else
        # Try to interpret as iterable of Bool.
        try
            if length(coords) == 0
                return Face(nn, falses(nn))
            end
            return Face(nn, Bool[x for x in coords])
        catch
        end
        # Try to interpret as iterable of Int (index list).
        try
            if length(coords) == 0
                return Face(nn, falses(nn))
            end
            return Face(nn, Int[x for x in coords])
        catch
        end
        error(
            "face(n, coords): coords must be BitVector, Vector{Bool}, Vector{Int}, " *
            "or an iterable producing Bool/Int.",
        )
    end
end

# ---------------------------------------------------------------------------
# IndFlat / IndInj
# ---------------------------------------------------------------------------

"""
    IndFlat(tau, b; id=:F)

An indexed flat in Z^n, specified by:

* `tau::Face`  : which coordinates are fixed (the mask in `tau.coords`)
* `b`          : translation tuple in Z^n
* `id::Symbol` : an arbitrary label (useful when multiple flats share the same
  underlying set)

Canonical constructor shape: `IndFlat(tau, b; id=...)`.
"""
struct IndFlat{N}
    b::NTuple{N,Int}
    tau::Face
    id::Symbol

    # Only allow canonical positional order (tau, b, id).
    # Defining any inner constructor suppresses Julia's default (b, tau, id) constructor.
    function IndFlat{N}(tau::Face, b::NTuple{N,Int}, id::Symbol) where {N}
        tau.n == N || error("IndFlat: face dimension $(tau.n) does not match b length $(N)")
        return new{N}(b, tau, id)
    end
end

function IndFlat(tau::Face, b::NTuple{N,<:Integer}; id::Symbol = :F) where {N}
    return IndFlat{N}(tau, ntuple(i -> Int(b[i]), N), id)
end

function IndFlat(tau::Face, b::AbstractVector{<:Integer}; id::Symbol = :F)
    return IndFlat{length(b)}(tau, ntuple(i -> Int(b[i]), length(b)), id)
end

"""
    IndInj(tau, b; id=:E)

An indexed injective in Z^n (the down-set analogue of `IndFlat`).

Canonical constructor shape: `IndInj(tau, b; id=...)`.
"""
struct IndInj{N}
    b::NTuple{N,Int}
    tau::Face
    id::Symbol

    # Only allow canonical positional order (tau, b, id).
    function IndInj{N}(tau::Face, b::NTuple{N,Int}, id::Symbol) where {N}
        tau.n == N || error("IndInj: face dimension $(tau.n) does not match b length $(N)")
        return new{N}(b, tau, id)
    end
end

function IndInj(tau::Face, b::NTuple{N,<:Integer}; id::Symbol = :E) where {N}
    return IndInj{N}(tau, ntuple(i -> Int(b[i]), N), id)
end

function IndInj(tau::Face, b::AbstractVector{<:Integer}; id::Symbol = :E)
    return IndInj{length(b)}(tau, ntuple(i -> Int(b[i]), length(b)), id)
end


# Internal predicate: do b and g agree on the fixed coordinates of tau?
@inline function _matches_on_face(b::NTuple{N,Int}, g::AbstractVector{<:Integer}, tau::Face)::Bool where {N}
    tau.n == N || error("_matches_on_face: face dimension $(tau.n) does not match b length $(N)")
    @inbounds for i in 1:N
        if tau.coords[i] && b[i] != g[i]
            return false
        end
    end
    return true
end

@inline function _matches_on_face(b::NTuple{N,Int}, g::NTuple{N,<:Integer}, tau::Face)::Bool where {N}
    tau.n == N || error("_matches_on_face: face dimension $(tau.n) does not match b length $(N)")
    @inbounds for i in 1:N
        if tau.coords[i] && b[i] != g[i]
            return false
        end
    end
    return true
end

"""
    active_flats(flats, g)

Return the indices of flats whose underlying sets contain the lattice point `g`.

This is used to form the local (fiberwise) matrix at `g` by selecting columns.
"""
@inline function _fill_active_flats!(active::Vector{Int},
                                     flats::Vector{IndFlat{N}},
                                     g::Union{Vector{Int},NTuple{N,Int}}) where {N}
    empty!(active)
    @inbounds for j in eachindex(flats)
        if in_flat(flats[j], g)
            push!(active, j)
        end
    end
    return active
end

function active_flats(flats::Vector{IndFlat{N}}, g::Vector{Int}) where {N}
    active = Int[]
    sizehint!(active, length(flats))
    _fill_active_flats!(active, flats, g)
    return active
end

function active_flats(flats::Vector{IndFlat{N}}, g::NTuple{N,Int}) where {N}
    active = Int[]
    sizehint!(active, length(flats))
    _fill_active_flats!(active, flats, g)
    return active
end

"""
    active_injectives(injectives, g)

Return the indices of injectives whose underlying sets contain the lattice point `g`.

This is used to form the local (fiberwise) matrix at `g` by selecting rows.
"""
@inline function _fill_active_injectives!(active::Vector{Int},
                                          injectives::Vector{IndInj{N}},
                                          g::Union{Vector{Int},NTuple{N,Int}}) where {N}
    empty!(active)
    @inbounds for i in eachindex(injectives)
        if in_inj(injectives[i], g)
            push!(active, i)
        end
    end
    return active
end

function active_injectives(injectives::Vector{IndInj{N}}, g::Vector{Int}) where {N}
    active = Int[]
    sizehint!(active, length(injectives))
    _fill_active_injectives!(active, injectives, g)
    return active
end

function active_injectives(injectives::Vector{IndInj{N}}, g::NTuple{N,Int}) where {N}
    active = Int[]
    sizehint!(active, length(injectives))
    _fill_active_injectives!(active, injectives, g)
    return active
end


# ---------------------------------------------------------------------------
# Membership predicates
# ---------------------------------------------------------------------------

"""
    in_flat(F, g)

Return `true` if the lattice point `g` lies in the underlying set of the flat `F`.

Convention (consistent with ZnEncoding):
* if `F.tau.coords[i] == true` then coordinate `i` is FREE (Z-direction): no constraint
* otherwise coordinate `i` is inequality-constrained: `g[i] >= F.b[i]`
"""
function in_flat(F::IndFlat, g::AbstractVector{<:Integer})::Bool
    n = F.tau.n
    if length(g) != n
        error("in_flat: point dimension does not match face dimension")
    end

    @inbounds for i in 1:n
        if !F.tau.coords[i] && g[i] < F.b[i]
            return false
        end
    end
    return true
end

"""
    in_inj(E, g)

Return `true` if the lattice point `g` lies in the underlying set of the injective `E`.

Convention (consistent with ZnEncoding):
* if `E.tau.coords[i] == true` then coordinate `i` is FREE (Z-direction): no constraint
* otherwise coordinate `i` is inequality-constrained: `g[i] <= E.b[i]`
"""
function in_inj(E::IndInj, g::AbstractVector{<:Integer})::Bool
    n = E.tau.n
    if length(g) != n
        error("in_inj: point dimension does not match face dimension")
    end

    @inbounds for i in 1:n
        if !E.tau.coords[i] && g[i] > E.b[i]
            return false
        end
    end
    return true
end

# Tuple overloads: ZnEncoding often represents lattice points as NTuples
# for performance (no allocations). Support that here.

@inline function in_flat(F::IndFlat, g::NTuple{N,T})::Bool where {N,T<:Integer}
    n = F.tau.n
    if N != n
        error("in_flat: point dimension does not match face dimension")
    end

    @inbounds for i in 1:n
        if !F.tau.coords[i] && g[i] < F.b[i]
            return false
        end
    end
    return true
end

@inline function in_inj(E::IndInj, g::NTuple{N,T})::Bool where {N,T<:Integer}
    n = E.tau.n
    if N != n
        error("in_inj: point dimension does not match face dimension")
    end

    @inbounds for i in 1:n
        if !E.tau.coords[i] && g[i] > E.b[i]
            return false
        end
    end
    return true
end

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

"""
    intersects(F, E)

Return `true` if the underlying sets of an `IndFlat` and an `IndInj` can intersect.

With the "free coordinates" convention above, the only obstruction is when a coordinate
is constrained for both generators and the bounds are incompatible:
    F.b[i] > E.b[i]  (for some i with !F.tau.coords[i] && !E.tau.coords[i])
"""
function intersects(F::IndFlat, E::IndInj)::Bool
    n = F.tau.n
    if E.tau.n != n
        error("intersects: face dimensions do not match")
    end

    @inbounds for i in 1:n
        if !F.tau.coords[i] && !E.tau.coords[i] && (F.b[i] > E.b[i])
            return false
        end
    end
    return true
end

# ---------------------------------------------------------------------------
# Flange
# ---------------------------------------------------------------------------

"""
    Flange(n, flats, injectives, phi)

A flange over Z^n.

Fields (canonical, no aliases):

* `n::Int`
* `flats::Vector{IndFlat{N}}`
* `injectives::Vector{IndInj{N}}`
* `phi::Matrix{K}`

The matrix `phi` is interpreted fiberwise: at a point `g in Z^n`, the active
injectives/active flats determine a submatrix, and the fiberwise "dimension" is
the rank of that submatrix.
"""
struct Flange{K, F<:AbstractCoeffField, N}
    field::F
    n::Int
    flats::Vector{IndFlat{N}}
    injectives::Vector{IndInj{N}}
    phi::Matrix{K}

    function Flange{K,F,N}(
        field::F,
        n::Int,
        flats::Vector{IndFlat{N}},
        injectives::Vector{IndInj{N}},
        phi::AbstractMatrix{K},
    ) where {K, F<:AbstractCoeffField, N}
        coeff_type(field) == K || error("Flange: coeff_type(field) != K")
        n == N || error("Flange: n=$n does not match flat/injective dimension $N")
        Phi = Matrix{K}(phi)
        if size(Phi, 1) != length(injectives) || size(Phi, 2) != length(flats)
            error(
                "Flange: phi must have size (#injectives, #flats) = " *
                "($(length(injectives)), $(length(flats))), got $(size(Phi))",
            )
        end

        # Enforce the monomial condition with a compiled SoA geometry view.
        kernel = _compile_flange_kernel(n, flats, injectives)
        _enforce_monomial_condition!(Phi, kernel)

        return new{K,F,N}(field, n, flats, injectives, Phi)
    end
end

# Outer constructors for convenience.
function Flange{K}(n::Int, flats::Vector{IndFlat{N}}, injectives::Vector{IndInj{N}}, phi::AbstractMatrix;
                   field::AbstractCoeffField=field_from_eltype(K)) where {K,N}
    return Flange{K, typeof(field), N}(field, n, flats, injectives, Matrix{K}(phi))
end

function Flange{K}(n::Int, flats::Vector{IndFlat{N}}, injectives::Vector{IndInj{N}}, phi::AbstractVector;
                   field::AbstractCoeffField=field_from_eltype(K)) where {K,N}
    Phi = reshape(collect(phi), length(injectives), length(flats))
    return Flange{K, typeof(field), N}(field, n, flats, injectives, Phi)
end

function Flange(n::Int, flats::Vector{IndFlat{N}}, injectives::Vector{IndInj{N}}, phi::AbstractMatrix{K};
                field::AbstractCoeffField=field_from_eltype(K)) where {K,N}
    return Flange{K, typeof(field), N}(field, n, flats, injectives, phi)
end


"""
    change_field(FG, field)

Return a flange obtained by coercing `FG.phi` into the target coefficient field.
"""
function change_field(FG::Flange{K,F,N}, field::AbstractCoeffField) where {K,F,N}
    K2 = coeff_type(field)
    Phi = Matrix{K2}(undef, size(FG.phi, 1), size(FG.phi, 2))
    @inbounds for j in 1:size(Phi, 2), i in 1:size(Phi, 1)
        Phi[i, j] = coerce(field, FG.phi[i, j])
    end
    return Flange{K2, typeof(field), N}(field, FG.n, FG.flats, FG.injectives, Phi)
end


# ---------------------------------------------------------------------------
# Compiled SoA kernel + dim cache
# ---------------------------------------------------------------------------

struct FlangeCompiledKernel{N}
    n::Int
    ncoord_words::Int
    nflat::Int
    ninj::Int
    flat_b::Matrix{Int}
    inj_b::Matrix{Int}
    flat_free_words::Matrix{UInt64}
    inj_free_words::Matrix{UInt64}
    coord_word::Vector{Int}
    coord_mask::Vector{UInt64}
end

struct _FlangeDimCacheEntry
    row_words::Vector{UInt64}
    col_words::Vector{UInt64}
    value::Int
end

mutable struct FlangeDimCache{K,F<:AbstractCoeffField,N}
    field::F
    flats_ref::Vector{IndFlat{N}}
    injectives_ref::Vector{IndInj{N}}
    phi_ref::Matrix{K}
    kernel::FlangeCompiledKernel{N}
    rows::Vector{Int}
    cols::Vector{Int}
    row_words::Vector{UInt64}
    col_words::Vector{UInt64}
    table::Dict{UInt64, Vector{_FlangeDimCacheEntry}}
    max_entries::Int
    nentries::Int
    hits::Int
    misses::Int
end

@inline _bit_word(i::Int) = ((i - 1) >>> 6) + 1
@inline _bit_mask(i::Int) = UInt64(1) << ((i - 1) & 63)

function _compile_flange_kernel(n::Int,
                                flats::Vector{IndFlat{N}},
                                injectives::Vector{IndInj{N}}) where {N}
    n == N || error("_compile_flange_kernel: n=$n does not match generator dimension $N")
    nflat = length(flats)
    ninj = length(injectives)
    ncoord_words = cld(N, 64)

    flat_b = Matrix{Int}(undef, N, nflat)
    inj_b = Matrix{Int}(undef, N, ninj)
    flat_free_words = Matrix{UInt64}(undef, ncoord_words, nflat)
    inj_free_words = Matrix{UInt64}(undef, ncoord_words, ninj)
    fill!(flat_free_words, zero(UInt64))
    fill!(inj_free_words, zero(UInt64))

    coord_word = Vector{Int}(undef, N)
    coord_mask = Vector{UInt64}(undef, N)
    @inbounds for d in 1:N
        coord_word[d] = _bit_word(d)
        coord_mask[d] = _bit_mask(d)
    end

    @inbounds for j in 1:nflat
        Fj = flats[j]
        for d in 1:N
            flat_b[d, j] = Fj.b[d]
            if Fj.tau.coords[d]
                flat_free_words[coord_word[d], j] |= coord_mask[d]
            end
        end
    end

    @inbounds for i in 1:ninj
        Ei = injectives[i]
        for d in 1:N
            inj_b[d, i] = Ei.b[d]
            if Ei.tau.coords[d]
                inj_free_words[coord_word[d], i] |= coord_mask[d]
            end
        end
    end

    return FlangeCompiledKernel{N}(n, ncoord_words, nflat, ninj,
                                   flat_b, inj_b, flat_free_words, inj_free_words,
                                   coord_word, coord_mask)
end

_compile_flange_kernel(FG::Flange{K,F,N}) where {K,F,N} =
    _compile_flange_kernel(FG.n, FG.flats, FG.injectives)

@inline function _kernel_generators_intersect(k::FlangeCompiledKernel{N},
                                              flat_idx::Int,
                                              inj_idx::Int)::Bool where {N}
    @inbounds for d in 1:N
        wd = k.coord_word[d]
        bm = k.coord_mask[d]
        flat_free = (k.flat_free_words[wd, flat_idx] & bm) != 0
        inj_free = (k.inj_free_words[wd, inj_idx] & bm) != 0
        if !flat_free && !inj_free && (k.flat_b[d, flat_idx] > k.inj_b[d, inj_idx])
            return false
        end
    end
    return true
end

function _enforce_monomial_condition!(Phi::Matrix{K},
                                      k::FlangeCompiledKernel) where {K}
    @inbounds for i in 1:k.ninj
        for j in 1:k.nflat
            _kernel_generators_intersect(k, j, i) || (Phi[i, j] = zero(K))
        end
    end
    return Phi
end

function FlangeDimCache(FG::Flange{K,F,N}; max_entries::Int=0) where {K,F<:AbstractCoeffField,N}
    max_entries >= 0 || error("FlangeDimCache: max_entries must be >= 0")
    kernel = _compile_flange_kernel(FG)
    rows = Int[]
    cols = Int[]
    sizehint!(rows, kernel.ninj)
    sizehint!(cols, kernel.nflat)
    row_words = Vector{UInt64}(undef, cld(kernel.ninj, 64))
    col_words = Vector{UInt64}(undef, cld(kernel.nflat, 64))
    fill!(row_words, zero(UInt64))
    fill!(col_words, zero(UInt64))
    return FlangeDimCache{K,F,N}(FG.field, FG.flats, FG.injectives, FG.phi, kernel,
                                 rows, cols, row_words, col_words,
                                 Dict{UInt64, Vector{_FlangeDimCacheEntry}}(),
                                 max_entries, 0, 0, 0)
end

function Base.empty!(cache::FlangeDimCache)
    empty!(cache.table)
    cache.nentries = 0
    cache.hits = 0
    cache.misses = 0
    return cache
end

@inline function _check_cache_compat(cache::FlangeDimCache{K,F,N},
                                     FG::Flange{K,F,N}) where {K,F,N}
    (cache.flats_ref === FG.flats &&
     cache.injectives_ref === FG.injectives &&
     cache.phi_ref === FG.phi) || error("FlangeDimCache: cache is not compatible with this Flange")
    return nothing
end

@inline function _contains_flat_kernel(k::FlangeCompiledKernel{N},
                                       j::Int,
                                       g::NTuple{N,<:Integer})::Bool where {N}
    @inbounds for d in 1:N
        wd = k.coord_word[d]
        bm = k.coord_mask[d]
        ((k.flat_free_words[wd, j] & bm) != 0) && continue
        g[d] >= k.flat_b[d, j] || return false
    end
    return true
end

@inline function _contains_inj_kernel(k::FlangeCompiledKernel{N},
                                      i::Int,
                                      g::NTuple{N,<:Integer})::Bool where {N}
    @inbounds for d in 1:N
        wd = k.coord_word[d]
        bm = k.coord_mask[d]
        ((k.inj_free_words[wd, i] & bm) != 0) && continue
        g[d] <= k.inj_b[d, i] || return false
    end
    return true
end

@inline function _contains_flat_kernel(k::FlangeCompiledKernel{N},
                                       j::Int,
                                       g::AbstractVector{<:Integer})::Bool where {N}
    length(g) == N || error("_contains_flat_kernel: point dimension does not match face dimension")
    @inbounds for d in 1:N
        wd = k.coord_word[d]
        bm = k.coord_mask[d]
        ((k.flat_free_words[wd, j] & bm) != 0) && continue
        g[d] >= k.flat_b[d, j] || return false
    end
    return true
end

@inline function _contains_inj_kernel(k::FlangeCompiledKernel{N},
                                      i::Int,
                                      g::AbstractVector{<:Integer})::Bool where {N}
    length(g) == N || error("_contains_inj_kernel: point dimension does not match face dimension")
    @inbounds for d in 1:N
        wd = k.coord_word[d]
        bm = k.coord_mask[d]
        ((k.inj_free_words[wd, i] & bm) != 0) && continue
        g[d] <= k.inj_b[d, i] || return false
    end
    return true
end

@inline function _mark_active!(words::Vector{UInt64}, idx::Int)
    wd = _bit_word(idx)
    bm = _bit_mask(idx)
    @inbounds words[wd] |= bm
    return nothing
end

@inline function _fill_active_injectives_kernel!(active::Vector{Int},
                                                 words::Vector{UInt64},
                                                 k::FlangeCompiledKernel{N},
                                                 g::NTuple{N,<:Integer}) where {N}
    empty!(active)
    fill!(words, zero(UInt64))
    @inbounds for i in 1:k.ninj
        if _contains_inj_kernel(k, i, g)
            push!(active, i)
            _mark_active!(words, i)
        end
    end
    return active
end

@inline function _fill_active_flats_kernel!(active::Vector{Int},
                                            words::Vector{UInt64},
                                            k::FlangeCompiledKernel{N},
                                            g::NTuple{N,<:Integer}) where {N}
    empty!(active)
    fill!(words, zero(UInt64))
    @inbounds for j in 1:k.nflat
        if _contains_flat_kernel(k, j, g)
            push!(active, j)
            _mark_active!(words, j)
        end
    end
    return active
end

@inline function _fill_active_injectives_kernel!(active::Vector{Int},
                                                 words::Vector{UInt64},
                                                 k::FlangeCompiledKernel{N},
                                                 g::AbstractVector{<:Integer}) where {N}
    empty!(active)
    fill!(words, zero(UInt64))
    @inbounds for i in 1:k.ninj
        if _contains_inj_kernel(k, i, g)
            push!(active, i)
            _mark_active!(words, i)
        end
    end
    return active
end

@inline function _fill_active_flats_kernel!(active::Vector{Int},
                                            words::Vector{UInt64},
                                            k::FlangeCompiledKernel{N},
                                            g::AbstractVector{<:Integer}) where {N}
    empty!(active)
    fill!(words, zero(UInt64))
    @inbounds for j in 1:k.nflat
        if _contains_flat_kernel(k, j, g)
            push!(active, j)
            _mark_active!(words, j)
        end
    end
    return active
end

@inline function _same_words(a::Vector{UInt64}, b::Vector{UInt64})::Bool
    length(a) == length(b) || return false
    @inbounds for i in eachindex(a, b)
        a[i] == b[i] || return false
    end
    return true
end

@inline function _active_words_hash(row_words::Vector{UInt64},
                                    col_words::Vector{UInt64})::UInt64
    h = UInt64(0x243f6a8885a308d3)
    @inbounds for w in row_words
        h = hash(w, h)
    end
    h = hash(UInt64(0x9e3779b97f4a7c15), h)
    @inbounds for w in col_words
        h = hash(w, h)
    end
    return h
end

@inline function _lookup_rank(cache::FlangeDimCache,
                              key::UInt64,
                              row_words::Vector{UInt64},
                              col_words::Vector{UInt64})::Int
    bucket = get(cache.table, key, nothing)
    bucket === nothing && return -1
    @inbounds for ent in bucket
        if _same_words(ent.row_words, row_words) && _same_words(ent.col_words, col_words)
            cache.hits += 1
            return ent.value
        end
    end
    return -1
end

@inline function _store_rank!(cache::FlangeDimCache,
                              key::UInt64,
                              row_words::Vector{UInt64},
                              col_words::Vector{UInt64},
                              value::Int)::Int
    cache.max_entries == 0 || cache.nentries < cache.max_entries || return value
    bucket = get!(cache.table, key, _FlangeDimCacheEntry[])
    push!(bucket, _FlangeDimCacheEntry(copy(row_words), copy(col_words), value))
    cache.nentries += 1
    return value
end

@inline function _dim_at_cached!(cache::FlangeDimCache{K,F,N},
                                 FG::Flange{K,F,N},
                                 g::Union{AbstractVector{<:Integer},NTuple{N,<:Integer}})::Int where {K,F,N}
    _check_cache_compat(cache, FG)
    k = cache.kernel
    _fill_active_injectives_kernel!(cache.rows, cache.row_words, k, g)
    _fill_active_flats_kernel!(cache.cols, cache.col_words, k, g)

    nr = length(cache.rows)
    nc = length(cache.cols)
    if nr == 0 || nc == 0
        return 0
    end

    key = _active_words_hash(cache.row_words, cache.col_words)
    cached = _lookup_rank(cache, key, cache.row_words, cache.col_words)
    cached >= 0 && return cached

    cache.misses += 1
    rk = FieldLinAlg.rank_restricted(FG.field, FG.phi, cache.rows, cache.cols)
    return _store_rank!(cache, key, cache.row_words, cache.col_words, rk)
end

@inline function _set_active_bit!(words::Vector{UInt64}, idx::Int)::Int
    wd = _bit_word(idx)
    bm = _bit_mask(idx)
    @inbounds old = words[wd]
    @inbounds words[wd] = old | bm
    return (old & bm) == 0 ? 1 : 0
end

@inline function _clear_active_bit!(words::Vector{UInt64}, idx::Int)::Int
    wd = _bit_word(idx)
    bm = _bit_mask(idx)
    @inbounds old = words[wd]
    @inbounds words[wd] = old & ~bm
    return (old & bm) == 0 ? 0 : 1
end

@inline function _any_active(words::Vector{UInt64})::Bool
    @inbounds for w in words
        w == 0 || return true
    end
    return false
end

@inline function _decode_active_words!(out::Vector{Int},
                                       words::Vector{UInt64},
                                       nmax::Int)
    empty!(out)
    @inbounds for wd in eachindex(words)
        w = words[wd]
        while w != 0
            tz = trailing_zeros(w)
            idx = ((wd - 1) << 6) + tz + 1
            idx <= nmax && push!(out, idx)
            w &= w - 1
        end
    end
    return out
end

@inline function _dim_from_active_words!(cache::FlangeDimCache{K,F,N},
                                         FG::Flange{K,F,N},
                                         row_words::Vector{UInt64},
                                         col_words::Vector{UInt64},
                                         row_count::Int,
                                         col_count::Int)::Int where {K,F,N}
    if row_count == 0 || col_count == 0 || !_any_active(row_words) || !_any_active(col_words)
        return 0
    end
    key = _active_words_hash(row_words, col_words)
    cached = _lookup_rank(cache, key, row_words, col_words)
    cached >= 0 && return cached

    cache.misses += 1
    _decode_active_words!(cache.rows, row_words, cache.kernel.ninj)
    _decode_active_words!(cache.cols, col_words, cache.kernel.nflat)
    rk = FieldLinAlg.rank_restricted(FG.field, FG.phi, cache.rows, cache.cols)
    return _store_rank!(cache, key, row_words, col_words, rk)
end

@inline function _line_pass_flat_other(k::FlangeCompiledKernel{N},
                                       j::Int,
                                       rest::NTuple{M,Int})::Bool where {N,M}
    @inbounds for d in 2:N
        wd = k.coord_word[d]
        bm = k.coord_mask[d]
        if ((k.flat_free_words[wd, j] & bm) == 0) && (rest[d - 1] < k.flat_b[d, j])
            return false
        end
    end
    return true
end

@inline function _line_pass_inj_other(k::FlangeCompiledKernel{N},
                                      i::Int,
                                      rest::NTuple{M,Int})::Bool where {N,M}
    @inbounds for d in 2:N
        wd = k.coord_word[d]
        bm = k.coord_mask[d]
        if ((k.inj_free_words[wd, i] & bm) == 0) && (rest[d - 1] > k.inj_b[d, i])
            return false
        end
    end
    return true
end

struct _BoxSweepScratch
    row_words::Vector{UInt64}
    col_words::Vector{UInt64}
    col_add_pos::Vector{Int}
    col_add_idx::Vector{Int}
    row_rem_pos::Vector{Int}
    row_rem_idx::Vector{Int}
    col_perm::Vector{Int}
    row_perm::Vector{Int}
    line_uids::Vector{Int}
end

function _box_sweep_scratch(k::FlangeCompiledKernel, xlen::Int)
    row_words = Vector{UInt64}(undef, cld(k.ninj, 64))
    col_words = Vector{UInt64}(undef, cld(k.nflat, 64))
    col_add_pos = Int[]; col_add_idx = Int[]
    row_rem_pos = Int[]; row_rem_idx = Int[]
    col_perm = Int[]; row_perm = Int[]
    line_uids = Vector{Int}(undef, xlen)
    sizehint!(col_add_pos, k.nflat); sizehint!(col_add_idx, k.nflat)
    sizehint!(row_rem_pos, k.ninj); sizehint!(row_rem_idx, k.ninj)
    sizehint!(col_perm, k.nflat); sizehint!(row_perm, k.ninj)
    return _BoxSweepScratch(row_words, col_words,
                            col_add_pos, col_add_idx,
                            row_rem_pos, row_rem_idx,
                            col_perm, row_perm,
                            line_uids)
end

@inline function _box_strides(axes::NTuple{N,UnitRange{Int}}) where {N}
    s = Vector{Int}(undef, N)
    s[1] = 1
    @inbounds for d in 2:N
        s[d] = s[d - 1] * length(axes[d - 1])
    end
    return s
end

@inline function _box_linear_index(p::NTuple{N,<:Integer},
                                   axes::NTuple{N,UnitRange{Int}},
                                   strides::Vector{Int}) where {N}
    idx = 1
    @inbounds for d in 1:N
        idx += (Int(p[d]) - first(axes[d])) * strides[d]
    end
    return idx
end

function _build_uid_lookup(unique_index::Dict{T,Int},
                           axes::NTuple{N,UnitRange{Int}},
                           strides::Vector{Int}) where {N,T<:NTuple{N,<:Integer}}
    total = prod(length(r) for r in axes)
    uid_lookup = Base.zeros(Int, total)
    for (p, uid) in unique_index
        li = _box_linear_index(p, axes, strides)
        @inbounds uid_lookup[li] = uid
    end
    return uid_lookup
end

@inline function _box_base_index(rest::NTuple{M,Int},
                                 axes::NTuple{N,UnitRange{Int}},
                                 strides::Vector{Int}) where {M,N}
    base = 1
    @inbounds for d in 2:N
        base += (rest[d - 1] - first(axes[d])) * strides[d]
    end
    return base
end

# Convenience wrappers accepting a flange directly.
active_flats(FG::Flange, g::Vector{Int}) = active_flats(FG.flats, g)
active_flats(FG::Flange, g::NTuple{N,Int}) where {N} = active_flats(FG.flats, g)
active_injectives(FG::Flange, g::Vector{Int}) = active_injectives(FG.injectives, g)
active_injectives(FG::Flange, g::NTuple{N,Int}) where {N} = active_injectives(FG.injectives, g)

const _DEGREE_SCRATCH_LOCK = ReentrantLock()
const _DEGREE_ROWS_SCRATCH = Vector{Vector{Int}}()
const _DEGREE_COLS_SCRATCH = Vector{Vector{Int}}()

@inline function _thread_degree_scratch(nrows_hint::Int, ncols_hint::Int)
    tid = Base.Threads.threadid()
    if tid > length(_DEGREE_ROWS_SCRATCH)
        lock(_DEGREE_SCRATCH_LOCK) do
            while length(_DEGREE_ROWS_SCRATCH) < tid
                push!(_DEGREE_ROWS_SCRATCH, Int[])
                push!(_DEGREE_COLS_SCRATCH, Int[])
            end
        end
    end
    rows = @inbounds _DEGREE_ROWS_SCRATCH[tid]
    cols = @inbounds _DEGREE_COLS_SCRATCH[tid]
    sizehint!(rows, nrows_hint)
    sizehint!(cols, ncols_hint)
    return rows, cols
end

"""
    degree_matrix(FG, g)

Return `(Phi_g, rows, cols)` where:

* `rows` are the active injective indices at `g`
* `cols` are the active flat indices at `g`
* `Phi_g = FG.phi[rows, cols]`
"""
function degree_matrix!(rows::Vector{Int}, cols::Vector{Int},
                        FG::Flange{K}, g::Vector{Int}) where {K}
    _fill_active_injectives!(rows, FG.injectives, g)
    _fill_active_flats!(cols, FG.flats, g)
    return (view(FG.phi, rows, cols), rows, cols)
end

function degree_matrix!(rows::Vector{Int}, cols::Vector{Int},
                        FG::Flange{K}, g::NTuple{N,Int}) where {K,N}
    _fill_active_injectives!(rows, FG.injectives, g)
    _fill_active_flats!(cols, FG.flats, g)
    return (view(FG.phi, rows, cols), rows, cols)
end

"""
    degree_matrix!(rows, cols, FG, g)

Fill caller-owned `rows` and `cols` buffers with active injective/flat indices and
return `(view(FG.phi, rows, cols), rows, cols)`.

This is the non-allocating variant for repeated-query workloads.
"""
degree_matrix!(rows::Vector{Int}, cols::Vector{Int},
               FG::Flange, g) = throw(MethodError(degree_matrix!, (rows, cols, FG, g)))

"""
    degree_matrix!(FG, g)

Thread-local scratch variant of `degree_matrix!` (non-allocating, ephemeral
`rows`/`cols` buffers). Use this in tight loops when results are consumed
immediately.
"""
function degree_matrix!(FG::Flange{K}, g::Vector{Int}) where {K}
    rows, cols = _thread_degree_scratch(length(FG.injectives), length(FG.flats))
    return degree_matrix!(rows, cols, FG, g)
end

function degree_matrix!(FG::Flange{K}, g::NTuple{N,Int}) where {K,N}
    rows, cols = _thread_degree_scratch(length(FG.injectives), length(FG.flats))
    return degree_matrix!(rows, cols, FG, g)
end

function degree_matrix(FG::Flange{K}, g::Vector{Int}) where {K}
    rows = Vector{Int}()
    cols = Vector{Int}()
    sizehint!(rows, length(FG.injectives))
    sizehint!(cols, length(FG.flats))
    return degree_matrix!(rows, cols, FG, g)
end

function degree_matrix(FG::Flange{K}, g::NTuple{N,Int}) where {K,N}
    rows = Vector{Int}()
    cols = Vector{Int}()
    sizehint!(rows, length(FG.injectives))
    sizehint!(cols, length(FG.flats))
    return degree_matrix!(rows, cols, FG, g)
end

@inline function _dim_at_auto(FG::Flange, g::Union{Vector{Int},NTuple})
    _, rows, cols = degree_matrix!(FG, g)
    nr = length(rows)
    nc = length(cols)
    if nr == 0 || nc == 0
        return 0
    end

    # QQ has a small-workload crossover where explicit tiny submatrices are faster.
    if FG.field isa QQField &&
       nr * nc <= FieldLinAlg.zn_qq_dimat_submatrix_work_threshold()
        return FieldLinAlg.rank(FG.field, FG.phi[rows, cols])
    end

    return FieldLinAlg.rank_restricted(FG.field, FG.phi, rows, cols)
end

@inline function _dim_at_rankfun!(FG::Flange,
                                  g::Union{Vector{Int},NTuple},
                                  rankfun::Function,
                                  rows::Vector{Int},
                                  cols::Vector{Int})
    A, rows_now, cols_now = degree_matrix!(rows, cols, FG, g)
    if isempty(rows_now) || isempty(cols_now)
        return 0
    end
    return rankfun(A)
end

"""
    dim_at(FG, g; rankfun=nothing, cache=nothing)

Fiberwise dimension of a flange at `g`.

The default path computes rank via a restricted linear algebra kernel to avoid
materializing the local submatrix `FG.phi[rows, cols]`.

If `cache::FlangeDimCache` is supplied (and `rankfun === nothing`), we use a
packed active-set cache keyed by row/column activity masks.
"""
function dim_at(FG::Flange{K}, g::Vector{Int}; rankfun=nothing, cache::Union{Nothing,FlangeDimCache}=nothing) where {K}
    if rankfun === nothing
        return cache === nothing ? _dim_at_auto(FG, g) : _dim_at_cached!(cache, FG, g)
    end
    if cache !== nothing
        rows = cache.rows
        cols = cache.cols
        return _dim_at_rankfun!(FG, g, rankfun, rows, cols)
    end
    Phi_g, _, _ = degree_matrix(FG, g)
    isempty(Phi_g) && return 0
    return rankfun(Phi_g)
end

function dim_at(FG::Flange{K}, g::NTuple{N,Int}; rankfun=nothing, cache::Union{Nothing,FlangeDimCache}=nothing) where {K,N}
    if rankfun === nothing
        return cache === nothing ? _dim_at_auto(FG, g) : _dim_at_cached!(cache, FG, g)
    end
    if cache !== nothing
        rows = cache.rows
        cols = cache.cols
        return _dim_at_rankfun!(FG, g, rankfun, rows, cols)
    end
    Phi_g, _, _ = degree_matrix(FG, g)
    isempty(Phi_g) && return 0
    return rankfun(Phi_g)
end

@inline _lex_sortable_points(points::AbstractVector) = eltype(points) <: NTuple
@inline _tuple_int_points(::AbstractVector{T}) where {N,T<:NTuple{N,<:Integer}} = true
@inline _tuple_int_points(::AbstractVector) = false

@inline _point_key(p::NTuple) = p
@inline _point_key(p::AbstractVector{<:Integer}) = Tuple(p)
@inline _point_key(p) = p

function _dedup_points(points::AbstractVector{T}) where {N,T<:NTuple{N,<:Integer}}
    uniq = Vector{T}()
    sizehint!(uniq, length(points))
    inv = Vector{Int}(undef, length(points))
    idx = Dict{T,Int}()
    @inbounds for i in eachindex(points)
        p = points[i]
        j = get(idx, p, 0)
        if j == 0
            j = length(uniq) + 1
            idx[p] = j
            push!(uniq, p)
        end
        inv[i] = j
    end
    return uniq, inv, idx
end

function _dedup_points(points::AbstractVector)
    uniq = Any[]
    sizehint!(uniq, length(points))
    inv = Vector{Int}(undef, length(points))
    idx = Dict{Any,Int}()
    @inbounds for i in eachindex(points)
        p = points[i]
        k = _point_key(p)
        j = get(idx, k, 0)
        if j == 0
            j = length(uniq) + 1
            idx[k] = j
            push!(uniq, p)
        end
        inv[i] = j
    end
    return uniq, inv, idx
end

@inline function _full_box_axes(points::AbstractVector{T},
                                unique_index::Dict{T,Int}) where {N,T<:NTuple{N,<:Integer}}
    isempty(points) && return nothing
    p0 = points[1]
    mins = collect(p0)
    maxs = collect(p0)
    @inbounds for i in 2:length(points)
        p = points[i]
        for d in 1:N
            pd = Int(p[d])
            pd < mins[d] && (mins[d] = pd)
            pd > maxs[d] && (maxs[d] = pd)
        end
    end
    expected = 1
    @inbounds for d in 1:N
        len = maxs[d] - mins[d] + 1
        len > 0 || return nothing
        expected = Base.checked_mul(expected, len)
        expected <= length(points) || return nothing
    end
    expected == length(points) || return nothing

    axes = ntuple(d -> mins[d]:maxs[d], N)
    for tup in Iterators.product(axes...)
        p = ntuple(d -> tup[d], N)
        haskey(unique_index, p) || return nothing
    end
    return axes
end

@inline function _evaluate_unique_serial!(vals::AbstractVector{Int},
                                          FG::Flange,
                                          unique_points::AbstractVector,
                                          order,
                                          cache::Union{Nothing,FlangeDimCache})
    cache0 = cache === nothing ? FlangeDimCache(FG) : cache
    @inbounds for oi in order
        vals[oi] = _dim_at_cached!(cache0, FG, unique_points[oi])
    end
    return vals
end

function _evaluate_unique_threaded!(vals::AbstractVector{Int},
                                    FG::Flange,
                                    unique_points::AbstractVector,
                                    order)
    nt = Threads.maxthreadid()
    caches = Vector{Any}(undef, nt)
    fill!(caches, nothing)
    Threads.@threads for k in eachindex(order)
        oi = order[k]
        tid = Threads.threadid()
        c = caches[tid]
        if c === nothing
            c = FlangeDimCache(FG)
            caches[tid] = c
        end
        @inbounds vals[oi] = _dim_at_cached!(c, FG, unique_points[oi])
    end
    return vals
end

function _evaluate_unique_points!(vals::AbstractVector{Int},
                                  FG::Flange,
                                  unique_points::AbstractVector,
                                  order;
                                  cache::Union{Nothing,FlangeDimCache}=nothing,
                                  threaded::Bool=false)
    if threaded && Threads.nthreads() > 1 && length(order) >= 1024
        return _evaluate_unique_threaded!(vals, FG, unique_points, order)
    end
    return _evaluate_unique_serial!(vals, FG, unique_points, order, cache)
end

@inline function _line_build_point(::Val{N}, x::Int, rest::NTuple{M,Int}) where {N,M}
    return ntuple(d -> d == 1 ? x : rest[d - 1], N)
end

function _eval_unique_box_line_sweep!(vals::Vector{Int},
                                      FG::Flange{K,F,N},
                                      cache::FlangeDimCache{K,F,N},
                                      scratch::_BoxSweepScratch,
                                      uid_lookup::Vector{Int},
                                      line_base::Int,
                                      axes::NTuple{N,UnitRange{Int}},
                                      rest::NTuple{M,Int}) where {K,F,N,M}
    k = cache.kernel
    xaxis = axes[1]
    xmin = first(xaxis)
    xmax = last(xaxis)
    xlen = length(xaxis)

    row_words = scratch.row_words
    col_words = scratch.col_words
    fill!(row_words, zero(UInt64))
    fill!(col_words, zero(UInt64))
    row_count = 0
    col_count = 0

    col_add_pos = scratch.col_add_pos
    col_add_idx = scratch.col_add_idx
    row_rem_pos = scratch.row_rem_pos
    row_rem_idx = scratch.row_rem_idx
    empty!(col_add_pos); empty!(col_add_idx)
    empty!(row_rem_pos); empty!(row_rem_idx)

    @inbounds for j in 1:k.nflat
        _line_pass_flat_other(k, j, rest) || continue
        free1 = (k.flat_free_words[k.coord_word[1], j] & k.coord_mask[1]) != 0
        bj = k.flat_b[1, j]
        if free1 || bj <= xmin
            col_count += _set_active_bit!(col_words, j)
        elseif bj <= xmax
            push!(col_add_pos, bj - xmin + 1)
            push!(col_add_idx, j)
        end
    end

    @inbounds for i in 1:k.ninj
        _line_pass_inj_other(k, i, rest) || continue
        free1 = (k.inj_free_words[k.coord_word[1], i] & k.coord_mask[1]) != 0
        bi = k.inj_b[1, i]
        if free1 || bi >= xmax
            row_count += _set_active_bit!(row_words, i)
        elseif bi >= xmin
            row_count += _set_active_bit!(row_words, i)
            rem = bi - xmin + 2
            if rem <= xlen
                push!(row_rem_pos, rem)
                push!(row_rem_idx, i)
            end
        end
    end

    col_perm = scratch.col_perm
    row_perm = scratch.row_perm
    resize!(col_perm, length(col_add_pos))
    @inbounds for i in eachindex(col_perm)
        col_perm[i] = i
    end
    sort!(col_perm; by=i -> @inbounds col_add_pos[i])
    resize!(row_perm, length(row_rem_pos))
    @inbounds for i in eachindex(row_perm)
        row_perm[i] = i
    end
    sort!(row_perm; by=i -> @inbounds row_rem_pos[i])
    pcol = 1
    prow = 1

    line_uids = scratch.line_uids
    @inbounds for xi in 1:xlen
        line_uids[xi] = uid_lookup[line_base + xi - 1]
    end

    xi = 1
    @inbounds while xi <= xlen
        while pcol <= length(col_perm) && col_add_pos[col_perm[pcol]] == xi
            col_count += _set_active_bit!(col_words, col_add_idx[col_perm[pcol]])
            pcol += 1
        end
        while prow <= length(row_perm) && row_rem_pos[row_perm[prow]] == xi
            row_count -= _clear_active_bit!(row_words, row_rem_idx[row_perm[prow]])
            prow += 1
        end

        next_col = pcol <= length(col_perm) ? col_add_pos[col_perm[pcol]] : (xlen + 1)
        next_row = prow <= length(row_perm) ? row_rem_pos[row_perm[prow]] : (xlen + 1)
        xstop = min(next_col, next_row) - 1
        xstop < xi && (xstop = xi)

        v = _dim_from_active_words!(cache, FG, row_words, col_words, row_count, col_count)
        for xj in xi:xstop
            vals[line_uids[xj]] = v
        end
        xi = xstop + 1
    end
    return nothing
end

function _eval_unique_box_sweep!(vals::Vector{Int},
                                 FG::Flange{K,F,N},
                                 unique_points::Vector{T},
                                 unique_index::Dict{T,Int},
                                 axes::NTuple{N,UnitRange{Int}};
                                 cache::Union{Nothing,FlangeDimCache{K,F,N}}=nothing,
                                 threaded::Bool=false) where {K,F,N,T<:NTuple{N,<:Integer}}
    cache0 = cache === nothing ? FlangeDimCache(FG) : cache
    _check_cache_compat(cache0, FG)
    fill!(vals, 0)
    strides = _box_strides(axes)
    uid_lookup = _build_uid_lookup(unique_index, axes, strides)
    xlen = length(axes[1])
    rest_ranges = ntuple(d -> axes[d + 1], N - 1)

    if threaded && Threads.nthreads() > 1
        rest_lines = collect(Iterators.product(rest_ranges...))
        if length(rest_lines) >= 2 * Threads.nthreads()
            nt = Threads.maxthreadid()
            caches = Vector{Any}(undef, nt)
            scratches = Vector{Any}(undef, nt)
            fill!(caches, nothing)
            fill!(scratches, nothing)
            Threads.@threads for li in eachindex(rest_lines)
                tid = Threads.threadid()
                c = caches[tid]
                if c === nothing
                    c = FlangeDimCache(FG)
                    caches[tid] = c
                end
                s = scratches[tid]
                if s === nothing
                    s = _box_sweep_scratch(c.kernel, xlen)
                    scratches[tid] = s
                end
                rest = rest_lines[li]
                line_base = _box_base_index(rest, axes, strides)
                _eval_unique_box_line_sweep!(vals, FG, c, s, uid_lookup, line_base, axes, rest)
            end
            return vals
        end
    end

    scratch0 = _box_sweep_scratch(cache0.kernel, xlen)
    for rest in Iterators.product(rest_ranges...)
        line_base = _box_base_index(rest, axes, strides)
        _eval_unique_box_line_sweep!(vals, FG, cache0, scratch0, uid_lookup, line_base, axes, rest)
    end
    return vals
end

"""
    dim_at_many!(out, FG, points; cache=nothing, sort_points=true,
                 dedup=true, threaded=false, sweep=:auto)

Compute `dim_at(FG, p)` for each point in `points` and write into `out`.

When `sort_points=true` and `points` is an `AbstractVector` of tuples, points are
processed in lexicographic order to increase active-set cache reuse.

`dedup=true` deduplicates query points before evaluation and scatters results back.

`threaded=true` evaluates unique points in parallel using per-thread `FlangeDimCache`.

`sweep=:auto|:none|:box` controls a line-sweep kernel for full box/grid tuple queries:
- `:auto` tries it opportunistically.
- `:none` disables it.
- `:box` requires full-box structure and throws if unavailable.
"""
function dim_at_many!(out::AbstractVector{Int},
                      FG::Flange,
                      points::AbstractVector;
                      cache::Union{Nothing,FlangeDimCache}=nothing,
                      sort_points::Bool=true,
                      dedup::Bool=true,
                      threaded::Bool=false,
                      sweep::Symbol=:auto)
    length(out) == length(points) || error("dim_at_many!: output length must match number of points")
    sweep in (:auto, :none, :box) || error("dim_at_many!: sweep must be :auto, :none, or :box")
    isempty(points) && return out

    unique_points = points
    scatter = Int[]
    unique_index = nothing
    if dedup
        unique_points, scatter, unique_index = _dedup_points(points)
    end
    nuniq = length(unique_points)
    nuniq == 0 && return out

    vals_unique = dedup ? Vector{Int}(undef, nuniq) : out
    did_sweep = false

    if sweep != :none &&
       dedup &&
       unique_index isa Dict &&
       _tuple_int_points(unique_points)
        axes = _full_box_axes(unique_points, unique_index)
        if axes !== nothing
            xlen = length(axes[1])
            # Auto mode: only use sweep when x-axis is long enough to amortize setup.
            use_sweep = sweep == :box || (xlen >= 96 && length(unique_points) >= 2048)
            if use_sweep
                _eval_unique_box_sweep!(vals_unique, FG, unique_points, unique_index, axes;
                                        cache=cache, threaded=threaded)
                did_sweep = true
            end
        elseif sweep == :box
            error("dim_at_many!: sweep=:box requires tuple points that form a full axis-aligned integer box")
        end
    elseif sweep == :box
        error("dim_at_many!: sweep=:box requires tuple points")
    end

    if !did_sweep
        order = if sort_points && _lex_sortable_points(unique_points)
            sortperm(eachindex(unique_points); by=i -> @inbounds(unique_points[i]))
        else
            collect(eachindex(unique_points))
        end
        cache_eval = threaded ? nothing : cache
        _evaluate_unique_points!(vals_unique, FG, unique_points, order;
                                 cache=cache_eval, threaded=threaded)
    end

    if dedup
        @inbounds for i in eachindex(points)
            out[i] = vals_unique[scatter[i]]
        end
    end
    return out
end

"""
    dim_at_many(FG, points; cache=nothing, sort_points=true,
                dedup=true, threaded=false, sweep=:auto)

Vector-returning convenience wrapper for `dim_at_many!`.
"""
function dim_at_many(FG::Flange,
                     points::AbstractVector;
                     cache::Union{Nothing,FlangeDimCache}=nothing,
                     sort_points::Bool=true,
                     dedup::Bool=true,
                     threaded::Bool=false,
                     sweep::Symbol=:auto)
    out = Vector{Int}(undef, length(points))
    return dim_at_many!(out, FG, points;
                        cache=cache,
                        sort_points=sort_points,
                        dedup=dedup,
                        threaded=threaded,
                        sweep=sweep)
end

# ---------------------------------------------------------------------------
# Unified API: dim_at on finite-poset modules (IndicatorResolutions.PModule)
#
# The umbrella module `PosetModules` exports `dim_at` from the flange layer
# (this file). Downstream code/tests also want to write `dim_at(M, q)` when `M`
# is a finite-poset module (a `PModule`).
#
# To avoid creating a second, unrelated `dim_at` function, we extend the *same*
# generic function here, where it is defined.
# ---------------------------------------------------------------------------

import ..IndicatorResolutions

"""
    dim_at(M::IndicatorResolutions.PModule, q::Integer) -> Int

Return the stalk (fiber) dimension of the finite-poset module `M` at vertex `q`.

This is the finite-encoding analogue of `dim_at(FG::Flange, g)`.
"""
@inline function dim_at(M::IndicatorResolutions.PModule, q::Integer)::Int
    qi = Int(q)
    1 <= qi <= length(M.dims) || error("dim_at: vertex index q=$(qi) out of range")
    @inbounds return M.dims[qi]
end


"""
    bounding_box(FG; margin=0)

Return an axis-aligned bounding box `(a, b)` for the region where the flange
can be nontrivial, based on the inequality coordinates of flats/injectives.

This is primarily used as a finite search window for encoding and testing.

For a coordinate `i`:

* flats contribute lower bounds `a[i] >= F.b[i] - margin` when `F.tau.coords[i] == false`
* injectives contribute upper bounds `b[i] <= E.b[i] + margin` when `E.tau.coords[i] == false`

Coordinates fixed by a face (`tau.coords[i] == true`) do not constrain the box.
"""
function bounding_box(FG::Flange; margin::Int = 0)
    n = FG.n
    a = fill(typemin(Int), n)
    b = fill(typemax(Int), n)

    for F in FG.flats
        @inbounds for i in 1:n
            if F.tau.coords[i]
                continue
            end
            a[i] = max(a[i], F.b[i] - margin)
        end
    end

    for E in FG.injectives
        @inbounds for i in 1:n
            if E.tau.coords[i]
                continue
            end
            b[i] = min(b[i], E.b[i] + margin)
        end
    end

    return (a, b)
end

# ---------------------------------------------------------------------------
# Minimization
# ---------------------------------------------------------------------------

struct _UnderlyingKey{N}
    face_mask::NTuple{N,Bool}
    b::NTuple{N,Int}
end

@inline function _face_mask_tuple(tau::Face, ::Val{N}) where {N}
    return ntuple(i -> @inbounds(tau.coords[i]), N)
end

# Helper: a hashable key for grouping flats/injectives by their underlying set.
@inline _underlying_key(F::IndFlat{N}) where {N} = _UnderlyingKey{N}(_face_mask_tuple(F.tau, Val(N)), F.b)
@inline _underlying_key(E::IndInj{N}) where {N} = _UnderlyingKey{N}(_face_mask_tuple(E.tau, Val(N)), E.b)

# Helper: test whether a vector is the zero vector.
@inline function _is_zero_vec(v)
    for x in v
        if !iszero(x)
            return false
        end
    end
    return true
end

# Helper: check whether w is a scalar multiple of v (including the all-zero case).
#
# This is used only inside `minimize`, where coefficients are expected to live in
# a field. If you work over a non-field coefficient type, you may want to disable
# minimization or replace this proportionality test.
function _is_proportional(v, w)::Bool
    if _is_zero_vec(v)
        return _is_zero_vec(w)
    end
    if _is_zero_vec(w)
        return true
    end

    # Find the first nonzero entry of v.
    idx = nothing
    for i in eachindex(v)
        if !iszero(v[i])
            idx = i
            break
        end
    end
    if idx === nothing
        # Should not happen (handled above), but keep it robust.
        return _is_zero_vec(w)
    end

    # Ratio c = w[idx] / v[idx]
    if iszero(w[idx])
        # w is not the zero vector here, so it cannot be proportional.
        return false
    end
    c = w[idx] / v[idx]

    @inbounds for i in eachindex(v)
        if w[i] != c * v[i]
            return false
        end
    end
    return true
end

"""
    minimize(FG; rankfun=rank)

Return a reduced flange obtained by removing proportional duplicate rows/columns
that correspond to *identical underlying* injectives/flats.

Rationale:

* If two flats have the same underlying set (same `tau` mask and same `b`) then
  they are either simultaneously active or inactive at every point `g`.
* If their columns in `phi` are proportional, dropping duplicates does not change
  the fiberwise rank function `dim_at(FG, g)`.

The same logic applies to injectives (rows).

This is primarily a performance optimization (smaller encodings).
"""
function minimize(FG::Flange{K,FT,N}; rankfun = rank) where {K,FT,N}
    flats = FG.flats
    injectives = FG.injectives
    Phi = FG.phi

    # --- Reduce columns (flats) ---
    col_groups = Dict{_UnderlyingKey{N}, Vector{Int}}()
    for (j, F) in enumerate(flats)
        key = _underlying_key(F)
        push!(get!(col_groups, key, Int[]), j)
    end

    keep_cols = Int[]
    for idxs in values(col_groups)
        # Preserve original order by processing indices as-is.
        j0 = idxs[1]
        push!(keep_cols, j0)

        for j in idxs[2:end]
            if _is_proportional(view(Phi, :, j0), view(Phi, :, j))
                # Drop proportional duplicates.
                continue
            end
            push!(keep_cols, j)
        end
    end
    sort!(keep_cols)

    flats2 = flats[keep_cols]
    Phi2 = Phi[:, keep_cols]

    # --- Reduce rows (injectives) ---
    row_groups = Dict{_UnderlyingKey{N}, Vector{Int}}()
    for (i, E) in enumerate(injectives)
        key = _underlying_key(E)
        push!(get!(row_groups, key, Int[]), i)
    end

    keep_rows = Int[]
    for idxs in values(row_groups)
        i0 = idxs[1]
        push!(keep_rows, i0)

        for i in idxs[2:end]
            if _is_proportional(view(Phi2, i0, :), view(Phi2, i, :))
                continue
            end
            push!(keep_rows, i)
        end
    end
    sort!(keep_rows)

    injectives2 = injectives[keep_rows]
    Phi3 = Phi2[keep_rows, :]

    return Flange{K}(FG.n, flats2, injectives2, Phi3)
end

# ---------------------------------------------------------------------------
# Canonical matrices
# ---------------------------------------------------------------------------

"""
    canonical_matrix(flats, injectives)

Build the (0,1) matrix with entry 1 exactly when the underlying flat and injective
can intersect.

This is useful for constructing small example flanges.
"""
function canonical_matrix(flats::Vector{IndFlat{N}}, injectives::Vector{IndInj{N}};
                          field::AbstractCoeffField = QQField()) where {N}
    m = length(injectives)
    n = length(flats)
    A = zeros(field, m, n)
    K = coeff_type(field)

    for (i, E) in enumerate(injectives)
        for (j, F) in enumerate(flats)
            if intersects(F, E)
                A[i, j] = one(K)
            end
        end
    end

    return A
end

"""A simple "identity" matrix between injectives and flats with identical underlying sets."""
function canonical_matrix(FG::Flange{K}) where {K}
    m = length(FG.injectives)
    n = length(FG.flats)
    A = zeros(K, m, n)

    for (i, E) in enumerate(FG.injectives)
        for (j, F) in enumerate(FG.flats)
            if E.tau.coords == F.tau.coords && E.b == F.b
                A[i, j] = one(K)
            end
        end
    end

    return A
end

#---------------------------------------------------------------------------
# Validation and cross checkers
#---------------------------------------------------------------------------

# Cross-validate a Z^n flange against an axis-aligned PL evaluation on the same
# lattice points in a convex-projection box [a,b].  No external Axis* module.

# Minimal axis-aligned shapes for cross-check (not exported elsewhere).
# IMPORTANT: we do not emulate +/- infinity with sentinels.  Instead we
# carry an explicit "free" bitmask that marks coordinates whose bound is
# ignored.  This keeps membership tests exact and avoids magic numbers.
struct AxisUpset{T}
    # a[i] is only used when free[i] == false; otherwise it is a dummy.
    a::Vector{T}      # lower thresholds
    free::BitVector   # coordinates that are free in Z^tau
end
struct AxisDownset{T}
    # b[i] is only used when free[i] == false; otherwise it is a dummy.
    b::Vector{T}      # upper thresholds
    free::BitVector   # coordinates that are free in Z^tau
end


struct AxisFringe{TU,TD,TF}
     n::Int
     births::Vector{AxisUpset{TU}}
     deaths::Vector{AxisDownset{TD}}
     Phi::Matrix{TF}  # same scalar "monomial" matrix
 end

"Translate Zn indecomposables to axis-aligned PL up/down sets for cross-checking."
function flange_to_axis(fr::Flange{K}) where {K}
    n = fr.n
    births = AxisUpset{Int}[]
    for F in fr.flats
        a = Vector{Int}(undef, n)
        for i in 1:n
            # When F.tau.coords[i] is true (free coordinate), a[i] is never read.
            # We still store an Int value to keep the vector concrete.
            a[i] = F.b[i]
        end
        push!(births, AxisUpset{Int}(a, BitVector(F.tau.coords)))
    end
    deaths = AxisDownset{Int}[]
    for E in fr.injectives
        b = Vector{Int}(undef, n)
        for i in 1:n
            # Same comment as above: b[i] is ignored when the coordinate is free.
            b[i] = E.b[i]
        end
        push!(deaths, AxisDownset{Int}(b, BitVector(E.tau.coords)))
    end
    Phi = Matrix{K}(fr.phi)
    return AxisFringe{Int,Int,K}(n, births, deaths, Phi)
end

"Test membership of a lattice point x in an axis-aligned upset or downset (generic and exact)."
_contains(U::AxisUpset{T}, x::AbstractVector{T}) where {T} =
    all(U.free[i] || (x[i] >= U.a[i]) for i in 1:length(x))
_contains(D::AxisDownset{T}, x::AbstractVector{T}) where {T} =
    all(D.free[i] || (x[i] <= D.b[i]) for i in 1:length(x))
_contains(U::AxisUpset{T}, x::NTuple{N,T}) where {N,T} =
    all(U.free[i] || (x[i] >= U.a[i]) for i in 1:N)
_contains(D::AxisDownset{T}, x::NTuple{N,T}) where {N,T} =
    all(D.free[i] || (x[i] <= D.b[i]) for i in 1:N)

"""
    cross_validate(fr::Flange; margin=1, rankfun=(A)->FieldLinAlg.rank(fr.field, A))

1. Build the convex-projection box [a,b] (heuristic 'bounding_box').
2. Evaluate 'dim M_g' for all integer 'g in [a,b]' via the flange.
3. Build an axis-aligned proxy and evaluate the same lattice points.
4. Compare; return '(all_equal?, report::Dict)'.
"""
function cross_validate(fr::Flange; margin=1,
                        rankfun = A -> FieldLinAlg.rank(fr.field, A))
    a, b = bounding_box(fr; margin)
    ranges = ntuple(i -> a[i]:b[i], fr.n)

    # Flange evaluation (use provided rank function).
    dims_Z = Dict{Tuple{Vararg{Int}}, Int}()
    rows_buf = Int[]
    cols_buf = Int[]
    sizehint!(rows_buf, length(fr.injectives))
    sizehint!(cols_buf, length(fr.flats))
    for t in Iterators.product(ranges...)
        dims_Z[t] = _dim_at_rankfun!(fr, t, rankfun, rows_buf, cols_buf)
    end

    # Axis-aligned proxy (compare apples-to-apples with the flange)
    afr = flange_to_axis(fr)
    dims_PL = Dict{Tuple{Vararg{Int}}, Int}()
    idxr = Int[]
    idxc = Int[]
    sizehint!(idxr, length(afr.deaths))
    sizehint!(idxc, length(afr.births))
    for t in Iterators.product(ranges...)
        empty!(idxr)
        empty!(idxc)
        @inbounds for i in eachindex(afr.deaths)
            if _contains(afr.deaths[i], t)
                push!(idxr, i)
            end
        end
        @inbounds for j in eachindex(afr.births)
            if _contains(afr.births[j], t)
                push!(idxc, j)
            end
        end
        if isempty(idxr) || isempty(idxc)
            dims_PL[t] = 0
        else
            # Rank exactly with the same provided rank function.
            Phi_x = @view afr.Phi[idxr, idxc]
            dims_PL[t] = rankfun(Phi_x)
        end
    end

    # Compare
    mism = Dict{Tuple{Vararg{Int}}, Tuple{Int,Int}}()
    for (k,v) in dims_Z
        if v != dims_PL[k]; mism[k] = (v, dims_PL[k]); end
    end
    ok = isempty(mism)
    tested = prod(length(r) for r in ranges)
    report = Dict("box" => (a,b), "mismatches" => mism,
                  "tested" => tested, "agree" => tested - length(mism))
    return ok, report
end

end # module
