# shared.jl -- owner-level cache/shared contracts for DerivedFunctors

using ..CoreModules: AbstractCoeffField, RealField, coeff_type, field_from_eltype, coerce,
                     AbstractHomSystemCache
using ..Options: EncodingOptions, ResolutionOptions, DerivedFunctorOptions
using ..FieldLinAlg
import ..Modules
import ..AbelianCategories
import ..Results: provenance
using ..Modules: PModule, PMorphism
using SparseArrays: sparse, SparseMatrixCSC
import Base.Threads

# Fixed-model computations cannot change representative coordinates merely by
# accepting a different `canon`. Validate before cache lookup as well as builds.
function _validate_native_derived_options(df::DerivedFunctorOptions, operation::AbstractString,
                                          model::Symbol)
    df.maxdeg >= 0 || throw(ArgumentError("$operation: maxdeg must be nonnegative."))
    df.model in (:auto, model) ||
        throw(ArgumentError("$operation: model must be :auto or :$model, got $(df.model)."))
    native_canon = model in (:projective, :injective) ? model : :none
    df.canon in (:auto, :none, native_canon) ||
        throw(ArgumentError("$operation: canon=$(df.canon) is incompatible with the :$model model; use :auto, :none$(native_canon === :none ? "" : ", or :$native_canon")."))
    return nothing
end

"""
    HomSystemCache{K}()
    HomSystemCache(K::Type)

Shared cache for expensive Hom-system setup reused across derived pipelines.
Lookup and publication are synchronized; Hom construction and coordinate solves
run outside the lock with independently leased scratch.

Stored entries:
- `hom`: `HomSpace` objects keyed by `(objectid(dom), objectid(cod))`
- `precompose`: typed coordinate matrices with weak input/basis identity witnesses
- `postcompose`: typed coordinate matrices with weak input/basis identity witnesses

Integer identity keys select candidate entries; every coordinate-result hit
also verifies its live owners. Keep source matrices read-only while cached.
"""
struct _HomKey2
    a::UInt
    b::UInt
end

struct _HomKey3
    a::UInt
    b::UInt
    c::UInt
end

# Integer cache keys are only lookup hints: recycled IDs and hash collisions
# must still identify the same live inputs before their result is reusable.
struct _ImmutableIdentityWitness
    value::Any
end

struct _ModuleIdentityWitness
    token::WeakRef
    field::AbstractCoeffField
    poset::Any
    dims::WeakRef
    edge_arrays::NTuple{4,WeakRef}
end

# PModule is an immutable wrapper: a WeakRef to a temporary boxed copy can
# expire while the same module value is still live. Witness its stable mutable
# storage instead. The field/poset/storage contract also distinguishes manually
# assembled modules that happen to share the same mutable memo owner.
@inline function _identity_witness(M::PModule)
    edges = M.edge_maps
    return _ModuleIdentityWitness(WeakRef(M.map_compose), M.field, M.Q, WeakRef(M.dims),
        map(WeakRef, (edges.preds, edges.succs, edges.maps_from_pred, edges.maps_to_succ)))
end
@inline function _identity_witness(owner)
    # Generic immutable inputs, such as matrix views, have no unique mutable
    # token. Retain that value itself; weakly retain ordinary mutable storage.
    return ismutable(owner) ? WeakRef(owner) : _ImmutableIdentityWitness(owner)
end

@inline _identity_witness_matches(w::Union{WeakRef,_ImmutableIdentityWitness}, owner) = w.value === owner
@inline function _identity_witness_matches(w::_ModuleIdentityWitness, M::PModule)
    edges = M.edge_maps
    return w.token.value === M.map_compose && w.field === M.field && w.poset === M.Q &&
        w.dims.value === M.dims && w.edge_arrays[1].value === edges.preds &&
        w.edge_arrays[2].value === edges.succs && w.edge_arrays[3].value === edges.maps_from_pred &&
        w.edge_arrays[4].value === edges.maps_to_succ
end
@inline _identity_witness_matches(::_ModuleIdentityWitness, owner) = false

struct _IdentityCacheEntry{N,C,V}
    owners::NTuple{N,Union{WeakRef,_ImmutableIdentityWitness,_ModuleIdentityWitness}}
    contract::C
    value::V
end
_identity_cache_entry(owners::Tuple, contract, value) =
    _IdentityCacheEntry(map(_identity_witness, owners), contract, value)
@inline function _identity_cache_matches(entry::_IdentityCacheEntry{N}, owners::NTuple{N,Any}) where {N}
    return all(i -> _identity_witness_matches(entry.owners[i], owners[i]), 1:N)
end

# PMorphism is immutable; its mutable component vector is the stable witness.
# Basis witnesses also invalidate coordinate results when a Hom basis is replaced.
@inline _hom_map_owners(Hdom, Hcod, f) =
    (Hdom, Hcod, Hdom.basis_matrix, Hcod.basis_matrix, f.dom, f.cod, f.comps)

# Derived-complex and tensor caches use this owner as a weak key. Its identity
# must therefore be mutable/finalizable, even though its fields are not replaced.
mutable struct HomSystemCache{HV,PV,QV} <: AbstractHomSystemCache
    lock::ReentrantLock
    hom::Dict{_HomKey2,HV}
    precompose::Dict{_HomKey3,_IdentityCacheEntry{7,Nothing,PV}}
    postcompose::Dict{_HomKey3,_IdentityCacheEntry{7,Nothing,QV}}
end

function HomSystemCache(::Type{HV}, ::Type{PV}, ::Type{QV}; shard_capacity::Int=256) where {HV,PV,QV}
    hom = Dict{_HomKey2,HV}()
    pre = Dict{_HomKey3,_IdentityCacheEntry{7,Nothing,PV}}()
    post = Dict{_HomKey3,_IdentityCacheEntry{7,Nothing,QV}}()
    if shard_capacity > 0
        sizehint!(hom, shard_capacity)
        sizehint!(pre, shard_capacity)
        sizehint!(post, shard_capacity)
    end
    return HomSystemCache(ReentrantLock(), hom, pre, post)
end

@inline function _cache_lookup(cache::HomSystemCache,
                              store::Dict{_HomKey3,_IdentityCacheEntry{7,Nothing,V}},
                              key::_HomKey3, owners::Tuple) where {V}
    return lock(cache.lock) do
        entry = get(store, key, nothing)
        entry !== nothing && _identity_cache_matches(entry, owners) ? entry.value : nothing
    end::Union{Nothing,V}
end

@inline function _cache_store_or_get!(cache::HomSystemCache,
                                     store::Dict{_HomKey3,_IdentityCacheEntry{7,Nothing,V}},
                                     key::_HomKey3, value::V, owners::Tuple) where {V}
    return lock(cache.lock) do
        entry = get(store, key, nothing)
        entry !== nothing && _identity_cache_matches(entry, owners) && return entry.value
        store[key] = _identity_cache_entry(owners, nothing, value)
        return value
    end::V
end

function clear_hom_system_cache!(cache::HomSystemCache)
    lock(cache.lock) do
        empty!(cache.hom)
        empty!(cache.precompose)
        empty!(cache.postcompose)
    end
    return nothing
end

@inline _cache_key2(a, b) = _HomKey2(UInt(objectid(a)), UInt(objectid(b)))
@inline _cache_key3(a, b, c) = _HomKey3(UInt(objectid(a)), UInt(objectid(b)), UInt(objectid(c)))

@inline function _cache_lookup(cache::HomSystemCache, store::AbstractDict{K,V}, key::K) where {K,V}
    return lock(cache.lock) do
        get(store, key, nothing)::Union{Nothing,V}
    end
end

@inline function _cache_store_or_get!(cache::HomSystemCache, store::AbstractDict{K,V}, key::K, value::V) where {K,V}
    return lock(cache.lock) do
        get!(store, key, value)::V
    end
end

# -----------------------------------------------------------------------------
# Owner-level semantic accessor generics
# -----------------------------------------------------------------------------

function resolution_terms end
function resolution_differentials end
function augmentation_map end
function coaugmentation_map end
function source_module end
function target_module end
function nonzero_degrees end
function degree_dimensions end
function total_dimension end
function page_dimensions end
function generator_degrees end
function algebra_field end
function resolution_summary end
function hom_summary end
function ext_summary end
function tor_summary end
function double_complex_summary end
function derived_les_summary end
function algebra_summary end
function parent_algebra end
function element_degree end
function element_coordinates end
function wrapped_spectral_sequence end
function underlying_ext_space end
function underlying_tor_space end
function cached_product_degrees end
function check_projective_resolution end
function check_injective_resolution end
function check_ext_spectral_sequence end
function check_tor_spectral_sequence end
function check_ext_algebra end
function check_tor_algebra end

"""
    DerivedFunctorValidationSummary

Compact wrapper around a validation report produced by the `DerivedFunctors`
UX-layer `check_*` helpers.

The wrapped report is a `NamedTuple` whose exact auxiliary fields depend on the
validated object, but every report contains at least:
- `kind`: symbolic object kind
- `valid`: overall validation result
- `issues`: a tuple of human-readable validation issues
"""
struct DerivedFunctorValidationSummary{R}
    report::R
end

@inline derived_functor_validation_summary(report::NamedTuple) = DerivedFunctorValidationSummary(report)

@inline function _derived_validation_report(
    kind::Symbol,
    valid::Bool;
    issues::AbstractVector{<:AbstractString}=String[],
    kwargs...,
)
    return (; kind, valid, issues=Tuple(String.(issues)), kwargs...)
end

@inline function _throw_invalid_derived_functor(fname::Symbol, issues::AbstractVector{<:AbstractString})
    msg = isempty(issues) ? "invalid object" : " - " * join(issues, "\n - ")
    Base.throw(ArgumentError(string(fname, ": validation failed\n", msg)))
end

function Base.show(io::IO, summary::DerivedFunctorValidationSummary)
    r = summary.report
    print(io, "DerivedFunctorValidationSummary(kind=", r.kind,
          ", valid=", r.valid,
          ", issues=", length(r.issues), ")")
end

function Base.show(io::IO, ::MIME"text/plain", summary::DerivedFunctorValidationSummary)
    r = summary.report
    println(io, "DerivedFunctorValidationSummary")
    println(io, "  kind: ", r.kind)
    println(io, "  valid: ", r.valid)
    println(io, "  issues: ", length(r.issues))
    if !isempty(r.issues)
        println(io, "  first_issue: ", first(r.issues))
    end
end

@inline _total_offset_aidx(a::Int, amin::Int) = a - amin + 1

function _build_total_offsets_grid(
    amin::Int, amax::Int,
    bmin::Int, bmax::Int,
    dims::AbstractMatrix{Int},
)
    tmin = amin + bmin
    tmax = amax + bmax
    offsets = [fill(-1, amax - amin + 1) for _ in tmin:tmax]
    dimsCt = zeros(Int, tmax - tmin + 1)

    for t in tmin:tmax
        off = 0
        row = offsets[t - tmin + 1]
        alo = max(amin, t - bmax)
        ahi = min(amax, t - bmin)
        for a in alo:ahi
            ai = _total_offset_aidx(a, amin)
            b = t - a
            bi = b - bmin + 1
            row[ai] = off
            off += dims[ai, bi]
        end
        dimsCt[t - tmin + 1] = off
    end

    return offsets, dimsCt, tmin, tmax
end

@inline function _total_offset_get(
    offsets::Vector{Vector{Int}},
    t::Int,
    tmin::Int,
    amin::Int,
    a::Int,
)
    v = offsets[t - tmin + 1][_total_offset_aidx(a, amin)]
    v >= 0 || error("_total_offset_get: invalid (t,a)=($t,$a)")
    return v
end

"""
Utils: shared low-level utilities for the derived-functors layer.

Intended contents (move here incrementally):
- small linear algebra helpers
- sparse-matrix manipulation helpers
- caching/memoization helpers local to DerivedFunctors
- generic composition and indexing helpers

Design rule:
- keep this dependency-light; higher-level constructions should depend on Utils,
  not the other way around.
"""
