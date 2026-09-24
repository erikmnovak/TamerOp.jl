# public_api.jl -- owner-level reexports, caches, and convenience wrappers

# -----------------------------------------------------------------------------
# Public surface reexports (parent module)
# -----------------------------------------------------------------------------

using .Utils: compose
import ..IndicatorResolutions: resolution_length
import ..ChainComplexes: describe, sequence_dimensions, sequence_maps, sequence_entry, spectral_sequence_summary

import .Resolutions:
    ProjectiveResolution, InjectiveResolution,
    projective_resolution, injective_resolution,
    betti, betti_table, bass, bass_table,
    minimality_report,
    ProjectiveMinimalityReport, InjectiveMinimalityReport,
    is_minimal, assert_minimal,
    lift_injective_chainmap,
    _coeff_matrix_upsets,
    _flatten_gens_at,
    _solve_downset_postcompose_coeff

import .ExtTorSpaces:
    Hom, HomSpace,
    degree_range,
    ExtSpaceProjective, ExtSpaceInjective, ExtSpace,
    Ext, ExtInjective,
    Tor, TorSpace, TorSpaceSecond,
    dim, basis, representative, cycles, boundaries, coordinates,
    comparison_isomorphism, comparison_isomorphisms,
    projective_model, injective_model,
    hom_ext_first_page, ext_dimensions_via_indicator_resolutions

import .Functoriality:
    ext_map_first, ext_map_second,
    tor_map_first, tor_map_second,
    connecting_hom, connecting_hom_first,
    ExtLongExactSequenceSecond, ExtLongExactSequenceFirst,
    TorLongExactSequenceFirst, TorLongExactSequenceSecond,
    _FUNCTORIALITY_USE_HOM_WORKSPACES,
    _PrecomposeWorkspace,
    _PostcomposeWorkspace,
    _precompose_matrix,
    _postcompose_matrix,
    _precompose_on_hom_cochains_from_projective_coeff,
    _tensor_map_on_tor_chains_from_projective_coeff,
    _tor_blockdiag_map_on_chains

import .Algebras:
    yoneda_product,
    ExtAlgebra, ExtElement,
    multiply, element, unit, precompute!,
    TorAlgebra, TorElement,
    set_chain_product!, set_chain_product_generator!,
    multiplication_matrix,
    ext_action_on_tor

import .SpectralSequences:
    ExtDoubleComplex, ExtSpectralSequence,
    TorDoubleComplex, TorSpectralSequence,
    TorSpectralPage

import .Backends:
    ExtZn, ExtRn,
    pmodule_on_box,
    projective_resolution_Zn, injective_resolution_Zn,
    projective_resolution_Rn, injective_resolution_Rn

using .HomExtEngine:
    build_hom_tot_complex,
    build_hom_bicomplex_data,
    ext_dims_via_resolutions, pi0_count

@inline function HomSystemCache(::Type{K}) where {K}
    MT = SparseMatrixCSC{K,Int}
    return HomSystemCache(HomSpace{K}, MT, MT)
end

HomSystemCache{K}() where {K} = HomSystemCache(K)

@inline _hom_with_cache(M::PModule{K}, N::PModule{K}, ::Nothing) where {K} = Hom(M, N)

function _hom_with_cache(
    M::PModule{K},
    N::PModule{K},
    cache::HomSystemCache{HomSpace{K},SparseMatrixCSC{K,Int},SparseMatrixCSC{K,Int}},
) where {K}
    key = _cache_key2(M, N)
    cached = _cache_lookup(cache, cache.hom, key)
    cached === nothing || return cached
    H = Hom(M, N)
    return _cache_store_or_get!(cache, cache.hom, key, H)
end

function _hom_with_cache(M::PModule{K}, N::PModule{K}, ::HomSystemCache) where {K}
    error("hom_with_cache: cache scalar type mismatch for coefficient type $(K).")
end

function hom_with_cache(M::PModule{K}, N::PModule{K}; cache::Union{Nothing,HomSystemCache}=nothing) where {K}
    return _hom_with_cache(M, N, cache)
end

@inline function _precompose_cached(Hdom::HomSpace{K}, Hcod::HomSpace{K}, f::PMorphism{K}, ::Nothing) where {K}
    return sparse(_precompose_matrix(Hdom, Hcod, f))
end

function _precompose_cached(
    Hdom::HomSpace{K},
    Hcod::HomSpace{K},
    f::PMorphism{K},
    cache::HomSystemCache{HomSpace{K},SparseMatrixCSC{K,Int},SparseMatrixCSC{K,Int}},
) where {K}
    key = _cache_key3(Hdom, Hcod, f)
    owners = _hom_map_owners(Hdom, Hcod, f)
    cached = _cache_lookup(cache, cache.precompose, key, owners)
    cached === nothing || return cached
    F = sparse(_precompose_matrix(Hdom, Hcod, f))
    return _cache_store_or_get!(cache, cache.precompose, key, F, owners)
end

function _precompose_cached(Hdom::HomSpace{K}, Hcod::HomSpace{K}, f::PMorphism{K}, ::HomSystemCache) where {K}
    error("precompose_matrix_cached: cache scalar type mismatch for coefficient type $(K).")
end

function precompose_matrix_cached(Hdom::HomSpace{K}, Hcod::HomSpace{K}, f::PMorphism{K}; cache::Union{Nothing,HomSystemCache}=nothing) where {K}
    return _precompose_cached(Hdom, Hcod, f, cache)
end

@inline function _postcompose_cached(Hdom::HomSpace{K}, Hcod::HomSpace{K}, g::PMorphism{K}, ::Nothing) where {K}
    return sparse(_postcompose_matrix(Hdom, Hcod, g))
end

function _postcompose_cached(
    Hdom::HomSpace{K},
    Hcod::HomSpace{K},
    g::PMorphism{K},
    cache::HomSystemCache{HomSpace{K},SparseMatrixCSC{K,Int},SparseMatrixCSC{K,Int}},
) where {K}
    key = _cache_key3(Hdom, Hcod, g)
    owners = _hom_map_owners(Hdom, Hcod, g)
    cached = _cache_lookup(cache, cache.postcompose, key, owners)
    cached === nothing || return cached
    F = sparse(_postcompose_matrix(Hdom, Hcod, g))
    return _cache_store_or_get!(cache, cache.postcompose, key, F, owners)
end

function _postcompose_cached(Hdom::HomSpace{K}, Hcod::HomSpace{K}, g::PMorphism{K}, ::HomSystemCache) where {K}
    error("postcompose_matrix_cached: cache scalar type mismatch for coefficient type $(K).")
end

function postcompose_matrix_cached(Hdom::HomSpace{K}, Hcod::HomSpace{K}, g::PMorphism{K}; cache::Union{Nothing,HomSystemCache}=nothing) where {K}
    return _postcompose_cached(Hdom, Hcod, g, cache)
end

# -----------------------------------------------------------------------------
# Public opts-default wrappers
# -----------------------------------------------------------------------------

Hom(M, N; cache::Union{Nothing,HomSystemCache}=nothing) =
    hom_with_cache(M, N; cache=cache)

projective_resolution(M; opts::ResolutionOptions=ResolutionOptions(), cache=nothing) =
    projective_resolution(M, opts; cache=cache)
injective_resolution(M; opts::ResolutionOptions=ResolutionOptions(), cache=nothing) =
    injective_resolution(M, opts; cache=cache)
betti(M; opts::ResolutionOptions=ResolutionOptions()) =
    betti(M, opts)
bass(M; opts::ResolutionOptions=ResolutionOptions()) =
    bass(M, opts)

Ext(M, N; opts::DerivedFunctorOptions=DerivedFunctorOptions(), cache=nothing) =
    Ext(M, N, opts; cache=cache)
ExtInjective(M, N; opts::DerivedFunctorOptions=DerivedFunctorOptions(), cache=nothing) =
    ExtInjective(M, N, opts; cache=cache)
ExtSpace(M, N; opts::DerivedFunctorOptions=DerivedFunctorOptions(), check::Bool=true, cache=nothing) =
    ExtSpace(M, N, opts; check=check, cache=cache)
Tor(Rop, L; opts::DerivedFunctorOptions=DerivedFunctorOptions(), res=nothing, cache=nothing) =
    Tor(Rop, L, opts; res=res, cache=cache)
ExtAlgebra(M; opts::DerivedFunctorOptions=DerivedFunctorOptions()) =
    ExtAlgebra(M, opts)
ext_action_on_tor(A, T, x; opts::DerivedFunctorOptions=DerivedFunctorOptions()) =
    ext_action_on_tor(A, T, x, opts)

TorSpectralSequence(Rop, L; maxlen=nothing, maxlenR=nothing, maxlenL=nothing,
                    first::Symbol=:vertical, cache=nothing) =
    TorSpectralSequence(Rop, L; maxlen=maxlen, maxlenR=maxlenR, maxlenL=maxlenL,
                        first=first, cache=cache)

ExtZn(FG1, FG2; enc::EncodingOptions=EncodingOptions(field=FG1.field), df::DerivedFunctorOptions=DerivedFunctorOptions(), kwargs...) =
    ExtZn(FG1, FG2, enc, df; kwargs...)
ExtRn(F1, F2; enc::EncodingOptions=EncodingOptions(), df::DerivedFunctorOptions=DerivedFunctorOptions()) =
    ExtRn(F1, F2, enc, df)

ExtLongExactSequenceSecond(M, A, B, C, i, p; opts::DerivedFunctorOptions=DerivedFunctorOptions()) =
    ExtLongExactSequenceSecond(M, A, B, C, i, p, opts)
ExtLongExactSequenceSecond(M, ses; opts::DerivedFunctorOptions=DerivedFunctorOptions()) =
    ExtLongExactSequenceSecond(M, ses, opts)

ExtLongExactSequenceFirst(A, B, C, N, i, p; opts::DerivedFunctorOptions=DerivedFunctorOptions()) =
    ExtLongExactSequenceFirst(A, B, C, N, i, p, opts)
ExtLongExactSequenceFirst(ses, N; opts::DerivedFunctorOptions=DerivedFunctorOptions()) =
    ExtLongExactSequenceFirst(ses, N, opts)

TorLongExactSequenceSecond(Rop, i, p; opts::DerivedFunctorOptions=DerivedFunctorOptions()) =
    TorLongExactSequenceSecond(Rop, i, p, opts)
TorLongExactSequenceSecond(Rop, ses; opts::DerivedFunctorOptions=DerivedFunctorOptions()) =
    TorLongExactSequenceSecond(Rop, ses, opts)

TorLongExactSequenceFirst(L, i, p; opts::DerivedFunctorOptions=DerivedFunctorOptions()) =
    TorLongExactSequenceFirst(L, i, p, opts)
TorLongExactSequenceFirst(L, ses; opts::DerivedFunctorOptions=DerivedFunctorOptions()) =
    TorLongExactSequenceFirst(L, ses, opts)

projective_resolution_Zn(FG; enc::EncodingOptions=EncodingOptions(field=FG.field), res::ResolutionOptions=ResolutionOptions(), return_encoding::Bool=false,
                         threads::Bool = (Threads.nthreads() > 1)) =
    projective_resolution_Zn(FG, enc, res;
                             return_encoding=return_encoding, threads=threads)
injective_resolution_Zn(FG; enc::EncodingOptions=EncodingOptions(field=FG.field), res::ResolutionOptions=ResolutionOptions(), return_encoding::Bool=false,
                        threads::Bool = (Threads.nthreads() > 1)) =
    injective_resolution_Zn(FG, enc, res;
                            return_encoding=return_encoding, threads=threads)

projective_resolution_Rn(FG; enc::EncodingOptions=EncodingOptions(), res::ResolutionOptions=ResolutionOptions(), return_encoding::Bool=false,
                         threads::Bool = (Threads.nthreads() > 1)) =
    projective_resolution_Rn(FG, enc, res;
                             return_encoding=return_encoding, threads=threads)
injective_resolution_Rn(FG; enc::EncodingOptions=EncodingOptions(), res::ResolutionOptions=ResolutionOptions(), return_encoding::Bool=false,
                        threads::Bool = (Threads.nthreads() > 1)) =
    injective_resolution_Rn(FG, enc, res;
                            return_encoding=return_encoding, threads=threads)

# -----------------------------------------------------------------------------
# Shared describe(...) bridge into ChainComplexes
# -----------------------------------------------------------------------------

# All entries retain the actual finite base by reference; no relation matrix,
# resolution comparison or representative is materialized for provenance.
function _finite_derived_provenance(P, field; degree, degree_convention, model,
        category=:finite_poset_representations, orientation=:forward,
        argument_variance=())
    return (; category, base_poset=P, field, degree, degree_convention, model,
            orientation, argument_variance, ambient_identification=:not_asserted)
end

provenance(H::HomSpace) = _finite_derived_provenance(H.dom.Q, H.dom.field;
    degree=0:0, degree_convention=:cohomological, model=:ordinary,
    argument_variance=(:contravariant, :covariant))

provenance(E::ExtSpaceProjective) = _finite_derived_provenance(E.M.Q, E.M.field;
    degree=degree_range(E), degree_convention=:cohomological, model=:projective,
    argument_variance=(:contravariant, :covariant))
provenance(E::ExtSpaceInjective) = _finite_derived_provenance(E.M.Q, E.M.field;
    degree=degree_range(E), degree_convention=:cohomological, model=:injective,
    argument_variance=(:contravariant, :covariant))
provenance(E::ExtSpace) = merge(_finite_derived_provenance(E.M.Q, E.M.field;
    degree=degree_range(E), degree_convention=:cohomological, model=:unified,
    argument_variance=(:contravariant, :covariant)), (canonical_model=E.canon,))

provenance(T::Union{TorSpace,TorSpaceSecond}) = _finite_derived_provenance(
    target_module(T).Q, source_module(T).field;
    category=:incidence_algebra_tensor, degree=degree_range(T),
    degree_convention=:homological, model=T isa TorSpace ? :first : :second,
    orientation=(right=:opposite, left=:forward),
    argument_variance=(:covariant, :covariant))

provenance(res::ProjectiveResolution) = _finite_derived_provenance(res.M.Q, res.M.field;
    degree=0:resolution_length(res), degree_convention=:homological, model=:projective)
provenance(res::InjectiveResolution) = _finite_derived_provenance(res.N.Q, res.N.field;
    degree=0:resolution_length(res), degree_convention=:cohomological, model=:injective)
provenance(A::ExtAlgebra) = merge(provenance(underlying_ext_space(A)), (product=:yoneda,))
provenance(A::TorAlgebra) = merge(provenance(underlying_tor_space(A)), (product=:supplied_chain_maps,))
provenance(x::Union{ExtElement,TorElement}) = merge(provenance(parent_algebra(x)), (degree=element_degree(x),))
provenance(les::Union{ExtLongExactSequenceFirst,ExtLongExactSequenceSecond}) =
    merge(provenance(les.EA), (degree=degree_range(les), construction=:long_exact_sequence))
provenance(les::Union{TorLongExactSequenceFirst,TorLongExactSequenceSecond}) =
    merge(provenance(les.TorA), (degree=degree_range(les), construction=:long_exact_sequence))

# Raw vector-space complexes erase the module/encoding from which they arose.
# They retain the coefficient field, including numerical tolerance policy.
function provenance(C::ChainComplexes.CochainComplex{K}) where {K}
    return (category=:vector_space_complexes, base_poset=nothing,
            field=C.field, field_source=:stored_field,
            degree=C.tmin:C.tmax, degree_convention=:cohomological,
            model=:stored_complex, ambient_identification=:not_asserted)
end
function provenance(DC::ChainComplexes.DoubleComplex{K}) where {K}
    return (category=:vector_space_complexes, base_poset=nothing,
            field=DC.field, field_source=:stored_field,
            degree=(DC.amin+DC.bmin):(DC.amax+DC.bmax), degree_convention=:cohomological,
            model=:stored_bicomplex, ambient_identification=:not_asserted)
end
provenance(ss::ChainComplexes.SpectralSequence) =
    merge(provenance(ss.DC), (model=:spectral_sequence_of_stored_bicomplex, filtration=ss.first))
function provenance(TSS::TorSpectralSequence)
    base = provenance(TSS.ss)
    return merge(base, (degree=(-last(base.degree)):(-first(base.degree)),
                       degree_convention=:homological, reindexing=:tor))
end

@inline function _describe_derived_element(kind::Symbol, x)
    coords = element_coordinates(x)
    return (
        kind=kind,
        provenance=provenance(x),
        field=algebra_field(parent_algebra(x)),
        degree=element_degree(x),
        coordinate_length=length(coords),
        nonzero_coordinates=count(y -> !iszero(y), coords),
        is_zero=all(iszero, coords),
    )
end

describe(res::ProjectiveResolution) = resolution_summary(res)
describe(res::InjectiveResolution) = resolution_summary(res)
describe(H::HomSpace) = hom_summary(H)
describe(E::ExtSpaceProjective) = ext_summary(E)
describe(E::ExtSpaceInjective) = ext_summary(E)
describe(E::ExtSpace) = ext_summary(E)
describe(T::TorSpace) = tor_summary(T)
describe(T::TorSpaceSecond) = tor_summary(T)
describe(A::ExtAlgebra) = algebra_summary(A)
describe(A::TorAlgebra) = algebra_summary(A)
describe(x::ExtElement) = _describe_derived_element(:ext_element, x)
describe(x::TorElement) = _describe_derived_element(:tor_element, x)
describe(les::ExtLongExactSequenceSecond) = derived_les_summary(les)
describe(les::ExtLongExactSequenceFirst) = derived_les_summary(les)
describe(les::TorLongExactSequenceSecond) = derived_les_summary(les)
describe(les::TorLongExactSequenceFirst) = derived_les_summary(les)
describe(TSS::TorSpectralSequence) = spectral_sequence_summary(TSS)
