module PosetModules
# =============================================================================
# PosetModules.jl
#
# Umbrella entrypoint for the library.
#
# This file has three jobs:
#   1) Load internal submodules in a stable include order.
#   2) Define the stable public API surface (curated imports + exports).
#   3) Define the opt-in broad surface for power users (PosetModules.Advanced).
#
# Notes:
# - CoreModules.jl now also defines the small sibling module PosetModules.Stats.
#   This reduces file count while preserving the module path PosetModules.Stats.
# - Workflow.jl defines the public narrative wrappers (encode/resolve/ext/tor/...).
#   Keep it loaded before the public export list below.
# =============================================================================

# 1) Core helpers, interface hooks, options/results, plus small stats helpers.
include("CoreModules.jl")   # defines PosetModules.CoreModules and PosetModules.Stats
include("RegionGeometry.jl")

# 2) Linear algebra engines (field-generic)
include("FieldLinAlg.jl")

# 3) Finite poset + indicator sets + fringe presentations
include("FiniteFringe.jl")
include("Modules.jl")
include("AbelianCategories.jl")

# 4) Shared record types for one-step indicator (co)presentations
#    NOTE: these are now defined in FiniteFringe.jl as `module IndicatorTypes`
#    to keep include-order constraints simple while preserving the public module
#    name `PosetModules.IndicatorTypes`.

# 5) Indicator resolutions (resolution builders)
include("IndicatorResolutions.jl")

# 6) Zn flange data structure
include("FlangeZn.jl")

# 7) ZnEncoding (needs FlangeZn + FiniteFringe)
include("ZnEncoding.jl")

# 8) PL backends for R^n (general polyhedra + axis-aligned boxes)
include("PLPolyhedra.jl")
include("PLBackend.jl")

# 9) Derived-functor and complexes layer
include("ChainComplexes.jl")
include("DerivedFunctors.jl")
include("ModuleComplexes.jl")
include("ChangeOfPosets.jl")

# 10) JSON IO layer (internal formats + external adapters)
include("Serialization.jl")

# 11) Invariants and summaries
include("Invariants.jl")

# 12) High-level workflow wrappers (encode/resolve/ext/tor/invariant, etc.)
include("Workflow.jl")

# 13) 2D visualization helpers
include("Viz2D.jl")

# =============================================================================
# PublicAPI.jl
#
# This file is the single source of truth for PosetModules' stable public
# surface. Everything not listed here remains accessible by qualification,
# e.g. 
#     PosetModules.Invariants.rank_invariant
#     PosetModules.DerivedFunctors.Ext
#
# Guiding principle:
#   - Most users should be productive with the workflow layer (encode/resolve/...).
#   - Power users should have a small set of composable "primitives" they can
#     use to build custom examples, do sanity checks, and pipeline/cache results.
# =============================================================================

# -----------------------------------------------------------------------------
# Stable public imports (types, options, results)
# -----------------------------------------------------------------------------

using .CoreModules: QQ,
                    EncodingOptions, ResolutionOptions, DerivedFunctorOptions, InvariantOptions,
                    ResolutionCache, clear_resolution_cache!,
                    SessionCache, EncodingCache, ModuleCache,
                    clear_session_cache!, clear_encoding_cache!, clear_module_cache!,
                    EncodingResult, ResolutionResult, InvariantResult, unwrap,
                    change_field, encode_from_data, ingest,
                    axes_from_encoding,
                    dimension,
                    representatives,
                    locate,
                    CompiledEncoding, compile_encoding, encoding_map

using .FlangeZn: Flange, face
using .PLPolyhedra: PLFringe
using .PLBackend: BoxUpset, BoxDownset


# Finite-poset primitives (a small algebra of objects)
# These are foundational and enable custom examples without running the full
# Zn/PL encoding pipeline.
using .FiniteFringe: FiniteFringeOptions,
                     AbstractPoset, FinitePoset, ProductOfChainsPoset, GridPoset, ProductPoset,
                     Upset, Downset,
                     principal_upset, principal_downset,
                     upset_from_generators, downset_from_generators, upset_closure, downset_closure,
                     FringeModule, one_by_one_fringe,
                     cover_edges, nvertices, leq, leq_matrix, upset_indices, downset_indices, upset_iter, downset_iter,
                     leq_row, leq_col, poset_equal, poset_equal_opposite

# Encoding-map layer (advanced but still friendly)
# This is the explicit bridge between "just call encode" and
# "manipulate/compress/refine an encoding poset in a controlled way".
using .Encoding: EncodingMap,
                              UptightEncoding,
                              build_uptight_encoding_from_fringe,
                              pullback_fringe_along_encoding,
                              pushforward_fringe_along_encoding



# JSON IO and cache/interoperability helpers
# These help users reproduce results, checkpoint expensive computations, and
# move between sessions/machines/notebooks cleanly.
using .Serialization: save_flange_json, load_flange_json,
                      save_encoding_json, load_encoding_json,
                      parse_flange_json, flange_from_m2,
                      save_mpp_decomposition_json, load_mpp_decomposition_json,
                      save_mpp_image_json, load_mpp_image_json,
                      TAMER_FEATURE_SCHEMA_VERSION,
                      feature_schema_header, validate_feature_metadata_schema,
                      save_dataset_json, load_dataset_json,
                      save_pipeline_json, load_pipeline_json,
                      load_gudhi_json, load_ripserer_json, load_eirene_json,
                      load_gudhi_txt, load_ripserer_txt, load_eirene_txt,
                      load_ripser_point_cloud, load_ripser_distance,
                      load_ripser_lower_distance, load_ripser_upper_distance,
                      load_ripser_sparse_triplet, load_ripser_binary_lower_distance,
                      load_dipha_distance_matrix,
                      load_boundary_complex_json, load_reduced_complex_json,
                      load_pmodule_json,
                      load_ripser_lower_distance_streaming

using .CoreModules: PointCloud, ImageNd, GraphData, EmbeddedPlanarGraph2D, GradedComplex,
                    FiltrationSpec, GridEncodingMap, grid_index
using .Workflow: poset_from_axes, fringe_presentation, flange_presentation,
    encode, coarsen, resolve, hom, ext, tor, ext_algebra, invariant, invariants,
    rhom, derived_tensor, hyperext, hypertor,
    rank_invariant, restricted_hilbert, euler_surface,
    slice_barcode, slice_barcodes, matching_distance,
    mp_landscape, mpp_decomposition, mpp_image,
    AbstractFeaturizerSpec, PersistenceImageSpec, LandscapeSpec, EulerSurfaceSpec,
    RankGridSpec, SlicedBarcodeSpec, SignedBarcodeImageSpec, ProjectedDistancesSpec,
    CompositeSpec, AbstractInvariantCache, ModuleInvariantCache, RestrictedHilbertInvariantCache, EncodingInvariantCache,
    BatchOptions,
    ExperimentIOConfig, ExperimentSpec, ExperimentArtifact, ExperimentResult,
    LoadedExperimentArtifact, LoadedExperimentResult,
    FeatureSet, FeatureSetWideTable, FeatureSetLongTable,
    EulerSurfaceLongTable, PersistenceImageLongTable, MPLandscapeLongTable, PointSignedMeasureLongTable,
    feature_table, euler_surface_table, persistence_image_table, mp_landscape_table, point_signed_measure_table,
    feature_names, feature_axes, nfeatures, supports, build_cache, cache_stats, transform, transform!, featurize,
    batch_transform, batch_transform!,
    mp_landscape_kernel_object, projected_kernel_object, mpp_image_kernel_object,
    point_signed_measure_kernel_object, rectangle_signed_barcode_kernel_object,
    matching_distance_metric, mp_landscape_distance_metric, projected_distance_metric,
    bottleneck_distance_metric, wasserstein_distance_metric, mpp_image_distance_metric,
    feature_metadata, default_feature_metadata_path, save_metadata_json, load_metadata_json,
    spec_from_metadata, load_spec_with_resolver, invariant_options_from_metadata,
    save_features, load_features,
    save_features_arrow, load_features_arrow, save_features_parquet, load_features_parquet,
    save_features_npz, load_features_npz, save_features_csv, load_features_csv,
    nsamples, asrowmajor, ascolmajor,
    TAMER_EXPERIMENT_SCHEMA_VERSION, run_experiment, load_experiment

# Tables are often the default thing users want to see for resolutions.
using .DerivedFunctors: betti_table, bass_table, HomSystemCache, clear_hom_system_cache!

using .Modules: PModule, PMorphism, ModuleOptions

using .ModuleComplexes: ModuleCochainComplex

using .DerivedFunctors.GradedSpaces: degree_range, dim, basis, coordinates, representative

using .Invariants: slice_chain, slice_chain_exact_2d, SliceSpec, collect_slices, save_slices_json, load_slices_json

# -----------------------------------------------------------------------------
# Stable public exports
# -----------------------------------------------------------------------------

export 
       # Core numeric types
       QQ,

       # Input presentations
       Flange, PLFringe, BoxUpset, BoxDownset,

       # Core algebraic data types
       AbstractPoset, FinitePoset, ProductOfChainsPoset, GridPoset, ProductPoset,
       FiniteFringeOptions,
       PModule, PMorphism, ModuleOptions, ModuleCochainComplex,

       # Results and options
       EncodingResult, ResolutionResult, InvariantResult, unwrap,
       EncodingOptions, ResolutionOptions, DerivedFunctorOptions, InvariantOptions,
       ResolutionCache, clear_resolution_cache!,
       SessionCache, EncodingCache, ModuleCache,
       clear_session_cache!, clear_encoding_cache!, clear_module_cache!,
       HomSystemCache, clear_hom_system_cache!,

       # Narrative workflow entrypoints
       encode, coarsen, resolve, hom, ext, tor, ext_algebra, invariant, invariants,
       slice_chain, slice_chain_exact_2d,
       SliceSpec, collect_slices, save_slices_json, load_slices_json,

       # Accessors (EncodingResult)
       poset, pmodule, classifier, backend, presentation,

       # Field coercion
       change_field,

       # Core finite-poset data types + basic constructors
       Upset, Downset, FringeModule, principal_upset, principal_downset,
       upset_from_generators, downset_from_generators, upset_closure, downset_closure,
       one_by_one_fringe, cover_edges,
       nvertices, leq, leq_matrix, upset_indices, downset_indices, upset_iter, downset_iter, leq_row, leq_col,
       poset_equal, poset_equal_opposite,

       # Encoding-map layer (controlled compression/refinement of encodings)
       EncodingMap, UptightEncoding, build_uptight_encoding_from_fringe,
       pullback_fringe_along_encoding, pushforward_fringe_along_encoding,
       CompiledEncoding, compile_encoding, encoding_map,

       # JSON IO / caching helpers
       save_flange_json, load_flange_json, save_encoding_json, load_encoding_json, save_mpp_decomposition_json, 
       load_mpp_decomposition_json, save_mpp_image_json, load_mpp_image_json, parse_flange_json, flange_from_m2,
       save_dataset_json, load_dataset_json, save_pipeline_json, load_pipeline_json,
       load_gudhi_json, load_ripserer_json, load_eirene_json,
       load_gudhi_txt, load_ripserer_txt, load_eirene_txt,
       load_ripser_point_cloud, load_ripser_distance,
       load_ripser_lower_distance, load_ripser_upper_distance,
       load_ripser_sparse_triplet, load_ripser_binary_lower_distance,
       load_dipha_distance_matrix,
       load_boundary_complex_json, load_reduced_complex_json, load_pmodule_json,
       load_ripser_lower_distance_streaming,

# Data ingestion layer (datasets + filtration specs)
       PointCloud, ImageNd, GraphData, EmbeddedPlanarGraph2D, GradedComplex,
       FiltrationSpec, GridEncodingMap, poset_from_axes, grid_index, axes_from_encoding,
       encode_from_data, ingest, fringe_presentation, flange_presentation,

       # Zn helpers
       face,

       # Homological algebra helpers (ResolutionResult)
       resolution, betti, minimality_report, is_minimal,

       # Complex-level entrypoints
       rhom, derived_tensor, hyperext, hypertor,

       # Graded-space query interface
       degree_range, dim, basis, coordinates, representative,

       # Resolution summaries (common user-facing "tables")
       betti_table, bass_table,

       # Curated invariants
       rank_invariant, restricted_hilbert, euler_surface,
       slice_barcode, slice_barcodes, matching_distance,
       mp_landscape, mpp_decomposition, mpp_image,

       # Statistics transition: typed featurizers + batch featurization
       AbstractFeaturizerSpec, PersistenceImageSpec, LandscapeSpec, EulerSurfaceSpec,
       RankGridSpec, SlicedBarcodeSpec, SignedBarcodeImageSpec, ProjectedDistancesSpec,
       CompositeSpec, AbstractInvariantCache, ModuleInvariantCache, RestrictedHilbertInvariantCache, EncodingInvariantCache,
       BatchOptions,
       ExperimentIOConfig, ExperimentSpec, ExperimentArtifact, ExperimentResult,
       LoadedExperimentArtifact, LoadedExperimentResult,
       FeatureSet, FeatureSetWideTable, FeatureSetLongTable,
       EulerSurfaceLongTable, PersistenceImageLongTable, MPLandscapeLongTable, PointSignedMeasureLongTable,
       feature_table, euler_surface_table, persistence_image_table, mp_landscape_table, point_signed_measure_table,
       TAMER_EXPERIMENT_SCHEMA_VERSION, run_experiment, load_experiment,
       TAMER_FEATURE_SCHEMA_VERSION, feature_names, feature_axes, nfeatures, supports, build_cache, cache_stats,
       transform, transform!, featurize, batch_transform, batch_transform!,
       mp_landscape_kernel_object, projected_kernel_object, mpp_image_kernel_object,
       point_signed_measure_kernel_object, rectangle_signed_barcode_kernel_object,
       matching_distance_metric, mp_landscape_distance_metric, projected_distance_metric,
       bottleneck_distance_metric, wasserstein_distance_metric, mpp_image_distance_metric,
       feature_schema_header, validate_feature_metadata_schema,
       feature_metadata, default_feature_metadata_path, save_metadata_json, load_metadata_json,
       spec_from_metadata, load_spec_with_resolver, invariant_options_from_metadata,
       save_features, load_features,
       save_features_arrow, load_features_arrow, save_features_parquet, load_features_parquet,
       save_features_npz, load_features_npz, save_features_csv, load_features_csv,
       nsamples, asrowmajor, ascolmajor,

       # Backend introspection
       has_polyhedra_backend, available_pl_backends, supports_pl_backend, choose_pl_backend

# =============================================================================
# AdvancedAPI.jl
#
# Opt-in broad surface for power users.
#
# The stable public surface lives in PublicAPI.jl and is exported by the top
# level `PosetModules` module.
#
# This file defines the nested module `PosetModules.Advanced`, intended for
# interactive and research use where convenience matters more than a small,
# curated namespace.
#
# Design goals:
# - `using PosetModules` stays clean: it exports only PublicAPI.jl.
# - `using PosetModules.Advanced` provides a "flat" namespace for most library
#   functionality, plus it always exports the major submodules so nothing is
#   hidden behind export decisions in internal files.
#
# Implementation strategy:
# - Always export the major submodules (CoreModules, DerivedFunctors, Invariants, ...).
# - Re-export ALL names exported by PosetModules itself (so Advanced is a strict
#   superset of the stable public API).
# - Then additionally lift a large set of package-owned symbols from the major
#   submodules into the Advanced namespace. This lifting:
#     * does NOT depend on submodule export lists,
#     * avoids pulling in external symbols (Base/LinearAlgebra/etc),
#     * is "first come wins" on collisions, with full access still available by
#       qualification via the submodule name.
# =============================================================================

module Advanced

# Root package module (PosetModules)
const _ROOT = parentmodule(@__MODULE__)

# -----------------------------------------------------------------------------
# Always expose the major submodules (users can always qualify).
# -----------------------------------------------------------------------------

import ..CoreModules
import ..RegionGeometry
import ..FiniteFringe
import ..IndicatorTypes
import ..Encoding
import ..Modules
import ..AbelianCategories
import ..IndicatorResolutions
import ..FlangeZn
import ..Serialization
import ..Viz2D
import ..PLPolyhedra
import ..PLBackend
import ..ChainComplexes
import ..ZnEncoding
import ..DerivedFunctors
import ..DerivedFunctors.Resolutions
import ..DerivedFunctors.ExtTorSpaces
import ..DerivedFunctors.Functoriality
import ..DerivedFunctors.Algebras
import ..DerivedFunctors.SpectralSequences
import ..DerivedFunctors.Backends
import ..DerivedFunctors.HomExtEngine
import ..DerivedFunctors.Utils
import ..ModuleComplexes
import ..ChangeOfPosets
import ..Invariants

export CoreModules, Stats, RegionGeometry, FiniteFringe, IndicatorTypes, Encoding,
       Modules, AbelianCategories, IndicatorResolutions, FlangeZn, Serialization, Viz2D,
       PLPolyhedra, PLBackend, ChainComplexes, ZnEncoding, DerivedFunctors,
       Resolutions, ExtTorSpaces, Functoriality, Algebras, SpectralSequences, Backends,
       HomExtEngine, Utils,
       ModuleComplexes, ChangeOfPosets, Invariants,
       idmap

# Convenience lift: identity cochain map for module complexes.
const idmap = ModuleComplexes.idmap

# -----------------------------------------------------------------------------
# Helper predicates for lifting bindings safely.
# -----------------------------------------------------------------------------

# Is module `m` nested under `root`?
function _is_submodule_of(m::Module, root::Module)::Bool
    m === root && return true
    while true
        pm = parentmodule(m)
        pm === m && return false
        pm === root && return true
        m = pm
    end
end

# Only lift identifiers we can safely bind as `const name = ...`.
# This excludes macros and compiler-generated names.
function _liftable_symbol(sym::Symbol)::Bool
    Base.isidentifier(sym) || return false
    s = String(sym)
    isempty(s) && return false
    s[1] == '#' && return false
    s[1] == '@' && return false
    return true
end

# True if `obj` "belongs" to PosetModules (by parentmodule chain).
function _is_internal_binding(obj)::Bool
    try
        pm = parentmodule(obj)
        return _is_submodule_of(pm, _ROOT)
    catch
        return false
    end
end

# Bind `sym` from module `M` into Advanced and export it.
function _bind_and_export!(M::Module, sym::Symbol)
    isdefined(@__MODULE__, sym) && return
    isdefined(M, sym) || return
    # Avoid relying on dotted syntax with interpolated module objects.
    @eval const $(sym) = getfield($M, $(QuoteNode(sym)))
    @eval export $(sym)
    return
end

# -----------------------------------------------------------------------------
# 1) Advanced is a strict superset of the stable public API:
#    lift and export everything exported by PosetModules itself.
# -----------------------------------------------------------------------------
for sym in names(_ROOT; all=false)
    _liftable_symbol(sym) || continue
    _bind_and_export!(_ROOT, sym)
end

# -----------------------------------------------------------------------------
# 2) Bulk lift package-owned symbols from the major internal modules.
#
# Rule:
# - If a name is "owned" by the module (not imported), lift it unconditionally.
#   This includes aliases like `QQ` even if the underlying object is from Base.
# - If a name is imported, lift it only if its parentmodule is inside PosetModules.
#   This avoids polluting Advanced with LinearAlgebra/Base names.
# -----------------------------------------------------------------------------

const _ADVANCED_MODULES = Module[
    _ROOT,
    CoreModules,
    RegionGeometry,
    FiniteFringe,
    IndicatorTypes,
    Modules,
    AbelianCategories,
    Encoding,
    IndicatorResolutions,
    FlangeZn,
    Serialization,
    Viz2D,
    PLPolyhedra,
    PLBackend,
    ChainComplexes,
    ZnEncoding,
    DerivedFunctors,
    DerivedFunctors.Resolutions,
    DerivedFunctors.ExtTorSpaces,
    DerivedFunctors.Functoriality,
    DerivedFunctors.Algebras,
    DerivedFunctors.SpectralSequences,
    DerivedFunctors.Backends,
    DerivedFunctors.HomExtEngine,
    DerivedFunctors.Utils,
    ModuleComplexes,
    ChangeOfPosets,
    Invariants,
]

for M in _ADVANCED_MODULES
    _owned = Set(names(M; all=true, imported=false))
    for sym in names(M; all=true, imported=true)
        _liftable_symbol(sym) || continue
        isdefined(@__MODULE__, sym) && continue
        isdefined(M, sym) || continue

        # Only lift imported names if they are package-owned.
        (sym in _owned) || _is_internal_binding(getfield(M, sym)) || continue

        _bind_and_export!(M, sym)
    end
end

end # module Advanced

end # module
