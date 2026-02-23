# APISurface.jl
#
# Canonical symbol contracts for the two curated public API tiers:
# - SIMPLE_API: default ergonomic surface for most users.
# - ADVANCED_API: strict superset for power users.
#
# This file is intentionally static and explicit. Do not auto-generate these
# lists from module exports.

const SIMPLE_API = (
    :QQ,
    :PointCloud, :ImageNd, :GraphData, :EmbeddedPlanarGraph2D,
    :GradedComplex, :MultiCriticalGradedComplex,
    :FiltrationSpec, :ConstructionBudget, :ConstructionOptions, :PipelineOptions,
    :encode, :coarsen, :resolve, :hom_dimension, :hom, :ext, :tor, :rhom, :derived_tensor, :hyperext, :hypertor, :ext_algebra, :invariant, :invariants,
    :rank_invariant, :restricted_hilbert, :euler_surface,
    :slice_barcode, :slice_barcodes, :matching_distance,
    :mp_landscape, :mpp_decomposition, :mpp_image,
    :AbstractFeaturizerSpec,
    :PersistenceImageSpec, :LandscapeSpec, :EulerSurfaceSpec, :RankGridSpec,
    :SlicedBarcodeSpec, :SignedBarcodeImageSpec, :ProjectedDistancesSpec, :CompositeSpec,
    :FeatureSet, :BatchOptions, :run_experiment, :load_experiment,
    :feature_names, :feature_axes, :nfeatures,
    :build_cache, :cache_stats,
    :transform, :transform!, :featurize, :batch_transform, :batch_transform!,
    :save_features, :load_features,
    :save_features_csv, :load_features_csv,
    :save_features_arrow, :load_features_arrow,
    :save_features_parquet, :load_features_parquet,
    :save_features_npz, :load_features_npz,
    :save_dataset_json, :load_dataset_json,
    :save_pipeline_json, :load_pipeline_json,
    :SessionCache,
    :EncodingResult, :CohomologyDimsResult, :ResolutionResult, :InvariantResult, :unwrap,
    :change_field,
    :one_criticalify, :criticality, :normalize_multicritical,
)

const SIMPLE_API_BINDINGS = (
    (:CoreModules, (
        :QQ,
        :PointCloud, :ImageNd, :GraphData, :EmbeddedPlanarGraph2D,
        :GradedComplex, :MultiCriticalGradedComplex,
        :FiltrationSpec, :ConstructionBudget, :ConstructionOptions, :PipelineOptions,
        :SessionCache,
        :EncodingResult, :CohomologyDimsResult, :ResolutionResult, :InvariantResult, :unwrap,
        :change_field,
    )),
    (:Workflow, (
        :encode, :coarsen, :resolve, :hom_dimension, :hom, :ext, :tor, :rhom, :derived_tensor, :hyperext, :hypertor, :ext_algebra, :invariant, :invariants,
        :rank_invariant, :restricted_hilbert, :euler_surface,
        :slice_barcode, :slice_barcodes,
        :mp_landscape, :mpp_decomposition, :mpp_image,
    )),
    (:DataIngestion, (
        :one_criticalify, :criticality, :normalize_multicritical,
    )),
    (:Featurizers, (
        :matching_distance,
        :AbstractFeaturizerSpec,
        :PersistenceImageSpec, :LandscapeSpec, :EulerSurfaceSpec, :RankGridSpec,
        :SlicedBarcodeSpec, :SignedBarcodeImageSpec, :ProjectedDistancesSpec, :CompositeSpec,
        :FeatureSet, :BatchOptions, :run_experiment, :load_experiment,
        :feature_names, :feature_axes, :nfeatures,
        :build_cache, :cache_stats,
        :transform, :transform!, :featurize, :batch_transform, :batch_transform!,
        :save_features, :load_features,
        :save_features_csv, :load_features_csv,
        :save_features_arrow, :load_features_arrow,
        :save_features_parquet, :load_features_parquet,
        :save_features_npz, :load_features_npz,
    )),
    (:Serialization, (
        :save_dataset_json, :load_dataset_json,
        :save_pipeline_json, :load_pipeline_json,
    )),
)

const ADVANCED_ONLY_API = (
    :PModule, :PMorphism, :ModuleOptions,
    :EncodingOptions, :ResolutionOptions, :InvariantOptions, :DerivedFunctorOptions,
    :FinitePoset, :ProductOfChainsPoset, :GridPoset, :ProductPoset,
    :Upset, :Downset, :FringeModule,
    :principal_upset, :principal_downset,
    :upset_from_generators, :downset_from_generators,
    :one_by_one_fringe, :cover_edges,
    :nvertices, :leq, :leq_matrix,
    :pmodule_from_fringe, :projective_cover, :injective_hull,
    :upset_resolution, :downset_resolution, :indicator_resolutions,
    :verify_upset_resolution, :verify_downset_resolution,
    :EncodingMap, :UptightEncoding,
    :fringe_presentation, :flange_presentation,
    :Flange, :PLFringe, :BoxUpset, :BoxDownset, :face,
    :ModuleCochainComplex, :ModuleCochainMap, :ModuleCochainHomotopy, :ModuleDistinguishedTriangle,
    :mapping_cone, :mapping_cone_triangle, :cohomology_module, :cohomology_module_data,
    :induced_map_on_cohomology_modules, :is_quasi_isomorphism,
    :RHomComplex, :RHom, :HyperExtSpace, :hyperExt,
    :DerivedTensorComplex, :DerivedTensor, :HyperTorSpace, :hyperTor,
    :rhom_map_first, :rhom_map_second,
    :hyperExt_map_first, :hyperExt_map_second,
    :derived_tensor_map_first, :derived_tensor_map_second,
    :hyperTor_map_first, :hyperTor_map_second,
    :betti_table, :bass_table,
    :dim, :degree_range, :basis, :representative, :coordinates,
    :HomSpace, :ExtSpace, :ExtInjective, :TorSpace,
    :ext_map_first, :ext_map_second, :tor_map_first, :tor_map_second,
    :ExtLongExactSequenceFirst, :ExtLongExactSequenceSecond,
    :TorLongExactSequenceFirst, :TorLongExactSequenceSecond,
    :ExtDoubleComplex, :ExtSpectralSequence, :TorDoubleComplex, :TorSpectralSequence,
    :yoneda_product,
    :slice_chain, :slice_chain_exact_2d, :SliceSpec, :collect_slices, :save_slices_json, :load_slices_json,
    :direct_sum, :direct_sum_with_maps, :zero_pmodule, :zero_morphism,
    :map_leq, :map_leq_many, :map_leq_many!,
    :kernel, :cokernel, :image,
    :kernel_with_inclusion, :image_with_inclusion,
    :cokernel_with_projection, :coimage_with_projection,
    :submodule, :image_submodule, :quotient, :coimage,
    :is_zero_morphism,
    :pushout, :pullback,
    :ShortExactSequence, :short_exact_sequence, :is_exact, :snake_lemma,
    :biproduct, :product, :coproduct, :equalizer, :coequalizer,
    :DiscretePairDiagram, :ParallelPairDiagram, :SpanDiagram, :CospanDiagram,
    :limit, :colimit,
    :pushforward_left, :pushforward_right, :restriction, :product_poset, :encode_pmodules_to_common_poset,
    :left_kan_extension, :right_kan_extension, :derived_pushforward_left, :derived_pushforward_right,
    :projective_resolution, :injective_resolution,
    :Hom, :Ext, :Tor, :ExtAlgebra,
    :save_flange_json, :load_flange_json,
    :IngestionPlan, :plan_ingestion, :run_ingestion,
)

const ADVANCED_ONLY_API_BINDINGS = (
    (:Modules, (
        :PModule, :PMorphism, :ModuleOptions,
        :direct_sum, :direct_sum_with_maps, :zero_pmodule, :zero_morphism,
        :map_leq, :map_leq_many, :map_leq_many!,
    )),
    (:CoreModules, (
        :EncodingOptions, :ResolutionOptions, :InvariantOptions, :DerivedFunctorOptions,
    )),
    (:ModuleComplexes, (
        :ModuleCochainComplex, :ModuleCochainMap, :ModuleCochainHomotopy, :ModuleDistinguishedTriangle,
        :mapping_cone, :mapping_cone_triangle, :cohomology_module, :cohomology_module_data,
        :induced_map_on_cohomology_modules, :is_quasi_isomorphism,
        :RHomComplex, :RHom, :HyperExtSpace, :hyperExt,
        :DerivedTensorComplex, :DerivedTensor, :HyperTorSpace, :hyperTor,
        :rhom_map_first, :rhom_map_second,
        :hyperExt_map_first, :hyperExt_map_second,
        :derived_tensor_map_first, :derived_tensor_map_second,
        :hyperTor_map_first, :hyperTor_map_second,
    )),
    (:FiniteFringe, (
        :FinitePoset, :ProductOfChainsPoset, :GridPoset, :ProductPoset,
        :Upset, :Downset, :FringeModule,
        :principal_upset, :principal_downset,
        :upset_from_generators, :downset_from_generators,
        :one_by_one_fringe, :cover_edges,
        :nvertices, :leq, :leq_matrix,
    )),
    (:Encoding, (
        :EncodingMap, :UptightEncoding,
    )),
    (:Workflow, (
        :fringe_presentation, :flange_presentation,
    )),
    (:FlangeZn, (
        :Flange, :face,
    )),
    (:PLPolyhedra, (
        :PLFringe,
    )),
    (:PLBackend, (
        :BoxUpset, :BoxDownset,
    )),
    (:DerivedFunctors, (
        :betti_table, :bass_table,
        :dim, :degree_range, :basis, :representative, :coordinates,
        :HomSpace, :ExtSpace, :ExtInjective, :TorSpace,
        :ext_map_first, :ext_map_second, :tor_map_first, :tor_map_second,
        :ExtLongExactSequenceFirst, :ExtLongExactSequenceSecond,
        :TorLongExactSequenceFirst, :TorLongExactSequenceSecond,
        :ExtDoubleComplex, :ExtSpectralSequence, :TorDoubleComplex, :TorSpectralSequence,
        :yoneda_product,
        :projective_resolution, :injective_resolution,
        :Hom, :Ext, :Tor, :ExtAlgebra,
    )),
    (:IndicatorResolutions, (
        :pmodule_from_fringe, :projective_cover, :injective_hull,
        :upset_resolution, :downset_resolution, :indicator_resolutions,
        :verify_upset_resolution, :verify_downset_resolution,
    )),
    (:DataIngestion, (
        :IngestionPlan, :plan_ingestion, :run_ingestion,
    )),
    (:Invariants, (
        :slice_chain, :slice_chain_exact_2d, :SliceSpec, :collect_slices, :save_slices_json, :load_slices_json,
    )),
    (:AbelianCategories, (
        :kernel, :cokernel, :image,
        :kernel_with_inclusion, :image_with_inclusion,
        :cokernel_with_projection, :coimage_with_projection,
        :submodule, :image_submodule, :quotient, :coimage,
        :is_zero_morphism,
        :pushout, :pullback,
        :ShortExactSequence, :short_exact_sequence, :is_exact, :snake_lemma,
        :biproduct, :product, :coproduct, :equalizer, :coequalizer,
        :DiscretePairDiagram, :ParallelPairDiagram, :SpanDiagram, :CospanDiagram,
        :limit, :colimit,
    )),
    (:ChangeOfPosets, (
        :pushforward_left, :pushforward_right, :restriction, :product_poset, :encode_pmodules_to_common_poset,
        :left_kan_extension, :right_kan_extension, :derived_pushforward_left, :derived_pushforward_right,
    )),
    (:Serialization, (
        :save_flange_json, :load_flange_json,
    )),
)

const ADVANCED_API = (SIMPLE_API..., ADVANCED_ONLY_API...)

function _bind_api_symbol!(target::Module, source::Module, sym::Symbol)
    isdefined(target, sym) && return
    isdefined(source, sym) || error("API surface mismatch: $(source).$(sym) is not defined.")
    Core.eval(target, :(const $(sym) = getfield($(source), $(QuoteNode(sym)))))
    Core.eval(target, :(export $(sym)))
    return
end

function _bind_api_list!(target::Module, source::Module, syms)
    for sym in syms
        _bind_api_symbol!(target, source, sym)
    end
    return
end

function _bind_api_bindings!(target::Module, bindings)
    for (modsym, syms) in bindings
        isdefined(target, modsym) || error("API surface mismatch: module $(target).$(modsym) is not defined.")
        source = getfield(target, modsym)
        source isa Module || error("API surface mismatch: $(target).$(modsym) is not a module.")
        _bind_api_list!(target, source, syms)
    end
    return
end

function _assert_api_list_defined!(target::Module, syms; label::AbstractString="API")
    for sym in syms
        isdefined(target, sym) || error("$(label) mismatch: $(target).$(sym) is not defined.")
    end
    return
end

function _export_api_symbol!(target::Module, sym::Symbol)
    isdefined(target, sym) || error("Export contract mismatch: $(target).$(sym) is not defined.")
    Core.eval(target, :(export $(sym)))
    return
end

function _export_api_list!(target::Module, syms)
    for sym in syms
        _export_api_symbol!(target, sym)
    end
    return
end
