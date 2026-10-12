"""
TamerOp.Visualization

Backend-agnostic visualization owner for encoding and invariant objects.

Design
------
- Core visualization data lives here as inspectable `VisualizationSpec` objects.
- Rendering backends live in `ext/` and register when explicitly imported.
- The canonical simple-surface workflow is:

  `visualize(obj)` -> backend-native figure/widget
  `save_visual(outdir, stem, obj)` -> choose an export target automatically
  `save_visuals(outdir, requests)` -> batch export without backend bookkeeping
  `available_visuals(obj)` -> supported visualization kinds

- The advanced exact-control export path remains available as:

  `save_visual(path, obj)` -> write to an explicit filename/extension

Cheap-first workflow
--------------------
- start with `available_visuals(obj)` to see the supported kinds,
- build an inspectable spec with `visual_spec(obj; kind=...)`,
- inspect it via `describe(spec)` / `visual_summary(spec)`,
- validate via `check_visual_spec(spec)` when you hand-build a spec,
- export with `save_visual(outdir, stem, obj)` or `save_visuals(outdir, requests)`
  when you want files without manual backend bookkeeping,
- use `save_visual(path, obj)` only when you want exact filename/backend control,
- render with `visualize(obj)` or `render(spec)` only when you actually want
  a backend figure.

This owner intentionally keeps rendering out of `src/` so notebook and script
UX can stay stable even when no Makie backend is installed.
"""
module Visualization

using Statistics
using Base64, SHA, JSON3
import ..Serialization
import ..Featurizers
using SparseArrays

import ..ChainComplexes: describe, cohomology_dims, dimensions
import ..ChainComplexes
import ..FiniteFringe
import ..Modules
import ..AbelianCategories
import ..DerivedFunctors
import ..ModuleComplexes
import ..Results
import ..CoreModules
import ..FieldLinAlg
import ..Invariants
import ..SignedMeasures
import ..InvariantCore
import ..IndicatorResolutions
import ..IndicatorTypes
using ..ExactReals: AlgebraicReal
using ..FiniteFringe: AbstractPoset, ProductOfChainsPoset, GridPoset, ProductPoset, nvertices, leq
import ..DataTypes
import ..DataIngestion
import ..OrdinaryPersistence
import ..Fibered2D
using ..EncodingCore: AbstractPLikeEncodingMap, CompiledEncoding, GridEncodingMap,
                      encoding_axes, encoding_map,
                      encoding_representatives, locate
using ..Encoding: EncodingMap, source_poset, target_poset, region_map
using ..Results: EncodingResult, CohomologyDimsResult, ModuleTranslationResult, InvariantResult,
                 invariant_value, translation_map, translation_kind, source_result
import ..ChangeOfPosets
using ..ChangeOfPosets: CommonRefinementTranslationResult, common_poset, projection_maps
using ..FlangeZn: Flange, active_flats, active_injectives, flats, injectives
using ..FieldLinAlg: rank_restricted
using ..InvariantCore: PackedBarcode
using ..Invariants: RankInvariantResult, value_at, source_poset, check_rank_query_points
using ..PLBackend: PLEncodingMapBoxes
using ..PLPolyhedra: PLEncodingMap
using ..ZnEncoding: ZnEncodingMap, critical_coordinates
using ..SliceInvariants: SliceBarcodesResult, slice_barcodes, slice_weights, slice_directions, slice_offsets,
                         bottleneck_distance, bottleneck_matching
using ..Fibered2D: FiberedArrangement2D, FiberedBarcodeCache2D, FiberedSliceFamily2D, FiberedSliceResult,
                   ProjectedArrangement, ProjectedBarcodesResult, ProjectedDistancesResult,
                   source_encoding, working_box, direction_representatives, slope_breaks,
                   ncells, computed_cell_count, source_arrangement, slice_direction,
                   slice_offset_interval, slice_offset, slice_chain_id, fibered_values, stores_values,
                   projections, projection_indices,
                   projection_directions, projected_barcodes, projected_distances, shared_arrangement,
                   fibered_query_summary, backend, fibered_barcode_cache_2d,
                   slice_chain, slice_values, slice_barcode, fibered_slice,
                   fibered_slice_family_2d, _arr2d_cell_offset_interval, _fibered_dir_cell_index, _arr2d_compute_cell!
using ..SignedMeasures: Rect, RectSignedBarcode, PointSignedMeasure, SignedMeasureDecomposition,
                        rectangles, weights, points, total_variation, total_mass,
                        components, component_names, has_rectangles,
                        has_euler_signed_measure, has_mpp_image,
                        euler_signed_measure, mpp_image
using ..MultiparameterImages: MPPLineSpec, MPPDecomposition, MPPImage, MPLandscape, AmbientLandscapeGrid,
                              line_direction, line_basepoint, line_offset, line_omega,
                              line_specs, summand_segments, nsummands, nlines, bounding_box,
                              summand_weights, image_xgrid, image_ygrid, image_values,
                              decomposition, landscape_grid, landscape_values,
                              landscape_layers, slice_weights, slice_directions,
                              slice_offsets, ndirections, noffsets, landscape_slice
include("visualization/style.jl")
include("visualization/types.jl")
include("visualization/interval_views.jl")
include("visualization/validation.jl")
include("visualization/rendering.jl")
include("visualization/region_geometry.jl")
include("visualization/builders_encoding.jl")
include("visualization/builders_invariants.jl")
include("visualization/builders_signed.jl")
include("visualization/hasse_layout.jl")
include("visualization/builders_modules.jl")
include("visualization/builders_presentations.jl")
include("visualization/builders_morphisms.jl")
include("visualization/builders_morphism_comparisons.jl")
include("visualization/builders_transport.jl")
include("visualization/builders_resolutions.jl")
include("visualization/builders_spectral.jl")
include("visualization/builders_rank_sections.jl")
include("visualization/builders_propagation.jl")
include("visualization/builders_slices.jl")
include("visualization/source_previews.jl")
include("visualization/builders_ingestion.jl")
include("visualization/builders_complexes.jl")
include("visualization/builders_source_chains.jl")
include("visualization/builders_chains.jl")
include("visualization/builders_geometric.jl")
include("visualization/builders_derived.jl")
include("visualization/builders_components.jl")
include("visualization/inspection_slices.jl")
include("visualization/inspection_session.jl")
include("visualization/inspection_intervals.jl")
include("visualization/builders_matchings.jl")
include("visualization/inspection_matchings.jl")
include("visualization/inspection_signed.jl")
include("visualization/inspection_spectral.jl")
include("visualization/inspection_complexes.jl")
include("visualization/inspection_chains.jl")
include("visualization/inspection_components.jl")
include("visualization/inspection_transport.jl")
include("visualization/builders_generalized_rank.jl")
include("visualization/inspection_generalized_rank.jl")
include("visualization/builders_fields.jl")
include("visualization/inspection_fields.jl")
include("visualization/inspection_rendering.jl")
include("visualization/composition.jl")
include("visualization/builders_comparisons.jl")
include("visualization/inspection_comparisons.jl")
include("visualization/builders_collections.jl")
include("visualization/inspection_collections.jl")
include("visualization/builders_ensembles.jl")
include("visualization/inspection_ensembles.jl")
include("visualization/module_editor.jl")
include("visualization/session_composition.jl")
include("visualization/session_html.jl")
include("visualization/saved_sessions.jl")
include("visualization/editor_export.jl")

end # module Visualization
