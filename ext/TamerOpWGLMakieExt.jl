module TamerOpWGLMakieExt

using WGLMakie
import TamerOp

const TO = TamerOp

include("visualization_makie_common.jl")

const _HANDLERS = _visual_makie_handlers(TO, WGLMakie; allow_save=false)

const VIZ = TO.Visualization
const _BASE_WGL_RENDER = _HANDLERS.render

include("visualization_wgl_inspection.jl")
include("visualization_wgl_intervals.jl")
include("visualization_wgl_volume.jl")
include("visualization_wgl_matchings.jl")
include("visualization_wgl_spectral.jl")
include("visualization_wgl_transport.jl")
include("visualization_wgl_chains.jl")
include("visualization_wgl_components.jl")
include("visualization_wgl_comparisons.jl")
include("visualization_wgl_collections.jl")
include("visualization_wgl_ensembles.jl")
include("visualization_wgl_complexes.jl")
include("visualization_wgl_signed.jl")
include("visualization_wgl_editor.jl")
include("visualization_wgl_generalized_rank.jl")
include("visualization_wgl_fields.jl")

function _render_wgl_visual(spec; kwargs...)
    WGLMakie.activate!(; use_html_widgets=true)
    if spec.kind === :slice_viewer && get(spec.interaction, :notebook, nothing) === :widget_viewer
        return _render_volume_viewer(spec; kwargs...)
    end
    if spec.kind === :linked_inspector
        spec.metadata.session isa VIZ.EnsembleInspectionSession && return _render_ensemble_inspector(spec;kwargs...)
        spec.metadata.session isa VIZ.FieldInspectionSession && return _render_field_inspector(spec;kwargs...)
        spec.metadata.session isa VIZ.GeneralizedRankInspectionSession && return _render_generalized_rank_inspector(spec;kwargs...)
        spec.metadata.session isa VIZ.ModuleEditorSession && return _render_module_editor(spec;kwargs...)
        spec.metadata.session isa VIZ.CollectionInspectionSession && return _render_collection_inspector(spec;kwargs...)
        spec.metadata.session isa VIZ.ComparisonInspectionSession && return _render_comparison_inspector(spec;kwargs...)
        spec.metadata.session isa VIZ.TransportInspectionSession && return _render_transport_inspector(spec;kwargs...)
        return spec.metadata.session isa VIZ.ComponentInspectionSession ? _render_component_inspector(spec;kwargs...) : spec.metadata.session isa VIZ.ChainInspectionSession ? _render_chain_inspector(spec;kwargs...) : spec.metadata.session isa VIZ.ComplexInspectionSession ? _render_complex_inspector(spec;kwargs...) : spec.metadata.session isa VIZ.SpectralInspectionSession ? _render_spectral_inspector(spec;kwargs...) : spec.metadata.session isa VIZ.SignedInspectionSession ? _render_signed_inspector(spec;kwargs...) : spec.metadata.session isa VIZ.MatchingInspectionSession ? _render_matching_inspector(spec;kwargs...) : spec.metadata.session isa VIZ.IntervalInspectionSession ?
            _render_interval_inspector(spec; kwargs...) : _render_linked_inspector(spec; kwargs...)
    end
    return Base.invokelatest(_BASE_WGL_RENDER, spec; kwargs...)
end

function _save_wgl_visual(path::AbstractString, spec; kwargs...)
    spec isa VIZ.VisualizationSpec || throw(ArgumentError("save_spec expected a VisualizationSpec, got $(typeof(spec))."))
    spec.kind === :linked_inspector && throw(ArgumentError("A linked inspector requires live Julia callbacks. Export inspection_snapshot(session) as a static figure instead."))
    WGLMakie.activate!(; use_html_widgets=true)
    fig = Base.invokelatest(_BASE_WGL_RENDER, spec; kwargs...)
    ext = lowercase(splitext(path)[2])
    if ext == ".html"
        app = WGLMakie.Bonito.App() do client
            dom = WGLMakie.Bonito.DOM.div(fig)
            _inspection_keyboard_navigation!(client, dom)
            dom
        end
        WGLMakie.Bonito.export_static(path, app)
    else
        WGLMakie.save(path, fig)
    end
    return path
end

function __init__()
    VIZ._register_visual_backend!(:wglmakie;
                                  render=_render_wgl_visual,
                                  save=_save_wgl_visual)
    return nothing
end

end # module TamerOpWGLMakieExt
