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
include("visualization_wgl_matchings.jl")

function _render_wgl_visual(spec; kwargs...)
    WGLMakie.activate!(; use_html_widgets=true)
    if spec.kind === :linked_inspector
        return spec.metadata.session isa VIZ.MatchingInspectionSession ? _render_matching_inspector(spec;kwargs...) : spec.metadata.session isa VIZ.IntervalInspectionSession ?
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
        app = WGLMakie.Bonito.App(fig)
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
