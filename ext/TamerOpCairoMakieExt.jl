module TamerOpCairoMakieExt

using CairoMakie
import TamerOp

const TO = TamerOp

include("visualization_makie_common.jl")
const _HANDLERS = _visual_makie_handlers(TO, CairoMakie)

function __init__()
    TO.Visualization._register_visual_backend!(:cairomakie;
        render=(spec; kwargs...) -> begin
            CairoMakie.activate!()
            _HANDLERS.render(spec; kwargs...)
        end,
        save=(path, spec; kwargs...) -> begin
            CairoMakie.activate!()
            _HANDLERS.save(path, spec; kwargs...)
        end)
    return nothing
end

end # module TamerOpCairoMakieExt
