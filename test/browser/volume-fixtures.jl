# Unequal axes and two independently fixed dimensions detect display transposes.
function _volume_browser_app(tick)
    B.App(; title="Array slice controls") do client
        volume = [1000i + 100j + 10k + l for i in 1:2, j in 1:3, k in 1:4, l in 1:2]
        viewer = V.visualize(TamerOp.DataTypes.ImageNd(volume); kind=:slice_viewer,
            view_dims=(3,1), slice_indices=Dict(2=>1,4=>2), backend=:wglmakie,
            colorrange=(1111,2342), size=(640,440))
        ext = Base.get_extension(TamerOp, :TamerOpWGLMakieExt)
        ui = ext._inspection_ui(viewer)
        M.update_state_before_display!(ui.figure)
        telemetry = map(client, tick) do _
            matrix = ui.heatmap[3][]
            JSON3.write((; closed=ui.closed[], fixed=Dict(string(k)=>v for (k,v) in ui.fixed_indices),
                rows=[[matrix[i,j] for j in axes(matrix,2)] for i in axes(matrix,1)],
                xlabel=ui.axis.xlabel[], ylabel=ui.axis.ylabel[],
                retained_source=ui.source[] !== nothing, observers=length(ui.observers),
                viewer_count=count(handle -> handle.session === nothing, values(ext._INSPECTION_UI_HANDLES))))
        end
        B.DOM.div(viewer, B.DOM.pre(telemetry; var"data-testid"="volume-state", style="display:none"))
    end
end
