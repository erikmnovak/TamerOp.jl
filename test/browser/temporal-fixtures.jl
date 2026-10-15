include("../fixtures/temporal.jl")
function _temporal_app(sessions,tick)
    instance=Ref{Any}(nothing)
    B.App(;title="Temporal persistence") do client
        if instance[]===nothing || V.inspection_summary(instance[]).closed
            s=V.temporal_explorer(temporal_cloud_fixture();occurrence=3,scale_index=3,interval=2)
            instance[]=_observe_fixture!(sessions,s,tick)
        end
        s=instance[];viewer=V.visualize(s;backend=:wglmakie)
        ext=Base.get_extension(TamerOp,:TamerOpWGLMakieExt);ui=ext._inspection_ui(viewer)
        telemetry=map(client,tick) do _
            snap=V.inspection_snapshot(s);sel=V.inspection_selection(s)
            targets=NamedTuple[];pointer=nothing
            if !ui.closed[]
                fig=ui.figure[];width,height=M.widths(M.viewport(fig.scene)[])
                pointer=(;pixels=Tuple(M.events(fig.scene).mouseposition[]),
                    axes=[(;index=i,data=Tuple(M.mouseposition(ax.scene)),inside=M.is_mouseinside(ax.scene))
                          for (i,ax) in enumerate(ui.chart_axes[]) if ax isa M.Axis])
                for t in ui.targets[]
                    ax=ui.chart_axes[][t.axis_index];ax isa M.Axis || continue
                    pixel=M.project(ax.scene,M.Point2d(t.point))+minimum(M.viewport(ax.scene)[])
                    push!(targets,merge(t,(;x=Float64(pixel[1]/width),y=Float64(1-pixel[2]/height))))
                end
            end
            JSON3.write(_json_value((;summary=V.inspection_summary(s),selection=sel,
                directions=snap.metadata.directions,times=snap.metadata.times,
                source_cycle=snap.metadata.source_cycle,landscape=snap.metadata.landscape,
                targets,pointer,ui=(;closed=ui.closed[],error=ui.error_text[],hover=ui.hover_text[],rebuilds=ui.rebuild_count[],updates=ui.update_count[]),
                viewer_count=count(u->u.session===s,values(ext._INSPECTION_UI_HANDLES)))))
        end
        B.DOM.div(viewer,B.DOM.pre(telemetry;var"data-testid"="julia-session-state",style="display:none"))
    end
end
