include("../fixtures/density_clustering.jl")
function _clustering_browser_app(sessions,tick;ties=false)
    instance=Ref{Any}(nothing)
    B.App(;title="Density clustering") do client
        if instance[]===nothing
            s=ties ? V.clustering_explorer(TamerOp.PointCloud([0. 0.;0. 0.;1. 0.;1. 1.]);max_diameter=2.,diameter=0.,neighbors=1) : _a129_session()
            instance[]=_observe_fixture!(sessions,s,tick)
        end
        s=instance[];viewer=V.visualize(s;backend=:wglmakie)
        ext=Base.get_extension(TamerOp,:TamerOpWGLMakieExt);ui=ext._inspection_ui(viewer)
        telemetry=map(client,tick) do _
            summary=V.inspection_summary(s);snap=V.inspection_snapshot(s);targets=NamedTuple[]
            if !ui.closed[]
                fig=ui.figure[];M.update_state_before_display!(fig);w,h=M.widths(M.viewport(fig.scene)[])
                for t in ui.targets[]
                    q=M.project(ui.chart_axes[][t.axis_index].scene,M.Point2d(t.point))+minimum(M.viewport(ui.chart_axes[][t.axis_index].scene)[])
                    push!(targets,(;t.selection,t.label,x=Float64(q[1]/w),y=Float64(1-q[2]/h)))
                end
            end
            JSON3.write(_json_value((;fixture=:clustering,summary,selection=summary.selection,labels=snap.metadata.labels,
                clusters=snap.metadata.clusters,gaps=snap.metadata.gaps,
                ui=(;closed=ui.closed[],error=ui.error_text[],pick_targets=targets,rebuild_count=ui.rebuild_count[],update_count=ui.update_count[],
                    observer_count=length(ui.observers)+length(ui.pointer_observers)),
                viewer_count=count(u->u.session===s,values(ext._INSPECTION_UI_HANDLES)))))
        end
        B.DOM.div(viewer,B.DOM.pre(telemetry;var"data-testid"="julia-session-state",style="display:none"))
    end
end
