include("../fixtures/stability.jl")
function _stability_browser_app(sessions,tick;landscape=false)
    instance=Ref{Any}(nothing)
    B.App(;title="Stability") do client
        if instance[]===nothing || V.inspection_summary(instance[]).closed
            s=landscape ? V.stability_explorer(_a51_landscape()) : _a51_session()
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
                    project(p)=begin
                        z=M.project(ui.chart_axes[][t.axis_index].scene,M.Point2d(p))+minimum(M.viewport(ui.chart_axes[][t.axis_index].scene)[])
                        (;x=Float64(z[1]/w),y=Float64(1-z[2]/h))
                    end
                    push!(targets,(;t.selection,t.label,x=Float64(q[1]/w),y=Float64(1-q[2]/h),
                        segment=(project(t.segment[1:2]),project(t.segment[3:4]))))
                end
            end
            JSON3.write(_json_value((;fixture=:stability,summary,selection=summary.selection,slopes=snap.metadata.slopes,boundary=snap.metadata.boundary,
                ui=(;closed=ui.closed[],error=ui.error_text[],pick_targets=targets,rebuild_count=ui.rebuild_count[],update_count=ui.update_count[],
                    observer_count=length(ui.observers)+length(ui.pointer_observers)),
                viewer_count=count(u->u.session===s,values(ext._INSPECTION_UI_HANDLES)))))
        end
        B.DOM.div(viewer,B.DOM.pre(telemetry;var"data-testid"="julia-session-state",style="display:none"))
    end
end
