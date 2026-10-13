include("../fixtures/source_laboratory.jl")
function _source_laboratory_browser_app(sessions,tick;limited=false,activity=false)
    instance=Ref{Any}(nothing)
    B.App(;title="Source laboratory") do client
        if instance[]===nothing
            lab=activity ? V.source_laboratory(TamerOp.DataTypes.PointCloud([0.0 0.0;1.0 0.0;4.0 0.0]);degree=0,dtm_mass=0.4,max_diameter=5.,codensity=1.,diameter=1.) : limited ? V.source_laboratory(_a124_cloud();max_diameter=0.1,max_cells=4) : _a124_session()
            instance[]=_observe_fixture!(sessions,lab,tick)
        end
        session=instance[];viewer=V.visualize(session;backend=:wglmakie)
        ext=Base.get_extension(TamerOp,:TamerOpWGLMakieExt);ui=ext._inspection_ui(viewer)
        telemetry=map(client,tick) do _
            summary=V.inspection_summary(session);snapshot=V.inspection_snapshot(session);targets=NamedTuple[]
            if !ui.closed[]
                figure=ui.figure[];M.update_state_before_display!(figure)
                width,height=M.widths(M.viewport(figure.scene)[]);ax=ui.axes[][2]
                for t in ui.targets[]
                    q=M.project(ax.scene,M.Point2d(t.point))+minimum(M.viewport(ax.scene)[])
                    push!(targets,(;id=t.id,x=Float64(q[1]/width),y=Float64(1-q[2]/height)))
                end
            end
            JSON3.write(_json_value((;fixture=:source_laboratory,summary,selection=summary.selection,
                draft=snapshot.metadata.draft,current=snapshot.metadata.current,
                source_activity=[p.metadata.active_source_ids for p in snapshot.panels[1:2]],
                bars=[p.metadata.records for p in snapshot.panels[3:4]],
                ui=(;closed=ui.closed[],dragging=ui.drag[]!==nothing,error=ui.error_text[],pick_targets=targets,rebuild_count=ui.rebuild_count[],update_count=ui.update_count[],
                    observer_count=length(ui.observers)+length(ui.pointer_observers)),
                viewer_count=count(u->u.session===session,values(ext._INSPECTION_UI_HANDLES)))))
        end
        B.DOM.div(viewer,B.DOM.pre(telemetry;var"data-testid"="julia-session-state",style="display:none"))
    end
end
