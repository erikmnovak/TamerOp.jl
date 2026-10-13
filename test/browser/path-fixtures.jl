include("../fixtures/persistence_paths.jl")
function _path_browser_app(sessions,tick;matched=false)
    instance=Ref{Any}(nothing)
    B.App(;title="Persistence paths") do client
        if instance[]===nothing
            path=matched ? _a96_supplied() : _a96_path()
            instance[]=_observe_fixture!(sessions,V.inspection_session(path),tick)
        end
        session=instance[];viewer=V.visualize(session;backend=:wglmakie)
        ext=Base.get_extension(TamerOp,:TamerOpWGLMakieExt);ui=ext._inspection_ui(viewer)
        nframes=length(session.object.frames)
        telemetry=map(client,tick) do _
            summary=V.inspection_summary(session);snapshot=V.inspection_snapshot(session)
            targets=NamedTuple[]
            if !ui.closed[]
                figure=ui.figure[];M.update_state_before_display!(figure)
                width,height=M.widths(M.viewport(figure.scene)[])
                for (i,ts) in enumerate(ui.targets[]),t in ts
                    ax=ui.axes[][i];q=M.project(ax.scene,M.Point2d(t.point))+minimum(M.viewport(ax.scene)[])
                    push!(targets,(;id=t.id,frame=t.frame,panel=i,x=Float64(q[1]/width),y=Float64(1-q[2]/height)))
                end
            end
            JSON3.write(_json_value((;fixture=:path,summary,selection=summary.selection,nframes,
                mode=snapshot.metadata.mode,records=snapshot.metadata.records,events=snapshot.metadata.events,
                representative=snapshot.metadata.representative!==nothing,
                source_representative_visible=any(p->get(p.metadata,:representative_visible,false),snapshot.panels),
                ui=(;closed=ui.closed[],error=ui.error_text[],playing=ui.playing[],timer_active=ui.timer[]!==nothing,
                    rebuild_count=ui.rebuild_count[],update_count=ui.update_count[],pick_targets=targets,
                    observer_count=length(ui.observers)+length(ui.pointer_observers)),
                viewer_count=count(u->u.session===session,values(ext._INSPECTION_UI_HANDLES)))))
        end
        B.DOM.div(viewer,B.DOM.pre(telemetry;var"data-testid"="julia-session-state",style="display:none"))
    end
end
