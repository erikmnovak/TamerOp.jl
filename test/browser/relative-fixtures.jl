include("../fixtures/spatial_pairs.jl")
function _relative_browser_app(sessions,tick;periodic=false)
    instance=Ref{Any}(nothing)
    B.App(;title="Spatial relative selection") do client
        if instance[]===nothing
            x=periodic ? TamerOp.DataIngestion.filtered_complex(zeros(Int,1,1);input=:vertices,periodic=true) : _a128_disk()
            p=TamerOp.spatial_pair(x;cells=periodic ? [4] : [4,5,6])
            instance[]=_observe_fixture!(sessions,V.inspection_session(p;degree=2),tick)
        end
        session=instance[];viewer=V.visualize(session;backend=:wglmakie)
        ext=Base.get_extension(TamerOp,:TamerOpWGLMakieExt);ui=ext._inspection_ui(viewer)
        telemetry=map(client,tick) do _
            summary=V.inspection_summary(session);snapshot=V.inspection_snapshot(session)
            targets=NamedTuple[];brush=NamedTuple[]
            if !ui.closed[] && ui.axis[]!==nothing
                figure=ui.figure[];axis=ui.axis[];M.update_state_before_display!(figure)
                width,height=M.widths(M.viewport(figure.scene)[])
                project(p)=let q=M.project(axis.scene,M.Point2d(p))+minimum(M.viewport(axis.scene)[])
                    (;x=Float64(q[1]/width),y=Float64(1-q[2]/height))
                end
                for t in ui.targets[];push!(targets,merge((;index=t.index),project(t.point)));end
                # A box strictly around the lower edge avoids pixel-rounding ties.
                for p in ((-1.1,-0.1),(5.1,0.1));push!(brush,project(p));end
            end
            JSON3.write(_json_value((;fixture=:relative,summary,selection=summary.selection,
                counts=snapshot.metadata.counts,selected=snapshot.metadata.selected,added=snapshot.metadata.added,
                table=snapshot.metadata.table,maps=snapshot.metadata.maps===nothing ? nothing : map(_matrix_state,snapshot.metadata.maps),
                representative=snapshot.metadata.representative===nothing ? nothing : (;index=snapshot.metadata.representative.index),
                ui=(;closed=ui.closed[],error=ui.error_text[],rebuild_count=ui.rebuild_count[],pick_targets=targets,brush,
                    observer_count=length(ui.observers)+length(ui.pointer_observers)),
                viewer_count=count(u->u.session===session,values(ext._INSPECTION_UI_HANDLES)))))
        end
        B.DOM.div(viewer,B.DOM.pre(telemetry;var"data-testid"="julia-session-state",style="display:none"))
    end
end
