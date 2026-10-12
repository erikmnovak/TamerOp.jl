function _editor_fixture(;presentation=false)
    if presentation
        return V.module_editor(;order=Bool[1 1 1;0 1 1;0 0 1],
            upsets=[[2,3]],downsets=[[1,2]],matrix=[1;;])
    end
    V.module_editor(;order=Bool[1 1 1 1;0 1 0 1;0 0 1 1;0 0 0 1],
        dimensions=ones(Int,4),maps=Dict(e=>ones(Int,1,1) for e in ((1,2),(1,3),(2,4),(3,4))))
end
function _editor_browser_app(sessions,tick;presentation=false)
    instance=Ref{Any}(nothing)
    B.App(;title="A99 finite editor") do client
        instance[]===nothing && (instance[]=_observe_fixture!(sessions,_editor_fixture(;presentation),tick))
        session=instance[];viewer=V.visualize(session;backend=:wglmakie)
        ext=Base.get_extension(TamerOp,:TamerOpWGLMakieExt);ui=ext._inspection_ui(viewer);revision=Ref(-1)
        telemetry=map(client,tick) do _
            summary=V.inspection_summary(session)
            if !ui.closed[] && revision[]!=summary.revision
                M.update_state_before_display!(ui.figure[]);revision[]=summary.revision
            end
            report=V.check_module_editor(session);w=report.witness
            JSON3.write(_json_value((;summary,selection=summary.selection,
                validation=(;valid=report.valid,issues=report.issues,witness=w===nothing ? nothing :
                    (;w.first_path,w.second_path,first=_matrix_state(w.first_composite),second=_matrix_state(w.second_composite))),
                draft=summary.closed ? nothing : V.editor_draft(session),
                viewer_count=count(u->u.session===session,values(ext._INSPECTION_UI_HANDLES)),
                ui=(;closed=ui.closed[],error=ui.error_text[]))))
        end
        B.DOM.div(viewer,B.DOM.pre(telemetry;var"data-testid"="julia-session-state",style="display:none"))
    end
end
