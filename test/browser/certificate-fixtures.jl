include("../fixtures/pruning.jl")

function _certificate_browser_app(sessions,tick)
    instance=Ref{Any}(nothing)
    return B.App(;title="A101 exact certificates") do client
        if instance[]===nothing
            pruning=TamerOp.epsilon_pruning(_pruning_box_fixture(),1//2)
            instance[]=_observe_fixture!(sessions,V.inspection_session(pruning;point=(1,1)),tick)
        end
        session=instance[];viewer=V.visualize(session;backend=:wglmakie)
        ext=Base.get_extension(TamerOp,:TamerOpWGLMakieExt);ui=ext._inspection_ui(viewer);revision=Ref(-1)
        telemetry=map(client,tick) do _
            summary=V.inspection_summary(session)
            if !ui.closed[] && revision[]!=summary.revision
                M.update_state_before_display!(ui.figure[]);revision[]=summary.revision
            end
            snapshot=V.inspection_snapshot(session)
            width,height=M.widths(M.viewport(ui.figure[].scene)[])
            targets=NamedTuple[]
            if ui.axis[]!==nothing
                for item in ui.targets[]
                    pixel=M.project(ui.axis[].scene,M.Point2d(item.point))+minimum(M.viewport(ui.axis[].scene)[])
                    push!(targets,(;vertex=item.vertex,label=item.label,x=Float64(pixel[1]/width),y=Float64(1-pixel[2]/height)))
                end
            end
            eq=snapshot.metadata.equations
            JSON3.write(_json_value((;fixture=:certificate,summary,selection=summary.selection,
                equations=(;forward=eq.forward.lhs==eq.forward.rhs,reverse=eq.reverse.lhs==eq.reverse.rhs),
                viewer_count=count(u->u.session===session,values(ext._INSPECTION_UI_HANDLES)),
                ui=(;closed=ui.closed[],error=ui.error_text[],pick_targets=targets,
                    mouseposition=M.events(ui.figure[].scene).mouseposition[],figure_size=(width,height)))))
        end
        B.DOM.div(viewer,B.DOM.pre(telemetry;var"data-testid"="julia-session-state",style="display:none"))
    end
end
