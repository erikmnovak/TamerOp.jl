function _comparison_browser_session(;ordinary=false)
    if ordinary
        G=TamerOp.DataTypes.GradedComplex([[1],[2]],[sparse(reshape([2],1,1))],[(0,),(1,)])
        d2=TamerOp.persistence_diagram(G;field=TamerOp.CoreModules.F2())
        d3=TamerOp.persistence_diagram(G;field=TamerOp.CoreModules.F3())
        c=V.compare_visuals((f2=d2,f3=d3);parameters=[0,1,2],
            correspondence="Same integral cellular boundary in each characteristic")
        return V.inspection_session(c;view=:barcode)
    end
    P=TamerOp.FiniteFringe.FinitePoset(BitMatrix([true true;false true]))
    function module_over(field)
        K=TamerOp.CoreModules.coeff_type(field)
        TamerOp.Modules.PModule{K}(P,[2,2],Dict((1,2)=>K[1 1;1 -1]);field)
    end
    c=V.compare_visuals((f2=module_over(TamerOp.CoreModules.F2()),f3=module_over(TamerOp.CoreModules.F3()));
        vertices=(f2=[1,2],f3=[1,2]),correspondence="Same integer matrix modulo each prime")
    V.inspection_session(c)
end
function _comparison_browser_app(sessions,tick;ordinary=false)
    instance=Ref{Any}(nothing)
    B.App(;title="A94 compare results") do client
        instance[]===nothing && (instance[]=_observe_fixture!(sessions,_comparison_browser_session(;ordinary),tick))
        session=instance[];viewer=V.visualize(session;backend=:wglmakie)
        ext=Base.get_extension(TamerOp,:TamerOpWGLMakieExt);ui=ext._inspection_ui(viewer);revision=Ref(-1)
        telemetry=map(client,tick) do _
            summary=V.inspection_summary(session)
            if !ui.closed[] && revision[]!=summary.revision
                M.update_state_before_display!(ui.figure[]);revision[]=summary.revision
            end
            snapshot=V.inspection_snapshot(session)
            width,height=M.widths(M.viewport(ui.figure[].scene)[]);targets=NamedTuple[]
            for item in ui.targets[]
                pixel=M.project(item.axis.scene,M.Point2d(item.point))+minimum(M.viewport(item.axis.scene)[])
                push!(targets,(;item.probe,item.label,axis_index=findfirst(==(item.axis),ui.axes),
                    x=Float64(pixel[1]/width),y=Float64(1-pixel[2]/height)))
            end
            JSON3.write(_json_value((;summary,selection=summary.selection,
                values=map(row->getproperty.(row,:value),snapshot.metadata.values),
                differences=map(row->getproperty.(row,:value),snapshot.metadata.differences),
                viewer_count=count(u->u.session===session,values(ext._INSPECTION_UI_HANDLES)),
                ui=(;closed=ui.closed[],error=ui.error_text[],pick_targets=targets,
                    mouseposition=M.events(ui.figure[].scene).mouseposition[],figure_size=(width,height)))))
        end
        B.DOM.div(viewer,B.DOM.pre(telemetry;var"data-testid"="julia-session-state",style="display:none"))
    end
end
