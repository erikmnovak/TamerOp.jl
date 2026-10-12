# A86 read-only telemetry and a computed finite-module cancellation fixture.
function _signed_browser_app(sessions,tick;style=TamerOp.VisualStyle())
    instance=Ref{Any}(nothing)
    B.App(;title="A86 signed cancellation") do client
        if instance[]===nothing
            F,E,D=TamerOp.FiniteFringe,TamerOp.EncodingCore,TamerOp.Modules
            xs=[0,2,5];P=F.ProductOfChainsPoset((3,3));pi=E.GridEncodingMap(P,(xs,xs))
            dims=Int[x<=2 || y<=2 for y in xs for x in xs]
            module_=D.PModule{QQ}(P,dims,Dict((a,b)=>ones(QQ,dims[b],dims[a]) for (a,b) in F.cover_edges(P));field=TamerOp.CoreModules.QQField())
            enc=TamerOp.Results.EncodingResult(P,module_,pi)
            measure=TamerOp.rank_signed_measure(enc;threads=false)
            direct=(p,q)->TamerOp.InvariantCore.rank_map(module_,E.locate(pi,collect(p)),E.locate(pi,collect(q)))
            instance[]=_observe_fixture!(sessions,V.inspection_session(measure;query=(p=(0,0),q=(2,2)),direct),tick)
        end
        s=instance[];viewer=V.visualize(s;backend=:wglmakie,style)
        ext=Base.get_extension(TamerOp,:TamerOpWGLMakieExt);ui=ext._inspection_ui(viewer)
        revision=Ref(-1)
        telemetry=map(client,tick) do _
            summary=V.inspection_summary(s)
            if !ui.closed[] && revision[]!=summary.revision
                M.update_state_before_display!(ui.figure[]);revision[]=summary.revision
            end
            shot=V.inspection_snapshot(s);width,height=M.widths(M.viewport(ui.figure[].scene)[])
            targets=NamedTuple[]
            for (kind,axis) in ui.axes,item in ui.targets[kind]
                pixel=M.project(axis.scene,M.Point2d(item.point))+minimum(M.viewport(axis.scene)[])
                push!(targets,(;id=item.id,panel=kind,x=Float64(pixel[1]/width),y=Float64(1-pixel[2]/height)))
            end
            JSON3.write(_json_value((;fixture=:signed,summary,selection=summary.selection,
                reconstructed=shot.metadata.reconstructed,direct=shot.metadata.direct_value,
                records=shot.metadata.records,viewer_count=count(u->u.session===s,values(ext._INSPECTION_UI_HANDLES)),
                ui=(;closed=ui.closed[],error=ui.error_text[],pick_targets=targets,
                    mouseposition=M.events(ui.figure[].scene).mouseposition[],figure_size=(width,height)))))
        end
        B.DOM.div(viewer,B.DOM.pre(telemetry;var"data-testid"="julia-session-state",style="display:none"))
    end
end
