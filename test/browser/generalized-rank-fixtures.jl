# Direct finite diagrams exercise region questions without a filtration route.
function _region_browser_fixture(;worm=false)
    FF,MD,EC=TamerOp.FiniteFringe,TamerOp.Modules,TamerOp.EncodingCore
    P=FF.ProductOfChainsPoset((3,3))
    supports=[collect(1:9),[2,3,5,6],[4,5,7,8]]
    active=[findall(s->q in s,supports) for q in 1:9]
    maps=Dict(e=>QQ[i==j for i in active[e[2]],j in active[e[1]]] for e in FF.cover_edges(P))
    module_=MD.PModule{QQ}(P,length.(active),maps;field=TamerOp.CoreModules.QQField())
    enc=TamerOp.EncodingResult(P,module_,EC.GridEncodingMap(P,([0,2,4],[0,3,6])))
    worm ? V.generalized_rank_session(enc;center=(3,4),width=1,levels=(1,2,3),lengths=(1,2,3)) :
        V.generalized_rank_session(enc;vertices=2:8,boundary=true)
end
function _region_browser_app(sessions,tick;worm=false)
    instance=Ref{Any}(nothing)
    B.App(;title="A54 generalized rank") do client
        instance[]===nothing && (instance[]=_observe_fixture!(sessions,_region_browser_fixture(;worm),tick))
        session=instance[];viewer=V.visualize(session;backend=:wglmakie)
        ext=Base.get_extension(TamerOp,:TamerOpWGLMakieExt);ui=ext._inspection_ui(viewer);revision=Ref(-1)
        telemetry=map(client,tick) do _
            summary=V.inspection_summary(session);targets=NamedTuple[];domain=NamedTuple[]
            if !ui.closed[]
                if revision[]!=summary.revision
                    M.update_state_before_display!(ui.figure[]);revision[]=summary.revision
                end
                width,height=M.widths(M.viewport(ui.figure[].scene)[])
                for item in ui.targets[]
                    pixel=M.project(item.axis.scene,M.Point2d(item.point))+minimum(M.viewport(item.axis.scene)[])
                    push!(targets,merge(Base.structdiff(item,(;axis=item.axis)),
                        (;x=Float64(pixel[1]/width),y=Float64(1-pixel[2]/height))))
                end
                if worm
                    # A known parameter target, used only to report its drawn pixel.
                    point=(1.,2.);ax=ui.axis[]
                    pixel=M.project(ax.scene,M.Point2d(point))+minimum(M.viewport(ax.scene)[])
                    push!(domain,(;point,x=Float64(pixel[1]/width),y=Float64(1-pixel[2]/height)))
                end
            end
            snapshot=V.inspection_snapshot(session)
            b=snapshot.metadata.boundary;r=snapshot.metadata.landscape
            JSON3.write(_json_value((;summary,selection=summary.selection,
                boundary=b===nothing ? nothing : (;path=TamerOp.Invariants.boundary_path(b),bars=TamerOp.Invariants.boundary_bars(b)),
                gril=r===nothing ? nothing : TamerOp.Invariants.gril_values(r),
                viewer_count=count(u->u.session===session,values(ext._INSPECTION_UI_HANDLES)),
                ui=(;closed=ui.closed[],error=ui.error_text[],pick_targets=targets,domain))))
        end
        B.DOM.div(viewer,B.DOM.pre(telemetry;var"data-testid"="julia-session-state",style="display:none"))
    end
end
