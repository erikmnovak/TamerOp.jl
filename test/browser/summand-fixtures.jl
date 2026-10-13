function _summand_browser_fixture()
    FF,MD,EC=TamerOp.FiniteFringe,TamerOp.Modules,TamerOp.EncodingCore
    P=FF.ProductOfChainsPoset((2,2));supports=[[1,2,3,4],[2,4],[4],[4]]
    active=[findall(I->q in I,supports) for q in 1:4]
    maps=Dict(e=>QQ[i==j for i in active[e[2]],j in active[e[1]]] for e in FF.cover_edges(P))
    module_=MD.PModule{QQ}(P,length.(active),maps;field=TamerOp.CoreModules.QQField())
    enc=TamerOp.EncodingResult(P,module_,EC.GridEncodingMap(P,([-3,2],[1,5])))
    result=TamerOp.interval_decomposition(module_;witnesses=true)
    V.summand_session(result;encoding=enc,box=([-4,0],[4,8]),vertex=4)
end
function _summand_browser_app(sessions,tick)
    instance=Ref{Any}(nothing);fixture_lock=ReentrantLock()
    B.App(;title="A87 checked summands") do client
        session=lock(fixture_lock) do
            instance[]===nothing && (instance[]=_observe_fixture!(sessions,_summand_browser_fixture(),tick))
            instance[]
        end
        viewer=V.visualize(session;backend=:wglmakie)
        ext=Base.get_extension(TamerOp,:TamerOpWGLMakieExt);ui=ext._inspection_ui(viewer);revision=Ref(-1)
        telemetry=map(client,tick) do _
            summary=V.inspection_summary(session);targets=NamedTuple[]
            if !ui.closed[]
                if revision[]!=summary.revision
                    M.update_state_before_display!(ui.figure[]);revision[]=summary.revision
                end
                width,height=M.widths(M.viewport(ui.figure[].scene)[])
                for item in ui.targets[]
                    pixel=M.project(item.axis.scene,M.Point2d(item.point))+minimum(M.viewport(item.axis.scene)[])
                    push!(targets,merge(Base.structdiff(item,(;axis=item.axis)),(;x=Float64(pixel[1]/width),y=Float64(1-pixel[2]/height))))
                end
            end
            snapshot=V.inspection_snapshot(session)
            JSON3.write(_json_value((;summary,selection=summary.selection,
                hom=snapshot.metadata.hom_dimensions,hom_component=_matrix_state(snapshot.metadata.hom_component),
                viewer_count=count(u->u.session===session,values(ext._INSPECTION_UI_HANDLES)),
                ui=(;closed=ui.closed[],error=ui.error_text[],pick_targets=targets))))
        end
        B.DOM.div(viewer,B.DOM.pre(telemetry;var"data-testid"="julia-session-state",style="display:none"))
    end
end
