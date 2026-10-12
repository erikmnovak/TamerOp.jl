# Read-only telemetry for actual A89 browser interactions.
function _chains_browser_session(kind)
    C=TamerOp.ChainComplexes
    if kind==:source
        return V.inspection_session(TamerOp.filtered_complex(zeros(Int,1,1));kind=:chain_inspector,degree=1,parameter=0)
    elseif kind==:cone
        source=C.CochainComplex{QQ}(0,0,[1],SparseMatrixCSC{QQ,Int}[])
        target=C.CochainComplex{QQ}(0,0,[0],SparseMatrixCSC{QQ,Int}[])
        return V.inspection_session(C.long_exact_sequence(C.CochainMap(source,target,[spzeros(QQ,0,1)]);output=:full);degree=-1)
    end
    complex=C.CochainComplex{QQ}(0,2,[1,3,1],
        [sparse(reshape(QQ[1,0,0],3,1)),sparse(reshape(QQ[0,0,1],1,3))])
    return V.inspection_session(complex;degree=1)
end
function _chains_browser_app(sessions,tick;kind=:cochain)
    instance=Ref{Any}(nothing)
    return B.App(;title="A89 chains and connecting maps") do client
        instance[]===nothing && (instance[]=_observe_fixture!(sessions,_chains_browser_session(kind),tick))
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
                    push!(targets,(;selection=item.selection,dimension=item.dimension,x=Float64(pixel[1]/width),y=Float64(1-pixel[2]/height)))
                end
            end
            selected=snapshot.metadata.selection
            JSON3.write(_json_value((;fixture=kind,summary,selection=summary.selection,selected,
                viewer_count=count(u->u.session===session,values(ext._INSPECTION_UI_HANDLES)),
                ui=(;closed=ui.closed[],error=ui.error_text[],pick_targets=targets,
                    mouseposition=M.events(ui.figure[].scene).mouseposition[],figure_size=(width,height)))))
        end
        B.DOM.div(viewer,B.DOM.pre(telemetry;var"data-testid"="julia-session-state",style="display:none"))
    end
end
