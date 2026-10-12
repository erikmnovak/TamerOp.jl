# A42d browser input: a rank-one d2 on two-dimensional terms, padded so E3
# is genuinely uncomputed before a query (it is below the filtration bound).
function _spectral_browser_session()
    C=TamerOp.ChainComplexes
    I2=sparse(QQ[1 0;0 1]);T=sparse(QQ[1 0;0 0])
    dc=C.filtered_cochain_complex(QQ;pieces=Dict((0,1)=>2,(1,1)=>2,(1,2)=>2,(2,2)=>2),
        d0=Dict((1,1)=>I2),d1=Dict((0,1)=>I2,(1,1)=>T),p_range=0:4)
    return V.inspection_session(C.spectral_sequence(dc;output=:full))
end
function _spectral_browser_app(sessions,tick)
    instance=Ref{Any}(nothing)
    return B.App(;title="A42d spectral pages") do client
        instance[] === nothing && (instance[]=_observe_fixture!(sessions,_spectral_browser_session(),tick))
        session=instance[]
        viewer=V.visualize(session;backend=:wglmakie)
        ext=Base.get_extension(TamerOp,:TamerOpWGLMakieExt)
        ui=ext._inspection_ui(viewer);revision=Ref(-1)
        telemetry=map(client,tick) do _
            summary=V.inspection_summary(session)
            if !ui.closed[] && revision[] != summary.revision
                M.update_state_before_display!(ui.figure[]);revision[]=summary.revision
            end
            snapshot=V.inspection_snapshot(session)
            width,height=M.widths(M.viewport(ui.figure[].scene)[])
            targets=NamedTuple[]
            if ui.axis[] !== nothing
                for item in ui.targets[]
                    pixel=M.project(ui.axis[].scene,M.Point2d(item.point))+minimum(M.viewport(ui.axis[].scene)[])
                    push!(targets,(;bidegree=item.bidegree,dimension=item.dimension,x=Float64(pixel[1]/width),y=Float64(1-pixel[2]/height)))
                end
            end
            selected=snapshot.metadata.selected
            detail=selected === nothing ? nothing : (;source=selected.source,target=selected.target,rank=selected.rank,
                kernel=selected.kernel_dimension,next=selected.next_dimension,
                target_next=selected.target_data === nothing ? nothing : selected.target_data.next_dimension,
                matrix=string.(selected.matrix))
            JSON3.write(_json_value((;fixture=:spectral,summary,selection=summary.selection,selected=detail,
                viewer_count=count(u->u.session===session,values(ext._INSPECTION_UI_HANDLES)),
                ui=(;closed=ui.closed[],error=ui.error_text[],pick_targets=targets,
                    mouseposition=M.events(ui.figure[].scene).mouseposition[],figure_size=(width,height)))))
        end
        B.DOM.div(viewer,B.DOM.pre(telemetry;var"data-testid"="julia-session-state",style="display:none"))
    end
end
