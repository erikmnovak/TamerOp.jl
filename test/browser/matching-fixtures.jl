# A85 self-contained browser fixtures and read-only telemetry.
function _matching_rectangles_session()
    F=TamerOp.FiniteFringe;E=TamerOp.EncodingCore;D=TamerOp.Modules
    coords=QQ[0,1,3,4];P=F.ProductOfChainsPoset((4,4));pi=E.GridEncodingMap(P,(coords,coords))
    modules=map([((0,1),(3,4)),((1,0),(4,4)),((1,1),(3,3))]) do (lo,hi)
        dims=Int[lo[1]<=x<hi[1] && lo[2]<=y<hi[2] for y in coords for x in coords]
        D.PModule{QQ}(P,dims,Dict((a,b)=>ones(QQ,dims[b],dims[a]) for (a,b) in F.cover_edges(P));field=TamerOp.CoreModules.QQField())
    end
    a=TamerOp.Results.EncodingResult(P,D.direct_sum(modules[1],modules[2]),pi)
    b=TamerOp.Results.EncodingResult(P,modules[3],pi)
    return V.inspection_session(a,b;opts=TamerOp.Options.InvariantOptions(box=([0,0],[4,4]),threads=false),
        samples=[(;basepoint=(0,h),direction=(1,1)) for h in (0,1)])
end
function _matching_browser_app(sessions,tick;fibered=false,style=TamerOp.VisualStyle())
    instance=Ref{Any}(nothing)
    return B.App(;title="A85 matching witness") do client
        if instance[] === nothing
            instance[]=_observe_fixture!(sessions,fibered ? _matching_rectangles_session() :
                V.inspection_session([(0.,2.),(0.,2.),(8.,10.),(4.,Inf)],[(0.,3.),(0.,3.),(5.,Inf)]),tick)
        end
        session=instance[]
        viewer=V.visualize(session;backend=:wglmakie,style)
        ext=Base.get_extension(TamerOp,:TamerOpWGLMakieExt)
        ui=ext._inspection_ui(viewer)
        revision=Ref(-1)
        telemetry=map(client,tick) do _
            summary=V.inspection_summary(session)
            if !ui.closed[] && revision[] != summary.revision
                M.update_state_before_display!(ui.figure[])
                revision[]=summary.revision
            end
            snapshot=V.inspection_snapshot(session)
            width,height=M.widths(M.viewport(ui.figure[].scene)[])
            targets=NamedTuple[]
            for (kind,axis) in ui.axes, item in ui.targets[kind]
                point=haskey(item,:segment) ? ((item.segment[1]+item.segment[3])/2,item.segment[2]) : item.point
                pixel=M.project(axis.scene,M.Point2d(point))+minimum(M.viewport(axis.scene)[])
                push!(targets,(;id=item.id,panel=kind,x=Float64(pixel[1]/width),y=Float64(1-pixel[2]/height)))
            end
            JSON3.write(_json_value((;fixture=fibered ? :matching_slices : :matching,summary,
                selection=summary.selection,records=snapshot.metadata.matching_records,
                context=snapshot.metadata.context,
                viewer_count=count(u->u.session===session,values(ext._INSPECTION_UI_HANDLES)),
                ui=(;closed=ui.closed[],error=ui.error_text[],pick_targets=targets,
                    mouseposition=M.events(ui.figure[].scene).mouseposition[],figure_size=(width,height)))))
        end
        B.DOM.div(B.jsrender(client,viewer),B.DOM.pre(telemetry;
            var"data-testid"="julia-session-state",style="display:none"))
    end
end
