function _transport_browser_session()
    A=TamerOp.Advanced;field=TamerOp.CoreModules.QQField()
    P=A.ProductOfChainsPoset((2,2))
    M=A.PModule{QQ}(P,ones(Int,4),Dict(e=>ones(QQ,1,1) for e in A.cover_edges(P));field)
    fine=A.EncodingMap(P,P,[1,2,3,4])
    # The right classifier is still genuinely planar (one x bin, two y bins).
    rightP=A.ProductOfChainsPoset((1,2))
    N=A.PModule{QQ}(rightP,ones(Int,2),Dict((1,2)=>ones(QQ,1,1));field)
    coarse=A.EncodingMap(P,rightP,[1,1,2,2]);joint=A.joint_encoding(fine,coarse)
    left=TamerOp.EncodingResult(P,M,A.GridEncodingMap(P,([0,1],[0,1])))
    right=TamerOp.EncodingResult(rightP,N,A.GridEncodingMap(rightP,([0],[0,1])))
    result=TamerOp.common_refinement(left,right,joint);mods=A.translated_modules(result)
    f=A.PMorphism(mods.left,mods.right,[ones(QQ,1,1) for _ in 1:4])
    V.inspection_session(result;sources=(left,right),joint,comparison=f,box=((-1,-1),(2,2)))
end
function _transport_browser_app(sessions,tick)
    instance=Ref{Any}(nothing)
    B.App(;title="A88 transport comparisons") do client
        instance[]===nothing && (instance[]=_observe_fixture!(sessions,_transport_browser_session(),tick))
        session=instance[];viewer=V.visualize(session;backend=:wglmakie)
        ext=Base.get_extension(TamerOp,:TamerOpWGLMakieExt);ui=ext._inspection_ui(viewer);revision=Ref(-1)
        telemetry=map(client,tick) do _
            summary=V.inspection_summary(session)
            if !ui.closed[] && revision[]!=summary.revision
                M.update_state_before_display!(ui.figure[]);revision[]=summary.revision
            end
            width,height=M.widths(M.viewport(ui.figure[].scene)[])
            targets=NamedTuple[]
            for t in ui.targets[]
                pixel=M.project(t.axis.scene,M.Point2d(t.point))+minimum(M.viewport(t.axis.scene)[])
                push!(targets,(;side=t.side,label=t.label,x=Float64(pixel[1]/width),y=Float64(1-pixel[2]/height)))
            end
            snap=V.inspection_snapshot(session)
            natural=snap.metadata.naturality
            square=natural===nothing ? nothing : natural.defined ? (;defined=true,valid=natural.valid,left=string.(natural.left),right=string.(natural.right)) : (;defined=false)
            JSON3.write(_json_value((;fixture=:transport,summary,selection=summary.selection,square,
                viewer_count=count(u->u.session===session,values(ext._INSPECTION_UI_HANDLES)),
                ui=(;closed=ui.closed[],error=ui.error_text[],pick_targets=targets,
                    mouseposition=M.events(ui.figure[].scene).mouseposition[],figure_size=(width,height)))))
        end
        B.DOM.div(viewer,B.DOM.pre(telemetry;var"data-testid"="julia-session-state",style="display:none"))
    end
end
