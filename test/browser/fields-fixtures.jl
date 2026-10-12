function _fields_browser_app(sessions,tick)
    instance=Ref{Any}(nothing)
    B.App(;title="A44 coordinate sections") do client
        if instance[]===nothing
            coords=([2.,5.],[-3.,0.,4.],[10.,20.,30.,40.],[7.,9.])
            tensor=[1000i+100j+10k+l+10000h for i in 1:2,j in 1:3,k in 1:4,l in 1:2,h in 1:2]
            a=V.ScalarField(coords,tensor;id="A",layers=["H0","H1"],units="rank",axis_labels=("a","b","c","d"))
            b=V.ScalarField(coords,tensor.-3;id="B",layers=["H0","H1"],units="rank",axis_labels=("a","b","c","d"))
            instance[]=_observe_fixture!(sessions,V.field_session([a,b];planes=[(3,1),(2,4)],indices=(1,2,3,2)),tick)
        end
        session=instance[];viewer=V.visualize(session;backend=:wglmakie)
        ext=Base.get_extension(TamerOp,:TamerOpWGLMakieExt);ui=ext._inspection_ui(viewer)
        telemetry=map(client,tick) do _
            summary=V.inspection_summary(session);snapshot=V.inspection_snapshot(session)
            target=nothing;native=Any[]
            if !ui.closed[]
                M.update_state_before_display!(ui.figure[])
                width,height=M.widths(M.viewport(ui.figure[].scene)[])
                ax=first(ui.axes);point=(20.,4.5)
                pixel=M.project(ax.scene,M.Point2d(point))+minimum(M.viewport(ax.scene)[])
                target=(;x=Float64(pixel[1]/width),y=Float64(1-pixel[2]/height))
                for axis in ui.axes
                    h=only(filter(p->p isa M.Heatmap,axis.scene.plots));matrix=h[3][]
                    push!(native,(;rows=[[matrix[i,j] for j in axes(matrix,2)] for i in axes(matrix,1)],
                        colorrange=Tuple(h.colorrange[]),xlabel=axis.xlabel[],ylabel=axis.ylabel[]))
                end
            end
            JSON3.write(_json_value((;summary,selection=summary.selection,target,native,
                viewer_count=count(u->u.session===session,values(ext._INSPECTION_UI_HANDLES)),
                ui=(;rebuilds=ui.rebuild_count[],updates=ui.update_count[],closed=ui.closed[],error=ui.error_text[],observers=length(ui.observers),pointer_observers=length(ui.pointer_observers)))))
        end
        B.DOM.div(viewer,B.DOM.pre(telemetry;var"data-testid"="julia-session-state",style="display:none"))
    end
end
