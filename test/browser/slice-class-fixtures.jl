include("../fixtures/slice_classes.jl")
function _slice_classes_app(sessions,tick)
    instance=Ref{Any}(nothing)
    B.App(;title="Slice classes") do client
        if instance[]===nothing || V.inspection_summary(instance[]).closed
            f=slice_class_triangle()
            s=V.inspection_session(f.enc;box=((0,0),(3,3)),slice=(basepoint=(0,0),direction=(1,1)),
                slice_classes=true,source_basis=f.basis,cache_limit=2)
            instance[]=_observe_fixture!(sessions,s,tick)
        end
        s=instance[];viewer=V.visualize(s;backend=:wglmakie)
        ext=Base.get_extension(TamerOp,:TamerOpWGLMakieExt);ui=ext._inspection_ui(viewer)
        telemetry=map(client,tick) do _
            c=V.slice_class(s);r=V.slice_correspondence(s)
            JSON3.write(_json_value((;summary=V.inspection_summary(s),selection=V.inspection_selection(s),
                class=c===nothing ? nothing : (;c.parameter,c.coordinates,cycle=c.source_cycle_data,filling=c.source_filling_data),
                correspondence=r===nothing ? nothing : (;r.status,r.source_point,r.target_point,r.join_point,r.coefficients,kernel_size=size(r.kernel)),
                ui=(;closed=ui.closed[],error=ui.error_text[]),
                viewer_count=count(u->u.session===s,values(ext._INSPECTION_UI_HANDLES)))))
        end
        B.DOM.div(viewer,B.DOM.pre(telemetry;var"data-testid"="julia-session-state",style="display:none"))
    end
end
