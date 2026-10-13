include("../fixtures/parameter_atlas.jl")

function _atlas_browser_app(sessions,tick;grid=false,module_input=false)
    instance=Ref{Any}(nothing)
    B.App(;title="Parameter atlas") do client
        if instance[]===nothing
            atlas=module_input ? TamerOp.parameter_atlas(_a125_zero_composite();parameters=1:3,witnesses=true) :
                grid ? TamerOp.parameter_atlas(_a125_triangle(bigraded=true);grid=([0,1,2],[0,1]),degree=1,witnesses=true) :
                TamerOp.parameter_atlas(_a125_triangle();parameters=[0,1,2],degree=1,witnesses=true)
            instance[]=_observe_fixture!(sessions,V.inspection_session(atlas;source=module_input || grid ? 1 : 2,target=3),tick)
        end
        session=instance[];viewer=V.visualize(session;backend=:wglmakie)
        ext=Base.get_extension(TamerOp,:TamerOpWGLMakieExt);ui=ext._inspection_ui(viewer)
        telemetry=map(client,tick) do _
            summary=V.inspection_summary(session);t=summary.transition
            targets=NamedTuple[]
            if !ui.closed[]
                M.update_state_before_display!(ui.figure)
                width,height=M.widths(M.viewport(ui.figure.scene)[])
                for (index,axis) in enumerate(ui.axes)
                    box=M.viewport(axis.scene)[];p=minimum(box)+M.widths(box)/2
                    push!(targets,(;index,x=Float64(p[1]/width),y=Float64(1-p[2]/height)))
                end
            end
            JSON3.write(_json_value((;fixture=:atlas,summary,selection=summary.selection,
                transition=t===nothing ? nothing : (;defined=t.defined,relation=t.relation,rank=t.rank,
                    kernel=t.kernel_dimension,cokernel=t.cokernel_dimension,matrix=_matrix_state(t.matrix),entries=[r.index for r in t.entries]),
                ui=(;closed=ui.closed[],error=ui.error_text[],rebuild_count=ui.rebuild_count[],pick_targets=targets,
                    figure_id=objectid(ui.figure),observer_count=length(ui.observers)+length(ui.pointer_observers)),
                viewer_count=count(u->u.session===session,values(ext._INSPECTION_UI_HANDLES)))))
        end
        B.DOM.div(viewer,B.DOM.pre(telemetry;var"data-testid"="julia-session-state",style="display:none"))
    end
end
