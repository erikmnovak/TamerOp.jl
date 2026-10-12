# Read-only telemetry for actual A91 canvas/control interactions.
function _components_browser_session(kind)
    D=TamerOp.DataTypes;O=TamerOp.OrdinaryPersistence
    if kind==:slice
        G=D.MultiCriticalGradedComplex([[1,2],[3]],[sparse(reshape([-1,1],2,1))],
            [[(0,0)],[(0,0)],[(1,3),(3,1)]])
        x=TamerOp.filtered_complex(G;geometry=[[(0,0)],[(1,0)],[(0,0),(1,0)]])
        return V.inspection_session(O.component_history(x;slice=(base=(0,0),direction=(1,2)));scale=1)
    end
    G=D.GradedComplex([[11,12,13,14],[101,102]],
        [sparse([-1 0;1 -1;0 1;0 0])],[(0,),(0,),(1,),(4,),(2,),(3,)])
    x=TamerOp.filtered_complex(G;geometry=[[(0,0)],[(1,1)],[(2,0)],[(3,1)],[(0,0),(1,1)],[(1,1),(2,0)]])
    V.inspection_session(O.component_history(x);scale=1)
end
function _components_browser_app(sessions,tick;kind=:graph)
    instance=Ref{Any}(nothing)
    B.App(;title="A91 components and merge history") do client
        instance[]===nothing && (instance[]=_observe_fixture!(sessions,_components_browser_session(kind),tick))
        session=instance[];viewer=V.visualize(session;backend=:wglmakie)
        ext=Base.get_extension(TamerOp,:TamerOpWGLMakieExt);ui=ext._inspection_ui(viewer);revision=Ref(-1)
        telemetry=map(client,tick) do _
            summary=V.inspection_summary(session)
            if !ui.closed[] && revision[]!=summary.revision
                M.update_state_before_display!(ui.figure[]);revision[]=summary.revision
            end
            snapshot=V.inspection_snapshot(session)
            width,height=M.widths(M.viewport(ui.figure[].scene)[]);targets=NamedTuple[]
            for item in ui.targets[]
                ax=ui.chart_axes[][item.axis_index]
                ax isa M.Axis || continue
                pixel=M.project(ax.scene,M.Point2d(item.point))+minimum(M.viewport(ax.scene)[])
                push!(targets,(;item.selection,item.axis_index,item.label,x=Float64(pixel[1]/width),y=Float64(1-pixel[2]/height)))
            end
            JSON3.write(_json_value((;fixture=kind,summary,selection=summary.selection,
                selected=snapshot.metadata.selection,partition=snapshot.metadata.partition,parameter=snapshot.metadata.parameter,
                viewer_count=count(u->u.session===session,values(ext._INSPECTION_UI_HANDLES)),
                ui=(;closed=ui.closed[],error=ui.error_text[],pick_targets=targets,
                    mouseposition=M.events(ui.figure[].scene).mouseposition[],figure_size=(width,height)))))
        end
        B.DOM.div(viewer,B.DOM.pre(telemetry;var"data-testid"="julia-session-state",style="display:none"))
    end
end
