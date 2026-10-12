# A92 uses the ring's top-cell model and a 3D skeleton; A93 links geometric data.
function _complex_browser_app(sessions,tick;volume=false,geometric=false)
    instance=Ref{Any}(nothing)
    return B.App(;title=geometric ? "Radius and depth" : "Filtered cells") do client
        if instance[]===nothing
            if geometric
                cells=TamerOp.filtered_complex(TamerOp.PointCloud([0 0;2 0]),TamerOp.RhomboidFiltration(depth_range=(1,1)))
                session=V.inspection_session(cells;kind=:geometric_filtration,parameter=(3//4,1),
                    view_dims=(1,2),invariant=:barcode,sample_resolution=15)
            else
                values=volume ? zeros(2,2,2) : [0 0 0;0 5 0;0 0 0]
                cells=TamerOp.filtered_complex(values;input=volume ? :vertices : :top_cells)
                session=V.inspection_session(cells)
            end
            instance[]=_observe_fixture!(sessions,session,tick)
        end
        session=instance[];viewer=V.visualize(session;backend=:wglmakie)
        ext=Base.get_extension(TamerOp,:TamerOpWGLMakieExt)
        ui=ext._inspection_ui(viewer);revision=Ref(-1)
        telemetry=map(client,tick) do _
            summary=V.inspection_summary(session)
            if !ui.closed[] && revision[]!=summary.revision
                M.update_state_before_display!(ui.figure[]);revision[]=summary.revision
            end
            width,height=M.widths(M.viewport(ui.figure[].scene)[])
            targets=NamedTuple[]
            if ui.axis[]!==nothing
                for item in ui.targets[]
                    point=length(item.point)==2 ? M.Point2d(item.point) : M.Point3d(item.point)
                    pixel=M.project(ui.axis[].scene,point)+minimum(M.viewport(ui.axis[].scene)[])
                    push!(targets,(;index=item.index,dimension=item.dimension,x=Float64(pixel[1]/width),y=Float64(1-pixel[2]/height)))
                end
            end
            snapshot=V.inspection_snapshot(session)
            selected=snapshot.metadata.selected
            details=selected===nothing ? nothing : (;index=selected.index,active=selected.active,
                dimension=selected.dimension,boundary=selected.boundary,source=selected.source)
            geometry=geometric ? (;betti=snapshot.metadata.betti,
                selected_sites=snapshot.panels[2].metadata.selected_sites,
                panel_parameter=snapshot.panels[3].metadata.parameter,
                radius_squared=snapshot.metadata.selected===nothing ? nothing : snapshot.metadata.selected.source.radius_squared) : nothing
            JSON3.write(_json_value((;fixture=:complex,summary,selection=summary.selection,selected=details,geometry,
                viewer_count=count(u->u.session===session,Base.values(ext._INSPECTION_UI_HANDLES)),
                ui=(;closed=ui.closed[],error=ui.error_text[],pick_targets=targets,
                    mouseposition=M.events(ui.figure[].scene).mouseposition[],figure_size=(width,height)))))
        end
        B.DOM.div(viewer,B.DOM.pre(telemetry;var"data-testid"="julia-session-state",style="display:none"))
    end
end
