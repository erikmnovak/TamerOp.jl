# Nonuniform probes expose coordinate mistakes; a selected source links by ID.
function _ensemble_browser_session()
    e=TamerOp.persistence_ensemble([0. 1. 4.;2. 3. 8.;4. 2. 3.];axes=([0,2,9],),sample_ids=["A","B","C"],
        statistic="Supplied dimensions",protocol="Fixed repeated inputs",population="Three fixtures")
    source=V.VisualizationSpec(:ensemble_source;title="Source points",layers=V.AbstractVisualizationLayer[
        V.PointLayer([(0.,0.),(1.,0.),(0.,1.)],:black,1.,12.)])
    V.inspection_session(TamerOp.summarize_ensemble(e;band=:range);samples=[(;source),(;source),(;source)])
end
function _ensemble_fields_browser_session()
    A=[1. 7.;2. 8.;3. 9.]
    fields=[V.ScalarField(([0.,2.,9.],[-3.,5.]),A;id="one"),V.ScalarField(([0.,2.,9.],[-3.,5.]),3A;id="three")]
    e=TamerOp.persistence_ensemble(fields;protocol="Fixed fields",population="Two fixtures")
    V.inspection_session(TamerOp.summarize_ensemble(e;threshold=5);sample=2)
end
function _ensemble_browser_app(sessions,tick;fields=false)
    instance=Ref{Any}(nothing)
    B.App(;title="A95 ensembles") do client
        instance[]===nothing && (instance[]=_observe_fixture!(sessions,fields ? _ensemble_fields_browser_session() : _ensemble_browser_session(),tick))
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
                pixel=M.project(item.axis.scene,M.Point2d(item.point))+minimum(M.viewport(item.axis.scene)[])
                push!(targets,(;item.probe,item.label,item.kind,x=Float64(pixel[1]/width),y=Float64(1-pixel[2]/height)))
            end
            JSON3.write(_json_value((;summary,selection=Base.structdiff(summary.selection,(;samples=summary.selection.samples)),selected=snapshot.metadata.selected,
                viewer_count=count(u->u.session===session,values(ext._INSPECTION_UI_HANDLES)),
                ui=(;closed=ui.closed[],error=ui.error_text[],pick_targets=targets,figure_size=(width,height)))))
        end
        B.DOM.div(viewer,B.DOM.pre(telemetry;var"data-testid"="julia-session-state",style="display:none"))
    end
end
