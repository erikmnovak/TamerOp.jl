# Supplied values with different row/column meanings; selection uses real events.
function _collection_browser_session()
    bars=[[(0.,2.)],[(0.,3.)],[(2.,4.)]]
    witness=TamerOp.bottleneck_matching(bars[1],bars[2])
    source=V.VisualizationSpec(:collection_source;title="Supplied source points",
        layers=V.AbstractVisualizationLayer[V.PointLayer([(0.,0.),(1.,0.),(0.,1.)],:black,1.,10.)])
    c=V.visual_collection(;sample_ids=["A","B","C"],features=[2. 1.;3. 1.;2. 0.],
        feature_names=["lifetime","dimension at 1"],definitions=["Death minus birth","Dimension at parameter 1"],
        distances=[0. 1. 1.;1. 0. 1.5;1. 1.5 0.],distance_semantics="Bottleneck distance, diagonal allowed",
        kernel=[1. -1. 0.;-1. 1. 0.;0. 0. 1.],kernel_semantics="Supplied similarity",
        embedding=(coordinates=[0. 0.;1. 2.;3. 1.],method="Supplied display coordinates",settings=(purpose="browser fixture",),seed=17),
        samples=[(;object=bar,invariant=V.visual_spec(bar),source,metadata=(degree=0,field="F2")) for bar in bars],
        witnesses=Dict(("A","B")=>witness),degree=0,field="F2")
    V.inspection_session(c)
end
function _collection_browser_app(sessions,tick)
    instance=Ref{Any}(nothing)
    B.App(;title="A98 collections") do client
        instance[]===nothing && (instance[]=_observe_fixture!(sessions,_collection_browser_session(),tick))
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
                push!(targets,(;item.sample,item.target,item.feature,item.label,item.kind,
                    x=Float64(pixel[1]/width),y=Float64(1-pixel[2]/height)))
            end
            JSON3.write(_json_value((;summary,selection=summary.selection,selected=snapshot.metadata.selected,
                pair_open=ui.pair_session[]!==nothing,
                viewer_count=count(u->u.session===session,values(ext._INSPECTION_UI_HANDLES)),
                ui=(;closed=ui.closed[],error=ui.error_text[],pick_targets=targets,
                    mouseposition=M.events(ui.figure[].scene).mouseposition[],figure_size=(width,height)))))
        end
        B.DOM.div(viewer,B.DOM.pre(telemetry;var"data-testid"="julia-session-state",style="display:none"))
    end
end
