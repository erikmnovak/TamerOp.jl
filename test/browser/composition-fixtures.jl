# Native figure controls exercise resize and axis changes. Telemetry only reads
# prepared data and rendered coordinates; it cannot move an annotation.
function _composition_browser_app(tick)
    B.App(;title="Semantic figure composition") do client
        image=zeros(Int,3,3);image[2,2]=5
        d=TamerOp.cubical_persistence(image;input=:top_cells,source=true,representatives=true)
        c=TamerOp.OrdinaryPersistence.persistence_representative(d;dim=1,scale=2).source_cycle
        note=V.correspondence_annotation(d,c;label="Retained cycle",curve=0.15)
        spec=V.compose_visuals((source=V.visual_panel(c),barcode=V.visual_panel(d;kind=:barcode,dim=1));
            title="The cycle and its interval",annotations=(note,))
        fig=V.visualize(spec;backend=:wglmakie,size=(1150,680))
        axes=filter(x->x isa M.Axis,fig.content)
        resize=B.Button("Resize figure");limits=B.Button("Change barcode limits");reset=B.Button("Reset limits")
        B.on(client,resize.value) do _;M.resize!(fig.scene,1450,820);end
        B.on(client,limits.value) do _;M.xlims!(axes[2],-1,8);end
        B.on(client,reset.value) do _;M.xlims!(axes[2],spec.panels[2].axes.xlimits...);end
        telemetry=map(client,tick) do _
            width,height=M.widths(M.viewport(fig.scene)[])
            lines=filter(p->p isa M.Lines,last(fig.scene.children).plots)
            points=lines[1][1][]
            expected=map(enumerate((note.from,note.to))) do (i,a)
                data=V._composition_anchor(spec.panels[i],a).point
                Tuple(M.project(axes[i].scene,M.Point2d(data))+minimum(M.viewport(axes[i].scene)[]))
            end
            center=minimum(M.viewport(axes[2].scene)[]) + 0.5*M.widths(M.viewport(axes[2].scene)[])
            JSON3.write(_json_value((;size=(width,height),actual=(Tuple(first(points)),Tuple(last(points))),expected,
                ink_probes=(Tuple(points[10]),Tuple(points[40])),
                barcode_center=(center[1]/width,1-center[2]/height),
                limits=Tuple(minimum(axes[2].finallimits[])),meaning=V.describe(note).meaning)))
        end
        dom=B.DOM.div(B.DOM.h1("Semantic figure composition"),B.DOM.p("Pan the barcode or change its window: the solid arrow follows its retained interval."),
            B.DOM.div(resize,limits,reset),
            B.DOM.div(fig;style="max-width:100%;overflow:auto",tabindex="0",var"aria-label"="Composed figure"),
            B.DOM.pre(telemetry;var"data-testid"="composition-state",style="display:none"))
        # A raw Figure leaves keyboard integration to its embedding host.
        ext=Base.get_extension(TamerOp,:TamerOpWGLMakieExt)
        ext._inspection_keyboard_navigation!(client,dom)
        dom
    end
end
