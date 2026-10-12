# Semantic composition tests: independent values, source witnesses, exact
# identities, incompatible frames, and actual native render/export attachment.
function _a102_samples(;context=nothing,values=[0. 1 2;1 2 3;2 3 4])
    V=TamerOp.Visualization
    ctx=context===nothing ? V.VisualContext(domain="parameter-plane",units=("radius","codensity"),
        field="F2",degree=1,provenance="square") : context
    V.visual_panel(values;x=[-2,0,3],y=[-1,2,5],quantity="dimension",context=ctx,title="Dimensions")
end
function _a102_ring(;field=CM.F2())
    O=TamerOp.OrdinaryPersistence
    values=fill(0,3,3);values[2,2]=5
    d=O.cubical_persistence(values;input=:top_cells,source=true,representatives=true,field)
    c=O.persistence_representative(d;dim=1,scale=2).source_cycle
    return d,c
end

@testset "A102 compatible layers and independent scales" begin
    V=TamerOp.Visualization;a=_a102_samples();ctx=a.metadata.composition_context
    b=_a102_samples(;values=2a.metadata.sample_values)
    s=V.overlay_visuals((background=a,other=b);title="Two fields")
    @test V.check_visual_spec(s).valid
    @test s.metadata.overlay_names==(:background,:other)
    @test length(s.layers)==2
    @test s.layers[1].values==[0 1 2;1 2 3;2 3 4]
    @test a.metadata.sample_axes==([-2,0,3],[-1,2,5])
    @test V.describe(ctx).orientation==(1,1)
    for kw in ((;domain="other"),(;units=("cm","s")),(;orientation=(-1,1)),(;field="F3"),(;degree=2),(;provenance="different sample"))
        changed=V.VisualContext(;merge(V.describe(ctx),kw)...)
        other=_a102_samples(;context=changed)
        @test_throws ArgumentError V.overlay_visuals((a=a,b=other))
        compared=V.overlay_visuals((a=a,b=other);comparison="compare these declared inputs")
        @test occursin("Intentional comparison",compared.subtitle)
        @test !isempty(compared.metadata.differences)
    end
    noctx=V.VisualizationSpec(:points;layers=V.AbstractVisualizationLayer[V.PointLayer([(0.,0.)],:red,1.,10.)])
    @test_throws ArgumentError V.overlay_visuals((a=a,b=noctx))
    @test_throws ArgumentError V.overlay_visuals((a=a,b=V.visual_panel(b;limits=(x=(-3,3),y=(-1,5)))))
    @test_throws ArgumentError V.VisualContext(domain="",units=("r","c"),provenance="data")
    @test_throws ArgumentError V.VisualContext(domain="d",units=("r","c"),orientation=(0,1),provenance="data")
    @test_throws ArgumentError V.visual_panel([1 2;3 4];x=[1,1],y=[0,1],quantity="rank",context=ctx)
    @test_throws ArgumentError V.visual_panel([1 Inf;3 4];x=[0,1],y=[0,1],quantity="rank",context=ctx)
    @test_throws ArgumentError V.visual_panel([1 2;3 4];x=[0,1],y=[0,1],quantity="",context=ctx)
    @test V.visual_panel([1 NaN;3 4];x=[0,1],y=[0,1],quantity="rank",context=ctx).layers[1].values[1,1]==1
    composed=V.compose_visuals((first=a,second=b);link_axes=((:first,:second),),link_colors=((:first,:second),))
    @test all(p->p.metadata.colorrange==(0.,8.),composed.panels)
    @test !haskey(a.metadata,:colorrange)
    @test composed.metadata.linked_axes==((:first,:second),)
    @test_throws ArgumentError V.compose_visuals((a=a,b=b);link_axes=((:a,:missing),))
    @test_throws ArgumentError V.compose_visuals((a=a,);columns=false)
    @test_throws ArgumentError V.compose_visuals((figure=a,))
    @test_throws ArgumentError V.compose_visuals((nested=composed,))
    @test_throws ArgumentError V.compose_visuals((a=a,b=b,c=b);link_colors=((:a,:b),(:b,:c)))
    @test_throws ArgumentError V.compose_visuals((a=a,b=b);link_axes=(:a,:b))
    @test_throws ArgumentError V.visual_panel([1 2;3 4];x=[big(2)^70,big(2)^70+1],y=[0,1],quantity="rank",context=ctx)
    @test V.VisualContext(;V.describe(ctx)...)==ctx
    @test hash(V.VisualContext(;V.describe(ctx)...))==hash(ctx)
    S=TamerOp.SignedMeasures;xs=[-2,0,3];ys=[-1,2,5]
    values=[Int(x>=0 && y>=2) for x in xs,y in ys]
    pm=S.point_signed_measure(values,(xs,ys))
    signed=V.visual_panel(pm;kind=:signed_atoms,context=ctx,limits=(x=(-2,3),y=(-1,5)))
    over=V.overlay_visuals((background=a,atoms=signed))
    @test over.metadata.overlay_specs[2].metadata.composition_object===pm
    @test V._composition_anchor(over,V.VisualAnchor(:field;target=(:overlay,:atoms,1,1))).point==(0.,2.)
    for (i,x) in enumerate(xs),(j,y) in enumerate(ys)
        @test sum((w for (p,w) in zip(S.points(pm),S.weights(pm)) if p[1]<=x && p[2]<=y);init=0)==values[i,j]
    end
    conflict=V._composition_copy(signed;legend=(;visible=true,entries=(;positive=:black)))
    @test_throws ArgumentError V.compose_visuals((a=signed,b=conflict))
    for symbol in (:VisualContext,:VisualAnchor,:VisualAnnotation,:visual_panel,:overlay_visuals,:compose_visuals,:correspondence_annotation)
        @test getproperty(TamerOp,symbol)===getproperty(V,symbol)
    end
end

@testset "A102 semantic anchors and checked correspondences" begin
    V=TamerOp.Visualization;a=_a102_samples()
    matrix=V.VisualizationSpec(:coefficients;layers=V.AbstractVisualizationLayer[
        V.MatrixLayer(["1" "0";"-1" "2"],["u","v"],["g1","g2"])],metadata=(;panel_style=:matrix))
    dot=V.VisualizationSpec(:generator;layers=V.AbstractVisualizationLayer[V.PointLayer([(0.,0.),(1.,2.)],:red,1.,12.)],
        axes=V._default_axes_2d(xlimits=(-1,2),ylimits=(-1,3)))
    first=V.VisualAnchor(:matrix;target=(:matrix,2,1));last=V.VisualAnchor(:support;target=(:layer,1,2))
    ann=V.VisualAnnotation(first,last;label="Generator coefficient",curve=0.15)
    @test V.describe(ann).meaning===:authored_explanation
    s=V.compose_visuals((matrix=matrix,support=dot);annotations=(ann,))
    @test V.check_visual_spec(s).valid
    @test V._composition_anchor(dot,last).point==(1.,2.)
    @test V._composition_anchor(matrix,first).entry==(2,1)
    for bad in (V.VisualAnchor(:missing;point=(0,0)),V.VisualAnchor(:matrix;target=(:matrix,3,1)),
                V.VisualAnchor(:support;target=(:layer,1,3)),V.VisualAnchor(:support;target=(:bogus,1)))
        @test_throws ArgumentError V.compose_visuals((matrix=matrix,support=dot);annotations=(V.VisualAnnotation(bad,last;label="bad"),))
    end
    figure_note=V.VisualAnnotation(V.VisualAnchor(:figure;point=(0.25,0.7),space=:figure),
        V.VisualAnchor(:support;point=(0.5,0.5),space=:panel);kind=:callout,label="Figure to panel")
    @test V.check_visual_spec(V.compose_visuals((support=dot,);annotations=(figure_note,))).valid
    @test_throws ArgumentError V.VisualAnchor(:support;point=(NaN,0))
    @test_throws ArgumentError V.VisualAnchor(:support;point=(1.5,0),space=:panel)
    @test_throws ArgumentError V.VisualAnchor(:support;point=(0,0),target=(:layer,1,1))
    @test_throws ArgumentError V.VisualAnnotation(first,last;label="")
    @test_throws ArgumentError V.VisualAnnotation(first,last;label="invalid",kind=:highlight)
    @test_throws ArgumentError V.VisualAnnotation(first,last;label="invalid",curve=Inf)
    @test_throws ArgumentError V.compose_visuals((matrix=matrix,support=dot);annotations=(
        V.VisualAnnotation(V.VisualAnchor(:matrix;point=(0,0)),last;label="No invented matrix coordinates"),))
    for kind in (:arrow,:callout,:bracket,:highlight)
        x=V.VisualAnchor(:field;point=(-1,0));y=V.VisualAnchor(:field;point=(2,4))
        item=V.VisualAnnotation(x,y;kind,label=string(kind))
        @test V.check_visual_spec(V.compose_visuals((field=a,);annotations=(item,))).valid
    end
    d,c=_a102_ring();source=V.visual_panel(c);bars=V.visual_panel(d;kind=:barcode,dim=1)
    checked=V.correspondence_annotation(d,c;label="Retained H1 cycle",curve=.15)
    @test V.describe(checked).meaning===:computed_correspondence
    composed=V.compose_visuals((source=source,barcode=bars);annotations=(checked,))
    @test V.check_visual_spec(composed).valid
    @test length(c.coefficients)>0
    @test all(iszero,DT.boundary_map(c.source.complex;dim=1)*c.coefficients)
    @test checked.to.target==(:interval,1,:finite,1)
    @test V._composition_anchor(bars,checked.to).point==(2.5,1.)
    @test_throws ArgumentError V._composition_anchor(bars,V.VisualAnchor(:barcode;target=(:interval,true,:finite,1)))
    another,_=_a102_ring()
    @test_throws ArgumentError V.correspondence_annotation(another,c)
    @test_throws ArgumentError V.compose_visuals((source=source,barcode=V.visual_panel(another;kind=:barcode,dim=1));annotations=(checked,))
    @test_throws ArgumentError V.compose_visuals((source=source,barcode=V.visual_panel(d;kind=:barcode,dim=0));annotations=(checked,))
    @test_throws ArgumentError V.visual_panel(d;kind=:barcode,dim=1,
        context=V.VisualContext(domain="scale",units=("s","row"),field="wrong",provenance="ring"))
    @test_throws ArgumentError V.visual_panel(V.visual_panel(c);
        context=V.VisualContext(domain="source",units=("x","y"),field="wrong",provenance="ring"))
    @test_throws ArgumentError V.visual_panel(c;
        context=V.VisualContext(domain="source",units=("x","y"),degree=0,provenance="ring"))
    @test_throws ArgumentError V._composition_anchor(source,V.VisualAnchor(:source;target=(:source_cell,typemax(Int))))
    for field in (CM.F3(),CM.Fp(5))
        other,cycle=_a102_ring(;field)
        link=V.correspondence_annotation(other,cycle)
        picture=V.compose_visuals((source=V.visual_panel(cycle),barcode=V.visual_panel(other;kind=:barcode,dim=1));annotations=(link,))
        @test V.check_visual_spec(picture).valid
        @test link.evidence.cycle.field==field
    end
    # Named endpoints survive insertion and reordering, independent of ordinal position.
    reordered=V.compose_visuals((barcode=bars,unrelated=a,source=source);annotations=(checked,),columns=3)
    @test V.check_visual_spec(reordered).valid
    @test reordered.metadata.annotations[1].from.panel==:source
    cropped=V.visual_panel(c;max_cells=1)
    @test V.check_visual_spec(V.compose_visuals((source=cropped,barcode=bars);annotations=(checked,))).valid
    omitted=findlast(!iszero,c.coefficients)
    cropped_note=V.correspondence_annotation(d,c;source_cell=omitted)
    @test_throws ArgumentError V.compose_visuals((source=cropped,barcode=bars);annotations=(cropped_note,))
end

@testset "A102 native rendering and export attachment" begin
    V=TamerOp.Visualization
    if Base.find_package("CairoMakie")!==nothing
        @eval using CairoMakie
        a=_a102_samples();d,c=_a102_ring()
        checked=V.correspondence_annotation(d,c;curve=.12)
        s=V.compose_visuals((source=V.visual_panel(c),barcode=V.visual_panel(d;kind=:barcode,dim=1));annotations=(checked,),title="Source and interval")
        mktempdir() do dir
            for ext in ("png","svg","pdf","html")
                path=joinpath(dir,"composition."*ext)
                saved=V.save_visual(path,s)
                @test isfile(path) && filesize(path)>400
                if ext=="html"
                    html=read(path,String)
                    @test occursin("Checked representative correspondence",html)
                    @test occursin("<svg",html)
                    @test !occursin("<script",html)
                    @test occursin("tabindex=\"0\" role=\"region\" aria-label=\"Composed figure\"",html)
                    # Scrolling wraps the complete SVG; its mathematical
                    # coordinate frame is identical to the standalone export.
                    raw_svg=read(joinpath(dir,"composition.svg"),String)
                    @test match(r"viewBox=\"([^\"]+)\"",html).captures ==
                          match(r"viewBox=\"([^\"]+)\"",raw_svg).captures
                    output = get(ENV, "TAMEROP_COMPOSITION_HTML", "")
                    if !isempty(output)
                        mkpath(dirname(abspath(output)))
                        cp(path, output; force=true)
                    end
                end
            end
            exported=V.save_visual(dir,"canonical",s;format=:html)
            @test V.export_backend(exported)===:cairomakie
            @test occursin("source_cell",read(V.export_path(exported),String))
        end
        fig=V.visualize(s;backend=:cairomakie)
        axes=filter(x->x isa CairoMakie.Axis,fig.content)
        lines=filter(p->p isa CairoMakie.Lines,last(fig.scene.children).plots)
        @test length(lines)==2
        original=copy(lines[1][1][])
        CairoMakie.xlims!(axes[2],-1,8)
        @test lines[1][1][]!=original
        before=copy(lines[1][1][])
        CairoMakie.resize!(fig.scene,1600,850)
        @test lines[1][1][]!=before
        CairoMakie.xlims!(axes[2],10,20)
        @test any(p->any(isnan,p),lines[1][1][])
        empty!(fig)
        for style in (V.VisualStyle(),V.VisualStyle(palette=:grayscale))
            annotation=V.VisualAnnotation(V.VisualAnchor(:left;point=(0,0),space=:panel),
                V.VisualAnchor(:right;point=(1,1),space=:panel);label="Compare",curve=-.1)
            spec=V.compose_visuals((left=a,right=a);annotations=(annotation,),link_axes=((:left,:right),))
            fig=V.visualize(spec;style,backend=:cairomakie)
            axes=filter(x->x isa CairoMakie.Axis,fig.content)
            CairoMakie.xlims!(axes[1],-1,1)
            @test axes[1].finallimits[]==axes[2].finallimits[]
            empty!(fig)
        end
        matrix=V.VisualizationSpec(:coefficients;layers=V.AbstractVisualizationLayer[
            V.MatrixLayer(["1" "0";"-1" "2"],["u","v"],["g1","g2"])],metadata=(;panel_style=:matrix))
        arrow=V.VisualAnnotation(V.VisualAnchor(:matrix;target=(:matrix,2,1)),
            V.VisualAnchor(:field;point=(0,2));label="Coefficient explanation",curve=-.1)
        fig=V.visualize(V.compose_visuals((matrix=matrix,field=a);annotations=(arrow,));backend=:cairomakie)
        coefficient=only(r for r in visual_text_records(fig,CairoMakie.Makie) if r.text=="-1")
        curve=first(filter(p->p isa CairoMakie.Lines,last(fig.scene.children).plots))
        for size in ((1000,640),(1400,900))
            CairoMakie.resize!(fig.scene,size...)
            center=coefficient.plot[1][][coefficient.index]
            @test isapprox(first(curve[1][]),center)
        end
        empty!(fig)
    else
        @test_skip false
    end
end
