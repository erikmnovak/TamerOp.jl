@testset "A38 retained figures update buffers and reject structural changes" begin
    V=TamerOp.Visualization
    for (backend,package,extension) in ((:cairomakie,"CairoMakie",:TamerOpCairoMakieExt),(:wglmakie,"WGLMakie",:TamerOpWGLMakieExt))
        if Base.find_package(package)===nothing
            @test_skip false;continue
        end
        backend===:cairomakie ? (@eval import CairoMakie) : (@eval import WGLMakie)
        M=backend===:cairomakie ? CairoMakie.Makie : WGLMakie.Makie
        E=Base.get_extension(TamerOp,extension);H=E._HANDLERS
        field=V.ScalarField(([1.,3.,8.],[-4.,2.],[10.,20.,40.]),
            [100i+10j+k for i in 1:3,j in 1:2,k in 1:3];id="asymmetric",units="rank")
        session=V.field_session(field;planes=[(3,1)],indices=(1,1,1))
        initial=V.inspection_snapshot(session)
        state=E._visual_retained_figure(M,V,H,initial;size=(760,640))
        fig=state.figure;ax=only(state.panels).axis;plots=copy(ax.scene.plots)
        M.xlims!(ax,12.,35.);M.ylims!(ax,2.,7.)
        reader_limits=ax.limits[]
        for index in (2,3,1,2)
            V.select_inspection!(session;indices=(index,2,index))
            next=V.inspection_snapshot(session)
            @test E._visual_update_retained!(M,V,H,state,next)
            @test state.figure===fig && only(state.panels).axis===ax
            @test ax.limits[]==reader_limits
            @test all(a===b for (a,b) in zip(plots,ax.scene.plots))
            heatmap=only(p for p in ax.scene.plots if p isa M.Heatmap)
            @test heatmap[3][] == [100i+20+k for k in 1:3,i in 1:3]
            @test Tuple(heatmap.colorrange[]) == (111.,323.)
            markers=filter(p->p isa M.Scatter,ax.scene.plots)
            @test all(p->Tuple(only(p[1][]))==(field.axes[3][index],field.axes[1][index]),markers)
            @test E._visual_update_retained!(M,V,H,state,next) # unchanged selection
        end
        old=state.spec
        V.select_inspection!(session;planes=[(1,2),(2,3)])
        @test !E._visual_update_retained!(M,V,H,state,V.inspection_snapshot(session))
        @test state.spec===old && state.figure===fig
        empty!(fig);V.close_inspection!(session)
        # A new slice can lose its infinity lane: automatic ticks must replace
        # the previous explicit tick list, rather than preserve stale labels.
        tickspec(ticks)=V.VisualizationSpec(:tick_transition;
            layers=V.AbstractVisualizationLayer[V.PointLayer([(0.,1.)],:blue,1.,8.)],
            axes=V._default_axes_2d(xlimits=(-1.,2.),ylimits=(-1.,2.),xticks=ticks))
        tickstate=E._visual_retained_figure(M,V,H,tickspec(([0.,1.],["zero","one"])))
        @test E._visual_update_retained!(M,V,H,tickstate,tickspec(nothing))
        @test only(tickstate.panels).axis.xticks[]===M.automatic
        empty!(tickstate.figure)
        # Selection decorations appear/disappear without replacing any plot.
        matching=V.inspection_session([(0.,2.),(3.,6.),(9.,Inf)],[(0.,3.),(4.,7.),(10.,Inf)])
        drawing(s)=V.VisualizationSpec(:matching_canvas;panels=filter(p->get(p.metadata,:panel_style,nothing)!==:text_only,V.inspection_snapshot(s).panels),metadata=(;panel_columns=2))
        state=E._visual_retained_figure(M,V,H,drawing(matching);size=(1000,800))
        original=[copy(h.plots) for h in state.panels]
        for pair in (1,2,3,0,1)
            V.select_inspection!(matching;pair)
            @test E._visual_update_retained!(M,V,H,state,drawing(matching))
            @test all(all(a===b for (a,b) in zip(ps,h.plots)) for (ps,h) in zip(original,state.panels))
        end
        empty!(state.figure);V.close_inspection!(matching)
    end
end

@testset "A38 batched geometry preserves coordinates and drawing bounds" begin
    V = TamerOp.Visualization
    tree_count(plot) = 1 + sum(tree_count,plot.plots;init=0)
    for (backend, name) in ((:cairomakie,"CairoMakie"),(:wglmakie,"WGLMakie"))
        if Base.find_package(name) === nothing
            @test_skip false
            continue
        end
        backend===:cairomakie ? (@eval import CairoMakie) : (@eval import WGLMakie)
        M = backend===:cairomakie ? CairoMakie.Makie : WGLMakie.Makie
        plot_counts = Dict{DataType,Int}()
        for n in (1,128)
            segments=[(Float64(i),-2.,Float64(i)+.5,3.) for i in 1:n]
            rectangles=[(Float64(i),-2.,Float64(i)+.5,3.) for i in 1:n]
            paths=[[(Float64(i),0.),(Float64(i)+.2,1.),(Float64(i)+.5,0.)] for i in 1:n]
            textpositions=[(Float64(i),Float64(i%3)) for i in 1:n]
            faces=[[(Float64(i),0.,0.),(Float64(i)+.5,0.,0.),
                    (Float64(i)+.5,0.,2.),(Float64(i),0.,2.)] for i in 1:n]
            layers=V.AbstractVisualizationLayer[
                V.SegmentLayer(segments,:blue,.7,2.,:dash),
                V.RectLayer(rectangles,:blue,:black,.3,1.),
                V.PolygonLayer(paths,:blue,:black,.3,1.),
                V.PolylineLayer(paths,:blue,.7,2.,true),
                V.TextLayer(string.(1:n),textpositions,:black,13.),
                V.BarcodeLayer([(0.,1.),(2.,4.)],[n,2],:blue,3.,-1.,.5),
                V.Segment3Layer([(s[1],s[2],0.,s[3],s[4],2.) for s in segments],:blue,.7,2.),
                V.Polygon3Layer(faces,:blue,.3)]
            for layer in layers
                before=deepcopy(layer)
                spec=V.VisualizationSpec(:a38;layers=V.AbstractVisualizationLayer[layer])
                fig=V.render(spec;backend,size=(700,500))
                ax=only(b for b in fig.content if b isa M.Axis || b isa M.Axis3)
                plots=ax.scene.plots
                @test length(plots) <= (layer isa V.PolylineLayer ? 2 : 1)
                total = sum(tree_count,plots;init=0)
                if n == 1
                    plot_counts[typeof(layer)] = total
                else
                    # Include recipe children, not only the root plot objects.
                    @test total == plot_counts[typeof(layer)]
                end
                if layer isa V.SegmentLayer
                    p=only(plots)
                    @test p isa M.LineSegments
                    @test Tuple.(p[1][])==[(s[k],s[k+1]) for s in segments for k in (1,3)]
                    @test p.linestyle[]==M.to_linestyle(:dash)
                elseif layer isa V.Segment3Layer
                    @test Tuple.(only(plots)[1][])==[(s[k],s[k+1],s[k+2]) for s in layer.segments for k in (1,4)]
                elseif layer isa V.PolylineLayer
                    p=only(p for p in plots if p isa M.Lines)
                    @test count(q->any(isnan,q),p[1][])==n-1
                    @test [Tuple(q) for q in p[1][] if all(isfinite,q)]==reduce(vcat,paths)
                    closing=only(p for p in plots if p isa M.LineSegments)
                    @test Tuple.(closing[1][])==[p for path in paths for p in (last(path),first(path))]
                elseif layer isa V.TextLayer
                    p=only(plots)
                    @test p.input_text[]==string.(1:n)
                    @test Tuple.(p[1][])==textpositions
                elseif layer isa V.BarcodeLayer
                    expected=[(x,-1.0 +(i-1)*.5) for (i,iv) in enumerate(vcat(fill((0.,1.),n),fill((2.,4.),2))) for x in iv]
                    @test Tuple.(only(plots)[1][])==expected
                elseif layer isa V.Polygon3Layer
                    mesh=only(plots)[1][]
                    @test Tuple.(M.GeometryBasics.coordinates(mesh))==reduce(vcat,faces)
                    @test length(M.GeometryBasics.faces(mesh))==2n
                    @test all(f->length(unique(Tuple(f)))==3,M.GeometryBasics.faces(mesh))
                    for (i,face) in enumerate(M.GeometryBasics.faces(mesh))
                        # Each pair triangulates its own four vertices, never a
                        # neighboring face after concatenating mesh buffers.
                        offset=4*div(i-1,2)
                        @test all(v->offset+1<=convert(Int,v)<=offset+4,face)
                    end
                end
                @test all(k->isequal(getfield(layer,k),getfield(before,k)),fieldnames(typeof(layer)))
                empty!(fig)
            end
        end
        # Overlapping outlines keep painter order; no outlines can batch safely.
        for stroke in (0.,1.)
            layer=V.RectLayer([(0.,0.,2.,2.),(1.,1.,3.,3.)],:blue,:red,.5,stroke)
            fig=V.render(V.VisualizationSpec(:a38;layers=V.AbstractVisualizationLayer[layer]);backend)
            ax=only(b for b in fig.content if b isa M.Axis)
            @test length(ax.scene.plots)==(stroke==0 ? 1 : 2)
            empty!(fig)
        end
        # Label order survives identical anchors. WGL requires a uniform stroke;
        # Cairo draws an interleaved background and foreground for each label.
        layer=V.TextLayer(["first","second"],[(0.,0.),(0.,0.)],:blue,12.)
        fig=V.render(V.VisualizationSpec(:a38;layers=V.AbstractVisualizationLayer[layer],metadata=(;text_halo=true));backend)
        ax=only(b for b in fig.content if b isa M.Axis)
        p=only(ax.scene.plots)
        @test p.input_text[]==(backend===:wglmakie ? ["first","second"] : ["first","first","second","second"])
        @test p.strokewidth[]==(backend===:wglmakie ? 1. : [1.,0.,1.,0.])
        empty!(fig)
    end
end

@testset "A38 actual batched exports" begin
    V=TamerOp.Visualization
    for (backend,name) in ((:cairomakie,"CairoMakie"),(:wglmakie,"WGLMakie"))
        if Base.find_package(name)===nothing
            @test_skip false
            continue
        end
        backend===:cairomakie ? (@eval import CairoMakie) : (@eval import WGLMakie)
        panels=V.VisualizationSpec[
            V.VisualizationSpec(:a38;title="Independent edges and supports",
                metadata=(;text_halo=true),
                axes=V._default_axes_2d(xlimits=(.5,9.),ylimits=(-.1,1.5)),
                layers=V.AbstractVisualizationLayer[
                V.RectLayer([(Float64(i),0.,Float64(i)+.6,1.) for i in 1:8],:blue,:black,.3,1.),
                V.SegmentLayer([(Float64(i),0.,Float64(i)+.6,1.) for i in 1:8],:red,1.,2.,:dash),
                V.TextLayer(string.(1:8),[(Float64(i),1.2) for i in 1:8],:black,13.)]),
            V.VisualizationSpec(:a38;title="Vertical faces remain visible",layers=V.AbstractVisualizationLayer[
                V.Polygon3Layer([[(Float64(i),0.,0.),(Float64(i)+.6,0.,0.),
                    (Float64(i)+.6,0.,2.),(Float64(i),0.,2.)] for i in 1:4],:blue,.7)]),
            V.visual_spec([(0.,2.),(1.,3.),(2.,Inf)];kind=:persistence_diagram),
            V.VisualizationSpec(:matrix;title="Exact coefficients",layers=V.AbstractVisualizationLayer[
                V.MatrixLayer(["1/3" "0" "-1";"0" "2" "0";"1" "0" "1"],["a","b","c"],["x","y","z"])])]
        spec=V.VisualizationSpec(:a38_export;panels,metadata=(;panel_columns=2,figure_size=(1000,760)))
        mktempdir() do dir
            for ext in (backend===:cairomakie ? ("png","svg","pdf") : ("html",))
                path=joinpath(dir,"batched.$ext")
                V.save_visual(path,spec;backend)
                @test filesize(path)>1000
                if backend===:wglmakie && !isempty(get(ENV,"TAMEROP_BATCHED_HTML",""))
                    target=abspath(ENV["TAMEROP_BATCHED_HTML"])
                    mkpath(dirname(target));cp(path,target;force=true)
                end
                if !isempty(get(ENV,"TAMEROP_A38_GALLERY",""))
                    target=joinpath(ENV["TAMEROP_A38_GALLERY"],"batched.$ext")
                    mkpath(dirname(target));cp(path,target;force=true)
                end
            end
        end
    end
end

@testset "A38 public interval recipes stay bounded and retain selected records" begin
    V=TamerOp.Visualization
    for (backend,name) in ((:cairomakie,"CairoMakie"),(:wglmakie,"WGLMakie"))
        if Base.find_package(name)===nothing
            @test_skip false
            continue
        end
        backend===:cairomakie ? (@eval import CairoMakie) : (@eval import WGLMakie)
        M=backend===:cairomakie ? CairoMakie.Makie : WGLMakie.Makie
        raw=vcat([(0.,Float64(i)) for i in 1:128],[(0.,Inf),(0.,Inf)])
        plotcount=nothing
        tree(p)=1+sum(tree,p.plots;init=0)
        for n in (8,128)
            spec=V.visual_spec(raw[1:n];kind=:persistence_diagram,max_intervals=n)
            fig=V.render(spec;backend,size=(900,650))
            ax=only(b for b in fig.content if b isa M.Axis)
            @test count(p->p isa M.Text,ax.scene.plots)==1
            @test !any(p->p isa M.Annotation,ax.scene.plots)
            countall=sum(tree,ax.scene.plots;init=0)
            n==8 ? (plotcount=countall) : (@test countall==plotcount)
            @test length(only(p for p in ax.scene.plots if p isa M.Text).input_text[])==n
            empty!(fig)
        end
        for kind in (:barcode,:persistence_diagram)
            spec=V.visual_spec(raw;kind,window=(0.,130.),max_intervals=24,interval=129)
            records=deepcopy(spec.metadata.records)
            fig=V.render(spec;backend,size=(900,650))
            ax=only(b for b in fig.content if b isa M.Axis)
            @test spec.metadata.selected_interval==129
            @test 129 in spec.metadata.interval_ids
            @test spec.metadata.displayed_groups==24
            @test length(spec.metadata.records)==129
            @test last(spec.metadata.records).multiplicity==2
            @test count(p->p isa M.Scatter,ax.scene.plots)==1
            if kind===:barcode
                @test length(ax.scene.plots)<=3
                strokes=only(p for p in ax.scene.plots if p isa M.LineSegments)
                @test Tuple.(strokes[1][])==[(s[k],s[k+1]) for s in spec.metadata.interval_segments for k in (1,3)]
                @test strokes.linewidth[][1:end-2]==fill(2.5,46)
                @test strokes.linewidth[][end-1:end]==[5.,5.]
                marks=only(p for p in ax.scene.plots if p isa M.Scatter)
                # The selected essential group occupies the final displayed row;
                # its triangle stays at the bar midpoint, not at an endpoint.
                selected_segment=last(spec.metadata.interval_segments)
                @test Tuple(last(marks[1][]))==((selected_segment[1]+selected_segment[3])/2,24.)
                @test last(marks.markersize[])==M.Vec2f(12)
                @test last(marks.marker[])==M.to_spritemarker(:utriangle)
            end
            @test spec.metadata.records==records
            empty!(fig)
        end
    end
end

@testset "A38 matrix batching retains exact text, measured spacing and styles" begin
    V=TamerOp.Visualization
    for (backend,name) in ((:cairomakie,"CairoMakie"),(:wglmakie,"WGLMakie"))
        if Base.find_package(name)===nothing
            @test_skip false
            continue
        end
        backend===:cairomakie ? (@eval import CairoMakie) : (@eval import WGLMakie)
        M=backend===:cairomakie ? CairoMakie.Makie : WGLMakie.Makie
        for n in (1,12), fontsize in (16.,24.)
            entries=[i==j ? "123456789/987654321" : "-1" for i in 1:n,j in 1:n]
            layer=V.MatrixLayer(entries,["target $i\nsecond line" for i in 1:n],["source $j" for j in 1:n])
            style=V.VisualStyle(;fontsize,palette=:grayscale)
            spec=V.VisualizationSpec(:matrix;layers=V.AbstractVisualizationLayer[layer],
                metadata=(;row_roles=fill(:target,n),column_roles=fill(:source,n)))
            fig=V.render(spec;backend,style,size=(1200,900))
            M.update_state_before_display!(fig)
            @test length(fig.content)==1
            records=visual_text_records(fig,M)
            nonempty=filter(r->!isempty(r.text),records)
            @test length(nonempty)==(n+1)^2
            @test count(r->r.text=="123456789/987654321",nonempty)==n
            @test count(r->occursin("[source]",r.text),nonempty)==n
            @test count(r->occursin("[target]",r.text),nonempty)==n
            table=only(unique(r.plot for r in nonempty))
            boxes=M.raw_string_boundingboxes(table)
            points=table[1][]
            for i in eachindex(points),j in 1:i-1
                a,b=boxes[i],boxes[j]
                alo=minimum(a)[1:2]+points[i];ahi=maximum(a)[1:2]+points[i]
                blo=minimum(b)[1:2]+points[j];bhi=maximum(b)[1:2]+points[j]
                @test ahi[1]<=blo[1] || bhi[1]<=alo[1] || ahi[2]<=blo[2] || bhi[2]<=alo[2]
            end
            entry=first(r for r in nonempty if r.text=="123456789/987654321")
            @test M.to_font(entry.font)==M.to_font(style.mono_font)
            @test entry.fontsize==(n>8 ? 12 : 14)*fontsize/16
            @test layer.entries==entries
            empty!(fig)
        end
    end
end
