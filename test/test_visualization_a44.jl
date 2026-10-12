function _a44_fields()
    V=TamerOp.Visualization
    ax=([2.,5.],[-3.,0.,4.],[10.,20.,30.,40.],[7.,9.])
    values=[1000i+100j+10k+l+10000h for i in 1:2,j in 1:3,k in 1:4,l in 1:2,h in 1:2]
    a=V.ScalarField(ax,values;id="A",layers=["H0","H1"],units="rank",axis_labels=("a","b","c","d"))
    b=V.ScalarField(ax,values .- 3;id="B",layers=["H0","H1"],units="rank",axis_labels=("a","b","c","d"))
    a,b
end
@testset "A44 controlled fields and independent coordinate oracle" begin
    V=TamerOp.Visualization;a,b=_a44_fields();fields=[a,b]
    @test TamerOp.ScalarField===V.ScalarField
    @test V.available_visuals(a)==(:coordinate_sections,)
    for plane in ((1,2),(2,1),(1,3),(3,1),(1,4),(4,1),(2,3),(3,2),(2,4),(4,2),(3,4),(4,3)),h in 1:2
        o=(;planes=[plane],indices=(2,3,4,1),layer="H$(h-1)")
        spec=V.visual_spec(fields;o...)
        @test V.check_visual_spec(spec).valid
        @test spec.metadata.scale.limits==(11108.,22342.)
        for (q,p) in enumerate(spec.panels)
            heat=only(filter(l->l isa V.HeatmapLayer,p.layers))
            @test size(heat.values)==(length(a.axes[plane[2]]),length(a.axes[plane[1]]))
            for j in eachindex(heat.y),i in eachindex(heat.x)
                indices=[2,3,4,1];indices[plane[1]]=i;indices[plane[2]]=j
                expected=sum(indices.*[1000,100,10,1])+10000h-(q-1)*3
                @test heat.values[j,i]==expected
            end
            @test p.metadata.selected_value==sum([2,3,4,1].*[1000,100,10,1])+10000h-(q-1)*3
            @test Tuple(x.axis for x in p.metadata.fixed)==Tuple(d for d in 1:4 if !(d in plane))
            @test occursin("layer H$(h-1)",p.subtitle)
        end
    end
    diff=V.visual_spec(fields;baseline="A",planes=[(3,1)],indices=(1,2,3,2))
    @test diff.metadata.scale.limits==(-3.,3.)
    @test diff.metadata.scale.normalization==:symmetric
    @test all(==(-3.),diff.panels[2].layers[1].values)
    @test_throws ArgumentError V.visual_spec(fields;baseline="A",normalization=:sequential)
    @test_throws ArgumentError V.visual_spec(fields;baseline="A",colorrange=(-2.,3.))
    hole=copy(b.values);hole[1,2,3,2,1]=NaN
    c=V.ScalarField(a.axes,hole;id="C",layers=a.layers,units=a.units,axis_labels=a.axis_labels)
    masked=V.visual_spec([a,c];baseline="A",planes=[(3,1)],indices=(1,2,3,2))
    @test isnan(masked.panels[2].metadata.selected_value)
    @test masked.panels[2].metadata.available==7
    @test occursin("unavailable",masked.panels[2].subtitle)
    @test V.visual_spec(V.ScalarField(([0.],[0.]),fill(NaN,1,1))).metadata.scale.limits==(0.,1.)
    # A linear scalar field has contour x+2y=2: test each segment endpoint.
    contour=V.visual_spec(V.ScalarField(([0.,1.,3.],[-1.,1.,2.]),[x+2y for x in [0.,1.,3.],y in [-1.,1.,2.]]);
        contours=[2.])
    seg=only(filter(l->l isa V.SegmentLayer,contour.panels[1].layers)).segments
    @test !isempty(seg)
    for level in (-20.,-10.,0.,0.5,1.,1.5,2.,10.,20.)
        @test V.check_visual_spec(V.visual_spec(a;planes=[(1,2)],contours=[level])).valid
    end
    with=V.visual_spec(a;planes=[(1,2)],contours=[1.,2.]).panels[1].legend.entries[2]
    alone=V.visual_spec(a;planes=[(1,2)],contours=[2.]).panels[1].legend.entries[1]
    @test with==alone
    for s in seg;@test isapprox(s[1]+2s[2],2);@test isapprox(s[3]+2s[4],2);end
    for kwargs in ((planes=[(1,1)],),(planes=[(1,5)],),(planes=[(true,2)],),(indices=(1,2),),
            (layer="absent",),(normalization=:log,),(contours=[2.,1.],),(contours=[Inf],),(max_cells=1,),
            (colorrange=(0,0),),(baseline="absent",),(unknown=1,))
        @test_throws Exception V.visual_spec(fields;kwargs...)
    end
    @test_throws ArgumentError V.visual_spec([a,a])
    @test_throws ArgumentError V.visual_spec([a,V.ScalarField(a.axes,b.values;id="C",layers=a.layers,units="seconds",axis_labels=a.axis_labels)])
    @test_throws ArgumentError V.ScalarField(([0,0],[0,1]),zeros(2,2))
    @test_throws ArgumentError V.ScalarField(([0,1],[0,1]),zeros(2,3))
    @test_throws ArgumentError V.ScalarField(([0,1],[0,1]),fill(Inf,2,2))
    @test_throws ArgumentError V.ScalarField(([big(2)^60,big(2)^60+1],[0,1]),zeros(2,2))
    s=TamerOp.field_session(fields;planes=[(3,1),(2,4)],indices=(1,2,3,2),layer="H0")
    notified=Ref(0);token=V._on_inspection(s,_ -> (notified[]+=1))
    scale=V.inspection_summary(s).scale
    V.select_inspection!(s;indices=(2,1,4,1),layer="H1")
    @test notified[]==1
    @test V.inspection_summary(s).scale==scale
    @test V.inspection_snapshot(s).panels[1].metadata.selected_value==22141.
    old=V.inspection_snapshot(s)
    @test !V.check_inspection_selection(s;indices=(0,1,1,1)).valid
    @test_throws ArgumentError V.select_inspection!(s;indices=(0,1,1,1))
    @test V.inspection_snapshot(s)===old
    @test V.check_inspection_session(s).valid
    @test V.check_visual_spec(V.visual_spec(s)).valid
    # Selections cannot be changed by a listener during a notification.
    nested=V._on_inspection(s,_ -> V.select_inspection!(s;layer="H0"))
    V.reset_inspection!(s)
    @test !isempty(V.inspection_summary(s).listener_errors)
    V._off_inspection!(s,nested)
    @test V.inspection_selection(s).indices==(1,2,3,2)
    @test V.inspection_snapshot(s).panels[1].metadata.selected_value==11232.
    V.close_inspection!(s);V.close_inspection!(s)
    @test s.fields===nothing&&isempty(s.listeners)
    @test V.check_inspection_session(s).valid
    @test_throws ArgumentError V.select_inspection!(s;layer="H1")
end

@testset "A44 ambient landscape interpolation, geometry and errors" begin
    I=TamerOp.MultiparameterImages;V=TamerOp.Visualization
    # One interval [0,4] on each declared line, and its empty second layer.
    tg=collect(0.:1.:4.);vals=zeros(1,3,2,5)
    for j in 1:3,t in eachindex(tg);vals[1,j,1,t]=min(tg[t],4-tg[t]);end
    L=I.MPLandscape(2,tg,vals,fill(1/3,1,3),[[2.,2.]],[-1.,0.,1.])
    bases=[(0.,-2.),(0.,0.),(0.,2.)]
    args=(;axes=([0.,1.,2.,3.,4.,8.,9.],[-2.,0.,1.,2.,3.,4.,6.,8.,9.]),basepoints=bases,source_window=(0.,4.),sample_error=0.)
    g=TamerOp.ambient_landscape(L;args...)
    for i in eachindex(g.axes[1]),j in eachindex(g.axes[2])
        x,y=g.axes[1][i],g.axes[2][j];covered=(y-x in (-2.,0.,2.))&&0<=x<=8
        @test g.coverage[i,j]==covered
        if covered
            @test isapprox(g.values[i,j,1],min(x,8-x))
            @test g.values[i,j,2]==0
            @test g.errors[i,j]==(isinteger(x/2) ? 0. : 1.)
        else
            @test isnan(g.values[i,j,1])&&isnan(g.errors[i,j])
        end
    end
    @test I.describe(g).interpretation==:diagonal_landscape
    @test I.describe(g).height_factor==2.
    @test I.check_ambient_landscape(g).valid
    @test I.ambient_coverage(g)===g.coverage
    @test I.ambient_errors(g)===g.errors
    @test I.ambient_landscape_summary(g)==I.describe(g)
    @test I.landscape_layers(g)==2
    @test V.check_visual_spec(V.visual_spec(V.ScalarField(g))).valid
    # Reparameterizing t -> 2t changes stored direction and heights together.
    L2=I.MPLandscape(2,2tg,2vals,ones(1,3),[[1.,1.]],[-1.,0.,1.])
    g2=I.ambient_landscape(L2;merge(args,(;source_window=(0.,8.)))...)
    @test isequal(g.values,g2.values)
    @test isequal(g.errors,g2.errors)
    unknown=I.ambient_landscape(L;merge(args,(;sample_error=nothing))...)
    @test all(isnan,unknown.errors)
    @test I.describe(unknown).error_bound==:unknown
    clipped=I.ambient_landscape(L;merge(args,(;source_window=(1.,3.)))...)
    @test all(!,clipped.coverage[1,:])
    inside=I.ambient_landscape(L;merge(args,(;axes=([1.],[1.]),source_window=(0.25,4.)))...)
    @test !only(inside.coverage) # Interpolation may not borrow a sample outside the source window.
    for kwargs in ((direction_index=2,),(sample_error=-1,),(max_points=1,),(line_tolerance=-1,),
            (basepoints=[(0,0)],),(basepoints=[(0,0),(1,1),(0,2)],),(source_window=(4,0),),
            (axes=([0,0],[0,1]),))
        @test_throws ArgumentError I.ambient_landscape(L;merge(args,kwargs)...)
    end
    # Unequal positive slopes are explicitly directional, not diagonal.
    other=I.MPLandscape(2,tg,vals,ones(1,3),[[2.,1.]],[-1.,0.,1.])
    @test I.describe(I.ambient_landscape(other;args...)).interpretation==:directional_landscape
end

@testset "A44 native coordinate sections and exports" begin
    V=TamerOp.Visualization;a,b=_a44_fields()
    spec=V.visual_spec([a,b];planes=[(3,1),(2,4)],indices=(1,2,3,2),layer="H1",contours=[22000.])
    for package in (:CairoMakie,:WGLMakie)
        if Base.find_package(string(package))===nothing;@test_skip false;continue;end
        @eval import $package
        mk=getfield(@__MODULE__,package);backend=package===:CairoMakie ? :cairomakie : :wglmakie
        fig=V.render(spec;backend)
        try
            mk.Makie.update_state_before_display!(fig)
            axs=filter(x->x isa mk.Axis,fig.content)
            @test length(axs)==4
            for (ax,p) in zip(axs,spec.panels)
                heat=only(filter(x->x isa mk.Heatmap,ax.scene.plots))
                @test heat[3][]==permutedims(p.layers[1].values)
                @test Tuple(heat.colorrange[])==spec.metadata.scale.limits
                @test ax.xlabel[]==p.axes.xlabel&&ax.ylabel[]==p.axes.ylabel
                @test any(block->block isa mk.Label&&block.text[]==p.subtitle,fig.content)
            end
            if package===:CairoMakie
                mktempdir() do dir
                    for suffix in ("png","svg","pdf")
                        path=joinpath(dir,"sections."*suffix);mk.save(path,fig);@test filesize(path)>1000
                    end
                end
            end
        finally;empty!(fig);end
        # Thin grids and wholly unavailable fields are valid scalar inputs.
        for values in (fill(NaN,1,1),zeros(1,1),reshape([2.,3.],1,2))
            thin=V.ScalarField((collect(1.:size(values,1)),collect(1.:size(values,2))),values)
            thinfig=V.render(V.visual_spec(thin);backend)
            try
                mk.Makie.update_state_before_display!(thinfig)
                @test length(filter(a->a isa mk.Axis,thinfig.content))==1
            finally;empty!(thinfig);end
        end
    end
end
