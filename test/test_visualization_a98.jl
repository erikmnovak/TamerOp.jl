# Supplied asymmetric feature values expose row/column transposition immediately.
@testset "A98 native rectangular features and supplied embedding" begin
    V=TamerOp.Visualization
    collection=V.visual_collection(;sample_ids=["a","b","c"],features=[11 23;37 41;59 67],
        feature_names=["first","second"],definitions=["First coordinate","Second coordinate"],
        embedding=(coordinates=[-2. 5.;1. -3.;4. 7.],method="Supplied coordinates",settings=(;),seed=nothing))
    spec=V.visual_spec(collection;sample="b",target="c",feature="second")
    for backend in (:cairomakie,:wglmakie)
        package=backend===:cairomakie ? :CairoMakie : :WGLMakie
        if Base.find_package(string(package))===nothing
            @test_skip false
            continue
        end
        @eval import $package
        mk=getfield(@__MODULE__,package)
        fig=V.render(spec;backend)
        try
            mk.Makie.update_state_before_display!(fig)
            axes=filter(a->a isa mk.Axis,fig.content)
            @test length(axes)==2
            matrix=only(filter(a->a.xlabel[]=="Feature",axes))
            hm=only(filter(p->p isa mk.Heatmap,matrix.scene.plots))
            @test hm[3][]==[11. 37. 59.;23. 41. 67.]
            @test collect(hm[1][])==[0.5,1.5,2.5]
            @test collect(hm[2][])==[0.5,1.5,2.5,3.5]
            embedding=only(filter(a->a.xlabel[]=="Coordinate 1",axes))
            points=filter(p->p isa mk.Scatter,embedding.scene.plots)
            @test length(points)==3
            @test Tuple.(points[1][1][])==[(-2.,5.),(1.,-3.),(4.,7.)]
            @test Tuple.(points[2][1][])==[(4.,7.)] # Target c.
            @test Tuple.(points[3][1][])==[(1.,-3.)] # Selected b.
            @test all(a->all(>(0),mk.Makie.widths(mk.Makie.viewport(a.scene)[])),axes)
            if backend===:cairomakie
                mktempdir() do dir
                    for extension in ("png","svg","pdf")
                        path=joinpath(dir,"collection."*extension)
                        mk.save(path,fig)
                        @test filesize(path)>1000
                    end
                end
            end
        finally
            empty!(fig)
        end
    end
end

@testset "A98 supplied collection contracts and exact readouts" begin
    V=TamerOp.Visualization
    X=[1//3 2//3; 5//7 -11//13;0//1 4//1]
    D=[0. 2. Inf;2. 0. 3.;Inf 3. 0.]
    K=[1 -2 0;-2 1 1;0 1 1] # Symmetric, deliberately indefinite.
    c=V.visual_collection(;sample_ids=["a","b","c"],features=X,feature_names=["x","y"],
        definitions=["First supplied coordinate","Second supplied coordinate"],distances=D,
        distance_semantics="Declared pair dissimilarity",kernel=K,kernel_semantics="Supplied indefinite similarity",
        embedding=(coordinates=[0. 0.;0. 0.;1. 5.],method="Supplied projection",settings=(axes=(1,2),),seed=nothing))
    @test TamerOp.visual_collection===V.visual_collection
    @test TamerOp.Advanced.CollectionInspectionSession===V.CollectionInspectionSession
    @test V.describe(c).all_pairs_computed===false
    @test !V.describe(c).kernel_psd_certified
    @test !V.check_visual_request(c;sample=999).valid
    for view in (:features,:distances,:kernel)
        spec=V.visual_spec(c;view,sample="b",target="c",feature="y")
        @test V.check_visual_spec(spec).valid
        @test spec.metadata.selected.value==c.tables[view][2,view===:features ? 2 : 3]
        @test first(first(spec.panels).layers).values[2,1]==Float64(c.tables[view][2,1])
    end
    @test V.visual_spec(c;sample=2,feature=2).metadata.selected.value===-11//13
    @test isnan(first(first(V.visual_spec(c;view=:distances).panels).layers).values[1,3])
    @test "+Inf" in V.visual_spec(c;view=:distances).panels[1].layers[2].labels
    @test c.embedding.coordinates[1,:]==c.embedding.coordinates[2,:] # Colliding points retain distinct IDs.
    @test occursin("3 samples",sprint(show,c))
    fs=TamerOp.Featurizers.FeatureSet(X,[:x,:y],["a","b","c"],(;normalization=:none))
    fc=V.visual_collection(fs;definitions=["x coordinate","y coordinate"])
    @test !haskey(fc.context.invariant_settings.feature_set,"session_cache")
    @test fc.tables.features==X && fc.context.normalization=="none" && fc.context.invariant_settings.feature_set["ids"]==["a","b","c"]
    X[1,1]=9//1
    @test fc.tables.features[1,1]==1//3 # Constructor snapshot.
    minimal=(sample_ids=["a","b"],distances=[0. 1.;1. 0.],distance_semantics="Declared distance")
    for changes in ((;sample_ids=["a","a"]),(;max_samples=1),(;max_cells=3),
            (;distances=[1. 1.;1. 0.]),(;distances=[0. -1.;-1. 0.]),
            (;distances=[0. 1.;2. 0.]),(;distances=[0. NaN;NaN 0.]),
            (;distances=[0. -Inf;-Inf 0.]),(;distance_semantics=""),
            (;embedding=(coordinates=ones(2,3),method="x",settings=(;),seed=0)),
            (;samples=[(;unknown=1),(;)]),(;degree=-1),(;normalization=""))
        @test_throws ArgumentError V.visual_collection(;merge(minimal,changes)...)
    end
    for A in (reshape([true,false],2,1),reshape([Inf,1.],2,1),reshape([NaN,1.],2,1))
        @test_throws ArgumentError V.visual_collection(;sample_ids=["a","b"],features=A,feature_names=["x"],definitions=["x"])
    end
    @test_throws ArgumentError V.visual_collection(;sample_ids=["a"],features=ones(1,1),feature_names=["x"])
    @test_throws ArgumentError V.visual_collection(;sample_ids=["a"])
    masked=V.visual_collection(;sample_ids=["a","b"],distances=[0 missing;missing 0],distance_semantics="Incomplete distances")
    @test ismissing(V.visual_spec(masked).metadata.selected.value)
    @test V.check_visual_spec(V.visual_spec(masked)).valid
    @test_throws ArgumentError V.visual_collection(;sample_ids=["a","b"],kernel=[1 Inf;Inf 1],kernel_semantics="bad")
end

@testset "A98 selection, sample linkage, pair witnesses and lifecycle" begin
    V=TamerOp.Visualization;SI=TamerOp.SliceInvariants
    barsA=[(0.,2.)];barsB=[(0.,4.)]
    witness=SI.bottleneck_matching(barsA,barsB)
    invariant=V.visual_spec(barsA)
    c=V.visual_collection(;sample_ids=["a","b"],features=[2;4;;],feature_names=["lifetime"],
        definitions=["Death minus birth for the single finite interval"],distances=[0. 2.;2. 0.],
        distance_semantics="Bottleneck distance",degree=0,field="F2",samples=[
            (;object=barsA,invariant,source=invariant,metadata=(window=(0,2),)),(;object=barsB,metadata=(window=(0,4),))],
        witnesses=Dict(("a","b")=>witness))
    s=V.inspection_session(c)
    @test V.check_visual_spec(V.visual_spec(s)).valid
    @test length(V.inspection_snapshot(s).panels)==4
    @test V.collection_sample(s).id=="a"
    old=V.inspection_selection(s);oldspec=V.inspection_snapshot(s)
    @test !V.check_inspection_selection(s;sample=0).valid
    @test_throws ArgumentError V.select_inspection!(s;sample=3)
    @test V.inspection_selection(s)==old && V.inspection_snapshot(s)===oldspec
    events=Ref(0);token=V._on_inspection(s,_ -> events[]+=1)
    V.select_inspection!(s;view=:distances,sample="b",target="a")
    @test events[]==1 && V.collection_sample(s).id=="b"
    @test length(V.inspection_snapshot(s).panels)==2
    @test V.inspection_snapshot(s).metadata.selected.sample_metadata.window==(0,4)
    @test_throws ArgumentError V.collection_matching(s) # No reversal of witness assignment.
    child=V.collection_matching(s;compute=true)
    @test child.payload.matching.distance==2
    V.close_inspection!(child)
    V.reset_inspection!(s)
    linked=V.collection_matching(s)
    @test linked.payload.matching.distance==2
    @test V.inspection_summary(s).all_pairs_computed===false
    @test V.check_inspection_session(s).valid
    V.close_inspection!(s)
    @test V.check_inspection_session(s).valid
    @test s.object===nothing && isempty(s.listeners)
    @test !V.inspection_summary(linked).closed
    @test V.check_visual_spec(V.inspection_snapshot(s)).valid
    @test_throws ArgumentError V.collection_sample(s)
    @test_throws ArgumentError V.collection_matching(s)
    @test_throws ArgumentError V.select_inspection!(s;sample=1)
    V.close_inspection!(s);V.close_inspection!(linked)
end

@testset "A98 masks, no hidden queries, rollback and nested sample views" begin
    V=TamerOp.Visualization
    # The retained object need not even implement a mathematical query API.
    c=V.visual_collection(;sample_ids=["a","b"],features=[missing -1//3;4//7 missing],
        feature_names=["p","q"],definitions=["p","q"],samples=[(;object=:unqueried),(;object=:unqueried)])
    s=V.inspection_session(c)
    @test ismissing(V.inspection_snapshot(s).metadata.selected.value)
    @test V.check_inspection_session(s).valid
    V.select_inspection!(s;sample=2,feature=1)
    @test V.inspection_snapshot(s).metadata.selected.value==4//7
    @test !V.inspection_summary(s).all_pairs_computed
    for kw in ((view=:kernel,),(target=0,),(feature="absent",),(sample=true,),(foo=1,))
        before=V.inspection_selection(s)
        @test !V.check_inspection_selection(s;kw...).valid
        @test_throws ArgumentError V.select_inspection!(s;kw...)
        @test V.inspection_selection(s)==before
    end
    token=V._on_inspection(s,_ -> error("deliberate listener failure"))
    V.select_inspection!(s;sample=1)
    @test !isempty(V.inspection_summary(s).listener_errors)
    V._off_inspection!(s,token);V.close_inspection!(s)
    a=V.visual_spec([(0.,1.)]);b=V.visual_spec([(1.,2.)])
    composite=V.VisualizationSpec(:supplied;panels=[a,b])
    c=V.visual_collection(;sample_ids=["a"],kernel=ones(1,1),kernel_semantics="Unit similarity",
        samples=[(;invariant=composite)])
    spec=V.visual_spec(c)
    @test length(spec.panels)==4 && all(isempty(p.panels) for p in spec.panels)
    @test V.check_visual_spec(spec).valid
    @test_throws ArgumentError V.collection_matching(c;sample=1,target=1)
    @test_throws ArgumentError V.visual_collection(;sample_ids=["a","b"],distances=[0 missing;1 0],distance_semantics="bad masks")
    @test_throws ArgumentError V.visual_collection(;sample_ids=["a"],features=ones(1,1),feature_names=["f"],definitions=[""])
    @test_throws ArgumentError V.visual_collection(;sample_ids=["a"],kernel=ones(1,1),kernel_semantics="k",embedding=(coordinates=ones(1,2),method="m",settings=(;),seed=true))
end

@testset "A98 feature metadata and one-interval bottleneck oracle" begin
    V=TamerOp.Visualization;F=TamerOp.Featurizers
    spec=F.RankGridSpec(;nvertices=2)
    cache=Ref(:must_not_be_retained)
    fs=F.FeatureSet([1 1 1;1 0 0],[:r11,:r12,:r22],["a","b"],(;spec,session_cache=cache,normalization="none"))
    c=V.visual_collection(fs;definitions=["rank 1 to 1","rank 1 to 2","rank 2 to 2"])
    metadata=V.describe(c).context.invariant_settings.feature_set
    @test metadata["spec"]["type"]=="RankGridSpec"
    @test metadata["spec"]["fields"]["nvertices"]==2
    @test !haskey(metadata,"session_cache")
    @test_throws ArgumentError V.visual_collection(fs;definitions=["a","b","c"],invariant_settings=(feature_set=1,))
    bars=[(0.,2.),(0.,3.),(2.,4.)]
    D=[0. 1. 1.;1. 0. 1.5;1. 1.5 0.]
    for i in 1:3,j in 1:3
        a,b=bars[i],bars[j]
        # Either match the two points or send both to the diagonal.
        oracle=min(max(abs(a[1]-b[1]),abs(a[2]-b[2])),max((a[2]-a[1])/2,(b[2]-b[1])/2))
        @test D[i,j]==oracle
    end
end
