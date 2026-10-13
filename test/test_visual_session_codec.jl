# Lossless drawing/data interchange, independent of any optional renderer.
function _a100_codec_roundtrip(value;kwargs...)
    S=TamerOp.Serialization
    return S.visual_data_from_json(S.JSON3.read(S.JSON3.write(S.visual_data_to_json(value;kwargs...)));kwargs...)
end

@testset "A100 declarative exact values and fields" begin
    S=TamerOp.Serialization; C=TamerOp.CoreModules; A=TamerOp.ExactReals.AlgebraicReal
    values=(nothing,missing,true,'q',:radius,"a </script> label",Int8(-7),UInt128(2)^100,
        big(2)^170,big(1)//big(3),Int16(2)//Int16(7),-0.0,Float32(1.25),Float16(0.5),
        Inf,-Inf,NaN,sqrt(A(2)),BigFloat("1.234567890123456789";precision=211))
    for value in values
        reopened=_a100_codec_roundtrip(value)
        @test typeof(reopened)===typeof(value)
        @test isequal(reopened,value)
        value isa BigFloat && @test precision(reopened)==211
    end
    for field in (C.QQField(),C.F2(),C.F3(),C.Fp(101),C.RealField(Float32;rtol=0.01f0,atol=0.001f0),
            C.RealField(BigFloat;rtol=BigFloat("0.01";precision=113),atol=BigFloat(0;precision=113)))
        reopened=_a100_codec_roundtrip(field)
        @test typeof(reopened)===typeof(field)
        @test reopened==field
    end
    for p in (2,3,101)
        x=C.coerce(C.Fp(p),p-1)
        @test _a100_codec_roundtrip(x)===x
        M=reshape([C.coerce(C.Fp(p),i) for i in 0:5],2,3)
        @test _a100_codec_roundtrip(M)==M
        @test eltype(_a100_codec_roundtrip(M))===eltype(M)
    end
    @test _a100_codec_roundtrip((x=1//3,y=(2,3),kind=:exact)).x==1//3
    @test _a100_codec_roundtrip(Tuple)===Tuple
    @test _a100_codec_roundtrip(Tuple{})===Tuple{}
    @test_throws ArgumentError S.visual_data_to_json(identity)
    @test_throws ArgumentError S.visual_data_to_json((nested=(callback=identity,),))
    @test_throws ArgumentError S.visual_data_to_json(Ref(3))
end

@testset "A100 typed arrays, sparse matrices and layout ranges" begin
    S=TamerOp.Serialization
    values=(zeros(0,3),reshape(Any[],2,0,4),reshape([1//2],()),
        [(context=(;),observation=(;))],Tuple{}[()],
        Union{Nothing,Int}[nothing,2],[(label=:one,ids=(1,2)),(label=:two,ids=(3,4))],
        NamedTuple{(:x,:y)}[(x=1,y=:a),(x=1//2,y=(1,2))],
        Dict{Symbol,Any}(:a=>[1,2],:b=>(0//1,2//3)),Int32(2):Int32(4),2:1,1:3:10,
        LinRange(1.,2.,3),BitMatrix([true false;false true]))
    for value in values
        restored=_a100_codec_roundtrip(value)
        @test isequal(restored,value)
        @test eltype(restored)===eltype(value)
        value isa AbstractArray && @test size(restored)==size(value)
    end
    for M in (sparse([1,2],[2,1],[1//3,2//5],3,4),spzeros(Rational{BigInt},0,3),
            sparse([1,2],[1,1],[TamerOp.CoreModules.coerce(TamerOp.CoreModules.F3(),2),TamerOp.CoreModules.coerce(TamerOp.CoreModules.F3(),1)],2,2))
        restored=_a100_codec_roundtrip(M)
        @test restored isa SparseMatrixCSC
        @test restored==M
        @test restored.colptr==M.colptr
        @test restored.rowval==M.rowval
        @test restored.nzval==M.nzval
    end
    data=S.visual_data_to_json([1 2;3 4]); data["size"]=[1,5]
    @test_throws ArgumentError S.visual_data_from_json(data)
    data=S.visual_data_to_json(Dict(:x=>1));push!(data["entries"],first(data["entries"]))
    @test_throws ArgumentError S.visual_data_from_json(data)
    data=S.visual_data_to_json((x=1,y=2));data["names"]=["x","x"]
    @test_throws ArgumentError S.visual_data_from_json(data)
    data=S.visual_data_to_json(sparse([1,2],[1,1],[1,2],2,2))
    data["row_indices"]=S.visual_data_to_json([2,1])
    @test_throws ArgumentError S.visual_data_from_json(data)
    forward=Dict("entry$i"=>i for i in 1:80)
    reverse=Dict("entry$i"=>i for i in 80:-1:1)
    @test S.visual_data_to_json(forward)==S.visual_data_to_json(reverse)
    data=S.visual_data_to_json(sparse([1,2],[1,1],[1,2],2,2))
    data["column_pointers"]=S.visual_data_to_json([0,3,3])
    @test_throws ArgumentError S.visual_data_from_json(data)
end

@testset "A100 layer and composition preservation" begin
    V=TamerOp.Visualization; S=TamerOp.Serialization
    role=V._VisualRole(:source)
    layers=V.AbstractVisualizationLayer[
        V.HeatmapLayer([0.,1.],[0.,1.],[1. NaN;2. 3.],:viridis,0.7,"dimension",true),
        V.RectLayer([(0.,0.,1.,2.)],role,:black,0.2,2.),
        V.PolygonLayer([[(0.,0.),(1.,0.),(0.,1.)]],role,:black,0.2,2.),
        V.Polygon3Layer([[(0.,0.,0.),(1.,0.,0.),(0.,1.,0.)]],role,0.2),
        V.SegmentLayer([(0.,0.,1.,1.)],role,0.7,2.,:dash),
        V.PolylineLayer([[(0.,0.),(1.,1.)]],role,0.7,2.,false),
        V.PointLayer([(0.,0.)],[2.],0.7,10.,:viridis,:pixel),
        V.Point3Layer([(0.,0.,0.)],role,0.7,10.,:viridis),
        V.TextLayer(["exact 1/3"],[(0.,0.)],role,12.),
        V.MatrixLayer(reshape(["1/3","-2/5"],1,2),["target"],["source 1","source 2"]),
        V.Segment3Layer([(0.,0.,0.,1.,1.,1.)],role,0.7,2.),
        V.BarcodeLayer([(0.,1.)],[2],role,3.,0.,1.)]
    for layer in layers
        restored=_a100_codec_roundtrip(layer)
        @test typeof(restored)===typeof(layer)
        for name in fieldnames(typeof(layer))
            a,b=getfield(restored,name),getfield(layer,name)
            @test isequal(S.visual_data_to_json(a),S.visual_data_to_json(b))
        end
    end
    style=V.VisualStyle(palette=:grayscale,fontsize=18,linewidth_scale=1.2,colors=(source="#112233",))
    @test _a100_codec_roundtrip(style)==style
    context=V.VisualContext(domain="parameter plane",units=("r","k"),field="F3",degree=1,provenance="sample 17")
    @test _a100_codec_roundtrip(context)==context
    spec=V.visual_panel([1 2;3 4];x=[0,1],y=[0,1],quantity="rank",context)
    annotation=V.VisualAnnotation(V.VisualAnchor(:left;point=(0.,0.)),V.VisualAnchor(:right;point=(0.,0.));label="same query")
    composed=V.compose_visuals((left=spec,right=spec);annotations=[annotation])
    composed=V._composition_copy(composed;metadata=merge(composed.metadata,(;panel_positions=[(1:1,1:1),(1:1,2:2)])))
    restored=_a100_codec_roundtrip(composed)
    @test V.check_visual_spec(restored).valid
    @test isequal(S.visual_data_to_json(restored),S.visual_data_to_json(composed))
    @test restored.metadata.panel_positions==composed.metadata.panel_positions
    @test restored.panels[1].layers[1].values==[1 2;3 4]
    @test restored.panels[1].metadata.composition_context==context
end

@testset "A100 strict data decoding and bounded resources" begin
    S=TamerOp.Serialization
    @test_throws ArgumentError S.visual_data_from_json(Dict("tag"=>"run","code"=>"error()"))
    @test_throws ArgumentError S.visual_data_from_json(Dict("tag"=>"type","name"=>"Function","parameters"=>[]))
    @test_throws ArgumentError S.visual_data_from_json(Dict("tag"=>"qq_field","extra"=>1))
    @test_throws ArgumentError S.visual_data_from_json(Dict("tag"=>"prime_scalar","characteristic"=>3,"value"=>3))
    @test_throws ArgumentError S.visual_data_from_json(Dict("tag"=>"prime_field","characteristic"=>4))
    @test_throws ArgumentError S.visual_data_from_json(Dict("tag"=>"scalar","value"=>S._pd_encode([1,2])))
    @test_throws ArgumentError S.visual_data_to_json((a=(b=(c=1,),),);max_depth=2)
    @test_throws ArgumentError S.visual_data_from_json(S.visual_data_to_json((a=(b=(c=1,),),));max_depth=2)
    @test_throws ArgumentError S.visual_data_to_json([1,2,3];max_entries=2)
    @test_throws ArgumentError S.visual_data_from_json(S.visual_data_to_json([1,2,3]);max_entries=2)
    @test_throws ArgumentError S.visual_data_to_json(1;max_depth=true)
    @test_throws ArgumentError S.visual_data_from_json(S.visual_data_to_json(1);max_entries=0)
    data=S.visual_data_to_json(TamerOp.Visualization.VisualStyle());data["name"]="SomeConstructor"
    @test_throws ArgumentError S.visual_data_from_json(data)
    data=S.visual_data_to_json([1,2]);data["size"]=[typemax(Int),typemax(Int)]
    @test_throws ArgumentError S.visual_data_from_json(data)
end

function _a100_source_module(field)
    C=TamerOp.CoreModules; F=TamerOp.FiniteFringe; M=TamerOp.Modules
    P=F.GridPoset(([0,1,2],))
    A=C.coerce.(Ref(field),[1 2;0 1]); B=C.coerce.(Ref(field),[2 1;1 1])
    object=M.PModule{C.coeff_type(field)}(P,[2,2,2],Dict((1,2)=>A,(2,3)=>B);field)
    return object,A,B
end

@testset "A100 owned mathematical source coordinates and identity" begin
    S=TamerOp.Serialization; C=TamerOp.CoreModules; M=TamerOp.Modules; F=TamerOp.FiniteFringe
    for field in (C.QQField(),C.F2(),C.F3(),C.Fp(101),C.RealField(Float64;rtol=1e-10,atol=1e-12))
        object,A,B=_a100_source_module(field)
        before=S.visual_source_identity(object)
        @test M.structure_map(object;source=1,target=3)==B*A
        @test S.visual_source_identity(object)==before
        restored=S.visual_source_from_json(S.visual_source_to_json(object))
        @test M.check_module_functoriality(restored).valid
        @test restored.field==field
        @test restored.dims==[2,2,2]
        @test M.structure_map(restored;source=1,target=2)==A
        @test M.structure_map(restored;source=2,target=3)==B
        @test M.structure_map(restored;source=1,target=3)==B*A
        @test S.visual_source_identity(restored)==before
        @test !isempty(sprint(show,restored))
    end
    object,A,B=_a100_source_module(C.QQField())
    changed=M.PModule{eltype(A)}(object.Q,[2,2,2],Dict((1,2)=>[1//1 0;2 1],(2,3)=>B);field=object.field)
    @test S.visual_source_identity(changed)!=S.visual_source_identity(object)
    @test_throws ArgumentError S.visual_source_to_json(object;max_vertices=2)
    @test_throws ArgumentError S.visual_source_from_json(S.visual_source_to_json(object);max_vertices=2)
    @test_throws ArgumentError S.visual_source_to_json(identity)
    zero_maps=M.PModule{Rational{BigInt}}(object.Q,[0,2,0],
        Dict((1,2)=>zeros(Rational{BigInt},2,0),(2,3)=>zeros(Rational{BigInt},0,2));field=C.QQField())
    zero_restored=S.visual_source_from_json(S.visual_source_to_json(zero_maps))
    @test size(M.structure_map(zero_restored;source=1,target=2))==(2,0)
    @test size(M.structure_map(zero_restored;source=2,target=3))==(0,2)
    sparse_maps=M.PModule{eltype(A)}(object.Q,[2,2,2],Dict((1,2)=>sparse(A),(2,3)=>sparse(B));field=object.field)
    @test S.visual_source_identity(S.visual_source_from_json(S.visual_source_to_json(sparse_maps)))==S.visual_source_identity(sparse_maps)
    raw=S.JSON3.read(S.visual_source_to_json(object),Dict{String,Any})
    pop!(raw["payload"]["module"]["edges"])
    @test_throws ArgumentError S.visual_source_from_json(S.JSON3.write(raw))
    raw=S.JSON3.read(S.visual_source_to_json(object),Dict{String,Any})
    push!(raw["payload"]["module"]["edges"],first(raw["payload"]["module"]["edges"]))
    @test_throws ArgumentError S.visual_source_from_json(S.JSON3.write(raw))
    raw=S.JSON3.read(S.visual_source_to_json(object),Dict{String,Any})
    raw["payload"]["module"]["poset"]["extra"]=true
    @test_throws ArgumentError S.visual_source_from_json(S.JSON3.write(raw))

    # A source file must not turn a noncommuting diamond into a module.
    P=F.FinitePoset(Bool[1 1 1 1;0 1 0 1;0 0 1 1;0 0 0 1])
    for field in (C.QQField(),C.F2(),C.F3(),C.RealField(Float64;rtol=1e-10,atol=1e-12))
        one=reshape([C.coerce(field,1)],1,1); zero=reshape([C.coerce(field,0)],1,1)
        good=M.PModule{C.coeff_type(field)}(P,ones(Int,4),Dict((1,2)=>one,(1,3)=>one,(2,4)=>one,(3,4)=>one);field)
        bad=M.PModule{C.coeff_type(field)}(P,ones(Int,4),Dict((1,2)=>one,(1,3)=>one,(2,4)=>one,(3,4)=>zero);field)
        @test M.check_module(bad).valid
        @test !M.check_module_functoriality(bad).valid
        @test_throws Exception M.check_module_functoriality(bad;throw=true)
        @test_throws Exception S.visual_source_to_json(bad)
        raw=S.JSON3.read(S.visual_source_to_json(good),Dict{String,Any})
        raw["payload"]["module"]["edges"][4]["matrix"]=S.visual_data_to_json(zero)
        @test_throws ArgumentError S.visual_source_from_json(S.JSON3.write(raw))
    end
    # Previously queried composites cannot conceal a changed internal edge.
    cover=[(1,2),(2,3),(3,6),(1,4),(4,5),(5,6)]
    L=Matrix{Bool}(I,6,6)
    for (u,v) in cover
        L[u,v]=true
    end
    for k in 1:6,u in 1:6,v in 1:6
        L[u,v] |= L[u,k] && L[k,v]
    end
    stale=M.PModule{Rational{BigInt}}(F.FinitePoset(L),ones(Int,6),
        Dict(edge=>reshape([big(1)//big(1)],1,1) for edge in cover);field=C.QQField())
    @test M.structure_map(stale;source=1,target=3)==ones(Rational{BigInt},1,1)
    stale.edge_maps[2,3][1,1]=0
    @test !M.check_module_functoriality(stale).valid
    @test_throws Exception S.visual_source_to_json(stale)
end

@testset "A100 owned encoding bases and ordinary result reuse" begin
    S=TamerOp.Serialization; C=TamerOp.CoreModules; R=TamerOp.Results; M=TamerOp.Modules
    for field in (C.QQField(),C.F3(),C.RealField(Float64))
        object,A,B=_a100_source_module(field)
        pi=TamerOp.EncodingCore.GridEncodingMap(object.Q,([0,1,2],))
        enc=R.EncodingResult(object.Q,object,pi;opts=TamerOp.Options.EncodingOptions(;field),
            backend=:test_source,meta=(;provenance=(;degree=1,degree_convention=:homological,window=:finite_fixture)))
        text=S.visual_source_to_json(enc)
        restored=S.visual_source_from_json(text)
        @test restored.P===restored.M.Q
        @test restored.pi.P===restored.P
        @test restored.backend==:test_source
        @test R.provenance(restored).degree==1
        @test M.structure_map(restored.M;source=1,target=2)==A
        @test M.structure_map(restored.M;source=1,target=3)==B*A
        @test S.visual_source_to_json(restored)==text
    end
    O=TamerOp.OrdinaryPersistence
    diagram=O.cubical_persistence([0//1,2//3,0//1];input=:vertices,source=true,representatives=true)
    restored=S.visual_source_from_json(S.visual_source_to_json(diagram))
    @test O.finite_intervals(restored;dim=0)==[(0//1,2//3)]
    @test O.essential_births(restored;dim=0)==[0//1]
    @test S.visual_source_identity(restored)==S.visual_source_identity(diagram)
    original=O.persistence_diagram_data(diagram)
    forward=Dict("entry$i"=>i for i in 1:80)
    reverse=Dict("entry$i"=>i for i in 80:-1:1)
    first_diagram=O.persistence_diagram_from_data(merge(original,(;meta=merge(original.meta,(;annotations=forward)))))
    second_diagram=O.persistence_diagram_from_data(merge(original,(;meta=merge(original.meta,(;annotations=reverse)))))
    @test S.visual_source_identity(first_diagram)==S.visual_source_identity(second_diagram)
    text=S.visual_source_to_json(first_diagram)
    @test S.visual_source_to_json(S.visual_source_from_json(text))==text
end
