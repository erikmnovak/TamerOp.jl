# Independent small complexes make the vector and quotient equations explicit.
function _a89_complex(field=CM.QQField())
    K=CM.coeff_type(field)
    # K --(1,0,0)--> K^3 --(0,0,1)--> K.
    CC.CochainComplex{K}(0,2,[1,3,1],
        [sparse(reshape(K[1,0,0],3,1)),sparse(reshape(K[0,0,1],1,3))];field)
end
function _a89_cone(field=CM.QQField())
    K=CM.coeff_type(field)
    C=CC.CochainComplex{K}(0,0,[1],SparseMatrixCSC{K,Int}[];field)
    D=CC.CochainComplex{K}(0,0,[0],SparseMatrixCSC{K,Int}[];field)
    f=CC.CochainMap(C,D,[spzeros(K,0,1)])
    CC.long_exact_sequence(f;output=:full)
end

@testset "A89 cycles, boundaries and noncycles" begin
    V=TamerOp.Visualization
    for field in (CM.QQField(),CM.F2(),CM.F3(),CM.Fp(5),CM.RealField(Float64))
        K=CM.coeff_type(field);C=_a89_complex(field)
        cheap=V.visual_spec(C;degree=1)
        @test !cheap.metadata.quotient_computed
        @test cheap.metadata.local_data.quotient===nothing
        @test V.check_visual_spec(cheap).valid
        for (z,cycle,boundary) in ((K[0,1,0],true,false),(K[1,0,0],true,true),(K[0,0,1],false,nothing),(zeros(K,3),true,true))
            s=V.visual_spec(C;degree=1,vector=z).metadata.selection
            @test s.is_cycle==cycle
            @test s.is_boundary===boundary
            @test s.differential==CC.differential(C,1)*z
            if cycle
                @test s.quotient_equation.valid
                @test s.vector==s.representative+CC.differential(C,0)*s.boundary_adjustment
                @test CC.differential(C,1)*s.representative==zeros(K,1)
                @test length(s.coordinates)==1
            else
                @test s.coordinates===nothing
            end
        end
        s=V.visual_spec(C;degree=1,vector=K[1],coordinates=:class)
        @test s.metadata.selection.is_cycle
        @test !s.metadata.selection.is_boundary
        @test s.metadata.local_data.quotient.dimensions==(ambient=3,cycles=2,boundaries=1,quotient=1)
        @test_throws DimensionMismatch V.visual_spec(C;degree=1,vector=K[1,2])
        @test_throws DimensionMismatch V.visual_spec(C;degree=1,vector=K[1,2],coordinates=:class)
        @test_throws ArgumentError V.visual_spec(C;degree=99)
        @test_throws ArgumentError V.visual_spec(C;vertex=1)
        @test_throws ArgumentError V.visual_spec(C;convention=:homological)
        @test_throws ArgumentError V.visual_spec(C;max_terms=2)
        @test_throws ArgumentError V.visual_spec(C;coordinates=:guess)
        @test_throws ArgumentError V.visual_spec(C;vector=[true,false,false],degree=1)
        @test_throws ArgumentError V.visual_spec(C;unknown_option=1)
    end
end

@testset "A89 field dependence, coordinate changes and zero spaces" begin
    V=TamerOp.Visualization
    for field in (CM.QQField(),CM.F2(),CM.F3())
        K=CM.coeff_type(field)
        C=CC.CochainComplex{K}(0,1,[1,1],[sparse(reshape(K[2],1,1))];field)
        s=V.visual_spec(C;degree=1,compute=true)
        @test s.metadata.local_data.quotient.dimensions.quotient==(field==CM.F2() ? 1 : 0)
        empty=CC.CochainComplex{K}(-2,-1,[0,0],[spzeros(K,0,0)];field)
        e=V.visual_spec(empty;degree=-2,vector=K[])
        @test e.metadata.selection.is_boundary
        @test size(e.metadata.local_data.quotient.representatives)==(0,0)
        @test V.check_visual_spec(e).valid
        # Different coordinates describe the same class and the same boundary.
        C=_a89_complex(field);P=K[1 1 0;0 1 0;0 0 1];Pinv=K[1 -1 0;0 1 0;0 0 1]
        D=CC.CochainComplex{K}(0,2,[1,3,1],[sparse(Pinv*CC.differential(C,0)),sparse(CC.differential(C,1)*P)];field)
        f=CC.CochainMap(C,D,[sparse(reshape(K[1],1,1)),sparse(Pinv),sparse(reshape(K[1],1,1))])
        s=V.visual_spec(f;degree=1,vector=K[1,1,0])
        a,b=s.metadata.selection,s.metadata.map_data.target.selected
        @test a.is_cycle && b.is_cycle
        @test a.is_boundary==b.is_boundary==false
        @test b.vector==Pinv*a.vector
        @test s.metadata.map_data.induced*a.coordinates==b.coordinates
        @test b.quotient_equation.valid
    end
    C=_a89_complex()
    @test_throws ArgumentError V.visual_spec(C;degree=1,vector=[0.0,1.0,0.0])
    R=_a89_complex(CM.RealField(Float64))
    @test_throws ArgumentError V.visual_spec(R;degree=1,vector=[0.0,NaN,0.0])
    @test_throws ArgumentError V.visual_spec(R;degree=1,vector=[0.0,Inf,0.0])
    @test V.visual_spec(_a89_complex(CM.F3());degree=1,vector=[0,1//2,0]).metadata.selection.is_cycle
    bad=deepcopy(C);bad.d[2][1,1]=1
    @test_throws Exception V.visual_spec(bad;degree=1)
end

@testset "A89 source chains and actual filling" begin
    V=TamerOp.Visualization;DI=TamerOp.DataIngestion;K=Rational{BigInt}
    source=DI.filtered_complex(reshape([0],1,1))
    # The exact zero product can contain explicitly stored cancellation entries.
    B1=TamerOp.DataTypes.boundary_map(source.complex;dim=1)
    B2=TamerOp.DataTypes.boundary_map(source.complex;dim=2)
    product=sparse(K.(B1))*sparse(K.(B2))
    @test nnz(product)>0 && all(iszero,nonzeros(product))
    @test CC._check_zero_matrix(product;field=CM.QQField())
    product[1,1]=one(K)
    @test !CC._check_zero_matrix(product;field=CM.QQField())
    for field in (CM.QQField(),CM.F2(),CM.F3())
        Kf=CM.coeff_type(field)
        B=Kf.(TamerOp.DataTypes.boundary_map(source.complex;dim=2));z=vec(B)
        s=V.visual_spec(source;kind=:chain_inspector,degree=1,parameter=0,field,vector=z)
        @test s.metadata.convention==:homological
        @test s.metadata.selection.is_boundary
        @test B*s.metadata.selection.boundary_adjustment==z
        @test count(p->p.kind==:source_chain_geometry,s.panels)==2
        corr=DI.source_correspondence(source,source;field)
        c=DI.lift_source_chain(corr;dim=1,coefficients=z,parameter=0,kind=:cycle)
        p=V.visual_spec(c;kind=:chain_inspector)
        @test p.metadata.selection.vector==z
        @test p.metadata.selection.is_boundary
        live=V.inspection_session(c)
        V.select_inspection!(live;vector=nothing)
        @test V.inspection_snapshot(live).metadata.selection===nothing
        V.reset_inspection!(live)
        @test V.inspection_snapshot(live).metadata.selection.vector==z
        V.close_inspection!(live)
        @test_throws ArgumentError V.visual_spec(c;kind=:chain_inspector,parameter=1)
        d=V.visual_spec(source;kind=:chain_inspector,degree=1,parameter=0,field,convention=:cohomological,vector=ones(Kf,4))
        @test d.metadata.convention==:cohomological
        @test d.metadata.selection.differential==transpose(B)*ones(Kf,4)
    end
    e=V.visual_spec(source;kind=:chain_inspector,parameter=-1,degree=1,compute=true)
    @test e.metadata.local_data.quotient.dimensions.ambient==0
    @test all(isempty,e.metadata.active_ids)
    # A multicritical edge enters through a union of appearances, not its box hull.
    G=TamerOp.DataTypes.MultiCriticalGradedComplex([[10,20],[30]],
        [sparse(reshape([-1,1],2,1))],[[(0,5)],[(0,5)],[(1,2),(3,4)]])
    x=DI.filtered_complex(G;geometry=[[(0,0)],[(1,0)],[(0,0),(1,0)]],orientation=(1,-1))
    for (p,h0) in (((2,3),2),((1,2),1),((3,4),1),((4,5),2))
        q=V.visual_spec(x;kind=:chain_inspector,degree=0,parameter=p,compute=true)
        @test q.metadata.local_data.quotient.dimensions.quotient==h0
    end
end

@testset "A89 connecting maps and shift conventions" begin
    V=TamerOp.Visualization
    for field in (CM.QQField(),CM.F2(),CM.F3())
        K=CM.coeff_type(field);les=_a89_cone(field)
        s=V.visual_spec(les;degree=-1,vector=K[1])
        @test s.metadata.map==:delta
        @test only(s.metadata.selection.image_coordinates)==one(K)
        @test s.metadata.selection.equation.valid
        @test all(c->c.exact && c.equation.valid,s.metadata.checks)
        @test V.check_visual_spec(s).valid
        @test_throws ArgumentError V.visual_spec(les;map=:p)
        @test_throws ArgumentError V.visual_spec(les;vertex=1)
        @test_throws DimensionMismatch V.visual_spec(les;degree=-1,vector=K[1,1])
        bad=deepcopy(les);bad.delta[1][1,1]=zero(K)
        @test_throws ArgumentError V.visual_spec(bad;degree=-1)
    end
end

@testset "A89 linked algebra lifecycle" begin
    V=TamerOp.Visualization;C=_a89_complex();s=V.inspection_session(C;degree=1)
    @test s isa TamerOp.Advanced.ChainInspectionSession
    @test V.check_inspection_session(s).valid
    @test V.check_visual_spec(V.visual_spec(s)).valid
    observed=Ref(0);token=V._on_inspection(s,_ -> (observed[]+=1))
    V.select_inspection!(s;vector=[1,1,0])
    @test observed[]==1
    @test V.inspection_snapshot(s).metadata.selection.is_cycle
    before=V.inspection_snapshot(s);revision=V.inspection_summary(s).revision
    @test_throws DimensionMismatch V.select_inspection!(s;vector=[1,2])
    @test V.inspection_snapshot(s)===before
    @test V.inspection_summary(s).revision==revision
    @test !V.check_inspection_selection(s;degree=7).valid
    V.select_inspection!(s;degree=0)
    @test V.inspection_selection(s).vector===nothing
    V.reset_inspection!(s)
    @test V.inspection_selection(s).degree==1
    V._off_inspection!(s,token)
    V.close_inspection!(s)
    @test V.check_inspection_session(s).valid
    @test s.object===nothing
    @test_throws ArgumentError V.select_inspection!(s;degree=0)
    @test V.check_visual_spec(V.inspection_snapshot(s)).valid
    @test V.close_inspection!(s)===s
    f=TamerOp.filtered_complex(zeros(1,1))
    @test V.inspection_session(f) isa V.ComplexInspectionSession
    c=V.inspection_session(f;kind=:chain_inspector,degree=1)
    @test c isa V.ChainInspectionSession
    V.close_inspection!(c)
end

@testset "A89 module stalks, induced maps and supplied homotopy" begin
    V,MC=TamerOp.Visualization,TamerOp.ModuleComplexes
    for field in (CM.QQField(),CM.F3(),CM.RealField(Float64))
        K=CM.coeff_type(field);P=FF.FinitePoset(trues(1,1))
        M=MD.PModule{K}(P,[2],Dict{Tuple{Int,Int},Matrix{K}}();field)
        N=MD.PModule{K}(P,[1],Dict{Tuple{Int,Int},Matrix{K}}();field)
        Z=MD.zero_pmodule(P;field)
        d=MD.PMorphism(M,N,[reshape(K[0,1],1,2)])
        C=MC.ModuleCochainComplex([M,N],[d])
        f=MC.ModuleCochainMap(C,C,[MD.id_morphism(M),MD.id_morphism(N)])
        g=MC.ModuleCochainMap(C,C,[MD.PMorphism(M,M,[K[1 0;0 0]]),MD.zero_morphism(N,N)])
        H=MC.ModuleCochainHomotopy(f,g,[MD.zero_morphism(M,Z),MD.PMorphism(N,M,[reshape(K[0,1],2,1)])])
        s=V.visual_spec(C;degree=0,vertex=1,vector=K[1,0])
        @test s.metadata.selection.is_cycle
        @test s.metadata.category==:finite_poset_stalk
        q=V.visual_spec(f;kind=:chain_inspector,degree=0,vector=K[1,0],homotopy=H)
        @test q.metadata.map_data.induced==reshape(K[1],1,1)
        @test q.metadata.map_data.witness.equation.valid
        @test q.metadata.map_data.target.incoming*q.metadata.map_data.witness.vector==q.metadata.map_data.witness.image_difference
        @test V.check_visual_spec(q).valid
        @test_throws ArgumentError V.visual_spec(g;kind=:chain_inspector,homotopy=H)
        @test_throws ArgumentError V.visual_spec(C;vertex=2)
        @test V.visual_spec(f;degree=0,vertex=1,induced=true).kind==:chain_map
        @test V.visual_spec(H;degree=0,vertex=1,induced=true).kind==:homotopy_comparison
    end
end

@testset "A89 snake connecting map" begin
    V,AC=TamerOp.Visualization,TamerOp.AbelianCategories
    for field in (CM.QQField(),CM.F3())
        K=CM.coeff_type(field);P=FF.FinitePoset(trues(1,1))
        M=MD.PModule{K}(P,[1],Dict{Tuple{Int,Int},Matrix{K}}();field)
        Z=MD.zero_pmodule(P;field)
        # 0 -> 0 -> K -> K -> 0 above 0 -> K -> K -> 0 -> 0.
        top=AC.ShortExactSequence(MD.zero_morphism(Z,M),MD.id_morphism(M);check=true)
        bottom=AC.ShortExactSequence(MD.id_morphism(M),MD.zero_morphism(M,Z);check=true)
        sn=AC.snake_lemma(top,bottom,MD.zero_morphism(Z,M),MD.id_morphism(M),MD.zero_morphism(M,Z))
        s=V.visual_spec(sn;vector=K[1])
        @test s.metadata.sequence_kind==:snake_sequence
        @test only(s.metadata.selection.image_coordinates)==one(K)
        @test all(c->c.exact,s.metadata.checks)
        @test V.check_visual_spec(s).valid
        @test_throws ArgumentError V.visual_spec(sn;degree=0)
        @test_throws ArgumentError V.visual_spec(sn;map=:f)
        live=V.inspection_session(sn)
        V.select_inspection!(live;vector=K[1])
        @test V.inspection_snapshot(live).metadata.selection.image_coordinates==K[1]
        V.close_inspection!(live)
    end
end

@testset "A89 native Cairo WGL exports and controls" begin
    V=TamerOp.Visualization
    if Base.find_package("CairoMakie")!==nothing && Base.find_package("WGLMakie")!==nothing
        @eval using CairoMakie,WGLMakie
        specs=(V.visual_spec(_a89_complex();degree=1,vector=[1,1,0]),
            V.visual_spec(_a89_cone();degree=-1,vector=[1]))
        mktempdir() do tmp
            for (i,spec) in enumerate(specs), extension in ("png","svg","pdf")
                path=joinpath(tmp,"chain-$i.$extension")
                V.save_visual(path,spec;backend=:cairomakie)
                @test filesize(path)>100
            end
        end
        for spec in specs,palette in (:accessible,:grayscale)
            a=V.render(spec;backend=:cairomakie,style=V.VisualStyle(;palette))
            b=V.render(spec;backend=:wglmakie,style=V.VisualStyle(;palette))
            @test a isa CairoMakie.Figure
            @test b isa WGLMakie.Figure
            empty!(a);empty!(b)
        end
        session=V.inspection_session(_a89_complex();degree=1)
        app=V.visualize(session;backend=:wglmakie)
        ext=Base.get_extension(TamerOp,:TamerOpWGLMakieExt);ui=ext._inspection_ui(app)
        old=ui.figure[]
        V.select_inspection!(session;vector=[1,1,0])
        @test isempty(V.inspection_summary(session).listener_errors)
        @test ui.figure[]!==old
        @test length(ui.pointer_observers)==2
        ui.controls.vector.value[]="1,2";ui.controls.select.value[]=true
        @test !isempty(ui.error_text[])
        @test V.inspection_selection(session).vector==[1,1,0]
        ui.controls.degree.value[]="0";ui.controls.apply_location.value[]=true
        @test isempty(ui.error_text[])
        @test V.inspection_selection(session).vector===nothing
        V.close_inspection!(session)
        @test ui.closed[] && isempty(ui.observers) && isempty(ui.pointer_observers)
        @test !haskey(ext._INSPECTION_UI_HANDLES,app)
    else
        @test_skip false
    end
end

@testset "A89 homotopy naturality and numerical tolerance" begin
    V,MC=TamerOp.Visualization,TamerOp.ModuleComplexes
    field=CM.QQField();K=CM.coeff_type(field);P=FF.FinitePoset(Bool[1 1;0 1])
    M=MD.PModule{K}(P,[1,1],Dict((1,2)=>reshape(K[1],1,1));field)
    Z=MD.zero_pmodule(P;field)
    C=MC.ModuleCochainComplex([M,M],[MD.zero_morphism(M,M)])
    z=MC.ModuleCochainMap(C,C,[MD.zero_morphism(M,M),MD.zero_morphism(M,M)])
    incompatible=MD.PMorphism(M,M,[reshape(K[1],1,1),reshape(K[2],1,1)])
    H=MC.ModuleCochainHomotopy(z,z,[MD.zero_morphism(M,Z),incompatible];check=false)
    @test MC.check_module_homotopy(H).valid
    @test_throws ArgumentError V.visual_spec(z;kind=:chain_inspector,homotopy=H)
    field=CM.RealField(Float64);K=Float64;P=FF.FinitePoset(trues(1,1))
    M=MD.PModule{K}(P,[1],Dict{Tuple{Int,Int},Matrix{K}}();field);Z=MD.zero_pmodule(P;field)
    C=MC.ModuleCochainComplex([M,M],[MD.id_morphism(M)])
    f=MC.ModuleCochainMap(C,C,[MD.id_morphism(M),MD.id_morphism(M)])
    g=MC.ModuleCochainMap(C,C,[MD.zero_morphism(M,M),MD.zero_morphism(M,M)])
    H=MC.ModuleCochainHomotopy(f,g,[MD.zero_morphism(M,Z),MD.PMorphism(M,M,[reshape([1+1e-10],1,1)])];check=false)
    q=V.visual_spec(f;kind=:chain_inspector,degree=1,homotopy=H,vector=[1.0])
    @test q.metadata.map_data.witness.equation.valid
    @test q.metadata.map_data.witness.equation.residual>0
end

@testset "A89 display budgets preserve the represented algebra" begin
    V=TamerOp.Visualization;K=Rational{BigInt}
    C=CC.CochainComplex{K}(-5,8,ones(Int,14),[spzeros(K,1,1) for _ in 1:13])
    s=V.visual_spec(C;degree=1,max_terms=3,vector=[1],matrix_limit=(1,1))
    @test first(s.panels).metadata.omitted_terms==11
    @test [t.degree for t in first(s.panels).metadata.pick_targets]==[0,1,2]
    @test s.metadata.local_data.quotient.dimensions.quotient==1
    source=TamerOp.filtered_complex(zeros(Int,1,1));z=K[-1,1,1,-1]
    p=V.visual_spec(source;kind=:chain_inspector,degree=1,parameter=0,vector=z,max_cells=2)
    @test p.metadata.selection.vector==z
    @test p.metadata.selection.quotient_equation.valid
    chart=first(filter(p->p.kind==:source_chain_geometry,p.panels))
    @test length(chart.metadata.displayed_indices)==2
end
