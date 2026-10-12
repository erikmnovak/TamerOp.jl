# Independent finite categories: the diamond relation and matrix multiplication.
function _a90_simple(P,v,field)
    K=CM.coeff_type(field);ds=[Int(i==v) for i in 1:FF.nvertices(P)]
    MD.PModule{K}(P,ds,Dict((a,b)=>zeros(K,ds[b],ds[a]) for (a,b) in FF.cover_edges(P));field)
end
function _a90_ext_fixture(field)
    P=diamond_poset();S=[_a90_simple(P,i,field) for i in 1:4]
    E(a,b)=TamerOp.ext(S[a],S[b];maxdeg=2,model=:projective)
    (;P,S,E24=E(2,4),E12=E(1,2),E34=E(3,4),E13=E(1,3),E14=E(1,4))
end
function _a90_matrix_algebra(field)
    K=CM.coeff_type(field);P=chain_poset(1)
    R=MD.PModule{K}(P,[1],Dict{Tuple{Int,Int},Matrix{K}}();field)
    L=MD.PModule{K}(P,[4],Dict{Tuple{Int,Int},Matrix{K}}();field)
    T=TamerOp.tor(R,L;maxdeg=1)
    mu=zeros(K,4,16)
    for i in 1:4,j in 1:4
        x=zeros(K,4);y=zeros(K,4);x[i]=1;y[j]=1
        mu[:,(i-1)*4+j]=vec(reshape(x,2,2)*reshape(y,2,2))
    end
    A=DF.TorAlgebra(T;mu_chain=Dict((0,0)=>sparse(mu)),unit_coords=K[1,0,0,1])
    (;A,T,mu,R,L)
end

@testset "A90 cheap dimensions and explicit representative models" begin
    V=TamerOp.Visualization
    with_fields(FIELDS_FULL) do field
        K=CM.coeff_type(field);a=_a90_ext_fixture(field)
        representatives_before=[getfield(h,:_Hrep) for h in a.E14.cohom]
        cheap=V.visual_spec(a.E14)
        @test cheap.metadata.dimensions==[0,0,1]
        @test !cheap.metadata.representatives_computed
        @test all(getfield(h,:_Hrep)===before for (h,before) in zip(a.E14.cohom,representatives_before))
        @test V.check_visual_spec(cheap).valid
        H=DF.Hom(a.S[1],a.S[1]);before=DF.hom_summary(H).basis_cached
        @test V.visual_spec(H;kind=:graded_dimensions).metadata.dimensions==[1]
        @test DF.hom_summary(H).basis_cached==before
        for (model,canon) in ((:projective,:auto),(:injective,:auto),(:unified,:projective),(:unified,:injective))
            E=TamerOp.ext(a.S[1],a.S[4];maxdeg=2,model,canon)
            spec=V.visual_spec(E;kind=:derived_class,degree=2,vector=[1])
            s=spec.metadata.selection
            @test V.check_visual_spec(spec).valid
            @test s.degree==s.complex_degree==2
            @test !s.zero_class
            @test s.cycle_equation.valid && s.coordinate_equation.valid && s.quotient_equation.valid
            @test s.representative==DF.representative(E,2,K[1])
            @test s.outgoing*s.representative==zeros(K,size(s.outgoing,1))
            z=V.visual_spec(E;kind=:derived_class,degree=0).metadata.selection
            @test z.zero_class && isempty(z.coordinates)
        end
        for model in (:first,:second)
            P=chain_poset(2);Pop=FF.FinitePoset(transpose(FF.leq_matrix(P)))
            R=_a90_simple(Pop,2,field);L=_a90_simple(P,1,field)
            T=TamerOp.tor(R,L;maxdeg=1,model)
            @test V.visual_spec(T).metadata.dimensions==[0,1]
            s=V.visual_spec(T;kind=:derived_class,degree=1,vector=[1]).metadata.selection
            @test s.complex_degree == -1
            @test !s.zero_class && s.cycle_equation.valid
            @test CC.degree_range(DF.representative_complex(T)) == -2:0
        end
    end
    @test TamerOp.Advanced.representative_complex===DF.representative_complex
    @test TamerOp.Advanced.multiplication_matrix===DF.multiplication_matrix
end

@testset "A90 Yoneda diamond signs and actual quotient witnesses" begin
    V=TamerOp.Visualization
    with_fields(FIELDS_FULL) do field
        K=CM.coeff_type(field);a=_a90_ext_fixture(field)
        specs=[V.visual_spec(L;kind=:yoneda_product,p=1,q=1,right=R,left_vector=[1],right_vector=[1],target=a.E14)
            for (L,R) in ((a.E24,a.E12),(a.E34,a.E13))]
        c,d=[s.metadata.selection.coordinates for s in specs]
        # The dual diamond relation is the sum of the two length-two paths.
        @test !iszero(c[1])
        @test field isa CM.RealField ? isapprox(c,-d;atol=field.atol,rtol=field.rtol) : c == -d
        for spec in specs
            @test V.check_visual_spec(spec).valid
            s=spec.metadata.selection
            @test s.cycle_equation.valid && s.quotient_equation.valid
            @test s.representative==s.canonical+s.incoming*s.boundary_adjustment
            @test spec.metadata.extension_sequence==:not_constructed
            @test spec.metadata.provenance.ambient_identification==:not_asserted
        end
        zero=V.visual_spec(a.E24;kind=:yoneda_product,p=1,q=1,right=a.E12,left_vector=[0],right_vector=[1])
        @test zero.metadata.selection.zero_class
        # Multiplying by degree-zero scalars fixes both convention and order.
        E11=TamerOp.ext(a.S[1],a.S[1];maxdeg=2)
        unit=V.visual_spec(a.E12;kind=:yoneda_product,p=1,q=0,right=E11,left_vector=[1],right_vector=[1],target=a.E12)
        @test unit.metadata.selection.coordinates==K[1]
    end
end

@testset "A90 exact tables, noncommutative ordering and bounded work" begin
    V=TamerOp.Visualization
    with_fields(FIELDS_FULL) do field
        K=CM.coeff_type(field);P=diamond_poset()
        # Direct sum of the four vertex simples: Ext dimensions 4,4,1.
        M=MD.PModule{K}(P,ones(Int,4),Dict(e=>zeros(K,1,1) for e in FF.cover_edges(P));field)
        A=DF.ExtAlgebra(M,OPT.DerivedFunctorOptions(maxdeg=2))
        @test isempty(DF.cached_product_degrees(A))
        @test V.visual_spec(A).metadata.dimensions==[4,4,1]
        @test isempty(DF.cached_product_degrees(A))
        @test !V.check_visual_request(A;kind=:product_table,p=1,q=1,max_products=15).valid
        @test isempty(DF.cached_product_degrees(A))
        spec=V.visual_spec(A;kind=:product_table,p=1,q=1,matrix_limit=(1,3))
        @test V.check_visual_spec(spec).valid
        mu=spec.metadata.matrix
        @test size(mu)==(1,16)
        @test count(!iszero,mu)==2
        @test iszero(sum(mu))
        @test spec.panels[1].metadata.truncated
        @test DF.cached_product_degrees(A)==[(1,1)]
        @test DF.multiplication_matrix(A,1,1)==mu
        mu[1,1]=CM.coerce(field,2)
        @test DF.multiplication_matrix(A,1,1)[1,1]==0 # owned visual snapshot
        e=DF.element(A,1,K[1,0,0,0])
        @test V.visual_spec(e).metadata.selection.coordinates==K[1,0,0,0]
        t=_a90_matrix_algebra(field)
        @test isempty(t.A.mu_H_cache)
        V.visual_spec(t.A)
        @test isempty(t.A.mu_H_cache)
        # E12 * E21 = E11, whereas E21 * E12 = E22.
        s=V.visual_spec(t.A;kind=:product_table,p=0,q=0,left=[0,0,1,0],right=[0,1,0,0],validate_algebra=true)
        @test s.metadata.matrix==t.mu
        @test s.metadata.selection.coordinates==K[1,0,0,0]
        @test s.metadata.algebra_validation.valid
        @test s.metadata.algebra_validation.verified_through==1
        rev=V.visual_spec(t.A;kind=:product_table,left=[0,1,0,0],right=[0,0,1,0])
        @test rev.metadata.selection.coordinates==K[0,0,0,1]
        z=V.visual_spec(t.A;kind=:product_table,p=1,q=0)
        @test size(z.metadata.matrix)==(0,0)
        @test V.check_visual_spec(z).valid
        @test V.visual_spec(DF.element(t.A,0,K[1,0,0,0])).metadata.selection.coordinates==K[1,0,0,0]
    end
end

@testset "A90 rejected contracts and invalid supplied products" begin
    V=TamerOp.Visualization;a=_a90_ext_fixture(CM.QQField());t=_a90_matrix_algebra(CM.QQField())
    for opts in ((degree=-1,), (degree=true,), (degree=2,vector=[1,2]),(degree=2,vector=[1.0]),(degree=2,vector=[true]),(matrix_limit=(0,2),))
        @test !V.check_visual_request(a.E14;kind=:derived_class,opts...).valid
        @test_throws ArgumentError V.visual_spec(a.E14;kind=:derived_class,opts...)
    end
    for opts in ((p=2,), (left=[1,0,0,0],), (max_products=15,), (max_entries=63,), (max_products=true,), (validate_algebra=:yes,))
        @test_throws ArgumentError V.visual_spec(t.A;kind=:product_table,opts...)
    end
    @test_throws ArgumentError V.visual_spec(t.T;kind=:product_table)
    @test_throws Exception V.visual_spec(DF.TorAlgebra(t.T);kind=:product_table)
    @test_throws ArgumentError V.visual_spec(a.E24;kind=:yoneda_product,right=a.E12,p=1,q=1)
    @test_throws Exception V.visual_spec(a.E24;kind=:yoneda_product,right=a.E13,p=1,q=1,left_vector=[1],right_vector=[1])
    @test_throws Exception V.visual_spec(a.E24;kind=:yoneda_product,right=a.E12,target=a.E13,p=1,q=1,left_vector=[1],right_vector=[1])
    # A boundary multiplied into a non-boundary cannot define a quotient product.
    field=CM.QQField();K=CM.coeff_type(field);P=chain_poset(2);Pop=FF.FinitePoset(transpose(FF.leq_matrix(P)))
    R=_a90_simple(Pop,2,field)
    L=MD.PModule{K}(P,[1,2],Dict((1,2)=>reshape(K[1,0],2,1));field)
    T=TamerOp.tor(R,L;maxdeg=1)
    A=DF.TorAlgebra(T;mu_chain=Dict((0,0)=>sparse(K[0 0 0 0;1 0 0 0])))
    @test_throws Exception V.visual_spec(A;kind=:product_table)
    real=_a90_ext_fixture(CM.RealField(Float64))
    @test_throws ArgumentError V.visual_spec(real.E14;kind=:derived_class,degree=2,vector=[NaN])
end

@testset "A90 nontrivial boundaries and coordinate changes" begin
    V=TamerOp.Visualization
    with_fields(FIELDS_FULL) do field
        K=CM.coeff_type(field);P=diamond_poset()
        # Each stalk contains its vertex simple and a constant summand.
        # Integral gauges mix the two across every field without singularity.
        gauges=[K[1 v;0 1] for v in 1:4]
        inverses=[K[1 -v;0 1] for v in 1:4]
        base=MD.PModule{K}(P,fill(2,4),Dict(e=>K[0 0;0 1] for e in FF.cover_edges(P));field)
        mixed=MD.PModule{K}(P,fill(2,4),Dict((u,v)=>gauges[v]*K[0 0;0 1]*inverses[u]
            for (u,v) in FF.cover_edges(P));field)
        E=TamerOp.ext(base,base;maxdeg=2);F=TamerOp.ext(mixed,mixed;maxdeg=2)
        iso=MD.PMorphism(base,mixed,gauges);back=MD.PMorphism(mixed,base,inverses)
        middle=TamerOp.ext(base,mixed;maxdeg=2)
        @test [DF.dim(E,t) for t in 0:2]==[7,4,1]
        # ext_map_first lists its target Ext model before its source model.
        transport=[DF.ext_map_first(F,middle,back;t)*DF.ext_map_second(E,middle,iso;t) for t in 0:2]
        c=zeros(K,DF.dim(E,1));c[1]=1
        s=V.visual_spec(E;kind=:derived_class,degree=1,vector=c).metadata.selection
        @test size(s.incoming,2)>0 && !iszero(s.incoming)
        boundary=DF.boundaries(E,1)[:,1]
        adjusted=V._derived_class_data(E,1,c;representative=s.representative+boundary)
        @test adjusted.coordinates==c
        @test adjusted.quotient_equation.valid
        @test adjusted.incoming*adjusted.boundary_adjustment==boundary
        f=V.visual_spec(F;kind=:derived_class,degree=1,vector=transport[2]*c).metadata.selection
        @test !f.zero_class && f.coordinate_equation.valid
        # Transport the entire product, not just its rank.
        for i in 1:4,j in 1:4
            b=zeros(K,4);a=zeros(K,4);b[i]=1;a[j]=1
            lhs=V.visual_spec(E;kind=:yoneda_product,right=E,p=1,q=1,left_vector=b,right_vector=a,target=E)
            rhs=V.visual_spec(F;kind=:yoneda_product,right=F,p=1,q=1,
                left_vector=transport[2]*b,right_vector=transport[2]*a,target=F)
            @test V._algebra_equation(field,transport[3]*lhs.metadata.selection.coordinates,rhs.metadata.selection.coordinates).valid
        end
    end
end

@testset "A90 native exports" begin
    if Base.find_package("CairoMakie")===nothing
        @test_skip false
    else
        println("A90: loading CairoMakie for native exports");flush(stdout)
        @eval using CairoMakie
        V=TamerOp.Visualization;a=_a90_ext_fixture(CM.QQField());t=_a90_matrix_algebra(CM.QQField())
        specs=[V.visual_spec(a.E14),V.visual_spec(a.E14;kind=:derived_class,degree=2),
            V.visual_spec(a.E24;kind=:yoneda_product,right=a.E12,p=1,q=1,left_vector=[1],right_vector=[1]),
            V.visual_spec(t.A;kind=:product_table,left=[0,0,1,0],right=[0,1,0,0])]
        mktempdir() do dir
            for (i,s) in enumerate(specs), ext in ("png","svg","pdf")
                println("A90: exporting view $i as $ext");flush(stdout)
                path=joinpath(dir,"derived-$i.$ext");V.save_visual(path,s)
                @test filesize(path)>100
            end
        end
        if Base.find_package("WGLMakie")!==nothing
            @eval using WGLMakie
            for s in specs
                fig=V.render(s;backend=:wglmakie)
                @test fig isa CairoMakie.Figure
                empty!(fig)
            end
        end
    end
end
