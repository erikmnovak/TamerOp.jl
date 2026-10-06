# Independent resolution oracles: simple objects on a diamond and a contractible
# extra summand on one vertex. All visualization math runs before rendering.
function _a42c_simple(P,v,field)
    K=CM.coeff_type(field)
    dims=[Int(q==v) for q in 1:FF.nvertices(P)]
    MD.PModule{K}(P,dims,Dict((a,b)=>zeros(K,dims[b],dims[a]) for (a,b) in FF.cover_edges(P));field)
end

@testset "A42c diamond Betti Bass and indicator parity" begin
    V=TamerOp.Visualization
    P=diamond_poset()
    for field in FIELDS_FULL
        M=_a42c_simple(P,1,field)
        N=_a42c_simple(P,4,field)
        rp=DF.projective_resolution(M,TO.ResolutionOptions(maxlen=3))
        ri=DF.injective_resolution(N,TO.ResolutionOptions(maxlen=3))
        for (res,expected,kind) in ((rp,[1 0 0 0;0 1 1 0;0 0 0 1],:betti_table),
                                  (ri,[0 0 0 1;0 1 1 0;1 0 0 0],:bass_table))
            cheap=V.visual_spec(res)
            @test cheap.kind==kind
            @test cheap.metadata.verification.minimality==:not_checked
            @test cheap.metadata.verification.completion==:not_checked
            @test isempty(cheap.metadata.verification.checks)
            @test cheap.metadata.counts[1:3,:]==expected
            @test all(iszero,cheap.metadata.counts[4:end,:])
            checked=V.visual_spec(res;verify=true)
            @test checked.metadata.verification.minimality==:minimal
            @test checked.metadata.verification.completion==:complete
            @test V.check_visual_spec(checked).valid
            s=V.visual_spec(res;kind=:resolution,degree=2,summand=1,vertex=4,verify=true,support_sheets=true)
            @test s.metadata.selection.summand==1
            @test s.metadata.category==:finite_poset_representations
            @test !s.metadata.ambient_free_resolution
            @test V.check_visual_spec(s).valid
            @test all(check -> check.equation.valid,s.metadata.verification.checks)
            @test s.metadata.support_sheet_total==1
            @test s.metadata.incidence.selected_column == (kind==:betti_table ? 1 : nothing)
            @test s.metadata.incidence.selected_row == (kind==:bass_table ? 1 : nothing)
        end
        for (res,expected) in ((IR.upset_resolution(M),[1 0 0 0;0 1 1 0;0 0 0 1]),
                               (IR.downset_resolution(N),[0 0 0 1;0 1 1 0;1 0 0 0]))
            s=V.visual_spec(res;verify=true)
            @test s.metadata.counts==expected
            @test s.metadata.verification.minimality==:minimal
            @test s.metadata.verification.completion==:complete
            @test V.check_visual_spec(V.visual_spec(res;kind=:resolution,degree=1,vertex=2)).valid
        end
        short=DF.projective_resolution(M,TO.ResolutionOptions(maxlen=0))
        @test V.visual_spec(short;verify=true).metadata.verification.completion==:truncated
        @test size(V.visual_spec(short).metadata.counts,1)==1
        @test_throws ArgumentError V.visual_spec(short;kind=:resolution,degree=1)
        z=MD.zero_pmodule(P;field)
        for res in (DF.projective_resolution(z,TO.ResolutionOptions(maxlen=0)),
                    DF.injective_resolution(z,TO.ResolutionOptions(maxlen=0)))
            @test V.visual_spec(res;verify=true).metadata.verification.completion==:complete
            @test V.visual_spec(res;verify=true).metadata.verification.minimality==:minimal
            @test V.check_visual_spec(V.visual_spec(res;kind=:resolution,vertex=1)).valid
            @test_throws ArgumentError V.visual_spec(res;kind=:resolution,summand=1)
        end
    end
end

@testset "A42c grade and request contracts" begin
    V=TamerOp.Visualization
    P=diamond_poset();M=_a42c_simple(P,1,FIELD_QQ)
    res=DF.projective_resolution(M,TO.ResolutionOptions(maxlen=2))
    grades=[(0//1,0//1),(2//3,0//1),(0//1,3//2),(2//3,3//2)]
    s=V.visual_spec(res;kind=:betti_degrees,grades,degree=1,verify=true)
    @test s.metadata.vertices==[2,3]
    @test s.metadata.multiplicities==[1,1]
    @test s.metadata.grades==grades
    @test V.check_visual_spec(s).valid
    @test length(only(filter(x->x isa V.PointLayer,s.layers)).points)==2
    for kwargs in ((;kind=:betti_degrees),(;kind=:betti_degrees,grades=[(0,0) for _ in 1:4]),
        (;kind=:betti_degrees,grades=reverse(grades)),(;kind=:betti_degrees,grades=[(NaN,0) for _ in 1:4]),
        (;kind=:resolution,degree=true),(;kind=:resolution,degree=-1),
        (;kind=:resolution,summand=false),(;kind=:resolution,summand=99),
        (;kind=:resolution,vertex=0),(;kind=:resolution,support_sheets=:yes),
        (;kind=:resolution,matrix_limit=(0,2)),(;verify=:yes),
        (;kind=:resolution,basis_change=(;source=ones(QQ,1,1),target=ones(QQ,1,1))))
        @test !V.check_visual_request(res;kwargs...).valid
        @test_throws ArgumentError V.visual_spec(res;kwargs...)
    end
    cropped=V.visual_spec(res;matrix_limit=(1,2))
    @test size(only(cropped.layers).entries)==(1,2)
    @test cropped.metadata.truncated
    @test size(cropped.metadata.counts)==(3,4)
    wrapped=TamerOp.Results.ResolutionResult(res)
    @test V.visual_spec(wrapped;verify=true).metadata.counts==V.visual_spec(res).metadata.counts
    @test V.available_visuals(wrapped)==V.available_visuals(res)
    huge=big(2)^60
    collisions=[(huge,huge),(huge+1,huge),(huge,huge+1),(huge+1,huge+1)]
    @test_throws ArgumentError V.visual_spec(res;kind=:betti_degrees,grades=collisions)
end

@testset "A42c nonminimal resolutions and invalid certificates" begin
    V=TamerOp.Visualization
    for field in FIELDS_FULL
        K=CM.coeff_type(field);P=chain_poset(1)
        M=MD.PModule{K}(P,[1],Dict{Tuple{Int,Int},Matrix{K}}();field)
        P0=MD.PModule{K}(P,[2],Dict{Tuple{Int,Int},Matrix{K}}();field)
        A=reshape(K[0,1],2,1)
        d=MD.PMorphism(M,P0,[A]);aug=MD.PMorphism(P0,M,[reshape(K[1,0],1,2)])
        r=DF.ProjectiveResolution(M,[P0,M],[[1,1],[1]],[d],[sparse(A)],aug)
        s=V.visual_spec(r;kind=:resolution,degree=1,summand=1,verify=true)
        @test s.metadata.verification.minimality==:nonminimal
        @test s.metadata.verification.completion==:complete
        @test s.metadata.counts==reshape([2,1],2,1)
        dots=V.visual_spec(r;kind=:betti_degrees,grades=[(0,0)],degree=0)
        @test dots.metadata.multiplicities==[2]
        @test length(only(filter(x->x isa V.PointLayer,dots.layers)).points)==1
        bad=DF.ProjectiveResolution(M,[P0,M],[[1,1],[1]],[d],[spzeros(K,2,1)],aug)
        @test_throws ArgumentError V.visual_spec(bad;verify=true)
        missing=DF.ProjectiveResolution(M,[P0,M],[[1,1],[1]],[d],typeof(sparse(A))[],aug)
        @test !V.check_visual_request(missing;verify=true).valid
        @test_throws ArgumentError V.visual_spec(missing;verify=true)
        ri=DF.InjectiveResolution(M,[P0,M],[[1,1],[1]],[aug],d)
        si=V.visual_spec(ri;verify=true)
        @test si.metadata.verification.minimality==:nonminimal
        @test si.metadata.verification.completion==:complete
        # A complex can satisfy d^2=0 without resolving its declared module.
        broken=DF.ProjectiveResolution(M,[P0,M],[[1,1],[1]],
            [MD.zero_morphism(M,P0)],[spzeros(K,2,1)],aug)
        @test_throws ArgumentError V.visual_spec(broken;verify=true)
        if field isa CM.RealField
            tolerant=CM.RealField(Float64;atol=1e-8,rtol=1e-8)
            m=MD.PModule{Float64}(P,[1],Dict{Tuple{Int,Int},Matrix{Float64}}();field=tolerant)
            p0=MD.PModule{Float64}(P,[2],Dict{Tuple{Int,Int},Matrix{Float64}}();field=tolerant)
            a=MD.PMorphism(p0,m,[reshape([1.0,0.0],1,2)])
            for (delta,valid) in ((1e-10,true),(1e-2,false))
                D=reshape([delta,1.0],2,1)
                near=DF.ProjectiveResolution(m,[p0,m],[[1,1],[1]],
                    [MD.PMorphism(m,p0,[D])],[sparse(D)],a)
                if valid
                    evidence=only(V.visual_spec(near;verify=true).metadata.verification.checks).equation
                    @test evidence.valid && !evidence.exact
                    @test 0<evidence.residual<evidence.tolerance
                else
                    @test_throws ArgumentError V.visual_spec(near;verify=true)
                end
            end
        end
    end
end

@testset "A42c presentation incidence, support and graded basis changes" begin
    V=TamerOp.Visualization;IT=TamerOp.IndicatorTypes
    P=chain_poset(2)
    for field in FIELDS_FULL
        K=CM.coeff_type(field)
        U=[FF.principal_upset(P,1),FF.principal_upset(P,2)]
        A=K[1 2;0 0]
        F=IT.UpsetPresentation{K}(P,U,U,copy(transpose(A)),nothing)
        S=K[1 1;0 1];T=K[1 0;0 1]
        s=V.visual_spec(F;degree=1,summand=2,vertex=2,grades=[(0,0),(1,1)],
            support_sheets=true,basis_change=(;source=S,target=T))
        @test s.metadata.coefficient_matrix==A
        @test s.metadata.construction==:cokernel
        @test s.metadata.represented_dimension==1
        @test s.metadata.basis_change.matrix==A*S
        @test s.metadata.basis_change.equation.valid
        @test !s.metadata.basis_change.whole_resolution_transformed
        @test s.metadata.support_sheet_indices==[1,2]
        panel=only(filter(p->p.title=="Differential coefficients",s.panels))
        @test panel.metadata.structural_zero_mask==Bool[0 0;1 0]
        @test endswith(only(panel.layers).entries[2,1],"\u2020")
        @test only(panel.layers).entries[2,2] == (field isa CM.RealField ? "0.0" : "0")
        @test occursin("(1, 1)",only(panel.layers).column_labels[2])
        @test V.check_visual_spec(s).valid
        @test only(filter(p->p.kind==:resolution_grades,s.panels)).metadata.selected_vertex==2
        focused=V.visual_spec(F;degree=1,summand=2,matrix_limit=(1,1),support_sheets=true)
        focused_panel=only(filter(p->p.title=="Differential coefficients",focused.panels))
        @test focused_panel.metadata.displayed_columns==[2]
        @test focused_panel.metadata.column_roles==[:selected]
        @test focused_panel.metadata.matrix==A
        @test focused.metadata.support_sheet_indices==[1,2]
        @test V.check_visual_spec(focused).valid
        @test_throws ArgumentError V.visual_spec(F;basis_change=(;source=K[1 0;1 1],target=T))
        @test_throws ArgumentError V.visual_spec(F;basis_change=(;source=zeros(K,2,2),target=T))
        @test_throws ArgumentError V.visual_spec(F;basis_change=(;source=ones(K,1,1),target=T))
        bad=IT.UpsetPresentation{K}(P,U,U,K[1 1;0 1],nothing)
        @test_throws ArgumentError V.visual_spec(bad)
        D=[FF.principal_downset(P,1),FF.principal_downset(P,2)]
        E=IT.DownsetCopresentation{K}(P,D,D,A,nothing)
        e=V.visual_spec(E;degree=0,summand=2,vertex=2)
        @test e.metadata.construction==:kernel
        @test e.metadata.represented_dimension==1  # active map [0] : K -> K
        @test e.metadata.stalk_matrix==zeros(K,1,1)
        @test V.check_visual_spec(e).valid
    end
    diamond=diamond_poset()
    nonprincipal=FF.Upset(diamond,BitVector([0,1,1,1]))
    F=IT.UpsetPresentation{QQ}(diamond,[nonprincipal],[nonprincipal],ones(QQ,1,1),nothing)
    @test !V.check_visual_request(F).valid
    @test_throws ArgumentError V.visual_spec(F)
end

@testset "A42c native resolution rendering and export" begin
    V=TamerOp.Visualization;P=diamond_poset()
    r=DF.projective_resolution(_a42c_simple(P,1,FIELD_QQ),TO.ResolutionOptions(maxlen=2))
    specs=[V.visual_spec(r;verify=true),
        V.visual_spec(r;kind=:resolution,degree=1,summand=1,vertex=4,verify=true,support_sheets=true),
        V.visual_spec(r;kind=:betti_degrees,degree=1,grades=[(0,0),(1,0),(0,1),(1,1)])]
    for backend in (:cairomakie,:wglmakie)
        name=backend==:cairomakie ? "CairoMakie" : "WGLMakie"
        if Base.find_package(name)===nothing;@test_skip false;continue;end
        if backend==:cairomakie;@eval import CairoMakie;mk=CairoMakie.Makie
        else;@eval import WGLMakie;mk=WGLMakie.Makie;end
        for s in specs
            fig=V.render(s;backend)
            mk.update_state_before_display!(fig)
            @test !isempty(fig.content)
            @test all(a->all(>(0),mk.widths(mk.viewport(a.scene)[])),filter(x->x isa mk.Axis,fig.content))
        end
        if backend==:cairomakie
            mktempdir() do dir
                for ext in ("png","svg","pdf")
                    path=joinpath(dir,"resolution.$ext")
                    V.save_visual(path,specs[2];backend)
                    @test filesize(path)>1000
                end
            end
        end
    end
end
