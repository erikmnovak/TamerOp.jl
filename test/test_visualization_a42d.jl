# Independent staircase oracles: d_1=...=d_(r-1)=0 but d_r is +/-T.
function _a42d_staircase(field; r=2, multiplicity=1, partial=false, horizontal=false)
    K=CM.coeff_type(field)
    dims=zeros(Int,r+3,r)
    dims[1,r]=dims[r+1,1]=multiplicity
    for a in 1:r-1
        dims[a+1,r-a]=dims[a+1,r-a+1]=multiplicity
    end
    dv=[spzeros(K,b<r ? dims[a,b+1] : 0,dims[a,b]) for a in 1:r+3,b in 1:r]
    dh=[spzeros(K,a<r+3 ? dims[a+1,b] : 0,dims[a,b]) for a in 1:r+3,b in 1:r]
    for a in 1:r-1;dv[a+1,r-a]=sparse(Matrix{K}(I,multiplicity,multiplicity));end
    for a in 0:r-1;dh[a+1,r-a]=sparse(Matrix{K}(I,multiplicity,multiplicity));end
    partial && (dh[r,1][end,end]=zero(K))
    if horizontal
        dc=CC.DoubleComplex{K}(0,r-1,0,r+2,permutedims(dims),permutedims(dh),permutedims(dv);field)
    else
        dc=CC.DoubleComplex{K}(0,r+2,0,r-1,dims,dv,dh;field)
    end
    return CC.spectral_sequence(dc;output=:full,first=horizontal ? :horizontal : :vertical)
end

@testset "A42d page and higher differential oracles" begin
    V=TamerOp.Visualization
    for field in FIELDS_FULL, horizontal in (false,true), r in (2,3)
        ss=_a42d_staircase(field;r,horizontal)
        source=horizontal ? (r-1,0) : (0,r-1)
        target=horizontal ? (0,r) : (r,0)
        before=(copy(keys(ss.page_cache)),copy(keys(ss.diff_cache)))
        s=V.visual_spec(ss)
        @test V.check_visual_spec(s).valid
        @test sum(values(s.metadata.dimensions))==2
        @test !s.metadata.limiting_dimensions
        @test (copy(keys(ss.page_cache)),copy(keys(ss.diff_cache)))==before
        unknown=V.visual_spec(ss;page=3,compute=false)
        @test unknown.metadata.page_status==:uncomputed
        @test all(t->t.status==:uncomputed,first(unknown.panels).metadata.terms)
        @test !haskey(ss.page_cache,3)
        d=V.visual_spec(ss;kind=:spectral_differential,page=r,source)
        m=d.metadata.selected
        @test m.target==target
        @test m.rank==1
        @test m.dimension==m.target_data.dimension==1
        @test m.kernel_dimension==m.next_dimension==m.target_data.next_dimension==0
        @test m.target_data.incoming_rank==1
        @test m.equation.valid && m.target_data.equation.valid
        @test V.check_visual_spec(d).valid
        @test d.metadata.extensions==:not_resolved
        @test all(iszero,m.matrix*m.kernel)
        @test all(iszero,m.target_data.matrix*m.image)
        after=V.visual_spec(ss;page=r+1)
        @test all(iszero,values(after.metadata.dimensions))
        @test after.metadata.limiting_dimensions
        limit=V.visual_spec(ss;page=:inf,truncated=true)
        @test limit.metadata.associated_graded && limit.metadata.input_truncated
        @test limit.metadata.extensions==:not_resolved
        tor=DF.TorSpectralSequence{CM.coeff_type(field)}(ss)
        t=V.visual_spec(tor;kind=:spectral_differential,page=r,source=(-source[1],-source[2]))
        @test t.metadata.convention==:homological
        @test t.metadata.selected.target==(-target[1],-target[2])
        @test t.metadata.selected.matrix==m.matrix
        @test CC.dr_source(tor,r,(-source[1],-source[2])) === nothing
        @test CC.dr_target(tor,r,(-target[1],-target[2])) === nothing
    end
end

@testset "A42d multiplicities and next-page quotients" begin
    V=TamerOp.Visualization
    for field in FIELDS_FULL
        ss=_a42d_staircase(field;multiplicity=2,partial=true)
        s=V.visual_spec(ss;kind=:spectral_differential,source=(0,1))
        m=s.metadata.selected
        @test m.dimension==m.target_data.dimension==2
        @test m.rank==m.kernel_dimension==m.next_dimension==m.target_data.next_dimension==1
        @test m.target_data.kernel_dimension==2
        @test size(m.quotient_representatives)==(2,1)
        @test size(m.image)==(2,1)
        @test first(s.panels).metadata.terms[2].dimension==0
        crop=V.visual_spec(ss;kind=:spectral_differential,source=(0,1),window=((0,1),(0,1)),matrix_limit=(1,1))
        @test crop.metadata.display_cropped
        @test crop.metadata.selected.matrix==m.matrix
        @test size(crop.panels[2].layers[1].entries)==(1,1)
        @test size(crop.panels[2].metadata.matrix)==(2,2)
        @test V.check_visual_spec(crop).valid
        zero=V.visual_spec(ss;kind=:spectral_differential,source=(0,0))
        @test size(zero.metadata.selected.matrix)==(0,0)
        @test zero.metadata.selected.rank==0
    end
end

@testset "A42d supplied maps and strict requests" begin
    V=TamerOp.Visualization;ss=_a42d_staircase(FIELD_QQ)
    s=V.visual_spec(ss;structure_maps=[(;source=(0,1),target=(2,0),matrix=ones(QQ,1,1),label="supplied u")])
    @test only(s.metadata.structure_maps).rank==1
    @test only(s.metadata.structure_maps).matrix==ones(QQ,1,1)
    @test V.check_visual_spec(s).valid
    for opts in ((;page=0),(;page=true),(;page=:infty),(;page="2"),(;compute=:yes),
        (;kind=:spectral_differential),(;source=(0,1)),(;kind=:spectral_differential,page=:inf,source=(0,1)),
        (;kind=:spectral_differential,source=(99,1)),(;window=((-1,1),(0,1))),
        (;window=((1,0),(0,1))),(;matrix_limit=(0,2)),(;max_terms=false),(;truncated=1))
        @test !V.check_visual_request(ss;opts...).valid
        @test_throws ArgumentError V.visual_spec(ss;opts...)
    end
    @test_throws ArgumentError V.visual_spec(ss;max_terms=1)
    @test_throws DimensionMismatch V.visual_spec(ss;structure_maps=[(;source=(0,1),target=(2,0),matrix=ones(QQ,2,1),label="bad")])
    @test_throws ArgumentError V.visual_spec(ss;structure_maps=[(;source=(0,1),target=(2,0),matrix=ones(1,1),label="wrong field")])
    @test_throws ArgumentError V.visual_spec(ss;page=3,compute=false,structure_maps=[(;source=(0,1),target=(2,0),matrix=ones(QQ,1,1),label="unknown")])
    missing=V.visual_spec(ss;kind=:spectral_differential,source=(0,1),compute=false)
    @test missing.metadata.differential_status==:uncomputed
    @test !haskey(ss.diff_cache,2)
    # A page beyond the finite-filtration bound uses the retained limit, without
    # overflowing either the bidegree displacement or the next-page index.
    far=V.visual_spec(ss;kind=:spectral_differential,page=typemax(Int),source=(0,1))
    @test far.metadata.associated_graded
    @test far.metadata.selected.target === nothing
    @test far.metadata.selected.next_dimension==0
end

@testset "A42d session atomicity, lazy navigation and lifecycle" begin
    V=TamerOp.Visualization;ss=_a42d_staircase(FIELD_QQ;r=3)
    s=V.inspection_session(ss;compute=false)
    @test TOA.SpectralInspectionSession===V.SpectralInspectionSession
    @test V.check_inspection_session(s).valid
    @test V.check_visual_spec(V.visual_spec(s)).valid
    V.select_inspection!(s;page=3)
    @test V.inspection_summary(s).page_status==:uncomputed
    @test !haskey(ss.page_cache,3)
    V.select_inspection!(s;compute=true,source=(0,2))
    @test V.inspection_snapshot(s).metadata.selected.rank==1
    token=V._on_inspection(s,_ -> error("fixture listener"))
    V.select_inspection!(s;page=4)
    @test V.inspection_selection(s).source===nothing
    @test length(V.inspection_summary(s).listener_errors)==1
    V._off_inspection!(s,token)
    before=V.inspection_snapshot(s);state=V.inspection_selection(s)
    @test_throws ArgumentError V.select_inspection!(s;source=(99,99))
    @test V.inspection_snapshot(s)===before
    @test V.inspection_selection(s)==state
    @test !V.check_inspection_selection(s;page=0).valid
    V.reset_inspection!(s)
    @test V.inspection_selection(s).page==2
    @test !V.inspection_selection(s).compute
    V.close_inspection!(s)
    @test V.check_inspection_session(s).valid
    @test s.object===nothing && isempty(s.listeners)
    @test V.inspection_snapshot(s).kind==:spectral_page
    @test_throws ArgumentError V.select_inspection!(s;page=1)
end

@testset "A42d native Cairo WGL exports and controls" begin
    V=TamerOp.Visualization;ss=_a42d_staircase(FIELD_QQ;multiplicity=2,partial=true)
    if Base.find_package("CairoMakie") !== nothing && Base.find_package("WGLMakie") !== nothing
        @eval using CairoMakie, WGLMakie
        native=V.visual_spec(ss;kind=:spectral_differential,source=(0,1))
        mktempdir() do tmp
            for ext in ("png","svg","pdf")
                path=joinpath(tmp,"spectral.$ext")
                V.save_visual(path,native;backend=:cairomakie)
                @test filesize(path)>100
            end
        end
        for palette in (:accessible,:grayscale)
            @test V.render(native;backend=:cairomakie,style=V.VisualStyle(palette=palette)) isa CairoMakie.Figure
            @test V.render(native;backend=:wglmakie,style=V.VisualStyle(palette=palette)) isa WGLMakie.Figure
        end
        session=V.inspection_session(ss)
        app=V.visualize(session;backend=:wglmakie)
        ext=Base.get_extension(TamerOp,:TamerOpWGLMakieExt)
        ui=ext._INSPECTION_UI_HANDLES[app]
        old=ui.figure[]
        V.select_inspection!(session;source=(0,1))
        @test isempty(V.inspection_summary(session).listener_errors)
        @test ui.figure[]!==old
        @test length(ui.pointer_observers)==2
        ui.controls.page.value[]="0";ui.controls.apply_page.value[]=true
        @test !isempty(ui.error_text[])
        @test V.inspection_selection(session).page==2
        V.select_inspection!(session;page=typemax(Int),source=nothing)
        ui.controls.next.value[]=true
        @test V.inspection_selection(session).page==V.inspection_summary(session).convergence_bound
        @test isempty(ui.error_text[])
        V.close_inspection!(session)
        @test ui.closed[] && isempty(ui.observers) && isempty(ui.pointer_observers)
        @test !haskey(ext._INSPECTION_UI_HANDLES,app)
    else
        @test_skip false
    end
end
