# Independent finite-diagram oracles and editor transactions.
function _a99_diamond(;field=TamerOp.CoreModules.QQField(),history_limit=32)
    TamerOp.Visualization.module_editor(;order=Bool[1 1 1 1;0 1 0 1;0 0 1 1;0 0 0 1],
        dimensions=ones(Int,4),maps=Dict(e=>ones(Int,1,1) for e in ((1,2),(1,3),(2,4),(3,4))),field,history_limit)
end
function _a99_all_paths(edges,u,v)
    u==v && return [[u]]
    result=Vector{Int}[]
    for (a,b) in edges
        a==u || continue
        append!(result,[[u;path] for path in _a99_all_paths(edges,b,v)])
    end
    result
end

@testset "A99 path witnesses agree with independent path enumeration" begin
    V,C,FF,MO=TamerOp.Visualization,TamerOp.CoreModules,TamerOp.FiniteFringe,TamerOp.Modules
    for field in (C.QQField(),C.F2(),C.F3(),C.Fp(101),C.RealField(Float64))
        editor=_a99_diamond(;field);object=V.editor_object(editor)
        @test MO.module_path_witness(object)===nothing
        @test MO.structure_map(object;source=1,target=4)==fill(C.coerce(field,1),1,1)
        V.edit_module!(editor;map=((3,4),fill(2,1,1)))
        report=V.check_module_editor(editor)
        @test !report.valid
        @test report.witness.first_path==[1,2,4]
        @test report.witness.second_path==[1,3,4]
        @test report.witness.first_composite==fill(C.coerce(field,1),1,1)
        @test report.witness.second_composite==fill(C.coerce(field,2),1,1)
        @test_throws ArgumentError V.editor_object(editor)
        @test_throws ArgumentError V.editor_code(editor)
        @test_throws ArgumentError V.check_module_editor(editor;throw=true)
        @test V.check_visual_spec(V.inspection_snapshot(editor)).valid
        # An invertible basis change at each vertex gives asymmetric 2x2 maps.
        P=FF.FinitePoset(Bool[1 1 1 1;0 1 0 1;0 0 1 1;0 0 0 1])
        basis=[C.coerce.(Ref(field),[1 i;0 1]) for i in 1:4]
        inverse=[C.coerce.(Ref(field),[1 -i;0 1]) for i in 1:4]
        maps=Dict((u,v)=>basis[v]*inverse[u] for (u,v) in FF.cover_edges(P))
        changed=MO.PModule{C.coeff_type(field)}(P,fill(2,4),maps;field)
        @test MO.module_path_witness(changed)===nothing
        @test MO.structure_map(changed;source=1,target=4)==basis[4]*inverse[1]
        # Random acyclic relations, closed independently, with non-topological IDs.
        rng=MersenneTwister(9901)
        for trial in 1:12
            n=6;L=Matrix{Bool}(I,n,n)
            for i in 1:n,j in i+1:n;L[i,j]=rand(rng,Bool);end
            for k in 1:n,i in 1:n,j in 1:n;L[i,j]|=L[i,k]&&L[k,j];end
            permutation=randperm(rng,n);L=L[permutation,permutation]
            Q=FF.FinitePoset(L);edges=collect(FF.cover_edges(Q))
            values=Dict(e=>reshape([C.coerce(field,rand(rng,0:3))],1,1) for e in edges)
            candidate=MO.PModule{C.coeff_type(field)}(Q,ones(Int,n),values;field)
            valid=true
            for u in 1:n,v in 1:n
                paths=_a99_all_paths(edges,u,v)
                isempty(paths) && continue
                products=[prod((only(values[a,b]) for (a,b) in zip(p[1:end-1],p[2:end]));init=one(C.coeff_type(field))) for p in paths]
                valid &= all(==(first(products)),products)
            end
            @test (MO.module_path_witness(candidate)===nothing)==valid
            @test MO.check_module_functoriality(candidate).valid==valid
        end
    end
    # Two unequal paths of lengths two and three: no elementary diamond.
    L=Bool[1 1 1 1 1;0 1 0 0 1;0 0 1 1 1;0 0 0 1 1;0 0 0 0 1]
    P=FF.FinitePoset(L);maps=Dict(e=>ones(C.QQ,1,1) for e in FF.cover_edges(P))
    maps[4,5].=2
    M=MO.PModule{C.QQ}(P,ones(Int,5),maps;field=C.QQField())
    w=MO.module_path_witness(M)
    @test w!==nothing
    @test sort([length(w.first_path),length(w.second_path)])==[3,4]
    @test Set([only(w.first_composite),only(w.second_composite)])==Set([1,2])
    # A previous cache hit cannot hide the subsequent raw-map mutation.
    MO.structure_map(M;source=1,target=5)
    @test MO.module_path_witness(M)!==nothing
end

@testset "A99 mathematical drafts, selection and bounded transactional history" begin
    V=TamerOp.Visualization
    e=_a99_diamond(;history_limit=2)
    @test TamerOp.module_editor===V.module_editor
    @test TamerOp.Advanced.undo_module_edit! === V.undo_module_edit!
    @test V.check_visual_request(e).valid
    @test V.check_visual_spec(V.visual_spec(e)).valid
    external=V.editor_draft(e);external.maps[3,4].=9
    @test V.editor_draft(e).maps[3,4]==ones(Int,1,1)
    owned=V.editor_object(e);owned.edge_maps[3,4].=7
    @test V.check_module_editor(e).valid
    initial=V.inspection_snapshot(e)
    V.select_inspection!(e;vertex=3,edge=(3,4))
    @test V.inspection_summary(e).undo_count==0
    V.edit_module!(e;map=((3,4),fill(2,1,1)))
    @test V.inspection_snapshot(e)!==initial
    @test V.check_inspection_session(e).valid
    V.undo_module_edit!(e);@test V.check_module_editor(e).valid
    V.redo_module_edit!(e);@test !V.check_module_editor(e).valid
    V.edit_module!(e;dimensions=[1,1,1,2])
    @test occursin("size",join(V.check_module_editor(e).issues))
    @test size(V.editor_draft(e).maps[3,4])==(1,1)
    V.edit_module!(e;map=((2,4),ones(Int,2,1)))
    V.edit_module!(e;map=((3,4),ones(Int,2,1)))
    @test V.check_module_editor(e).valid
    @test size(TamerOp.Modules.structure_map(V.editor_object(e);source=1,target=4))==(2,1)
    @test V.inspection_summary(e).undo_count==2
    V.undo_module_edit!(e);V.edit_module!(e;map=((3,4),ones(Int,2,1)))
    @test V.inspection_summary(e).redo_count==0
    @test_throws ArgumentError V.redo_module_edit!(e)
    before=V.inspection_selection(e)
    @test_throws ArgumentError V.edit_module!(e;dimensions=[999999,1,1,1])
    @test_throws ArgumentError V.edit_module!(e;map=((1,2),["run"] ))
    @test_throws ArgumentError V.edit_module!(e;map=(("1",2),[1;;]))
    @test_throws ArgumentError V.edit_module!(e;map=((1,2),zeros(Int,100000,0)))
    @test_throws ArgumentError V.select_inspection!(e;vertex=5)
    @test V.inspection_selection(e)==before
    V.reset_inspection!(e)
    cyclic=V.editor_draft(e).order;cyclic[4,1]=true
    V.edit_module!(e;order=cyclic)
    @test !V.check_module_editor(e).valid
    @test V.check_visual_spec(V.inspection_snapshot(e)).valid
    V.undo_module_edit!(e)
    nontransitive=V.editor_draft(e).order;nontransitive[1,4]=false
    V.edit_module!(e;order=nontransitive)
    @test !V.check_module_editor(e).valid
    V.undo_module_edit!(e)
    V.edit_module!(e;map=((1,2),nothing));@test !V.check_module_editor(e).valid
    V.undo_module_edit!(e)
    V.edit_module!(e;map=((1,4),ones(Int,1,1)));@test !V.check_module_editor(e).valid
    V.undo_module_edit!(e)
    V.edit_module!(e;dimensions=[-1,1,1,1]);@test !V.check_module_editor(e).valid
    V.undo_module_edit!(e)
    # Failed listeners do not roll back the valid committed draft; reentry fails.
    token=V._on_inspection(e,s->V.edit_module!(s;map=((3,4),ones(Int,1,1))))
    V.edit_module!(e;map=((3,4),fill(2,1,1)))
    @test !isempty(V.inspection_summary(e).listener_errors)
    @test !V.check_module_editor(e).valid
    V._off_inspection!(e,token)
    V.close_inspection!(e);V.close_inspection!(e)
    @test V.check_inspection_session(e).valid
    @test V.check_visual_spec(V.inspection_snapshot(e)).valid
    @test_throws ArgumentError V.editor_draft(e)
    @test_throws ArgumentError V.undo_module_edit!(e)
    @test_throws ArgumentError V.editor_object(e)
    @test occursin("ModuleEditorSession",sprint(show,e))
    nohistory=_a99_diamond(;history_limit=0)
    V.edit_module!(nohistory;map=((3,4),fill(2,1,1)))
    @test V.inspection_summary(nohistory).undo_count==0
    @test_throws ArgumentError V.undo_module_edit!(nohistory)
    orderedit=V.module_editor(;order=Bool[1 1;0 1],dimensions=[1,1],maps=Dict((1,2)=>[1;;]))
    V.edit_module!(orderedit;order=Bool[1 0;0 1],maps=Dict())
    @test V.check_module_editor(orderedit).valid
    @test isempty(TamerOp.FiniteFringe.cover_edges(V.editor_object(orderedit).Q))
    V.undo_module_edit!(orderedit)
    @test V.inspection_selection(orderedit).edge==(1,2)
end

@testset "A99 matrix edits publish one coherent selection to linked viewers" begin
    V,C=TamerOp.Visualization,TamerOp.CoreModules
    for field in (C.QQField(),C.F2(),C.F3(),C.Fp(5),C.RealField(Float64))
        editor=_a99_diamond(;field)
        before=V.inspection_selection(editor)
        observed=NamedTuple[]
        token=V._on_inspection(editor,s -> begin
            push!(observed,(;selection=V.inspection_selection(s),draft=V.editor_draft(s)))
        end)
        V._edit_module!(editor,true;map=((3,4),fill(2,1,1)))
        @test length(observed)==1
        @test only(observed).selection.edge==(3,4)
        @test only(observed).selection.revision==before.revision+1
        @test only(observed).draft.maps[(3,4)]==fill(2,1,1)
        @test !V.check_module_editor(editor).valid
        @test_throws ArgumentError V._edit_module!(editor,true;map=((0,4),fill(2,1,1)))
        @test length(observed)==1
        @test V.inspection_selection(editor)==only(observed).selection
        V.undo_module_edit!(editor)
        @test V.inspection_selection(editor).edge==before.edge
        @test V.editor_draft(editor).maps[(3,4)]==ones(Int,1,1)
        V.redo_module_edit!(editor)
        @test V.inspection_selection(editor).edge==(3,4)
        @test V.editor_draft(editor).maps[(3,4)]==fill(2,1,1)
        # Programmatic edits retain the caller's selection as before.
        V.edit_module!(editor;map=((1,2),fill(2,1,1)))
        @test V.inspection_selection(editor).edge==(3,4)
        @test V.check_module_editor(editor).valid
        V._off_inspection!(editor,token)
        V.close_inspection!(editor)
    end
end

@testset "A99 presentations, exact text, field reinterpretation and empty shapes" begin
    V,C,FF=TamerOp.Visualization,TamerOp.CoreModules,TamerOp.FiniteFringe
    order=Bool[1 1 1;0 1 1;0 0 1]
    e=V.module_editor(;order,upsets=[[2,3]],downsets=[[1,2]],matrix=fill(1,1,1))
    @test V.check_module_editor(e).valid
    H=V.editor_object(e)
    @test TamerOp.dimensions(H).fibers==[0,1,0]
    V.select_inspection!(e;vertex=2)
    @test V.check_visual_spec(V.inspection_snapshot(e)).valid
    @test V.check_module_editor(V.module_editor(H)).valid
    V.edit_module!(e;matrix=zeros(Int,1,1))
    @test TamerOp.dimensions(V.editor_object(e)).fibers==[0,0,0]
    V.undo_module_edit!(e)
    V.edit_module!(e;upsets=[[2]])
    @test !V.check_module_editor(e).valid
    V.undo_module_edit!(e)
    V.edit_module!(e;upsets=[[3]],downsets=[[1]])
    @test !V.check_module_editor(e).valid
    V.edit_module!(e;matrix=zeros(Int,1,1))
    @test V.check_module_editor(e).valid
    V.edit_module!(e;upsets=[[0,3]])
    @test !V.check_module_editor(e).valid
    V.reset_inspection!(e)
    V.edit_module!(e;matrix=fill(1//2,1,1),field=C.F2())
    @test !V.check_module_editor(e).valid
    V.edit_module!(e;field=C.F3())
    @test only(FF.fringe_coefficients(V.editor_object(e)))==C.coerce(C.F3(),2)
    @test V._editor_parse_matrix("1/3 2; -4.5 1e-2")==[1//3 2;-9//2 1//100]
    @test size(V._editor_parse_matrix("";shape=(2,0)))==(2,0)
    for text in ("1; 2 3","1/0","Inf","NaN","run(`echo no`)","begin 1 end")
        @test_throws ArgumentError V._editor_parse_matrix(text)
    end
    @test_throws ArgumentError V._editor_parse_matrix("1";shape=(10^8,10^8))
    @test_throws ArgumentError V._editor_parse_matrix("";shape=(10^8,0))
    # Non-square and zero-dimensional matrices retain orientation and shape.
    for field in (C.QQField(),C.F2(),C.F3(),C.Fp(101),C.RealField(Float64))
        editor=V.module_editor(;order=Bool[1 1;0 1],dimensions=[3,2],maps=Dict((1,2)=>[1 2 3;4 5 6]),field)
        A=TamerOp.Modules.structure_map(V.editor_object(editor);source=1,target=2)
        @test A==C.coerce.(Ref(field),[1 2 3;4 5 6])
        V.edit_module!(editor;dimensions=[0,2],maps=Dict((1,2)=>zeros(Int,2,0)))
        @test size(TamerOp.Modules.structure_map(V.editor_object(editor);source=1,target=2))==(2,0)
        @test V.check_visual_spec(V.inspection_snapshot(editor)).valid
        exported=Base.include_string(Module(:EmptyAxisCode),V.editor_code(editor))
        @test size(TamerOp.Modules.structure_map(exported;source=1,target=2))==(2,0)
    end
    empty=V.module_editor(;order=falses(0,0),dimensions=Int[],maps=Dict())
    @test V.check_module_editor(empty).valid
    @test V.check_visual_spec(V.inspection_snapshot(empty)).valid
    numerical=V.module_editor(;order=Bool[1 1;0 1],dimensions=[1,1],maps=Dict((1,2)=>[Inf;;]),field=C.RealField(Float64))
    @test !V.check_module_editor(numerical).valid
    raw=TamerOp.Modules.PModule{Float64}(FF.FinitePoset(Bool[1 1;0 1]),[1,1],Dict((1,2)=>[Inf;;]);field=C.RealField(Float64))
    @test_throws ArgumentError TamerOp.Modules.module_path_witness(raw)
end

@testset "A99 code and saved draft round trips" begin
    V,C=TamerOp.Visualization,TamerOp.CoreModules
    for field in (C.QQField(),C.F2(),C.F3(),C.Fp(101),C.RealField(Float64))
        e=_a99_diamond(;field)
        code=V.editor_code(e)
        result=Base.include_string(Module(:EditorCodeFixture),code)
        @test TamerOp.Modules.check_module_functoriality(result).valid
        @test TamerOp.Modules.structure_map(result;source=1,target=4)==fill(C.coerce(field,1),1,1)
        @test V.check_module_editor(V.module_editor(result)).valid
    end
    H=V.module_editor(;order=Bool[1 1;0 1],upsets=[[1,2],[2]],downsets=[[1],[1,2]],matrix=[1 0;0 2])
    restored=Base.include_string(Module(:EditorPresentationCode),V.editor_code(H))
    @test TamerOp.FiniteFringe.fringe_coefficients(restored)==[1 0;0 2]
    @test TamerOp.dimensions(restored).fibers==[1,1]
    mktempdir() do dir
        for e in (_a99_diamond(),H)
            path=joinpath(dir,"draft.json")
            V.save_visual_session(path,e)
            fresh=V.module_editor(V.load_visual_session(path))
            @test V.check_module_editor(fresh).valid
            @test V.editor_draft(fresh)==V.editor_draft(e)
            @test V.inspection_summary(fresh).undo_count==0
        end
        e=_a99_diamond();V.edit_module!(e;map=((3,4),fill(2,1,1)))
        V.save_visual_session(joinpath(dir,"invalid.json"),e)
        fresh=V.module_editor(V.load_visual_session(joinpath(dir,"invalid.json")))
        @test !V.check_module_editor(fresh).valid
        @test V.check_module_editor(fresh).witness.first_composite==ones(C.QQ,1,1)
        @test_throws ArgumentError V.editor_object(fresh)
    end
end

@testset "A99 native editor rendering and figure exports" begin
    V=TamerOp.Visualization
    if Base.find_package("CairoMakie")===nothing
        @test_skip false
    else
        @eval using CairoMakie
        e=_a99_diamond();V.edit_module!(e;map=((3,4),fill(2,1,1)))
        snapshot=V.inspection_snapshot(e)
        @test snapshot.panels[3].metadata.matrix==ones(Int,1,1)
        @test snapshot.panels[4].metadata.matrix==fill(2,1,1)
        @test length(filter(x->x isa V.SegmentLayer,snapshot.panels[1].layers))>=3
        @test V.check_visual_spec(snapshot).valid
        mktempdir() do dir
            for suffix in ("svg","png","pdf")
                path=joinpath(dir,"failed-paths."*suffix)
                V.save_visual(path,snapshot;backend=:cairomakie)
                @test isfile(path) && filesize(path)>1000
            end
            path=joinpath(dir,"draft.html")
            V.save_visual_session(path,e)
            @test !V.check_module_editor(V.module_editor(V.load_visual_session(path))).valid
        end
        if Base.find_package("WGLMakie")!==nothing
            @eval using WGLMakie
            app=V.visualize(e;backend=:wglmakie)
            ext=Base.get_extension(TamerOp,:TamerOpWGLMakieExt)
            ui=ext._inspection_ui(app)
            @test isempty(V.inspection_summary(e).listener_errors)
            @test !ui.closed[]
            ext._dispose_inspection_ui!(app)
            @test V.inspection_summary(e).closed
            for T in (Float16,Float32,Float64)
                numeric=_a99_diamond(;field=TamerOp.CoreModules.RealField(T))
                viewer=V.visualize(numeric;backend=:wglmakie)
                @test ext._inspection_ui(viewer).controls.field.value[]=="Real($T)"
                ext._dispose_inspection_ui!(viewer)
            end
        end
    end
end
