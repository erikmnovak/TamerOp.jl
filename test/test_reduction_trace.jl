include("fixtures/reduction_trace.jl")

# Independent dense row elimination for homology/rank-map oracles. The traced
# algorithm uses column additions and does not call this oracle.
function _a130_kernel_rank(A)
    B=copy(A);K=eltype(B);nr,nc=size(B);pivots=Int[];row=1
    for j in 1:nc
        row>nr && break
        k=findfirst(i->!iszero(B[i,j]),row:nr)
        k===nothing && continue
        i=row+k-1;B[row,:],B[i,:]=copy(B[i,:]),copy(B[row,:])
        B[row,:]./=B[row,j]
        for k in 1:nr
            k==row && continue
            a=B[k,j];B[k,:].-=a.*B[row,:]
        end
        push!(pivots,j);row+=1
    end
    free=setdiff(1:nc,pivots);Z=zeros(K,nc,length(free))
    for (k,j) in enumerate(free)
        Z[j,k]=one(K)
        for (i,p) in enumerate(pivots);Z[p,k]=-B[i,j];end
    end
    Z,length(pivots)
end
function _a130_rank_map(D,cells,a,b,d)
    early=findall(c->c.index<=a && c.dimension==d,cells)
    later=findall(c->c.index<=b && c.dimension==d,cells)
    lower=findall(c->c.index<=a && c.dimension==d-1,cells)
    higher=findall(c->c.index<=b && c.dimension==d+1,cells)
    Z,_=_a130_kernel_rank(D[lower,early]);embedded=zeros(eltype(D),length(later),size(Z,2))
    for (i,k) in enumerate(early);embedded[findfirst(==(k),later),:]=Z[i,:];end
    B=D[later,higher]
    _a130_kernel_rank(hcat(embedded,B))[2]-_a130_kernel_rank(B)[2]
end

@testset "A130 exact reference reduction and independent homology" begin
    O=TamerOp.OrdinaryPersistence;C=TamerOp.CoreModules
    for field in (C.F2(),C.F3(),C.Fp(101)),order in (:sublevel,:superlevel),tied in (false,true)
        G=_a130_complex(;source=false,order,tied)
        r=O.persistence_reduction(G;field,order);K=C.coeff_type(field)
        n=length(r.cells);pairs=O.reduction_pairs(r);ops=O.reduction_operations(r)
        @test O.check_persistence_reduction(r;throw=true).ordinary_checked
        @test [(p.birth_cell,p.death_cell) for p in pairs]==[(1,0),(2,4),(3,5),(6,7)]
        @test count(p->p.zero_length,pairs)==(tied ? 3 : 0)
        @test any(o->o.kind===:add,ops)
        for k in 0:length(ops)
            s=O.reduction_step(r;step=k)
            @test s.D*s.V==s.R
            @test all(i->s.V[i,i]==one(K),1:n)
            @test all(iszero(s.V[i,j]) for i in 1:n for j in 1:i-1)
            @test all(p->p.birth_cell<=s.prefix && p.death_cell<=s.prefix,s.pairs)
            for d in 0:2
                @test count(p->p.dimension==d && p.death_cell==0,s.pairs)==_a130_rank_map(s.D,r.cells,s.prefix,s.prefix,d)
            end
            if s.active>0
                @test s.prefix==s.active-1
                @test all(p->p.death_cell!=s.active,s.pairs)
            end
            for p in s.pairs
                w=O.reduction_witness(r;step=k,birth=p.birth_cell)
                @test iszero(s.D*w.cycle_vector)
                @test all(i->iszero(w.cycle_vector[i]) || i<=p.birth_cell,1:n)
                if w.filling_vector!==nothing
                    @test s.D*w.filling_vector==w.cycle_vector
                    @test all(i->iszero(w.filling_vector[i]) || i<=p.death_cell,1:n)
                end
            end
        end
        # All ranks of all prefix inclusion maps determine the barcode.
        D=r.boundary
        for a in 0:n,b in a:n,d in 0:2
            expected=_a130_rank_map(D,r.cells,a,b,d)
            @test count(p->p.dimension==d && p.birth_cell<=a && (p.death_cell==0 || p.death_cell>b),pairs)==expected
        end
        d=O.persistence_diagram(r)
        sign=order===:sublevel ? 1 : -1
        @test O.finite_intervals(d;dim=1)==(tied ? Tuple{Rational{Int},Rational{Int}}[] : [(sign*4//1,sign*7//1)])
        @test O.essential_births(d;dim=0)==[tied ? 0//1 : sign*(-2//1)]
        @test O.finite_intervals(d;dim=2)==[]
        # Public records and snapshots must not expose mutable retained storage.
        cells=O.reduction_cells(r);empty!(cells);ops=O.reduction_operations(r);empty!(ops)
        s=O.reduction_step(r);fill!(s.D,one(K))
        @test O.check_persistence_reduction(r).valid
    end
    # A degree-two attachment of degree two distinguishes F2 from odd fields.
    G=TamerOp.GradedComplex([[1],[1],[1]],[spzeros(Int,1,1),sparse(reshape([2],1,1))],[(0,),(1,),(2,)])
    r2=O.persistence_reduction(G;field=C.F2());r3=O.persistence_reduction(G;field=C.F3())
    @test O.essential_births(O.persistence_diagram(r2);dim=1)==[1]
    @test O.essential_births(O.persistence_diagram(r2);dim=2)==[2]
    @test O.finite_intervals(O.persistence_diagram(r3);dim=1)==[(1,2)]
    @test O.reduction_witness(r3;birth=2).cycle.coefficients==[C.coeff_type(C.F3())(2)]
    # Nonunit pivot cancellation checks division rather than F2-style XOR.
    G=TamerOp.GradedComplex([[1,2],[1,2]],[sparse([-2 -1;2 1])],[(0,),(0,),(1,),(2,)])
    r=O.persistence_reduction(G;field=C.Fp(101))
    @test only(filter(o->o.kind===:add,O.reduction_operations(r))).coefficient==C.coeff_type(C.Fp(101))(50)
    @test O.essential_births(O.persistence_diagram(r);dim=1)==[2]
    @test O.check_persistence_reduction(r).valid
    # The boundary of a tetrahedron births H2; the solid tetrahedron fills it.
    simplices=[[[i] for i in 1:4],[[i,j] for i in 1:3 for j in i+1:4],
        [[i,j,k] for i in 1:2 for j in i+1:3 for k in j+1:4],[[1,2,3,4]]]
    boundaries=SparseMatrixCSC{Int,Int}[]
    for d in 1:3
        B=zeros(Int,length(simplices[d]),length(simplices[d+1]))
        for (j,simplex) in enumerate(simplices[d+1]),k in eachindex(simplex)
            face=simplex[setdiff(eachindex(simplex),[k])]
            B[findfirst(==(face),simplices[d]),j]=isodd(k) ? 1 : -1
        end
        push!(boundaries,sparse(B))
    end
    G=TamerOp.GradedComplex([collect(1:length(s)) for s in simplices],boundaries,
        vcat(fill((0,),4),fill((1,),6),fill((2,),4),[(5,)]))
    for field in (C.F2(),C.F3(),C.Fp(101))
        r=O.persistence_reduction(G;field);pairs=O.reduction_pairs(r)
        @test O.finite_intervals(O.persistence_diagram(r);dim=2)==[(2,5)]
        @test O.check_persistence_reduction(r).valid
        for a in 0:15,b in a:15,d in 0:3
            @test count(p->p.dimension==d && p.birth_cell<=a && (p.death_cell==0 || p.death_cell>b),pairs)==
                _a130_rank_map(r.boundary,r.cells,a,b,d)
        end
    end
end

@testset "A130 trace budgets, malformed data and API contracts" begin
    O=TamerOp.OrdinaryPersistence;C=TamerOp.CoreModules
    r=_a130_trace();G=_a130_complex(;source=false)
    @test TamerOp.persistence_reduction===O.persistence_reduction
    @test TamerOp.Advanced.reduction_step===O.reduction_step
    @test O.reduction_summary(r).geometry
    @test occursin("reference",sprint(show,r))
    @test O.filtration_order(r)===:sublevel
    @test O.field(r)==C.F3()
    report=O.check_persistence_reduction(r)
    @test isequal(report,O.check_persistence_reduction(r))
    @test hash(report)==hash(O.check_persistence_reduction(r))
    @test O.describe(report).valid
    @test_throws ArgumentError O.ReductionBudget(max_cells=true)
    @test_throws ArgumentError O.persistence_reduction(G;budget=O.ReductionBudget(max_cells=2))
    @test_throws ArgumentError O.persistence_reduction(G;budget=O.ReductionBudget(max_matrix_entries=2))
    @test_throws ArgumentError O.persistence_reduction(G;budget=O.ReductionBudget(max_operations=2))
    @test_throws ArgumentError O.persistence_reduction(G;field=C.QQField())
    @test_throws ArgumentError O.persistence_reduction(_a130_complex();order=:superlevel)
    @test_throws ArgumentError O.reduction_step(r;step=-1)
    @test_throws ArgumentError O.reduction_step(r;step=true)
    @test_throws ArgumentError O.reduction_step(r;step=length(r.operations)+1)
    @test_throws ArgumentError O.reduction_witness(r;birth=7)
    @test_throws ArgumentError O.reduction_witness(r;birth=1,step=0)
    for corrupt in (x->pop!(x.operations),x->(x.boundary[1,1]=one(eltype(x.boundary))),
            x->reverse!(x.cells),x->(x.source.geometry[1][1]=(999.,0.)),
            x->(x.source.complex.grades[1]=(99//1,)))
        bad=deepcopy(r);corrupt(bad)
        # Pure geometry changes are valid supplied geometry; chain/source grade changes are not.
        if bad.source.geometry[1][1]==(999.,0.)
            @test O.check_persistence_reduction(bad).valid
        else
            @test !O.check_persistence_reduction(bad).valid
            @test_throws ArgumentError O.check_persistence_reduction(bad;throw=true)
        end
    end
    bad=deepcopy(r);i=findfirst(o->o.kind===:add,bad.operations);o=bad.operations[i]
    bad.operations[i]=typeof(o)(o.kind,o.column,o.source,zero(o.coefficient))
    @test !O.check_persistence_reduction(bad).valid
    # Empty complexes and isolated cells need no special fabricated operation.
    emptyG=TamerOp.GradedComplex([Int[]],SparseMatrixCSC{Int,Int}[],Tuple{Int}[])
    e=O.persistence_reduction(emptyG)
    @test isempty(O.reduction_operations(e))
    @test O.reduction_step(e).complete
    @test O.check_persistence_reduction(e).valid
    @test O.essential_births(O.persistence_diagram(e);dim=0)==[]
    invalid=TamerOp.GradedComplex([[1],[2]],[sparse(reshape([1],1,1))],[(2,),(1,)])
    @test_throws ArgumentError O.persistence_reduction(invalid)
    @test_throws ArgumentError O.persistence_reduction(TamerOp.GradedComplex([[1,1]],SparseMatrixCSC{Int,Int}[],[(0,),(1,)]))
    invalid=TamerOp.GradedComplex([[1],[2],[3]],[sparse(reshape([1],1,1)),sparse(reshape([1],1,1))],[(0,),(1,),(2,)])
    @test_throws ArgumentError O.persistence_reduction(invalid)
end

@testset "A130 reduction visualization and renderer contracts" begin
    O=TamerOp.OrdinaryPersistence;V=TamerOp.Visualization
    r=_a130_trace();s=V.reduction_explorer(r);last=length(r.operations)
    @test TamerOp.reduction_explorer===V.reduction_explorer
    @test V.available_visuals(r)==(:persistence_reduction,)
    @test V.check_visual_spec(V.visual_spec(s)).valid
    @test V.inspection_snapshot(s).metadata.prefix==0
    @test isempty(V.inspection_snapshot(s).metadata.pairs)
    @test_throws ArgumentError V.reduction_explorer(r;max_cells=2)
    @test_throws ArgumentError V.visual_spec(r;matrix=:unknown)
    V.select_inspection!(s;step=last,birth=6,cell=7,chain=:filling)
    snap=V.inspection_snapshot(s)
    @test V.check_visual_spec(snap).valid
    @test snap.panels[1].metadata.chain_coefficients==Dict(7=>"1")
    @test snap.panels[1].metadata.geometry
    @test snap.panels[2].metadata.matrix==snap.metadata.R
    @test snap.panels[3].metadata.parameter===:cell_order
    @test snap.metadata.witness.pair.birth==4//1
    @test snap.metadata.witness.pair.death==7//1
    before=V.inspection_selection(s)
    @test_throws ArgumentError V.select_inspection!(s;step=last+1)
    @test V.inspection_selection(s)==before
    @test !V.check_inspection_selection(s;birth=7).valid
    @test !V.check_inspection_selection(s;cell=true).valid
    @test !V.check_inspection_selection(s;chain=:shortest).valid
    catalog=V.visual_spec(O.persistence_reduction(_a130_complex(;source=false));step=2)
    @test !catalog.panels[1].metadata.geometry
    V.select_inspection!(s;step=0)
    @test V.inspection_selection(s).birth==0
    # Session input is independent of the caller, including source geometry.
    r.source.geometry[1][1]=(999.,999.)
    @test s.object.source.geometry[1][1]==(-2.,1.)
    for backend in (:cairomakie,:wglmakie)
        ext=Base.get_extension(TamerOp,backend===:cairomakie ? :TamerOpCairoMakieExt : :TamerOpWGLMakieExt)
        if ext===nothing;@test_skip false;continue;end
        M=backend===:cairomakie ? ext.CairoMakie.Makie : ext.WGLMakie.Makie
        fig=V.render(snap;backend);axes=filter(a->a isa M.Axis,fig.content)
        M.update_state_before_display!(fig)
        @test all(label->minimum(label.layoutobservables.computedbbox[])[2]>=0,filter(a->a isa M.Label,fig.content))
        @test axes[2].xlabel[]=="Column"
        @test axes[2].ylabel[]=="Row"
        heat=only(filter(plot->plot isa M.Heatmap,axes[2].scene.plots))
        @test heat[3][]==permutedims(Float64.(.!iszero.(reverse(snap.metadata.R;dims=1))))
        points=[Tuple(p) for plot in axes[1].scene.plots if plot isa M.Scatter for p in plot[1][]]
        @test (-2.,1.) in points
        @test (3.,1.) in points
        texts=[plot for plot in axes[2].scene.plots if plot isa M.Text]
        @test !isempty(texts)
        empty!(fig)
        if backend===:wglmakie
            ui=ext._inspection_ui(V.visualize(s;backend));canvas=ui.figure[]
            ui.controls.finish.value[]=true;wait(s.requests.task)
            @test V.inspection_selection(s).step==last
            ui.controls.chain.option_index[]=2;wait(s.requests.task)
            ui.controls.birth.value[]="6"
            ui.controls.apply.value[]=true;wait(s.requests.task)
            @test V.inspection_selection(s).birth==6
            ui.controls.previous.value[]=true;wait(s.requests.task)
            @test V.inspection_selection(s).step==last-1
            ui.controls.step.value[]="-1";ui.controls.apply.value[]=true
            @test occursin("step",lowercase(ui.error_text[])) || (wait(s.requests.task);occursin("step",lowercase(ui.error_text[])))
            V.reset_inspection!(s)
            @test ui.figure[]===canvas && ui.rebuild_count[]==1
            @test ui.update_count[]>0
            ui.dispose();@test isempty(ui.observers) && isempty(ui.pointer_observers)
        end
    end
    if Base.get_extension(TamerOp,:TamerOpCairoMakieExt)!==nothing
        mktempdir() do dir
            for suffix in ("svg","png","pdf")
                path=joinpath(dir,"reduction.$suffix");V.save_visual(path,snap;backend=:cairomakie)
                @test filesize(path)>1000
            end
            V.reset_inspection!(s);start=V.visual_state(s)
            addition=findfirst(o->o.kind===:add,s.object.operations)
            V.select_inspection!(s;step=addition);during=V.visual_state(s)
            V.select_inspection!(s;step=last,birth=6,chain=:filling);finished=V.visual_state(s)
            file=get(ENV,"TAMEROP_REDUCTION_HTML",joinpath(dir,"reduction.html"));mkpath(dirname(file))
            V.save_visual_session(file,(;start,during,finished));@test filesize(file)>1000
        end
    end
    V.close_inspection!(s);@test V.check_inspection_session(s).valid
    @test s.object===nothing
    @test_throws ArgumentError V.select_inspection!(s;step=0)
end
