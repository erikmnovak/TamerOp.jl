# Independent dense arithmetic on actual quotients, with no cone reduction.
function _extended_oracle_rref(A,p)
    R=mod.(BigInt.(A),p);pivots=Int[];row=1
    for col in axes(R,2)
        row>size(R,1) && break
        k=findfirst(i->!iszero(R[i,col]),row:size(R,1));k===nothing && continue
        k+=row-1;R[row,:],R[k,:]=copy(R[k,:]),copy(R[row,:])
        R[row,:]=mod.(R[row,:].*invmod(R[row,col],big(p)),p)
        for j in axes(R,1)
            j==row && continue
            R[j,:]=mod.(R[j,:].-R[j,col].*R[row,:],p)
        end
        push!(pivots,col);row+=1
    end
    return R,pivots
end
_extended_oracle_rank(A,p)=length(last(_extended_oracle_rref(A,p)))
function _extended_oracle_kernel(A,p)
    R,pivots=_extended_oracle_rref(A,p);free=setdiff(collect(axes(A,2)),pivots)
    Z=zeros(BigInt,size(A,2),length(free))
    for (j,col) in enumerate(free)
        Z[col,j]=1
        for (i,v) in enumerate(pivots);Z[v,j]=mod(-R[i,col],p);end
    end
    return Z
end

function _extended_oracle_complex(simplices,values)
    # Explicitly enumerate all nonempty subsets, independent of the production
    # face-closure queue; fixtures have at most four vertices per simplex.
    cells=Set{Tuple{Vararg{Int}}}((i,) for i in eachindex(values))
    for s in simplices,mask in 1:(2^length(s)-1)
        push!(cells,Tuple(sort([s[j] for j in eachindex(s) if (mask>>(j-1))&1==1])))
    end
    groups=[sort!([s for s in cells if length(s)==d]) for d in 1:maximum(length,cells;init=1)]
    B=Matrix{Int}[]
    for d in 2:length(groups)
        A=zeros(Int,length(groups[d-1]),length(groups[d]))
        for (j,s) in enumerate(groups[d]), (i,t) in enumerate(groups[d-1])
            if issubset(t,s)
                missing=only(setdiff(s,t));k=findfirst(==(missing),s)
                A[i,j]=(-1)^(k-1)
            end
        end
        push!(B,A)
    end
    return groups,B
end

function _extended_oracle_snapshot(groups,B,values,order,phase,t,dim)
    firstactive(v)=order===:sublevel ? v<=t : v>=t
    secondactive(v)=order===:sublevel ? v>=t : v<=t
    ids=[findall(s->phase===:ordinary ? all(v->firstactive(values[v]),s) :
        !all(v->secondactive(values[v]),s),g) for g in groups]
    n=length(ids[dim+1]);D=dim==0 ? zeros(Int,0,n) : B[dim][ids[dim],ids[dim+1]]
    U=dim+2>length(groups) ? zeros(Int,n,0) : B[dim+1][ids[dim+1],ids[dim+2]]
    return ids[dim+1],D,U
end

@testset "A119 known height functions, essential classes and ties" begin
    OP=TamerOp.OrdinaryPersistence
    circle=[[1,2],[2,3],[3,4],[1,4]]
    for F in (CM.F2(),CM.F3(),CM.Fp(101)),order in (:sublevel,:superlevel)
        d=OP.extended_persistence(circle,[0,1,3,2];field=F,order)
        a=only(OP.extended_intervals(d;dim=0));b=only(OP.extended_intervals(d;dim=1))
        @test (a.birth,a.death)==(order===:sublevel ? (0,3) : (3,0))
        @test (b.birth,b.death)==(order===:sublevel ? (3,0) : (0,3))
        @test a.kind===:extended_positive && b.kind===:extended_negative
        @test a.birth_phase===b.birth_phase===:ordinary
        @test a.death_phase===b.death_phase===:relative
        @test a.birth_included && !a.death_included
        @test OP.check_extended_persistence(d;recompute=true).valid
        @test OP.field(d)==F && OP.filtration_order(d)==order
        # Constant functions have nonempty cross-phase bars, in every Betti degree.
        c=OP.extended_persistence(circle,fill(7//3,4);field=F,order)
        @test OP.extended_persistence_summary(c).types.extended_zero==2
        @test all(r->r.birth==r.death==7//3 && r.birth_step<r.death_step,
            [OP.extended_intervals(c;dim=0);OP.extended_intervals(c;dim=1)])
        # Fill the circle by a disk: the top-dimensional extended class disappears.
        disk=OP.extended_persistence([[1,2,3],[1,3,4]],[0,1,3,2];field=F,order)
        @test isempty(OP.extended_intervals(disk;dim=1))
        @test isempty(OP.extended_intervals(disk;dim=2))
        # A path with two low ends has an ordinary H0 bar and one extended H0 bar.
        path=OP.extended_persistence([[1,2],[2,3]],[0,3,1];field=F,order)
        if order===:sublevel
            @test [(r.birth,r.death) for r in OP.extended_intervals(path;dim=0,kind=:ordinary)]==[(1,3)]
        else
            @test [(r.birth,r.death) for r in OP.extended_intervals(path;dim=1,kind=:relative)]==[(1,3)]
        end
        points=OP.extended_persistence(Vector{Int}[],[0,4];field=F,order)
        @test length(OP.extended_intervals(points;dim=0,kind=:extended_zero))==2
        empty=OP.extended_persistence(Vector{Int}[],Int[];field=F,order)
        @test isempty(OP.extended_intervals(empty;dim=0))
    end
end

@testset "A119 independent persistent quotient-map oracle" begin
    OP=TamerOp.OrdinaryPersistence
    rng=MersenneTwister(119)
    fixtures=[([[1,2],[2,3],[3,4],[1,4]],[0,2,5,3]),
        ([[1,2],[2,3],[2,4]],[0,3,1,4]),
        ([[1,2,3],[1,2,4],[1,3,4],[2,3,4]],[0,1,1,3]),
        ([[1,2,3,4]],[0,3,2,1])]
    for _ in 1:5
        edges=[collect(c) for c in ((1,2),(1,3),(1,4),(2,3),(2,4),(3,4)) if rand(rng,Bool)]
        push!(fixtures,(edges,rand(rng,-2:3,4)))
    end
    for (simplices,values) in fixtures,F in (CM.F2(),CM.F3(),CM.Fp(101)),order in (:sublevel,:superlevel)
        d=OP.extended_persistence(simplices,values;field=F,order)
        groups,B=_extended_oracle_complex(simplices,values)
        events=OP.extended_steps(d)
        snapshots=[(;step=0,phase=:ordinary,value=order===:sublevel ? minimum(values)-1 : maximum(values)+1);events]
        for dim in 0:length(groups)-1
            rows=OP.extended_intervals(d;dim)
            models=[_extended_oracle_snapshot(groups,B,values,order,s.phase,s.value,dim) for s in snapshots]
            for i in eachindex(snapshots),j in i:length(snapshots)
                si,ti=snapshots[i],snapshots[j];sids,Ds,_=models[i];tids,_,Bt=models[j]
                Z=_extended_oracle_kernel(Ds,F.p)
                m=zeros(BigInt,length(tids),length(sids))
                for (a,id) in enumerate(sids)
                    b=findfirst(==(id),tids);b===nothing || (m[b,a]=1)
                end
                expected=_extended_oracle_rank(hcat(Bt,m*Z),F.p)-_extended_oracle_rank(Bt,F.p)
                @test count(r->r.birth_step<=si.step && ti.step<r.death_step,rows)==expected
                # Selected cross-junction and adjacent maps exercise the exposed
                # actual matrices without constructing O(n^2) homology models.
                if j==i || j==i+1 || (si.phase===:ordinary && ti.phase===:relative && j==length(snapshots)-1)
                    actual=OP.extended_structure_map(d;source=(si.phase,si.value),target=(ti.phase,ti.value),dim)
                    @test _extended_oracle_rank(getfield.(Matrix(actual),:val),F.p)==expected
                    cm=OP.extended_structure_map(d;source=(si.phase,si.value),target=(ti.phase,ti.value),dim,chains=true)
                    @test getfield.(Matrix(cm),:val)==mod.(m,F.p)
                end
            end
        end
    end
end

@testset "A119 cubical functions, signs, exact grades and contracts" begin
    OP=TamerOp.OrdinaryPersistence
    for F in (CM.F2(),CM.F3(),CM.Fp(101)),order in (:sublevel,:superlevel)
        rp2=[[1,2,3],[1,2,4],[1,3,5],[1,4,6],[1,5,6],[2,3,6],[2,4,5],[2,5,6],[3,4,5],[3,4,6]]
        r=OP.extended_persistence(rp2,fill(0,6);field=F,order)
        @test length.(r.intervals)==(F.p==2 ? [1,1,1] : [1,0,0])
        # A periodic one-vertex cube has cancelling incidences; no vertex
        # support may be inferred from its zero boundary columns.
        d=OP.extended_persistence(fill(4,1,1);periodic=true,field=F,order)
        @test length.(d.intervals)==[1,2,1]
        @test all(r->r.kind===:extended_zero,reduce(vcat,d.intervals))
        c=OP.extended_persistence([0,3,1];field=F,order)
        s=OP.extended_persistence([[1,2],[2,3]],[0,3,1];field=F,order)
        @test c.intervals==s.intervals
    end
    for values in ([big(2)^80,big(2)^80+1],[1//7,5//3],setprecision(256) do;[BigFloat(1),BigFloat(1)+big"1e-60"];end)
        d=OP.extended_persistence([[1,2]],values)
        @test only(OP.extended_intervals(d;dim=0)).birth==values[1]
        @test only(OP.extended_intervals(d;dim=0)).death==values[2]
    end
    d=OP.extended_persistence([[1,2]],[0,2]);f=d.filtration
    # The original one-parameter grades do not determine an upper star. An
    # explicitly supplied vertex function replaces them on the same cells.
    G=DT.GradedComplex([[91,92],[83]],[sparse(reshape([-1,1],2,1))],[(100,),(200,),(300,)])
    supplied=OP.extended_persistence(G;vertex_values=[0,2],cell_vertices=[[1],[2],[1,2]])
    @test supplied.intervals==d.intervals
    @test OP.provenance(supplied).source_cells==((91,92),(83,))
    @test_throws ArgumentError OP.extended_filtration(G;vertex_values=[0,2],cell_vertices=[[1],[2],[1]])
    @test_throws ArgumentError OP.extended_filtration(G;vertex_values=[0,2],cell_vertices=[[2],[1],[1,2]])
    tree=DT.SimplexTreeMulti([1,2,3,5],[1,2,1,2],[0,0,1],[1,3,4],[1,2,3,4],[(10,),(20,),(30,)])
    @test OP.extended_persistence(tree;vertex_values=[0,2]).intervals==d.intervals
    @test_throws ArgumentError OP.extended_persistence([[1,2]],[0,NaN])
    @test_throws ArgumentError OP.extended_persistence([[1,2]],[false,true])
    @test_throws ArgumentError OP.extended_persistence([[1,2]],[0,2];field=CM.QQField())
    @test_throws ArgumentError OP.extended_persistence([[1,2]],[0,2];order=:other)
    @test_throws ArgumentError OP.extended_persistence([[1,1]],[0,2])
    @test_throws ArgumentError OP.extended_persistence([[1,3]],[0,2])
    @test_throws ArgumentError OP.extended_persistence([[1,2,3]],1:3;max_cells=6)
    @test_throws ArgumentError OP.extended_intervals(d;dim=-1)
    @test_throws ArgumentError OP.extended_intervals(d;dim=true)
    @test_throws ArgumentError OP.extended_intervals(d;dim=0,kind=:finite)
    @test_throws ArgumentError OP.extended_structure_map(d;source=(:ordinary,1),target=(:ordinary,0),dim=0)
    @test_throws ArgumentError OP.extended_structure_map(d;source=(:ordinary,"bad"),target=(:ordinary,0),dim=0)
    @test_throws ArgumentError OP.extended_structure_map(d;source=(:relative,4),target=(:ordinary,5),dim=0)
    @test_throws ArgumentError OP.extended_homology(d;phase=:foo,scale=0)
    pair=OP.extended_pair(f);pair.grades[1][1]=0
    @test OP.check_extended_persistence(f).valid
    corrupted=deepcopy(f);corrupted.pair.grades[1][1]=0
    @test !OP.check_extended_persistence(corrupted).valid
    @test_throws ArgumentError OP.check_extended_persistence(corrupted;throw=true)
    @test OP.describe(OP.ExtendedPersistenceValidationSummary(f)).valid
    @test OP.ExtendedPersistenceValidationSummary(f)==OP.ExtendedPersistenceValidationSummary(f)
    @test occursin("Extended",sprint(show,d))
    @test TamerOp.extended_persistence===OP.extended_persistence
    @test TOA.extended_structure_map===OP.extended_structure_map
    a=OP.extended_homology(f;phase=:ordinary,scale=2)
    b=OP.extended_homology(f;phase=:relative,scale=2)
    c=OP.extended_homology(f;phase=:relative,scale=0)
    @test OP.relative_structure_map(b,c;dim=0)*OP.relative_structure_map(a,b;dim=0)==OP.relative_structure_map(a,c;dim=0)
    @test all(iszero,OP.dimensions(c))
    outputs=fetch.([Threads.@spawn(OP.extended_persistence(f).intervals) for _ in 1:4])
    @test all(==(d.intervals),outputs)
end

@testset "A119 native extended rendering and export" begin
    OP=TamerOp.OrdinaryPersistence;V=TamerOp.Visualization
    d=OP.extended_persistence([[1,2],[2,3],[3,4],[1,4],[2,5]],[0,2,5,3,4];field=CM.F3())
    dir=get(ENV,"TAMEROP_A119_OUTPUT",joinpath(@__DIR__,"browser","artifacts","extended"));mkpath(dir)
    states=Pair{Symbol,V.VisualSessionState}[]
    for package in (:CairoMakie,:WGLMakie)
        if Base.find_package(string(package))===nothing
            @test_skip false
            continue
        end
        @eval import $package
        mk=getfield(@__MODULE__,package);backend=package===:CairoMakie ? :cairomakie : :wglmakie
        for dim in (0,1),kind in (:extended_persistence,:extended_barcode)
            spec=V.visual_spec(d;dim,kind)
            fig=V.render(spec;backend,size=kind===:extended_persistence ? (1500,600) : (1500,500))
            try
                mk.Makie.update_state_before_display!(fig)
                axes=filter(x->x isa mk.Axis,fig.content)
                @test length(axes)==(kind===:extended_persistence ? 3 : 1)
                @test all(a->a.xlabel[]==(kind===:extended_persistence ? "birth value" : "tagged parameter event"),axes)
                @test all(a->all(>(0),mk.Makie.widths(mk.Makie.viewport(a.scene)[])),axes)
                if package===:CairoMakie
                    for extension in ("png","svg","pdf")
                        path=joinpath(dir,"$(kind)-$(dim).$(extension)");mk.save(path,fig)
                        @test filesize(path)>1000
                    end
                    push!(states,Symbol("$(kind)_$(dim)")=>V.visual_state(spec))
                end
            finally
                empty!(fig)
            end
        end
        if package===:CairoMakie
            V.save_visual_session(joinpath(dir,"extended.html"),(;states...);title="Extended persistence",
                caption="Ordinary and relative parameter copies, with exact typed members. Saved views do not run Julia.",size=(1500,600))
            @test filesize(joinpath(dir,"extended.html"))>1000
        end
    end
end

@testset "A119 typed analysis and owned serialization" begin
    OP=TamerOp.OrdinaryPersistence;S=TamerOp.Serialization
    a=OP.extended_persistence([[1,2],[2,3],[1,3]],[0,1,3];field=CM.F3())
    b=OP.extended_persistence([[1,2],[2,3],[1,3]],[0,1,4];field=CM.F3())
    @test S.visual_source_from_json(S.visual_source_to_json(a)).intervals==a.intervals
    for dim in (0,1),kind in (:ordinary,:relative,:extended_positive,:extended_negative,:extended_zero)
        r=OP.extended_analysis(a;dim,kind)
        @test r.intervals==OP.extended_intervals(a;dim,kind)
        @test all(bd->bd[1]<=bd[2],r.value)
        @test r.context.field==CM.F3()
        @test r.context.interval_type===kind
        for operation in (:landscape,:image,:silhouette,:entropy,:summary)
            args=operation in (:landscape,:silhouette) ? (;tgrid=-5:0.1:5) : (;)
            @test OP.extended_analysis(a;dim,kind,operation,args...).context==r.context
        end
    end
    for dim in (0,1)
        @test OP.extended_distance(a,a;dim).value==0
        @test OP.extended_distance(a,b;dim).value==1
        @test OP.extended_distance(a,b;dim,metric=:wasserstein).value==1
    end
    @test_throws ArgumentError OP.extended_analysis(a;dim=0,kind=:all)
    @test_throws ArgumentError OP.extended_analysis(a;dim=0,kind=:ordinary,scale=0)
    @test_throws ArgumentError OP.extended_distance(a,OP.extended_persistence([[1,2]],[0,1]);dim=0)
    mktempdir() do dir
        for d in (a,OP.extended_persistence([[1,2]],[big(2)^80,big(2)^80+1]),
                OP.extended_persistence([[1,2]],[1//3,2//3];order=:superlevel),
                OP.extended_persistence([[1,2]],setprecision(256) do;[BigFloat(1),BigFloat(1)+big"1e-60"];end))
            path=joinpath(dir,"extended.json")
            S.save_extended_persistence_json(path,d)
            restored=S.load_extended_persistence_json(path)
            @test restored.intervals==d.intervals
            @test OP.provenance(restored)==OP.provenance(d)
            @test S.check_extended_persistence_json(path).valid
            @test S.describe(S.extended_persistence_json_summary(path)).kind=="ExtendedPersistenceDiagram"
            @test_throws ArgumentError S.load_persistence_diagram_json(path)
            data=OP.extended_persistence_data(d)
            data.intervals[1][1]=merge(data.intervals[1][1],(;kind=:relative))
            @test_throws ArgumentError OP.extended_persistence_from_data(data)
        end
    end
end

@testset "A119 exact typed visualization specifications" begin
    OP=TamerOp.OrdinaryPersistence;V=TamerOp.Visualization
    for order in (:sublevel,:superlevel),values in ([0,1,5],[2,2,2])
        d=OP.extended_persistence([[1,2],[2,3],[1,3]],values;order)
        for dim in (0,1,3),kind in (:extended_persistence,:extended_barcode)
            spec=V.visual_spec(d;dim,kind)
            @test V.check_visual_spec(spec).valid
            @test spec.metadata.intervals==OP.describe.(OP.extended_intervals(d;dim))
            if kind===:extended_persistence
                for panel in spec.panels
                    points=reduce(vcat,[l.points for l in panel.layers if l isa V.PointLayer];init=NTuple{2,Float64}[])
                    @test sort(points)==sort([(Float64(r.birth),Float64(r.death)) for r in panel.metadata.intervals])
                end
            else
                @test spec.metadata.coordinate_space===:ordinal_two_copy_order
                @test spec.axes.xticks[2]==["$(s.phase===:ordinary ? "O" : "R"):$(s.value)" for s in OP.extended_steps(d)]
            end
        end
        @test_throws ArgumentError V.visual_spec(d;dim=-1)
        @test_throws ArgumentError V.visual_spec(d;dim=0,unknown=true)
    end
    exact=OP.extended_persistence([[1,2]],[big(2)^80,big(2)^80+1])
    @test_throws ArgumentError V.visual_spec(exact)
end
