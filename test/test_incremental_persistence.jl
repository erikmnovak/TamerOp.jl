# Independent dense BigInt arithmetic, fresh reductions, and explicit maps for A121.
module IncrementalPersistenceOracles
using Test, Random, SparseArrays, LinearAlgebra
import TamerOp
const OP=TamerOp.OrdinaryPersistence
const DT=TamerOp.DataTypes
const CM=TamerOp.CoreModules

function complex(cells)
    T=isempty(cells) ? Int : typeof(first(cells).grade)
    nd=maximum((c.dimension for c in cells);init=0)+1
    groups=[sort([c for c in cells if c.dimension==d];by=c->c.id) for d in 0:nd-1]
    Bs=SparseMatrixCSC{Int,Int}[]
    for d in 2:nd
        lookup=Dict(c.id=>j for (j,c) in enumerate(groups[d-1]))
        rows=Int[];cols=Int[];values=Int[]
        for (j,c) in enumerate(groups[d]),e in c.boundary
            push!(rows,lookup[first(e)]);push!(cols,j);push!(values,last(e))
        end
        push!(Bs,sparse(rows,cols,values,length(groups[d-1]),length(groups[d])))
    end
    DT.GradedComplex([Int[c.id for c in group] for group in groups],Bs,
        Tuple{T}[(c.grade,) for group in groups for c in group])
end
function simplex(n;maxdim=n-1,grades=:dimension)
    supports=[findall(i->!iszero(mask & (1<<(i-1))),1:n) for mask in 1:(1<<n)-1 if count_ones(mask)<=maxdim+1]
    sort!(supports;by=x->(length(x),Tuple(x)))
    lookup=Dict(Tuple(v)=>j for (j,v) in enumerate(supports))
    cells=[(;id=j,dimension=length(v)-1,grade=grades===:zero ? 0 : length(v)-1,
        boundary=length(v)==1 ? Pair{Int,Int}[] : [lookup[Tuple(v[setdiff(eachindex(v),[k])])] => (-1)^(k-1) for k in eachindex(v)]) for (j,v) in enumerate(supports)]
    complex(cells),supports
end

function matrix(cells,p)
    lookup=Dict(c.id=>j for (j,c) in enumerate(cells));D=zeros(BigInt,length(cells),length(cells))
    for (j,c) in enumerate(cells),e in c.boundary;D[lookup[first(e)],j]+=last(e);end
    mod.(D,p)
end
function rankmod(A,p)
    B=mod.(map(x->BigInt(x isa Integer ? x : x.val),A),p);r=0
    for j in axes(B,2)
        k=findfirst(i->B[i,j]!=0,r+1:size(B,1));k===nothing && continue
        k+=r;r+=1;B[r,:],B[k,:]=copy(B[k,:]),copy(B[r,:])
        B[r,:]=mod.(B[r,:].*invmod(B[r,j],BigInt(p)),p)
        for i in r+1:size(B,1);B[i,:]=mod.(B[i,:].-B[i,j].*B[r,:],p);end
    end
    r
end
function pairs(cells,p)
    R=matrix(cells,p);pivots=Dict{Int,Int}();positive=Int[]
    for j in axes(R,2)
        while true
            k=findlast(!iszero,R[:,j]);k===nothing && (push!(positive,j);break)
            previous=get(pivots,k,0)
            if previous==0;pivots[k]=j;break;end
            scale=mod(R[k,j]*invmod(R[k,previous],BigInt(p)),p)
            R[:,j]=mod.(R[:,j].-scale.*R[:,previous],p)
        end
    end
    Set((cells[j].dimension,cells[j].id,haskey(pivots,j) ? cells[pivots[j]].id : 0) for j in positive)
end
function equivalent(s;representatives=false)
    cells=OP.persistence_cells(s);p=s.field.p
    @test OP.check_persistence_state(s).valid
    oracle=pairs(cells,p)
    if s.retained
        @test Set((r.dimension,r.birth_cell,r.death_cell) for r in OP.persistence_pairing(s))==oracle
    end
    G=complex(cells);fresh=OP.persistence_diagram(G;field=s.field,order=s.order)
    result=OP.persistence_diagram(s;representatives)
    @test OP.check_persistence_diagram(result).valid
    nd=maximum((c.dimension for c in cells);init=0)
    grades=Dict(c.id=>c.grade for c in cells)
    for d in 0:nd
        @test OP.finite_intervals(result;dim=d)==OP.finite_intervals(fresh;dim=d)
        @test OP.essential_births(result;dim=d)==OP.essential_births(fresh;dim=d)
        @test OP.finite_intervals(result;dim=d)==sort([(grades[b],grades[e]) for (dim,b,e) in oracle
            if dim==d && e!=0 && grades[b]!=grades[e]];rev=s.order===:superlevel)
        @test OP.essential_births(result;dim=d)==sort([grades[b] for (dim,b,e) in oracle
            if dim==d && e==0];rev=s.order===:superlevel)
    end
    if representatives
        D=matrix(cells,p);lookup=Dict(c.id=>j for (j,c) in enumerate(cells))
        vecchain(chain)=begin
            v=zeros(BigInt,length(cells))
            for (id,c) in zip(chain.cell_ids,chain.coefficients);v[lookup[id]]=c;end
            v
        end
        for d in 0:nd,kind in (:finite,:essential)
            bars=kind===:finite ? OP.finite_intervals(result;dim=d) : OP.essential_births(result;dim=d)
            for j in eachindex(bars)
                r=OP.persistence_representative(result;dim=d,kind,index=j)
                @test r.available
                z=vecchain(r.cycle);@test all(iszero,mod.(D*z,p))
                birth=kind===:finite ? bars[j][1] : bars[j]
                active=[k for (k,c) in enumerate(cells) if c.dimension==d+1 && (s.order===:sublevel ? c.grade<=birth : c.grade>=birth)]
                @test rankmod(hcat(D[:,active],z),p)==rankmod(D[:,active],p)+1
                if kind===:finite
                    @test mod.(D*vecchain(r.bounding_chain),p)==z
                end
            end
        end
    end
    result
end
function fingerprint(s)
    (OP.persistence_cells(s),s.retained ? OP.persistence_pairing(s) : nothing,
        [OP.persistence_column(s;cell=c.id,kind=k) for c in OP.persistence_cells(s),k in (s.retained ? (:boundary,:reduced,:basis) : (:boundary,))],
        OP.persistence_state_summary(s))
end
function mapcheck(m)
    p=m.field.p
    @test all(iszero,m.source_boundary*m.source_cycles)
    @test all(iszero,m.target_boundary*m.target_cycles)
    difference=m.chain_map*m.source_cycles-m.target_cycles*m.homology_map
    @test rankmod(hcat(m.target_incoming,difference),p)==rankmod(m.target_incoming,p)
end
end

@testset "A121 incremental growth and representatives" begin
    O=IncrementalPersistenceOracles;OP=O.OP
    for field in (CM.F2(),CM.F3(),CM.Fp(5),CM.Fp(101)),order in (:sublevel,:superlevel)
        G,supports=O.simplex(4);source=O.OP.persistence_cells(OP.persistence_state(G;retain_reduction=true,field))
        emptyG=O.complex(NamedTuple[]);s=OP.persistence_state(emptyG;retain_reduction=true,field,order)
        olddiagram=OP.persistence_diagram(s)
        for c in source
            grade=order===:sublevel ? c.grade : -c.grade
            u=OP.update_persistence!(s;append=[(;dimension=c.dimension,grade,boundary=c.boundary)],witnesses=true)
            @test u.added==[c.id]
            @test OP.check_persistence_update(u).valid
            @test OP.persistence_update_summary(u).full_recomputations==0
            O.equivalent(s;representatives=true)
            O.mapcheck(OP.persistence_update_map(u;dim=c.dimension,scale=order===:sublevel ? 10 : -10))
        end
        @test isempty(OP.essential_births(olddiagram;dim=0))
        @test OP.essential_births(OP.persistence_diagram(s);dim=0)==[0]
        @test OP.persistence_state_summary(s).history_length==0
    end
end

@testset "A121 adjacent vineyard matrix repair independent oracle" begin
    O=IncrementalPersistenceOracles;OP=O.OP;cases=Set{Tuple{Bool,Bool}}()
    for field in (CM.F2(),CM.F3(),CM.Fp(5),CM.Fp(101))
        G,_=O.simplex(5;maxdim=3,grades=:zero);s=OP.persistence_state(G;retain_reduction=true,field);rng=MersenneTwister(121)
        for _ in 1:160
            cells=OP.persistence_cells(s)
            admissible=[j for j in 1:length(cells)-1 if all(e->first(e)!=cells[j].id,cells[j+1].boundary)]
            j=rand(rng,admissible);a,b=cells[j].id,cells[j+1].id
            push!(cases,(isempty(OP.persistence_column(s;cell=a)),isempty(OP.persistence_column(s;cell=b))))
            u=OP.transpose_persistence!(s;position=j)
            @test OP.check_persistence_update(u).valid
            @test length(u.events)==1
            @test u.events[1].cells==(a,b)
            @test u.events[1].tie===:explicit_tie
            O.equivalent(s)
        end
    end
    @test cases==Set([(true,true),(false,false),(true,false),(false,true)])
end

@testset "A121 exact grade homotopies and tied crossings" begin
    O=IncrementalPersistenceOracles;OP=O.OP
    edge=O.complex([(id=1,dimension=0,grade=0,boundary=Pair{Int,Int}[]),
        (id=2,dimension=0,grade=1,boundary=Pair{Int,Int}[]),
        (id=3,dimension=1,grade=2,boundary=[1=>-1,2=>1])])
    for field in (CM.F2(),CM.F3(),CM.Fp(101))
        s=OP.persistence_state(edge;retain_reduction=true,field)
        u=OP.update_persistence!(s;continuation=true,grades=[1=>1,2=>0],witnesses=true)
        @test length(u.events)==1
        @test only(u.events).time==1//2
        @test Set(only(u.events).continuation)==Set([(1,2),(2,1)])
        @test OP.check_persistence_update(u).valid
        O.equivalent(s;representatives=true)
        @test_throws ArgumentError OP.persistence_update_map(u;dim=0,scale=1//2)
        m=OP.persistence_update_map(u;dim=0,scale=2);O.mapcheck(m)
        @test size(m.homology_map)==(1,1)
        @test O.rankmod(m.homology_map,field.p)==1
        for order in (:sublevel,:superlevel)
            G,supports=O.simplex(5;maxdim=2)
            order===:superlevel && (G=DT.GradedComplex(G.cells_by_dim,G.boundaries,[(-g[1],) for g in G.grades]))
            s=OP.persistence_state(G;retain_reduction=true,field,order);rng=MersenneTwister(54)
            for _ in 1:16
                heights=randperm(rng,5)
                grades=[j=>(order===:sublevel ? 1 : -1)*(10maximum(heights[v])+length(v)) for (j,v) in enumerate(supports)]
                u=OP.update_persistence!(s;continuation=true,grades)
                @test OP.check_persistence_update(u).valid
                @test issorted([e.time for e in u.events])
                O.equivalent(s;representatives=true)
            end
        end
    end
    # Exact Float64 crossing times, not epsilon grouping or rounded events.
    G=DT.GradedComplex([[1,2,3]],SparseMatrixCSC{Int,Int}[],[(0.0,),(0.5,),(1.0,)])
    s=OP.persistence_state(G;retain_reduction=true)
    u=OP.update_persistence!(s;continuation=true,grades=[1=>1.0,2=>0.5,3=>0.0])
    @test length(u.events)==3
    @test all(e->e.time==1//2 && e.tie===:simultaneous,u.events)
    @test OP.check_persistence_update(u).valid
    O.equivalent(s)
end

@testset "A121 extreme integer crossing times" begin
    O=IncrementalPersistenceOracles;OP=O.OP
    # Three distinct crossings which all round to 0.5 in Float64. Neither
    # machine-integer subtraction nor floating event sorting is safe here.
    lo,hi=typemin(Int),typemax(Int)
    G=DT.GradedComplex([[1,2,3]],SparseMatrixCSC{Int,Int}[],[(lo,),(0,),(hi,)])
    s=OP.persistence_state(G;retain_reduction=true)
    u=OP.update_persistence!(s;continuation=true,grades=[1=>hi,3=>lo])
    @test length(unique(e.time for e in u.events))==3
    @test all(e->e.tie===:isolated,u.events)
    @test all(e->Float64(e.time)==0.5,u.events)
    @test OP.check_persistence_update(u).valid
    O.equivalent(s)
end

@testset "A121 insertion deletion maps and bounded retention" begin
    O=IncrementalPersistenceOracles;OP=O.OP
    for field in (CM.F2(),CM.F3(),CM.Fp(5))
        G,_=O.simplex(3;grades=:dimension);s=OP.persistence_state(G;retain_reduction=true,field)
        original=O.equivalent(s;representatives=true)
        u=OP.update_persistence!(s;delete=[7],witnesses=true)
        @test OP.check_persistence_update(u).valid
        d=O.equivalent(s;representatives=true)
        @test OP.essential_births(d;dim=1)==[1]
        m=OP.persistence_update_map(u;dim=1,scale=3);O.mapcheck(m)
        @test m.direction===:after_to_before
        @test size(m.homology_map)==(0,1)
        @test isempty(OP.essential_births(original;dim=1))
        u=OP.update_persistence!(s;insert=[(dimension=2,grade=1,boundary=[4=>1,5=>-1,6=>1])],witnesses=true)
        @test u.added==[8]
        O.equivalent(s;representatives=true)
        @test isempty(OP.finite_intervals(OP.persistence_diagram(s);dim=1))
        @test_throws ArgumentError OP.update_persistence!(s;delete=[1])
        @test_throws ArgumentError OP.update_persistence!(s;insert=[(id=7,dimension=0,grade=0,boundary=[])])
        # Remove a closed star while retaining nonterminal cells; no tombstones.
        u=OP.update_persistence!(s;delete=[1,4,5,8],witnesses=true)
        @test sort(u.removed)==[1,4,5,8]
        @test all(e->e.tie===:algebraic_edit && isempty(e.continuation),u.events)
        @test OP.check_persistence_update(u).valid
        O.equivalent(s;representatives=true)
        O.mapcheck(OP.persistence_update_map(u;dim=0,scale=3))
        for _ in 1:100
            u=OP.update_persistence!(s;insert=[(dimension=0,grade=-1,boundary=[])])
            OP.update_persistence!(s;delete=u.added)
            @test length(s.ids)==3 && length(s.rank)==3 && length(s.slots)==3
        end
        O.equivalent(s)
        @test s.next_id==109
        kept=OP.persistence_diagram(s;representatives=true)
        OP.close_persistence!(s);OP.close_persistence!(s)
        @test OP.check_persistence_state(s).valid
        @test OP.persistence_state_summary(s).stored_entries==0
        @test_throws ArgumentError OP.persistence_diagram(s)
        @test OP.check_persistence_diagram(kept).valid
    end
end

@testset "A121 atomic rejection exact fields and validation" begin
    O=IncrementalPersistenceOracles;OP=O.OP;G,_=O.simplex(3)
    s=OP.persistence_state(G;retain_reduction=true)
    for bad in (()->OP.update_persistence!(s),
        ()->OP.update_persistence!(s;continuation=true,grades=[true=>2]),
        ()->OP.update_persistence!(s;continuation=true,grades=[1=>99]),
        ()->OP.update_persistence!(s;continuation=true,grades=[1=>0,1=>0]),
        ()->OP.update_persistence!(s;append=[(dimension=0,grade=-1,boundary=[])]),
        ()->OP.update_persistence!(s;append=[(dimension=3,grade=3,boundary=[7=>1])]),
        ()->OP.update_persistence!(s;append=[(dimension=0,grade=4,boundary=[]),(dimension=1,grade=5,boundary=[999=>1])]),
        ()->OP.update_persistence!(s;delete=[1,1]),
        ()->OP.transpose_persistence!(s;position=3),
        ()->OP.transpose_persistence!(s;position=true))
        before=O.fingerprint(s);@test_throws ArgumentError bad();@test O.fingerprint(s)==before
        @test OP.check_persistence_state(s).valid
    end
    @test_throws ArgumentError OP.persistence_state(G;retain_reduction=true,field=CM.QQField())
    @test_throws ArgumentError OP.PersistenceUpdateBudget(max_cells=true)
    @test_throws ArgumentError OP.PersistenceUpdateBudget(max_swaps=0)
    isolated=DT.GradedComplex([[1,2,3]],SparseMatrixCSC{Int,Int}[],[(0,),(1,),(2,)])
    for budget in (OP.PersistenceUpdateBudget(max_swaps=1),)
        limited=OP.persistence_state(isolated;retain_reduction=true,budget);before=O.fingerprint(limited)
        @test_throws ArgumentError OP.update_persistence!(limited;continuation=true,grades=[1=>2,2=>1,3=>0])
        @test O.fingerprint(limited)==before
    end
    limited=OP.persistence_state(isolated;retain_reduction=true,budget=OP.PersistenceUpdateBudget(max_entries=4))
    before=O.fingerprint(limited)
    @test_throws ArgumentError OP.update_persistence!(limited;append=[(dimension=1,grade=3,boundary=[1=>-1,2=>1])])
    @test O.fingerprint(limited)==before
    @test_throws ArgumentError OP.update_persistence!(limited;continuation=true,grades=[1=>0],witnesses=true)
    # Nonunit boundaries and characteristic dependence, independent of geometry.
    torsion=DT.GradedComplex([[1],[1]],[sparse(reshape([2],1,1))],[(0,),(1,)])
    for p in (2,3,5)
        s=OP.persistence_state(torsion;retain_reduction=true,field=CM.Fp(p));O.equivalent(s;representatives=true)
        @test length(OP.essential_births(OP.persistence_diagram(s);dim=0))==(p==2 ? 1 : 0)
        @test [c.label for c in OP.persistence_cells(s)]==[1,1]
    end
    corrupt=deepcopy(s);empty!(corrupt.change[1])
    @test !OP.check_persistence_state(corrupt).valid
    @test_throws ArgumentError OP.check_persistence_state(corrupt;throw=true)
end

@testset "A121 concurrent independent queries" begin
    O=IncrementalPersistenceOracles;OP=O.OP;G,_=O.simplex(4);s=OP.persistence_state(G;retain_reduction=true)
    @sync begin
        Threads.@spawn for i in 1:15
            u=OP.update_persistence!(s;append=[(dimension=0,grade=10+i,boundary=[])])
            OP.update_persistence!(s;delete=u.added)
        end
        for _ in 1:3
            Threads.@spawn for _ in 1:15
                @test OP.check_persistence_state(s).valid
                @test OP.check_persistence_diagram(OP.persistence_diagram(s;representatives=true)).valid
            end
        end
    end
    O.equivalent(s)
end

@testset "A121 based-chain edits nonunit coefficients and compacted IDs" begin
    O=IncrementalPersistenceOracles;OP=O.OP
    for p in (3,5,101),order in (:sublevel,:superlevel)
        G,_=O.simplex(5;maxdim=3,grades=:zero);rng=MersenneTwister(9121)
        cells=OP.persistence_cells(OP.persistence_state(G;retain_reduction=true,field=CM.Fp(p)))
        weights=rand(rng,1:p-1,length(cells))
        cells=[merge(c,(boundary=[first(e)=>mod(last(e)*weights[c.id]*invmod(weights[first(e)],p),p) for e in c.boundary],)) for c in cells]
        s=OP.persistence_state(O.complex(cells);retain_reduction=true,field=CM.Fp(p),order)
        for step in 1:35
            # These changes of chain basis are not just simplicial signs.
            current=OP.persistence_cells(s)
            admissible=[i for i in 1:length(current)-1 if all(e->first(e)!=current[i].id,current[i+1].boundary)]
            OP.transpose_persistence!(s;position=rand(rng,admissible))
            O.equivalent(s;representatives=true)
            used=Set(first(e) for c in current for e in c.boundary)
            c=rand(rng,[c for c in current if !(c.id in used)])
            u=OP.update_persistence!(s;delete=[c.id],witnesses=true)
            @test isempty(u.events) # Structural edits do not invent a vine path.
            O.equivalent(s;representatives=true)
            O.mapcheck(OP.persistence_update_map(u;dim=max(c.dimension-1,0),scale=0))
            u=OP.update_persistence!(s;insert=[(;c.dimension,c.grade,c.boundary)],witnesses=true)
            @test only(u.added)>c.id
            @test isempty(u.events)
            O.equivalent(s;representatives=true)
            O.mapcheck(OP.persistence_update_map(u;dim=c.dimension,scale=0))
        end
        # Deleting everything, including nonterminal columns, releases all slots.
        OP.update_persistence!(s;delete=[c.id for c in OP.persistence_cells(s)])
        @test isempty(s.ids) && s.entries==0
        O.equivalent(s)
        u=OP.update_persistence!(s;append=[(dimension=0,grade=0,boundary=[])])
        @test only(u.added)>length(cells)
        O.equivalent(s;representatives=true)
    end
end

@testset "A121 public surface exact rational paths and bounded repair" begin
    O=IncrementalPersistenceOracles;OP=O.OP
    for name in (:persistence_state,:update_persistence!,:persistence_state_summary)
        @test getfield(TamerOp,name)===getfield(OP,name)
    end
    for name in (:PersistenceState,:PersistenceUpdate,:PersistenceUpdateBudget,:VineyardEvent,
            :persistence_pairing,:persistence_cells,:persistence_column,:persistence_update_map,
            :check_persistence_state,:check_persistence_update,:transpose_persistence!,:close_persistence!)
        @test getfield(TamerOp.Advanced,name)===getfield(OP,name)
    end
    G=DT.GradedComplex([[1,2,3]],SparseMatrixCSC{Int,Int}[],[(0//1,),(1//3,),(1//1,)])
    s=OP.persistence_state(G;retain_reduction=true)
    u=OP.update_persistence!(s;continuation=true,grades=[1=>2//3,2=>0//1])
    @test only(u.events).time==1//3
    @test OP.check_persistence_update(u).valid
    malformed=deepcopy(u);e=only(malformed.events)
    malformed.events[1]=OP.VineyardEvent(e.cells,e.time,e.before,e.after,e.continuation,:simultaneous)
    @test !OP.check_persistence_update(malformed).valid
    @test_throws ArgumentError OP.check_persistence_update(malformed;throw=true)
    O.equivalent(s;representatives=true)
    @test_throws ArgumentError OP.persistence_update_map(u;dim=0,scale=0)
    before=O.fingerprint(s)
    @test_throws ArgumentError OP.update_persistence!(s;continuation=true,grades=[1=>Inf])
    @test O.fingerprint(s)==before
    @test isempty(OP.update_persistence!(s;continuation=true,grades=[]).events)
    # A new earliest isolated cell leaves all existing matrix columns untouched.
    oldR=copy(s.reduced);oldV=copy(s.change)
    u=OP.update_persistence!(s;insert=[(dimension=0,grade=-1//1,boundary=[])])
    @test all(s.reduced[j]===oldR[j] && s.change[j]===oldV[j] for j in eachindex(oldR))
    @test u.column_additions==0 && u.reduced_columns==1
    G,_=O.simplex(4;maxdim=2)
    limited=OP.persistence_state(G;retain_reduction=true,field=CM.F3())
    limited.budget=OP.PersistenceUpdateBudget(max_column_additions=1)
    before=O.fingerprint(limited)
    # Removing this early edge and its cofaces requires more than one repair.
    cells=OP.persistence_cells(limited);removed=Set([5])
    for c in cells
        any(e->first(e) in removed,c.boundary) && push!(removed,c.id)
    end
    @test_throws ArgumentError OP.update_persistence!(limited;delete=collect(removed))
    @test O.fingerprint(limited)==before
    @test OP.check_persistence_state(limited).valid
end

@testset "A121 inclusion witnesses compose with A120 zigzags" begin
    O=IncrementalPersistenceOracles;OP=O.OP;ZZ=TamerOp.ZigzagPersistence
    for field in (CM.F2(),CM.F3(),CM.Fp(101))
        G=DT.GradedComplex([[1,2,3]],SparseMatrixCSC{Int,Int}[],[(0,),(0,),(0,)])
        s=OP.persistence_state(G;retain_reduction=true,field)
        first=OP.update_persistence!(s;append=[(dimension=1,grade=1,boundary=[1=>-1,2=>1])],witnesses=true)
        second=OP.update_persistence!(s;delete=first.added,witnesses=true)
        a=OP.persistence_update_map(first;dim=0,scale=2)
        b=OP.persistence_update_map(second;dim=0,scale=2)
        @test a.direction===:before_to_after && b.direction===:after_to_before
        @test a.target_cells==b.target_cells && a.target_cycles==b.target_cycles
        z=ZZ.ZigzagModule([3,2,3],[a.homology_map,b.homology_map];directions=[:forward,:backward],field)
        r=ZZ.zigzag_persistence(z;representatives=true)
        @test sort([(i.first,i.last) for i in ZZ.zigzag_intervals(r)])==[(1,1),(1,3),(1,3),(3,3)]
        @test ZZ.check_zigzag(r).valid
    end
end

@testset "A121 endpoint routing and explicit continuation" begin
    O=IncrementalPersistenceOracles;OP=O.OP
    for field in (CM.F2(),CM.F3(),CM.Fp(5),CM.Fp(101)),order in (:sublevel,:superlevel)
        G,supports=O.simplex(4;maxdim=2)
        cells=OP.persistence_cells(OP.persistence_state(G;field))
        sign=order===:sublevel ? 1 : -1
        G=O.complex([merge(c,(grade=sign*(10maximum(supports[c.id])+c.dimension),)) for c in cells])
        s=OP.persistence_state(G;field,order)
        v=OP.persistence_state(G;field,order,retain_reduction=true)
        @test OP.persistence_state_summary(s).reduction===:none
        @test isempty(s.reduced) && isempty(s.change) && isempty(s.pivot)
        @test_throws ArgumentError OP.persistence_pairing(s)
        @test_throws ArgumentError OP.persistence_column(s;cell=1)
        @test isempty(OP.persistence_column(s;cell=1,kind=:boundary))
        O.equivalent(s;representatives=true)
        rng=MersenneTwister(121013)
        for step in 1:8
            heights=randperm(rng,4)
            grades=[c.id=>sign*(10maximum(heights[supports[c.id]])+c.dimension) for c in cells]
            u=OP.update_persistence!(s;grades,witnesses=true)
            path=OP.update_persistence!(v;grades,continuation=true)
            @test OP.check_persistence_update(u).valid && OP.check_persistence_update(path).valid
            @test u.method===:endpoint_regrade && isempty(u.events) && isempty(u.start_grades)
            @test OP.persistence_update_summary(u).correspondence===:none
            @test OP.persistence_update_summary(path).correspondence===:vineyard_pairing
            @test u.column_additions==u.reduced_columns==u.reduction_builds==0
            a=O.equivalent(s;representatives=true);b=O.equivalent(v)
            @test a.finite_by_dim==b.finite_by_dim && a.essential_by_dim==b.essential_by_dim
            @test a.meta.update_query===:fresh_reduction && b.meta.update_query===:retained_reduction
            @test isempty(s.reduced) && isempty(s.change) && isempty(s.reduced_rows)
            @test !s.retained
            O.mapcheck(OP.persistence_update_map(u;dim=1,scale=sign*100))
        end
        # A request for continuation prepares only the current starting endpoint.
        before=OP.persistence_cells(s)
        u=OP.update_persistence!(s;grades=[c.id=>c.grade+sign for c in before],continuation=true)
        @test u.reduction_builds==1 && u.method===:vineyard && isempty(u.events)
        @test Dict(u.start_grades)==Dict(c.id=>c.grade for c in before)
        @test s.retained && OP.check_persistence_update(u).valid
        O.equivalent(s;representatives=true)
        # Changes preserving total order can reuse columns without a path claim.
        columns=copy(s.reduced)
        u=OP.update_persistence!(s;grades=[c.id=>c.grade+2sign for c in OP.persistence_cells(s)])
        @test u.method===:reuse_order && u.retained && u.reduction_builds==0
        @test OP.persistence_update_summary(u).correspondence===:none
        @test all(s.reduced[j]===columns[j] for j in eachindex(columns))
        O.equivalent(s;representatives=true)
    end
end

@testset "A121 endpoint structural edits and stable identities" begin
    O=IncrementalPersistenceOracles;OP=O.OP
    for field in (CM.F2(),CM.F3(),CM.Fp(101)),retained in (false,true)
        G,_=O.simplex(4;maxdim=2)
        s=OP.persistence_state(G;field,retain_reduction=retained)
        # Broad endpoint changes discard even an explicitly prepared reduction.
        # Reverse vertex order; positive-dimensional cells remain later.
        grades=[c.id=>(c.dimension==0 ? 5-c.id : 100c.dimension+c.id) for c in OP.persistence_cells(s)]
        u=OP.update_persistence!(s;grades)
        @test !s.retained && u.method===:endpoint_regrade && OP.check_persistence_update(u).valid
        kept=O.equivalent(s;representatives=true)
        removed=Set([1])
        for c in OP.persistence_cells(s)
            any(e->first(e) in removed,c.boundary) && push!(removed,c.id)
        end
        u=OP.update_persistence!(s;delete=collect(removed),witnesses=true)
        @test u.method===:source_edit && u.column_additions==u.reduced_columns==0
        @test OP.check_persistence_update(u).valid
        O.equivalent(s;representatives=true)
        O.mapcheck(OP.persistence_update_map(u;dim=0,scale=1000))
        next=s.next_id
        for j in 1:8
            u=OP.update_persistence!(s;insert=[(dimension=0,grade=-j,boundary=[],label=-10)])
            @test only(u.added)==next+j-1 && u.method===:source_edit
            O.equivalent(s;representatives=true)
            OP.update_persistence!(s;delete=u.added)
            @test OP.persistence_state_summary(s).history_length==0
        end
        u=OP.update_persistence!(s;append=[(dimension=0,grade=1000,boundary=[]),(dimension=1,grade=1001,boundary=[2=>-1,(next+8)=>1])])
        @test OP.check_persistence_update(u).valid
        O.equivalent(s;representatives=true)
        OP.update_persistence!(s;delete=[c.id for c in OP.persistence_cells(s)])
        O.equivalent(s;representatives=true)
        @test isempty(s.boundary) && isempty(s.cofaces) && s.entries==0
        OP.close_persistence!(s)
        @test OP.check_persistence_state(s).valid && OP.check_persistence_diagram(kept).valid
    end
end

@testset "A121 routing budgets rollback and invalid contracts" begin
    O=IncrementalPersistenceOracles;OP=O.OP
    G=DT.GradedComplex([[9,9,9]],SparseMatrixCSC{Int,Int}[],[(0,),(1,),(2,)])
    for retained in (false,true)
        s=OP.persistence_state(G;retain_reduction=retained,budget=OP.PersistenceUpdateBudget(max_swaps=1))
        before=O.fingerprint(s)
        @test_throws ArgumentError OP.update_persistence!(s;grades=[1=>2,3=>0],continuation=true)
        @test O.fingerprint(s)==before && OP.check_persistence_state(s).valid
        # Endpoint-only work is independent of the crossing budget.
        u=OP.update_persistence!(s;grades=[1=>2,3=>0])
        @test u.method===:endpoint_regrade && isempty(u.events) && !s.retained
        @test OP.check_persistence_update(u).valid
        @test [c.label for c in OP.persistence_cells(s)]==[9,9,9]
        O.equivalent(s;representatives=true)
        for bad in (()->OP.update_persistence!(s;grades=[],continuation=:auto),
            ()->OP.update_persistence!(s;delete=[],continuation=true),
            ()->OP.update_persistence!(s;grades=[999=>0]),
            ()->OP.update_persistence!(s;grades=[1=>0,1=>1]),
            ()->OP.update_persistence!(s;grades=[1=>0],witnesses=1),
            ()->OP.update_persistence!(s;insert=[(dimension=0,grade=-1,boundary=[]),(dimension=1,grade=3,boundary=[999=>1])]))
            before=O.fingerprint(s);@test_throws ArgumentError bad();@test O.fingerprint(s)==before
        end
        # Merely changing a result's route label cannot certify continuation.
        malformed=typeof(u)((name===:method ? :vineyard : getfield(u,name) for name in fieldnames(typeof(u)))...)
        @test !OP.check_persistence_update(malformed).valid
        @test_throws ArgumentError OP.check_persistence_update(malformed;throw=true)
    end
    @test_throws ArgumentError OP.persistence_state(G;retain_reduction=1)
    # Preparation and repair share the same per-request budget; no partial cache remains.
    G,_=O.simplex(4;maxdim=2)
    s=OP.persistence_state(G;field=CM.F3(),budget=OP.PersistenceUpdateBudget(max_column_additions=1))
    before=O.fingerprint(s)
    @test_throws ArgumentError OP.update_persistence!(s;grades=[],continuation=true)
    @test O.fingerprint(s)==before && !s.retained
    @test_throws ArgumentError OP.transpose_persistence!(s;position=1)
    @test O.fingerprint(s)==before
    # Explicit transposition also prepares a cold state and reports that cost.
    tied=DT.GradedComplex([[1,2]],SparseMatrixCSC{Int,Int}[],[(0.0,),(0.0,)])
    s=OP.persistence_state(tied);u=OP.transpose_persistence!(s;position=1)
    @test u.reduction_builds==1 && u.method===:vineyard && OP.check_persistence_update(u).valid
    O.equivalent(s;representatives=true)
    OP.update_persistence!(s;delete=[1,2]);O.equivalent(s;representatives=true)
    s=OP.persistence_state(tied)
    OP.update_persistence!(s;delete=[1,2])
    @test OP.persistence_diagram(s).finite_by_dim==[Tuple{Float64,Float64}[]]
    @test !s.retained
    OP.update_persistence!(s;append=[(dimension=0,grade=1.0,boundary=[])])
    @test OP.essential_births(OP.persistence_diagram(s);dim=0)==[1.0]
end
