# Independent H0 oracle: connected components and their inclusion maps on every
# finite label. This deliberately uses no graph-presentation helper.
function _a48_module(P,vg,edges,eg,field)
    K=CM.coeff_type(field); n=FF.nvertices(P)
    component=Vector{Dict{Int,Int}}(undef,n); representatives=Vector{Vector{Int}}(undef,n)
    for v in 1:n
        active=findall(g->FF.leq(P,g,v),vg)
        adjacency=Dict(i=>Int[] for i in active)
        for (i,(a,b)) in enumerate(edges)
            FF.leq(P,eg[i],v) || continue
            push!(adjacency[a],b);push!(adjacency[b],a)
        end
        c=Dict{Int,Int}(); reps=Int[]
        for a in active
            haskey(c,a) && continue
            push!(reps,a); todo=[a];c[a]=length(reps)
            while !isempty(todo)
                x=pop!(todo)
                for b in adjacency[x]
                    haskey(c,b) && continue
                    c[b]=length(reps);push!(todo,b)
                end
            end
        end
        component[v]=c;representatives[v]=reps
    end
    maps=Dict{Tuple{Int,Int},Matrix{K}}()
    for (u,v) in FF.cover_edges(P)
        A=zeros(K,length(representatives[v]),length(representatives[u]))
        for (j,r) in enumerate(representatives[u]);A[component[v][r],j]=one(K);end
        maps[u,v]=A
    end
    return MD.PModule{K}(P,length.(representatives),maps;field),component
end
function _a48_check(P,vg,edges,eg,field;syzygies=false)
    HP=TamerOp.H0Presentations
    p=HP.h0_presentation(P;vertex_grades=vg,edges,edge_grades=eg,field,syzygies)
    M,components=_a48_module(P,vg,edges,eg,field)
    @test HP.check_h0_presentation(p).valid
    # Keep the independent characteristic-zero oracle as well as the checked
    # numerical resolution, including higher differentials on the 3D fixture.
    real=field isa CM.RealField
    oracle=real ? first(_a48_module(P,vg,edges,eg,CM.QQField())) : M
    opts=OPT.ResolutionOptions(maxlen=3,minimal=true)
    reference=DF.projective_resolution(oracle,OPT.ResolutionOptions(maxlen=3,minimal=true);threads=false)
    if real
        numerical=DF.projective_resolution(M,opts;threads=false)
        @test DF.check_projective_resolution(numerical).valid
        @test DF.betti_table(numerical)==DF.betti_table(reference)
        maps=vcat([numerical.aug],numerical.d_mor)
        for v in 1:FF.nvertices(P), i in 1:length(maps)-1
            A,B=maps[i].comps[v],maps[i+1].comps[v]
            @test norm(A*B)<=1e-12*norm(A)*norm(B)
            @test FL.rank(field,A)+FL.rank(field,B)==size(A,2)
        end
    end
    @test HP.betti_table(p;opts)==DF.betti_table(reference)
    for len in (0,1,2), pad in (nothing,0,1,4)
        ropts=OPT.ResolutionOptions(maxlen=len)
        expected=DF.betti_table(DF.projective_resolution(oracle,ropts;threads=false);pad_to=pad)
        @test HP.betti_table(p;opts=ropts,pad_to=pad)==expected
    end
    f=HP.presentation_map(p); K=CM.coeff_type(field)
    comps=Matrix{K}[]
    for v in 1:FF.nvertices(P)
        ids=findall(g->FF.leq(P,g,v),p.generators)
        A=zeros(K,M.dims[v],length(ids))
        for (j,i) in enumerate(ids); A[components[v][p.original_vertices[i]],j]=one(K); end
        push!(comps,A)
        @test iszero(A*f.comps[v])
        @test FL.rank(field,A)==M.dims[v]
        @test FL.rank(field,f.comps[v])+M.dims[v]==length(ids)
    end
    @test MD.check_morphism(MD.PMorphism(f.cod,M,comps)).valid
    @test HP.presentation_module(p).dims==M.dims
    if syzygies
        res=HP.projective_resolution(p,OPT.ResolutionOptions(maxlen=3,minimal=true))
        @test DF.check_projective_resolution(res).valid
        @test MD.check_morphism(res.aug).valid
        @test DF.betti_table(res)==DF.betti_table(reference)
    end
    return p
end

@testset "A48 exact graph presentations and independent Betti oracles" begin
    HP=TamerOp.H0Presentations
    for field in (CM.QQField(),CM.F2(),CM.F3(),CM.Fp(101),CM.RealField(Float64))
        P=FF.ProductOfChainsPoset((2,2))
        p=_a48_check(P,[1,1],[(1,2),(1,2)],[2,3],field;syzygies=true)
        @test HP.betti_table(p)==[2 0 0 0;0 1 1 0;0 0 0 1]
        @test HP.presentation_matrix(p)==p.matrix
        @test HP.generator_labels(p)==[1,1]
        @test sort(HP.relation_labels(p))==[2,3]
        _a48_check(P,[1,2,4,1],[(1,2),(2,3),(1,3),(4,4)],[2,4,4,1],field;syzygies=true)
        _a48_check(P,Int[],Tuple{Int,Int}[],Int[],field;syzygies=true)
        _a48_check(P,[1,2,3,4],Tuple{Int,Int}[],Int[],field;syzygies=true)
        Q=FF.ProductOfChainsPoset((4,))
        _a48_check(Q,[1,2,1,3],[(1,2),(2,3),(3,4),(1,4)],[2,3,3,4],field;syzygies=true)
        rng=MersenneTwister(48001)
        for trial in 1:8
            P=FF.ProductOfChainsPoset((3,3))
            vg=rand(rng,1:9,5);edges=[(rand(rng,1:5),rand(rng,1:5)) for _ in 1:10]
            eg=Int[]
            for (a,b) in edges
                candidates=[q for q in 1:9 if FF.leq(P,vg[a],q)&&FF.leq(P,vg[b],q)]
                push!(eg,rand(rng,candidates))
            end
            _a48_check(P,vg,edges,eg,field;syzygies=true)
        end
        # Relabeled nonlattice indexing poset, and a genuine 3D grid: no
        # bigraded claim is substituted for higher projective Betti numbers.
        order=Bool[1 0 0 1;0 1 0 1;0 0 1 1;0 0 0 1]
        perm=[4,2,1,3];Q=FF.FinitePoset(order[perm,perm])
        p=_a48_check(Q,[2,3,4],[(1,2),(2,3),(1,3)],[1,1,1],field)
        @test p.algorithm==:poset_graph && p.second===nothing
        Q=FF.ProductOfChainsPoset((2,2,2))
        _a48_check(Q,[1,1],[(1,2),(1,2),(1,2)],[2,3,5],field)
        # Signed product order: same graph after reversing both coordinate axes.
        Q=FF.ProductOfChainsPoset((2,2);orientation=(-1,-1))
        _a48_check(Q,[4,4],[(1,2),(1,2)],[2,3],field;syzygies=true)
    end
end

@testset "A48 contracts witnesses and graph invariance" begin
    HP=TamerOp.H0Presentations;P=FF.ProductOfChainsPoset((3,3))
    p=HP.h0_presentation(P;vertex_grades=[1,1],edges=[(1,2),(1,2)],edge_grades=[3,7],syzygies=true)
    @test HP.h0_presentation_summary(p).syzygies_retained
    @test occursin("2 generators",sprint(show,p))
    @test HP.describe(p).algorithm==:bigraded_forest
    @test HP.provenance(p).base_poset===P
    @test TO.h0_presentation===HP.h0_presentation
    @test all(s->s in names(TamerOp.Advanced),(:H0Presentation,:H0PresentationValidationSummary,:check_h0_presentation))
    @test TamerOp.betti_table(p)==HP.betti_table(p)
    r=TamerOp.resolve(p;minimality=true)
    @test TamerOp.betti_table(r)==HP.betti_table(p)
    @test HP.betti_table(HP.h0_presentation(P;vertex_grades=[1,1],edges=[(2,1),(1,2)],edge_grades=[7,3]))==HP.betti_table(p)
    for kwargs in ((vertex_grades=[0],edges=[],edge_grades=[]),
                   (vertex_grades=[1],edges=[(1,2)],edge_grades=[1]),
                   (vertex_grades=[9,1],edges=[(1,2)],edge_grades=[1]))
        @test_throws ArgumentError HP.h0_presentation(P;kwargs...)
    end
    @test_throws DimensionMismatch HP.h0_presentation(P;vertex_grades=[1],edges=[(1,1)],edge_grades=[])
    @test_throws ArgumentError HP.h0_presentation(P;vertex_grades=[1,1],edges=[(1,2),(1,2)],edge_grades=[3,7],syzygies=true,max_witness_entries=1)
    @test_throws ArgumentError HP.betti_table(p;pad_to=-1)
    bad=deepcopy(p);bad.matrix[1,1]=0
    @test !HP.check_h0_presentation(bad).valid
    @test_throws ArgumentError HP.check_h0_presentation(bad;throw=true)
    bad=deepcopy(p);bad.syzygies[1,1]=0
    @test !HP.check_h0_presentation(bad).valid
    @test all(t->fetch(t)==HP.betti_table(p),[Threads.@spawn HP.betti_table(p) for _ in 1:8])
end

@testset "A48 encoding Betti queries preserve finite-grid semantics" begin
    HP=TamerOp.H0Presentations
    B=sparse([1,2,1,2],[1,1,2,2],[-1,1,-1,1],2,2)
    G=DT.GradedComplex([[1,2],[3,4]],[B],[(0,0),(0,0),(1,0),(0,1)])
    for field in (CM.QQField(),CM.F2(),CM.F3(),CM.Fp(101),CM.RealField(Float64))
        enc=TamerOp.encode(G,DI.GradedFiltration();field)
        @test enc.M.cached_module===nothing
        p=TamerOp.h0_presentation(enc;syzygies=true)
        @test HP.check_h0_presentation(p).valid
        @test TamerOp.betti_table(enc)==HP.betti_table(p)
        @test enc.M.cached_module===nothing
        @test TamerOp.betti_table(enc;cache=TamerOp.SessionCache())==HP.betti_table(p)
        @test_throws ArgumentError TamerOp.betti_table(enc;cache=:invalid)
        @test_throws ArgumentError TamerOp.h0_presentation(enc;max_expanded_cells=2)
    end
    Gm=DT.MultiCriticalGradedComplex([[1,2],[3]],[B[:,1:1]],
        [[(0,1),(1,0)],[(0,0)],[(1,1)]])
    for semantics in (:union,:intersection)
        enc=TamerOp.encode(Gm,TamerOp.FiltrationSpec(kind=:graded,multicritical=semantics))
        p=TamerOp.h0_presentation(enc;syzygies=true)
        @test HP.check_h0_presentation(p).valid
        @test TamerOp.betti_table(enc)==TamerOp.betti_table(TamerOp.resolve(enc))
    end
    # Non-incidence chain columns retain the existing field-aware algebra.
    Gbad=DT.GradedComplex([[1,2],[3]],[sparse(reshape([1,1],2,1))],[(0,0),(0,0),(1,1)])
    enc=TamerOp.encode(Gbad,DI.GradedFiltration())
    @test_throws ArgumentError TamerOp.h0_presentation(enc)
    @test TamerOp.betti_table(enc)==TamerOp.betti_table(TamerOp.resolve(enc))
end

@testset "A48 dynamic forest agrees with independent minimax paths" begin
    HP=TamerOp.H0Presentations
    rng=MersenneTwister(48002)
    for n in (2,7,25)
        edges=[(rand(rng,1:n),rand(rng,1:n)) for _ in 1:120]
        weights=rand(rng,1:40,length(edges))
        T=HP._H0Forest(vcat(zeros(Int,n),weights))
        distance=fill(typemax(Int),n,n)
        for i in 1:n; distance[i,i]=0; end
        for (e,(u,v)) in enumerate(edges)
            connected=HP._h0_findroot!(T,u)==HP._h0_findroot!(T,v)
            @test connected==(distance[u,v]<typemax(Int))
            old=0
            if connected
                HP._h0_makeroot!(T,u);HP._h0_access!(T,v)
                old=T.maximum[v]
                @test T.weight[old]==distance[u,v]
                path=HP._h0_path_nodes(T,v)
                @test first(path)==u && last(path)==v
                @test all(j->Set(edges[path[j]-n])==Set((path[j-1],path[j+1])),2:2:length(path)-1)
            end
            if !connected || T.weight[old]>weights[e]
                if connected
                    a,b=edges[old-n]
                    HP._h0_cut!(T,a,old);HP._h0_cut!(T,b,old)
                end
                HP._h0_link!(T,u,n+e);HP._h0_link!(T,n+e,v)
            end
            distance[u,v]=distance[v,u]=min(distance[u,v],weights[e])
            for k in 1:n,a in 1:n,b in 1:n
                distance[a,b]=min(distance[a,b],max(distance[a,k],distance[k,b]))
            end
        end
    end
end

@testset "A48 ownership and encoding boundary cases" begin
    HP=TamerOp.H0Presentations
    P=FF.ProductOfChainsPoset((2,2));vg=[1,1];es=[(1,2),(1,2)];eg=[2,3]
    p=HP.h0_presentation(P;vertex_grades=vg,edges=es,edge_grades=eg,syzygies=true)
    vg[1]=4;es[1]=(1,1);eg[1]=4
    @test HP.check_h0_presentation(p).valid
    A=HP.presentation_matrix(p);A[1,1]=0
    @test HP.check_h0_presentation(p).valid
    s=HP.check_h0_presentation(p)
    @test s==deepcopy(s) && isequal(s,deepcopy(s)) && hash(s)==hash(deepcopy(s))
    bad=deepcopy(p);bad.original_vertices[1]=2
    @test !HP.check_h0_presentation(bad).valid
    Q=FF.ProductOfChainsPoset((2,2,2))
    @test_throws ArgumentError HP.h0_presentation(Q;vertex_grades=[1],edges=[],edge_grades=[],syzygies=true)
    B=sparse([1,2,1,2],[1,1,2,2],[-1,1,-1,1],2,2)
    G=DT.GradedComplex([[1,2],[3,4]],[B],[(0,0),(0,0),(2,0),(0,2)])
    for pipeline in (OPT.PipelineOptions(axes_policy=:coarsen,max_axis_len=1),
                     OPT.PipelineOptions(axes_policy=:coarsen,max_axis_len=2))
        enc=TamerOp.encode(G,DI.GradedFiltration();pipeline)
        @test TamerOp.betti_table(enc)==TamerOp.betti_table(TamerOp.resolve(enc))
    end
    reverseG=DT.GradedComplex([[1,2],[3,4]],[B],[(2,2),(2,2),(0,2),(2,0)])
    enc=TamerOp.encode(reverseG,TamerOp.FiltrationSpec(kind=:graded,orientation=(-1,-1)))
    @test TamerOp.betti_table(enc)==TamerOp.betti_table(TamerOp.resolve(enc))
    @test DI._h0_presentation_input(enc.M.lazy;max_expanded_cells=1,allow_fallback=true)===nothing
    enc1=TamerOp.encode(G,DI.GradedFiltration();degree=1)
    @test_throws ArgumentError TamerOp.h0_presentation(enc1)
    @test TamerOp.betti_table(enc1)==TamerOp.betti_table(TamerOp.resolve(enc1))
end

@testset "A48 higher Betti numbers retain characteristic dependence" begin
    HP=TamerOp.H0Presentations
    # Face poset of RP2 with an added maximum. The interval below that maximum
    # has H1 and H2 over F2, while both reduced rational homology groups vanish.
    facets=[(1,2,3),(1,2,6),(1,3,5),(1,4,5),(1,4,6),
            (2,3,4),(2,4,5),(2,5,6),(3,4,6),(3,5,6)]
    faces=Set{Tuple}()
    for f in facets, mask in 1:7
        push!(faces,Tuple(f[i] for i in 1:3 if !iszero(mask & (1<<(i-1)))))
    end
    faces=sort!(collect(faces);by=f->(length(f),f));n=length(faces)+1
    order=falses(n,n)
    for a in 1:n-1,b in 1:n-1;order[a,b]=issubset(faces[a],faces[b]);end
    order[:,n].=true
    P=FF.FinitePoset(order)
    eg=findall(f->length(f)==2,faces);es=NTuple{2,Int}[faces[e] for e in eg]
    for field in (CM.QQField(),CM.F2(),CM.F3())
        p=HP.h0_presentation(P;vertex_grades=collect(1:6),edges=es,edge_grades=eg,field)
        @test HP.check_h0_presentation(p).valid
        B=HP.betti_table(p;pad_to=3)
        @test B[1,1:6]==ones(Int,6)
        @test B[2,eg]==ones(Int,length(eg))
        @test B[3,22:31]==ones(Int,10)
        @test B[3:4,n]==(field==CM.F2() ? [1,1] : [0,0])
    end
end

@testset "A48 RealField general resolution and cache regression" begin
    # Original sparse_16_Real benchmark input. Retain its seed and draw order.
    rng=MersenneTwister(48100);n=16;m=32
    edges=[(rand(rng,1:n),rand(rng,1:n)) for _ in 1:m]
    eg=[(rand(rng,1:5),rand(rng,1:5)) for _ in edges]
    B=sparse(vcat(first.(edges),last.(edges)),vcat(1:m,1:m),vcat(fill(-1,m),fill(1,m)),n,m)
    G=DT.GradedComplex([collect(1:n),collect(1:m)],[B],vcat(fill((0,0),n),eg))
    exact_enc=TamerOp.encode(G,DI.GradedFiltration();field=CM.QQField())
    expected=TamerOp.betti_table(TamerOp.resolve(exact_enc))
    field=CM.RealField(Float64)
    enc=TamerOp.encode(G,DI.GradedFiltration();field)
    checked=TamerOp.resolve(enc)
    res=TamerOp.resolution_object(checked)
    @test TamerOp.betti_table(checked)==expected==TamerOp.betti_table(enc)
    @test DF.check_projective_resolution(res;check_cover=true,throw=true).valid
    @test DF.betti_table(DF.projective_resolution(res.M,OPT.ResolutionOptions();threads=false))==expected
    @test DF.betti_table(DF.projective_resolution(res.M,OPT.ResolutionOptions();threads=true))==expected
    # Check actual maps and exactness independently of the repaired validator.
    for f in vcat([res.aug],res.d_mor)
        @test MD.check_morphism(f).valid
    end
    maps=vcat([res.aug],res.d_mor)
    for v in 1:FF.nvertices(res.M.Q)
        @test FL.rank(field,res.aug.comps[v])==res.M.dims[v]
        for i in 1:length(maps)-1
            A,C=maps[i].comps[v],maps[i+1].comps[v]
            @test norm(A*C)<=1e-12*norm(A)*norm(C)
            @test FL.rank(field,A)+FL.rank(field,C)==size(A,2)
        end
        @test FL.rank(field,last(maps).comps[v])==size(last(maps).comps[v],2)
    end
    cache=TamerOp.SessionCache()
    first_result=TamerOp.resolve(enc;cache)
    @test TamerOp.betti_table(enc;cache)==expected
    @test TamerOp.resolution_object(TamerOp.resolve(enc;cache))===TamerOp.resolution_object(first_result)
    CM._clear_session_cache!(cache)
    @test TamerOp.resolution_object(TamerOp.resolve(enc;cache))!==TamerOp.resolution_object(first_result)
    @test TamerOp.betti_table(enc;cache)==expected
    @test all(t -> fetch(t)==expected,[Threads.@spawn TamerOp.betti_table(enc;cache) for _ in 1:4])

    # Transposition on the opposite poset gives an independent dual oracle.
    Q=FF.FinitePoset(transpose(FF.leq_matrix(res.M.Q)))
    function dual(M,field)
        K=CM.coeff_type(field)
        maps=Dict((v,u)=>Matrix{K}(transpose(A)) for ((u,v),A) in M.edge_maps)
        return MD.PModule{K}(Q,copy(M.dims),maps;field)
    end
    exact_source=TamerOp.resolution_object(TamerOp.resolve(exact_enc)).M
    opts=OPT.ResolutionOptions(maxlen=3)
    exact_dual=DF.injective_resolution(dual(exact_source,CM.QQField()),opts;threads=false)
    numerical_dual=DF.injective_resolution(dual(res.M,field),opts;threads=false)
    @test DF.check_injective_resolution(numerical_dual;check_hull=true,throw=true).valid
    @test DF.bass_table(numerical_dual)==DF.bass_table(exact_dual)==expected
end
