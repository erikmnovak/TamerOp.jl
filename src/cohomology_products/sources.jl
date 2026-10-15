# Simplicial inclusions on finite posets and scalar/multiparameter filtrations.

function _simplicial_diagram(P,complexes,F,budget)
    Q=_cup_poset(P,budget);K=coeff_type(_cup_field(F))
    length(complexes)==nvertices(Q) || throw(ArgumentError("one simplicial complex per poset vertex is required"))
    w=_CupWork(budget);As=CohomologyAlgebra{K,typeof(F)}[]
    for cells in complexes
        push!(As,_simplicial_algebra(cells,F,w))
    end
    D=maximum(length(A.dims) for A in As)-1
    maps=Dict{Tuple{Int,Int},Vector{Matrix{K}}}()
    for (u,v) in cover_edges(Q)
        A=As[v];B=As[u];ms=Matrix{K}[]
        issubset(Set(complexes[u]),Set(complexes[v])) || throw(ArgumentError("simplicial complexes must be nested along every poset cover"))
        for d in 0:D
            na=_hdim(A,d);nb=_hdim(B,d)
            _spend!(w;entries=big(na)*nb)
            if na==0 || nb==0
                push!(ms,zeros(K,nb,na));continue
            end
            source=Dict(s=>j for (j,s) in enumerate(A.cochains.cells[d+1]))
            rows=[source[s] for s in B.cochains.cells[d+1]]
            push!(ms,_quotient_coordinates(F,B.cochains,A.cochains.representatives[d+1][rows,:],d))
        end
        maps[v,u]=ms
    end
    _cup_report(_restriction_issues(Q,As,maps,w),true)
    PersistentCupProducts(Q,As,maps,nothing,:poset,F,:simplicial_inclusion_diagram)
end

"""
    persistent_cup_products(P, complexes; field=F2(), budget=CupProductBudget())

Compute cohomology algebras and their restrictions for simplicial complexes
indexed by a finite poset. Each entry lists simplices using a common set of
positive vertex IDs; faces are added automatically. For every u <= v, the
complex at u must be a subcomplex of the one at v. Products use the ordered
Alexander--Whitney diagonal, with exact quotient coordinates over the field.
"""
function persistent_cup_products(P::AbstractPoset,complexes::AbstractVector;
        field=F2(),budget=CupProductBudget())
    Q=_cup_poset(P,budget)
    length(complexes)==nvertices(Q) || throw(ArgumentError("one complex per vertex is required"))
    closed=[first(_simplicial_filtration(c,zeros(Int,length(c)),:sublevel,budget)) for c in complexes]
    _simplicial_diagram(Q,closed,field,budget)
end

"""
    persistent_cup_products(simplices; grades, field=F2(), order=:sublevel,
                            budget=CupProductBudget())

Compute the ring diagram at every scalar filtration event. Faces enter at
their earliest coface grade. Tuple grades instead construct the finite product
grid of all critical coordinates, including their coordinatewise joins.
The matrix work is bounded before constructing the grid. Multiparameter output
retains this finite base; `cup_pmodule` exposes its image modules.
"""
function persistent_cup_products(simplices::AbstractVector;grades,field=F2(),
        order=:sublevel,budget=CupProductBudget())
    if !isempty(grades) && first(grades) isa Tuple
        return _graded_cup_products(simplices,[[g] for g in grades],field,order,budget;closed=false)
    end
    cells,gs=_simplicial_filtration(simplices,grades,order,budget)
    isempty(gs) && throw(ArgumentError("empty filtration has no event parameter; supply a zero algebra at a declared parameter"))
    ts=sort!(unique(gs);rev=order===:superlevel);n=length(ts)
    n<=budget.max_stages && big(n)^2<=budget.max_entries || throw(ArgumentError("cup chain exceeds budget"))
    P=FinitePoset([i<=j for i in 1:n,j in 1:n])
    complexes=[[s for (s,g) in zip(cells,gs) if order===:sublevel ? g<=t : g>=t] for t in ts]
    R=_simplicial_diagram(P,complexes,field,budget)
    PersistentCupProducts(R.poset,R.algebras,R.restrictions,ts,order,R.field,:ordered_simplicial_filtration)
end

function _graded_cup_products(simplices,grade_sets,F,order,budget;closed)
    _cup_order(order);_cup_field(F)
    length(simplices)==length(grade_sets) || throw(ArgumentError("one grade set per simplex is required"))
    0<length(simplices)<=budget.max_cells || throw(ArgumentError("nonempty source within max_cells is required"))
    all(g->!isempty(g),grade_sets) || throw(ArgumentError("every simplex needs a nonempty grade set"))
    grades=collect(Iterators.flatten(grade_sets));N=length(first(grades))
    N>0 && all(g->g isa Tuple && length(g)==N && all(t->t isa Real && !(t isa Bool) && isfinite(t),g),grades) ||
        throw(ArgumentError("grades must be finite real tuples of a common positive dimension"))
    T=mapreduce(typeof,promote_type,Iterators.flatten(grades))
    before(a,b)=all(k->order===:sublevel ? a[k]<=b[k] : a[k]>=b[k],1:N)
    faces=Dict{Tuple{Vararg{Int}},Vector{NTuple{N,T}}}()
    for (simplex,gs) in zip(simplices,grade_sets)
        cs,_=_simplicial_filtration([simplex],[0],:sublevel,budget)
        for g in gs
            t=ntuple(k->iszero(g[k]) ? zero(T) : convert(T,g[k]),N)
            all(k->t[k]==g[k],1:N) || throw(ArgumentError("grade conversion loses precision"))
            for s in cs
                length(faces)<budget.max_cells || haskey(faces,s) || throw(ArgumentError("face closure exceeds max_cells"))
                births=get!(faces,s,NTuple{N,T}[])
                any(b->before(b,t),births) && continue
                filter!(b->!before(t,b),births);push!(births,t)
            end
        end
    end
    if closed
        originals=Dict(Tuple(sort!(collect(s)))=>gs for (s,gs) in zip(simplices,grade_sets))
        length(originals)==length(simplices)==length(faces) || throw(ArgumentError("simplex tree must contain distinct simplices and all faces"))
        for (s,bs) in faces,b in bs
            any(g->before(g,b),originals[s]) || throw(ArgumentError("a face enters after one of its cofaces"))
        end
    end
    coords=ntuple(k->sort!(unique(T[g[k] for gs in values(faces) for g in gs]);rev=order===:superlevel),N)
    shape=length.(coords);n=prod(big.(shape))
    n<=budget.max_stages && n^2<=budget.max_entries || throw(ArgumentError("critical-coordinate product grid exceeds cup budget"))
    # Bound cell membership storage as well as the subsequent algebra matrices.
    n*length(faces)<=budget.max_entries || throw(ArgumentError("simplicial grid membership exceeds cup budget"))
    ts=[ntuple(k->coords[k][i[k]],N) for i in CartesianIndices(shape)] |> vec
    cells=sort!(collect(keys(faces));by=s->(length(s),s))
    complexes=[[s for s in cells if any(g->before(g,t),faces[s])] for t in ts]
    R=_simplicial_diagram(ProductOfChainsPoset(shape),complexes,F,budget)
    parameters=N==1 ? [only(t) for t in ts] : ts
    PersistentCupProducts(R.poset,R.algebras,R.restrictions,parameters,order,R.field,:ordered_simplicial_filtration)
end

"""
    persistent_cup_products(tree::SimplexTreeMulti; field=F2(), order=:sublevel,
                            budget=CupProductBudget())

Compute products on the full finite critical-coordinate grid of a simplicial
multifiltration, in any parameter dimension. A simplex with several grades is
present when at least one of them precedes the query. Validate all face grades
in the declared order; no new face or earlier entry is added to a tree input.
"""
function persistent_cup_products(tree::DataTypes.SimplexTreeMulti;field=F2(),
        order=:sublevel,budget=CupProductBudget())
    DataTypes.check_simplex_tree_multi(tree;throw=true)
    n=DataTypes.simplex_count(tree)
    n<=budget.max_cells || throw(ArgumentError("supplied simplices exceed max_cells"))
    cells=[Tuple(DataTypes.simplex_vertices(tree,i)) for i in 1:n]
    gs=[collect(DataTypes.simplex_grades(tree,i)) for i in 1:n]
    _graded_cup_products(cells,gs,field,order,budget;closed=true)
end

function persistent_cup_products(::PModule;kwargs...)
    throw(ArgumentError("a vector-space persistence module does not determine cup products; supply compatible cohomology algebras or a simplicial filtration"))
end
