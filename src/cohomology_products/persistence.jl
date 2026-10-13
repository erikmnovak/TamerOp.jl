# Contravariant algebra diagrams and actual cup-image submodules on finite chains.

"""A checked chain of graded algebras with restrictions from stage i+1 to i.
Parameters are in filtration order. Between events the object is constant,
zero before the first event, and constant after the last. Source simplicial
filtrations include every event; supplied diagrams declare this extension."""
struct PersistentCupProducts{K,F,T}
    algebras::Vector{CohomologyAlgebra{K,F}}
    restrictions::Vector{Vector{Matrix{K}}}
    parameters::Vector{T}
    order::Symbol
    field::F
    origin::Symbol
end
field(R::PersistentCupProducts)=R.field
"""Exact event parameters, ordered in filtration order."""
cup_parameters(R::PersistentCupProducts)=copy(R.parameters)
function _stage(R,i)
    i=_cup_int(i;positive=true)
    i<=length(R.algebras) || throw(ArgumentError("stage outside the represented filtration"))
    i
end
"""A copy of the algebra at a one-based filtration stage."""
cohomology_algebra(R::PersistentCupProducts;stage)=deepcopy(R.algebras[_stage(R,stage)])
_ring_degree(R)=maximum(length(A.dims) for A in R.algebras)-1

function _restriction_issues(As,maps,w)
    K=coeff_type(field(first(As)));D=maximum(length(A.dims) for A in As)-1
    length(maps)==length(As)-1 || return ["one restriction is required per adjacent stage"]
    for i in eachindex(maps)
        A=As[i+1];B=As[i];fs=maps[i]
        length(fs)==D+1 || return ["restriction does not cover all degrees"]
        all(d->size(fs[d+1])==(_hdim(B,d),_hdim(A,d)),0:D) || return ["restriction matrix shape disagrees"]
        fs[1]*A.unit==B.unit || return ["restriction does not preserve the unit"]
        for p in 0:D,q in 0:D-p
            np=_hdim(A,p);nq=_hdim(A,q)
            _spend!(w;products=big(np)*nq)
            table=get(A.products,(p,q),nothing)
            for j in 1:nq,k in 1:np
                lhs=table===nothing ? zeros(K,_hdim(B,p+q)) : fs[p+q+1]*view(table,:,k+(j-1)*np)
                lhs==_multiply(B,view(fs[p+1],:,k),view(fs[q+1],:,j),p,q) ||
                    return ["restriction $i does not preserve products in degrees ($p,$q)"]
            end
        end
    end
    String[]
end

"""
    persistent_cup_products(algebras; parameters, restrictions,
                            order=:sublevel, budget=CupProductBudget())

Build a checked contravariant ring diagram. `restrictions[i][d]` is a matrix
H^d(stage i+1) -> H^d(stage i), keyed by zero-based degree. Omitted degree
matrices are zero, not identities. All algebras must use the same exact field;
restriction maps preserve degree, unit, and products. Parameters are strictly
ordered (increasing for sublevels, decreasing for superlevels). The input is
copied. A supplied algebra diagram need not have a geometric realization.
"""
function persistent_cup_products(algebras::AbstractVector{<:CohomologyAlgebra};
        parameters,restrictions,order=:sublevel,budget=CupProductBudget())
    isempty(algebras) && throw(ArgumentError("at least one algebra is required"))
    length(algebras)<=budget.max_stages || throw(ArgumentError("cup stage budget exceeded"))
    ts=_cup_parameters(parameters,order)
    length(ts)==length(algebras) || throw(ArgumentError("one parameter per algebra is required"))
    length(restrictions)==length(algebras)-1 || throw(ArgumentError("one restriction per adjacent stage is required"))
    F=field(first(algebras));K=coeff_type(F);_cup_field(F);w=_CupWork(budget)
    all(A->field(A)==F,algebras) || throw(ArgumentError("all algebras must use the same field"))
    for A in algebras
        _spend!(w;entries=sum(big(length(M)) for M in values(A.products);init=big(0))+sum(big(d)^2 for d in A.dims))
        _cup_report(_algebra_issues(A,w),true)
    end
    As=CohomologyAlgebra{K,typeof(F)}[deepcopy(A) for A in algebras]
    D=maximum(length(A.dims) for A in As)-1;maps=Vector{Matrix{K}}[]
    for (i,input) in enumerate(restrictions)
        input isa AbstractDict || throw(ArgumentError("restrictions must be dictionaries keyed by degree"))
        all(d->d isa Integer && !(d isa Bool) && 0<=d<=D,keys(input)) || throw(ArgumentError("invalid restriction degree"))
        ms=Matrix{K}[]
        for d in 0:D
            shape=(_hdim(As[i],d),_hdim(As[i+1],d))
            _spend!(w;entries=big(shape[1])*shape[2])
            M=get(input,d,nothing)
            M===nothing || (M isa AbstractMatrix && size(M)==shape) || throw(ArgumentError("restriction has wrong shape"))
            push!(ms,M===nothing ? zeros(K,shape) : Matrix{K}(_cup_coefficients(F,M)))
        end
        push!(maps,ms)
    end
    _cup_report(_restriction_issues(As,maps,w),true)
    PersistentCupProducts(As,maps,ts,order,F,:supplied_algebra_diagram)
end

"""
    persistent_cup_products(simplices; grades, field=F2(), order=:sublevel,
                            budget=CupProductBudget())
    persistent_cup_products(tree::SimplexTreeMulti{1}; field=F2(), order=:sublevel,
                            budget=CupProductBudget())

Compute rings and actual restriction maps at every distinct simplex grade.
The simplex-list form closes under faces; each face enters at the earliest
grade of any supplied coface (minimum for sublevels, maximum for superlevels).
Tied entries form one event. The simplex-tree form requires a valid one-critical
filtration and preserves its supplied grades. No arbitrary multiparameter ring
construction or cellular diagonal is inferred. Uses exact cochain quotients;
this route does not claim the incremental complexity of Dey--Rathod (2024).
"""
function persistent_cup_products(simplices::AbstractVector;grades,field=F2(),
                                 order=:sublevel,budget=CupProductBudget())
    F=_cup_field(field);K=coeff_type(F)
    cells,gs=_simplicial_filtration(simplices,grades,order,budget)
    isempty(gs) && throw(ArgumentError("empty filtration has no event parameter; supply a zero algebra at a declared parameter"))
    ts=sort!(unique(gs);rev=order===:superlevel)
    length(ts)<=budget.max_stages || throw(ArgumentError("cup stage budget exceeded"))
    w=_CupWork(budget);As=CohomologyAlgebra{K,typeof(F)}[]
    for t in ts
        active=[s for (s,g) in zip(cells,gs) if order===:sublevel ? g<=t : g>=t]
        push!(As,_simplicial_algebra(active,F,w))
    end
    D=maximum(length(A.dims) for A in As)-1;maps=Vector{Matrix{K}}[]
    for i in 1:length(As)-1
        A=As[i+1];B=As[i];ms=Matrix{K}[]
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
        push!(maps,ms)
    end
    _cup_report(_restriction_issues(As,maps,w),true)
    PersistentCupProducts(As,maps,ts,order,F,:ordered_simplicial_filtration)
end

function persistent_cup_products(tree::DataTypes.SimplexTreeMulti{1};field=F2(),
                                 order=:sublevel,budget=CupProductBudget())
    DataTypes.check_simplex_tree_multi(tree;throw=true)
    n=DataTypes.simplex_count(tree)
    cells=[Tuple(DataTypes.simplex_vertices(tree,i)) for i in 1:n]
    all(i->length(DataTypes.simplex_grades(tree,i))==1,1:n) || throw(ArgumentError("a one-critical simplex tree is required"))
    gs=[only(DataTypes.simplex_grades(tree,i))[1] for i in 1:n]
    closed,values=_simplicial_filtration(cells,gs,order,budget)
    recorded=Dict(Tuple(sort!(collect(s)))=>g for (s,g) in zip(cells,gs))
    length(closed)==n && all(haskey(recorded,s) && recorded[s]==g for (s,g) in zip(closed,values)) ||
        throw(ArgumentError("simplex grades must already define a face-closed filtration in the declared order"))
    persistent_cup_products(cells;grades=gs,field,order,budget)
end

function persistent_cup_products(::DataTypes.SimplexTreeMulti;kwargs...)
    throw(ArgumentError("cup persistence requires a one-parameter simplicial filtration; supply an explicit chain restriction first"))
end

function persistent_cup_products(::PModule;kwargs...)
    throw(ArgumentError("a vector-space persistence module does not determine cup products; supply compatible cohomology algebras or a simplicial filtration"))
end

"""Compose actual cohomology restrictions from the later `source` stage to the
earlier `target` stage, in a named degree. Stages are indices, not scalar grades."""
function cohomology_restriction(R::PersistentCupProducts{K};source,target,degree) where {K}
    s=_stage(R,source);t=_stage(R,target);d=_cup_int(degree)
    t<=s || throw(ArgumentError("cohomology restrictions run from later to earlier filtration stages"))
    d<=_ring_degree(R) || throw(ArgumentError("degree outside the stored range"))
    M=eye(field(R),_hdim(R.algebras[s],d))
    for i in s-1:-1:t;M=R.restrictions[i][d+1]*M;end
    M
end

function _chain_module(F,dims,maps)
    n=length(dims);P=FinitePoset([i>=j for i in 1:n,j in 1:n])
    K=coeff_type(F)
    PModule{K}(P,Int[d for d in dims],Dict{Tuple{Int,Int},Matrix{K}}((i+1,i)=>M for (i,M) in enumerate(maps));field=F)
end
"""The additive degree-d cohomology PModule. Vertex i is filtration stage i;
its poset is the opposite chain, with arrows i+1 -> i."""
function cohomology_module(R::PersistentCupProducts;degree)
    d=_cup_int(degree);d<=_ring_degree(R) || throw(ArgumentError("degree outside stored range"))
    _chain_module(field(R),[_hdim(A,d) for A in R.algebras],[copy(ms[d+1]) for ms in R.restrictions])
end

function _span_products(A::CohomologyAlgebra{K},left,right,p,q,w) where {K}
    n=big(size(left,2))*size(right,2);h=_hdim(A,p+q)
    _spend!(w;entries=n*h,products=n)
    out=zeros(K,h,Int(n))
    for j in axes(right,2),i in axes(left,2)
        out[:,i+(j-1)*size(left,2)]=_multiply(A,view(left,:,i),view(right,:,j),p,q)
    end
    Matrix{K}(FieldLinAlg.colspace(field(A),out))
end

function _power_images(A::CohomologyAlgebra{K},k,w;generators=nothing) where {K}
    D=length(A.dims)-1
    base=generators===nothing ? [d==0 ? zeros(K,A.dims[1],0) : eye(field(A),_hdim(A,d)) for d in 0:D] : generators
    k>D && return [zeros(K,_hdim(A,d),0) for d in 0:D]
    images=deepcopy(base)
    for power in 2:k
        next=[zeros(K,_hdim(A,d),0) for d in 0:D]
        for d in power:D
            blocks=Matrix{K}[]
            for p in power-1:d-1
                push!(blocks,_span_products(A,images[p+1],base[d-p+1],p,d-p,w))
            end
            next[d+1]=Matrix{K}(FieldLinAlg.colspace(field(A),hcat(blocks...)))
        end
        images=next
    end
    images
end
function _partition_images(A::CohomologyAlgebra{K},parts,w) where {K}
    D=length(A.dims)-1;out=[zeros(K,_hdim(A,d),0) for d in 0:D]
    sum(big.(parts))>D && return out
    degree=first(parts);V=eye(field(A),_hdim(A,degree))
    for q in parts[2:end]
        V=_span_products(A,V,eye(field(A),_hdim(A,q)),degree,q,w);degree+=q
    end
    out[degree+1]=V;out
end

"""A cup-image submodule with its actual embeddings in a retained ring diagram.
Use `cup_module`, `partition_module`, `cup_intervals`, `cup_embedding`,
`cup_pmodule`, `dimensions`, and `structure_map`. Vertex numbers remain original
filtration stages; maps point from later to earlier stages."""
struct CupModule{K,R,M}
    ring::R
    images::Vector{Vector{Matrix{K}}}
    pmodule::M
    power::Int
    partition::Union{Nothing,Vector{Int}}
    degree::Union{Nothing,Int}
end
field(V::CupModule)=field(V.ring)
dimensions(V::CupModule)=copy(V.pmodule.dims)
"""Return the exact image PModule on the opposite stage chain (an owned copy)."""
cup_pmodule(V::CupModule)=deepcopy(V.pmodule)
cup_parameters(V::CupModule)=cup_parameters(V.ring)
"""Actual image-submodule map from later source to earlier target stage."""
function structure_map(V::CupModule;source,target)
    s=_stage(V.ring,source);t=_stage(V.ring,target)
    t<=s || throw(ArgumentError("cup maps run from later to earlier stages"))
    copy(structure_map(V.pmodule;source=s,target=t))
end

function _image_embedding(A::CohomologyAlgebra{K},images) where {K}
    out=zeros(K,sum(A.dims),sum(size(M,2) for M in images));r=0;c=0
    for (d,M) in enumerate(images)
        out[r+1:r+size(M,1),c+1:c+size(M,2)]=M
        r+=size(M,1);c+=size(M,2)
    end
    out
end
"""Embedding columns in the direct sum H^0 + H^1 + ... of a stage algebra.
Rows and columns are grouped by increasing degree; `dimensions(algebra)` gives
the ambient row blocks. This is an actual map, not a dimension-only summary."""
function cup_embedding(V::CupModule;stage)
    i=_stage(V.ring,stage);_image_embedding(V.ring.algebras[i],V.images[i])
end

function _image_module(R::PersistentCupProducts{K},power,parts,degree,budget) where {K}
    _cup_int(power;positive=true)
    parts===nothing || (!isempty(parts) && all(p->p isa Integer && !(p isa Bool) && p>0,parts)) ||
        throw(ArgumentError("partition must consist of positive degrees"))
    length(R.algebras)<=budget.max_stages || throw(ArgumentError("cup stage budget exceeded"))
    degree===nothing || _cup_int(degree;positive=true)
    D=_ring_degree(R);degree===nothing || degree<=D || throw(ArgumentError("requested degree exceeds the stored ring"))
    w=_CupWork(budget);images=Vector{Matrix{K}}[]
    for A in R.algebras
        _spend!(w;entries=sum(big(d)^2 for d in A.dims))
        bs=parts===nothing ? _power_images(A,power,w) : _partition_images(A,parts,w)
        if degree!==nothing
            for d in 0:length(bs)-1;d==degree || (bs[d+1]=zeros(K,_hdim(A,d),0));end
        end
        push!(images,bs)
    end
    dims=[sum(size(B,2) for B in bs) for bs in images];maps=Matrix{K}[]
    for i in eachindex(R.restrictions)
        _spend!(w;entries=big(dims[i])*dims[i+1])
        M=zeros(K,dims[i],dims[i+1]);r=0;c=0
        for d in 0:D
            B=d<length(images[i]) ? images[i][d+1] : zeros(K,0,0)
            C=d<length(images[i+1]) ? images[i+1][d+1] : zeros(K,0,0)
            nr=size(B,2);nc=size(C,2)
            M[r+1:r+nr,c+1:c+nc]=FieldLinAlg.solve_fullcolumn(field(R),B,R.restrictions[i][d+1]*C;cache=false)
            r+=nr;c+=nc
        end
        push!(maps,M)
    end
    _spend!(w;entries=big(length(dims))^2)
    CupModule(R,images,_chain_module(field(R),dims,maps),power,parts,degree)
end
"""
    cup_module(R; factors=2, degree=nothing, budget=CupProductBudget())

The image spanned by products of `factors` positive-degree cohomology classes,
with restrictions inherited from R. `factors=1` is all positive cohomology.
Omit `degree` for the direct sum across all total degrees, or select one total
degree. H^0 factors are excluded so units cannot make cup-length infinite.
"""
function cup_module(R::PersistentCupProducts;factors=2,degree=nothing,budget=CupProductBudget())
    k=_cup_int(factors;positive=true)
    _image_module(R,k,nothing,degree,budget)
end
"""Image of products with the specified positive factor degrees. Repeated
degrees are allowed; order changes at most signs and therefore not the span.
Unlike a k-cup module, this fixes the full degree partition."""
function partition_module(R::PersistentCupProducts,parts;budget=CupProductBudget())
    isempty(parts) && throw(ArgumentError("a degree partition must be nonempty"))
    ps=sort!(Int[_cup_int(p;positive=true) for p in parts])
    _image_module(R,length(ps),ps,nothing,budget)
end

"""Intervals in original parameter units, with `death=nothing` for terminal
classes under the declared constant continuation. Includes exact first/last
stage indices and multiplicities as separate records. Birth is included and
finite death excluded in filtration order, including superlevels. The
underlying algebra maps remain contravariant."""
function cup_intervals(V::CupModule;budget=CupProductBudget())
    ds=dimensions(V);n=length(ds);K=coeff_type(field(V))
    maps=Matrix{K}[structure_map(V;source=i+1,target=i) for i in 1:n-1]
    b=ZigzagPersistence.ZigzagBudget(max_vertices=budget.max_stages,
        max_dimension=budget.max_entries,max_matrix_entries=budget.max_entries)
    z=ZigzagPersistence.ZigzagModule(ds,maps;directions=fill(:backward,n-1),field=field(V),budget=b)
    intervals=ZigzagPersistence.zigzag_intervals(ZigzagPersistence.zigzag_persistence(z;budget=b))
    ts=V.ring.parameters
    [(;birth=ts[x.first],death=x.last==n ? nothing : ts[x.last+1],
       first_stage=x.first,last_stage=x.last,birth_included=true,death_included=false,
       order=V.ring.order) for x in intervals]
end

function _cup_length(A,w;generators=nothing)
    D=length(A.dims)-1;best=0
    _spend!(w;entries=sum(big(d)^2 for d in A.dims))
    for k in 1:D
        images=_power_images(A,k,w;generators)
        any(M->size(M,2)>0,images) || break
        best=k
    end
    best
end
"""Maximum length of a nonzero product of positive-degree classes; zero if
there are none. For a persistence interval use the ring-diagram method."""
cup_length(A::CohomologyAlgebra;budget=CupProductBudget())=_cup_length(A,_CupWork(budget))
"""
    cup_length(R; source=length(cup_parameters(R)), target=source, budget=CupProductBudget())

Cup-length of the image ring of the restriction from source to target stage.
With equal stages this is the pointwise cup-length. In general it is not the
minimum of the endpoint cup-lengths: products must survive the actual map.
"""
function cup_length(R::PersistentCupProducts{K};source=length(R.algebras),target=source,
                    budget=CupProductBudget()) where {K}
    s=_stage(R,source);t=_stage(R,target);t<=s || throw(ArgumentError("source must be a later or equal stage"))
    A=R.algebras[t];w=_CupWork(budget)
    _spend!(w;entries=sum(big(_hdim(A,d))*_hdim(R.algebras[s],d) for d in 0:length(A.dims)-1))
    generators=[d==0 ? zeros(K,_hdim(A,0),0) : Matrix{K}(FieldLinAlg.colspace(field(R),
        cohomology_restriction(R;source=s,target=t,degree=d))) for d in 0:length(A.dims)-1]
    _cup_length(A,w;generators)
end

provenance(R::PersistentCupProducts)=(;source=R.origin,field=field(R),variance=:contravariant,
    indexing=:opposite_filtration_chain,order=R.order,continuation=:zero_before_constant_after,
    arithmetic=:exact,product_convention=:positive_degree_factors,
    basis_choice=:noncanonical,geometry=R.origin===:supplied_algebra_diagram ? :not_asserted : :abstract_simplicial)
describe(R::PersistentCupProducts)=(;kind=:persistent_cup_products,stages=length(R.parameters),
    parameters=Tuple(R.parameters),dimensions=Tuple(Tuple(A.dims) for A in R.algebras),provenance=provenance(R))
describe(V::CupModule)=(;kind=:cup_module,factors=V.power,partition=V.partition===nothing ? nothing : Tuple(V.partition),
    degree=V.degree,dimensions=Tuple(dimensions(V)),provenance=provenance(V.ring))
"""Cheap dimensions, factor/partition choices, and mathematical provenance."""
cup_product_summary(x::Union{CohomologyAlgebra,PersistentCupProducts,CupModule})=describe(x)
Base.show(io::IO,R::PersistentCupProducts)=print(io,"PersistentCupProducts(stages=",length(R.parameters),", field=",field(R),", order=",R.order,")")
Base.show(io::IO,V::CupModule)=print(io,"CupModule(factors=",V.power,", partition=",V.partition,", dimensions=",Tuple(dimensions(V)),")")

"""Check every algebra and actual restriction. No geometric realization is asserted."""
function check_persistent_cup_products(R::PersistentCupProducts;throw=false,budget=CupProductBudget())
    issues=String[];w=_CupWork(budget)
    if isempty(R.algebras) || length(R.algebras)!=length(R.parameters)
        push!(issues,"algebras and parameters disagree")
    else
        try
            _cup_parameters(R.parameters,R.order)
        catch err
            err isa ArgumentError || rethrow();push!(issues,sprint(showerror,err))
        end
        for A in R.algebras
            field(A)==field(R) || push!(issues,"coefficient fields disagree")
            append!(issues,_algebra_issues(A,w))
        end
        isempty(issues) && append!(issues,_restriction_issues(R.algebras,R.restrictions,w))
    end
    _cup_report(issues,throw)
end
"""Recompute the declared product-image spaces and verify their spans and
embedding squares. This checks a hand-built or retained module, not just ranks."""
function check_cup_module(V::CupModule;throw=false,budget=CupProductBudget())
    issues=copy(check_persistent_cup_products(V.ring;budget).issues)
    append!(issues,check_module(V.pmodule).issues)
    n=length(V.ring.parameters)
    if nvertices(V.pmodule.Q)!=n || any(leq(V.pmodule.Q,i,j)!=(i>=j) for i in 1:n,j in 1:n)
        push!(issues,"cup module must use the opposite stage chain")
    end
    V.pmodule.field==field(V.ring) || push!(issues,"cup module and ring fields disagree")
    V.power>0 || push!(issues,"cup factors must be positive")
    V.partition===nothing || (!isempty(V.partition) && all(>(0),V.partition) && length(V.partition)==V.power) ||
        push!(issues,"invalid degree partition")
    V.degree===nothing || (!isempty(V.ring.algebras) && 0<V.degree<=_ring_degree(V.ring)) || push!(issues,"invalid total degree")
    if isempty(issues)
        expected=_image_module(V.ring,V.power,V.partition,V.degree,budget)
        if length(V.images)!=length(expected.images) || dimensions(V)!=dimensions(expected)
            push!(issues,"cup module dimensions or stage count disagree")
        else
            for i in eachindex(V.images)
                if length(V.images[i])!=length(expected.images[i]) ||
                   any(size(A,1)!=size(B,1) for (A,B) in zip(V.images[i],expected.images[i]))
                    push!(issues,"cup image blocks have wrong shapes");break
                end
                A=cup_embedding(V;stage=i);B=cup_embedding(expected;stage=i)
                if size(A,2)!=size(B,2) || FieldLinAlg.rank(field(V),A)!=size(A,2) ||
                   FieldLinAlg.rank(field(V),hcat(A,B))!=size(B,2)
                    push!(issues,"cup image is not the declared product span");break
                end
            end
            if isempty(issues)
                for i in 1:length(V.images)-1
                    source=V.images[i+1];target=V.images[i];r=0;c=0
                    M=structure_map(V;source=i+1,target=i)
                    for d in 0:_ring_degree(V.ring)
                        K=coeff_type(field(V))
                        A=d<length(target) ? target[d+1] : zeros(K,0,0)
                        B=d<length(source) ? source[d+1] : zeros(K,0,0)
                        nr=size(A,2);nc=size(B,2)
                        A*M[r+1:r+nr,c+1:c+nc]==V.ring.restrictions[i][d+1]*B ||
                            push!(issues,"cup restriction does not commute with its embedding")
                        r+=nr;c+=nc
                    end
                    # Off-degree blocks must vanish as well.
                    E=cup_embedding(V;stage=i);S=cup_embedding(V;stage=i+1)
                    diagonal=zeros(eltype(M),size(E,1),size(S,1));r=0;c=0
                    for d in 0:_ring_degree(V.ring)
                        F=V.ring.restrictions[i][d+1];nr,nc=size(F)
                        diagonal[r+1:r+nr,c+1:c+nc]=F;r+=nr;c+=nc
                    end
                    E*M==diagonal*S || push!(issues,"cup embedding square fails")
                end
            end
        end
    end
    _cup_report(issues,throw)
end
