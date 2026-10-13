# Bjerkevik, Definition 4.15 and Lemmas 5.2-5.3:
# https://doi.org/10.1007/s10208-025-09695-w
# Pointwise fixed points use a FULL Hom basis, not just the structure map.
"""Exact pruning (I/K)(-epsilon), with its maximal I, minimal K, inclusions and quotient projection."""
struct PruningResult{A,F,I,K,P}
    source::A
    refined::A
    shifted::A
    epsilon::QQ
    shift_map::F
    image::I
    kernel::K
    projection::P
    unshifted::A
    pruned::A
    hom_dimension::Int
    iterations::Tuple{Vector{Int},Vector{Int}}
end

function _limit(n, budget, label)
    n <= budget || throw(ArgumentError("Pruning $label budget exceeded ($n > $budget)."))
end

function _points(A)
    axes=map(encoding_axes(A)) do c
        isempty(c) && return QQ[0]
        A.domain === :integer && return [c[1]-1;c]
        v=QQ[c[1]-1]
        for i in eachindex(c)
            push!(v,c[i],i==length(c) ? c[i]+1 : (c[i]+c[i+1])/2)
        end
        v
    end
    vec(collect(Iterators.product(axes...)))
end

function _prepare(M,epsilon,budget)
    epsilon isa Union{Integer,Rational,AbstractFloat} && isfinite(epsilon) && epsilon>=0 ||
        throw(ArgumentError("epsilon must be a finite nonnegative scalar."))
    e=QQ(epsilon)
    shifted=parameter_shift(M,2e;budget)
    # This also enforces integral epsilon on lattice modules (the final shift).
    parameter_shift(M,-e;budget)
    cuts=map((a,b)->sort!(unique([a;b])),encoding_axes(M),encoding_axes(shifted))
    R=refine_parameters(M,cuts;budget)
    T=refine_parameters(shifted,cuts;budget)
    A=encoding_module(R); B0=encoding_module(T); K=coeff_type(A.field)
    _limit(sum(big(d)^2 for d in A.dims;init=big(0)),budget.max_matrix_entries,"subspace bases")
    B=PModule{K}(A.Q,copy(B0.dims),Dict(e=>Matrix{K}(B0.edge_maps[e...]) for e in cover_edges(A.Q));field=A.field)
    f=PMorphism(A,B,Matrix{K}[structure_map(M;source=x,target=x.+2e) for x in _points(R)])
    check_morphism(f;throw=true)
    variables=sum(big(a)*b for (a,b) in zip(A.dims,B.dims);init=big(0))
    _limit(variables,budget.max_hom_variables,"Hom variables")
    _limit(variables^2,budget.max_matrix_entries,"Hom workspace")
    H=basis(Hom(A,B))
    return R,T,f,H,e
end

# Columns in the domain whose image lies in col(B), including ker(A).
function _preimage(field,A,B)
    N=FieldLinAlg.nullspace(field,hcat(A,-B))
    return Matrix(FieldLinAlg.colspace(field,N[1:size(A,2),:]))
end

function _pair(field,t,fs)
    K=eltype(t); n=size(t,2)
    I=eye(field,n); icount=0
    for iteration in 1:n+1
        image=FieldLinAlg.colspace(field,t*I)
        J=I
        for f in fs
            J=J*_preimage(field,f*J,image)
        end
        size(J,2)==size(I,2) && break
        I=J;icount+=1
    end
    tI=t*I
    kernel=zeros(K,size(I,2),0);kcount=0
    for iteration in 1:size(I,2)+1
        # f(K) is contained in im(t|I). On this image, inverse image
        # commutes with sums; hence a Hom basis suffices also for K.
        images=zeros(K,size(t,1),0)
        for f in fs
            images=hcat(images,f*I*kernel)
        end
        J=_preimage(field,tI,FieldLinAlg.colspace(field,images))
        size(J,2)==size(kernel,2) && break
        kernel=J;kcount+=1
    end
    return I,I*kernel,icount,kcount
end

function _subobject(M::PModule{K},columns) where K
    maps=Dict{Tuple{Int,Int},Matrix{K}}()
    for (u,v) in cover_edges(M.Q)
        maps[u,v]=FieldLinAlg.solve_fullcolumn(M.field,columns[v],M.edge_maps[u,v]*columns[u])
    end
    S=PModule{K}(M.Q,[size(B,2) for B in columns],maps;field=M.field)
    inclusion=PMorphism(S,M,columns)
    check_morphism(inclusion;throw=true)
    return submodule(inclusion)
end

"""Construct the paper-defined (I/K)(-epsilon), retaining both inclusions and the quotient map."""
function epsilon_pruning(M::RectilinearModule,epsilon;budget=InterleavingBudget())
    R,T,t,H,e=_prepare(M,epsilon,budget)
    A=encoding_module(R);K=coeff_type(A.field)
    pairs=[_pair(A.field,t.comps[v],[f.comps[v] for f in H]) for v in eachindex(A.dims)]
    Is=Matrix{K}[p[1] for p in pairs]; Ks=Matrix{K}[p[2] for p in pairs]
    I=_subobject(A,Is)
    Kcoords=Matrix{K}[FieldLinAlg.solve_fullcolumn(A.field,Is[v],Ks[v]) for v in eachindex(Is)]
    kernel=_subobject(submodule_object(I),Kcoords)
    Q,q=quotient_with_projection(kernel)
    U=RectilinearModule(Q,encoding_axes(R);domain=M.domain,budget,
        origin=(;kind=:pruning_subquotient,epsilon=e,source=provenance(M)))
    P=parameter_shift(U,-e;budget)
    source=RectilinearModule(encoding_module(M),encoding_axes(M);domain=M.domain,budget,origin=M.origin)
    return PruningResult{typeof(source),typeof(t),typeof(I),typeof(kernel),typeof(q)}(
        source,R,T,e,t,I,kernel,q,U,P,length(H),([p[3] for p in pairs],[p[4] for p in pairs]))
end
function epsilon_pruning(M::EncodingResult,e;extension=:none,budget=InterleavingBudget())
    epsilon_pruning(ambient_module(M;extension,budget),e;budget)
end
epsilon_pruning(::PModule,e;kwargs...)=throw(ArgumentError("A bare finite poset has no ambient epsilon or inverse translation. Supply a global RectilinearModule or supported EncodingResult."))

pruned_module(r::PruningResult)=r.pruned
pruning_pair(r::PruningResult)=(;image=r.image,kernel=r.kernel)
pruning_maps(r::PruningResult)=(;inclusion=inclusion_map(r.image),
    kernel_inclusion=inclusion_map(r.kernel),projection=r.projection,shift=r.shift_map)
encoding_module(r::PruningResult)=encoding_module(r.pruned)
provenance(r::PruningResult)=(;kind=:epsilon_pruning,epsilon=r.epsilon,
    source=provenance(r.source),definition=:bjerkevik_4_15,exact=true,
    final_translation=-r.epsilon,hom_dimension=r.hom_dimension)
pruning_summary(r::PruningResult)=(;kind=:epsilon_pruning,epsilon=r.epsilon,
    max_stalk_dimension=maximum(encoding_module(r.source).dims;init=0),
    hom_dimension=r.hom_dimension,vertices=nvertices(encoding_module(r.pruned).Q),
    image_dimensions=copy(submodule_object(r.image).dims),
    kernel_dimensions=copy(submodule_object(r.kernel).dims),
    quotient_dimensions=copy(encoding_module(r.unshifted).dims),provenance=provenance(r))
describe(r::PruningResult)=pruning_summary(r)
Base.show(io::IO,r::PruningResult)=print(io,"PruningResult(epsilon=",r.epsilon,", Hom dimension=",r.hom_dimension,")")
