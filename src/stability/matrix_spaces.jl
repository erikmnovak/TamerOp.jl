# Exact weighted subrepresentation optimization. Cheng's path-matrix reduction
# followed by a certified Wong sequence on tensor blowups (Fersztand--Jendrysiak,
# arXiv:2603.23560, Section 4). Commuting poset paths need only one structure map.
# Randomness proposes a matrix; containment, descent and optimality are exact.
struct _WeightSpace{K,F}
    field::F
    source_dims::Vector{Int}
    target_dims::Vector{Int}
    source_copies::Vector{Int}
    target_copies::Vector{Int}
    maps::Vector{Tuple{Int,Int,Matrix{K}}}
end
function _block_ranges(dims,copies,p)
    offset=0;out=Vector{UnitRange{Int}}[]
    for (d,c) in zip(dims,copies)
        push!(out,[offset+(i-1)*d+1:offset+i*d for i in 1:c*p])
        offset+=d*c*p
    end
    out
end
function _space_image(S::_WeightSpace{K},p,U,cols,rows) where K
    inputs=[_basis(S.field,reduce(hcat,(U[r,:] for r in rs);init=zeros(K,d,0)))
        for (d,rs) in zip(S.source_dims,cols)]
    images=[zeros(K,d,0) for d in S.target_dims]
    for (i,j,A) in S.maps
        images[j]=_basis(S.field,hcat(images[j],A*inputs[i]))
    end
    nr=sum(S.target_dims.*S.target_copies)*p
    W=zeros(K,nr,sum(size(A,2)*length(rs) for (A,rs) in zip(images,rows);init=0))
    at=0
    for (A,rs) in zip(images,rows),r in rs
        k=size(A,2);W[r,at+1:at+k]=A;at+=k
    end
    W
end
function _random_matrix(S::_WeightSpace{K},p,cols,rows,rng) where K
    nr=sum(S.target_dims.*S.target_copies)*p
    nc=sum(S.source_dims.*S.source_copies)*p
    A=zeros(K,nr,nc)
    # A local reproducible stream does not change the caller's random state.
    span=S.field isa PrimeField ? (0:S.field.p-1) : (-4max(nr,nc,2):4max(nr,nc,2))
    for (i,j,B) in S.maps,r in rows[j],c in cols[i]
        a=K(rand(rng,span));iszero(a) && continue
        A[r,c].+=a.*B
    end
    A
end
function _wong_certificate(S,p,A,cols,rows,b,w)
    F=S.field;K=coeff_type(F);nc=size(A,2)
    R,ps=FL.rref(F,A);pivots=collect(Int,ps)
    length(pivots)==nc && return zeros(K,nc,0),0
    free=setdiff(1:nc,pivots)
    kernel=zeros(K,nc,length(free))
    kernel[free,:]=eye(F,length(free))
    kernel[pivots,:]=-R[1:length(pivots),free]
    image=A[:,pivots];U=kernel;last_dimension=0
    while true
        _tick!(w,b)
        W=_space_image(S,p,U,cols,rows)
        # Failure is inconclusive. In particular, a singular random matrix is
        # never itself evidence that the module is unstable.
        _contains(F,image,W) || return nothing
        if size(W,2)==last_dimension
            size(U,2)-size(W,2)==nc-length(pivots) || error("invalid Wong defect certificate")
            return U,nc-length(pivots)
        end
        last_dimension=size(W,2)
        X=FL.solve_fullcolumn(F,image,W;cache=false)
        lift=zeros(K,nc,size(W,2));lift[pivots,:]=X
        U=hcat(kernel,lift)
    end
end
# Regular representation of F_(q^g) in g-by-g matrices over F_q. A monic
# irreducible polynomial is proposed from the local RNG and tested exactly.
# This does not add extension-field coefficients to the public module API.
function _extension_basis(F::PrimeField,g,b,w)
    g>=1 || throw(ArgumentError("extension degree must be positive"))
    K=coeff_type(F);g==1 && return [ones(K,1,1)]
    _bound(g,b.max_extension_degree,"extension degree")
    _bound(big(g)^3,b.max_entries,"extension basis entries")
    R,_=Nemo.polynomial_ring(Nemo.GF(F.p),:t;cached=false)
    coefficients=Int[]
    while true
        _tick!(w,b)
        coefficients=vcat([rand(w.rng,0:F.p-1) for _ in 1:g],1)
        iszero(first(coefficients)) && continue
        Nemo.is_irreducible(R(coefficients)) && break
    end
    C=zeros(K,g,g)
    for i in 1:g-1;C[i+1,i]=one(K);end
    C[:,g]=-K.(coefficients[1:g])
    powers=[eye(F,g)]
    for _ in 2:g;push!(powers,last(powers)*C);end
    powers
end
function _extension_degree(F,p,dimension)
    # A nonzero maximal minor has degree at most p*dimension. Sampling from
    # at least eight times that many elements bounds its vanishing probability
    # by 1/8, conditional on this tensor size attaining the optimal rank.
    target=8big(p)*dimension;g=1;order=big(F.p)
    while order<target;g+=1;order*=F.p;end
    g
end
function _extension_matrix(S::_WeightSpace{K},p,powers,cols,rows,rng) where K
    g=length(powers);q=S.field.p
    nr=sum(S.target_dims.*S.target_copies)*p*g
    nc=sum(S.source_dims.*S.source_copies)*p*g
    A=zeros(K,nr,nc)
    for (i,j,B) in S.maps,r in 1:g:length(rows[j]),c in 1:g:length(cols[i])
        coefficient=zeros(K,g,g)
        for P in powers;coefficient .+=K(rand(rng,0:q-1)).*P;end
        for u in 1:g,v in 1:g
            value=coefficient[u,v];iszero(value) && continue
            A[rows[j][r+u-1],cols[i][c+v-1]].+=value.*B
        end
    end
    A
end
function _minimum_shrunk(S::_WeightSpace,b,w)
    base_rows=sum(big.(S.target_dims).*S.target_copies)
    base_cols=sum(big.(S.source_dims).*S.source_copies)
    dimension=max(base_rows,base_cols,1)
    # Four cheap base-field trials precede extension sampling. Each subsequent
    # tensor size has four independent proposals; the global trial budget wins.
    sizes=S.field isa PrimeField ? Iterators.flatten((((1,1),),((p,_extension_degree(S.field,p,dimension)) for p in 1:b.max_blowup))) :
        ((p,1) for p in 1:b.max_blowup)
    seen=Set{Tuple{Int,Int}}()
    for (p,g) in sizes
        (p,g) in seen && continue;push!(seen,(p,g))
        big(p)*g<=b.max_blowup || break
        total=p*g
        g==1 || _bound(g,b.max_extension_degree,"extension degree")
        _bound(6*(dimension*total)^2+big(g)^3,b.max_entries,"matrix-space workspace entries")
        rows=_block_ranges(S.target_dims,S.target_copies,total)
        cols=_block_ranges(S.source_dims,S.source_copies,total)
        _bound(w.matrix_trials+1,b.max_trials,"certified matrix trials")
        powers=g>1 ? _extension_basis(S.field,g,b,w) : nothing
        for _ in 1:4
            _bound(w.matrix_trials+1,b.max_trials,"certified matrix trials")
            _tick!(w,b)
            w.matrix_trials+=1;w.largest_blowup=max(w.largest_blowup,total)
            w.largest_matrix=max(w.largest_matrix,Int(dimension*total))
            w.largest_extension=max(w.largest_extension,g)
            g>1 && (w.extension_trials+=1)
            A=g>1 ? _extension_matrix(S,p,powers,cols,rows,w.rng) : _random_matrix(S,p,cols,rows,w.rng)
            result=_wong_certificate(S,total,A,cols,rows,b,w)
            result===nothing && continue
            U,defect=result
            bases=[_basis(S.field,reduce(hcat,(U[r,:] for r in rs);
                init=zeros(eltype(U),d,0))) for (d,rs) in zip(S.source_dims,cols)]
            sum(size(B,2)*length(rs) for (B,rs) in zip(bases,cols);init=0)==size(U,2) ||
                error("Wong certificate did not descend to the original matrix space")
            rem(defect,total)==0 || error("nonintegral descended defect")
            return bases,div(defect,total)
        end
    end
    throw(ArgumentError("HN tensor/extension budget exhausted without an optimality certificate; increase max_blowup/max_trials"))
end
function _integer_weights(theta,dims)
    denominators=[denominator(theta[i]) for i in eachindex(dims) if dims[i]>0]
    scale=foldl(lcm,denominators;init=big(1))
    weights=[dims[i]==0 ? big(0) : numerator(theta[i]*scale) for i in eachindex(dims)]
    divisor=foldl(gcd,weights;init=big(0))
    iszero(divisor) ? weights : div.(weights,divisor)
end
# Thin stalks reduce to maximum-weight closure. Exact integral residual
# capacities avoid weight replication entirely, including very large weights.
function _thin_weight_submodule(M::PModule{K},weights,b,w) where K
    n=length(weights);source=n+1;sink=n+2
    adjacency=[Int[] for _ in 1:n+2];targets=Int[];capacity=BigInt[]
    function edge(u,v,c)
        push!(targets,v);push!(capacity,c);push!(adjacency[u],length(targets))
        push!(targets,u);push!(capacity,big(0));push!(adjacency[v],length(targets))
    end
    total=sum(max(x,0) for x in weights;init=big(0))
    for v in 1:n
        weights[v]>0 && edge(source,v,weights[v])
        weights[v]<0 && edge(v,sink,-weights[v])
    end
    for ((u,v),A) in M.edge_maps
        iszero(A) || edge(u,v,total+1)
    end
    _bound(3*length(targets)+3*(n+2),b.max_entries,"closure workspace entries")
    flow=big(0);parent=zeros(Int,n+2)
    while true
        _tick!(w,b);fill!(parent,0);parent[source]=-1;queue=[source];at=1
        while at<=length(queue)
            u=queue[at];at+=1
            for e in adjacency[u]
                v=targets[e]
                capacity[e]>0 && parent[v]==0 || continue
                parent[v]=e;push!(queue,v)
            end
        end
        parent[sink]==0 && break
        value=total+1;v=sink
        while v!=source
            e=parent[v];value=min(value,capacity[e]);v=targets[isodd(e) ? e+1 : e-1]
        end
        v=sink
        while v!=source
            e=parent[v];reverse=isodd(e) ? e+1 : e-1
            capacity[e]-=value;capacity[reverse]+=value;v=targets[reverse]
        end
        flow+=value
    end
    [parent[v]!=0 ? eye(M.field,M.dims[v]) : zeros(K,M.dims[v],0) for v in 1:n],total-flow
end
function _maximum_weight_submodule(M::PModule{K},theta,b,w) where K
    weights=_integer_weights(theta,M.dims)
    all(<=(1),M.dims) && return _thin_weight_submodule(M,weights,b,w)
    pos=findall(>(0),weights);neg=findall(<(0),weights)
    bases=[zeros(K,d,0) for d in M.dims]
    isempty(pos) && return bases,zero(BigInt)
    total=max(sum(weights[v]*M.dims[v] for v in pos;init=big(0)),
        -sum(weights[v]*M.dims[v] for v in neg;init=big(0)))
    maps=Tuple{Int,Int,Matrix{K}}[];entries=big(0)
    for (i,u) in enumerate(pos),(j,v) in enumerate(neg)
        leq(M.Q,u,v) || continue
        entries+=big(M.dims[u])*M.dims[v]
        _bound(entries,b.max_entries,"weighted path-map entries")
        A=Matrix{K}(map_leq(M,u,v));iszero(A) || push!(maps,(i,j,A))
    end
    chosen,defect=if isempty(maps)
        _tick!(w,b)
        [eye(M.field,M.dims[v]) for v in pos],sum(weights[v]*M.dims[v] for v in pos)
    elseif length(unique(first.(maps)))==length(maps)==length(unique(getindex.(maps,2)))
        # Independent one-map pieces have a closed-form minimum optimizer:
        # alpha*dim(U)-beta*rank(AU) = alpha*dim(U intersect ker(A))
        #                              +(alpha-beta)*rank(AU).
        # Use ker(A) when alpha<=beta (including the minimum tie), otherwise
        # the entire source. This avoids unlikely simultaneous random ranks
        # on many disconnected pieces, as well as large weight replication.
        _tick!(w,b)
        single_bases=[eye(M.field,M.dims[v]) for v in pos]
        for (i,j,A) in maps
            weights[pos[i]]<=-weights[neg[j]] && (single_bases[i]=Matrix{K}(FL.nullspace(M.field,A)))
        end
        value=sum(weights[pos[i]]*size(single_bases[i],2) for i in eachindex(pos))-
            sum(-weights[neg[j]]*_rank(M.field,A*single_bases[i]) for (i,j,A) in maps)
        single_bases,value
    else
        _bound(6*total^2,b.max_entries,"matrix-space workspace entries")
        S=_WeightSpace(M.field,M.dims[pos],M.dims[neg],Int.(weights[pos]),Int.(-weights[neg]),maps)
        _minimum_shrunk(S,b,w)
    end
    for (i,v) in enumerate(pos);bases[v]=chosen[i];end
    # Generate the least subrepresentation containing the positive stalks.
    order=sort(collect(eachindex(M.dims));by=v->count(u->leq(M.Q,u,v),eachindex(M.dims)))
    for v in order,u in M.edge_maps.preds[v]
        bases[v]=_basis(M.field,hcat(bases[v],M.edge_maps[u,v]*bases[u]))
    end
    sum(weights[v]*size(bases[v],2) for v in eachindex(weights))==defect ||
        error("subrepresentation does not realize the certified optimum")
    bases,defect
end
function _highest_general(M,z,b,w)
    C=M;inclusions=[eye(M.field,d) for d in M.dims]
    while true
        mass=_mass(z,C.dims);degree=_degree(z,C.dims)
        theta=mass.*z.degree.-degree.*z.mass
        B,defect=_maximum_weight_submodule(C,theta,b,w)
        iszero(defect) && return inclusions,degree/mass
        N,_=_subobject(C,B)
        0<sum(N.dims)<sum(C.dims) || error("destabilizing subobject did not decrease dimension")
        inclusions=[inclusions[v]*B[v] for v in eachindex(B)]
        C=N
    end
end
