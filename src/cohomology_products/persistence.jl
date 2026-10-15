# Product-image modules, cup length, and chain intervals.

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
Use `cup_module`, `partition_module`, `cup_embedding`, `cup_pmodule`,
`dimensions`, and `structure_map`. Vertex labels agree with the ring diagram;
maps follow the opposite indexing order. Scalar chains also have `cup_intervals`."""
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
"""Return the exact image PModule on the opposite indexing poset (an owned copy)."""
cup_pmodule(V::CupModule)=deepcopy(V.pmodule)
cup_parameters(V::CupModule)=cup_parameters(V.ring)
cup_poset(V::CupModule)=cup_poset(V.ring)
"""Actual image-submodule map from later source to earlier target stage."""
function structure_map(V::CupModule;source,target)
    s=_stage(V.ring,source);t=_stage(V.ring,target)
    leq(V.ring.poset,t,s) || throw(ArgumentError("cup maps require target <= source in the indexing poset"))
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
    dims=[sum(size(B,2) for B in bs) for bs in images];maps=Dict{Tuple{Int,Int},Matrix{K}}()
    for ((j,i),restriction) in R.restrictions
        _spend!(w;entries=big(dims[i])*dims[j])
        M=zeros(K,dims[i],dims[j]);r=0;c=0
        for d in 0:D
            B=d<length(images[i]) ? images[i][d+1] : zeros(K,0,0)
            C=d<length(images[j]) ? images[j][d+1] : zeros(K,0,0)
            nr=size(B,2);nc=size(C,2)
            M[r+1:r+nr,c+1:c+nc]=FieldLinAlg.solve_fullcolumn(field(R),B,restriction[d+1]*C;cache=false)
            r+=nr;c+=nc
        end
        maps[j,i]=M
    end
    _spend!(w;entries=big(length(dims))^2)
    CupModule(R,images,_diagram_module(R.poset,field(R),dims,maps),power,parts,degree)
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
    _parameter_chain(V.ring) || throw(ArgumentError("intervals require a scalar-parameter chain; use cup_pmodule for a general finite-poset result"))
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
    cup_length(R; source=nothing, target=source, budget=CupProductBudget())

Cup-length of the image ring of the restriction from source to target stage.
With equal stages this is the pointwise cup-length. In general it is not the
minimum of the endpoint cup-lengths: products must survive the actual map.
An omitted source selects the last event of a scalar chain. For a general
finite poset, supply the source vertex explicitly.
"""
function cup_length(R::PersistentCupProducts{K};source=nothing,target=source,
                    budget=CupProductBudget()) where {K}
    if source===nothing
        _parameter_chain(R) || throw(ArgumentError("choose a source vertex for cup length on a general finite poset"))
        source=length(R.algebras)
    end
    target===nothing && (target=source)
    s=_stage(R,source);t=_stage(R,target);leq(R.poset,t,s) || throw(ArgumentError("cup length requires target <= source in the indexing poset"))
    A=R.algebras[t];w=_CupWork(budget)
    _spend!(w;entries=sum(big(_hdim(A,d))*_hdim(R.algebras[s],d) for d in 0:length(A.dims)-1))
    generators=[d==0 ? zeros(K,_hdim(A,0),0) : Matrix{K}(FieldLinAlg.colspace(field(R),
        cohomology_restriction(R;source=s,target=t,degree=d))) for d in 0:length(A.dims)-1]
    _cup_length(A,w;generators)
end

provenance(R::PersistentCupProducts)=(;source=R.origin,field=field(R),variance=:contravariant,
    indexing=_parameter_chain(R) ? :opposite_filtration_chain : :opposite_finite_poset,
    order=R.order,continuation=_parameter_chain(R) ? :zero_before_constant_after : :finite_base,
    arithmetic=:exact,product_convention=:positive_degree_factors,
    basis_choice=:noncanonical,geometry=R.origin===:supplied_algebra_diagram ? :not_asserted : :abstract_simplicial)
describe(R::PersistentCupProducts)=(;kind=:persistent_cup_products,stages=length(R.algebras),
    parameters=R.parameters===nothing ? nothing : Tuple(R.parameters),
    dimensions=Tuple(Tuple(A.dims) for A in R.algebras),provenance=provenance(R))
describe(V::CupModule)=(;kind=:cup_module,factors=V.power,partition=V.partition===nothing ? nothing : Tuple(V.partition),
    degree=V.degree,dimensions=Tuple(dimensions(V)),provenance=provenance(V.ring))
"""Cheap dimensions, factor/partition choices, and mathematical provenance."""
cup_product_summary(x::Union{CohomologyAlgebra,PersistentCupProducts,CupModule})=describe(x)
Base.show(io::IO,R::PersistentCupProducts)=print(io,"PersistentCupProducts(stages=",length(R.algebras),", field=",field(R),", order=",R.order,")")
Base.show(io::IO,V::CupModule)=print(io,"CupModule(factors=",V.power,", partition=",V.partition,", dimensions=",Tuple(dimensions(V)),")")

"""Recompute the declared product-image spaces and verify their spans and
embedding squares. This checks a hand-built or retained module, not just ranks."""
function check_cup_module(V::CupModule;throw=false,budget=CupProductBudget())
    issues=copy(check_persistent_cup_products(V.ring;budget).issues)
    append!(issues,check_module(V.pmodule).issues)
    n=length(V.ring.algebras)
    if nvertices(V.pmodule.Q)!=n || any(leq(V.pmodule.Q,i,j)!=leq(V.ring.poset,j,i) for i in 1:n,j in 1:n)
        push!(issues,"cup module must use the opposite indexing poset")
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
                for ((j,i),restriction) in V.ring.restrictions
                    source=V.images[j];target=V.images[i];r=0;c=0
                    M=structure_map(V;source=j,target=i)
                    for d in 0:_ring_degree(V.ring)
                        K=coeff_type(field(V))
                        A=d<length(target) ? target[d+1] : zeros(K,0,0)
                        B=d<length(source) ? source[d+1] : zeros(K,0,0)
                        nr=size(A,2);nc=size(B,2)
                        A*M[r+1:r+nr,c+1:c+nc]==restriction[d+1]*B ||
                            push!(issues,"cup restriction does not commute with its embedding")
                        r+=nr;c+=nc
                    end
                    # Off-degree blocks must vanish as well.
                    E=cup_embedding(V;stage=i);S=cup_embedding(V;stage=j)
                    diagonal=zeros(eltype(M),size(E,1),size(S,1));r=0;c=0
                    for d in 0:_ring_degree(V.ring)
                        F=restriction[d+1];nr,nc=size(F)
                        diagonal[r+1:r+nr,c+1:c+nc]=F;r+=nr;c+=nc
                    end
                    E*M==diagonal*S || push!(issues,"cup embedding square fails")
                end
            end
        end
    end
    _cup_report(issues,throw)
end
