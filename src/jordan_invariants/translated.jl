# The ambient source supplies all exterior strata. Translating a finite pattern
# preserves its points as distinct summands even when their encoding labels agree.
"""A lazy family of images of powers of the translated Jordan operator.
Use `jordan_filtered_rank` for selected ranks and `jordan_module` to construct
an entire degree of the family on its exact finite encoding.
"""
struct JordanFiltration{N,M}
    source::M
    slices::Vector{Vector{NTuple{N,QQ}}}
end

function _ambient_budget(b)
    InterleavingBudget(max_vertices=max(1,b.max_cells),max_matrix_entries=max(1,b.max_entries))
end
function _parameter(x,n,domain)
    x isa Union{Tuple,AbstractVector} && length(x)==n ||
        throw(ArgumentError("Supply a parameter tuple/vector of length $n"))
    all(t->t isa Union{Integer,Rational,AbstractFloat} && !(t isa Bool) && isfinite(t),x) ||
        throw(ArgumentError("Parameters must be finite real coordinates"))
    y=Tuple(QQ.(x))
    domain===:integer && !all(isinteger,y) && throw(ArgumentError("Integer-domain parameters must be integral"))
    y
end
function _pattern(slices,n,domain,b)
    slices isa Union{Tuple,AbstractVector} && 2<=length(slices)<=b.max_slices ||
        throw(ArgumentError("Supply between 2 and max_slices ordered finite subsets"))
    total=sum(big(length(s)) for s in slices if s isa Union{Tuple,AbstractVector,AbstractSet};init=big(0))
    _bound(total,b.max_vertices,"pattern points")
    result=Vector{NTuple{n,QQ}}[];seen=Set{NTuple{n,QQ}}()
    for s in slices
        s isa Union{Tuple,AbstractVector,AbstractSet} || throw(ArgumentError("Each slice must be a finite collection of coordinate tuples"))
        points=sort!([_parameter(x,n,domain) for x in s])
        for x in points
            x in seen && throw(ArgumentError("Geometric slices must be pairwise disjoint without repeated points"))
            push!(seen,x)
        end
        push!(result,points)
    end
    result
end
"""
    jordan_filtration(M; slices, budget=JordanBudget())
    jordan_filtration(enc::EncodingResult; slices, extension=:none, budget=JordanBudget())

Form the family J^i(x) = im(T(M[x], slices)^i) on R^d or Z^d, for a global
`RectilinearModule` or a supported ambient encoding. Slices contain geometric
coordinate tuples, not finite-poset labels, and remain disjoint before encoding.
Grid encodings require an explicit extension declaration. No translation grid
is materialized here. The source and pattern are copied into the result.
"""
function jordan_filtration(M::RectilinearModule{N};slices,budget::JordanBudget=JordanBudget()) where N
    pattern=_pattern(slices,N,M.domain,budget)
    owned=RectilinearModule(encoding_module(M),encoding_axes(M);domain=M.domain,
        origin=provenance(M).origin,budget=_ambient_budget(budget))
    JordanFiltration{N,typeof(owned)}(owned,pattern)
end
function jordan_filtration(enc::EncodingResult;slices,extension=:none,budget::JordanBudget=JordanBudget())
    jordan_filtration(ambient_module(enc;extension,budget=_ambient_budget(budget));slices,budget)
end
function jordan_filtration(::PModule;kwargs...)
    throw(ArgumentError("A bare finite poset supplies no ambient translations; supply an encoding with declared continuation or a RectilinearModule"))
end
field(f::JordanFiltration)=encoding_module(f.source).field
jordan_slices(f::JordanFiltration)=map(copy,f.slices)
provenance(f::JordanFiltration)=(;construction=:translated_jordan_images,translated=true,
    domain=f.source.domain,parameters=length(f.source.cuts),slices=jordan_slices(f),
    source=provenance(f.source),scope=:global,coefficients=field(f))
describe(f::JordanFiltration)=(;kind=:jordan_filtration,slices=length(f.slices),
    points=sum(length,f.slices),powers=0:length(f.slices),provenance=provenance(f))
jordan_summary(f::JordanFiltration)=describe(f)
Base.show(io::IO,f::JordanFiltration)=print(io,"JordanFiltration(",length(f.slices)," slices, ",length(f.source.cuts)," parameters, ",f.source.domain,")")
"""Check geometric disjointness, ambient source functoriality and size limits."""
function check_jordan_filtration(f::JordanFiltration;budget::JordanBudget=JordanBudget(),throw=false)
    errors=String[]
    try
        _pattern(f.slices,length(f.source.cuts),f.source.domain,budget)
        RectilinearModule(encoding_module(f.source),encoding_axes(f.source);domain=f.source.domain,budget=_ambient_budget(budget))
    catch e
        e isa InterruptException && rethrow()
        push!(errors,sprint(showerror,e))
    end
    _report(errors,throw)
end
function _power(f,p)
    p isa Integer && !(p isa Bool) && 0<=p<=length(f.slices) ||
        throw(ArgumentError("power must be an integer from 0 through the number of slices"))
    Int(p)
end
function _at(f,x)
    _parameter(x,length(f.source.cuts),f.source.domain)
end
function _stalk_data(f,x,b)
    M=encoding_module(f.source)
    ids=[[locate(f.source,x.+z) for z in s] for s in f.slices]
    ds=[M.dims[s] for s in ids];dims=[sum(big,d;init=big(0)) for d in ds]
    _bound(sum(length,ids),b.max_vertices,"pattern points")
    _bound(length(ids),b.max_slices,"slices")
    _bound(sum(dims;init=big(0)),b.max_dimension,"translated dimension")
    n=length(dims)
    adjacent=sum(dims[i]*dims[i+1] for i in 1:n-1;init=big(0))
    largest=maximum((dims[i]*dims[j] for i in 1:n for j in i+1:n);init=big(0))
    _bound(adjacent+2largest,b.max_entries,"translated block/product entries")
    _bound(div(big(n)*(n-1),2),b.max_products,"translated products")
    (;ids,ds,dims=Int.(dims))
end
function _translated_blocks(f,data)
    M=encoding_module(f.source);K=coeff_type(field(f));blocks=Matrix{K}[]
    for i in 1:length(f.slices)-1
        A=zeros(K,data.dims[i+1],data.dims[i]);col=0
        for (j,u) in enumerate(f.slices[i])
            row=0
            for (k,v) in enumerate(f.slices[i+1])
                # Encoding can identify distinct comparable points. Their map
                # is then the identity, not zero and not a duplicate-label error.
                if all(u .<= v)
                    A[row+1:row+data.ds[i+1][k],col+1:col+data.ds[i][j]]=map_leq(M,data.ids[i][j],data.ids[i+1][k])
                end
                row+=data.ds[i+1][k]
            end
            col+=data.ds[i][j]
        end
        push!(blocks,A)
    end
    blocks
end
function _diagonal(maps,K)
    A=zeros(K,sum(size(m,1) for m in maps;init=0),sum(size(m,2) for m in maps;init=0))
    row=0;col=0
    for m in maps
        A[row+1:row+size(m,1),col+1:col+size(m,2)]=m
        row+=size(m,1);col+=size(m,2)
    end
    A
end
function _slice_transports(f,a,b,budget)
    M=encoding_module(f.source);K=coeff_type(field(f))
    _bound(sum(big(a.dims[i])*b.dims[i] for i in eachindex(a.dims)),budget.max_entries,"transport entries")
    [_diagonal([map_leq(M,u,v) for (u,v) in zip(a.ids[i],b.ids[i])],K) for i in eachindex(f.slices)]
end
"""
    jordan_filtered_rank(f; source, target, power=nothing, budget=JordanBudget())

Exact ranks of J^i(source -> target). By default return powers 0 through n,
including the terminal zero. `power=i` returns one rank. Incomparable parameters
have value Inf, following the extended rank-invariant convention. Computation
uses consecutive subset products followed by the source-to-target maps; ranks
at the two endpoints alone do not determine this answer.
"""
function jordan_filtered_rank(f::JordanFiltration;source,target,power=nothing,budget::JordanBudget=JordanBudget())
    x=_at(f,source);y=_at(f,target);n=length(f.slices)
    p=power===nothing ? nothing : _power(f,power)
    all(x .<= y) || return p===nothing ? fill(Inf,n+1) : Inf
    a=_stalk_data(f,x,budget);b=_stalk_data(f,y,budget)
    transports=_slice_transports(f,a,b,budget)
    ranks=zeros(Int,n+1)
    if p===nothing || p==0
        ranks[1]=sum(FieldLinAlg.rank_dim(field(f),A) for A in transports;init=0)
    end
    if p===nothing || 0<p<n
        blocks=_translated_blocks(f,a)
        for first in 1:n-1
            product=blocks[first]
            for last in first:n-1
                last==first || (product=blocks[last]*product)
                exponent=last-first+1
                if p===nothing || exponent==p
                    _bound(big(size(transports[last+1],1))*size(product,2),budget.max_entries,"transported product entries")
                    ranks[exponent+1]+=FieldLinAlg.rank_dim(field(f),transports[last+1]*product)
                end
                p===nothing || exponent<p || break
            end
        end
    end
    p===nothing ? ranks : ranks[p+1]
end


"""Power ranks at one translation, including powers zero through n."""
jordan_power_ranks(f::JordanFiltration;at,budget::JordanBudget=JordanBudget()) =
    jordan_filtered_rank(f;source=at,target=at,budget)
"""Jordan block multiplicities at one translation, obtained from power ranks."""
jordan_multiplicities(f::JordanFiltration;at,budget::JordanBudget=JordanBudget()) =
    _multiplicities(jordan_power_ranks(f;at,budget))

"""Validate a translated rank query and its selected-stalk resource limits."""
function check_jordan_query(f::JordanFiltration;source,target,power=nothing,
        budget::JordanBudget=JordanBudget(),throw=false)
    errors=String[]
    try
        x=_at(f,source);y=_at(f,target)
        power===nothing || _power(f,power)
        _stalk_data(f,x,budget);_stalk_data(f,y,budget)
    catch e
        e isa InterruptException && rethrow()
        push!(errors,sprint(showerror,e))
    end
    _report(errors,throw)
end

# Each power has at most one block arriving in each slice. Keeping bases by
# slice avoids allocating the nilpotent operator on the entire direct sum.
function _image_data(f,x,p,b)
    data=_stalk_data(f,x,b);K=coeff_type(field(f));n=length(f.slices)
    bases=[zeros(K,d,0) for d in data.dims]
    if p==0
        _bound(sum(big(d)^2 for d in data.dims;init=big(0)),b.max_entries,"identity image bases")
        bases=[Matrix{K}(I,d,d) for d in data.dims]
    elseif p<n
        blocks=_translated_blocks(f,data)
        for first in 1:n-p
            product=blocks[first]
            for last in first+1:first+p-1;product=blocks[last]*product;end
            bases[first+p]=Matrix{K}(FieldLinAlg.colspace(field(f),product))
        end
    end
    (;data,bases)
end
"""Basis of im(T^power) at a translation, in slice/point/stalk order."""
function jordan_basis(f::JordanFiltration;at,power,budget::JordanBudget=JordanBudget())
    x=_at(f,at);p=_power(f,power);data=_image_data(f,x,p,budget)
    _bound(big(sum(data.data.dims))*sum(size(B,2) for B in data.bases),budget.max_entries,"image embedding")
    _diagonal(data.bases,coeff_type(field(f)))
end
function _image_map(f,a,b,transports,budget)
    K=coeff_type(field(f))
    ds=[size(A,2) for A in a.bases];dt=[size(B,2) for B in b.bases]
    _bound(big(sum(ds))*sum(dt),budget.max_entries,"Jordan structure map")
    maps=Matrix{K}[]
    for i in eachindex(ds)
        Y=transports[i]*a.bases[i]
        X=FieldLinAlg.solve_fullcolumn(field(f),b.bases[i],Y;cache=false)
        push!(maps,Matrix{K}(X))
    end
    _diagonal(maps,K)
end
"""
    jordan_structure_map(f; source, target, power, target_power=power)

Map between the image bases returned by `jordan_basis`. `target_power <= power`
combines parameter transport with the inclusion in the Jordan filtration.
"""
function jordan_structure_map(f::JordanFiltration;source,target,power,target_power=power,budget::JordanBudget=JordanBudget())
    x=_at(f,source);y=_at(f,target);p=_power(f,power);q=_power(f,target_power)
    all(x .<= y) && q<=p || throw(ArgumentError("Require source <= target and target_power <= power"))
    a=_image_data(f,x,p,budget);b=_image_data(f,y,q,budget)
    _image_map(f,a,b,_slice_transports(f,a.data,b.data,budget),budget)
end

"""All critical hyperplanes of the translated family, including exterior strata."""
function encoding_axes(f::JordanFiltration;budget::JordanBudget=JordanBudget())
    n=length(f.source.cuts);points=reduce(vcat,f.slices;init=NTuple{n,QQ}[])
    cuts=encoding_axes(f.source)
    _bound(sum(big(length(c))*length(points) for c in cuts;init=big(0)),budget.max_events,"translated cut candidates")
    ntuple(i->sort!(unique(QQ[c-z[i] for c in cuts[i] for z in points])),n)
end
function _representatives(c,domain)
    isempty(c) && return QQ[0]
    domain===:integer && return [c[1]-1;c]
    a=QQ[c[1]-1]
    for i in eachindex(c)
        push!(a,c[i],i==length(c) ? c[i]+1 : (c[i]+c[i+1])/2)
    end
    a
end
function _parameter_grid(c,domain,b)
    axes=map(x->_representatives(x,domain),c)
    _bound(prod(big(length(a)) for a in axes),b.max_cells,"parameter strata")
    Q=GridPoset(map(a->collect(1:length(a)),axes))
    Q,vec(collect(Iterators.product(axes...)))
end
"""One complete Jordan image module, optionally retaining its embedding bases."""
struct JordanModuleResult{F,M,K}
    filtration::F
    ambient::M
    power::Int
    bases::Union{Nothing,Vector{Matrix{K}}}
end
"""
    jordan_module(f; power, witness=false, budget=JordanBudget())

Construct the entire image module on the common refinement of every translated
source cut. Singleton strata, open cells and exterior stalks are retained for
real parameters; integer parameters use exact bins. `witness=true` retains
embeddings into the direct sum of translated source stalks.
"""
function jordan_module(f::JordanFiltration;power,witness=false,budget::JordanBudget=JordanBudget())
    p=_power(f,power);cuts=encoding_axes(f;budget);Q,points=_parameter_grid(cuts,f.source.domain,budget)
    _bound(big(length(points))*div(big(length(f.slices))*(length(f.slices)-1),2),budget.max_products,"family products")
    K=coeff_type(field(f))
    first_data=_image_data(f,first(points),p,budget)
    data=typeof(first_data)[];bases_entries=big(0)
    for (i,x) in enumerate(points)
        d=i==1 ? first_data : _image_data(f,x,p,budget)
        bases_entries+=sum(big(length(B)) for B in d.bases;init=big(0))
        _bound(bases_entries,budget.max_entries,"retained construction bases")
        push!(data,d)
    end
    dims=[sum(size(B,2) for B in d.bases) for d in data]
    entries=sum(big(dims[u])*dims[v] for (u,v) in cover_edges(Q);init=big(0))
    _bound(entries+bases_entries,budget.max_entries,"encoded module and bases")
    maps=Dict{Tuple{Int,Int},Matrix{K}}()
    for (u,v) in cover_edges(Q)
        maps[u,v]=_image_map(f,data[u],data[v],_slice_transports(f,data[u].data,data[v].data,budget),budget)
    end
    M=PModule{K}(Q,dims,maps;field=field(f))
    ambient=RectilinearModule(M,cuts;domain=f.source.domain,budget=_ambient_budget(budget),
        origin=(;kind=:jordan_image,power=p,slices=jordan_slices(f),source=provenance(f.source)))
    if witness
        _bound(entries+sum(big(sum(d.data.dims))*dims[i] for (i,d) in enumerate(data);init=big(0)),budget.max_entries,"retained image embeddings")
    end
    bases=witness ? [_diagonal(d.bases,K) for d in data] : nothing
    JordanModuleResult{typeof(f),typeof(ambient),K}(f,ambient,p,bases)
end
ambient_module(r::JordanModuleResult)=r.ambient
encoding_module(r::JordanModuleResult)=encoding_module(r.ambient)
encoding_axes(r::JordanModuleResult)=encoding_axes(r.ambient)
structure_map(r::JordanModuleResult;source,target)=structure_map(r.ambient;source,target)
field(r::JordanModuleResult)=field(r.filtration)
dimensions(r::JordanModuleResult)=copy(encoding_module(r).dims)
provenance(r::JordanModuleResult)=(;provenance(r.filtration)...,power=r.power,representation=:complete_finite_encoding)
describe(r::JordanModuleResult)=(;kind=:jordan_module,power=r.power,vertices=nvertices(encoding_module(r).Q),witness=r.bases!==nothing,provenance=provenance(r))
jordan_summary(r::JordanModuleResult)=describe(r)
Base.show(io::IO,r::JordanModuleResult)=print(io,"JordanModuleResult(power=",r.power,", ",nvertices(encoding_module(r).Q)," strata)")
function jordan_basis(r::JordanModuleResult;at)
    r.bases===nothing && throw(ArgumentError("Request witness=true to retain image bases"))
    copy(r.bases[locate(r.ambient,at)])
end
"""Reconstruct and check the encoded image module and any retained embeddings."""
function check_jordan_module(r::JordanModuleResult;budget::JordanBudget=JordanBudget(),throw=false)
    errors=String[]
    try
        check_jordan_filtration(r.filtration;budget,throw=true)
        expected=jordan_module(r.filtration;power=r.power,witness=r.bases!==nothing,budget)
        A=encoding_module(r);B=encoding_module(expected)
        RectilinearModule(A,encoding_axes(r);domain=r.ambient.domain,budget=_ambient_budget(budget))
        encoding_axes(r)==encoding_axes(expected) && r.ambient.domain==expected.ambient.domain || push!(errors,"Image parameter encoding differs")
        A.field==B.field && A.dims==B.dims && A.edge_maps==B.edge_maps || push!(errors,"Image module differs from the defining construction")
        r.bases==expected.bases || push!(errors,"Image embeddings differ")
        check_module_functoriality(A;throw=true)
    catch e
        e isa InterruptException && rethrow()
        push!(errors,sprint(showerror,e))
    end
    _report(errors,throw)
end
