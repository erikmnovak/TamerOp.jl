"""
Jordan type and multirank of exact finite-poset modules on selected disjoint
subsets. Consecutive subsets determine a nilpotent block operator; ranks of
its powers determine the multiplicities of its Jordan blocks.
"""
module JordanInvariants

using SparseArrays
using ..CoreModules: QQField, PrimeField, coeff_type
using ..FiniteFringe: nvertices, leq
using ..Modules: PModule, map_leq, check_module, check_module_functoriality
import ..FieldLinAlg
import ..ChainComplexes: describe, dimensions
import ..FiniteFringe: field
import ..Results: provenance

"""Bounds on selected stalks, block matrices and consecutive block products."""
struct JordanBudget
    max_vertices::Int
    max_slices::Int
    max_dimension::Int
    max_entries::Int
    max_products::Int
    function JordanBudget(;max_vertices=1024,max_slices=256,max_dimension=4096,
            max_entries=8_000_000,max_products=100000)
        xs=(max_vertices,max_slices,max_dimension,max_entries,max_products)
        all(x->x isa Integer && !(x isa Bool) && 0<=x<=typemax(Int),xs) ||
            throw(ArgumentError("Jordan budgets must be nonnegative integers fitting Int"))
        new(Int.(xs)...)
    end
end
_bound(n,b,label)=n<=b || throw(ArgumentError("Jordan $label budget exceeded ($n > $b)"))

"""Block multiplicities and power ranks, with optional consecutive-subset maps."""
struct JordanTypeResult{K,F}
    field::F
    source_vertices::Int
    slices::Vector{Vector{Int}}
    stalk_dimensions::Vector{Vector{Int}}
    multiplicities::Vector{Int}
    power_ranks::Vector{Int}
    blocks::Union{Nothing,Vector{Matrix{K}}}
end
"""Validation of a selected-subset query or its retained Jordan data."""
struct JordanValidationSummary
    valid::Bool
    errors::Vector{String}
end
Base.:(==)(a::JordanValidationSummary,b::JordanValidationSummary)=a.valid==b.valid && a.errors==b.errors
Base.isequal(a::JordanValidationSummary,b::JordanValidationSummary)=isequal(a.valid,b.valid)&&isequal(a.errors,b.errors)
Base.hash(x::JordanValidationSummary,h::UInt)=hash((x.valid,Tuple(x.errors)),h)
Base.show(io::IO,x::JordanValidationSummary)=print(io,"JordanValidationSummary(valid=",x.valid,", errors=",length(x.errors),")")
describe(x::JordanValidationSummary)=(valid=x.valid,errors=copy(x.errors))
function _report(errors,throw)
    throw && !isempty(errors) && Base.throw(ArgumentError(join(errors,"; ")))
    return JordanValidationSummary(isempty(errors),errors)
end
function _query(M,slices,budget)
    errors=String[]
    M.field isa Union{QQField,PrimeField} || push!(errors,"Jordan type requires QQ or a prime field")
    slices isa Union{Tuple,AbstractVector} || return vcat(errors,["slices must be a finite ordered tuple or vector"]),Vector{Int}[],Vector{Int}[]
    2<=length(slices)<=budget.max_slices || push!(errors,"slice count must be between 2 and max_slices")
    length(slices)<=budget.max_slices || return errors,Vector{Int}[],Vector{Int}[]
    labels=Vector{Int}[];ds=Vector{Int}[];seen=Set{Int}()
    for slice in slices
        slice isa Union{AbstractVector,AbstractSet,Tuple} || return vcat(errors,["each slice must be a finite collection of labels"]),labels,ds
        length(seen)+length(slice)<=budget.max_vertices || return vcat(errors,["selected vertex budget exceeded"]),labels,ds
        all(v->v isa Integer && !(v isa Bool) && 1<=v<=nvertices(M.Q),slice) || return vcat(errors,["invalid selected vertex"]),labels,ds
        ids=sort!(Int[v for v in slice])
        length(unique(ids))==length(ids) && all(v->!(v in seen),ids) || push!(errors,"slices must be pairwise disjoint with no repeated labels")
        union!(seen,ids);push!(labels,ids);push!(ds,M.dims[ids])
    end
    dims=[sum(big,d;init=big(0)) for d in ds]
    any(d->any(<(0),d),ds) && push!(errors,"negative stalk dimension")
    sum(dims;init=big(0))<=budget.max_dimension || push!(errors,"total selected dimension budget exceeded")
    n=length(dims)
    # Keep all adjacent blocks and at most two consecutive product matrices.
    adjacent=sum(dims[i]*dims[i+1] for i in 1:n-1;init=big(0))
    largest=maximum((dims[i]*dims[j] for i in 1:n for j in i+1:n);init=big(0))
    adjacent+2largest<=budget.max_entries || push!(errors,"block/product matrix budget exceeded")
    div(big(n)*(n-1),2)<=budget.max_products || push!(errors,"block product budget exceeded")
    return errors,labels,ds
end
"""
    check_jordan_query(M; slices, budget=JordanBudget(), throw=false)

Check exact coefficients, pairwise disjoint finite subsets and work limits.
Empty subsets are permitted. Subsets need not be antichains or follow an order;
only strictly comparable pairs from one subset to the next contribute a map.
This validates the query; `jordan_type` checks module functoriality by default.
"""
function check_jordan_query(M::PModule;slices,budget::JordanBudget=JordanBudget(),throw::Bool=false)
    structural=check_module(M)
    structural.valid || return _report(copy(structural.issues),throw)
    errors,_,_=_query(M,slices,budget)
    return _report(errors,throw)
end
function _blocks(M::PModule{K},slices,ds) where K
    n=length(slices);blocks=Matrix{K}[]
    for i in 1:n-1
        a=ds[i];b=ds[i+1];A=zeros(K,sum(b),sum(a));col=0
        for (j,u) in enumerate(slices[i])
            row=0
            for (k,v) in enumerate(slices[i+1])
                if leq(M.Q,u,v) && a[j]>0 && b[k]>0
                    A[row+1:row+b[k],col+1:col+a[j]]=map_leq(M,u,v)
                end
                row+=b[k]
            end
            col+=a[j]
        end
        push!(blocks,A)
    end
    return blocks
end
function _power_ranks(F,blocks,dims)
    n=length(dims);ranks=zeros(Int,n+1);ranks[1]=sum(dims)
    for first in 1:n-1
        product=blocks[first]
        for last in first:n-1
            last==first || (product=blocks[last]*product)
            rank=FieldLinAlg.rank_dim(F,product)
            ranks[last-first+2]+=rank
            iszero(rank) && break
        end
    end
    return ranks
end
function _multiplicities(ranks)
    n=length(ranks)-1
    return [ranks[j]-2ranks[j+1]+(j<n ? ranks[j+2] : 0) for j in 1:n]
end
"""
    jordan_type(M; slices, witness=false, check=true, budget=JordanBudget())

Compute the Jordan block multiplicities of the selected-subset operator of
Chindris, Kang and Kline (arXiv:2510.22116, Definition 3). The block from slice
`i` to `i+1` contains each structure map for a strictly comparable pair, with
zero blocks for other pairs. Its nth power vanishes for n slices.

For `r[j] = rank(T^j)`, size-j blocks have multiplicity
`r[j-1] - 2r[j] + r[j+1]`. Computation sums the ranks of consecutive block
products, avoiding a dense matrix on the whole direct sum. QQ and prime fields
are supported; characteristic-dependent path cancellations are retained.
Set `witness=true` to retain the actual consecutive blocks for inspection.
This finite query does not compute translated Jordan-filtered ranks.
"""
function jordan_type(M::PModule{K};slices,witness::Bool=false,check::Bool=true,
        budget::JordanBudget=JordanBudget()) where K
    check_jordan_query(M;slices,budget,throw=true)
    _,labels,ds=_query(M,slices,budget)
    check && _report(check_module_functoriality(M).issues,true)
    blocks=_blocks(M,labels,ds)
    ranks=_power_ranks(M.field,blocks,sum.(ds));counts=_multiplicities(ranks)
    all(>=(0),counts) || error("invalid Jordan multiplicities")
    return JordanTypeResult{K,typeof(M.field)}(M.field,nvertices(M.Q),labels,ds,counts,ranks,witness ? blocks : nothing)
end
"""
    multirank(M; sources, targets, check=true, budget=JordanBudget())

Rank of the block map from the direct sum of the selected source stalks to the
selected target stalks. The two collections must be disjoint. This is the
size-two multiplicity of their Jordan type, rather than a sum of pairwise ranks.
"""
function multirank(M::PModule;sources,targets,check::Bool=true,budget::JordanBudget=JordanBudget())
    return jordan_type(M;slices=(sources,targets),check,budget).power_ranks[2]
end
"""Return owned block counts, indexed by block size 1 through the number of slices."""
jordan_multiplicities(r::JordanTypeResult)=copy(r.multiplicities)
"""Return owned ranks of powers zero through n, including the terminal zero."""
jordan_power_ranks(r::JordanTypeResult)=copy(r.power_ranks)
"""Return copies of the ordered slices, whose internal labels are sorted."""
jordan_slices(r::JordanTypeResult)=map(copy,r.slices)
"""Return copies of the retained consecutive-subset matrices."""
function jordan_subset_maps(r::JordanTypeResult)
    r.blocks===nothing && throw(ArgumentError("subset maps were not retained; request witness=true"))
    return map(copy,r.blocks)
end
"""Materialize the retained nilpotent operator as a sparse matrix, within a logical matrix budget."""
function jordan_operator(r::JordanTypeResult{K};budget::JordanBudget=JordanBudget()) where K
    r.blocks===nothing && throw(ArgumentError("operator was not retained; request witness=true"))
    ds=sum.(r.stalk_dimensions);d=sum(ds)
    _bound(d,budget.max_dimension,"operator dimension");_bound(big(d)^2,budget.max_entries,"operator entries")
    offsets=cumsum(vcat(0,ds));rows=Int[];cols=Int[];vals=K[]
    for (i,A) in enumerate(r.blocks),col in axes(A,2),row in axes(A,1)
        iszero(A[row,col]) && continue
        push!(rows,offsets[i+1]+row);push!(cols,offsets[i]+col);push!(vals,A[row,col])
    end
    return sparse(rows,cols,vals,d,d)
end
field(r::JordanTypeResult)=r.field
dimensions(r::JordanTypeResult)=sum.(r.stalk_dimensions)
provenance(r::JordanTypeResult)=(category=:finite_poset,field=r.field,source_vertices=r.source_vertices,
    construction=:consecutive_subset_operator,slices=jordan_slices(r),translated=false)
"""Describe the selected finite diagram, power ranks and Jordan block counts."""
jordan_summary(r::JordanTypeResult)=describe(r)
describe(r::JordanTypeResult)=(kind=:jordan_type,field=r.field,slices=length(r.slices),
    vertices=sum(length,r.slices),dimension=sum(dimensions(r)),
    multiplicities=jordan_multiplicities(r),power_ranks=jordan_power_ranks(r),
    category=:finite_poset,witness=r.blocks!==nothing)
Base.show(io::IO,r::JordanTypeResult)=print(io,"JordanTypeResult(multiplicities=",r.multiplicities,", field=",r.field,")")
Base.show(io::IO,b::JordanBudget)=print(io,"JordanBudget(max_vertices=",b.max_vertices,", max_dimension=",b.max_dimension,")")
describe(b::JordanBudget)=(max_vertices=b.max_vertices,max_slices=b.max_slices,max_dimension=b.max_dimension,max_entries=b.max_entries,max_products=b.max_products)

"""
    check_jordan_type(result; source=nothing, budget=JordanBudget(), throw=false)

Check ranks, multiplicities and total dimension. Retained subset maps are
independently recomputed when `source` is supplied and their power ranks are
checked whenever present. A result without maps is checked for internal
numerical consistency only; supply the original module to check its provenance.
"""
function check_jordan_type(r::JordanTypeResult{K};source=nothing,
        budget::JordanBudget=JordanBudget(),throw::Bool=false) where K
    errors=String[];n=length(r.slices)
    n>=2 && length(r.stalk_dimensions)==n && length(r.multiplicities)==n && length(r.power_ranks)==n+1 ||
        return _report(["Jordan result dimensions disagree"],throw)
    all(i->length(r.slices[i])==length(r.stalk_dimensions[i]),1:n) || return _report(["stalk dimensions disagree with slices"],throw)
    labels=reduce(vcat,r.slices;init=Int[])
    all(v->1<=v<=r.source_vertices,labels) && length(unique(labels))==length(labels) || push!(errors,"invalid or repeated source labels")
    all(ds->all(>=(0),ds),r.stalk_dimensions) || return _report(["negative stalk dimension"],throw)
    bigdims=[sum(big,d;init=big(0)) for d in r.stalk_dimensions]
    sum(bigdims;init=big(0))<=budget.max_dimension && n<=budget.max_slices && length(labels)<=budget.max_vertices ||
        return _report(["Jordan validation budget exceeded"],throw)
    ds=Int.(bigdims)
    r.field isa Union{QQField,PrimeField} && coeff_type(r.field)==K || push!(errors,"invalid exact coefficient field")
    all(>=(0),r.multiplicities) && all(>=(0),r.power_ranks) || push!(errors,"negative Jordan data")
    r.power_ranks[1]==sum(ds) && iszero(r.power_ranks[end]) || push!(errors,"invalid zeroth or terminal power rank")
    r.multiplicities==_multiplicities(r.power_ranks) || push!(errors,"multiplicities disagree with power ranks")
    sum(big(i)*r.multiplicities[i] for i in 1:n)==sum(big,ds) || push!(errors,"Jordan blocks do not recover total dimension")
    for j in 0:n
        sum(big(max(i-j,0))*r.multiplicities[i] for i in 1:n)==r.power_ranks[j+1] || push!(errors,"Jordan blocks do not recover power rank $j")
    end
    if r.blocks!==nothing
        length(r.blocks)==n-1 && all(i->size(r.blocks[i])==(ds[i+1],ds[i]),1:n-1) ||
            return _report(vcat(errors,["retained block shapes disagree"]),throw)
        if isempty(errors)
            sum(big,ds)<=budget.max_dimension && n<=budget.max_slices &&
                sum(big(length(A)) for A in r.blocks)+2maximum((big(ds[i])*ds[j] for i in 1:n for j in i+1:n);init=big(0))<=budget.max_entries && div(big(n)*(n-1),2)<=budget.max_products ||
                return _report(["Jordan validation budget exceeded"],throw)
            _power_ranks(r.field,r.blocks,ds)==r.power_ranks || push!(errors,"retained blocks disagree with power ranks")
        end
    end
    if source!==nothing && isempty(errors)
        source isa PModule || return _report(["source must be a finite module"],throw)
        query=check_jordan_query(source;slices=r.slices,budget)
        append!(errors,query.errors)
        if isempty(errors)
            source.field==r.field && nvertices(source.Q)==r.source_vertices && [source.dims[s] for s in r.slices]==r.stalk_dimensions ||
                push!(errors,"source field, labels or stalk dimensions disagree")
            if isempty(errors)
                valid=check_module_functoriality(source);append!(errors,valid.issues)
            end
            if isempty(errors)
                b=_blocks(source,r.slices,r.stalk_dimensions)
                r.blocks===nothing || b==r.blocks || push!(errors,"source structure maps disagree with witness")
                _power_ranks(source.field,b,ds)==r.power_ranks || push!(errors,"source power ranks disagree")
            end
        end
    end
    return _report(errors,throw)
end

end # module JordanInvariants
