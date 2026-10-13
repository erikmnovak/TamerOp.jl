# Validation retains maximality/minimality, not merely one feasible pair.
"""Validation of the defining pruning pair, ambient shifts and exact subquotient maps."""
struct PruningValidationSummary
    valid::Bool
    issues::Vector{String}
end
Base.show(io::IO,v::PruningValidationSummary)=print(io,"PruningValidationSummary(valid=",v.valid,", issues=",length(v.issues),")")
describe(v::PruningValidationSummary)=(;kind=:pruning_validation,valid=v.valid,issues=copy(v.issues))
Base.:(==)(a::PruningValidationSummary,b::PruningValidationSummary)=a.valid==b.valid && a.issues==b.issues
Base.isequal(a::PruningValidationSummary,b::PruningValidationSummary)=a==b
Base.hash(a::PruningValidationSummary,h::UInt)=hash((a.valid,Tuple(a.issues)),h)

function _same_module(A,B)
    A.field==B.field && A.dims==B.dims && collect(cover_edges(A.Q))==collect(cover_edges(B.Q)) &&
        all(e->A.edge_maps[e...]==B.edge_maps[e...],cover_edges(A.Q))
end
function _same_ambient(A,B;budget=InterleavingBudget())
    A.domain==B.domain && length(encoding_axes(A))==length(encoding_axes(B)) || return false
    cuts=map((a,b)->sort!(unique([a;b])),encoding_axes(A),encoding_axes(B))
    _same_module(encoding_module(refine_parameters(A,cuts;budget)),encoding_module(refine_parameters(B,cuts;budget)))
end
function _same_space(field,A,B)
    size(A,1)==size(B,1) && size(A,2)==size(B,2) &&
        FieldLinAlg.rank(field,hcat(A,B))==size(A,2)
end

function check_pruning(r::PruningResult;throw::Bool=false,budget=InterleavingBudget())
    issues=String[]
    try
        R,T,t,H,e=_prepare(r.source,r.epsilon,budget)
        A=encoding_module(R);field=A.field
        encoding_axes(R)==encoding_axes(r.refined) && _same_module(A,encoding_module(r.refined)) ||
            push!(issues,"Refined source differs from the global input.")
        _same_ambient(T,r.shifted;budget) || push!(issues,"Shifted source differs from M(2 epsilon).")
        t.comps==r.shift_map.comps && _same_module(t.dom,r.shift_map.dom) && _same_module(t.cod,r.shift_map.cod) ||
            push!(issues,"Canonical shift map differs from the ambient structure maps.")
        r.hom_dimension==length(H) || push!(issues,"Full Hom dimension differs.")
        append!(issues,check_submodule(r.image).issues)
        append!(issues,check_submodule(r.kernel).issues)
        i=inclusion_map(r.image);k=inclusion_map(r.kernel);q=r.projection
        _same_module(i.cod,A) && k.cod===i.dom && q.dom===i.dom ||
            push!(issues,"The pruning inclusions/projection have incompatible domains.")
        append!(issues,check_morphism(q).issues)
        _same_module(q.cod,encoding_module(r.unshifted)) && encoding_axes(r.unshifted)==encoding_axes(R) ||
            push!(issues,"The unshifted output differs from the quotient.")
        _same_ambient(parameter_shift(r.unshifted,-e;budget),r.pruned;budget) ||
            push!(issues,"The output differs from (I/K)(-epsilon).")
        for v in eachindex(A.dims)
            expectedI,expectedK,_,_=_pair(field,t.comps[v],[f.comps[v] for f in H])
            _same_space(field,i.comps[v],expectedI) || push!(issues,"I is not the maximal pruning subspace at vertex $v.")
            _same_space(field,i.comps[v]*k.comps[v],expectedK) || push!(issues,"K is not the minimal pruning subspace at vertex $v.")
            all(iszero,q.comps[v]*k.comps[v]) && FieldLinAlg.rank(field,q.comps[v])==q.cod.dims[v] &&
                q.cod.dims[v]==i.dom.dims[v]-k.dom.dims[v] || push!(issues,"The quotient is not exact at vertex $v.")
        end
    catch err
        err isa InterruptException && rethrow()
        push!(issues,sprint(showerror,err))
    end
    v=PruningValidationSummary(isempty(issues),issues)
    throw && !v.valid && Base.throw(ArgumentError(join(issues,"\n")))
    return v
end

"""Report the proved refinement bound; optionally validate a supplied epsilon-interleaving to another module."""
function pruning_guarantee(r::PruningResult;against=nothing,budget=InterleavingBudget())
    reference=:source
    if against!==nothing
        check_interleaving(against;throw=true,budget)
        p=against.problem
        p.left isa RectilinearModule && p.metadata.epsilon==r.epsilon && _same_ambient(p.left,r.source;budget) ||
            throw(ArgumentError("Supply an epsilon-interleaving whose left module is the pruning source."))
        reference=:source_and_compared_module
    end
    rmax=maximum(encoding_module(r.source).dims;init=0)
    # If M=0 and N is epsilon-trivial, N need not be zero. Its comparison
    # bound is epsilon, not the vacuous numeric product 2*0*epsilon.
    bound=against!==nothing && rmax==0 ? r.epsilon : 2rmax*r.epsilon
    return (;kind=:refinement, bound,epsilon=r.epsilon,
        max_stalk_dimension=rmax,reference,theorem=rmax==0 ? :zero_module : :bjerkevik_theorem_5_4,
        pruning_distance_stability=:not_asserted,
        comparison_maps=against===nothing ? :constructible_on_request : :supplied_input_interleaving,
        source=provenance(r.source))
end

# A consistent solve with free coordinates zero. The chosen lift is immaterial
# after quotienting by K, which contains ker(M -> M(2 epsilon)) on I.
function _lift(field,A,B)
    n=size(A,2);R,p=FieldLinAlg.rref(field,hcat(A,B))
    any(>(n),p) && throw(ArgumentError("The pruning comparison has no lift."))
    X=zeros(eltype(A),n,size(B,2))
    for (row,col) in enumerate(p)
        X[col,:]=R[row,n+1:end]
    end
    A*X==B || error("Pruning lift failed its exact equation.")
    return X
end

"""Construct and validate actual interleaving maps from M to its pruning at the proved 2r*epsilon bound."""
function pruning_certificate(r::PruningResult;budget=InterleavingBudget())
    check_pruning(r;throw=true,budget)
    b=pruning_guarantee(r).bound;e=r.epsilon
    p=interleaving_problem(r.source,r.pruned,b;budget)
    A=encoding_module(r.refined);field=A.field;K=coeff_type(field)
    cuts=p.metadata.cuts
    model=RectilinearModule(p.sources[1],cuts;domain=r.source.domain,budget)
    forward=Matrix{K}[];reverse=Matrix{K}[]
    i=inclusion_map(r.image);q=r.projection
    for x in _points(model)
        y=x.+b.-e;v=locate(r.refined,y)
        target=structure_map(r.source;source=x,target=y.+2e)
        lift=_lift(field,r.shift_map.comps[v]*i.comps[v],target)
        push!(forward,q.comps[v]*lift)
        z=x.-e;w=locate(r.refined,z)
        B=structure_map(r.source;source=z,target=x.+b)*i.comps[w]
        push!(reverse,Matrix(transpose(FieldLinAlg.solve_fullcolumn(field,Matrix(transpose(q.comps[w])),Matrix(transpose(B))))))
    end
    return interleaving_certificate(p,forward,reverse;budget)
end
