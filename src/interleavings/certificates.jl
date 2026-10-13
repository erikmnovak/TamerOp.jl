struct InterleavingProblem{K,L,R,P}
    left::L
    right::R
    sources::Tuple{PModule{K},PModule{K}}
    targets::Tuple{PModule{K},PModule{K}}
    at::Vector{Int}
    shifted_at::Vector{Int}
    canonical::Tuple{Vector{Matrix{K}},Vector{Matrix{K}}}
    points::Vector{P}
    metadata::NamedTuple
end

function _same_field(M,N)
    M.field == N.field || throw(ArgumentError("Interleaving coefficient fields must agree."))
end

function _witness_budget(sources,targets,budget)
    entries=sum(big(sources[s].dims[i])*targets[s].dims[i]
                for s in 1:2 for i in eachindex(sources[s].dims);init=big(0))
    _limit(entries,budget.max_matrix_entries,"witness matrices")
end

"""
    interleaving_problem(M, N, delta)

Construct the finite naturality and composition problem for an ambient
nonnegative translation. Scalar delta means a diagonal translation. Real
singleton strata are retained, including in the twice-shifted refinement.
For two PModules on the same finite poset, delta is instead the vertex vector
of a monotone extensive endomap; that version carries no physical epsilon.
"""
function interleaving_problem(A::RectilinearModule, B::RectilinearModule, delta;
                              budget=InterleavingBudget())
    A.domain == B.domain && length(A.cuts) == length(B.cuts) ||
        throw(ArgumentError("Ambient domains must agree."))
    _same_field(A.finite,B.finite)
    A=RectilinearModule(A.finite,A.cuts;domain=A.domain,origin=A.origin,budget)
    B=RectilinearModule(B.finite,B.cuts;domain=B.domain,origin=B.origin,budget)
    n = length(A.cuts); d = _vector_shift(delta,n,A.domain;nonnegative=true)
    # A common refinement for both Hom spaces, followed by its translated
    # refinement for the two equations. No sampling window is introduced.
    c = ntuple(i -> sort!(unique([A.cuts[i]; B.cuts[i]; A.cuts[i].-d[i]; B.cuts[i].-d[i]])),n)
    Q = _grid(c,A.domain,budget); points = _points(c,A.domain)
    z = ntuple(_ -> QQ(0),n)
    M = _pull(A,Q,points,z,budget); N = _pull(B,Q,points,z,budget)
    Ns = _pull(B,Q,points,d,budget); Ms = _pull(A,Q,points,d,budget)
    _witness_budget((M,N),(Ns,Ms),budget)
    cc = ntuple(i -> sort!(unique([c[i]; c[i].-d[i]])),n)
    _grid(cc,A.domain,budget)
    checks = _points(cc,A.domain)
    at = [_label(c,A.domain,x) for x in checks]
    shifted = [_label(c,A.domain,x.+d) for x in checks]
    entries = sum(big(M.dims[at[i]])*Ms.dims[shifted[i]] +
                  big(N.dims[at[i]])*Ns.dims[shifted[i]] for i in eachindex(at);init=big(0))
    _limit(entries,budget.max_matrix_entries,"composition matrices")
    K = coeff_type(M.field)
    canonical = (Matrix{K}[structure_map(A;source=x,target=x.+2 .*d) for x in checks],
                 Matrix{K}[structure_map(B;source=x,target=x.+2 .*d) for x in checks])
    meta = (; kind=:ambient_interleaving, domain=A.domain, translation=d,
            epsilon=all(==(d[1]),d) ? d[1] : nothing, field=M.field, window=nothing,
            left=provenance(A), right=provenance(B), cuts=c)
    return InterleavingProblem(A,B,(M,N),(Ns,Ms),at,shifted,canonical,checks,meta)
end

function interleaving_problem(A::PModule, B::PModule, translation::AbstractVector{<:Integer};
                              budget=InterleavingBudget())
    A.Q === B.Q || throw(ArgumentError("Finite-poset modules must have the same poset object."))
    _same_field(A,B); _module_check(A,budget); _module_check(B,budget)
    Q = A.Q; n = nvertices(Q); t = Int.(translation)
    length(t) == n && all(i -> 1 <= i <= n,t) || throw(ArgumentError("Invalid translation vertex map."))
    all(i -> leq(Q,i,t[i]),1:n) || throw(ArgumentError("The translation must be extensive: x <= T(x)."))
    all(e -> leq(Q,t[e[1]],t[e[2]]),cover_edges(Q)) || throw(ArgumentError("The translation must be monotone."))
    M,N = _snapshot(A),_snapshot(B)
    Ns,Ms = _pull(N,Q,t,budget),_pull(M,Q,t,budget)
    _witness_budget((M,N),(Ns,Ms),budget)
    _limit(sum(big(M.dims[i])*M.dims[t[t[i]]] + big(N.dims[i])*N.dims[t[t[i]]] for i in 1:n;init=big(0)),
        budget.max_matrix_entries,"composition matrices")
    K = coeff_type(M.field)
    canonical = (Matrix{K}[map_leq(M,i,t[t[i]]) for i in 1:n],
                 Matrix{K}[map_leq(N,i,t[t[i]]) for i in 1:n])
    meta = (;kind=:finite_poset_interleaving,domain=:finite_poset,translation=t,
             epsilon=nothing,field=M.field,window=nothing)
    return InterleavingProblem(M,N,(M,N),(Ns,Ms),collect(1:n),t,canonical,collect(1:n),meta)
end

function interleaving_problem(A::EncodingResult,B::EncodingResult,delta;
                              extension=:none,budget=InterleavingBudget())
    return interleaving_problem(ambient_module(A;extension,budget),
                               ambient_module(B;extension,budget),delta;budget)
end

interleaving_spaces(p::InterleavingProblem) =
    (;forward=(source=p.sources[1],target=p.targets[1]),
      reverse=(source=p.sources[2],target=p.targets[2]))
provenance(p::InterleavingProblem) = p.metadata
describe(p::InterleavingProblem) = (;kind=:interleaving_problem,
    vertices=nvertices(p.sources[1].Q),equation_locations=length(p.at),provenance=provenance(p))
Base.show(io::IO,p::InterleavingProblem) = print(io,"InterleavingProblem(",p.metadata.domain,", ",length(p.at)," equation locations)")

function _same_module(A,B)
    A.field==B.field && A.dims==B.dims &&
        sort!(collect(cover_edges(A.Q)))==sort!(collect(cover_edges(B.Q))) || return false
    return all(e -> A.edge_maps[e...]==B.edge_maps[e...],cover_edges(A.Q))
end

# Rebuild from the owned ambient data, not from stored equation right-hand
# sides. A mutated refinement, translation, or canonical map cannot turn a
# failed equation into a certificate merely by editing its expected answer.
function check_interleaving_problem(p::InterleavingProblem; throw::Bool=false,
                                   budget=InterleavingBudget())
    issues=String[]
    try
        expected=interleaving_problem(p.left,p.right,p.metadata.translation;budget)
        p.metadata==expected.metadata || push!(issues,"Problem provenance differs from its ambient construction.")
        p.at==expected.at && p.shifted_at==expected.shifted_at && p.points==expected.points ||
            push!(issues,"The equation refinement or translated labels have changed.")
        p.canonical==expected.canonical || push!(issues,"Canonical structure maps have changed.")
        for side in 1:2
            _same_module(p.sources[side],expected.sources[side]) || push!(issues,"A source module differs from its encoding.")
            _same_module(p.targets[side],expected.targets[side]) || push!(issues,"A shifted module differs from its encoding.")
        end
    catch err
        err isa InterruptException && rethrow()
        push!(issues,sprint(showerror,err))
    end
    v=InterleavingValidationSummary(isempty(issues),issues,Int[],Int[])
    throw && !v.valid && Base.throw(ArgumentError(join(issues,"\n")))
    return v
end

struct InterleavingValidationSummary
    valid::Bool
    issues::Vector{String}
    forward_failures::Vector{Int}
    reverse_failures::Vector{Int}
end
Base.show(io::IO,v::InterleavingValidationSummary) = print(io,"InterleavingValidationSummary(valid=",v.valid,", issues=",length(v.issues),")")
describe(v::InterleavingValidationSummary) = (;kind=:interleaving_validation,valid=v.valid,
    issues=copy(v.issues),forward_failures=copy(v.forward_failures),reverse_failures=copy(v.reverse_failures))

struct InterleavingCertificate{P,F,G}
    problem::P
    forward::F
    reverse::G
end

function _components(p,side,maps)
    M,N = p.sources[side],p.targets[side]
    K = coeff_type(M.field)
    if maps isa PMorphism
        maps.dom === M && maps.cod === N || throw(ArgumentError("Use a morphism on this problem's source and target modules."))
        return PMorphism(M,N,Matrix{K}[Matrix{K}(A) for A in maps.comps])
    end
    length(maps)==length(M.dims) &&
        all(i -> maps[i] isa AbstractMatrix && size(maps[i])==(N.dims[i],M.dims[i]),eachindex(M.dims)) ||
        throw(ArgumentError("Supply one witness matrix of the correct shape per finite vertex."))
    return PMorphism(M,N,Matrix{K}[_exact_matrix(M.field,A) for A in maps])
end

function interleaving_certificate(p::InterleavingProblem, forward, reverse; budget=InterleavingBudget())
    _witness_budget(p.sources,p.targets,budget)
    c = InterleavingCertificate(p,_components(p,1,forward),_components(p,2,reverse))
    check_interleaving(c;throw=true,budget)
    return c
end

interleaving_maps(c::InterleavingCertificate) = (;forward=c.forward,reverse=c.reverse)
provenance(c::InterleavingCertificate) = (;provenance(c.problem)...,guarantee=:exact_interleaving,
    confidence=nothing,geometry_assumptions=:declared_parameter_model)
describe(c::InterleavingCertificate) = (;kind=:interleaving_certificate,provenance=provenance(c))
Base.show(io::IO,c::InterleavingCertificate) = print(io,"InterleavingCertificate(",c.problem.metadata.domain,", translation=",c.problem.metadata.translation,")")

function interleaving_equations(c::InterleavingCertificate; vertex::Integer)
    p=c.problem; i=Int(vertex)
    1 <= i <= length(p.at) || throw(ArgumentError("Equation vertex is out of range."))
    a,b=p.at[i],p.shifted_at[i]
    return (;point=p.points[i],
        forward=(lhs=c.reverse.comps[b]*c.forward.comps[a],rhs=copy(p.canonical[1][i])),
        reverse=(lhs=c.forward.comps[b]*c.reverse.comps[a],rhs=copy(p.canonical[2][i])))
end

function check_interleaving(c::InterleavingCertificate; throw::Bool=false,budget=InterleavingBudget())
    issues=String[]; ff=Int[]; gf=Int[]; p=c.problem
    try
        append!(issues,check_interleaving_problem(p;budget).issues)
        for side in 1:2
            f=side==1 ? c.forward : c.reverse
            f.dom === p.sources[side] && f.cod === p.targets[side] ||
                push!(issues,"Witness source or target differs from the problem.")
            append!(issues,check_module_functoriality(p.sources[side]).issues)
            append!(issues,check_module_functoriality(p.targets[side]).issues)
            append!(issues,check_morphism(f).issues)
        end
        if isempty(issues)
            for i in eachindex(p.at)
                e=interleaving_equations(c;vertex=i)
                e.forward.lhs==e.forward.rhs || push!(ff,i)
                e.reverse.lhs==e.reverse.rhs || push!(gf,i)
            end
            isempty(ff) || push!(issues,"The reverse-after-forward equation fails.")
            isempty(gf) || push!(issues,"The forward-after-reverse equation fails.")
        end
    catch err
        err isa InterruptException && rethrow()
        push!(issues,sprint(showerror,err))
    end
    v=InterleavingValidationSummary(isempty(issues),issues,ff,gf)
    throw && !v.valid && Base.throw(ArgumentError(join(issues,"\n")))
    return v
end
