"""Integral lifting and harmonic circular coordinates on finite filtered sources.

The integer cocycle and least-squares construction follows de Silva, Morozov and
Vejdemo-Johansson, Persistent Cohomology and Circular Coordinates (2011),
https://doi.org/10.1007/s00454-011-9344-x. Multiple selected classes give a map
to a product of circles. This does not assert an embedding or a preferred basis.
"""
module CohomologicalCoordinates
using LinearAlgebra, SparseArrays
import Nemo
import ..DataIngestion, ..DataTypes, ..OrdinaryPersistence, ..FieldLinAlg
using ..CoreModules: Fp, PrimeField, coeff_type
import ..ChainComplexes: describe
import ..Results: provenance, result_summary

"""Size limits for active cells, selected classes and dense integer-lifting work."""
Base.@kwdef struct CoordinateBudget
    max_vertices::Int = 20_000
    max_edges::Int = 100_000
    max_faces::Int = 200_000
    max_classes::Int = 16
    max_integer_entries::Int = 4_000_000
end

"""Checked integer lifts and numerical circle coordinates on active source vertices.

Use `coordinate_values`, `coordinate_summary` and `provenance` to inspect a
result. Retained source cells, cochains and solver arrays are read-only while
an explorer uses them. Values are turns modulo one, not an embedding certificate.
"""
struct CohomologicalCoordinateResult{S,D,T}
    source::S
    diagram::D
    scale::T
    field::PrimeField
    classes::Vector{NamedTuple}
    vertices::Vector{Int}
    edges::Vector{Int}
    delta0::SparseMatrixCSC{Int,Int}
    delta1::SparseMatrixCSC{Int,Int}
    cocycles::Matrix{BigInt}
    lifts::Matrix{BigInt}
    potentials::Matrix{Float64}
    values::Matrix{Float64}
    harmonic::Matrix{Float64}
    weights::Vector{Float64}
    anchors::Vector{Int}
    components::Vector{Int}
    provenance::NamedTuple
end

"""Validity and concrete failures of the lift, smoothing and gauge equations."""
struct CoordinateValidationSummary
    valid::Bool
    issues::Vector{String}
end
Base.show(io::IO,s::CoordinateValidationSummary)=print(io,"CoordinateValidationSummary(valid=",s.valid,", issues=",length(s.issues),")")
Base.:(==)(a::CoordinateValidationSummary,b::CoordinateValidationSummary)=a.valid==b.valid && a.issues==b.issues
Base.isequal(a::CoordinateValidationSummary,b::CoordinateValidationSummary)=isequal(a.valid,b.valid) && isequal(a.issues,b.issues)
Base.hash(a::CoordinateValidationSummary,h::UInt)=hash(a.issues,hash(a.valid,h))

coordinate_values(r::CohomologicalCoordinateResult)=copy(r.values)
function coordinate_summary(r::CohomologicalCoordinateResult)
    (;kind=:cohomological_coordinates,vertices=length(r.vertices),edges=length(r.edges),
        coordinates=size(r.values,2),scale=r.scale,field=r.field,classes=copy(r.classes),
        components=maximum(r.components;init=0),anchors=copy(r.anchors),
        energy=vec(sum(r.weights .* abs2.(r.harmonic);dims=1)),provenance=r.provenance)
end
describe(r::CohomologicalCoordinateResult)=coordinate_summary(r)
result_summary(r::CohomologicalCoordinateResult)=coordinate_summary(r)
provenance(r::CohomologicalCoordinateResult)=merge(r.provenance,(;field=r.field,scale=r.scale,classes=copy(r.classes),anchors=copy(r.anchors)))
Base.show(io::IO,r::CohomologicalCoordinateResult)=print(io,"CohomologicalCoordinateResult(",length(r.vertices)," vertices, ",size(r.values,2)," circle coordinates, scale=",r.scale,")")

function _budget(b::CoordinateBudget,n,e,f,k)
    all(name->getfield(b,name)>0,fieldnames(CoordinateBudget)) || throw(ArgumentError("Coordinate budgets must be positive."))
    n<=b.max_vertices && e<=b.max_edges && f<=b.max_faces && k<=b.max_classes ||
        throw(ArgumentError("Cohomological coordinate size budget exceeded."))
end

function _integer_lift(D, alpha, p, budget)
    A=copy(alpha);defect=D*A
    all(x->iszero(mod(x,p)),defect) || throw(ArgumentError("Input cochains are not cocycles over the selected prime field."))
    iszero(defect) && return A, :balanced_integral_lift
    m,n=size(D)
    big(m)*n+big(m)^2+big(n)^2<=budget.max_integer_entries ||
        throw(ArgumentError("Integral-lift Smith normal form exceeds max_integer_entries."))
    # S=U D V: divisibility, including zero rows, decides whether an integral
    # correction exists. Rational least squares cannot decide this obstruction.
    S,U,V=Nemo.snf_with_transform(Nemo.matrix(Nemo.ZZ,Matrix(D)))
    rhs=U*Nemo.matrix(Nemo.ZZ,div.(defect,p))
    y=Nemo.zero_matrix(Nemo.ZZ,n,size(A,2))
    for i in 1:m,j in axes(A,2)
        s=i<=min(m,n) ? S[i,i] : Nemo.ZZ(0)
        if iszero(s)
            iszero(rhs[i,j]) || throw(ArgumentError("Selected cohomology class has an integral-lifting obstruction at this scale."))
        else
            iszero(rem(rhs[i,j],s)) || throw(ArgumentError("Selected cohomology class has an integral-lifting obstruction at this scale."))
            y[i,j]=div(rhs[i,j],s)
        end
    end
    correction=Matrix{BigInt}(V*y)
    A .-= BigInt(p).*correction
    iszero(D*A) || error("Integral lifting failed its exact equation.")
    return A,:smith_normal_form_correction
end

function _edge_vertices(D::SparseMatrixCSC)
    ends=[Int[] for _ in axes(D,1)]
    for v in axes(D,2),ptr in nzrange(D,v)
        iszero(D.nzval[ptr]) || push!(ends[D.rowval[ptr]],v)
    end
    ends
end

function _components(D)
    n=size(D,2);parent=collect(1:n)
    root(i)=begin;while parent[i]!=i;parent[i]=parent[parent[i]];i=parent[i];end;i;end
    endpoints=_edge_vertices(D)
    for e in axes(D,1)
        vs=endpoints[e]
        if length(vs)==2 && sort(Int.(D[e,vs]))==[-1,1]
            a,b=root(vs[1]),root(vs[2]);parent[max(a,b)]=min(a,b)
        elseif !isempty(vs)
            throw(ArgumentError("Circular coordinates require oriented graph incidence in degree one."))
        end
    end
    roots=root.(1:n);labels=sort!(unique(roots));index=Dict(v=>i for (i,v) in enumerate(labels))
    return [index[r] for r in roots],labels
end

function _smooth(D, lifts, weights, anchors)
    n=size(D,2);free=setdiff(1:n,anchors);potential=zeros(n,size(lifts,2))
    A=Float64.(lifts);all(isfinite,A) || throw(ArgumentError("Integral lift cannot be represented by the numerical smoothing solver."))
    if !isempty(free)
        E=Float64.(D[:,free]);W=spdiagm(0=>weights)
        L=Symmetric(transpose(E)*W*E)
        potential[free,:]=cholesky(L) \ (-transpose(E)*(weights.*A))
    end
    harmonic=A+D*potential
    residual=transpose(D)*(weights.*harmonic)
    norm(residual,Inf)<=1e-8*max(1.,norm(weights.*A,Inf)) ||
        throw(ArgumentError("Harmonic solver failed its residual check; rescale edge weights."))
    return potential,harmonic
end

"""
    toroidal_coordinates(source; scale, classes=[(kind=:finite,index=1)], field=Fp(41), ...)

Lift selected persistent H1 cocycles to integers and minimize their weighted
edge energy. Return vertex values in R/Z, in the selected class order. `source`
is a one-parameter FilteredComplex; `diagram` may reuse a cocycle-bearing result
on that same source. Alternatively `cocycles` supplies one integer column per
class, indexed by all original edges, and reduced modulo `field`.

`edge_weights` are positive weights on active edges. `anchors` supplies one
active original vertex index per connected component; its potential is zero.
The default anchors are the first vertices. Inactive vertices are omitted,
and disconnected components have independent phase origins. An integral
lifting obstruction is an error. Numerical smoothing follows an exact lift;
coordinates are not exact arithmetic, an embedding, or a canonical torus basis.
"""
function toroidal_coordinates(source::DataIngestion.FilteredComplex;scale,
        classes=[(;kind=:finite,index=1)],field=Fp(41),diagram=nothing,
        cocycles=nothing,edge_weights=nothing,anchors=nothing,budget=CoordinateBudget())
    DataIngestion.check_filtered_complex(source;throw=true)
    G=DataIngestion.graded_complex(source)
    G isa DataTypes.GradedComplex && DataTypes.parameter_dim(G)==1 ||
        throw(ArgumentError("Coordinates require an ordinary one-parameter source complex."))
    field isa PrimeField || throw(ArgumentError("Persistent coordinates require a prime field."))
    scale isa Real && isfinite(scale) || throw(ArgumentError("scale must be finite."))
    counts=DataTypes.cell_counts(G);length(counts)>=2 || throw(ArgumentError("Source has no edges."))
    offsets=cumsum([0;counts]);order=only(source.orientation)==1 ? :sublevel : :superlevel
    active(d)=findall(i->order===:sublevel ? G.grades[offsets[d+1]+i][1]<=scale : G.grades[offsets[d+1]+i][1]>=scale,1:counts[d+1])
    vs,es=active(0),active(1);fs=length(counts)>=3 ? active(2) : Int[]
    isempty(vs) && throw(ArgumentError("No active vertices at this scale."))
    D0=sparse(transpose(G.boundaries[1][vs,es]))
    D1=isempty(fs) ? spzeros(Int,0,length(es)) : sparse(transpose(G.boundaries[2][es,fs]))
    selected=NamedTuple[]
    if cocycles===nothing
        isempty(classes) && throw(ArgumentError("Select at least one persistent class."))
        _budget(budget,length(vs),length(es),length(fs),length(classes))
        if diagram===nothing
            diagram=OrdinaryPersistence.persistence_diagram(source;field,order,cocycles=true)
        else
            diagram.retained_source!==nothing && diagram.retained_source.source===source ||
                throw(ArgumentError("The supplied diagram must retain this same source object (source=true or persistence_diagram(source))."))
            diagram.field==field && diagram.order==order || throw(ArgumentError("Diagram field/orientation disagrees with coordinate request."))
        end
        A=zeros(BigInt,length(es),length(classes));positions=Dict(e=>i for (i,e) in enumerate(es))
        for (j,sel) in enumerate(classes)
            q=OrdinaryPersistence.persistence_cocycle(diagram;dim=1,scale,kind=sel.kind,index=sel.index)
            q.available || throw(ArgumentError("Compute the supplied diagram with cocycles=true."))
            if q.source_cochain!==nothing
                # Retractions pull cochains back to the original source. The
                # computed complex's local edge numbers are not source IDs.
                for e in es
                    A[positions[e],j]=BigInt(q.source_cochain.coefficients[e].val)
                end
            else
                for (e,a) in zip(q.cochain.cell_indices,q.cochain.coefficients)
                    haskey(positions,e) || throw(ArgumentError("Cocycle refers to an inactive source edge."))
                    A[positions[e],j]=BigInt(a)
                end
            end
            push!(selected,(;kind=sel.kind,index=Int(sel.index),interval=q.interval))
        end
    else
        diagram===nothing || throw(ArgumentError("Supply either a diagram or explicit cocycles."))
        cocycles isa AbstractMatrix && size(cocycles,1)==counts[2] && size(cocycles,2)>0 ||
            throw(ArgumentError("cocycles must have one row per original edge and at least one column."))
        all(x->x isa Integer,cocycles) || throw(ArgumentError("Cocycle coefficients must be integers."))
        _budget(budget,length(vs),length(es),length(fs),size(cocycles,2))
        A=BigInt.(cocycles[es,:]);selected=NamedTuple[(;kind=:supplied,index=j,interval=nothing) for j in axes(A,2)]
    end
    A=mod.(A,BigInt(field.p));balanced=map(a->2a>field.p ? a-field.p : a,A)
    lift,method=_integer_lift(D1,balanced,field.p,budget)
    comps,firsts=_components(D0)
    chosen=anchors===nothing ? vs[firsts] : collect(anchors)
    length(chosen)==length(firsts) && all(v->v isa Integer && !(v isa Bool) && v in vs,chosen) &&
        sort(comps[[findfirst(==(v),vs) for v in chosen]])==collect(1:length(firsts)) ||
        throw(ArgumentError("anchors must select one active vertex per connected component."))
    localanchors=Int[findfirst(==(v),vs) for v in chosen]
    weights=edge_weights===nothing ? ones(length(es)) : Float64.(edge_weights)
    length(weights)==length(es) && all(w->isfinite(w) && w>0,weights) ||
        throw(ArgumentError("edge_weights must be positive finite values on all active edges."))
    potential,harmonic=_smooth(D0,lift,weights,localanchors)
    # Exact cocycle class independence, using a spanning forest gauge rather
    # than a dense edge-by-vertex rank calculation.
    K=coeff_type(field);gauged=K.(A);known=falses(length(vs));values=zeros(K,length(vs),size(A,2))
    endpoints=_edge_vertices(D0)
    for root in firsts
        known[root]=true;queue=[root]
        while !isempty(queue)
            v=pop!(queue)
            for ptr in nzrange(D0,v)
                e=D0.rowval[ptr];ends=endpoints[e];length(ends)==2 || continue
                w=ends[1]==v ? ends[2] : ends[1];known[w] && continue
                values[w,:]=values[v,:]+K(-D0[e,v]).*gauged[e,:]
                known[w]=true;push!(queue,w)
            end
        end
    end
    periods=gauged-K.(D0)*values
    FieldLinAlg.rank_dim(field,periods)==size(A,2) ||
        throw(ArgumentError("Selected cocycles must be independent nonzero cohomology classes at this scale."))
    p=(;construction=:integral_harmonic_coordinates,lift=method,
        solver=:grounded_weighted_sparse_cholesky,gauge=:one_zero_potential_per_component,
        units=:turns_mod_one,basis=:selected_classes,embedding=:not_asserted,
        source_order=order,exactness=:exact_integer_lift_numerical_smoothing)
    r=CohomologicalCoordinateResult(source,diagram,scale,field,selected,vs,es,D0,D1,A,lift,potential,
        mod.(potential,1.),harmonic,weights,Int.(chosen),comps,p)
    check_coordinates(r;throw=true)
    return r
end

function toroidal_coordinates(diagram::OrdinaryPersistence.PersistenceDiagram;kwargs...)
    diagram.retained_source===nothing && throw(ArgumentError("Compute the diagram with source=true and cocycles=true to retain the coordinate domain."))
    toroidal_coordinates(diagram.retained_source.source;diagram,field=diagram.field,kwargs...)
end

function toroidal_coordinates(data,filtration::DataIngestion.AbstractFiltration;kwargs...)
    toroidal_coordinates(DataIngestion.filtered_complex(data,filtration);kwargs...)
end

"""Compute one circular coordinate from a selected H1 class. See `toroidal_coordinates`."""
function circular_coordinates(source,args...;scale,kind=:finite,index=1,kwargs...)
    r=toroidal_coordinates(source,args...;scale,classes=[(;kind,index)],kwargs...)
    size(r.values,2)==1 || throw(ArgumentError("circular_coordinates requires exactly one class."))
    r
end

function check_coordinates(r::CohomologicalCoordinateResult;throw=false)
    issues=String[]
    try
        n,k=size(r.values);e=length(r.edges)
        n==length(r.vertices)==length(r.components) && k>0 || error("Vertex/class dimensions disagree.")
        size(r.cocycles)==size(r.lifts)==size(r.harmonic)==(e,k) || error("Cochain shapes disagree.")
        size(r.delta0)==(e,n) && size(r.delta1,2)==e || error("Coboundary dimensions disagree.")
        iszero(r.delta1*r.delta0) || error("Coboundary squared is nonzero.")
        iszero(r.delta1*r.lifts) || error("Integral lift is not closed.")
        all(iszero,mod.(r.lifts-r.cocycles,BigInt(r.field.p))) || error("Integral lift changes the selected finite-field cocycle.")
        all(x->isfinite(x) && 0<=x<1,r.values) || error("Coordinates must be finite turns in [0,1).")
        size(r.potentials)==size(r.values) && r.values==mod.(r.potentials,1.) || error("Coordinates disagree with potentials.")
        all(v->v in r.vertices,r.anchors) || error("Invalid anchor identities.")
        comps,firsts=_components(r.delta0)
        comps==r.components && length(r.anchors)==length(firsts) &&
            sort(comps[[findfirst(==(v),r.vertices) for v in r.anchors]])==collect(1:length(firsts)) || error("Gauge must cover each component exactly once.")
        all(iszero,r.potentials[[findfirst(==(v),r.vertices) for v in r.anchors],:]) || error("Gauge anchor is nonzero.")
        length(r.weights)==e && all(w->isfinite(w) && w>0,r.weights) || error("Invalid smoothing weights.")
        isapprox(r.harmonic,Float64.(r.lifts)+r.delta0*r.potentials;rtol=1e-10,atol=1e-10) || error("Harmonic representative disagrees with lift and potential.")
        norm(transpose(r.delta0)*(r.weights.*r.harmonic),Inf)<=1e-8*max(1.,norm(r.weights.*Float64.(r.lifts),Inf)) || error("Weighted harmonic equation failed.")
    catch err
        err isa InterruptException && rethrow();push!(issues,sprint(showerror,err))
    end
    s=CoordinateValidationSummary(isempty(issues),issues)
    throw && !s.valid && Base.throw(ArgumentError(join(issues,"\n")))
    s
end
end
