# Ordered simplicial Alexander--Whitney products in exact quotient coordinates.
# Product-image/partition invariants: Dey and Rathod, SoCG 2024,
# https://doi.org/10.4230/LIPIcs.SoCG.2024.50 . This is a direct exact
# cochain/quotient implementation, not their specialized incremental algorithm.

function _cup_order(order)
    order in (:sublevel,:superlevel) || throw(ArgumentError("order must be :sublevel or :superlevel"))
    order
end
function _cup_parameters(parameters,order)
    _cup_order(order)
    isempty(parameters) && throw(ArgumentError("at least one parameter is required"))
    all(t->t isa Real && !(t isa Bool) && isfinite(t),parameters) ||
        throw(ArgumentError("parameters must be finite real numbers"))
    T=mapreduce(typeof,promote_type,parameters)
    ts=T[t for t in parameters]
    all(ts[i]==parameters[i] for i in eachindex(ts)) || throw(ArgumentError("parameter conversion loses precision"))
    all(i->order===:sublevel ? ts[i]<ts[i+1] : ts[i]>ts[i+1],1:length(ts)-1) ||
        throw(ArgumentError("parameters must be strictly ordered in filtration order"))
    ts
end

function _simplicial_filtration(simplices,grades,order,budget)
    _cup_order(order)
    length(simplices)==length(grades) || throw(ArgumentError("one grade per supplied simplex is required"))
    length(simplices)<=budget.max_cells || throw(ArgumentError("supplied simplices exceed max_cells"))
    all(t->t isa Real && !(t isa Bool) && isfinite(t),grades) || throw(ArgumentError("grades must be finite real numbers"))
    T=isempty(grades) ? Int : mapreduce(typeof,promote_type,grades)
    gs=T[iszero(t) ? zero(T) : t for t in grades]
    all(gs[i]==grades[i] for i in eachindex(gs)) || throw(ArgumentError("grade conversion loses precision"))
    faces=Dict{Tuple{Vararg{Int}},T}()
    for (s,t) in zip(simplices,gs)
        (s isa AbstractVector || s isa Tuple) && !isempty(s) && allunique(s) &&
            all(v->v isa Integer && !(v isa Bool) && 0<v<=typemax(Int),s) ||
            throw(ArgumentError("simplices must contain distinct positive integer vertex IDs"))
        length(s)<8sizeof(Int)-1 && (big(1)<<length(s))-1<=budget.max_cells ||
            throw(ArgumentError("simplex face closure exceeds max_cells"))
        simplex=Tuple(sort!(Int.(collect(s))))
        # A repeated face receives its earliest grade in filtration order.
        queue=Tuple{Vararg{Int}}[simplex]
        while !isempty(queue)
            f=pop!(queue)
            if haskey(faces,f)
                (order===:sublevel ? faces[f]<=t : faces[f]>=t) && continue
            else
                length(faces)<budget.max_cells || throw(ArgumentError("face closure exceeds max_cells"))
            end
            faces[f]=t
            if length(f)>1
                for j in eachindex(f);push!(queue,Tuple(f[k] for k in eachindex(f) if k!=j));end
            end
        end
    end
    ordered=sort!(collect(keys(faces));by=s->(length(s),s))
    ordered,T[faces[s] for s in ordered]
end

function _quotient_coordinates(F,C::_SimplicialCochains,Y,d)
    0<=d<length(C.cells) || throw(ArgumentError("cochain degree outside stored complex"))
    size(Y,1)==length(C.cells[d+1]) || throw(ArgumentError("cochain dimension disagrees with source cells"))
    all(iszero,C.deltas[d+1]*Y) || throw(ArgumentError("cochain is not a cocycle"))
    X=FieldLinAlg.solve_fullcolumn(F,C.quotient_bases[d+1],Y;cache=false)
    X[size(C.boundaries[d+1],2)+1:end,:]
end

function _cochain_product(C::_SimplicialCochains{K},x,y,p,q) where {K}
    p+q>=length(C.cells) && return K[]
    left=Dict(s=>i for (i,s) in enumerate(C.cells[p+1]))
    right=Dict(s=>i for (i,s) in enumerate(C.cells[q+1]))
    out=zeros(K,length(C.cells[p+q+1]))
    for (i,s) in enumerate(C.cells[p+q+1])
        out[i]=x[left[s[1:p+1]]]*y[right[s[p+1:end]]]
    end
    out
end

function _simplicial_algebra(simplices,F,w)
    K=coeff_type(F);D=maximum(length,simplices;init=1)-1
    cells=[Tuple{Vararg{Int}}[s for s in simplices if length(s)==d+1] for d in 0:D]
    ns=length.(cells)
    # Dense exact quotients are bounded before allocating matrices or bases.
    _spend!(w;entries=6sum(big(n)^2 for n in ns)+2sum((big(ns[i])*ns[i+1] for i in 1:D);init=big(0)))
    deltas=Matrix{K}[]
    for d in 0:D
        M=zeros(K,d==D ? 0 : ns[d+2],ns[d+1])
        if d<D
            lookup=Dict(s=>i for (i,s) in enumerate(cells[d+1]))
            for (i,s) in enumerate(cells[d+2]),j in eachindex(s)
                M[i,lookup[Tuple(s[k] for k in eachindex(s) if k!=j)]]=isodd(j) ? one(K) : -one(K)
            end
        end
        push!(deltas,M)
    end
    boundaries=Matrix{K}[];reps=Matrix{K}[];bases=Matrix{K}[]
    for d in 0:D
        B=d==0 ? zeros(K,ns[1],0) : Matrix{K}(FieldLinAlg.colspace(F,deltas[d]))
        Z=Matrix{K}(FieldLinAlg.nullspace(F,deltas[d+1]))
        all(iszero,deltas[d+1]*B) || error("simplicial coboundary does not square to zero")
        together=hcat(B,Z);_,pivs=FieldLinAlg.rref(F,together)
        H=together[:,[j for j in pivs if j>size(B,2)]]
        push!(boundaries,B);push!(reps,H);push!(bases,hcat(B,H))
    end
    C=_SimplicialCochains(cells,deltas,boundaries,reps,bases)
    dims=[size(H,2) for H in reps];tables=Dict{Tuple{Int,Int},Matrix{K}}()
    for p in 0:D,q in 0:D-p
        n=big(dims[p+1])*dims[q+1]
        _spend!(w;entries=n*(ns[p+q+1]+dims[p+q+1]),products=n)
        # Omit zero-dimensional factors but retain zero products with nonzero sources.
        n==0 && continue
        values=zeros(K,ns[p+q+1],Int(n))
        for j in 1:dims[q+1],i in 1:dims[p+1]
            values[:,i+(j-1)*dims[p+1]]=_cochain_product(C,view(reps[p+1],:,i),view(reps[q+1],:,j),p,q)
        end
        tables[(p,q)]=_quotient_coordinates(F,C,values,p+q)
    end
    unit=vec(_quotient_coordinates(F,C,ones(K,ns[1],1),0))
    A=CohomologyAlgebra(dims,tables,unit,F,C)
    _cup_report(_algebra_issues(A,w),true);A
end

"""
    cohomology_algebra(simplices; field=F2(), budget=CupProductBudget())

Compute the unreduced cohomology ring of a finite abstract simplicial complex.
Input may contain maximal faces; all nonempty faces are included. Positive
integer vertex IDs are ordered increasingly to orient simplices. Exact QQ and
prime fields are supported. Cochain products use the front/back-face
Alexander--Whitney formula, reduced modulo coboundaries. No products are
inferred from a bare module or a generic cellular boundary matrix.
"""
function cohomology_algebra(simplices::AbstractVector;field=F2(),budget=CupProductBudget())
    F=_cup_field(field)
    cells,_=_simplicial_filtration(simplices,zeros(Int,length(simplices)),:sublevel,budget)
    _simplicial_algebra(cells,F,_CupWork(budget))
end

function _source_cochains(A)
    A.cochains===nothing && throw(ArgumentError("no simplicial cochains are retained for this supplied algebra"))
    A.cochains
end
"""Copy the ordered simplex labels, cocycle representatives (columns), and
coboundary basis in a degree. These are noncanonical bases in original
cochain coordinates, not persistent interval representatives."""
function cohomology_representatives(A::CohomologyAlgebra;degree)
    d=_cup_int(degree);C=_source_cochains(A)
    d<length(C.cells) || throw(ArgumentError("degree outside stored complex"))
    (;cells=copy(C.cells[d+1]),representatives=copy(C.representatives[d+1]),
      coboundaries=copy(C.boundaries[d+1]),coboundary=copy(C.deltas[d+1]))
end
"""Project cocycle columns into the retained cohomology basis. Rejects noncocycles.
Adding any coboundary leaves these coordinates unchanged."""
function cohomology_coordinates(A::CohomologyAlgebra,Y::AbstractVecOrMat;degree)
    d=_cup_int(degree);C=_source_cochains(A)
    input=_cup_coefficients(field(A),Y)
    result=_quotient_coordinates(field(A),C,Y isa AbstractVector ? reshape(input,:,1) : input,d)
    Y isa AbstractVector ? vec(result) : result
end
"""Alexander--Whitney product of arbitrary cochains on the retained ordered
simplices. This operation is defined before taking cohomology; cochains need
not be cocycles and do not in general commute. Degree-zero units are included."""
function cochain_cup(A::CohomologyAlgebra,x::AbstractVector,y::AbstractVector;
                     left_degree,right_degree)
    p=_cup_int(left_degree);q=_cup_int(right_degree);C=_source_cochains(A)
    p<length(C.cells) && q<length(C.cells) || throw(ArgumentError("cochain degree outside stored complex"))
    length(x)==length(C.cells[p+1]) && length(y)==length(C.cells[q+1]) ||
        throw(ArgumentError("cochain dimensions disagree"))
    _cochain_product(C,_cup_coefficients(field(A),x),_cup_coefficients(field(A),y),p,q)
end
