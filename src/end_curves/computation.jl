# Diagonal kernel/cokernel, exact spread-curve decompositions and Koszul Betti data.
function _grid(M,origin,budget)
    M.field isa Union{QQField,PrimeField} || throw(ArgumentError("end-curves require QQ or a prime field"))
    Q=M.Q
    Q isa Union{ProductOfChainsPoset{2},GridPoset{2}} || throw(ArgumentError("prepare a two-dimensional finite lattice grid module before computing end-curves"))
    m,n=Q.sizes;b=budget.decomposition
    _bound((big(m)+1)*(big(n)+1),b.max_vertices,"vertices including zero halo")
    _bound(sum(big,M.dims;init=big(0)),b.max_dimension,"total dimension")
    _bound((big(m)+1)^2*(big(n)+1)^2,b.max_matrix_entries,"padded order matrix")
    entries=sum(x->big(x)^2,M.dims;init=big(0))
    entries+=sum(big(M.dims[u])*M.dims[v] for (u,v) in cover_edges(Q);init=big(0))
    _bound(8entries,b.max_matrix_entries,"input and diagonal algebra")
    orientation=Q isa ProductOfChainsPoset ? Q.orientation : (1,1)
    if Q isa GridPoset
        for xs in Q.coords
            all(x->x isa Union{Integer,Rational} && !(x isa Bool) && isinteger(x),xs) &&
                all(xs[k+1]-xs[k]==1 for k in 1:length(xs)-1) ||
                throw(ArgumentError("end-curves require consecutive integer lattice coordinates; nonuniform samples need an explicit lattice model"))
        end
        origin===nothing && (origin=(Int(first(Q.coords[1])),Int(first(Q.coords[2]))))
    end
    origin===nothing && (origin=(0,0))
    shape,o=_shape_origin((m,n),origin)
    ii=orientation[1]==1 ? (1:m) : (m:-1:1);jj=orientation[2]==1 ? (1:n) : (n:-1:1)
    ids=[i+(j-1)*m for i in ii,j in jj]
    check_module_functoriality(M;throw=true)
    (;shape,origin=o,orientation,ids)
end
function _count_terms(M,g)
    m,n=g.shape;ids=g.ids;F=M.field
    rx=ry=rxy=0
    for j in 1:n,i in 1:m
        i<m && (rx+=FieldLinAlg.rank(F,map_leq(M,ids[i,j],ids[i+1,j])))
        j<n && (ry+=FieldLinAlg.rank(F,map_leq(M,ids[i,j],ids[i,j+1])))
        i<m && j<n && (rxy+=FieldLinAlg.rank(F,map_leq(M,ids[i,j],ids[i+1,j+1])))
    end
    (;dimension=sum(M.dims),x_rank=rx,y_rank=ry,xy_rank=rxy)
end
_count(t)=t.dimension-t.x_rank-t.y_rank+t.xy_rank
"""Compute `dim M - rank x - rank y + rank xy` on a finite lattice grid,
extended by zero. This cheap query does not compute the curves. Product grids
use unit coordinates in module order; GridPoset coordinates must be consecutive
integers. The invariant depends on this declared lattice model.
"""
function two_parameter_count(M::PModule;origin=nothing,budget::EndCurveBudget=EndCurveBudget())
    _count(_count_terms(M,_grid(M,origin,budget)))
end
two_parameter_count(e::Results.EncodingResult;kwargs...)=two_parameter_count(Results.encoding_module(e);kwargs...)

function _diagonal(M::PModule{K},g) where {K}
    m,n=g.shape;a,b=m+1,n+1;Q=ProductOfChainsPoset((a,b));ids=g.ids
    original(i,j)=1<=i<=m && 1<=j<=n ? ids[i,j] : 0
    labels=[original(i,j) for j in 1:b for i in 1:a]
    shifted=[original(i-1,j-1) for j in 1:b for i in 1:a]
    dims(ls)=[k==0 ? 0 : M.dims[k] for k in ls]
    function copied(ls)
        ds=dims(ls)
        maps=Dict((u,v)=>(ls[u]==0 || ls[v]==0 ? zeros(K,ds[v],ds[u]) : Matrix(map_leq(M,ls[u],ls[v]))) for (u,v) in cover_edges(Q))
        PModule{K}(Q,ds,maps;field=M.field)
    end
    target=copied(labels);source=copied(shifted)
    maps=[labels[q]==0 || shifted[q]==0 ? zeros(K,target.dims[q],source.dims[q]) :
        Matrix(map_leq(M,shifted[q],labels[q])) for q in eachindex(labels)]
    PMorphism(source,target,maps)
end

function _betti(M::PModule{K}) where {K}
    a,b=M.Q.sizes;F=M.field;data=zeros(Int,a,b,3)
    dim(i,j)=i<1 || j<1 ? 0 : M.dims[i+(j-1)*a]
    function edge(i,j,k,l)
        i<1 || j<1 || k<1 || l<1 ? zeros(K,dim(k,l),dim(i,j)) : Matrix(map_leq(M,i+(j-1)*a,k+(l-1)*a))
    end
    for j in 1:b,i in 1:a
        d0=hcat(edge(i-1,j,i,j),edge(i,j-1,i,j))
        d1=vcat(edge(i-1,j-1,i-1,j),-edge(i-1,j-1,i,j-1))
        data[i,j,1]=dim(i,j)-FieldLinAlg.rank(F,d0)
        data[i,j,3]=dim(i-1,j-1)-FieldLinAlg.rank(F,d1)
        data[i,j,2]=data[i,j,1]+data[i,j,3]-dim(i,j)+dim(i-1,j)+dim(i,j-1)-dim(i-1,j-1)
    end
    data
end
function _curves(d,kind,origin)
    a=first(IntervalDecompositions.source_module(d).Q.sizes)
    [EndCurve(kind,[(origin[1]+(mod1(q,a)-1),origin[2]+(cld(q,a)-1)) for q in vertices];multiplicity=m)
        for (vertices,m) in zip(IntervalDecompositions.interval_family(d),IntervalDecompositions.interval_multiplicities(d))]
end
"""Compute the birth and death curve multisets of a finite lattice module with
zero extension. Births decompose `coker(xy)`; deaths decompose `ker(xy)` in the
shifted grades, including the upper/right halo. Exact QQ and prime fields are
supported. `witnesses=true` retains the diagonal sequence and splitting maps.
The bounded exact interval-summand algorithm is used for decomposition; no
linear-time decomposition or canonical boundary-component grouping is claimed.
"""
function end_curves(M::PModule;origin=nothing,witnesses::Bool=false,budget::EndCurveBudget=EndCurveBudget())
    g=_grid(M,origin,budget);terms=_count_terms(M,g);f=_diagonal(M,g)
    D,inc=AbelianCategories.kernel_with_inclusion(f)
    B,prj=AbelianCategories.cokernel_with_projection(f)
    bd=IntervalDecompositions.interval_decomposition(B;witnesses,budget=budget.decomposition)
    dd=IntervalDecompositions.interval_decomposition(D;witnesses,budget=budget.decomposition)
    IntervalDecompositions.is_interval_decomposable(bd)===true && IntervalDecompositions.is_interval_decomposable(dd)===true ||
        error("an ephemeral module failed to decompose into spread curves")
    births=_coalesce(_curves(bd,:birth,g.origin),:birth);deaths=_coalesce(_curves(dd,:death,g.origin),:death)
    sum(c->c.multiplicity,births;init=0)==sum(c->c.multiplicity,deaths;init=0)==_count(terms) ||
        error("curve multiplicities disagree with the independent two-parameter count")
    w=witnesses ? (;diagonal=f,birth_projection=prj,death_inclusion=inc,birth_decomposition=bd,death_decomposition=dd) : nothing
    EndCurveResult(births,deaths,g.shape,g.origin,M.field,_betti(f.cod),terms,w,:computed,g.orientation)
end
end_curves(e::Results.EncodingResult;kwargs...)=end_curves(Results.encoding_module(e);kwargs...)

"""Reconstruct the ephemeral birth or death module from the curve multiset.
This is not reconstruction of the original module, which end-curves do not determine.
"""
function end_curve_module(r::EndCurveResult;kind::Symbol,budget::EndCurveBudget=EndCurveBudget())
    kind in (:birth,:death) || throw(ArgumentError("kind must be :birth or :death"))
    curves=kind===:birth ? r.births : r.deaths;a,b=r.shape.+1
    _bound(big(a)*b,budget.decomposition.max_vertices,"reconstruction vertices")
    _bound(sum(c->big(length(c.vertices))*c.multiplicity,curves;init=big(0)),budget.decomposition.max_dimension,"reconstruction dimension")
    _bound(big(two_parameter_count(r))^2*2a*b,budget.decomposition.max_matrix_entries,"reconstruction matrices")
    Q=ProductOfChainsPoset((a,b));K=coeff_type(r.field)
    active=[[(k,l) for (k,c) in enumerate(curves) if (r.origin[1]+i-1,r.origin[2]+j-1) in c.vertices for l in 1:c.multiplicity] for j in 1:b for i in 1:a]
    maps=Dict((u,v)=>K[s==t for t in active[v],s in active[u]] for (u,v) in cover_edges(Q))
    PModule{K}(Q,length.(active),maps;field=r.field)
end

function check_end_curves(r::EndCurveResult;throw::Bool=false,budget::EndCurveBudget=EndCurveBudget())
    issues=String[];checked=false
    try
        s,o=_shape_origin(r.shape,r.origin)
        r.field isa Union{QQField,PrimeField} || error("unsupported coefficient field")
        r.scope in (:supplied,:computed) || error("invalid result scope")
        all(x->x in (-1,1),r.orientation) || error("invalid source orientation")
        for (kind,curves) in ((:birth,r.births),(:death,r.deaths))
            curves==_coalesce(curves,kind) || error("curve multiset is not canonical")
            offset=kind===:birth ? 0 : 1
            for c in curves,p in c.vertices,k in 1:2
                o[k]+offset<=p[k]<=o[k]+(s[k]-1+offset) || error("$kind curve lies outside its declared lattice support")
            end
        end
        n=two_parameter_count(r)
        n==sum(c->big(c.multiplicity),r.deaths;init=big(0)) || error("birth and death counts differ")
        r.count_terms===nothing || n==_count(r.count_terms) || error("rank formula disagrees with curves")
        r.betti===nothing || (size(r.betti)==(s[1]+1,s[2]+1,3) && all(>=(0),r.betti)) || error("invalid Betti data")
        if r.witnesses!==nothing
            w=r.witnesses;check_morphism(w.diagonal;throw=true)
            check_morphism(w.birth_projection;throw=true);check_morphism(w.death_inclusion;throw=true)
            w.birth_projection.dom===w.diagonal.cod && w.death_inclusion.cod===w.diagonal.dom || error("diagonal sequence domains disagree")
            B=w.birth_projection.cod;D=w.death_inclusion.dom;F=r.field
            IntervalDecompositions.source_module(w.birth_decomposition)===B &&
                IntervalDecompositions.source_module(w.death_decomposition)===D || error("decomposition sources disagree with the diagonal sequence")
            for q in eachindex(B.dims)
                f=w.diagonal.comps[q];inc=w.death_inclusion.comps[q];prj=w.birth_projection.comps[q]
                all(iszero,f*inc) && all(iszero,prj*f) || error("diagonal sequence does not compose to zero")
                FieldLinAlg.rank(F,inc)==D.dims[q]==size(f,2)-FieldLinAlg.rank(F,f) || error("incorrect diagonal kernel")
                FieldLinAlg.rank(F,prj)==B.dims[q]==size(f,1)-FieldLinAlg.rank(F,f) || error("incorrect diagonal cokernel")
            end
            for (d,kind,cs) in ((w.birth_decomposition,:birth,r.births),(w.death_decomposition,:death,r.deaths))
                IntervalDecompositions.check_interval_decomposition(d;budget=budget.decomposition,throw=true)
                _coalesce(_curves(d,kind,r.origin),kind)==cs || error("curve supports disagree with splitting witnesses")
            end
            _betti(w.diagonal.cod)==r.betti || error("Betti data disagree with the source")
            checked=true
        end
    catch err
        err isa InterruptException && rethrow()
        push!(issues,sprint(showerror,err))
    end
    result=EndCurveValidationSummary(isempty(issues),issues,checked)
    throw && !result.valid && Base.throw(ArgumentError(join(issues,"; ")))
    result
end
