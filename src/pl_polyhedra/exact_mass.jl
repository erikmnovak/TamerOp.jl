# Exact rational volume by pulling triangulation of the clipped vertex set.
# This route is intended for bounded, modest-dimensional selected queries.
# Both vertex enumeration and face traversal have explicit work limits.
import ..FieldLinAlg as _VolumeLinAlg
function _exact_poly_volume(hp,a,b,work,max_states,max_entries)
    n=hp.n;m=size(hp.A,1)
    count=_binomial_big(m+2n,n)
    work[]+count<=max_states || throw(ArgumentError("exact polyhedral vertex budget exceeded"))
    work[]+=Int(count)
    (big(m)+2n)*(n+1)<=max_entries || throw(ArgumentError("exact polyhedral workspace budget exceeded"))
    verts=_vertices_of_hpoly_in_box(hp,a,b;max_combinations=Int(count),
        max_vertices=div(max_entries,max(n+m+2n,1)))
    verts===nothing && throw(ArgumentError("exact polyhedral vertex storage budget exceeded"))
    length(verts)>=n+1 || return zero(QQ)
    V=reduce(hcat,collect.(sort!(collect(verts))))
    _VolumeLinAlg.rank(QQField(),V[:,2:end].-V[:,1])==n || return zero(QQ)
    A=vcat(hp.A,Matrix{QQ}(I,n,n),-Matrix{QQ}(I,n,n));rhs=vcat(_relaxed_b(hp),b,-a)
    big(length(V))+big(size(A,1))*size(V,2)<=max_entries ||
        throw(ArgumentError("exact polyhedral face storage budget exceeded"))
    tight=[findall(j->dot(A[i,:],V[:,j])==rhs[i],axes(V,2)) for i in axes(A,1)]
    function triangulate(ids,d,prefix)
        work[]+=1;work[]<=max_states || throw(ArgumentError("exact polyhedral triangulation budget exceeded"))
        if d==0
            simplex=vcat(prefix,first(ids));base=V[:,first(simplex)]
            return abs(det(V[:,simplex[2:end]].-base))/factorial(big(n))
        end
        apex=first(ids);seen=Set{Tuple{Vararg{Int}}}();volume=zero(QQ)
        for face in tight
            sub=intersect(ids,face)
            (apex in sub || length(sub)<d || length(sub)==length(ids)) && continue
            key=Tuple(sub);key in seen && continue;push!(seen,key)
            work[]+=1;work[]<=max_states || throw(ArgumentError("exact polyhedral face budget exceeded"))
            _VolumeLinAlg.rank(QQField(),V[:,sub[2:end]].-V[:,first(sub)])==d-1 || continue
            volume+=triangulate(sub,d-1,vcat(prefix,apex))
        end
        volume
    end
    triangulate(collect(axes(V,2)),n,Int[])
end
function region_weights_exact(pi::PLEncodingMap;box,max_states=1_000_000,max_entries=8_000_000)
    a,b=map(x->QQ.(collect(x)),box)
    length(a)==length(b)==pi.n && all(isfinite,a) && all(isfinite,b) && all(a .<= b) ||
        throw(ArgumentError("supply finite ordered box corners of the encoding dimension"))
    max_states>=0 && max_entries>=0 || throw(ArgumentError("geometry budgets must be nonnegative"))
    length(pi.regions)<=max_entries || throw(ArgumentError("exact volume output budget exceeded"))
    any(a .== b) && return zeros(QQ,length(pi.regions))
    work=Ref(0)
    [_exact_poly_volume(hp,a,b,work,max_states,max_entries) for hp in pi.regions]
end
region_weights_exact(pi::CompiledEncoding{<:PLEncodingMap};kwargs...)=
    region_weights_exact(_unwrap_encoding(pi);kwargs...)
