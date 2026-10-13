# Right-filtration propagation, Carlsson--de Silva, section 4.3:
# https://arxiv.org/abs/0812.0197 . Forward cokernel births append; backward
# kernel births prepend. The order is a filtration order, not chronological.

function _reduce_columns(A::Matrix{K}) where {K}
    R=copy(A); T=Matrix{K}(I,size(A,2),size(A,2))
    owners=zeros(Int,size(A,1)); pivots=zeros(Int,size(A,2))
    for j in axes(R,2)
        while true
            p=findlast(!iszero,view(R,:,j))
            p===nothing && break
            k=owners[p]
            if k==0
                owners[p]=j; pivots[j]=p
                break
            end
            c=R[p,j]/R[p,k]
            for i in axes(R,1); R[i,j]-=c*R[i,k]; end
            for i in axes(T,1); T[i,j]-=c*T[i,k]; end
        end
    end
    R,T,pivots
end

# Apply a filtration-preserving triangular basis change to all retained
# occurrences, restricting each column to its own interval support.
function _change_histories!(histories, births, U)
    histories===nothing && return
    old=copy(histories)
    for j in eachindex(births)
        histories[j]=[begin
            v=zeros(eltype(U),length(old[j][t-births[j]+1]))
            for k in 1:j
                iszero(U[k,j]) && continue
                t<births[k] && continue
                v .+= U[k,j].*old[k][t-births[k]+1]
            end
            v
        end for t in births[j]:births[j]+length(old[j])-1]
    end
end

function _finish!(supports,retained,births,histories,indices,last)
    for j in indices
        push!(supports,(births[j],last))
        retained===nothing || push!(retained,histories[j])
    end
end

function zigzag_persistence(input::ZigzagModule; representatives::Bool=false,
                            budget::ZigzagBudget=ZigzagBudget())
    # Copy and validate before computation so subsequent caller mutation cannot
    # change either the input contract or the retained witness.
    z=ZigzagModule(input.dims,input.maps;directions=input.directions,field=input.field,times=input.times,budget)
    _decompose(z,representatives,budget,nothing)
end

function _decompose(z::ZigzagModule{K},reps,budget,source) where {K}
    n=length(z.dims); F=z.field
    # At each vertex exactly dim(V) vectors of length dim(V) are retained.
    reps && _bound(sum(big(d)^2 for d in z.dims),budget.max_representative_entries,"representative entries")
    B=Matrix{K}(I,z.dims[1],z.dims[1]); births=fill(1,z.dims[1])
    histories=reps ? [[copy(B[:,j])] for j in axes(B,2)] : nothing
    retained=reps ? Vector{Vector{K}}[] : nothing
    supports=Tuple{Int,Int}[]
    for e in 1:n-1
        A=z.maps[e]
        if z.directions[e]===:forward
            R,U,pivots=_reduce_columns(A*B)
            keep=findall(!iszero,pivots); dead=findall(iszero,pivots)
            _change_histories!(histories,births,U)
            _finish!(supports,retained,births,histories,dead,e)
            C=R[:,keep]
            # Complete without altering the surviving interval columns.
            augmented=hcat(C,Matrix{K}(I,z.dims[e+1],z.dims[e+1]))
            _,cols=FieldLinAlg.rref(F,augmented)
            extra=Int[j for j in cols if j>length(keep)]
            B=hcat(C,augmented[:,extra])
            births=[births[keep];fill(e+1,length(extra))]
            if reps
                histories=histories[keep]
                for j in eachindex(keep); push!(histories[j],copy(C[:,j])); end
                append!(histories,[[copy(augmented[:,j])] for j in extra])
            end
        else
            D=Matrix{K}(FieldLinAlg.solve_fullcolumn(F,B,A))
            R,T,pivots=_reduce_columns(D)
            cols=sort(findall(!iszero,pivots);by=j->pivots[j])
            keep=pivots[cols]; dead=setdiff(collect(eachindex(births)),keep)
            U=Matrix{K}(I,length(births),length(births))
            lifts=zeros(K,z.dims[e+1],length(cols))
            for (j,c) in enumerate(cols)
                p=pivots[c]; scale=R[p,c]
                U[:,p]=R[:,c]/scale
                lifts[:,j]=T[:,c]/scale
            end
            _change_histories!(histories,births,U)
            _finish!(supports,retained,births,histories,dead,e)
            kernel=T[:,findall(iszero,pivots)]
            B=hcat(kernel,lifts)
            births=[fill(e+1,size(kernel,2));births[keep]]
            if reps
                survivors=histories[keep]
                for j in eachindex(keep); push!(survivors[j],copy(lifts[:,j])); end
                histories=[[[copy(kernel[:,j])] for j in axes(kernel,2)];survivors]
            end
        end
    end
    _finish!(supports,retained,births,histories,eachindex(births),n)
    order=sortperm(supports)
    ZigzagResult{K,typeof(z),typeof(source)}(z,supports[order],reps ? retained[order] : nothing,source)
end

function _check_representatives(r::ZigzagResult{K},budget) where {K}
    z=r.diagram; F=z.field; n=length(z.dims)
    _bound(sum(big(d)^2 for d in z.dims),budget.max_representative_entries,"representative entries")
    length(r.representatives)==length(r.supports) || throw(ArgumentError("representative count disagrees"))
    for (j,(a,b)) in enumerate(r.supports)
        h=r.representatives[j]
        length(h)==b-a+1 || throw(ArgumentError("representative occurrence count disagrees"))
        all(length(h[t-a+1])==z.dims[t] for t in a:b) || throw(ArgumentError("representative dimension disagrees"))
        for e in max(1,a-1):min(n-1,b)
            s=z.directions[e]===:forward ? e : e+1
            a<=s<=b || continue
            x=a<=e<=b ? h[e-a+1] : zeros(K,z.dims[e])
            y=a<=e+1<=b ? h[e-a+2] : zeros(K,z.dims[e+1])
            ok=z.directions[e]===:forward ? z.maps[e]*x==y : z.maps[e]*y==x
            ok || throw(ArgumentError("interval $j fails the map equation at edge $e"))
        end
    end
    for t in 1:n
        ids=findall(ab->ab[1]<=t<=ab[2],r.supports)
        B=zeros(K,z.dims[t],length(ids))
        for (c,j) in enumerate(ids); B[:,c]=r.representatives[j][t-r.supports[j][1]+1]; end
        FieldLinAlg.rank(F,B)==z.dims[t] || throw(ArgumentError("representatives are not a basis at vertex $t"))
    end
    nothing
end
