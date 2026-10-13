# Graph minimization: Algorithms 1-3 of Luo--Morozov--Scoccola,
# arXiv:2410.22242v3. Grades are labels of the declared finite indexing poset.
@inline function _h0_find!(parent,x)
    while parent[x] != x
        parent[x]=parent[parent[x]]; x=parent[x]
    end
    return x
end
function _h0_collapse(P,vg,edges,eg)
    n=length(vg); parent=collect(1:n); sizes=ones(Int,n)
    removed=falses(length(edges))
    for (i,(u,v)) in enumerate(edges)
        vg[u]==vg[v]==eg[i] || continue
        a,b=_h0_find!(parent,u),_h0_find!(parent,v)
        a==b && continue
        if sizes[a]<sizes[b]; a,b=b,a; end
        parent[b]=a; sizes[a]+=sizes[b]; removed[i]=true
    end
    roots=[_h0_find!(parent,v) for v in 1:n]
    upward=[Tuple{Int,Int}[] for _ in 1:n]; indegree=zeros(Int,n)
    for (i,(u,v)) in enumerate(edges)
        removed[i] && continue
        a,b=roots[u],roots[v]
        a==b && continue
        if eg[i]==vg[a]
            push!(upward[b],(a,i)); indegree[a]+=1
        elseif eg[i]==vg[b]
            push!(upward[a],(b,i)); indegree[b]+=1
        end
    end
    representatives=zeros(Int,n)
    for r in 1:n
        roots[r]==r && indegree[r]==0 || continue
        stack=[(r,0)]
        while !isempty(stack)
            v,e=pop!(stack)
            representatives[v]!=0 && continue
            representatives[v]=r
            e==0 || (removed[e]=true)
            append!(stack,upward[v])
        end
    end
    original=[representatives[roots[v]] for v in 1:n]
    all(!iszero,original) || error("H0 collapse: missing root")
    survivors=findall(v->original[v]==v,1:n)
    index=zeros(Int,n)
    for (i,v) in enumerate(survivors); index[v]=i; end
    images=[index[v] for v in original]
    kept=findall(!,removed)
    return survivors,images,[(images[edges[i][1]],images[edges[i][2]]) for i in kept],eg[kept]
end

_h0_grid_shape(::AbstractPoset)=nothing
_h0_grid_shape(P::ProductOfChainsPoset{N}) where {N}=(P.sizes,P.strides,P.orientation)
_h0_grid_shape(P::GridPoset{N}) where {N}=(P.sizes,P.strides,ntuple(_->1,N))
@inline function _h0_coords(shape,v)
    sizes,strides,orientation=shape
    return ntuple(i->begin
        c=div(v-1,strides[i])%sizes[i]+1
        orientation[i]==1 ? c : sizes[i]+1-c
    end,length(sizes))
end
@inline function _h0_label(shape,coords)
    sizes,strides,orientation=shape
    return 1+sum(((orientation[i]==1 ? coords[i] : sizes[i]+1-coords[i])-1)*strides[i] for i in eachindex(sizes))
end

function _h0_reachable(P,adj,edges,grades,active,u,v,g,skip,seen,queue,stamp)
    u==v && return true
    empty!(queue); push!(queue,u); seen[u]=stamp
    cursor=1
    while cursor<=length(queue)
        a=queue[cursor]; cursor+=1
        for e in adj[a]
            (e==skip || !active[e] || !leq(P,grades[e],g)) && continue
            x,y=edges[e]; b=x==a ? y : x
            b==v && return true
            if seen[b]!=stamp
                seen[b]=stamp; push!(queue,b)
            end
        end
    end
    return false
end
function _h0_minimal_edges(P,n,edges,grades)
    adj=[Int[] for _ in 1:n]
    for (i,(u,v)) in enumerate(edges)
        push!(adj[u],i); push!(adj[v],i)
    end
    active=trues(length(edges)); seen=zeros(Int,n); queue=Int[]
    for (i,(u,v)) in enumerate(edges)
        if _h0_reachable(P,adj,edges,grades,active,u,v,grades[i],i,seen,queue,i)
            active[i]=false
        end
    end
    return findall(active)
end

function _h0_sweep(P,n,edges,grades; witnesses=false, max_witness_entries=10_000_000)
    shape=_h0_grid_shape(P)
    xy=[length(shape[1])==1 ? (_h0_coords(shape,g)[1],1) : _h0_coords(shape,g) for g in grades]
    order=sortperm(eachindex(grades);by=i->(xy[i],i))
    T=_H0Forest(vcat(zeros(Int,n),[c[2] for c in xy]))
    kept=Int[]; second=Int[]; relation_index=zeros(Int,length(edges))
    si,sj,sv=Int[],Int[],Int[]
    for e in order
        u,v=edges[e]
        u==v && continue
        connected=_h0_findroot!(T,u)==_h0_findroot!(T,v)
        old=0
        if connected
            _h0_makeroot!(T,u); _h0_access!(T,v)
            old=T.maximum[v]
            T.weight[old]<=xy[e][2] && continue
        end
        push!(kept,e); relation_index[e]=length(kept)
        if connected
            push!(second,_h0_label(shape,(xy[e][1],T.weight[old])))
            if witnesses
                path=_h0_path_nodes(T,v)
                length(sv)+div(length(path),2)+1<=max_witness_entries || throw(ArgumentError("h0_presentation: syzygy entry budget exceeded"))
                for j in 2:2:length(path)-1
                    edge=path[j]-n
                    sign=edges[edge]==(path[j-1],path[j+1]) ? -1 : 1
                    push!(si,relation_index[edge]); push!(sj,length(second)); push!(sv,sign)
                end
                push!(si,length(kept)); push!(sj,length(second)); push!(sv,1)
            end
            a,b=edges[old-n]
            _h0_cut!(T,a,old); _h0_cut!(T,b,old)
        end
        _h0_link!(T,u,n+e); _h0_link!(T,n+e,v)
    end
    S=witnesses ? sparse(si,sj,sv,length(kept),length(second)) : nothing
    return kept,second,S
end
