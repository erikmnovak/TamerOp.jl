
# Sixteen forest-deletion patterns for an ordinary square. This immutable
# table contains only geometry positions, never input-dependent pairings.
const _CUBICAL_SQUARE_PATTERNS=ntuple(16) do index
    mask=index-1
    positions=Tuple(i for i in 1:4 if !iszero(mask & (1<<(i-1))))
    ntuple(i->Int8(i==5 ? length(positions) : i<=length(positions) ? positions[i] : 0),5)
end
@inline function _cubical_square_pair(g,m,localid,axis,rank)
    edge_mask=xor(m,1<<(axis-1))
    lo=g.offsets[edge_mask+1]+localid+
        div(localid-1,g.strides[m+1][axis]*g.shapes[m+1][axis])*g.strides[edge_mask+1][axis]
    hi=lo+g.strides[edge_mask+1][axis]
    @inbounds return rank[lo],rank[hi]
end


# Roots carry negative component sizes; nonroots carry a parent index.
# Compact arrays reduce graph-sweep traffic; Int remains the large-grid fallback.
@inline function _cubical_compact_root!(parent,vertex::Int)
    @inbounds while parent[vertex]>0
        next=Int(parent[vertex]);after=parent[next]
        after>0 && (parent[vertex]=after)
        vertex=next
    end
    return vertex
end

# Grid-native ordinary barcodes. No cell-coordinate catalogue or boundary
# matrix is constructed. The rectangular, nonperiodic domain is contractible.
# H0 uses the primal grid graph; the top finite bars use its dual graph with
# an exterior vertex. In 3D only the remaining square boundaries are reduced.
# This is independent code; no competitor implementation is incorporated.

struct _CubicalBarcodeGrid{N,T}
    shapes::Vector{NTuple{N,Int}}
    strides::Vector{NTuple{N,Int}}
    offsets::Vector{Int}
    grades::Vector{Vector{T}}
end

function _cubical_barcode_grid(values::AbstractArray{T,N}, input, order) where {T,N}
    dims = ntuple(a -> size(values,a) + (input === :top_cells), N)
    shapes = [ntuple(a -> dims[a] - ((m >> (a-1)) & 1), N) for m in 0:(1<<N)-1]
    strides = [ntuple(a -> prod(s[1:a-1]), N) for s in shapes]
    masks = [[m for m in 0:(1<<N)-1 if count_ones(m)==d] for d in 0:N]
    offsets = zeros(Int,1<<N)
    counts = zeros(Int,N+1)
    for d in 0:N, m in masks[d+1]
        offsets[m+1] = counts[d+1]
        counts[d+1] += prod(shapes[m+1])
    end
    grades = [Vector{T}(undef,n) for n in counts]
    increasing = order === :sublevel
    if input === :vertices
        copyto!(grades[1], vec(values))
        for m in 1:(1<<N)-1
            axis = trailing_zeros(m)+1
            parent = m & (m-1)
            dst, src = grades[count_ones(m)+1], grades[count_ones(parent)+1]
            stride = strides[parent+1]
            for (i,c) in enumerate(CartesianIndices(shapes[m+1]))
                j = offsets[parent+1] + 1 + sum((c[a]-1)*stride[a] for a in 1:N)
                @inbounds x,y = src[j],src[j+stride[axis]]
                @inbounds dst[offsets[m+1]+i] = (increasing ? x < y : x > y) ? y : x
            end
        end
    else
        copyto!(grades[end],vec(values))
        for m in (1<<N)-2:-1:0
            axis = trailing_zeros(xor(m,(1<<N)-1))+1
            parent = m | (1<<(axis-1))
            dst,src = grades[count_ones(m)+1],grades[count_ones(parent)+1]
            stride,shape = strides[parent+1],shapes[parent+1]
            for (i,c) in enumerate(CartesianIndices(shapes[m+1]))
                lo = max(1,c[axis]-1); hi = min(shape[axis],c[axis])
                j = offsets[parent+1]+1+sum(((a==axis ? lo : c[a])-1)*stride[a] for a in 1:N)
                @inbounds x,y = src[j],src[j+(hi-lo)*stride[axis]]
                @inbounds dst[offsets[m+1]+i] = (increasing ? y < x : y > x) ? y : x
            end
        end
    end
    return _CubicalBarcodeGrid(shapes,strides,offsets,grades)
end

# Stable ordering by grade, then dimension-local cell ID. Bounded integer
# grades (including exact integers stored as floats) admit counting sort.
# Other exact real types retain their ordinary comparison semantics.
function _cubical_grade_order(values::Vector{T}, order) where {T}
    n=length(values)
    cap=min(65535,max(255,n))
    if T <: Union{Integer,AbstractFloat} && all(v -> 0 <= v <= cap && isinteger(v), values)
        counts=zeros(Int, isempty(values) ? 0 : Int(maximum(values))+1)
        for v in values
            @inbounds counts[Int(v)+1]+=1
        end
        position=1
        buckets=order===:sublevel ? eachindex(counts) : reverse(eachindex(counts))
        for b in buckets
            count=counts[b];counts[b]=position;position+=count
        end
        perm=Vector{Int}(undef,n)
        for i in eachindex(values)
            b=Int(values[i])+1
            @inbounds perm[counts[b]]=i
            @inbounds counts[b]+=1
        end
        return perm
    end
    # Explicit equality preserves the filtration's signed-zero convention.
    perm=collect(1:n)
    sort!(perm;lt=(i,j)->values[i]==values[j] ? i<j :
        (order===:sublevel ? values[i]<values[j] : values[i]>values[j]))
    return perm
end

# Only edge and codimension-one blocks are requested in dimensions 1-3.
# Fixed offsets resolve their orientation, including empty thin-grid blocks.
@inline function _cubical_grid_mask(g::_CubicalBarcodeGrid{N},d,id) where {N}
    if N==1
        return 1
    elseif N==2
        return id>g.offsets[3] ? 2 : 1
    elseif d==1
        return id>g.offsets[5] ? 4 : id>g.offsets[3] ? 2 : 1
    else
        return id>g.offsets[7] ? 6 : id>g.offsets[6] ? 5 : 3
    end
end

@inline function _cubical_edge_vertices(g,edge)
    m=_cubical_grid_mask(g,1,edge)
    axis=trailing_zeros(m)+1
    localid=edge-g.offsets[m+1]
    step=g.strides[1][axis]
    width=g.strides[m+1][axis]*g.shapes[m+1][axis]
    v=localid+div(localid-1,width)*step
    return v,v+step
end

@inline _cubical_older(a,b,grades,increasing) = grades[a]==grades[b] ? a<b :
    (increasing ? grades[a]<grades[b] : grades[a]>grades[b])

function _cubical_grid_h0!(bars,essential,g,edges,order)
    vertex_grades=g.grades[1]
    n=length(vertex_grades)
    I=n<=typemax(Int32) ? Int32 : Int
    parent=fill(-one(I),n);elder=collect(I,1:n)
    # Retain negative edges for dual pruning and filtered cycle coordinates.
    forest=falses(length(edges))
    for edge in edges
        a,b=_cubical_edge_vertices(g,edge)
        ra,rb=_cubical_compact_root!(parent,a),_cubical_compact_root!(parent,b)
        ra==rb && continue
        forest[edge]=true
        ea,eb=elder[ra],elder[rb]
        older,younger=_cubical_older(ea,eb,vertex_grades,order===:sublevel) ? (ea,eb) : (eb,ea)
        birth,death=vertex_grades[younger],g.grades[2][edge]
        birth==death || push!(bars,(birth,death))
        if parent[ra]>parent[rb];ra,rb=rb,ra;end
        parent[ra]+=parent[rb];parent[rb]=ra;elder[ra]=older
    end
    for v in 1:n
        parent[v]<0 && push!(essential,vertex_grades[elder[v]])
    end
    return forest
end

function _cubical_grid_top!(bars,g::_CubicalBarcodeGrid{N},faces,order,skip=nothing) where {N}
    topgrades=g.grades[end];ntop=length(topgrades)
    cleared=falses(length(g.grades[N]))
    ntop==0 && return cleared
    outside=ntop+1
    I=outside<=typemax(Int32) ? Int32 : Int
    parent=fill(-one(I),outside);elder=collect(I,1:outside)
    topmask=(1<<N)-1
    for face in Iterators.reverse(faces)
        skip!==nothing && skip[face] && continue
        m=_cubical_grid_mask(g,N-1,face)
        axis=trailing_zeros(xor(m,topmask))+1
        localid=face-g.offsets[m+1]
        step=g.strides[end][axis]
        width=g.shapes[m+1][axis]
        layer=div(localid-1,step)
        coordinate=rem(layer,width)
        topid=localid-div(layer,width)*step
        a=coordinate<width-1 ? topid : outside
        b=coordinate>0 ? topid-step : outside
        ra,rb=_cubical_compact_root!(parent,a),_cubical_compact_root!(parent,b)
        ra==rb && continue
        ea,eb=elder[ra],elder[rb]
        aolder=ea==outside || (eb!=outside && !_cubical_older(ea,eb,topgrades,order===:sublevel))
        older,younger=aolder ? (ea,eb) : (eb,ea)
        birth,death=g.grades[N][face],topgrades[younger]
        birth==death || push!(bars,(birth,death))
        cleared[face]=true
        if parent[ra]>parent[rb];ra,rb=rb,ra;end
        parent[ra]+=parent[rb];parent[rb]=ra;elder[ra]=older
    end
    return cleared
end


# In this grid kernel, a zero span count stores an implicit square ID;
# ID zero instead marks a canceled unit relation. Explicit columns retain
# the shared sparse/packed representation.
@inline function _cubical_replay_square!(active,g,rank,face)
    m=_cubical_grid_mask(g,2,face)
    localid=face-g.offsets[m+1]
    first=trailing_zeros(m)+1
    last=8*sizeof(Int)-leading_zeros(m)
    a,b=_cubical_square_pair(g,m,localid,first,rank)
    c,d=_cubical_square_pair(g,m,localid,last,rank)
    a!=0 && _toggle_entry!(active,a)
    b!=0 && _toggle_entry!(active,b)
    c!=0 && _toggle_entry!(active,c)
    d!=0 && _toggle_entry!(active,d)
    return nothing
end

function _cubical_grid_diagram(values::AbstractArray{T,N},input,order,field,construction) where {T,N}
    N == 3 && return _cubical_local_diagram(values,input,order,field,construction)
    per=ntuple(_->false,N)
    if input===:top_cells
        _top_cell_preflight(values,per,construction)
    else
        _vertex_cubical_preflight(values,per,construction)
    end
    _validate_cubical_values(values)
    grid=_cubical_barcode_grid(values,input,order)
    bars=[Tuple{T,T}[] for _ in 0:N];essential=[T[] for _ in 0:N]
    edges=_cubical_grade_order(grid.grades[2],order)
    forest=_cubical_grid_h0!(bars[1],essential[1],grid,edges,order)
    if N==2
        _cubical_grid_top!(bars[2],grid,edges,order,forest)

    end
    for group in (bars,essential), entries in group
        sort!(entries;rev=order===:superlevel)
    end
    kind=input===:vertices ? :cubical_vertices : :cubical_top_cells
    meta=(construction=(requested=kind,effective=kind,substitution=:none),
        source=(shape=size(values),periodic=per),grade_arithmetic=:exact_input_values,
        geometry=:cubical_cells,chain_validation=:cubical_tensor_identity,
        backend=:f2_cubical_grid,approximation=:none_in_reduction,discretization=:none)
    return PersistenceDiagram(bars,essential,field,order,meta)
end

include("cubical_local.jl")
