# Finite relative chain plans for nonperiodic 3D F2 barcodes.
# Generated independently by tools/generate_cubical_tables.py. Constant plans
# are input-independent; ranks, forests, pairings and reductions are per query.
# Local elimination retains full boundary combinations, including earlier owners.

struct _CubicalLocalGeometry
    slots::Vector{NTuple{3,Int}}
    masks::Vector{Int}
    anchors::Vector{NTuple{3,Int}}
    keys::Vector{UInt32}
    ids::Vector{UInt16}
    negative_edges::Vector{UInt32}
    negative_faces::Vector{UInt32}
    pair_offsets::Vector{UInt32}
    pairs::Vector{UInt32}
end
@inline _cubical_local_hash(mask,cap) = Int(((UInt64(mask)*UInt64(0x9e3779b97f4a7c15)) >> 32) & UInt64(cap-1))+1
@inline function _cubical_local_planindex(table,mask)
    cap=length(table.keys);slot=_cubical_local_hash(mask,cap)
    @inbounds while table.keys[slot]!=mask
        table.keys[slot]==0 && error("invalid cubical neighborhood")
        slot=(slot & (cap-1))+1
    end
    @inbounds return Int(table.ids[slot])
end
function _cubical_read_local_table(io,vertices)
    slots=NTuple{3,Int}[(a,b,c) for a in -1:1 for b in -1:1 for c in -1:1]
    masks=[sum(1<<(a-1) for a in 1:3 if vertices ? s[a]!=0 : s[a]==0;init=0) for s in slots]
    anchors=[ntuple(a->vertices ? min(s[a],0) : Int(s[a]==1),3) for s in slots]
    n=Int(ltoh(read(io,UInt32)))
    n==15935 || error("invalid cubical table size")
    cap=nextpow(2,2*n);keys=zeros(UInt32,cap);ids=zeros(UInt16,cap)
    negative_edges=Vector{UInt32}(undef,n);negative_faces=similar(negative_edges)
    offsets=UInt32[1];pairs=UInt32[]
    for i in 1:n
        mask=ltoh(read(io,UInt32));mask!=0 || error("invalid cubical table key")
        slot=_cubical_local_hash(mask,cap)
        while keys[slot]!=0
            keys[slot]!=mask || error("duplicate cubical table key")
            slot=(slot & (cap-1))+1
        end
        keys[slot]=mask;ids[slot]=i
        negative_edges[i]=ltoh(read(io,UInt32));negative_faces[i]=ltoh(read(io,UInt32))
        count=Int(ltoh(read(io,UInt32)))
        count<=27 || error("invalid cubical table pair count")
        for _ in 1:count;push!(pairs,ltoh(read(io,UInt32)));end
        push!(offsets,UInt32(length(pairs)+1))
    end
    _CubicalLocalGeometry(slots,masks,anchors,keys,ids,negative_edges,negative_faces,offsets,pairs)
end
function _cubical_read_local_tables(path)
    open(path,"r") do io
        read(io,8)==Vector{UInt8}(codeunits("CT3D0001")) || error("invalid cubical table format")
        tables=(_cubical_read_local_table(io,true),_cubical_read_local_table(io,false))
        eof(io) || error("unexpected trailing cubical table data")
        tables
    end
end
# Loaded during package precompilation, then serialized in the package image.
# No file reads or lazy mutable cache are needed during a cubical query.
const _CUBICAL_LOCAL_TABLES = let path=joinpath(@__DIR__,"cubical_tables.bin")
    include_dependency(path)
    _cubical_read_local_tables(path)
end

# Propagate an axis at a time through contiguous blocks. Cell order and
# the comparison/tie choice match the tensor-grid construction exactly.
function _cubical_grade_up!(dst,src,dst_offset,src_offset,step,width,blocks,increasing)
    @inbounds for block in 0:blocks-1
        target=dst_offset+block*step*width
        source=src_offset+block*step*(width+1)
        for layer in 0:width-1
            a=source+layer*step;b=target+layer*step
            @simd for i in 1:step
                x,y=src[a+i],src[a+i+step]
                dst[b+i]=(increasing ? x<y : x>y) ? y : x
            end
        end
    end
end
function _cubical_grade_down!(dst,src,dst_offset,src_offset,step,width,blocks,increasing)
    @inbounds for block in 0:blocks-1
        target=dst_offset+block*step*(width+1)
        source=src_offset+block*step*width
        @simd for i in 1:step
            dst[target+i]=src[source+i]
            dst[target+width*step+i]=src[source+(width-1)*step+i]
        end
        for layer in 1:width-1
            a=source+(layer-1)*step;b=target+layer*step
            @simd for i in 1:step
                x,y=src[a+i],src[a+i+step]
                dst[b+i]=(increasing ? y<x : y>x) ? y : x
            end
        end
    end
end

function _cubical_owner_grid(values::Array{T,3}, input) where {T<:Integer}
    N=3
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
    source_degree=input===:vertices ? 1 : N+1
    grades = [d==source_degree ? vec(values) : Vector{T}(undef,n) for (d,n) in enumerate(counts)]
    increasing = true
    if input === :vertices
        for m in 1:(1<<N)-1
            axis = trailing_zeros(m)+1
            parent = m & (m-1)
            dst, src = grades[count_ones(m)+1], grades[count_ones(parent)+1]
            step=strides[parent+1][axis];width=shapes[m+1][axis]
            blocks=prod(shapes[m+1][axis+1:end])
            _cubical_grade_up!(dst,src,offsets[m+1],offsets[parent+1],step,width,blocks,increasing)
        end
    else
        for m in (1<<N)-2:-1:0
            axis = trailing_zeros(xor(m,(1<<N)-1))+1
            parent = m | (1<<(axis-1))
            dst,src = grades[count_ones(m)+1],grades[count_ones(parent)+1]
            step=strides[parent+1][axis];width=shapes[parent+1][axis]
            blocks=prod(shapes[m+1][axis+1:end])
            _cubical_grade_down!(dst,src,offsets[m+1],offsets[parent+1],step,width,blocks,increasing)
        end
    end
    return _CubicalBarcodeGrid(shapes,strides,offsets,grades)
end


@inline function _cubical_local_cell_id(table::_CubicalLocalGeometry,g,c,slot)
    N=3
    mask=table.masks[slot];a=table.anchors[slot];strides=g.strides[mask+1]
    id=g.offsets[mask+1]+1
    @inbounds for axis in 1:N
        id+=(c[axis]+a[axis]-1)*strides[axis]
    end
    id
end
function _cubical_local_pattern(table::_CubicalLocalGeometry,ranks,c,id,dims,shifts,vertices)
    @inbounds begin
        owner=ranks[id]
        mask=UInt32(8192)
        if mask & UInt32(8192) == UInt32(8192)
            present=(c[1]>1) ? (vertices ? ranks[id+shifts[5]]<owner : ranks[id+shifts[5]]>owner) : !vertices
            present && (mask |= UInt32(16))
        end
        if mask & UInt32(8192) == UInt32(8192)
            present=(c[2]>1) ? (vertices ? ranks[id+shifts[11]]<owner : ranks[id+shifts[11]]>owner) : !vertices
            present && (mask |= UInt32(1024))
        end
        if mask & UInt32(8192) == UInt32(8192)
            present=(c[3]>1) ? (vertices ? ranks[id+shifts[13]]<owner : ranks[id+shifts[13]]>owner) : !vertices
            present && (mask |= UInt32(4096))
        end
        if mask & UInt32(8192) == UInt32(8192)
            present=(c[3]<dims[3]) ? (vertices ? ranks[id+shifts[15]]<owner : ranks[id+shifts[15]]>owner) : !vertices
            present && (mask |= UInt32(16384))
        end
        if mask & UInt32(8192) == UInt32(8192)
            present=(c[2]<dims[2]) ? (vertices ? ranks[id+shifts[17]]<owner : ranks[id+shifts[17]]>owner) : !vertices
            present && (mask |= UInt32(65536))
        end
        if mask & UInt32(8192) == UInt32(8192)
            present=(c[1]<dims[1]) ? (vertices ? ranks[id+shifts[23]]<owner : ranks[id+shifts[23]]>owner) : !vertices
            present && (mask |= UInt32(4194304))
        end
        if mask & UInt32(1040) == UInt32(1040)
            present=(c[1]>1 && c[2]>1) ? (vertices ? ranks[id+shifts[2]]<owner : ranks[id+shifts[2]]>owner) : !vertices
            present && (mask |= UInt32(2))
        end
        if mask & UInt32(4112) == UInt32(4112)
            present=(c[1]>1 && c[3]>1) ? (vertices ? ranks[id+shifts[4]]<owner : ranks[id+shifts[4]]>owner) : !vertices
            present && (mask |= UInt32(8))
        end
        if mask & UInt32(16400) == UInt32(16400)
            present=(c[1]>1 && c[3]<dims[3]) ? (vertices ? ranks[id+shifts[6]]<owner : ranks[id+shifts[6]]>owner) : !vertices
            present && (mask |= UInt32(32))
        end
        if mask & UInt32(65552) == UInt32(65552)
            present=(c[1]>1 && c[2]<dims[2]) ? (vertices ? ranks[id+shifts[8]]<owner : ranks[id+shifts[8]]>owner) : !vertices
            present && (mask |= UInt32(128))
        end
        if mask & UInt32(5120) == UInt32(5120)
            present=(c[2]>1 && c[3]>1) ? (vertices ? ranks[id+shifts[10]]<owner : ranks[id+shifts[10]]>owner) : !vertices
            present && (mask |= UInt32(512))
        end
        if mask & UInt32(17408) == UInt32(17408)
            present=(c[2]>1 && c[3]<dims[3]) ? (vertices ? ranks[id+shifts[12]]<owner : ranks[id+shifts[12]]>owner) : !vertices
            present && (mask |= UInt32(2048))
        end
        if mask & UInt32(69632) == UInt32(69632)
            present=(c[2]<dims[2] && c[3]>1) ? (vertices ? ranks[id+shifts[16]]<owner : ranks[id+shifts[16]]>owner) : !vertices
            present && (mask |= UInt32(32768))
        end
        if mask & UInt32(81920) == UInt32(81920)
            present=(c[2]<dims[2] && c[3]<dims[3]) ? (vertices ? ranks[id+shifts[18]]<owner : ranks[id+shifts[18]]>owner) : !vertices
            present && (mask |= UInt32(131072))
        end
        if mask & UInt32(4195328) == UInt32(4195328)
            present=(c[1]<dims[1] && c[2]>1) ? (vertices ? ranks[id+shifts[20]]<owner : ranks[id+shifts[20]]>owner) : !vertices
            present && (mask |= UInt32(524288))
        end
        if mask & UInt32(4198400) == UInt32(4198400)
            present=(c[1]<dims[1] && c[3]>1) ? (vertices ? ranks[id+shifts[22]]<owner : ranks[id+shifts[22]]>owner) : !vertices
            present && (mask |= UInt32(2097152))
        end
        if mask & UInt32(4210688) == UInt32(4210688)
            present=(c[1]<dims[1] && c[3]<dims[3]) ? (vertices ? ranks[id+shifts[24]]<owner : ranks[id+shifts[24]]>owner) : !vertices
            present && (mask |= UInt32(8388608))
        end
        if mask & UInt32(4259840) == UInt32(4259840)
            present=(c[1]<dims[1] && c[2]<dims[2]) ? (vertices ? ranks[id+shifts[26]]<owner : ranks[id+shifts[26]]>owner) : !vertices
            present && (mask |= UInt32(33554432))
        end
        if mask & UInt32(522) == UInt32(522)
            present=(c[1]>1 && c[2]>1 && c[3]>1) ? (vertices ? ranks[id+shifts[1]]<owner : ranks[id+shifts[1]]>owner) : !vertices
            present && (mask |= UInt32(1))
        end
        if mask & UInt32(2082) == UInt32(2082)
            present=(c[1]>1 && c[2]>1 && c[3]<dims[3]) ? (vertices ? ranks[id+shifts[3]]<owner : ranks[id+shifts[3]]>owner) : !vertices
            present && (mask |= UInt32(4))
        end
        if mask & UInt32(32904) == UInt32(32904)
            present=(c[1]>1 && c[2]<dims[2] && c[3]>1) ? (vertices ? ranks[id+shifts[7]]<owner : ranks[id+shifts[7]]>owner) : !vertices
            present && (mask |= UInt32(64))
        end
        if mask & UInt32(131232) == UInt32(131232)
            present=(c[1]>1 && c[2]<dims[2] && c[3]<dims[3]) ? (vertices ? ranks[id+shifts[9]]<owner : ranks[id+shifts[9]]>owner) : !vertices
            present && (mask |= UInt32(256))
        end
        if mask & UInt32(2621952) == UInt32(2621952)
            present=(c[1]<dims[1] && c[2]>1 && c[3]>1) ? (vertices ? ranks[id+shifts[19]]<owner : ranks[id+shifts[19]]>owner) : !vertices
            present && (mask |= UInt32(262144))
        end
        if mask & UInt32(8914944) == UInt32(8914944)
            present=(c[1]<dims[1] && c[2]>1 && c[3]<dims[3]) ? (vertices ? ranks[id+shifts[21]]<owner : ranks[id+shifts[21]]>owner) : !vertices
            present && (mask |= UInt32(1048576))
        end
        if mask & UInt32(35684352) == UInt32(35684352)
            present=(c[1]<dims[1] && c[2]<dims[2] && c[3]>1) ? (vertices ? ranks[id+shifts[25]]<owner : ranks[id+shifts[25]]>owner) : !vertices
            present && (mask |= UInt32(16777216))
        end
        if mask & UInt32(42074112) == UInt32(42074112)
            present=(c[1]<dims[1] && c[2]<dims[2] && c[3]<dims[3]) ? (vertices ? ranks[id+shifts[27]]<owner : ranks[id+shifts[27]]>owner) : !vertices
            present && (mask |= UInt32(67108864))
        end
        return mask
    end
end
function _cubical_local_classify(g,table::_CubicalLocalGeometry,ranks,input,dims)
    N=3
    strides=ntuple(a->prod(dims[1:a-1]),N)
    shifts=[sum(s[a]*strides[a] for a in 1:N) for s in table.slots]
    n=length(g.grades[1]);I=n<=typemax(Int32) ? Int32 : Int
    parent=fill(-one(I),n);elder=collect(I,1:n);forest=falses(length(g.grades[2]))
    J=length(ranks)<=typemax(Int32) ? Int32 : Int
    owner=zeros(J,length(g.grades[2]));combination=zeros(UInt32,length(owner))
    negative_faces=falses(length(g.grades[3]))
    for (id,c) in enumerate(CartesianIndices(dims))
        mask=_cubical_local_pattern(table,ranks,c,id,dims,shifts,input===:vertices)
        plan=_cubical_local_planindex(table,mask)
        bits=table.negative_edges[plan]
        while bits!=0
            slot=trailing_zeros(bits)+1;bits &= bits-one(bits)
            edge=_cubical_local_cell_id(table,g,c,slot)
            a,b=_cubical_edge_vertices(g,edge)
            ra,rb=_cubical_compact_root!(parent,a),_cubical_compact_root!(parent,b)
            @assert ra!=rb
            ea,eb=elder[ra],elder[rb];older=_cubical_older(ea,eb,g.grades[1],true) ? ea : eb
            if parent[ra]>parent[rb];ra,rb=rb,ra;end
            parent[ra]+=parent[rb];parent[rb]=ra;elder[ra]=older;forest[edge]=true
        end
        bits=table.negative_faces[plan]
        while bits!=0
            slot=trailing_zeros(bits)+1;bits &= bits-one(bits)
            negative_faces[_cubical_local_cell_id(table,g,c,slot)]=true
        end
        for i in Int(table.pair_offsets[plan]):Int(table.pair_offsets[plan+1])-1
            packed=table.pairs[i];slot=Int(packed & UInt32(31))+1;chain=packed>>5
            edge=_cubical_local_cell_id(table,g,c,slot)
            owner[edge]=id;combination[edge]=chain
        end
    end
    (parent,elder,forest),owner,combination,negative_faces
end
function _cubical_local_cell_order(grades,positions)
    fill!(positions,0)
    @inbounds for v in grades;positions[v]+=1;end
    next=1
    for i in eachindex(positions);count=positions[i];positions[i]=next;next+=count;end
    perm=Vector{Int}(undef,length(grades))
    @inbounds for i in eachindex(grades)
        bucket=grades[i];perm[positions[bucket]]=i;positions[bucket]+=1
    end
    perm
end
function _cubical_local_h0!(bars,essential,g,edges,prepared,local_combinations)
    vg=g.grades[1];n=length(vg);parent,elder,forest=prepared
    for edge in edges
        (forest[edge] || local_combinations[edge]!=0) && continue
        a,b=_cubical_edge_vertices(g,edge)
        ra,rb=_cubical_compact_root!(parent,a),_cubical_compact_root!(parent,b)
        ra==rb && continue
        forest[edge]=true;ea,eb=elder[ra],elder[rb]
        older,younger=_cubical_older(ea,eb,vg,true) ? (ea,eb) : (eb,ea)
        birth,death=vg[younger],g.grades[2][edge]
        birth==death || push!(bars,(birth,death))
        if parent[ra]>parent[rb];ra,rb=rb,ra;end
        parent[ra]+=parent[rb];parent[rb]=ra;elder[ra]=older
    end
    for v in 1:n;parent[v]<0 && push!(essential,vg[elder[v]]);end
    forest
end
function _cubical_local_replay!(active,g,rank,table,coords,chain)
    while chain!=0
        slot=trailing_zeros(chain)+1;chain &= chain-one(chain)
        face=_cubical_local_cell_id(table,g,coords,slot)
        _cubical_replay_square!(active,g,rank,face)
    end
end
function _cubical_local_h1!(bars,g,faces,cleared,edges,forest,table,owners,combinations,negative_faces,dims)
    rank=zeros(Int,length(edges));kept=0
    for edge in edges
        if !forest[edge];kept+=1;edges[kept]=edge;end
    end
    resize!(edges,kept);positive=edges
    for (i,e) in enumerate(positive);rank[e]=i;end
    n=length(positive);active=_ParityColumn(n);stored=_BarcodeColumns(n)
    has_reduced=BitVector(combinations[e]!=0 for e in positive);column=Int[]
    indices=CartesianIndices(dims)
    for face in faces
        (cleared[face] || negative_faces[face]) && continue
        m=_cubical_grid_mask(g,2,face);localid=face-g.offsets[m+1]
        first=trailing_zeros(m)+1;last=8*sizeof(Int)-leading_zeros(m)
        a,b=_cubical_square_pair(g,m,localid,first,rank);c,d=_cubical_square_pair(g,m,localid,last,rank)
        rows=(a,b,c,d);mask=Int(a!=0)+2*Int(b!=0)+4*Int(c!=0)+8*Int(d!=0)
        pattern=_CUBICAL_SQUARE_PATTERNS[mask+1];pivot=max(a,b,c,d)
        pivot==0 && continue
        if !has_reduced[pivot]
            if pattern[5]==1 && g.grades[2][positive[pivot]]==g.grades[3][face]
                rank[positive[pivot]]=0;stored.spans[pivot]=(0,0);has_reduced[pivot]=true;continue
            end
            stored.spans[pivot]=(face,0);has_reduced[pivot]=true
            birth,death=g.grades[2][positive[pivot]],g.grades[3][face]
            birth==death || push!(bars,(birth,death));continue
        end
        for i in 1:pattern[5];_toggle_entry!(active,rows[pattern[i]]);end
        while pivot!=0 && has_reduced[pivot]
            edge=positive[pivot];chain=combinations[edge]
            if chain!=0
                _cubical_local_replay!(active,g,rank,table,indices[owners[edge]],chain)
            else
                start,count=stored.spans[pivot]
                if count==0
                    start==0 ? _toggle_entry!(active,pivot) : _cubical_replay_square!(active,g,rank,start)
                else
                    _add_barcode_column!(active,stored,pivot)
                end
            end
            pivot=_column_pivot(active,pivot)
        end
        pivot==0 && continue
        _drain_column!(column,active);_store_barcode_column!(stored,pivot,column);has_reduced[pivot]=true
        birth,death=g.grades[2][positive[pivot]],g.grades[3][face]
        birth==death || push!(bars,(birth,death))
    end
end

# Refining ties by input position gives each cell a unique owner. A local
# pair has the same owner at both ends, so it contributes only a zero bar.
# Push the remaining ordinal endpoints back to the original filtration and
# discard any additional bars made zero by undoing the tie refinement.
# Existing construction preflights conservatively cover ranks, owner grades,
# order arrays and local setup (less than the general cell/CSC allowance).
# Fixed package tables and subsequent reduction storage are not per-input
# construction allocations or part of its advertised memory estimate.
function _cubical_local_diagram(values::AbstractArray{T,3},input,order,field,construction) where T
    N=3
    per=ntuple(_->false,N)
    input===:top_cells ? _top_cell_preflight(values,per,construction) : _vertex_cubical_preflight(values,per,construction)
    _validate_cubical_values(values)
    sorted=_cubical_grade_order(collect(vec(values)),order)
    I=length(values)<=typemax(Int32) ? Int32 : Int
    ranks=Vector{I}(undef,length(values));for (i,v) in enumerate(sorted);ranks[v]=i;end
    g=_cubical_owner_grid(reshape(ranks,size(values)),input)
    table=_CUBICAL_LOCAL_TABLES[input===:vertices ? 1 : 2]
    forced,owners,combinations,negative_faces=_cubical_local_classify(g,table,ranks,input,size(values))
    bars=[Tuple{Int,Int}[] for _ in 0:N];essential=[Int[] for _ in 0:N]
    positions=zeros(Int,length(values))
    edges=_cubical_local_cell_order(g.grades[2],positions)
    forest=_cubical_local_h0!(bars[1],essential[1],g,edges,forced,combinations)
    faces=_cubical_local_cell_order(g.grades[3],positions)
    cleared=_cubical_grid_top!(bars[3],g,faces,:sublevel,negative_faces)
    _cubical_local_h1!(bars[2],g,faces,cleared,edges,forest,table,owners,combinations,negative_faces,size(values))
    mapped=[Tuple{T,T}[] for _ in 0:N];births=[T[] for _ in 0:N]
    for d in eachindex(bars)
        for (a,b) in bars[d]
            x,y=values[sorted[a]],values[sorted[b]];x==y || push!(mapped[d],(x,y))
        end
        append!(births[d],(values[sorted[i]] for i in essential[d]))
        sort!(mapped[d];rev=order===:superlevel);sort!(births[d];rev=order===:superlevel)
    end
    kind=input===:vertices ? :cubical_vertices : :cubical_top_cells
    meta=(construction=(requested=kind,effective=kind,substitution=:none),
        source=(shape=size(values),periodic=per),grade_arithmetic=:exact_input_values,
        geometry=:cubical_cells,chain_validation=:cubical_tensor_identity,
        backend=:f2_cubical_grid,local_reduction=:relative_tables,
        approximation=:none_in_reduction,discretization=:none)
    PersistenceDiagram(mapped,births,field,order,meta)
end
