# A48: retain the encoding's actual finite-grid births, including coarsening,
# orientation and union/intersection semantics. Source geometry is unnecessary.
function _h0_presentation_input(L::LazyModuleCochainComplex;max_expanded_cells::Int=1_000_000, allow_fallback::Bool=false)
    max_expanded_cells>=0 || throw(ArgumentError("h0_presentation: max_expanded_cells must be nonnegative"))
    isempty(L.births_by_dim) && return nothing
    N=length(L.axes); sizes=length.(L.axes)
    strides=ntuple(i->prod(sizes[1:i-1];init=1),N)
    label(b)=1+sum((b[i]-1)*strides[i] for i in 1:N)
    grade_set(b)=b isa Tuple ? [b] : L.multicritical==:intersection ?
        [ntuple(i->maximum(g[i] for g in b),N)] : b
    birth0=L.births_by_dim[1]
    vg=Int[]; copies=[Int[] for _ in eachindex(birth0)]
    copygrades=[grade_set(b) for b in birth0]
    # Reject an oversized expansion before allocating it. Automatic queries
    # retain the general solver when this optional representation is too large.
    remaining=max_expanded_cells
    for grades in copygrades
        k=length(grades)
        if k>remaining || div(widemul(k,k-1),2)+k>remaining
            allow_fallback && return nothing
            throw(ArgumentError("h0_presentation: expanded graph exceeds max_expanded_cells"))
        end
        remaining-=k+div(k*(k-1),2)
    end
    if !isempty(L.boundaries)
        for e in axes(L.boundaries[1],2)
            any(k->!iszero(L.boundaries[1].nzval[k]),nzrange(L.boundaries[1],e)) || continue
            remaining-=length(grade_set(L.births_by_dim[2][e]))
            if remaining<0
                allow_fallback && return nothing
                throw(ArgumentError("h0_presentation: expanded graph exceeds max_expanded_cells"))
            end
        end
    end
    edges=NTuple{2,Int}[]; eg=Int[]
    function add_edge(u,v,b)
        length(vg)+length(edges)<max_expanded_cells || throw(ArgumentError("h0_presentation: expanded graph exceeds max_expanded_cells"))
        push!(edges,(u,v));push!(eg,label(b))
    end
    for v in eachindex(birth0)
        for b in copygrades[v]
            length(vg)+length(edges)<max_expanded_cells || throw(ArgumentError("h0_presentation: expanded graph exceeds max_expanded_cells"))
            push!(vg,label(b));push!(copies[v],length(vg))
        end
        # Pairwise joins identify every simultaneously active copy. This works
        # in any finite product lattice, including more than two parameters.
        for j in 2:length(copies[v]),i in 1:j-1
            a,b=copygrades[v][i],copygrades[v][j]
            add_edge(copies[v][i],copies[v][j],ntuple(k->max(a[k],b[k]),N))
        end
    end
    if !isempty(L.boundaries)
        B=L.boundaries[1]; birth1=L.births_by_dim[2]
        for e in axes(B,2)
            ids=[k for k in nzrange(B,e) if !iszero(B.nzval[k])]
            isempty(ids) && continue # Loops/zero columns have no H0 relation.
            length(ids)==2 || return nothing
            a,b=ids
            abs(B.nzval[a])==1 && B.nzval[a]==-B.nzval[b] || return nothing
            u,v=B.rowval[a],B.rowval[b]
            for g in grade_set(birth1[e])
                ui=findfirst(b->_grid_birth_leq(b,g),copygrades[u])
                vi=findfirst(b->_grid_birth_leq(b,g),copygrades[v])
                ui!==nothing && vi!==nothing || return nothing
                add_edge(copies[u][ui],copies[v][vi],g)
            end
        end
    end
    return (vertex_grades=vg,edges=edges,edge_grades=eg)
end

function h0_presentation(enc::EncodingResult{P,M};syzygies::Bool=false,
                         max_expanded_cells::Int=1_000_000,
                         max_witness_entries::Int=10_000_000) where {P,M<:_LazyEncodedModule}
    enc.M.degree==0 || throw(ArgumentError("h0_presentation: encoding must represent homology degree zero"))
    input=_h0_presentation_input(enc.M.lazy;max_expanded_cells)
    input===nothing && throw(ArgumentError("h0_presentation: source boundaries must be ordinary graph incidence columns"))
    return h0_presentation(enc.P;input...,field=enc.M.lazy.field,syzygies,max_witness_entries)
end

function _specialized_betti_table(enc::EncodingResult{P,M},opts,pad_to) where {P,M<:_LazyEncodedModule}
    enc.M.degree==0 || return nothing
    shape=H0Presentations._h0_grid_shape(enc.P)
    shape!==nothing && length(shape[1])<=2 || return nothing
    input=_h0_presentation_input(enc.M.lazy;allow_fallback=true)
    input===nothing && return nothing
    p=h0_presentation(enc.P;input...,field=enc.M.lazy.field)
    opts.minimal && opts.check && H0Presentations.check_h0_presentation(p;throw=true)
    return H0Presentations.betti_table(p;opts,pad_to)
end
