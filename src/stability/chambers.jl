# Exact planar lower envelopes of inverse-slope polynomials. In a fixed source
# cell their xy coefficients agree, leaving rational affine comparisons.
"""Closed rational polygon supporting a planar HN chamber interior.
`subspaces` contains candidate IDs local to the source cell; boundary flags
must be obtained by `skyscraper_chamber_at`, which resolves exact ties.
"""
struct HNChamber
    cell::Int
    polygon::Vector{NTuple{2,QQ}}
    inequalities::Vector{NTuple{3,QQ}}
    subspaces::Vector{Int}
end
describe(c::HNChamber)=(;cell=c.cell,polygon=copy(c.polygon),inequalities=copy(c.inequalities),
    interior_subspaces=copy(c.subspaces),scope=:closed_support_of_interior)
Base.show(io::IO,c::HNChamber)=print(io,"HNChamber(cell=",c.cell,", ",length(c.polygon)," vertices)")
struct SkyscraperChambers{G,K}
    grid::G
    cells::Vector{Int}
    chambers::Vector{HNChamber}
    flags::Vector{_AnchorData{K}}
    statistics::NamedTuple
end
_affine(h,p)=h[1]+h[2]*p[1]+h[3]*p[2]
function _clip_polygon(poly,h)
    isempty(poly) && return poly
    out=NTuple{2,QQ}[]
    a=last(poly);va=_affine(h,a)
    for b in poly
        vb=_affine(h,b)
        if (va<=0)!=(vb<=0)
            t=va/(va-vb)
            push!(out,(a[1]+t*(b[1]-a[1]),a[2]+t*(b[2]-a[2])))
        end
        vb<=0 && push!(out,b)
        a=b;va=vb
    end
    unique(out)
end
function _area(poly)
    length(poly)<3 && return zero(QQ)
    abs(sum(poly[i][1]*poly[mod1(i+1,length(poly))][2]-poly[i][2]*poly[mod1(i+1,length(poly))][1] for i in eachindex(poly)))/2
end
function _mass_polynomials(g,cell)
    source=CartesianIndices(g.shape)[cell]
    vec(map(CartesianIndices(g.shape)) do I
        if any(I[j]<source[j] for j in 1:2)
            (QQ(0),QQ(0),QQ(0),QQ(0))
        else
            a=g.axes[1][I[1]+1]-(I[1]==source[1] ? 0 : g.axes[1][I[1]])
            b=g.axes[2][I[2]+1]-(I[2]==source[2] ? 0 : g.axes[2][I[2]])
            x=QQ(I[1]==source[1]);y=QQ(I[2]==source[2])
            Tuple(g.density[LinearIndices(g.shape)[I]].*(a*b,-x*b,-y*a,x*y))
        end
    end)
end
function _quotient_poly(data,masses,a,b)
    k=size(data.spaces[b],2)-size(data.spaces[a],2)
    ntuple(j->sum((masses[v][j]*(data.dimensions[b][v]-data.dimensions[a][v]) for v in eachindex(masses));init=zero(QQ))/k,4)
end
# A complete candidate table gives affine inverse-slope comparisons in 2D.
function _chamber_subdivision(data,masses,cell,poly,F,budget,w)
    result=HNChamber[]
    zero_id=findfirst(U->size(U,2)==0,data.spaces);d=size(first(data.maps),2)
    function subdivide(current,polygon,inequalities,flag)
        if size(data.spaces[current],2)==d
            _bound(length(result)+1,budget.max_chambers,"chambers")
            push!(result,HNChamber(cell,polygon,inequalities,flag));return
        end
        ids=Int[];ps=NTuple{4,QQ}[]
        for j in eachindex(data.spaces)
            _tick!(w,budget)
            size(data.spaces[j],2)>size(data.spaces[current],2) &&
                _contains(F,data.spaces[j],data.spaces[current]) || continue
            p=_quotient_poly(data,masses,current,j)
            equal=findfirst(==(p),ps)
            if equal===nothing
                push!(ids,j);push!(ps,p)
            elseif size(data.spaces[j],2)>size(data.spaces[ids[equal]],2)
                ids[equal]=j
            end
        end
        for i in eachindex(ids)
            region=polygon;hs=copy(inequalities)
            for j in eachindex(ids)
                i==j && continue
                _tick!(w,budget)
                ps[i][4]==ps[j][4] || error("inverse-slope leading coefficients disagree")
                h=ntuple(k->ps[i][k]-ps[j][k],3)
                region=_clip_polygon(region,h);push!(hs,h)
                iszero(_area(region)) && break
            end
            iszero(_area(region)) && continue
            subdivide(ids[i],region,hs,vcat(flag,ids[i]))
        end
    end
    subdivide(zero_id,poly,NTuple{3,QQ}[],Int[])
    result
end

# Certify a proposed subdivision at every polygon vertex. Every possible
# subspace has an affine inverse-slope difference on a fixed source cell.
# At a vertex its full HN flag supplies the lower convex envelope of
# (anchor dimension, mass). If no new flags appear, every affine comparison
# holds at the vertices and hence throughout each polygon. Corner masses may
# vanish; the inverse-slope oracle handles them without an epsilon or rounding.
function _discover_chambers(g,cell,masses,poly,b,w)
    M=g.finite;F=M.field;K=coeff_type(F);d=M.dims[cell]
    maps=_anchor_maps(M,cell,b)
    spaces=Matrix{K}[zeros(K,d,0),eye(F,d)]
    d==0 && pop!(spaces)
    _bound(w.subspaces+length(spaces),b.max_subspaces,"chamber anchor subspaces")
    w.subspaces+=length(spaces)
    data=_AnchorData(maps,spaces,[[size(U,2)==0 ? 0 : _rank(F,A) for A in maps] for U in spaces])
    seen=Set{NTuple{2,QQ}}()
    while true
        chambers=_chamber_subdivision(data,masses,cell,poly,F,b,w)
        added=false
        for chamber in chambers,p in chamber.polygon
            p in seen && continue
            push!(seen,p)
            weights=QQ[q[1]+q[2]*p[1]+q[3]*p[2]+q[4]*p[1]*p[2] for q in masses]
            all(>=(0),weights) || error("negative mass at a closed cell corner")
            oracle,_=_general_anchor_data(M,cell,weights,b,w;maps)
            for (U,dims) in zip(oracle.spaces,oracle.dimensions)
                any(V->size(U,2)==size(V,2) && _contains(F,V,U),data.spaces) && continue
                _bound(w.subspaces+1,b.max_subspaces,"discovered anchor subspaces")
                retained=sum(big(length(V)) for V in data.spaces;init=big(0))+length(U)+
                    big(length(data.spaces)+1)*length(M.dims)+2*length(seen)
                _bound(retained,b.max_entries,"chamber candidate entries")
                push!(data.spaces,U);push!(data.dimensions,dims);w.subspaces+=1;added=true
            end
        end
        added || return chambers,data
    end
end

"""Compute exact planar HN chamber interiors and their closed polygon supports.
Arbitrary finite stalk dimensions over QQ and prime fields are supported within
explicit budgets. Small complete candidate families use direct subdivision;
larger ones use matrix-space oracles and exact polygon-vertex certificates.
Retained flags support exact subsequent queries without another matrix search.
At a boundary, equal slopes merge; closures are not disjoint strata. Optional
`cells` restricts the computation to finite source labels.
"""
function skyscraper_chambers(g::SkyscraperGrid{2};cells=collect(1:length(g.finite.dims)),
        budget::HNBudget=HNBudget())
    _input(g.finite,budget);cs=sort!(unique([_vertex(g.finite,c) for c in cells]))
    w=_Work(0,0);result=HNChamber[];F=g.finite.field;K=coeff_type(F)
    flags=_AnchorData{K}[];retained=big(0)
    for cell in cs
        masses=_mass_polynomials(g,cell)
        a,b=Tuple(CartesianIndices(g.shape)[cell]);x0,x1=g.axes[1][a:a+1];y0,y1=g.axes[2][b:b+1]
        poly=NTuple{2,QQ}[(x0,y0),(x1,y0),(x1,y1),(x0,y1)]
        chambers,data=if _small_anchor(F,g.finite.dims[cell])
            data=_anchor_data(g.finite,cell,budget,w)
            _chamber_subdivision(data,masses,cell,poly,F,budget,w),data
        else
            _discover_chambers(g,cell,masses,poly,budget,w)
        end
        _bound(length(result)+length(chambers),budget.max_chambers,"chambers")
        retained+=sum(big(length(A)) for A in data.maps;init=big(0))
        zero_id=findfirst(U->size(U,2)==0,data.spaces)
        for chamber in chambers
            ids=vcat(zero_id,chamber.subspaces)
            retained+=sum(big(length(data.spaces[i]))+length(data.dimensions[i]) for i in ids)+
                2*length(chamber.polygon)+3*length(chamber.inequalities)
            _bound(retained,budget.max_entries,"retained chamber entries")
            push!(flags,_AnchorData(data.maps,data.spaces[ids],data.dimensions[ids]))
        end
        append!(result,chambers)
    end
    SkyscraperChambers(g,cs,result,flags,_statistics(w))
end
function skyscraper_chambers(g::SkyscraperGrid;kwargs...)
    throw(ArgumentError("complete chamber geometry is implemented for two parameters; higher-dimensional walls need not be affine. Use selected-anchor skyscraper or filtered_landscape queries instead"))
end
function _in_polygon(poly,p)
    all((poly[mod1(i+1,length(poly))][1]-poly[i][1])*(p[2]-poly[i][2])-
        (poly[mod1(i+1,length(poly))][2]-poly[i][2])*(p[1]-poly[i][1])>=0 for i in eachindex(poly))
end
function skyscraper_chamber_at(r::SkyscraperChambers;anchor,witnesses=false,budget=HNBudget())
    cell,p=_grid_label(r.grid,anchor)
    cell in r.cells || throw(ArgumentError("anchor cell was not included in this chamber computation"))
    ids=findall(c->c.cell==cell && _in_polygon(c.polygon,p),r.chambers)
    isempty(ids) && error("chamber subdivision does not cover this anchor")
    boundary=length(ids)>1 || any(i->any(h->iszero(_affine(h,p)),r.chambers[i].inequalities),ids)
    # A chamber's flag refines its boundary HN flags: affine inequalities stay
    # valid on the closure, and equal adjacent slopes merge there. Thus any
    # containing chamber supplies all required terms, including exact ties.
    g=r.grid;_input(g.finite,budget);M=_snapshot(g.finite);w=_Work(0,0)
    context=(;category=:bounded_cartesian_encoding,measure=_measure(g),density=copy(g.density),anchor=p,
        axes=map(copy,g.axes),shape=g.shape,extension=:zero,cells=:left_closed_right_open,
        route=:certified_chamber)
    filtration=_anchor_result(M,cell,_grid_weights(g,p),r.flags[first(ids)],budget,w,context,witnesses)
    (;chambers=ids,boundary,filtration)
end
function skyscraper_chamber_polygons(r::SkyscraperChambers)
    [(;cell=c.cell,polygon=copy(c.polygon),inequalities=copy(c.inequalities),
        interior_subspaces=copy(c.subspaces)) for c in r.chambers]
end
describe(r::SkyscraperChambers)=(;kind=:skyscraper_chambers,cells=copy(r.cells),
    chambers=length(r.chambers),geometry=:exact_rational,scope=:closed_supports_of_interiors,
    boundaries=:exact_query,statistics=r.statistics)
Base.show(io::IO,r::SkyscraperChambers)=print(io,"SkyscraperChambers(",length(r.chambers)," regions, ",length(r.cells)," source cells)")

function skyscraper_chambers(g::SkyscraperEncoding;kwargs...)
    throw(ArgumentError("complete chambers require a planar SkyscraperGrid; use selected skyscraper or filtered_landscape queries for this encoding"))
end
