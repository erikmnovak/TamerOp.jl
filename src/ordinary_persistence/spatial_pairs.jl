# Source selection constructs a mathematical pair before any drawing is made.
"""A selected one-parameter coordinate pair with retained source identities.
Construct with `spatial_pair`; obtain its checked pair with `filtered_pair`, and
inspect both entry grades through `spatial_cells`. Geometric closure requires
retained face incidence; `:chain_closure` asserts only a based chain subcomplex.
"""
struct SpatialPair{S,P,B}
    source::S
    pair::P
    seeds::Vector{Int}
    selected::Vector{Int}
    added::Vector{Int}
    rule::Symbol
    brush::B
end
_spatial_complex(x::DataIngestion.FilteredComplex)=DataIngestion.graded_complex(x)
_spatial_complex(x::GradedComplex)=x

function _spatial_ids(ids,n)
    ids isa Union{AbstractVector,Tuple,AbstractSet} && all(i->i isa Integer && !(i isa Bool) && 1<=i<=n,ids) ||
        throw(ArgumentError("cells must contain global cell indices in 1:$n."))
    allunique(ids) || throw(ArgumentError("Repeated cell indices are not allowed."))
    sort(Int.(collect(ids)))
end
function _spatial_faces(x,rule,field)
    G=_spatial_complex(x);n=length(G.grades)
    if rule!==:chain_closure
        x isa DataIngestion.FilteredComplex || throw(ArgumentError("Geometric selection requires retained source cells and face incidence."))
        faces=DataIngestion.face_incidence(x)
        faces===nothing && throw(ArgumentError("Geometric selection requires retained face incidence; use faces= in filtered_complex, or explicitly choose :chain_closure."))
        return faces
    end
    K=coeff_type(field);offsets=G.dim_offsets;faces=[Int[] for _ in 1:n]
    for d in eachindex(G.boundaries)
        B=G.boundaries[d]
        for j in axes(B,2), p in nzrange(B,j)
            iszero(K(B.nzval[p])) || push!(faces[offsets[d+1]+j-1],offsets[d]+B.rowval[p]-1)
        end
    end
    faces
end

function _spatial_brush(x,brush,rule)
    x isa DataIngestion.FilteredComplex || throw(ArgumentError("A spatial brush requires retained source geometry."))
    rule===:chain_closure && throw(ArgumentError("A spatial brush needs an explicit geometric selection rule."))
    brush isa NamedTuple && keys(brush)==(:dims,:lower,:upper) || throw(ArgumentError("brush must be (;dims=(1,2),lower=(x0,y0),upper=(x1,y1))."))
    dims,lo,hi=brush.dims,brush.lower,brush.upper
    dims isa Tuple && length(dims) in (2,3) && allunique(dims) && all(d->d isa Int && 1<=d<=x.provenance.ambient_dim,dims) ||
        throw(ArgumentError("brush dims must select two or three distinct spatial coordinates."))
    lo isa Tuple && hi isa Tuple && length(lo)==length(hi)==length(dims) &&
        all(t->t isa Real && !(t isa Bool) && isfinite(t),(lo...,hi...)) && all(lo .<= hi) ||
        throw(ArgumentError("brush bounds must be finite, ordered coordinates."))
    ids=Int[]
    for i in eachindex(x.geometry)
        rule===:induced && x.complex.cell_dims[i]!=0 && continue
        vertices=x.geometry[i]
        # Divide integers exactly; Float coordinates keep their supplied precision.
        coordinate(k)=let values=[p[dims[k]] for p in vertices], total=all(v->v isa Integer,values) ? sum(BigInt,values) : sum(values)
            total isa Integer ? total//length(vertices) : total/length(vertices)
        end
        all(k->lo[k]<=coordinate(k)<=hi[k],eachindex(dims)) && push!(ids,i)
    end
    ids
end

"""
    spatial_pair(source; cells=Int[], source_ids=nothing, brush=nothing,
                 rule=:face_closure, grades=nothing, field=F2(), order=nothing)

Select global cells, exact source IDs, or a closed rectangular spatial brush.
`:face_closure` selects cell vertex-centroids inside the brush, then all faces.
`:induced` selects vertices and every cell whose vertices are selected.
`:chain_closure` closes nonzero boundaries in the chosen field and makes no
geometric assertion; it also accepts a matrix-only `GradedComplex`.

Brush coordinates are `(;dims=(1,2),lower=(x0,y0),upper=(x1,y1))`, with both
boundaries included. Projected brushes are extruded along omitted coordinates.
Periodic cells use the retained unwrapped drawing; no periodic wrapping or
cutting/subdivision is performed. Exact source IDs must identify unique cells.

`grades` is an optional dictionary from selected global indices (including
closure-added cells) to A-entry grades. Other entries inherit their K-grades.
Both filtrations and geometric face grades, when available, are validated.
The returned object copies inputs and does not compute persistence until asked.
"""
function spatial_pair(x::Union{DataIngestion.FilteredComplex,GradedComplex};cells=Int[],source_ids=nothing,
        brush=nothing,rule=:face_closure,grades=nothing,field=F2(),order=nothing)
    rule in (:face_closure,:induced,:chain_closure) || throw(ArgumentError("rule must be :face_closure, :induced or :chain_closure."))
    G=_spatial_complex(x)
    G isa GradedComplex{1} || throw(ArgumentError("Spatial pairs support one-parameter, single-grade complexes."))
    _normalize_field(field)
    expected=x isa DataIngestion.FilteredComplex && x.orientation==( -1, ) ? :superlevel : :sublevel
    order=order===nothing ? expected : order
    _normalize_order(order)
    x isa DataIngestion.FilteredComplex && order!=expected && throw(ArgumentError("order disagrees with source grade orientation."))
    x isa DataIngestion.FilteredComplex && DataIngestion.check_filtered_complex(x;throw=true)
    _validate_complex(G,order,coeff_type(field))
    n=length(G.grades);seeds=_spatial_ids(cells,n)
    (source_ids!==nothing)+(brush!==nothing)+!isempty(seeds)<=1 || throw(ArgumentError("Choose cells, source_ids or brush, not multiple selectors."))
    if source_ids!==nothing
        source_ids isa Union{AbstractVector,Tuple,AbstractSet} || throw(ArgumentError("source_ids must be a collection of exact identities."))
        x isa DataIngestion.FilteredComplex || throw(ArgumentError("source_ids requires retained source identity."))
        seeds=Int[]
        for id in source_ids
            matches=findall(isequal(id),x.sources)
            length(matches)==1 || throw(ArgumentError("Each source ID must identify exactly one cell; use global indices for ambiguous IDs."))
            push!(seeds,only(matches))
        end
        seeds=_spatial_ids(seeds,n)
    elseif brush!==nothing
        seeds=_spatial_brush(x,brush,rule)
    end
    faces=_spatial_faces(x,rule,field);chosen=falses(n)
    if rule===:induced
        all(i->G.cell_dims[i]==0,seeds) || throw(ArgumentError("An induced selection accepts only vertices."))
        chosen[seeds].=true
        for i in 1:n
            G.cell_dims[i]==0 && continue
            isempty(faces[i]) && throw(ArgumentError("Induced selection requires a vertex-bearing face structure for every positive-dimensional cell."))
            chosen[i]=all(j->chosen[j],faces[i])
        end
    else
        stack=copy(seeds)
        while !isempty(stack)
            i=pop!(stack);chosen[i] && continue
            chosen[i]=true;append!(stack,faces[i])
        end
    end
    selected=findall(chosen)
    grades===nothing || (grades isa AbstractDict && all(i->i isa Int && i in selected,keys(grades))) ||
        throw(ArgumentError("grades must map selected global cell indices to entry values."))
    offsets=G.dim_offsets
    ids=[[i-offsets[d]+1 for i in selected if G.cell_dims[i]==d-1] for d in eachindex(G.cells_by_dim)]
    values=[[grades===nothing ? G.grades[offsets[d]+i-1][1] : get(grades,offsets[d]+i-1,G.grades[offsets[d]+i-1][1]) for i in ids[d]] for d in eachindex(ids)]
    pair=filtered_pair(G,ids;grades=values,field,order)
    if rule!==:chain_closure
        entries=Dict(offsets[d]+i-1=>pair.grades[d][j] for d in eachindex(ids) for (j,i) in enumerate(ids[d]))
        for i in selected,j in faces[i]
            (order===:sublevel ? entries[j]<=entries[i] : entries[j]>=entries[i]) ||
                throw(ArgumentError("A-entry grades violate retained geometric face incidence."))
        end
    end
    SpatialPair(deepcopy(x),pair,copy(seeds),selected,setdiff(selected,seeds),rule,deepcopy(brush))
end
filtered_pair(s::SpatialPair)=deepcopy(s.pair)
spatial_pair_summary(s::SpatialPair)=merge(filtered_pair_summary(s.pair),(;kind=:spatial_pair,rule=s.rule,
    selected_count=length(s.selected),seed_count=length(s.seeds),added_count=length(s.added),
    geometry=s.rule===:chain_closure ? :not_asserted : :retained_face_incidence,brush=s.brush))
describe(s::SpatialPair)=spatial_pair_summary(s)
Base.show(io::IO,s::SpatialPair)=print(io,"SpatialPair(",s.rule,", ",length(s.selected)," cells, ",length(s.added)," added)")
"""Cell table with source identity, selection role and both exact entry grades."""
function spatial_cells(s::SpatialPair)
    G=s.pair.ambient;offsets=G.dim_offsets
    entries=Dict(offsets[d]+i-1=>s.pair.grades[d][j] for d in eachindex(s.pair.indices) for (j,i) in enumerate(s.pair.indices[d]))
    seeds=Set(s.seeds)
    [(;index=i,dimension=G.cell_dims[i],source=s.source isa DataIngestion.FilteredComplex ? deepcopy(s.source.sources[i]) : (G.cell_dims[i],G.cell_ids[i]),
       selected=haskey(entries,i),role=i in seeds ? :seed : haskey(entries,i) ? :added : :ambient,
       ambient_grade=G.grades[i][1],subcomplex_grade=get(entries,i,nothing)) for i in eachindex(G.grades)]
end
function check_spatial_pair(s::SpatialPair;throw=false)
    issues=String[]
    try
        entries=Dict(r.index=>r.subcomplex_grade for r in spatial_cells(s) if r.selected)
        rebuilt=spatial_pair(s.source;cells=s.seeds,rule=s.rule,grades=entries,field=s.pair.field,order=s.pair.order)
        _same_filtered_pair(rebuilt.pair,s.pair) && rebuilt.selected==s.selected && rebuilt.added==s.added || error("Selection and filtered pair disagree.")
        s.brush===nothing || _spatial_brush(s.source,s.brush,s.rule)==s.seeds || error("Brush and seed cells disagree.")
    catch err
        err isa InterruptException && rethrow();push!(issues,sprint(showerror,err))
    end
    throw && !isempty(issues) && Base.throw(ArgumentError(join(issues,"\n")))
    (;kind=:spatial_pair_validation,valid=isempty(issues),issues)
end
struct SpatialPairValidationSummary{R}
    report::R
end
SpatialPairValidationSummary(s::SpatialPair)=SpatialPairValidationSummary(check_spatial_pair(s))
describe(s::SpatialPairValidationSummary)=s.report
Base.show(io::IO,s::SpatialPairValidationSummary)=print(io,"SpatialPairValidationSummary(valid=",s.report.valid,")")
function persistence_diagram(s::SpatialPair;part=:relative)
    _relative_part(part)
    part===:relative && return persistence_diagram(s.pair)
    p=s.pair
    part===:ambient && return persistence_diagram(p.ambient;field=p.field,order=p.order)
    ids=p.indices
    G=GradedComplex([p.ambient.cells_by_dim[d][ids[d]] for d in eachindex(ids)],
        [p.ambient.boundaries[d][ids[d],ids[d+1]] for d in eachindex(p.ambient.boundaries)],
        [(g,) for group in p.grades for g in group])
    persistence_diagram(G;field=p.field,order=p.order)
end
relative_complex(s::SpatialPair;kwargs...)=relative_complex(s.pair;kwargs...)
relative_homology(s::SpatialPair;kwargs...)=relative_homology(s.pair;kwargs...)
