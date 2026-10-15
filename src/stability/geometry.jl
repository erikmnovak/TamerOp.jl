# Ambient selected queries reuse the finite algebra with exact clipped masses.
"""A finite encoding with an explicit bounded observation window and measure.
`SkyscraperEncoding(M, pi; box, density=1)` accepts box or rational polyhedral
classifiers matching the vertices of `M`. An encoding result can replace `M,pi`.
The window is left closed and right open, with zero extension. Positive rational
`density` is constant on each encoding region (a scalar applies everywhere).
Classifier storage is read-only. Lower-dimensional strata have zero Lebesgue
mass; a query with a nonzero zero-mass generated factor is rejected.
"""
struct SkyscraperEncoding{N,M,P}
    finite::M
    classifier::P
    box::Tuple{NTuple{N,QQ},NTuple{N,QQ}}
    density::Vector{QQ}
    function SkyscraperEncoding(M::PModule,pi;box,density=1,budget::HNBudget=HNBudget())
        _input(M,budget);raw=EC.encoding_map(pi)
        raw isa Union{PLB.PLEncodingMapBoxes,PLP.PLEncodingMap} ||
            throw(ArgumentError("geometric skyscraper queries require a box or rational polyhedral encoding"))
        N=EC.dimension(raw);lo,hi=map(x->Tuple(_rational.(x)),box)
        length(lo)==length(hi)==N && all(lo .< hi) ||
            throw(ArgumentError("supply finite increasing box corners of the encoding dimension"))
        n=length(M.dims)
        length(EC.representatives(raw))==n || throw(DimensionMismatch("encoding regions do not match module vertices"))
        signatures=[region_signature(raw,v) for v in 1:n]
        Q=SignaturePoset([BitVector(s.y) for s in signatures],[BitVector(s.z) for s in signatures])
        Set(cover_edges(Q))==Set(cover_edges(M.Q)) || throw(ArgumentError("encoding and module posets disagree"))
        rho=_density(density,n)
        masses=RG.region_weights_exact(raw;box=(lo,hi),max_states=budget.max_states,max_entries=budget.max_entries)
        sum(masses)==prod(hi.-lo) || throw(ArgumentError("encoding regions do not cover the observation window exactly"))
        source=_snapshot(M)
        new{N,typeof(source),typeof(raw)}(source,raw,(lo,hi),rho)
    end
end
SkyscraperEncoding(enc::Results.EncodingResult;kwargs...)=
    SkyscraperEncoding(Results.encoding_module(enc),Results.encoding_map(enc);kwargs...)
_support_box(g::SkyscraperGrid)=(map(first,g.axes),map(last,g.axes))
_support_box(g::SkyscraperEncoding)=g.box
function _grid_label(g::SkyscraperEncoding{N},x) where N
    x isa Union{Tuple,AbstractVector} && length(x)==N || throw(ArgumentError("wrong parameter dimension"))
    p=Tuple(_rational.(x));lo,hi=g.box
    all(lo .<= p) && all(p .< hi) || return 0,p
    v=EC.locate(g.classifier,p;mode=:verified)
    v>0 || throw(ArgumentError("anchor is not covered by the encoding"))
    v,p
end
function skyscraper(g::SkyscraperEncoding;anchor,witnesses::Bool=false,budget::HNBudget=HNBudget())
    _input(g.finite,budget);a,p=_grid_label(g,anchor)
    a>0 || throw(ArgumentError("anchor is outside the declared half-open observation window"))
    weights=g.density.*RG.region_weights_exact(g.classifier;box=(p,g.box[2]),
        max_states=budget.max_states,max_entries=budget.max_entries)
    M=_snapshot(g.finite);w=_Work(0,0);data,flag,route=_anchor_query_data(M,a,weights,budget,w)
    context=(;category=:bounded_geometric_encoding,measure=_measure(g),density=copy(g.density),
        anchor=p,box=g.box,extension=:zero,window=:left_closed_right_open,
        generated_support=:upper_orthant,route)
    _anchor_result(M,a,weights,data,budget,w,context,witnesses;flag)
end
function skyscraper_rank(g::SkyscraperEncoding;source,target,threshold=0,budget::HNBudget=HNBudget())
    a,p=_grid_label(g,source);b,q=_grid_label(g,target)
    all(p .<= q) || throw(ArgumentError("source must precede target coordinatewise"))
    _rational(threshold)
    (a==0 || b==0) && return 0
    skyscraper_rank(skyscraper(g;anchor=p,budget);target=b,threshold)
end
describe(g::SkyscraperEncoding)=(;kind=:skyscraper_encoding,box=g.box,regions=length(g.density),
    measure=_measure(g),density=copy(g.density),extension=:zero,window=:left_closed_right_open,field=g.finite.field)
Base.show(io::IO,g::SkyscraperEncoding)=print(io,"SkyscraperEncoding(",length(g.density)," regions, ",length(g.box[1])," parameters)")

field(g::Union{SkyscraperGrid,SkyscraperEncoding})=g.finite.field
dimensions(g::Union{SkyscraperGrid,SkyscraperEncoding})=copy(g.finite.dims)
provenance(g::Union{SkyscraperGrid,SkyscraperEncoding})=describe(g)
function check_skyscraper(g::Union{SkyscraperGrid,SkyscraperEncoding};
        budget::HNBudget=HNBudget(),throw::Bool=false)
    issues=String[]
    try
        if g isa SkyscraperGrid
            SkyscraperGrid(g.finite,g.axes;density=g.density,budget)
        else
            SkyscraperEncoding(g.finite,g.classifier;box=g.box,density=g.density,budget)
        end
    catch err
        err isa InterruptException && rethrow()
        push!(issues,sprint(showerror,err))
    end
    throw && !isempty(issues) && Base.throw(ArgumentError(join(issues,"; ")))
    HNValidationSummary(isempty(issues),issues,false)
end
