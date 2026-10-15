# Rational cell volumes for exact geometric invariants. Box coordinates retain
# their represented binary values; no tolerance-based rationalization occurs.
function region_weights_exact(pi::PLEncodingMapBoxes;box,max_states=1_000_000,max_entries=8_000_000)
    a,b=map(x->QQ.(collect(x)),box)
    length(a)==length(b)==pi.n && all(isfinite,a) && all(isfinite,b) && all(a .<= b) ||
        throw(ArgumentError("supply finite ordered box corners of the encoding dimension"))
    max_states>=0 && max_entries>=0 || throw(ArgumentError("geometry budgets must be nonnegative"))
    length(pi.sig_y)<=max_entries || throw(ArgumentError("exact volume output budget exceeded"))
    weights=zeros(QQ,length(pi.sig_y));states=0
    walls=ntuple(j->sort!(unique(vcat(a[j],QQ[x for x in pi.coords[j] if a[j]<x<b[j]],b[j]))),pi.n)
    shape=map(x->length(x)-1,walls)
    prod(big.(shape))<=max_states || throw(ArgumentError("exact volume cell budget exceeded"))
    for I in CartesianIndices(shape)
        states+=1
        lo=ntuple(j->walls[j][I[j]],pi.n);hi=ntuple(j->walls[j][I[j]+1],pi.n)
        point=ntuple(j->(lo[j]+hi[j])/2,pi.n)
        r=locate(pi,point;mode=:verified)
        r>0 || throw(ArgumentError("encoding does not cover the observation window"))
        weights[r]+=prod(hi.-lo)
    end
    weights
end
region_weights_exact(pi::CompiledEncoding{<:PLEncodingMapBoxes};kwargs...)=
    region_weights_exact(_unwrap_encoding(pi);kwargs...)
