# Explicit identifications between finite-module coordinates and source homology.
"""
    source_homology_basis(source; degree, basis)

Retain the source complex and an explicit coordinate identification for a slice
inspector. `basis(label, point)` must return a matrix whose columns are source
cycles in the order of the encoded stalk coordinates. Rows index all source
cells of `degree`, including inactive cells (whose coefficients must be zero).

The inspector checks completeness modulo boundaries and compatibility with
every requested structure map. Equal dimensions alone do not identify bases.
The supplier must be deterministic and the source must be treated as read only.
Nothing is evaluated until source classes are requested.
"""
struct SourceHomologyBasis
    source::FilteredComplex
    degree::Int
    # A user callback is an orchestration boundary, not a numeric kernel.
    # Erase its type so every inspector helper need not specialize on it.
    basis::Any
end
function source_homology_basis(source::FilteredComplex;degree,basis)
    degree isa Integer && !(degree isa Bool) && 0<=degree<length(_source_counts(source)) ||
        throw(ArgumentError("degree must index a retained source chain group."))
    check_filtered_complex(source;throw=true)
    SourceHomologyBasis(source,Int(degree),basis)
end
describe(b::SourceHomologyBasis)=(;kind=:source_homology_basis,degree=b.degree,
    cells=_source_counts(b.source)[b.degree+1],identification=:supplied,
    validation=:at_requested_parameters_and_maps)
Base.show(io::IO,b::SourceHomologyBasis)=print(io,"SourceHomologyBasis(degree=",b.degree,", supplied coordinates)")

function _slice_source_basis(b::SourceHomologyBasis,label,point,dimension,field,budget)
    x=b.source;d=b.degree;K=coeff_type(field);counts=_source_counts(x)
    field isa Union{QQField,PrimeField} || throw(ArgumentError("Source classes require an exact field."))
    p=_source_parameter(x,point);off=_source_offsets(x)
    big(counts[d+1])*(dimension+counts[d+1])<=budget ||
        throw(ArgumentError("Source basis exceeds the slice algebra entry budget."))
    raw=b.basis(label,p)
    raw isa AbstractMatrix && size(raw)==(counts[d+1],dimension) ||
        throw(ArgumentError("Source basis must have one row per degree-$d cell and one column per stalk coordinate."))
    all(a->a isa Integer || (field isa QQField && a isa Rational) || a isa K,raw) ||
        throw(ArgumentError("Source basis coefficients must belong to the declared exact field."))
    Z=K.(raw)
    for j in axes(Z,2)
        check_source_chain(SourceChain(x,Z[:,j],d,p,field,:cycle,(;));throw=true)
    end
    active=[i for i in 1:counts[d+1] if _source_active(x,off[d+1]+i,p)]
    next=d+2>length(counts) ? Int[] : [i for i in 1:counts[d+2] if _source_active(x,off[d+2]+i,p)]
    big(counts[d+1])*length(next)<=budget || throw(ArgumentError("Source boundaries exceed the slice algebra entry budget."))
    B=_source_boundary(x,d+1,K)[:,next]
    rb=FieldLinAlg.rank(field,B)
    h=length(active)-FieldLinAlg.rank(field,_source_boundary(x,d,K)[:,active])-rb
    dimension==h && FieldLinAlg.rank(field,hcat(B,Z))==rb+h ||
        throw(ArgumentError("Supplied source cycles are not a complete independent homology basis at $p."))
    (;point=p,label,basis=Z,boundaries=B,next)
end

function _slice_source_map(field,a,b,A)
    difference=a.basis-b.basis*A
    FieldLinAlg.rank(field,hcat(b.boundaries,difference))==FieldLinAlg.rank(field,b.boundaries) ||
        throw(ArgumentError("Encoded structure map disagrees with the supplied source bases modulo boundaries."))
    nothing
end

function _slice_source_cycle(b::SourceHomologyBasis,data,coordinates,field)
    c=SourceChain(b.source,vec(data.basis*coordinates),b.degree,data.point,field,:cycle,
        (;choice=:slice_interval_basis,homology_basis=:verified,basis_identification=:supplied,
          source_equation=:checked,optimization=:none))
    check_source_chain(c;throw=true)
    c
end

function _slice_source_filling(b::SourceHomologyBasis,data,difference,field)
    B=data.boundaries;_,pivots=FieldLinAlg.rref(field,B)
    columns=Int[p for p in pivots]
    coordinates=vec(FieldLinAlg.solve_fullcolumn(field,B[:,columns],reshape(difference,:,1)))
    counts=_source_counts(b.source);d=b.degree+1;K=coeff_type(field)
    d<length(counts) || return nothing
    v=zeros(K,counts[d+1]);v[data.next[columns]]=coordinates
    _source_boundary(b.source,d,K)*v==difference || error("Source filling failed its boundary equation.")
    c=SourceChain(b.source,v,d,data.point,field,:chain,(;choice=:verified_boundary_filling))
    check_source_chain(c;throw=true)
    c
end
