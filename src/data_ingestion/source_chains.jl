# Exact source correspondences. Geometry is owned by FilteredComplex; these
# records add actual chain maps, never a match inferred from nearby coordinates.

"""
    SourceCorrespondence

A retained chain map from a computed complex to source cells. A chain map
commutes with taking boundaries, so it sends a computed cycle to a source
cycle. Construct the correspondence with `source_correspondence`.

An optional retraction maps source chains back to computed chains and recovers
each computed chain after its lift. Its transpose lets a computed cocycle
evaluate source chains. Use `describe` to inspect retained counts, the field
and retraction availability, and `check_source_correspondence` to validate the
maps. Treat retained arrays as read only.
"""
struct SourceCorrespondence{C,S,K,F}
    computed::C
    source::S
    lift::Vector{SparseMatrixCSC{K,Int}}
    retraction::Union{Nothing,Vector{SparseMatrixCSC{K,Int}}}
    field::F
    certificate::Symbol
end

"""
    SourceChain

A chain or cochain on retained source cells at one parameter. Obtain it through
`lift_source_chain` or an ordinary persistence representative query. A chain
is a linear combination of cells; a cochain assigns values to those cells and
evaluates chains by linearity. The `kind` records whether the object is an
arbitrary chain, a cycle with zero boundary, or a cocycle vanishing on active
boundaries.

`describe` gives the degree, parameter, field and support size.
`source_chain_data` returns the nonzero coefficients and corresponding source
identities and geometry. `visualize` draws that retained geometry without
computing a new homology basis.
"""
struct SourceChain{S,K,F,P,M}
    source::S
    coefficients::Vector{K}
    dimension::Int
    parameter::P
    field::F
    kind::Symbol
    metadata::M
end

"""
    SourceValidationSummary

A validation result for a source correspondence or chain. `valid` reports
whether its required checks passed, and `issues` lists any failures. Use
`describe` for a structured result containing the validation kind as well.
"""
struct SourceValidationSummary
    kind::Symbol
    valid::Bool
    issues::Vector{String}
end
Base.show(io::IO,x::SourceValidationSummary)=print(io,"SourceValidationSummary(",x.kind,", valid=",x.valid,", issues=",length(x.issues),")")
describe(x::SourceValidationSummary)=(;kind=x.kind,valid=x.valid,issues=copy(x.issues))

function _source_validation(f,kind,do_throw)
    issues=String[]
    try; f()
    catch err
        err isa InterruptException && rethrow()
        push!(issues,sprint(showerror,err))
    end
    do_throw && !isempty(issues) && throw(ArgumentError(join(issues,"\n")))
    return SourceValidationSummary(kind,isempty(issues),issues)
end

_source_field_label(field)=field isa PrimeField ? "F$(field.p)" : "QQ"
_source_coefficient(field,a)=field isa PrimeField ? a.val : a

_source_counts(x)=DataTypes.cell_counts(graded_complex(x))
_source_offsets(x)=cumsum(vcat(0,_source_counts(x)))
_source_boundary(x,d,K)=d==0 ? spzeros(K,0,first(_source_counts(x))) :
    d>=length(_source_counts(x)) ? spzeros(K,last(_source_counts(x)),0) :
    SparseMatrixCSC{K,Int}(DataTypes.boundary_map(graded_complex(x);dim=d))
_source_active(x,i,p)=any(g->all(k->x.orientation[k]==1 ? g[k]<=p[k] : g[k]>=p[k],eachindex(p)),
    DataTypes.cell_grade_set(graded_complex(x),i))
function _source_parameter(x,parameter)
    N=DataTypes.parameter_dim(graded_complex(x))
    p=parameter isa Real && N==1 ? (parameter,) : parameter
    p isa Union{Tuple,AbstractVector} && length(p)==N && all(v->v isa Real && !(v isa Bool) && isfinite(v),p) ||
        throw(ArgumentError("parameter must contain $N finite real coordinates."))
    return Tuple(p)
end

function _check_filtered_maps(from,to,maps,K)
    a,b=_source_counts(from),_source_counts(to)
    length(a)==length(b)==length(maps) || error("Source maps must cover the same chain degrees (pad empty degrees explicitly).")
    from.orientation==to.orientation || error("Source maps need matching parameter orientations.")
    oa,ob=_source_offsets(from),_source_offsets(to)
    for d in eachindex(a)
        M=maps[d]
        size(M)==(b[d],a[d]) || error("Wrong map shape in dimension $(d-1).")
        d>1 && _source_boundary(to,d-1,K)*M!=maps[d-1]*_source_boundary(from,d-1,K) &&
            error("The map does not commute with boundary in dimension $(d-1).")
        rows,cols,vals=findnz(M)
        for (r,c,v) in zip(rows,cols,vals)
            iszero(v) && continue
            all(g->_source_active(to,ob[d]+r,g),DataTypes.cell_grade_set(graded_complex(from),oa[d]+c)) ||
                error("The map sends a cell to source cells not yet active at its appearance.")
        end
    end
end

"""
    check_source_correspondence(c; throw=false)

Check the two filtered complexes and their map equations over the retained
field. Each map must commute with taking boundaries and send active cells to
active cells. If a retraction is retained, also check
`retraction * lift == identity` in each degree.

Return a `SourceValidationSummary` containing `valid` and `issues`, or raise an
`ArgumentError` on failure when `throw=true`. The splitting equation makes the
lift injective on homology; it does not exclude additional source classes.
"""
function check_source_correspondence(c::SourceCorrespondence;throw=false)
    _source_validation(:source_correspondence,throw) do
        c.field isa Union{PrimeField,QQField} || error("Source correspondences require an exact rational or prime field.")
        check_filtered_complex(c.computed;throw=true);check_filtered_complex(c.source;throw=true)
        K=coeff_type(c.field)
        _check_filtered_maps(c.computed,c.source,c.lift,K)
        if c.retraction!==nothing
            _check_filtered_maps(c.source,c.computed,c.retraction,K)
            for d in eachindex(c.lift)
                c.retraction[d]*c.lift[d]==spdiagm(0=>ones(K,size(c.lift[d],2))) ||
                    error("Retraction does not split the lift in dimension $(d-1).")
            end
        end
    end
end

"""
    source_correspondence(computed, source=computed; field=PrimeField(2),
                          lift=nothing, retraction=nothing)

Retain checked maps between `FilteredComplex` objects. With one object, both
maps are identities. With two objects and no explicit maps, match exact source
identities and construct a retraction by canceling complementary cells with
equal appearance grade sets. The computed cells must form a subcomplex of the
source with the same oriented boundaries, and its complement must admit these
cancellations.

An explicit `lift` may instead be any filtered chain map: it commutes with
boundaries and respects cell appearances. A supplied retraction must recover
each computed chain after lifting it. This certifies an injection on homology
but permits additional source classes outside the image. Map rows index
destination cells and columns index domain cells within each degree. Supply
all retained degrees, including zero, and pad absent top degrees explicitly.

Exact rational and prime fields are supported. The result keeps both complexes;
algebraic cancellation can increase the number of nonzero matrix entries.
Use `describe` for counts and retained map information, and `lift_source_chain`
for a cycle or cocycle at one parameter. Several parameters and several
appearance grades per cell are supported. A module encoding without a retained
cell correspondence does not provide the data needed by this constructor.
"""
function source_correspondence(computed::FilteredComplex,source::FilteredComplex=computed;
        field=PrimeField(2),lift=nothing,retraction=nothing)
    field isa Union{PrimeField,QQField} || throw(ArgumentError("Source correspondences require an exact rational or prime field."))
    check_filtered_complex(computed;throw=true);check_filtered_complex(source;throw=true)
    K=coeff_type(field);a,b=_source_counts(computed),_source_counts(source)
    length(a)==length(b) || throw(ArgumentError("Pad empty top chain degrees before comparing complexes."))
    automatic=lift===nothing
    if automatic
        retraction===nothing || throw(ArgumentError("Supply lift together with an explicit retraction."))
        oa,ob=_source_offsets(computed),_source_offsets(source)
        lift=SparseMatrixCSC{K,Int}[]
        for d in eachindex(a)
            src=source.sources[ob[d]+1:ob[d+1]]
            allunique(src) || throw(ArgumentError("Source identities must be unique within each dimension for automatic lifting."))
            lookup=Dict(s=>i for (i,s) in enumerate(src))
            rows=[get(lookup,s,0) for s in computed.sources[oa[d]+1:oa[d+1]]]
            all(>(0),rows) && allunique(rows) || throw(ArgumentError("Computed cells do not inject into the supplied source identities."))
            push!(lift,sparse(rows,collect(1:a[d]),ones(K,a[d]),b[d],a[d]))
        end
    end
    maps=SparseMatrixCSC{K,Int}[SparseMatrixCSC{K,Int}(M) for M in lift]
    _source_validation(:source_correspondence,true) do
        _check_filtered_maps(computed,source,maps,K)
    end
    retract=automatic && computed===source ? copy(maps) : automatic ? _source_relative_retraction(computed,source,maps,K) :
        retraction===nothing ? nothing : SparseMatrixCSC{K,Int}[SparseMatrixCSC{K,Int}(M) for M in retraction]
    c=SourceCorrespondence(computed,source,maps,retract,field,
        automatic ? :equal_grade_contraction : :supplied_filtered_chain_map)
    check_source_correspondence(c;throw=true)
    return c
end

# Cancel an invertible relative incidence a <- b at exactly the same grade.
# The target is a subcomplex, hence these operations fix it pointwise. P tracks
# the actual retraction, including off-diagonal coefficients and orientations.
function _source_relative_retraction(computed,source,lift,::Type{K}) where K
    counts=_source_counts(source);off=_source_offsets(source);n=last(off)
    D=spzeros(K,n,n)
    for d in 1:length(counts)-1
        D[off[d]+1:off[d+1],off[d+1]+1:off[d+2]]=_source_boundary(source,d,K)
    end
    P=spdiagm(0=>ones(K,n));kept=Int[]
    for d in eachindex(lift)
        append!(kept,off[d].+Int[only(findall(!iszero,lift[d][:,j])) for j in axes(lift[d],2)])
    end
    extra=trues(n);extra[kept].=false
    grades=[Set(DataTypes.cell_grade_set(source.complex,i)) for i in 1:n]
    while any(extra)
        pair=nothing
        for b in findall(extra), z in nzrange(D,b)
            a=rowvals(D)[z]
            extra[a] && !iszero(nonzeros(D)[z]) && grades[a]==grades[b] || continue
            pair=(a,b);break
        end
        pair===nothing && throw(ArgumentError("The source complement has no equal-grade contraction; supply verified maps explicitly."))
        a,b=pair;v=D[:,b]/D[a,b]
        P=P-v*P[a:a,:];P[a,:].=zero(K);P[b,:].=zero(K);dropzeros!(P)
        D=D-v*D[a:a,:]
        D[a,:].=zero(K);D[b,:].=zero(K);D[:,a].=zero(K);D[:,b].=zero(K);dropzeros!(D)
        extra[a]=extra[b]=false
    end
    ca=_source_counts(computed);co=_source_offsets(computed)
    return SparseMatrixCSC{K,Int}[P[kept[co[d]+1:co[d+1]],off[d]+1:off[d+1]] for d in eachindex(ca)]
end

describe(c::SourceCorrespondence)=(;kind=:source_correspondence,computed_counts=_source_counts(c.computed),
    source_counts=_source_counts(c.source),field=c.field,certificate=c.certificate,
    cochain_lift_available=c.retraction!==nothing,orientation=c.source.orientation)
Base.show(io::IO,c::SourceCorrespondence)=print(io,"SourceCorrespondence(",sum(_source_counts(c.computed))," -> ",sum(_source_counts(c.source))," cells, ",_source_field_label(c.field),", ",c.certificate,")")

"""
    source_correspondence(data, filtration; field=PrimeField(2), cache=nothing)

Construct and retain source geometry through `filtered_complex`. When ingestion
collapses cells, retain the same selected model before collapse and check a
correspondence to it. Sampling and sparsification remain part of that model;
restoring collapsed cells does not restore discarded points or omitted edges.

The result is a `SourceCorrespondence`, which can be inspected with `describe`
and queried with `lift_source_chain`. Inputs without automatic geometry require
a supplied `FilteredComplex` with geometry and the required maps.
"""
function source_correspondence(data,filtration::AbstractFiltration;field=PrimeField(2),cache=nothing)
    if filtration isa Union{RipsFiltration,LandmarkRipsFiltration}
        actual,spec,selection=_rips_landmark_input(data,_filtration_spec(filtration))
        if selection!==nothing
            c=source_correspondence(actual,to_filtration(spec);field,cache)
            ids=landmark_indices(selection)
            function remap(x)
                sources=[Tuple(ids[v] for v in simplex) for simplex in x.sources]
                return filtered_complex(x.complex;geometry=x.geometry,source_ids=sources,
                    orientation=x.orientation,construction=string(x.provenance.effective),model=x.provenance.model,
                    provenance=merge(x.provenance,(;source_vertex_ids=ids,sampled=true,landmarks=selection)))
            end
            computed,source=remap(c.computed),remap(c.source)
            result=SourceCorrespondence(computed,source,c.lift,c.retraction,c.field,c.certificate)
            check_source_correspondence(result;throw=true)
            return result
        end
    end
    applicable(filtered_complex,data,filtration) ||
        throw(ArgumentError("This input has no automatic source geometry; supply a FilteredComplex with explicit coordinates."))
    computed=filtered_complex(data,filtration;cache)
    construction=construction_mode(filtration)
    construction.collapse==:none && return source_correspondence(computed;field)
    spec=_filtration_spec(filtration)
    uncollapsed=ConstructionOptions(;sparsify=construction.sparsify,collapse=:none,
        output_stage=construction.output_stage,budget=construction.budget)
    original=filtered_complex(data,to_filtration(FiltrationSpec(spec.kind,merge(spec.params,(;construction=uncollapsed))));cache)
    return source_correspondence(computed,original;field)
end

"""
    source_chain_data(chain)

Return the nonzero coefficients of a source chain together with their cell
indices, source identities, oriented geometry and appearance grades. The result
also gives the degree, chain kind, parameter, field, construction information
and verification or optimization metadata. Over a prime field, coefficients
are represented by their canonical integer residues.

Cell indices are positions within the chain degree. Source identities refer
to the input model and are retained separately from those positions.
"""
function source_chain_data(c::SourceChain)
    ids=findall(!iszero,c.coefficients);off=_source_offsets(c.source)[c.dimension+1]
    return (;dimension=c.dimension,kind=c.kind,parameter=c.parameter,field=c.field,
        cell_indices=Tuple(ids),cell_ids=Tuple(DataTypes.cells(c.source.complex;dim=c.dimension)[ids]),
        coefficients=Tuple(_source_coefficient(c.field,c.coefficients[i]) for i in ids),
        source_ids=Tuple(deepcopy(c.source.sources[off+i]) for i in ids),
        geometry=Tuple(deepcopy(c.source.geometry[off+i]) for i in ids),
        grades=Tuple(Tuple(DataTypes.cell_grade_set(c.source.complex,off+i)) for i in ids),
        construction=c.source.provenance,metadata=c.metadata)
end
describe(c::SourceChain)=(;kind=:source_chain,chain_kind=c.kind,dimension=c.dimension,parameter=c.parameter,
    field=c.field,cell_count=length(c.coefficients),support=count(!iszero,c.coefficients),metadata=c.metadata)
Base.show(io::IO,c::SourceChain)=print(io,"SourceChain(",c.kind,", dimension=",c.dimension,", support=",count(!iszero,c.coefficients),", ",_source_field_label(c.field),")")

function _source_chain_equations(c)
    x=c.source;K=coeff_type(c.field);d=c.dimension;counts=_source_counts(x)
    0<=d<length(counts) || error("Chain dimension is outside the complex.")
    length(c.coefficients)==counts[d+1] || error("Chain coefficient count is wrong.")
    c.kind in (:chain,:cycle,:cocycle) || error("kind must be :chain, :cycle or :cocycle.")
    p=_source_parameter(x,c.parameter);off=_source_offsets(x)
    all(i->iszero(c.coefficients[i]) || _source_active(x,off[d+1]+i,p),eachindex(c.coefficients)) ||
        error("Chain support includes inactive source cells.")
    if c.kind==:cycle
        iszero(_source_boundary(x,d,K)*c.coefficients) || error("The source chain is not a cycle.")
    elseif c.kind==:cocycle
        B=_source_boundary(x,d+1,K)
        active=d+2>length(counts) ? Int[] : [i for i in 1:counts[d+2] if _source_active(x,off[d+2]+i,p)]
        iszero(transpose(B[:,active])*c.coefficients) || error("The source cochain is not a cocycle at the declared parameter.")
    end
end

"""
    check_source_chain(chain; throw=false)

Check the source complex, support activity and the equation required by the
chain kind. A cycle must have zero boundary; a cocycle must vanish on every
active boundary. An arbitrary `:chain`, such as a death filling, need not
satisfy either equation on its own.

Return a `SourceValidationSummary` with `valid` and `issues`, or raise an
`ArgumentError` on failure when `throw=true`.
"""
function check_source_chain(c::SourceChain;throw=false)
    _source_validation(:source_chain,throw) do
        check_filtered_complex(c.source;throw=true)
        _source_chain_equations(c)
    end
end

"""
    propagation_source_chain(r::VectorPropagation; target, basis, column=1)

Express one propagated column as a source-cell cycle using an explicitly
supplied homology basis at the target. `basis` contains one `SourceChain` cycle
per target stalk coordinate, on the identical complex, field, degree and
parameter. Activity, cycle equations and completeness/independence modulo
boundaries are checked. The caller declares that their order identifies the
ambient module's target coordinates; equal dimensions alone cannot certify
that identification. The returned metadata records this declaration.

This is an explicit A43 source link, not recovery of a discarded basis and not
a contravariant cocycle transport. An empty basis cannot identify source cells.
The retained source-cycle route requires QQ or prime-field coefficients.
"""
function propagation_source_chain(r::AbelianCategories.VectorPropagation;
        target, basis, column=1)
    r.ambient.field isa Union{QQField,PrimeField} ||
        throw(ArgumentError("Retained source-cycle links require exact QQ or prime-field coefficients."))
    AbelianCategories.check_propagation(r;throw=true)
    A=AbelianCategories.propagation_image(r;target)
    A===nothing && throw(ArgumentError("No forward map exists to the selected target."))
    column isa Integer && !(column isa Bool) && 1<=column<=size(A,2) ||
        throw(ArgumentError("column must select one supplied generator column."))
    basis isa AbstractVector && !isempty(basis) && all(b->b isa SourceChain,basis) ||
        throw(ArgumentError("basis must be a nonempty vector of retained SourceChain cycles."))
    length(basis)==size(A,1) || throw(DimensionMismatch("One source cycle is required per target coordinate."))
    b=first(basis);x=b.source;d=b.dimension;field=r.ambient.field;K=coeff_type(field)
    b.field==field || throw(ArgumentError("Source cycles and propagation must use the same field."))
    for c in basis
        c.source===x && c.dimension==d && c.parameter==b.parameter && c.field==field && c.kind===:cycle ||
            throw(ArgumentError("Basis cycles must share source, degree, parameter and field."))
        check_source_chain(c;throw=true)
    end
    counts=_source_counts(x);off=_source_offsets(x)
    active=[i for i in 1:counts[d+1] if _source_active(x,off[d+1]+i,b.parameter)]
    next=d+2>length(counts) ? Int[] : [i for i in 1:counts[d+2] if _source_active(x,off[d+2]+i,b.parameter)]
    B=_source_boundary(x,d+1,K)[:,next]
    Z=hcat((c.coefficients for c in basis)...)
    rb=FieldLinAlg.rank(field,B)
    h=length(active)-FieldLinAlg.rank(field,_source_boundary(x,d,K)[:,active])-rb
    length(basis)==h && FieldLinAlg.rank(field,hcat(B,Z))==rb+h ||
        throw(ArgumentError("The supplied cycles are not a complete independent homology basis."))
    out=SourceChain(x,collect(Z*A[:,column]),d,b.parameter,field,:cycle,
        (;choice=:selected_propagated_column,source_vertex=r.source,target_vertex=Int(target),
          column=Int(column),basis_identification=:supplied_by_caller,
          homology_basis=:verified,source_equation=:checked,optimization=:none))
    check_source_chain(out;throw=true)
    return out
end

"""
    lift_source_chain(c; dim, coefficients, parameter, kind=:cycle)

Apply a retained `SourceCorrespondence` to a coefficient vector indexed by
computed cells of degree `dim`. Check activity and the required cycle or
cocycle equation before and after mapping, then return a `SourceChain` on the
source cells at `parameter`.

With `kind=:cocycle`, use the transpose of the retained retraction and restrict
the result to active source cells. A computed cocycle evaluates a source chain
by first mapping that chain back to the computed complex. A lift without a
retraction cannot perform this operation. Use `kind=:chain` for arbitrary
chains, including fillings; this mode checks activity without imposing a cycle
equation.

Coefficients must be integers, elements of the retained field, or rational
numbers over `QQField()`. Supply one coefficient per computed cell in the
chosen degree and a parameter tuple with one entry per filtration coordinate;
a scalar is accepted for one parameter. The supplied maps determine the basis
correspondence.
"""
function lift_source_chain(c::SourceCorrespondence;dim,coefficients,parameter,kind=:cycle)
    dim isa Int && 0<=dim<length(c.lift) || throw(ArgumentError("dim must index a retained chain degree."))
    check_source_correspondence(c;throw=true)
    K=coeff_type(c.field);p=_source_parameter(c.computed,parameter)
    coefficients isa Union{Tuple,AbstractVector} &&
        length(coefficients)==_source_counts(c.computed)[dim+1] &&
        all(a->a isa Integer || (c.field isa QQField && a isa Rational) || a isa K,coefficients) ||
        throw(ArgumentError("coefficients must be a dimension-local vector in the declared exact field (integers, matching field elements, or rationals over QQ)."))
    v=K.(collect(coefficients))
    input=SourceChain(c.computed,v,dim,p,c.field,kind,(;))
    check_source_chain(input;throw=true)
    if kind==:cocycle
        c.retraction===nothing && throw(ArgumentError("Cocycle lifting needs a retained verified retraction."))
        w=transpose(c.retraction[dim+1])*v
        off=_source_offsets(c.source)[dim+1]
        for i in eachindex(w);_source_active(c.source,off+i,p) || (w[i]=zero(K));end
    else
        w=c.lift[dim+1]*v
    end
    out=SourceChain(c.source,collect(w),dim,p,c.field,kind,
        (;correspondence=:verified_chain_map,certificate=c.certificate,choice=:supplied_coordinates,
          equation=:checked_on_source,optimization=:none))
    check_source_chain(out;throw=true)
    return out
end

"""
    optimize_representative(c; weights=nothing, max_candidates=100000)

Find a cycle of minimum weighted support in the homology class of the nonzero
`SourceChain` `c`, at its declared parameter and over its prime field. Weights
are finite nonnegative real numbers, one per source cell of that degree; the
default gives every cell weight one. The objective sums weights at cells with
nonzero coefficients, regardless of coefficient magnitude.

Enumerate the active boundary image exactly, refusing the problem before
search if the number of candidates exceeds `max_candidates`. Ties keep the
first enumerated chain. The returned source cycle records its weights,
objective, candidate count and an explicit chain whose boundary is the
difference from the original cycle. This certifies the unchanged nonzero class.
The minimum uses the declared cell weights and active complex; it need not
minimize Euclidean length or remain minimal at another parameter.
"""
function optimize_representative(c::SourceChain;weights=nothing,max_candidates=100000)
    check_source_chain(c;throw=true)
    c.kind==:cycle || throw(ArgumentError("Optimization requires a cycle."))
    c.field isa PrimeField || throw(ArgumentError("Finite representative enumeration requires a prime field."))
    max_candidates isa Int && max_candidates>0 || throw(ArgumentError("max_candidates must be a positive Int."))
    x=c.source;d=c.dimension;K=coeff_type(c.field);n=length(c.coefficients);counts=_source_counts(x);off=_source_offsets(x)
    w=weights===nothing ? ones(Int,n) : collect(weights)
    length(w)==n && all(v->v isa Real && !(v isa Bool) && isfinite(v) && v>=0,w) ||
        throw(ArgumentError("weights must be finite nonnegative real values, one per source cell in this degree."))
    # Rationalizing the represented floating value makes comparisons exact.
    all(v->v isa Union{Integer,Rational,AbstractFloat},w) || throw(ArgumentError("Exact support objectives accept integer, rational or floating weights."))
    exact=Rational{BigInt}.(w)
    B=_source_boundary(x,d+1,K)
    active=d+2>length(counts) ? Int[] : [i for i in 1:counts[d+2] if _source_active(x,off[d+2]+i,c.parameter)]
    A=B[:,active];_,pivot_tuple=FieldLinAlg.rref(c.field,A)
    pivots=collect(pivot_tuple)
    basis=A[:,pivots];r=length(pivots)
    FieldLinAlg.rank(c.field,hcat(basis,c.coefficients))==r &&
        throw(ArgumentError("The input represents the zero homology class at this parameter."))
    candidates=big(c.field.p)^r
    candidates<=max_candidates || throw(ArgumentError("Exact optimization needs $candidates candidates, exceeding max_candidates=$max_candidates."))
    objective(v)=sum((exact[i] for i in eachindex(v) if !iszero(v[i]));init=zero(Rational{BigInt}))
    best=copy(c.coefficients);cost=objective(best);winner=zeros(K,r)
    for code in 1:Int(candidates)-1
        q=code;coordinates=zeros(K,r)
        for j in 1:r;coordinates[j]=K(q%c.field.p);q=div(q,c.field.p);end
        candidate=c.coefficients+basis*coordinates;value=objective(candidate)
        if value<cost;best=candidate;cost=value;winner=coordinates;end
    end
    witness=zeros(K,size(B,2));witness[active[pivots]]=winner
    B*witness==best-c.coefficients || error("Internal optimization witness failure.")
    metadata=merge(c.metadata,(;optimization=:exact_weighted_support,objective=cost,weights=Tuple(exact),
        candidates=Int(candidates),admissible_chains=:active_source_complex,nonzero_class=true,
        boundary_adjustment=Tuple(getfield.(witness,:val)),original_coefficients=Tuple(getfield.(c.coefficients,:val))))
    out=SourceChain(x,best,d,c.parameter,c.field,:cycle,metadata)
    check_source_chain(out;throw=true)
    return out
end
