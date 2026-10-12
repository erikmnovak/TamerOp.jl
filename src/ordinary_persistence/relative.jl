# Checked filtered coordinate subcomplexes, relative persistence, and the
# covariant maps of quotient chains and their long exact homology sequence.

"""
    FilteredPair

A finite one-parameter pair `(K_t, A_t)` over a prime field. Construct it with
`filtered_pair(K, cells; grades=nothing, field=F2(), order=:sublevel)`.
The subcomplex is specified by dimension-local positions in K, not cell IDs.
Inputs are copied. Treat stored arrays as read-only; `check_filtered_pair`
rechecks the field, filtration and subcomplex conditions.
"""
struct FilteredPair{T<:Real}
    ambient::GradedComplex{1,T}
    indices::Vector{Vector{Int}}
    grades::Vector{Vector{T}}
    field::PrimeField
    order::Symbol
end

"""
    filtered_pair(K, cells; grades=nothing, field=F2(), order=:sublevel)

Construct a checked filtered pair. `K` is a one-parameter `GradedComplex`.
`cells[d+1]` lists the distinct dimension-local positions of A's d-cells in K;
give one vector per ambient chain degree, including empty vectors. IDs need
not be unique. The positions are sorted together with their supplied grades.

By default A inherits each selected cell's K-grade. To allow a cell to enter A
later, give `grades[d+1]` in the same order as `cells[d+1]`. Values must convert
exactly to K's stored grade type. Sublevels require K-grade <= A-grade;
superlevels reverse this inequality. Boundary support and boundary grades must
close in A over the chosen field. This is a based chain-subcomplex contract;
geometric realizability is not inferred from integer cell IDs.

Use `persistence_diagram(pair)` for intervals; request `relative_complex` or
`relative_homology` at a scale only when explicit chains or maps are needed.
"""
function filtered_pair(G::GradedComplex, cells::AbstractVector;
                       grades=nothing, field=F2(), order::Symbol=:sublevel)
    _normalize_field(field); _normalize_order(order)
    offsets = _validate_complex(G, order, coeff_type(field))
    counts = diff(offsets)
    length(cells) == length(counts) || throw(ArgumentError("give one cell-index vector per ambient chain degree"))
    grades === nothing || (grades isa AbstractVector && length(grades) == length(counts)) ||
        throw(ArgumentError("give one grade vector per ambient chain degree"))
    return _filtered_pair(G, cells, grades, field, order, counts, offsets)
end

function _filtered_pair(G::GradedComplex{1,T}, cells, grades, field, order, counts, offsets) where {T}
    indices = Vector{Int}[]
    values = Vector{T}[]
    for d in eachindex(counts)
        cs = cells[d]
        cs isa AbstractVector && all(i -> i isa Integer && !(i isa Bool) && 1 <= i <= counts[d], cs) ||
            throw(ArgumentError("invalid dimension-local subcomplex indices in degree $(d-1)"))
        ids = Int.(cs)
        allunique(ids) || throw(ArgumentError("duplicate subcomplex index in degree $(d-1)"))
        gs = grades === nothing ? [G.grades[offsets[d]+i-1][1] for i in ids] : grades[d]
        gs isa AbstractVector && length(gs) == length(ids) ||
            throw(ArgumentError("subcomplex grade count disagrees in degree $(d-1)"))
        typed = T[]
        for g in gs
            g isa Real && isfinite(g) || throw(ArgumentError("subcomplex grades must be finite real values"))
            v = try convert(T, g) catch; throw(ArgumentError("subcomplex grade is not representable in $T")) end
            v == g || throw(ArgumentError("subcomplex grade conversion to $T loses precision"))
            push!(typed, v)
        end
        perm = sortperm(ids)
        push!(indices, ids[perm]); push!(values, typed[perm])
    end
    pair = FilteredPair(deepcopy(G), indices, values, field, order)
    check_filtered_pair(pair; throw=true)
    return pair
end

function _validate_filtered_pair(pair::FilteredPair)
    _normalize_field(pair.field); _normalize_order(pair.order)
    K = coeff_type(pair.field)
    offsets = _validate_complex(pair.ambient, pair.order, K)
    counts = diff(offsets)
    isempty(counts) && throw(ArgumentError("a filtered pair needs at least its degree-zero cell vector"))
    length(pair.indices) == length(pair.grades) == length(counts) ||
        throw(ArgumentError("subcomplex degree counts disagree"))
    positions = Vector{Dict{Int,Int}}()
    before(a,b) = pair.order === :sublevel ? a <= b : a >= b
    for d in eachindex(counts)
        ids, grades = pair.indices[d], pair.grades[d]
        length(ids) == length(grades) && issorted(ids) && allunique(ids) &&
            all(i -> 1 <= i <= counts[d], ids) && all(isfinite, grades) ||
            throw(ArgumentError("invalid subcomplex indices or grades in degree $(d-1)"))
        all(j -> before(pair.ambient.grades[offsets[d]+ids[j]-1][1], grades[j]), eachindex(ids)) ||
            throw(ArgumentError("a subcomplex cell enters before its ambient cell in degree $(d-1)"))
        push!(positions, Dict(i=>j for (j,i) in enumerate(ids)))
    end
    for d in 2:length(counts)
        B = pair.ambient.boundaries[d-1]
        for (j, col) in enumerate(pair.indices[d]), ptr in nzrange(B, col)
            iszero(K(B.nzval[ptr])) && continue
            row = B.rowval[ptr]
            i = get(positions[d-1], row, 0)
            i != 0 || throw(ArgumentError("subcomplex is not boundary-closed in degree $(d-1)"))
            before(pair.grades[d-1][i], pair.grades[d][j]) ||
                throw(ArgumentError("subcomplex boundary grades violate the $(pair.order) filtration"))
        end
    end
    return nothing
end

"""`check_filtered_pair(pair; throw=false)` checks boundaries, field and both filtrations."""
function check_filtered_pair(pair::FilteredPair; throw::Bool=false)
    issues = String[]
    try
        _validate_filtered_pair(pair)
    catch e
        e isa ArgumentError || rethrow()
        push!(issues, sprint(showerror,e))
    end
    throw && !isempty(issues) && Base.throw(ArgumentError(join(issues,"; ")))
    return (kind=:filtered_pair, valid=isempty(issues), issues=Tuple(issues))
end

"""Readable validation result for a filtered pair; `describe` returns its report."""
struct FilteredPairValidationSummary{R<:NamedTuple}
    report::R
end
FilteredPairValidationSummary(pair::FilteredPair) = FilteredPairValidationSummary(check_filtered_pair(pair))
describe(s::FilteredPairValidationSummary) = s.report
Base.:(==)(a::FilteredPairValidationSummary,b::FilteredPairValidationSummary) = a.report == b.report
Base.isequal(a::FilteredPairValidationSummary,b::FilteredPairValidationSummary) = isequal(a.report,b.report)
Base.hash(s::FilteredPairValidationSummary,h::UInt) = hash(s.report,h)
Base.show(io::IO,s::FilteredPairValidationSummary) = print(io,"FilteredPairValidationSummary(valid=",s.report.valid,", issues=",s.report.issues,")")

"""`filtered_pair_summary(pair)` reports field, order and ambient/subcomplex cell counts."""
filtered_pair_summary(pair::FilteredPair) = (kind=:filtered_pair, field=pair.field, order=pair.order,
    ambient_cells=Tuple(diff(getfield(pair.ambient,:dim_offsets))), subcomplex_cells=Tuple(length.(pair.indices)))
describe(pair::FilteredPair) = filtered_pair_summary(pair)
field(pair::FilteredPair) = pair.field
filtration_order(pair::FilteredPair) = pair.order
Base.show(io::IO,pair::FilteredPair) = print(io,"FilteredPair(",pair.order,", ",pair.field,
    ", K=",filtered_pair_summary(pair).ambient_cells,", A=",filtered_pair_summary(pair).subcomplex_cells,")")

function _relative_provenance(pair::FilteredPair)
    return (kind=:filtered_pair, field=pair.field, order=pair.order,
        ambient_cell_ids=Tuple(Tuple(v) for v in pair.ambient.cells_by_dim),
        ambient_cell_grades=Tuple(g[1] for g in pair.ambient.grades),
        subcomplex_indices=Tuple(Tuple(v) for v in pair.indices),
        subcomplex_cell_ids=Tuple(Tuple(pair.ambient.cells_by_dim[d][v]) for (d,v) in enumerate(pair.indices)),
        subcomplex_grades=Tuple(Tuple(v) for v in pair.grades),
        indexing=:dimension_local_ambient_cells, connecting_sign=:boundary_of_ambient_lift,
        geometry=:not_asserted)
end
provenance(pair::FilteredPair) = _relative_provenance(pair)

function _relative_reduction_complex(pair::FilteredPair{T}) where {T}
    G = pair.ambient
    counts = diff(getfield(G,:dim_offsets)); offsets = getfield(G,:dim_offsets)
    inherited = all(d -> all(j -> pair.grades[d][j] == G.grades[offsets[d]+pair.indices[d][j]-1][1],
                             eachindex(pair.indices[d])), eachindex(counts))
    if inherited
        keep = [setdiff(collect(1:counts[d]),pair.indices[d]) for d in eachindex(counts)]
        boundaries = [G.boundaries[d][keep[d],keep[d+1]] for d in eachindex(G.boundaries)]
        grades = NTuple{1,T}[G.grades[offsets[d]+i-1] for d in eachindex(counts) for i in keep[d]]
        return GradedComplex([G.cells_by_dim[d][keep[d]] for d in eachindex(counts)],boundaries,grades), :quotient
    end
    # Cone(i)_d = K_d + A_(d-1), d(k,a) = (d_K k + i(a), -d_A a).
    # Its projection to K/A is a natural quasi-isomorphism at every scale.
    n = length(counts)
    acounts = length.(pair.indices)
    conecounts = [ (d <= n ? counts[d] : 0) + (d > 1 ? acounts[d-1] : 0) for d in 1:n+1 ]
    boundaries = SparseMatrixCSC{Int,Int}[]
    for d in 1:n
        rows, cols, vals = Int[],Int[],Int[]
        if d < n
            B = G.boundaries[d]
            for j in axes(B,2), ptr in nzrange(B,j)
                push!(rows,B.rowval[ptr]); push!(cols,j); push!(vals,B.nzval[ptr])
            end
        end
        coloffset = d < n ? counts[d+1] : 0
        for (j,i) in enumerate(pair.indices[d])
            push!(rows,i); push!(cols,coloffset+j); push!(vals,1)
        end
        if d > 1
            B = G.boundaries[d-1][pair.indices[d-1],pair.indices[d]]
            for j in axes(B,2), ptr in nzrange(B,j)
                v = mod(B.nzval[ptr],pair.field.p)
                v == 0 && continue
                push!(rows,counts[d]+B.rowval[ptr]); push!(cols,coloffset+j); push!(vals,pair.field.p-v)
            end
        end
        push!(boundaries,sparse(rows,cols,vals,conecounts[d],conecounts[d+1]))
    end
    grades = NTuple{1,T}[]
    for d in 1:n+1
        d <= n && append!(grades,G.grades[offsets[d]:offsets[d+1]-1])
        d > 1 && append!(grades,((v,) for v in pair.grades[d-1]))
    end
    return GradedComplex([collect(1:c) for c in conecounts],boundaries,grades), :mapping_cone
end

"""
    persistence_diagram(pair::FilteredPair)

Compute relative homology intervals H_d(K_t,A_t) in the pair's field/order.
Inherited A-grades use direct quotient columns. Later A-entry uses the filtered
mapping cone of inclusion, naturally quasi-isomorphic to the quotient at every
scale. Both routes preserve exact stored grades and omit zero-length bars.

The result uses the ordinary `PersistenceDiagram` interval/analysis/export
contract, with explicit pair/source provenance. It does not expose artificial
cone cells as source representatives. For actual quotient bases, projection,
inclusion and connecting maps use `relative_homology(pair; scale)` and
`relative_maps`. These scale-specific bases are not an interval matching.
"""
function persistence_diagram(pair::FilteredPair)
    check_filtered_pair(pair; throw=true)
    G, route = _relative_reduction_complex(pair)
    d = persistence_diagram(G; field=pair.field, order=pair.order)
    n = length(pair.indices)
    all(isempty,d.finite_by_dim[n+1:end]) && all(isempty,d.essential_by_dim[n+1:end]) ||
        error("relative cone has unexpected homology above the ambient dimension")
    meta = merge(d.meta,(construction=(requested=:relative_pair,effective=route,
        substitution=route === :mapping_cone ? :natural_quasi_isomorphism : :none),
        source=:filtered_pair,pair=_relative_provenance(pair),homology=:relative,
        representative_scope=:use_relative_homology_at_scale))
    return PersistenceDiagram(d.finite_by_dim[1:n],d.essential_by_dim[1:n],pair.field,pair.order,meta)
end

"""
    RelativeComplex

Snapshot of the exact sequence of chain complexes A -> K -> K/A at one scale.
Use `relative_complex(pair; scale)`, `relative_cells`, `relative_boundary`,
`dimensions` and `relative_maps`. Matrices have target rows and source columns.
"""
struct RelativeComplex{T<:Real,K,S<:Real}
    pair::FilteredPair{T}
    scale::S
    indices::NamedTuple{(:ambient,:subcomplex,:relative),Tuple{Vector{Vector{Int}},Vector{Vector{Int}},Vector{Vector{Int}}}}
    boundaries::NamedTuple{(:ambient,:subcomplex,:relative),Tuple{Vector{SparseMatrixCSC{K,Int}},Vector{SparseMatrixCSC{K,Int}},Vector{SparseMatrixCSC{K,Int}}}}
end

function _relative_part(part::Symbol)
    part in (:ambient,:subcomplex,:relative) || throw(ArgumentError("part must be :ambient, :subcomplex or :relative"))
    return part
end
function _relative_dim(c::RelativeComplex,dim::Integer; boundary=false)
    dim isa Bool && throw(ArgumentError("homological dimension must be an integer, not Bool"))
    0 <= dim < length(c.pair.indices) + boundary || throw(ArgumentError("homological dimension is outside this pair"))
    return Int(dim)
end

"""
    relative_complex(pair; scale)

Construct C(A_scale), C(K_scale) and C(K_scale)/C(A_scale), including quotient
differentials. `scale` is finite and compared to exact stored grades without
rounding. Cells active in K and subsequently entering A disappear from the
quotient. This heavier snapshot copies its source pair to prevent later input
mutation from changing its maps. Use `persistence_diagram(pair)` for barcodes.
"""
function relative_complex(pair::FilteredPair; scale::Real)
    isfinite(scale) || throw(ArgumentError("relative scale must be finite"))
    check_filtered_pair(pair; throw=true)
    pair = deepcopy(pair)
    G = pair.ambient; F = coeff_type(pair.field)
    offsets = getfield(G,:dim_offsets)
    active(g) = pair.order === :sublevel ? g <= scale : g >= scale
    ambient = [findall(g -> active(g[1]),G.grades[offsets[d]:offsets[d+1]-1]) for d in eachindex(pair.indices)]
    subcomplex = [pair.indices[d][findall(active,pair.grades[d])] for d in eachindex(pair.indices)]
    relative = [setdiff(ambient[d],subcomplex[d]) for d in eachindex(pair.indices)]
    indices = (;ambient,subcomplex,relative)
    boundaries = map(indices) do ids
        SparseMatrixCSC{F,Int}[F.(G.boundaries[d][ids[d],ids[d+1]]) for d in eachindex(G.boundaries)]
    end
    return RelativeComplex(pair,scale,indices,boundaries)
end

"""`relative_cells(c; dim, part=:relative)` returns ambient positions, IDs and part-specific grades, in matrix row/column order."""
function relative_cells(c::RelativeComplex; dim::Integer, part::Symbol=:relative)
    d = _relative_dim(c,dim)+1; _relative_part(part)
    ids = c.indices[part][d]; G = c.pair.ambient
    grades = part === :subcomplex ? c.pair.grades[d][searchsortedfirst.(Ref(c.pair.indices[d]),ids)] :
        [G.grades[getfield(G,:dim_offsets)[d]+i-1][1] for i in ids]
    return (dimension=d-1,cell_indices=Tuple(ids),cell_ids=Tuple(G.cells_by_dim[d][ids]),cell_grades=Tuple(grades))
end

"""`relative_boundary(c; dim, part=:relative)` copies d_dim, with target rows and source columns. Degree zero and the differential above the top degree are zero matrices."""
function relative_boundary(c::RelativeComplex{T,K}; dim::Integer, part::Symbol=:relative) where {T,K}
    d = _relative_dim(c,dim;boundary=true); _relative_part(part)
    ids = c.indices[part]
    d == 0 && return spzeros(K,0,length(ids[1]))
    d == length(ids) && return spzeros(K,length(ids[end]),0)
    return copy(c.boundaries[part][d])
end

"""`dimensions(c::RelativeComplex; part=:relative)` gives chain dimensions in degrees 0,1,... ."""
dimensions(c::RelativeComplex;part::Symbol=:relative) = length.(c.indices[_relative_part(part)])
describe(c::RelativeComplex) = (kind=:relative_complex,scale=c.scale,field=c.pair.field,order=c.pair.order,
    ambient=dimensions(c;part=:ambient),subcomplex=dimensions(c;part=:subcomplex),relative=dimensions(c))
Base.show(io::IO,c::RelativeComplex) = print(io,"RelativeComplex(scale=",c.scale,", chain dimensions=",dimensions(c),")")
field(c::RelativeComplex) = field(c.pair)
provenance(c::RelativeComplex) = merge(provenance(c.pair),(scale=c.scale,))

function _relative_coordinate_map(::Type{K},source_ids,target_ids) where {K}
    lookup = Dict(i=>j for (j,i) in enumerate(target_ids))
    rows,cols = Int[],Int[]
    for (j,id) in enumerate(source_ids)
        i = get(lookup,id,0)
        i == 0 && continue
        push!(rows,i); push!(cols,j)
    end
    return sparse(rows,cols,ones(K,length(rows)),length(target_ids),length(source_ids))
end

"""
    relative_maps(c::RelativeComplex; dim)

Copy the inclusion C_dim(A)->C_dim(K) and quotient projection C_dim(K)->C_dim(K,A).
These are chain maps of degree zero. Connecting maps are defined on homology:
call `relative_maps(relative_homology(c); dim)` for that sequence.
"""
function relative_maps(c::RelativeComplex{T,K};dim::Integer) where {T,K}
    d = _relative_dim(c,dim)+1
    return (inclusion=_relative_coordinate_map(K,c.indices.subcomplex[d],c.indices.ambient[d]),
        projection=_relative_coordinate_map(K,c.indices.ambient[d],c.indices.relative[d]))
end

"""A scale-specific model of H(A), H(K) and H(K,A). Use `relative_homology`, `dimensions`, `basis`, `relative_maps` and `relative_structure_map`. Bases are chosen, not canonical interval identities."""
struct RelativeHomology{C<:RelativeComplex,K}
    complex::C
    groups::NamedTuple{(:ambient,:subcomplex,:relative),Tuple{Vector{ChainComplexes.HomologyData{K}},Vector{ChainComplexes.HomologyData{K}},Vector{ChainComplexes.HomologyData{K}}}}
end

"""`relative_complex(h::RelativeHomology)` returns the retained chain snapshot. Treat its stored arrays as read-only; query helpers return copies."""
relative_complex(h::RelativeHomology) = h.complex

"""
    relative_homology(pair; scale)
    relative_homology(c::RelativeComplex)

Compute explicit homology spaces of all three complexes, with noncanonical
cycle bases. This performs exact linear algebra over the pair's prime field
and can use dense storage. Inspect dimensions before requesting maps or bases.
For intervals alone use `persistence_diagram(pair)`.
"""
relative_homology(pair::FilteredPair;scale::Real) = relative_homology(relative_complex(pair;scale))
function relative_homology(c::RelativeComplex{T,K}) where {T,K}
    groups = map(keys(c.indices)) do part
        [ChainComplexes.homology_data(relative_boundary(c;dim=d+1,part),
             relative_boundary(c;dim=d,part),d;field=c.pair.field) for d in 0:length(c.pair.indices)-1]
    end
    return RelativeHomology(c,NamedTuple{keys(c.indices)}(groups))
end

"""`dimensions(h::RelativeHomology; part=:relative)` gives homology dimensions in degrees 0,1,... ."""
dimensions(h::RelativeHomology;part::Symbol=:relative) = [dimensions(g).homology for g in h.groups[_relative_part(part)]]
"""`basis(h::RelativeHomology; dim, part=:relative)` copies cycle representatives as columns in `relative_cells(h.complex; dim, part)` order."""
function basis(h::RelativeHomology;dim::Integer,part::Symbol=:relative)
    d = _relative_dim(h.complex,dim)+1
    return copy(basis(h.groups[_relative_part(part)][d]))
end
# Advanced.basis uses the graded-space generic, as for HomologyData itself.
DerivedFunctors.basis(h::RelativeHomology;dim::Integer,part::Symbol=:relative) = basis(h;dim,part)
describe(h::RelativeHomology) = merge(describe(h.complex),(kind=:relative_homology,
    ambient=dimensions(h;part=:ambient),subcomplex=dimensions(h;part=:subcomplex),relative=dimensions(h)))
field(h::RelativeHomology) = field(h.complex)
provenance(h::RelativeHomology) = provenance(h.complex)
Base.show(io::IO,h::RelativeHomology) = print(io,"RelativeHomology(scale=",h.complex.scale,", dimensions=",dimensions(h),")")

"""
    relative_maps(h::RelativeHomology; dim)

Return inclusion H_dim(A)->H_dim(K), projection H_dim(K)->H_dim(K,A), and
connecting H_dim(K,A)->H_(dim-1)(A). Matrices use the bases returned by `basis`;
target rows and source columns. The connecting sign is delta[z]=[d_K lift(z)].
For degree zero the connecting map has zero rows (unreduced homology).
"""
function relative_maps(h::RelativeHomology{C,K};dim::Integer) where {C,K}
    c = h.complex; d = _relative_dim(c,dim)+1
    maps = relative_maps(c;dim)
    inclusion = ChainComplexes.induced_map_on_homology(h.groups.subcomplex[d],h.groups.ambient[d],maps.inclusion)
    projection = ChainComplexes.induced_map_on_homology(h.groups.ambient[d],h.groups.relative[d],maps.projection)
    connecting = if dim == 0
        zeros(K,0,dimensions(h.groups.relative[d]).homology)
    else
        lift = _relative_coordinate_map(K,c.indices.relative[d],c.indices.ambient[d])
        restrict = _relative_coordinate_map(K,c.indices.ambient[d-1],c.indices.subcomplex[d-1])
        boundary = relative_boundary(c;dim,part=:ambient) * lift * basis(h.groups.relative[d])
        ChainComplexes.homology_coordinates(h.groups.subcomplex[d-1],restrict * boundary)
    end
    return (;inclusion,projection,connecting)
end

function _same_filtered_pair(a::FilteredPair,b::FilteredPair)
    return a.field == b.field && a.order == b.order && a.indices == b.indices && a.grades == b.grades &&
        getfield(a.ambient,:cell_ids) == getfield(b.ambient,:cell_ids) &&
        getfield(a.ambient,:dim_offsets) == getfield(b.ambient,:dim_offsets) &&
        a.ambient.grades == b.ambient.grades && a.ambient.boundaries == b.ambient.boundaries
end

"""
    relative_structure_map(source, target; dim, part=:relative)

Induced filtration map between snapshots of the same filtered pair. Two
`RelativeComplex` inputs return a chain map; two `RelativeHomology` inputs
return its homology matrix in their chosen bases. Part is :relative, :ambient
or :subcomplex. Target rows, source columns. A quotient cell entering A maps
to zero. Source <= target is required for sublevels (>= for superlevels).
Pair contents, coefficient fields and orientation must agree exactly.
"""
function relative_structure_map(a::RelativeComplex,b::RelativeComplex;dim::Integer,part::Symbol=:relative)
    _relative_part(part); d = _relative_dim(a,dim)+1; _relative_dim(b,dim)
    _same_filtered_pair(a.pair,b.pair) || throw(ArgumentError("snapshots belong to different filtered pairs"))
    (a.pair.order === :sublevel ? a.scale <= b.scale : a.scale >= b.scale) ||
        throw(ArgumentError("scales run against the filtration order"))
    return _relative_coordinate_map(coeff_type(a.pair.field),a.indices[part][d],b.indices[part][d])
end
function relative_structure_map(a::RelativeHomology,b::RelativeHomology;dim::Integer,part::Symbol=:relative)
    m = relative_structure_map(a.complex,b.complex;dim,part)
    return ChainComplexes.induced_map_on_homology(a.groups[part][dim+1],b.groups[part][dim+1],m)
end
