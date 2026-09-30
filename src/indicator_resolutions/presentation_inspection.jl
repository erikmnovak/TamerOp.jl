# Selected fibers of a fringe image and the maps induced by downset projection.
# This uses the same descriptors and coordinate solver as pmodule_from_fringe,
# without constructing the remaining fibers or the whole module.

"""
    PresentationStalk

A selected stalk of the image of a finite-fringe presentation. Construct it with
[`presentation_stalk`](@ref), normally with `basis=false` for a rank-only query.

[`active_rows`](@ref) and [`active_columns`](@ref) are stable indices into the
original downset and upset lists. [`presentation_matrix`](@ref) has exactly these
rows and columns. If requested, [`image_basis`](@ref) embeds stalk coordinates
in the active downset coordinates; its columns use the same choice as
[`pmodule_from_fringe`](@ref).

The selected IDs, coefficients and basis are owned by this snapshot. Accessors
return that storage without an additional copy. The ambient poset is shared;
mutating it invalidates the snapshot's indexing contract. Use
[`presentation_summary`](@ref) for cheap inspection and
[`check_presentation_stalk`](@ref) to validate the stored algebra.
"""
struct PresentationStalk{K,P<:AbstractPoset,F<:AbstractCoeffField,
                         A<:AbstractMatrix{K},B<:Union{Nothing,Matrix{K}}}
    poset::P
    field::F
    vertex::Int
    presentation_size::Tuple{Int,Int}
    rows::Vector{Int}
    columns::Vector{Int}
    coefficients::A
    dimension::Int
    embedded_basis::B
end

"""
    PresentationMap

A structure map between two selected stalks of a fringe image. Obtain it with
`presentation_map(H; source=u, target=v)` for `u <= v` in the ambient poset.

[`source_stalk`](@ref) and [`target_stalk`](@ref) carry embedded image bases.
With these bases `B_source`, `B_target`, [`ambient_projection`](@ref) `R`, and
[`induced_map`](@ref) `C`, the defining equation is
`B_target * C = R * B_source`. It holds exactly over exact fields and under the
coefficient field's numerical solve tolerance over `RealField`.

This snapshot materializes only the selected pair. Empty maps retain their
source and target dimensions. Use [`presentation_summary`](@ref) and
[`check_presentation_map`](@ref) for inspection and validation.
"""
struct PresentationMap{K,S<:PresentationStalk{K},T<:PresentationStalk{K}}
    source::S
    target::T
    projection::SparseMatrixCSC{K,Int}
    matrix::Matrix{K}
end

"""
    active_rows(stalk::PresentationStalk)

Return the increasing original downset-row IDs active at the selected vertex.
These IDs label the rows of `presentation_matrix(stalk)`. An active row need
not contain a nonzero coefficient.
"""
active_rows(s::PresentationStalk) = s.rows

"""
    active_columns(stalk::PresentationStalk)

Return the increasing original upset-column IDs active at the selected vertex.
These IDs label the columns of `presentation_matrix(stalk)`. An active column
need not contain a nonzero coefficient.
"""
active_columns(s::PresentationStalk) = s.columns

"""
    presentation_vertex(stalk::PresentationStalk)

Return the selected vertex ID in `ambient_poset(stalk)`.
"""
presentation_vertex(s::PresentationStalk) = s.vertex

"""
    presentation_matrix(stalk::PresentationStalk)

Return the restricted coefficient matrix, with rows indexed by
`active_rows(stalk)` and columns by `active_columns(stalk)`. Its image is the
stalk; it is not an induced map between stalk bases. Sparse input retains sparse
storage, and zero-row/zero-column restrictions retain both dimensions.
"""
presentation_matrix(s::PresentationStalk) = s.coefficients

"""
    image_basis(stalk::PresentationStalk)

Return the embedded image-basis matrix, or `nothing` when the snapshot was made
with `basis=false`. Rows are active downset coordinates and columns are the
chosen stalk coordinates. This accessor never computes a missing basis; request
`presentation_stalk(H; vertex=q, basis=true)` explicitly when it is needed.
"""
image_basis(s::PresentationStalk) = s.embedded_basis

"""
    source_stalk(map::PresentationMap)

Return the source snapshot, including the embedded basis used by
`induced_map(map)`. Its stalk coordinates index the columns of that matrix.
"""
source_stalk(m::PresentationMap) = m.source

"""
    target_stalk(map::PresentationMap)

Return the target snapshot, including the embedded basis used by
`induced_map(map)`. Its stalk coordinates index the rows of that matrix.
"""
target_stalk(m::PresentationMap) = m.target

"""
    ambient_projection(map::PresentationMap)

Return the sparse coordinate projection from active source downsets to active
target downsets. Its shape is `length(active_rows(target_stalk(map)))` by
`length(active_rows(source_stalk(map)))`. This ambient projection differs from
the map between image-basis coordinates returned by `induced_map(map)`.
"""
ambient_projection(m::PresentationMap) = m.projection

"""
    induced_map(map::PresentationMap)

Return the structure-map matrix in the bases stored in the source and target
snapshots. It has target dimension rows and source dimension columns.
"""
induced_map(m::PresentationMap) = m.matrix

FiniteFringe.field(s::PresentationStalk) = s.field
FiniteFringe.field(m::PresentationMap) = FiniteFringe.field(source_stalk(m))
FiniteFringe.base_poset(s::PresentationStalk) = s.poset
FiniteFringe.base_poset(m::PresentationMap) = FiniteFringe.base_poset(source_stalk(m))
FiniteFringe.ambient_poset(s::PresentationStalk) = s.poset
FiniteFringe.ambient_poset(m::PresentationMap) = FiniteFringe.ambient_poset(source_stalk(m))

function _presentation_vertex(P::AbstractPoset, vertex, name::Symbol)
    vertex isa Integer && !(vertex isa Bool) ||
        throw(ArgumentError("$name must be an integer vertex ID, not $(repr(vertex))."))
    1 <= vertex <= nvertices(P) ||
        throw(ArgumentError("$name must lie in 1:$(nvertices(P)); got $vertex."))
    return Int(vertex)
end

function _presentation_active_data(H::FiniteFringe.FringeModule, q::Int)
    rows = Int[i for i in eachindex(H.D) if H.D[i].mask[q]]
    cols = Int[i for i in eachindex(H.U) if H.U[i].mask[q]]
    return rows, cols, H.phi[rows, cols]
end

function _presentation_stalk_with_descriptor(H::FiniteFringe.FringeModule{K}, q::Int) where {K}
    rows, cols, coefficients = _presentation_active_data(H, q)
    # Preserve the descriptor input used by pmodule_from_fringe, including its
    # empty-column convention and field-aware backend selection on the view.
    phi_q = isempty(cols) || isempty(rows) ? zeros(K, length(rows), 0) :
            view(H.phi, rows, cols)
    desc = _build_fringe_fiber_descriptor(H.field, rows, phi_q, K)
    stalk = PresentationStalk(H.P, H.field, q, size(H.phi), rows, cols,
                              coefficients, size(desc.basis, 2), desc.basis)
    return stalk, desc
end

"""
    presentation_stalk(H::FiniteFringe.FringeModule; vertex, basis=false)

Inspect `im(phi_vertex)` at one vertex of the presentation's ambient poset.
The returned [`PresentationStalk`](@ref) contains the active original row and
column IDs, their coefficient submatrix, the field and the image dimension.

The default computes rank but no image basis. Set `basis=true` to obtain the
embedded image basis used by `pmodule_from_fringe(H)`. Only the selected fiber
is inspected; this does not construct all stalks or structure maps. The field
object, including `RealField` tolerances, is retained unchanged.

`vertex` must be an integer ID in `1:nvertices(ambient_poset(H))`; Boolean,
noninteger and out-of-range values raise `ArgumentError`. Empty active sets
produce the correctly shaped coefficient matrix and a zero-dimensional image.

```julia
s = presentation_stalk(H; vertex=2)
presentation_summary(s)             # dimension and active matrix size
presentation_matrix(s)              # rows/columns identified by active_* IDs
sb = presentation_stalk(H; vertex=2, basis=true)
image_basis(sb)                     # image embedded in active downsets
```
"""
function presentation_stalk(H::FiniteFringe.FringeModule; vertex, basis::Bool=false)
    q = _presentation_vertex(H.P, vertex, :vertex)
    if basis
        stalk, _ = _presentation_stalk_with_descriptor(H, q)
        return stalk
    end
    rows, cols, coefficients = _presentation_active_data(H, q)
    d = isempty(rows) || isempty(cols) ? 0 :
        FieldLinAlg.rank(H.field, view(H.phi, rows, cols))
    return PresentationStalk(H.P, H.field, q, size(H.phi), rows, cols,
                              coefficients, d, nothing)
end

# Results is loaded before this owner. Its fallback does not inspect arbitrary
# retained objects; this concrete method recognizes a matching fringe witness.
function Results._encoding_presentation(H::FiniteFringe.FringeModule, P, field)
    return FiniteFringe.ambient_poset(H) === P && FiniteFringe.field(H) == field ? H : nothing
end

function _retained_presentation(enc::Results.EncodingResult)
    H = Results.encoding_presentation(enc)
    H === nothing && throw(ArgumentError(
        "presentation inspection requires a retained finite-fringe presentation on the encoding's poset and coefficient field."))
    return H
end

"""
    presentation_stalk(enc::Results.EncodingResult; vertex, basis=false)

Inspect the retained finite-fringe presentation returned by
`encoding_presentation(enc)`. The vertex is an ID in the encoding poset, not an
ambient parameter point. Missing or incompatible presentation data raise
`ArgumentError`; this query does not materialize the encoding's module.

This inspects the retained presentation witness. Matching its poset and field
does not certify agreement with an independently hand-built `enc.M`.
"""
function presentation_stalk(enc::Results.EncodingResult; vertex, basis::Bool=false)
    return presentation_stalk(_retained_presentation(enc); vertex=vertex, basis=basis)
end

function _presentation_projection(::Type{K}, rows_src::Vector{Int}, rows_tgt::Vector{Int}) where {K}
    slots = _row_projection_slots(rows_src, rows_tgt)
    return sparse(collect(eachindex(slots)), slots, fill(one(K), length(slots)),
                  length(rows_tgt), length(rows_src))
end

_presentation_basis_equation_holds(::AbstractCoeffField, B, C, Y) = B * C == Y

function _presentation_basis_equation_holds(field::RealField, B, C, Y;
                                          residual_atol=field.atol)
    all(isfinite, B) && all(isfinite, C) && all(isfinite, Y) || return false
    product = B * C
    all(isfinite, product) || return false
    # Match solve_fullcolumn's per-RHS contract and evaluation order. In
    # particular, rtol=0 must not multiply an overflowing unused norm by zero.
    relative = !iszero(field.rtol)
    scaled_norm_B = relative ? field.rtol * norm(B) : zero(field.rtol)
    for j in axes(Y, 2)
        residual = norm(view(product, :, j) - view(Y, :, j))
        bound = relative ? residual_atol + scaled_norm_B * norm(view(C, :, j)) +
                           field.rtol * norm(view(Y, :, j)) : residual_atol
        isfinite(bound) && residual <= bound || return false
    end
    return true
end

"""
    presentation_map(H::FiniteFringe.FringeModule)
    presentation_map(H::FiniteFringe.FringeModule; source, target)

With no keywords, return the full fringe coefficient matrix. With both vertex
IDs, return a [`PresentationMap`](@ref) for the comparable pair `source <= target`.
The latter explicitly computes two image bases, the downset-coordinate
projection and the induced structure map; it does not materialize a `PModule`.

Supply both IDs or neither. Invalid IDs, reverse/incomparable pairs and a lone
keyword raise `ArgumentError`. An equal pair is allowed and gives the identity
in the chosen image basis, including the `0 x 0` identity of a zero stalk.

The result satisfies `B_target * C = R * B_source` over the exact coefficient
field, or the field's per-column numerical solve tolerance for `RealField`.
For example:

```julia
m = presentation_map(H; source=1, target=3)
presentation_summary(m)
induced_map(m)
check_presentation_map(m; throw=true)
```
"""
function presentation_map(H::FiniteFringe.FringeModule{K}; source=nothing, target=nothing) where {K}
    source === nothing && target === nothing && return FiniteFringe.fringe_coefficients(H)
    (source === nothing || target === nothing) &&
        throw(ArgumentError("presentation_map: supply both source and target vertex IDs."))
    u = _presentation_vertex(H.P, source, :source)
    v = _presentation_vertex(H.P, target, :target)
    leq(H.P, u, v) ||
        throw(ArgumentError("presentation_map: source $u must be <= target $v in the ambient poset."))
    s, ds = _presentation_stalk_with_descriptor(H, u)
    t, dt = u == v ? (s, ds) : _presentation_stalk_with_descriptor(H, v)
    R = _presentation_projection(K, ds.rows, dt.rows)
    C = if u == v
        eye(H.field, s.dimension)
    elseif s.dimension == 0 || t.dimension == 0
        zeros(K, t.dimension, s.dimension)
    else
        slots = _row_projection_slots(ds.rows, dt.rhs_rows)
        _indicator_cover_map_from_basis(H.field, ds.basis, dt.basis, slots, dt.factor)
    end
    _presentation_basis_equation_holds(H.field, dt.basis, C, R * ds.basis) ||
        throw(ArgumentError("presentation_map: projected source image is not contained in the target image at the field tolerance."))
    return PresentationMap(s, t, R, C)
end

"""
    presentation_map(enc::Results.EncodingResult; source, target)
    presentation_map(enc::Results.EncodingResult)

Inspect the map induced by the encoding's retained fringe presentation, or its
full coefficient matrix when no vertex pair is supplied. The same index/order
contract as `presentation_map(H; source, target)` applies. Missing or
incompatible retained data raise `ArgumentError`.

The result concerns that retained witness; it does not certify its agreement
with arbitrary independently constructed module data in the encoding.
"""
function presentation_map(enc::Results.EncodingResult; source=nothing, target=nothing)
    return presentation_map(_retained_presentation(enc); source=source, target=target)
end

"""
    presentation_summary(stalk::PresentationStalk)
    presentation_summary(map::PresentationMap)

Return a cheap summary of a selected fringe stalk or map. A stalk summary
includes its image `dimension`, active row/column counts and `basis_available`.
A map summary reports the selected pair, source/target dimensions and the shapes
of both ambient projection and induced matrix. Neither method computes a basis,
a new rank, or any other stalk.
"""
function presentation_summary(s::PresentationStalk)
    return (kind=:presentation_stalk, vertex=s.vertex, field=s.field,
            dimension=s.dimension, active_rows=length(s.rows),
            active_columns=length(s.columns), matrix_size=size(s.coefficients),
            basis_available=s.embedded_basis !== nothing)
end

function presentation_summary(m::PresentationMap)
    return (kind=:presentation_map, source=m.source.vertex, target=m.target.vertex,
            field=FiniteFringe.field(m), source_dimension=m.source.dimension,
            target_dimension=m.target.dimension, matrix_size=size(m.matrix),
            projection_size=size(m.projection))
end

function Base.show(io::IO, s::PresentationStalk)
    print(io, "PresentationStalk(vertex=", s.vertex, ", dimension=", s.dimension,
          ", matrix_size=", size(s.coefficients), ", basis=", s.embedded_basis !== nothing, ")")
end

function Base.show(io::IO, ::MIME"text/plain", s::PresentationStalk)
    show(io, s)
    print(io, "\n  field: ", s.field, "\n  active downset rows: ", s.rows,
          "\n  active upset columns: ", s.columns)
end

function Base.show(io::IO, m::PresentationMap)
    print(io, "PresentationMap(", m.source.vertex, " <= ", m.target.vertex,
          ", matrix_size=", size(m.matrix), ")")
end

function Base.show(io::IO, ::MIME"text/plain", m::PresentationMap)
    show(io, m)
    print(io, "\n  field: ", FiniteFringe.field(m),
          "\n  ambient projection: ", size(m.projection),
          "\n  source dimension: ", m.source.dimension,
          "\n  target dimension: ", m.target.dimension)
end

function _presentation_ids_valid(ids::Vector{Int}, bound::Int)
    return all(i -> 1 <= i <= bound, ids) &&
           all(i -> ids[i-1] < ids[i], 2:length(ids))
end

"""
    check_presentation_stalk(stalk::PresentationStalk; throw=false)

Validate stored indexing, coefficient shape, field, rank and (when present) the
embedded basis. The report has `valid` and `issues`; use
`indicator_resolution_validation_summary(report)` for compact display.
`throw=true` raises `ArgumentError` for an invalid snapshot.

This checks the snapshot's algebra, not equality to a subsequently modified
original presentation. It may recompute rank and solve for basis coordinates;
`presentation_summary` is the cheap inspection path.

For `RealField`, the image basis may discard directions below the restricted
matrix's rank tolerance `atol + rtol * opnorm(A, 1)`. Validation allows that
per-column truncation residual, in addition to numerical solve error. Induced
structure maps retain the stricter solve tolerance in `check_presentation_map`.
"""
function check_presentation_stalk(s::PresentationStalk; throw::Bool=false)
    issues = String[]
    1 <= s.vertex <= nvertices(s.poset) || push!(issues, "vertex is outside the ambient poset")
    nr, nc = s.presentation_size
    nr >= 0 && nc >= 0 || push!(issues, "presentation size must be nonnegative")
    _presentation_ids_valid(s.rows, nr) || push!(issues, "active row IDs must be increasing, unique and in range")
    _presentation_ids_valid(s.columns, nc) || push!(issues, "active column IDs must be increasing, unique and in range")
    size(s.coefficients) == (length(s.rows), length(s.columns)) ||
        push!(issues, "coefficient shape does not match the active IDs")
    coeff_type(s.field) == eltype(s.coefficients) || push!(issues, "coefficient type does not match the field")
    if s.field isa RealField
        isfinite(s.field.atol) && isfinite(s.field.rtol) && s.field.atol >= 0 && s.field.rtol >= 0 ||
            push!(issues, "RealField tolerances must be finite and nonnegative")
    end
    0 <= s.dimension <= min(length(s.rows), length(s.columns)) ||
        push!(issues, "image dimension is outside the possible rank range")
    B = s.embedded_basis
    B === nothing || size(B) == (length(s.rows), s.dimension) ||
        push!(issues, "embedded basis has the wrong shape")
    if isempty(issues)
        try
            d = isempty(s.rows) || isempty(s.columns) ? 0 : FieldLinAlg.rank(s.field, s.coefficients)
            d == s.dimension || push!(issues, "stored dimension differs from the restricted matrix rank")
            if B !== nothing
                size(B, 2) == 0 || FieldLinAlg.rank(s.field, B) == s.dimension ||
                    push!(issues, "embedded basis columns are not independent")
                # Literal equality is already a span certificate. Solving an
                # identity coordinate problem can introduce roundoff that
                # violates a deliberately strict absolute-only tolerance.
                if isempty(issues) && B != s.coefficients
                    coordinates = s.dimension == 0 ? zeros(eltype(B), 0, size(s.coefficients, 2)) :
                                  FieldLinAlg.solve_fullcolumn(s.field, B, s.coefficients;
                                                              check_rhs=!(s.field isa RealField))
                    spans_image = if s.field isa RealField
                        # QR image selection uses the whole input's rank scale.
                        # A discarded small column need not pass the stricter
                        # per-RHS consistency contract for structure maps.
                        image_atol = isempty(s.coefficients) || iszero(s.field.rtol) ? s.field.atol :
                                     s.field.atol + s.field.rtol * opnorm(s.coefficients, 1)
                        _presentation_basis_equation_holds(s.field, B, coordinates, s.coefficients;
                                                           residual_atol=image_atol)
                    else
                        _presentation_basis_equation_holds(s.field, B, coordinates, s.coefficients)
                    end
                    spans_image ||
                        push!(issues, "embedded basis does not span the restricted image")
                end
            end
        catch err
            err isa InterruptException && rethrow()
            push!(issues, "stalk algebra validation failed: " * sprint(showerror, err))
        end
    end
    valid = isempty(issues)
    throw && !valid && _throw_invalid_indicator_resolution(:presentation_stalk, issues)
    return (kind=:presentation_stalk, valid=valid, issues=issues,
            vertex=s.vertex, dimension=s.dimension, basis_available=B !== nothing)
end

"""
    check_presentation_map(map::PresentationMap; throw=false)

Validate both stalk snapshots, their common field/poset and order, the labelled
downset projection, and `B_target * C = R * B_source`. Active upset columns must
persist from source to target, and overlapping original coefficients must agree
exactly, including over `RealField`: they are restrictions of the same stored
matrix. Equal vertex IDs must have identical active row and column lists.
These checks establish consistency of the selected data, not equality to an
unretained original presentation. Numerical basis equations use the retained
`RealField` per-column solve tolerance, not exact equality.
Return a structured report, or raise `ArgumentError` when `throw=true` and the
snapshot is invalid. This is an explicit algebra check, not a cheap summary.
"""
function check_presentation_map(m::PresentationMap; throw::Bool=false)
    issues = String[]
    s, t = m.source, m.target
    for (name, stalk) in (("source", s), ("target", t))
        report = check_presentation_stalk(stalk)
        append!(issues, ["$name: $issue" for issue in report.issues])
    end
    s.poset === t.poset || push!(issues, "source and target must share the ambient poset")
    isequal(s.field, t.field) || push!(issues, "source and target coefficient fields differ")
    s.presentation_size == t.presentation_size || push!(issues, "source and target presentation sizes differ")
    s.embedded_basis !== nothing && t.embedded_basis !== nothing ||
        push!(issues, "map inspection requires both embedded image bases")
    size(m.projection) == (length(t.rows), length(s.rows)) || push!(issues, "ambient projection has the wrong shape")
    size(m.matrix) == (t.dimension, s.dimension) || push!(issues, "induced map has the wrong shape")
    if isempty(issues)
        try
            leq(s.poset, s.vertex, t.vertex) || push!(issues, "source is not <= target in the ambient poset")
            R = _presentation_projection(eltype(m.matrix), s.rows, t.rows)
            m.projection == R || push!(issues, "ambient projection does not preserve the active downset IDs")
            if s.vertex == t.vertex
                s.rows == t.rows && s.columns == t.columns ||
                    push!(issues, "equal vertices must have identical active row and column IDs")
            end
            target_column_slots = [searchsortedfirst(t.columns, j) for j in s.columns]
            columns_persist = all(eachindex(s.columns)) do j
                slot = target_column_slots[j]
                slot <= length(t.columns) && t.columns[slot] == s.columns[j]
            end
            if columns_persist
                source_row_slots = _row_projection_slots(s.rows, t.rows)
                view(s.coefficients, source_row_slots, :) ==
                    view(t.coefficients, :, target_column_slots) ||
                    push!(issues, "overlapping presentation coefficients disagree")
            else
                push!(issues, "active source upset columns must remain active at the target")
            end
            _presentation_basis_equation_holds(s.field, t.embedded_basis, m.matrix,
                                                R * s.embedded_basis) ||
                push!(issues, "induced map does not satisfy the embedded image equation")
        catch err
            err isa InterruptException && rethrow()
            push!(issues, "map algebra validation failed: " * sprint(showerror, err))
        end
    end
    valid = isempty(issues)
    throw && !valid && _throw_invalid_indicator_resolution(:presentation_map, issues)
    return (kind=:presentation_map, valid=valid, issues=issues,
            source=s.vertex, target=t.vertex, matrix_size=size(m.matrix))
end
