# Images of selected stalk coordinates and their generated subobject.
import ..Results

function _propagation_equal(field, A, B)
    size(A)==size(B) || return false
    field isa RealField || return A==B
    all(isfinite,A) && all(isfinite,B) || return false
    scale=max(maximum(abs,A;init=0.0),maximum(abs,B;init=0.0))
    return maximum(abs,A-B;init=0.0)<=field.atol+field.rtol*scale
end

"""
    VectorPropagation

Images of supplied vector columns at one stalk of a finite-poset module.
Construct with `propagate(M; source, vectors)`. Columns remain in the ambient
module's stored bases, including dependent and zero columns. `propagation_image`
returns these columns; `propagation_summary` reports ranks and nonzero support.
A `nothing` image means there is no forward map, not that its image is zero.
Request `generated_submodule(r)` explicitly for the subobject and inclusion.
The retained ambient module and encoding must remain read-only.
"""
struct VectorPropagation{K,M,C}
    ambient::M
    source::Int
    vectors::Matrix{K}
    images::Vector{Union{Nothing,Matrix{K}}}
    image_dimensions::Vector{Union{Nothing,Int}}
    context::C
end

function _propagation_vertex(M, q)
    q isa Integer && !(q isa Bool) && 1 <= q <= nvertices(M.Q) ||
        throw(ArgumentError("Expected a finite-poset vertex in 1:$(nvertices(M.Q))."))
    return Int(q)
end

"""
    propagate(M::PModule; source, vectors) -> VectorPropagation

Follow a vector or the columns of a matrix from the declared source stalk.
Coordinates are converted to the module's coefficient field. The number of rows
must be `dim_at(M, source)`; dependent/zero columns and an empty subspace are
allowed. Storage is checked, then columns are propagated through cover maps;
every alternate route is checked on those columns. The ambient object must be
a valid module; unselected vectors are not certified by this computation.
No generated module, quotient, or direct-summand certificate is computed.
Exact fields give exact images/ranks; RealField uses its rank tolerances.
For an encoding use the root workflow, which retains the classifier as well.
"""
function propagate(M::PModule{K}; source, vectors) where {K}
    p = _propagation_vertex(M, source)
    vectors isa Union{AbstractVector,AbstractMatrix} ||
        throw(ArgumentError("vectors must be a vector or a matrix of column coordinates."))
    X = vectors isa AbstractVector ? reshape(collect(vectors), :, 1) : vectors
    size(X, 1) == M.dims[p] || throw(DimensionMismatch("Selected coordinates must have $(M.dims[p]) rows."))
    B = Matrix{K}(X)
    M.field isa RealField && !all(isfinite, B) && throw(ArgumentError("Selected coordinates must be finite."))
    Modules.check_module(M; throw=true)
    images = Union{Nothing,Matrix{K}}[nothing for _ in M.dims]
    images[p] = copy(B)
    queue = [p]; head = 1
    # Retain one spanning path, but check every alternative on these columns.
    # Propagate only the selected columns, never a dense all-pairs map table.
    while head <= length(queue)
        u = queue[head]; head += 1
        for (j, v) in enumerate(M.edge_maps.succs[u])
            Y = Matrix{K}(M.edge_maps.maps_to_succ[u][j] * images[u])
            M.field isa RealField && !all(isfinite,Y) && throw(ArgumentError("Propagation produced nonfinite coordinates."))
            if images[v] !== nothing
                _propagation_equal(M.field,Y,images[v]) ||
                    throw(ArgumentError("Selected columns are path-dependent at cover ($u,$v)."))
                continue
            end
            images[v] = Y
            push!(queue, v)
        end
    end
    dims = Union{Nothing,Int}[A === nothing ? nothing : FieldLinAlg.rank(M.field, A) for A in images]
    return VectorPropagation(M, p, B, images, dims,
        (; encoding=nothing, basis_convention=:ambient_stalk_coordinates))
end

"""
    propagation_image(r; target)

Copy the propagated columns at `target`, or return `nothing` if no forward map
exists from the selected source. A defined zero image retains its full matrix
shape. This query performs no further algebra.
"""
function propagation_image(r::VectorPropagation; target)
    q = _propagation_vertex(r.ambient, target)
    A = r.images[q]
    return A === nothing ? nothing : copy(A)
end

"""
    propagation_summary(r)

Return the source vertex, coefficient field, coordinate convention, copied
image dimensions (with `nothing` outside the upper ideal), nonzero support and
retained encoding. Ranks refer to the chosen subspace, not the whole map.
"""
function propagation_summary(r::VectorPropagation)
    (; kind=:vector_propagation, source=r.source, field=r.ambient.field,
       basis_convention=r.context.basis_convention, columns=size(r.vectors,2),
       image_dimensions=copy(r.image_dimensions),
       nonzero_support=findall(d -> d !== nothing && d > 0, r.image_dimensions),
       applicable_vertices=findall(!isnothing, r.images),
       encoding=r.context.encoding, exact=!(r.ambient.field isa RealField),
       path_independence=:selected_columns_checked, summand=:not_certified)
end
ambient_module(r::VectorPropagation) = r.ambient

"""
    provenance(r::VectorPropagation)

Return the finite base, field, source vertex, copied source coordinates and
their basis convention, together with retained encoding provenance when
available. Path checks concern the selected columns; no homology-basis or
direct-summand identification is inferred.
"""
Results.provenance(r::VectorPropagation) = (;category=:finite_poset_representations,
    base_poset=r.ambient.Q,field=r.ambient.field,source_vertex=r.source,
    source_coordinates=copy(r.vectors),basis_convention=r.context.basis_convention,
    encoding=r.context.encoding===nothing ? nothing : Results.provenance(r.context.encoding),
    path_independence=:selected_columns_checked,summand=:not_certified)
Base.show(io::IO, r::VectorPropagation) = print(io, "VectorPropagation(source=", r.source,
    ", columns=", size(r.vectors,2), ", nonzero_stalks=", count(d -> d !== nothing && d > 0, r.image_dimensions), ")")

"""
    check_propagation(r; throw=false)

Validate retained shapes, ranks, the source columns, and every applicable cover
equation against the retained ambient module. Exact for exact fields; numerical
checks use RealField tolerances. Returns an abelian `ValidationSummary` report.
"""
function check_propagation(r::VectorPropagation; throw::Bool=false)
    issues = String[]
    try
        M = r.ambient; p = _propagation_vertex(M, r.source)
        Modules.check_module(M; throw=true)
        r.context.basis_convention===:ambient_stalk_coordinates || error("Unknown coordinate convention.")
        enc=r.context.encoding
        enc===nothing || (enc isa Results.EncodingResult && Results.encoding_module(enc)===M) ||
            error("Retained encoding must contain the identical ambient module.")
        n = nvertices(M.Q); k = size(r.vectors,2)
        length(r.images) == length(r.image_dimensions) == n || error("Retained vertex counts disagree.")
        size(r.vectors,1) == M.dims[p] || error("Source coordinate shape disagrees.")
        for q in 1:n
            A = r.images[q]; d = r.image_dimensions[q]
            applicable = FiniteFringe.leq(M.Q,p,q)
            if !applicable
                A === nothing && d === nothing || error("Vertex $q has no forward map.")
                continue
            end
            A !== nothing && size(A) == (M.dims[q],k) || error("Invalid image shape at $q.")
            M.field isa RealField && !all(isfinite,A) && error("Nonfinite image at $q.")
            d == FieldLinAlg.rank(M.field,A) || error("Incorrect image dimension at $q.")
        end
        _propagation_equal(M.field, r.images[p], r.vectors) || error("Source columns disagree.")
        for (u,v) in cover_edges(M.Q)
            r.images[u] === nothing && continue
            lhs = Modules.structure_map(M;source=u,target=v)*r.images[u]
            rhs = r.images[v]
            _propagation_equal(M.field,lhs,rhs) || error("Propagation fails on ($u,$v).")
        end
    catch err
        err isa InterruptException && rethrow()
        push!(issues,sprint(showerror,err))
    end
    report = (;kind=:vector_propagation,valid=isempty(issues),issues)
    throw && !report.valid && _abelian_validation_error("check_propagation",issues)
    return validation_summary(report)
end

"""
    generated_submodule(r::VectorPropagation; cache=:auto) -> Submodule

Construct the smallest finite-poset submodule containing the chosen source
columns. Outside the source upper ideal its stalk is zero. Access its module
with `submodule_object`, its inclusion with `inclusion_map`, and request
`quotient_with_projection(S)` separately. No splitting is asserted.

The construction takes the image of the natural map from copies of the
principal projective at the selected vertex; it reuses the checked image and
quotient engines. It allocates bases and cover maps only on this explicit call.
"""
function generated_submodule(r::VectorPropagation{K}; cache::Union{Symbol,CoverCache}=:auto) where {K}
    check_propagation(r;throw=true)
    M = r.ambient; k = size(r.vectors,2)
    dims = [A === nothing ? 0 : k for A in r.images]
    maps = Dict{Tuple{Int,Int},Matrix{K}}((u,v) => (dims[u] == 0 ? zeros(K,dims[v],0) : eye(M.field,k)) for (u,v) in cover_edges(M.Q))
    F = PModule{K}(M.Q,dims,maps;field=M.field)
    f = PMorphism(F,M,[A === nothing ? zeros(K,M.dims[q],0) : copy(A) for (q,A) in enumerate(r.images)])
    return image_submodule(f;cache)
end
