"""
    filtered_complex(values; input=:top_cells, order=:sublevel, periodic=false,
                     construction=ConstructionOptions())

Construct the cubical cell model used by `cubical_persistence`, retaining
geometry and exact grades without computing persistence. Both input conventions
support any positive array dimension. Top-cell faces inherit the minimum
incident sublevel value (maximum for superlevels); vertex cells use the maximum
incident sublevel value (minimum for superlevels). `construction.budget` limits
construction as for `cubical_persistence`. Periodic boundary identifications
are retained. One-dimensional geometry is drawn on a line in the plane;
higher-dimensional data retains all coordinates for explicit projection.
"""
function DataIngestion.filtered_complex(values::AbstractArray{<:Real}; input=:top_cells,
        order=:sublevel, periodic=false, construction::ConstructionOptions=ConstructionOptions())
    _normalize_order(order)
    per = _periodic_tuple(periodic, Val(ndims(values)))
    G = if input === :top_cells
        _top_cell_complex(values, per, order, construction)
    elseif input === :vertices
        _validate_cubical_values(values)
        _vertex_cubical_complex(values, per, order, construction, nothing)
    else
        throw(ArgumentError("input must be :top_cells or :vertices."))
    end
    return _ordinary_source_cubical(G, size(values), per, order, input)
end
