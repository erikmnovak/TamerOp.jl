# Supplied algebraic maps and witnesses. Construction/validation happens while
# making the specification; a renderer only receives copied coefficients.

available_visuals(::DerivedFunctors.HomSpace) = (:hom_basis,)
available_visuals(::ModuleComplexes.ModuleCochainMap) = (:chain_map,)
available_visuals(::ModuleComplexes.ModuleCochainHomotopy) = (:homotopy_comparison,)


_visual_request_keywords(::DerivedFunctors.HomSpace, ::Symbol) =
    (:basis_index, :vertex, :pair, :matrix_limit)
_visual_request_keywords(::ModuleComplexes.ModuleCochainMap, ::Symbol) =
    (:degree, :vertex, :matrix_limit, :induced)
_visual_request_keywords(::ModuleComplexes.ModuleCochainHomotopy, ::Symbol) =
    (:degree, :vertex, :matrix_limit, :induced, :induced_maps)
_visual_request_keywords(::DerivedFunctors.ProjectiveResolution, kind::Symbol) =
    kind !== :resolution_lift ? _resolution_request_keywords(kind) :
    (:target_resolution, :morphism, :lift, :comparison_lift, :homotopy,
     :degree, :vertex, :matrix_limit)

_visual_request_cost(::DerivedFunctors.HomSpace, ::Symbol) =
    (; work=:selected_hom_basis_morphism, basis=:explicit_selection_materializes_basis,
       algebra=:already_computed_hom_space, timing=:not_measured)
_visual_request_cost(::ModuleComplexes.ModuleCochainMap, ::Symbol) =
    (; work=:supplied_cochain_map_validation, selected_matrices=:degree_and_vertex,
       induced=:explicit_opt_in_computes_cohomology_modules, timing=:not_measured)
_visual_request_cost(::ModuleComplexes.ModuleCochainHomotopy, ::Symbol) =
    (; work=:supplied_cochain_homotopy_validation,
       induced=:explicit_opt_in_computes_and_compares_cohomology_maps,
       timing=:not_measured)
_visual_request_cost(::DerivedFunctors.ProjectiveResolution, kind::Symbol) =
    kind !== :resolution_lift ? _resolution_request_cost(kind) :
    (; work=:supplied_augmented_lift_and_homotopy_validation,
       algebra=:no_resolution_or_lift_construction,
       selected_matrices=:degree_and_vertex, timing=:not_measured)

function _comparison_matrix_limit!(issues, limit)
    ((limit isa Tuple || limit isa AbstractVector) && length(limit) == 2 &&
        all(x -> x isa Integer && !(x isa Bool) && x > 0, limit)) ||
        push!(issues, "matrix_limit must contain two positive integers.")
end

function _comparison_vertex!(issues, P, q)
    q isa Integer && !(q isa Bool) && 1 <= q <= nvertices(P) ||
        push!(issues, "vertex must be a finite-poset ID in 1:$(nvertices(P)).")
end

function _comparison_degree!(issues, t, degrees)
    t isa Integer && !(t isa Bool) && t in degrees ||
        push!(issues, "degree must be an integer in $(degrees).")
end

function _append_visual_request_issues!(issues::Vector{String}, H::DerivedFunctors.HomSpace, kind::Symbol; kwargs...)
    i = get(kwargs, :basis_index, nothing)
    n = DerivedFunctors.dim(H)
    (i === nothing || (i isa Integer && !(i isa Bool) && 1 <= i <= n)) ||
        push!(issues, n == 0 ? "The zero Hom space has no basis element to select." :
              "basis_index must be an integer in 1:$n.")
    _check_module_selection!(issues, DerivedFunctors.source_module(H), :module_inspector;
        vertex=get(kwargs, :vertex, nothing), pair=get(kwargs, :pair, nothing),
        matrix_limit=get(kwargs, :matrix_limit, (12, 12)))
    return issues
end

"""
    visual_spec(H::HomSpace; kind=:hom_basis, basis_index=nothing,
                vertex=nothing, pair=nothing, matrix_limit=(12,12))

Inspect an actual selected basis morphism in the already computed ordinary
`Hom` space. Selecting a basis element explicitly materializes the stored Hom
basis. The basis depends on the algebra's coordinate choices. This is neither
a support-overlap heuristic nor a quotient of Hom by maps vanishing at a point.
With no explicit index, the first basis element is selected when one exists;
the zero Hom space instead displays an informative no-basis view.
"""
function _visual_spec(H::DerivedFunctors.HomSpace, kind::Symbol;
                      basis_index=nothing, vertex=nothing, pair=nothing, matrix_limit=(12, 12))
    kind === :hom_basis || throw(ArgumentError("Unsupported Hom visualization $kind."))
    before = DerivedFunctors.hom_summary(H).basis_cached
    if DerivedFunctors.dim(H) == 0
        return VisualizationSpec(:hom_basis; title="The zero Hom space",
            subtitle="There is no basis element to select; the zero map is its only member.",
            panels=[_algebra_module_panel(DerivedFunctors.source_module(H), "Source module"),
                    _algebra_module_panel(DerivedFunctors.target_module(H), "Target module")],
            metadata=(; basis_index=nothing, hom_dimension=0, basis_materialized=false,
                basis_cached_before=before, provenance=DerivedFunctors.provenance(H),
                panel_columns=2, figure_size=(1100, 500)))
    end
    basis_index = basis_index === nothing ? 1 : Int(basis_index)
    f = DerivedFunctors.basis(H)[basis_index]
    base = _morphism_visual_spec(f, :morphism_inspector; vertex, pair, matrix_limit)
    return VisualizationSpec(:hom_basis; title="Selected Hom basis map",
        subtitle="Basis element $basis_index of $(DerivedFunctors.dim(H)); ordinary Hom in the finite-poset category",
        panels=base.panels, layers=base.layers, axes=base.axes, legend=base.legend,
        interaction=base.interaction,
        metadata=merge(base.metadata, (; basis_index=Int(basis_index),
            hom_dimension=DerivedFunctors.dim(H), basis_convention=:computed_hom_basis,
            basis_cached_before=before, basis_materialized=!before,
            provenance=DerivedFunctors.provenance(H))))
end

_cochain_plot_poset(f::ModuleComplexes.ModuleCochainMap) =
    ModuleComplexes.provenance(ModuleComplexes.source(f)).base_poset
_cochain_plot_field(f::ModuleComplexes.ModuleCochainMap) =
    ModuleComplexes.provenance(ModuleComplexes.source(f)).field
_cochain_plot_degrees(f::ModuleComplexes.ModuleCochainMap) =
    ModuleComplexes.degree_range(f)
function _cochain_plot_degrees(H::ModuleComplexes.ModuleCochainHomotopy)
    f, g = ModuleComplexes.source_map(H), ModuleComplexes.target_map(H)
    rf, rg, rh = _cochain_plot_degrees(f), _cochain_plot_degrees(g), ModuleComplexes.degree_range(H)
    return (min(first(rf), first(rg), first(rh)) - 1):max(last(rf), last(rg), last(rh))
end

function _append_visual_request_issues!(issues::Vector{String}, f::ModuleComplexes.ModuleCochainMap, kind::Symbol; kwargs...)
    degrees = _cochain_plot_degrees(f)
    _comparison_degree!(issues, get(kwargs, :degree, first(degrees)), degrees)
    _comparison_vertex!(issues, _cochain_plot_poset(f), get(kwargs, :vertex, 1))
    _comparison_matrix_limit!(issues, get(kwargs, :matrix_limit, (12, 12)))
    get(kwargs, :induced, false) isa Bool || push!(issues, "induced must be true or false.")
    return issues
end

function _append_visual_request_issues!(issues::Vector{String}, H::ModuleComplexes.ModuleCochainHomotopy, kind::Symbol; kwargs...)
    f = ModuleComplexes.source_map(H)
    degrees = _cochain_plot_degrees(H)
    _comparison_degree!(issues, get(kwargs, :degree, first(ModuleComplexes.degree_range(H))), degrees)
    _comparison_vertex!(issues, _cochain_plot_poset(f), get(kwargs, :vertex, 1))
    _comparison_matrix_limit!(issues, get(kwargs, :matrix_limit, (12, 12)))
    get(kwargs, :induced, false) isa Bool || push!(issues, "induced must be true or false.")
    supplied = get(kwargs, :induced_maps, nothing)
    if supplied !== nothing
        supplied isa Tuple && length(supplied) == 2 && all(x -> x isa Modules.PMorphism, supplied) ||
            push!(issues, "induced_maps must be a pair of supplied PMorphisms in the owner's cohomology bases.")
    end
    return issues
end

function _comparison_checked_report(report, label)
    report.valid || throw(ArgumentError("$label is invalid: " * join(report.issues, "; ")))
    return report
end

function _cochain_checked_map(f)
    _comparison_checked_report(ModuleComplexes.check_module_complex(
        ModuleComplexes.source(f)), "Source complex")
    _comparison_checked_report(ModuleComplexes.check_module_complex(
        ModuleComplexes.target(f)), "Target complex")
    for C in (ModuleComplexes.source(f), ModuleComplexes.target(f))
        for t in ModuleComplexes.degree_range(C)
            _comparison_checked_report(Modules.check_module(ModuleComplexes.component(C, t)),
                "Complex term in degree $t")
            _comparison_checked_report(_algebra_morphism_validation(ModuleComplexes.differential(C, t)),
                "Complex differential in degree $t")
        end
    end
    report = _comparison_checked_report(ModuleComplexes.check_module_complex_map(f), "Cochain map")
    for t in ModuleComplexes.degree_range(f)
        _comparison_checked_report(_algebra_morphism_validation(ModuleComplexes.component(f, t)),
            "Cochain component f^$t")
    end
    return report
end

function _comparison_equation_panel(title, equation, report)
    label = report.exact ? (report.valid ? "Verified in the coefficient field." : "The identity fails.") :
        "Residual $(report.residual); tolerance $(report.tolerance); within tolerance: $(report.valid)."
    return _inspection_text_panel(title, [equation, label]; metadata=(; equation, report))
end

# Size the bounded displayed data, not the full matrices retained in metadata.
# Tiny fixtures should remain notebook-sized; a twelve-column coefficient table
# still needs enough width for its actual labels and literal coefficients.
function _comparison_panel_width(panel)
    for layer in panel.layers
        layer isa MatrixLayer || continue
        nr, nc = size(layer.entries)
        nc == 0 && return 380
        row_width = maximum(length, layer.row_labels; init=0) * 7 + 30
        return row_width + sum(max(length(layer.column_labels[j]) * 7,
            maximum(i -> length(layer.entries[i,j]), 1:nr; init=0) * 8) + 16 for j in 1:nc) + 40
    end
    return 380
end

function _comparison_panel_height(panel, width)
    heading_lines = max(1, cld(length(panel.title) * 10, width))
    subtitle_lines = sum(max(1, cld(length(line) * 7, width)) for line in split(panel.subtitle, '\n'))
    heading_height = 30 * heading_lines + 21 * subtitle_lines + 28
    for layer in panel.layers
        if layer isa MatrixLayer
            return max(175, heading_height + 30 * (size(layer.entries, 1) + 1))
        elseif layer isa TextLayer
            text_lines = sum(max(1, cld(length(line) * 8, width)) for label in layer.labels for line in split(label, '\n'))
            return max(175, heading_height + 24 * text_lines)
        end
    end
    return 260
end

function _comparison_layout(panels; hero=false)
    first_data = hero ? 2 : 1
    data = panels[first_data:end]
    panel_width = maximum(_comparison_panel_width, data; init=380)
    columns = min(length(data), panel_width <= 420 ? 3 : panel_width <= 620 ? 2 : 1)
    columns = max(columns, 1)
    width = max(columns == 3 ? 1280 : columns == 2 ? 1200 : 1100,
        columns * (panel_width + 20) + 40)
    cell_width = div(width - 40, columns) - 20
    positions = Tuple{UnitRange{Int},UnitRange{Int}}[]
    weights = Int[]
    if hero
        push!(positions, (1:1, 1:columns))
        push!(weights, 250)
    end
    offset = hero ? 1 : 0
    for (i, panel) in enumerate(data)
        row, col = offset + 1 + div(i-1, columns), 1 + mod(i-1, columns)
        push!(positions, (row:row, col:col))
        while length(weights) < row
            push!(weights, 175)
        end
        weights[row] = max(weights[row], _comparison_panel_height(panel, cell_width))
    end
    return (; panel_columns=columns, panel_positions=positions,
        panel_row_weights=weights, figure_size=(width, 120 + sum(weights) + 16 * length(weights)))
end

_cochain_stalk(f, t, q) = copy(Modules.component(ModuleComplexes.component(f, t), q))
_cochain_differential_stalk(C, t, q) =
    copy(Modules.component(ModuleComplexes.differential(C, t), q))

"""
    visual_spec(f::ModuleCochainMap; kind=:chain_map, degree, vertex=1,
                induced=false, matrix_limit=(12,12))

Display a supplied cochain map and its selected differential square. Degrees
increase under the differential: `d_D^t f^t = f^(t+1) d_C^t`.
`induced=true` explicitly computes and displays the map on cohomology modules
in the selected degree, including its chosen quotient bases. No lift or
cohomology computation runs inside a rendering backend.
"""
function _visual_spec(f::ModuleComplexes.ModuleCochainMap, kind::Symbol;
                      degree=first(_cochain_plot_degrees(f)), vertex=1,
                      matrix_limit=(12, 12), induced=false)
    kind === :chain_map || throw(ArgumentError("Unsupported cochain-map visualization $kind."))
    validation = _cochain_checked_map(f)
    C, D = ModuleComplexes.source(f), ModuleComplexes.target(f)
    field = _cochain_plot_field(f)
    t, q = Int(degree), Int(vertex)
    F, Fnext = _cochain_stalk(f, t, q), _cochain_stalk(f, t + 1, q)
    DC, DD = _cochain_differential_stalk(C, t, q), _cochain_differential_stalk(D, t, q)
    left, right = DD * F, Fnext * DC
    equation = _algebra_equation(field, left, right)
    panels = VisualizationSpec[
        _algebra_square_diagram(t, t+1, (size(F, 2), size(Fnext, 2)),
            (size(F, 1), size(Fnext, 1)); title="The selected cochain-map square",
            labels=["C^$t($q)", "D^$t($q)", "C^$(t+1)($q)", "D^$(t+1)($q)"],
            edge_labels=["f^$t", "f^$(t+1)", "d_C^$t", "d_D^$t"]),
        _algebra_matrix_panel(F, "f^$t at vertex $q", field, matrix_limit;
            source="C^$t", target="D^$t"),
        _algebra_matrix_panel(Fnext, "f^$(t+1) at vertex $q", field, matrix_limit;
            source="C^$(t+1)", target="D^$(t+1)"),
        _algebra_matrix_panel(DC, "Source differential d_C^$t", field, matrix_limit;
            source="C^$t", target="C^$(t+1)"),
        _algebra_matrix_panel(DD, "Target differential d_D^$t", field, matrix_limit;
            source="D^$t", target="D^$(t+1)"),
        _algebra_matrix_panel(left, "d_D^$t f^$t", field, matrix_limit),
        _algebra_matrix_panel(right, "f^$(t+1) d_C^$t", field, matrix_limit),
        _comparison_equation_panel("Cochain-map equation", "d_D^t f^t = f^(t+1) d_C^t", equation),
    ]
    induced_map = if induced
        h = ModuleComplexes.induced_map_on_cohomology_modules(f, t)
        push!(panels, _algebra_matrix_panel(Modules.component(h, q), "H^$t(f) at vertex $q",
            field, matrix_limit; source="H^$t(C)", target="H^$t(D)",
            subtitle="Computed cycle/quotient bases; not the chain-level coordinates"))
        h
    else
        nothing
    end
    layout = _comparison_layout(panels; hero=true)
    return VisualizationSpec(:chain_map; title="A supplied cochain map",
        subtitle="Cohomological degree $t; finite vertex $q; $(_inspection_field_label(field))",
        panels, metadata=(; degree=t, vertex=q, field, equation, validation,
            matrices=(; component=F, next_component=Fnext, source_differential=DC,
                      target_differential=DD, left, right),
            induced_map, induced_computed=induced, basis_convention=:stored_module_coordinates,
            degree_convention=:cohomological, provenance=ModuleComplexes.provenance(f), layout...))
end

function _comparison_same_module(field, M, N)
    M.Q === N.Q && M.field == N.field && dimensions(M).stalks == dimensions(N).stalks || return false
    return all(_algebra_equation(field,
        Modules.structure_map(M; source=u, target=v),
        Modules.structure_map(N; source=u, target=v)).valid for (u, v) in FiniteFringe.cover_edges(M.Q))
end

function _comparison_same_morphism(field, a, b)
    _comparison_same_module(field, a.dom, b.dom) && _comparison_same_module(field, a.cod, b.cod) || return false
    return all(_algebra_equation(field, Modules.component(a, q), Modules.component(b, q)).valid
               for q in 1:nvertices(a.dom.Q))
end

function _homotopy_equations(H)
    f, g = ModuleComplexes.source_map(H), ModuleComplexes.target_map(H)
    C, D = ModuleComplexes.source(f), ModuleComplexes.target(f)
    ModuleComplexes.source(g) === C && ModuleComplexes.target(g) === D ||
        throw(ArgumentError("Homotopic maps must have the same source and target complexes."))
    field = _cochain_plot_field(f)
    # The owner validator supplies structural/endpoint diagnostics. Its current
    # homotopy equality is literal; RealField additionally gets residual tests.
    owner = ModuleComplexes.check_module_homotopy(H)
    if !(field isa CoreModules.RealField)
        _comparison_checked_report(owner, "Supplied homotopy")
    end
    for t in ModuleComplexes.degree_range(H)
        h = ModuleComplexes.component(H, t)
        _comparison_checked_report(_algebra_morphism_validation(h), "Homotopy component h^$t")
        _comparison_same_module(field, h.dom, ModuleComplexes.component(C, t)) &&
        _comparison_same_module(field, h.cod, ModuleComplexes.component(D, t - 1)) ||
            throw(ArgumentError("Homotopy component h^$t has incompatible source or target coordinates."))
    end
    reports = NamedTuple[]
    for t in _cochain_plot_degrees(H), q in 1:nvertices(_cochain_plot_poset(f))
        F, G = _cochain_stalk(f, t, q), _cochain_stalk(g, t, q)
        A = _cochain_differential_stalk(D, t - 1, q) * _cochain_stalk(H, t, q)
        B = _cochain_stalk(H, t + 1, q) * _cochain_differential_stalk(C, t, q)
        report = _algebra_equation(field, F - G, A + B)
        push!(reports, (; degree=t, vertex=q, report))
    end
    all(x -> x.report.valid, reports) || throw(ArgumentError("The supplied homotopy does not satisfy f-g=d h+h d over the represented degrees and vertices."))
    return (; owner, equations=reports, valid=true)
end

"""
    visual_spec(H::ModuleCochainHomotopy; kind=:homotopy_comparison,
                degree, vertex=1, induced=false, induced_maps=nothing,
                matrix_limit=(12,12))

Compare the two supplied cochain maps and the actual witness
`f^t-g^t = d_D^(t-1) h^t + h^(t+1) d_C^t`. The whole represented witness is
validated; selected matrices, residuals and exact/numerical status are retained.

`induced=true` computes their cohomology maps. Alternatively, `induced_maps=(a,b)`
supplies maps to be checked against that same computation in the owner's chosen
cycle/quotient bases. A change of quotient bases requires an explicit conversion
before this comparison. With neither option, induced maps are not computed and
the picture says so. A witness is never inferred from equal dimensions.
"""
function _visual_spec(H::ModuleComplexes.ModuleCochainHomotopy, kind::Symbol;
                      degree=first(ModuleComplexes.degree_range(H)), vertex=1,
                      matrix_limit=(12, 12), induced=false, induced_maps=nothing)
    kind === :homotopy_comparison || throw(ArgumentError("Unsupported homotopy visualization $kind."))
    f, g = ModuleComplexes.source_map(H), ModuleComplexes.target_map(H)
    _cochain_checked_map(f)
    _cochain_checked_map(g)
    validation = _homotopy_equations(H)
    C, D = ModuleComplexes.source(f), ModuleComplexes.target(f)
    field = _cochain_plot_field(f)
    t, q = Int(degree), Int(vertex)
    F, G = _cochain_stalk(f, t, q), _cochain_stalk(g, t, q)
    h, hnext = _cochain_stalk(H, t, q), _cochain_stalk(H, t + 1, q)
    DC, DDprev = _cochain_differential_stalk(C, t, q), _cochain_differential_stalk(D, t - 1, q)
    left, right = F - G, DDprev * h + hnext * DC
    equation = _algebra_equation(field, left, right)
    panels = VisualizationSpec[
        _algebra_matrix_panel(F, "First lift f^$t", field, matrix_limit; source="C^$t", target="D^$t"),
        _algebra_matrix_panel(G, "Second lift g^$t", field, matrix_limit; source="C^$t", target="D^$t"),
        _algebra_matrix_panel(h, "Witness h^$t", field, matrix_limit; source="C^$t", target="D^$(t-1)"),
        _algebra_matrix_panel(hnext, "Witness h^$(t+1)", field, matrix_limit; source="C^$(t+1)", target="D^$t"),
        _algebra_matrix_panel(left, "f^$t - g^$t", field, matrix_limit),
        _algebra_matrix_panel(right, "d_D^$(t-1) h^$t + h^$(t+1) d_C^$t", field, matrix_limit),
        _comparison_equation_panel("Supplied homotopy", "f-g = d h+h d; cohomological convention", equation),
    ]
    computed = induced || induced_maps !== nothing
    maps, induced_equality = nothing, nothing
    if computed
        actual = (ModuleComplexes.induced_map_on_cohomology_modules(f, t),
                  ModuleComplexes.induced_map_on_cohomology_modules(g, t))
        if induced_maps !== nothing
            all(_comparison_same_morphism(field, induced_maps[i], actual[i]) for i in 1:2) ||
                throw(ArgumentError("Supplied induced maps do not match the actual maps in the owner's cohomology coordinates."))
        end
        maps = induced_maps === nothing ? actual : induced_maps
        A, B = copy(Modules.component(maps[1], q)), copy(Modules.component(maps[2], q))
        induced_equality = _algebra_equation(field, A, B)
        push!(panels, _algebra_matrix_panel(A, "H^$t(f)", field, matrix_limit;
            source="H^$t(C)", target="H^$t(D)"))
        push!(panels, _algebra_matrix_panel(B, "H^$t(g)", field, matrix_limit;
            source="H^$t(C)", target="H^$t(D)"))
        push!(panels, _comparison_equation_panel("Induced maps in compatible quotient bases",
            "Compare the two induced maps in the same cycle/quotient bases.", induced_equality))
    else
        push!(panels, _inspection_text_panel("Induced maps", [
            "No cohomology maps were computed for this view.",
            "Use induced=true to compute and compare them in the selected degree."]))
    end
    layout = _comparison_layout(panels)
    return VisualizationSpec(:homotopy_comparison; title="Comparing lifts with a supplied homotopy",
        subtitle="Cohomological degree $t; finite vertex $q; $(_inspection_field_label(field))",
        panels, metadata=(; degree=t, vertex=q, field, validation, equation,
            matrices=(; first=F, second=G, witness=h, next_witness=hnext, left, right),
            induced_maps=maps, induced_equality, induced_computed=computed,
            induced_basis_convention=:owner_cycle_quotient_bases,
            degree_convention=:cohomological, provenance=ModuleComplexes.provenance(H), layout...))
end

# A projective coefficient matrix represents a natural transformation between
# sums of principal upsets. Rows/columns are in the resolution's recorded
# summand order; the stalk matrix is the corresponding active submatrix.
function _resolution_coefficient_panel(A, title, field, P, dom_generators, cod_generators, limit)
    forbidden = BitMatrix([!leq(P, v, u) for v in cod_generators, u in dom_generators])
    rows, cols = 1:min(size(A, 1), limit[1]), 1:min(size(A, 2), limit[2])
    roles = [forbidden[i,j] ? :muted : :foreground for i in rows, j in cols]
    panel = _presentation_matrix_panel(A, title,
        ["P@$(v) #$i" for (i,v) in enumerate(cod_generators)],
        ["P@$(v) #$j" for (j,v) in enumerate(dom_generators)], limit;
        subtitle="$(_inspection_field_label(field)); @ labels the finite base vertex, not a geometric coordinate\n\u2020 forced zero by order; an unmarked zero is permitted",
        metadata=(; field, basis_convention=:stored_principal_upset_summands,
            source_generators=copy(dom_generators), target_generators=copy(cod_generators),
            structural_zero_mask=forbidden, cell_roles=roles,
            matrix_corner="target / source summand"))
    layer = only(panel.layers)::MatrixLayer
    entries = copy(layer.entries)
    for (ii,i) in enumerate(rows), (jj,j) in enumerate(cols)
        forbidden[i,j] && (entries[ii,jj] *= "\u2020")
    end
    return VisualizationSpec(panel.kind; title=panel.title, subtitle=panel.subtitle,
        layers=AbstractVisualizationLayer[MatrixLayer(entries, layer.row_labels, layer.column_labels)],
        metadata=panel.metadata)
end

function _resolution_coefficient_map(dom, cod, dom_generators, cod_generators, A, label)
    K = CoreModules.coeff_type(dom.field)
    A isa AbstractMatrix{K} || throw(ArgumentError("$label must be a coefficient matrix over $(dom.field)."))
    size(A) == (length(cod_generators), length(dom_generators)) ||
        throw(ArgumentError("$label has incompatible generator-matrix dimensions."))
    dom.Q === cod.Q && dom.field == cod.field ||
        throw(ArgumentError("$label must use the same finite poset and coefficient field."))
    Q = dom.Q
    for j in eachindex(dom_generators), i in eachindex(cod_generators)
        if !leq(Q, cod_generators[i], dom_generators[j]) && !iszero(A[i, j])
            throw(ArgumentError("$label has a nonzero coefficient forbidden by the principal-upset grades."))
        end
    end
    components = Matrix{K}[]
    for q in 1:nvertices(Q)
        rows = findall(v -> leq(Q, v, q), cod_generators)
        cols = findall(v -> leq(Q, v, q), dom_generators)
        push!(components, Matrix(A[rows, cols]))
    end
    f = Modules.PMorphism(dom, cod, components)
    # Public validators give useful structural diagnostics. Numerical identities
    # are also checked with the declared field tolerances rather than symbolic ==.
    report = Modules.check_morphism(f)
    if !(dom.field isa CoreModules.RealField)
        _comparison_checked_report(report, label)
    else
        for (u, v) in FiniteFringe.cover_edges(Q)
            lhs = Modules.structure_map(cod; source=u, target=v) * Modules.component(f, u)
            rhs = Modules.component(f, v) * Modules.structure_map(dom; source=u, target=v)
            _algebra_equation(dom.field, lhs, rhs).valid ||
                throw(ArgumentError("$label fails naturality on the cover $u \u2192 $v."))
        end
    end
    return f
end

function _append_visual_request_issues!(issues::Vector{String}, res::DerivedFunctors.ProjectiveResolution, kind::Symbol; kwargs...)
    kind === :resolution_lift || return _check_resolution_request!(issues, res, kind; kwargs...)
    target = get(kwargs, :target_resolution, nothing)
    f = get(kwargs, :morphism, nothing)
    lift = get(kwargs, :lift, nothing)
    target isa DerivedFunctors.ProjectiveResolution || push!(issues, "target_resolution must be a supplied ProjectiveResolution.")
    f isa Modules.PMorphism || push!(issues, "morphism must be the supplied map between the resolved modules.")
    lift isa AbstractVector && !isempty(lift) || push!(issues, "lift must be a nonempty vector of supplied coefficient matrices, starting at homological degree zero.")
    if lift isa AbstractVector && !isempty(lift)
        _comparison_degree!(issues, get(kwargs, :degree, 0), 0:(length(lift) - 1))
        length(lift) <= length(DerivedFunctors.resolution_terms(res)) || push!(issues, "The lift exceeds the stored source resolution.")
        if target isa DerivedFunctors.ProjectiveResolution
            length(lift) <= length(DerivedFunctors.resolution_terms(target)) || push!(issues, "The lift exceeds the stored target resolution.")
        end
    end
    second = get(kwargs, :comparison_lift, nothing)
    witness = get(kwargs, :homotopy, nothing)
    if second !== nothing
        second isa AbstractVector && lift isa AbstractVector && length(second) == length(lift) ||
            push!(issues, "comparison_lift must have the same degree range as lift.")
    end
    if witness !== nothing
        second === nothing && push!(issues, "A homotopy requires comparison_lift.")
        witness isa AbstractVector && lift isa AbstractVector && length(witness) == length(lift) ||
            push!(issues, "homotopy must contain h_k : P_k \u2192 Q_(k+1) for every supplied lift degree; an absent top target uses a 0-row matrix.")
    end
    _comparison_vertex!(issues, DerivedFunctors.source_module(res).Q, get(kwargs, :vertex, 1))
    _comparison_matrix_limit!(issues, get(kwargs, :matrix_limit, (12, 12)))
    return issues
end

function _resolution_validate_lift(res, target, f, coefficients, name)
    dom_terms, cod_terms = DerivedFunctors.resolution_terms(res), DerivedFunctors.resolution_terms(target)
    maps = [_resolution_coefficient_map(dom_terms[k], cod_terms[k], res.gens[k], target.gens[k],
                                        coefficients[k], "$name degree $(k-1)") for k in eachindex(coefficients)]
    aug_source, aug_target = DerivedFunctors.augmentation_map(res), DerivedFunctors.augmentation_map(target)
    ds, dt = DerivedFunctors.resolution_differentials(res), DerivedFunctors.resolution_differentials(target)
    field = f.dom.field
    checks = NamedTuple[]
    for q in 1:nvertices(f.dom.Q)
        lhs = Modules.component(aug_target, q) * Modules.component(maps[1], q)
        rhs = Modules.component(f, q) * Modules.component(aug_source, q)
        push!(checks, (; degree=0, vertex=q, kind=:augmentation, equation=_algebra_equation(field, lhs, rhs)))
        for k in 2:length(maps)
            lhs = Modules.component(dt[k-1], q) * Modules.component(maps[k], q)
            rhs = Modules.component(maps[k-1], q) * Modules.component(ds[k-1], q)
            push!(checks, (; degree=k-1, vertex=q, kind=:chain_map, equation=_algebra_equation(field, lhs, rhs)))
        end
    end
    all(x -> x.equation.valid, checks) || throw(ArgumentError("$name fails an augmented-chain-map equation."))
    return maps, checks
end

"""
    visual_spec(res::ProjectiveResolution; kind=:resolution_lift,
                target_resolution, morphism, lift, comparison_lift=nothing,
                homotopy=nothing, degree=0, vertex=1, matrix_limit=(12,12))

Inspect supplied projective-resolution lift coefficients (for example, the
output of `lift_chainmap`). `lift[k+1] : P_k \u2192 Q_k` uses the stored indicator
summand bases. The recipe checks naturality, every supplied chain equation,
and `\u03b5_Q lift[1] = morphism \u03b5_P` at every finite vertex. It constructs neither
resolutions nor lifts and makes no ambient-category identification.

An optional second lift is checked against the same supplied module map.
`homotopy[k+1] : P_k \u2192 Q_(k+1)` is a supplied homological witness for
`lift_k-comparison_k = d_Q h_k + h_(k-1) d_P`, with `h_(-1)=0`.
Its range matches the lift; use a correctly shaped zero matrix when a top
target term is absent. Verification is restricted to the supplied degrees:
a truncated lift is not presented as a complete or unique resolution map.
Different coefficient matrices may therefore lift the same module map.
"""
function _resolution_lift_spec(res::DerivedFunctors.ProjectiveResolution;
                      target_resolution, morphism, lift, comparison_lift=nothing,
                      homotopy=nothing, degree=0, vertex=1, matrix_limit=(12, 12))
    target, f = target_resolution, morphism
    field = DerivedFunctors.source_module(res).field
    f.dom === DerivedFunctors.source_module(res) && f.cod === DerivedFunctors.source_module(target) ||
        throw(ArgumentError("morphism must have exactly the modules resolved by source and target as its endpoints."))
    validation = (source=DerivedFunctors.check_projective_resolution(res),
                  target=DerivedFunctors.check_projective_resolution(target),
                  morphism=Modules.check_morphism(f))
    for (label, report) in (("Source resolution", validation.source), ("Target resolution", validation.target))
        _comparison_checked_report(report, label)
    end
    _comparison_checked_report(_algebra_morphism_validation(f), "Underlying module morphism")
    for augmentation in (DerivedFunctors.augmentation_map(res), DerivedFunctors.augmentation_map(target))
        _comparison_checked_report(_algebra_morphism_validation(augmentation), "Resolution augmentation")
        for q in 1:nvertices(f.dom.Q)
            FieldLinAlg.rank(field, Modules.component(augmentation, q)) == augmentation.cod.dims[q] ||
                throw(ArgumentError("A supplied augmentation is not pointwise surjective; it cannot identify the displayed underlying module map."))
        end
    end
    if !(field isa CoreModules.RealField)
        _comparison_checked_report(validation.morphism, "Underlying module morphism")
    end
    maps, checks = _resolution_validate_lift(res, target, f, lift, "First lift")
    second_maps, second_checks = comparison_lift === nothing ? (nothing, nothing) :
        _resolution_validate_lift(res, target, f, comparison_lift, "Second lift")
    witness_maps, witness_checks = nothing, nothing
    n = length(lift)
    if homotopy !== nothing
        dom_terms, cod_terms = DerivedFunctors.resolution_terms(res), DerivedFunctors.resolution_terms(target)
        witness_maps = [_resolution_coefficient_map(dom_terms[k],
            k < length(cod_terms) ? cod_terms[k+1] : Modules.zero_pmodule(f.cod.Q; field),
            res.gens[k], k < length(target.gens) ? target.gens[k+1] : Int[],
            homotopy[k], "Homotopy h_$(k-1)") for k in 1:n]
        ds, dt = DerivedFunctors.resolution_differentials(res), DerivedFunctors.resolution_differentials(target)
        witness_checks = NamedTuple[]
        for k in 1:n, q in 1:nvertices(f.dom.Q)
            left = Modules.component(maps[k], q) - Modules.component(second_maps[k], q)
            right = zeros(eltype(left), size(left))
            k <= length(dt) && (right += Modules.component(dt[k], q) * Modules.component(witness_maps[k], q))
            k > 1 && (right += Modules.component(witness_maps[k-1], q) * Modules.component(ds[k-1], q))
            push!(witness_checks, (; degree=k-1, vertex=q,
                equation=_algebra_equation(field, left, right), left=copy(left), right=copy(right)))
        end
        all(x -> x.equation.valid, witness_checks) || throw(ArgumentError("The supplied resolution homotopy fails f-g=d h+h d."))
    end
    k, q = Int(degree) + 1, Int(vertex)
    A = copy(Modules.component(maps[k], q))
    panels = VisualizationSpec[
        _resolution_coefficient_panel(lift[k], "Lift coefficients in degree $(k-1)", field,
            f.dom.Q, res.gens[k], target.gens[k], matrix_limit),
        _algebra_matrix_panel(A, "Lift at degree $(k-1), vertex $q", field, matrix_limit;
            source="P_$(k-1)($q)", target="Q_$(k-1)($q)"),
        _algebra_matrix_panel(Modules.component(f, q), "Induced module map at vertex $q", field, matrix_limit;
            source="M($q)", target="N($q)", subtitle="Supplied map; verified through both augmentations"),
    ]
    selected_check = only(filter(x -> x.degree == k-1 && x.vertex == q, checks))
    square_labels = k == 1 ? ["P_0($q)", "Q_0($q)", "M($q)", "N($q)"] :
        ["P_$(k-1)($q)", "Q_$(k-1)($q)", "P_$(k-2)($q)", "Q_$(k-2)($q)"]
    square_edges = k == 1 ? ["F_0", "f", "\u03b5_P", "\u03b5_Q"] :
        ["F_$(k-1)", "F_$(k-2)", "d_P", "d_Q"]
    pushfirst!(panels, _algebra_square_diagram(k-1, k-2, (0, 0), (0, 0);
        title=k == 1 ? "The lift and its underlying module map" : "The selected resolution-lift square",
        labels=square_labels, edge_labels=square_edges))
    push!(panels, _comparison_equation_panel("Augmented-chain compatibility",
        k == 1 ? "\u03b5_Q F_0 = f \u03b5_P" : "d_Q F_$(k-1) = F_$(k-2) d_P", selected_check.equation))
    if second_maps !== nothing
        B = copy(Modules.component(second_maps[k], q))
        push!(panels, _resolution_coefficient_panel(comparison_lift[k], "Second lift coefficients in degree $(k-1)",
            field, f.dom.Q, res.gens[k], target.gens[k], matrix_limit))
        push!(panels, _algebra_matrix_panel(B, "Second lift at degree $(k-1), vertex $q", field, matrix_limit))
        push!(panels, _inspection_text_panel("Same underlying module map", [
            "Both supplied lifts satisfy \u03b5_Q F_0 = f \u03b5_P at every vertex.",
            "Different chain coordinates do not imply different module maps.",
            homotopy === nothing ? "No homotopy witness was supplied." : "A supplied homotopy was verified over the displayed degree range."]))
    end
    if witness_maps !== nothing
        selected = only(filter(x -> x.degree == k-1 && x.vertex == q, witness_checks))
        push!(panels, _resolution_coefficient_panel(homotopy[k], "Witness coefficients h_$(k-1)",
            field, f.dom.Q, res.gens[k], k < length(target.gens) ? target.gens[k+1] : Int[], matrix_limit))
        push!(panels, _algebra_matrix_panel(Modules.component(witness_maps[k], q), "Witness h_$(k-1) at vertex $q",
            field, matrix_limit; source="P_$(k-1)($q)", target="Q_$k($q)"))
        push!(panels, _algebra_matrix_panel(selected.left, "F_$(k-1) - G_$(k-1)", field, matrix_limit))
        push!(panels, _algebra_matrix_panel(selected.right, "d_Q h_$(k-1) + h_$(k-2) d_P", field, matrix_limit))
        push!(panels, _comparison_equation_panel("Supplied homological homotopy",
            "F_k-G_k = d_Q h_k + h_(k-1) d_P; h_(-1)=0", selected.equation))
    end
    layout = _comparison_layout(panels; hero=true)
    return VisualizationSpec(:resolution_lift; title="Resolution lifts and their module map",
        subtitle="Homological degree $(k-1); finite vertex $q; checked degrees 0:$(n-1); $(_inspection_field_label(field))",
        panels, metadata=(; degree=k-1, vertex=q, field, checked_degrees=0:(n-1),
            validation, lift_checks=checks, comparison_checks=second_checks,
            homotopy_checks=witness_checks, homotopy_supplied=homotopy !== nothing,
            selected_matrix=A, induced_matrix=copy(Modules.component(f, q)),
            degree_convention=:homological, basis_convention=:stored_principal_upset_summands,
            provenance=DerivedFunctors.provenance(res), lift_uniqueness=:not_asserted,
            completeness=:only_supplied_degrees_checked, layout...))
end
