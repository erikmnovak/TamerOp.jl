# Schematic Hasse diagrams, independent of ambient coordinates and module maps.

function _hasse_positions(n::Int, edges::Vector{Tuple{Int,Int}})
    successors = [Int[] for _ in 1:n]
    predecessors = [Int[] for _ in 1:n]
    for (u, v) in edges
        push!(successors[u], v)
        push!(predecessors[v], u)
    end
    remaining = length.(predecessors)
    ready = findall(iszero, remaining)
    ranks = zeros(Int, n)
    cursor = 1
    while cursor <= length(ready)
        u = ready[cursor]
        for v in successors[u]
            ranks[v] = max(ranks[v], ranks[u] + 1)
            remaining[v] -= 1
            iszero(remaining[v]) && push!(ready, v)
        end
        cursor += 1
    end
    length(ready) == n || throw(ArgumentError("Hasse diagram requires an acyclic cover relation"))

    # Keep weakly connected components apart. Vertex identifiers break layout
    # ties; they never determine the order relation or vertical direction.
    components = Vector{Int}[]
    visited = falses(n)
    for seed in 1:n
        visited[seed] && continue
        component = Int[]
        stack = [seed]
        visited[seed] = true
        while !isempty(stack)
            u = pop!(stack)
            push!(component, u)
            for neighbors in (successors[u], predecessors[u]), v in neighbors
                if !visited[v]
                    visited[v] = true
                    push!(stack, v)
                end
            end
        end
        push!(components, sort!(component))
    end

    xs = zeros(Float64, n)
    offset = 0.0
    for component in components
        levels = [Int[] for _ in 0:maximum(ranks[v] for v in component)]
        for v in component
            push!(levels[ranks[v] + 1], v)
        end
        for level in levels, (i, v) in enumerate(level)
            xs[v] = 1.7 * (i - (length(level) + 1) / 2)
        end
        # Two barycentric sweeps reduce crossings without an external graph
        # layout dependency. All tie breaking remains deterministic.
        for _ in 1:2
            for level in levels
                sort!(level; by=v -> (isempty(predecessors[v]) ? xs[v] :
                    sum(xs[u] for u in predecessors[v]) / length(predecessors[v]), v))
                for (i, v) in enumerate(level)
                    xs[v] = 1.7 * (i - (length(level) + 1) / 2)
                end
            end
            for level in Iterators.reverse(levels)
                sort!(level; by=v -> (isempty(successors[v]) ? xs[v] :
                    sum(xs[u] for u in successors[v]) / length(successors[v]), v))
                for (i, v) in enumerate(level)
                    xs[v] = 1.7 * (i - (length(level) + 1) / 2)
                end
            end
        end
        width = 1.7 * (maximum(length, levels) - 1)
        for v in component
            xs[v] += offset + width / 2
        end
        offset += width + 2.0
    end
    if n > 0
        center = (minimum(xs) + maximum(xs)) / 2
        xs .-= center
    end
    positions = NTuple{2,Float64}[(xs[v], 1.35 * ranks[v]) for v in 1:n]
    return (; positions, ranks, components)
end

function _hasse_arrow!(segments::Vector{NTuple{4,Float64}},
                       heads::Vector{Vector{NTuple{2,Float64}}},
                       source::NTuple{2,Float64}, target::NTuple{2,Float64};
                       start_gap::Float64=0.13, end_gap::Float64=0.17)
    dx, dy = target[1] - source[1], target[2] - source[2]
    distance = hypot(dx, dy)
    distance > start_gap + end_gap || return nothing
    ux, uy = dx / distance, dy / distance
    start = (source[1] + start_gap * ux, source[2] + start_gap * uy)
    tip = (target[1] - end_gap * ux, target[2] - end_gap * uy)
    base = (tip[1] - 0.13 * ux, tip[2] - 0.13 * uy)
    push!(segments, (start[1], start[2], base[1], base[2]))
    push!(heads, [tip, (base[1] - 0.055 * uy, base[2] + 0.055 * ux),
                       (base[1] + 0.055 * uy, base[2] - 0.055 * ux)])
    return nothing
end

function _hasse_spec(P::AbstractPoset; dims=nothing, vertex=nothing, pair=nothing,
                     field_label=nothing)
    n = nvertices(P)
    vertex === nothing || (vertex isa Integer && !(vertex isa Bool) && 1 <= vertex <= n) ||
        throw(ArgumentError("vertex must be a poset vertex identifier in 1:$n"))
    pair === nothing || ((pair isa Tuple || pair isa AbstractVector) && length(pair) == 2 &&
        all(v -> v isa Integer && !(v isa Bool) && 1 <= v <= n, pair)) ||
        throw(ArgumentError("pair must contain two poset vertex identifiers in 1:$n"))
    dims === nothing || (length(dims) == n && all(d -> d isa Integer && d >= 0, dims)) ||
        throw(ArgumentError("dimensions must contain one nonnegative integer per vertex"))
    selected_vertex = vertex === nothing ? nothing : Int(vertex)
    selected_pair = pair === nothing ? nothing : (Int(pair[1]), Int(pair[2]))
    dimensions = dims === nothing ? nothing : Int[d for d in dims]
    edges = sort!(Tuple{Int,Int}[(u, v) for (u, v) in FiniteFringe.cover_edges(P)])
    layout = _hasse_positions(n, edges)
    positions = layout.positions
    relation = if selected_pair === nothing
        :none
    else
        u, v = selected_pair
        u == v ? :equal : leq(P, u, v) ? (selected_pair in edges ? :cover : :comparable) :
            leq(P, v, u) ? :reverse_comparable : :incomparable
    end

    segments = NTuple{4,Float64}[]
    heads = Vector{NTuple{2,Float64}}[]
    for (u, v) in edges
        _hasse_arrow!(segments, heads, positions[u], positions[v])
    end
    layers = AbstractVisualizationLayer[
        SegmentLayer(segments, :gray45, 1.0, 1.4),
        PolygonLayer(heads, :gray45, :gray45, 1.0, 0.0),
    ]
    legend_entries = [(; label="cover relation (upward)", color=:gray45, style=:line)]
    if relation in (:cover, :comparable)
        u, v = selected_pair
        selected_segments = NTuple{4,Float64}[]
        selected_heads = Vector{NTuple{2,Float64}}[]
        if relation === :cover
            _hasse_arrow!(selected_segments, selected_heads, positions[u], positions[v])
        else
            # Bend the explicitly selected non-cover relation away from a
            # vertical chain. It is never added to the cover-edge metadata.
            source, target = positions[u], positions[v]
            bend = (min(source[1], target[1]) - 0.65, (source[2] + target[2]) / 2)
            direction = hypot(bend[1] - source[1], bend[2] - source[2])
            start = (source[1] + 0.13 * (bend[1] - source[1]) / direction,
                     source[2] + 0.13 * (bend[2] - source[2]) / direction)
            push!(selected_segments, (start[1], start[2], bend[1], bend[2]))
            _hasse_arrow!(selected_segments, selected_heads, bend, target; start_gap=0.0)
        end
        push!(layers, SegmentLayer(selected_segments, :navy, 1.0, 2.6,
                                   relation === :cover ? :solid : :dash))
        push!(layers, PolygonLayer(selected_heads, :navy, :navy, 1.0, 0.0))
        push!(legend_entries, (; label=relation === :cover ? "selected cover" :
            "selected comparable pair (not a cover)", color=:navy, style=:line))
    end
    selected_vertex === nothing || push!(layers,
        PointLayer([positions[selected_vertex]], :black, 1.0, 29.0))
    if selected_pair !== nothing
        u, v = selected_pair
        push!(layers, PointLayer([positions[u]], :navy, 1.0, 27.0))
        u == v || push!(layers, PointLayer([positions[v]], :firebrick3, 1.0, 27.0))
    end
    labels = String[]
    label_positions = NTuple{2,Float64}[]
    for v in 1:n
        push!(layers, PointLayer([positions[v]], _box_region_color(v), 1.0, 17.0))
        label = string(v)
        dimensions === nothing || (label *= "\ndim = $(dimensions[v])")
        if selected_pair !== nothing
            u, w = selected_pair
            v == u && (label *= u == w ? "\nsource = target" : "\nsource")
            v == w && u != w && (label *= "\ntarget")
        end
        push!(labels, label)
        # Leave room for upward arrows beside multiline dimension/role labels.
        push!(label_positions, (positions[v][1] + 0.5, positions[v][2] + 0.07))
    end
    push!(layers, TextLayer(labels, label_positions, :black, 12.0))
    n == 0 && push!(layers, TextLayer(["empty poset"], [(0.0, 0.0)], :gray40, 14.0))

    subtitle = "Schematic order layout; not ambient coordinates"
    field_label === nothing || (subtitle *= "\nField: $(field_label)")
    relation === :equal && (subtitle *= "\nEqual finite labels: identity at this finite vertex")
    relation === :reverse_comparable && (subtitle *= "\nSource is above target; no forward map")
    relation === :incomparable && (subtitle *= "\nIncomparable finite labels; no structure map")
    xlimits = n == 0 ? (-1.0, 1.0) : (minimum(first, positions) - 1.0, maximum(first, positions) + 1.45)
    ylimits = n == 0 ? (-1.0, 1.0) : (-0.65, maximum(last, positions) + 0.85)
    return VisualizationSpec(:hasse;
        title=dimensions === nothing ? "Finite poset" : "Finite poset and stalk dimensions",
        subtitle, layers,
        axes=_default_axes_2d(; xlabel="", ylabel="", xlimits, ylimits,
                              xticks=(Float64[], String[]), yticks=(Float64[], String[])),
        legend=_default_legend(; visible=!isempty(edges) || relation in (:cover, :comparable),
                                entries=legend_entries),
        interaction=_default_interaction(; labels=true),
        metadata=(; object=:finite_poset, layout=:schematic, hide_decorations=true,
                    vertex_ids=collect(1:n), positions, cover_edges=edges, dimensions,
                    selected_vertex, selected_pair, relation, field_label,
                    ranks=layout.ranks, components=layout.components,
                    figure_size=(760, 560), legend_position=:bottom))
end
