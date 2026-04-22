using Dates
using Random
using SHA
using Statistics
using TOML

const _THESIS_FALLBACK_SOURCE_MODE = let source_mode = false
    try
        @eval using TamerOp
    catch
        include(joinpath(@__DIR__, "..", "..", "src", "TamerOp.jl"))
        @eval using .TamerOp
        source_mode = true
    end
    source_mode
end

const _THESIS_HAVE_NEARESTNEIGHBORS = let ok = false
    try
        @eval using NearestNeighbors
        ok = true
    catch
        ok = false
    end
    ok
end

if _THESIS_FALLBACK_SOURCE_MODE && _THESIS_HAVE_NEARESTNEIGHBORS
    include(joinpath(@__DIR__, "..", "..", "ext", "TamerOpNearestNeighborsExt.jl"))
end

const DI = TamerOp.DataIngestion
const DT = TamerOp.DataTypes
const EC = TamerOp.EncodingCore
const IC = TamerOp.InvariantCore
const IM = TamerOp.MultiparameterImages
const INV = TamerOp.Invariants
const PLP = TamerOp.PLPolyhedra
const SM = TamerOp.SignedMeasures
const SI = TamerOp.SliceInvariants
const OPT = TamerOp.Options
const FF = TamerOp.FiniteFringe
const FZ = TamerOp.FlangeZn
const SD = TamerOp.SyntheticData
const SER = TamerOp.Serialization

const _PRIMARY_INVARIANTS = [
    :euler_signed_measure,
    :euler_surface,
    :rank_signed_measure,
    :rank_invariant,
    :restricted_hilbert,
    :rectangle_signed_barcode,
    :slice_barcodes,
    :mp_landscape,
    :mpp_decomposition,
    :mpp_image,
]

const _DEGREE_AGNOSTIC_INVARIANTS = Set([:euler_signed_measure, :euler_surface])
const _DEGREE_SENSITIVE_INVARIANTS = Set(inv for inv in _PRIMARY_INVARIANTS if !(inv in _DEGREE_AGNOSTIC_INVARIANTS))
const _MPP_INVARIANTS = Set([:mpp_decomposition, :mpp_image])
const _PRESENTATION_FAMILIES = Set([:pl_fringe, :flange])
const _BUILTIN_FILTRATION_FAMILIES = Set([
    :alpha,
    :clique_lower_star,
    :core,
    :core_delaunay,
    :cubical,
    :degree_rips,
    :delaunay_lower_star,
    :edge_weighted,
    :function_delaunay,
    :function_rips,
    :graph_centrality,
    :graph_function_geodesic_bifiltration,
    :graph_geodesic,
    :graph_lower_star,
    :graph_weight_threshold,
    :image_distance_bifiltration,
    :landmark_rips,
    :lower_star,
    :rhomboid,
    :rips,
    :rips_codensity,
    :rips_density,
    :rips_lowerstar,
    :wing_vein_bifiltration,
])

const _RAW_FAMILY_CASES = [
    (family=:rips, family_case=:rips, source_templates=[:pc_annulus2d_rips, :pc_clusters8d_rips], nparams=1, degree_labels=[:H0, :H1, :H2]),
    (family=:landmark_rips, family_case=:landmark_rips, source_templates=[:pc_annulus2d_rips, :pc_clusters8d_rips], nparams=1, degree_labels=[:H0, :H1, :H2]),
    (family=:rips_density, family_case=:rips_density, source_templates=[:pc_annulus2d_rips, :pc_clusters8d_rips], nparams=2, degree_labels=[:H0, :H1, :H2]),
    (family=:rips_codensity, family_case=:rips_codensity, source_templates=[:pc_annulus2d_rips, :pc_clusters8d_rips], nparams=2, degree_labels=[:H0, :H1, :H2]),
    (family=:rips_lowerstar, family_case=:rips_lowerstar, source_templates=[:pc_annulus2d_rips, :pc_clusters8d_rips], nparams=2, degree_labels=[:H0, :H1, :H2]),
    (family=:function_rips, family_case=:function_rips, source_templates=[:pc_annulus2d_rips, :pc_clusters8d_rips], nparams=2, degree_labels=[:H0, :H1, :H2]),
    (family=:degree_rips, family_case=:degree_rips, source_templates=[:pc_annulus2d_rips, :pc_clusters8d_rips], nparams=2, degree_labels=[:H0, :H1, :H2]),
    (family=:alpha, family_case=:alpha, source_templates=[:pc_annulus2d_heavy], nparams=1, degree_labels=[:H0, :H1]),
    (family=:delaunay_lower_star, family_case=:delaunay_lower_star, source_templates=[:pc_annulus2d_heavy], nparams=1, degree_labels=[:H0, :H1]),
    (family=:function_delaunay, family_case=:function_delaunay, source_templates=[:pc_annulus2d_heavy], nparams=2, degree_labels=[:H0, :H1]),
    (family=:core_delaunay, family_case=:core_delaunay, source_templates=[:pc_annulus2d_heavy], nparams=2, degree_labels=[:H0, :H1]),
    (family=:core, family_case=:point_core, source_templates=[:pc_annulus2d_heavy], nparams=2, degree_labels=[:H0, :H1]),
    (family=:rhomboid, family_case=:rhomboid, source_templates=[:pc_annulus2d_heavy], nparams=2, degree_labels=[:H0, :H1]),
    (family=:graph_lower_star, family_case=:graph_lower_star, source_templates=[:graph_sparse_grid], nparams=1, degree_labels=[:H0, :H1]),
    (family=:graph_centrality, family_case=:graph_centrality, source_templates=[:graph_sparse_grid], nparams=1, degree_labels=[:H0, :H1]),
    (family=:graph_geodesic, family_case=:graph_geodesic, source_templates=[:graph_sparse_grid], nparams=1, degree_labels=[:H0, :H1]),
    (family=:graph_function_geodesic_bifiltration, family_case=:graph_function_geodesic_bifiltration, source_templates=[:graph_sparse_grid], nparams=2, degree_labels=[:H0, :H1]),
    (family=:graph_weight_threshold, family_case=:graph_weight_threshold_graph, source_templates=[:graph_sparse_grid], nparams=1, degree_labels=[:H0, :H1]),
    (family=:edge_weighted, family_case=:edge_weighted, source_templates=[:graph_sparse_grid], nparams=1, degree_labels=[:H0, :H1]),
    (family=:core, family_case=:graph_core, source_templates=[:graph_sparse_grid], nparams=2, degree_labels=[:H0, :H1]),
    (family=:clique_lower_star, family_case=:clique_lower_star, source_templates=[:graph_clique_grid], nparams=1, degree_labels=[:H0, :H1, :H2]),
    (family=:graph_weight_threshold, family_case=:graph_weight_threshold_clique, source_templates=[:graph_clique_grid], nparams=1, degree_labels=[:H0, :H1, :H2]),
    (family=:lower_star, family_case=:lower_star, source_templates=[:image_checkerboard], nparams=1, degree_labels=[:H0, :H1]),
    (family=:cubical, family_case=:cubical, source_templates=[:image_checkerboard], nparams=1, degree_labels=[:H0, :H1]),
    (family=:image_distance_bifiltration, family_case=:image_distance_bifiltration, source_templates=[:image_checkerboard], nparams=2, degree_labels=[:H0, :H1]),
    (family=:wing_vein_bifiltration, family_case=:wing_vein_bifiltration, source_templates=[:embedded_grid], nparams=2, degree_labels=[:H0, :H1]),
]

const _PRESENTATION_CASES = [
    (family=:pl_fringe, family_case=:pl_fringe_axis, source_templates=[:pl_axis], nparams=2, degree_labels=[:na]),
    (family=:pl_fringe, family_case=:pl_fringe_coupled, source_templates=[:pl_coupled], nparams=2, degree_labels=[:na]),
    (family=:flange, family_case=:flange_friendly, source_templates=[:flange_friendly], nparams=2, degree_labels=[:na]),
    (family=:flange, family_case=:flange_adversarial, source_templates=[:flange_adversarial], nparams=2, degree_labels=[:na]),
]

const _SOURCE_TEMPLATES = Dict{Symbol,NamedTuple}(
    :pc_annulus2d_rips => (source_kind=:point_cloud, variant=:annulus2d, size_group=:point_rips_2d),
    :pc_clusters8d_rips => (source_kind=:point_cloud, variant=:clusters8d, size_group=:point_rips_8d),
    :pc_annulus2d_heavy => (source_kind=:point_cloud, variant=:annulus2d_heavy, size_group=:point_heavy_2d),
    :graph_sparse_grid => (source_kind=:graph, variant=:grid_sparse, size_group=:graph_sparse),
    :graph_clique_grid => (source_kind=:graph, variant=:grid_clique, size_group=:graph_clique),
    :image_checkerboard => (source_kind=:image, variant=:checkerboard, size_group=:image),
    :embedded_grid => (source_kind=:embedded_graph, variant=:embedded_grid, size_group=:embedded),
    :pl_axis => (source_kind=:pl_fringe, variant=:axis, size_group=:pl_fringe),
    :pl_coupled => (source_kind=:pl_fringe, variant=:coupled, size_group=:pl_fringe),
    :flange_friendly => (source_kind=:flange, variant=:friendly, size_group=:flange),
    :flange_adversarial => (source_kind=:flange, variant=:adversarial, size_group=:flange),
)

const _SIZE_LADDERS = Dict{Symbol,Dict{Symbol,Dict{Symbol,Vector{Int}}}}(
    :thesis => Dict(
        :point_rips_2d => Dict(:na => [10_000, 50_000], :H0 => [10_000, 50_000], :H1 => [2_500, 10_000], :H2 => [500, 2_000]),
        :point_rips_8d => Dict(:na => [10_000, 50_000], :H0 => [10_000, 50_000], :H1 => [2_500, 10_000], :H2 => [500, 2_000]),
        :point_heavy_2d => Dict(:na => [1_000, 5_000], :H0 => [1_000, 5_000], :H1 => [500, 2_000]),
        :graph_sparse => Dict(:na => [10_000, 50_000], :H0 => [10_000, 50_000], :H1 => [2_500, 10_000]),
        :graph_clique => Dict(:na => [2_500, 10_000], :H0 => [2_500, 10_000], :H1 => [1_000, 2_500], :H2 => [250, 1_000]),
        :image => Dict(:na => [128, 256], :H0 => [128, 256], :H1 => [128, 256]),
        :embedded => Dict(:na => [256, 1024], :H0 => [256, 1024], :H1 => [256, 1024]),
        :pl_fringe => Dict(:na => [1_000, 10_000, 50_000]),
        :flange => Dict(:na => [1_000, 10_000, 50_000]),
    ),
    :smoke => Dict(
        :point_rips_2d => Dict(:na => [80], :H0 => [80], :H1 => [48], :H2 => [24]),
        :point_rips_8d => Dict(:na => [80], :H0 => [80], :H1 => [48], :H2 => [24]),
        :point_heavy_2d => Dict(:na => [40], :H0 => [40], :H1 => [24]),
        :graph_sparse => Dict(:na => [64], :H0 => [64], :H1 => [40]),
        :graph_clique => Dict(:na => [36], :H0 => [36], :H1 => [24], :H2 => [12]),
        :image => Dict(:na => [24], :H0 => [24], :H1 => [24]),
        :embedded => Dict(:na => [36], :H0 => [36], :H1 => [36]),
        :pl_fringe => Dict(:na => [64, 256]),
        :flange => Dict(:na => [64, 256]),
    ),
)

@inline function _arg(args::Vector{String}, key::String, default::String)
    prefix = key * "="
    for a in args
        startswith(a, prefix) || continue
        return split(a, "=", limit=2)[2]
    end
    return default
end

@inline _arg_int(args::Vector{String}, key::String, default::Int) = parse(Int, _arg(args, key, string(default)))

@inline function _parse_symbol_list(raw::AbstractString)
    s = strip(raw)
    lowercase(s) == "all" && return nothing
    isempty(s) && return nothing
    out = Symbol[]
    for part in split(s, ',')
        token = Symbol(replace(lowercase(strip(part)), '-' => '_'))
        push!(out, token)
    end
    return unique(out)
end

function _profile_defaults(profile::Symbol)
    if profile == :desktop
        return (reps=4, trim_between_reps=true, trim_between_cases=true)
    elseif profile == :balanced
        return (reps=5, trim_between_reps=false, trim_between_cases=true)
    elseif profile == :stress
        return (reps=9, trim_between_reps=false, trim_between_cases=false)
    elseif profile == :probe
        return (reps=3, trim_between_reps=false, trim_between_cases=true)
    elseif profile == :smoke
        return (reps=1, trim_between_reps=true, trim_between_cases=true)
    end
    error("--profile must be one of: desktop, balanced, stress, probe, smoke")
end

@inline function _memory_relief!()
    GC.gc()
    GC.gc(true)
    try
        Base.Libc.malloc_trim(0)
    catch
    end
    return nothing
end

@inline _degree_symbol_to_int(sym::Symbol) = sym === :H0 ? 0 : sym === :H1 ? 1 : sym === :H2 ? 2 : nothing
@inline _degree_string(sym::Symbol) = lowercase(String(sym))

function _point_knn(n::Int)
    return max(1, min(12, n - 1))
end

function _point_budget(n::Int)
    return (
        max_simplices=max(200_000, 60 * n),
        max_edges=max(200_000, 40 * n),
        memory_budget_bytes=8_000_000_000,
    )
end

function _landmark_indices(n::Int)
    target = min(n, max(16, ceil(Int, sqrt(n))))
    step = max(1, fld(n, max(1, target)))
    idx = collect(1:step:n)
    length(idx) > target && resize!(idx, target)
    isempty(idx) && push!(idx, 1)
    return idx
end

function _grid_points_exact(n::Int)
    n >= 1 || error("grid source requires n >= 1")
    nrows = max(1, floor(Int, sqrt(n)))
    ncols = cld(n, nrows)
    coords = Vector{Vector{Float64}}(undef, n)
    for idx in 1:n
        i = fld(idx - 1, ncols) + 1
        j = ((idx - 1) % ncols) + 1
        coords[idx] = [Float64(j - 1), Float64(i - 1)]
    end
    return coords, nrows, ncols
end

function _grid_edges_exact(n::Int; diagonals::Bool=false)
    coords, nrows, ncols = _grid_points_exact(n)
    index(i, j) = (i - 1) * ncols + j
    edges = Tuple{Int,Int}[]
    seen = Set{Tuple{Int,Int}}()
    for idx in 1:n
        i = fld(idx - 1, ncols) + 1
        j = ((idx - 1) % ncols) + 1
        if j < ncols
            nbr = index(i, j + 1)
            if nbr <= n
                e = (idx, nbr)
                push!(edges, e)
                push!(seen, e)
            end
        end
        if i < nrows
            nbr = index(i + 1, j)
            if nbr <= n
                e = (idx, nbr)
                if !(e in seen)
                    push!(edges, e)
                    push!(seen, e)
                end
            end
        end
        if diagonals && i < nrows && j < ncols
            nbr = index(i + 1, j + 1)
            if nbr <= n
                e = (min(idx, nbr), max(idx, nbr))
                if !(e in seen)
                    push!(edges, e)
                    push!(seen, e)
                end
            end
        end
        if diagonals && i < nrows && j > 1
            nbr = index(i + 1, j - 1)
            if nbr <= n
                e = (min(idx, nbr), max(idx, nbr))
                if !(e in seen)
                    push!(edges, e)
                    push!(seen, e)
                end
            end
        end
    end
    sort!(edges)
    weights = Float64[hypot(coords[u][1] - coords[v][1], coords[u][2] - coords[v][2]) + 1e-3 for (u, v) in edges]
    return coords, edges, weights
end

function _grid_graph_data(n::Int; diagonals::Bool=false)
    coords, edges, weights = _grid_edges_exact(n; diagonals=diagonals)
    return TamerOp.GraphData(n, edges; coords=coords, weights=weights)
end

function _embedded_grid_graph(n::Int; diagonals::Bool=false)
    coords, edges, weights = _grid_edges_exact(n; diagonals=diagonals)
    xs = map(p -> p[1], coords)
    ys = map(p -> p[2], coords)
    bbox = (minimum(xs), maximum(xs), minimum(ys), maximum(ys))
    return TamerOp.EmbeddedPlanarGraph2D(coords, edges; bbox=bbox), weights
end

function _checkerboard_with_mask(side::Int)
    blocks = (max(4, fld(side, 16)), max(4, fld(side, 16)))
    img = SD.checkerboard_image(size=(side, side), blocks=blocks, low=0.0, high=1.0)
    data = getfield(img, :data)
    mask = data .> 0.5
    return img, mask
end

function _grid_bars_for_target(target::Int; float_coords::Bool=true)
    target >= 1 || error("presentation target must be positive")
    nbars = max(4, ceil(Int, sqrt(target)))
    side = max(2, ceil(Int, sqrt(nbars)))
    bars = Vector{Tuple{Vector{Float64},Vector{Float64}}}(undef, nbars)
    step = 1.0
    width = 1.5
    height = 1.25
    for idx in 1:nbars
        gx = Float64((idx - 1) % side)
        gy = Float64(fld(idx - 1, side))
        lo = [0.5 * gx * step, 0.5 * gy * step]
        hi = [lo[1] + width, lo[2] + height]
        bars[idx] = (lo, hi)
    end
    return bars
end

function _upper_bidiagonal_phi(n::Int)
    phi = zeros(TamerOp.QQ, n, n)
    for i in 1:n
        phi[i, i] = TamerOp.QQ(1)
        if i < n
            phi[i, i + 1] = TamerOp.QQ(-1)
        end
    end
    return phi
end

function _coupled_orthant_flange(target::Int)
    bars = _grid_bars_for_target(target)
    dim = 2
    tau = FZ.face(dim, Int[])
    flats = Vector{FZ.IndFlat{2}}(undef, length(bars))
    injectives = Vector{FZ.IndInj{2}}(undef, length(bars))
    for i in eachindex(bars)
        lo, hi = bars[i]
        flats[i] = FZ.IndFlat(tau, (round(Int, lo[1]), round(Int, lo[2])); id=Symbol(:F, i))
        injectives[i] = FZ.IndInj(tau, (round(Int, hi[1]), round(Int, hi[2])); id=Symbol(:E, i))
    end
    phi = _upper_bidiagonal_phi(length(bars))
    return FZ.Flange{TamerOp.QQ}(dim, flats, injectives, phi; field=TamerOp.QQField())
end

function _presentation_encoding_options(target::Int)
    return OPT.EncodingOptions(
        backend=:auto,
        max_regions=max(100_000, 8 * target),
        poset_kind=:signature,
        field=TamerOp.CoreModules.QQField(),
    )
end

function generate_source_object(source_case::AbstractDict{String,<:Any})
    variant = Symbol(source_case["source_variant"])
    requested_size = Int(source_case["requested_size"])
    seed = Int(source_case["seed"])
    rng = Random.MersenneTwister(seed)
    if variant === :annulus2d || variant === :annulus2d_heavy
        n_annulus = max(1, round(Int, 0.85 * requested_size))
        n_noise = max(0, requested_size - n_annulus)
        return SD.noisy_annulus(
            n_annulus=n_annulus,
            n_noise=n_noise,
            r_inner=0.7,
            r_outer=1.3,
            dim=2,
            rng=rng,
        )
    elseif variant === :clusters8d
        a = fld(requested_size, 2)
        b = requested_size - a
        centers = [zeros(8), fill(2.0, 8)]
        return SD.gaussian_clusters(counts=[a, b], centers=centers, std=0.35, rng=rng)
    elseif variant === :grid_sparse
        return _grid_graph_data(requested_size; diagonals=false)
    elseif variant === :grid_clique
        return _grid_graph_data(requested_size; diagonals=true)
    elseif variant === :checkerboard
        img, _ = _checkerboard_with_mask(requested_size)
        return img
    elseif variant === :embedded_grid
        graph, _ = _embedded_grid_graph(requested_size; diagonals=true)
        return graph
    elseif variant === :axis
        bars = _grid_bars_for_target(requested_size)
        return SD.pl_box_fringe(bars=bars)
    elseif variant === :coupled
        bars = _grid_bars_for_target(requested_size)
        return SD.coupled_pl_fringe(bars=bars)
    elseif variant === :friendly
        bars = [([round(Int, lo[1]), round(Int, lo[2])], [round(Int, hi[1]), round(Int, hi[2])]) for (lo, hi) in _grid_bars_for_target(requested_size)]
        return SD.orthant_bar_flange(bars=bars)
    elseif variant === :adversarial
        return _coupled_orthant_flange(requested_size)
    end
    error("Unsupported source variant $(variant)")
end

function _fixture_extension(source_kind::Symbol)
    if source_kind in (:point_cloud, :graph, :image, :embedded_graph)
        return ".dataset.json"
    elseif source_kind === :pl_fringe
        return ".pl_fringe.json"
    elseif source_kind === :flange
        return ".flange.json"
    end
    error("Unsupported source kind $(source_kind)")
end

function source_fixture_relpath(source_case::AbstractDict{String,<:Any})
    source_kind = Symbol(source_case["source_kind"])
    return joinpath("fixtures", string(source_case["id"]) * _fixture_extension(source_kind))
end

function save_source_fixture(path::AbstractString, source_case::AbstractDict{String,<:Any})
    mkpath(dirname(path))
    obj = generate_source_object(source_case)
    source_kind = Symbol(source_case["source_kind"])
    if source_kind in (:point_cloud, :graph, :image, :embedded_graph)
        SER.save_dataset_json(path, obj; profile=:compact)
    elseif source_kind === :pl_fringe
        SER.save_pl_fringe_json(path, obj; profile=:compact)
    elseif source_kind === :flange
        SER.save_flange_json(path, obj; profile=:compact)
    else
        error("Unsupported source kind $(source_kind)")
    end
    return obj
end

function load_source_fixture(path::AbstractString, source_case::AbstractDict{String,<:Any})
    source_kind = Symbol(source_case["source_kind"])
    if source_kind in (:point_cloud, :graph, :image, :embedded_graph)
        return SER.load_dataset_json(path)
    elseif source_kind === :pl_fringe
        return SER.load_pl_fringe_json(path)
    elseif source_kind === :flange
        return SER.load_flange_json(path)
    end
    error("Unsupported source kind $(source_kind)")
end

function _point_values(data)
    mat = DT.point_matrix(data)
    n = size(mat, 1)
    out = Vector{Float64}(undef, n)
    for i in 1:n
        x1 = mat[i, 1]
        x2 = size(mat, 2) >= 2 ? mat[i, 2] : 0.0
        out[i] = x1 + 0.35 * x2
    end
    return out
end

function _graph_values(data)
    coords = DT.coord_matrix(data)
    n = size(coords, 1)
    out = Vector{Float64}(undef, n)
    for i in 1:n
        x1 = coords[i, 1]
        x2 = size(coords, 2) >= 2 ? coords[i, 2] : 0.0
        out[i] = x1 + 0.25 * x2
    end
    return out
end

function _graph_or_embedded_bbox(data)
    if data isa DT.GraphData
        coords = DT.coord_matrix(data)
        return (minimum(coords[:, 1]), maximum(coords[:, 1]), minimum(coords[:, 2]), maximum(coords[:, 2]))
    elseif data isa DT.EmbeddedPlanarGraph2D
        return data.bbox
    end
    error("bbox helper requires graph-like data")
end

function _wing_grid_for_size(requested_size::Int)
    side = clamp(round(Int, sqrt(requested_size)), 32, 128)
    return (side, side)
end

function _max_dim_for_job(family_case::Symbol, degree::Union{Nothing,Int})
    if family_case in (:rips, :landmark_rips, :rips_density, :rips_codensity, :rips_lowerstar, :function_rips, :degree_rips)
        # The sparse kNN construction used for thesis-scale Rips-family jobs is
        # currently edge-driven, so degree-agnostic lanes must stay on the
        # 1-skeleton unless they opt into a different construction policy.
        return degree === nothing ? 1 : (degree + 1)
    elseif family_case in (:alpha, :delaunay_lower_star, :function_delaunay, :core_delaunay, :point_core, :rhomboid)
        return 2
    elseif family_case in (:clique_lower_star, :graph_weight_threshold_clique)
        return 2
    end
    return nothing
end

function build_filtration_spec(data, job::AbstractDict{String,<:Any})
    family_case = Symbol(job["family_case"])
    degree_label = Symbol(job["degree_label"])
    degree = _degree_symbol_to_int(degree_label)
    max_dim = _max_dim_for_job(family_case, degree)
    if family_case in (:rips, :landmark_rips, :rips_density, :rips_codensity, :rips_lowerstar, :function_rips, :degree_rips)
        n = DT.npoints(data)
        knn = _point_knn(n)
        budget = _point_budget(n)
        if family_case === :rips
            return TamerOp.FiltrationSpec(kind=:rips, max_dim=max_dim, knn=knn, nn_backend=:auto,
                                          construction=TamerOp.ConstructionOptions(; sparsify=:knn, budget=budget))
        elseif family_case === :landmark_rips
            return TamerOp.FiltrationSpec(kind=:landmark_rips, max_dim=max_dim, landmarks=_landmark_indices(n),
                                          construction=TamerOp.ConstructionOptions(; budget=budget))
        elseif family_case === :rips_density
            return TamerOp.FiltrationSpec(kind=:rips_density, max_dim=max_dim, density_k=knn, knn=knn, nn_backend=:auto,
                                          construction=TamerOp.ConstructionOptions(; sparsify=:knn, budget=budget))
        elseif family_case === :rips_codensity
            return TamerOp.FiltrationSpec(kind=:rips_codensity, max_dim=max_dim, knn=knn, dtm_mass=0.15, nn_backend=:auto,
                                          construction=TamerOp.ConstructionOptions(; sparsify=:knn, budget=budget))
        elseif family_case === :rips_lowerstar
            return TamerOp.FiltrationSpec(kind=:rips_lowerstar, max_dim=max_dim, knn=knn, coord=1, nn_backend=:auto,
                                          construction=TamerOp.ConstructionOptions(; sparsify=:knn, budget=budget))
        elseif family_case === :function_rips
            return TamerOp.FiltrationSpec(kind=:function_rips, max_dim=max_dim, vertex_values=_point_values(data), simplex_agg=:max,
                                          knn=knn, nn_backend=:auto,
                                          construction=TamerOp.ConstructionOptions(; sparsify=:knn, budget=budget))
        elseif family_case === :degree_rips
            return TamerOp.FiltrationSpec(kind=:degree_rips, max_dim=max_dim, knn=knn, nn_backend=:auto,
                                          construction=TamerOp.ConstructionOptions(; sparsify=:knn, budget=budget))
        end
    elseif family_case in (:alpha, :delaunay_lower_star, :function_delaunay, :core_delaunay, :point_core, :rhomboid)
        pvals = _point_values(data)
        if family_case === :alpha
            return TamerOp.FiltrationSpec(kind=:alpha, max_dim=max_dim)
        elseif family_case === :delaunay_lower_star
            return TamerOp.FiltrationSpec(kind=:delaunay_lower_star, max_dim=max_dim, vertex_values=pvals, simplex_agg=:max)
        elseif family_case === :function_delaunay
            return TamerOp.FiltrationSpec(kind=:function_delaunay, max_dim=max_dim, vertex_values=pvals, simplex_agg=:max)
        elseif family_case === :core_delaunay
            return TamerOp.FiltrationSpec(kind=:core_delaunay, max_dim=max_dim)
        elseif family_case === :point_core
            return TamerOp.FiltrationSpec(kind=:core, knn=max(1, min(10, DT.npoints(data) - 1)), vertex_values=pvals)
        elseif family_case === :rhomboid
            return TamerOp.FiltrationSpec(kind=:rhomboid, max_dim=max_dim, vertex_values=pvals)
        end
    elseif family_case in (:graph_lower_star, :graph_centrality, :graph_geodesic, :graph_function_geodesic_bifiltration,
                           :graph_weight_threshold_graph, :edge_weighted, :graph_core,
                           :clique_lower_star, :graph_weight_threshold_clique)
        gvals = _graph_values(data)
        if family_case === :graph_lower_star
            return TamerOp.FiltrationSpec(kind=:graph_lower_star, vertex_values=gvals, simplex_agg=:max)
        elseif family_case === :graph_centrality
            return TamerOp.FiltrationSpec(kind=:graph_centrality, centrality=:degree, lift=:lower_star)
        elseif family_case === :graph_geodesic
            return TamerOp.FiltrationSpec(kind=:graph_geodesic, sources=[1], metric=:hop, lift=:lower_star)
        elseif family_case === :graph_function_geodesic_bifiltration
            return TamerOp.FiltrationSpec(kind=:graph_function_geodesic_bifiltration, sources=[1], metric=:hop,
                                          vertex_values=gvals, lift=:lower_star, simplex_agg=:max)
        elseif family_case === :graph_weight_threshold_graph
            return TamerOp.FiltrationSpec(kind=:graph_weight_threshold, lift=:graph, edge_weights=DT.edge_weights(data))
        elseif family_case === :edge_weighted
            return TamerOp.FiltrationSpec(kind=:edge_weighted, edge_weights=DT.edge_weights(data))
        elseif family_case === :graph_core
            return TamerOp.FiltrationSpec(kind=:core, vertex_values=gvals)
        elseif family_case === :clique_lower_star
            return TamerOp.FiltrationSpec(kind=:clique_lower_star, max_dim=2, vertex_values=gvals, simplex_agg=:max)
        elseif family_case === :graph_weight_threshold_clique
            return TamerOp.FiltrationSpec(kind=:graph_weight_threshold, lift=:clique, max_dim=2, edge_weights=DT.edge_weights(data))
        end
    elseif family_case in (:lower_star, :cubical, :image_distance_bifiltration)
        if family_case === :lower_star
            return TamerOp.FiltrationSpec(kind=:lower_star)
        elseif family_case === :cubical
            return TamerOp.FiltrationSpec(kind=:cubical)
        elseif family_case === :image_distance_bifiltration
            mask = getfield(data, :data) .> 0.5
            return TamerOp.FiltrationSpec(kind=:image_distance_bifiltration, mask=mask)
        end
    elseif family_case === :wing_vein_bifiltration
        return TamerOp.FiltrationSpec(kind=:wing_vein_bifiltration, grid=_wing_grid_for_size(Int(job["requested_size"])), bbox=_graph_or_embedded_bbox(data))
    end
    error("Unsupported family case $(family_case)")
end

function build_encoding_result(data, source_case::AbstractDict{String,<:Any}, job::AbstractDict{String,<:Any}; cache=:auto)
    source_kind = Symbol(source_case["source_kind"])
    if source_kind in (:point_cloud, :graph, :image, :embedded_graph)
        degree = _degree_symbol_to_int(Symbol(job["degree_label"]))
        spec = build_filtration_spec(data, job)
        if degree === nothing
            return TamerOp.encode(data, spec; cache=cache, stage=:encoding_result)
        end
        return TamerOp.encode(data, spec; degree=degree, cache=cache, stage=:encoding_result)
    elseif source_kind === :pl_fringe || source_kind === :flange
        enc_opts = _presentation_encoding_options(Int(source_case["requested_size"]))
        return TamerOp.encode(data, enc_opts; cache=cache)
    end
    error("Unsupported source kind $(source_kind)")
end

function _encoding_bbox(enc)
    axes = TamerOp.encoding_axes(enc)
    if axes !== nothing
        lo = Float64[Float64(axis[1]) for axis in axes]
        hi = Float64[Float64(axis[end]) for axis in axes]
        return lo, hi
    end
    reps = TamerOp.encoding_representatives(enc)
    reps === nothing && error("Cannot infer encoding bounding box: representatives unavailable")
    first_rep = Float64[Float64(x) for x in reps[1]]
    lo = copy(first_rep)
    hi = copy(first_rep)
    for rep in reps
        vals = Float64[Float64(x) for x in rep]
        for d in eachindex(vals)
            lo[d] = min(lo[d], vals[d])
            hi[d] = max(hi[d], vals[d])
        end
    end
    return lo, hi
end

function slice_query_defaults(enc)
    lo, hi = _encoding_bbox(enc)
    n = length(lo)
    if n == 1
        return (directions=[[1.0]], offsets=[copy(lo)], tgrid=collect(range(0.0, stop=max(1.0, hi[1] - lo[1]), length=32)), kmax=3)
    elseif n == 2
        mid = [(lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2]
        span = max(hi[1] - lo[1], hi[2] - lo[2], 1.0)
        return (
            directions=[[1.0, 1.0], [1.0, 0.5], [0.5, 1.0]],
            offsets=[copy(lo), mid],
            tgrid=collect(range(0.0, stop=span, length=40)),
            kmax=3,
        )
    end
    error("slice/mp defaults only implemented for 1D and 2D encodings")
end

function _supports_exact_path(enc, invariant_kind::Symbol)
    opts = OPT.InvariantOptions()
    if invariant_kind === :rank_signed_measure
        return IC._supports_exact_rank_signed_measure(enc; opts=opts)
    elseif invariant_kind === :restricted_hilbert
        return IC._supports_exact_restricted_hilbert(enc; opts=opts)
    elseif invariant_kind === :rectangle_signed_barcode
        return IC._supports_exact_rectangle_signed_barcode(enc; opts=opts)
    elseif invariant_kind === :euler_signed_measure
        return IC._supports_exact_euler_signed_measure(enc; opts=opts)
    elseif invariant_kind === :slice_barcodes || invariant_kind === :mp_landscape
        q = slice_query_defaults(enc)
        return IC._supports_exact_slice_barcodes(enc; opts=opts,
                                                 directions=q.directions,
                                                 offsets=q.offsets,
                                                 normalize_dirs=:none,
                                                 values=nothing,
                                                 packed=false)
    end
    return false
end

function _drop_degenerate_slice_intervals(result::SI.SliceBarcodesResult)
    bars = SI.slice_barcodes(result)
    cleaned = similar(bars)
    for I in eachindex(bars)
        src = bars[I]
        dst = typeof(src)()
        for ((birth, death), mult) in pairs(src)
            birth < death || continue
            iszero(mult) && continue
            dst[(Float64(birth), Float64(death))] = mult
        end
        cleaned[I] = dst
    end
    return SI.SliceBarcodesResult(cleaned, SI.slice_weights(result), SI.slice_directions(result), SI.slice_offsets(result))
end

function backend_label(enc, invariant_kind::Symbol, degree_label::Symbol)
    exact = _supports_exact_path(enc, invariant_kind)
    if exact
        base = degree_label === :na ? "exact_lazy" : "exact_" * lowercase(String(degree_label)) * "_lazy"
        if invariant_kind === :mp_landscape
            return "workflow_from_" * base
        end
        return base
    end
    if invariant_kind in (:mp_landscape, :euler_surface, :rank_invariant, :mpp_decomposition, :mpp_image)
        return "workflow_fallback"
    end
    return "generic_module"
end

function run_invariant_on_encoding(enc, invariant_kind::Symbol)
    opts = OPT.InvariantOptions()
    if invariant_kind === :euler_signed_measure
        return TamerOp.euler_signed_measure(enc; opts=opts, cache=:auto)
    elseif invariant_kind === :euler_surface
        return TamerOp.euler_surface(enc; opts=opts, cache=:auto)
    elseif invariant_kind === :rank_signed_measure
        return TamerOp.rank_signed_measure(enc; opts=opts, cache=:auto)
    elseif invariant_kind === :rank_invariant
        return TamerOp.rank_invariant(enc; opts=opts, cache=:auto)
    elseif invariant_kind === :restricted_hilbert
        return TamerOp.restricted_hilbert(enc; opts=opts, cache=:auto)
    elseif invariant_kind === :rectangle_signed_barcode
        return TamerOp.rectangle_signed_barcode(enc; opts=opts, cache=:auto)
    elseif invariant_kind === :slice_barcodes
        q = slice_query_defaults(enc)
        return TamerOp.slice_barcodes(enc; opts=opts, cache=:auto,
                                      directions=q.directions,
                                      offsets=q.offsets,
                                      threads=false)
    elseif invariant_kind === :mp_landscape
        q = slice_query_defaults(enc)
        bars = TamerOp.slice_barcodes(enc; opts=opts, cache=:auto,
                                      directions=q.directions,
                                      offsets=q.offsets,
                                      threads=false)
        bars = _drop_degenerate_slice_intervals(bars)
        return IM._mp_landscape_from_slice_barcodes(
            bars;
            kmax=q.kmax,
            tgrid=q.tgrid,
            normalize_weights=false,
            threads=false,
        )
    elseif invariant_kind === :mpp_decomposition
        return TamerOp.mpp_decomposition(enc; opts=opts, cache=:auto, N=8, delta=:auto, q=1.0, tie_break=:center)
    elseif invariant_kind === :mpp_image
        return TamerOp.mpp_image(enc; opts=opts, cache=:auto, N=8, delta=:auto, q=1.0, tie_break=:center, resolution=24, sigma=0.1)
    end
    error("Unsupported invariant kind $(invariant_kind)")
end

function _summary_digest(payload)
    return bytes2hex(SHA.sha1(repr(payload)))
end

function _measure_summary(pm::SM.PointSignedMeasure)
    d = TamerOp.describe(pm; nlargest=3)
    return (
        output_kind="point_signed_measure",
        output_term_count=Int(d.nterms),
        output_abs_mass=Float64(d.total_variation),
        output_shape=repr(d.axis_lengths),
        output_digest=_summary_digest(d),
        output_signature=repr(d),
    )
end

function _rect_barcode_summary(sb::SM.RectSignedBarcode)
    d = TamerOp.describe(sb; nlargest=3)
    return (
        output_kind="rectangle_signed_barcode",
        output_term_count=Int(d.nterms),
        output_abs_mass=Float64(d.total_variation),
        output_shape=repr(d.axis_lengths),
        output_digest=_summary_digest(d),
        output_signature=repr(d),
    )
end

function _rank_invariant_summary(res)
    d = TamerOp.describe(res)
    sample = sort!(collect(pairs(res)); by=first)
    sample = sample[1:min(end, 8)]
    payload = (; describe=d, sample=sample)
    abs_mass = sum(abs, values(res))
    return (
        output_kind="rank_invariant",
        output_term_count=Int(INV.nentries(res)),
        output_abs_mass=Float64(abs_mass),
        output_shape=string("nentries=", INV.nentries(res)),
        output_digest=_summary_digest(payload),
        output_signature=repr(payload),
    )
end

function _vector_summary(kind::String, v::AbstractVector)
    sample_head = collect(v[1:min(end, 8)])
    sample_tail = collect(v[max(1, end - 7):end])
    payload = (
        kind=kind,
        length=length(v),
        sum=sum(Float64, v),
        abs_sum=sum(abs, Float64.(v)),
        minimum=isempty(v) ? nothing : minimum(v),
        maximum=isempty(v) ? nothing : maximum(v),
        head=sample_head,
        tail=sample_tail,
    )
    return (
        output_kind=kind,
        output_term_count=count(!iszero, v),
        output_abs_mass=sum(abs, Float64.(v)),
        output_shape=string(length(v)),
        output_digest=_summary_digest(payload),
        output_signature=repr(payload),
    )
end

function _matrix_summary(kind::String, A::AbstractMatrix)
    flat = vec(A)
    sample_head = collect(flat[1:min(end, 8)])
    sample_tail = collect(flat[max(1, end - 7):end])
    payload = (
        kind=kind,
        size=size(A),
        sum=sum(Float64, A),
        abs_sum=sum(abs, Float64.(A)),
        minimum=isempty(flat) ? nothing : minimum(flat),
        maximum=isempty(flat) ? nothing : maximum(flat),
        head=sample_head,
        tail=sample_tail,
    )
    return (
        output_kind=kind,
        output_term_count=count(!iszero, flat),
        output_abs_mass=sum(abs, Float64.(flat)),
        output_shape=repr(size(A)),
        output_digest=_summary_digest(payload),
        output_signature=repr(payload),
    )
end

function _slice_barcodes_summary(result)
    bars = SI.slice_barcodes(result)
    ndirs, noffs = size(bars)
    pair_counts = Int[]
    total_mass = 0.0
    for i in 1:ndirs, j in 1:noffs
        count_ij = 0
        for (_, mult) in pairs(bars[i, j])
            count_ij += Int(mult)
            total_mass += abs(Float64(mult))
        end
        push!(pair_counts, count_ij)
    end
    payload = (
        kind=:slice_barcodes,
        ndirections=ndirs,
        noffsets=noffs,
        total_pairs=sum(pair_counts),
        pair_counts=pair_counts[1:min(end, 12)],
    )
    return (
        output_kind="slice_barcodes",
        output_term_count=sum(pair_counts),
        output_abs_mass=total_mass,
        output_shape=repr(size(bars)),
        output_digest=_summary_digest(payload),
        output_signature=repr(payload),
    )
end

function _mp_landscape_summary(result)
    d = TamerOp.describe(result)
    vals = IM.landscape_values(result)
    payload = (
        describe=d,
        sum=sum(vals),
        abs_sum=sum(abs, vals),
        maxval=isempty(vals) ? 0.0 : maximum(vals),
    )
    return (
        output_kind="mp_landscape",
        output_term_count=count(!iszero, vals),
        output_abs_mass=sum(abs, vals),
        output_shape=repr(size(vals)),
        output_digest=_summary_digest(payload),
        output_signature=repr(payload),
    )
end

function _mpp_decomposition_summary(result)
    d = TamerOp.describe(result)
    payload = (
        describe=d,
        first_line=isempty(result.lines) ? nothing : result.lines[1],
        first_weight=isempty(result.weights) ? nothing : result.weights[1],
    )
    return (
        output_kind="mpp_decomposition",
        output_term_count=Int(d.total_segments),
        output_abs_mass=Float64(d.weight_sum),
        output_shape=string("nsummands=", d.nsummands),
        output_digest=_summary_digest(payload),
        output_signature=repr(payload),
    )
end

function _mpp_image_summary(result)
    d = TamerOp.describe(result)
    payload = (
        describe=d,
        image_sum=sum(result.img),
        image_abs_sum=sum(abs, result.img),
        minval=isempty(result.img) ? 0.0 : minimum(result.img),
        maxval=isempty(result.img) ? 0.0 : maximum(result.img),
    )
    return (
        output_kind="mpp_image",
        output_term_count=count(!iszero, result.img),
        output_abs_mass=sum(abs, result.img),
        output_shape=repr(size(result.img)),
        output_digest=_summary_digest(payload),
        output_signature=repr(payload),
    )
end

function summarize_output(result, invariant_kind::Symbol)
    if result isa SM.PointSignedMeasure
        return _measure_summary(result)
    elseif result isa SM.RectSignedBarcode
        return _rect_barcode_summary(result)
    elseif invariant_kind === :rank_invariant
        return _rank_invariant_summary(result)
    elseif invariant_kind === :restricted_hilbert
        return _vector_summary("restricted_hilbert", result)
    elseif invariant_kind === :euler_surface
        return _matrix_summary("euler_surface", result)
    elseif invariant_kind === :slice_barcodes
        return _slice_barcodes_summary(result)
    elseif invariant_kind === :mp_landscape
        return _mp_landscape_summary(result)
    elseif invariant_kind === :mpp_decomposition
        return _mpp_decomposition_summary(result)
    elseif invariant_kind === :mpp_image
        return _mpp_image_summary(result)
    elseif result isa AbstractVector
        return _vector_summary(String(invariant_kind), result)
    elseif result isa AbstractMatrix
        return _matrix_summary(String(invariant_kind), result)
    end
    payload = (kind=String(invariant_kind), type=string(typeof(result)), summary=repr(TamerOp.describe(result)))
    return (
        output_kind=String(invariant_kind),
        output_term_count=0,
        output_abs_mass=0.0,
        output_shape=string(typeof(result)),
        output_digest=_summary_digest(payload),
        output_signature=repr(payload),
    )
end

function source_complexity_summary(data)
    if data isa DT.PointCloud
        return (source_size=DT.npoints(data), ambient_dim=DT.ambient_dim(data), source_aux=0, source_shape=string(DT.npoints(data), "x", DT.ambient_dim(data)))
    elseif data isa DT.GraphData
        return (source_size=DT.nvertices(data), ambient_dim=DT.ambient_dim(data), source_aux=DT.nedges(data), source_shape=string(DT.nvertices(data), "v/", DT.nedges(data), "e"))
    elseif data isa DT.ImageNd
        shp = size(getfield(data, :data))
        return (source_size=prod(shp), ambient_dim=length(shp), source_aux=0, source_shape=repr(shp))
    elseif data isa DT.EmbeddedPlanarGraph2D
        return (source_size=DT.nvertices(data), ambient_dim=2, source_aux=DT.nedges(data), source_shape=string(DT.nvertices(data), "v/", DT.nedges(data), "e"))
    elseif data isa PLP.PLFringe
        return (source_size=PLP.nupsets(data) + PLP.ndownsets(data), ambient_dim=PLP.ambient_dim(data), source_aux=PLP.nupsets(data) * PLP.ndownsets(data), source_shape=string("ups=", PLP.nupsets(data), ",downs=", PLP.ndownsets(data)))
    elseif data isa FZ.Flange
        return (source_size=length(data.flats) + length(data.injectives), ambient_dim=FZ.ambient_dim(data), source_aux=length(data.flats) * length(data.injectives), source_shape=string("flats=", length(data.flats), ",injectives=", length(data.injectives)))
    end
    return (source_size=0, ambient_dim=0, source_aux=0, source_shape=string(typeof(data)))
end

function encoding_complexity_summary(enc)
    reps = TamerOp.encoding_representatives(enc)
    axes = TamerOp.encoding_axes(enc)
    q = TamerOp.encoding_poset(enc)
    return (
        encoding_vertices=FF.nvertices(q),
        encoding_nparams=axes === nothing ? (reps === nothing ? 0 : length(reps[1])) : length(axes),
        encoding_axis_lengths=axes === nothing ? "" : repr(map(length, axes)),
    )
end

function benchmark_notes(enc, invariant_kind::Symbol, degree_label::Symbol)
    label = backend_label(enc, invariant_kind, degree_label)
    return string("backend=", enc.backend, ";path=", label)
end

source_case_lookup(manifest::AbstractDict{String,<:Any}) =
    Dict(String(case["id"]) => case for case in manifest["source_cases"])

function filter_source_cases(source_cases::AbstractVector;
                             source_case_ids=nothing,
                             source_kinds=nothing,
                             limit::Union{Nothing,Int}=nothing)
    out = collect(source_cases)
    if source_case_ids !== nothing
        wanted = Set(String.(source_case_ids))
        out = [case for case in out if String(case["id"]) in wanted]
    end
    if source_kinds !== nothing
        wanted = Set(Symbol.(source_kinds))
        out = [case for case in out if Symbol(case["source_kind"]) in wanted]
    end
    limit === nothing || (out = out[1:min(end, limit)])
    return out
end

function filter_jobs(jobs::AbstractVector,
                     source_cases_by_id::AbstractDict{String,<:Any};
                     job_ids=nothing,
                     source_case_ids=nothing,
                     families=nothing,
                     invariants=nothing,
                     degrees=nothing,
                     source_kinds=nothing,
                     limit::Union{Nothing,Int}=nothing)
    out = collect(jobs)
    if job_ids !== nothing
        wanted = Set(String.(job_ids))
        out = [job for job in out if String(job["id"]) in wanted]
    end
    if source_case_ids !== nothing
        wanted = Set(String.(source_case_ids))
        out = [job for job in out if String(job["source_case_id"]) in wanted]
    end
    if families !== nothing
        wanted = Set(Symbol.(families))
        out = [job for job in out if Symbol(job["family"]) in wanted || Symbol(job["family_case"]) in wanted]
    end
    if invariants !== nothing
        wanted = Set(Symbol.(invariants))
        out = [job for job in out if Symbol(job["invariant_kind"]) in wanted]
    end
    if degrees !== nothing
        wanted = Set(Symbol.(degrees))
        out = [job for job in out if Symbol(job["degree_label"]) in wanted]
    end
    if source_kinds !== nothing
        wanted = Set(Symbol.(source_kinds))
        out = [
            job for job in out
            if Symbol(source_cases_by_id[String(job["source_case_id"])]["source_kind"]) in wanted
        ]
    end
    limit === nothing || (out = out[1:min(end, limit)])
    return out
end

function build_catalog(profile::Symbol; fixtures_dir::String="fixtures")
    haskey(_SIZE_LADDERS, profile) || error("Unknown thesis macro size profile $(profile)")
    available = Set(TamerOp.available_filtrations())
    missing = setdiff(_BUILTIN_FILTRATION_FAMILIES, available)
    isempty(missing) || error("Harness expects filtration families missing from available_filtrations(): $(collect(missing))")
    raw_covered = Set(fc.family for fc in _RAW_FAMILY_CASES)
    missing_cov = setdiff(_BUILTIN_FILTRATION_FAMILIES, raw_covered)
    isempty(missing_cov) || error("Harness raw family metadata is missing built-in families: $(collect(missing_cov))")

    source_cases = Vector{Dict{String,Any}}()
    source_index = Dict{Tuple{Symbol,Int},String}()
    seed0 = 0x5A17_2604
    for (template_name, meta) in sort(collect(_SOURCE_TEMPLATES); by=first)
        ladders = _SIZE_LADDERS[profile][meta.size_group]
        sizes_all = sort!(unique(vcat(values(ladders)...)))
        for (idx, requested_size) in enumerate(sizes_all)
            id = string(template_name, "__n", requested_size)
            source_case = Dict{String,Any}(
                "id" => id,
                "source_template" => String(template_name),
                "source_kind" => String(meta.source_kind),
                "source_variant" => String(meta.variant),
                "size_group" => String(meta.size_group),
                "size_tier" => string("n", requested_size),
                "requested_size" => requested_size,
                "seed" => Int(seed0 + length(source_cases) + idx),
                "fixture_relpath" => joinpath(fixtures_dir, id * _fixture_extension(meta.source_kind)),
            )
            push!(source_cases, source_case)
            source_index[(template_name, requested_size)] = id
        end
    end

    jobs = Vector{Dict{String,Any}}()
    family_cases = vcat(_RAW_FAMILY_CASES, _PRESENTATION_CASES)
    for fc in family_cases
        invariants = fc.nparams == 2 ? _PRIMARY_INVARIANTS : [inv for inv in _PRIMARY_INVARIANTS if !(inv in _MPP_INVARIANTS)]
        for template_name in fc.source_templates
            meta = _SOURCE_TEMPLATES[template_name]
            ladders = _SIZE_LADDERS[profile][meta.size_group]
            degrees = fc.family in _PRESENTATION_FAMILIES ? [:na] : vcat([:na], fc.degree_labels)
            for degree_label in degrees
                haskey(ladders, degree_label) || continue
                for requested_size in ladders[degree_label]
                    source_id = source_index[(template_name, requested_size)]
                    for invariant_kind in invariants
                        if fc.family in _PRESENTATION_FAMILIES
                            degree_label === :na || continue
                            degree_int = ""
                        elseif invariant_kind in _DEGREE_AGNOSTIC_INVARIANTS
                            degree_label === :na || continue
                            degree_int = ""
                        else
                            degree_label === :na && continue
                            degree_int = _degree_symbol_to_int(degree_label)
                        end
                        job_id = join([
                            source_id,
                            String(fc.family_case),
                            String(invariant_kind),
                            degree_label === :na ? "na" : lowercase(String(degree_label)),
                        ], "__")
                        push!(jobs, Dict{String,Any}(
                            "id" => job_id,
                            "source_case_id" => source_id,
                            "family" => String(fc.family),
                            "family_case" => String(fc.family_case),
                            "invariant_kind" => String(invariant_kind),
                            "degree_label" => String(degree_label),
                            "degree" => degree_int,
                            "nparams" => fc.nparams,
                            "requested_size" => requested_size,
                            "size_tier" => string(lowercase(String(degree_label)), "_", requested_size),
                        ))
                    end
                end
            end
        end
    end

    return Dict(
        "harness" => "thesis_macro_v1",
        "profile" => String(profile),
        "generated_utc" => string(Dates.now(Dates.UTC)),
        "fixtures_dir" => fixtures_dir,
        "primary_invariants" => String.(collect(_PRIMARY_INVARIANTS)),
        "source_cases" => source_cases,
        "jobs" => jobs,
    )
end
