using Test
using SparseArrays
using Random
using JSON3

const IR = TamerOp.IndicatorResolutions
const DI = TamerOp.DataIngestion
const IC = TamerOp.InvariantCore
const DFI = TamerOp.DataFileIO
const SER = TamerOp.Serialization
const FF = TamerOp.FiniteFringe
const FZ = TamerOp.FlangeZn
const PLB = TamerOp.PLBackend
const MC = TamerOp.ModuleComplexes
const MD = TamerOp.Modules
const Inv = TamerOp.Invariants
const FL = TamerOp.FieldLinAlg
const AC = TamerOp.AbelianCategories

if !isdefined(@__MODULE__, :TestTriGradeFiltration)
struct TestTriGradeFiltration{P<:NamedTuple} <: DI.AbstractFiltration
    params::P
end

TestTriGradeFiltration(; shift::Real=0.0,
                       scale::Real=1.0,
                       construction::OPT.ConstructionOptions=OPT.ConstructionOptions()) =
    TestTriGradeFiltration((;
        shift=Float64(shift),
        scale=Float64(scale),
        construction,
    ))

DI.filtration_kind(::Type{<:TestTriGradeFiltration}) = :test_trigrade
DI.filtration_arity(::TestTriGradeFiltration, _data=nothing) = 3

function _test_trigrade_builder(data::DT.PointCloud,
                                filtration::TestTriGradeFiltration;
                                cache::Union{Nothing,CM.EncodingCache}=nothing)
    points = data.points
    n = length(points)
    shift = Float64(get(filtration.params, :shift, 0.0))
    scale = Float64(get(filtration.params, :scale, 1.0))
    grades = Vector{NTuple{3,Float64}}(undef, n)
    @inbounds for i in 1:n
        x = Float64(points[i][1])
        grades[i] = (x + shift, scale * x * x, 1.0)
    end
    cells = [collect(1:n)]
    G = DT.GradedComplex(cells, SparseMatrixCSC{Int,Int}[], grades)
    ax1 = unique(Float64[g[1] for g in grades]); sort!(ax1)
    ax2 = unique(Float64[g[2] for g in grades]); sort!(ax2)
    ax3 = unique(Float64[g[3] for g in grades]); sort!(ax3)
    return G, (ax1, ax2, ax3), (1, 1, 1)
end

const _TEST_TRIGRADE_SCHEMA = (
    defaults=(shift=0.0, scale=1.0),
    types=(shift=Real, scale=Real),
    checks=(scale=((x)->x > 0.0, "test_trigrade expects `scale > 0`."),),
)
end

if !isdefined(@__MODULE__, :BadUXFiltration)
struct BadUXFiltration <: DI.AbstractFiltration end
end

@inline _enc_module(enc::RES.EncodingResult) = DI.materialize_module(enc.M)
@inline _enc_dims(enc::RES.EncodingResult) = DI.module_dims(enc.M)
@inline _canon_simplex_tree(st::DI.SimplexTreeMulti) = sort([
    (Tuple(collect(DI.simplex_vertices(st, i))), Tuple(collect(DI.simplex_grades(st, i))))
    for i in 1:DI.simplex_count(st)
])

with_fields(FIELDS_FULL) do field
K = CM.coeff_type(field)
@inline c(x) = CM.coerce(field, x)

@testset "DataTypes packed storage constructors" begin
    empty_complex = DT.GradedComplex([Int[],Int[]], [spzeros(Int,0,0)], NTuple{2,Float64}[])
    @test DT.parameter_dim(empty_complex) == 2
    @test DT.cell_counts(empty_complex) == [0,0]
    @test DT.check_graded_complex(empty_complex;throw=true).valid
    empty_tree = DI._simplex_tree_multi_from_complex(empty_complex)
    @test DT.cell_counts(empty_tree) == [0,0]
    @test DT.parameter_dim(empty_tree) == 2
    @test DI._graded_complex_from_simplex_tree(empty_tree).boundaries == empty_complex.boundaries
    @test_throws ArgumentError DT.GradedComplex([Int[]], SparseMatrixCSC{Int,Int}[], Vector{Float64}[])
    @test_throws ArgumentError DT.GradedComplex([Int[]], SparseMatrixCSC{Int,Int}[], Tuple[])
    @test_throws ArgumentError DT.GradedComplex([Int[],Int[]], SparseMatrixCSC{Int,Int}[], NTuple{2,Float64}[])
    @test_throws ArgumentError DT.GradedComplex([Int[],Int[]], [spzeros(Int,1,1)], NTuple{2,Float64}[])
    @test_throws ArgumentError DT.GradedComplex([Int[]], SparseMatrixCSC{Int,Int}[], Tuple{}[])
    for offsets in (Int[1,2,1], Int[2,1])
        @test_throws ArgumentError DT.SimplexTreeMulti(Int[1],Int[],Int[],offsets,Int[1],NTuple{2,Float64}[])
    end
    pts = [0.0 1.0; 2.0 3.0]
    pc = DT.PointCloud(pts)
    @test DT.point_matrix(pc) === pts
    @test length(pc.points) == 2
    @test pc.points[2] == [2.0, 3.0]

    coords = [0.0 0.0; 1.0 0.5; 2.0 1.0]
    edge_u = [1, 2]
    edge_v = [2, 3]
    weights = [1.0, 2.0]
    g = DT.GraphData(3, edge_u, edge_v; coords=coords, weights=weights, copy=false)
    @test DT.coord_matrix(g) === coords
    @test DT.edge_columns(g)[1] === edge_u
    @test DT.edge_columns(g)[2] === edge_v
    @test collect(g.edges) == [(1, 2), (2, 3)]
    @test g.coords[2] == [1.0, 0.5]

    g2 = DT.GraphData(3, [(1, 2), (2, 3)]; coords=coords, weights=weights, copy=false)
    @test DT.coord_matrix(g2) === coords
    @test collect(g2.edges) == [(1, 2), (2, 3)]

    verts = [0.0 0.0; 1.0 0.0; 1.0 1.0]
    poly_points = [0.0 0.0; 1.0 0.0]
    poly_offsets = [1, 3]
    emb = DT.EmbeddedPlanarGraph2D(
        verts,
        [1, 2],
        [2, 3];
        polyline_offsets=poly_offsets,
        polyline_points=poly_points,
        bbox=(0.0, 1.0, 0.0, 1.0),
        copy=false,
    )
    @test DT.vertex_matrix(emb) === verts
    @test DT.edge_columns(emb)[1] == [1, 2]
    @test DT.edge_columns(emb)[2] == [2, 3]
    @test emb.vertices[3] == [1.0, 1.0]
    @test collect(emb.edges) == [(1, 2), (2, 3)]
    @test emb.polylines !== nothing
    @test emb.polylines[1][1] == [0.0, 0.0]
    @test emb.polylines[1][2] == [1.0, 0.0]

    cells = [Int[10, 11], Int[20]]
    boundaries = [spzeros(Int, 2, 1)]
    grades = [(0.0,), (1.0,), (2.0,)]
    gc = DT.GradedComplex(cells, boundaries, grades)
    @test getfield(gc, :cell_ids) == [10, 11, 20]
    @test getfield(gc, :dim_offsets) == [1, 3, 4]
    @test collect(gc.cells_by_dim[1]) == [10, 11]
    @test collect(gc.cells_by_dim[2]) == [20]
    @test collect(gc.cell_dims) == [0, 0, 1]
    gc2 = DT.GradedComplex(gc.cells_by_dim, gc.boundaries, gc.grades; cell_dims=gc.cell_dims)
    @test gc2.grades == gc.grades
    @test collect(gc2.cell_dims) == [0, 0, 1]

    multi_grades = [[(0.0, 0.0)], [(1.0, 1.0)], [(2.0, 2.0), (2.5, 3.0)]]
    mgc = DT.MultiCriticalGradedComplex(cells, boundaries, multi_grades)
    @test getfield(mgc, :cell_ids) == [10, 11, 20]
    @test getfield(mgc, :dim_offsets) == [1, 3, 4]
    @test getfield(mgc, :grade_offsets) == [1, 2, 3, 5]
    @test collect(mgc.grades[3]) == [(2.0, 2.0), (2.5, 3.0)]
    @test collect(mgc.cell_dims) == [0, 0, 1]
    mgc2 = DT.MultiCriticalGradedComplex(mgc.cells_by_dim, mgc.boundaries, mgc.grades; cell_dims=mgc.cell_dims)
    @test collect(mgc2.grades[3]) == [(2.0, 2.0), (2.5, 3.0)]

    dup_multi = [[(0.0, 0.0)], [(1.0, 1.0)], [(2.0, 2.0), (2.0, 2.0)]]
    @test_throws ErrorException DT.MultiCriticalGradedComplex(cells, boundaries, dup_multi)
end

@testset "Data pipeline: packed-matrix brute-force point-cloud builder parity" begin
    pts = TamerOp.PointCloud([
        [0.0, 0.0],
        [0.5, 0.1],
        [1.0, 0.0],
        [1.6, 0.4],
        [2.0, 0.1],
        [2.6, 0.5],
    ])
    rows = pts.points
    mat = DT.point_matrix(pts)

    e_knn_rows, d_knn_rows, k_knn_rows = DI._point_cloud_knn_graph(rows, 2; backend=:bruteforce, approx_candidates=0)
    e_knn_mat, d_knn_mat, k_knn_mat = DI._point_cloud_knn_graph(mat, 2; backend=:bruteforce, approx_candidates=0)
    @test e_knn_mat == e_knn_rows
    @test isapprox(d_knn_mat, d_knn_rows; atol=1e-12, rtol=1e-12)
    @test isapprox(k_knn_mat, k_knn_rows; atol=1e-12, rtol=1e-12)

    e_rad_rows, d_rad_rows = DI._point_cloud_radius_graph(rows, 0.8; backend=:bruteforce, approx_candidates=0)
    e_rad_mat, d_rad_mat = DI._point_cloud_radius_graph(mat, 0.8; backend=:bruteforce, approx_candidates=0)
    @test e_rad_mat == e_rad_rows
    @test isapprox(d_rad_mat, d_rad_rows; atol=1e-12, rtol=1e-12)

    e_idx_rows, d_idx_rows = DI._point_cloud_edges_within_radius_indexed(rows, [1, 3, 4, 6], 1.3)
    e_idx_mat, d_idx_mat = DI._point_cloud_edges_within_radius_indexed(mat, [1, 3, 4, 6], 1.3)
    @test e_idx_mat == e_idx_rows
    @test isapprox(d_idx_mat, d_idx_rows; atol=1e-12, rtol=1e-12)

    spec_knn = TamerOp.FiltrationSpec(
        kind=:rips,
        knn=2,
        nn_backend=:bruteforce,
        construction=OPT.ConstructionOptions(; sparsify=:knn, output_stage=:simplex_tree),
    )
    construction_knn = DI._construction_from_params(spec_knn.params)
    sparse_rows = DI._point_cloud_sparsify_edge_driven(rows, spec_knn, construction_knn)
    sparse_mat = DI._point_cloud_sparsify_edge_driven(mat, spec_knn, construction_knn)
    @test sparse_mat[1] == sparse_rows[1]
    @test isapprox(sparse_mat[2], sparse_rows[2]; atol=1e-12, rtol=1e-12)
    @test isapprox(sparse_mat[3], sparse_rows[3]; atol=1e-12, rtol=1e-12)

    spec_lm = TamerOp.FiltrationSpec(kind=:landmark_rips, radius=1.3, nn_backend=:bruteforce)
    ec = CM.EncodingCache()
    packed_rows = DI._landmark_radius_subgraph_cached(rows, [1, 3, 4, 6], 1.3, spec_lm; cache=ec)
    packed_mat = DI._landmark_radius_subgraph_cached(mat, [1, 3, 4, 6], 1.3, spec_lm; cache=ec)
    @test packed_mat === packed_rows
    @test packed_mat.edges == e_idx_mat
    @test isapprox(packed_mat.dists, d_idx_mat; atol=1e-12, rtol=1e-12)
end

@testset "Data pipeline: JSON round-trips" begin
    mktemp() do path, io
        close(io)
        data = TamerOp.PointCloud([[0.0], [1.0]])
        SER.save_dataset_json(path, data)
        obj = JSON3.read(read(path, String))
        @test obj["layout"] == SER._DATASET_COLUMN_LAYOUT
        @test haskey(obj, "points_flat")
        @test !haskey(obj, "points")
        @test Vector{Float64}(obj["points_flat"]) == collect(vec(DT.point_matrix(data)))
        data2 = SER.load_dataset_json(path)
        @test length(data2.points) == 2
        @test data2.points[2][1] == 1.0
    end

    mktempdir() do dir
        data = TamerOp.PointCloud([[0.0], [1.0], [2.0]])
        compact_path = joinpath(dir, "compact.json")
        pretty_path = joinpath(dir, "pretty.json")
        SER.save_dataset_json(compact_path, data; profile=:compact)
        SER.save_dataset_json(pretty_path, data; profile=:debug)
        @test filesize(compact_path) < filesize(pretty_path)
        @test SER.inspect_json(compact_path).profile_hint == :compact
        @test SER.inspect_json(pretty_path).profile_hint == :debug
        @test SER.load_dataset_json(compact_path; validation=:strict).points == SER.load_dataset_json(compact_path; validation=:trusted).points
    end

    mktemp() do path, io
        close(io)
        data = TamerOp.ImageNd([0.0 1.0; 2.0 3.0])
        SER.save_dataset_json(path, data)
        data2 = SER.load_dataset_json(path)
        @test size(data2.data) == (2, 2)
        @test data2.data[2, 2] == 3.0
    end

    mktemp() do path, io
        close(io)
        data = TamerOp.GraphData(3, [(1, 2), (2, 3)]; coords=[[0.0], [1.0], [2.0]], weights=[1.0, 2.0])
        SER.save_dataset_json(path, data)
        obj = JSON3.read(read(path, String))
        @test obj["layout"] == SER._DATASET_COLUMN_LAYOUT
        @test haskey(obj, "edges_u")
        @test haskey(obj, "edges_v")
        @test !haskey(obj, "edges")
        @test Vector{Float64}(obj["coords_flat"]) == collect(vec(DT.coord_matrix(data)))
        data2 = SER.load_dataset_json(path)
        @test data2.n == 3
        @test length(data2.edges) == 2
        @test data2.weights[2] == 2.0
    end

    mktemp() do path, io
        close(io)
        write(path, "{\"kind\":\"PointCloud\",\"points\":[[0.0,1.0],[2.0,3.0]]}")
        @test_throws ErrorException SER.load_dataset_json(path)
    end

    mktemp() do path, io
        close(io)
        write(path, "{\"kind\":\"PointCloud\",\"layout\":\"columnar_v1\",\"n\":2,\"d\":2,\"points_flat\":[0.0,1.0,2.0,3.0]}")
        @test_throws ErrorException SER.load_dataset_json(path)
    end

    mktemp() do path, io
        close(io)
        write(path, "{\"kind\":\"GraphData\",\"n\":3,\"edges\":[[1,2],[2,3]],\"coords\":null,\"weights\":null}")
        @test_throws ErrorException SER.load_dataset_json(path)
    end

    mktemp() do path, io
        close(io)
        write(path, "{\"kind\":\"GraphData\",\"layout\":\"columnar_v1\",\"n\":3,\"edges_u\":[1,2],\"edges_v\":[2,3],\"coords_dim\":null,\"coords_flat\":null,\"weights\":null}")
        @test_throws ErrorException SER.load_dataset_json(path)
    end

    mktemp() do path, io
        close(io)
        data = TamerOp.EmbeddedPlanarGraph2D([[0.0, 0.0], [1.0, 0.0]], [(1, 2)])
        SER.save_dataset_json(path, data)
        data2 = SER.load_dataset_json(path)
        @test length(data2.vertices) == 2
        @test data2.edges[1] == (1, 2)
    end

    mktemp() do path, io
        close(io)
        cells = [Int[1]]
        boundaries = SparseMatrixCSC{Int,Int}[]
        grades = [Float64[0.0]]
        data = TamerOp.GradedComplex(cells, boundaries, grades)
        SER.save_dataset_json(path, data)
        data2 = SER.load_dataset_json(path)
        @test length(data2.cells_by_dim) == 1
        @test length(data2.grades) == 1
    end

    mktemp() do path, io
        close(io)
        cells = [Int[1, 2], Int[1]]
        boundaries = [sparse([1, 2], [1, 1], [1, -1], 2, 1)]
        grades = [
            [Float64[0.0, 0.0]],
            [Float64[0.0, 0.0]],
            [Float64[1.0, 0.0], Float64[0.0, 1.0]],
        ]
        data = TamerOp.MultiCriticalGradedComplex(cells, boundaries, grades)
        SER.save_dataset_json(path, data)
        data2 = SER.load_dataset_json(path)
        @test data2 isa TamerOp.MultiCriticalGradedComplex
        @test length(data2.grades) == 3
        @test length(data2.grades[3]) == 2
    end

    mktemp() do path, io
        close(io)
        data = TamerOp.PointCloud([[0.0], [1.0]])
        spec = TamerOp.FiltrationSpec(
            kind=:rips,
            max_dim=1,
            construction=TamerOp.ConstructionOptions(; output_stage=:simplex_tree),
        )
        st = TamerOp.encode(data, spec; degree=0)
        @test st isa DI.SimplexTreeMulti
        SER.save_dataset_json(path, st)
        st2 = SER.load_dataset_json(path)
        @test st2 isa DI.SimplexTreeMulti
        @test DI.simplex_count(st2) == DI.simplex_count(st)
        @test collect(DI.simplex_vertices(st2, 3)) == [1, 2]
    end

    mktemp() do path, io
        close(io)
        data = TamerOp.PointCloud([[0.0], [1.0]])
        spec = TamerOp.FiltrationSpec(kind=:rips, max_dim=1, axes=([0.0, 1.0],))
        SER.save_pipeline_json(path, data, spec; degree=1)
        data2, spec2, degree2, popts = SER.load_pipeline_json(path)
        @test length(data2.points) == 2
        @test spec2.kind == :rips
        @test spec2.params[:max_dim] == 1
        @test degree2 == 1
        @test popts isa TamerOp.PipelineOptions
        @test popts.axes_policy == :encoding
        @test popts.poset_kind == :signature

        data3, spec3, degree3, popts3 = SER.load_pipeline_json(path; validation=:trusted)
        @test data3.points == data2.points
        @test spec3.kind == spec2.kind
        @test degree3 == degree2
        @test popts3 == popts
    end

    mktemp() do path, io
        close(io)
        data = TamerOp.PointCloud([[0.0], [1.0], [2.0]])
        spec = TamerOp.FiltrationSpec(
            kind=:rips,
            max_dim=1,
            construction=TamerOp.ConstructionOptions(;
                sparsify=:knn,
                collapse=:none,
                output_stage=:encoding_result,
                budget=(nothing, 12, 2_000_000),
            ),
        )
        SER.save_pipeline_json(path, data, spec; degree=0)
        _, spec2, degree2, _ = SER.load_pipeline_json(path)
        @test degree2 == 0
        @test haskey(spec2.params, :construction)
        cons = spec2.params[:construction]
        @test get(cons, "sparsify", get(cons, :sparsify, nothing)) == "knn" ||
              get(cons, "sparsify", get(cons, :sparsify, nothing)) == :knn
    end

    mktemp() do path, io
        close(io)
        data = TamerOp.PointCloud([[0.0], [1.0]])
        filt = DI.RipsFiltration(
            max_dim=1,
            construction=TamerOp.ConstructionOptions(
                ;
                sparsify=:knn,
                collapse=:none,
                output_stage=:encoding_result,
                budget=(max_simplices=nothing, max_edges=16, memory_budget_bytes=1_000_000),
            ),
        )
        popts = TamerOp.PipelineOptions(;
            orientation=(1,),
            axes_policy=:coarsen,
            axis_kind=:zn,
            eps=0.25,
            poset_kind=:signature,
            field=CM.F2(),
            max_axis_len=8,
        )
        SER.save_pipeline_json(path, data, filt; degree=1, pipeline_opts=popts)
        data2, spec2, degree2, popts2 = SER.load_pipeline_json(path)
        @test length(data2.points) == 2
        @test spec2.kind == :rips
        @test spec2.params[:max_dim] == 1
        @test haskey(spec2.params, :construction)
        cons = spec2.params[:construction]
        @test get(cons, "sparsify", get(cons, :sparsify, nothing)) == "knn" ||
              get(cons, "sparsify", get(cons, :sparsify, nothing)) == :knn
        b = get(cons, "budget", get(cons, :budget, nothing))
        @test get(b, "max_edges", get(b, :max_edges, nothing)) == 16
        @test degree2 == 1
        @test popts2 isa TamerOp.PipelineOptions
        @test popts2.orientation == (1,)
        @test popts2.axes_policy == :coarsen
        @test popts2.axis_kind == :zn
        @test popts2.eps == 0.25
        @test popts2.field == CM.F2()
        @test popts2.max_axis_len == 8
    end

    mktemp() do path, io
        close(io)
        data = TamerOp.GraphData(3, [(1, 2), (2, 3)])
        spec = TamerOp.FiltrationSpec(kind=:graph_lower_star, vertex_grades=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        SER.save_pipeline_json(path, data, spec; degree=0)
        _, spec2, degree2, popts = SER.load_pipeline_json(path)
        @test spec2.kind == :graph_lower_star
        @test length(spec2.params[:vertex_grades]) == 3
        @test degree2 == 0
        @test popts.axes_policy == :encoding
    end

    @testset "Serialization artifact summaries and validation" begin
        mktempdir() do dir
            dataset_path = joinpath(dir, "dataset.json")
            pipeline_path = joinpath(dir, "pipeline.json")

            data = TamerOp.PointCloud([[0.0], [1.0]])
            spec = TamerOp.FiltrationSpec(kind=:rips, max_dim=1, axes=([0.0, 1.0],))

            SER.save_dataset_json(dataset_path, data)
            dataset_info = SER.inspect_json(dataset_path)
            @test dataset_info isa SER.JSONArtifactSummary
            @test dataset_info.path == dataset_path
            @test SER.artifact_kind(dataset_info) == "PointCloud"
            @test SER.schema_version(dataset_info) === nothing
            @test SER.artifact_field(dataset_info) === nothing
            @test SER.artifact_poset_kind(dataset_info) === nothing
            @test SER.artifact_path(dataset_info) == dataset_path
            @test SER.artifact_profile_hint(dataset_info) == :compact
            @test SER.artifact_data_kind(dataset_info) == "PointCloud"
            @test SER.has_encoding_map(dataset_info) === nothing
            @test SER.has_dense_leq(dataset_info) === nothing
            @test SER.artifact_size_bytes(dataset_info) isa Integer
            @test SER.json_artifact_summary(dataset_path).kind == "PointCloud"
            @test SER.dataset_json_summary(dataset_path).kind == "PointCloud"
            @test TamerOp.describe(dataset_info).path == dataset_path
            @test occursin("JSONArtifactSummary", sprint(show, MIME"text/plain"(), dataset_info))

            dataset_report = SER.check_dataset_json(dataset_path)
            @test dataset_report.valid
            @test dataset_report.summary isa SER.JSONArtifactSummary
            @test dataset_report.artifact_kind == "PointCloud"
            dataset_validation = SER.serialization_validation_summary(dataset_report)
            @test dataset_validation isa SER.SerializationValidationSummary
            @test TamerOp.describe(dataset_validation).valid
            @test occursin("SerializationValidationSummary", sprint(show, MIME"text/plain"(), dataset_validation))

            SER.save_pipeline_json(pipeline_path, data, spec; degree=1)
            pipeline_info = SER.pipeline_json_summary(pipeline_path)
            @test pipeline_info isa SER.JSONArtifactSummary
            @test SER.artifact_kind(pipeline_info) == "PipelineJSON"
            @test SER.artifact_path(pipeline_info) == pipeline_path
            @test SER.artifact_data_kind(pipeline_info) == "PointCloud"
            @test pipeline_info.data_kind == "PointCloud"
            @test SER.check_pipeline_json(pipeline_path).valid

            meta = SER.feature_schema_header(format=:npz)
            meta_info = SER.feature_metadata_summary(meta)
            @test meta_info isa SER.JSONArtifactSummary
            @test SER.artifact_kind(meta_info) == "features"
            @test SER.schema_version(meta_info) == string(SER.TAMER_FEATURE_SCHEMA_VERSION)
            @test meta_info.format == "npz"
            @test SER.check_feature_metadata_json(meta).valid

            @test TOA.JSONArtifactSummary === SER.JSONArtifactSummary
            @test TOA.SerializationValidationSummary === SER.SerializationValidationSummary
            @test TOA.artifact_kind === SER.artifact_kind
            @test TOA.schema_version === SER.schema_version
            @test TOA.artifact_field === SER.artifact_field
            @test TOA.artifact_poset_kind === SER.artifact_poset_kind
            @test TOA.artifact_path === SER.artifact_path
            @test TOA.artifact_profile_hint === SER.artifact_profile_hint
            @test TOA.artifact_data_kind === SER.artifact_data_kind
            @test TOA.has_encoding_map === SER.has_encoding_map
            @test TOA.has_dense_leq === SER.has_dense_leq
            @test TOA.artifact_size_bytes === SER.artifact_size_bytes
            @test TOA.json_artifact_summary === SER.json_artifact_summary
            @test TOA.dataset_json_summary === SER.dataset_json_summary
            @test TOA.pipeline_json_summary === SER.pipeline_json_summary
            @test TOA.feature_metadata_summary === SER.feature_metadata_summary
            @test TOA.check_feature_metadata_json === SER.check_feature_metadata_json
            @test TOA.serialization_validation_summary === SER.serialization_validation_summary
            @test TOA.check_dataset_json === SER.check_dataset_json
            @test TOA.check_pipeline_json === SER.check_pipeline_json

            @test SER.check_json_save_profile(:compact).valid
            @test SER.check_json_save_profile(:compact).normalized == (pretty=false,)
            @test SER.check_json_save_profile(:debug).normalized == (pretty=true,)
            @test !SER.check_json_save_profile(:portable).valid
            @test !SER.check_json_save_profile(:bad).valid
            @test_throws ArgumentError SER.check_json_save_profile(:portable; throw=true)
            @test_throws ArgumentError SER.check_json_save_profile(:bad; throw=true)

            @test SER.check_encoding_save_profile(:compact).valid
            @test SER.check_encoding_save_profile(:compact).normalized == (include_pi=true, include_leq=:auto, pretty=false)
            @test !SER.check_encoding_save_profile(:bad).valid
            @test_throws ArgumentError SER.check_encoding_save_profile(:bad; throw=true)

            @test SER.check_include_leq_option(:auto).valid
            @test SER.check_include_leq_option(true).valid
            @test !SER.check_include_leq_option(:bad).valid
            @test_throws ArgumentError SER.check_include_leq_option(:bad; throw=true)

            @test SER.check_serialization_validation_mode(:strict).valid
            @test SER.check_serialization_validation_mode(:strict).normalized === true
            @test !SER.check_serialization_validation_mode(:bad).valid
            @test_throws ArgumentError SER.check_serialization_validation_mode(:bad; throw=true)

            @test SER.check_encoding_output_mode(:fringe).valid
            @test SER.check_encoding_output_mode(:fringe).normalized === :fringe
            @test !SER.check_encoding_output_mode(:bad).valid
            @test_throws ArgumentError SER.check_encoding_output_mode(:bad; throw=true)

            @test TOA.check_json_save_profile === SER.check_json_save_profile
            @test TOA.check_encoding_save_profile === SER.check_encoding_save_profile
            @test TOA.check_include_leq_option === SER.check_include_leq_option
            @test TOA.check_serialization_validation_mode === SER.check_serialization_validation_mode
            @test TOA.check_encoding_output_mode === SER.check_encoding_output_mode
        end

        mktemp() do path, io
            close(io)
            write(path, JSON3.write(Dict("metadata" => SER.feature_schema_header(format=:npz))))
            report = SER.check_feature_metadata_json(path)
            @test report.valid
            @test report.summary isa SER.JSONArtifactSummary
        end

        @test !SER.check_feature_metadata_json(Dict("kind" => "features")).valid
        @test_throws ArgumentError SER.check_feature_metadata_json(Dict("kind" => "features"); throw=true)

        mktemp() do path, io
            close(io)
            write(path, "{}")
            report = SER.check_dataset_json(path)
            @test !report.valid
            @test !isempty(report.issues)
            @test_throws ArgumentError SER.check_dataset_json(path; throw=true)
            @test_throws ArgumentError SER.dataset_json_summary(path)
        end

        mktemp() do path, io
            close(io)
            data = TamerOp.PointCloud([[0.0], [1.0]])
            spec = TamerOp.FiltrationSpec(kind=:rips, max_dim=1, axes=([0.0, 1.0],))
            SER.save_pipeline_json(path, data, spec; degree=1)
            raw = JSON3.read(read(path, String), Dict{String,Any})
            raw["schema_version"] = 0
            write(path, JSON3.write(raw))
            report = SER.check_pipeline_json(path)
            @test !report.valid
            @test any(issue -> occursin("schema_version", issue), report.issues)
            @test_throws ArgumentError SER.check_pipeline_json(path; throw=true)
        end

        mktemp() do path, io
            close(io)
            data = TamerOp.PointCloud([[0.0], [1.0]])
            spec = TamerOp.FiltrationSpec(kind=:rips, max_dim=1, axes=([0.0, 1.0],))
            SER.save_pipeline_json(path, data, spec; degree=1)
            obj = JSON3.read(read(path, String))
            hacked = Dict{String,Any}(String(k) => v for (k, v) in pairs(obj))
            pop!(hacked, "pipeline_options", nothing)
            hacked["schema_version"] = 0
            write(path, JSON3.write(hacked))
            @test_throws ErrorException SER.load_pipeline_json(path; validation=:strict)
            _, spec_rt, degree_rt, popts_rt = SER.load_pipeline_json(path; validation=:trusted)
            @test spec_rt.kind == :rips
            @test degree_rt == 1
            @test popts_rt.axes_policy == :encoding
            @test popts_rt.poset_kind == :signature
        end

        mktemp() do path, io
            close(io)
            data = TamerOp.PointCloud([[0.0], [1.0]])
            @test_throws ArgumentError SER.save_dataset_json(path, data; profile=:portable)
            spec = TamerOp.FiltrationSpec(kind=:rips, max_dim=1, axes=([0.0, 1.0],))
            @test_throws ArgumentError SER.save_pipeline_json(path, data, spec; profile=:portable)
        end
    end

    mktemp() do path, io
        close(io)
        data = TamerOp.PointCloud([[0.0], [1.0]])
        spec = TamerOp.FiltrationSpec(kind=:rips, max_dim=1, axes=([0.0, 1.0],))
        enc = TamerOp.encode(data, spec; degree=0)
        SER.save_encoding_json(path, enc)
        H2, pi2 = SER.load_encoding_json(path; output=:fringe_with_pi)
        @test FF.nvertices(H2.P) == FF.nvertices(enc.P)
        @test EC.axes_from_encoding(pi2) == EC.axes_from_encoding(enc.pi)
        enc2 = SER.load_encoding_json(path; output=:encoding_result)
        @test enc2 isa RES.EncodingResult
        @test FF.nvertices(enc2.P) == FF.nvertices(enc.P)
    end

    mktemp() do path, io
        close(io)
        F = FZ.face(1, [1])
        flats = [FZ.IndFlat(F, [0]; id=:F)]
        injectives = [FZ.IndInj(F, [0]; id=:E)]
        FG = FZ.Flange{K}(1, flats, injectives, [c(1)]; field=field)
        enc = TamerOp.encode(FG; backend=:zn)
        @test enc.pi isa EC.CompiledEncoding
        @test enc.pi.meta isa CM.EncodingCache
        SER.save_encoding_json(path, enc; include_pi=false)
        obj = JSON3.read(read(path, String))
        @test !haskey(obj, "pi")
        info = SER.inspect_json(path)
        @test !info.has_pi
        @test_throws ErrorException SER.load_encoding_json(path; output=:fringe_with_pi)
        SER.save_encoding_json(path, enc; include_pi=true)
        obj = JSON3.read(read(path, String))
        @test obj["pi"]["kind"] == "ZnEncodingMap"
        @test obj["pi"]["sig_y"]["kind"] == "packed_words_v1"
        @test obj["pi"]["sig_z"]["kind"] == "packed_words_v1"
        H2, pi2 = SER.load_encoding_json(path; output=:fringe_with_pi)
        @test EC.axes_from_encoding(pi2) == EC.axes_from_encoding(enc.pi)
    end

    mktemp() do path, io
        close(io)
        Ups = [PLB.BoxUpset([0.0, 0.0])]
        Downs = [PLB.BoxDownset([1.0, 1.0])]
        enc = TamerOp.encode(Ups, Downs; backend=:pl_backend)
        @test enc.pi isa EC.CompiledEncoding
        @test enc.pi.meta isa CM.EncodingCache
        SER.save_encoding_json(path, enc.P, enc.H, enc.pi)
        obj = JSON3.read(read(path, String))
        @test obj["pi"]["kind"] == "PLEncodingMapBoxes"
        @test obj["pi"]["sig_y"]["kind"] == "packed_words_v1"
        @test obj["pi"]["sig_z"]["kind"] == "packed_words_v1"
        H2, pi2 = SER.load_encoding_json(path; output=:fringe_with_pi)
        @test EC.axes_from_encoding(pi2) == EC.axes_from_encoding(enc.pi)
    end

    mktemp() do path, io
        close(io)
        sig_y = BitVector[
            BitVector([false, false]),
            BitVector([true,  false]),
            BitVector([true,  true]),
        ]
        sig_z = BitVector[
            BitVector([false]),
            BitVector([false]),
            BitVector([false]),
        ]
        P = TamerOp.ZnEncoding.SignaturePoset(sig_y, sig_z)
        U = FF.upset_closure(P, trues(FF.nvertices(P)))
        D = FF.downset_closure(P, trues(FF.nvertices(P)))
        H = FF.FringeModule{K}(P, [U], [D], reshape([c(1)], 1, 1); field=field)
        SER.save_encoding_json(path, H)
        obj = JSON3.read(read(path, String))
        @test Int(obj["schema_version"]) == TamerOp.Serialization.ENCODING_SCHEMA_VERSION
        @test obj["U"]["kind"] == "packed_words_v1"
        @test obj["D"]["kind"] == "packed_words_v1"
        @test obj["phi"]["kind"] == "qq_chunks_v1" || obj["phi"]["kind"] == "fp_flat_v1" || obj["phi"]["kind"] == "real_flat_v1"
        @test obj["poset"]["sig_y"]["kind"] == "packed_words_v1"
        @test obj["poset"]["sig_z"]["kind"] == "packed_words_v1"
        @test !haskey(obj["poset"], "leq")
        H2 = SER.load_encoding_json(path; output=:fringe)
        @test H2.P isa TamerOp.ZnEncoding.SignaturePoset
        @test H2.P.sig_y isa TamerOp.ZnEncoding.PackedSignatureRows
        @test H2.P.sig_z isa TamerOp.ZnEncoding.PackedSignatureRows
        @test FF.nvertices(H2.P) == FF.nvertices(P)
        @test FF.leq_matrix(H2.P) == FF.leq_matrix(P)

        SER.save_encoding_json(path, H; include_leq=false)
        H2 = SER.load_encoding_json(path; output=:fringe)
        @test H2.P isa TamerOp.ZnEncoding.SignaturePoset
        @test FF.nvertices(H2.P) == FF.nvertices(P)
        @test FF.leq_matrix(H2.P) == FF.leq_matrix(P)
    end

    mktemp() do path, io
        close(io)
        P = FF.FinitePoset(BitMatrix([1 1; 0 1]))
        U = FF.principal_upset(P, 2)
        D = FF.principal_downset(P, 2)
        H = FF.FringeModule{K}(P, [U], [D], reshape([c(1)], 1, 1); field=field)
        SER.save_encoding_json(path, H)
        obj = JSON3.read(read(path, String))
        @test obj["U"]["kind"] == "packed_words_v1"
        @test obj["D"]["kind"] == "packed_words_v1"
        @test obj["poset"]["leq"]["kind"] == "packed_words_v1"
        @test haskey(obj["poset"], "leq")
    end

    # Multi-input flange contract: tuple/vector only (no 2/3-arg varargs wrappers).
    let
        F = FZ.face(1, [1])
        flats = [FZ.IndFlat(F, [0]; id=:F)]
        injectives = [FZ.IndInj(F, [1]; id=:E)]
        FG = FZ.Flange{K}(1, flats, injectives, [c(1)]; field=field)

        out_tuple = TamerOp.encode((FG, FG); backend=:zn)
        out_vec = TamerOp.encode(FZ.Flange{K}[FG, FG]; backend=:zn)
        @test length(out_tuple) == 2
        @test length(out_vec) == 2
        @test_throws MethodError TamerOp.encode(FG, FG; backend=:zn)
        @test_throws MethodError TamerOp.encode(FG, FG, FG; backend=:zn)
    end
end

@testset "Data pipeline: DataFileIO file loading" begin
    mktempdir() do dir
        pc_path = joinpath(dir, "points.csv")
        write(pc_path, "x,y,z\n0.0,1.0,2.0\n1.0,2.0,3.0\n2.0,3.0,4.0\n")

        opts = TamerOp.DataFileOptions(; cols=(:x, :z), header=true)
        data = TamerOp.load_data(pc_path; kind=:point_cloud, opts=opts)
        @test data isa TamerOp.PointCloud
        @test length(data.points) == 3
        @test data.points[2] == [1.0, 3.0]
        @test_throws ArgumentError TamerOp.load_data(pc_path; kind=:point_cloud, opts=opts, max_dim=1)

        info = TamerOp.inspect_data_file(pc_path)
        @test info isa DFI.DataFileInspectionSummary
        @test info.format == :csv
        @test :point_cloud in info.candidate_kinds
        @test info.ncols == 3
        @test DFI.kind(info) == :table
        @test DFI.is_table_file(info)
        @test !DFI.is_dataset_json(info)
        @test DFI.is_ambiguous(info)
        @test DFI.requires_explicit_kind(info)
        @test DFI.suggested_kind(info) === nothing
        @test DFI.resolved_format(info) == :csv
        @test DFI.resolved_kind(info) === nothing
        @test DFI.columns(info) == (:x, :y, :z)
        @test length(DFI.sample_rows(info)) == 3
        @test DFI.detail(info) isa DFI.DelimitedTableInspection
        @test describe(info).kind == :data_file_inspection
        @test describe(DFI.detail(info)).kind == :delimited_table_inspection
        @test DFI.data_file_summary(info).format == :csv
        @test occursin("DataFileInspectionSummary", sprint(show, info))
    end

    mktempdir() do dir
        g_path = joinpath(dir, "graph.tsv")
        write(g_path, "u\tv\tw\n1\t2\t1.5\n2\t3\t2.5\n")

        opts = TamerOp.DataFileOptions(;
            header=true,
            u_col=:u,
            v_col=:v,
            weight_col=:w,
        )
        g = TamerOp.load_data(g_path; kind=:graph, format=:tsv, opts=opts)
        @test g isa TamerOp.GraphData
        @test g.n == 3
        @test g.edges == [(1, 2), (2, 3)]
        @test g.weights == [1.5, 2.5]
    end

    mktempdir() do dir
        g_path = joinpath(dir, "graph_no_header.csv")
        write(g_path, "1,2\n2,4\n")
        g = TamerOp.load_data(g_path; kind=:graph, format=:csv)
        @test g isa TamerOp.GraphData
        @test g.edges == [(1, 2), (2, 4)]
        @test g.n == 4
    end

    mktempdir() do dir
        img_path = joinpath(dir, "img.txt")
        write(img_path, "0 1 2\n3 4 5\n")
        img = TamerOp.load_data(img_path; kind=:image, format=:txt)
        @test img isa TamerOp.ImageNd
        @test size(img.data) == (2, 3)
        @test img.data[2, 3] == 5.0
    end

    mktempdir() do dir
        d_path = joinpath(dir, "dist.csv")
        write(d_path, "0,1,2\n1,0,3\n2,3,0\n")
        G = TamerOp.load_data(
            d_path;
            kind=:distance_matrix,
            format=:csv,
            max_dim=1,
            construction=TamerOp.ConstructionOptions(),
        )
        @test G isa TamerOp.GradedComplex
    end

    mktempdir() do dir
        bad_path = joinpath(dir, "ambiguous.csv")
        write(bad_path, "1,2\n3,4\n")
        @test_throws ArgumentError TamerOp.load_data(bad_path)
    end
end

@testset "Data pipeline: DataFileIO summaries and validators" begin
    mktempdir() do dir
        pc_path = joinpath(dir, "points.csv")
        write(pc_path, "x,y,z\n0.0,1.0,2.0\n1.0,2.0,3.0\n2.0,3.0,4.0\n")
        g_path = joinpath(dir, "graph.tsv")
        write(g_path, "u\tv\tw\n1\t2\t1.5\n2\t3\t2.5\n")
        nonsquare_path = joinpath(dir, "nonsquare.csv")
        write(nonsquare_path, "0,1,2\n1,0,3\n")
        ragged_path = joinpath(dir, "ragged.csv")
        write(ragged_path, "1,2\n3\n")
        json_path = joinpath(dir, "pts.json")
        TamerOp.save_dataset_json(json_path, TamerOp.PointCloud([[0.0], [1.0], [2.0]]))
        weird_path = joinpath(dir, "data.weird")
        write(weird_path, "1,2,3\n")

        info_json = TamerOp.inspect_data_file(json_path)
        @test info_json isa DFI.DataFileInspectionSummary
        @test DFI.is_dataset_json(info_json)
        @test !DFI.is_table_file(info_json)
        @test DFI.detail(info_json) isa DFI.DatasetFileInspection
        @test DFI.resolved_kind(info_json) == :point_cloud
        @test !DFI.is_ambiguous(info_json)
        @test !DFI.requires_explicit_kind(info_json)
        @test DFI.suggested_kind(info_json) == :point_cloud
        @test hasproperty(describe(info_json), :schema_version)
        @test isnothing(DFI.schema_version(info_json))
        @test describe(DFI.detail(info_json)).kind == :dataset_file_inspection

        file_ok = DFI.check_data_file(pc_path)
        @test file_ok isa DFI.DataFileValidationSummary
        @test DFI.ok(file_ok)
        @test DFI.validation_kind(file_ok) == :data_file_validation
        @test isempty(DFI.issues(file_ok))
        @test DFI.inspection(file_ok) isa DFI.DataFileInspectionSummary
        @test file_ok.candidate_kinds == (:point_cloud, :graph, :image, :distance_matrix)
        @test occursin("DataFileValidationSummary", sprint(show, file_ok))
        @test DFI.data_file_validation_summary(file_ok) === file_ok
        @test DFI.data_file_validation_summary(pc_path).ok

        file_bad = DFI.check_data_file(ragged_path)
        @test !DFI.ok(file_bad)
        @test !isempty(DFI.issues(file_bad))
        @test_throws ArgumentError DFI.check_data_file(ragged_path; throw=true)

        file_fmt_bad = DFI.check_data_file(weird_path)
        @test !file_fmt_bad.ok
        @test occursin("could not infer format", first(DFI.issues(file_fmt_bad)))

        col_ok = DFI.check_table_columns(g_path; kind=:graph, format=:tsv,
                                         opts=TamerOp.DataFileOptions(; header=true, u_col=:u, v_col=:v, weight_col=:w))
        @test col_ok isa DFI.TableColumnValidationSummary
        @test DFI.ok(col_ok)
        @test DFI.validation_kind(col_ok) == :table_column_validation
        @test DFI.resolved_columns(col_ok) == (; u=:u, v=:v, weight=:w)
        @test describe(col_ok).kind == :table_column_validation
        @test DFI.table_column_validation_summary(col_ok) === col_ok

        col_bad = DFI.check_table_columns(g_path; kind=:graph, format=:tsv,
                                          opts=TamerOp.DataFileOptions(; header=true, u_col=:src, v_col=:dst))
        @test !DFI.ok(col_bad)
        @test occursin("not found", first(DFI.issues(col_bad)))
        @test_throws ArgumentError DFI.check_table_columns(g_path; kind=:graph, format=:tsv,
                                                           opts=TamerOp.DataFileOptions(; header=true, u_col=:src, v_col=:dst),
                                                           throw=true)

        dist_bad = DFI.check_table_columns(nonsquare_path; kind=:distance_matrix, format=:csv)
        @test !dist_bad.ok
        @test occursin("n x n", first(DFI.issues(dist_bad)))

        load_bad = DFI.check_load_data(pc_path)
        @test load_bad isa DFI.LoadDataValidationSummary
        @test !DFI.ok(load_bad)
        @test occursin("kind=:auto is ambiguous", first(DFI.issues(load_bad)))

        load_ok = DFI.check_load_data(pc_path; kind=:point_cloud,
                                      opts=TamerOp.DataFileOptions(; header=true, cols=(:x, :y)))
        @test DFI.ok(load_ok)
        @test DFI.validation_kind(load_ok) == :load_data_validation
        @test DFI.inspection(load_ok) isa DFI.DataFileInspectionSummary
        @test DFI.resolved_columns(load_ok) == (; coords=(:x, :y))
        @test occursin("LoadDataValidationSummary", sprint(show, load_ok))
        @test DFI.load_data_validation_summary(load_ok) === load_ok

        @test TOA.DataFileInspectionSummary === DFI.DataFileInspectionSummary
        @test TOA.DelimitedTableInspection === DFI.DelimitedTableInspection
        @test TOA.DatasetFileInspection === DFI.DatasetFileInspection
        @test TOA.DataFileValidationSummary === DFI.DataFileValidationSummary
        @test TOA.LoadDataValidationSummary === DFI.LoadDataValidationSummary
        @test TOA.TableColumnValidationSummary === DFI.TableColumnValidationSummary
        @test TOA.data_file_summary === DFI.data_file_summary
        @test TOA.data_file_validation_summary === DFI.data_file_validation_summary
        @test TOA.load_data_validation_summary === DFI.load_data_validation_summary
        @test TOA.table_column_validation_summary === DFI.table_column_validation_summary
        @test TOA.check_data_file === DFI.check_data_file
        @test TOA.check_load_data === DFI.check_load_data
        @test TOA.check_table_columns === DFI.check_table_columns
    end
end

@testset "Data pipeline: encode from data files" begin
    mktempdir() do dir
        pc_path = joinpath(dir, "pts.csv")
        write(pc_path, "x,y\n0.0,0.0\n1.0,0.0\n0.0,1.0\n")
        spec = TamerOp.FiltrationSpec(kind=:rips, max_dim=1)
        opts = TamerOp.DataFileOptions(; header=true, cols=(:x, :y))
        enc_path = TamerOp.encode(pc_path, spec; kind=:point_cloud, file_opts=opts, degree=0)
        enc_mem = TamerOp.encode(TamerOp.PointCloud([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), spec; degree=0)
        @test enc_path isa RES.EncodingResult
        @test enc_mem isa RES.EncodingResult
        @test FF.nvertices(enc_path.P) == FF.nvertices(enc_mem.P)
        @test TamerOp.Results.module_dims(TamerOp.Results.materialize_module(enc_path.M)) ==
              TamerOp.Results.module_dims(TamerOp.Results.materialize_module(enc_mem.M))
    end

    mktempdir() do dir
        json_path = joinpath(dir, "pts.json")
        data = TamerOp.PointCloud([[0.0], [1.0], [2.0]])
        TamerOp.save_dataset_json(json_path, data)
        spec = TamerOp.FiltrationSpec(kind=:rips, max_dim=1)
        enc = TamerOp.encode(json_path, spec; degree=0)
        @test enc isa RES.EncodingResult
        @test FF.nvertices(enc.P) > 0
    end
end

@testset "Data pipeline: poset_from_axes kind=:grid" begin
    axes = (Float64[0.0, 1.0], Float64[0.0, 2.0, 4.0])
    P = DI.poset_from_axes(axes; kind=:grid)
    @test P isa FF.ProductOfChainsPoset
    @test FF.nvertices(P) == 2 * 3
    @test FF.leq(P, 1, 6)
    @test !FF.leq(P, 6, 1)
end

end # with_fields
@testset "Data pipeline: poset_from_axes kind=:grid with orientation -1" begin
    axes = (Float64[0.0, 1.0], Float64[0.0, 2.0, 4.0])
    P = DI.poset_from_axes(axes; orientation=(1, -1), kind=:grid)
    @test P isa FF.FinitePoset
    @test FF.nvertices(P) == 2 * 3
end

@testset "Data pipeline: auto axes respect orientation signs" begin
    cells = [Int[1, 2]]
    boundaries = SparseMatrixCSC{Int,Int}[]
    grades = [Float64[1.0], Float64[2.0]]
    G = TamerOp.GradedComplex(cells, boundaries, grades)

    spec_auto = TamerOp.FiltrationSpec(kind=:graded, orientation=(-1,))
    enc_auto = TamerOp.encode(G, spec_auto; degree=0)
    axes_auto = EC.axes_from_encoding(enc_auto.pi)
    @test axes_auto == (Float64[-2.0, -1.0],)

    # Explicit axes must remain unchanged even with negative orientation.
    explicit_axes = (Float64[-3.0, -2.0, -1.0],)
    spec_explicit = TamerOp.FiltrationSpec(kind=:graded, orientation=(-1,), axes=explicit_axes)
    enc_explicit = TamerOp.encode(G, spec_explicit; degree=0)
    @test EC.axes_from_encoding(enc_explicit.pi) == explicit_axes
end

@testset "Data pipeline: point cloud auto axes honor orientation" begin
    data = TamerOp.PointCloud([[0.0], [1.0]])
    spec = TamerOp.FiltrationSpec(kind=:rips, max_dim=1, orientation=(-1,))
    enc = TamerOp.encode(data, spec; degree=0)
    ax = EC.axes_from_encoding(enc.pi)[1]
    @test minimum(ax) <= 0.0
    @test maximum(ax) <= 0.0
end

@testset "Data pipeline: point cloud dim2 packed kernel parity" begin
    pts = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [3.0, 0.0],
        [3.0, 1.0],
    ]
    data = TamerOp.PointCloud(pts)
    spec = TamerOp.FiltrationSpec(
        kind=:rips,
        max_dim=2,
        construction=TamerOp.ConstructionOptions(;
            sparsify=:none,
            collapse=:none,
            output_stage=:simplex_tree,
        ),
    )

    old_dim2 = DI._POINTCLOUD_DIM2_PACKED_KERNEL[]
    st_base = nothing
    st_fast = nothing
    try
        DI._POINTCLOUD_DIM2_PACKED_KERNEL[] = false
        st_base = TamerOp.encode(data, spec; degree=0)

        DI._POINTCLOUD_DIM2_PACKED_KERNEL[] = true
        st_fast = TamerOp.encode(data, spec; degree=0)
    finally
        DI._POINTCLOUD_DIM2_PACKED_KERNEL[] = old_dim2
    end

    @test st_base isa DI.SimplexTreeMulti
    @test st_fast isa DI.SimplexTreeMulti
    @test DI.simplex_count(st_fast) == DI.simplex_count(st_base)
    @test st_fast.simplex_dims == st_base.simplex_dims
    @test st_fast.simplex_vertices == st_base.simplex_vertices
    @test st_fast.grade_data == st_base.grade_data

    spec_radius = TamerOp.FiltrationSpec(
        kind=:rips,
        max_dim=2,
        radius=1.5,
        construction=TamerOp.ConstructionOptions(;
            sparsify=:none,
            collapse=:none,
            output_stage=:simplex_tree,
        ),
    )
    st_radius = TamerOp.encode(data, spec_radius; degree=0)
    edge_lo = st_fast.dim_offsets[2]
    edge_hi = st_fast.dim_offsets[3] - 1
    edge_lo_r = st_radius.dim_offsets[2]
    edge_hi_r = st_radius.dim_offsets[3] - 1
    @test edge_hi_r <= edge_hi
    for sid in edge_lo_r:edge_hi_r
        g = st_radius.grade_data[st_radius.grade_offsets[sid]]
        @test g[1] <= 1.5 + 1e-12
    end
    for sid in edge_lo:edge_hi
        g = st_fast.grade_data[st_fast.grade_offsets[sid]]
        @test g[1] >= 0.0
    end

    spec_radius_d1 = TamerOp.FiltrationSpec(
        kind=:rips,
        max_dim=1,
        radius=1.1,
        construction=TamerOp.ConstructionOptions(;
            sparsify=:none,
            collapse=:none,
            output_stage=:simplex_tree,
        ),
    )
    st_radius_d1 = TamerOp.encode(data, spec_radius_d1; degree=0)
    edge_lo_d1 = st_radius_d1.dim_offsets[2]
    edge_hi_d1 = st_radius_d1.dim_offsets[3] - 1
    @test (edge_hi_d1 - edge_lo_d1 + 1) < 15
    for sid in edge_lo_d1:edge_hi_d1
        g = st_radius_d1.grade_data[st_radius_d1.grade_offsets[sid]]
        @test g[1] <= 1.1 + 1e-12
    end
end

@testset "Interop adapters: GUDHI/Ripserer/Eirene" begin
    mktemp() do path, io
        close(io)
        open(path, "w") do f
            write(f, "{\"simplices\": [[0],[1],[0,1]], \"filtration\": [0.0,0.0,1.0]}")
        end
        G = SER.load_gudhi_json(path)
        @test length(G.grades) == 3
        @test length(G.boundaries) == 1
    end

    mktemp() do path, io
        close(io)
        open(path, "w") do f
            write(f, "{\"simplices\": [[0],[1],[0,1]], \"filtration\": [0.0,0.0,1.0]}")
        end
        G = SER.load_ripserer_json(path)
        @test length(G.grades) == 3
        @test length(G.boundaries) == 1
    end

    mktemp() do path, io
        close(io)
        open(path, "w") do f
            write(f, "{\"simplices\": [[0],[1],[0,1]], \"filtration\": [0.0,0.0,1.0]}")
        end
        G = SER.load_eirene_json(path)
        @test length(G.grades) == 3
        @test length(G.boundaries) == 1
    end

    mktemp() do path, io
        close(io)
        open(path, "w") do f
            write(f, "# dim v1 v2 filtration\n")
            write(f, "0 0 0.0\n")
            write(f, "0 1 0.0\n")
            write(f, "1 0 1 1.0\n")
        end
        G = SER.load_gudhi_txt(path)
        @test length(G.grades) == 3
        @test length(G.boundaries) == 1
    end
end

@testset "Interop adapters: Ripser/DIPHA matrix formats" begin
    mktemp() do path, io
        close(io)
        open(path, "w") do f
            write(f, "0.0 1.0\n1.0 0.0\n")
        end
        G = SER.load_ripser_distance(path; max_dim=1)
        @test length(G.grades) == 3
        @test maximum(getindex.(G.grades, 1)) == 1.0
    end

    mktemp() do path, io
        close(io)
        open(path, "w") do f
            write(f, "0.0 1.0\n1.0 0.0\n")
        end
        G = SER.load_ripser_distance(
            path;
            max_dim=1,
            radius=1.0,
            construction=TamerOp.ConstructionOptions(; sparsify=:radius, budget=(nothing, 4, nothing)),
        )
        @test length(G.grades) == 3
        @test_throws MethodError SER.load_ripser_distance(path; max_dim=1, sparse_rips=true, radius=1.0)
        @test_throws MethodError SER.load_ripser_distance(path; max_dim=1, approx_rips=true, radius=1.0)
    end

    mktemp() do path, io
        close(io)
        open(path, "w") do f
            write(f, "0.0\n1.0 0.0\n")
        end
        G = SER.load_ripser_lower_distance(path; max_dim=1)
        @test length(G.grades) == 3
        @test maximum(getindex.(G.grades, 1)) == 1.0
        @test_throws Exception SER.load_ripser_lower_distance(
            path;
            max_dim=1,
            construction=TamerOp.ConstructionOptions(; collapse=:acyclic),
        )
    end

    mktemp() do path, io
        close(io)
        open(path, "w") do f
            write(f, "0.0 1.0\n0.0\n")
        end
        G = SER.load_ripser_upper_distance(path; max_dim=1)
        @test length(G.grades) == 3
        @test maximum(getindex.(G.grades, 1)) == 1.0
    end

    mktemp() do path, io
        close(io)
        open(path, "w") do f
            write(f, "0 1 1.0\n")
        end
        G = SER.load_ripser_sparse_triplet(path; max_dim=1)
        @test length(G.grades) == 3
        @test maximum(getindex.(G.grades, 1)) == 1.0
    end

    mktemp() do path, io
        close(io)
        open(path, "w") do f
            write(f, "0.0 0.0\n1.0 0.0\n")
        end
        pc = SER.load_ripser_point_cloud(path)
        @test length(pc.points) == 2
        @test length(pc.points[1]) == 2
    end

    mktemp() do path, io
        close(io)
        open(path, "w") do f
            write(f, Float64[0.0, 1.0, 0.0])
        end
        G = SER.load_ripser_binary_lower_distance(path; max_dim=1)
        @test length(G.grades) == 3
        @test maximum(getindex.(G.grades, 1)) == 1.0
    end

    mktemp() do path, io
        close(io)
        open(path, "w") do f
            write(f, Int64(8067171840))
            write(f, Int64(7))
            write(f, Int64(2))
            write(f, Float64[0.0, 1.0, 1.0, 0.0])
        end
        G = SER.load_dipha_distance_matrix(path; max_dim=1)
        @test length(G.grades) == 3
        @test maximum(getindex.(G.grades, 1)) == 1.0
    end
end

@testset "Interop adapters: fixture round-trips" begin
    fixtures = joinpath(@__DIR__, "fixtures", "interop")

    G = SER.load_gudhi_json(joinpath(fixtures, "gudhi.json"))
    @test length(G.grades) == 7
    @test length(G.boundaries) == 2

    G = SER.load_ripserer_json(joinpath(fixtures, "ripserer.json"))
    @test length(G.grades) == 7
    @test length(G.boundaries) == 2

    G = SER.load_eirene_json(joinpath(fixtures, "eirene.json"))
    @test length(G.grades) == 3
    @test length(G.boundaries) == 1

    G = SER.load_gudhi_txt(joinpath(fixtures, "gudhi.txt"))
    @test length(G.grades) == 3
    @test length(G.boundaries) == 1

    G = SER.load_ripserer_txt(joinpath(fixtures, "ripserer.txt"))
    @test length(G.grades) == 3
    @test length(G.boundaries) == 1

    G = SER.load_eirene_txt(joinpath(fixtures, "eirene.txt"))
    @test length(G.grades) == 3
    @test length(G.boundaries) == 1

    G = SER.load_ripser_distance(joinpath(fixtures, "ripser_distance.txt"); max_dim=1)
    @test length(G.grades) == 6
    @test maximum(getindex.(G.grades, 1)) == 3.0

    G = SER.load_ripser_lower_distance(joinpath(fixtures, "ripser_lower_distance.txt"); max_dim=1)
    @test length(G.grades) == 6
    @test maximum(getindex.(G.grades, 1)) == 3.0

    G = SER.load_ripser_upper_distance(joinpath(fixtures, "ripser_upper_distance.txt"); max_dim=1)
    @test length(G.grades) == 6
    @test maximum(getindex.(G.grades, 1)) == 3.0

    G = SER.load_ripser_sparse_triplet(joinpath(fixtures, "ripser_sparse_triplet.txt"); max_dim=1)
    @test length(G.grades) == 6
    @test maximum(getindex.(G.grades, 1)) == 3.0

    pc = SER.load_ripser_point_cloud(joinpath(fixtures, "ripser_point_cloud.txt"))
    @test length(pc.points) == 3
    @test length(pc.points[1]) == 2

    G = SER.load_ripser_binary_lower_distance(joinpath(fixtures, "ripser_binary_lower_distance.bin"); max_dim=1)
    @test length(G.grades) == 6
    @test maximum(getindex.(G.grades, 1)) == 3.0

    G = SER.load_dipha_distance_matrix(joinpath(fixtures, "dipha_distance_matrix.bin"); max_dim=1)
    @test length(G.grades) == 6
    @test maximum(getindex.(G.grades, 1)) == 3.0
end

@testset "Interop adapters: boundary complex + PModule JSON" begin
    mktemp() do path, io
        close(io)
        open(path, "w") do f
            write(f, """{
  "counts_by_dim": [2,1],
  "boundaries": [
    {"m":2,"n":1,"I":[1,2],"J":[1,1],"V":[1,-1]}
  ],
  "grades": [[0.0],[0.0],[1.0]]
}""")
        end
        G = SER.load_boundary_complex_json(path)
        @test length(G.grades) == 3
        @test length(G.boundaries) == 1
    end

    mktemp() do path, io
        close(io)
        open(path, "w") do f
            write(f, """{
  "poset": {"n": 2, "leq": [[true,true],[false,true]]},
  "dims": [1,1],
  "edges": [
    {"src": 1, "dst": 2, "mat": [[1]]}
  ]
}""")
        end
        M = SER.load_pmodule_json(path)
        @test M.dims == [1, 1]
    end
end

@testset "Interop adapters: streaming lower-triangular distance" begin
    fixtures = joinpath(@__DIR__, "fixtures", "interop")
    G = SER.load_ripser_lower_distance_streaming(
        joinpath(fixtures, "ripser_lower_distance.txt"); radius=2.5
    )
    @test length(G.grades) == 5
    @test maximum(getindex.(G.grades, 1)) == 2.0
end

@testset "Data pipeline: JSON negative cases" begin
    mktemp() do path, io
        close(io)
        open(path, "w") do f
            write(f, "{\"points\": [[0.0]]}")
        end
        @test_throws Exception SER.load_dataset_json(path)
    end

    mktemp() do path, io
        close(io)
        open(path, "w") do f
            write(f, "{\"dataset\": {\"kind\": \"PointCloud\", \"points\": [[0.0]]}, \"spec\": {}}")
        end
        @test_throws Exception SER.load_pipeline_json(path)
    end

    data = TamerOp.PointCloud([[0.0], [1.0]])
    spec = TamerOp.FiltrationSpec(kind=:rips, max_dim=1, axes=([1.0, 0.0],))
    @test_throws Exception TamerOp.encode(data, spec; degree=0)

    cells = [Int[1]]
    boundaries = SparseMatrixCSC{Int,Int}[]
    grades = [Float64[0.0], Float64[1.0]]
    @test_throws Exception TamerOp.GradedComplex(cells, boundaries, grades)

    data = TamerOp.PointCloud([[0.0], [1.0]])
    spec = TamerOp.FiltrationSpec(kind=:rips, max_dim=1, axes=([0.0, 1.5],))
    enc = TamerOp.encode(data, spec; degree=0)
    @test FF.nvertices(enc.P) > 0

    spec = TamerOp.FiltrationSpec(kind=:rips, max_dim=1, axes=([0, 1],), axis_kind=:rn)
    @test_throws Exception TamerOp.encode(data, spec; degree=0)

    spec = TamerOp.FiltrationSpec(kind=:rips, max_dim=1, axes=([0.0, 1.0],), axis_kind=:zn)
    @test_throws Exception TamerOp.encode(data, spec; degree=0)
end

@testset "Data pipeline: quantization + coarsen axes" begin
    cells = [Int[1], Int[]]
    boundaries = SparseMatrixCSC{Int,Int}[]
    grades = [Float64[0.05]]
    G = TamerOp.GradedComplex(cells, boundaries, grades)
    spec = TamerOp.FiltrationSpec(kind=:graded, eps=0.1)
    enc = TamerOp.encode(G, spec; degree=0)
    @test EC.axes_from_encoding(enc.pi)[1] == [0.0]

    data = TamerOp.PointCloud([[0.0], [1.0], [2.0], [3.0]])
    spec = TamerOp.FiltrationSpec(kind=:rips, max_dim=1, axes_policy=:coarsen, max_axis_len=2)
    enc = TamerOp.encode(data, spec; degree=0)
    @test length(EC.axes_from_encoding(enc.pi)[1]) == 2
end

@testset "Data pipeline: session cache reuse" begin
    data = TamerOp.PointCloud([[0.0], [1.0]])
    spec = TamerOp.FiltrationSpec(kind=:rips, max_dim=1, axes=([0.0, 1.0],))
    sc = CM.SessionCache()
    enc1 = TamerOp.encode(data, spec; degree=0, cache=sc)
    enc2 = TamerOp.encode(data, spec; degree=0, cache=sc)
    @test enc1.P === enc2.P
    @test typeof(enc1.M) === typeof(enc2.M)
    @test DI.module_dims(enc1.M) == DI.module_dims(enc2.M)
    @test enc1.H === enc2.H

    enc_deg1 = TamerOp.encode(data, spec; degree=1, cache=sc)
    @test enc_deg1.M !== enc1.M

    CM._clear_session_cache!(sc)
    enc3 = TamerOp.encode(data, spec; degree=0, cache=sc)
    @test enc3.P !== enc1.P
end

@testset "Data pipeline: point cloud guardrails + landmark_rips" begin
    data = TamerOp.PointCloud([[0.0], [1.0], [2.0], [3.0]])
    spec = TamerOp.FiltrationSpec(
        kind=:rips,
        max_dim=2,
        construction=TamerOp.ConstructionOptions(; budget=(5, nothing, nothing)),
    )
    @test_throws Exception TamerOp.encode(data, spec; degree=0)

    data_big = TamerOp.PointCloud([[Float64(i)] for i in 1:180])
    spec_precheck = TamerOp.FiltrationSpec(
        kind=:rips,
        max_dim=3,
        construction=TamerOp.ConstructionOptions(; budget=(15_000, nothing, nothing)),
    )
    err = try
        TamerOp.encode(data_big, spec_precheck; degree=0)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("before enumeration", sprint(showerror, err))

    # Even one cube in affine dimension 3 needs 3^4=81 cells.
    rhomboid_cloud = DT.PointCloud([0 0 0; 1 0 0; 0 1 0; 0 0 1])
    spec_rhomboid_precheck = OPT.FiltrationSpec(kind=:rhomboid,
        construction=OPT.ConstructionOptions(; budget=(80, nothing, nothing)))
    @test_throws ArgumentError TamerOp.encode(rhomboid_cloud, spec_rhomboid_precheck; stage=:graded_complex)

    spec = TamerOp.FiltrationSpec(
        kind=:rips,
        max_dim=1,
        radius=1.1,
        construction=TamerOp.ConstructionOptions(; sparsify=:radius),
    )
    enc = TamerOp.encode(data, spec; degree=0)
    @test FF.nvertices(enc.P) > 0

    spec = TamerOp.FiltrationSpec(kind=:landmark_rips, max_dim=1, landmarks=[1, 3])
    enc = TamerOp.encode(data, spec; degree=0)
    @test FF.nvertices(enc.P) > 0

    # landmark_rips must preserve radius/knn backend knobs through FiltrationSpec
    # -> typed filtration -> FiltrationSpec round-trips.
    spec_lm_radius = TamerOp.FiltrationSpec(
        kind=:landmark_rips,
        max_dim=1,
        landmarks=[1, 2, 3],
        radius=0.6,
        nn_backend=:auto,
        construction=TamerOp.ConstructionOptions(; output_stage=:simplex_tree),
    )
    typed_lm = DI.to_filtration(spec_lm_radius)
    roundtrip_lm = DI._filtration_spec(typed_lm)
    @test get(roundtrip_lm.params, :radius, nothing) == 0.6
    @test get(roundtrip_lm.params, :nn_backend, nothing) == :auto

    st_lm = TamerOp.encode(data, spec_lm_radius; degree=0, stage=:simplex_tree)
    lm_points = [data.points[i] for i in [1, 2, 3]]
    lm_edges_expected, _ = DI._point_cloud_edges_within_radius(lm_points, 0.6)
    @test count(==(1), st_lm.simplex_dims) == length(lm_edges_expected)

    # landmark+radii should normalize to sparse radius construction by default.
    spec_lm_radius_explicit = TamerOp.FiltrationSpec(
        kind=:landmark_rips,
        max_dim=1,
        landmarks=[1, 2, 3],
        radius=0.6,
        construction=TamerOp.ConstructionOptions(; sparsify=:radius, output_stage=:simplex_tree),
    )
    st_lm_explicit = TamerOp.encode(data, spec_lm_radius_explicit; degree=0, stage=:simplex_tree)
    @test _canon_simplex_tree(st_lm) == _canon_simplex_tree(st_lm_explicit)

    # landmark subgraph cache (session-level geometry cache) should reuse packed edge lists.
    ec = CM.EncodingCache()
    packed1 = DI._landmark_radius_subgraph_cached(data.points, [1, 2, 3], 0.6, spec_lm_radius; cache=ec)
    packed2 = DI._landmark_radius_subgraph_cached(data.points, [1, 2, 3], 0.6, spec_lm_radius; cache=ec)
    @test packed1 === packed2
    @test length(packed1.edges) == length(lm_edges_expected)

    spec = TamerOp.FiltrationSpec(
        kind=:rips,
        max_dim=1,
        knn=2,
        construction=TamerOp.ConstructionOptions(; sparsify=:knn, budget=(nothing, 8, nothing)),
    )
    enc = TamerOp.encode(data, spec; degree=0)
    @test FF.nvertices(enc.P) > 0

    spec = TamerOp.FiltrationSpec(
        kind=:rips,
        max_dim=1,
        construction=TamerOp.ConstructionOptions(; budget=(nothing, 1, nothing)),
    )
    @test_throws Exception TamerOp.encode(data, spec; degree=0)
end

@testset "Data pipeline: estimate_ingestion preflight" begin
    data = TamerOp.PointCloud([[0.0], [1.0], [2.0], [3.0]])
    spec = TamerOp.FiltrationSpec(kind=:rips, max_dim=2, axes=([0.0, 1.0],))
    est = DI.estimate_ingestion(data, spec)
    @test DI.estimated_cells(est) == big(14)  # C(4,1)+C(4,2)+C(4,3)
    @test DI.cell_counts_by_dim(est) == BigInt[4, 6, 4]
    @test DI.estimated_axis_sizes(est) == (2,)
    @test DI.estimated_poset_size(est) == big(2)
    @test DI.estimated_nnz(est) == big(24)
    @test DI.estimated_dense_bytes(est) == big(192)

    est_warn = DI.estimate_ingestion(data,
                                               TamerOp.FiltrationSpec(
                                                   kind=:rips,
                                                   max_dim=2,
                                                   construction=(budget=(max_simplices=5, max_edges=nothing, memory_budget_bytes=nothing),),
                                               );
                                               poset_threshold=1)
    @test !isempty(DI.estimate_warnings(est_warn))
    @test any(occursin("max_simplices", w) for w in DI.estimate_warnings(est_warn))
    @test any(occursin("|P|", w) || occursin("axis_sizes unavailable", w) for w in DI.estimate_warnings(est_warn))

    est_edges = DI.estimate_ingestion(
        data,
        TamerOp.FiltrationSpec(kind=:rips, max_dim=1,
                                    construction=(budget=(max_simplices=nothing, max_edges=2, memory_budget_bytes=nothing),)),
    )
    @test any(occursin("max_edges", w) for w in DI.estimate_warnings(est_edges))

    @test_throws ArgumentError DI.estimate_ingestion(
        data,
        TamerOp.FiltrationSpec(kind=:delaunay_lower_star, max_dim=2, highdim_policy=:error);
        strict=true,
    )
end

@testset "Data pipeline: serialization fast-load parity" begin
    function _pointcloud_from_flat_rowmajor(n::Int, d::Int, flat::Vector{Float64})
        length(flat) == n * d || error("PointCloud points_flat length mismatch.")
        pts = Matrix{Float64}(undef, n, d)
        t = 1
        @inbounds for i in 1:n, j in 1:d
            pts[i, j] = flat[t]
            t += 1
        end
        return DT.PointCloud(pts; copy=false)
    end

    function _coords_from_flat_rowmajor(n::Int, d::Int, flat::Vector{Float64})
        d >= 0 || error("GraphData coords_dim must be nonnegative.")
        d == 0 && return Matrix{Float64}(undef, n, 0)
        length(flat) == n * d || error("GraphData coords_flat length mismatch.")
        out = Matrix{Float64}(undef, n, d)
        t = 1
        @inbounds for i in 1:n, j in 1:d
            out[i, j] = flat[t]
            t += 1
        end
        return out
    end

    function _load_dataset_json_baseline(path::AbstractString)
        raw = read(path, String)
        kind_hdr = JSON3.read(raw, NamedTuple{(:kind,),Tuple{String}})
        kind = kind_hdr.kind
        if kind == "PointCloud"
            obj = JSON3.read(raw, SER._PointCloudColumnarJSON)
            return _pointcloud_from_flat_rowmajor(obj.n, obj.d, obj.points_flat)
        end
        if kind == "GraphData"
            obj = JSON3.read(raw, SER._GraphDataColumnarJSON)
            coords = if obj.coords_dim === nothing || obj.coords_flat === nothing
                nothing
            else
                _coords_from_flat_rowmajor(obj.n, obj.coords_dim, obj.coords_flat)
            end
            return DT.GraphData(obj.n, obj.edges_u, obj.edges_v;
                                coords=coords,
                                weights=obj.weights,
                                T=Float64,
                                copy=false)
        end
        return SER._dataset_from_obj(JSON3.read(raw))
    end

    function _load_encoding_json_strict_baseline(path::AbstractString)
        raw = read(path, String)
        obj = JSON3.read(raw, SER._FiniteEncodingFringeJSONV1)
        obj.kind == "FiniteEncodingFringe" || error("Unsupported encoding JSON kind: $(obj.kind)")
        obj.schema_version == SER.ENCODING_SCHEMA_VERSION ||
            error("Unsupported encoding JSON schema_version: $(obj.schema_version)")
        P = SER._parse_poset_from_typed(obj.poset)
        n = FF.nvertices(P)
        Umasks = SER._decode_masks(obj.U, "U", n)
        Dmasks = SER._decode_masks(obj.D, "D", n)
        U = SER._build_upsets(P, Umasks, true)
        D = SER._build_downsets(P, Dmasks, true)
        saved_field = SER._field_from_typed(obj.coeff_field)
        m = length(D)
        k = length(U)
        Phi = SER._decode_phi(obj.phi, saved_field, saved_field, m, k)
        H = FF.FringeModule{CM.coeff_type(saved_field)}(P, U, D, Phi; field=saved_field)
        obj.pi === nothing && error("baseline strict load requires stored pi.")
        return H, SER._pi_from_typed(P, obj.pi)
    end

    mktemp() do path, io
        close(io)
        data = TamerOp.PointCloud([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        old_path = path * "_old"
        SER._json_write(old_path, Dict("kind" => "PointCloud",
                                       "layout" => "columnar_v1",
                                       "n" => size(DT.point_matrix(data), 1),
                                       "d" => size(DT.point_matrix(data), 2),
                                       "points_flat" => begin
                                           flat = Float64[]
                                           pts = DT.point_matrix(data)
                                           @inbounds for i in axes(pts, 1), j in axes(pts, 2)
                                               push!(flat, pts[i, j])
                                           end
                                           flat
                                       end); pretty=false)
        SER.save_dataset_json(path, data)
        new_data = SER.load_dataset_json(path)
        old_data = _load_dataset_json_baseline(old_path)
        @test new_data.coords == old_data.coords
    end

    mktemp() do path, io
        close(io)
        data = TamerOp.GraphData(4, [(1, 2), (2, 3), (3, 4)];
                                      coords=[[0.0], [1.0], [2.0], [3.0]],
                                      weights=[1.0, 2.0, 3.0])
        old_path = path * "_old"
        coords = DT.coord_matrix(data)
        coords_flat = Float64[]
        @inbounds for i in axes(coords, 1), j in axes(coords, 2)
            push!(coords_flat, coords[i, j])
        end
        SER._json_write(old_path, Dict("kind" => "GraphData",
                                       "layout" => "columnar_v1",
                                       "n" => data.n,
                                       "edges_u" => collect(DT.edge_columns(data)[1]),
                                       "edges_v" => collect(DT.edge_columns(data)[2]),
                                       "coords_dim" => size(coords, 2),
                                       "coords_flat" => coords_flat,
                                       "weights" => data.weights === nothing ? nothing : collect(data.weights));
                        pretty=false)
        SER.save_dataset_json(path, data)
        new_data = SER.load_dataset_json(path)
        old_data = _load_dataset_json_baseline(old_path)
        @test new_data.edge_u == old_data.edge_u
        @test new_data.edge_v == old_data.edge_v
        @test new_data.coord_matrix == old_data.coord_matrix
        @test new_data.weights == old_data.weights
    end

    mktemp() do path, io
        close(io)
        data = TamerOp.PointCloud([[0.0], [1.0]])
        spec = TamerOp.FiltrationSpec(kind=:rips, max_dim=1, axes=([0.0, 1.0],))
        enc = TamerOp.encode(data, spec; degree=0)
        SER.save_encoding_json(path, enc)
        H_new, pi_new = SER.load_encoding_json(path; output=:fringe_with_pi, validation=:strict)
        H_old, pi_old = _load_encoding_json_strict_baseline(path)
        @test H_new.phi == H_old.phi
        @test [U.mask for U in H_new.U] == [U.mask for U in H_old.U]
        @test [D.mask for D in H_new.D] == [D.mask for D in H_old.D]
        @test EC.axes_from_encoding(pi_new) == EC.axes_from_encoding(pi_old)
    end
end

@testset "Data pipeline: construction contract is strict" begin
    data = TamerOp.PointCloud([[0.0], [1.0], [2.0]])
    bad = TamerOp.FiltrationSpec(kind=:rips, max_dim=1, sparse_rips=true, radius=1.0)
    @test_throws ArgumentError TamerOp.encode(data, bad; degree=0)

    @test_throws ArgumentError TamerOp.ConstructionOptions(; sparsify=:approx)
    @test_throws ArgumentError TamerOp.ConstructionOptions(; collapse=:legacy)
    @test TamerOp.ConstructionOptions(; output_stage=:simplex_tree).output_stage == :simplex_tree
    @test_throws ArgumentError TamerOp.ConstructionOptions(; output_stage=:raw)
end

@testset "Data pipeline: point-cloud sparse large-n contract" begin
    n = 5_001
    data = TamerOp.PointCloud([[Float64(i)] for i in 1:n])
    dense_spec = TamerOp.FiltrationSpec(kind=:rips, max_dim=1)
    @test_throws ArgumentError TamerOp.encode(data, dense_spec; degree=0)
end

@testset "Data pipeline: edge-driven sparse point-cloud path" begin
    data = TamerOp.PointCloud([[0.0], [1.0], [2.0], [3.0]])
    dense = TamerOp.FiltrationSpec(
        kind=:rips,
        max_dim=1,
        construction=TamerOp.ConstructionOptions(; output_stage=:graded_complex),
    )
    sparse_radius = TamerOp.FiltrationSpec(
        kind=:rips,
        max_dim=1,
        radius=10.0,
        construction=TamerOp.ConstructionOptions(;
            sparsify=:radius,
            output_stage=:graded_complex,
            budget=(max_simplices=nothing, max_edges=32, memory_budget_bytes=nothing),
        ),
    )
    G_dense = TamerOp.encode(data, dense; degree=0)
    G_sparse = TamerOp.encode(data, sparse_radius; degree=0)
    @test G_dense isa TamerOp.GradedComplex
    @test G_sparse isa TamerOp.GradedComplex
    @test G_dense.cells_by_dim == G_sparse.cells_by_dim
    @test G_dense.grades == G_sparse.grades
end

@testset "Data pipeline: NN backend contract for sparse point-cloud path" begin
    data = TamerOp.PointCloud([[0.0], [0.5], [1.5], [3.0], [4.5]])
    spec_auto = TamerOp.FiltrationSpec(
        kind=:rips,
        max_dim=1,
        knn=2,
        nn_backend=:auto,
        construction=TamerOp.ConstructionOptions(; sparsify=:knn, output_stage=:graded_complex),
    )
    spec_nn = TamerOp.FiltrationSpec(
        kind=:rips,
        max_dim=1,
        knn=2,
        nn_backend=:nearestneighbors,
        construction=TamerOp.ConstructionOptions(; sparsify=:knn, output_stage=:graded_complex),
    )
    spec_apx = TamerOp.FiltrationSpec(
        kind=:rips,
        max_dim=1,
        knn=2,
        nn_backend=:approx,
        nn_approx_candidates=8,
        construction=TamerOp.ConstructionOptions(; sparsify=:knn, output_stage=:graded_complex),
    )
    if DI._have_pointcloud_nn_backend()
        @test DI._pointcloud_nn_backend(spec_auto) == :auto
        G_auto = TamerOp.encode(data, spec_auto; degree=0)
        G_nn = TamerOp.encode(data, spec_nn; degree=0)
        @test G_auto.cells_by_dim == G_nn.cells_by_dim
        @test G_auto.grades == G_nn.grades
        @test TamerOp.encode(data, spec_nn; degree=0) isa TamerOp.GradedComplex
        @test TamerOp.encode(data, spec_apx; degree=0) isa TamerOp.GradedComplex
    else
        @test DI._pointcloud_nn_backend(spec_auto) == :bruteforce
        spec_brute = TamerOp.FiltrationSpec(
            kind=:rips,
            max_dim=1,
            knn=2,
            nn_backend=:bruteforce,
            construction=TamerOp.ConstructionOptions(; sparsify=:knn, output_stage=:graded_complex),
        )
        G_auto = TamerOp.encode(data, spec_auto; degree=0)
        G_brute = TamerOp.encode(data, spec_brute; degree=0)
        @test G_auto.cells_by_dim == G_brute.cells_by_dim
        @test G_auto.grades == G_brute.grades
        @test_throws ArgumentError TamerOp.encode(data, spec_nn; degree=0)
        @test_throws ArgumentError TamerOp.encode(data, spec_apx; degree=0)
    end
end

@testset "Data pipeline: NN runtime backend resolution cache contract" begin
    old_cache = DI._POINTCLOUD_BACKEND_RESOLVE_CACHE_ENABLED[]
    try
        DI._POINTCLOUD_BACKEND_RESOLVE_CACHE_ENABLED[] = true
        if DI._have_pointcloud_nn_backend()
            @test DI._resolve_pointcloud_runtime_backend(:auto, 2000, 32, 1) == :approx
            @test DI._resolve_pointcloud_runtime_backend(:auto, 2000, 32, 2) == :nearestneighbors
            @test DI._resolve_pointcloud_runtime_backend(:auto, 320, 8, 1) == :nearestneighbors
        else
            @test DI._resolve_pointcloud_runtime_backend(:auto, 2000, 32, 1) == :bruteforce
            @test DI._resolve_pointcloud_runtime_backend(:auto, 2000, 32, 2) == :bruteforce
            @test DI._resolve_pointcloud_runtime_backend(:auto, 320, 8, 1) == :bruteforce
        end
        @test DI._resolve_pointcloud_runtime_backend(:bruteforce, 2000, 32, 1) == :bruteforce
        @test DI._resolve_pointcloud_runtime_backend(:nearestneighbors, 2000, 32, 1) == :nearestneighbors
    finally
        DI._POINTCLOUD_BACKEND_RESOLVE_CACHE_ENABLED[] = old_cache
    end
end

@testset "Data pipeline: Delaunay backend contract/parity + cache" begin
    pts = TamerOp.PointCloud([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.25, 0.55],
        [0.65, 0.25],
    ])
    construction = TamerOp.ConstructionOptions(; output_stage=:graded_complex)
    spec_auto = TamerOp.FiltrationSpec(kind=:alpha, max_dim=2, delaunay_backend=:auto, construction=construction)
    spec_naive = TamerOp.FiltrationSpec(kind=:alpha, max_dim=2, delaunay_backend=:naive, construction=construction)
    spec_fast = TamerOp.FiltrationSpec(kind=:alpha, max_dim=2, delaunay_backend=:fast, construction=construction)

    @test_throws ArgumentError DI._pointcloud_delaunay_backend(
        TamerOp.FiltrationSpec(kind=:alpha, max_dim=2, delaunay_backend=:bad),
    )

    auto_backend = DI._pointcloud_delaunay_backend(spec_auto)
    if auto_backend == :fast
        spec_fast_st = TamerOp.FiltrationSpec(
            kind=:alpha,
            max_dim=2,
            delaunay_backend=:fast,
            construction=TamerOp.ConstructionOptions(; output_stage=:simplex_tree),
        )
        spec_naive_st = TamerOp.FiltrationSpec(
            kind=:alpha,
            max_dim=2,
            delaunay_backend=:naive,
            construction=TamerOp.ConstructionOptions(; output_stage=:simplex_tree),
        )
        st_fast = TamerOp.encode(pts, spec_fast_st; degree=0)
        st_naive = TamerOp.encode(pts, spec_naive_st; degree=0)
        @test _canon_simplex_tree(st_fast) == _canon_simplex_tree(st_naive)
    else
        @test auto_backend == :naive
        G_auto = TamerOp.encode(pts, spec_auto; degree=0)
        G_naive = TamerOp.encode(pts, spec_naive; degree=0)
        @test G_auto.cells_by_dim == G_naive.cells_by_dim
        @test G_auto.grades == G_naive.grades
        @test_throws ArgumentError TamerOp.encode(pts, spec_fast; degree=0)
    end

    old_cache = DI._POINTCLOUD_DELAUNAY_CACHE_ENABLED[]
    try
        DI._POINTCLOUD_DELAUNAY_CACHE_ENABLED[] = true
        DI._clear_pointcloud_delaunay_cache!()
        p1 = DI._packed_delaunay_simplices(pts.points, spec_naive; max_dim=2)
        p2 = DI._packed_delaunay_simplices(pts.points, spec_naive; max_dim=2)
        @test p1 === p2
        e1 = DI._packed_delaunay_entry(pts.points, spec_naive; max_dim=2)
        @test e1.edge_boundary === nothing
        @test e1.tri_boundary === nothing
        DI._ensure_packed_delaunay_boundaries!(e1, length(pts.points), 2)
        b1 = e1.edge_boundary
        b2 = e1.tri_boundary
        @test b1 !== nothing
        @test b2 !== nothing
        e2 = DI._packed_delaunay_entry(pts.points, spec_naive; max_dim=2)
        @test e1 === e2
        @test e2.edge_boundary === b1
        @test e2.tri_boundary === b2
        G1 = TamerOp.encode(pts, spec_naive; degree=0, stage=:graded_complex)
        G2 = TamerOp.encode(pts, spec_naive; degree=0, stage=:graded_complex)
        @test G1.boundaries[1] === G2.boundaries[1]
        @test G1.boundaries[2] === G2.boundaries[2]
        TamerOp.encode(pts, spec_naive; degree=0)
        n1 = lock(DI._POINTCLOUD_DELAUNAY_CACHE_LOCK) do
            length(DI._POINTCLOUD_DELAUNAY_CACHE)
        end
        TamerOp.encode(pts, spec_naive; degree=0)
        n2 = lock(DI._POINTCLOUD_DELAUNAY_CACHE_LOCK) do
            length(DI._POINTCLOUD_DELAUNAY_CACHE)
        end
        @test n1 == 1
        @test n2 == 1
    finally
        DI._POINTCLOUD_DELAUNAY_CACHE_ENABLED[] = old_cache
        DI._clear_pointcloud_delaunay_cache!()
    end
end

@testset "Data pipeline: Delaunay packed materialization preserves max_dim" begin
    pts = TamerOp.PointCloud([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.25, 0.55],
        [0.65, 0.25],
    ])
    for md in (0, 1, 2)
        spec = TamerOp.FiltrationSpec(
            kind=:alpha,
            max_dim=md,
            construction=TamerOp.ConstructionOptions(; output_stage=:graded_complex),
        )
        G = TamerOp.encode(pts, spec; degree=0)
        @test G isa TamerOp.GradedComplex
        @test length(G.cells_by_dim) == md + 1
        @test length(G.boundaries) == md
    end
end

@testset "Data pipeline: alpha uses true squared-radius edge grades" begin
    pts = TamerOp.PointCloud([
        [0.0, 0.0],
        [2.0, 0.0],
        [0.5, 0.1],
    ])
    spec = TamerOp.FiltrationSpec(
        kind=:alpha,
        max_dim=2,
        delaunay_backend=:naive,
        construction=TamerOp.ConstructionOptions(; output_stage=:simplex_tree),
    )

    packed = DI._packed_delaunay_simplices(pts.points, spec; max_dim=2)
    @test packed.edges == [(1, 2), (1, 3), (2, 3)]
    @test packed.triangles == [(1, 2, 3)]

    edge_alpha_sq = DI._alpha_edge_grades_sq(pts.points, packed)
    tri_alpha_sq = packed.tri_radius[1]^2

    half_sq_long = DI._euclidean_distance(pts.points[1], pts.points[2])^2 / 4
    half_sq_short1 = DI._euclidean_distance(pts.points[1], pts.points[3])^2 / 4
    half_sq_short2 = DI._euclidean_distance(pts.points[2], pts.points[3])^2 / 4

    @test isapprox(edge_alpha_sq[1], tri_alpha_sq; atol=1e-10, rtol=0.0)
    @test edge_alpha_sq[1] > half_sq_long
    @test isapprox(edge_alpha_sq[2], half_sq_short1; atol=1e-10, rtol=0.0)
    @test isapprox(edge_alpha_sq[3], half_sq_short2; atol=1e-10, rtol=0.0)

    st = TamerOp.encode(pts, spec; degree=0, stage=:simplex_tree)
    grade_map = Dict(
        Tuple(collect(DI.simplex_vertices(st, i))) => first(first(collect(DI.simplex_grades(st, i))))
        for i in 1:DI.simplex_count(st)
    )

    @test grade_map[(1,)] == 0.0
    @test grade_map[(2,)] == 0.0
    @test grade_map[(3,)] == 0.0
    @test isapprox(grade_map[(1, 2)], tri_alpha_sq; atol=1e-10, rtol=0.0)
    @test isapprox(grade_map[(1, 3)], half_sq_short1; atol=1e-10, rtol=0.0)
    @test isapprox(grade_map[(2, 3)], half_sq_short2; atol=1e-10, rtol=0.0)
    @test isapprox(grade_map[(1, 2, 3)], tri_alpha_sq; atol=1e-10, rtol=0.0)
end

@testset "Data pipeline: Delaunay packed simplices are canonical/unique" begin
    DI._have_pointcloud_delaunay_backend() || begin
        @test true
        return
    end

    rng = Random.MersenneTwister(0xD4B4)
    pts = TamerOp.PointCloud([randn(rng, 2) for _ in 1:128])
    spec = TamerOp.FiltrationSpec(kind=:alpha, max_dim=2, delaunay_backend=:fast)
    packed = DI._packed_delaunay_simplices(pts.points, spec; max_dim=2)

    @test length(packed.edge_radius) == length(packed.edges)
    @test length(packed.tri_radius) == length(packed.triangles)
    @test all(e -> e[1] < e[2], packed.edges)
    @test all(t -> (t[1] < t[2] && t[2] < t[3]), packed.triangles)
    @test length(unique(packed.edges)) == length(packed.edges)
    @test length(unique(packed.triangles)) == length(packed.triangles)
end

@testset "Data pipeline: construction output_stage routing" begin
    data = TamerOp.PointCloud([[0.0], [1.0]])

    filt_st = DI.RipsFiltration(
        max_dim=1,
        construction=TamerOp.ConstructionOptions(; output_stage=:simplex_tree),
    )
    ST = TamerOp.encode(data, filt_st; degree=0)
    @test ST isa DI.SimplexTreeMulti
    @test DI.simplex_count(ST) == 3
    @test DI.max_simplex_dim(ST) == 1
    @test collect(DI.simplex_vertices(ST, 1)) == [1]
    @test collect(DI.simplex_vertices(ST, 2)) == [2]
    @test collect(DI.simplex_vertices(ST, 3)) == [1, 2]
    @test collect(DI.simplex_grades(ST, 3)) == [(1.0,)]

    filt_gc = DI.RipsFiltration(
        max_dim=1,
        construction=TamerOp.ConstructionOptions(; output_stage=:graded_complex),
    )
    G = TamerOp.encode(data, filt_gc; degree=0)
    @test G isa TamerOp.GradedComplex

    filt_cc = DI.RipsFiltration(
        max_dim=1,
        construction=TamerOp.ConstructionOptions(; output_stage=:cochain),
    )
    C = TamerOp.encode(data, filt_cc; degree=0)
    @test C isa MC.ModuleCochainComplex
    for (u, v) in FF.cover_edges(C.terms[1].Q)
        @test C.terms[1].edge_maps[u, v] isa AbstractMatrix
    end
    @test C.diffs[1].comps[1] isa AbstractMatrix

    filt_mod = DI.RipsFiltration(
        max_dim=1,
        construction=TamerOp.ConstructionOptions(; output_stage=:module),
    )
    M = TamerOp.encode(data, filt_mod; degree=0)
    @test M isa MD.PModule

    spec_mod = TamerOp.FiltrationSpec(
        kind=:rips,
        max_dim=1,
        construction=(sparsify=:none, collapse=:none, output_stage=:module,
                      budget=(max_simplices=nothing, max_edges=nothing, memory_budget_bytes=nothing)),
    )
    M2 = TamerOp.encode(data, spec_mod; degree=0)
    @test M2 isa MD.PModule
end

@testset "Data pipeline: simplex-tree stage rejects non-simplicial cubical ingestion" begin
    img = TamerOp.ImageNd([0.0 1.0; 2.0 3.0])
    spec = TamerOp.FiltrationSpec(
        kind=:lower_star,
        construction=TamerOp.ConstructionOptions(; output_stage=:simplex_tree),
    )
    @test_throws ArgumentError TamerOp.encode(img, spec; degree=0)
end

@testset "Data pipeline: flange emission (Zn)" begin
    data = TamerOp.PointCloud([[0.0], [1.0]])
    spec = TamerOp.FiltrationSpec(kind=:rips, max_dim=1, axes=([0, 1],), axis_kind=:zn)
    FG = TamerOp.encode(data, spec; degree=0, stage=:flange)
    @test FG.n == 1

    spec = TamerOp.FiltrationSpec(kind=:rips, max_dim=1, axes=([0.0, 1.0],), axis_kind=:rn)
    @test_throws Exception TamerOp.encode(data, spec; degree=0, stage=:flange)

    spec = TamerOp.FiltrationSpec(kind=:rips, max_dim=1, axes=([0.0, 1.0],))
    @test_throws Exception TamerOp.encode(data, spec; degree=0, stage=:flange)
end

@testset "Data pipeline: graded complex escape hatch" begin
    # single vertex
    cells = [Int[1]]
    boundaries = SparseMatrixCSC{Int,Int}[]
    grades = [Float64[0.0]]
    G = TamerOp.GradedComplex(cells, boundaries, grades)
    spec = TamerOp.FiltrationSpec(kind=:graded, axes=([0.0, 1.0],))
    enc = TamerOp.encode(G, spec; degree=0)
    @test DI.module_dims(enc.M) == [1, 1]

    # single edge between two vertices
    cells = [Int[1, 2], Int[1]]
    B1 = sparse([1, 2], [1, 1], [1, -1], 2, 1)
    boundaries = [B1]
    grades = [Float64[0.0], Float64[0.0], Float64[1.0]]
    G = TamerOp.GradedComplex(cells, boundaries, grades)
    enc = TamerOp.encode(G, spec; degree=0)
    @test DI.module_dims(enc.M) == [2, 1]
    ri = Inv.rank_invariant(DI.materialize_module(enc.M), OPT.InvariantOptions(); store_zeros=true)
    @test ri[(1, 2)] == 1

    # filled triangle: H1 appears at 1 and dies at 2
    cells = [Int[1, 2, 3], Int[1, 2, 3], Int[1]]
    B1 = sparse([1, 2, 1, 3, 2, 3],
                [1, 1, 2, 2, 3, 3],
                [-1, 1, -1, 1, -1, 1], 3, 3)
    B2 = sparse([1, 2, 3], [1, 1, 1], [1, -1, 1], 3, 1)
    boundaries = [B1, B2]
    grades = [Float64[0.0], Float64[0.0], Float64[0.0],
              Float64[1.0], Float64[1.0], Float64[1.0],
              Float64[2.0]]
    G = TamerOp.GradedComplex(cells, boundaries, grades)
    spec2 = TamerOp.FiltrationSpec(kind=:graded, axes=([0.0, 1.0, 2.0],))
    enc1 = TamerOp.encode(G, spec2; degree=1)
    @test DI.module_dims(enc1.M) == [0, 1, 0]
end

@testset "Data pipeline: cellular boundaries are natural under inclusion" begin
    # At 0 there is one edge; at 1 a third vertex and the filled triangle enter.
    # Extending a chain by zero is covariant and commutes with the boundary.
    # Extending a cochain by zero does not commute with the transposed boundary.
    B1 = sparse([-1 -1 0; 1 0 -1; 0 1 1])
    B2 = sparse(reshape([1, -1, 1], 3, 1))
    G = DT.GradedComplex([[1, 2, 3], [1, 2, 3], [1]], [B1, B2],
        [(0.0,), (0.0,), (1.0,), (0.0,), (1.0,), (1.0,), (1.0,)])
    axes = ([0.0, 1.0],)
    spec = OPT.FiltrationSpec(kind=:graded, axes=axes)
    for field in FIELDS_FULL
        K = CM.coeff_type(field)
        C = DI.encode(G, spec; field=field, stage=:cochain)
        encoded = DI.encode(G, spec; field=field, stage=:encoded_complex)
        L = TO.encoding_complex(encoded)
        materialized = RES._materialize_complex(L)
        @test L.tmin == -2
        @test L.tmax == 0
        @test DI._lazy_term(L, -1).dims == [1, 3]
        @test Matrix(DI._lazy_diff(L, -1).comps[2]) == K[CM.coerce(field, x) for x in B1]

        for chain_cochain in (C, materialized)
            @test MC.degree_range(chain_cochain) == -2:0
            @test MC.component(chain_cochain, -2).dims == [0, 1]
            @test MC.component(chain_cochain, -1).dims == [1, 3]
            @test MC.component(chain_cochain, 0).dims == [2, 3]
            @test Matrix(MC.differential(chain_cochain, -1).comps[1]) ==
                  reshape(K[CM.coerce(field, -1), CM.coerce(field, 1)], 2, 1)
            @test Matrix(MC.differential(chain_cochain, -1).comps[2]) ==
                  K[CM.coerce(field, x) for x in B1]
            @test Matrix(MC.differential(chain_cochain, -2).comps[2]) ==
                  K[CM.coerce(field, x) for x in B2]
            @test MC.check_module_complex(chain_cochain; throw=true).valid
            for t in -2:-1
                d = MC.differential(chain_cochain, t)
                @test MD.check_morphism(d; throw=true).valid
                @test d.comps[2] * MD.structure_map(d.dom; source=1, target=2) ==
                      MD.structure_map(d.cod; source=1, target=2) * d.comps[1]
            end
            H0 = MC.cohomology_module(chain_cochain, 0)
            @test H0.dims == [1, 1]
            @test FL.rank(field, MD.structure_map(H0; source=1, target=2)) == 1
            @test MC.cohomology_module(chain_cochain, -1).dims == [0, 0]
        end
        @test DI.encode(G, spec; field=field, degree=0, stage=:module).dims == [1, 1]
        @test_throws ArgumentError DI.encode(G, spec; field=field, degree=-1)
    end
end

@testset "Data pipeline: homology classes are born, persist and die under inclusion" begin
    # A path at 0 closes into a circle at 1, persists at 2, and is filled at 3.
    # Every grid value is a true query: all cell birth values occur on the grid.
    simplices = [[[1], [2], [3]], [[1, 2], [1, 3], [2, 3]], [[1, 2, 3]]]
    grades = [(0.0,), (0.0,), (0.0,), (0.0,), (0.0,), (1.0,), (3.0,)]
    B1 = sparse([-1 -1 0; 1 0 -1; 0 1 1])
    B2 = sparse(reshape([1, -1, 1], 3, 1))
    G = DT.GradedComplex([[1, 2, 3], [1, 2, 3], [1]], [B1, B2], grades)
    ST = DI._simplex_tree_multi_from_simplices(simplices, grades)
    axes = ([0.0, 1.0, 2.0, 3.0],)
    spec = OPT.FiltrationSpec(kind=:graded, axes=axes)
    expected_h1 = [0, 1, 1, 0]
    function check_circle_interval(M, field)
        @test M.dims == expected_h1
        @test MD.check_module(M; throw=true).valid
        for u in 1:4, v in u:4
            expected_rank = (u in (2, 3) && v in (2, 3)) ? 1 : 0
            @test FL.rank(field, MD.structure_map(M; source=u, target=v)) == expected_rank
        end
    end
    for field in FIELDS_FULL
        for data in (G, ST)
            for cache in (nothing, CM.SessionCache())
                C = DI.encode(data, spec; degree=1, field=field, stage=:cochain, cache=cache)
                @test MC.degree_range(C) == -2:0
                check_circle_interval(MC.cohomology_module(C, -1), field)
                @test MC.cohomology_module(C, 1).dims == zeros(Int, 4)
                encoded = DI.encode(data, spec; degree=1, field=field, cache=cache)
                check_circle_interval(_enc_module(encoded), field)
                check_circle_interval(DI.encode(data, spec; degree=1, field=field,
                                               stage=:module, cache=cache), field)
                dims = DI.encode(data, spec; degree=1, field=field,
                                 stage=:cohomology_dims, cache=cache)
                @test dims.dims == expected_h1
                complex_result = DI.encode(data, spec; degree=1, field=field,
                                           stage=:encoded_complex, cache=cache)
                L = TO.encoding_complex(complex_result)
                @test DI._cohomology_dims_from_lazy(L, 1) == expected_h1
                check_circle_interval(DI._cohomology_module_from_lazy_generic(L, 1), field)
                check_circle_interval(MC.cohomology_module(RES._materialize_complex(L), -1), field)
                repeated = DI.encode(data, spec; degree=1, field=field, cache=cache)
                check_circle_interval(_enc_module(repeated), field)
                if cache !== nothing
                    @test repeated.H === encoded.H
                end
            end
        end

        # Removing the filling face isolates the graph shortcut: H1 is ker d_1.
        graph = DT.GradedComplex([[1, 2, 3], [1, 2, 3]], [B1], grades[1:6])
        old_gate = DI._H1_KERNEL_FASTPATH[]
        try
            for enabled in (false, true)
                DI._H1_KERNEL_FASTPATH[] = enabled
                H1 = DI.encode(graph, spec; degree=1, field=field, stage=:module)
                @test H1.dims == [0, 1, 1, 1]
                @test FL.rank(field, MD.structure_map(H1; source=2, target=4)) == 1
            end
        finally
            DI._H1_KERNEL_FASTPATH[] = old_gate
        end
    end
end

@testset "Data pipeline: simplex-tree escape hatch" begin
    data = TamerOp.PointCloud([[0.0], [1.0], [2.0]])
    spec_st = TamerOp.FiltrationSpec(
        kind=:rips,
        max_dim=1,
        construction=TamerOp.ConstructionOptions(; output_stage=:simplex_tree),
    )
    st = TamerOp.encode(data, spec_st; degree=0)
    @test st isa DI.SimplexTreeMulti
    @test DI.simplex_count(st) == 6
    @test DI.max_simplex_dim(st) == 1

    spec_enc = TamerOp.FiltrationSpec(kind=:rips, axes=([0.0, 1.0, 2.0],))
    enc = TamerOp.encode(st, spec_enc; degree=0)
    @test enc isa RES.EncodingResult
    @test DI.module_dims(enc.M) == [3, 1, 1]
end

@testset "Data pipeline: simplex-tree cochain/module parity" begin
    data = TamerOp.PointCloud([[0.0], [1.0], [2.0]])
    st = TamerOp.encode(
        data,
        TamerOp.FiltrationSpec(
            kind=:rips,
            max_dim=1,
            construction=TamerOp.ConstructionOptions(; output_stage=:simplex_tree),
        );
        degree=0,
    )
    @test st isa DI.SimplexTreeMulti

    spec_mod = TamerOp.FiltrationSpec(
        kind=:rips,
        max_dim=1,
        axes=([0.0, 1.0, 2.0],),
        construction=TamerOp.ConstructionOptions(; output_stage=:module),
    )
    M_raw = TamerOp.encode(data, spec_mod; degree=0)
    M_tree = TamerOp.encode(st, spec_mod; degree=0)
    @test M_tree isa MD.PModule
    @test M_tree.dims == M_raw.dims
    for (u, v) in FF.cover_edges(M_raw.Q)
        @test Array(M_tree.edge_maps[u, v]) == Array(M_raw.edge_maps[u, v])
    end

    spec_cc = TamerOp.FiltrationSpec(
        kind=:rips,
        max_dim=1,
        axes=([0.0, 1.0, 2.0],),
        construction=TamerOp.ConstructionOptions(; output_stage=:cochain),
    )
    C_raw = TamerOp.encode(data, spec_cc; degree=0)
    C_tree = TamerOp.encode(st, spec_cc; degree=0)
    @test C_tree isa MC.ModuleCochainComplex
    @test length(C_tree.terms) == length(C_raw.terms)
    @test C_tree.terms[1].dims == C_raw.terms[1].dims
    @test C_tree.terms[2].dims == C_raw.terms[2].dims
    @test all(Array(C_tree.diffs[1].comps[i]) == Array(C_raw.diffs[1].comps[i]) for i in eachindex(C_raw.diffs[1].comps))
end

@testset "Data pipeline: lazy default parity vs explicit cochain" begin
    data = TamerOp.PointCloud([[0.0], [1.0], [2.0]])
    spec_default = TamerOp.FiltrationSpec(kind=:rips, max_dim=1, axes=([0.0, 1.0, 2.0],))
    enc_lazy = TamerOp.encode(data, spec_default; degree=0)

    spec_cochain = TamerOp.FiltrationSpec(
        kind=:rips,
        max_dim=1,
        axes=([0.0, 1.0, 2.0],),
        construction=TamerOp.ConstructionOptions(; output_stage=:cochain),
    )
    C_full = TamerOp.encode(data, spec_cochain; degree=0)
    M_full = MC.cohomology_module(C_full, 0)

    M_lazy = _enc_module(enc_lazy)
    @test M_lazy.dims == M_full.dims
    for (u, v) in FF.cover_edges(M_lazy.Q)
        A_lazy = Array(M_lazy.edge_maps[u, v])
        A_full = Array(M_full.edge_maps[u, v])
        @test size(A_lazy) == size(A_full)
        @test TamerOp.FieldLinAlg.rank(M_lazy.field, A_lazy) == TamerOp.FieldLinAlg.rank(M_full.field, A_full)
    end
end

@testset "Data pipeline: graded-complex lazy parity vs explicit cochain" begin
    cells = [Int[1, 2], Int[1]]
    boundaries = [sparse([1, 2], [1, 1], [1, -1], 2, 1)]
    grades = [Float64[0.0], Float64[0.0], Float64[1.0]]
    G = TamerOp.GradedComplex(cells, boundaries, grades)

    spec_default = TamerOp.FiltrationSpec(kind=:graded, axes=([0.0, 1.0],))
    enc_lazy = TamerOp.encode(G, spec_default; degree=0)
    axes = ([0.0, 1.0],)
    P = DI.poset_from_axes(axes)
    C_full = DI.cochain_complex_from_graded_complex(G, P, axes; field=CM.QQField())
    M_full = MC.cohomology_module(C_full, 0)

    M_lazy = _enc_module(enc_lazy)
    @test M_lazy.dims == M_full.dims
    for (u, v) in FF.cover_edges(M_lazy.Q)
        A_lazy = Array(M_lazy.edge_maps[u, v])
        A_full = Array(M_full.edge_maps[u, v])
        @test size(A_lazy) == size(A_full)
        @test TamerOp.FieldLinAlg.rank(M_lazy.field, A_lazy) == TamerOp.FieldLinAlg.rank(M_full.field, A_full)
    end
end

@testset "Data pipeline: low-dim H0 fast path parity" begin
    data = TamerOp.PointCloud([[0.0], [0.4], [0.9], [1.3]])
    spec_gc = TamerOp.FiltrationSpec(
        kind=:rips,
        max_dim=1,
        axes=([0.0, 0.5, 1.0, 1.5],),
        construction=OPT.ConstructionOptions(; output_stage=:graded_complex),
    )
    G = TamerOp.encode(data, spec_gc; degree=0)
    axes = spec_gc.params[:axes]
    P = DI.poset_from_axes(axes)

    L_fast = DI._lazy_cochain_complex_from_graded_complex(G, P, axes; field=CM.QQField())
    M_fast = DI._cohomology_module_from_lazy(L_fast, 0)
    L_generic = DI._lazy_cochain_complex_from_graded_complex(G, P, axes; field=CM.QQField())
    M_generic = DI._cohomology_module_from_lazy_generic(L_generic, 0)
    @test M_fast.dims == M_generic.dims
    for (u, v) in FF.cover_edges(M_fast.Q)
        @test FL.rank(CM.QQField(), M_fast.edge_maps[u, v]) ==
              FL.rank(CM.QQField(), M_generic.edge_maps[u, v])
    end

    # Non-edge boundary columns still agree with the generic local cohomology path.
    cells_bad = [Int[1, 2], Int[1]]
    boundaries_bad = [sparse([1], [1], [1], 2, 1)]
    grades_bad = [Float64[0.0], Float64[0.0], Float64[1.0]]
    G_bad = TamerOp.GradedComplex(cells_bad, boundaries_bad, grades_bad)
    axes_bad = ([0.0, 1.0],)
    P_bad = DI.poset_from_axes(axes_bad)
    L_bad = DI._lazy_cochain_complex_from_graded_complex(G_bad, P_bad, axes_bad; field=CM.QQField())
    M_bad = DI._cohomology_module_from_lazy(L_bad, 0)
    L_bad_generic = DI._lazy_cochain_complex_from_graded_complex(G_bad, P_bad, axes_bad; field=CM.QQField())
    M_bad_generic = DI._cohomology_module_from_lazy_generic(L_bad_generic, 0)
    @test M_bad.dims == M_bad_generic.dims
    for (u, v) in FF.cover_edges(M_bad.Q)
        @test FL.rank(CM.QQField(), M_bad.edge_maps[u, v]) ==
              FL.rank(CM.QQField(), M_bad_generic.edge_maps[u, v])
    end
end

@testset "Data pipeline: H0 cokernel path parity for max_dim>1" begin
    data = TamerOp.PointCloud([[0.0], [0.25], [0.5], [0.75], [1.0]])
    spec_gc = TamerOp.FiltrationSpec(
        kind=:rips,
        max_dim=2,
        axes=([0.0, 0.4, 0.8, 1.2],),
        construction=OPT.ConstructionOptions(; output_stage=:graded_complex),
    )
    G = TamerOp.encode(data, spec_gc; degree=0)
    axes = spec_gc.params[:axes]
    P = DI.poset_from_axes(axes)

    old_chain = DI._H0_CHAIN_SWEEP_FASTPATH[]
    old_min_pos = DI._H0_UNIONFIND_MIN_POS_VERTICES[]
    old_min_v = DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_VERTICES[]
    old_min_e = DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_EDGES[]
    try
        DI._H0_CHAIN_SWEEP_FASTPATH[] = false
        DI._H0_UNIONFIND_MIN_POS_VERTICES[] = typemax(Int)
        DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_VERTICES[] = typemax(Int)
        DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_EDGES[] = typemax(Int)
        L_fast = DI._lazy_cochain_complex_from_graded_complex(G, P, axes; field=CM.QQField())
        M_fast = DI._cohomology_module_from_lazy(L_fast, 0)
        L_generic = DI._lazy_cochain_complex_from_graded_complex(G, P, axes; field=CM.QQField())
        M_generic = DI._cohomology_module_from_lazy_generic(L_generic, 0)
        @test M_fast.dims == M_generic.dims
        for (u, v) in FF.cover_edges(M_fast.Q)
            @test FL.rank(CM.QQField(), M_fast.edge_maps[u, v]) ==
                  FL.rank(CM.QQField(), M_generic.edge_maps[u, v])
        end
    finally
        DI._H0_CHAIN_SWEEP_FASTPATH[] = old_chain
        DI._H0_UNIONFIND_MIN_POS_VERTICES[] = old_min_pos
        DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_VERTICES[] = old_min_v
        DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_EDGES[] = old_min_e
    end
end

@testset "Data pipeline: H1 kernel fast path parity" begin
    n = 30
    cells0 = collect(1:n)
    cells1 = collect(1:n) # path edges + one closing edge
    I = Int[]
    J = Int[]
    V = Int[]
    sizehint!(I, 2n)
    sizehint!(J, 2n)
    sizehint!(V, 2n)
    for e in 1:(n - 1)
        push!(I, e); push!(J, e); push!(V, 1)
        push!(I, e + 1); push!(J, e); push!(V, -1)
    end
    # Closing edge (n -> 1) creates a 1-cycle.
    push!(I, 1); push!(J, n); push!(V, 1)
    push!(I, n); push!(J, n); push!(V, -1)
    B = sparse(I, J, V, n, n)

    grades = [i <= n ? [0.0] : [0.5] for i in 1:(n + n)]
    G = TamerOp.GradedComplex([cells0, cells1], [B], grades)
    axes = (collect(range(0.0, stop=1.0, length=20)),)
    P = DI.poset_from_axes(axes)

    old_h1 = DI._H1_KERNEL_FASTPATH[]
    try
        DI._H1_KERNEL_FASTPATH[] = true
        L_fast = DI._lazy_cochain_complex_from_graded_complex(G, P, axes; field=CM.F2())
        M_fast = DI._cohomology_module_from_lazy(L_fast, 1)

        DI._H1_KERNEL_FASTPATH[] = false
        L_generic = DI._lazy_cochain_complex_from_graded_complex(G, P, axes; field=CM.F2())
        M_generic = DI._cohomology_module_from_lazy(L_generic, 1)

        @test M_fast.dims == M_generic.dims
        for (u, v) in FF.cover_edges(M_fast.Q)
            @test TamerOp.FieldLinAlg.rank(CM.F2(), M_fast.edge_maps[u, v]) ==
                  TamerOp.FieldLinAlg.rank(CM.F2(), M_generic.edge_maps[u, v])
        end
    finally
        DI._H1_KERNEL_FASTPATH[] = old_h1
    end
end

@testset "Data pipeline: solve-check fast path parity (generic H1)" begin
    n = 30
    cells0 = collect(1:n)
    cells1 = collect(1:n)
    I = Int[]
    J = Int[]
    V = Int[]
    sizehint!(I, 2n)
    sizehint!(J, 2n)
    sizehint!(V, 2n)
    for e in 1:(n - 1)
        push!(I, e); push!(J, e); push!(V, 1)
        push!(I, e + 1); push!(J, e); push!(V, -1)
    end
    push!(I, 1); push!(J, n); push!(V, 1)
    push!(I, n); push!(J, n); push!(V, -1)
    B = sparse(I, J, V, n, n)
    grades = [i <= n ? [0.0] : [0.5] for i in 1:(n + n)]
    G = TamerOp.GradedComplex([cells0, cells1], [B], grades)
    axes = (collect(range(0.0, stop=1.0, length=20)),)
    P = DI.poset_from_axes(axes)

    old_h1 = DI._H1_KERNEL_FASTPATH[]
    old_solve = AC._FAST_SOLVE_NO_CHECK[]
    try
        DI._H1_KERNEL_FASTPATH[] = false

        AC._FAST_SOLVE_NO_CHECK[] = true
        L_fast = DI._lazy_cochain_complex_from_graded_complex(G, P, axes; field=CM.F2())
        M_fast = DI._cohomology_module_from_lazy(L_fast, 1)

        AC._FAST_SOLVE_NO_CHECK[] = false
        L_base = DI._lazy_cochain_complex_from_graded_complex(G, P, axes; field=CM.F2())
        M_base = DI._cohomology_module_from_lazy(L_base, 1)

        @test M_fast.dims == M_base.dims
        for (u, v) in FF.cover_edges(M_fast.Q)
            @test FL.rank(CM.F2(), M_fast.edge_maps[u, v]) ==
                  FL.rank(CM.F2(), M_base.edge_maps[u, v])
        end
    finally
        DI._H1_KERNEL_FASTPATH[] = old_h1
        AC._FAST_SOLVE_NO_CHECK[] = old_solve
    end
end

@testset "Data pipeline: H0 active-chain incremental parity" begin
    old_inc = DI._H0_ACTIVE_CHAIN_INCREMENTAL[]
    old_min_pos = DI._H0_UNIONFIND_MIN_POS_VERTICES[]
    old_min_v = DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_VERTICES[]
    old_min_e = DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_EDGES[]
    try
        DI._H0_UNIONFIND_MIN_POS_VERTICES[] = 0
        DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_VERTICES[] = 0
        DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_EDGES[] = 0
        data = TamerOp.PointCloud([[0.0], [0.3], [0.8], [1.1], [1.6], [2.0]])
        spec_gc = TamerOp.FiltrationSpec(
            kind=:rips,
            max_dim=1,
            axes=([0.0, 0.4, 0.8, 1.2, 1.6, 2.0],),
            construction=OPT.ConstructionOptions(; output_stage=:graded_complex),
        )
        G = TamerOp.encode(data, spec_gc; degree=0)
        axes = spec_gc.params[:axes]
        P = DI.poset_from_axes(axes)

        DI._H0_ACTIVE_CHAIN_INCREMENTAL[] = true
        L_fast = DI._lazy_cochain_complex_from_graded_complex(G, P, axes; field=CM.F2())
        M_fast = DI._cohomology_module_from_lazy(L_fast, 0)

        DI._H0_ACTIVE_CHAIN_INCREMENTAL[] = false
        L_base = DI._lazy_cochain_complex_from_graded_complex(G, P, axes; field=CM.F2())
        M_base = DI._cohomology_module_from_lazy(L_base, 0)

        @test M_fast.dims == M_base.dims
        for (u, v) in FF.cover_edges(M_fast.Q)
            @test FL.rank(CM.F2(), M_fast.edge_maps[u, v]) ==
                  FL.rank(CM.F2(), M_base.edge_maps[u, v])
        end
    finally
        DI._H0_ACTIVE_CHAIN_INCREMENTAL[] = old_inc
        DI._H0_UNIONFIND_MIN_POS_VERTICES[] = old_min_pos
        DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_VERTICES[] = old_min_v
        DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_EDGES[] = old_min_e
    end
end

@testset "Data pipeline: H0 active-chain incremental heuristic contract" begin
    old_min_pos = DI._H0_ACTIVE_CHAIN_INCREMENTAL_MIN_POS_VERTICES[]
    old_min_v = DI._H0_ACTIVE_CHAIN_INCREMENTAL_MIN_TOTAL_ACTIVE_VERTICES[]
    old_min_e = DI._H0_ACTIVE_CHAIN_INCREMENTAL_MIN_TOTAL_ACTIVE_EDGES[]
    try
        DI._H0_ACTIVE_CHAIN_INCREMENTAL_MIN_POS_VERTICES[] = 10
        DI._H0_ACTIVE_CHAIN_INCREMENTAL_MIN_TOTAL_ACTIVE_VERTICES[] = 100
        DI._H0_ACTIVE_CHAIN_INCREMENTAL_MIN_TOTAL_ACTIVE_EDGES[] = 200
        @test !DI._use_h0_active_chain_incremental(9, 1_000, 1_000)
        @test !DI._use_h0_active_chain_incremental(10, 99, 199)
        @test DI._use_h0_active_chain_incremental(10, 120, 50)
        @test DI._use_h0_active_chain_incremental(10, 10, 220)
    finally
        DI._H0_ACTIVE_CHAIN_INCREMENTAL_MIN_POS_VERTICES[] = old_min_pos
        DI._H0_ACTIVE_CHAIN_INCREMENTAL_MIN_TOTAL_ACTIVE_VERTICES[] = old_min_v
        DI._H0_ACTIVE_CHAIN_INCREMENTAL_MIN_TOTAL_ACTIVE_EDGES[] = old_min_e
    end
end

@testset "Data pipeline: degree-local t>=2 fast path parity" begin
    verts = 1:5
    edges = [(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (3, 4), (4, 5)]
    triangles = [(1, 2, 3), (1, 3, 4), (1, 4, 5)]

    I1 = Int[]
    J1 = Int[]
    V1 = Int[]
    for (j, (a, b)) in enumerate(edges)
        push!(I1, a); push!(J1, j); push!(V1, 1)
        push!(I1, b); push!(J1, j); push!(V1, -1)
    end
    B1 = sparse(I1, J1, V1, length(verts), length(edges))
    B2 = spzeros(Int, length(edges), length(triangles))

    cells = [collect(verts), collect(1:length(edges)), collect(1:length(triangles))]
    boundaries = [B1, B2]
    grades = vcat([Float64[0.0] for _ in verts],
                  [Float64[0.4] for _ in edges],
                  [Float64[0.8] for _ in triangles])
    G = TamerOp.GradedComplex(cells, boundaries, grades)
    axes = (collect(range(0.0, stop=1.0, length=20)),)
    P = DI.poset_from_axes(axes)

    old_local = DI._COHOMOLOGY_DEGREE_LOCAL_FASTPATH[]
    try
        DI._COHOMOLOGY_DEGREE_LOCAL_FASTPATH[] = true
        L_fast = DI._lazy_cochain_complex_from_graded_complex(G, P, axes; field=CM.F2())
        M_fast = DI._cohomology_module_from_lazy(L_fast, 2)

        DI._COHOMOLOGY_DEGREE_LOCAL_FASTPATH[] = false
        L_base = DI._lazy_cochain_complex_from_graded_complex(G, P, axes; field=CM.F2())
        M_base = DI._cohomology_module_from_lazy(L_base, 2)

        @test M_fast.dims == M_base.dims
        for (u, v) in FF.cover_edges(M_fast.Q)
            @test FL.rank(CM.F2(), M_fast.edge_maps[u, v]) ==
                  FL.rank(CM.F2(), M_base.edge_maps[u, v])
        end
    finally
        DI._COHOMOLOGY_DEGREE_LOCAL_FASTPATH[] = old_local
    end
end

@testset "Data pipeline: degree-local t=1 fast path parity" begin
    verts = 1:5
    edges = [(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (3, 4), (4, 5)]
    triangles = [(1, 2, 3), (1, 3, 4), (1, 4, 5)]

    I1 = Int[]
    J1 = Int[]
    V1 = Int[]
    for (j, (a, b)) in enumerate(edges)
        push!(I1, a); push!(J1, j); push!(V1, 1)
        push!(I1, b); push!(J1, j); push!(V1, -1)
    end
    B1 = sparse(I1, J1, V1, length(verts), length(edges))
    B2 = spzeros(Int, length(edges), length(triangles))

    cells = [collect(verts), collect(1:length(edges)), collect(1:length(triangles))]
    boundaries = [B1, B2]
    grades = vcat([Float64[0.0] for _ in verts],
                  [Float64[0.4] for _ in edges],
                  [Float64[0.8] for _ in triangles])
    G = TamerOp.GradedComplex(cells, boundaries, grades)
    axes = (collect(range(0.0, stop=1.0, length=28)),)
    P = DI.poset_from_axes(axes)

    old_local = DI._COHOMOLOGY_DEGREE_LOCAL_FASTPATH[]
    old_all_t = DI._COHOMOLOGY_DEGREE_LOCAL_ALL_T[]
    old_t1_min_pos = DI._COHOMOLOGY_DEGREE_LOCAL_T1_MIN_POS_VERTICES[]
    old_t1_min_dim1 = DI._COHOMOLOGY_DEGREE_LOCAL_T1_MIN_TOTAL_ACTIVE_DIM1[]
    old_t1_min_dim2 = DI._COHOMOLOGY_DEGREE_LOCAL_T1_MIN_TOTAL_ACTIVE_DIM2[]
    old_h1 = DI._H1_KERNEL_FASTPATH[]
    try
        DI._H1_KERNEL_FASTPATH[] = false
        DI._COHOMOLOGY_DEGREE_LOCAL_FASTPATH[] = true
        DI._COHOMOLOGY_DEGREE_LOCAL_ALL_T[] = true
        DI._COHOMOLOGY_DEGREE_LOCAL_T1_MIN_POS_VERTICES[] = 1
        DI._COHOMOLOGY_DEGREE_LOCAL_T1_MIN_TOTAL_ACTIVE_DIM1[] = 0
        DI._COHOMOLOGY_DEGREE_LOCAL_T1_MIN_TOTAL_ACTIVE_DIM2[] = 0
        L_fast = DI._lazy_cochain_complex_from_graded_complex(G, P, axes; field=CM.F2())
        M_fast = DI._cohomology_module_from_lazy(L_fast, 1)

        DI._COHOMOLOGY_DEGREE_LOCAL_ALL_T[] = false
        L_base = DI._lazy_cochain_complex_from_graded_complex(G, P, axes; field=CM.F2())
        M_base = DI._cohomology_module_from_lazy(L_base, 1)

        @test M_fast.dims == M_base.dims
        for (u, v) in FF.cover_edges(M_fast.Q)
            @test FL.rank(CM.F2(), M_fast.edge_maps[u, v]) ==
                  FL.rank(CM.F2(), M_base.edge_maps[u, v])
        end
    finally
        DI._H1_KERNEL_FASTPATH[] = old_h1
        DI._COHOMOLOGY_DEGREE_LOCAL_FASTPATH[] = old_local
        DI._COHOMOLOGY_DEGREE_LOCAL_ALL_T[] = old_all_t
        DI._COHOMOLOGY_DEGREE_LOCAL_T1_MIN_POS_VERTICES[] = old_t1_min_pos
        DI._COHOMOLOGY_DEGREE_LOCAL_T1_MIN_TOTAL_ACTIVE_DIM1[] = old_t1_min_dim1
        DI._COHOMOLOGY_DEGREE_LOCAL_T1_MIN_TOTAL_ACTIVE_DIM2[] = old_t1_min_dim2
    end
end

@testset "Data pipeline: monotone rank-update dims parity" begin
    verts = 1:6
    edges = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (1, 6)]
    I1 = Int[]
    J1 = Int[]
    V1 = Int[]
    for (j, (a, b)) in enumerate(edges)
        push!(I1, a); push!(J1, j); push!(V1, 1)
        push!(I1, b); push!(J1, j); push!(V1, -1)
    end
    B1 = sparse(I1, J1, V1, length(verts), length(edges))
    cells = [collect(verts), collect(1:length(edges))]
    boundaries = [B1]
    grades = vcat([Float64[0.0] for _ in verts], [Float64[0.5] for _ in edges])
    G = TamerOp.GradedComplex(cells, boundaries, grades)
    axes = (collect(range(0.0, stop=1.0, length=80)),)
    P = DI.poset_from_axes(axes)

    old_flag = DI._COHOMOLOGY_DIMS_MONOTONE_RANK_FASTPATH[]
    try
        DI._COHOMOLOGY_DIMS_MONOTONE_RANK_FASTPATH[] = true
        L_fast = DI._lazy_cochain_complex_from_graded_complex(G, P, axes; field=CM.F2())
        d_fast = DI._cohomology_dims_from_lazy(L_fast, 1)

        DI._COHOMOLOGY_DIMS_MONOTONE_RANK_FASTPATH[] = false
        L_base = DI._lazy_cochain_complex_from_graded_complex(G, P, axes; field=CM.F2())
        d_base = DI._cohomology_dims_from_lazy(L_base, 1)

        @test d_fast == d_base
    finally
        DI._COHOMOLOGY_DIMS_MONOTONE_RANK_FASTPATH[] = old_flag
    end
end

@testset "Data pipeline: direct restricted-rank dims parity" begin
    verts = 1:7
    edges = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (1, 7)]
    I1 = Int[]
    J1 = Int[]
    V1 = Int[]
    for (j, (a, b)) in enumerate(edges)
        push!(I1, a); push!(J1, j); push!(V1, 1)
        push!(I1, b); push!(J1, j); push!(V1, -1)
    end
    B1 = sparse(I1, J1, V1, length(verts), length(edges))
    cells = [collect(verts), collect(1:length(edges))]
    boundaries = [B1]
    grades = vcat([Float64[0.0] for _ in verts], [Float64[0.4] for _ in edges])
    G = TamerOp.GradedComplex(cells, boundaries, grades)
    axes = (collect(range(0.0, stop=1.0, length=96)),)
    P = DI.poset_from_axes(axes)

    old_mon = DI._COHOMOLOGY_DIMS_MONOTONE_RANK_FASTPATH[]
    old_direct = DI._COHOMOLOGY_DIMS_USE_DIRECT_RESTRICTED_RANK[]
    try
        DI._COHOMOLOGY_DIMS_MONOTONE_RANK_FASTPATH[] = true
        DI._COHOMOLOGY_DIMS_USE_DIRECT_RESTRICTED_RANK[] = true
        L_fast = DI._lazy_cochain_complex_from_graded_complex(G, P, axes; field=CM.F2())
        d_fast = DI._cohomology_dims_from_lazy(L_fast, 1)

        DI._COHOMOLOGY_DIMS_USE_DIRECT_RESTRICTED_RANK[] = false
        L_base = DI._lazy_cochain_complex_from_graded_complex(G, P, axes; field=CM.F2())
        d_base = DI._cohomology_dims_from_lazy(L_base, 1)

        @test d_fast == d_base
    finally
        DI._COHOMOLOGY_DIMS_MONOTONE_RANK_FASTPATH[] = old_mon
        DI._COHOMOLOGY_DIMS_USE_DIRECT_RESTRICTED_RANK[] = old_direct
    end
end

@testset "Data pipeline: structural inclusion kernels parity" begin
    F = CM.QQField()
    A = DI._StructuralInclusionMap{CM.QQ}(6, 4, [1, 3, 4, 6])
    B = DI._StructuralInclusionMap{CM.QQ}(4, 3, [1, 2, 4])
    C = A * B
    @test C isa DI._StructuralInclusionMap
    @test Matrix(C) == Matrix(A) * Matrix(B)

    @test FL.rank_dim(F, A) == FL.rank_dim(F, Matrix(A))

    Y = Matrix{CM.QQ}(undef, 6, 2)
    fill!(Y, CM.QQ(0))
    Y[1, 1] = CM.QQ(2)
    Y[1, 2] = CM.QQ(1)
    Y[3, 1] = CM.QQ(5)
    Y[3, 2] = CM.QQ(0)
    Y[4, 1] = CM.QQ(7)
    Y[4, 2] = CM.QQ(3)
    Y[6, 1] = CM.QQ(11)
    Y[6, 2] = CM.QQ(13)

    X_struct = FL.solve_fullcolumn(F, A, Y; check_rhs=true)
    X_dense = FL.solve_fullcolumn(F, Matrix(A), Y; check_rhs=true)
    @test X_struct == X_dense
end

@testset "Data pipeline: packed edge-list clique parity" begin
    n = 8
    edges = [
        (1, 2), (1, 3), (2, 3),
        (3, 4), (3, 5), (4, 5),
        (5, 6), (6, 7), (7, 8), (6, 8),
        (2, 4), (2, 5),
    ]
    spec = TamerOp.FiltrationSpec(
        kind=:clique_lower_star,
        max_dim=3,
        construction=OPT.ConstructionOptions(; collapse=:none, sparsify=:none),
    )
    old_flag = DI._GRAPH_PACKED_EDGELIST_BACKEND[]
    old_cache = DI._GRAPH_BACKEND_WINNER_CACHE_ENABLED[]
    old_probe = DI._GRAPH_BACKEND_WINNER_CACHE_PROBE[]
    old_mode = DI._GRAPH_CLIQUE_ENUM_MODE[]
    try
        DI._GRAPH_PACKED_EDGELIST_BACKEND[] = true
        DI._GRAPH_BACKEND_WINNER_CACHE_ENABLED[] = true
        DI._GRAPH_BACKEND_WINNER_CACHE_PROBE[] = false
        DI._clear_graph_backend_winner_cache!()
        @test DI._use_packed_edge_list_backend(160, 365, 3)
        @test !DI._use_packed_edge_list_backend(360, 771, 3)
        @test DI._graph_backend_winner_cache_size() == 0

        # Bucket winner cache should memoize decisions for repeated size buckets.
        choice = DI._select_packed_edge_list_backend(edges, n, 3)
        @test choice isa Bool
        cache_sz = DI._graph_backend_winner_cache_size()
        @test cache_sz >= 1
        @test DI._select_packed_edge_list_backend(edges, n, 3) == choice
        @test DI._graph_backend_winner_cache_size() == cache_sz

        c3_packed = DI._enumerate_cliques_k(edges, n, 3, spec, big(n); context="test")
        c4_packed = DI._enumerate_cliques_k(edges, n, 4, spec, big(n); context="test")

        DI._GRAPH_PACKED_EDGELIST_BACKEND[] = false
        c3_base = DI._enumerate_cliques_k(edges, n, 3, spec, big(n); context="test")
        c4_base = DI._enumerate_cliques_k(edges, n, 4, spec, big(n); context="test")

        # Cached path reuses packed/adjacency representations across k-calls.
        DI._GRAPH_PACKED_EDGELIST_BACKEND[] = true
        c3_cached, packed, adj = DI._enumerate_cliques_k_cached(edges, n, 3, spec, big(n); context="test")
        total2 = big(n) + big(length(c3_cached))
        c4_cached, packed2, adj2 = DI._enumerate_cliques_k_cached(
            edges, n, 4, spec, total2;
            context="test",
            packed=packed,
            adj_lists=adj,
        )
        @test packed2 === packed
        @test adj2 === adj

        normalize(cs) = sort([Tuple(sort(c)) for c in cs])
        @test normalize(c3_packed) == normalize(c3_base)
        @test normalize(c4_packed) == normalize(c4_base)
        @test normalize(c3_cached) == normalize(c3_base)
        @test normalize(c4_cached) == normalize(c4_base)

        # Specialized triangle enumerator (packed dim<=2 path) parity.
        t_before = big(n) + big(length(edges))
        DI._GRAPH_PACKED_EDGELIST_BACKEND[] = true
        tris_packed, _, _ = DI._enumerate_triangles_cached(edges, n, spec, t_before; context="test")
        DI._GRAPH_PACKED_EDGELIST_BACKEND[] = false
        tris_base, _, _ = DI._enumerate_triangles_cached(edges, n, spec, t_before; context="test")
        normalize_tris(ts) = sort([Tuple(t) for t in ts])
        @test normalize_tris(tris_packed) == normalize_tris(tris_base)
        @test normalize_tris(tris_base) == sort([Tuple(sort(c)) for c in c3_base])

        # End-to-end clique_lower_star max_dim=2 parity across enum modes.
        data = TamerOp.GraphData(
            n,
            edges;
            coords=[[Float64(i), Float64(mod(i, 3))] for i in 1:n],
        )
        vals = [Float64(i) / n for i in 1:n]
        spec2 = TamerOp.FiltrationSpec(
            kind=:clique_lower_star,
            max_dim=2,
            vertex_values=vals,
            simplex_agg=:max,
            construction=OPT.ConstructionOptions(; collapse=:none, sparsify=:none),
        )
        DI._GRAPH_CLIQUE_ENUM_MODE[] = :intersection
        st_inter = TamerOp.encode(data, spec2; degree=0, cache=:auto, stage=:simplex_tree)
        DI._GRAPH_CLIQUE_ENUM_MODE[] = :combinations
        st_comb = TamerOp.encode(data, spec2; degree=0, cache=:auto, stage=:simplex_tree)
        @test _canon_simplex_tree(st_inter) == _canon_simplex_tree(st_comb)
    finally
        DI._GRAPH_PACKED_EDGELIST_BACKEND[] = old_flag
        DI._GRAPH_BACKEND_WINNER_CACHE_ENABLED[] = old_cache
        DI._GRAPH_BACKEND_WINNER_CACHE_PROBE[] = old_probe
        DI._GRAPH_CLIQUE_ENUM_MODE[] = old_mode
    end
end

@testset "Data pipeline: cohomology_dims stage parity + invariant shortcut" begin
    data = TamerOp.PointCloud([[0.0], [0.3], [0.8], [1.1], [1.6], [2.0]])
    spec = TamerOp.FiltrationSpec(
        kind=:rips,
        max_dim=1,
        axes=([0.0, 0.4, 0.8, 1.2, 1.6, 2.0],),
        construction=OPT.ConstructionOptions(; output_stage=:encoding_result),
    )
    d = TamerOp.encode(data, spec; degree=0, stage=:cohomology_dims, cache=:auto)
    M = TamerOp.encode(data, spec; degree=0, stage=:module, cache=:auto)
    enc = TamerOp.encode(data, spec; degree=0, stage=:encoding_result, cache=:auto)

    @test d isa RES.CohomologyDimsResult
    @test FF.nvertices(d.P) == FF.nvertices(M.Q)
    for u in 1:FF.nvertices(d.P), v in 1:FF.nvertices(d.P)
        @test FF.leq(d.P, u, v) == FF.leq(M.Q, u, v)
    end
    @test d.dims == M.dims

    h_mod = Inv.restricted_hilbert(M)
    h_dims = TamerOp.restricted_hilbert(d)
    @test h_mod == h_dims

    # Same dims-only invariant should work on both result types.
    h_enc = TamerOp.invariant(enc; which=:restricted_hilbert).value
    h_cdr = TamerOp.invariant(d; which=:restricted_hilbert).value
    @test h_cdr == h_enc

    e_opts = OPT.InvariantOptions(; axes=([0.0, 0.8, 1.6],), axes_policy=:as_given, threads=false)
    e_enc = TamerOp.invariant(enc; which=:euler_surface, opts=e_opts).value
    e_cdr = TamerOp.invariant(d; which=:euler_surface, opts=e_opts).value
    @test e_cdr == e_enc

    # Unsupported invariants fail cleanly on dims-only objects.
    @test_throws ErrorException TamerOp.invariant(d; which=:rank_invariant)
end

@testset "Data pipeline: encoded_complex stage exact Euler route" begin
    data = TamerOp.PointCloud([[0.0], [0.3], [0.8], [1.1], [1.6], [2.0]])
    spec = TamerOp.FiltrationSpec(
        kind=:rips,
        max_dim=1,
        axes=([0.0, 0.4, 0.8, 1.2, 1.6, 2.0],),
        construction=OPT.ConstructionOptions(; output_stage=:encoding_result),
    )
    enc_complex = TamerOp.encode(data, spec; degree=0, stage=:encoded_complex, cache=:auto)
    C = TamerOp.encode(data, spec; degree=0, stage=:cochain, cache=:auto)
    d = TamerOp.encode(data, spec; degree=0, stage=:cohomology_dims, cache=:auto)

    @test enc_complex isa RES.EncodedComplexResult
    @test TO.describe(enc_complex).kind == :encoded_complex_result
    @test TO.encoding_complex(enc_complex) isa DI.LazyModuleCochainComplex
    @test MC.describe(RES._materialize_complex(TO.encoding_complex(enc_complex))).degree_range == MC.describe(C).degree_range
    @test TO.encoding_map(enc_complex) isa EC.CompiledEncoding
    @test FF.nvertices(TO.encoding_poset(enc_complex)) == FF.nvertices(d.P)
    for u in 1:FF.nvertices(d.P), v in 1:FF.nvertices(d.P)
        @test FF.leq(TO.encoding_poset(enc_complex), u, v) == FF.leq(d.P, u, v)
    end

    e_opts = OPT.InvariantOptions(; axes=([0.0, 0.8, 1.6],), axes_policy=:as_given, threads=false)
    e_direct = SM.euler_signed_measure(TO.encoding_complex(enc_complex), TO.encoding_map(enc_complex), e_opts)
    e_workflow = TamerOp.euler_signed_measure(enc_complex; opts=e_opts)
    e_invariant = TamerOp.invariant(enc_complex; which=:euler_signed_measure, opts=e_opts).value
    @test Base.axes(e_workflow) == Base.axes(e_direct)
    @test SM.support_indices(e_workflow) == SM.support_indices(e_direct)
    @test SM.weights(e_workflow) == SM.weights(e_direct)
    @test SM.support_indices(e_invariant) == SM.support_indices(e_direct)
    @test SM.weights(e_invariant) == SM.weights(e_direct)

    s_direct = SM.euler_surface(TO.encoding_complex(enc_complex), TO.encoding_map(enc_complex), e_opts)
    s_workflow = TamerOp.euler_surface(enc_complex; opts=e_opts)
    s_invariant = TamerOp.invariant(enc_complex; which=:euler_surface, opts=e_opts).value
    @test s_workflow == s_direct
    @test s_invariant == s_direct
end

@testset "Data pipeline: lazy 1D Euler bypasses active lists" begin
    data = TamerOp.PointCloud([[0.0], [0.3], [0.8], [1.1], [1.6], [2.0]])
    spec = TamerOp.FiltrationSpec(
        kind=:rips,
        max_dim=1,
        axes=([0.0, 0.4, 0.8, 1.2, 1.6, 2.0],),
        construction=OPT.ConstructionOptions(; output_stage=:encoding_result),
    )
    enc_complex = TamerOp.encode(data, spec; degree=0, stage=:encoded_complex, cache=:auto)
    lazy = TO.encoding_complex(enc_complex)
    e_opts = OPT.InvariantOptions(; axes=([0.0, 0.8, 1.6],), axes_policy=:as_given, threads=false)

    @test all(isnothing, lazy.active_by_dim)
    pm = TamerOp.euler_signed_measure(enc_complex; opts=e_opts)
    @test all(isnothing, lazy.active_by_dim)

    surf = SM.surface_from_point_signed_measure(pm)
    @test surf == TamerOp.euler_surface(enc_complex; opts=e_opts)
end

@testset "Data pipeline: lazy 1D Euler direct measure path on encoding axes" begin
    data = TamerOp.PointCloud([
        [0.0, 0.0],
        [1.0, 0.1],
        [0.2, 0.95],
        [1.1, 0.85],
        [0.55, 0.42],
    ])
    spec = TamerOp.FiltrationSpec(
        kind=:alpha,
        max_dim=2,
        construction=OPT.ConstructionOptions(; output_stage=:encoding_result),
    )
    enc_complex = TamerOp.encode(data, spec; degree=0, stage=:encoded_complex, cache=:auto)
    lazy = TO.encoding_complex(enc_complex)
    e_opts = OPT.InvariantOptions(; threads=false)

    @test all(isnothing, lazy.active_by_dim)
    pm = TamerOp.euler_signed_measure(enc_complex; opts=e_opts)
    @test all(isnothing, lazy.active_by_dim)

    surf = TamerOp.euler_surface(enc_complex; opts=e_opts)
    @test SM.surface_from_point_signed_measure(pm) == surf
end

@testset "Data pipeline: lazy 2D Euler direct measure path on encoding axes" begin
    data = TamerOp.PointCloud([
        [0.0, 0.0],
        [0.8, 0.2],
        [1.6, 0.6],
        [2.2, 0.4],
    ])
    spec = TamerOp.FiltrationSpec(
        kind=:rips_lowerstar,
        max_dim=1,
        radius=2.5,
        coord=1,
        construction=OPT.ConstructionOptions(; output_stage=:encoding_result),
    )
    enc_complex = TamerOp.encode(data, spec; degree=0, stage=:encoded_complex, cache=:auto)
    lazy = TO.encoding_complex(enc_complex)
    e_opts = OPT.InvariantOptions(; threads=false)

    @test lazy.vertex_idxs === nothing
    @test TO.encoding_map(enc_complex).reps === nothing
    enc_cached = RES._encoding_with_session_cache(enc_complex, CM.SessionCache())
    @test TO.encoding_map(enc_cached).reps === nothing
    @test all(isnothing, lazy.active_by_dim)
    pm = TamerOp.euler_signed_measure(enc_complex; opts=e_opts)
    @test lazy.vertex_idxs === nothing
    @test all(isnothing, lazy.active_by_dim)

    surf = TamerOp.euler_surface(enc_complex; opts=e_opts)
    @test SM.surface_from_point_signed_measure(pm) == surf
end

@testset "Data pipeline: lazy 2D active-list fallback materializes vertex indices on demand" begin
    data = TamerOp.PointCloud([
        [0.0, 0.0],
        [0.8, 0.2],
        [1.6, 0.6],
        [2.2, 0.4],
    ])
    spec = TamerOp.FiltrationSpec(
        kind=:rips_codensity,
        max_dim=1,
        radius=2.5,
        dtm_mass=0.5,
        construction=OPT.ConstructionOptions(; output_stage=:encoding_result),
    )
    enc_complex = TamerOp.encode(data, spec; degree=0, stage=:encoded_complex, cache=:auto)
    lazy = TO.encoding_complex(enc_complex)

    @test lazy.vertex_idxs === nothing
    _ = RES._materialize_complex(lazy)
    @test lazy.vertex_idxs !== nothing
end

@testset "Data pipeline: encoding_result lazy module parity" begin
    data = TamerOp.PointCloud([[0.0], [0.25], [0.7], [1.1], [1.6], [2.0]])
    spec = TamerOp.FiltrationSpec(
        kind=:rips,
        max_dim=2,
        construction=OPT.ConstructionOptions(; output_stage=:encoding_result),
    )
    old_lazy = DI._ENCODING_RESULT_LAZY_MODULE[]
    try
        DI._ENCODING_RESULT_LAZY_MODULE[] = true
        enc_lazy = TamerOp.encode(data, spec; degree=1, stage=:encoding_result, cache=:auto)
        @test enc_lazy isa RES.EncodingResult
        @test enc_lazy.M isa DI._LazyEncodedModule

        M_lazy = TamerOp.Workflow.pmodule(enc_lazy)
        @test M_lazy isa MD.PModule
        @test TamerOp.Workflow.pmodule(enc_lazy) === M_lazy

        DI._ENCODING_RESULT_LAZY_MODULE[] = false
        enc_eager = TamerOp.encode(data, spec; degree=1, stage=:encoding_result, cache=:auto)
        @test enc_eager.M isa MD.PModule

        h_lazy = TamerOp.restricted_hilbert(enc_lazy)
        h_eager = TamerOp.restricted_hilbert(enc_eager)
        @test h_lazy == h_eager

        e_opts = OPT.InvariantOptions(; axes=([0.0, 0.8, 1.6],), axes_policy=:as_given, threads=false)
        e_lazy = TamerOp.euler_surface(enc_lazy; opts=e_opts)
        e_eager = TamerOp.euler_surface(enc_eager; opts=e_opts)
        @test e_lazy == e_eager
    finally
        DI._ENCODING_RESULT_LAZY_MODULE[] = old_lazy
    end
end

@testset "Data pipeline: encoding_result lazy module defers representative materialization" begin
    data = TamerOp.PointCloud([
        [0.0, 0.0],
        [0.8, 0.2],
        [1.6, 0.6],
        [2.2, 0.4],
    ])
    spec = TamerOp.FiltrationSpec(
        kind=:rips_lowerstar,
        max_dim=1,
        radius=2.5,
        coord=1,
        construction=OPT.ConstructionOptions(; output_stage=:encoding_result),
    )
    old_lazy = DI._ENCODING_RESULT_LAZY_MODULE[]
    try
        DI._ENCODING_RESULT_LAZY_MODULE[] = true
        enc = TamerOp.encode(data, spec; degree=0, stage=:encoding_result, cache=:auto)
        @test enc.M isa DI._LazyEncodedModule
        @test TO.encoding_map(enc).reps === nothing
        @test EC.encoding_summary(TO.encoding_map(enc)).has_representatives
        @test TO.encoding_map(enc).reps === nothing

        enc_cached = RES._encoding_with_session_cache(enc, CM.SessionCache())
        @test TO.encoding_map(enc_cached).reps === nothing
        @test EC.encoding_summary(TO.encoding_map(enc_cached)).has_representatives
        @test TO.encoding_map(enc_cached).reps === nothing

        reps = TO.encoding_representatives(enc)
        @test length(reps) == prod(length.(EC.axes_from_encoding(enc.pi)))
        @test TO.encoding_map(enc).reps === nothing
    finally
        DI._ENCODING_RESULT_LAZY_MODULE[] = old_lazy
    end
end

@testset "Data pipeline: rectangle_signed_barcode uses lazy H0 direct path" begin
    data = TamerOp.PointCloud([
        [0.0, 0.0],
        [0.8, 0.2],
        [1.6, 0.6],
        [2.2, 0.4],
    ])
    spec = TamerOp.FiltrationSpec(
        kind=:rips_lowerstar,
        max_dim=1,
        radius=2.5,
        coord=1,
        construction=OPT.ConstructionOptions(; output_stage=:encoding_result),
    )
    old_lazy = DI._ENCODING_RESULT_LAZY_MODULE[]
    try
        DI._ENCODING_RESULT_LAZY_MODULE[] = true
        enc_lazy = TamerOp.encode(data, spec; degree=0, stage=:encoding_result, cache=:auto)
        @test enc_lazy.M isa DI._LazyEncodedModule
        rect_opts = OPT.InvariantOptions(; threads=false)
        @test IC._supports_exact_rectangle_signed_barcode(enc_lazy; opts=rect_opts)

        sb_lazy = TamerOp.rectangle_signed_barcode(enc_lazy; opts=rect_opts, cache=CM.SessionCache(), threads=false)
        @test enc_lazy.M.cached_module === nothing

        DI._ENCODING_RESULT_LAZY_MODULE[] = false
        enc_eager = TamerOp.encode(data, spec; degree=0, stage=:encoding_result, cache=:auto)
        sb_eager = TamerOp.rectangle_signed_barcode(enc_eager; opts=rect_opts, cache=CM.SessionCache(), threads=false)

        @test Dict(zip(sb_lazy.rects, sb_lazy.weights)) ==
              Dict(zip(sb_eager.rects, sb_eager.weights))
        @test TamerOp.SignedMeasures.rectangle_signed_barcode_rank(sb_lazy; zero_noncomparable=true, threads=false) ==
              TamerOp.SignedMeasures.rectangle_signed_barcode_rank(sb_eager; zero_noncomparable=true, threads=false)
    finally
        DI._ENCODING_RESULT_LAZY_MODULE[] = old_lazy
    end
end

@testset "Data pipeline: exact lazy H0 rank table bypasses rectangle decomposition" begin
    function _serialize_rank_table(axes, table)
        nd = length(axes)
        dims = ntuple(i -> length(axes[i]), nd)
        io = IOBuffer()
        first = true
        for pCI in CartesianIndices(dims)
            p = pCI.I
            q_ranges = ntuple(k -> p[k]:dims[k], nd)
            for qCI in CartesianIndices(q_ranges)
                q = qCI.I
                val = @inbounds table[pCI, qCI]
                iszero(val) && continue
                first || write(io, ';')
                first = false
                print(io,
                      join((repr(Float64(axes[k][p[k]])) for k in 1:nd), "|"),
                      "||",
                      join((repr(Float64(axes[k][q[k]])) for k in 1:nd), "|"),
                      "=>",
                      val)
            end
        end
        return String(take!(io))
    end

    data = TamerOp.PointCloud([
        [0.0, 0.0],
        [0.8, 0.2],
        [1.6, 0.6],
        [2.2, 0.4],
    ])
    spec = TamerOp.FiltrationSpec(
        kind=:rips_lowerstar,
        max_dim=1,
        radius=2.5,
        coord=1,
        construction=OPT.ConstructionOptions(; output_stage=:encoding_result),
    )
    old_lazy = DI._ENCODING_RESULT_LAZY_MODULE[]
    try
        DI._ENCODING_RESULT_LAZY_MODULE[] = true
        enc_lazy = TamerOp.encode(data, spec; degree=0, stage=:encoding_result, cache=:auto)
        rank_opts = OPT.InvariantOptions(; threads=false)
        @test IC._supports_exact_rank_query_table(enc_lazy; opts=rank_opts)
        @test IC._supports_exact_rank_signed_measure(enc_lazy; opts=rank_opts)

        direct = IC._exact_rank_query_table(enc_lazy; opts=rank_opts, threads=false)
        @test direct !== nothing
        @test getproperty(direct, :direct_rank_table)
        @test enc_lazy.M.cached_module === nothing

        direct_measure = IC._exact_rank_signed_measure(enc_lazy; opts=rank_opts, threads=false)
        @test direct_measure !== nothing
        @test getproperty(direct_measure, :direct_rank_measure)
        @test enc_lazy.M.cached_module === nothing

        sb_lazy = TamerOp.rectangle_signed_barcode(enc_lazy; opts=rank_opts, cache=CM.SessionCache(), threads=false)
        pi0 = TamerOp.encoding_map(enc_lazy)
        raw_pi = pi0 isa EC.CompiledEncoding ? TamerOp.encoding_map(pi0) : pi0
        birth_axes, _ = SM._rectangle_signed_barcode_grid_semantic_axes(raw_pi, rank_opts; keep_endpoints=true)
        rank_table = SM.rectangle_signed_barcode_rank(sb_lazy; zero_noncomparable=true, threads=false)

        @test getproperty(direct, :rank_query_axes) == birth_axes
        # The direct sweep emits target-major rows; the independent dense
        # reconstruction emits source-major rows. Compare the complete entries.
        @test sort(split(getproperty(direct, :rank_table_canonical), ';')) ==
              sort(split(_serialize_rank_table(birth_axes, rank_table), ';'))
        sem_birth, sem_death = SM._rectangle_signed_barcode_grid_semantic_axes(raw_pi, rank_opts; keep_endpoints=true)
        direct_terms = Dict(
            (
                direct_measure.axes[1][idx[1]],
                direct_measure.axes[2][idx[2]],
                direct_measure.axes[3][idx[3]],
                direct_measure.axes[4][idx[4]],
            ) => wt for (idx, wt) in zip(direct_measure.inds, direct_measure.wts)
        )
        sb_terms = Dict(
            (
                sem_birth[1][rect.lo[1]],
                sem_birth[2][rect.lo[2]],
                sem_death[1][rect.hi[1]],
                sem_death[2][rect.hi[2]],
            ) => wt for (rect, wt) in zip(sb_lazy.rects, sb_lazy.weights)
        )
        @test direct_terms == sb_terms
        # Four components are born at the four vertex function values. Each
        # later component merges with its predecessor at their edge length.
        expected_terms = Dict(
            (0.0, 0.0, Inf, Inf) => 1,
            (0.0, 0.8, hypot(0.8, 0.2), Inf) => 1,
            (0.0, 1.6, hypot(0.8, 0.4), Inf) => 1,
            (0.0, 2.2, hypot(2.2 - 1.6, 0.4 - 0.6), Inf) => 1,
        )
        @test Dict(map(x -> round(x; digits=12), key) => wt for (key, wt) in direct_terms) ==
              Dict(map(x -> round(x; digits=12), key) => wt for (key, wt) in expected_terms)
    finally
        DI._ENCODING_RESULT_LAZY_MODULE[] = old_lazy
    end
end

@testset "Data pipeline: exact lazy H0 rank measure covers alpha" begin
    data = TamerOp.PointCloud([
        [0.0, 0.0],
        [0.9, 0.1],
        [0.2, 1.0],
        [1.1, 0.9],
        [0.55, 0.45],
    ])
    spec = TamerOp.FiltrationSpec(
        kind=:alpha,
        max_dim=2,
        construction=OPT.ConstructionOptions(; output_stage=:encoding_result),
    )
    old_lazy = DI._ENCODING_RESULT_LAZY_MODULE[]
    try
        DI._ENCODING_RESULT_LAZY_MODULE[] = true
        enc_lazy = TamerOp.encode(data, spec; degree=0, stage=:encoding_result, cache=:auto)
        rank_opts = OPT.InvariantOptions(; threads=false)
        @test IC._supports_exact_rank_signed_measure(enc_lazy; opts=rank_opts, threads=false)

        direct_measure = IC._exact_rank_signed_measure(enc_lazy; opts=rank_opts, threads=false)
        @test direct_measure !== nothing
        @test getproperty(direct_measure, :direct_rank_measure)
        @test enc_lazy.M.cached_module === nothing

        sb_lazy = TamerOp.rectangle_signed_barcode(enc_lazy; opts=rank_opts, cache=CM.SessionCache(), threads=false)
        pi0 = TamerOp.encoding_map(enc_lazy)
        raw_pi = pi0 isa EC.CompiledEncoding ? TamerOp.encoding_map(pi0) : pi0
        sem_birth, sem_death = SM._rectangle_signed_barcode_grid_semantic_axes(raw_pi, rank_opts; keep_endpoints=true)
        direct_terms = Dict(
            (
                direct_measure.axes[1][idx[1]],
                direct_measure.axes[2][idx[2]],
            ) => wt for (idx, wt) in zip(direct_measure.inds, direct_measure.wts)
        )
        sb_terms = Dict(
            (
                sem_birth[1][rect.lo[1]],
                sem_death[1][rect.hi[1]],
            ) => wt for (rect, wt) in zip(sb_lazy.rects, sb_lazy.weights)
        )
        @test direct_terms == sb_terms
    finally
        DI._ENCODING_RESULT_LAZY_MODULE[] = old_lazy
    end
end

@testset "Data pipeline: exact lazy H0 restricted Hilbert backend" begin
    alpha_data = TamerOp.PointCloud([
        [0.0, 0.0],
        [0.9, 0.1],
        [0.2, 1.0],
        [1.1, 0.9],
        [0.55, 0.45],
    ])
    alpha_spec = TamerOp.FiltrationSpec(
        kind=:alpha,
        max_dim=2,
        construction=OPT.ConstructionOptions(; output_stage=:encoding_result),
    )
    rips_data = TamerOp.PointCloud([
        [0.0, 0.0],
        [0.8, 0.2],
        [1.6, 0.6],
        [2.2, 0.4],
    ])
    rips_spec = TamerOp.FiltrationSpec(
        kind=:rips_lowerstar,
        max_dim=1,
        radius=2.5,
        coord=1,
        construction=OPT.ConstructionOptions(; output_stage=:encoding_result),
    )
    inv_opts = OPT.InvariantOptions(; threads=false)
    old_lazy = DI._ENCODING_RESULT_LAZY_MODULE[]
    try
        DI._ENCODING_RESULT_LAZY_MODULE[] = true
        enc_alpha = TamerOp.encode(alpha_data, alpha_spec; degree=0, stage=:encoding_result, cache=:auto)
        enc_rips = TamerOp.encode(rips_data, rips_spec; degree=0, stage=:encoding_result, cache=:auto)

        @test IC._supports_exact_restricted_hilbert(enc_alpha; opts=inv_opts, threads=false)
        @test IC._supports_exact_restricted_hilbert(enc_rips; opts=inv_opts, threads=false)

        alpha_direct = IC._exact_restricted_hilbert(enc_alpha; opts=inv_opts, threads=false)
        rips_direct = IC._exact_restricted_hilbert(enc_rips; opts=inv_opts, threads=false)
        @test alpha_direct !== nothing
        @test rips_direct !== nothing
        @test enc_alpha.M.cached_module === nothing
        @test enc_rips.M.cached_module === nothing
        @test enc_alpha.M.dims !== nothing
        @test enc_rips.M.dims !== nothing

        DI._ENCODING_RESULT_LAZY_MODULE[] = false
        enc_alpha_eager = TamerOp.encode(alpha_data, alpha_spec; degree=0, stage=:encoding_result, cache=:auto)
        enc_rips_eager = TamerOp.encode(rips_data, rips_spec; degree=0, stage=:encoding_result, cache=:auto)

        @test alpha_direct == TamerOp.restricted_hilbert(enc_alpha_eager; opts=inv_opts, cache=CM.SessionCache())
        @test rips_direct == TamerOp.restricted_hilbert(enc_rips_eager; opts=inv_opts, cache=CM.SessionCache())
    finally
        DI._ENCODING_RESULT_LAZY_MODULE[] = old_lazy
    end
end

@testset "Data pipeline: Workflow wrappers preserve lazy exact paths" begin
    function _pm_terms(pm)
        return Dict(
            ntuple(i -> pm.axes[i][idx[i]], length(pm.axes)) => wt
            for (idx, wt) in zip(pm.inds, pm.wts)
        )
    end

    function _same_mp_landscape(a, b)
        return a.kmax == b.kmax &&
               a.tgrid == b.tgrid &&
               a.values == b.values &&
               a.weights == b.weights &&
               a.directions == b.directions &&
               a.offsets == b.offsets
    end

    data = TamerOp.PointCloud([
        [0.0, 0.0],
        [0.8, 0.2],
        [1.6, 0.6],
        [2.2, 0.4],
    ])
    spec = TamerOp.FiltrationSpec(
        kind=:rips_lowerstar,
        max_dim=1,
        radius=2.5,
        coord=1,
        construction=OPT.ConstructionOptions(; output_stage=:encoding_result),
    )
    dirs = [[1.0, 0.0], [1.0, 1.0]]
    offs = [[0.0, 0.0], [0.5, 0.0]]
    tg = collect(range(-0.5, 3.0; length=17))
    inv_opts = OPT.InvariantOptions(; threads=false)
    old_lazy = DI._ENCODING_RESULT_LAZY_MODULE[]
    try
        DI._ENCODING_RESULT_LAZY_MODULE[] = true
        enc_lazy = TamerOp.encode(data, spec; degree=0, stage=:encoding_result, cache=:auto)
        @test enc_lazy.M isa DI._LazyEncodedModule

        hilbert_lazy = TamerOp.restricted_hilbert(enc_lazy; opts=inv_opts, cache=CM.SessionCache())
        @test enc_lazy.M.cached_module === nothing

        euler_lazy = TamerOp.euler_signed_measure(enc_lazy; opts=inv_opts, cache=CM.SessionCache())
        @test enc_lazy.M.cached_module === nothing

        rank_lazy = TamerOp.rank_signed_measure(enc_lazy; opts=inv_opts, cache=CM.SessionCache(), threads=false)
        @test enc_lazy.M.cached_module === nothing

        slices_lazy = TamerOp.slice_barcodes(
            enc_lazy;
            opts=inv_opts,
            cache=CM.SessionCache(),
            directions=dirs,
            offsets=offs,
            normalize_dirs=:none,
            ts=tg,
            direction_weight=:uniform,
            normalize_weights=true,
            drop_unknown=true,
            dedup=true,
            threads=false,
        )
        @test enc_lazy.M.cached_module === nothing

        mp_lazy = TamerOp.mp_landscape(
            enc_lazy;
            opts=inv_opts,
            cache=CM.SessionCache(),
            directions=dirs,
            offsets=offs,
            tgrid=tg,
            direction_weight=:uniform,
            normalize_weights=true,
            threads=false,
        )
        @test enc_lazy.M.cached_module === nothing

        inv_euler = TamerOp.invariant(enc_lazy; which=:euler_signed_measure, opts=inv_opts, cache=CM.SessionCache())
        @test enc_lazy.M.cached_module === nothing
        @test _pm_terms(TamerOp.invariant_value(inv_euler)) == _pm_terms(euler_lazy)

        inv_hilbert = TamerOp.invariant(enc_lazy; which=:restricted_hilbert, opts=inv_opts, cache=CM.SessionCache())
        @test enc_lazy.M.cached_module === nothing
        @test TamerOp.invariant_value(inv_hilbert) == hilbert_lazy

        inv_rank = TamerOp.invariant(enc_lazy; which=:rank_signed_measure, opts=inv_opts, cache=CM.SessionCache(), threads=false)
        @test enc_lazy.M.cached_module === nothing
        @test _pm_terms(TamerOp.invariant_value(inv_rank)) == _pm_terms(rank_lazy)

        inv_slices = TamerOp.invariant(
            enc_lazy;
            which=:slice_barcodes,
            opts=inv_opts,
            cache=CM.SessionCache(),
            directions=dirs,
            offsets=offs,
            normalize_dirs=:none,
            ts=tg,
            direction_weight=:uniform,
            normalize_weights=true,
            drop_unknown=true,
            dedup=true,
            threads=false,
        )
        @test enc_lazy.M.cached_module === nothing
        @test TamerOp.invariant_value(inv_slices).barcodes == slices_lazy.barcodes

        inv_mp = TamerOp.invariant(
            enc_lazy;
            which=:mp_landscape,
            opts=inv_opts,
            cache=CM.SessionCache(),
            directions=dirs,
            offsets=offs,
            tgrid=tg,
            direction_weight=:uniform,
            normalize_weights=true,
            threads=false,
        )
        @test enc_lazy.M.cached_module === nothing
        @test _same_mp_landscape(TamerOp.invariant_value(inv_mp), mp_lazy)

        DI._ENCODING_RESULT_LAZY_MODULE[] = false
        enc_eager = TamerOp.encode(data, spec; degree=0, stage=:encoding_result, cache=:auto)

        hilbert_eager = TamerOp.restricted_hilbert(enc_eager; opts=inv_opts, cache=CM.SessionCache())
        euler_eager = TamerOp.euler_signed_measure(enc_eager; opts=inv_opts, cache=CM.SessionCache())
        rank_eager = TamerOp.rank_signed_measure(enc_eager; opts=inv_opts, cache=CM.SessionCache(), threads=false)
        slices_eager = TamerOp.slice_barcodes(
            enc_eager;
            opts=inv_opts,
            cache=CM.SessionCache(),
            directions=dirs,
            offsets=offs,
            normalize_dirs=:none,
            ts=tg,
            direction_weight=:uniform,
            normalize_weights=true,
            drop_unknown=true,
            dedup=true,
            threads=false,
        )
        mp_eager = TamerOp.mp_landscape(
            enc_eager;
            opts=inv_opts,
            cache=CM.SessionCache(),
            directions=dirs,
            offsets=offs,
            tgrid=tg,
            direction_weight=:uniform,
            normalize_weights=true,
            threads=false,
        )

        @test hilbert_lazy == hilbert_eager
        @test _pm_terms(euler_lazy) == _pm_terms(euler_eager)
        @test _pm_terms(rank_lazy) == _pm_terms(rank_eager)
        @test slices_lazy.barcodes == slices_eager.barcodes
        @test slices_lazy.weights == slices_eager.weights
        @test slices_lazy.dirs == slices_eager.dirs
        @test slices_lazy.offs == slices_eager.offs
        @test _same_mp_landscape(mp_lazy, mp_eager)
    finally
        DI._ENCODING_RESULT_LAZY_MODULE[] = old_lazy
    end
end

with_fields(FIELDS_FULL) do field
@testset "Data pipeline: lazy sampled H0 slices respect sample times and grid births" begin
    # Two components merge at radius 1. The sample at 1.5 first sees the merge;
    # the last sampled interval is censored at 2.5, not at infinity.
    B = sparse([1, 2], [1, 1], [-1, 1], 2, 1)
    G = DT.GradedComplex([Int[1, 2], Int[1]], [B], [(0.0,), (0.0,), (1.0,)])
    spec = OPT.FiltrationSpec(kind=:graded, axes=([0.0, 1.0, 2.0],))
    samples = [0.0, 0.5, 1.5]
    query = (; directions=[[1.0]], offsets=[[0.0]], ts=samples, dedup=false)
    opts = OPT.InvariantOptions(threads=false)
    expected = Dict((0.0, 1.5) => 1, (0.0, 2.5) => 1)
    old_lazy = DI._ENCODING_RESULT_LAZY_MODULE[]
    try
        DI._ENCODING_RESULT_LAZY_MODULE[] = true
        enc = TamerOp.encode(G, spec; degree=0, field=field, stage=:encoding_result)
        session = CM.SessionCache()
        sampled = TamerOp.slice_barcodes(enc; opts=opts, cache=session, query...)
        @test sampled.barcodes[1, 1] == expected
        @test enc.M.cached_module === nothing
        plan_cache = CM._session_slice_plan_cache(session)
        @test length(plan_cache.plans) == 1
        @test TamerOp.slice_barcodes(enc; opts=opts, cache=session, query...).barcodes == sampled.barcodes
        @test length(plan_cache.plans) == 1
        packed = TamerOp.slice_barcodes(enc; opts=opts, cache=session, packed=true, query...)
        @test TamerOp.SliceInvariants._barcode_from_packed(packed.barcodes[1, 1]) == expected

        deduplicated = TamerOp.slice_barcodes(enc; opts=opts,
            directions=[[1.0]], offsets=[[0.0]], ts=samples, dedup=true)
        @test deduplicated.barcodes[1, 1] == Dict((0.0, 1.5) => 1, (0.0, 3.0) => 1)
        continuous = TamerOp.slice_barcodes(enc; opts=opts,
            directions=[[1.0]], offsets=[[0.0]])
        @test continuous.barcodes[1, 1] == Dict((0.0, 1.0) => 1, (0.0, Inf) => 1)
        normalized = TamerOp.slice_barcodes(enc; opts=opts,
            directions=[[2.0]], offsets=[[0.0]], normalize_dirs=:L1,
            ts=samples, dedup=false)
        @test normalized.barcodes == sampled.barcodes

        # Explicit coarsening puts the edge in the grid cell born at zero.
        coarse_spec = OPT.FiltrationSpec(kind=:graded, axes=([0.0, 2.0],))
        coarse = TamerOp.encode(G, coarse_spec; degree=0, field=field, stage=:encoding_result)
        @test TamerOp.slice_barcodes(coarse; opts=opts, query...).barcodes[1, 1] == Dict((0.0, 2.5) => 1)
        @test coarse.M.cached_module === nothing
        empty_sample = TamerOp.slice_barcodes(enc; opts=opts,
            directions=[[1.0]], offsets=[[0.0]], ts=[-2.0, -1.0])
        @test isempty(empty_sample.barcodes[1, 1])

        # Landscape evaluation points and persistence samples are distinct.
        tg = [0.25, 0.75, 1.25, 1.75, 2.25]
        landscape = TamerOp.mp_landscape(enc; opts=opts,
            directions=[[1.0]], offsets=[[0.0]], ts=samples, dedup=false,
            tgrid=tg, kmax=2, threads=false)
        @test vec(landscape.values[1, 1, 1, :]) == [0.25, 0.75, 1.25, 0.75, 0.25]
        @test vec(landscape.values[1, 1, 2, :]) == [0.25, 0.75, 0.25, 0.0, 0.0]
        @test enc.M.cached_module === nothing

        DI._ENCODING_RESULT_LAZY_MODULE[] = false
        eager = TamerOp.encode(G, spec; degree=0, field=field, stage=:encoding_result)
        @test TamerOp.slice_barcodes(eager; opts=opts, query...).barcodes == sampled.barcodes
        eager_landscape = TamerOp.mp_landscape(eager; opts=opts,
            directions=[[1.0]], offsets=[[0.0]], ts=samples, dedup=false,
            tgrid=tg, kmax=2, threads=false)
        @test eager_landscape.values == landscape.values
        # Unknown labels are never silently treated as empty H0 spaces.
        for strict in (false, true)
            unknown_query = (; directions=[[1.0]], offsets=[[0.0]],
                ts=[-0.5, 0.0, 1.5], tmin=-1.0, tmax=2.0,
                drop_unknown=false, dedup=false)
            strict_opts = OPT.InvariantOptions(threads=false, strict=strict)
            @test_throws Exception TamerOp.slice_barcodes(enc; opts=strict_opts, unknown_query...)
            @test_throws Exception TamerOp.slice_barcodes(eager; opts=strict_opts, unknown_query...)
        end
        @test enc.M.cached_module === nothing
    finally
        DI._ENCODING_RESULT_LAZY_MODULE[] = old_lazy
    end
end

@testset "Data pipeline: weighted edge boundaries use field-aware H0" begin
    # The boundary 2(v2-v1) vanishes in F2, but identifies the two vertices
    # over the other supported fields. It must not enter a union-find route.
    B = sparse([1, 2], [1, 1], [-2, 2], 2, 1)
    @test DI._edge_endpoints_from_boundary(B) === nothing
    @test DI._edge_endpoints_from_boundary(sparse([1, 2], [1, 1], [1, 1], 2, 1)) === nothing
    G = DT.GradedComplex([Int[1, 2], Int[1]], [B], [(0.0,), (0.0,), (1.0,)])
    spec = OPT.FiltrationSpec(kind=:graded, axes=([0.0, 1.0, 2.0],))
    old_lazy = DI._ENCODING_RESULT_LAZY_MODULE[]
    try
        DI._ENCODING_RESULT_LAZY_MODULE[] = true
        enc = TamerOp.encode(G, spec; degree=0, field=field, stage=:encoding_result)
        query = (; opts=OPT.InvariantOptions(threads=false), directions=[[1.0]],
            offsets=[[0.0]], ts=[0.0, 0.5, 1.5], dedup=false)
        @test IC._exact_slice_barcodes(enc; query...) === nothing
        @test enc.M.cached_module === nothing
        sampled = TamerOp.slice_barcodes(enc; query...)
        expected_dim = iszero(CM.coerce(field, 2)) ? 2 : 1
        expected = expected_dim == 2 ? Dict((0.0, 2.5) => 2) :
            Dict((0.0, 1.5) => 1, (0.0, 2.5) => 1)
        @test sampled.barcodes[1, 1] == expected
        M = DI.materialize_module(enc.M)
        @test M.dims == [2, expected_dim, expected_dim]
        @test FL.rank(field, MD.map_leq(M, 1, 2)) == expected_dim
        @test FL.rank(field, MD.map_leq(M, 2, 3)) == expected_dim
        if expected_dim == 1
            A = Matrix(MD.map_leq(M, 1, 2))
            @test A[:, 1] == A[:, 2]
        end
    finally
        DI._ENCODING_RESULT_LAZY_MODULE[] = old_lazy
    end
end

@testset "Data pipeline: Euler distinguishes chosen homology from the whole complex" begin
    # The unfilled triangle has three isolated vertices at zero and one
    # connected cycle at one: H0=(3,1), H1=(0,1), total Euler=(3,0).
    B = sparse([1, 2, 1, 3, 2, 3], [1, 1, 2, 2, 3, 3],
               [-1, 1, -1, 1, -1, 1], 3, 3)
    G = DT.GradedComplex([Int[1, 2, 3], Int[1, 2, 3]], [B],
        [(0.0,), (0.0,), (0.0,), (1.0,), (1.0,), (1.0,)])
    spec = OPT.FiltrationSpec(kind=:graded, axes=([0.0, 1.0],))
    opts = OPT.InvariantOptions(threads=false)
    terms(pm) = Dict(pm.axes[1][idx[1]] => wt for (idx, wt) in zip(pm.inds, pm.wts))
    old_lazy = DI._ENCODING_RESULT_LAZY_MODULE[]
    try
        DI._ENCODING_RESULT_LAZY_MODULE[] = true
        h0 = TamerOp.encode(G, spec; degree=0, field=field, stage=:encoding_result)
        @test terms(TamerOp.euler_signed_measure(h0; opts=opts)) == Dict(0.0 => 3, 1.0 => -2)
        @test h0.M.cached_module === nothing
        h1 = TamerOp.encode(G, spec; degree=1, field=field, stage=:encoding_result)
        @test terms(TamerOp.euler_signed_measure(h1; opts=opts)) == Dict(1.0 => 1)
        whole = TamerOp.encode(G, spec; field=field, stage=:encoded_complex)
        @test terms(TamerOp.euler_signed_measure(whole; opts=opts)) == Dict(0.0 => 3, 1.0 => -3)
        DI._ENCODING_RESULT_LAZY_MODULE[] = false
        eager = TamerOp.encode(G, spec; degree=0, field=field, stage=:encoding_result)
        @test terms(TamerOp.euler_signed_measure(eager; opts=opts)) == Dict(0.0 => 3, 1.0 => -2)
    finally
        DI._ENCODING_RESULT_LAZY_MODULE[] = old_lazy
    end
end
end # with_fields

@testset "Data pipeline: degree-local all-t keeps local term materialization" begin
    data = TamerOp.PointCloud([[0.0], [0.3], [0.8], [1.4], [1.9], [2.2]])
    spec = TamerOp.FiltrationSpec(
        kind=:rips,
        max_dim=3,
        construction=OPT.ConstructionOptions(; output_stage=:graded_complex),
    )
    G = TamerOp.encode(data, spec; degree=0, stage=:graded_complex, cache=:auto)
    axes = get(spec.params, :axes, (collect(range(0.0, stop=2.2, length=18)),))
    P = DI.poset_from_axes(axes)

    old_fast = DI._COHOMOLOGY_DEGREE_LOCAL_FASTPATH[]
    old_allt = DI._COHOMOLOGY_DEGREE_LOCAL_ALL_T[]
    try
        DI._COHOMOLOGY_DEGREE_LOCAL_FASTPATH[] = true
        DI._COHOMOLOGY_DEGREE_LOCAL_ALL_T[] = true
        L = DI._lazy_cochain_complex_from_graded_complex(G, P, axes; field=CM.F2())
        M = DI._cohomology_module_from_lazy(L, 2)
        @test M isa MD.PModule
        @test L.terms[1] === nothing
        @test L.terms[2] !== nothing
        @test L.terms[3] !== nothing
        @test L.terms[4] !== nothing
        @test L.diffs[1] === nothing
        @test L.diffs[2] !== nothing
        @test L.diffs[3] !== nothing
    finally
        DI._COHOMOLOGY_DEGREE_LOCAL_FASTPATH[] = old_fast
        DI._COHOMOLOGY_DEGREE_LOCAL_ALL_T[] = old_allt
    end
end

@testset "Data pipeline: low-dim H0 union-find forced parity" begin
    old_min_pos = DI._H0_UNIONFIND_MIN_POS_VERTICES[]
    old_min_v = DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_VERTICES[]
    old_min_e = DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_EDGES[]
    try
        DI._H0_UNIONFIND_MIN_POS_VERTICES[] = 0
        DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_VERTICES[] = 0
        DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_EDGES[] = 0

        data = TamerOp.PointCloud([[0.0], [0.4], [0.9], [1.3], [1.8]])
        spec_gc = TamerOp.FiltrationSpec(
            kind=:rips,
            max_dim=1,
            axes=([0.0, 0.5, 1.0, 1.5, 2.0],),
            construction=OPT.ConstructionOptions(; output_stage=:graded_complex),
        )
        G = TamerOp.encode(data, spec_gc; degree=0)
        axes = spec_gc.params[:axes]
        P = DI.poset_from_axes(axes)
        L_uf = DI._lazy_cochain_complex_from_graded_complex(G, P, axes; field=CM.F2())
        M_uf = DI._cohomology_module_from_lazy(L_uf, 0)
        L_generic = DI._lazy_cochain_complex_from_graded_complex(G, P, axes; field=CM.F2())
        M_generic = DI._cohomology_module_from_lazy_generic(L_generic, 0)
        @test M_uf.dims == M_generic.dims
        for (u, v) in FF.cover_edges(M_uf.Q)
            @test M_uf.edge_maps[u, v] isa DI._StructuralInclusionMap
        end
        for (u, v) in FF.cover_edges(M_uf.Q)
            @test FL.rank(CM.F2(), M_uf.edge_maps[u, v]) ==
                  FL.rank(CM.F2(), M_generic.edge_maps[u, v])
        end

        # Invalid edge boundary columns force union-find path to fall back safely.
        cells_bad = [Int[1, 2], Int[1]]
        boundaries_bad = [sparse([1], [1], [1], 2, 1)]
        grades_bad = [Float64[0.0], Float64[0.0], Float64[1.0]]
        G_bad = TamerOp.GradedComplex(cells_bad, boundaries_bad, grades_bad)
        axes_bad = ([0.0, 1.0],)
        P_bad = DI.poset_from_axes(axes_bad)
        L_bad = DI._lazy_cochain_complex_from_graded_complex(G_bad, P_bad, axes_bad; field=CM.F2())
        M_bad = DI._cohomology_module_from_lazy(L_bad, 0)
        L_bad_generic = DI._lazy_cochain_complex_from_graded_complex(G_bad, P_bad, axes_bad; field=CM.F2())
        M_bad_generic = DI._cohomology_module_from_lazy_generic(L_bad_generic, 0)
        @test M_bad.dims == M_bad_generic.dims
        for (u, v) in FF.cover_edges(M_bad.Q)
            @test FL.rank(CM.F2(), M_bad.edge_maps[u, v]) ==
                  FL.rank(CM.F2(), M_bad_generic.edge_maps[u, v])
        end
    finally
        DI._H0_UNIONFIND_MIN_POS_VERTICES[] = old_min_pos
        DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_VERTICES[] = old_min_v
        DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_EDGES[] = old_min_e
    end
end

@testset "Data pipeline: H0 chain-sweep fast path parity" begin
    old_chain = DI._H0_CHAIN_SWEEP_FASTPATH[]
    old_min_pos = DI._H0_UNIONFIND_MIN_POS_VERTICES[]
    old_min_v = DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_VERTICES[]
    old_min_e = DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_EDGES[]
    try
        # Keep union-find in lazy path enabled so the comparison isolates chain-sweep
        # vs existing low-dim H0 handling.
        DI._H0_UNIONFIND_MIN_POS_VERTICES[] = 0
        DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_VERTICES[] = 0
        DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_EDGES[] = 0

        data = TamerOp.PointCloud([[0.0], [0.4], [0.9], [1.3], [1.7]])
        spec = TamerOp.FiltrationSpec(
            kind=:rips,
            max_dim=1,
            axes=([0.0, 0.5, 1.0, 1.5, 2.0],),
            construction=OPT.ConstructionOptions(; output_stage=:encoding_result),
        )
        DI._H0_CHAIN_SWEEP_FASTPATH[] = false
        M_base = TamerOp.encode(data, spec; degree=0, stage=:module, cache=:auto)
        DI._H0_CHAIN_SWEEP_FASTPATH[] = true
        M_fast = TamerOp.encode(data, spec; degree=0, stage=:module, cache=:auto)
        @test M_fast.dims == M_base.dims
        for (u, v) in FF.cover_edges(M_fast.Q)
            @test FL.rank(M_fast.field, M_fast.edge_maps[u, v]) ==
                  FL.rank(M_base.field, M_base.edge_maps[u, v])
        end

        edges = [(1, 2), (2, 3), (3, 4), (4, 5), (1, 5)]
        weights = [0.2, 0.4, 0.7, 1.0, 1.3]
        gdata = TamerOp.GraphData(5, edges; weights=weights)
        gspec = TamerOp.FiltrationSpec(
            kind=:graph_weight_threshold,
            max_dim=1,
            construction=OPT.ConstructionOptions(; output_stage=:encoding_result),
        )
        DI._H0_CHAIN_SWEEP_FASTPATH[] = false
        Mg_base = TamerOp.encode(gdata, gspec; degree=0, stage=:module, cache=:auto)
        DI._H0_CHAIN_SWEEP_FASTPATH[] = true
        Mg_fast = TamerOp.encode(gdata, gspec; degree=0, stage=:module, cache=:auto)
        @test Mg_fast.dims == Mg_base.dims
        for (u, v) in FF.cover_edges(Mg_fast.Q)
            @test FL.rank(Mg_fast.field, Mg_fast.edge_maps[u, v]) ==
                  FL.rank(Mg_base.field, Mg_base.edge_maps[u, v])
        end
    finally
        DI._H0_CHAIN_SWEEP_FASTPATH[] = old_chain
        DI._H0_UNIONFIND_MIN_POS_VERTICES[] = old_min_pos
        DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_VERTICES[] = old_min_v
        DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_EDGES[] = old_min_e
    end
end

@testset "Data pipeline: simplicial-boundary specialized kernels parity" begin
    old_specialized = DI._SIMPLICIAL_BOUNDARY_SPECIALIZED[]
    try
        # K=2: edges -> vertices
        faces0 = [[1], [2], [3], [4]]
        simplices1 = [[1, 2], [2, 3], [3, 4], [1, 4]]
        DI._SIMPLICIAL_BOUNDARY_SPECIALIZED[] = false
        B_hash_1 = DI._simplicial_boundary(simplices1, faces0)
        DI._SIMPLICIAL_BOUNDARY_SPECIALIZED[] = true
        B_spec_1 = DI._simplicial_boundary(simplices1, faces0)
        @test Matrix(B_spec_1) == Matrix(B_hash_1)

        # K=3: triangles -> edges
        faces1 = [[1, 2], [1, 3], [2, 3], [1, 4], [3, 4]]
        simplices2 = [[1, 2, 3], [1, 3, 4]]
        DI._SIMPLICIAL_BOUNDARY_SPECIALIZED[] = false
        B_hash_2 = DI._simplicial_boundary(simplices2, faces1)
        DI._SIMPLICIAL_BOUNDARY_SPECIALIZED[] = true
        B_spec_2 = DI._simplicial_boundary(simplices2, faces1)
        @test Matrix(B_spec_2) == Matrix(B_hash_2)
    finally
        DI._SIMPLICIAL_BOUNDARY_SPECIALIZED[] = old_specialized
    end
end

@testset "Data pipeline: NN backend parity (extension-aware)" begin
    if DI._have_pointcloud_nn_backend()
        n = 64
        d = 24
        pts = [collect(range(0.0, stop=1.0, length=d)) .+ 0.01 * i for i in 1:n]
        data = TamerOp.PointCloud(pts)
        cons = OPT.ConstructionOptions(; sparsify=:knn, output_stage=:simplex_tree)
        spec_bf = TamerOp.FiltrationSpec(
            kind=:rips,
            max_dim=1,
            knn=8,
            nn_backend=:bruteforce,
            construction=cons,
        )
        spec_nn = TamerOp.FiltrationSpec(
            kind=:rips,
            max_dim=1,
            knn=8,
            nn_backend=:nearestneighbors,
            construction=cons,
        )
        spec_ap = TamerOp.FiltrationSpec(
            kind=:rips,
            max_dim=1,
            knn=8,
            nn_backend=:approx,
            nn_approx_candidates=n, # candidate=n forces exact parity path
            construction=cons,
        )

        st_bf = TamerOp.encode(data, spec_bf; degree=0, stage=:simplex_tree, cache=:auto)
        st_nn = TamerOp.encode(data, spec_nn; degree=0, stage=:simplex_tree, cache=:auto)
        st_ap = TamerOp.encode(data, spec_ap; degree=0, stage=:simplex_tree, cache=:auto)

        function _edge_signature(st)
            out = Tuple{Int,Int,Float64}[]
            for sid in 1:DT.simplex_count(st)
                st.simplex_dims[sid] == 1 || continue
                verts = DT.simplex_vertices(st, sid)
                a, b = Int(verts[1]), Int(verts[2])
                a > b && ((a, b) = (b, a))
                g = DT.simplex_grades(st, sid)
                push!(out, (a, b, Float64(g[1][1])))
            end
            sort!(out)
            return out
        end

        @test _edge_signature(st_nn) == _edge_signature(st_bf)
        @test _edge_signature(st_ap) == _edge_signature(st_bf)
    else
        @test true
    end
end

@testset "Data pipeline: monotone incremental rank parity" begin
    n = 48
    cells0 = collect(1:n)
    cells1 = collect(1:(n - 1))
    I = Int[]
    J = Int[]
    V = Int[]
    @inbounds for e in 1:(n - 1)
        push!(I, e); push!(J, e); push!(V, 1)
        push!(I, e + 1); push!(J, e); push!(V, -1)
    end
    B = sparse(I, J, V, n, n - 1)
    grades = vcat([Float64[0.0] for _ in 1:n], [Float64[0.6] for _ in 1:(n - 1)])
    data = TamerOp.GradedComplex([cells0, cells1], [B], grades)
    axes = (collect(range(0.0, stop=1.0, length=40)),)
    spec = TamerOp.FiltrationSpec(
        kind=:graded,
        axes=axes,
        construction=OPT.ConstructionOptions(; output_stage=:encoding_result),
    )
    pipeline = TamerOp.PipelineOptions(field=CM.F2())

    old_fast = DI._COHOMOLOGY_DIMS_MONOTONE_RANK_FASTPATH[]
    old_inc = DI._COHOMOLOGY_DIMS_MONOTONE_INCREMENTAL_RANK[]
    try
        DI._COHOMOLOGY_DIMS_MONOTONE_RANK_FASTPATH[] = true
        DI._COHOMOLOGY_DIMS_MONOTONE_INCREMENTAL_RANK[] = true
        d_inc = TamerOp.encode(data, spec; degree=1, stage=:cohomology_dims, pipeline=pipeline, cache=:auto)
        DI._COHOMOLOGY_DIMS_MONOTONE_INCREMENTAL_RANK[] = false
        d_base = TamerOp.encode(data, spec; degree=1, stage=:cohomology_dims, pipeline=pipeline, cache=:auto)
        @test d_inc.dims == d_base.dims
    finally
        DI._COHOMOLOGY_DIMS_MONOTONE_RANK_FASTPATH[] = old_fast
        DI._COHOMOLOGY_DIMS_MONOTONE_INCREMENTAL_RANK[] = old_inc
    end
end

@testset "Data pipeline: structural-map kernel parity" begin
    field = CM.F2()
    A = DI._StructuralInclusionMap{CM.coeff_type(field)}(6, 5, [1, 3, 3, 0, 6])
    rows = [1, 2, 3, 5, 6]
    cols = [1, 2, 3, 5]
    old_struct = DI._STRUCTURAL_MAP_FAST_KERNELS[]
    try
        DI._STRUCTURAL_MAP_FAST_KERNELS[] = true
        rf = FL.rank_restricted(field, A, rows, cols)
        Zf = FL.nullspace(field, A)
        Cf = FL.colspace(field, A)

        DI._STRUCTURAL_MAP_FAST_KERNELS[] = false
        rb = FL.rank_restricted(field, A, rows, cols)
        Zb = FL.nullspace(field, A)
        Cb = FL.colspace(field, A)

        @test rf == rb
        @test size(Zf, 2) == size(Zb, 2)
        @test Matrix(A) * Matrix(Zf) == zeros(CM.coeff_type(field), size(A, 1), size(Zf, 2))
        @test FL.rank(field, Cf) == FL.rank(field, Cb)
    finally
        DI._STRUCTURAL_MAP_FAST_KERNELS[] = old_struct
    end
end

@testset "Data pipeline: active-list chain fast path parity" begin
    vertex_idxs = [(i,) for i in 1:8]
    orientation = (1,)
    births = [(1,), (3,), (2,), (5,), (5,), (7,), (2,)]
    mbirths = [[(1,), (4,)], [(2,)], [(3,), (5,)], [(6,), (7,)]]
    old_active = DI._ACTIVE_LISTS_CHAIN_FASTPATH[]
    try
        DI._ACTIVE_LISTS_CHAIN_FASTPATH[] = true
        a_fast = DI._active_lists(births, vertex_idxs, orientation; multicritical=:union)
        au_fast = DI._active_lists(mbirths, vertex_idxs, orientation; multicritical=:union)
        ai_fast = DI._active_lists(mbirths, vertex_idxs, orientation; multicritical=:intersection)

        DI._ACTIVE_LISTS_CHAIN_FASTPATH[] = false
        a_base = DI._active_lists(births, vertex_idxs, orientation; multicritical=:union)
        au_base = DI._active_lists(mbirths, vertex_idxs, orientation; multicritical=:union)
        ai_base = DI._active_lists(mbirths, vertex_idxs, orientation; multicritical=:intersection)

        @test a_fast == a_base
        @test au_fast == au_base
        @test ai_fast == ai_base
    finally
        DI._ACTIVE_LISTS_CHAIN_FASTPATH[] = old_active
    end
end

@testset "Data pipeline: lazy diff threaded parity" begin
    if Threads.nthreads() > 1
        data = TamerOp.PointCloud([[0.0], [0.25], [0.5], [0.75], [1.0], [1.25]])
        spec_gc = TamerOp.FiltrationSpec(
            kind=:rips,
            max_dim=2,
            axes=([0.0, 0.3, 0.6, 0.9, 1.2],),
            construction=OPT.ConstructionOptions(; output_stage=:graded_complex),
        )
        G = TamerOp.encode(data, spec_gc; degree=0)
        axes = spec_gc.params[:axes]
        P = DI.poset_from_axes(axes)
        L = DI._lazy_cochain_complex_from_graded_complex(G, P, axes; field=CM.QQField())
        DI._lazy_ensure_active!(L, 1)
        DI._lazy_ensure_active!(L, 2)
        comps_serial = DI._lazy_diff_components(L, 1; threaded=false)
        comps_threaded = DI._lazy_diff_components(L, 1; threaded=true)
        @test length(comps_serial) == length(comps_threaded)
        for i in eachindex(comps_serial)
            @test Array(comps_serial[i]) == Array(comps_threaded[i])
        end
    else
        @test true
    end
end

@testset "Data pipeline: structural inclusion map term-builder contract" begin
    data = TamerOp.PointCloud([[0.0], [0.4], [0.9], [1.3]])
    spec_gc = TamerOp.FiltrationSpec(
        kind=:rips,
        max_dim=1,
        axes=([0.0, 0.5, 1.0, 1.5],),
        construction=OPT.ConstructionOptions(; output_stage=:graded_complex),
    )
    G = TamerOp.encode(data, spec_gc; degree=0)
    axes = spec_gc.params[:axes]
    P = DI.poset_from_axes(axes)

    L = DI._lazy_cochain_complex_from_graded_complex(G, P, axes; field=CM.QQField())
    T = DI._lazy_term_idx!(L, 1)
    for (u, v) in FF.cover_edges(T.Q)
        @test T.edge_maps[u, v] isa DI._StructuralInclusionMap
    end
end

@testset "Data pipeline: dense non-sparse point-cloud streaming distance parity" begin
    old_stream = DI._POINTCLOUD_STREAM_DIST_NONSPARSE[]
    try
        data = TamerOp.PointCloud([[0.0], [0.25], [0.5], [0.75], [1.0], [1.25]])
        spec = TamerOp.FiltrationSpec(
            kind=:rips,
            max_dim=2,
            construction=OPT.ConstructionOptions(; sparsify=:none, output_stage=:graded_complex),
        )
        DI._POINTCLOUD_STREAM_DIST_NONSPARSE[] = true
        G_stream = TamerOp.encode(data, spec; degree=0, stage=:graded_complex)
        DI._POINTCLOUD_STREAM_DIST_NONSPARSE[] = false
        G_packed = TamerOp.encode(data, spec; degree=0, stage=:graded_complex)
        @test G_stream.cells_by_dim == G_packed.cells_by_dim
        @test G_stream.boundaries == G_packed.boundaries
        @test G_stream.grades == G_packed.grades
    finally
        DI._POINTCLOUD_STREAM_DIST_NONSPARSE[] = old_stream
    end
end

@testset "Data pipeline: lowdim finite-radius streaming parity (rips_density)" begin
    old_stream = DI._POINTCLOUD_LOWDIM_RADIUS_STREAMING[]
    try
        data = TamerOp.PointCloud([[0.0], [0.2], [0.55], [0.9], [1.3], [1.8]])
        spec = TamerOp.FiltrationSpec(
            kind=:rips_density,
            max_dim=1,
            radius=0.75,
            density_k=2,
            nn_backend=:bruteforce,
            construction=OPT.ConstructionOptions(; sparsify=:none, output_stage=:graded_complex),
        )
        DI._POINTCLOUD_LOWDIM_RADIUS_STREAMING[] = true
        G_stream = TamerOp.encode(data, spec; degree=0, stage=:graded_complex)
        DI._POINTCLOUD_LOWDIM_RADIUS_STREAMING[] = false
        G_dense = TamerOp.encode(data, spec; degree=0, stage=:graded_complex)
        @test G_stream.cells_by_dim == G_dense.cells_by_dim
        @test G_stream.boundaries == G_dense.boundaries
        @test G_stream.grades == G_dense.grades
    finally
        DI._POINTCLOUD_LOWDIM_RADIUS_STREAMING[] = old_stream
    end
end

@testset "Data pipeline: graph clique enumeration parity" begin
    old_enum = DI._GRAPH_CLIQUE_ENUM_MODE[]
    try
        data = TamerOp.GraphData(
            6,
            [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (3, 5), (4, 5), (4, 6), (5, 6)],
        )
        vg = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8]
        spec = TamerOp.FiltrationSpec(
            kind=:clique_lower_star,
            max_dim=2,
            vertex_grades=vg,
            simplex_agg=:max,
            construction=OPT.ConstructionOptions(; output_stage=:graded_complex),
        )
        DI._GRAPH_CLIQUE_ENUM_MODE[] = :intersection
        G_fast = TamerOp.encode(data, spec; degree=0, stage=:graded_complex)
        DI._GRAPH_CLIQUE_ENUM_MODE[] = :combinations
        G_base = TamerOp.encode(data, spec; degree=0, stage=:graded_complex)
        DI._GRAPH_CLIQUE_ENUM_MODE[] = :auto
        G_auto = TamerOp.encode(data, spec; degree=0, stage=:graded_complex)
        @test G_fast.cells_by_dim == G_base.cells_by_dim
        @test G_fast.boundaries == G_base.boundaries
        @test G_fast.grades == G_base.grades
        @test G_auto.cells_by_dim == G_base.cells_by_dim
        @test G_auto.boundaries == G_base.boundaries
        @test G_auto.grades == G_base.grades

        w = [1.0 + 0.1 * i for i in eachindex(data.edges)]
        spec_w = TamerOp.FiltrationSpec(
            kind=:graph_weight_threshold,
            lift=:clique,
            max_dim=2,
            edge_weights=w,
            construction=OPT.ConstructionOptions(; output_stage=:graded_complex),
        )
        DI._GRAPH_CLIQUE_ENUM_MODE[] = :intersection
        Gw_fast = TamerOp.encode(data, spec_w; degree=0, stage=:graded_complex)
        DI._GRAPH_CLIQUE_ENUM_MODE[] = :combinations
        Gw_base = TamerOp.encode(data, spec_w; degree=0, stage=:graded_complex)
        DI._GRAPH_CLIQUE_ENUM_MODE[] = :auto
        Gw_auto = TamerOp.encode(data, spec_w; degree=0, stage=:graded_complex)
        @test Gw_fast.cells_by_dim == Gw_base.cells_by_dim
        @test Gw_fast.boundaries == Gw_base.boundaries
        @test Gw_fast.grades == Gw_base.grades
        @test Gw_auto.cells_by_dim == Gw_base.cells_by_dim
        @test Gw_auto.boundaries == Gw_base.boundaries
        @test Gw_auto.grades == Gw_base.grades
    finally
        DI._GRAPH_CLIQUE_ENUM_MODE[] = old_enum
    end
end

@testset "Data pipeline: point-cloud graded_complex stage returns graded complex" begin
    data = TamerOp.PointCloud([[0.0], [1.0], [2.0]])
    spec = TamerOp.FiltrationSpec(
        kind=:rips,
        max_dim=1,
        construction=TamerOp.ConstructionOptions(; output_stage=:graded_complex),
    )
    G = TamerOp.encode(data, spec; degree=0)
    @test G isa TamerOp.GradedComplex
    @test !isempty(G.cells_by_dim)
end

@testset "Data pipeline: simplex-tree eps quantization parity" begin
    data = TamerOp.PointCloud([[0.0], [0.41], [0.93]])
    st = TamerOp.encode(
        data,
        TamerOp.FiltrationSpec(
            kind=:rips,
            max_dim=1,
            construction=TamerOp.ConstructionOptions(; output_stage=:simplex_tree),
        );
        degree=0,
    )
    @test st isa DI.SimplexTreeMulti

    spec_eps = TamerOp.FiltrationSpec(kind=:graded, eps=0.25)
    enc_tree = TamerOp.encode(st, spec_eps; degree=0)
    G = DI._graded_complex_from_simplex_tree(st)
    enc_grad = TamerOp.encode(G, spec_eps; degree=0)
    M_tree = _enc_module(enc_tree)
    M_grad = _enc_module(enc_grad)
    @test M_tree.dims == M_grad.dims
    for (u, v) in FF.cover_edges(M_tree.Q)
        @test Array(M_tree.edge_maps[u, v]) == Array(M_grad.edge_maps[u, v])
    end
end

@testset "Data pipeline: simplex-tree one_critical parity" begin
    cells = [Int[1, 2], Int[1]]
    B1 = sparse([1, 2], [1, 1], [1, -1], 2, 1)
    grades = [
        [Float64[0.0, 0.0]],
        [Float64[0.0, 0.0]],
        [Float64[1.0, 0.0], Float64[0.0, 1.0], Float64[1.0, 1.0]],
    ]
    Gm = TamerOp.MultiCriticalGradedComplex(cells, [B1], grades)
    st = DI._simplex_tree_multi_from_complex(Gm)
    @test st isa DI.SimplexTreeMulti

    spec_one = TamerOp.FiltrationSpec(
        kind=:graded,
        multicritical=:one_critical,
        onecritical_selector=:lexmin,
        onecritical_enforce_boundary=true,
    )
    enc_tree = TamerOp.encode(st, spec_one; degree=0)
    enc_grad = TamerOp.encode(Gm, spec_one; degree=0)
    M_tree = _enc_module(enc_tree)
    M_grad = _enc_module(enc_grad)
    @test M_tree.dims == M_grad.dims
    for (u, v) in FF.cover_edges(M_tree.Q)
        @test Array(M_tree.edge_maps[u, v]) == Array(M_grad.edge_maps[u, v])
    end
end

@testset "Data pipeline: packed simplex-tree complex conversions" begin
    cells = [Int[1, 2, 3], Int[1, 2, 3], Int[1]]
    B1 = sparse(
        [1, 2, 2, 3, 1, 3],
        [1, 1, 2, 2, 3, 3],
        [1, -1, 1, -1, 1, -1],
        3, 3,
    )
    B2 = sparse([1, 2, 3], [1, 1, 1], [1, -1, 1], 3, 1)
    Gg = TamerOp.GradedComplex(
        cells,
        [B1, B2],
        [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (2.0, 2.0)],
    )
    Gm = TamerOp.MultiCriticalGradedComplex(
        cells,
        [B1, B2],
        [
            [(0.0, 0.0)],
            [(1.0, 0.0)],
            [(0.0, 1.0)],
            [(1.0, 0.0)],
            [(1.0, 1.0)],
            [(0.0, 1.0)],
            [(1.0, 1.0), (2.0, 2.0)],
        ],
    )

    stg = DI._simplex_tree_multi_from_complex(Gg)
    stm = DI._simplex_tree_multi_from_complex(Gm)
    @test stg.grade_offsets == collect(1:(length(Gg.grades) + 1))
    @test stg.grade_data == Gg.grades
    @test stm.grade_offsets == getfield(Gm, :grade_offsets)
    @test stm.grade_data == getfield(Gm, :grade_data)

    Gg_rt = DI._graded_complex_from_simplex_tree(stg)
    Gm_rt = DI._graded_complex_from_simplex_tree(stm)
    @test Gg_rt.grades == Gg.grades
    @test collect(Gm_rt.grades[7]) == collect(Gm.grades[7])
    @test DI._axes_from_simplex_tree(stm; orientation=(1, 1)) == ([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
end

@testset "Data pipeline: graded complex" begin
    cells = [Int[1]]
    boundaries = SparseMatrixCSC{Int,Int}[]
    grades = [Float64[0.0]]
    G = TamerOp.GradedComplex(cells, boundaries, grades)
    spec = TamerOp.FiltrationSpec(kind=:graded, axes=([0.0, 1.0],))
    enc = TamerOp.encode(G, spec; degree=0)
    @test _enc_dims(enc) == [1, 1]

    H = TamerOp.Workflow.fringe_presentation(DI.materialize_module(enc.M))
    Mp = IR.pmodule_from_fringe(H)
    @test Mp.dims == _enc_dims(enc)
end

@testset "Data pipeline: point cloud rips" begin
    data = TamerOp.PointCloud([[0.0], [1.0]])
    spec = TamerOp.FiltrationSpec(kind=:rips, max_dim=1, axes=([0.0, 1.0],))
    enc = TamerOp.encode(data, spec; degree=0)
    @test _enc_dims(enc) == [2, 1]
    bc = Inv.slice_barcode(_enc_module(enc), [1, 2])
    @test bc[(1, 2)] == 1
    @test bc[(1, 3)] == 1
end

@testset "Data pipeline: point cloud rips higher-dim" begin
    data = TamerOp.PointCloud([[0.0], [1.0], [2.0]])
    spec = TamerOp.FiltrationSpec(kind=:rips, max_dim=2, axes=([0.0, 1.0, 2.0],))
    enc = TamerOp.encode(data, spec; degree=0)
    @test _enc_dims(enc) == [3, 1, 1]
end

@testset "Data pipeline: point cloud dense rips d2 oracle" begin
    data = TamerOp.PointCloud([[0.0], [1.0], [3.0], [6.0]])
    spec = TamerOp.FiltrationSpec(
        kind=:rips,
        max_dim=2,
        construction=TamerOp.ConstructionOptions(; sparsify=:none, output_stage=:graded_complex),
    )
    G = TamerOp.encode(data, spec; stage=:graded_complex)
    @test G isa TamerOp.GradedComplex
    @test length(G.cells_by_dim) == 3
    @test length(G.cells_by_dim[1]) == 4
    @test length(G.cells_by_dim[2]) == 6
    @test length(G.cells_by_dim[3]) == 4
    @test G.grades[1:4] == [(0.0,), (0.0,), (0.0,), (0.0,)]
    @test G.grades[5:10] == [(1.0,), (3.0,), (6.0,), (2.0,), (5.0,), (3.0,)]
    @test G.grades[11:14] == [(3.0,), (6.0,), (6.0,), (5.0,)]
end

@testset "Data pipeline: packed pairwise distance oracle" begin
    points = [[0.0, 0.0], [1.0, 0.0], [1.0, 2.0], [3.0, 2.0]]
    packed = DI._point_cloud_pairwise_packed(points)
    n = length(points)
    dist = Matrix{Float64}(undef, n, n)
    @inbounds for i in 1:n
        dist[i, i] = 0.0
        pi = points[i]
        for j in (i + 1):n
            pj = points[j]
            s = 0.0
            for k in eachindex(pi)
                d = Float64(pi[k]) - Float64(pj[k])
                s += d * d
            end
            dij = sqrt(s)
            dist[i, j] = dij
            dist[j, i] = dij
        end
    end
    @test length(packed) == div(n * (n - 1), 2)
    for i in 1:n
        for j in 1:n
            dij = DI._packed_pair_distance(packed, n, i, j)
            @test isapprox(dij, dist[i, j]; atol=1e-12, rtol=0.0)
        end
    end
end

@testset "Data pipeline: low-dim point-cloud oracle kernels" begin
    data = TamerOp.PointCloud([[0.0], [2.0], [5.0]])

    rips_spec = TamerOp.FiltrationSpec(
        kind=:rips,
        max_dim=1,
        construction=TamerOp.ConstructionOptions(; sparsify=:none, output_stage=:graded_complex),
    )
    G_rips = TamerOp.encode(data, rips_spec; stage=:graded_complex)
    @test G_rips isa TamerOp.GradedComplex
    @test length(G_rips.cells_by_dim[1]) == 3
    @test length(G_rips.cells_by_dim[2]) == 3
    @test G_rips.grades[1:3] == [(0.0,), (0.0,), (0.0,)]
    @test G_rips.grades[4:6] == [(2.0,), (5.0,), (3.0,)]
    Br = G_rips.boundaries[1]
    @test size(Br) == (3, 3)
    @test Br[1, 1] == -1 && Br[2, 1] == 1
    @test Br[1, 2] == -1 && Br[3, 2] == 1
    @test Br[2, 3] == -1 && Br[3, 3] == 1

    fr_spec = TamerOp.FiltrationSpec(
        kind=:function_rips,
        max_dim=1,
        vertex_values=[1.0, 4.0, 10.0],
        simplex_agg=:sum,
        construction=TamerOp.ConstructionOptions(; sparsify=:none, output_stage=:graded_complex),
    )
    G_fr = TamerOp.encode(data, fr_spec; stage=:graded_complex)
    @test G_fr.grades[1:3] == [(0.0, 1.0), (0.0, 4.0), (0.0, 10.0)]
    @test G_fr.grades[4:6] == [(2.0, 5.0), (5.0, 11.0), (3.0, 14.0)]

    rd_spec = TamerOp.FiltrationSpec(
        kind=:rips_density,
        max_dim=1,
        density_k=1,
        construction=TamerOp.ConstructionOptions(; sparsify=:none, output_stage=:graded_complex),
    )
    G_rd = TamerOp.encode(data, rd_spec; stage=:graded_complex)
    @test G_rd.grades[1:3] == [(0.0, 2.0), (0.0, 2.0), (0.0, 3.0)]
    @test G_rd.grades[4:6] == [(2.0, 2.0), (5.0, 3.0), (3.0, 3.0)]

    rc_spec = TamerOp.FiltrationSpec(
        kind=:rips_codensity,
        max_dim=1,
        dtm_mass=0.5,
        construction=TamerOp.ConstructionOptions(; sparsify=:none, output_stage=:graded_complex),
    )
    G_rc = TamerOp.encode(data, rc_spec; stage=:graded_complex)
    rc_expected_vertices = [(0.0, sqrt(2.0)), (0.0, sqrt(2.0)), (0.0, 3 / sqrt(2.0))]
    rc_expected_edges = [(2.0, sqrt(2.0)), (5.0, 3 / sqrt(2.0)), (3.0, 3 / sqrt(2.0))]
    @test all(
        all(isapprox(gi, ei; atol=1e-12, rtol=0.0) for (gi, ei) in zip(g, e))
        for (g, e) in zip(G_rc.grades[1:3], rc_expected_vertices)
    )
    @test all(
        all(isapprox(gi, ei; atol=1e-12, rtol=0.0) for (gi, ei) in zip(g, e))
        for (g, e) in zip(G_rc.grades[4:6], rc_expected_edges)
    )


end

@testset "Data pipeline: typed filtration dispatch" begin
    data = TamerOp.PointCloud([[0.0], [1.0]])
    fspec = TamerOp.FiltrationSpec(kind=:rips, max_dim=1, axes=([0.0, 1.0],))
    ftyped = DI.RipsFiltration(max_dim=1)
    enc_spec = TamerOp.encode(data, fspec; degree=0)
    enc_typed = TamerOp.encode(data, ftyped; degree=0)
    @test _enc_dims(enc_typed) == _enc_dims(enc_spec)
    @test EC.axes_from_encoding(enc_typed.pi) == EC.axes_from_encoding(enc_spec.pi)

    g = TamerOp.GraphData(3, [(1, 2), (2, 3)])
    gfilt = DI.GraphLowerStarFiltration(vertex_values=[0.0, 1.0, 2.0], simplex_agg=:max)
    gspec = TamerOp.FiltrationSpec(kind=:graph_lower_star, vertex_values=[0.0, 1.0, 2.0], simplex_agg=:max)
    enc_g = TamerOp.encode(g, gfilt; degree=0)
    enc_gspec = TamerOp.encode(g, gspec; degree=0)
    @test _enc_dims(enc_g) == _enc_dims(enc_gspec)

    codensity_data = TamerOp.PointCloud([[0.0], [2.0], [5.0]])
    codensity_spec = TamerOp.FiltrationSpec(kind=:rips_codensity, max_dim=1, dtm_mass=0.5)
    codensity_ref = TamerOp.FiltrationSpec(
        kind=:function_rips,
        max_dim=1,
        vertex_values=[sqrt(2.0), sqrt(2.0), 3 / sqrt(2.0)],
        simplex_agg=:max,
    )
    G_codensity = TamerOp.encode(codensity_data, codensity_spec; stage=:graded_complex)
    G_codensity_ref = TamerOp.encode(codensity_data, codensity_ref; stage=:graded_complex)
    @test G_codensity.grades == G_codensity_ref.grades

    ffilt = DI.to_filtration(TamerOp.FiltrationSpec(kind=:rips_density, max_dim=1, density_k=2))
    @test ffilt isa DI.RipsDensityFiltration
    cdfilt = DI.to_filtration(TamerOp.FiltrationSpec(kind=:rips_codensity, max_dim=1, dtm_mass=0.25))
    @test cdfilt isa DI.RipsCodensityFiltration
    lsfilt = DI.to_filtration(TamerOp.FiltrationSpec(kind=:rips_lowerstar, max_dim=1, coord=1))
    @test lsfilt isa DI.RipsLowerStarFiltration

    afilt = DI.to_filtration(TamerOp.FiltrationSpec(kind=:alpha, max_dim=2))
    @test afilt isa DI.AlphaFiltration
    corefilt = DI.to_filtration(TamerOp.FiltrationSpec(kind=:core_delaunay, max_dim=2))
    @test corefilt isa DI.CoreDelaunayFiltration
    drfilt = DI.to_filtration(TamerOp.FiltrationSpec(kind=:degree_rips, max_dim=1))
    @test drfilt isa DI.DegreeRipsFiltration
    cubfilt = DI.to_filtration(TamerOp.FiltrationSpec(kind=:cubical, periodic=(true, false)))
    @test cubfilt isa DI.CubicalFiltration
    @test DI.filtration_parameters(cubfilt).periodic == (true, false)
end

@testset "Data pipeline: periodic cubical ingestion" begin
    img = TamerOp.ImageNd([0.0 1.0; 2.0 3.0])
    G = TamerOp.encode(
        img,
        TamerOp.FiltrationSpec(kind=:cubical, periodic=(true, true));
        stage=:graded_complex,
    )
    @test DT.cell_counts(G) == [4, 8, 4]
    @test size(G.boundaries[1]) == (4, 8)
    @test size(G.boundaries[2]) == (8, 4)
    @test all(g == (3.0,) for g in G.grades[13:16])
    est = DI.estimate_ingestion(
        img,
        TamerOp.FiltrationSpec(kind=:cubical, periodic=(true, true)),
    )
    @test DI.cell_counts_by_dim(est) == BigInt[4, 8, 4]

end

@testset "Data pipeline: distance-matrix Rips parity path" begin
    D2 = [0.0 2.0; 2.0 0.0]
    build = TamerOp.build_graded_complex(D2, TamerOp.RipsFiltration(max_dim=1))
    G2 = DI.graded_complex(build)
    @test DT.cell_counts(G2) == [2, 1]
    @test G2.grades == [(0.0,), (0.0,), (2.0,)]

    D3 = [0.0 1.0 1.0; 1.0 0.0 1.0; 1.0 1.0 0.0]
    G3 = DI.graded_complex(
        TamerOp.build_graded_complex(D3, TamerOp.RipsFiltration(max_dim=2, radius=1.1)),
    )
    @test DT.cell_counts(G3) == [3, 3, 1]
    st3 = TamerOp.encode(
        D3,
        TamerOp.FiltrationSpec(kind=:rips, max_dim=2, radius=1.1);
        stage=:simplex_tree,
    )
    @test DT.cell_counts(st3) == [3, 3, 1]
    @test_throws ArgumentError TamerOp.build_graded_complex([0.0 1.0; 2.0 0.0], TamerOp.RipsFiltration(max_dim=1))
end

@testset "Data pipeline: Delaunay/function-Delaunay filtrations" begin
    pts = TamerOp.PointCloud([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    dspec = TamerOp.FiltrationSpec(kind=:delaunay_lower_star, vertex_values=[0.0, 1.0, 2.0], max_dim=2)
    dtyped = DI.to_filtration(dspec)
    @test dtyped isa DI.DelaunayLowerStarFiltration
    enc_d = TamerOp.encode(pts, dspec; degree=0)
    ax_d = EC.axes_from_encoding(enc_d.pi)
    @test length(ax_d) == 1
    @test ax_d[1] == [0.0, 1.0, 2.0]

    fspec = TamerOp.FiltrationSpec(kind=:function_delaunay, vertex_values=[0.0, 1.0, 2.0], max_dim=2)
    ftyped = DI.to_filtration(fspec)
    @test ftyped isa DI.FunctionDelaunayFiltration
    enc_f = TamerOp.encode(pts, fspec; degree=0)
    ax_f = EC.axes_from_encoding(enc_f.pi)
    @test length(ax_f) == 2
    @test 0.0 in ax_f[1]
    @test 0.0 in ax_f[2] && 2.0 in ax_f[2]

    aspec = TamerOp.FiltrationSpec(kind=:alpha, max_dim=2)
    atyped = DI.to_filtration(aspec)
    @test atyped isa DI.AlphaFiltration
    enc_a = TamerOp.encode(pts, aspec; degree=0)
    ax_a = EC.axes_from_encoding(enc_a.pi)
    @test length(ax_a) == 1
    @test 0.0 in ax_a[1]

    cdspec = TamerOp.FiltrationSpec(kind=:core_delaunay, max_dim=2)
    cdtyped = DI.to_filtration(cdspec)
    @test cdtyped isa DI.CoreDelaunayFiltration
    enc_cd = TamerOp.encode(pts, cdspec; degree=0)
    ax_cd = EC.axes_from_encoding(enc_cd.pi)
    @test length(ax_cd) == 2
    @test 0.0 in ax_cd[1]
    @test !isempty(ax_cd[2])
end

@testset "Data pipeline: Delaunay high-dimensional fallback policy" begin
    pts3d = TamerOp.PointCloud([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    spec_ls = TamerOp.FiltrationSpec(kind=:delaunay_lower_star,
                                          vertex_values=[0.0, 1.0, 2.0, 3.0],
                                          max_dim=2,
                                          highdim_policy=:rips)
    enc_ls = TamerOp.encode(pts3d, spec_ls; degree=0)
    ax_ls = EC.axes_from_encoding(enc_ls.pi)
    @test length(ax_ls) == 1
    @test 0.0 in ax_ls[1] && 3.0 in ax_ls[1]

    spec_fn = TamerOp.FiltrationSpec(kind=:function_delaunay,
                                          vertex_values=[0.0, 1.0, 2.0, 3.0],
                                          max_dim=2)
    @test_throws ArgumentError TamerOp.encode(pts3d, spec_fn; degree=0)
    @test DI.to_filtration(spec_fn) isa DI.FunctionDelaunayFiltration

    spec_err = TamerOp.FiltrationSpec(kind=:delaunay_lower_star,
                                           vertex_values=[0.0, 1.0, 2.0, 3.0],
                                           max_dim=2,
                                           highdim_policy=:error)
    @test_throws ArgumentError TamerOp.encode(pts3d, spec_err; degree=0)

    spec_alpha = TamerOp.FiltrationSpec(kind=:alpha, max_dim=2, highdim_policy=:rips)
    enc_alpha = TamerOp.encode(pts3d, spec_alpha; degree=0)
    @test length(EC.axes_from_encoding(enc_alpha.pi)) == 1

    spec_alpha_err = TamerOp.FiltrationSpec(kind=:alpha, max_dim=2, highdim_policy=:error)
    @test_throws ArgumentError TamerOp.encode(pts3d, spec_alpha_err; degree=0)

    spec_core_del = TamerOp.FiltrationSpec(kind=:core_delaunay, max_dim=2)
    @test_throws ArgumentError TamerOp.encode(pts3d, spec_core_del; degree=0)
    for kind in (:function_delaunay, :core_delaunay), policy in (:rips, :error)
        old_spec = TamerOp.FiltrationSpec(; kind, max_dim=2, highdim_policy=policy)
        @test_throws ArgumentError DI.to_filtration(old_spec)
    end

end

@testset "Data pipeline: graph-core filtrations" begin
    g = TamerOp.GraphData(4, [(1, 2), (2, 3), (1, 3), (3, 4)])
    cspec = TamerOp.FiltrationSpec(kind=:graph_core)
    ctyped = DI.to_filtration(cspec)
    @test ctyped isa DI.GraphCoreFiltration
    enc_c = TamerOp.encode(g, cspec; degree=0)
    ax_c = EC.axes_from_encoding(enc_c.pi)
    @test length(ax_c) == 2
    @test -1.0 in ax_c[2] && -2.0 in ax_c[2]

    p = TamerOp.PointCloud([[0.0], [1.0], [2.0], [3.0]])
    enc_cp = TamerOp.encode(
        p,
        TamerOp.FiltrationSpec(kind=:graph_core, knn=1, vertex_values=[0.0, 0.0, 0.0, 0.0]);
        degree=0,
    )
    @test length(EC.axes_from_encoding(enc_cp.pi)) == 2


end

@testset "Data pipeline: core packed dim01 + edge-only builder parity" begin
    pts = TamerOp.PointCloud([[0.0], [0.3], [0.8], [1.4], [2.1], [2.9], [3.2], [4.0]])

    # Core now routes through edge-only builders; verify edge parity vs full builders.
    spec_knn = TamerOp.FiltrationSpec(kind=:graph_core, knn=3, nn_backend=:bruteforce)
    e_core_knn = DI._core_edges_from_point_cloud(pts.points, spec_knn)
    e_full_knn, _, _ = DI._point_cloud_knn_graph(pts.points, 3; backend=:bruteforce, approx_candidates=0)
    @test sort(e_core_knn) == sort(e_full_knn)

    spec_rad = TamerOp.FiltrationSpec(kind=:graph_core, radius=1.25, nn_backend=:bruteforce)
    e_core_rad = DI._core_edges_from_point_cloud(pts.points, spec_rad)
    e_full_rad, _ = DI._point_cloud_radius_graph(pts.points, 1.25; backend=:bruteforce, approx_candidates=0)
    @test sort(e_core_rad) == sort(e_full_rad)

    if DI._have_pointcloud_nn_backend()
        spec_knn_nn = TamerOp.FiltrationSpec(kind=:graph_core, knn=3, nn_backend=:nearestneighbors)
        e_core_knn_nn = DI._core_edges_from_point_cloud(pts.points, spec_knn_nn)
        e_full_knn_nn, _, _ = DI._point_cloud_knn_graph(pts.points, 3; backend=:nearestneighbors, approx_candidates=0)
        @test sort(e_core_knn_nn) == sort(e_full_knn_nn)

        spec_rad_nn = TamerOp.FiltrationSpec(kind=:graph_core, radius=1.25, nn_backend=:nearestneighbors)
        e_core_rad_nn = DI._core_edges_from_point_cloud(pts.points, spec_rad_nn)
        e_full_rad_nn, _ = DI._point_cloud_radius_graph(pts.points, 1.25; backend=:nearestneighbors, approx_candidates=0)
        @test sort(e_core_rad_nn) == sort(e_full_rad_nn)
    end

    # Point-core path now uses packed dim01 materialization.
    pvals = [0.0, 0.2, 0.1, 0.9, 0.4, 0.8, 0.6, 0.7]
    spec_point = TamerOp.FiltrationSpec(
        kind=:graph_core,
        knn=3,
        nn_backend=:bruteforce,
        vertex_values=pvals,
        construction=OPT.ConstructionOptions(; output_stage=:graded_complex),
    )
    Gp = TamerOp.encode(pts, spec_point; degree=0, stage=:graded_complex, cache=:auto)
    @test Gp isa DT.GradedComplex
    @test length(Gp.cells_by_dim) == 2
    @test length(Gp.cells_by_dim[1]) == length(pts.points)
    @test length(Gp.cells_by_dim[2]) == length(e_core_knn)
    @test size(Gp.boundaries[1], 1) == length(pts.points)
    @test size(Gp.boundaries[1], 2) == length(e_core_knn)

    # Graph-core path now also uses packed dim01 materialization.
    n = 9
    edges = [(1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (7, 9)]
    g = TamerOp.GraphData(n, edges)
    gvals = [Float64(i) / n for i in 1:n]
    spec_graph = TamerOp.FiltrationSpec(
        kind=:graph_core,
        vertex_values=gvals,
        construction=OPT.ConstructionOptions(; output_stage=:graded_complex),
    )
    Gg = TamerOp.encode(g, spec_graph; degree=0, stage=:graded_complex, cache=:auto)
    @test Gg isa DT.GradedComplex
    @test length(Gg.cells_by_dim) == 2
    @test length(Gg.cells_by_dim[1]) == n
    @test length(Gg.cells_by_dim[2]) == length(edges)
    @test size(Gg.boundaries[1], 1) == n
    @test size(Gg.boundaries[1], 2) == length(edges)
end

@testset "Data pipeline: core_numbers oracle parity" begin
    function _core_numbers_naive(n::Int, edges::Vector{NTuple{2,Int}})
        adj = [Int[] for _ in 1:n]
        for (u, v) in edges
            u == v && continue
            push!(adj[u], v)
            push!(adj[v], u)
        end
        deg = [length(adj[v]) for v in 1:n]
        alive = trues(n)
        core = zeros(Int, n)
        remaining = n
        k = 0
        while remaining > 0
            peeled = false
            for v in 1:n
                if alive[v] && deg[v] <= k
                    alive[v] = false
                    core[v] = k
                    remaining -= 1
                    peeled = true
                    for w in adj[v]
                        alive[w] && (deg[w] -= 1)
                    end
                end
            end
            peeled || (k += 1)
        end
        return core
    end

    # hand-computable examples
    @test DI._core_numbers(5, NTuple{2,Int}[(1, 2), (1, 3), (1, 4), (1, 5)]) == [1, 1, 1, 1, 1]
    @test DI._core_numbers(4, NTuple{2,Int}[(1, 2), (2, 3), (1, 3), (3, 4)]) == [2, 2, 2, 1]

    # randomized differential parity with naive baseline
    rng = Random.MersenneTwister(0xD4B1)
    for n in (6, 10, 14)
        for p in (0.18, 0.33, 0.55)
            edges = NTuple{2,Int}[]
            for i in 1:(n - 1), j in (i + 1):n
                rand(rng) < p || continue
                push!(edges, (i, j))
            end
            @test DI._core_numbers(n, edges) == _core_numbers_naive(n, edges)
        end
    end
end

@testset "Data pipeline: degree_rips and cubical filtrations" begin
    p = TamerOp.PointCloud([[0.0], [2.0], [5.0]])
    dr_spec = TamerOp.FiltrationSpec(
        kind=:degree_rips,
        max_dim=1,
        construction=TamerOp.ConstructionOptions(; sparsify=:none, output_stage=:graded_complex),
    )
    G_dr = TamerOp.encode(p, dr_spec; stage=:graded_complex)
    @test G_dr.grades[1:3] == [(0.0, 2.0), (0.0, 2.0), (0.0, 2.0)]
    @test G_dr.grades[4:6] == [(2.0, 2.0), (5.0, 2.0), (3.0, 2.0)]

    img = TamerOp.ImageNd([0.0 1.0; 2.0 3.0])
    spec_cub = TamerOp.FiltrationSpec(kind=:cubical)
    spec_ls = TamerOp.FiltrationSpec(kind=:lower_star)
    G_cub = TamerOp.encode(img, spec_cub; stage=:graded_complex)
    G_ls = TamerOp.encode(img, spec_ls; stage=:graded_complex)
    @test G_cub.cells_by_dim == G_ls.cells_by_dim
    @test G_cub.boundaries == G_ls.boundaries
    @test G_cub.grades == G_ls.grades

    # 2D cubical fast path must be exact-parity with the generic cubical kernel.
    img2 = TamerOp.ImageNd([0.1 0.9 1.2 0.7; 0.4 1.3 0.2 1.1; 0.8 0.6 1.5 0.3])
    spec_cub2 = TamerOp.FiltrationSpec(kind=:cubical)
    spec_bi2 = TamerOp.FiltrationSpec(kind=:image_distance_bifiltration, mask=img2.data .> 0.75)
    old_fast = DI._CUBICAL_2D_FASTPATH[]
    try
        DI._CUBICAL_2D_FASTPATH[] = false
        G_cub_ref = TamerOp.encode(img2, spec_cub2; stage=:graded_complex)
        G_bi_ref = TamerOp.encode(img2, spec_bi2; stage=:graded_complex)

        DI._CUBICAL_2D_FASTPATH[] = true
        G_cub_fast = TamerOp.encode(img2, spec_cub2; stage=:graded_complex)
        G_bi_fast = TamerOp.encode(img2, spec_bi2; stage=:graded_complex)

        @test G_cub_fast.cells_by_dim == G_cub_ref.cells_by_dim
        @test G_cub_fast.boundaries == G_cub_ref.boundaries
        @test G_cub_fast.grades == G_cub_ref.grades

        @test G_bi_fast.cells_by_dim == G_bi_ref.cells_by_dim
        @test G_bi_fast.boundaries == G_bi_ref.boundaries
        @test G_bi_fast.grades == G_bi_ref.grades
    finally
        DI._CUBICAL_2D_FASTPATH[] = old_fast
    end

    # 2D EDT fast path must be exact-parity with a hand-written naive reference.
    function _distance_transform_naive(mask::AbstractMatrix{Bool})
        nx, ny = size(mask)
        out = Matrix{Float64}(undef, nx, ny)
        true_pts = Tuple{Int,Int}[]
        @inbounds for j in 1:ny, i in 1:nx
            mask[i, j] && push!(true_pts, (i, j))
        end
        @inbounds for j in 1:ny, i in 1:nx
            if mask[i, j]
                out[i, j] = 0.0
                continue
            end
            best = Inf
            for (ti, tj) in true_pts
                dx = i - ti
                dy = j - tj
                s = dx * dx + dy * dy
                s < best && (best = s)
            end
            out[i, j] = sqrt(best)
        end
        return out
    end

    rng = Random.MersenneTwister(0xED71)
    for dims in ((4, 5), (6, 7))
        for _ in 1:6
            mask = rand(rng, Bool, dims...)
            dt = DI._distance_transform(mask)
            ref = _distance_transform_naive(mask)
            @test size(dt) == size(ref)
            @inbounds for idx in eachindex(dt)
                if isinf(ref[idx])
                    @test isinf(dt[idx])
                else
                    @test isapprox(dt[idx], ref[idx]; atol=1.0e-12, rtol=0.0)
                end
            end
        end
    end

    @test all(isinf, DI._distance_transform(falses(5, 6)))
    @test DI._distance_transform(trues(3, 4)) == zeros(Float64, 3, 4)

    # Distance-transform caching should reuse the cached array object for repeated calls.
    dt_cache = CM.EncodingCache()
    dt_mask = img2.data .> 0.75
    dt1 = DI._distance_transform_cached(dt_mask; cache=dt_cache)
    dt2 = DI._distance_transform_cached(dt_mask; cache=dt_cache)
    @test dt1 === dt2
    @test !isempty(dt_cache.geometry)
end

@testset "Data pipeline: custom filtration extensibility" begin
    struct ToyPointCloudFiltration{P<:NamedTuple} <: DI.AbstractFiltration
        params::P
    end
    ToyPointCloudFiltration(; construction::OPT.ConstructionOptions=OPT.ConstructionOptions()) =
        ToyPointCloudFiltration((; construction))
    DI.filtration_kind(::Type{<:ToyPointCloudFiltration}) = :toy_point_cloud
    DI.filtration_arity(::ToyPointCloudFiltration, _data=nothing) = 1

    function _toy_builder(data::TamerOp.PointCloud,
                          filtration::ToyPointCloudFiltration;
                          cache::Union{Nothing,CM.EncodingCache}=nothing)
        construction = get(filtration.params, :construction, OPT.ConstructionOptions())
        return DI._graded_complex_from_data(data, DI.RipsFiltration(max_dim=1, construction=construction); cache=cache)
    end
    DI._build_graded_complex_tuple(data::TamerOp.PointCloud,
                                   filtration::ToyPointCloudFiltration;
                                   cache::Union{Nothing,CM.EncodingCache}=nothing) =
        _toy_builder(data, filtration; cache=cache)

    DI.register_filtration_family!(
        kind=:toy_point_cloud,
        ctor=spec -> ToyPointCloudFiltration(construction=DI._construction_from_params(spec.params)),
        builder=_toy_builder,
        arity=1,
    )

    data = TamerOp.PointCloud([[0.0], [1.0]])
    enc = TamerOp.encode(data, ToyPointCloudFiltration(); degree=0)
    @test _enc_dims(enc) == [2, 1]
    enc2 = TamerOp.encode(data, TamerOp.FiltrationSpec(kind=:toy_point_cloud); degree=0)
    @test _enc_dims(enc2) == [2, 1]
end

@testset "Data pipeline: ingestion planning protocol" begin
    data = TamerOp.PointCloud([[0.0], [1.0], [2.0]])
    filt = DI.RipsFiltration(max_dim=1, knn=2)
    spec = TamerOp.FiltrationSpec(kind=:rips, max_dim=1, knn=2)

    construction = OPT.ConstructionOptions(;
        sparsify=:knn,
        collapse=:none,
        output_stage=:encoding_result,
        budget=(max_simplices=100, max_edges=32, memory_budget_bytes=2_000_000),
    )
    pipeline = OPT.PipelineOptions(;
        orientation=(1,),
        axes_policy=:coarsen,
        max_axis_len=3,
        axis_kind=:rn,
    )

    plan_a = DI.plan_ingestion(data, spec;
                               construction=construction,
                               pipeline=pipeline,
                               cache=nothing)
    plan_b = DI.plan_ingestion(data, spec;
                               construction=construction,
                               pipeline=pipeline,
                               cache=nothing)
    @test plan_a.construction == plan_b.construction
    @test plan_a.pipeline == plan_b.pipeline
    @test plan_a.stage == plan_b.stage
    @test plan_a.route_hint == plan_b.route_hint
    @test isnothing(plan_a.preflight)
    @test plan_a.preflight == plan_b.preflight
    @test plan_a.spec == plan_b.spec

    plan_with_preflight = DI.plan_ingestion(data, spec;
                                            construction=construction,
                                            pipeline=pipeline,
                                            cache=nothing,
                                            preflight=true)
    @test !isnothing(plan_with_preflight.preflight)
    @test DI.estimated_cells(plan_with_preflight.preflight) == big(6)

    G_direct = TamerOp.encode(data, filt;
                                   degree=0,
                                   construction=construction,
                                   pipeline=pipeline,
                                   stage=:graded_complex)
    G_plan = DI.run_ingestion(plan_a; stage=:graded_complex)
    @test G_plan.cells_by_dim == G_direct.cells_by_dim
    @test G_plan.grades == G_direct.grades

    enc_direct = TamerOp.encode(data, filt;
                                     degree=0,
                                     construction=construction,
                                     pipeline=pipeline)
    enc_plan = TamerOp.encode(plan_a; degree=0)
    @test isnothing(enc_direct.H)
    @test isnothing(enc_plan.H)
    @test _enc_dims(enc_plan) == _enc_dims(enc_direct)
    @test EC.axes_from_encoding(enc_plan.pi) == EC.axes_from_encoding(enc_direct.pi)

    H_direct = TamerOp.encode(data, filt;
                                   degree=0,
                                   construction=construction,
                                   pipeline=pipeline,
                                   stage=:fringe)
    @test H_direct isa FF.FringeModule

    plan_auto = DI.plan_ingestion(data, filt;
                                  construction=construction,
                                  pipeline=pipeline,
                                  cache=:auto)
    enc_auto = TamerOp.encode(plan_auto; degree=0)
    @test _enc_dims(enc_auto) == _enc_dims(enc_direct)

    old_plan_norm = DI._INGESTION_PLAN_NORM_CACHE[]
    try
        sc = CM.SessionCache()
        DI._INGESTION_PLAN_NORM_CACHE[] = true
        plan_cached_a = DI.plan_ingestion(data, spec;
                                          construction=construction,
                                          pipeline=pipeline,
                                          cache=sc)
        plan_cached_b = DI.plan_ingestion(data, spec;
                                          construction=construction,
                                          pipeline=pipeline,
                                          cache=sc)
        key = DI._ingestion_plan_norm_key(plan_cached_a.spec, plan_cached_a.stage, plan_cached_a.field)
        ec = CM._workflow_encoding_cache(sc)
        cached_norm = DI._get_geometry_cached(ec, key)
        @test !isnothing(cached_norm)
        @test plan_cached_a.filtration == plan_cached_b.filtration

        DI._INGESTION_PLAN_NORM_CACHE[] = false
        plan_uncached_a = DI.plan_ingestion(data, spec;
                                            construction=construction,
                                            pipeline=pipeline,
                                            cache=sc)
        plan_uncached_b = DI.plan_ingestion(data, spec;
                                            construction=construction,
                                            pipeline=pipeline,
                                            cache=sc)
        @test plan_uncached_a.filtration == plan_uncached_b.filtration
    finally
        DI._INGESTION_PLAN_NORM_CACHE[] = old_plan_norm
    end

    @test_throws ErrorException TamerOp.encode(data, filt; stage=:not_a_stage)

    tiny_budget = OPT.ConstructionOptions(;
        sparsify=:none,
        collapse=:none,
        output_stage=:encoding_result,
        budget=(max_simplices=1, max_edges=1, memory_budget_bytes=64),
    )
    @test_throws ArgumentError DI.plan_ingestion(data, spec;
                                                 construction=tiny_budget,
                                                 strict_preflight=true)
    @test_throws TypeError DI.plan_ingestion(data, spec; preflight=:on)
end

@testset "Data pipeline: ingestion UX surface" begin
    data = TamerOp.PointCloud([[0.0], [1.0], [2.0]])
    field = CM.QQField()
    construction = OPT.ConstructionOptions(; sparsify=:knn, output_stage=:graded_complex)
    filt = DI.RipsFiltration(max_dim=1, knn=2, construction=construction)
    spec = TamerOp.FiltrationSpec(kind=:rips, max_dim=1, construction=construction)

    filt_desc = describe(filt)
    @test filt_desc.kind == :rips
    @test filt_desc.arity == 1
    @test filt_desc.construction_mode.output_stage == :graded_complex
    @test DI.filtration_summary(filt) == filt_desc
    @test DI.filtration_kind(filt) == :rips
    @test DI.filtration_parameters(filt).max_dim == 1
    @test DI.construction_mode(filt).output_stage == :graded_complex
    @test occursin("RipsFiltration(", sprint(show, filt))
    @test occursin("construction_mode", sprint(show, MIME"text/plain"(), filt))

    spec_desc = describe(spec)
    @test spec_desc.kind == :filtration_spec
    @test spec_desc.filtration_kind == :rips
    @test DI.filtration_kind(spec) == :rips
    @test DI.filtration_arity(spec) == 1
    @test DI.filtration_parameters(spec).max_dim == 1
    @test DI.construction_mode(spec).output_stage == :graded_complex
    @test DI.filtration_spec_summary(spec) == spec_desc
    @test DI.filtration_family_summary(:rips).filtration_kind == :rips
    @test DI.filtration_family_summary(spec).provided_parameters == (:max_dim, :construction)
    @test DI.registered_filtration_families() isa Vector{Symbol}

    est = DI.estimate_ingestion(data, filt)
    @test est isa DI.IngestionEstimate
    @test describe(est).kind == :ingestion_estimate
    @test DI.ingestion_estimate_summary(est).estimated_cells == big(6)
    @test DI.estimated_cells(est) == big(6)
    @test DI.cell_counts_by_dim(est) == BigInt[3, 3]
    @test occursin("IngestionEstimate(", sprint(show, est))
    @test occursin("estimated_cells", sprint(show, MIME"text/plain"(), est))

    build = DI.build_graded_complex(data, filt)
    @test build isa DI.GradedComplexBuildResult
    @test describe(build).kind == :graded_complex_build_result
    @test DI.graded_complex_build_summary(build).kind == :graded_complex_build_result
    @test DI.graded_complex(build) isa DT.GradedComplex
    @test length(DI.grade_axes(build)) == 1
    @test DI.grade_orientation(build) == (1,)
    @test occursin("GradedComplexBuildResult(", sprint(show, build))

    plan = DI.plan_ingestion(data, filt; field=field, cache=nothing, preflight=true)
    @test plan isa DI.IngestionPlan
    @test describe(plan).kind == :ingestion_plan
    @test DI.ingestion_plan_summary(plan).planned_stage == :graded_complex
    @test DI.source_data(plan) === data
    @test DI.plan_filtration(plan) == filt
    @test DI.plan_spec(plan) isa OPT.FiltrationSpec
    @test DI.plan_construction(plan) == construction
    @test DI.planned_stage(plan) == :graded_complex
    @test DI.plan_field(plan) === field
    @test DI.has_preflight(plan)
    @test DI.preflight_estimate(plan) isa DI.IngestionEstimate
    @test DI.route_hint(plan) == :simplex_tree_first
    @test DI.multicritical_mode(plan) == :union
    @test DI.onecritical_selector(plan) == :lexmin
    @test DI.enforce_boundary(plan)
    @test occursin("IngestionPlan(", sprint(show, plan))
    @test occursin("planned_stage", sprint(show, MIME"text/plain"(), plan))

    report_f = DI.check_filtration(filt; throw=false)
    report_spec = DI.check_filtration_spec(spec; throw=false)
    report_pair = DI.check_data_filtration(data, filt; throw=false)
    report_est = DI.check_ingestion_estimate(est; throw=false)
    report_plan = DI.check_ingestion_plan(plan; throw=false)
    report_build = DI.check_graded_complex_build_result(build; throw=false)
    report_stage = DI.check_ingestion_stage(:graded_complex; throw=false)
    report_construction = DI.check_construction_options(data, construction; throw=false)
    report_preflight = DI.check_preflight_mode(true; throw=false)
    @test report_f.valid
    @test report_spec.valid
    @test report_pair.valid
    @test report_est.valid
    @test report_plan.valid
    @test report_build.valid
    @test report_stage.valid
    @test report_construction.valid
    @test report_preflight.valid
    @test occursin("IngestionValidationSummary(", sprint(show, DI.ingestion_validation_summary(report_plan)))

    bad_filt = BadUXFiltration()
    @test !DI.check_filtration(bad_filt; throw=false).valid
    @test_throws ArgumentError DI.check_filtration(bad_filt; throw=true)

    bad_spec = TamerOp.FiltrationSpec(kind=:__no_such_filtration__)
    @test !DI.check_filtration_spec(bad_spec; throw=false).valid
    @test_throws ArgumentError DI.check_filtration_spec(bad_spec; throw=true)

    bad_pair = DI.check_data_filtration(data, DI.WingVeinBifiltration(); throw=false)
    @test !bad_pair.valid
    @test_throws ArgumentError DI.check_data_filtration(data, DI.WingVeinBifiltration(); throw=true)

    bad_build = DI.GradedComplexBuildResult(DI.graded_complex(build), DI.grade_axes(build), (1, 1))
    @test !DI.check_graded_complex_build_result(bad_build; throw=false).valid
    @test_throws ArgumentError DI.check_graded_complex_build_result(bad_build; throw=true)

    @test !DI.check_ingestion_stage(:raw; throw=false).valid
    @test_throws ArgumentError DI.check_ingestion_stage(:raw; throw=true)

    gdata = TamerOp.GraphData(2, [(1, 2)])
    @test !DI.check_construction_options(gdata, construction; throw=false).valid
    @test_throws ArgumentError DI.check_construction_options(gdata, construction; throw=true)

    @test !DI.check_preflight_mode(:yes; throw=false).valid
    @test_throws ArgumentError DI.check_preflight_mode(:yes; throw=true)

    bad_plan = DI.IngestionPlan(
        data,
        filt,
        DI.plan_spec(plan),
        construction,
        OPT.PipelineOptions(),
        :not_a_stage,
        field,
        nothing,
        nothing,
        :simplex_tree_first,
        :union,
        :lexmin,
        true,
    )
    @test !DI.check_ingestion_plan(bad_plan; throw=false).valid
    @test_throws ArgumentError DI.check_ingestion_plan(bad_plan; throw=true)

    @test TOA.IngestionPlan === DI.IngestionPlan
    @test TOA.IngestionEstimate === DI.IngestionEstimate
    @test TOA.GradedComplexBuildResult === DI.GradedComplexBuildResult
    @test TOA.registered_filtration_families === DI.registered_filtration_families
    @test TOA.filtration_spec_summary === DI.filtration_spec_summary
    @test TOA.filtration_family_summary === DI.filtration_family_summary
    @test TOA.construction_mode === DI.construction_mode
    @test TOA.estimate_ingestion === DI.estimate_ingestion
    @test TOA.source_data === DI.source_data
    @test TOA.estimated_cells === DI.estimated_cells
    @test TOA.graded_complex === DI.graded_complex
    @test TOA.check_filtration === DI.check_filtration
    @test TOA.check_filtration_spec === DI.check_filtration_spec
    @test TOA.check_data_filtration === DI.check_data_filtration
    @test TOA.check_graded_complex_build_result === DI.check_graded_complex_build_result
    @test TOA.check_ingestion_stage === DI.check_ingestion_stage
    @test TOA.check_construction_options === DI.check_construction_options
    @test TOA.check_preflight_mode === DI.check_preflight_mode
    @test TOA.ingestion_plan_summary === DI.ingestion_plan_summary
    @test TOA.graded_complex_build_summary === DI.graded_complex_build_summary
    @test TOA.PersistenceDiagram === OP.PersistenceDiagram
    @test TOA.persistence_diagram === OP.persistence_diagram
    @test TOA.cubical_persistence === OP.cubical_persistence
    @test TOA.persistence_intervals === OP.persistence_intervals
    @test TOA.finite_intervals === OP.finite_intervals
    @test TOA.essential_births === OP.essential_births
    @test TOA.persistence_diagram_summary === OP.persistence_diagram_summary
    @test TOA.check_torus_persistence === OP.check_torus_persistence

    typed_filtration_ctors = (
        :GradedFiltration,
        :RipsFiltration, :RipsDensityFiltration, :RipsCodensityFiltration, :RipsLowerStarFiltration, :FunctionRipsFiltration, :LandmarkRipsFiltration,
        :GraphLowerStarFiltration, :CliqueLowerStarFiltration, :EdgeWeightedFiltration,
        :GraphCentralityFiltration, :GraphGeodesicFiltration, :GraphFunctionGeodesicBifiltration,
        :GraphWeightThresholdFiltration,
        :ImageLowerStarFiltration, :ImageDistanceBifiltration, :WingVeinBifiltration,
        :DelaunayLowerStarFiltration, :AlphaFiltration, :FunctionDelaunayFiltration,
        :CoreDelaunayFiltration, :CoreFiltration, :GraphCoreFiltration, :DegreeRipsFiltration, :CubicalFiltration,
        :RhomboidFiltration,
    )
    for sym in typed_filtration_ctors
        @test sym in TamerOp.SIMPLE_API
        @test isdefined(TamerOp, sym)
        @test getfield(TamerOp, sym) === getfield(DI, sym)
    end
    @test TamerOp.AlphaFiltration(; max_dim=2) isa DI.AlphaFiltration
    @test TamerOp.GraphFunctionGeodesicBifiltration(; sources=[1]) isa DI.GraphFunctionGeodesicBifiltration
end

@testset "Data pipeline: function-Rips (point cloud)" begin
    data = TamerOp.PointCloud([[0.0], [1.0]])
    spec_vals = TamerOp.FiltrationSpec(kind=:function_rips,
                                            max_dim=1,
                                            vertex_values=[0.0, 2.0],
                                            simplex_agg=:max)
    enc_vals = TamerOp.encode(data, spec_vals; degree=0)
    @test FF.nvertices(enc_vals.P) == 4
    @test MD.dim_at(_enc_module(enc_vals), EC.locate(enc_vals.pi, [0.0, 0.0])) == 1
    @test MD.dim_at(_enc_module(enc_vals), EC.locate(enc_vals.pi, [0.0, 2.0])) == 2
    @test MD.dim_at(_enc_module(enc_vals), EC.locate(enc_vals.pi, [1.0, 2.0])) == 1

    spec_fun = TamerOp.FiltrationSpec(kind=:function_rips,
                                           max_dim=1,
                                           vertex_function=(p, i) -> (i == 1 ? 0.0 : 2.0),
                                           simplex_agg=:max)
    enc_fun = TamerOp.encode(data, spec_fun; degree=0)
    @test _enc_dims(enc_fun) == _enc_dims(enc_vals)
    @test EC.axes_from_encoding(enc_fun.pi) == EC.axes_from_encoding(enc_vals.pi)
end

@testset "Data pipeline: typed rips_lowerstar (point cloud)" begin
    data = TamerOp.PointCloud([[0.0, 2.0], [1.0, 0.5], [2.0, 1.5]])
    filt = DI.RipsLowerStarFiltration(; max_dim=1, radius=3.0, coord=1)
    spec = TamerOp.FiltrationSpec(kind=:rips_lowerstar, max_dim=1, radius=3.0, coord=1)
    enc_f = TamerOp.encode(data, filt; degree=0)
    enc_s = TamerOp.encode(data, spec; degree=0)
    @test _enc_dims(enc_f) == _enc_dims(enc_s)
    @test EC.axes_from_encoding(enc_f.pi) == EC.axes_from_encoding(enc_s.pi)
end

@testset "Data pipeline: graph vertex-values UX parity" begin
    g = TamerOp.GraphData(3, [(1, 2), (2, 3)])
    spec_old = TamerOp.FiltrationSpec(kind=:graph_lower_star,
                                           vertex_grades=[[0.0], [1.0], [2.0]],
                                           simplex_agg=:max)
    spec_new = TamerOp.FiltrationSpec(kind=:graph_lower_star,
                                           vertex_values=[0.0, 1.0, 2.0],
                                           simplex_agg=:max)
    spec_fun = TamerOp.FiltrationSpec(kind=:graph_lower_star,
                                           vertex_function=(arg, i) -> i - 1,
                                           simplex_agg=:max)
    enc_old = TamerOp.encode(g, spec_old; degree=0)
    enc_new = TamerOp.encode(g, spec_new; degree=0)
    enc_fun = TamerOp.encode(g, spec_fun; degree=0)
    @test _enc_dims(enc_new) == _enc_dims(enc_old)
    @test _enc_dims(enc_fun) == _enc_dims(enc_old)
    @test EC.axes_from_encoding(enc_new.pi) == EC.axes_from_encoding(enc_old.pi)
    @test EC.axes_from_encoding(enc_fun.pi) == EC.axes_from_encoding(enc_old.pi)
end

@testset "Data pipeline: graph centrality/geodesic/threshold filtrations" begin
    g = TamerOp.GraphData(3, [(1, 2), (2, 3)]; weights=[1.0, 2.0])

    f_cent = DI.GraphCentralityFiltration(centrality=:degree, lift=:lower_star)
    enc_cent = TamerOp.encode(g, f_cent; degree=0)
    ax_cent = EC.axes_from_encoding(enc_cent.pi)
    @test length(ax_cent) == 1
    @test ax_cent[1] == [1.0, 2.0]

    spec_cent = TamerOp.FiltrationSpec(kind=:graph_centrality, centrality=:closeness, metric=:hop, lift=:lower_star)
    typed_cent = DI.to_filtration(spec_cent)
    @test typed_cent isa DI.GraphCentralityFiltration
    enc_close = TamerOp.encode(g, spec_cent; degree=0)
    close_vals = EC.axes_from_encoding(enc_close.pi)[1]
    @test any(isapprox(v, 2 / 3; atol=1e-6) for v in close_vals)
    @test any(isapprox(v, 1.0; atol=1e-8) for v in close_vals)

    f_geo = DI.GraphGeodesicFiltration(sources=[1], metric=:hop, lift=:lower_star)
    enc_geo = TamerOp.encode(g, f_geo; degree=0)
    ax_geo = EC.axes_from_encoding(enc_geo.pi)
    @test length(ax_geo) == 1
    @test ax_geo[1] == [0.0, 1.0, 2.0]

    spec_geo = TamerOp.FiltrationSpec(kind=:graph_geodesic, sources=[1], metric=:weighted, lift=:lower_star)
    typed_geo = DI.to_filtration(spec_geo)
    @test typed_geo isa DI.GraphGeodesicFiltration
    enc_geo_w = TamerOp.encode(g, spec_geo; degree=0)
    @test EC.axes_from_encoding(enc_geo_w.pi)[1] == [0.0, 1.0, 3.0]

    spec_bi = TamerOp.FiltrationSpec(
        kind=:graph_function_geodesic_bifiltration,
        sources=[1],
        metric=:hop,
        vertex_values=[10.0, 20.0, 30.0],
        lift=:lower_star,
        simplex_agg=:max,
    )
    typed_bi = DI.to_filtration(spec_bi)
    @test typed_bi isa DI.GraphFunctionGeodesicBifiltration
    enc_bi = TamerOp.encode(g, spec_bi; degree=0)
    ax_bi = EC.axes_from_encoding(enc_bi.pi)
    @test length(ax_bi) == 2
    @test ax_bi[1] == [0.0, 1.0, 2.0]
    @test ax_bi[2] == [10.0, 20.0, 30.0]

    f_thr = DI.GraphWeightThresholdFiltration(edge_weights=[0.3, 0.8], lift=:graph)
    enc_thr = TamerOp.encode(g, f_thr; degree=0)
    @test EC.axes_from_encoding(enc_thr.pi)[1] == [0.0, 0.3, 0.8]

    gtri = TamerOp.GraphData(3, [(1, 2), (2, 3), (1, 3)]; weights=[0.3, 0.8, 0.5])
    spec_thr = TamerOp.FiltrationSpec(kind=:graph_weight_threshold, lift=:clique, max_dim=2)
    typed_thr = DI.to_filtration(spec_thr)
    @test typed_thr isa DI.GraphWeightThresholdFiltration
    enc_thr_clique = TamerOp.encode(gtri, spec_thr; degree=0)
    @test EC.axes_from_encoding(enc_thr_clique.pi)[1] == [0.0, 0.3, 0.5, 0.8]
    gtri_unsorted = TamerOp.GraphData(3, [(2, 1), (3, 2), (3, 1)]; weights=[0.3, 0.8, 0.5])
    enc_thr_clique_unsorted = TamerOp.encode(gtri_unsorted, spec_thr; degree=0)
    @test EC.axes_from_encoding(enc_thr_clique_unsorted.pi)[1] == [0.0, 0.3, 0.5, 0.8]

    est = DI.estimate_ingestion(gtri, TamerOp.FiltrationSpec(kind=:graph_centrality, lift=:clique, max_dim=2))
    @test DI.cell_counts_by_dim(est) == BigInt[3, 3, 1]
end

@testset "Data pipeline: graph new-family contracts" begin
    g = TamerOp.GraphData(3, [(1, 2), (2, 3)])
    @test_throws Exception TamerOp.encode(
        g,
        TamerOp.FiltrationSpec(kind=:graph_geodesic, sources=[1], metric=:weighted, lift=:lower_star);
        degree=0,
    )
    @test_throws Exception TamerOp.encode(
        g,
        TamerOp.FiltrationSpec(kind=:graph_weight_threshold, lift=:clique, max_dim=2);
        degree=0,
    )

    nbig = 80
    epath = [(i, j) for i in 1:(nbig - 1) for j in (i + 1):nbig]
    gpath = TamerOp.GraphData(nbig, epath; weights=fill(1.0, length(epath)))

    spec_clique_precheck = TamerOp.FiltrationSpec(
        kind=:graph_centrality,
        centrality=:degree,
        lift=:clique,
        max_dim=3,
        construction=TamerOp.ConstructionOptions(; budget=(15_000, nothing, nothing)),
    )
    err = try
        TamerOp.encode(gpath, spec_clique_precheck; degree=0)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("clique enumeration", sprint(showerror, err))

    spec_thr_clique_precheck = TamerOp.FiltrationSpec(
        kind=:graph_weight_threshold,
        lift=:clique,
        max_dim=3,
        edge_weights=fill(1.0, length(epath)),
        construction=TamerOp.ConstructionOptions(; budget=(15_000, nothing, nothing)),
    )
    err = try
        TamerOp.encode(gpath, spec_thr_clique_precheck; degree=0)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("clique enumeration", sprint(showerror, err))

    @test_throws ArgumentError TamerOp.encode(gpath, DI.RhomboidFiltration(); degree=0)

end

@testset "Data pipeline: multi-critical graded complex" begin
    cells = [Int[1, 2], Int[1]]
    B1 = sparse([1, 2], [1, 1], [1, -1], 2, 1)
    grades = [
        [Float64[0.0, 0.0]],
        [Float64[0.0, 0.0]],
        [Float64[1.0, 0.0], Float64[0.0, 1.0]],
    ]
    G = TamerOp.MultiCriticalGradedComplex(cells, [B1], grades)
    spec = TamerOp.FiltrationSpec(kind=:graded)
    enc = TamerOp.encode(G, spec; degree=0)
    @test MD.dim_at(_enc_module(enc), EC.locate(enc.pi, [0.0, 0.0])) == 2
    @test MD.dim_at(_enc_module(enc), EC.locate(enc.pi, [1.0, 0.0])) == 1
    @test MD.dim_at(_enc_module(enc), EC.locate(enc.pi, [0.0, 1.0])) == 1
end

@testset "Data pipeline: one_criticalify" begin
    cells = [Int[1, 2], Int[1]]
    B1 = sparse([1, 2], [1, 1], [1, -1], 2, 1)
    grades = [
        [Float64[0.0, 0.0]],
        [Float64[2.0, 2.0]],
        [Float64[1.0, 0.0], Float64[0.0, 1.0]],
    ]
    Gm = TamerOp.MultiCriticalGradedComplex(cells, [B1], grades)

    G1 = DI.one_criticalify(Gm)
    @test G1 isa TamerOp.GradedComplex
    @test length(G1.grades) == 3
    @test G1.grades[3] == (2.0, 2.0)  # lifted to dominate boundary-face grades

    G1_raw = DI.one_criticalify(Gm; enforce_boundary=false)
    @test G1_raw.grades[3] == (0.0, 1.0)  # default selector=:lexmin

    G1_max = DI.one_criticalify(Gm; selector=:lexmax, enforce_boundary=false)
    @test G1_max.grades[3] == (1.0, 0.0)

    Gs = TamerOp.GradedComplex(cells, [B1], [Float64[0.0, 0.0], Float64[2.0, 2.0], Float64[2.0, 2.0]])
    @test DI.one_criticalify(Gs) === Gs
end

@testset "Data pipeline: multi-critical algebra policies" begin
    cells = [Int[1, 2], Int[1]]
    B1 = sparse([1, 2], [1, 1], [1, -1], 2, 1)
    grades = [
        [Float64[0.0, 0.0]],
        [Float64[0.0, 0.0]],
        [Float64[1.0, 0.0], Float64[0.0, 1.0], Float64[1.0, 1.0]],
    ]
    G = TamerOp.MultiCriticalGradedComplex(cells, [B1], grades)
    @test DI.criticality(G) == 3
    @test DI.criticality(DI.one_criticalify(G)) == 1

    Gn = DI.normalize_multicritical(G; keep=:minimal)
    @test DI.criticality(Gn) == 2
    @test length(Gn.grades[3]) == 2

    spec_union = TamerOp.FiltrationSpec(kind=:graded, multicritical=:union)
    spec_inter = TamerOp.FiltrationSpec(kind=:graded, multicritical=:intersection)
    spec_one = TamerOp.FiltrationSpec(kind=:graded, multicritical=:one_critical,
                                           onecritical_selector=:lexmin,
                                           onecritical_enforce_boundary=false)

    enc_union = TamerOp.encode(G, spec_union; degree=0)
    enc_inter = TamerOp.encode(G, spec_inter; degree=0)
    enc_one = TamerOp.encode(G, spec_one; degree=0)

    q10 = EC.locate(enc_union.pi, [1.0, 0.0])
    q01 = EC.locate(enc_union.pi, [0.0, 1.0])
    q11 = EC.locate(enc_union.pi, [1.0, 1.0])
    @test MD.dim_at(_enc_module(enc_union), q10) == 1
    @test MD.dim_at(_enc_module(enc_union), q01) == 1
    @test MD.dim_at(_enc_module(enc_union), q11) == 1

    q10i = EC.locate(enc_inter.pi, [1.0, 0.0])
    q01i = EC.locate(enc_inter.pi, [0.0, 1.0])
    q11i = EC.locate(enc_inter.pi, [1.0, 1.0])
    @test MD.dim_at(_enc_module(enc_inter), q10i) == 2
    @test MD.dim_at(_enc_module(enc_inter), q01i) == 2
    @test MD.dim_at(_enc_module(enc_inter), q11i) == 1

    q10o = EC.locate(enc_one.pi, [1.0, 0.0])
    q01o = EC.locate(enc_one.pi, [0.0, 1.0])
    q11o = EC.locate(enc_one.pi, [1.0, 1.0])
    @test MD.dim_at(_enc_module(enc_one), q10o) == 2
    @test MD.dim_at(_enc_module(enc_one), q01o) == 1
    @test MD.dim_at(_enc_module(enc_one), q11o) == 1
end

@testset "Data pipeline: packed complex metadata consumers" begin
    cells = [Int[10, 11, 12], Int[20, 21, 22], Int[30]]
    B1 = sparse(
        [1, 2, 2, 3, 1, 3],
        [1, 1, 2, 2, 3, 3],
        [1, -1, 1, -1, 1, -1],
        3, 3,
    )
    B2 = sparse([1, 2, 3], [1, 1, 1], [1, -1, 1], 3, 1)

    Gs = TamerOp.GradedComplex(
        cells,
        [B1, B2],
        [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (2.0, 2.0)],
    )
    Gm = TamerOp.MultiCriticalGradedComplex(
        cells,
        [B1, B2],
        [
            [(0.0, 0.0)],
            [(1.0, 0.0)],
            [(0.0, 1.0)],
            [(1.0, 0.0)],
            [(1.0, 1.0)],
            [(0.0, 1.0)],
            [(1.0, 1.0), (2.0, 2.0)],
        ],
    )
    spec = TamerOp.FiltrationSpec(kind=:graded)

    @test DI._estimate_cell_counts(Gs, spec; exact_pairwise_limit=16, warnings=String[], strict=false) ==
          BigInt[3, 3, 1]
    @test DI._estimate_cell_counts(Gm, spec; exact_pairwise_limit=16, warnings=String[], strict=false) ==
          BigInt[3, 3, 1]

    axes_s = DI._axes_from_complex_grades(Gs, (1, 1))
    axes_m = DI._axes_from_complex_grades(Gm, (1, 1))
    @test axes_s == ([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
    @test axes_m == axes_s

    @test DI.criticality(Gm) == 2
    @test DI.normalize_multicritical(Gm; keep=:unique) === Gm

    Gn_min = DI.normalize_multicritical(Gm; keep=:minimal)
    Gn_max = DI.normalize_multicritical(Gm; keep=:maximal)
    @test collect(Gn_min.grades[7]) == [(1.0, 1.0)]
    @test collect(Gn_max.grades[7]) == [(2.0, 2.0)]

    grades_s = DI._grades_by_dim(Gs)
    grades_m = DI._grades_by_dim(Gm)
    @test length.(grades_s) == [3, 3, 1]
    @test length.(grades_m) == [3, 3, 1]
    @test grades_s[2][2] == (1.0, 1.0)
    @test grades_m[3][1] == [(1.0, 1.0), (2.0, 2.0)]

    expected_simplices = [
        [[1], [2], [3]],
        [[1, 2], [2, 3], [1, 3]],
        [[1, 2, 3]],
    ]
    @test DI._simplices_from_complex(Gs) == expected_simplices
    @test DI._simplices_from_complex(Gm) == expected_simplices

    Gm_out, axes_out, orient_out = DI._graded_complex_from_data(Gm, spec; cache=nothing)
    @test Gm_out === Gm
    @test axes_out == axes_m
    @test orient_out == (1, 1)
end

@testset "Interop adapters: RIVET bifiltration + FIRep" begin
    mktemp() do path, io
        close(io)
        open(path, "w") do f
            write(f, "--datatype bifiltration\n")
            write(f, "0 ; 0 0\n")
            write(f, "1 ; 0 0\n")
            write(f, "0 1 ; 1 0 0 1\n")
        end
        G = SER.load_rivet_bifiltration(path)
        @test G isa TamerOp.MultiCriticalGradedComplex
        @test length(G.grades) == 3
        @test length(G.grades[3]) == 2
        enc = TamerOp.encode(G, TamerOp.FiltrationSpec(kind=:graded); degree=0)
        @test FF.nvertices(enc.P) == 4
    end

    mktemp() do path, io
        close(io)
        open(path, "w") do f
            write(f, "bifiltration\n")
            write(f, "s\n")
            write(f, "2\n")
            write(f, "3\n")
            write(f, "0 ; 0 0\n")
            write(f, "1 ; 0 0\n")
            write(f, "0 1 ; 1 0 0 1\n")
        end
        @test_throws ErrorException SER.load_rivet_bifiltration(path)
    end

    mktemp() do path, io
        close(io)
        open(path, "w") do f
            write(f, "--datatype firep\n")
            write(f, "1 2 2\n")
            write(f, "1.0 1.0 ; 0 1\n")
            write(f, "0.0 0.0 ; 0\n")
            write(f, "0.0 1.0 ; 0 1\n")
        end
        G = SER.load_rivet_firep(path)
        @test G isa TamerOp.GradedComplex
        @test length(G.cells_by_dim) == 3
        @test size(G.boundaries[1]) == (2, 2)
        @test size(G.boundaries[2]) == (2, 1)
    end
end

@testset "Data pipeline: image lower-star" begin
    img = [0.0 1.0; 2.0 3.0]
    data = TamerOp.ImageNd(img)
    spec = TamerOp.FiltrationSpec(kind=:lower_star, axes=([0.0, 1.0, 2.0, 3.0],))
    enc = TamerOp.encode(data, spec; degree=0)
    @test _enc_dims(enc) == fill(1, 4)
end

@testset "Data pipeline: 3D cubical lower-star" begin
    img = reshape(Float64.(1:8), (2, 2, 2))
    data = TamerOp.ImageNd(img)
    spec = TamerOp.FiltrationSpec(kind=:lower_star, axes=([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],))
    enc = TamerOp.encode(data, spec; degree=0)
    @test _enc_dims(enc) == fill(1, 8)
end

@testset "Data pipeline: embedded planar graph toy" begin
    verts = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
    edges = [(1, 2), (2, 3)]
    data = TamerOp.EmbeddedPlanarGraph2D(verts, edges)
    vgrades = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
    spec = TamerOp.FiltrationSpec(kind=:graph_lower_star, vertex_grades=vgrades)
    enc = TamerOp.encode(data, spec; degree=0)
    @test FF.nvertices(enc.P) > 0
    @test EC.locate(enc.pi, [0.0, 0.0]) > 0
    H = enc.H === nothing ? TamerOp.Workflow.fringe_presentation(DI.materialize_module(enc.M)) : enc.H
    @test H isa FF.FringeModule
end

@testset "Data pipeline: wing distance bifiltration" begin
    verts = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
    edges = [(1, 2), (2, 3)]
    data = TamerOp.EmbeddedPlanarGraph2D(verts, edges)
    spec = TamerOp.FiltrationSpec(
        kind=:wing_vein_bifiltration,
        grid=(8, 8),
        bbox=(0.0, 1.0, 0.0, 1.0),
        orientation=(-1, 1),
    )
    enc = TamerOp.encode(data, spec; degree=0)
    H = enc.H === nothing ? TamerOp.Workflow.fringe_presentation(DI.materialize_module(enc.M)) : enc.H
    @test H isa FF.FringeModule
    @test EC.locate(enc.pi, [0.0, 0.0]) > 0
    @test EC.locate(enc.pi, [-1.0, 0.0]) > 0
    @test EC.locate(enc.pi, [-0.5, 0.5]) > 0
    ri = Inv.rank_invariant(_enc_module(enc), OPT.InvariantOptions(); store_zeros=true)
    @test ri[(1, 1)] >= 0

    opts = OPT.InvariantOptions(axes_policy=:encoding, strict=false, box=:auto)
    chain, _ = Inv.slice_chain(enc.pi, [-1.0, 0.0], [1.0, 1.0], opts; nsteps=25, check_chain=true)
    @test length(chain) > 0
    ri_chain = Inv.rank_invariant(_enc_module(enc), OPT.InvariantOptions(); store_zeros=true)
    for a in 1:length(chain)
        for b in a:length(chain)
            qa = chain[a]
            qb = chain[b]
            @test get(ri_chain, (qa, qb), 0) >= 0
        end
    end
end

@testset "Data pipeline: invariants compatibility" begin
    # Graded complex
    cells = [Int[1]]
    boundaries = SparseMatrixCSC{Int,Int}[]
    grades = [Float64[0.0]]
    G = TamerOp.GradedComplex(cells, boundaries, grades)
    spec = TamerOp.FiltrationSpec(kind=:graded, axes=([0.0, 1.0],))
    enc = TamerOp.encode(G, spec; degree=0)
    @test SM.euler_surface(_enc_module(enc), enc.pi; opts=OPT.InvariantOptions(axes_policy=:encoding)) isa AbstractArray
    @test Inv.rank_invariant(_enc_module(enc), OPT.InvariantOptions()) isa Inv.RankInvariantResult

    # Point cloud
    data = TamerOp.PointCloud([[0.0], [1.0]])
    spec = TamerOp.FiltrationSpec(kind=:rips, max_dim=1, axes=([0.0, 1.0],))
    enc = TamerOp.encode(data, spec; degree=0)
    @test SM.euler_surface(_enc_module(enc), enc.pi; opts=OPT.InvariantOptions(axes_policy=:encoding)) isa AbstractArray
    @test Inv.rank_invariant(_enc_module(enc), OPT.InvariantOptions()) isa Inv.RankInvariantResult

    # Image (2D)
    img = [0.0 1.0; 2.0 3.0]
    data = TamerOp.ImageNd(img)
    spec = TamerOp.FiltrationSpec(kind=:lower_star, axes=([0.0, 1.0, 2.0, 3.0],))
    enc = TamerOp.encode(data, spec; degree=0)
    @test SM.euler_surface(_enc_module(enc), enc.pi; opts=OPT.InvariantOptions(axes_policy=:encoding)) isa AbstractArray

    # Graph (2D) for slice_chain
    data = TamerOp.GraphData(3, [(1, 2), (2, 3)])
    vgrades = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
    spec = TamerOp.FiltrationSpec(kind=:graph_lower_star, vertex_grades=vgrades)
    enc = TamerOp.encode(data, spec; degree=0)
    opts = OPT.InvariantOptions(axes_policy=:encoding, strict=false, box=:auto)
    chain, tvals = Inv.slice_chain(enc.pi, [0.0, 0.0], [1.0, 1.0], opts; nsteps=5)
    @test length(chain) > 0
    @test length(chain) == length(tvals)

    # Embedded planar graph (2D) for slice_chain
    verts = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
    edges = [(1, 2), (2, 3)]
    data = TamerOp.EmbeddedPlanarGraph2D(verts, edges)
    vgrades = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
    spec = TamerOp.FiltrationSpec(kind=:graph_lower_star, vertex_grades=vgrades)
    enc = TamerOp.encode(data, spec; degree=0)
    chain, tvals = Inv.slice_chain(enc.pi, [0.0, 0.0], [1.0, 1.0], opts; nsteps=5)
    @test length(chain) > 0
    @test length(chain) == length(tvals)
end

@testset "Data pipeline: custom filtration registry and schema" begin
    DI.register_filtration_family!(
        kind=:test_trigrade,
        ctor = spec -> TestTriGradeFiltration(;
            shift=Float64(get(spec.params, :shift, 0.0)),
            scale=Float64(get(spec.params, :scale, 1.0)),
            construction=DI._construction_from_params(spec.params),
        ),
        builder = _test_trigrade_builder,
        arity = 3,
        schema = _TEST_TRIGRADE_SCHEMA,
    )

    data = TamerOp.PointCloud([[0.0], [1.0], [2.0]])
    tf = TestTriGradeFiltration(; shift=0.5, scale=2.0)

    @test :test_trigrade in DI.available_filtrations()
    @test :test_trigrade in DI.registered_filtration_families()
    @test DI.filtration_arity(tf, data) == 3
    @test DI.filtration_kind(typeof(tf)) == :test_trigrade
    @test DI.filtration_kind(tf) == :test_trigrade

    sig = DI.filtration_signature(:test_trigrade)
    @test sig.kind == :test_trigrade
    @test sig.registered == true
    @test sig.arity == 3
    @test haskey(sig.defaults, :shift)
    @test haskey(sig.defaults, :scale)
    @test DI.filtration_family_summary(:test_trigrade).registered
    @test DI.filtration_family_summary(:test_trigrade).arity == 3

    fp = DI.filtration_parameters(:test_trigrade)
    @test fp.defaults[:shift] == 0.0
    @test fp.defaults[:scale] == 1.0
    @test haskey(fp.types, :shift)
    @test haskey(fp.checks, :scale)

    # Roundtrip parity: typed filtration -> FiltrationSpec -> typed filtration.
    spec_from_typed = DI._filtration_spec(tf)
    @test spec_from_typed.kind == :test_trigrade
    @test spec_from_typed.params[:shift] == 0.5
    @test spec_from_typed.params[:scale] == 2.0
    @test describe(spec_from_typed).filtration_kind == :test_trigrade
    @test DI.filtration_spec_summary(spec_from_typed).arity == 3
    @test DI.filtration_family_summary(spec_from_typed).provided_parameters == (:shift, :scale, :construction)
    @test DI.check_filtration_spec(spec_from_typed; throw=false).valid
    tf2 = DI.to_filtration(spec_from_typed)
    @test tf2 isa TestTriGradeFiltration
    @test tf2.params.shift == 0.5
    @test tf2.params.scale == 2.0

    # Schema defaults apply when spec omits optional params.
    tf_default = DI.to_filtration(TamerOp.FiltrationSpec(kind=:test_trigrade))
    @test tf_default.params.shift == 0.0
    @test tf_default.params.scale == 1.0

    # Stage parity from typed filtration and FiltrationSpec.
    spec_direct = TamerOp.FiltrationSpec(kind=:test_trigrade, shift=0.5, scale=2.0)
    G_typed = TamerOp.encode(data, tf; stage=:graded_complex)
    G_spec = TamerOp.encode(data, spec_direct; stage=:graded_complex)
    @test G_typed.grades == G_spec.grades
    @test G_typed.cells_by_dim == G_spec.cells_by_dim

    enc_typed = TamerOp.encode(data, tf; stage=:encoding_result, degree=0)
    enc_spec = TamerOp.encode(data, spec_direct; stage=:encoding_result, degree=0)
    @test DI.module_dims(enc_typed.M) == DI.module_dims(enc_spec.M)

    # Schema contract failures.
    @test_throws ArgumentError DI.to_filtration(TamerOp.FiltrationSpec(kind=:test_trigrade, shift="bad"))
    @test_throws ArgumentError DI.to_filtration(TamerOp.FiltrationSpec(kind=:test_trigrade, scale=0.0))
end

@testset "A06: certified filtered edge collapses preserve inclusion homology" begin
    reduction = TamerOp.SimplicialReduction

    # Independent oracle: enumerate vertex subsets, rather than using the
    # production flag expansion or its simplex grading helpers.
    function a06_cells(edges, edge_grades, vertex_grades, max_dim, query)
        n = length(vertex_grades)
        cells = [Vector{Int}[] for _ in 0:max_dim]
        for mask in 1:((1 << n) - 1)
            vertices = [v for v in 1:n if !iszero(mask & (1 << (v - 1)))]
            length(vertices) <= max_dim + 1 || continue
            all(v -> all(vertex_grades[v][k] <= query[k] for k in eachindex(query)),
                vertices) || continue
            present = true
            for a in 1:length(vertices), b in (a + 1):length(vertices)
                u, v = vertices[a], vertices[b]
                idx = findfirst(e -> e == (u, v) || e == (v, u), edges)
                if idx === nothing ||
                   !all(edge_grades[idx][k] <= query[k] for k in eachindex(query))
                    present = false
                    break
                end
            end
            present && push!(cells[length(vertices)], vertices)
        end
        return cells
    end

    function a06_boundary(cells, degree, field)
        ncols = degree < length(cells) ? length(cells[degree + 1]) : 0
        nrows = degree == 0 ? 0 : length(cells[degree])
        matrix = zeros(CM.coeff_type(field), nrows, ncols)
        degree == 0 && return matrix
        for (j, simplex) in enumerate(degree < length(cells) ? cells[degree + 1] : Vector{Int}[])
            for k in eachindex(simplex)
                face = [simplex[i] for i in eachindex(simplex) if i != k]
                row = findfirst(==(face), cells[degree])
                @test row !== nothing
                row === nothing && continue
                matrix[row, j] = CM.coerce(field, isodd(k) ? 1 : -1)
            end
        end
        return matrix
    end

    function a06_inclusion(original, reduced, field)
        matrix = zeros(CM.coeff_type(field), length(original), length(reduced))
        for (j, simplex) in enumerate(reduced)
            i = findfirst(==(simplex), original)
            @test i !== nothing
            i === nothing && continue
            matrix[i, j] = CM.coerce(field, 1)
        end
        return matrix
    end

    function a06_zero(matrix, field)
        return field isa CM.RealField ? all(x -> abs(x) <= 1e-10, matrix) : all(iszero, matrix)
    end

    function a06_betti(cells, field)
        return [length(cells[q + 1]) - FL.rank(field, a06_boundary(cells, q, field)) -
                FL.rank(field, a06_boundary(cells, q + 1, field)) for q in 0:(length(cells) - 1)]
    end

    function a06_check_inclusion(edges, edge_grades, vertex_grades, max_dim,
                                 retained, query, field)
        original = a06_cells(edges, edge_grades, vertex_grades, max_dim, query)
        reduced = a06_cells(edges[retained], edge_grades[retained], vertex_grades,
                            max_dim, query)
        previous_inclusion = zeros(CM.coeff_type(field), 0, 0)
        for q in 0:max_dim
            outgoing_original = a06_boundary(original, q, field)
            incoming_original = a06_boundary(original, q + 1, field)
            outgoing_reduced = a06_boundary(reduced, q, field)
            incoming_reduced = a06_boundary(reduced, q + 1, field)
            inclusion = a06_inclusion(original[q + 1], reduced[q + 1], field)
            cycles = Matrix(FL.nullspace(field, outgoing_reduced))
            included_cycles = inclusion * cycles
            boundary_rank = FL.rank(field, incoming_original)
            betti_original = length(original[q + 1]) -
                             FL.rank(field, outgoing_original) - boundary_rank
            betti_reduced = length(reduced[q + 1]) -
                            FL.rank(field, outgoing_reduced) - FL.rank(field, incoming_reduced)
            @test betti_original == betti_reduced
            # Equality of Betti numbers alone would miss a bad induced map.
            @test FL.rank(field, hcat(incoming_original, included_cycles)) - boundary_rank == betti_original
            @test a06_zero(outgoing_original * included_cycles, field)
            @test a06_zero(outgoing_original * incoming_original, field)
            @test a06_zero(outgoing_original * inclusion - previous_inclusion * outgoing_reduced, field)
            previous_inclusion = inclusion
        end
        return nothing
    end

    @testset "Hand-computable filtration and truncation counterexamples" begin
        triangle = [(1, 2), (1, 3), (2, 3)]
        triangle_grades = fill((1.0,), 3)
        vertex_grades = fill((0.0,), 3)
        triangle_copy = copy(triangle)
        grades_copy = copy(triangle_grades)
        kept = reduction._collapse_dominated_edges(triangle, triangle_grades, 3, 2)
        @test length(kept) == 2
        @test issorted(kept)
        @test triangle == triangle_copy
        @test triangle_grades == grades_copy
        @test reduction._collapse_dominated_edges(triangle, triangle_grades, 3, 1) == [1, 2, 3]
        @test reduction._collapse_dominated_edges(triangle, triangle_grades, 3, 0) == [1, 2, 3]

        # A geometrically nearby isolated vertex supplies no actual witness edges.
        @test reduction._collapse_dominated_edges([(1, 2)], [(1.0,)], 3, 2) == [1]

        # The shortest detour uses vertex 3, absent until the second parameter 10.
        late_vertices = [(0.0, 0.0), (0.0, 0.0), (0.0, 10.0)]
        late_grades = [(2.0, 0.0), (1.0, 10.0), (1.0, 10.0)]
        late_kept = reduction._collapse_dominated_edges(triangle, late_grades, 3, 2)
        @test late_kept == [1, 2, 3]

        # Stored-grade inequalities must not turn into a tolerance-based witness.
        almost_one = nextfloat(1.0)
        close_grades = [(1.0,), (almost_one,), (almost_one,)]
        close_kept = reduction._collapse_dominated_edges(triangle, close_grades, 3, 2)
        @test 1 in close_kept
        @test length(close_kept) == 2

        # The boundary of a tetrahedron is not a filled tetrahedron. Its H2 must
        # survive when max_dim=2, although full-flag domination would delete edges.
        tetrahedron = [(u, v) for u in 1:4 for v in (u + 1):4]
        tetra_grades = fill((1.0,), length(tetrahedron))
        tetra_vertices = fill((0.0,), 4)
        tetra_kept = reduction._collapse_dominated_edges(tetrahedron, tetra_grades, 4, 2)
        tetra_full_kept = reduction._collapse_dominated_edges(tetrahedron, tetra_grades, 4, 3)
        @test tetra_kept == collect(eachindex(tetrahedron))
        @test length(tetra_full_kept) == 3

        # At 1.5 this graph is a four-cycle; at 2 its two triangles fill a disk.
        # The shared edge (1,2) has two nonadjacent common neighbors, not a cone.
        diamond = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4)]
        diamond_grades = [(2.0,), (1.5,), (1.5,), (1.5,), (1.5,)]
        diamond_kept = reduction._collapse_dominated_edges(diamond, diamond_grades, 4, 2)
        @test diamond_kept == collect(eachindex(diamond))

        for field in FIELDS_FULL
            @testset "field=$(field)" begin
                @test a06_betti(a06_cells(triangle, triangle_grades, vertex_grades, 2, (1.0,)), field) == [1, 0, 0]
                @test a06_betti(a06_cells(triangle, triangle_grades, vertex_grades, 1, (1.0,)), field) == [1, 1]
                @test a06_betti(a06_cells(tetrahedron, tetra_grades, tetra_vertices, 2, (1.0,)), field) == [1, 0, 1]
                @test a06_betti(a06_cells(tetrahedron, tetra_grades, tetra_vertices, 3, (1.0,)), field) == [1, 0, 0, 0]
                @test a06_betti(a06_cells(diamond, diamond_grades, tetra_vertices, 2, (1.5,)), field) == [1, 1, 0]
                @test a06_betti(a06_cells(diamond, diamond_grades, tetra_vertices, 2, (2.0,)), field) == [1, 0, 0]
                @test a06_betti(a06_cells(triangle, late_grades, late_vertices, 2, (2.0, 0.0)), field) == [1, 0, 0]
                # Independent reproduction of the wrong distance-MST answer.
                @test a06_betti(a06_cells(triangle[2:3], late_grades[2:3], late_vertices, 2, (2.0, 0.0)), field) == [2, 0, 0]
                @test a06_betti(a06_cells(triangle, close_grades, vertex_grades, 2, (1.0,)), field) == [2, 0, 0]
                for query in ((-1.0,), (0.0,), (1.0,))
                    a06_check_inclusion(triangle, triangle_grades, vertex_grades, 2, kept, query, field)
                    a06_check_inclusion(tetrahedron, tetra_grades, tetra_vertices, 2, tetra_kept, query, field)
                    a06_check_inclusion(tetrahedron, tetra_grades, tetra_vertices, 3, tetra_full_kept, query, field)
                end
                for query in ((0.0,), (1.0,), (almost_one,))
                    a06_check_inclusion(triangle, close_grades, vertex_grades, 2, close_kept, query, field)
                end
                for query in ((0.0, 0.0), (2.0, 0.0), (1.0, 10.0), (2.0, 10.0))
                    a06_check_inclusion(triangle, late_grades, late_vertices, 2, late_kept, query, field)
                end
                for query in ((0.0,), (1.5,), (2.0,))
                    a06_check_inclusion(diamond, diamond_grades, tetra_vertices, 2, diamond_kept, query, field)
                end
            end
        end

        if Threads.nthreads() > 1
            results = Vector{Vector{Int}}(undef, 32)
            Threads.@threads for i in eachindex(results)
                results[i] = reduction._collapse_dominated_edges(tetrahedron, tetra_grades, 4, 3)
            end
            @test all(==(tetra_full_kept), results)
            @test tetrahedron == [(u, v) for u in 1:4 for v in (u + 1):4]
            @test tetra_grades == fill((1.0,), 6)
        end
    end

    @testset "All critical grid grades: independent inclusion oracle" begin
        rng = MersenneTwister(0xa0606)
        for nparameters in 1:3, trial in 1:2
            n = trial + 3
            vertex_grades = NTuple{nparameters,Int}[ntuple(_ -> rand(rng, 0:1), nparameters) for _ in 1:n]
            edges = [(u, v) for u in 1:n for v in (u + 1):n if rand(rng) < 0.8]
            edge_grades = NTuple{nparameters,Int}[
                ntuple(k -> max(vertex_grades[u][k], vertex_grades[v][k], rand(rng, 0:1)),
                       nparameters) for (u, v) in edges]
            # Every possible critical coordinate, plus one grade below all births.
            # Even coordinates absent from one fixture do no harm to exhaustiveness.
            grid_axes = ntuple(_ -> (-1, 0, 1), nparameters)
            for max_dim in 1:3
                kept = reduction._collapse_dominated_edges(edges, edge_grades, n, max_dim)
                @test issorted(kept)
                @test length(unique(kept)) == length(kept)
                @test all(i -> 1 <= i <= length(edges), kept)
                # Relabeling the input order cannot change the chosen graph.
                reversed_kept = reduction._collapse_dominated_edges(reverse(edges), reverse(edge_grades), n, max_dim)
                @test Set(edges[kept]) == Set(reverse(edges)[reversed_kept])
                for field in FIELDS_FULL, query in Iterators.product(grid_axes...)
                    a06_check_inclusion(edges, edge_grades, vertex_grades,
                                         max_dim, kept, query, field)
                end
            end
        end
    end

    @testset "Public ingestion stages, grades, and contract rejection" begin
        points = DT.PointCloud([[0.0], [1.0], [2.0]])
        construction = OPT.ConstructionOptions(; collapse=:dominated_edges)
        typed = DI.RipsFiltration(; max_dim=2, construction=construction)
        build = DI.build_graded_complex(points, typed)
        complex = DI.graded_complex(build)
        @test length.(complex.cells_by_dim) == [3, 2, 0]
        @test complex.grades[1:3] == fill((0.0,), 3)
        @test DI.check_data_filtration(points, typed; throw=false).valid
        @test DI.check_data_filtration(points, typed; throw=true).valid
        @test DI.check_filtration_spec(DI._filtration_spec(typed); throw=false).valid

        tree = DI.encode(points, typed; stage=:simplex_tree)
        @test DI.simplex_count(tree) == 5
        @test count(i -> length(DI.simplex_vertices(tree, i)) == 2, 1:DI.simplex_count(tree)) == 2
        from_spec = DI.encode(points, DI._filtration_spec(typed); stage=:graded_complex)
        @test from_spec.cells_by_dim == complex.cells_by_dim
        @test from_spec.grades == complex.grades
        @test from_spec.boundaries == complex.boundaries

        # A fixed grid permits direct module comparison even when deletion removes
        # a now-redundant critical value from the automatically inferred axes.
        fixed_spec = OPT.FiltrationSpec(; kind=:rips, max_dim=2,
            axes=([0.0, 1.0, 2.0],), construction=construction)
        fixed_reference = OPT.FiltrationSpec(; kind=:rips, max_dim=2,
            axes=([0.0, 1.0, 2.0],))
        for field in FIELDS_FULL
            cache = CM.SessionCache()
            reduced_module = DI.encode(points, fixed_spec; stage=:module, degree=0, field=field, cache=cache)
            reference_module = DI.encode(points, fixed_reference; stage=:module, degree=0, field=field)
            @test DI.module_dims(reduced_module) == [3, 1, 1]
            @test DI.module_dims(reduced_module) == DI.module_dims(reference_module)
            for (edge, matrix) in reference_module.edge_maps
                @test FL.rank(field, reduced_module.edge_maps[edge...]) == FL.rank(field, matrix)
            end
            # Warm cache use must return the same mathematical answer.
            cached = DI.encode(points, fixed_spec; stage=:cohomology_dims, degree=0, field=field, cache=cache)
            @test cached.dims == [3, 1, 1]
            h1 = DI.encode(points, fixed_spec; stage=:cohomology_dims, degree=1, field=field)
            @test h1.dims == [0, 0, 0]
            encoded = DI.encode(points, fixed_spec; stage=:encoded_complex, field=field)
            @test encoded isa RES.EncodedComplexResult
        end

        # All vertices have degree two in the input triangle. A reduced tree has
        # degrees 1,2,1; recomputing these scores would change the bifiltration.
        degree_build = DI.build_graded_complex(points,
            DI.DegreeRipsFiltration(; max_dim=2, construction=construction))
        degree_complex = DI.graded_complex(degree_build)
        @test length.(degree_complex.cells_by_dim) == [3, 2, 0]
        @test degree_complex.grades[1:3] == fill((0.0, 2.0), 3)

        late_points = DT.PointCloud([[0.0], [2.0], [1.0]])
        late_filtration = DI.FunctionRipsFiltration(; max_dim=2,
            vertex_values=[0.0, 0.0, 10.0], construction=construction)
        late_complex = DI.graded_complex(DI.build_graded_complex(late_points, late_filtration))
        @test length.(late_complex.cells_by_dim) == [3, 3, 1]
        @test late_complex.grades[1:3] == [(0.0, 0.0), (0.0, 0.0), (0.0, 10.0)]

        # Sparse Rips now expands its selected graph to the requested dimension;
        # comparison is against that same graph, not against a dense surrogate.
        distance = [0.0 1.0 2.0; 1.0 0.0 1.0; 2.0 1.0 0.0]
        for sparsify in (:radius, :knn), mode in (:none, :dominated_edges)
            options = OPT.ConstructionOptions(; sparsify=sparsify, collapse=mode)
            filt = sparsify == :radius ?
                DI.RipsFiltration(; max_dim=2, radius=2.0, construction=options) :
                DI.RipsFiltration(; max_dim=2, knn=2, construction=options)
            expected = mode == :none ? [3, 3, 1] : [3, 2, 0]
            point_complex = DI.graded_complex(DI.build_graded_complex(points, filt))
            matrix_complex = DI.graded_complex(DI.build_graded_complex(distance, filt))
            @test length.(point_complex.cells_by_dim) == expected
            @test length.(matrix_complex.cells_by_dim) == expected
            @test point_complex.grades == matrix_complex.grades
            @test point_complex.boundaries == matrix_complex.boundaries
        end

        @test_throws ArgumentError DI.RhomboidFiltration(; construction=construction)
        unsupported = (
            (points, DI.AlphaFiltration(; construction=construction)),
            (DT.GraphData(3, [(1, 2), (2, 3)]),
             DI.CliqueLowerStarFiltration(; vertex_values=[0.0, 0.0, 0.0], construction=construction)),
            (DT.ImageNd([0.0 1.0; 2.0 3.0]), DI.ImageLowerStarFiltration(; construction=construction)),
            (points, DI.FunctionRipsFiltration(; max_dim=2, vertex_values=[0.0, 0.0, 0.0],
                                               simplex_agg=:mean, construction=construction)),
        )
        for (data, filt) in unsupported
            @test !DI.check_filtration_spec(DI._filtration_spec(filt); throw=false).valid
            @test !DI.check_data_filtration(data, filt; throw=false).valid
            @test_throws ArgumentError DI.check_data_filtration(data, filt; throw=true)
            @test_throws ArgumentError DI.plan_ingestion(data, filt)
            @test_throws ArgumentError DI.build_graded_complex(data, filt)
            @test_throws ArgumentError DI.encode(data, filt; stage=:graded_complex)
        end
        invalid_orientation = OPT.FiltrationSpec(; kind=:rips, max_dim=2,
            orientation=(-1,), construction=construction)
        @test !DI.check_filtration_spec(invalid_orientation; throw=false).valid
        @test_throws ArgumentError DI.check_filtration_spec(invalid_orientation; throw=true)
        @test_throws ArgumentError DI.plan_ingestion(points, invalid_orientation)
        @test_throws ArgumentError DI.encode(points, invalid_orientation; stage=:simplex_tree)

        @test_throws ArgumentError OPT.ConstructionOptions(; collapse=:acyclic)
        hand_built = OPT.ConstructionOptions(:none, :acyclic, :encoding_result, OPT.ConstructionBudget())
        @test !DI.check_construction_options(points, hand_built; throw=false).valid
        @test_throws ArgumentError DI.check_construction_options(points, hand_built; throw=true)
        invalid_options = DI.RipsFiltration(; max_dim=2, construction=hand_built)
        @test !DI.check_filtration_spec(DI._filtration_spec(invalid_options); throw=false).valid
        @test_throws ArgumentError DI.plan_ingestion(points, invalid_options)
        @test_throws ArgumentError DI.build_graded_complex(points, invalid_options)

        # Radius applies in every requested dimension. Four nearby vertices span
        # K4, while vertex 5 stays isolated. Its 2-skeleton has a genuine H2 class;
        # its 3-skeleton fills that class, without connecting the isolated vertex.
        square = DT.PointCloud([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0],
                                [0.0, 1.0], [10.0, 10.0]])
        for d in (2, 3), mode in (:none, :dominated_edges)
            opts = OPT.ConstructionOptions(; collapse=mode)
            filt = DI.RipsFiltration(; max_dim=d, radius=1.5, construction=opts)
            G = DI.graded_complex(DI.build_graded_complex(square, filt))
            expected_counts = d == 2 ? [5, 6, 4] :
                              mode == :none ? [5, 6, 4, 1] : [5, 5, 2, 0]
            @test length.(G.cells_by_dim) == expected_counts
            @test all(g -> g[1] <= 1.5, G.grades)
            for field in FIELDS_FULL
                counts = length.(G.cells_by_dim)
                ranks = [FL.rank(field, CM.coerce.(Ref(field), Matrix(B))) for B in G.boundaries]
                betti = [counts[q + 1] - (q == 0 ? 0 : ranks[q]) -
                         (q == d ? 0 : ranks[q + 1]) for q in 0:d]
                @test betti == (d == 2 ? [2, 0, 1] : [2, 0, 0, 0])
            end
        end
    end

    @testset "Input contracts" begin
        @test_throws ArgumentError reduction._collapse_dominated_edges([(1, 2)], NTuple{1,Float64}[], 2, 2)
        @test_throws ArgumentError reduction._collapse_dominated_edges([(1, 1)], [(0.0,)], 2, 2)
        @test_throws ArgumentError reduction._collapse_dominated_edges([(0, 2)], [(0.0,)], 2, 2)
        @test_throws ArgumentError reduction._collapse_dominated_edges([(1, 3)], [(0.0,)], 2, 2)
        @test_throws ArgumentError reduction._collapse_dominated_edges([(1, 2), (2, 1)], [(0.0,), (0.0,)], 2, 2)
        @test_throws ArgumentError reduction._collapse_dominated_edges([(1, 2)], [(NaN,)], 2, 2)
        @test_throws ArgumentError reduction._collapse_dominated_edges([(1, 2)], [(0.0,)], 2, -1)
        @test_throws ArgumentError reduction._collapse_dominated_edges(NTuple{2,Int}[], NTuple{1,Float64}[], -1, 2)
        @test isempty(reduction._collapse_dominated_edges(NTuple{2,Int}[], NTuple{1,Float64}[], 0, 2))
        # Endpoint ordering is immaterial; returned indices still refer to inputs.
        @test length(reduction._collapse_dominated_edges([(2, 1), (3, 1), (3, 2)], fill((1.0,), 3), 3, 2)) == 2
    end
end

@testset "A06 certified collapse: external distance adapters and budgets" begin
    collapse_options = TamerOp.Options.ConstructionOptions(collapse=:dominated_edges)
    D = [0.0 1.0 1.0; 1.0 0.0 1.0; 1.0 1.0 0.0]
    counts(G) = length.(G.cells_by_dim)
    mktempdir() do dir
        full = joinpath(dir, "full.txt")
        lower = joinpath(dir, "lower.txt")
        upper = joinpath(dir, "upper.txt")
        triplet = joinpath(dir, "triplet.txt")
        binary = joinpath(dir, "lower.bin")
        dipha = joinpath(dir, "dipha.bin")
        write(full, "0 1 1\n1 0 1\n1 1 0\n")
        write(lower, "0\n1 0\n1 1 0\n")
        write(upper, "0 1 1\n0 1\n0\n")
        write(triplet, "0 1 1\n0 2 1\n1 2 1\n")
        open(binary, "w") do io
            write(io, Float64[0,1,0,1,1,0])
        end
        open(dipha, "w") do io
            write(io, Int64[8067171840, 7, 3])
            write(io, vec(D))
        end
        for (loader, path) in ((SER.load_ripser_distance, full),
                               (SER.load_ripser_lower_distance, lower),
                               (SER.load_ripser_upper_distance, upper),
                               (SER.load_ripser_sparse_triplet, triplet),
                               (SER.load_ripser_binary_lower_distance, binary),
                               (SER.load_dipha_distance_matrix, dipha))
            for d in 0:3, sparsify in (:none, :radius, :knn)
                construction = TamerOp.Options.ConstructionOptions(collapse=:dominated_edges, sparsify=sparsify)
                kwargs = sparsify == :radius ? (; radius=1.0) : sparsify == :knn ? (; knn=2) : (;)
                G = loader(path; max_dim=d, construction=construction, kwargs...)
                expected = d == 0 ? [3] : d == 1 ? [3,3] : vcat([3,2], zeros(Int,d-1))
                @test counts(G) == expected
                @test all(isfinite(g[1]) for g in G.grades)
                # Signed boundary rank gives H0=1; graph triangle H1=1 is retained.
                if d >= 1
                    @test TamerOp.FieldLinAlg.rank_dim(FIELD_QQ, CM.coeff_type(FIELD_QQ).(Matrix(G.boundaries[1]))) == 2
                end
                f = DI.RipsFiltration(; max_dim=d, construction=construction, kwargs...)
                direct = DI.graded_complex(DI.build_graded_complex(D, f))
                @test counts(direct) == counts(G)
                @test direct.grades == G.grades
                @test direct.boundaries == G.boundaries
            end
        end
        write(triplet, "0 1 1\n2 2 0\n")
        @test counts(SER.load_ripser_sparse_triplet(triplet; max_dim=2, construction=collapse_options)) == [3,1,0]
        for contents in ("0 1 NaN\n", "0 1 -1\n", "0 0 2\n", "0.5 1 1\n")
            write(triplet, contents)
            @test_throws ArgumentError SER.load_ripser_sparse_triplet(triplet; construction=collapse_options)
        end
        write(lower, "1\n1 0\n1 1 0\n")
        @test_throws ArgumentError SER.load_ripser_lower_distance(lower; construction=collapse_options)
        write(upper, "0 1 1\n1 1\n0\n")
        @test_throws ArgumentError SER.load_ripser_upper_distance(upper; construction=collapse_options)
    end
    for invalid in ([0.0 NaN; NaN 0.0], [0.0 -1.0; -1.0 0.0],
                    [0.0 1.0; 2.0 0.0], [1.0 1.0; 1.0 0.0])
        @test_throws ArgumentError SER._graded_complex_from_distance_matrix(invalid; construction=collapse_options)
    end
    # The collapse cannot hide an excessive original edge budget; retained
    # simplex and memory limits are enforced before clique insertion.
    for budget in ((max_edges=2,), (max_simplices=4,), (memory_budget_bytes=47,))
        opts = TamerOp.Options.ConstructionOptions(collapse=:dominated_edges, budget=budget)
        @test_throws ArgumentError SER._graded_complex_from_distance_matrix(D; max_dim=2, construction=opts)
        @test_throws ArgumentError DI.build_graded_complex(D, DI.RipsFiltration(max_dim=2, construction=opts))
    end
    # Exact retained-budget boundary: original triangle has six graph cells;
    # the certified output has five and a 3-by-2 boundary (48 estimated bytes).
    tight = TamerOp.Options.ConstructionOptions(collapse=:dominated_edges,
        budget=(max_simplices=5, memory_budget_bytes=48))
    @test counts(SER._graded_complex_from_distance_matrix(D; max_dim=2, construction=tight)) == [3,2,0]
    @test counts(DI.graded_complex(DI.build_graded_complex(D, DI.RipsFiltration(max_dim=2, construction=tight)))) == [3,2,0]
    @test counts(DI.graded_complex(DI.build_graded_complex(DT.PointCloud([[0.0],[1.0],[2.0]]), DI.RipsFiltration(max_dim=2, construction=tight)))) == [3,2,0]
    SR = TamerOp.SimplicialReduction
    k4 = [(u,v) for u in 1:4 for v in (u+1):4]
    @test_throws ArgumentError SR._flag_simplices(k4,4,3;max_simplices=14)
    @test_throws ArgumentError SR._flag_simplices(k4,4,3;memory_budget_bytes=100)
end

@testset "A06: serialized construction and file-loading boundaries" begin
    mktempdir() do dir
        data = DT.PointCloud([0.0 0.0; 1.0 0.0; 0.5 0.75])
        pipeline_path = joinpath(dir, "pipeline.json")
        for mode in (:acyclic, :unknown_collapse)
            spec = OPT.FiltrationSpec(;
                kind=:rips, max_dim=2,
                construction=(sparsify=:none, collapse=mode,
                              output_stage=:encoding_result),
            )
            SER.save_pipeline_json(pipeline_path, data, spec)
            @test_throws ArgumentError SER.load_pipeline_json(pipeline_path)
            @test_throws ArgumentError SER.load_pipeline_json(pipeline_path; validation=:trusted)
            @test !SER.check_pipeline_json(pipeline_path).valid
            @test_throws ArgumentError SER.check_pipeline_json(pipeline_path; throw=true)
        end
        spec = OPT.FiltrationSpec(;
            kind=:rips, max_dim=2,
            construction=OPT.ConstructionOptions(; collapse=:dominated_edges),
        )
        SER.save_pipeline_json(pipeline_path, data, spec)
        for validation in (:strict, :trusted)
            _, loaded_spec, _, _ = SER.load_pipeline_json(pipeline_path; validation=validation)
            @test DI.construction_mode(loaded_spec).collapse == :dominated_edges
        end
        @test SER.check_pipeline_json(pipeline_path).valid

        point_path = joinpath(dir, "points.txt")
        write(point_path, "0.0 0.0\n1.0 0.0\n0.5 0.75\n")
        @test DT.point_matrix(DFI.load_data(point_path; format=:ripser_point_cloud)) == DT.point_matrix(data)
        @test_throws ArgumentError DFI.load_data(
            point_path; format=:ripser_point_cloud,
            construction=OPT.ConstructionOptions(; collapse=:dominated_edges),
        )
        @test_throws ArgumentError DFI.load_data(point_path; format=:ripser_point_cloud, max_dim=2)
    end
end

@testset "A08: neighbor-distance core bifiltrations" begin
    # This oracle constructs signed boundaries directly from active simplex
    # vertex sets. Activation uses the mathematical (radius, reverse-k) order,
    # independently of the ingestion planner and encoded-module routes.
    function _a08_betti(tree, radius, k, field)
        dimension = DT.max_dim(tree)
        simplices = [Tuple[] for _ in 0:dimension]
        for sid in 1:DT.simplex_count(tree)
            any(g -> g[1] <= radius && g[2] >= k, DT.simplex_grades(tree, sid)) || continue
            simplex = Tuple(DT.simplex_vertices(tree, sid))
            push!(simplices[length(simplex)], simplex)
        end
        counts = length.(simplices)
        ranks = zeros(Int, dimension)
        scalar = CM.coeff_type(field)
        for dim in 1:dimension
            rows = Dict(face => i for (i, face) in enumerate(simplices[dim]))
            matrix = zeros(scalar, counts[dim], counts[dim + 1])
            for (j, simplex) in enumerate(simplices[dim + 1])
                for omit in eachindex(simplex)
                    face = Tuple(simplex[t] for t in eachindex(simplex) if t != omit)
                    @test haskey(rows, face)
                    matrix[rows[face], j] = CM.coerce(field, isodd(omit) ? 1 : -1)
                end
            end
            ranks[dim] = FL.rank(field, matrix)
        end
        return [counts[dim + 1] - (dim == 0 ? 0 : ranks[dim]) -
                (dim == dimension ? 0 : ranks[dim + 1]) for dim in 0:dimension]
    end
    _a08_births(tree) = Dict(Tuple(DT.simplex_vertices(tree, sid)) =>
                            collect(DT.simplex_grades(tree, sid))
                            for sid in 1:DT.simplex_count(tree))

    @testset "Self-inclusive neighbor distances and minimal births" begin
        points = reshape([0.0, 1.0, 3.0], :, 1)
        ks = DI._core_k_values(nothing, 3)
        @test ks == [1, 2, 3]
        @test DI._core_k_values((3, 1), 3) == [1, 3]
        distances = DI._core_neighbor_distances(points, ks)
        @test distances == [0.0 1.0 3.0; 0.0 1.0 2.0; 0.0 2.0 3.0]
        @test DI._core_neighbor_distances(points, [1, 3]) == distances[:, [1, 3]]
        @test DI._core_neighbor_distances(2 .* points, ks) == 2 .* distances
        @test DI._core_neighbor_distances(reshape([8.0], 1, 1), [1]) == zeros(1, 1)
        @test DI._core_simplex_multigrades((1, 2), 0.5, distances, ks, 1.0) ==
              [(0.5, 1.0), (1.0, 2.0), (3.0, 3.0)]
        @test DI._core_simplex_multigrades((1, 2), 0.5, distances, ks, 0.5) ==
              [(0.5, 2.0), (1.5, 3.0)]
        @test DI._core_simplex_multigrades((1, 2), 3.0, distances, ks, 1.0) == [(3.0, 3.0)]
        @test DI._core_simplex_multigrades((1, 2), 0.5, distances[:, [1, 3]], [1, 3], 1.0) ==
              [(0.5, 1.0), (3.0, 3.0)]
        almost_equal = [1.0 nextfloat(1.0); 1.0 nextfloat(1.0)]
        @test DI._core_simplex_multigrades((1,), 0.0, almost_equal, [1, 2], 1.0) ==
              [(1.0, 1.0), (nextfloat(1.0), 2.0)]
        @test DI._core_multigrades([[[1], [2], [3]], [[1, 2]]],
            [[0.0, 0.0, 0.0], [0.5]], distances, ks, 1.0)[4] ==
            [(0.5, 1.0), (1.0, 2.0), (3.0, 3.0)]

        # Independent scalar selection oracle on several nonuniform clouds.
        rng = MersenneTwister(808)
        for n in (2, 5, 11)
            coordinates = randn(rng, n, 2)
            selected = unique(sort([1, max(1, div(n, 2)), n]))
            actual = DI._core_neighbor_distances(coordinates, selected)
            for v in 1:n
                reference = sort([sqrt(sum((coordinates[v, a] - coordinates[w, a])^2
                                          for a in 1:2)) for w in 1:n])
                @test isapprox(actual[v, :], reference[selected]; atol=1e-14, rtol=1e-14)
            end
        end
        for beta in (0, -1, Inf, NaN, true, "one")
            @test_throws ArgumentError DI._core_beta(beta)
        end
        for selected in (Int[], [0], [4], [1, 1], [1.5], [true], "all")
            @test_throws ArgumentError DI._core_k_values(selected, 3)
        end
        @test_throws ArgumentError DI._core_k_values(nothing, 0)
        @test_throws ArgumentError DI._core_neighbor_distances(points, [2, 1])
        @test_throws ArgumentError DI._core_neighbor_distances(fill(NaN, 2, 1), [1])
        @test_throws ArgumentError DI._core_neighbor_distances(zeros(2, 0), [1])
        @test_throws ArgumentError DI._core_simplex_multigrades((0,), 0.0, distances, ks, 1.0)
        @test_throws ArgumentError DI._core_simplex_multigrades((), 0.0, distances, ks, 1.0)
        @test_throws ArgumentError DI._core_simplex_multigrades((1,), -1.0, distances, ks, 1.0)
        @test_throws DimensionMismatch DI._core_simplex_multigrades((1,), 0.0, distances, [1], 1.0)
        @test_throws DimensionMismatch DI._core_multigrades([[[1]]], [[0.0, 1.0]], distances, ks, 1.0)
    end

    @testset "Exact line births, stages, fields, and cache reuse" begin
        points = DT.PointCloud(reshape([0.0, 1.0, 3.0], :, 1))
        expected_vertices = Dict(
            (1,) => [(0.0, 1.0), (1.0, 2.0), (3.0, 3.0)],
            (2,) => [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)],
            (3,) => [(0.0, 1.0), (2.0, 2.0), (3.0, 3.0)],
        )
        axes = ([0.0, 0.5, 1.0, 2.0, 3.0], [-3.0, -2.0, -1.0])
        for (kind, filtration) in ((:core, DI.CoreFiltration(max_dim=2)),
                                   (:core_delaunay, DI.CoreDelaunayFiltration(max_dim=2)))
            tree = DI.encode(points, filtration; stage=:simplex_tree)
            births = _a08_births(tree)
            for (simplex, expected) in expected_vertices
                @test births[simplex] == expected
            end
            @test births[(1, 2)] == [(0.5, 1.0), (1.0, 2.0), (3.0, 3.0)]
            @test births[(2, 3)] == [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
            if kind == :core
                @test births[(1, 3)] == [(1.5, 1.0), (2.0, 2.0), (3.0, 3.0)]
                @test births[(1, 2, 3)] == births[(1, 3)]
            else
                @test !haskey(births, (1, 3))
                @test !haskey(births, (1, 2, 3))
            end
            build = DI.build_graded_complex(points, filtration)
            complex = DI.graded_complex(build)
            @test complex isa DT.MultiCriticalGradedComplex
            @test DI.grade_orientation(build) == (1, -1)
            @test _canon_simplex_tree(DI._simplex_tree_multi_from_complex(complex)) == _canon_simplex_tree(tree)
            @test _canon_simplex_tree(DI.encode(points, DI._filtration_spec(filtration); stage=:simplex_tree)) ==
                  _canon_simplex_tree(tree)
            @test DI.check_data_filtration(points, filtration; throw=true).valid
            @test DI.filtration_kind(filtration) == kind
            @test DI.filtration_parameters(filtration).beta == 1.0
            if Threads.nthreads() > 1
                jobs = [Threads.@spawn(DI.encode(points, filtration; stage=:simplex_tree)) for _ in 1:8]
                @test all(job -> _canon_simplex_tree(fetch(job)) == _canon_simplex_tree(tree), jobs)
            end

            fixed_spec = OPT.FiltrationSpec(; kind, max_dim=2, beta=1.0, axes)
            for field in FIELDS_FULL
                cache = CM.SessionCache()
                encoded = DI.encode(points, fixed_spec; degree=0, field=field, cache=cache)
                dims = DI.encode(points, fixed_spec; stage=:cohomology_dims, degree=0, field=field, cache=cache)
                again = DI.encode(points, fixed_spec; stage=:cohomology_dims, degree=0, field=field, cache=cache)
                @test dims.dims == again.dims == _enc_dims(encoded)
                h1 = DI.encode(points, fixed_spec; stage=:cohomology_dims, degree=1, field=field)
                @test all(iszero, h1.dims)
                for (radius, k, expected) in ((0.0, 1.0, 3), (0.5, 1.0, 2),
                                              (1.0, 1.0, 1), (0.0, 2.0, 0),
                                              (1.0, 2.0, 1), (2.0, 2.0, 1),
                                              (1.0, 3.0, 0), (2.0, 3.0, 1), (3.0, 3.0, 1))
                    label = EC.locate(encoded.pi, [radius, k])
                    @test label > 0
                    @test _enc_dims(encoded)[label] == expected
                    @test _a08_betti(tree, radius, k, field)[1:2] == [expected, 0]
                end
                encoded_complex = DI.encode(points, fixed_spec; stage=:encoded_complex, field=field)
                @test encoded_complex isa RES.EncodedComplexResult
            end
        end
    end

    @testset "Cech circles, complete Voronoi nerves, and different core spaces" begin
        square = DT.PointCloud([0.0 0.0; 1.0 0.0; 1.0 1.0; 0.0 1.0])
        cech = DI.encode(square, DI.CoreFiltration(max_dim=3, k_values=[1]); stage=:simplex_tree)
        delaunay = DI.encode(square, DI.CoreDelaunayFiltration(max_dim=3, k_values=[1]); stage=:simplex_tree)
        @test DT.cell_counts(cech) == [4, 6, 4, 1]
        @test DT.cell_counts(delaunay) == [4, 6, 4, 1]
        births = _a08_births(delaunay)
        for edge in ((1, 2), (2, 3), (3, 4), (1, 4))
            @test births[edge] == [(0.5, 1.0)]
        end
        for simplex in ((1, 3), (2, 4), (1, 2, 3), (1, 2, 4),
                        (1, 3, 4), (2, 3, 4), (1, 2, 3, 4))
            @test length(births[simplex]) == 1
            @test isapprox(births[simplex][1][1], sqrt(0.5); atol=1e-14)
            @test births[simplex][1][2] == 1.0
        end
        for field in FIELDS_FULL, tree in (cech, delaunay)
            @test _a08_betti(tree, 0.49, 1, field) == [4, 0, 0, 0]
            @test _a08_betti(tree, 0.6, 1, field) == [1, 1, 0, 0]
            @test _a08_betti(tree, 0.71, 1, field) == [1, 0, 0, 0]
        end
        # Voronoi restriction and density selection do not commute. The center
        # is inactive here, although the two active outer clusters' balls meet.
        split = DT.PointCloud(reshape([-2.0, -1.9, 0.0, 1.9, 2.0], :, 1))
        full = DI.encode(split, DI.CoreFiltration(max_dim=2, beta=2.0, k_values=[2]); stage=:simplex_tree)
        restricted = DI.encode(split, DI.CoreDelaunayFiltration(max_dim=2, beta=2.0, k_values=[2]); stage=:simplex_tree)
        for field in FIELDS_FULL
            @test _a08_betti(full, 2.0, 2, field)[1:2] == [1, 0]
            @test _a08_betti(restricted, 2.0, 2, field)[1:2] == [2, 0]
            @test _a08_betti(restricted, 4.0, 2, field)[1:2] == [1, 0]
        end
    end

    @testset "JSON replay, query contracts, and explicit graph core" begin
        points = DT.PointCloud(reshape([0.0, 1.0, 3.0], :, 1))
        mktempdir() do dir
            for kind in (:core, :core_delaunay)
                spec = OPT.FiltrationSpec(; kind, max_dim=2, beta=0.5, k_values=[1, 3])
                reference = DI.encode(points, spec; stage=:simplex_tree)
                path = joinpath(dir, "$(kind)_pipeline.json")
                SER.save_pipeline_json(path, points, spec; degree=1)
                for validation in (:strict, :trusted)
                    loaded_points, loaded_spec, degree, _ = SER.load_pipeline_json(path; validation)
                    @test degree == 1
                    @test DI.filtration_parameters(loaded_spec).beta == 0.5
                    @test DI.filtration_parameters(loaded_spec).k_values == [1, 3]
                    @test _canon_simplex_tree(DI.encode(loaded_points, loaded_spec; stage=:simplex_tree)) ==
                          _canon_simplex_tree(reference)
                end
                @test SER.check_pipeline_json(path; throw=true).valid
                dataset_path = joinpath(dir, "$(kind)_complex.json")
                complex = DI.encode(points, spec; stage=:graded_complex)
                SER.save_dataset_json(dataset_path, complex)
                loaded_complex = SER.load_dataset_json(dataset_path)
                @test _canon_simplex_tree(DI._simplex_tree_multi_from_complex(loaded_complex)) ==
                      _canon_simplex_tree(reference)
            end
        end
        for kind in (:core, :core_delaunay)
            for params in ((beta=0.0,), (beta=-1.0,), (beta=Inf,), (beta=NaN,),
                           (k_values=Int[],), (k_values=[0],), (k_values=[4],),
                           (k_values=[1, 1],), (k_values=[1.5],), (max_dim=-1,),
                           (orientation=(1, 1),), (orientation=(-1, -1),),
                           (knn=2,), (radius=1.0,), (vertex_values=[0, 0, 0],),
                           (simplex_agg=:max,), (highdim_policy=:rips,))
                spec = OPT.FiltrationSpec(; kind, params...)
                @test !DI.check_filtration_spec(spec; throw=false).valid ||
                      !DI.check_data_filtration(points, DI.to_filtration(spec); throw=false).valid
                @test_throws ArgumentError DI.plan_ingestion(points, spec)
                @test_throws ArgumentError DI.encode(points, spec; stage=:simplex_tree)
            end
            spec = OPT.FiltrationSpec(; kind, max_dim=2)
            @test_throws ArgumentError DI.encode(points, spec; stage=:graded_complex,
                pipeline=OPT.PipelineOptions(orientation=(1, 1)))
            for bad_data in (DT.PointCloud([0.0 0.0 0.0; 1.0 1.0 1.0]),
                             DT.PointCloud(reshape([0.0, NaN], :, 1)),
                             DT.GraphData(3, [(1, 2), (2, 3)]))
                @test_throws ArgumentError DI.plan_ingestion(bad_data, spec)
            end
            for budget in ((max_edges=0,), (max_simplices=2,), (memory_budget_bytes=1,))
                limited = OPT.FiltrationSpec(; kind, max_dim=2,
                    construction=OPT.ConstructionOptions(; budget))
                @test_throws ArgumentError DI.encode(points, limited; stage=:simplex_tree)
            end
            for mode in (OPT.ConstructionOptions(sparsify=:knn),
                         OPT.ConstructionOptions(collapse=:dominated_edges))
                limited = OPT.FiltrationSpec(; kind, max_dim=2, construction=mode)
                @test_throws ArgumentError DI.plan_ingestion(points, limited)
            end
        end
        graph = DT.GraphData(4, [(1, 2), (1, 3), (2, 3), (3, 4)])
        graph_filtration = DI.GraphCoreFiltration(vertex_values=zeros(4))
        graph_build = DI.build_graded_complex(graph, graph_filtration)
        @test DI.grade_orientation(graph_build) == (1, -1)
        graph_complex = DI.graded_complex(graph_build)
        @test graph_complex.grades[1:4] == [(0.0, 2.0), (0.0, 2.0), (0.0, 2.0), (0.0, 1.0)]
        @test graph_complex.grades[5:8] == [(0.0, 2.0), (0.0, 2.0), (0.0, 2.0), (0.0, 1.0)]
        graph_tree = DI.encode(graph, graph_filtration; stage=:simplex_tree)
        for field in FIELDS_FULL
            @test _a08_betti(graph_tree, 0.0, 2, field) == [1, 1]
            @test _a08_betti(graph_tree, 0.0, 1, field) == [1, 1]
            @test _a08_betti(graph_tree, 0.0, 3, field) == [0, 0]
        end
    end
end

@testset "A07: incremental function-Delaunay-Cech oracles" begin
    # Independent chain-matrix oracle: decide membership from birth inequalities
    # and assemble every boundary by deleting one vertex, without ingestion's
    # boundary constructors or persistence algorithms.
    function a07_betti(tree, radius, level, field; density=false)
        simplices = [Tuple[] for _ in DT.cell_counts(tree)]
        for sid in 1:DT.simplex_count(tree)
            active = any(g -> g[1] <= radius && (density ? g[2] >= level : g[2] <= level),
                         DT.simplex_grades(tree, sid))
            active || continue
            push!(simplices[DT.simplex_dimension(tree, sid) + 1], Tuple(DT.simplex_vertices(tree, sid)))
        end
        ranks = Int[0]
        for d in 2:length(simplices)
            rows = Dict(s => i for (i, s) in enumerate(simplices[d - 1]))
            boundary = zeros(CM.coeff_type(field), length(rows), length(simplices[d]))
            for (column, simplex) in enumerate(simplices[d]), omit in 1:d
                face = Tuple(simplex[i] for i in 1:d if i != omit)
                @test haskey(rows, face)
                boundary[rows[face], column] = CM.coerce(field, isodd(omit) ? 1 : -1)
            end
            push!(ranks, FL.rank(field, boundary))
        end
        push!(ranks, 0)
        return [length(simplices[d]) - ranks[d] - ranks[d + 1] for d in eachindex(simplices)]
    end
    function a07_tree(points, values; max_dim=3, backend=:naive, cache=:auto)
        return TamerOp.encode(DT.PointCloud(points), DI.FunctionDelaunayFiltration(
            vertex_values=values, max_dim=max_dim, delaunay_backend=backend);
            stage=:simplex_tree, cache=cache)
    end

    @testset "Changing sublevels and insertion cofaces" begin
        points = [[0.0], [1.0], [2.0]]
        values = [0.0, 1.0, 0.0]
        tree = a07_tree(points, values)
        births = Dict(Tuple(DT.simplex_vertices(tree, i)) => only(DT.simplex_grades(tree, i))
                      for i in 1:DT.simplex_count(tree))
        @test births[(1,3)] == (1.0, 0.0)
        @test births[(1,2,3)] == (1.0, 1.0)
        @test DT.cell_counts(tree) == [3,3,1,0]
        for field in FIELDS_FULL
            @test a07_betti(tree, 0.99, 0.0, field) == [2,0,0,0]
            @test a07_betti(tree, 1.1, 0.0, field) == [1,0,0,0]
            @test a07_betti(tree, 1.1, 1.0, field) == [1,0,0,0]
            data = DT.PointCloud(points)
            filtration = DI.FunctionDelaunayFiltration(vertex_values=values, delaunay_backend=:naive)
            enc = TamerOp.encode(data, filtration; degree=0, field=field)
            module_ = _enc_module(enc)
            @test MD.dim_at(module_, EC.locate(enc.pi, [1.1,0.0])) == 1
        end
        # Ambient 2D collinear prefixes use the same interval insertion rule.
        linear2d = a07_tree([[0.,0.], [1.,1.], [2.,2.]], values)
        @test DT.cell_counts(linear2d) == [3,3,1,0]
        @test a07_betti(linear2d, 1.5, 1., CM.F2()) == [1,0,0,0]

        planar = [[-1.,0.], [1.,0.], [0.,2.], [0.,0.5]]
        planar_tree = a07_tree(planar, [0.,0.,0.,1.])
        @test DT.cell_counts(planar_tree) == [4,6,4,1]
        for field in FIELDS_FULL
            @test a07_betti(planar_tree, 1.3, 0., field) == [1,0,0,0]
            @test a07_betti(planar_tree, 1.3, 1., field) == [1,0,0,0]
            # Truncating the tetrahedron intentionally exposes top-degree H2.
            truncated = a07_tree(planar, [0.,0.,0.,1.]; max_dim=2)
            @test a07_betti(truncated, 1.3, 1., field) == [1,0,1]
        end
        # The MEB of this obtuse triangle is the diameter disk of its long
        # edge (radius 1), not its circumdisk (radius approximately 3.833).
        obtuse = a07_tree([[0.,0.],[2.,0.],[0.5,0.1]], zeros(3))
        triangle = findfirst(i -> DT.simplex_dimension(obtuse,i) == 2, 1:DT.simplex_count(obtuse))
        @test only(DT.simplex_grades(obtuse,triangle)) == (1.0,0.0)
        for field in FIELDS_FULL
            @test a07_betti(obtuse,1.1,0.,field) == [1,0,0,0]
        end
    end

    @testset "Geometry, stages, caches and contracts" begin
        pts = DT.PointCloud([[-1.,0.], [1.,0.], [0.,2.], [0.,0.5]])
        filtration = DI.FunctionDelaunayFiltration(vertex_values=[0.,0.,0.,1.], delaunay_backend=:naive)
        spec = DI._filtration_spec(filtration)
        direct = DI.build_graded_complex(pts, filtration)
        @test DI.grade_orientation(direct) == (1,1)
        @test DI.check_filtration(filtration).valid
        @test DI.check_filtration_spec(spec).valid
        @test DI.filtration_kind(DI.to_filtration(spec)) == :function_delaunay
        @test occursin("FunctionDelaunay", sprint(show,filtration))
        @test DI.filtration_summary(filtration).kind == :function_delaunay
        @test DI.cell_counts_by_dim(DI.estimate_ingestion(pts,filtration)) == BigInt[4,6,4,1]
        session = CM.SessionCache()
        for cache in (:auto, session)
            tree = TamerOp.encode(pts,filtration; stage=:simplex_tree,cache=cache)
            again = TamerOp.encode(pts,filtration; stage=:simplex_tree,cache=cache)
            @test _canon_simplex_tree(tree) == _canon_simplex_tree(again)
            reconstructed = DI._graded_complex_from_simplex_tree(tree)
            @test reconstructed.grades == DI.graded_complex(direct).grades
            @test reconstructed.boundaries == DI.graded_complex(direct).boundaries
        end
        if Threads.nthreads() > 1
            trees = Vector{Any}(undef,8)
            Threads.@threads for i in eachindex(trees)
                trees[i] = TamerOp.encode(pts,filtration;stage=:simplex_tree,cache=session)
            end
            @test all(t -> _canon_simplex_tree(t) == _canon_simplex_tree(first(trees)), trees)
        end
        # Fixed-triangulation helper must retain interior edges in its
        # 1-skeleton; a planar interior point has three incident edges.
        p1 = DI._packed_delaunay_entry(pts.points, spec;max_dim=1).packed
        p2 = DI._packed_delaunay_entry(pts.points, spec;max_dim=2).packed
        @test p1.edges == p2.edges
        @test length(p1.edges) == 6
        square = DT.PointCloud([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
        packed = DI._packed_delaunay_entry(square.points,spec;max_dim=2).packed
        @test length(packed.edges) == 5
        @test length(packed.triangles) == 2
        @test_throws ArgumentError a07_tree(square.points, collect(1.:4.))
        @test_throws ArgumentError a07_tree([[0.],[0.]], [0.,1.])
        @test_throws ArgumentError a07_tree([[NaN],[1.]], [0.,1.])
        @test_throws ArgumentError a07_tree([[0.,0.,0.],[1.,0.,0.]], [0.,1.])
        @test_throws ArgumentError DI.FunctionDelaunayFiltration(vertex_values=[0.,1.], max_dim=-1)
        @test_throws ArgumentError DI.FunctionDelaunayFiltration(vertex_values=[0.,1.], max_dim=4)
        for params in ((simplex_agg=:mean,), (highdim_policy=:rips,), (orientation=(1,-1),))
            bad = OPT.FiltrationSpec(; kind=:function_delaunay, vertex_values=[0.,0.,0.,1.], params...)
            @test !DI.check_filtration_spec(bad).valid
            @test_throws ArgumentError DI.plan_ingestion(pts,bad)
        end
        @test_throws ArgumentError DI.plan_ingestion(pts,filtration;pipeline=OPT.PipelineOptions(orientation=(1,-1)))
        for budget in (OPT.ConstructionBudget(max_edges=2), OPT.ConstructionBudget(max_simplices=5),
                       OPT.ConstructionBudget(memory_budget_bytes=1))
            @test_throws ArgumentError TamerOp.encode(pts,filtration;stage=:simplex_tree,
                construction=OPT.ConstructionOptions(budget=budget))
        end
        # Geometry predicates must not erase small-scale or translated cells.
        for scale in (1e-8,1.,1e8)
            scaled = [[scale*x for x in p] for p in pts.points]
            @test DT.cell_counts(a07_tree(scaled,[0.,0.,0.,1.])) == [4,6,4,1]
        end
        if DI._have_pointcloud_delaunay_backend()
            fast = a07_tree(pts.points,[0.,0.,0.,1.];backend=:fast)
            naive = a07_tree(pts.points,[0.,0.,0.,1.])
            fast_births = Dict(Tuple(DT.simplex_vertices(fast,i)) => only(DT.simplex_grades(fast,i)) for i in 1:DT.simplex_count(fast))
            naive_births = Dict(Tuple(DT.simplex_vertices(naive,i)) => only(DT.simplex_grades(naive,i)) for i in 1:DT.simplex_count(naive))
            @test keys(fast_births) == keys(naive_births)
            @test all(k -> all(isapprox.(fast_births[k],naive_births[k];atol=1e-12)), keys(fast_births))
        end
    end

    @testset "Differential sublevel-offset homology" begin
        rng = MersenneTwister(708)
        for ambient in (1,2), trial in 1:8
            points = [randn(rng,ambient) for _ in 1:6]
            values = rand(rng,6)
            tree = a07_tree(points,values)
            for level in sort(values), radius in (0.,0.25,0.75,2.)
                active = findall(<=(level), values)
                cech = TamerOp.encode(DT.PointCloud(points[active]),
                    DI.CoreFiltration(max_dim=3,k_values=[1]);stage=:simplex_tree)
                for field in (CM.F2(),CM.QQField())
                    obtained = a07_betti(tree,radius,level,field)
                    expected = a07_betti(cech,radius,1,field;density=true)
                    @test obtained[1] == expected[1]
                    @test obtained[2] == (length(expected) >= 2 ? expected[2] : 0)
                    @test obtained[3] == (length(expected) >= 3 ? expected[3] : 0)
                end
            end
        end
    end
end


@testset "A07/A08: mutable-coordinate cache lifecycle and geometric extremes" begin
    @testset "Global Delaunay cache follows all coordinate edits" begin
        was_enabled = DI._POINTCLOUD_DELAUNAY_CACHE_ENABLED[]
        DI._POINTCLOUD_DELAUNAY_CACHE_ENABLED[] = true
        DI._clear_pointcloud_delaunay_cache!()
        try
            coords = [0.0 0.0; 2.0 0.0; 0.0 2.0; 0.2 0.2]
            data = DT.PointCloud(coords)
            spec = OPT.FiltrationSpec(kind=:core_delaunay, max_dim=2,
                                      k_values=[1], delaunay_backend=:naive)
            @test isempty(DI._POINTCLOUD_DELAUNAY_CACHE)
            key_before = DI._delaunay_cache_key(data.points, 2, :naive)
            first_entry = DI._packed_delaunay_entry(data.points, spec; max_dim=2)
            @test haskey(DI._POINTCLOUD_DELAUNAY_CACHE, key_before)
            @test DI._packed_delaunay_entry(data.points, spec; max_dim=2) === first_entry
            @test length(DI._POINTCLOUD_DELAUNAY_CACHE) == 1
            @test length(first_entry.packed.edges) == 6
            tree_before = DI.encode(data, spec; stage=:simplex_tree)

            # Move the interior vertex outside the original triangle. Both
            # connectivity and radii change without replacing the dataset.
            coords[4, :] .= 3.0
            key_after = DI._delaunay_cache_key(data.points, 2, :naive)
            @test key_after != key_before
            changed_entry = DI._packed_delaunay_entry(data.points, spec; max_dim=2)
            @test changed_entry !== first_entry
            @test length(changed_entry.packed.edges) == 5
            @test changed_entry.packed.edge_radius != first_entry.packed.edge_radius
            @test DI._packed_delaunay_entry(data.points, spec; max_dim=2) === changed_entry
            tree_after = DI.encode(data, spec; stage=:simplex_tree)
            @test _canon_simplex_tree(tree_after) != _canon_simplex_tree(tree_before)
            DI._POINTCLOUD_DELAUNAY_CACHE_ENABLED[] = false
            reference = DI.encode(data, spec; stage=:simplex_tree, cache=nothing)
            @test _canon_simplex_tree(tree_after) == _canon_simplex_tree(reference)
            DI._POINTCLOUD_DELAUNAY_CACHE_ENABLED[] = true

            DI._clear_pointcloud_delaunay_cache!()
            @test isempty(DI._POINTCLOUD_DELAUNAY_CACHE)
            @test isempty(DI._POINTCLOUD_DELAUNAY_CACHE_ORDER)
            rebuilt = DI._packed_delaunay_entry(data.points, spec; max_dim=2)
            @test rebuilt !== changed_entry
            @test rebuilt.packed.edges == changed_entry.packed.edges
            @test rebuilt.packed.edge_radius == changed_entry.packed.edge_radius
            @test length(DI._POINTCLOUD_DELAUNAY_CACHE) == 1

            # Every coordinate participates, including positions a sampled
            # array hash could omit. Geometry need not be built for this check.
            packed_coords = reshape(collect(1.0:128.0), 64, 2)
            fingerprint_data = DT.PointCloud(packed_coords)
            original_key = DI._delaunay_cache_key(fingerprint_data.points, 2, :naive)
            for index in eachindex(packed_coords)
                old = packed_coords[index]
                packed_coords[index] = old + 0.25
                @test DI._delaunay_cache_key(fingerprint_data.points, 2, :naive) != original_key
                packed_coords[index] = old
            end
            @test DI._delaunay_cache_key(fingerprint_data.points, 2, :naive) == original_key
        finally
            DI._POINTCLOUD_DELAUNAY_CACHE_ENABLED[] = was_enabled
            DI._clear_pointcloud_delaunay_cache!()
        end
    end

    @testset "Session poset/module caches follow coordinate edits" begin
        for kind in (:core, :core_delaunay, :function_delaunay), field in FIELDS_FULL
            coords = [0.0 0.0; 1.0 0.0]
            data = DT.PointCloud(coords)
            density = kind != :function_delaunay
            axes = ([0.0, 0.5, 0.75, 1.5], density ? [-1.0] : [0.0])
            spec = if density
                OPT.FiltrationSpec(; kind, max_dim=1, k_values=[1], axes)
            else
                OPT.FiltrationSpec(; kind, max_dim=1, vertex_values=zeros(2),
                                   delaunay_backend=:naive, axes)
            end
            session = CM.SessionCache()
            before = DI.encode(data, spec; stage=:module, degree=0, field, cache=session)
            @test DI.module_dims(before) == [2, 1, 1, 1]
            @test DI.encode(data, spec; stage=:module, degree=0, field, cache=session) === before

            # The axes stay fixed: an identity-only module key would silently
            # return the old connected component counts at radii 0.5 and 0.75.
            coords[2, 1] = 3.0
            after = DI.encode(data, spec; stage=:module, degree=0, field, cache=session)
            reference = DI.encode(data, spec; stage=:module, degree=0, field, cache=nothing)
            @test after !== before
            @test DI.module_dims(after) == DI.module_dims(reference) == [2, 2, 2, 1]
            @test DI.module_dims(before) == [2, 1, 1, 1]
            @test DI.encode(data, spec; stage=:module, degree=0, field, cache=session) === after

            CM._clear_session_cache!(session)
            rebuilt = DI.encode(data, spec; stage=:module, degree=0, field, cache=session)
            @test rebuilt !== after
            @test DI.module_dims(rebuilt) == [2, 2, 2, 1]
        end
    end

    @testset "Vertex truncation, distinct points, and subnormal radii" begin
        # The circumradius exceeds Float64, but a vertex-only request has no
        # need to construct a triangle or calculate that radius.
        nearly_collinear = DT.PointCloud([0.0 0.0; 1.0 0.0; 0.5 1e-320])
        DI._clear_pointcloud_delaunay_cache!()
        for filtration in (DI.CoreFiltration(max_dim=0, k_values=[1]),
                           DI.CoreDelaunayFiltration(max_dim=0, k_values=[1], delaunay_backend=:naive))
            tree = DI.encode(nearly_collinear, filtration; stage=:simplex_tree)
            @test DT.cell_counts(tree) == [3]
            @test all(i -> only(DT.simplex_grades(tree, i)) == (0.0, 1.0), 1:3)
        end
        @test isempty(DI._POINTCLOUD_DELAUNAY_CACHE)

        for coords in (reshape([0.0, 0.0], 2, 1), [0.0 -0.0; 0.0 0.0],
                       reshape([big(1)//1, (big(2)^60 + 1)//big(2)^60], 2, 1))
            duplicate = DT.PointCloud(coords)
            for filtration in (DI.CoreFiltration(), DI.CoreDelaunayFiltration(delaunay_backend=:naive),
                               DI.FunctionDelaunayFiltration(vertex_values=zeros(2)))
                @test !DI.check_data_filtration(duplicate, filtration; throw=false).valid
                @test_throws ArgumentError DI.check_data_filtration(duplicate, filtration; throw=true)
                @test_throws ArgumentError DI.estimate_ingestion(duplicate, filtration)
                @test_throws ArgumentError DI.plan_ingestion(duplicate, filtration)
                @test_throws ArgumentError DI.encode(duplicate, filtration; stage=:simplex_tree)
            end
        end

        unit = nextfloat(0.0)
        @test DI._half_point_distance([3unit], [unit]) == unit
        @test DI._half_point_distance([unit], [3unit]) == unit
        @test DI._half_point_distance([0.0, 0.0], [unit, unit]) == unit
        for x in 0:5, y in 0:5
            expected = Float64(hypot(BigFloat(x) * unit, BigFloat(y) * unit) / 2)
            @test DI._half_point_distance([0.0, 0.0], [x * unit, y * unit]) == expected
        end
        @test DI._half_point_distance([floatmax(Float64)], [-floatmax(Float64)]) == floatmax(Float64)
        for coords in (reshape([unit, 3unit], 2, 1), [unit 0.0; 3unit 0.0], [0.0 0.0; unit unit])
            data = DT.PointCloud(coords)
            for filtration in (DI.CoreFiltration(max_dim=1, k_values=[1]),
                               DI.CoreDelaunayFiltration(max_dim=1, k_values=[1], delaunay_backend=:naive),
                               DI.FunctionDelaunayFiltration(max_dim=1, vertex_values=zeros(2), delaunay_backend=:naive))
                tree = DI.encode(data, filtration; stage=:simplex_tree)
                @test DT.cell_counts(tree) == [2, 1]
                edge = findfirst(i -> DT.simplex_dimension(tree, i) == 1, 1:DT.simplex_count(tree))
                @test only(DT.simplex_grades(tree, edge))[1] == unit
            end
        end
    end
end

@testset "A08: oriented grids apply the sign once" begin
    for field in FIELDS_FULL
        # Physical superlevels x>=s correspond to increasing coordinates -s.
        data = DT.GradedComplex([[1,2]], SparseMatrixCSC{Int,Int}[], [(1.0,), (2.0,)])
        enc = DI.encode(data, OPT.FiltrationSpec(kind=:graded, orientation=(-1,));field=field)
        M = _enc_module(enc)
        first_label, second_label = EC.locate(enc.pi,[2.0]), EC.locate(enc.pi,[1.0])
        @test MD.dim_at(M,first_label) == 1
        @test MD.dim_at(M,second_label) == 2
        @test FF.leq(enc.P,first_label,second_label)
        @test !FF.leq(enc.P,second_label,first_label)
        @test FL.rank(field,MD.structure_map(M;source=first_label,target=second_label)) == 1

        points = DT.PointCloud([[0.],[1.],[3.]])
        for filtration in (DI.CoreFiltration(), DI.CoreDelaunayFiltration(delaunay_backend=:naive))
            encoded = DI.encode(points,filtration;field=field,degree=0)
            module_ = _enc_module(encoded)
            start = EC.locate(encoded.pi,[0.,1.])
            merged = EC.locate(encoded.pi,[1.,1.])
            dense = EC.locate(encoded.pi,[1.,2.])
            @test MD.dim_at(module_,start) == 3
            @test MD.dim_at(module_,merged) == 1
            @test FF.leq(encoded.P,start,merged)
            @test FF.leq(encoded.P,dense,merged)
            @test !FF.leq(encoded.P,merged,dense)
            @test FL.rank(field,MD.structure_map(module_;source=start,target=merged)) == 1
            @test FL.rank(field,MD.structure_map(module_;source=dense,target=merged)) == 1
            @test MD.check_module(module_;throw=true).valid
        end
    end
end

@testset "A11 lazy cochains and H0 own their parallel lookup scratch" begin
    # Four isolated vertices merge along a path as edges appear. Repeated axes
    # exercise more than one chunk while retaining a hand-computable H0 oracle.
    cells = [collect(1:4), collect(1:3)]
    B = sparse([1,2,2,3,3,4], [1,1,2,2,3,3], [-1,1,-1,1,-1,1], 4, 3)
    G = TamerOp.GradedComplex(cells, [B],
        [[0.0], [0.0], [0.0], [0.0], [1.0], [2.0], [3.0]])
    axes = (collect(0.0:0.1:3.0),)
    P = DI.poset_from_axes(axes)
    expected = [4 - min(3, floor(Int, t + 1e-9)) for t in axes[1]]
    old_gate = DI._LAZY_DIFF_THREADS_MIN_VERTICES[]
    try
        DI._LAZY_DIFF_THREADS_MIN_VERTICES[] = 1
        with_fields(FIELDS_FULL) do field
            L = DI._lazy_cochain_complex_from_graded_complex(G, P, axes; field=field)
            DI._lazy_ensure_active!(L, 1)
            DI._lazy_ensure_active!(L, 2)
            active0, active1 = L.active_by_dim[1], L.active_by_dim[2]
            K = CM.coeff_type(field)
            oracle_diff = [K[CM.coerce(field, B[v,e]) for v in active0[i], e in active1[i]]
                for i in eachindex(expected)]
            function exercise_lazy()
                yield()
                diffs = DI._lazy_diff_components(L, 1; threaded=true)
                dims = DI._cohomology_dims_h0_unionfind_from_lazy(L, active0, active1)
                module_ = DI._cohomology_module_h0_unionfind_from_lazy(L, active0, active1)
                return (diffs=Matrix.(diffs), dims=dims, module_dims=module_.dims)
            end
            direct = exercise_lazy()
            @test direct.diffs == oracle_diff
            @test direct.dims == expected
            @test direct.module_dims == expected
            @test Matrix.(DI._lazy_diff_components(L, 1; threaded=false)) == oracle_diff
            tasks = [Threads.@spawn exercise_lazy() for _ in 1:4]
            for task in tasks
                result = fetch(task)
                @test result.diffs == oracle_diff
                @test result.dims == expected
                @test result.module_dims == expected
            end
            nested = Vector{Bool}(undef, 3)
            Threads.@threads for i in eachindex(nested)
                local result = exercise_lazy()
                nested[i] = result.diffs == oracle_diff && result.dims == expected && result.module_dims == expected
            end
            @test all(nested)
            if Threads.nthreads(:interactive) > 0
                result = fetch(Threads.@spawn :interactive exercise_lazy())
                @test result.diffs == oracle_diff
                @test result.dims == expected
                @test result.module_dims == expected
            end
        end
    finally
        DI._LAZY_DIFF_THREADS_MIN_VERTICES[] = old_gate
    end
end

@testset "A09: interval multicover components and inclusion-rank oracles" begin
    # In R, the k-fold cover is the union, over consecutive k sites, of
    # [x[i+k-1]-r, x[i]+r]. This oracle uses no rhomboid or chain-complex code.
    function interval_multicover_components(sites, radius, depth)
        depth == 0 && return [(-Inf, Inf)]
        intervals = Tuple{Float64,Float64}[]
        for i in 1:(length(sites) - depth + 1)
            lo = sites[i + depth - 1] - radius
            hi = sites[i] + radius
            lo <= hi && push!(intervals, (lo, hi))
        end
        sort!(intervals)
        local components = Tuple{Float64,Float64}[]
        for (lo, hi) in intervals
            if !isempty(components) && lo <= last(components)[2]
                previous_lo, previous_hi = pop!(components)
                push!(components, (previous_lo, max(previous_hi, hi)))
            else
                push!(components, (lo, hi))
            end
        end
        return components
    end
    function interval_inclusion_rank(source, target)
        return count(target) do interval
            any(source) do component
                interval[1] <= component[1] && component[2] <= interval[2]
            end
        end
    end

    sites = [0.0, 2.0, 5.0, 9.0]
    radii = sort!(unique!([0.0; [(sites[j]-sites[i])/2 for i in 1:4 for j in i+1:4]]))
    depths = collect(0:4)
    queries = [(r, k) for k in depths for r in radii]
    components = [interval_multicover_components(sites, r, k) for (r, k) in queries]
    @test interval_multicover_components(sites, 1.0, 2) == [(1.0, 1.0)]
    @test interval_multicover_components(sites, 1.5, 2) == [(0.5, 1.5), (3.5, 3.5)]
    @test interval_multicover_components(sites, 4.5, 4) == [(4.5, 4.5)]
    axes = (radii, sort!(-Float64.(depths)))
    data = DT.PointCloud(reshape(sites, :, 1))
    spec = OPT.FiltrationSpec(kind=:rhomboid, axes=axes)
    tree = DI.encode(data, spec; stage=:simplex_tree)
    tree_spec = OPT.FiltrationSpec(kind=:graded, axes=axes, orientation=(1, -1))

    for field in FIELDS_FULL
        cellular = DI.encode(data, spec; degree=0, field=field)
        simplicial = DI.encode(tree, tree_spec; degree=0, field=field)
        for encoded in (cellular, simplicial)
            module_ = _enc_module(encoded)
            labels = [EC.locate(encoded.pi, [r, Float64(k)]) for (r, k) in queries]
            @test all(!=(0), labels)
            @test [MD.dim_at(module_, label) for label in labels] == length.(components)
            for source in eachindex(queries), target in eachindex(queries)
                rs, ks = queries[source]
                rt, kt = queries[target]
                rs <= rt && ks >= kt || continue
                @test FF.leq(encoded.P, labels[source], labels[target])
                expected = interval_inclusion_rank(components[source], components[target])
                actual = FL.rank(field, MD.structure_map(module_;
                    source=labels[source], target=labels[target]))
                @test actual == expected
            end
            @test MD.check_module(module_; throw=true).valid
        end
        # Every interval-cover component is contractible. This also checks that
        # the genuine 2-cells kill cycles introduced by the tiling's graph.
        h1 = DI.encode(data, spec; degree=1, field=field)
        @test all(iszero, _enc_dims(h1))
    end
end

@testset "A09: rhomboid geometry, cellular boundaries and public contracts" begin
    function rhomboid_betti(G, radius, depth, field)
        counts = DT.cell_counts(G)
        offsets = cumsum([0; counts])
        active = [[j for j in 1:counts[s]
                   if G.grades[offsets[s] + j][1] <= radius &&
                      G.grades[offsets[s] + j][2] >= depth] for s in eachindex(counts)]
        K = CM.coeff_type(field)
        ranks = [0; [FL.rank(field, K[CM.coerce(field,B[i,j]) for i in active[s], j in active[s+1]])
                     for (s,B) in enumerate(G.boundaries)]; 0]
        return [length(active[s]) - ranks[s] - ranks[s+1] for s in eachindex(counts)]
    end
    f = DI.RhomboidFiltration()
    spec = DI._filtration_spec(f)
    @test spec.kind === :rhomboid
    @test spec.params.orientation == (1,-1)
    @test DI.check_filtration_spec(spec; throw=true).valid
    @test DI.to_filtration(OPT.FiltrationSpec(kind=:rhomboid, max_dim=nothing)) isa DI.RhomboidFiltration
    @test occursin("Rhomboid", sprint(show, f))
    @test get(DI.filtration_parameters(f), :max_dim, nothing) === nothing

    @testset "Exact cubes in arbitrary affine dimensions" begin
        for d in 0:4
            points = zeros(Int, d+1, max(d,1))
            for j in 1:d
                points[j+1,j] = 2
            end
            data = DT.PointCloud(points)
            result = DI.build_graded_complex(data, f)
            G = DI.graded_complex(result)
            @test DT.cell_counts(G) == [binomial(d+1,q)*2^(d+1-q) for q in 0:d+1]
            @test DI.check_graded_complex_build_result(result; throw=true).valid
            @test DI.grade_orientation(result) == (1,-1)
            for q in 2:length(G.boundaries)
                @test iszero(G.boundaries[q-1]*G.boundaries[q])
            end
            for field in FIELDS_FULL
                @test rhomboid_betti(G, 0.0, 1, field) == [d+1; zeros(Int,d+1)]
                @test rhomboid_betti(G, 0.0, 0, field) == [1; zeros(Int,d+1)]
                @test rhomboid_betti(G, 10.0, d+1, field) == [1; zeros(Int,d+1)]
            end
        end
        pair = DT.PointCloud([0 0 0; 2 0 0])
        G = DI.graded_complex(DI.build_graded_complex(pair, f))
        @test DT.cell_counts(G) == [4,4,1]
        @test G.grades[1:4] == [(0.,0.),(0.,1.),(0.,1.),(1.,2.)]
        @test only(G.grades[end:end]) == (1.,0.)
        for field in FIELDS_FULL
            @test rhomboid_betti(G, 0.5, 1, field) == [2,0,0]
            @test rhomboid_betti(G, 1.0, 1, field) == [1,0,0]
            @test rhomboid_betti(G, 0.5, 2, field) == [0,0,0]
            @test rhomboid_betti(G, 1.0, 2, field) == [1,0,0]
        end
        # Exact-coordinate predicates do not conflate points rounded together
        # by Float64. Their small positive separation remains representable.
        epsilon = QQ(1, big(2)^60)
        exact = DT.PointCloud(reshape(QQ[1,1+epsilon], :, 1))
        precise = DI.graded_complex(DI.build_graded_complex(exact, f))
        @test maximum(first, precise.grades) == Float64(epsilon/2)
        @test DI._rhomboid_affine_dimension(DI._rhomboid_points(exact)) == 1
        @test DI._rhomboid_radius(QQ(1,big(2)^2200)) == QQ(1,big(2)^1100)
        @test DI._rhomboid_radius(QQ(big(2)^2200)) == QQ(big(2)^1100)
    end

    @testset "Mask backends and high-site exact sphere predicates" begin
        X = reshape(QQ[0,2,5,9], :, 1)
        word_cells, word_radii = DI._rhomboid_geometry(X, 1, spec, UInt64)
        big_cells, big_radii = DI._rhomboid_geometry(X, 1, spec, BigInt)
        word_complex = DI._rhomboid_cellular_complex(word_cells, word_radii, 4)
        big_complex = DI._rhomboid_cellular_complex(big_cells, big_radii, 4)
        @test word_complex.grades == big_complex.grades
        @test word_complex.boundaries == big_complex.boundaries
        @test DT.cell_counts(word_complex) == DT.cell_counts(big_complex)
        for cell in keys(word_radii)
            converted = DI._RhomboidCell(BigInt(cell.inside), BigInt(cell.on))
            @test word_radii[cell] == big_radii[converted]
        end
        for (count_sites, mask_type) in ((64,UInt64),(65,BigInt))
            line = reshape(QQ.(0:count_sites-1), :, 1)
            support = one(mask_type) | (one(mask_type) << (count_sites-1))
            sphere = DI._rhomboid_sphere(line, support)
            @test sphere.radius2 == QQ((count_sites-1)^2,4)
            @test sphere.on == support
            @test sphere.inside == (one(mask_type) << (count_sites-1)) - 2
            @test DI._rhomboid_indices(sphere.on,count_sites) == [1,count_sites]
            @test DI._rhomboid_indices(sphere.inside,count_sites) == collect(2:count_sites-1)
        end
    end

    @testset "Cellular dataset serialization preserves cubical incidence" begin
        G = DI.graded_complex(DI.build_graded_complex(DT.PointCloud([0 0;4 0;2 3]),f))
        mktemp() do path, io
            close(io)
            SER.save_dataset_json(path,G)
            for validation in (:strict,:trusted)
                restored = SER.load_dataset_json(path; validation)
                @test restored isa DT.GradedComplex
                @test DT.cell_counts(restored) == [8,12,6,1]
                @test restored.grades == G.grades
                @test restored.boundaries == G.boundaries
                for field in FIELDS_FULL
                    @test rhomboid_betti(restored,2.05,1,field) == [1,1,0,0]
                    @test rhomboid_betti(restored,2.2,1,field) == [1,0,0,0]
                end
            end
        end
    end

    @testset "Planar multicover topology and genuine inclusion maps" begin
        # Pairwise disks meet before their common triple intersection. At
        # radius 2.05 the union has a hole; the double cover has three lenses.
        data = DT.PointCloud([0 0; 4 0; 2 3])
        G = DI.graded_complex(DI.build_graded_complex(data, f))
        ST = DI.encode(data, f; stage=:simplex_tree)
        triangulated = DI._graded_complex_from_simplex_tree(ST)
        @test DT.cell_counts(G) == [8,12,6,1]
        @test DT.cell_counts(triangulated) == [8,19,18,6]
        radii = sort!(unique!([first.(G.grades); [0.0, 2.0, 2.05, 2.1, 2.2]]))
        depths = [0.0,1.0,2.0,3.0]
        for field in FIELDS_FULL
            for model in (G, triangulated)
                @test rhomboid_betti(model, 2.05, 1, field) == [1,1,0,0]
                @test rhomboid_betti(model, 2.05, 2, field) == [3,0,0,0]
                @test rhomboid_betti(model, 2.05, 3, field) == [0,0,0,0]
                for depth in 0:3
                    @test rhomboid_betti(model, 2.2, depth, field) == [1,0,0,0]
                end
            end
            enc = DI.encode(data, OPT.FiltrationSpec(kind=:rhomboid, axes=(radii,sort!(-depths))); degree=1, field)
            labels = [EC.locate(enc.pi, [r,1.0]) for r in (2.05,2.1,2.2)]
            M = _enc_module(enc)
            @test [MD.dim_at(M,i) for i in labels] == [1,1,0]
            @test FL.rank(field, MD.structure_map(M; source=labels[1], target=labels[2])) == 1
            @test FL.rank(field, MD.structure_map(M; source=labels[1], target=labels[3])) == 0
        end
        estimate = DI.estimate_ingestion(data, f)
        @test DI.cell_counts_by_dim(estimate) == BigInt[8,12,6,1]
        @test DI.describe(DI.plan_ingestion(data, f)).route_hint === :graded_complex_only
        truncated = DI.graded_complex(DI.build_graded_complex(data, DI.RhomboidFiltration(max_dim=1)))
        @test truncated.grades == G.grades[1:20]
        @test truncated.boundaries == G.boundaries[1:1]
        cut = DI.graded_complex(DI.build_graded_complex(data, DI.RhomboidFiltration(radius=2.05)))
        @test all(g -> g[1] <= 2.05, cut.grades)
        for field in FIELDS_FULL, k in 0:3
            @test rhomboid_betti(cut, 2.05, k, field) == rhomboid_betti(G, 2.05, k, field)
        end
    end

    @testset "Strict geometry, budgets and coordinate-cache lifecycle" begin
        data = DT.PointCloud([0.0; 2.0;;])
        for p in ((; max_dim=-1), (; radius=-1), (; radius=Inf), (; radius=true),
                  (; vertex_values=[0.,1.]), (; knn=1), (; orientation=(1,1)))
            invalid = OPT.FiltrationSpec(; kind=:rhomboid, p...)
            @test !DI.check_filtration_spec(invalid; throw=false).valid
            @test_throws ArgumentError DI.to_filtration(invalid)
            @test_throws ArgumentError DI.plan_ingestion(data,invalid)
        end
        for option in (OPT.ConstructionOptions(sparsify=:knn), OPT.ConstructionOptions(collapse=:dominated_edges))
            @test_throws ArgumentError DI.RhomboidFiltration(construction=option)
        end
        @test_throws MethodError DI.RhomboidFiltration(vertex_values=[0.,1.])
        @test_throws ArgumentError DI.build_graded_complex(DT.GraphData(2,[(1,2)]),f)
        for points in ([0.0; 0.0;;], [0.0; Inf;;], [0 0;1 0;1 1;0 1])
            @test_throws ArgumentError DI.build_graded_complex(DT.PointCloud(points),f)
        end
        # A radius cutoff or skeleton request must not hide cospherical input.
        square = DT.PointCloud([0 0;1 0;1 1;0 1])
        @test_throws ArgumentError DI.build_graded_complex(square, DI.RhomboidFiltration(max_dim=0,radius=0))
        for budget in ((8,nothing,nothing),(nothing,3,nothing),(nothing,nothing,1))
            limited = DI.RhomboidFiltration(construction=OPT.ConstructionOptions(; budget))
            @test_throws ArgumentError DI.build_graded_complex(data,limited)
        end
        cache = CM.SessionCache()
        f0 = OPT.FiltrationSpec(kind=:rhomboid,axes=([0.,1.,2.],[-2.,-1.,0.]))
        before = DI.encode(data,f0; degree=0,cache)
        repeated = DI.encode(data,f0; degree=0,cache)
        @test before.P === repeated.P
        @test _enc_dims(before) == _enc_dims(repeated)
        DT.point_matrix(data)[2,1] = 4.0
        after = DI.encode(data,f0; degree=0,cache)
        @test _enc_dims(before) != _enc_dims(after)
        @test _enc_dims(after) == _enc_dims(DI.encode(data,f0; degree=0))
        CM._clear_session_cache!(cache)
        @test _enc_dims(after) == _enc_dims(DI.encode(data,f0; degree=0,cache))
    end
end


@testset "A12 lazy result inspection and explicit computation lifecycle" begin
    # The three edges form a cycle at (1,0), filled by the triangle at (2,1).
    # Thus H1 has dimensions [0,1,1] on y=0 and [0,1,0] on y=1.
    boundary1 = sparse([-1 -1 0; 1 0 -1; 0 1 1])
    boundary2 = sparse(reshape([1,-1,1], 3, 1))
    graded = DT.GradedComplex([Int[1,2,3], Int[1,2,3], Int[1]],
        [boundary1,boundary2],
        [(0.,0.),(0.,0.),(0.,0.),(1.,0.),(1.,0.),(1.,0.),(2.,1.)])
    spec = OPT.FiltrationSpec(kind=:graded, axes=([0.,1.,2.],[0.,1.]))
    cache_state(L) = (terms=map(!isnothing, L.terms), diffs=map(!isnothing, L.diffs),
        active=map(!isnothing, L.active_by_dim), positions=map(!isnothing, L.pos_by_dim),
        boundaries=map(!isnothing, L.boundaries_field), vertices=L.vertex_idxs !== nothing)
    with_fields(FIELDS_FULL) do field
        enc = DI.encode(graded, spec; degree=1, field=field, stage=:encoding_result)
        lazy = enc.M
        @test lazy isa DI._LazyEncodedModule
        @test lazy.cached_module === nothing && lazy.dims === nothing
        before = cache_state(lazy.lazy)
        @test !before.vertices
        for obj in (enc, lazy)
            @test !isempty(sprint(show, obj))
            @test !isempty(sprint(show, MIME"text/plain"(), obj))
            @test !DI.describe(obj).materialized
            @test DI.describe(obj).module_dims === nothing
        end
        @test occursin("not computed", sprint(show, MIME"text/plain"(), enc))
        @test !RES.result_summary(enc).materialized
        @test RES.encoding_poset(enc) === enc.P
        @test RES.encoding_map(enc) === enc.pi
        @test EC.encoding_axes(enc) == spec.params[:axes]
        @test EC.encoding_axes(EC.compile_encoding(enc)) == spec.params[:axes]
        @test RES.provenance(enc).degree == 1
        @test enc.pi.reps === nothing
        session_enc = RES._encoding_with_session_cache(enc, CM.SessionCache())
        @test session_enc.M === lazy
        @test session_enc.pi.reps === nothing
        supplied_axes = EC.encoding_axes(enc)
        supplied_reps = [(Float64(i),Float64(j)) for j in 0:1 for i in 0:2]
        supplied_map = EC.compile_encoding(enc.P, enc.pi.pi; axes=supplied_axes, reps=supplied_reps)
        supplied = RES.EncodingResult(enc.P, lazy, supplied_map)
        rewrapped = RES._encoding_with_session_cache(supplied, CM.SessionCache())
        @test EC.encoding_axes(rewrapped) === supplied_axes
        @test EC.encoding_representatives(rewrapped) === supplied_reps
        @test cache_state(lazy.lazy) == before
        @test lazy.cached_module === nothing && lazy.dims === nothing

        expected = [0,1,1,0,1,0]
        dims = TamerOp.dimensions(enc)
        @test dims == expected
        @test TamerOp.dimensions(enc) === dims
        @test lazy.cached_module === nothing
        @test all(isnothing, lazy.lazy.terms) && all(isnothing, lazy.lazy.diffs)
        after_dims = cache_state(lazy.lazy)
        @test RES.result_summary(enc).module_dims === dims
        @test !RES.result_summary(enc).materialized
        sprint(show, MIME"text/plain"(), enc)
        sprint(show, MIME"text/plain"(), lazy)
        @test cache_state(lazy.lazy) == after_dims

        module_ = RES.encoding_module(enc)
        @test module_.dims == expected
        @test TamerOp.Workflow.pmodule(enc) === module_
        @test TamerOp.dimensions(enc) === module_.dims
        @test RES.result_summary(enc).materialized
        @test DI.describe(lazy).materialized
        labels = [EC.locate(enc.pi, x) for x in ((1.,0.),(2.,0.),(1.,1.),(2.,1.))]
        for (s,t,rank) in ((1,2,1),(1,3,1),(2,4,0),(3,4,0))
            @test FL.rank(field, MD.structure_map(module_; source=labels[s], target=labels[t])) == rank
        end

        # A direct module request also fills the dimension cache; a subsequent
        # dimension query reuses that result instead of doing another reduction.
        direct = DI.encode(graded, spec; degree=1, field=field, stage=:encoding_result)
        full = RES.encoding_module(direct)
        @test TamerOp.dimensions(direct) === full.dims
        @test full.dims == expected

        complex = DI.encode(graded, spec; field=field, stage=:encoded_complex)
        L = RES.encoding_complex(complex)
        before_complex = cache_state(L)
        @test !RES.result_summary(complex).materialized
        @test DI.describe(L).cell_counts == (3,3,1)
        @test DI.describe(L).degree_range == -2:0
        @test MC.module_complex_summary(L) == DI.describe(L)
        for obj in (complex, L)
            @test !isempty(sprint(show, obj))
            @test !isempty(sprint(show, MIME"text/plain"(), obj))
        end
        @test EC.encoding_axes(complex) == spec.params[:axes]
        @test cache_state(L) == before_complex
        @test complex.pi.reps === nothing
        vertices = DI._lazy_term(L, 0)
        @test vertices.dims == fill(3, 6)
        partial = cache_state(L)
        @test DI.describe(L).materialized_terms == 1
        @test !RES.result_summary(complex).materialized
        sprint(show, MIME"text/plain"(), complex)
        sprint(show, MIME"text/plain"(), L)
        @test cache_state(L) == partial
        C = RES._materialize_complex(L)
        @test RES.result_summary(complex).materialized
        @test DI.describe(L).materialized_differentials == 2
        @test MC.component(C, 0) === vertices
        @test MC.component(C, -1).dims == [0,3,3,0,3,3]
        @test MC.component(C, -2).dims == [0,0,0,0,0,1]
    end
    # Large cached dimension vectors must not become enormous notebook output.
    # A single point has zero H1 at every vertex of this 40,000-region grid.
    point = DT.GradedComplex([Int[1]], SparseMatrixCSC{Int,Int}[], [(0.,0.)])
    wide = DI.encode(point, OPT.FiltrationSpec(kind=:graded,
        axes=(collect(0.:199.), collect(0.:199.))); degree=1, field=FIELD_QQ)
    @test TamerOp.dimensions(wide) == zeros(Int, 40_000)
    @test wide.M.cached_module === nothing
    @test wide.pi.reps === nothing
    @test length(sprint(show, MIME"text/plain"(), wide)) < 3000
    @test length(sprint(show, MIME"text/plain"(), wide.M)) < 1000
    @test wide.M.cached_module === nothing
end

@testset "A75 ingestion composition and cache content oracles" begin
    if !isdefined(@__MODULE__, :_a72_rref)
        include(joinpath(@__DIR__, "helpers", "geometric_persistence_oracles.jl"))
    end
    # Two loops acquire the relation e1 + 2e2 at (1,1). The map from the
    # one-loop stalk to the quotient is multiplication by -2: rank alone
    # cannot distinguish this from an incorrect coefficient or sign.
    G = DT.GradedComplex([Int[1], Int[1,2], Int[1]],
        [spzeros(Int,1,2), sparse([1,2],[1,1],[1,2],2,1)],
        [(0.,0.), (0.,0.), (1.,0.), (1.,1.)])
    spec = OPT.FiltrationSpec(kind=:graded, axes=([0.,1.],[0.,1.]))
    queries = [(0.,0.), (1.,0.), (0.,1.), (1.,1.)]
    active = [_a72_source_active(G, p, (1,1)) for p in queries]
    old_lazy = DI._ENCODING_RESULT_LAZY_MODULE[]
    try
        for field in (FIELDS_FULL..., CM.RealField(Float64; atol=1e-11, rtol=1e-10))
            K = CM.coeff_type(field)
            coerce_matrix(A) = K[CM.coerce(field,x) for x in A]
            reference = [_a72_reference_homology(G, a, 1, field) for a in active]
            # This identity is independent of the oracle's quotient basis.
            @test _a72_equal(_a72_coordinates(reference[4], coerce_matrix(reshape([1,0],2,1))),
                  CM.coerce(field,-2) * _a72_coordinates(reference[4], coerce_matrix(reshape([0,1],2,1))))
            for cache in (nothing, CM.SessionCache())
                C = DI.encode(G, spec; stage=:cochain, field, cache)
                H = MC.cohomology_module_data(C, -1)
                for lazy in (true, false)
                    DI._ENCODING_RESULT_LAZY_MODULE[] = lazy
                    enc = DI.encode(G, spec; degree=1, field, cache)
                    p = RES.provenance(enc)
                    @test p.field == field
                    @test p.degree == 1 && p.degree_convention === :homological
                    @test p.orientation == (1,1)
                    @test p.window.lower == (0.,0.) && p.window.upper == (1.,1.)
                    @test p.discretization.grade_placement === :critical_grades
                    @test DI.module_dims(enc.M) == [1,2,1,1]
                    M = RES.encoding_module(enc)
                    labels = [EC.locate(enc.pi, q) for q in queries]
                    comparisons = [_a72_coordinates(reference[i], _a72_lift_homology(H,u))
                                   for (i,u) in enumerate(labels)]
                    for (i,u) in enumerate(labels)
                        @test length(last(_a72_rref(comparisons[i]))) == M.dims[u]
                        @test M.field == H.H.field == field
                        for (j,v) in enumerate(labels)
                            all(queries[i][a] <= queries[j][a] for a in 1:2) || continue
                            inclusion = zeros(K, length(active[j][2]), length(active[i][2]))
                            for (col,cell) in enumerate(active[i][2])
                                inclusion[findfirst(==(cell),active[j][2]),col] = one(K)
                            end
                            expected = _a72_coordinates(reference[j], inclusion * reference[i].basis)
                            actual = MD.structure_map(M; source=u, target=v)
                            @test _a72_equal(actual, MD.structure_map(H.H; source=u, target=v))
                            @test _a72_equal(comparisons[j]*actual, expected*comparisons[i])
                        end
                    end
                    again = DI.encode(G, spec; degree=1, field, cache, stage=:module)
                    @test all(_a72_equal(MD.structure_map(M; source=u,target=v),
                        MD.structure_map(again; source=u,target=v))
                        for u in labels, v in labels if FF.leq(M.Q,u,v))
                    if cache !== nothing
                        CM._clear_session_cache!(cache)
                        cleared = DI.encode(G,spec; degree=1,field,cache,stage=:module)
                        @test cleared.dims == M.dims
                        @test all(_a72_equal(MD.structure_map(M;source=u,target=v),
                            MD.structure_map(cleared;source=u,target=v))
                            for u in labels,v in labels if FF.leq(M.Q,u,v))
                    end
                end
            end
        end
    finally
        DI._ENCODING_RESULT_LAZY_MODULE[] = old_lazy
    end

    @testset "Mutable input cannot reuse a different boundary or birth" begin
        # The grid and all dimensions stay unchanged after this boundary edit;
        # only the persistence map changes. An object-identity cache misses it.
        session = CM.SessionCache()
        first_module = DI.encode(G,spec;degree=1,stage=:module,cache=session)
        first_map = copy(MD.structure_map(first_module;source=1,target=4))
        pending = DI.encode(G,spec;degree=1,cache=nothing)
        @test pending.M isa DI._LazyEncodedModule
        @test pending.M.cached_module === nothing
        G.boundaries[2][2,1] = 3
        @test MD.structure_map(RES.encoding_module(pending);source=1,target=4) == first_map
        changed = DI.encode(G,spec;degree=1,stage=:module,cache=session)
        fresh = DI.encode(G,spec;degree=1,stage=:module,cache=nothing)
        @test changed !== first_module
        @test changed.dims == first_module.dims
        @test MD.structure_map(changed;source=1,target=4) ==
              MD.structure_map(fresh;source=1,target=4)
        @test MD.structure_map(changed;source=1,target=4) != first_map
        @test DI.encode(G,spec;degree=1,stage=:module,cache=session) === changed
        G.grades[3] = (0.,0.)
        new_birth = DI.encode(G,spec;degree=1,stage=:module,cache=session)
        @test new_birth.dims == [2,2,2,1]
        @test first_module.dims == [1,2,1,1]
        axes = ([0.,1.],[0.,1.])
        windowed = DI.encode(G,OPT.FiltrationSpec(kind=:graded,axes=axes);degree=1,cache=nothing)
        label = EC.locate(windowed.pi,(1.,0.))
        axes[1][2] = 2.
        @test EC.locate(windowed.pi,(1.,0.)) == label
        @test RES.provenance(windowed).window.upper == (1.,1.)
        if Threads.nthreads() > 1
            copies = fetch.([Threads.@spawn DI.encode(G,spec;degree=1,stage=:module,cache=session)
                             for _ in 1:8])
            @test all(M -> M === new_birth, copies)
            @test all(M -> MD.structure_map(M;source=1,target=4) ==
                           MD.structure_map(new_birth;source=1,target=4), copies)
        end
    end
end

@testset "A13 ingestion mathematical provenance" begin
    # Two vertices merge at grade 1. Exact critical-grid homology has dimensions
    # [2,1,1]; floor-snapping the edge to 0 changes them to [1,1]. Sampling the
    # original module on [0,2] would instead produce [2,1].
    boundary = sparse([1,2], [1,1], [-1,1], 2, 1)
    graded = DT.GradedComplex([Int[1,2], Int[1]], [boundary], [(0.,), (0.,), (1.,)])
    with_fields(FIELDS_FULL) do field
        fine_spec = OPT.FiltrationSpec(kind=:graded, axes=([0.,1.,2.],))
        fine = TamerOp.encode(graded, fine_spec; degree=0, field=field)
        p = RES.provenance(fine)
        @test p.category === :finite_poset_representations
        @test p.field == field
        @test p.base_poset === fine.P
        @test p.degree == 0
        @test p.degree_convention === :homological
        @test p.orientation == (1,)
        @test p.window.lower == (0.,) && p.window.upper == (2.,)
        @test p.window.coordinates === :oriented
        @test p.window.upper_extension === :constant
        @test p.discretization.grade_placement === :critical_grades
        @test p.discretization.sampling === false
        @test fine.M.cached_module === nothing
        @test fine.M.dims === nothing
        @test DI.describe(fine).provenance.degree == 0
        @test fine.M.cached_module === nothing
        @test fine.M.dims === nothing
        @test _enc_dims(fine) == [2,1,1]

        coarse_spec = OPT.FiltrationSpec(kind=:graded, axes=([0.,2.],))
        coarse = TamerOp.encode(graded, coarse_spec; field=field)
        cp = RES.provenance(coarse)
        @test cp.discretization.grade_placement === :floor_snapped
        @test cp.discretization.sampling === false
        @test cp.reconstruction === :floor_snapped_graded_complex
        @test _enc_dims(coarse) == [1,1]

        dims = TamerOp.encode(graded, fine_spec; degree=0, field=field, stage=:cohomology_dims)
        complex = TamerOp.encode(graded, fine_spec; field=field, stage=:encoded_complex)
        @test dims.dims == [2,1,1]
        @test RES.provenance(dims).field == field
        @test RES.provenance(dims).degree == 0
        @test RES.provenance(complex).field == field
        @test RES.provenance(complex).degree === nothing
        @test RES.provenance(complex).degree_range == -1:0
        @test RES.provenance(complex).degree_convention === :cohomological
        @test RES.provenance(complex).discretization == p.discretization

        graph = DT.GraphData(2, [(1,2)])
        graph_result = TamerOp.encode(graph, OPT.FiltrationSpec(kind=:graph_lower_star,
            vertex_values=[0.,1.]); field=field)
        @test RES.provenance(graph_result).construction.effective === :graph_lower_star
        @test _enc_dims(graph_result) == [1,1]
        image = DT.ImageNd(reshape([0.,1.], 2, 1))
        image_result = TamerOp.encode(image, DI.ImageLowerStarFiltration(); field=field)
        @test RES.provenance(image_result).construction.effective === :lower_star
        @test RES.provenance(image_result).approximation.grade_arithmetic === :float64
        @test _enc_dims(image_result) == [1,1]
    end

    implicit_axes = TamerOp.encode(graded, OPT.FiltrationSpec(kind=:graded, axes=nothing))
    @test RES.provenance(implicit_axes).discretization.axes_source === :computed_grades
    @test RES.provenance(implicit_axes).discretization.grade_placement === :critical_grades
    @test _enc_dims(implicit_axes) == [2,1]

    # Cellular chains of RP2: d_2=2, d_1=0. Reinterpreting the already computed
    # rational H_1 module gives zero over F2, whereas recomputing H_1 gives F2.
    torsion = DT.GradedComplex([Int[1], Int[1], Int[1]],
        [spzeros(Int,1,1), sparse([1],[1],[2],1,1)], [(0.,),(0.,),(0.,)])
    rational = TamerOp.encode(torsion, DI.GradedFiltration(); degree=1, field=FIELD_QQ)
    @test rational.M isa DI._LazyEncodedModule
    @test rational.M.cached_module === nothing
    interpreted = CM.change_field(rational, FIELD_F2)
    recomputed = TamerOp.encode(torsion, DI.GradedFiltration(); degree=1, field=FIELD_F2)
    @test interpreted.M isa MD.PModule
    @test interpreted.M.field == FIELD_F2
    @test _enc_dims(interpreted) == [0]
    @test _enc_dims(recomputed) == [1]
    ip = RES.provenance(interpreted)
    @test ip.field == FIELD_F2
    @test ip.degree === nothing
    @test ip.reconstruction === :stored_matrix_reinterpretation
    @test ip.coefficient_change.semantics === :reinterpret_stored_module_matrices
    @test ip.source.field == FIELD_QQ
    @test ip.source.degree == 1
    @test interpreted.H === nothing && interpreted.presentation === nothing

    # Rounded grades and floor placement are separate operations.
    shifted = DT.GradedComplex([Int[1,2], Int[1]], [boundary], [(0.,), (0.,), (0.76,)])
    rounded = TamerOp.encode(shifted, OPT.FiltrationSpec(kind=:graded, eps=0.5))
    rp = RES.provenance(rounded)
    @test rp.discretization.quantization === :nearest_multiple
    @test rp.discretization.eps == 0.5
    @test rp.discretization.grade_placement === :critical_grades
    @test EC.axes_from_encoding(rounded.pi) == ([0.,1.],)
    @test _enc_dims(rounded) == [2,1]

    # Reversed axes are recorded in oriented coordinates, never mislabeled as
    # ordinary ascending physical coordinates.
    graph = DT.GraphData(2, [(1,2)])
    reversed = TamerOp.encode(graph, DI.GraphCoreFiltration())
    @test RES.provenance(reversed).orientation == (1,-1)
    @test RES.provenance(reversed).window.coordinates === :oriented
end

@testset "A13 explicit Delaunay substitution at every stage" begin
    points = DT.PointCloud([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [0.,0.,1.]])
    stages = (:simplex_tree, :graded_complex, :cochain, :encoded_complex,
              :cohomology_dims, :module, :fringe, :encoding_result)
    for kind in (:alpha, :delaunay_lower_star)
        parameters = kind === :alpha ? NamedTuple() : (vertex_values=[0.,1.,2.,3.],)
        default_spec = OPT.FiltrationSpec(; kind, max_dim=1, parameters...)
        explicit_spec = OPT.FiltrationSpec(; kind, max_dim=1, highdim_policy=:rips, parameters...)
        typed_default = DI.to_filtration(default_spec)
        @test DI.filtration_parameters(typed_default).highdim_policy === :error
        @test_throws ArgumentError DI.plan_ingestion(points, default_spec)
        @test_throws ArgumentError DI.estimate_ingestion(points, default_spec)
        @test_throws ArgumentError DI.estimate_ingestion(points, default_spec; strict=true)
        for filtration in (default_spec, typed_default), stage in stages
            @test_throws ArgumentError TamerOp.encode(points, filtration; stage=stage)
        end
        @test_throws ArgumentError DI.to_filtration(OPT.FiltrationSpec(; kind, highdim_policy=:invalid, parameters...))
        @test !DI.check_filtration_spec(OPT.FiltrationSpec(; kind, highdim_policy=:invalid, parameters...)).valid
        estimate = DI.estimate_ingestion(points, explicit_spec)
        @test any(occursin("substitutes a different Rips", warning) for warning in DI.estimate_warnings(estimate))
        for filtration in (explicit_spec, DI.to_filtration(explicit_spec))
            result = TamerOp.encode(points, filtration)
            p = RES.provenance(result)
            @test p.construction.requested === kind
            @test p.construction.effective === (kind === :alpha ? :rips : :function_rips)
            @test p.construction.projection === (kind === :alpha ? :none : :function_coordinate)
            @test p.construction.substitution === :explicit_highdim_rips
            @test p.construction.grade_scale === (kind === :alpha ? :diameter : :filtration_values)
            @test p.approximation.construction_substitution === :explicit_highdim_rips
            tree = TamerOp.encode(points, filtration; stage=:simplex_tree)
            @test DT.cell_counts(tree) == [4,6]
            if kind === :alpha
                rips = TamerOp.encode(points, DI.RipsFiltration(max_dim=1); stage=:simplex_tree)
                @test _canon_simplex_tree(tree) == _canon_simplex_tree(rips)
            end
            plan = DI.plan_ingestion(points, filtration)
            @test DI.describe(plan).construction.effective === p.construction.effective
            @test DI.describe(plan).backend_status === :not_executed
        end
    end
    @test_throws ArgumentError DI.AlphaFiltration(highdim_policy=:invalid)
    @test_throws ArgumentError DI.DelaunayLowerStarFiltration(highdim_policy=:invalid)
end

@testset "A13 actual backend evidence and cache reuse" begin
    points = DT.PointCloud([[0.,0.], [1.,0.], [0.,1.]])
    # A requested fast extension may decline this input. The actual fallback
    # must be recorded, and survive triangulation-cache reuse.
    old_impl = DI._POINTCLOUD_DELAUNAY_2D_IMPL[]
    try
        DI._set_pointcloud_delaunay_2d_impl!((points; max_dim=2) -> nothing)
        spec = OPT.FiltrationSpec(kind=:alpha, max_dim=1, delaunay_backend=:fast)
        for iteration in 1:2
            result = TamerOp.encode(points, spec)
            p = RES.provenance(result)
            @test p.backend.requested.delaunay === :fast
            @test (:delaunay => :naive) in p.backend.effective
            @test !((:delaunay => :fast) in p.backend.effective)
            @test p.construction.grade_scale === :squared_radius
            @test _enc_dims(result) == [3,1,1]
        end
    finally
        DI._set_pointcloud_delaunay_2d_impl!(old_impl)
    end
    sparse_spec = OPT.FiltrationSpec(kind=:rips, max_dim=1, knn=1,
        nn_backend=:bruteforce, construction=OPT.ConstructionOptions(sparsify=:knn))
    sparse_result = TamerOp.encode(points, sparse_spec)
    sp = RES.provenance(sparse_result)
    @test sp.approximation.sparsify === :knn
    @test sp.approximation.neighbor_search === :exact_search
    @test (:knn_graph => :bruteforce) in sp.backend.effective
    @test _enc_dims(sparse_result) == [3,1]

    refs = (DI._POINTCLOUD_KNN_GRAPH_IMPL, DI._POINTCLOUD_KNN_DISTANCES_IMPL,
            DI._POINTCLOUD_RADIUS_GRAPH_IMPL, DI._POINTCLOUD_RADIUS_GRAPH_EDGES_IMPL)
    saved = map(ref -> ref[], refs)
    try
        decline(args...; kwargs...) = nothing
        DI._set_pointcloud_nn_impl!(knn_graph=decline, knn_distances=decline,
                                   radius_graph=decline, radius_graph_edges=decline)
        result = TamerOp.encode(points, DI.GraphCoreFiltration(radius=1., nn_backend=:approx))
        p = RES.provenance(result)
        @test p.backend.requested.neighbors === :approx
        @test (:radius_graph => :bruteforce) in p.backend.effective
        @test p.approximation.neighbor_search === :exact_search
        @test all(pair -> last(pair) !== :approx, p.backend.effective)
    finally
        for (ref, value) in zip(refs, saved)
            ref[] = value
        end
        DI._set_pointcloud_nn_impl!() # invalidate backend-resolution decisions
    end
end

@testset "A64: exact rhomboid radii, physical cutoffs and cellular grades" begin
    # On a line, every interval of consecutive sites has radius half its span.
    # The second adjacent pair is born strictly after the first, although the
    # two radii have the same Float64 representation.
    epsilon = QQ(1, big(2)^70)
    points = DT.PointCloud(reshape(QQ[0, 2, 4 + epsilon], :, 1))
    filtration = DI.RhomboidFiltration()
    G = DI.graded_complex(DI.build_graded_complex(points, filtration))
    radii = sort!(unique(first.(G.grades)))
    expected = QQ[0, 1, 1 + epsilon / 2, 2 + epsilon / 2]
    @test radii == expected
    @test length(radii) == 4
    @test radii[2] < radii[3]
    @test Float64(radii[2]) == Float64(radii[3]) == 1.0
    @test all(r -> r^2 in expected .^ 2, radii)
    @test DT.cell_counts(G) == [7, 9, 3]
    @test iszero(G.boundaries[1] * G.boundaries[2])

    @testset "Default critical encoding and exact batched membership" begin
        query_radii = QQ[0, 1, 1 + epsilon/4, 1 + epsilon/2, 2 + epsilon/2]
        queries = vcat(permutedims(query_radii), ones(QQ, 1, length(query_radii)))
        for field in FIELDS_FULL
            enc = DI.encode(points, filtration; degree=0, field, cache=nothing)
            @test EC.axes_from_encoding(enc.pi)[1] == expected
            @test length(EC.axes_from_encoding(enc.pi)[1]) == 4
            labels = [EC.locate(enc.pi, (radius, QQ(1))) for radius in query_radii]
            @test labels[2] == labels[3]
            @test labels[3] != labels[4]
            for classifier in (enc.pi, enc.pi.pi)
                batched = zeros(Int, length(query_radii))
                EC.locate_many!(batched, classifier, queries)
                @test batched == labels
            end
            @test TamerOp.dimensions(enc)[labels] == [3, 2, 2, 1, 1]
            evidence = RES.provenance(enc)
            @test evidence.approximation.grade_arithmetic === :exact_real_algebraic
            @test evidence.construction.grade_scale === :radius
            @test evidence.discretization.grade_placement === :critical_grades
            @test enc.M.cached_module === nothing
        end
    end

    # `radius` remains a physical-radius cutoff, including for a rational
    # value lying strictly between critical values with equal float displays.
    between = 1 + epsilon / 4
    cut = DI.graded_complex(DI.build_graded_complex(points,
        DI.RhomboidFiltration(radius=between)))
    @test DT.cell_counts(cut) == [5, 5, 1]
    @test maximum(first, cut.grades) == 1
    @test all(g -> g[1] <= between, cut.grades)
    closed = DI.graded_complex(DI.build_graded_complex(points,
        DI.RhomboidFiltration(radius=1 + epsilon / 2)))
    @test DT.cell_counts(closed) == [6, 7, 2]
    @test maximum(first, closed.grades) == 1 + epsilon / 2
    @test iszero(closed.boundaries[1] * closed.boundaries[2])

    ST = DI.encode(points, filtration; stage=:simplex_tree, cache=nothing)
    triangulated = DI._graded_complex_from_simplex_tree(ST)
    @test sort!(unique(first.(triangulated.grades))) == expected
    @test iszero(triangulated.boundaries[1] * triangulated.boundaries[2])
    cut_tree = DI.encode(points, DI.RhomboidFiltration(radius=between);
                         stage=:simplex_tree, cache=nothing)
    @test maximum(first, DI._graded_complex_from_simplex_tree(cut_tree).grades) == 1

    # Irrational radii retain their exact rational square. This fixture's
    # unique pair has distance sqrt(8), hence physical birth radius sqrt(2).
    diagonal = DT.PointCloud([0 0; 2 2])
    diagonal_G = DI.graded_complex(DI.build_graded_complex(diagonal, filtration))
    radius = maximum(first, diagonal_G.grades)
    @test radius^2 == 2
    @test 7//5 < radius < 10//7
    exact_cut = DI.graded_complex(DI.build_graded_complex(diagonal,
        DI.RhomboidFiltration(radius=radius)))
    @test exact_cut.grades == diagonal_G.grades
    @test exact_cut.boundaries == diagonal_G.boundaries
    @test DT.cell_counts(DI.graded_complex(DI.build_graded_complex(diagonal,
        DI.RhomboidFiltration(radius=7//5)))) == [3, 2, 0]
    @test DT.cell_counts(DI.graded_complex(DI.build_graded_complex(diagonal,
        DI.RhomboidFiltration(radius=10//7)))) == [4, 4, 1]

    @testset "Exact continuous H0 events and rank-measure support" begin
        AR = TamerOp.ExactReals.AlgebraicReal
        # Three components merge at the two adjacent-pair radii. This chain
        # complex is written independently of the rhomboid geometry builder.
        a, b = AR(1), AR(1 + epsilon / 2)
        boundary = sparse([-1 0; 1 -1; 0 1])
        line = DT.GradedComplex([[1, 2, 3], [1, 2]], [boundary],
            [(AR(0),), (AR(0),), (AR(0),), (a,), (b,)])
        expected_bars = Dict((AR(0), a) => 1, (AR(0), b) => 1, (AR(0), Inf) => 1)
        for field in FIELDS_FULL
            enc = DI.encode(line, OPT.FiltrationSpec(kind=:graded); degree=0, field, cache=nothing)
            @test enc.M.cached_module === nothing
            for threads in (false, true)
                slices = TamerOp.slice_barcodes(enc; directions=[[1]], offsets=[[0]], threads,
                                               cache=nothing)
                @test only(slices.barcodes) == expected_bars
                @test length(only(slices.barcodes)) == 3
            end
            @test enc.M.cached_module === nothing
            measure = TamerOp.rank_signed_measure(enc; cache=nothing)
            supports = Dict((measure.axes[1][i[1]], measure.axes[2][i[2]]) => w
                            for (i, w) in zip(measure.inds, measure.wts))
            @test supports == expected_bars
            @test length(measure.axes[2]) == 3
            @test measure.axes[2][1] < measure.axes[2][2] < measure.axes[2][3]
            @test enc.M.cached_module === nothing
            @test TamerOp.dimensions(enc) == [3, 2, 1]
            # Supplying sample times switches to the finite-grid H0 backend.
            # Its final endpoint follows the documented last-step extension;
            # neither of the two actual deaths may round into the other.
            for threads in (false, true)
                sampled = TamerOp.slice_barcodes(enc; directions=[[1]], offsets=[[0]],
                    ts=AR[0,a,b], threads, cache=nothing)
                @test only(sampled.barcodes) == Dict((AR(0), a) => 1,
                    (AR(0), b) => 1, (AR(0), 2b-a) => 1)
            end
            @test enc.M.cached_module === nothing
        end
        # Positive direction components are not discarded as numerical zero.
        @test DI._line_time_from_grade((a,), AR[0], AR[epsilon]) == a / epsilon
        @test DI._line_time_from_grade((b,), AR[a], AR[0]) == Inf
        # The ordinary Float64 lane also respects distinct represented event
        # values; the previous tolerance grouping erased this short interval.
        float_bars = DI._h0_line_barcode([0.0, 0.0, 0.0], [(1, 2), (2, 3)],
                                        [1.0, nextfloat(1.0)])
        @test float_bars == Dict((0.0, 1.0) => 1, (0.0, nextfloat(1.0)) => 1,
                                 (0.0, Inf) => 1)
    end

    @testset "Shared ingestion caches distinguish conjugate cutoffs" begin
        AR = TamerOp.ExactReals.AlgebraicReal
        small = sqrt(AR(3)) - sqrt(AR(2))
        large = sqrt(AR(3)) + sqrt(AR(2))
        @test 0 < small < 1 < large
        @test hash(small) == hash(large)
        pair = DT.PointCloud(reshape(QQ[0, 2], :, 1))
        cutoffs = (small, large)
        # The two balls merge at physical radius 1. A smaller construction
        # cutoff leaves two components; a larger one includes their merger.
        for field in FIELDS_FULL, order in ((1, 2), (2, 1))
            session = CM.SessionCache()
            bucket = CM._workflow_encoding_cache(session)
            @test bucket isa CM.EncodingCache
            for index in order
                cutoff_filtration = DI.RhomboidFiltration(radius=cutoffs[index])
                encoded = DI.encode(pair, cutoff_filtration; field, cache=session)
                reference = DI.encode(pair, cutoff_filtration; field, cache=nothing)
                expected_radius_axis = index == 1 ? AR[0, small] : AR[0, 1, large]
                @test EC.axes_from_encoding(encoded.pi)[1] == expected_radius_axis
                @test EC.axes_from_encoding(encoded.pi) == EC.axes_from_encoding(reference.pi)
                @test TamerOp.dimensions(encoded) == TamerOp.dimensions(reference)
                label = EC.locate(encoded.pi, (QQ(index - 1), QQ(1)))
                @test TamerOp.dimensions(encoded)[label] == (index == 1 ? 2 : 1)
            end
            @test !isempty(bucket.geometry)

            # The same explicit grid makes both requests share one encoding
            # poset and lie inside both requested windows. These are explicitly
            # floor-snapped models: the large-cutoff merger moves from radius 1
            # to small, whereas the small-cutoff construction omits it. Their
            # depth-one arrows from radius 0 to small have ranks 2 and 1.
            # Include every oriented depth grade before selecting that row:
            # a singleton axis would also floor depth-zero cell births there.
            common_axes = (AR[0, small], AR[-2, -1, 0])
            requests = map(radius -> OPT.FiltrationSpec(kind=:rhomboid, radius=radius,
                axes=common_axes), cutoffs)
            modules = Dict{Int,Any}()
            for index in order
                module_ = DI.encode(pair, requests[index]; stage=:module, field, cache=session)
                reference = DI.encode(pair, requests[index]; stage=:module, field, cache=nothing)
                expected = index == 1 ? [2, 2] : [2, 1]
                classifier = EC.GridEncodingMap(module_.Q, common_axes; orientation=(1, -1))
                source = EC.locate(classifier, AR[0, 1])
                target = EC.locate(classifier, AR[small, 1])
                @test DI.module_dims(module_) == DI.module_dims(reference)
                @test DI.module_dims(module_)[[source, target]] == expected
                arrow = MD.structure_map(module_; source, target)
                @test arrow == MD.structure_map(reference; source, target)
                @test FL.rank(field, arrow) == expected[2]
                modules[index] = module_
            end
            @test modules[1] !== modules[2]
            @test modules[1].Q === modules[2].Q
            for index in order
                @test DI.encode(pair, requests[index]; stage=:module, field, cache=session) === modules[index]
            end
        end
    end

    # Computational grades are not bounded by Float64's display range.
    for radius in (QQ(1, big(2)^1100), QQ(big(2)^1100))
        pair = DT.PointCloud(reshape(QQ[0, 2radius], :, 1))
        extreme = DI.graded_complex(DI.build_graded_complex(pair, filtration))
        @test maximum(first, extreme.grades) == radius
        @test maximum(first, extreme.grades)^2 == radius^2
        @test DT.cell_counts(extreme) == [4, 4, 1]
    end
end

@testset "A72: independent geometric persistence-module comparisons" begin
    if !isdefined(@__MODULE__, :_a72_rref)
        include(joinpath(@__DIR__, "helpers", "geometric_persistence_oracles.jl"))
    end

    @testset "R3 multicover H2, signed subdivision, and naturality" begin
        # A regular tetrahedron has edge, face and tetrahedron minimum squared
        # radii 2, 8/3 and 3. Adding its center cones off the ordinary Cech
        # complex, but the depth-two subdivision still has H2 between 8/3 and
        # 3. Thus these fixtures see positive higher homology at TWO depths,
        # its exact birth/death, nonzero radius transport, and death under depth
        # maps. The core fixtures below also have nonzero depth transports.
        # Subdivision-Cech's multicover interpretation follows the persistent
        # nerve theorem; see Corbet et al., arXiv:2103.07823, Sections 1.1,
        # 2.2 and 4.4. The chain comparison below is explicitly constructed and
        # tested, rather than inferring a natural isomorphism from equal ranks.
        A = TamerOp.ExactReals.AlgebraicReal
        radii = [A(3//2), sqrt(A(8//3)), A(17//10), sqrt(A(3))]
        depths = [1, 2, 3]
        tetrahedron = QQ[1 1 1; 1 -1 -1; -1 1 -1; -1 -1 1]
        for with_center in (false, true)
            X = with_center ? vcat(tetrahedron, zeros(QQ,1,3)) : tetrahedron
            data = DT.PointCloud(X)
            reference = _a72_subdivision_cech(X)
            @test reference.radii2[1+0b0011] == 2
            @test reference.radii2[1+0b0111] == 8//3
            @test reference.radii2[1+0b1111] == 3
            if with_center
                @test reference.radii2[1+0b10111] == 8//3
                @test reference.radii2[1+0b11111] == 3
            end
            spec = OPT.FiltrationSpec(kind=:rhomboid, axes=(radii, -reverse(depths)))
            G = DI.encode(data, spec; stage=:graded_complex)
            # Production cell labels identify the domain generators only.
            # Reference geometry, simplices, boundaries and elimination are
            # independently constructed above, with rational certificates.
            cells, _ = DI._rhomboid_geometry(X, 3, spec, UInt64)
            comparison = _a72_rhomboid_comparison(cells, reference)
            offsets = cumsum([0; DT.cell_counts(G)])
            for q in 1:3
                @test reference.boundaries[q]*comparison[q+1] == comparison[q]*G.boundaries[q]
            end
            for q in 2:3
                @test iszero(reference.boundaries[q-1]*reference.boundaries[q])
            end
            for slot in eachindex(comparison)
                rows, columns, _ = findnz(comparison[slot])
                for (row, column) in zip(rows, columns)
                    simplex = reference.simplices[slot][row]
                    grade = G.grades[offsets[slot]+column]
                    @test reference.radii2[last(simplex)+1] <= grade[1]^2
                    @test count_ones(first(simplex)) >= grade[2]
                end
            end
            queries = [(r,k) for r in radii for k in depths]
            active_reference = [[[i for (i,s) in enumerate(group)
                                  if reference.radii2[last(s)+1] <= r^2 && count_ones(first(s)) >= k]
                                 for group in reference.simplices] for (r,k) in queries]
            active_source = [_a72_source_active(G, p, (1,-1)) for p in queries]
            for field in FIELDS_FULL, degree in (1,2)
                expected = map(queries) do (r,k)
                    before_birth = r^2 < 8//3
                    after_death = r^2 >= 3
                    after_death && return 0
                    if degree == 2
                        return !before_birth && k == (with_center ? 2 : 1) ? 1 : 0
                    end
                    active_depth = (before_birth ? 1 : 2) + (with_center ? 1 : 0)
                    return k == active_depth ? 3 : 0
                end
                _a72_check_module_comparison(data, spec, G, reference, comparison,
                    queries, active_reference, active_source, degree, expected, field)
            end
        end
    end

    @testset "Function-sublevel offsets and core nerves have natural H1 comparisons" begin
        radii = [1.9, 2.05, 2.1, 2.2]
        triangle = QQ[0 0; 4 0; 2 3]
        for family in (:function_delaunay, :core, :core_delaunay)
            is_function = family === :function_delaunay
            X = is_function ? vcat(triangle, QQ[2 1]) : triangle
            data = DT.PointCloud(X)
            levels = is_function ? [0,1,2] : [1,2,3]
            values = [0,0,0,2]
            orientation = is_function ? (1,1) : (1,-1)
            base = is_function ? DI.FunctionDelaunayFiltration(vertex_values=values, delaunay_backend=:naive) :
                   family === :core ? DI.CoreFiltration(max_dim=2,beta=0.5) :
                   DI.CoreDelaunayFiltration(max_dim=2,beta=0.5,delaunay_backend=:naive)
            params = merge(DI._filtration_spec(base).params,
                           (; axes=(radii, is_function ? levels : -reverse(levels))))
            spec = OPT.FiltrationSpec(; kind=family, params...)
            reference = _a72_cech(X; maxdim=is_function ? 3 : 2)
            @test reference.radii2[(1,2,3)] == 169//36
            tree = DI.encode(data, spec; stage=:simplex_tree)
            G = DI.encode(data, spec; stage=:graded_complex)
            comparison = _a72_simplicial_comparison(tree, reference)
            @test G.boundaries == DI._graded_complex_from_simplex_tree(tree).boundaries
            for q in eachindex(G.boundaries)
                @test reference.boundaries[q]*comparison[q+1] == comparison[q]*G.boundaries[q]
            end
            # For this acute triangle each edge midpoint lies in the common
            # full-cloud Voronoi boundary, and the circumcenter is inside the
            # triangle. Therefore restricted-ball (alpha) and Cech thresholds
            # coincide, independently certifying the CoreDelaunay reference.
            distances2 = [sort([sum((X[i,a]-X[j,a])^2 for a in axes(X,2))
                                  for j in axes(X,1)]) for i in axes(X,1)]
            queries = [(r,k) for r in radii for k in levels]
            active_reference = map(queries) do (r,k)
                r2 = QQ(r)^2
                [[i for (i,s) in enumerate(group) if reference.radii2[s] <= r2 &&
                  (is_function ? maximum(values[v] for v in s) <= k :
                   all(v -> distances2[v][k]/4 <= r2, s))]
                 for group in reference.simplices]
            end
            active_source = [_a72_source_active(G,p,orientation) for p in queries]
            expected = [2 <= r < 13/6 && (!is_function || k < 2) ? 1 : 0 for (r,k) in queries]
            for field in FIELDS_FULL
                _a72_check_module_comparison(data, spec, G, reference, comparison,
                    queries, active_reference, active_source, 1, expected, field)
            end
        end
    end
end

@testset "A64: exact coordinate serialization" begin
    AR = TamerOp.ExactReals.AlgebraicReal
    # Slice intersections can generate higher-degree algebraic numbers. All
    # four real conjugates share a minimal polynomial; its root index must
    # distinguish them without relying on a displayed approximation.
    conjugates = sort!([s*sqrt(AR(2)) + t*sqrt(AR(3)) for s in (-1,1) for t in (-1,1)])
    conjugate_complex = DT.GradedComplex([collect(1:4)], SparseMatrixCSC{Int,Int}[],
                                         [(x,) for x in conjugates])
    mktemp() do path, io
        close(io)
        SER.save_dataset_json(path, conjugate_complex)
        for validation in (:strict, :trusted)
            restored = SER.load_dataset_json(path; validation)
            @test first.(restored.grades) == conjugates
            @test all(diff(first.(restored.grades)) .> 0)
        end
    end
    q = one(QQ) + one(QQ) / big(10)^40
    r1, r2 = AR(1), sqrt(AR(q))
    @test r1 < r2
    @test Float64(r1) == Float64(r2)
    @test r2^2 == q
    cells = [[1, 2], [3]]
    boundaries = [sparse([1, 2], [1, 1], [-1, 1], 2, 1)]
    for (a, b) in ((one(QQ), q), (r1, r2))
        T = typeof(a)
        grades = [(zero(T), zero(T)), (a, zero(T)), (b, zero(T))]
        G = DT.GradedComplex(cells, boundaries, grades)
        GM = DT.MultiCriticalGradedComplex(cells, boundaries,
            [[grades[1]], [grades[2]], [grades[3], (a, one(T))]])
        ST = DI._simplex_tree_multi_from_complex(G)
        for original in (G, GM, ST), mode in (:strict, :trusted)
            mktemp() do path, io
                close(io)
                SER.save_dataset_json(path, original)
                loaded = SER.load_dataset_json(path; validation=mode)
                @test loaded isa typeof(original)
                if loaded isa DT.SimplexTreeMulti
                    @test collect.(DT.simplex_grades(loaded)) == collect.(DT.simplex_grades(original))
                    @test loaded.simplex_offsets == original.simplex_offsets
                    @test loaded.simplex_vertices == original.simplex_vertices
                else
                    @test collect.(DT.cell_grades(loaded)) == collect.(DT.cell_grades(original))
                    @test loaded.boundaries == original.boundaries
                end
            end
        end
    end

    # Source data and exact spec cutoffs survive too; JSON numeric coercion
    # would make these two supplied rational coordinates identical.
    cloud = DT.PointCloud(reshape(QQ[1, q], 2, 1))
    mktemp() do path, io
        close(io)
        original = DT.PointCloud(reshape(Float32[0.1], 1, 1))
        SER.save_dataset_json(path, original)
        @test DT.point_matrix(SER.load_dataset_json(path)) == Float64.(DT.point_matrix(original))
    end
    for mode in (:strict, :trusted)
        mktemp() do path, io
            close(io)
            SER.save_dataset_json(path, cloud)
            loaded = SER.load_dataset_json(path; validation=mode)
            @test DT.point_matrix(loaded) == DT.point_matrix(cloud)
            @test DT.point_matrix(loaded)[1, 1] < DT.point_matrix(loaded)[2, 1]
        end
    end
    image_data = DT.ImageNd(reshape(QQ[1, q], 2, 1))
    graph = DT.GraphData(2, [(1, 2)]; coords=reshape(QQ[1, q], 2, 1), weights=QQ[q], T=QQ)
    embedded = DT.EmbeddedPlanarGraph2D([QQ[0, 0], QQ[q, 1]], [(1, 2)]; bbox=(QQ(0), q, QQ(0), QQ(1)))
    for original in (image_data, graph, embedded)
        mktemp() do path, io
            close(io)
            SER.save_dataset_json(path, original)
            loaded = SER.load_dataset_json(path)
            @test loaded isa typeof(original)
            if original isa DT.ImageNd
                @test loaded.data == original.data
            elseif original isa DT.GraphData
                @test DT.coord_matrix(loaded) == DT.coord_matrix(original)
                @test loaded.weights == original.weights
            else
                @test collect.(loaded.vertices) == collect.(original.vertices)
                @test loaded.bbox == original.bbox
            end
        end
    end
    spec = OPT.FiltrationSpec(kind=:rhomboid, radius=r2,
                              axes=([AR(0), r1, r2], AR[0, 1]))
    mktemp() do path, io
        close(io)
        SER.save_pipeline_json(path, cloud, spec; degree=0)
        for mode in (:strict, :trusted)
            data2, spec2, degree2, _ = SER.load_pipeline_json(path; validation=mode)
            @test DT.point_matrix(data2) == DT.point_matrix(cloud)
            @test spec2.kind == :rhomboid
            @test spec2.params.radius == r2
            @test spec2.params.axes[1] == spec.params.axes[1]
            @test spec2.params.axes[1][2] < spec2.params.axes[1][3]
            @test degree2 == 0
        end
    end

    # An explicit constant diagram supplies a map-level oracle independent of
    # geometric predicates and of coordinate rounding. Both near-colliding
    # radius cells must remain distinguishable after finite-encoding I/O.
    axes = (AR[0, r1, r2], AR[0])
    P = FF.GridPoset(axes)
    pi = EC.GridEncodingMap(P, axes)
    H = FF.one_by_one_fringe(P, FF.principal_upset(P, 1), FF.principal_downset(P, 3), 1; field=CM.QQField())
    M = IR.pmodule_from_fringe(H)
    enc = RES.EncodingResult(P, M, pi; H=H,
        meta=(provenance=(construction=(grade_scale=:radius,), orientation=(1, 1),
                          approximation=(grade_arithmetic=:exact_real_algebraic,)),))
    mktemp() do path, io
        close(io)
        SER.save_encoding_json(path, enc)
        for mode in (:strict, :trusted)
            loaded = SER.load_encoding_json(path; validation=mode)
            @test EC.axes_from_encoding(loaded.pi) == axes
            @test loaded.P.coords == P.coords
            @test EC.locate(loaded.pi, AR[(r1 + r2)/2, 0]) == 2
            @test EC.locate(loaded.pi, AR[r2, 0]) == 3
            @test loaded.M.dims == [1, 1, 1]
            @test MD.map_leq(loaded.M, 1, 3) == ones(QQ, 1, 1)
            p = RES.provenance(loaded)
            @test p.base_poset === loaded.P
            @test p.field == CM.QQField()
            @test p.coordinate_semantics == (grade_scale=:radius, orientation=(1, 1),
                                               grade_arithmetic=:exact_real_algebraic)
            SER.save_encoding_json(path, loaded)
            @test RES.provenance(SER.load_encoding_json(path)).coordinate_semantics == p.coordinate_semantics
        end
    end
end

@testset "A64: invalid exact coordinate payloads" begin
    AR = TamerOp.ExactReals.AlgebraicReal
    G = DT.GradedComplex([[1]], SparseMatrixCSC{Int,Int}[], [(sqrt(AR(2)),)])
    mktemp() do path, io
        close(io)
        SER.save_dataset_json(path, G)
        original = JSON3.read(read(path, String), Dict{String,Any})
        alterations = (
            obj -> (obj["grades"] = [[1.0]]),
            obj -> (obj["exact_grades"]["scalar_kind"] = "approximate"),
            obj -> (obj["exact_grades"]["offsets"] = [2, 2]),
            obj -> (obj["exact_grades"]["algebraic_values"][1]["real_root_index"] = 0),
            obj -> (obj["exact_grades"]["algebraic_values"][1]["polynomial"] = ["1", "0", "1"]),
            obj -> (obj["exact_grades"]["algebraic_values"][1]["polynomial"] = ["-4", "0", "2"]),
            obj -> (obj["exact_grades"]["algebraic_values"][1]["polynomial"] = ["-4", "0", "1"]),
        )
        for alter in alterations
            obj = deepcopy(original)
            alter(obj)
            write(path, JSON3.write(obj))
            for mode in (:strict, :trusted)
                @test_throws ArgumentError SER.load_dataset_json(path; validation=mode)
            end
        end
        Gq = DT.GradedComplex([[1]], SparseMatrixCSC{Int,Int}[], [(one(QQ),)])
        SER.save_dataset_json(path, Gq)
        rational_obj = JSON3.read(read(path, String), Dict{String,Any})
        for value in ("1/0", "2/2", "1/-1", "garbage")
            obj = deepcopy(rational_obj)
            obj["exact_grades"]["rational_values"][1] = value
            write(path, JSON3.write(obj))
            @test_throws ArgumentError SER.load_dataset_json(path)
        end
    end
end

@testset "A66 capped depth windows: independent two-site and triangle maps" begin
    isdefined(@__MODULE__, :_a72_rref) || include(joinpath(@__DIR__, "helpers", "geometric_persistence_oracles.jl"))
    isdefined(@__MODULE__, :_a66_carrier_comparison) || include(joinpath(@__DIR__, "helpers", "rhomboid_extension_oracles.jl"))
    AR = TamerOp.ExactReals.AlgebraicReal
    fixtures = (
        (X=reshape(QQ[0, 2], :, 1), radii=AR[0, 1//2, 1, 3//2],
         windows=((1, 1), (0, 1), (1, 2), (2, 2)), degrees=(0,), maxdim=2),
        (X=QQ[0 0; 2 0; 1 2], radii=AR[0, 1, 9//8, 5//4, 3//2],
         windows=((1, 1), (1, 2), (2, 2), (0, 3)), degrees=(0, 1), maxdim=3),
    )
    for fixture in fixtures
        X, radii = fixture.X, fixture.radii
        n = size(X, 1)
        reference = _a72_subdivision_cech(X; maxdim=fixture.maxdim)
        full_spec = OPT.FiltrationSpec(kind=:rhomboid, backend=:exhaustive, max_dim=fixture.maxdim)
        full_G, full_unions = _a66_native_fixture(X, full_spec)
        full_comparison = _a66_carrier_comparison(full_G.boundaries, full_unions, reference)
        _a66_check_carrier_grades(full_G, reference, full_comparison)
        for window in fixture.windows
            lo, hi = window
            queries = [(r, k) for r in radii for k in lo:hi]
            active_reference = _a66_active_reference(reference, queries)
            active_full = [_a72_source_active(full_G, p, (1, -1)) for p in queries]
            spec = OPT.FiltrationSpec(kind=:rhomboid, backend=:exhaustive,
                                      max_dim=fixture.maxdim, depth_range=window)
            G, unions = _a66_native_fixture(X, spec)
            comparison = _a66_carrier_comparison(G.boundaries, unions, reference)
            _a66_check_carrier_grades(G, reference, comparison)
            active_source = [_a72_source_active(G, p, (1, -1)) for p in queries]
            for field in FIELDS_FULL, degree in fixture.degrees
                expected = map(queries) do (r, k)
                    k == 0 && return degree == 0 ? 1 : 0
                    if n == 2
                        return k == 1 ? (r < 1 ? 2 : 1) : (r < 1 ? 0 : 1)
                    elseif degree == 1
                        return k == 1 && 5//4 <= r^2 < 25//16 ? 1 : 0
                    elseif k == 1
                        return r < 1 ? 3 : r^2 < 5//4 ? 2 : 1
                    elseif k == 2
                        return r < 1 ? 0 : r^2 < 5//4 ? 1 : r < 5//4 ? 3 : 1
                    end
                    return r < 5//4 ? 0 : 1
                end
                complete = _a72_check_module_comparison(DT.PointCloud(X), full_spec, full_G,
                    reference, full_comparison, queries, active_reference, active_full, degree, expected, field)
                capped = _a72_check_module_comparison(DT.PointCloud(X), spec, G,
                    reference, comparison, queries, active_reference, active_source, degree, expected, field)
                _a66_compare_through_reference(complete, capped, queries)
            end
        end
    end
end

@testset "A66 capped tetrahedron: higher homology and signed comparison maps" begin
    isdefined(@__MODULE__, :_a72_rref) || include(joinpath(@__DIR__, "helpers", "geometric_persistence_oracles.jl"))
    isdefined(@__MODULE__, :_a66_carrier_comparison) || include(joinpath(@__DIR__, "helpers", "rhomboid_extension_oracles.jl"))
    AR = TamerOp.ExactReals.AlgebraicReal
    X = QQ[1 1 1; 1 -1 -1; -1 1 -1; -1 -1 1]
    radii = AR[3//2, sqrt(AR(8//3)), 17//10, sqrt(AR(3))]
    reference = _a72_subdivision_cech(X; maxdim=3)
    @test reference.radii2[1+0b0111] == 8//3
    @test reference.radii2[1+0b1111] == 3
    full_spec = OPT.FiltrationSpec(kind=:rhomboid, backend=:exhaustive, max_dim=3)
    full_G, full_unions = _a66_native_fixture(X, full_spec)
    full_comparison = _a66_carrier_comparison(full_G.boundaries, full_unions, reference)
    for window in ((1, 1), (1, 2), (2, 2))
        queries = [(r, k) for r in radii for k in first(window):last(window)]
        active_reference = _a66_active_reference(reference, queries)
        active_full = [_a72_source_active(full_G, p, (1, -1)) for p in queries]
        spec = OPT.FiltrationSpec(kind=:rhomboid, backend=:exhaustive, max_dim=3, depth_range=window)
        G, unions = _a66_native_fixture(X, spec)
        comparison = _a66_carrier_comparison(G.boundaries, unions, reference)
        _a66_check_carrier_grades(G, reference, comparison)
        active_source = [_a72_source_active(G, p, (1, -1)) for p in queries]
        for field in FIELDS_FULL, degree in (1, 2)
            expected = map(queries) do (r, k)
                r^2 >= 3 && return 0
                degree == 2 && return k == 1 && r^2 >= 8//3 ? 1 : 0
                return k == (r^2 < 8//3 ? 1 : 2) ? 3 : 0
            end
            complete = _a72_check_module_comparison(DT.PointCloud(X), full_spec, full_G,
                reference, full_comparison, queries, active_reference, active_full, degree, expected, field)
            capped = _a72_check_module_comparison(DT.PointCloud(X), spec, G,
                reference, comparison, queries, active_reference, active_source, degree, expected, field)
            _a66_compare_through_reference(complete, capped, queries)
        end
    end
end

@testset "A67 exact incremental carrier and radius oracle" begin
    # Exhaustive supports are a genuinely different discovery algorithm. Exact
    # equality here includes all original cells, constrained radii and signed
    # boundaries, so the native module and every inclusion matrix coincide.
    tetrahedron = QQ[1 1 1; 1 -1 -1; -1 1 -1; -1 -1 1]
    clouds = (
        reshape(QQ[7], 1, 1),
        reshape(QQ[0, 1, 3, 6, 10], :, 1),
        QQ[0 0; 2 0; 1 2],
        QQ[0 0; 2 0; 1 2; 3 3],
        QQ[0 0 1; 2 0 1; 1 2 1; 3 3 1],
        tetrahedron,
        vcat(tetrahedron, QQ[3//10 1//5 1//10]),
        vcat(tetrahedron, QQ[3//10 1//5 1//10; 17//5 11//7 13//11]),
    )
    for X in clouds, window in (nothing, (1, 1), (1, 2))
        window !== nothing && last(window) > size(X, 1) && continue
        dimension = DI._rhomboid_affine_dimension(X)
        spec = OPT.FiltrationSpec(kind=:rhomboid, depth_range=window)
        exhaustive, eradii = DI._rhomboid_geometry(X, dimension, spec, UInt64; backend=:exhaustive)
        incremental, iradii = DI._rhomboid_geometry(X, dimension, spec, UInt64; backend=:incremental)
        @test incremental == exhaustive
        @test all(iradii[cell] == eradii[cell] for group in exhaustive for cell in group)
        if window === nothing
            E = DI._rhomboid_cellular_complex(exhaustive, eradii, size(X, 1))
            I = DI._rhomboid_cellular_complex(incremental, iradii, size(X, 1))
        else
            egroups, E = DI._rhomboid_depth_model(exhaustive, eradii, size(X, 1), spec)
            igroups, I = DI._rhomboid_depth_model(incremental, iradii, size(X, 1), spec)
            @test egroups == igroups
        end
        @test E.grades == I.grades
        @test E.boundaries == I.boundaries
        # Verify the actual public construction dispatch reaches this result.
        public_spec = OPT.FiltrationSpec(kind=:rhomboid, backend=:incremental, depth_range=window)
        public = DI.encode(DT.PointCloud(X), public_spec; stage=:graded_complex)
        @test public.grades == E.grades
        @test public.boundaries == E.boundaries
    end
    # Small fixtures force the arbitrary-length mask path without needing an
    # infeasible >64-site reference enumeration.
    X = QQ[0 0; 2 0; 1 2; 3 3]
    spec = OPT.FiltrationSpec(kind=:rhomboid, backend=:incremental)
    small, rs = DI._rhomboid_geometry(X, 2, spec, UInt64; backend=:incremental)
    large, rl = DI._rhomboid_geometry(X, 2, spec, BigInt; backend=:incremental)
    @test [[(BigInt(c.inside), BigInt(c.on), rs[c]) for c in group] for group in small] ==
          [[(c.inside, c.on, rl[c]) for c in group] for group in large]
    # Discovery itself must budget intermediate descriptors, even if its
    # consumer retains no cells. The three input sites already fill this
    # allowance, so the first top-cell descriptor must fail before emission.
    triangle = QQ[0 0; 2 0; 1 2]
    limited = OPT.FiltrationSpec(kind=:rhomboid, backend=:incremental,
        construction=OPT.ConstructionOptions(budget=OPT.ConstructionBudget(max_simplices=3)))
    calls = Ref(0)
    @test_throws ArgumentError DI._rhomboid_incremental_top!(
        (_, _) -> (calls[] += 1), triangle, 2, limited, UInt64)
    @test calls[] == 0
    if Threads.nthreads() > 1
        # Each task builds its own exact polyhedra. Compare complete native
        # grades and signed incidences, exercising concurrent CDD calls without
        # sharing a polyhedron or relying on diagnostic global counters.
        cases = [(X, window) for X in (triangle, QQ[0 0; 2 0; 1 2; 3 3], tetrahedron)
                 for window in (nothing, (1, 2))]
        build_case(case) = DI.encode(DT.PointCloud(case[1]),
            OPT.FiltrationSpec(kind=:rhomboid, backend=:incremental, depth_range=case[2]);
            stage=:graded_complex)
        serial = build_case.(cases)
        jobs = [let case = cases[mod1(i, length(cases))]
                    Threads.@spawn build_case(case)
                end for i in 1:12]
        for (i, job) in enumerate(jobs)
            actual = fetch(job)
            expected = serial[mod1(i, length(cases))]
            @test actual.grades == expected.grades
            @test actual.boundaries == expected.boundaries
        end
    end
end

@testset "A67 automatic enumeration uses the measured bounded regime" begin
    # Positive parabola parameters have no four cocircular sites: the four
    # roots of a circle intersection polynomial would have sum zero.
    # Test both sides of the measured threshold through the public entrypoint.
    for n in (16, 32)
        X = hcat(QQ[i//n for i in 1:n], QQ[(i*i)//(n*n) for i in 1:n])
        points = DT.PointCloud(X)
        outputs = [DI.encode(points, DI.RhomboidFiltration(;
                       backend, depth_range=(1,2)); stage=:graded_complex, cache=nothing)
                   for backend in (:exhaustive, :incremental, :auto)]
        @test outputs[1].grades == outputs[2].grades == outputs[3].grades
        @test outputs[1].boundaries == outputs[2].boundaries == outputs[3].boundaries
        # A small explicit grid checks the recorded decision and the known
        # final H0 depth map. Coarse-axis floor snapping changes earlier
        # births, so only query beyond every original birth in this fixture.
        spec = OPT.FiltrationSpec(kind=:rhomboid, backend=:auto,
            depth_range=(1,2), axes=([0,2],[-2,-1]))
        enc = DI.encode(points, spec; field=CM.F3(), cache=nothing)
        evidence = RES.provenance(enc)
        @test evidence.backend.requested.multicover === :auto
        @test (:multicover => (n < 32 ? :exhaustive : :incremental)) in evidence.backend.effective
        M = RES.encoding_module(enc)
        labels = [EC.locate(enc.pi, point) for point in ((2,1),(2,2))]
        @test M.dims[labels] == [1,1]
        @test Matrix(MD.map_leq(M, labels[2], labels[1])) == ones(CM.coeff_type(CM.F3()),1,1)
    end
end

@testset "A66-A68 public contracts, empty windows and recorded models" begin
    budget = OPT.ConstructionBudget(max_simplices=20_000)
    construction = OPT.ConstructionOptions(; budget)
    points = DT.PointCloud(reshape([0,2],2,1))
    for backend in (:auto,:exhaustive,:incremental,:subdivision_cech)
        f = DI.RhomboidFiltration(;backend,depth_range=(1,2),construction)
        spec = DI._filtration_spec(f)
        @test DI.check_filtration_spec(spec;throw=true).valid
        @test DI.filtration_parameters(DI.to_filtration(spec)) == DI.filtration_parameters(f)
        @test DI.filtration_parameters(f).backend === backend
        @test DI.filtration_parameters(f).depth_range == (1,2)
        direct = DI.encode(points,f;stage=:graded_complex)
        from_spec = DI.encode(points,spec;stage=:graded_complex)
        @test direct.grades == from_spec.grades
        @test DT.boundary_maps(direct) == DT.boundary_maps(from_spec)
        @test all(B -> all(iszero,nonzeros(B)),
                  [direct.boundaries[i-1]*direct.boundaries[i] for i in 2:length(direct.boundaries)])
        for field in FIELDS_FULL
            zero_window = DI.RhomboidFiltration(;backend,depth_range=(2,2),radius=0,construction)
            for stage in (:graded_complex,:simplex_tree)
                empty = DI.encode(points,zero_window;stage,field)
                @test sum(DT.cell_counts(empty)) == 0
            end
            enc = DI.encode(points,zero_window;field,cache=CM.SessionCache())
            @test all(iszero,RES.module_dims(enc.M))
            p = RES.provenance(enc)
            @test p.window.lower == (0,-2) && p.window.upper == (0,-2)
            @test p.construction.depth_range == (2,2)
            @test p.construction.model === (backend === :subdivision_cech ? :subdivision_cech : :sliced_rhomboid)
            @test p.backend.requested.multicover === backend
            @test (:multicover => (backend === :auto ? :exhaustive : backend)) in p.backend.effective
            depths = DI.RhomboidFiltration(;backend,depth_range=(0,2),radius=0,construction)
            all_depths = DI.encode(points,depths;field)
            @test RES.provenance(all_depths).window.lower == (0,-2)
            @test RES.provenance(all_depths).window.upper == (0,0)
            @test sort(RES.module_dims(all_depths.M)) == [0,1,2]
        end
    end
    for depths in ((-1,1),(2,1),(0,true),(0,1.0),(0,),[0,1],0:1)
        @test_throws ArgumentError DI.RhomboidFiltration(depth_range=depths)
        @test !DI.check_filtration_spec(OPT.FiltrationSpec(kind=:rhomboid,depth_range=depths)).valid
    end
    @test_throws ArgumentError DI.plan_ingestion(points,DI.RhomboidFiltration(depth_range=(0,3)))
    for axes in (([0,1],[-3,-1]), ([0,1],[-1,0]), ([0,2],[-2,-1]), (Int[],[-2,-1]), (0,[-2,-1]))
        invalid_axes = OPT.FiltrationSpec(kind=:rhomboid, depth_range=(1,2), radius=1, axes=axes)
        @test_throws ArgumentError DI.plan_ingestion(points,invalid_axes)
    end
    @test_throws ArgumentError DI.RhomboidFiltration(backend=:legacy)
    @test_throws ArgumentError DI.RhomboidFiltration(backend=:subdivision_cech)
    bad = OPT.FiltrationSpec(kind=:rhomboid,backend=:subdivision_cech)
    @test !DI.check_filtration_spec(bad).valid
    @test_throws ArgumentError DI.plan_ingestion(points,bad)
    for limited in (OPT.ConstructionBudget(max_simplices=2),
                    OPT.ConstructionBudget(max_simplices=100,max_edges=0),
                    OPT.ConstructionBudget(max_simplices=100,memory_budget_bytes=1))
        f = DI.RhomboidFiltration(backend=:subdivision_cech,
            construction=OPT.ConstructionOptions(budget=limited))
        @test_throws ArgumentError DI.encode(points,f;stage=:graded_complex)
    end
    # Subdivision-Cech of two sites including the depth-zero vertex is the
    # order complex of a Boolean lattice: 4 vertices, 5 edges and 2 triangles.
    f = DI.RhomboidFiltration(backend=:subdivision_cech,construction=construction)
    estimated = DI.describe(DI.estimate_ingestion(points,f))
    @test estimated.cell_counts_by_dim == [4,5,2]
    @test DT.cell_counts(DI.encode(points,f;stage=:graded_complex)) == [4,5,2]
    # Coincident labels must survive into coverage, and auto must not silently
    # substitute the explicitly requested degenerate-input model.
    duplicate = DT.PointCloud(reshape([0,0,2],3,1))
    for backend in (:auto,:exhaustive,:incremental)
        @test_throws ArgumentError DI.plan_ingestion(duplicate,DI.RhomboidFiltration(;backend))
    end
    # Arbitrary-length indexed masks, without exponentially many subsets:
    # all 65 coincident labels meet at radius zero in coverage depth 65.
    many_labels = DT.PointCloud(zeros(Int,65,1))
    high_depth = DI.RhomboidFiltration(backend=:subdivision_cech,depth_range=(65,65),
        construction=OPT.ConstructionOptions(budget=OPT.ConstructionBudget(max_simplices=100)))
    high_complex = DI.encode(many_labels,high_depth;stage=:graded_complex)
    @test DT.cell_counts(high_complex) == [1]
    @test high_complex.grades == [(0,65)]
    @test RES.module_dims(DI.encode(many_labels,high_depth).M) == [1]
    empty_filtration = DI.RhomboidFiltration(depth_range=(2,2),radius=0)
    quantized_empty = DI.encode(points,empty_filtration;pipeline=OPT.PipelineOptions(eps=0.5))
    @test RES.module_dims(quantized_empty.M) == [0]
    @test RES.provenance(quantized_empty).window.lower == (0,-2)
    four_space = DT.PointCloud(vcat(zeros(Int,1,4),Matrix{Int}(I,4,4)))
    @test_throws ArgumentError DI.plan_ingestion(four_space,DI.RhomboidFiltration(backend=:incremental))
    @test_throws ArgumentError DI.estimate_ingestion(four_space,DI.RhomboidFiltration(backend=:incremental))
end

@testset "A68 degenerate square and repeated sites: independent persistence maps" begin
    isdefined(@__MODULE__, :_a72_rref) || include(joinpath(@__DIR__, "helpers", "geometric_persistence_oracles.jl"))
    isdefined(@__MODULE__, :_a66_carrier_comparison) || include(joinpath(@__DIR__, "helpers", "rhomboid_extension_oracles.jl"))
    AR = TamerOp.ExactReals.AlgebraicReal
    construction = OPT.ConstructionOptions(budget=OPT.ConstructionBudget(max_simplices=50_000))
    fixtures = (
        (X=QQ[-1 -1; 1 -1; 1 1; -1 1], radii=AR[0, 1, 6//5, sqrt(AR(2)), 3//2], depth=4),
        (X=reshape(QQ[0, 0, 2], :, 1), radii=AR[0, 1//2, 1, 3//2], depth=3),
    )
    for fixture in fixtures
        X = fixture.X
        spec = OPT.FiltrationSpec(kind=:rhomboid, backend=:subdivision_cech,
            max_dim=2, depth_range=(1, fixture.depth), construction=construction)
        reference = _a72_subdivision_cech(X; maxdim=2)
        G, flags, masks, radii2 = _a68_flag_fixture(X, spec)
        @test Int.(masks) == collect(1:(1 << size(X, 1))-1)
        @test sort.(flags) == [[flag for flag in group if first(flag) != 0]
                              for group in reference.simplices]
        # Compare every independent convex-ball optimum, including all tied
        # radii and distinct subset labels at coincident geometric locations.
        @test all(radii2[i] == reference.radii2[Int(mask)+1] for (i, mask) in enumerate(masks))
        comparison = _a66_flag_carrier_comparison(flags, reference)
        for degree in 1:length(comparison)-1
            @test reference.boundaries[degree] * comparison[degree+1] == comparison[degree] * G.boundaries[degree]
        end
        _a66_check_carrier_grades(G, reference, comparison)
        queries = [(r, k) for r in fixture.radii for k in 1:fixture.depth]
        active_reference = _a66_active_reference(reference, queries)
        active_source = [_a72_source_active(G, p, (1, -1)) for p in queries]
        for field in FIELDS_FULL, degree in (0, 1)
            expected = map(queries) do (r, k)
                if fixture.depth == 4
                    if degree == 1
                        return k == 1 && 1 <= r^2 < 2 ? 1 : 0
                    elseif r^2 >= 2
                        return 1
                    elseif k == 1
                        return r < 1 ? 4 : 1
                    elseif k == 2
                        return r < 1 ? 0 : 4
                    end
                    return 0
                end
                degree == 1 && return 0
                return r >= 1 ? 1 : k == 1 ? 2 : k == 2 ? 1 : 0
            end
            _a72_check_module_comparison(DT.PointCloud(X), spec, G, reference, comparison,
                queries, active_reference, active_source, degree, expected, field)
        end
    end
end

@testset "A68 exact hexagon: ordinary Cech carrier and nonzero H1 transport" begin
    isdefined(@__MODULE__, :_a72_rref) || include(joinpath(@__DIR__, "helpers", "geometric_persistence_oracles.jl"))
    isdefined(@__MODULE__, :_a66_carrier_comparison) || include(joinpath(@__DIR__, "helpers", "rhomboid_extension_oracles.jl"))
    AR = TamerOp.ExactReals.AlgebraicReal
    # Exact regular hexagon in x+y+z=0. Float64 trigonometric coordinates would
    # conceal its actual cosphericity and are deliberately not used.
    X = QQ[1 -1 0; 1 0 -1; 0 1 -1; -1 1 0; -1 0 1; 0 -1 1]
    @test all(sum(abs2, X[i, :]) == 2 for i in axes(X, 1))
    construction = OPT.ConstructionOptions(budget=OPT.ConstructionBudget(max_simplices=50_000))
    spec = OPT.FiltrationSpec(kind=:rhomboid, backend=:subdivision_cech, max_dim=2,
                              depth_range=(1, 1), construction=construction)
    reference = _a72_cech(X; maxdim=2)
    @test reference.radii2[(1, 2)] == 1//2
    G, flags, _, _ = _a68_flag_fixture(X, spec)
    comparison = _a68_ordinary_cech_comparison(flags, reference)
    for degree in 1:2
        @test reference.boundaries[degree] * comparison[degree+1] == comparison[degree] * G.boundaries[degree]
    end
    queries = [(r, 1) for r in AR[0, 3//4, 13//10, sqrt(AR(2))]]
    active_reference = [[[i for (i, s) in enumerate(group) if reference.radii2[s] <= r^2]
                         for group in reference.simplices] for (r, _) in queries]
    active_source = [_a72_source_active(G, p, (1, -1)) for p in queries]
    for field in (CM.QQField(), CM.F3())
        _a72_check_module_comparison(DT.PointCloud(X), spec, G, reference, comparison,
            queries, active_reference, active_source, 1, [0, 1, 1, 0], field)
    end
end

@testset "A66-A68 pipeline serialization preserves depth and backend contracts" begin
    points = DT.PointCloud(reshape(QQ[0, 2], :, 1))
    construction = OPT.ConstructionOptions(budget=OPT.ConstructionBudget(max_simplices=1_000))
    mktemp() do path, io
        close(io)
        for backend in (:auto, :exhaustive, :incremental, :subdivision_cech), depths in (nothing, (1, 1), (1, 2))
            filtration = DI.RhomboidFiltration(; backend, depth_range=depths, construction)
            spec = DI._filtration_spec(filtration)
            original = DI.encode(points, filtration; stage=:graded_complex)
            SER.save_pipeline_json(path, points, spec; degree=0)
            for validation in (:strict, :trusted)
                loaded, restored, degree, _ = SER.load_pipeline_json(path; validation=validation)
                @test get(restored.params, :depth_range, nothing) == depths
                @test restored.params.backend === backend
                @test degree == 0
                @test DT.point_matrix(loaded) == DT.point_matrix(points)
                @test DI.check_filtration_spec(restored; throw=true).valid
                rebuilt = DI.encode(loaded, restored; stage=:graded_complex)
                @test rebuilt.grades == original.grades
                @test rebuilt.boundaries == original.boundaries
            end
        end
        base = JSON3.read(read(path, String), Dict{String,Any})
        # JSON numbers such as 1.0 denote integral values and JSON3 canonically
        # parses them as Int. Fractional values and Boolean endpoints are invalid.
        for depths in ([-1, 1], [2, 1], Any[0, true], [0.0, 1.5], [0], "0:1")
            altered = deepcopy(base)
            altered["spec"]["params"]["depth_range"] = depths
            write(path, JSON3.write(altered))
            @test !SER.check_pipeline_json(path).valid
            for validation in (:strict, :trusted)
                @test_throws ArgumentError SER.load_pipeline_json(path; validation=validation)
            end
        end
        for backend in ("legacy", "EXHAUSTIVE", 1, nothing)
            altered = deepcopy(base)
            altered["spec"]["params"]["backend"] = backend
            write(path, JSON3.write(altered))
            @test !SER.check_pipeline_json(path).valid
            for validation in (:strict, :trusted)
                @test_throws ArgumentError SER.load_pipeline_json(path; validation=validation)
            end
        end
    end
end

@testset "A66 simplex-stage carrier maps agree with native capped modules" begin
    isdefined(@__MODULE__, :_a72_rref) || include(joinpath(@__DIR__, "helpers", "geometric_persistence_oracles.jl"))
    isdefined(@__MODULE__, :_a66_carrier_comparison) || include(joinpath(@__DIR__, "helpers", "rhomboid_extension_oracles.jl"))
    AR = TamerOp.ExactReals.AlgebraicReal
    X = QQ[0 0; 2 0; 1 2]
    spec = OPT.FiltrationSpec(kind=:rhomboid, backend=:exhaustive, depth_range=(1, 2), max_dim=3)
    reference = _a72_subdivision_cech(X; maxdim=3)
    cells, radii2 = DI._rhomboid_geometry(X, 2, spec, UInt64)
    groups, G = DI._rhomboid_depth_model(cells, radii2, 3, spec)
    unions = [[cell.carrier.inside | cell.carrier.on for cell in group] for group in groups]
    cellular = _a66_carrier_comparison(G.boundaries, unions, reference)
    tree = DI.encode(DT.PointCloud(X), spec; stage=:simplex_tree)
    simplicial_G = DI._graded_complex_from_simplex_tree(tree)
    global_carriers = reduce(vcat, unions)
    flags = [Tuple[] for _ in DT.cell_counts(simplicial_G)]
    for id in 1:DT.simplex_count(tree)
        push!(flags[DT.simplex_dimension(tree, id)+1],
              Tuple(Int(global_carriers[v]) for v in DT.simplex_vertices(tree, id)))
    end
    simplicial = _a66_flag_carrier_comparison(flags, reference)
    for degree in 1:length(simplicial)-1
        @test reference.boundaries[degree] * simplicial[degree+1] == simplicial[degree] * simplicial_G.boundaries[degree]
    end
    _a66_check_carrier_grades(simplicial_G, reference, simplicial)
    queries = [(r, k) for r in AR[1, 9//8, 5//4] for k in 1:2]
    active_reference = _a66_active_reference(reference, queries)
    active_native = [_a72_source_active(G, p, (1, -1)) for p in queries]
    active_simplicial = [_a72_source_active(simplicial_G, p, (1, -1)) for p in queries]
    for field in FIELDS_FULL, degree in (0, 1)
        expected = map(queries) do (r, k)
            degree == 1 && return k == 1 && 5//4 <= r^2 < 25//16 ? 1 : 0
            return k == 1 ? (r == 1 ? 2 : 1) : (r == 9//8 ? 3 : 1)
        end
        native = _a72_check_module_comparison(DT.PointCloud(X), spec, G, reference, cellular,
            queries, active_reference, active_native, degree, expected, field)
        flag_model = _a72_check_module_comparison(tree, OPT.FiltrationSpec(kind=:graded, orientation=(1, -1)),
            simplicial_G, reference, simplicial, queries, active_reference, active_simplicial, degree, expected, field)
        _a66_compare_through_reference(native, flag_model, queries)
    end
end

@testset "A66 omitted carriers and constrained radius minima" begin
    isdefined(@__MODULE__, :_a72_rref) || include(joinpath(@__DIR__, "helpers", "geometric_persistence_oracles.jl"))
    isdefined(@__MODULE__, :_a66_carrier_comparison) || include(joinpath(@__DIR__, "helpers", "rhomboid_extension_oracles.jl"))
    AR = TamerOp.ExactReals.AlgebraicReal
    # The unconstrained circle through sites1,2 has center(2,0), radius2,
    # and incorrectly contains site3. Excluding it forces center(2,-1) and
    # squared radius5. A top cell with inside={3} is omitted by the depth1
    # construction, so this checks actual coface pruning, unlike a simplex.
    X = QQ[0 0; 4 0; 1 1; 0 3]
    full_spec = OPT.FiltrationSpec(kind=:rhomboid, backend=:exhaustive, max_dim=3)
    cells, radii2 = DI._rhomboid_geometry(X, 2, full_spec, UInt64)
    constrained = DI._RhomboidCell(UInt64(0), UInt64(0b0011))
    @test radii2[constrained] == 5
    reference = _a72_subdivision_cech(X; maxdim=3)
    @test reference.radii2[0b0011+1] == 4
    full_G, full_unions = _a66_native_fixture(X, full_spec)
    full_map = _a66_carrier_comparison(full_G.boundaries, full_unions, reference)
    radii = AR[0, 1, 3//2, sqrt(AR(5)), 5//2]
    for window in ((1, 1), (1, 2))
        spec = OPT.FiltrationSpec(kind=:rhomboid, backend=:exhaustive, max_dim=3, depth_range=window)
        retained, rradii = DI._rhomboid_geometry(X, 2, spec, UInt64)
        @test rradii[constrained] == 5
        # Compare the actual clipped cellular models before quotienting:
        # pruning original carriers must preserve every cell and incidence.
        full_groups, clipped_full = DI._rhomboid_depth_model(cells, radii2, 4, spec)
        retained_groups, clipped_retained = DI._rhomboid_depth_model(retained, rradii, 4, spec)
        @test retained_groups == full_groups
        @test clipped_retained.grades == clipped_full.grades
        @test clipped_retained.boundaries == clipped_full.boundaries
        if last(window) == 1
            @test length(last(retained)) < length(last(cells))
            @test sum(length, retained) < sum(length, cells)
        end
        G, unions = _a66_native_fixture(X, spec)
        comparison = _a66_carrier_comparison(G.boundaries, unions, reference)
        _a66_check_carrier_grades(G, reference, comparison)
        queries = [(r, k) for r in radii for k in first(window):last(window)]
        active_reference = _a66_active_reference(reference, queries)
        active_full = [_a72_source_active(full_G, p, (1, -1)) for p in queries]
        active_source = [_a72_source_active(G, p, (1, -1)) for p in queries]
        for field in FIELDS_FULL, degree in (0, 1)
            expected = map(queries) do (r, k)
                degree == 1 && return 0
                k == 2 && return r == 0 ? 0 : 1
                return r == 0 ? 4 : r == 1 ? 3 : r == 3//2 ? 2 : 1
            end
            complete = _a72_check_module_comparison(DT.PointCloud(X), full_spec, full_G,
                reference, full_map, queries, active_reference, active_full, degree, expected, field)
            capped = _a72_check_module_comparison(DT.PointCloud(X), spec, G,
                reference, comparison, queries, active_reference, active_source, degree, expected, field)
            _a66_compare_through_reference(complete, capped, queries)
        end
    end
end

@testset "A66-A68 encoding-result roundtrip preserves the multicover module and provenance" begin
    isdefined(@__MODULE__, :_a72_rref) || include(joinpath(@__DIR__, "helpers", "geometric_persistence_oracles.jl"))
    AR = TamerOp.ExactReals.AlgebraicReal
    birth = sqrt(AR(2))
    radii = AR[0, 1, birth, 3//2]
    oriented_axes = (radii, AR[-2, -1])
    # Two sites distance 2sqrt(2): depth-one components merge exactly at
    # sqrt(2), when the depth-two component is born and maps isomorphically
    # to the merged component. This fixture has an exact irrational grade.
    points = DT.PointCloud(QQ[0 0; 2 2])
    construction = OPT.ConstructionOptions(budget=OPT.ConstructionBudget(max_simplices=1000))
    for backend in (:exhaustive, :subdivision_cech), field in FIELDS_FULL
        K = CM.coeff_type(field)
        spec = OPT.FiltrationSpec(kind=:rhomboid, backend=backend, depth_range=(1, 2),
            radius=last(radii), max_dim=2, axes=oriented_axes, construction=construction)
        enc = DI.encode(points, spec; field=field)
        M = RES.encoding_module(enc)
        recorded = RES.provenance(enc)
        @test recorded.construction.model === (backend === :exhaustive ? :sliced_rhomboid : :subdivision_cech)
        @test recorded.construction.depth_range == (1, 2)
        @test recorded.window.lower == (0, -2)
        @test recorded.window.upper == (3//2, -1)
        @test recorded.approximation.grade_arithmetic === :exact_real_algebraic
        @test EC.axes_from_encoding(enc.pi) == oriented_axes
        anchor = EC.locate(enc.pi, AR[0, 1])
        terminal = EC.locate(enc.pi, AR[last(radii), 1])
        # Independently identify a diagram of this shape with the component
        # oracle. Its sole nontrivial merge is [1 1]; all other nonzero maps
        # are identities. No equality of serialized stalk bases is assumed.
        function identify(N)
            merge = MD.map_leq(N, anchor, terminal)
            @test size(merge) == (1, 2)
            pivot = findfirst(x -> field isa CM.RealField ? abs(x) > 1e-10 : !iszero(x), vec(merge))
            @test pivot !== nothing
            row = zeros(K, 1, 2)
            row[1, 3-pivot] = one(K)
            first_basis = vcat(merge-row, row)
            @test !iszero(first_basis[1, 1]*first_basis[2, 2] - first_basis[1, 2]*first_basis[2, 1])
            return [d == 0 ? zeros(K, 0, 0) : d == 1 ? MD.map_leq(N, q, terminal) :
                    first_basis * _a72_solve(MD.map_leq(N, anchor, q), Matrix{K}(I, 2, 2))
                    for (q, d) in enumerate(N.dims)]
        end
        original_identification = identify(M)
        # Replay the owned pipeline as well: JSON stores the axes in an array,
        # whereas the grid builder consumes a tuple. This must preserve the
        # complete diagram, including user-specified noncritical sample axes.
        mktemp() do path, io
            close(io)
            SER.save_pipeline_json(path, points, spec; degree=0)
            for validation in (:strict, :trusted)
                restored_points, restored_spec, degree, _ = SER.load_pipeline_json(path; validation=validation)
                rebuilt = DI.encode(restored_points, restored_spec; field=field, degree=degree)
                rebuilt_M = RES.encoding_module(rebuilt)
                @test EC.axes_from_encoding(rebuilt.pi) == oriented_axes
                @test RES.provenance(rebuilt).window == recorded.window
                @test RES.provenance(rebuilt).construction == recorded.construction
                @test rebuilt_M.dims == M.dims
                RJ = identify(rebuilt_M)
                for u in eachindex(RJ), v in eachindex(RJ)
                    FF.leq(rebuilt.P, u, v) || continue
                    Ju = _a72_solve(original_identification[u], RJ[u])
                    Jv = _a72_solve(original_identification[v], RJ[v])
                    @test _a72_equal(Jv * MD.map_leq(rebuilt_M, u, v), MD.map_leq(M, u, v) * Ju)
                end
            end
        end
        mktemp() do path, io
            close(io)
            SER.save_encoding_json(path, enc)
            for validation in (:strict, :trusted)
                loaded = SER.load_encoding_json(path; validation=validation)
                @test SER.check_encoding_json(path).valid
                @test loaded.M.field == field
                @test loaded.opts.field == field
                @test EC.axes_from_encoding(loaded.pi) == oriented_axes
                restored = RES.provenance(loaded)
                for key in (:degree, :degree_convention, :window, :orientation, :construction,
                            :discretization, :approximation, :backend, :reconstruction)
                    @test restored[key] == recorded[key]
                end
                @test restored.serialization.identification === :natural_isomorphism
                @test restored.base_poset === loaded.P
                for r in radii, depth in 1:2
                    q = EC.locate(loaded.pi, AR[r, depth])
                    @test q == EC.locate(enc.pi, AR[r, depth])
                    expected = r < birth ? (depth == 1 ? 2 : 0) : 1
                    @test loaded.M.dims[q] == expected
                    @test M.dims[q] == expected
                end
                J = identify(loaded.M)
                # The explicit stalkwise identification also compares the
                # loaded diagram to the actual pre-save homology module.
                comparison = [_a72_solve(original_identification[q], J[q]) for q in eachindex(J)]
                for u in eachindex(J), v in eachindex(J)
                    FF.leq(loaded.P, u, v) || continue
                    du, dv = loaded.M.dims[u], loaded.M.dims[v]
                    expected = du == 0 || dv == 0 ? zeros(K, dv, du) :
                               du == dv ? Matrix{K}(I, du, du) : ones(K, 1, 2)
                    @test _a72_equal(J[v] * MD.map_leq(loaded.M, u, v), expected * J[u])
                    @test _a72_equal(original_identification[v] * MD.map_leq(M, u, v),
                                     expected * original_identification[u])
                    @test _a72_equal(comparison[v] * MD.map_leq(loaded.M, u, v),
                                     MD.map_leq(M, u, v) * comparison[u])
                end
            end
        end
    end
end

@testset "A75 empty graded dataset serialization preserves mathematical shape" begin
    AR = TamerOp.ExactReals.AlgebraicReal
    for (T, scalar_kind) in ((Float64, "float64"), (QQ, "rational"), (AR, "algebraic_real")), N in (1, 2)
        grades = NTuple{N,T}[]
        datasets = (
            DT.GradedComplex([Int[], Int[]], [spzeros(Int, 0, 0)], grades),
            DT.SimplexTreeMulti(Int[1], Int[], Int[], Int[1, 1, 1], Int[1], grades),
        )
        axes = ntuple(_ -> T[0, 1], N)
        spec = OPT.FiltrationSpec(kind=:graded, axes=axes)
        for data in datasets
            key = data isa DT.GradedComplex ? "grades" : "grade_data"
            mktemp() do path, io
                close(io)
                SER.save_dataset_json(path, data)
                payload = JSON3.read(read(path, String), Dict{String,Any})
                @test payload["empty_grade_type"] == Dict("parameter_dim" => N, "scalar_kind" => scalar_kind)
                @test payload[key] == []
                @test !haskey(payload, "exact_" * key)
                @test SER.check_dataset_json(path).valid
                for validation in (:strict, :trusted)
                    restored = SER.load_dataset_json(path; validation=validation)
                    @test DT.parameter_dim(restored) == N
                    @test DT.cell_counts(restored) == DT.cell_counts(data) == [0, 0]
                    restored_grades = restored isa DT.GradedComplex ? restored.grades : restored.grade_data
                    @test eltype(restored_grades) == NTuple{N,T}
                    for field in FIELDS_FULL
                        M = DI.encode(restored, spec; stage=:module, field=field)
                        @test M.field == field
                        @test M.dims == zeros(Int, 2^N)
                        for u in eachindex(M.dims), v in eachindex(M.dims)
                            FF.leq(M.Q, u, v) || continue
                            @test MD.map_leq(M, u, v) == zeros(CM.coeff_type(field), 0, 0)
                        end
                    end
                end
                # These are malformed mathematical-shape records even on the
                # trusted path. Neither arity nor scalar kind can be guessed.
                invalid = Dict{String,Any}[]
                missing = deepcopy(payload); delete!(missing, "empty_grade_type"); push!(invalid, missing)
                for member in ("parameter_dim", "scalar_kind")
                    missing = deepcopy(payload); delete!(missing["empty_grade_type"], member); push!(invalid, missing)
                end
                for dimension in (0, -1, 1.5, true, "2")
                    bad = deepcopy(payload); bad["empty_grade_type"]["parameter_dim"] = dimension; push!(invalid, bad)
                end
                for scalar in ("Float64", "unknown", "Main.ArbitraryType", 1)
                    bad = deepcopy(payload); bad["empty_grade_type"]["scalar_kind"] = scalar; push!(invalid, bad)
                end
                bad = deepcopy(payload); bad["empty_grade_type"]["extra"] = 1; push!(invalid, bad)
                bad = deepcopy(payload); bad[key] = Dict(); push!(invalid, bad)
                bad = deepcopy(payload); bad[key] = [zeros(N)]; push!(invalid, bad)
                bad = deepcopy(payload)
                bad["exact_" * key] = Dict("scalar_kind" => "rational", "offsets" => [1], "rational_values" => [])
                push!(invalid, bad)
                if data isa DT.GradedComplex
                    for count in (0, 2)
                        bad = deepcopy(payload)
                        bad["boundaries"] = [deepcopy(first(payload["boundaries"])) for _ in 1:count]
                        push!(invalid, bad)
                    end
                    for shape in ((1, 0), (0, 1), (1, 1))
                        bad = deepcopy(payload)
                        bad["boundaries"][1]["m"], bad["boundaries"][1]["n"] = shape
                        push!(invalid, bad)
                    end
                else
                    for offsets in ([1, 2, 1], [2, 1])
                        bad = deepcopy(payload); bad["dim_offsets"] = offsets; push!(invalid, bad)
                    end
                end
                for bad in invalid
                    write(path, JSON3.write(bad))
                    @test !SER.check_dataset_json(path).valid
                    for validation in (:strict, :trusted)
                        @test_throws ArgumentError SER.load_dataset_json(path; validation=validation)
                    end
                end
            end
        end
    end
    # Ordinary nonempty coordinate payloads remain canonical without an empty
    # shape record, including their existing exact rational/algebraic tags.
    for T in (Float64, QQ, AR)
        data = DT.GradedComplex([Int[1]], SparseMatrixCSC{Int,Int}[], [(T(0), T(1))])
        mktemp() do path, io
            close(io)
            SER.save_dataset_json(path, data)
            payload = JSON3.read(read(path, String), Dict{String,Any})
            @test !haskey(payload, "empty_grade_type")
            @test SER.load_dataset_json(path).grades == data.grades
        end
    end
    # Exercise the exact empty outputs that motivated this boundary repair,
    # through both the native sliced model and the degenerate-input backend.
    points = DT.PointCloud(reshape(QQ[0, 2], :, 1))
    construction = OPT.ConstructionOptions(budget=OPT.ConstructionBudget(max_simplices=1_000))
    for backend in (:exhaustive, :subdivision_cech), stage in (:graded_complex, :simplex_tree)
        data = DI.encode(points, DI.RhomboidFiltration(; backend, depth_range=(2, 2), radius=0, construction); stage=stage)
        @test sum(DT.cell_counts(data)) == 0
        mktemp() do path, io
            close(io)
            SER.save_dataset_json(path, data)
            restored = SER.load_dataset_json(path)
            @test DT.parameter_dim(restored) == 2
            @test DT.cell_counts(restored) == DT.cell_counts(data)
            restored_grades = restored isa DT.GradedComplex ? restored.grades : restored.grade_data
            @test eltype(restored_grades) == NTuple{2,AR}
            spec = OPT.FiltrationSpec(kind=:graded, axes=(AR[0], AR[-2]), orientation=(1, -1))
            M = DI.encode(restored, spec; stage=:module)
            @test M.dims == [0]
            @test MD.map_leq(M, 1, 1) == zeros(QQ, 0, 0)
        end
    end
end

@testset "A66 integer depth and dimension contracts normalize before geometry" begin
    points = DT.PointCloud(reshape([0,2],2,1))
    construction = OPT.ConstructionOptions(budget=OPT.ConstructionBudget(max_simplices=20_000))
    axes = ([0,1],[-2,-1])
    for backend in (:auto,:exhaustive,:incremental,:subdivision_cech)
        reference_spec = OPT.FiltrationSpec(;kind=:rhomboid,backend,depth_range=(1,2),max_dim=1,axes,construction)
        reference = DI.encode(points,reference_spec;stage=:graded_complex,cache=nothing)
        reference_module = DI.encode(points,reference_spec;stage=:module,cache=nothing)
        reference_counts = DI.cell_counts_by_dim(DI.estimate_ingestion(points,reference_spec))
        for T in (Int32,BigInt,UInt)
            depths = (T(1),T(2))
            typed = DI.RhomboidFiltration(;backend,depth_range=depths,max_dim=1,construction)
            @test DI.filtration_parameters(typed).depth_range isa Tuple{Int,Int}
            @test DI.filtration_parameters(typed).depth_range == (1,2)
            direct = DI.encode(points,typed;stage=:graded_complex,cache=nothing)
            @test direct.grades == reference.grades
            @test direct.boundaries == reference.boundaries
            raw = OPT.FiltrationSpec(;kind=:rhomboid,backend,depth_range=depths,max_dim=T(1),axes,construction)
            @test DI.check_filtration_spec(raw;throw=true).valid
            restored = DI.to_filtration(raw)
            @test DI.filtration_parameters(restored).depth_range isa Tuple{Int,Int}
            @test DI.filtration_parameters(restored).max_dim isa Int
            plan = DI.plan_ingestion(points,raw;preflight=true,cache=nothing)
            @test DI.plan_spec(plan).params.depth_range isa Tuple{Int,Int}
            @test DI.plan_spec(plan).params.max_dim isa Int
            @test DI.cell_counts_by_dim(DI.estimate_ingestion(points,raw)) == reference_counts
            built = DI.encode(points,raw;stage=:graded_complex,cache=nothing)
            @test built.grades == reference.grades
            @test built.boundaries == reference.boundaries
            encoded = DI.encode(points,raw;cache=nothing)
            @test RES.provenance(encoded).construction.depth_range isa Tuple{Int,Int}
            actual = RES.encoding_module(encoded)
            @test actual.dims == reference_module.dims
            for u in eachindex(actual.dims),v in eachindex(actual.dims)
                FF.leq(actual.Q,u,v) || continue
                @test MD.map_leq(actual,u,v) == MD.map_leq(reference_module,u,v)
            end
        end
    end
    too_large = big(typemax(Int)) + 1
    for depths in ((0,too_large),(-too_large,0),(UInt(0),UInt(typemax(Int))+UInt(1)),
                   (false,2),(1,true),(1.0,2),(2,1))
        @test_throws ArgumentError DI.RhomboidFiltration(depth_range=depths)
        raw = OPT.FiltrationSpec(kind=:rhomboid,depth_range=depths)
        @test !DI.check_filtration_spec(raw).valid
        @test_throws ArgumentError DI.to_filtration(raw)
        @test_throws ArgumentError DI.plan_ingestion(points,raw)
        @test_throws ArgumentError DI.estimate_ingestion(points,raw)
        @test_throws ArgumentError DI.encode(points,raw;stage=:graded_complex)
    end
    for dimension in (too_large,-too_large,true)
        raw = OPT.FiltrationSpec(kind=:rhomboid,max_dim=dimension)
        @test !DI.check_filtration_spec(raw).valid
        @test_throws ArgumentError DI.to_filtration(raw)
        @test_throws ArgumentError DI.plan_ingestion(points,raw)
        @test_throws ArgumentError DI.estimate_ingestion(points,raw)
    end
    # The JSON boundary validates representability before converting numbers;
    # this payload is a legitimate UInt64 JSON integer on 64-bit Julia.
    mktemp() do path,io
        close(io)
        spec = OPT.FiltrationSpec(kind=:rhomboid,depth_range=(1,2))
        SER.save_pipeline_json(path,points,spec)
        payload = JSON3.read(read(path,String),Dict{String,Any})
        payload["spec"]["params"]["depth_range"] = [UInt(0),UInt(typemax(Int))+UInt(1)]
        write(path,JSON3.write(payload))
        @test !SER.check_pipeline_json(path).valid
        for validation in (:strict,:trusted)
            @test_throws ArgumentError SER.load_pipeline_json(path;validation)
        end
    end
end

@testset "A75 point ownership and geometry cache counterexamples" begin
    @testset "PointCloud copy owns its coordinate matrix" begin
        coordinates = [0.0 0.0; 1.0 0.0]
        copied = DT.PointCloud(coordinates; copy=true)
        borrowed = DT.PointCloud(coordinates; copy=false)
        @test DT.point_matrix(copied) == coordinates
        @test DT.point_matrix(copied) !== coordinates
        coordinates[2,1] = 3.0
        @test DT.point_matrix(copied)[2,1] == 1.0
        @test DT.point_matrix(borrowed)[2,1] == 3.0
    end
    for columnar in (false,true)
        @testset "Embedded planar copy owns its matrix ($columnar)" begin
            vertices = [0.0 0.0; 1.0 0.0]
            copied = columnar ? DT.EmbeddedPlanarGraph2D(vertices,[1],[2];copy=true) :
                                DT.EmbeddedPlanarGraph2D(vertices,[(1,2)];copy=true)
            @test DT.vertex_matrix(copied) == vertices
            @test DT.vertex_matrix(copied) !== vertices
            vertices[2,1] = 3.0
            @test DT.vertex_matrix(copied)[2,1] == 1.0
        end
    end
    @testset "Alpha distinguishes algebraic conjugates after an in-place edit" begin
        AR = TamerOp.ExactReals.AlgebraicReal
        small,large = sqrt(AR(3))-sqrt(AR(2)),sqrt(AR(3))+sqrt(AR(2))
        @test hash(small) == hash(large)
        cloud = DT.PointCloud(AR[0 0; small 0])
        filtration = DI.AlphaFiltration(max_dim=1,delaunay_backend=:naive)
        enabled = DI._POINTCLOUD_DELAUNAY_CACHE_ENABLED[]
        try
            DI._POINTCLOUD_DELAUNAY_CACHE_ENABLED[] = true
            DI._clear_pointcloud_delaunay_cache!()
            initial = DI.encode(cloud,filtration;stage=:graded_complex,cache=nothing)
            @test isapprox(last(initial.grades)[1], Float64(small)^2/4)
            DT.point_matrix(cloud)[2,1] = large
            changed = DI.encode(cloud,filtration;stage=:graded_complex,cache=nothing)
            fresh = DI.encode(DT.PointCloud(copy(DT.point_matrix(cloud))),filtration;
                              stage=:graded_complex,cache=nothing)
            @test isapprox(last(changed.grades)[1], Float64(large)^2/4)
            @test changed.grades == fresh.grades
            @test changed.boundaries == fresh.boundaries
        finally
            DI._POINTCLOUD_DELAUNAY_CACHE_ENABLED[] = enabled
            DI._clear_pointcloud_delaunay_cache!()
        end
    end
    @testset "Landmark radius graphs follow coordinate edits" begin
        cloud = DT.PointCloud([0.0 0.0;1.0 0.0])
        filtration = DI.LandmarkRipsFiltration(landmarks=[1,2],radius=2.0,max_dim=1,
                                              nn_backend=:bruteforce)
        session = CM.SessionCache()
        initial = DI.encode(cloud,filtration;stage=:graded_complex,cache=session)
        @test DT.cell_counts(initial) == [2,1]
        DT.point_matrix(cloud)[2,1] = 3.0
        changed = DI.encode(cloud,filtration;stage=:graded_complex,cache=session)
        fresh = DI.encode(cloud,filtration;stage=:graded_complex,cache=nothing)
        @test DT.cell_counts(fresh) == [2,0]
        @test DT.cell_counts(changed) == [2,0]
        @test changed.grades == fresh.grades
        @test changed.boundaries == fresh.boundaries
    end
end

@testset "A75 point geometry cache lifecycle and mathematical maps" begin
    # Two isolated vertices followed by their merger have identity maps before
    # the event and the augmentation [1 1] across it, independently of any
    # geometry or quotient implementation.
    function check_pair_maps(M, expected, field)
        @test M.dims == expected
        K = CM.coeff_type(field)
        for u in eachindex(expected), v in u:length(expected)
            target = expected[v] == expected[u] ? Matrix{K}(I,expected[v],expected[u]) :
                                                ones(K,expected[v],expected[u])
            actual = Matrix(MD.map_leq(M,u,v))
            if field isa CM.RealField
                @test isapprox(actual,target;atol=1e-12,rtol=0)
            else
                @test actual == target
            end
        end
    end
    AR = TamerOp.ExactReals.AlgebraicReal
    small,large = sqrt(AR(3))-sqrt(AR(2)),sqrt(AR(3))+sqrt(AR(2))
    small_grade,large_grade = Float64(small)^2/4,Float64(large)^2/4
    enabled = DI._POINTCLOUD_DELAUNAY_CACHE_ENABLED[]
    try
        DI._POINTCLOUD_DELAUNAY_CACHE_ENABLED[] = true
        DI._clear_pointcloud_delaunay_cache!()
        for field in FIELDS_FULL
            cloud = DT.PointCloud(AR[0 0;small 0])
            spec = OPT.FiltrationSpec(kind=:alpha,max_dim=1,delaunay_backend=:naive,
                                     axes=([0.0,small_grade,large_grade],))
            session = CM.SessionCache()
            initial = DI.encode(cloud,spec;stage=:module,field,cache=session)
            check_pair_maps(initial,[2,1,1],field)
            @test DI.encode(cloud,spec;stage=:module,field,cache=session) === initial
            DT.point_matrix(cloud)[2,1] = large
            changed = DI.encode(cloud,spec;stage=:module,field,cache=session)
            check_pair_maps(changed,[2,2,1],field)
            @test changed !== initial
            @test DI.encode(cloud,spec;stage=:module,field,cache=session) === changed
            DI._POINTCLOUD_DELAUNAY_CACHE_ENABLED[] = false
            direct = DI.encode(cloud,spec;stage=:module,field,cache=nothing)
            check_pair_maps(direct,[2,2,1],field)
            DI._POINTCLOUD_DELAUNAY_CACHE_ENABLED[] = true
            DT.point_matrix(cloud)[2,1] = small
            @test DI.encode(cloud,spec;stage=:module,field,cache=session) === initial
            CM._clear_session_cache!(session)
            cleared = DI.encode(cloud,spec;stage=:module,field,cache=session)
            @test cleared !== initial
            check_pair_maps(cleared,[2,1,1],field)
        end
        # Global triangulation ownership: first population, unchanged reuse,
        # explicit clearing, and collision-safe keys with immutable snapshots.
        cloud = DT.PointCloud(AR[0 0;small 0])
        spec = OPT.FiltrationSpec(kind=:alpha,max_dim=1,delaunay_backend=:naive)
        DI._clear_pointcloud_delaunay_cache!()
        first_key = DI._delaunay_cache_key(cloud.points,2,:naive)
        first_entry = DI._packed_delaunay_entry(cloud.points,spec;max_dim=2)
        @test length(DI._POINTCLOUD_DELAUNAY_CACHE) == 1
        @test DI._packed_delaunay_entry(cloud.points,spec;max_dim=2) === first_entry
        DT.point_matrix(cloud)[2,1] = large
        next_key = DI._delaunay_cache_key(cloud.points,2,:naive)
        @test hash(first_key) == hash(next_key)
        @test first_key != next_key
        second_entry = DI._packed_delaunay_entry(cloud.points,spec;max_dim=2)
        @test second_entry !== first_entry
        @test length(DI._POINTCLOUD_DELAUNAY_CACHE) == 2
        @test first_entry.packed.edge_radius == [Float64(small)/2]
        @test second_entry.packed.edge_radius == [Float64(large)/2]
        DI._clear_pointcloud_delaunay_cache!()
        @test isempty(DI._POINTCLOUD_DELAUNAY_CACHE)
        @test isempty(DI._POINTCLOUD_DELAUNAY_CACHE_ORDER)
        @test DI._packed_delaunay_entry(cloud.points,spec;max_dim=2) !== second_entry

        for field in FIELDS_FULL
            cloud = DT.PointCloud([0.0 0.0;1.0 0.0;4.0 0.0])
            landmarks = [1,2]
            spec = OPT.FiltrationSpec(kind=:landmark_rips,max_dim=1,landmarks=landmarks,
                radius=2.0,nn_backend=:bruteforce,axes=([0.0,1.0,2.0],))
            session = CM.SessionCache()
            initial = DI.encode(cloud,spec;stage=:module,field,cache=session)
            check_pair_maps(initial,[2,1,1],field)
            @test DI.encode(cloud,spec;stage=:module,field,cache=session) === initial
            DT.point_matrix(cloud)[2,1] = 3.0
            changed = DI.encode(cloud,spec;stage=:module,field,cache=session)
            check_pair_maps(changed,[2,2,2],field)
            check_pair_maps(DI.encode(cloud,spec;stage=:module,field,cache=nothing),[2,2,2],field)
            @test changed !== initial
            landmarks .= [2,3]
            selected = DI.encode(cloud,spec;stage=:module,field,cache=session)
            check_pair_maps(selected,[2,1,1],field)
            check_pair_maps(DI.encode(cloud,spec;stage=:module,field,cache=nothing),[2,1,1],field)
            CM._clear_session_cache!(session)
            cleared = DI.encode(cloud,spec;stage=:module,field,cache=session)
            @test cleared !== selected
            check_pair_maps(cleared,[2,1,1],field)
        end

        # Only the selected coordinates affect this radius graph. Matrix and
        # packed row views share keys, whereas reordered labels and changed
        # coordinates produce separate snapshots.
        cloud = DT.PointCloud([0.0 0.0;1.0 0.0;4.0 0.0])
        landmarks = [1,2]
        spec = OPT.FiltrationSpec(kind=:landmark_rips,landmarks=landmarks,radius=2.0,nn_backend=:bruteforce)
        cache = CM.EncodingCache()
        first_graph = DI._landmark_radius_subgraph_cached(cloud.points,landmarks,2.0,spec;cache)
        @test first_graph.edges == [(1,2)]
        @test first_graph.dists == [1.0]
        @test DI._landmark_radius_subgraph_cached(DT.point_matrix(cloud),landmarks,2.0,spec;cache) === first_graph
        @test length(cache.geometry) == 1
        DT.point_matrix(cloud)[3,1] = 99.0
        @test DI._landmark_radius_subgraph_cached(cloud.points,landmarks,2.0,spec;cache) === first_graph
        landmarks .= [2,1]
        reordered = DI._landmark_radius_subgraph_cached(cloud.points,landmarks,2.0,spec;cache)
        @test reordered !== first_graph
        @test reordered.edges == first_graph.edges
        @test reordered.dists == first_graph.dists
        DT.point_matrix(cloud)[2,1] = 1.5
        changed = DI._landmark_radius_subgraph_cached(cloud.points,landmarks,2.0,spec;cache)
        @test changed.dists == [1.5]
        @test first_graph.dists == [1.0]
        @test DI._landmark_radius_subgraph_cached(cloud.points,landmarks,1.0,spec;cache).edges == []

        if Threads.nthreads() > 1
            # Concurrent cold population of shared caches, with deterministic
            # independent inputs; the actual persistence arrows are checked.
            DI._clear_pointcloud_delaunay_cache!()
            shared = CM.SessionCache()
            clouds = [DT.PointCloud([0.0 0.0;x 0.0]) for x in (1.0,3.0)]
            alpha = OPT.FiltrationSpec(kind=:alpha,max_dim=1,delaunay_backend=:naive,
                                       axes=([0.0,0.25,2.25],))
            landmark = OPT.FiltrationSpec(kind=:landmark_rips,max_dim=1,landmarks=[1,2],
                                          radius=2.0,nn_backend=:bruteforce,axes=([0.0,1.0,2.0],))
            tasks = [Threads.@spawn begin
                family = isodd(div(i-1,2)) ? landmark : alpha
                DI.encode(clouds[mod1(i,2)],family;stage=:module,field=CM.QQField(),cache=shared)
            end for i in 1:12]
            for (i,task) in enumerate(tasks)
                is_landmark = isodd(div(i-1,2))
                expected = isodd(i) ? [2,1,1] : is_landmark ? [2,2,2] : [2,2,1]
                check_pair_maps(fetch(task),expected,CM.QQField())
            end
        end
    finally
        DI._POINTCLOUD_DELAUNAY_CACHE_ENABLED[] = enabled
        DI._clear_pointcloud_delaunay_cache!()
    end
end

@testset "A65 compiled rhomboid cache ownership, budgets and lifecycle" begin
    function geometry_entries(session, tag=:rhomboid_geometry)
        bucket = CM._workflow_encoding_cache(session)
        lock(bucket.lock) do
            [v.value for (k,v) in bucket.geometry if first(k) === tag]
        end
    end
    for backend in (:exhaustive,:incremental,:subdivision_cech)
        X = backend === :subdivision_cech ? reshape(QQ[0,0,2],3,1) : reshape(QQ[0,2],2,1)
        cloud = DT.PointCloud(X;copy=false)
        sc = CM.SessionCache()
        construction = OPT.ConstructionOptions(budget=OPT.ConstructionBudget(max_simplices=100_000))
        filtration = DI.RhomboidFiltration(;backend,depth_range=(1,2),construction)
        tag = backend === :subdivision_cech ? :subdivision_cech_geometry : :rhomboid_geometry
        @test isempty(geometry_entries(sc,tag))
        original = TamerOp.Workflow.encode(cloud,filtration;stage=:graded_complex,cache=sc)
        first_geometry = only(geometry_entries(sc,tag))
        @test sort!(unique!(collect(values(first_geometry.radii2)))) == QQ[0,1]
        again = TamerOp.Workflow.encode(cloud,filtration;stage=:graded_complex,cache=sc)
        direct = DI.encode(cloud,filtration;stage=:graded_complex,cache=nothing)
        @test only(geometry_entries(sc,tag)) === first_geometry
        @test again.grades == direct.grades == original.grades
        @test again.boundaries == direct.boundaries == original.boundaries
        @test again.grades !== original.grades
        encoded = DI.encode(cloud,filtration;stage=:encoded_complex,cache=sc)
        @test (:multicover => backend) in RES.provenance(encoded).backend.effective
        @test only(geometry_entries(sc,tag)) === first_geometry
        # Equal coordinates in another PointCloud are compatible content, while
        # shape/order/backend/depth remain part of the mathematical request.
        DI.encode(DT.PointCloud(copy(X)),filtration;stage=:graded_complex,cache=sc)
        @test only(geometry_entries(sc,tag)) === first_geometry
        for max_dim in (0,1,nothing), radius in (0,1,nothing), stage in (:simplex_tree,:graded_complex)
            request = DI.RhomboidFiltration(;backend,depth_range=(1,2),construction,max_dim,radius)
            cached = DI.encode(cloud,request;stage,cache=sc)
            fresh = DI.encode(cloud,request;stage,cache=nothing)
            cached = stage === :simplex_tree ? DI._graded_complex_from_simplex_tree(cached) : cached
            fresh = stage === :simplex_tree ? DI._graded_complex_from_simplex_tree(fresh) : fresh
            @test cached.grades == fresh.grades
            @test cached.boundaries == fresh.boundaries
            @test only(geometry_entries(sc,tag)) === first_geometry
        end
        if backend === :subdivision_cech
            larger = DI.RhomboidFiltration(;backend,depth_range=(1,3),construction)
            cached = DI.encode(cloud,larger;stage=:graded_complex,cache=sc)
            fresh = DI.encode(cloud,larger;stage=:graded_complex,cache=nothing)
            @test cached.grades == fresh.grades
            @test cached.boundaries == fresh.boundaries
            @test only(geometry_entries(sc,tag)) === first_geometry
        end
        # Neither editing a returned grade nor its boundary may poison geometry.
        original.grades[1] = (TamerOp.ExactReals.AlgebraicReal(10),TamerOp.ExactReals.AlgebraicReal(10))
        isempty(original.boundaries) || fill!(nonzeros(original.boundaries[1]),0)
        restored = DI.encode(cloud,filtration;stage=:graded_complex,cache=sc)
        @test restored.grades == direct.grades
        @test restored.boundaries == direct.boundaries
        for budget in (OPT.ConstructionBudget(max_simplices=1),
                       OPT.ConstructionBudget(max_simplices=100_000,max_edges=0),
                       OPT.ConstructionBudget(max_simplices=100_000,memory_budget_bytes=1))
            restricted = DI.RhomboidFiltration(;backend,depth_range=(1,2),
                construction=OPT.ConstructionOptions(;budget))
            @test_throws ArgumentError DI.encode(cloud,restricted;stage=:graded_complex,cache=sc)
            @test_throws ArgumentError DI.encode(cloud,restricted;stage=:graded_complex,cache=nothing)
        end
        @test only(geometry_entries(sc,tag)) === first_geometry
        X[end,1] = 4
        changed = DI.encode(cloud,filtration;stage=:graded_complex,cache=sc)
        changed_direct = DI.encode(cloud,filtration;stage=:graded_complex,cache=nothing)
        @test changed.grades == changed_direct.grades
        @test changed.boundaries == changed_direct.boundaries
        @test changed.grades != direct.grades
        @test length(geometry_entries(sc,tag)) == 2
        if backend !== :subdivision_cech
            X[end,1] = 0
            @test_throws ArgumentError DI.encode(cloud,filtration;stage=:graded_complex,cache=sc)
            @test length(geometry_entries(sc,tag)) == 2
        end
        X[end,1] = 2
        @test DI.encode(cloud,filtration;stage=:graded_complex,cache=sc).grades == direct.grades
        @test first_geometry in geometry_entries(sc,tag)
        CM._clear_session_cache!(sc)
        @test isempty(geometry_entries(sc,tag))
        @test DI.encode(cloud,filtration;stage=:graded_complex,cache=sc).grades == direct.grades
        @test only(geometry_entries(sc,tag)) !== first_geometry
    end

    @testset "Backend, depth and shape identities; arbitrary-size masks" begin
        sc = CM.SessionCache()
        cloud = DT.PointCloud(reshape(QQ[0,2],2,1))
        for backend in (:exhaustive,:incremental), depth_range in ((1,1),(1,2))
            filt = DI.RhomboidFiltration(;backend,depth_range)
            cached = DI.encode(cloud,filt;stage=:graded_complex,cache=sc)
            fresh = DI.encode(cloud,filt;stage=:graded_complex,cache=nothing)
            @test cached.grades == fresh.grades
            @test cached.boundaries == fresh.boundaries
        end
        @test length(geometry_entries(sc)) == 4
        full = DI.RhomboidFiltration(backend=:exhaustive)
        pair = DI.encode(cloud,full;stage=:graded_complex,cache=sc)
        point = DI.encode(DT.PointCloud(reshape(QQ[0,2],1,2)),full;stage=:graded_complex,cache=sc)
        @test DT.cell_counts(pair) == [4,4,1]
        @test DT.cell_counts(point) == [2,1]
        @test length(geometry_entries(sc)) == 6
        skeleton = DI.encode(cloud,DI.RhomboidFiltration(backend=:exhaustive,max_dim=0);
                             stage=:graded_complex,cache=sc)
        @test DT.cell_counts(skeleton) == [4]
        clipped = DI.encode(cloud,DI.RhomboidFiltration(backend=:exhaustive,radius=0);
                            stage=:graded_complex,cache=sc)
        @test DT.cell_counts(clipped) == [3,2,0]
        recovered = DI.encode(cloud,full;stage=:graded_complex,cache=sc)
        @test recovered.grades == pair.grades
        @test recovered.boundaries == pair.boundaries
        @test length(geometry_entries(sc)) == 6
        # A zero-depth cap has a single vertex, even with more than 64 labels.
        many = DT.PointCloud(reshape(QQ.(1:65),65,1))
        filt = DI.RhomboidFiltration(backend=:exhaustive,depth_range=(0,0))
        G = DI.encode(many,filt;stage=:graded_complex,cache=sc)
        @test sum(DT.cell_counts(G)) == 1
        @test G.grades == [(TamerOp.ExactReals.AlgebraicReal(0),TamerOp.ExactReals.AlgebraicReal(0))]
        @test DI.encode(many,filt;stage=:graded_complex,cache=sc).grades == G.grades
        @test any(e -> e isa DI._CompiledRhomboidGeometry{BigInt},geometry_entries(sc))
    end

    @testset "Failed native construction does not publish partial geometry" begin
        sc = CM.SessionCache()
        square = DT.PointCloud(QQ[-1 -1;1 -1;1 1;-1 1])
        @test_throws ArgumentError DI.encode(square,DI.RhomboidFiltration(backend=:exhaustive);
                                            stage=:graded_complex,cache=sc)
        @test isempty(geometry_entries(sc))
        cloud = DT.PointCloud(reshape(QQ[0,2],2,1))
        too_small = DI.RhomboidFiltration(construction=OPT.ConstructionOptions(
            budget=OPT.ConstructionBudget(max_simplices=1)))
        @test_throws ArgumentError DI.encode(cloud,too_small;stage=:graded_complex,cache=sc)
        @test isempty(geometry_entries(sc))
    end
    @testset "One complete publication, concurrent readers and clear" begin
        # Exhaustive geometry avoids making any new claim about the dependency's
        # mixed-owner CDD concurrency contract (tracked independently by A78).
        X = QQ[0 0;2 0;1 2]
        spec = DI._filtration_spec(DI.RhomboidFiltration(backend=:exhaustive,depth_range=(1,2)))
        sc = CM.SessionCache()
        bucket = CM._workflow_encoding_cache(sc)
        entries = fetch.([Threads.@spawn DI._compiled_rhomboid_geometry(copy(X),spec,bucket) for _ in 1:8])
        @test all(e -> e === first(entries),entries)
        @test length(geometry_entries(sc)) == 1
        expected = DI.encode(DT.PointCloud(X),spec;stage=:graded_complex,cache=nothing)
        outputs = fetch.([Threads.@spawn DI.encode(DT.PointCloud(copy(X)),spec;stage=:graded_complex,cache=sc) for _ in 1:8])
        @test all(G -> G.grades == expected.grades && G.boundaries == expected.boundaries,outputs)
        # Holding the actual publication lock makes the order deterministic:
        # clear cannot finish midway through a publication critical section.
        entered, proceed = Channel{Nothing}(1),Channel{Nothing}(1)
        publication = Threads.@spawn DI._rhomboid_cached_geometry(bucket,(:a65_clear_probe,)) do
            put!(entered,nothing); take!(proceed); first(entries)
        end
        take!(entered)
        clearing = Threads.@spawn CM._clear_session_cache!(sc)
        yield()
        @test !istaskdone(clearing)
        put!(proceed,nothing)
        @test fetch(publication) === first(entries)
        fetch(clearing)
        @test isempty(bucket.geometry)
        @test isempty(geometry_entries(sc))
        rebuilt = DI._compiled_rhomboid_geometry(X,spec,CM._workflow_encoding_cache(sc))
        @test rebuilt !== first(entries)
        @test rebuilt.cells == first(entries).cells
        @test rebuilt.radii2 == first(entries).radii2
    end
end

@testset "A65 cached geometry preserves independent homology and persistence maps" begin
    isdefined(@__MODULE__, :_a72_rref) || include(joinpath(@__DIR__, "helpers", "geometric_persistence_oracles.jl"))
    isdefined(@__MODULE__, :_a66_carrier_comparison) || include(joinpath(@__DIR__, "helpers", "rhomboid_extension_oracles.jl"))
    # The triangle has H1 for 5/4 <= r^2 < 25/16 at depth one. Subdivision-Cech
    # minimum balls, boundaries, quotient coordinates and comparison maps are
    # independently constructed by the test helpers.
    X = QQ[0 0;2 0;1 2]
    cloud = DT.PointCloud(X)
    spec = OPT.FiltrationSpec(kind=:rhomboid,backend=:exhaustive,depth_range=(1,2),max_dim=3)
    G,unions = _a66_native_fixture(X,spec)
    reference = _a72_subdivision_cech(X;maxdim=3)
    comparison = _a66_carrier_comparison(G.boundaries,unions,reference)
    queries = [(r,k) for r in QQ[1,9//8,6//5,5//4] for k in 1:2]
    active_reference = _a66_active_reference(reference,queries)
    active_source = [_a72_source_active(G,p,(1,-1)) for p in queries]
    sc = CM.SessionCache()
    for field in FIELDS_FULL, degree in (0,1)
        expected = map(queries) do (r,k)
            degree == 1 && return k == 1 && 5//4 <= r^2 < 25//16 ? 1 : 0
            return k == 1 ? (r == 1 ? 2 : 1) : (r in (9//8,6//5) ? 3 : 1)
        end
        _a72_check_module_comparison(cloud,spec,G,reference,comparison,queries,
            active_reference,active_source,degree,expected,field;cache=sc)
    end
    bucket = CM._workflow_encoding_cache(sc)
    @test count(k -> first(k) === :rhomboid_geometry,keys(bucket.geometry)) == 1
end

@testset "A76 sliced-cell facets have independent geometric orientations" begin
    isdefined(@__MODULE__, :_a72_rref) || include(joinpath(@__DIR__, "helpers", "geometric_persistence_oracles.jl"))
    isdefined(@__MODULE__, :_a76_check_slice_facets) || include(joinpath(@__DIR__, "helpers", "rhomboid_slice_oracles.jl"))
    _a76_check_slice_facets()
end


@testset "A77 independent carrier certificates and deterministic coverage" begin
    isdefined(@__MODULE__, :_a72_rref) || include(joinpath(@__DIR__, "helpers", "geometric_persistence_oracles.jl"))
    isdefined(@__MODULE__, :_a66_carrier_comparison) || include(joinpath(@__DIR__, "helpers", "rhomboid_extension_oracles.jl"))
    isdefined(@__MODULE__, :_a76_slice_vertices) || include(joinpath(@__DIR__, "helpers", "rhomboid_slice_oracles.jl"))
    isdefined(@__MODULE__, :_a77_clouds) || include(joinpath(@__DIR__, "helpers", "generated_rhomboid_oracles.jl"))
    cases = _a77_clouds()
    @test length(cases) == 38
    @test allunique(c.name for c in cases)
    @test count(c -> c.family === :multiset,cases) == 10
    X = QQ[0 0; 4 0; 1 1; 0 3]
    @test _a77_carrier_feasible(X,0,0b0011)
    @test _a77_carrier_radius_squared(X,0,0b0011) == 5
    @test _a72_minimum_ball_squared(X,Int(0b0011)) == 4
    @test !_a77_carrier_feasible(reshape(QQ[0,1,2],:,1),0b101,0)
    @test !_a77_native_admissible(QQ[0 0;1 0;1 1;0 1])
    @test !_a77_native_admissible(reshape(QQ[0,0,1],:,1))
    @test _a77_native_admissible(QQ[0 0 0;2 0 2;1 2 5;3 3 9])
end

@testset "A77 generated native grades, constrained minima and signed windows" begin
    isdefined(@__MODULE__, :_a72_rref) || include(joinpath(@__DIR__, "helpers", "geometric_persistence_oracles.jl"))
    isdefined(@__MODULE__, :_a76_slice_vertices) || include(joinpath(@__DIR__, "helpers", "rhomboid_slice_oracles.jl"))
    isdefined(@__MODULE__, :_a77_clouds) || include(joinpath(@__DIR__, "helpers", "generated_rhomboid_oracles.jl"))
    clouds = windows = builds = carriers = 0
    for case in _a77_clouds()
        _a77_native_admissible(case.X) || continue
        println("A77 native cloud=",case.name); flush(stdout)
        reference = _a77_carriers(case.X)
        coverage = _a77_check_native_windows(case.X,reference;family=case.family)
        clouds += 1; windows += coverage.windows
        builds += coverage.cutoff_builds; carriers += coverage.carriers
    end
    @test clouds == 27
    @test windows == 552
    @test builds == 2_664
    @test carriers == 841
    println("A77 native totals clouds=",clouds," windows=",windows," cutoff_builds=",builds," carriers=",carriers)
end

@testset "A77 generated fallback grades, labeled multiplicity and signed flags" begin
    isdefined(@__MODULE__, :_a72_rref) || include(joinpath(@__DIR__, "helpers", "geometric_persistence_oracles.jl"))
    isdefined(@__MODULE__, :_a66_carrier_comparison) || include(joinpath(@__DIR__, "helpers", "rhomboid_extension_oracles.jl"))
    isdefined(@__MODULE__, :_a77_clouds) || include(joinpath(@__DIR__, "helpers", "generated_rhomboid_oracles.jl"))
    builds = 0
    for case in _a77_clouds()
        println("A77 fallback cloud=",case.name); flush(stdout)
        reference = _a72_subdivision_cech(case.X;maxdim=3)
        builds += _a77_check_fallback_windows(case.X,reference;family=case.family)
    end
    @test builds == 1_372
    println("A77 fallback totals clouds=38 cutoff_builds=",builds)
end

@testset "A77 generated line and repeated-site persistence maps" begin
    isdefined(@__MODULE__, :_a72_rref) || include(joinpath(@__DIR__, "helpers", "geometric_persistence_oracles.jl"))
    isdefined(@__MODULE__, :_a66_carrier_comparison) || include(joinpath(@__DIR__, "helpers", "rhomboid_extension_oracles.jl"))
    isdefined(@__MODULE__, :_a77_clouds) || include(joinpath(@__DIR__, "helpers", "generated_rhomboid_oracles.jl"))
    for (index,case) in enumerate(_a77_clouds())
        case.family in (:line,:multiset) || continue
        println("A77 module cloud=",case.name); flush(stdout)
        _a77_check_generated_persistence(case,index)
    end
end

@testset "A77 generated planar, oblique and tetrahedral persistence maps" begin
    isdefined(@__MODULE__, :_a72_rref) || include(joinpath(@__DIR__, "helpers", "geometric_persistence_oracles.jl"))
    isdefined(@__MODULE__, :_a66_carrier_comparison) || include(joinpath(@__DIR__, "helpers", "rhomboid_extension_oracles.jl"))
    isdefined(@__MODULE__, :_a77_clouds) || include(joinpath(@__DIR__, "helpers", "generated_rhomboid_oracles.jl"))
    for (index,case) in enumerate(_a77_clouds())
        case.family in (:line,:multiset) && continue
        println("A77 module cloud=",case.name); flush(stdout)
        _a77_check_generated_persistence(case,index)
    end
end

@testset "A14 pipeline poset representations preserve modules and maps" begin
    # Two vertices merge along one edge. Mixed-radix grid labels are
    # (0,0), (1,0), (0,1), (1,1), with H0 dimensions 1,2,1,1.
    B = sparse(reshape([-1, 1], 2, 1))
    axes = ([0, 1], [0, 1])
    expected_dims = [1, 2, 1, 1]
    for field in FIELDS_FULL, orientation in ((1, 1), (1, -1))
        G = DT.GradedComplex([[1, 2], [1]], [B],
            [(0, 0), (1, 0), (1, orientation[2])])
        spec = OPT.FiltrationSpec(kind=:graded, axes=axes)
        sc = CM.SessionCache()
        built = Dict{Symbol,Any}()
        for kind in (:signature, :dense)
            pipeline = OPT.PipelineOptions(; poset_kind=kind, orientation, field)
            plan = DI.plan_ingestion(G, spec; pipeline, cache=sc, stage=:module)
            @test DI.plan_field(plan) == field
            @test DI.ingestion_plan_summary(plan).poset_kind == kind
            @test DI.check_ingestion_plan(plan; throw=true).valid
            M = DI.run_ingestion(plan)
            @test M isa MD.PModule
            @test M.dims == expected_dims
            @test M.field == field
            @test kind === :signature ? M.Q isa FF.ProductOfChainsPoset : M.Q isa FF.FinitePoset
            @test DI.run_ingestion(plan) === M
            uncached = DI.encode(G, spec; pipeline, cache=nothing, stage=:module)
            fresh = CM.SessionCache()
            encoded = DI.encode(G, spec; pipeline, cache=fresh)
            @test encoded.opts.poset_kind == kind
            @test encoded.opts.field == field
            @test _enc_dims(encoded) == expected_dims
            lazyM = _enc_module(encoded)
            for u in 1:4, v in 1:4
                FF.leq(M.Q, u, v) || continue
                arrow = MD.structure_map(M; source=u, target=v)
                @test arrow == MD.structure_map(uncached; source=u, target=v)
                @test arrow == MD.structure_map(lazyM; source=u, target=v)
                @test FL.rank(field, arrow) == (u == v ? expected_dims[u] : 1)
            end
            @test Matrix(MD.structure_map(M; source=2, target=4)) ==
                  reshape([CM.coerce(field, 1), CM.coerce(field, 1)], 1, 2)
            @test Matrix(MD.structure_map(M; source=1, target=2)) ==
                  reshape([CM.coerce(field, 1), CM.coerce(field, 0)], 2, 1)
            dims = DI.run_ingestion(plan; stage=:cohomology_dims)
            @test dims.dims == expected_dims
            @test typeof(dims.P) == typeof(M.Q)
            C = DI.run_ingestion(plan; stage=:cochain)
            E = DI.run_ingestion(plan; stage=:encoded_complex)
            for complex in (C, RES._materialize_complex(E.C))
                @test MC.component(complex, 0).dims == [1, 2, 1, 2]
                @test MC.component(complex, -1).dims == [0, 0, 0, 1]
                @test MC.check_module_complex(complex; throw=true).valid
                H = MC.cohomology_module(complex, 0)
                @test H.dims == expected_dims
                for u in 1:4, v in 1:4
                    FF.leq(M.Q, u, v) || continue
                    @test FL.rank(field, MD.structure_map(H; source=u, target=v)) ==
                          FL.rank(field, MD.structure_map(M; source=u, target=v))
                end
            end
            H = DI.run_ingestion(plan; stage=:fringe)
            @test IR.pmodule_from_fringe(H).dims == expected_dims
            built[kind] = M
        end
        @test built[:signature].Q !== built[:dense].Q
        for u in 1:4, v in 1:4
            @test FF.leq(built[:signature].Q, u, v) == FF.leq(built[:dense].Q, u, v)
            FF.leq(built[:signature].Q, u, v) || continue
            @test MD.structure_map(built[:signature]; source=u, target=v) ==
                  MD.structure_map(built[:dense]; source=u, target=v)
        end
        CM._clear_session_cache!(sc)
        again = DI.encode(G, spec; cache=sc, stage=:module,
            pipeline=OPT.PipelineOptions(poset_kind=:dense, orientation=orientation, field=field))
        @test again.Q !== built[:dense].Q
        @test again.dims == expected_dims
    end
end

@testset "A14 pipeline fields change mathematics and survive serialization" begin
    # The standard CW complex of RP^2 has d_1=0 and d_2=2. Its H1
    # survives the 2-cell exactly in characteristic two, detecting ignored fields.
    G = DT.GradedComplex([[1], [1], [1]],
        [spzeros(Int, 1, 1), sparse(reshape([2], 1, 1))], [(0,), (0,), (1,)])
    spec = OPT.FiltrationSpec(kind=:graded, axes=([0, 1],))
    sc = CM.SessionCache()
    for field in FIELDS_FULL
        expected = field == CM.F2() ? [1, 1] : [1, 0]
        pipeline = OPT.PipelineOptions(; field, poset_kind=:dense)
        plan = DI.plan_ingestion(G, spec; pipeline, cache=sc)
        encoded = DI.encode(plan; degree=1)
        @test DI.plan_field(plan) == field
        @test _enc_dims(encoded) == expected
        @test encoded.opts.field == field
        @test FL.rank(field, MD.structure_map(_enc_module(encoded); source=1, target=2)) == expected[2]
        explicit = DI.encode(G, spec; pipeline, field=CM.QQField(), cache=sc, degree=1)
        @test explicit.opts.field == CM.QQField()
        @test _enc_dims(explicit) == [1, 0]
        specfield = OPT.FiltrationSpec(kind=:graded, axes=([0, 1],), field=field)
        @test _enc_dims(DI.encode(G, specfield; cache=sc, degree=1)) == expected
        mktemp() do path, io
            close(io)
            SER.save_pipeline_json(path, G, specfield; degree=1, pipeline_opts=pipeline)
            payload = JSON3.read(read(path, String), Dict{String,Any})
            @test payload["schema_version"] == SER.PIPELINE_SCHEMA_VERSION
            for validation in (:strict, :trusted)
                G2, spec2, degree2, pipeline2 = SER.load_pipeline_json(path; validation)
                @test pipeline2.field == field
                @test pipeline2.poset_kind == :dense
                replay = DI.encode(G2, spec2; degree=degree2, pipeline=pipeline2)
                @test replay.opts.field == field
                @test replay.P isa FF.FinitePoset
                @test _enc_dims(replay) == expected
                @test MD.structure_map(_enc_module(replay); source=1, target=2) ==
                      MD.structure_map(_enc_module(encoded); source=1, target=2)
            end
            old = deepcopy(payload)
            old["schema_version"] = 2
            write(path, JSON3.write(old))
            @test_throws ErrorException SER.load_pipeline_json(path)
            for section in ("pipeline_options", "spec")
                malformed = deepcopy(payload)
                if section == "pipeline_options"
                    malformed[section]["field"] = "F2"
                else
                    malformed[section]["params"]["field"] = "F2"
                end
                write(path, JSON3.write(malformed))
                for validation in (:strict, :trusted)
                    @test_throws ArgumentError SER.load_pipeline_json(path; validation)
                end
            end
        end
    end
end

@testset "A14 stage, quantization and grid controls have observable behavior" begin
    data = DT.PointCloud([[0.0], [0.6]])
    filtration = DI.RipsFiltration(max_dim=1)
    pipeline = OPT.PipelineOptions(eps=0.5)
    plan = DI.plan_ingestion(data, filtration; pipeline, stage=:graded_complex)
    G = DI.run_ingestion(plan)
    @test G isa DT.GradedComplex
    @test G.grades == [(0.0,), (0.0,), (0.5,)]
    ST = DI.run_ingestion(plan; stage=:simplex_tree)
    @test ST.grade_data == [(0.0,), (0.0,), (0.5,)]
    encoded = DI.encode(plan)
    @test EC.axes_from_encoding(encoded.pi) == ([0.0, 0.5],)
    @test _enc_dims(encoded) == [2, 1]
    @test G.boundaries == DI.encode(data, filtration; stage=:graded_complex).boundaries

    full_axes = ([0.0, 0.25, 0.5, 0.75, 1.0],)
    spec = OPT.FiltrationSpec(kind=:rips, max_dim=1, axes=full_axes)
    as_given = DI.encode(data, spec; pipeline=OPT.PipelineOptions(axes_policy=:as_given))
    # The declared grid contract floors noncritical births: 0.6 is placed at 0.5.
    @test EC.axes_from_encoding(as_given.pi) == full_axes
    @test _enc_dims(as_given) == [2, 2, 1, 1, 1]
    coarse = DI.encode(data, spec;
        pipeline=OPT.PipelineOptions(axes_policy=:coarsen, max_axis_len=3))
    @test EC.axes_from_encoding(coarse.pi) == ([0.0, 0.5, 1.0],)
    @test _enc_dims(coarse) == [2, 1, 1]
    single = DI.encode(data, spec;
        pipeline=OPT.PipelineOptions(axes_policy=:coarsen, max_axis_len=1))
    @test EC.axes_from_encoding(single.pi) == ([0.0],)
    @test _enc_dims(single) == [1]
    @test MD.structure_map(_enc_module(coarse); source=1, target=3) ==
          MD.structure_map(_enc_module(as_given); source=1, target=5)
    # Quantization is not allowed to erase an explicitly requested grid.
    quantified = DI.encode(data, spec; pipeline=OPT.PipelineOptions(eps=0.5))
    @test EC.axes_from_encoding(quantified.pi) == full_axes
    @test _enc_dims(quantified) == [2, 2, 1, 1, 1]

    for stage in (:simplex_tree, :graded_complex, :cochain, :encoded_complex,
                  :module, :fringe, :cohomology_dims, :encoding_result)
        opts = OPT.ConstructionOptions(output_stage=stage)
        configured = DI.plan_ingestion(data, filtration; construction=opts)
        @test DI.planned_stage(configured) == stage
        value = DI.run_ingestion(configured)
        @test stage === :simplex_tree ? value isa DT.SimplexTreeMulti :
              stage === :graded_complex ? value isa DT.GradedComplex :
              stage === :cochain ? value isa MC.ModuleCochainComplex :
              stage === :encoded_complex ? value isa RES.EncodedComplexResult :
              stage === :module ? value isa MD.PModule :
              stage === :fringe ? value isa FF.FringeModule :
              stage === :cohomology_dims ? value isa RES.CohomologyDimsResult :
              value isa RES.EncodingResult
    end
    integer_spec = OPT.FiltrationSpec(kind=:rips, max_dim=1, axes=([0, 1],))
    @test DI.encode(data, integer_spec; pipeline=OPT.PipelineOptions(axis_kind=:zn), stage=:flange) isa TamerOp.FlangeZn.Flange
    @test_throws ArgumentError DI.plan_ingestion(data, filtration;
        pipeline=OPT.PipelineOptions(axes_policy=:as_given))
    for kwargs in ((poset_kind=:regions,), (poset_kind=:grid,), (field=:F2,),
                   (axis_kind=:unknown,), (orientation=(1, 0),), (eps=0,),
                   (eps=-1,), (eps=Inf,), (eps=(0.5, NaN),),
                   (max_axis_len=3,), (axes_policy=:coarsen,),
                   (axes_policy=:coarsen, max_axis_len=0))
        @test_throws ArgumentError OPT.PipelineOptions(; kwargs...)
    end
    bad = OPT.PipelineOptions(nothing, :encoding, nothing, nothing, :regions, nothing, nothing)
    @test_throws ArgumentError DI.plan_ingestion(data, filtration; pipeline=bad)
end

@testset "A14 construction controls reject unused modes and enforce budgets" begin
    data = DT.PointCloud([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    image = DT.ImageNd([0.0 0.0; 0.0 0.0])
    graph = DT.GraphData(3, [(1, 2), (2, 3)])
    for kwargs in ((max_simplices=-1,), (max_edges=-1,), (memory_budget_bytes=-1,))
        @test_throws ArgumentError OPT.ConstructionBudget(; kwargs...)
    end
    for (input, spec) in ((data, OPT.FiltrationSpec(kind=:alpha)),
                          (image, OPT.FiltrationSpec(kind=:cubical)),
                          (graph, OPT.FiltrationSpec(kind=:edge_weighted, edge_weights=[0.0, 1.0])))
        @test_throws ArgumentError DI.encode(input, spec;
            construction=OPT.ConstructionOptions(sparsify=:knn), stage=:graded_complex)
    end
    for (input, spec, counts) in ((image, OPT.FiltrationSpec(kind=:cubical), (4, 4, 1)),
                                  (graph, OPT.FiltrationSpec(kind=:edge_weighted, edge_weights=[0.0, 1.0]), (3, 2)))
        G = DI.encode(input, spec; stage=:graded_complex)
        @test Tuple(DT.cell_counts(G)) == counts
        for budget in (OPT.ConstructionBudget(max_simplices=sum(counts)-1),
                       OPT.ConstructionBudget(max_edges=counts[2]-1),
                       OPT.ConstructionBudget(memory_budget_bytes=1))
            @test_throws ArgumentError DI.encode(input, spec; stage=:graded_complex,
                construction=OPT.ConstructionOptions(; budget))
            @test_throws ArgumentError DI.encode(G, OPT.FiltrationSpec(kind=:graded); stage=:graded_complex,
                construction=OPT.ConstructionOptions(; budget))
        end
    end
end

@testset "A14 data-file controls are applied or rejected" begin
    FIO = TamerOp.DataFileIO
    mktempdir() do dir
        json = joinpath(dir, "points.json")
        SER.save_dataset_json(json, DT.PointCloud([[0.0], [1.0]]))
        for kwargs in ((header=true,), (delimiter=';',), (comment_prefix=nothing,),
                       (missing_policy=:drop_rows,), (cols=(1,),), (u_col=:source,),
                       (v_col=:target,), (weight_col=:weight,))
            options = OPT.DataFileOptions(; kwargs...)
            @test_throws ArgumentError FIO.load_data(json; opts=options)
            @test !FIO.ok(FIO.check_load_data(json; opts=options))
            @test_throws ArgumentError FIO.load_data("unused.ripser";
                format=:ripser_point_cloud, opts=options)
        end
        table = joinpath(dir, "points.csv")
        write(table, "# comment\nx;y\n1;5\n;6\n2;7\n")
        options = OPT.DataFileOptions(kind=:point_cloud, format=:csv, header=true,
            delimiter=';', missing_policy=:drop_rows, cols=(:y,))
        # Missingness in an unselected column does not delete a point.
        points = FIO.load_data(table; opts=options)
        @test DT.point_matrix(points) == reshape([5.0, 6.0, 7.0], 3, 1)
        @test FIO.ok(FIO.check_load_data(table; opts=options))
        graph = joinpath(dir, "graph.csv")
        write(graph, "a,b,w\n1,2,4\n2,3,5\n")
        edges = FIO.load_data(graph; kind=:graph,
            opts=OPT.DataFileOptions(header=true, u_col=:a, v_col=:b, weight_col=:w))
        @test collect(edges.edges) == [(1, 2), (2, 3)]
        @test edges.weights == [4.0, 5.0]
        for (kind, options) in ((:image, OPT.DataFileOptions(cols=(1,))),
                                (:distance_matrix, OPT.DataFileOptions(cols=(1,))),
                                (:point_cloud, OPT.DataFileOptions(u_col=1)),
                                (:graph, OPT.DataFileOptions(cols=(1, 2))))
            @test_throws ArgumentError FIO.load_data(table; kind, opts=options)
        end
    end
end

@testset "A14 per-axis quantization survives canonical pipeline JSON" begin
    G = DT.GradedComplex([[1, 2], [1]], [sparse(reshape([-1, 1], 2, 1))],
        [(0.1, 0.2), (0.6, 0.6), (1.1, 1.1)])
    steps = (0.5, 1.0)
    expected_grades = [(0.0, 0.0), (0.5, 1.0), (1.0, 1.0)]
    expected_axes = ([0.0, 0.5, 1.0], [0.0, 1.0])
    expected_dims = [1, 1, 1, 1, 2, 1]
    field = CM.F3()
    opts = OPT.PipelineOptions(eps=steps, orientation=(1, 1), field=field, poset_kind=:dense)
    for spec_options in (false, true)
        spec = spec_options ?
            OPT.FiltrationSpec(kind=:graded, eps=steps, orientation=(1, 1), field=field, poset_kind=:dense) :
            OPT.FiltrationSpec(kind=:graded)
        pipeline = spec_options ? nothing : opts
        original = DI.encode(G, spec; pipeline)
        original_module = _enc_module(original)
        @test _enc_dims(original) == expected_dims
        @test EC.axes_from_encoding(original.pi) == expected_axes
        @test DI.encode(G, spec; pipeline, stage=:graded_complex).grades == expected_grades
        @test Matrix(MD.structure_map(original_module; source=5, target=6)) ==
              reshape([CM.coerce(field, 1), CM.coerce(field, 1)], 1, 2)
        mktemp() do path, io
            close(io)
            SER.save_pipeline_json(path, G, spec; pipeline_opts=pipeline)
            payload = JSON3.read(read(path, String), Dict{String,Any})
            for validation in (:strict, :trusted)
                G2, spec2, _, opts2 = SER.load_pipeline_json(path; validation)
                @test opts2.eps === steps
                @test opts2.orientation === (1, 1)
                @test opts2.field == field
                @test DI.encode(G2, spec2; pipeline=opts2, stage=:graded_complex).grades == expected_grades
                replay = DI.encode(G2, spec2; pipeline=opts2)
                @test _enc_dims(replay) == expected_dims
                @test EC.axes_from_encoding(replay.pi) == expected_axes
                replay_module = _enc_module(replay)
                for u in 1:6, v in 1:6
                    FF.leq(original.P, u, v) || continue
                    @test MD.structure_map(replay_module; source=u, target=v) ==
                          MD.structure_map(original_module; source=u, target=v)
                end
                if spec_options
                    @test spec2.params.eps === steps
                    @test DI.encode(G2, spec2; stage=:graded_complex).grades == expected_grades
                    spec_replay = _enc_module(DI.encode(G2, spec2))
                    @test spec_replay.dims == expected_dims
                    for u in 1:6, v in 1:6
                        FF.leq(original.P, u, v) || continue
                        @test MD.structure_map(spec_replay; source=u, target=v) ==
                              MD.structure_map(original_module; source=u, target=v)
                    end
                end
            end
            # JSON3 normalizes exactly integral JSON numbers, so decimal
            # spelling does not change a mathematical orientation sign.
            for section in ("pipeline_options", "spec")
                normalized = deepcopy(payload)
                target = section == "spec" ? normalized[section]["params"] : normalized[section]
                target["orientation"] = [1.0, -1.0]
                write(path, JSON3.write(normalized))
                for validation in (:strict, :trusted)
                    _, normalized_spec, _, normalized_opts = SER.load_pipeline_json(path; validation)
                    signs = section == "spec" ? normalized_spec.params.orientation : normalized_opts.orientation
                    @test signs === (1, -1)
                end
            end
            for section in ("pipeline_options", "spec"), orientation in (1, Any[true, 1], [1.5, 1.0], [0, 1], [2, 1])
                malformed = deepcopy(payload)
                target = section == "spec" ? malformed[section]["params"] : malformed[section]
                target["orientation"] = orientation
                write(path, JSON3.write(malformed))
                for validation in (:strict, :trusted)
                    @test_throws ArgumentError SER.load_pipeline_json(path; validation)
                end
            end
            for section in ("pipeline_options", "spec")
                malformed = deepcopy(payload)
                target = section == "spec" ? malformed[section]["params"] : malformed[section]
                target["eps"] = Any[true, 0.5]
                write(path, JSON3.write(malformed))
                for validation in (:strict, :trusted)
                    @test_throws ArgumentError SER.load_pipeline_json(path; validation)
                end
            end
        end
    end
end
