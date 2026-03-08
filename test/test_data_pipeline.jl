using Test
using SparseArrays
using Random
using JSON3

const IR = PosetModules.IndicatorResolutions
const DI = PosetModules.DataIngestion
const DFI = PosetModules.DataFileIO
const SER = PosetModules.Serialization
const FF = PosetModules.FiniteFringe
const FZ = PosetModules.FlangeZn
const PLB = PosetModules.PLBackend
const MC = PosetModules.ModuleComplexes
const MD = PosetModules.Modules
const Inv = PosetModules.Invariants
const FL = PosetModules.FieldLinAlg
const AC = PosetModules.AbelianCategories

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

@inline _enc_module(enc::RES.EncodingResult) = DI.materialize_module(enc.M)
@inline _enc_dims(enc::RES.EncodingResult) = DI.module_dims(enc.M)
@inline _canon_simplex_tree(st::DI.SimplexTreeMulti) = sort([
    (Tuple(collect(DI.simplex_vertices(st, i))), Tuple(collect(DI.simplex_grades(st, i))))
    for i in 1:DI.simplex_count(st)
])

with_fields(FIELDS_FULL) do field
K = CM.coeff_type(field)
@inline c(x) = CM.coerce(field, x)

@testset "Data pipeline: JSON round-trips" begin
    mktemp() do path, io
        close(io)
        data = PosetModules.PointCloud([[0.0], [1.0]])
        SER.save_dataset_json(path, data)
        obj = JSON3.read(read(path, String))
        @test haskey(obj, "points_flat")
        @test !haskey(obj, "points")
        data2 = SER.load_dataset_json(path)
        @test length(data2.points) == 2
        @test data2.points[2][1] == 1.0
    end

    mktempdir() do dir
        data = PosetModules.PointCloud([[0.0], [1.0], [2.0]])
        compact_path = joinpath(dir, "compact.json")
        pretty_path = joinpath(dir, "pretty.json")
        SER.save_dataset_json(compact_path, data; pretty=false)
        SER.save_dataset_json(pretty_path, data; pretty=true)
        @test filesize(compact_path) < filesize(pretty_path)
    end

    mktemp() do path, io
        close(io)
        data = PosetModules.ImageNd([0.0 1.0; 2.0 3.0])
        SER.save_dataset_json(path, data)
        data2 = SER.load_dataset_json(path)
        @test size(data2.data) == (2, 2)
        @test data2.data[2, 2] == 3.0
    end

    mktemp() do path, io
        close(io)
        data = PosetModules.GraphData(3, [(1, 2), (2, 3)]; coords=[[0.0], [1.0], [2.0]], weights=[1.0, 2.0])
        SER.save_dataset_json(path, data)
        obj = JSON3.read(read(path, String))
        @test haskey(obj, "edges_u")
        @test haskey(obj, "edges_v")
        @test !haskey(obj, "edges")
        data2 = SER.load_dataset_json(path)
        @test data2.n == 3
        @test length(data2.edges) == 2
        @test data2.weights[2] == 2.0
    end

    mktemp() do path, io
        close(io)
        write(path, "{\"kind\":\"PointCloud\",\"points\":[[0.0,1.0],[2.0,3.0]]}")
        data2 = SER.load_dataset_json(path)
        @test length(data2.points) == 2
        @test data2.points[1] == [0.0, 1.0]
    end

    mktemp() do path, io
        close(io)
        write(path, "{\"kind\":\"GraphData\",\"n\":3,\"edges\":[[1,2],[2,3]],\"coords\":null,\"weights\":null}")
        data2 = SER.load_dataset_json(path)
        @test data2.n == 3
        @test data2.edges == [(1, 2), (2, 3)]
    end

    mktemp() do path, io
        close(io)
        data = PosetModules.EmbeddedPlanarGraph2D([[0.0, 0.0], [1.0, 0.0]], [(1, 2)])
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
        data = PosetModules.GradedComplex(cells, boundaries, grades)
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
        data = PosetModules.MultiCriticalGradedComplex(cells, boundaries, grades)
        SER.save_dataset_json(path, data)
        data2 = SER.load_dataset_json(path)
        @test data2 isa PosetModules.MultiCriticalGradedComplex
        @test length(data2.grades) == 3
        @test length(data2.grades[3]) == 2
    end

    mktemp() do path, io
        close(io)
        data = PosetModules.PointCloud([[0.0], [1.0]])
        spec = PosetModules.FiltrationSpec(
            kind=:rips,
            max_dim=1,
            construction=PosetModules.ConstructionOptions(; output_stage=:simplex_tree),
        )
        st = PosetModules.encode(data, spec; degree=0)
        @test st isa DI.SimplexTreeMulti
        SER.save_dataset_json(path, st)
        st2 = SER.load_dataset_json(path)
        @test st2 isa DI.SimplexTreeMulti
        @test DI.simplex_count(st2) == DI.simplex_count(st)
        @test collect(DI.simplex_vertices(st2, 3)) == [1, 2]
    end

    mktemp() do path, io
        close(io)
        data = PosetModules.PointCloud([[0.0], [1.0]])
        spec = PosetModules.FiltrationSpec(kind=:rips, max_dim=1, axes=([0.0, 1.0],))
        SER.save_pipeline_json(path, data, spec; degree=1)
        data2, spec2, degree2, popts = SER.load_pipeline_json(path)
        @test length(data2.points) == 2
        @test spec2.kind == :rips
        @test spec2.params[:max_dim] == 1
        @test degree2 == 1
        @test popts isa PosetModules.PipelineOptions
        @test popts.axes_policy == :encoding
        @test popts.poset_kind == :signature
    end

    mktemp() do path, io
        close(io)
        data = PosetModules.PointCloud([[0.0], [1.0], [2.0]])
        spec = PosetModules.FiltrationSpec(
            kind=:rips,
            max_dim=1,
            construction=PosetModules.ConstructionOptions(;
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
        data = PosetModules.PointCloud([[0.0], [1.0]])
        filt = DI.RipsFiltration(
            max_dim=1,
            construction=PosetModules.ConstructionOptions(
                ;
                sparsify=:knn,
                collapse=:none,
                output_stage=:encoding_result,
                budget=(max_simplices=nothing, max_edges=16, memory_budget_bytes=1_000_000),
            ),
        )
        popts = PosetModules.PipelineOptions(;
            orientation=(1,),
            axes_policy=:coarsen,
            axis_kind=:zn,
            eps=0.25,
            poset_kind=:signature,
            field=:F2,
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
        @test popts2 isa PosetModules.PipelineOptions
        @test popts2.orientation == (1,)
        @test popts2.axes_policy == :coarsen
        @test popts2.axis_kind == :zn
        @test popts2.eps == 0.25
        @test popts2.field == :F2
        @test popts2.max_axis_len == 8
    end

    mktemp() do path, io
        close(io)
        data = PosetModules.GraphData(3, [(1, 2), (2, 3)])
        spec = PosetModules.FiltrationSpec(kind=:graph_lower_star, vertex_grades=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        SER.save_pipeline_json(path, data, spec; degree=0)
        _, spec2, degree2, popts = SER.load_pipeline_json(path)
        @test spec2.kind == :graph_lower_star
        @test length(spec2.params[:vertex_grades]) == 3
        @test degree2 == 0
        @test popts.axes_policy == :encoding
    end

    mktemp() do path, io
        close(io)
        data = PosetModules.PointCloud([[0.0], [1.0]])
        spec = PosetModules.FiltrationSpec(kind=:rips, max_dim=1, axes=([0.0, 1.0],))
        enc = PosetModules.encode(data, spec; degree=0)
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
        enc = PosetModules.encode(FG; backend=:zn)
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
        enc = PosetModules.encode(Ups, Downs; backend=:pl_backend)
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
        P = PosetModules.ZnEncoding.SignaturePoset(sig_y, sig_z)
        U = FF.upset_closure(P, trues(FF.nvertices(P)))
        D = FF.downset_closure(P, trues(FF.nvertices(P)))
        H = FF.FringeModule{K}(P, [U], [D], reshape([c(1)], 1, 1); field=field)
        SER.save_encoding_json(path, H)
        obj = JSON3.read(read(path, String))
        @test Int(obj["schema_version"]) == PosetModules.Serialization.ENCODING_SCHEMA_VERSION
        @test obj["U"]["kind"] == "packed_words_v1"
        @test obj["D"]["kind"] == "packed_words_v1"
        @test obj["phi"]["kind"] == "qq_chunks_v1" || obj["phi"]["kind"] == "fp_flat_v1" || obj["phi"]["kind"] == "real_flat_v1"
        @test obj["poset"]["sig_y"]["kind"] == "packed_words_v1"
        @test obj["poset"]["sig_z"]["kind"] == "packed_words_v1"
        @test !haskey(obj["poset"], "leq")
        H2 = SER.load_encoding_json(path; output=:fringe)
        @test H2.P isa PosetModules.ZnEncoding.SignaturePoset
        @test H2.P.sig_y isa PosetModules.ZnEncoding.PackedSignatureRows
        @test H2.P.sig_z isa PosetModules.ZnEncoding.PackedSignatureRows
        @test FF.nvertices(H2.P) == FF.nvertices(P)
        @test FF.leq_matrix(H2.P) == FF.leq_matrix(P)

        SER.save_encoding_json(path, H; include_leq=false)
        H2 = SER.load_encoding_json(path; output=:fringe)
        @test H2.P isa PosetModules.ZnEncoding.SignaturePoset
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

        out_tuple = PosetModules.encode((FG, FG); backend=:zn)
        out_vec = PosetModules.encode(FZ.Flange{K}[FG, FG]; backend=:zn)
        @test length(out_tuple) == 2
        @test length(out_vec) == 2
        @test_throws MethodError PosetModules.encode(FG, FG; backend=:zn)
        @test_throws MethodError PosetModules.encode(FG, FG, FG; backend=:zn)
    end
end

@testset "Data pipeline: DataFileIO file loading" begin
    mktempdir() do dir
        pc_path = joinpath(dir, "points.csv")
        write(pc_path, "x,y,z\n0.0,1.0,2.0\n1.0,2.0,3.0\n2.0,3.0,4.0\n")

        opts = PosetModules.DataFileOptions(; cols=(:x, :z), header=true)
        data = PosetModules.load_data(pc_path; kind=:point_cloud, opts=opts)
        @test data isa PosetModules.PointCloud
        @test length(data.points) == 3
        @test data.points[2] == [1.0, 3.0]
        @test_throws ArgumentError PosetModules.load_data(pc_path; kind=:point_cloud, opts=opts, max_dim=1)

        info = PosetModules.inspect_data_file(pc_path)
        @test info.format == :csv
        @test :point_cloud in info.candidate_kinds
        @test info.ncols == 3
    end

    mktempdir() do dir
        g_path = joinpath(dir, "graph.tsv")
        write(g_path, "u\tv\tw\n1\t2\t1.5\n2\t3\t2.5\n")

        opts = PosetModules.DataFileOptions(;
            header=true,
            u_col=:u,
            v_col=:v,
            weight_col=:w,
        )
        g = PosetModules.load_data(g_path; kind=:graph, format=:tsv, opts=opts)
        @test g isa PosetModules.GraphData
        @test g.n == 3
        @test g.edges == [(1, 2), (2, 3)]
        @test g.weights == [1.5, 2.5]
    end

    mktempdir() do dir
        g_path = joinpath(dir, "graph_no_header.csv")
        write(g_path, "1,2\n2,4\n")
        g = PosetModules.load_data(g_path; kind=:graph, format=:csv)
        @test g isa PosetModules.GraphData
        @test g.edges == [(1, 2), (2, 4)]
        @test g.n == 4
    end

    mktempdir() do dir
        img_path = joinpath(dir, "img.txt")
        write(img_path, "0 1 2\n3 4 5\n")
        img = PosetModules.load_data(img_path; kind=:image, format=:txt)
        @test img isa PosetModules.ImageNd
        @test size(img.data) == (2, 3)
        @test img.data[2, 3] == 5.0
    end

    mktempdir() do dir
        d_path = joinpath(dir, "dist.csv")
        write(d_path, "0,1,2\n1,0,3\n2,3,0\n")
        G = PosetModules.load_data(
            d_path;
            kind=:distance_matrix,
            format=:csv,
            max_dim=1,
            construction=PosetModules.ConstructionOptions(),
        )
        @test G isa PosetModules.GradedComplex
    end

    mktempdir() do dir
        bad_path = joinpath(dir, "ambiguous.csv")
        write(bad_path, "1,2\n3,4\n")
        @test_throws ArgumentError PosetModules.load_data(bad_path)
    end
end

@testset "Data pipeline: encode from data files" begin
    mktempdir() do dir
        pc_path = joinpath(dir, "pts.csv")
        write(pc_path, "x,y\n0.0,0.0\n1.0,0.0\n0.0,1.0\n")
        spec = PosetModules.FiltrationSpec(kind=:rips, max_dim=1)
        opts = PosetModules.DataFileOptions(; header=true, cols=(:x, :y))
        enc_path = PosetModules.encode(pc_path, spec; kind=:point_cloud, file_opts=opts, degree=0)
        enc_mem = PosetModules.encode(PosetModules.PointCloud([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), spec; degree=0)
        @test enc_path isa RES.EncodingResult
        @test enc_mem isa RES.EncodingResult
        @test FF.nvertices(enc_path.P) == FF.nvertices(enc_mem.P)
        @test PosetModules.Results.module_dims(PosetModules.Results.materialize_module(enc_path.M)) ==
              PosetModules.Results.module_dims(PosetModules.Results.materialize_module(enc_mem.M))
    end

    mktempdir() do dir
        json_path = joinpath(dir, "pts.json")
        data = PosetModules.PointCloud([[0.0], [1.0], [2.0]])
        PosetModules.save_dataset_json(json_path, data)
        spec = PosetModules.FiltrationSpec(kind=:rips, max_dim=1)
        enc = PosetModules.encode(json_path, spec; degree=0)
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
    G = PosetModules.GradedComplex(cells, boundaries, grades)

    spec_auto = PosetModules.FiltrationSpec(kind=:graded, orientation=(-1,))
    enc_auto = PosetModules.encode(G, spec_auto; degree=0)
    axes_auto = EC.axes_from_encoding(enc_auto.pi)
    @test axes_auto == (Float64[-2.0, -1.0],)

    # Explicit axes must remain unchanged even with negative orientation.
    explicit_axes = (Float64[-3.0, -2.0, -1.0],)
    spec_explicit = PosetModules.FiltrationSpec(kind=:graded, orientation=(-1,), axes=explicit_axes)
    enc_explicit = PosetModules.encode(G, spec_explicit; degree=0)
    @test EC.axes_from_encoding(enc_explicit.pi) == explicit_axes
end

@testset "Data pipeline: point cloud auto axes honor orientation" begin
    data = PosetModules.PointCloud([[0.0], [1.0]])
    spec = PosetModules.FiltrationSpec(kind=:rips, max_dim=1, orientation=(-1,))
    enc = PosetModules.encode(data, spec; degree=0)
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
    data = PosetModules.PointCloud(pts)
    spec = PosetModules.FiltrationSpec(
        kind=:rips,
        max_dim=2,
        construction=PosetModules.ConstructionOptions(;
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
        st_base = PosetModules.encode(data, spec; degree=0)

        DI._POINTCLOUD_DIM2_PACKED_KERNEL[] = true
        st_fast = PosetModules.encode(data, spec; degree=0)
    finally
        DI._POINTCLOUD_DIM2_PACKED_KERNEL[] = old_dim2
    end

    @test st_base isa DI.SimplexTreeMulti
    @test st_fast isa DI.SimplexTreeMulti
    @test DI.simplex_count(st_fast) == DI.simplex_count(st_base)
    @test st_fast.simplex_dims == st_base.simplex_dims
    @test st_fast.simplex_vertices == st_base.simplex_vertices
    @test st_fast.grade_data == st_base.grade_data

    spec_radius = PosetModules.FiltrationSpec(
        kind=:rips,
        max_dim=2,
        radius=1.5,
        construction=PosetModules.ConstructionOptions(;
            sparsify=:none,
            collapse=:none,
            output_stage=:simplex_tree,
        ),
    )
    st_radius = PosetModules.encode(data, spec_radius; degree=0)
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

    spec_radius_d1 = PosetModules.FiltrationSpec(
        kind=:rips,
        max_dim=1,
        radius=1.1,
        construction=PosetModules.ConstructionOptions(;
            sparsify=:none,
            collapse=:none,
            output_stage=:simplex_tree,
        ),
    )
    st_radius_d1 = PosetModules.encode(data, spec_radius_d1; degree=0)
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
            construction=PosetModules.ConstructionOptions(; sparsify=:radius, budget=(nothing, 4, nothing)),
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
            construction=PosetModules.ConstructionOptions(; collapse=:acyclic),
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

    data = PosetModules.PointCloud([[0.0], [1.0]])
    spec = PosetModules.FiltrationSpec(kind=:rips, max_dim=1, axes=([1.0, 0.0],))
    @test_throws Exception PosetModules.encode(data, spec; degree=0)

    cells = [Int[1]]
    boundaries = SparseMatrixCSC{Int,Int}[]
    grades = [Float64[0.0], Float64[1.0]]
    G = PosetModules.GradedComplex(cells, boundaries, grades)
    spec = PosetModules.FiltrationSpec(kind=:graded, axes=([0.0, 1.0],))
    @test_throws Exception PosetModules.encode(G, spec; degree=0)

    data = PosetModules.PointCloud([[0.0], [1.0]])
    spec = PosetModules.FiltrationSpec(kind=:rips, max_dim=1, axes=([0.0, 1.5],))
    enc = PosetModules.encode(data, spec; degree=0)
    @test FF.nvertices(enc.P) > 0

    spec = PosetModules.FiltrationSpec(kind=:rips, max_dim=1, axes=([0, 1],), axis_kind=:rn)
    @test_throws Exception PosetModules.encode(data, spec; degree=0)

    spec = PosetModules.FiltrationSpec(kind=:rips, max_dim=1, axes=([0.0, 1.0],), axis_kind=:zn)
    @test_throws Exception PosetModules.encode(data, spec; degree=0)
end

@testset "Data pipeline: quantization + coarsen axes" begin
    cells = [Int[1], Int[]]
    boundaries = SparseMatrixCSC{Int,Int}[]
    grades = [Float64[0.05]]
    G = PosetModules.GradedComplex(cells, boundaries, grades)
    spec = PosetModules.FiltrationSpec(kind=:graded, eps=0.1)
    enc = PosetModules.encode(G, spec; degree=0)
    @test EC.axes_from_encoding(enc.pi)[1] == [0.0]

    data = PosetModules.PointCloud([[0.0], [1.0], [2.0], [3.0]])
    spec = PosetModules.FiltrationSpec(kind=:rips, max_dim=1, axes_policy=:coarsen, max_axis_len=2)
    enc = PosetModules.encode(data, spec; degree=0)
    @test length(EC.axes_from_encoding(enc.pi)[1]) == 2
end

@testset "Data pipeline: session cache reuse" begin
    data = PosetModules.PointCloud([[0.0], [1.0]])
    spec = PosetModules.FiltrationSpec(kind=:rips, max_dim=1, axes=([0.0, 1.0],))
    sc = CM.SessionCache()
    enc1 = PosetModules.encode(data, spec; degree=0, cache=sc)
    enc2 = PosetModules.encode(data, spec; degree=0, cache=sc)
    @test enc1.P === enc2.P
    @test typeof(enc1.M) === typeof(enc2.M)
    @test DI.module_dims(enc1.M) == DI.module_dims(enc2.M)
    @test enc1.H === enc2.H

    enc_deg1 = PosetModules.encode(data, spec; degree=1, cache=sc)
    @test enc_deg1.M !== enc1.M

    CM._clear_session_cache!(sc)
    enc3 = PosetModules.encode(data, spec; degree=0, cache=sc)
    @test enc3.P !== enc1.P
end

@testset "Data pipeline: point cloud guardrails + landmark_rips" begin
    data = PosetModules.PointCloud([[0.0], [1.0], [2.0], [3.0]])
    spec = PosetModules.FiltrationSpec(
        kind=:rips,
        max_dim=2,
        construction=PosetModules.ConstructionOptions(; budget=(5, nothing, nothing)),
    )
    @test_throws Exception PosetModules.encode(data, spec; degree=0)

    data_big = PosetModules.PointCloud([[Float64(i)] for i in 1:180])
    spec_precheck = PosetModules.FiltrationSpec(
        kind=:rips,
        max_dim=3,
        construction=PosetModules.ConstructionOptions(; budget=(15_000, nothing, nothing)),
    )
    err = try
        PosetModules.encode(data_big, spec_precheck; degree=0)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("before enumeration", sprint(showerror, err))

    spec_rhomboid_precheck = PosetModules.FiltrationSpec(
        kind=:rhomboid,
        max_dim=3,
        vertex_values=fill(0.0, length(data_big.points)),
        construction=PosetModules.ConstructionOptions(; budget=(15_000, nothing, nothing)),
    )
    err = try
        PosetModules.encode(data_big, spec_rhomboid_precheck; degree=0)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("before enumeration", sprint(showerror, err))

    spec = PosetModules.FiltrationSpec(
        kind=:rips,
        max_dim=1,
        radius=1.1,
        construction=PosetModules.ConstructionOptions(; sparsify=:radius),
    )
    enc = PosetModules.encode(data, spec; degree=0)
    @test FF.nvertices(enc.P) > 0

    spec = PosetModules.FiltrationSpec(kind=:landmark_rips, max_dim=1, landmarks=[1, 3])
    enc = PosetModules.encode(data, spec; degree=0)
    @test FF.nvertices(enc.P) > 0

    # landmark_rips must preserve radius/knn backend knobs through FiltrationSpec
    # -> typed filtration -> FiltrationSpec round-trips.
    spec_lm_radius = PosetModules.FiltrationSpec(
        kind=:landmark_rips,
        max_dim=1,
        landmarks=[1, 2, 3],
        radius=0.6,
        nn_backend=:auto,
        construction=PosetModules.ConstructionOptions(; output_stage=:simplex_tree),
    )
    typed_lm = DI.to_filtration(spec_lm_radius)
    roundtrip_lm = DI._filtration_spec(typed_lm)
    @test get(roundtrip_lm.params, :radius, nothing) == 0.6
    @test get(roundtrip_lm.params, :nn_backend, nothing) == :auto

    st_lm = PosetModules.encode(data, spec_lm_radius; degree=0, stage=:simplex_tree)
    lm_points = [data.points[i] for i in [1, 2, 3]]
    lm_edges_expected, _ = DI._point_cloud_edges_within_radius(lm_points, 0.6)
    @test count(==(1), st_lm.simplex_dims) == length(lm_edges_expected)

    # landmark+radii should normalize to sparse radius construction by default.
    spec_lm_radius_explicit = PosetModules.FiltrationSpec(
        kind=:landmark_rips,
        max_dim=1,
        landmarks=[1, 2, 3],
        radius=0.6,
        construction=PosetModules.ConstructionOptions(; sparsify=:radius, output_stage=:simplex_tree),
    )
    st_lm_explicit = PosetModules.encode(data, spec_lm_radius_explicit; degree=0, stage=:simplex_tree)
    @test _canon_simplex_tree(st_lm) == _canon_simplex_tree(st_lm_explicit)

    # landmark subgraph cache (session-level geometry cache) should reuse packed edge lists.
    ec = CM.EncodingCache()
    packed1 = DI._landmark_radius_subgraph_cached(data.points, [1, 2, 3], 0.6, spec_lm_radius; cache=ec)
    packed2 = DI._landmark_radius_subgraph_cached(data.points, [1, 2, 3], 0.6, spec_lm_radius; cache=ec)
    @test packed1 === packed2
    @test length(packed1.edges) == length(lm_edges_expected)

    spec = PosetModules.FiltrationSpec(
        kind=:rips,
        max_dim=1,
        knn=2,
        construction=PosetModules.ConstructionOptions(; sparsify=:knn, budget=(nothing, 8, nothing)),
    )
    enc = PosetModules.encode(data, spec; degree=0)
    @test FF.nvertices(enc.P) > 0

    spec = PosetModules.FiltrationSpec(
        kind=:rips,
        max_dim=1,
        construction=PosetModules.ConstructionOptions(; budget=(nothing, 1, nothing)),
    )
    @test_throws Exception PosetModules.encode(data, spec; degree=0)
end

@testset "Data pipeline: estimate_ingestion preflight" begin
    data = PosetModules.PointCloud([[0.0], [1.0], [2.0], [3.0]])
    spec = PosetModules.FiltrationSpec(kind=:rips, max_dim=2, axes=([0.0, 1.0],))
    est = DI.estimate_ingestion(data, spec)
    @test est.n_cells_est == big(14)  # C(4,1)+C(4,2)+C(4,3)
    @test est.cell_counts_by_dim == BigInt[4, 6, 4]
    @test est.axis_sizes == (2,)
    @test est.poset_size == big(2)
    @test est.nnz_est == big(24)
    @test est.dense_bytes_est == big(192)

    est_warn = DI.estimate_ingestion(data,
                                               PosetModules.FiltrationSpec(
                                                   kind=:rips,
                                                   max_dim=2,
                                                   construction=(budget=(max_simplices=5, max_edges=nothing, memory_budget_bytes=nothing),),
                                               );
                                               poset_threshold=1)
    @test !isempty(est_warn.warnings)
    @test any(occursin("max_simplices", w) for w in est_warn.warnings)
    @test any(occursin("|P|", w) || occursin("axis_sizes unavailable", w) for w in est_warn.warnings)

    est_edges = DI.estimate_ingestion(
        data,
        PosetModules.FiltrationSpec(kind=:rips, max_dim=1,
                                    construction=(budget=(max_simplices=nothing, max_edges=2, memory_budget_bytes=nothing),)),
    )
    @test any(occursin("max_edges", w) for w in est_edges.warnings)

    @test_throws ArgumentError DI.estimate_ingestion(
        data,
        PosetModules.FiltrationSpec(kind=:delaunay_lower_star, max_dim=2, highdim_policy=:error);
        strict=true,
    )
end

@testset "Data pipeline: construction contract is strict" begin
    data = PosetModules.PointCloud([[0.0], [1.0], [2.0]])
    bad = PosetModules.FiltrationSpec(kind=:rips, max_dim=1, sparse_rips=true, radius=1.0)
    @test_throws ArgumentError PosetModules.encode(data, bad; degree=0)

    @test_throws ErrorException PosetModules.ConstructionOptions(; sparsify=:approx)
    @test_throws ErrorException PosetModules.ConstructionOptions(; collapse=:legacy)
    @test PosetModules.ConstructionOptions(; output_stage=:simplex_tree).output_stage == :simplex_tree
    @test_throws ErrorException PosetModules.ConstructionOptions(; output_stage=:raw)
end

@testset "Data pipeline: point-cloud sparse large-n contract" begin
    n = 5_001
    data = PosetModules.PointCloud([[Float64(i)] for i in 1:n])
    dense_spec = PosetModules.FiltrationSpec(kind=:rips, max_dim=1)
    @test_throws ArgumentError PosetModules.encode(data, dense_spec; degree=0)
end

@testset "Data pipeline: edge-driven sparse point-cloud path" begin
    data = PosetModules.PointCloud([[0.0], [1.0], [2.0], [3.0]])
    dense = PosetModules.FiltrationSpec(
        kind=:rips,
        max_dim=1,
        construction=PosetModules.ConstructionOptions(; output_stage=:graded_complex),
    )
    sparse_radius = PosetModules.FiltrationSpec(
        kind=:rips,
        max_dim=1,
        radius=10.0,
        construction=PosetModules.ConstructionOptions(;
            sparsify=:radius,
            output_stage=:graded_complex,
            budget=(max_simplices=nothing, max_edges=32, memory_budget_bytes=nothing),
        ),
    )
    G_dense = PosetModules.encode(data, dense; degree=0)
    G_sparse = PosetModules.encode(data, sparse_radius; degree=0)
    @test G_dense isa PosetModules.GradedComplex
    @test G_sparse isa PosetModules.GradedComplex
    @test G_dense.cells_by_dim == G_sparse.cells_by_dim
    @test G_dense.grades == G_sparse.grades
end

@testset "Data pipeline: NN backend contract for sparse point-cloud path" begin
    data = PosetModules.PointCloud([[0.0], [0.5], [1.5], [3.0], [4.5]])
    spec_auto = PosetModules.FiltrationSpec(
        kind=:rips,
        max_dim=1,
        knn=2,
        nn_backend=:auto,
        construction=PosetModules.ConstructionOptions(; sparsify=:knn, output_stage=:graded_complex),
    )
    spec_nn = PosetModules.FiltrationSpec(
        kind=:rips,
        max_dim=1,
        knn=2,
        nn_backend=:nearestneighbors,
        construction=PosetModules.ConstructionOptions(; sparsify=:knn, output_stage=:graded_complex),
    )
    spec_apx = PosetModules.FiltrationSpec(
        kind=:rips,
        max_dim=1,
        knn=2,
        nn_backend=:approx,
        nn_approx_candidates=8,
        construction=PosetModules.ConstructionOptions(; sparsify=:knn, output_stage=:graded_complex),
    )
    if DI._have_pointcloud_nn_backend()
        @test DI._pointcloud_nn_backend(spec_auto) == :auto
        G_auto = PosetModules.encode(data, spec_auto; degree=0)
        G_nn = PosetModules.encode(data, spec_nn; degree=0)
        @test G_auto.cells_by_dim == G_nn.cells_by_dim
        @test G_auto.grades == G_nn.grades
        @test PosetModules.encode(data, spec_nn; degree=0) isa PosetModules.GradedComplex
        @test PosetModules.encode(data, spec_apx; degree=0) isa PosetModules.GradedComplex
    else
        @test DI._pointcloud_nn_backend(spec_auto) == :bruteforce
        spec_brute = PosetModules.FiltrationSpec(
            kind=:rips,
            max_dim=1,
            knn=2,
            nn_backend=:bruteforce,
            construction=PosetModules.ConstructionOptions(; sparsify=:knn, output_stage=:graded_complex),
        )
        G_auto = PosetModules.encode(data, spec_auto; degree=0)
        G_brute = PosetModules.encode(data, spec_brute; degree=0)
        @test G_auto.cells_by_dim == G_brute.cells_by_dim
        @test G_auto.grades == G_brute.grades
        @test_throws ArgumentError PosetModules.encode(data, spec_nn; degree=0)
        @test_throws ArgumentError PosetModules.encode(data, spec_apx; degree=0)
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
    pts = PosetModules.PointCloud([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.25, 0.55],
        [0.65, 0.25],
    ])
    construction = PosetModules.ConstructionOptions(; output_stage=:graded_complex)
    spec_auto = PosetModules.FiltrationSpec(kind=:alpha, max_dim=2, delaunay_backend=:auto, construction=construction)
    spec_naive = PosetModules.FiltrationSpec(kind=:alpha, max_dim=2, delaunay_backend=:naive, construction=construction)
    spec_fast = PosetModules.FiltrationSpec(kind=:alpha, max_dim=2, delaunay_backend=:fast, construction=construction)

    @test_throws ArgumentError DI._pointcloud_delaunay_backend(
        PosetModules.FiltrationSpec(kind=:alpha, max_dim=2, delaunay_backend=:bad),
    )

    auto_backend = DI._pointcloud_delaunay_backend(spec_auto)
    if auto_backend == :fast
        spec_fast_st = PosetModules.FiltrationSpec(
            kind=:alpha,
            max_dim=2,
            delaunay_backend=:fast,
            construction=PosetModules.ConstructionOptions(; output_stage=:simplex_tree),
        )
        spec_naive_st = PosetModules.FiltrationSpec(
            kind=:alpha,
            max_dim=2,
            delaunay_backend=:naive,
            construction=PosetModules.ConstructionOptions(; output_stage=:simplex_tree),
        )
        st_fast = PosetModules.encode(pts, spec_fast_st; degree=0)
        st_naive = PosetModules.encode(pts, spec_naive_st; degree=0)
        @test _canon_simplex_tree(st_fast) == _canon_simplex_tree(st_naive)
    else
        @test auto_backend == :naive
        G_auto = PosetModules.encode(pts, spec_auto; degree=0)
        G_naive = PosetModules.encode(pts, spec_naive; degree=0)
        @test G_auto.cells_by_dim == G_naive.cells_by_dim
        @test G_auto.grades == G_naive.grades
        @test_throws ArgumentError PosetModules.encode(pts, spec_fast; degree=0)
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
        G1 = PosetModules.encode(pts, spec_naive; degree=0, stage=:graded_complex)
        G2 = PosetModules.encode(pts, spec_naive; degree=0, stage=:graded_complex)
        @test G1.boundaries[1] === G2.boundaries[1]
        @test G1.boundaries[2] === G2.boundaries[2]
        PosetModules.encode(pts, spec_naive; degree=0)
        n1 = lock(DI._POINTCLOUD_DELAUNAY_CACHE_LOCK) do
            length(DI._POINTCLOUD_DELAUNAY_CACHE)
        end
        PosetModules.encode(pts, spec_naive; degree=0)
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
    pts = PosetModules.PointCloud([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.25, 0.55],
        [0.65, 0.25],
    ])
    for md in (0, 1, 2)
        spec = PosetModules.FiltrationSpec(
            kind=:alpha,
            max_dim=md,
            construction=PosetModules.ConstructionOptions(; output_stage=:graded_complex),
        )
        G = PosetModules.encode(pts, spec; degree=0)
        @test G isa PosetModules.GradedComplex
        @test length(G.cells_by_dim) == md + 1
        @test length(G.boundaries) == md
    end
end

@testset "Data pipeline: Delaunay packed simplices are canonical/unique" begin
    DI._have_pointcloud_delaunay_backend() || begin
        @test true
        return
    end

    rng = Random.MersenneTwister(0xD4B4)
    pts = PosetModules.PointCloud([randn(rng, 2) for _ in 1:128])
    spec = PosetModules.FiltrationSpec(kind=:alpha, max_dim=2, delaunay_backend=:fast)
    packed = DI._packed_delaunay_simplices(pts.points, spec; max_dim=2)

    @test length(packed.edge_radius) == length(packed.edges)
    @test length(packed.tri_radius) == length(packed.triangles)
    @test all(e -> e[1] < e[2], packed.edges)
    @test all(t -> (t[1] < t[2] && t[2] < t[3]), packed.triangles)
    @test length(unique(packed.edges)) == length(packed.edges)
    @test length(unique(packed.triangles)) == length(packed.triangles)
end

@testset "Data pipeline: construction output_stage routing" begin
    data = PosetModules.PointCloud([[0.0], [1.0]])

    filt_st = DI.RipsFiltration(
        max_dim=1,
        construction=PosetModules.ConstructionOptions(; output_stage=:simplex_tree),
    )
    ST = PosetModules.encode(data, filt_st; degree=0)
    @test ST isa DI.SimplexTreeMulti
    @test DI.simplex_count(ST) == 3
    @test DI.max_simplex_dim(ST) == 1
    @test collect(DI.simplex_vertices(ST, 1)) == [1]
    @test collect(DI.simplex_vertices(ST, 2)) == [2]
    @test collect(DI.simplex_vertices(ST, 3)) == [1, 2]
    @test collect(DI.simplex_grades(ST, 3)) == [(1.0,)]

    filt_gc = DI.RipsFiltration(
        max_dim=1,
        construction=PosetModules.ConstructionOptions(; output_stage=:graded_complex),
    )
    G = PosetModules.encode(data, filt_gc; degree=0)
    @test G isa PosetModules.GradedComplex

    filt_cc = DI.RipsFiltration(
        max_dim=1,
        construction=PosetModules.ConstructionOptions(; output_stage=:cochain),
    )
    C = PosetModules.encode(data, filt_cc; degree=0)
    @test C isa MC.ModuleCochainComplex
    for (u, v) in FF.cover_edges(C.terms[1].Q)
        @test C.terms[1].edge_maps[u, v] isa AbstractMatrix
    end
    @test C.diffs[1].comps[1] isa AbstractMatrix

    filt_mod = DI.RipsFiltration(
        max_dim=1,
        construction=PosetModules.ConstructionOptions(; output_stage=:module),
    )
    M = PosetModules.encode(data, filt_mod; degree=0)
    @test M isa MD.PModule

    spec_mod = PosetModules.FiltrationSpec(
        kind=:rips,
        max_dim=1,
        construction=(sparsify=:none, collapse=:none, output_stage=:module,
                      budget=(max_simplices=nothing, max_edges=nothing, memory_budget_bytes=nothing)),
    )
    M2 = PosetModules.encode(data, spec_mod; degree=0)
    @test M2 isa MD.PModule
end

@testset "Data pipeline: simplex-tree stage rejects non-simplicial cubical ingestion" begin
    img = PosetModules.ImageNd([0.0 1.0; 2.0 3.0])
    spec = PosetModules.FiltrationSpec(
        kind=:lower_star,
        construction=PosetModules.ConstructionOptions(; output_stage=:simplex_tree),
    )
    @test_throws ArgumentError PosetModules.encode(img, spec; degree=0)
end

@testset "Data pipeline: flange emission (Zn)" begin
    data = PosetModules.PointCloud([[0.0], [1.0]])
    spec = PosetModules.FiltrationSpec(kind=:rips, max_dim=1, axes=([0, 1],), axis_kind=:zn)
    FG = PosetModules.encode(data, spec; degree=0, stage=:flange)
    @test FG.n == 1

    spec = PosetModules.FiltrationSpec(kind=:rips, max_dim=1, axes=([0.0, 1.0],), axis_kind=:rn)
    @test_throws Exception PosetModules.encode(data, spec; degree=0, stage=:flange)

    spec = PosetModules.FiltrationSpec(kind=:rips, max_dim=1, axes=([0.0, 1.0],))
    @test_throws Exception PosetModules.encode(data, spec; degree=0, stage=:flange)
end

@testset "Data pipeline: graded complex escape hatch" begin
    # single vertex
    cells = [Int[1]]
    boundaries = SparseMatrixCSC{Int,Int}[]
    grades = [Float64[0.0]]
    G = PosetModules.GradedComplex(cells, boundaries, grades)
    spec = PosetModules.FiltrationSpec(kind=:graded, axes=([0.0, 1.0],))
    enc = PosetModules.encode(G, spec; degree=0)
    @test DI.module_dims(enc.M) == [1, 1]

    # single edge between two vertices
    cells = [Int[1, 2], Int[1]]
    B1 = sparse([1, 2], [1, 1], [1, -1], 2, 1)
    boundaries = [B1]
    grades = [Float64[0.0], Float64[0.0], Float64[1.0]]
    G = PosetModules.GradedComplex(cells, boundaries, grades)
    enc = PosetModules.encode(G, spec; degree=0)
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
    G = PosetModules.GradedComplex(cells, boundaries, grades)
    spec2 = PosetModules.FiltrationSpec(kind=:graded, axes=([0.0, 1.0, 2.0],))
    enc1 = PosetModules.encode(G, spec2; degree=1)
    @test DI.module_dims(enc1.M) == [0, 1, 0]
end

@testset "Data pipeline: simplex-tree escape hatch" begin
    data = PosetModules.PointCloud([[0.0], [1.0], [2.0]])
    spec_st = PosetModules.FiltrationSpec(
        kind=:rips,
        max_dim=1,
        construction=PosetModules.ConstructionOptions(; output_stage=:simplex_tree),
    )
    st = PosetModules.encode(data, spec_st; degree=0)
    @test st isa DI.SimplexTreeMulti
    @test DI.simplex_count(st) == 6
    @test DI.max_simplex_dim(st) == 1

    spec_enc = PosetModules.FiltrationSpec(kind=:rips, axes=([0.0, 1.0, 2.0],))
    enc = PosetModules.encode(st, spec_enc; degree=0)
    @test enc isa RES.EncodingResult
    @test DI.module_dims(enc.M) == [3, 1, 1]
end

@testset "Data pipeline: simplex-tree cochain/module parity" begin
    data = PosetModules.PointCloud([[0.0], [1.0], [2.0]])
    st = PosetModules.encode(
        data,
        PosetModules.FiltrationSpec(
            kind=:rips,
            max_dim=1,
            construction=PosetModules.ConstructionOptions(; output_stage=:simplex_tree),
        );
        degree=0,
    )
    @test st isa DI.SimplexTreeMulti

    spec_mod = PosetModules.FiltrationSpec(
        kind=:rips,
        max_dim=1,
        axes=([0.0, 1.0, 2.0],),
        construction=PosetModules.ConstructionOptions(; output_stage=:module),
    )
    M_raw = PosetModules.encode(data, spec_mod; degree=0)
    M_tree = PosetModules.encode(st, spec_mod; degree=0)
    @test M_tree isa MD.PModule
    @test M_tree.dims == M_raw.dims
    for (u, v) in FF.cover_edges(M_raw.Q)
        @test Array(M_tree.edge_maps[u, v]) == Array(M_raw.edge_maps[u, v])
    end

    spec_cc = PosetModules.FiltrationSpec(
        kind=:rips,
        max_dim=1,
        axes=([0.0, 1.0, 2.0],),
        construction=PosetModules.ConstructionOptions(; output_stage=:cochain),
    )
    C_raw = PosetModules.encode(data, spec_cc; degree=0)
    C_tree = PosetModules.encode(st, spec_cc; degree=0)
    @test C_tree isa MC.ModuleCochainComplex
    @test length(C_tree.terms) == length(C_raw.terms)
    @test C_tree.terms[1].dims == C_raw.terms[1].dims
    @test C_tree.terms[2].dims == C_raw.terms[2].dims
    @test all(Array(C_tree.diffs[1].comps[i]) == Array(C_raw.diffs[1].comps[i]) for i in eachindex(C_raw.diffs[1].comps))
end

@testset "Data pipeline: lazy default parity vs explicit cochain" begin
    data = PosetModules.PointCloud([[0.0], [1.0], [2.0]])
    spec_default = PosetModules.FiltrationSpec(kind=:rips, max_dim=1, axes=([0.0, 1.0, 2.0],))
    enc_lazy = PosetModules.encode(data, spec_default; degree=0)

    spec_cochain = PosetModules.FiltrationSpec(
        kind=:rips,
        max_dim=1,
        axes=([0.0, 1.0, 2.0],),
        construction=PosetModules.ConstructionOptions(; output_stage=:cochain),
    )
    C_full = PosetModules.encode(data, spec_cochain; degree=0)
    M_full = MC.cohomology_module(C_full, 0)

    M_lazy = _enc_module(enc_lazy)
    @test M_lazy.dims == M_full.dims
    for (u, v) in FF.cover_edges(M_lazy.Q)
        @test Array(M_lazy.edge_maps[u, v]) == Array(M_full.edge_maps[u, v])
    end
end

@testset "Data pipeline: graded-complex lazy parity vs explicit cochain" begin
    cells = [Int[1, 2], Int[1]]
    boundaries = [sparse([1, 2], [1, 1], [1, -1], 2, 1)]
    grades = [Float64[0.0], Float64[0.0], Float64[1.0]]
    G = PosetModules.GradedComplex(cells, boundaries, grades)

    spec_default = PosetModules.FiltrationSpec(kind=:graded, axes=([0.0, 1.0],))
    enc_lazy = PosetModules.encode(G, spec_default; degree=0)
    axes = ([0.0, 1.0],)
    P = DI.poset_from_axes(axes)
    C_full = DI.cochain_complex_from_graded_complex(G, P, axes; field=CM.QQField())
    M_full = MC.cohomology_module(C_full, 0)

    M_lazy = _enc_module(enc_lazy)
    @test M_lazy.dims == M_full.dims
    for (u, v) in FF.cover_edges(M_lazy.Q)
        @test Array(M_lazy.edge_maps[u, v]) == Array(M_full.edge_maps[u, v])
    end
end

@testset "Data pipeline: low-dim H0 fast path parity" begin
    data = PosetModules.PointCloud([[0.0], [0.4], [0.9], [1.3]])
    spec_gc = PosetModules.FiltrationSpec(
        kind=:rips,
        max_dim=1,
        axes=([0.0, 0.5, 1.0, 1.5],),
        construction=OPT.ConstructionOptions(; output_stage=:graded_complex),
    )
    G = PosetModules.encode(data, spec_gc; degree=0)
    axes = spec_gc.params[:axes]
    P = DI.poset_from_axes(axes)

    L_fast = DI._lazy_cochain_complex_from_graded_complex(G, P, axes; field=CM.QQField())
    M_fast = DI._cohomology_module_from_lazy(L_fast, 0)
    L_generic = DI._lazy_cochain_complex_from_graded_complex(G, P, axes; field=CM.QQField())
    M_generic = DI._cohomology_module_from_lazy_generic(L_generic, 0)
    @test M_fast.dims == M_generic.dims
    for (u, v) in FF.cover_edges(M_fast.Q)
        @test Array(M_fast.edge_maps[u, v]) == Array(M_generic.edge_maps[u, v])
    end

    # Non-edge boundary columns still agree with the generic local cohomology path.
    cells_bad = [Int[1, 2], Int[1]]
    boundaries_bad = [sparse([1], [1], [1], 2, 1)]
    grades_bad = [Float64[0.0], Float64[0.0], Float64[1.0]]
    G_bad = PosetModules.GradedComplex(cells_bad, boundaries_bad, grades_bad)
    axes_bad = ([0.0, 1.0],)
    P_bad = DI.poset_from_axes(axes_bad)
    L_bad = DI._lazy_cochain_complex_from_graded_complex(G_bad, P_bad, axes_bad; field=CM.QQField())
    M_bad = DI._cohomology_module_from_lazy(L_bad, 0)
    L_bad_generic = DI._lazy_cochain_complex_from_graded_complex(G_bad, P_bad, axes_bad; field=CM.QQField())
    M_bad_generic = DI._cohomology_module_from_lazy_generic(L_bad_generic, 0)
    @test M_bad.dims == M_bad_generic.dims
    for (u, v) in FF.cover_edges(M_bad.Q)
        @test Array(M_bad.edge_maps[u, v]) == Array(M_bad_generic.edge_maps[u, v])
    end
end

@testset "Data pipeline: H0 kernel path parity for max_dim>1" begin
    data = PosetModules.PointCloud([[0.0], [0.25], [0.5], [0.75], [1.0]])
    spec_gc = PosetModules.FiltrationSpec(
        kind=:rips,
        max_dim=2,
        axes=([0.0, 0.4, 0.8, 1.2],),
        construction=OPT.ConstructionOptions(; output_stage=:graded_complex),
    )
    G = PosetModules.encode(data, spec_gc; degree=0)
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
            @test Array(M_fast.edge_maps[u, v]) == Array(M_generic.edge_maps[u, v])
        end
    finally
        DI._H0_CHAIN_SWEEP_FASTPATH[] = old_chain
        DI._H0_UNIONFIND_MIN_POS_VERTICES[] = old_min_pos
        DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_VERTICES[] = old_min_v
        DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_EDGES[] = old_min_e
    end
end

@testset "Data pipeline: H1 cokernel fast path parity" begin
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
    G = PosetModules.GradedComplex([cells0, cells1], [B], grades)
    axes = (collect(range(0.0, stop=1.0, length=20)),)
    P = DI.poset_from_axes(axes)

    old_h1 = DI._H1_COKERNEL_FASTPATH[]
    try
        DI._H1_COKERNEL_FASTPATH[] = true
        L_fast = DI._lazy_cochain_complex_from_graded_complex(G, P, axes; field=CM.F2())
        M_fast = DI._cohomology_module_from_lazy(L_fast, 1)

        DI._H1_COKERNEL_FASTPATH[] = false
        L_generic = DI._lazy_cochain_complex_from_graded_complex(G, P, axes; field=CM.F2())
        M_generic = DI._cohomology_module_from_lazy(L_generic, 1)

        @test M_fast.dims == M_generic.dims
        for (u, v) in FF.cover_edges(M_fast.Q)
            @test PosetModules.FieldLinAlg.rank(CM.F2(), M_fast.edge_maps[u, v]) ==
                  PosetModules.FieldLinAlg.rank(CM.F2(), M_generic.edge_maps[u, v])
        end
    finally
        DI._H1_COKERNEL_FASTPATH[] = old_h1
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
    G = PosetModules.GradedComplex([cells0, cells1], [B], grades)
    axes = (collect(range(0.0, stop=1.0, length=20)),)
    P = DI.poset_from_axes(axes)

    old_h1 = DI._H1_COKERNEL_FASTPATH[]
    old_solve = AC._FAST_SOLVE_NO_CHECK[]
    try
        DI._H1_COKERNEL_FASTPATH[] = false

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
        DI._H1_COKERNEL_FASTPATH[] = old_h1
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
        data = PosetModules.PointCloud([[0.0], [0.3], [0.8], [1.1], [1.6], [2.0]])
        spec_gc = PosetModules.FiltrationSpec(
            kind=:rips,
            max_dim=1,
            axes=([0.0, 0.4, 0.8, 1.2, 1.6, 2.0],),
            construction=OPT.ConstructionOptions(; output_stage=:graded_complex),
        )
        G = PosetModules.encode(data, spec_gc; degree=0)
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
            @test Array(M_fast.edge_maps[u, v]) == Array(M_base.edge_maps[u, v])
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
    G = PosetModules.GradedComplex(cells, boundaries, grades)
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
    G = PosetModules.GradedComplex(cells, boundaries, grades)
    axes = (collect(range(0.0, stop=1.0, length=28)),)
    P = DI.poset_from_axes(axes)

    old_local = DI._COHOMOLOGY_DEGREE_LOCAL_FASTPATH[]
    old_all_t = DI._COHOMOLOGY_DEGREE_LOCAL_ALL_T[]
    old_t1_min_pos = DI._COHOMOLOGY_DEGREE_LOCAL_T1_MIN_POS_VERTICES[]
    old_t1_min_dim1 = DI._COHOMOLOGY_DEGREE_LOCAL_T1_MIN_TOTAL_ACTIVE_DIM1[]
    old_t1_min_dim2 = DI._COHOMOLOGY_DEGREE_LOCAL_T1_MIN_TOTAL_ACTIVE_DIM2[]
    old_h1 = DI._H1_COKERNEL_FASTPATH[]
    try
        DI._H1_COKERNEL_FASTPATH[] = false
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
        DI._H1_COKERNEL_FASTPATH[] = old_h1
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
    G = PosetModules.GradedComplex(cells, boundaries, grades)
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
    G = PosetModules.GradedComplex(cells, boundaries, grades)
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
    spec = PosetModules.FiltrationSpec(
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
        data = PosetModules.GraphData(
            n,
            edges;
            coords=[[Float64(i), Float64(mod(i, 3))] for i in 1:n],
        )
        vals = [Float64(i) / n for i in 1:n]
        spec2 = PosetModules.FiltrationSpec(
            kind=:clique_lower_star,
            max_dim=2,
            vertex_values=vals,
            simplex_agg=:max,
            construction=OPT.ConstructionOptions(; collapse=:none, sparsify=:none),
        )
        DI._GRAPH_CLIQUE_ENUM_MODE[] = :intersection
        st_inter = PosetModules.encode(data, spec2; degree=0, cache=:auto, stage=:simplex_tree)
        DI._GRAPH_CLIQUE_ENUM_MODE[] = :combinations
        st_comb = PosetModules.encode(data, spec2; degree=0, cache=:auto, stage=:simplex_tree)
        @test _canon_simplex_tree(st_inter) == _canon_simplex_tree(st_comb)
    finally
        DI._GRAPH_PACKED_EDGELIST_BACKEND[] = old_flag
        DI._GRAPH_BACKEND_WINNER_CACHE_ENABLED[] = old_cache
        DI._GRAPH_BACKEND_WINNER_CACHE_PROBE[] = old_probe
        DI._GRAPH_CLIQUE_ENUM_MODE[] = old_mode
    end
end

@testset "Data pipeline: cohomology_dims stage parity + invariant shortcut" begin
    data = PosetModules.PointCloud([[0.0], [0.3], [0.8], [1.1], [1.6], [2.0]])
    spec = PosetModules.FiltrationSpec(
        kind=:rips,
        max_dim=1,
        axes=([0.0, 0.4, 0.8, 1.2, 1.6, 2.0],),
        construction=OPT.ConstructionOptions(; output_stage=:encoding_result),
    )
    d = PosetModules.encode(data, spec; degree=0, stage=:cohomology_dims, cache=:auto)
    M = PosetModules.encode(data, spec; degree=0, stage=:module, cache=:auto)
    enc = PosetModules.encode(data, spec; degree=0, stage=:encoding_result, cache=:auto)

    @test d isa RES.CohomologyDimsResult
    @test FF.nvertices(d.P) == FF.nvertices(M.Q)
    for u in 1:FF.nvertices(d.P), v in 1:FF.nvertices(d.P)
        @test FF.leq(d.P, u, v) == FF.leq(M.Q, u, v)
    end
    @test d.dims == M.dims

    h_mod = Inv.restricted_hilbert(M)
    h_dims = PosetModules.restricted_hilbert(d)
    @test h_mod == h_dims

    # Same dims-only invariant should work on both result types.
    h_enc = PosetModules.invariant(enc; which=:restricted_hilbert).value
    h_cdr = PosetModules.invariant(d; which=:restricted_hilbert).value
    @test h_cdr == h_enc

    e_opts = OPT.InvariantOptions(; axes=([0.0, 0.8, 1.6],), axes_policy=:as_given, threads=false)
    e_enc = PosetModules.invariant(enc; which=:euler_surface, opts=e_opts).value
    e_cdr = PosetModules.invariant(d; which=:euler_surface, opts=e_opts).value
    @test e_cdr == e_enc

    # Unsupported invariants fail cleanly on dims-only objects.
    @test_throws ErrorException PosetModules.invariant(d; which=:rank_invariant)
end

@testset "Data pipeline: encoding_result lazy module parity" begin
    data = PosetModules.PointCloud([[0.0], [0.25], [0.7], [1.1], [1.6], [2.0]])
    spec = PosetModules.FiltrationSpec(
        kind=:rips,
        max_dim=2,
        construction=OPT.ConstructionOptions(; output_stage=:encoding_result),
    )
    old_lazy = DI._ENCODING_RESULT_LAZY_MODULE[]
    try
        DI._ENCODING_RESULT_LAZY_MODULE[] = true
        enc_lazy = PosetModules.encode(data, spec; degree=1, stage=:encoding_result, cache=:auto)
        @test enc_lazy isa RES.EncodingResult
        @test enc_lazy.M isa DI._LazyEncodedModule

        M_lazy = PosetModules.Workflow.pmodule(enc_lazy)
        @test M_lazy isa MD.PModule
        @test PosetModules.Workflow.pmodule(enc_lazy) === M_lazy

        DI._ENCODING_RESULT_LAZY_MODULE[] = false
        enc_eager = PosetModules.encode(data, spec; degree=1, stage=:encoding_result, cache=:auto)
        @test enc_eager.M isa MD.PModule

        h_lazy = PosetModules.restricted_hilbert(enc_lazy)
        h_eager = PosetModules.restricted_hilbert(enc_eager)
        @test h_lazy == h_eager

        e_opts = OPT.InvariantOptions(; axes=([0.0, 0.8, 1.6],), axes_policy=:as_given, threads=false)
        e_lazy = PosetModules.euler_surface(enc_lazy; opts=e_opts)
        e_eager = PosetModules.euler_surface(enc_eager; opts=e_opts)
        @test e_lazy == e_eager
    finally
        DI._ENCODING_RESULT_LAZY_MODULE[] = old_lazy
    end
end

@testset "Data pipeline: degree-local all-t keeps local term materialization" begin
    data = PosetModules.PointCloud([[0.0], [0.3], [0.8], [1.4], [1.9], [2.2]])
    spec = PosetModules.FiltrationSpec(
        kind=:rips,
        max_dim=3,
        construction=OPT.ConstructionOptions(; output_stage=:graded_complex),
    )
    G = PosetModules.encode(data, spec; degree=0, stage=:graded_complex, cache=:auto)
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

        data = PosetModules.PointCloud([[0.0], [0.4], [0.9], [1.3], [1.8]])
        spec_gc = PosetModules.FiltrationSpec(
            kind=:rips,
            max_dim=1,
            axes=([0.0, 0.5, 1.0, 1.5, 2.0],),
            construction=OPT.ConstructionOptions(; output_stage=:graded_complex),
        )
        G = PosetModules.encode(data, spec_gc; degree=0)
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
        G_bad = PosetModules.GradedComplex(cells_bad, boundaries_bad, grades_bad)
        axes_bad = ([0.0, 1.0],)
        P_bad = DI.poset_from_axes(axes_bad)
        L_bad = DI._lazy_cochain_complex_from_graded_complex(G_bad, P_bad, axes_bad; field=CM.F2())
        M_bad = DI._cohomology_module_from_lazy(L_bad, 0)
        L_bad_generic = DI._lazy_cochain_complex_from_graded_complex(G_bad, P_bad, axes_bad; field=CM.F2())
        M_bad_generic = DI._cohomology_module_from_lazy_generic(L_bad_generic, 0)
        @test M_bad.dims == M_bad_generic.dims
        for (u, v) in FF.cover_edges(M_bad.Q)
            @test Array(M_bad.edge_maps[u, v]) == Array(M_bad_generic.edge_maps[u, v])
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

        data = PosetModules.PointCloud([[0.0], [0.4], [0.9], [1.3], [1.7]])
        spec = PosetModules.FiltrationSpec(
            kind=:rips,
            max_dim=1,
            axes=([0.0, 0.5, 1.0, 1.5, 2.0],),
            construction=OPT.ConstructionOptions(; output_stage=:encoding_result),
        )
        DI._H0_CHAIN_SWEEP_FASTPATH[] = false
        M_base = PosetModules.encode(data, spec; degree=0, stage=:module, cache=:auto)
        DI._H0_CHAIN_SWEEP_FASTPATH[] = true
        M_fast = PosetModules.encode(data, spec; degree=0, stage=:module, cache=:auto)
        @test M_fast.dims == M_base.dims
        for (u, v) in FF.cover_edges(M_fast.Q)
            @test Array(M_fast.edge_maps[u, v]) == Array(M_base.edge_maps[u, v])
        end

        edges = [(1, 2), (2, 3), (3, 4), (4, 5), (1, 5)]
        weights = [0.2, 0.4, 0.7, 1.0, 1.3]
        gdata = PosetModules.GraphData(5, edges; weights=weights)
        gspec = PosetModules.FiltrationSpec(
            kind=:graph_weight_threshold,
            max_dim=1,
            construction=OPT.ConstructionOptions(; output_stage=:encoding_result),
        )
        DI._H0_CHAIN_SWEEP_FASTPATH[] = false
        Mg_base = PosetModules.encode(gdata, gspec; degree=0, stage=:module, cache=:auto)
        DI._H0_CHAIN_SWEEP_FASTPATH[] = true
        Mg_fast = PosetModules.encode(gdata, gspec; degree=0, stage=:module, cache=:auto)
        @test Mg_fast.dims == Mg_base.dims
        for (u, v) in FF.cover_edges(Mg_fast.Q)
            @test Array(Mg_fast.edge_maps[u, v]) == Array(Mg_base.edge_maps[u, v])
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
        data = PosetModules.PointCloud(pts)
        cons = OPT.ConstructionOptions(; sparsify=:knn, output_stage=:simplex_tree)
        spec_bf = PosetModules.FiltrationSpec(
            kind=:rips,
            max_dim=1,
            knn=8,
            nn_backend=:bruteforce,
            construction=cons,
        )
        spec_nn = PosetModules.FiltrationSpec(
            kind=:rips,
            max_dim=1,
            knn=8,
            nn_backend=:nearestneighbors,
            construction=cons,
        )
        spec_ap = PosetModules.FiltrationSpec(
            kind=:rips,
            max_dim=1,
            knn=8,
            nn_backend=:approx,
            nn_approx_candidates=n, # candidate=n forces exact parity path
            construction=cons,
        )

        st_bf = PosetModules.encode(data, spec_bf; degree=0, stage=:simplex_tree, cache=:auto)
        st_nn = PosetModules.encode(data, spec_nn; degree=0, stage=:simplex_tree, cache=:auto)
        st_ap = PosetModules.encode(data, spec_ap; degree=0, stage=:simplex_tree, cache=:auto)

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
    data = PosetModules.GradedComplex([cells0, cells1], [B], grades)
    axes = (collect(range(0.0, stop=1.0, length=40)),)
    spec = PosetModules.FiltrationSpec(
        kind=:graded,
        axes=axes,
        construction=OPT.ConstructionOptions(; output_stage=:encoding_result),
    )
    pipeline = PosetModules.PipelineOptions(field=CM.F2())

    old_fast = DI._COHOMOLOGY_DIMS_MONOTONE_RANK_FASTPATH[]
    old_inc = DI._COHOMOLOGY_DIMS_MONOTONE_INCREMENTAL_RANK[]
    try
        DI._COHOMOLOGY_DIMS_MONOTONE_RANK_FASTPATH[] = true
        DI._COHOMOLOGY_DIMS_MONOTONE_INCREMENTAL_RANK[] = true
        d_inc = PosetModules.encode(data, spec; degree=1, stage=:cohomology_dims, pipeline=pipeline, cache=:auto)
        DI._COHOMOLOGY_DIMS_MONOTONE_INCREMENTAL_RANK[] = false
        d_base = PosetModules.encode(data, spec; degree=1, stage=:cohomology_dims, pipeline=pipeline, cache=:auto)
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
        data = PosetModules.PointCloud([[0.0], [0.25], [0.5], [0.75], [1.0], [1.25]])
        spec_gc = PosetModules.FiltrationSpec(
            kind=:rips,
            max_dim=2,
            axes=([0.0, 0.3, 0.6, 0.9, 1.2],),
            construction=OPT.ConstructionOptions(; output_stage=:graded_complex),
        )
        G = PosetModules.encode(data, spec_gc; degree=0)
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
    data = PosetModules.PointCloud([[0.0], [0.4], [0.9], [1.3]])
    spec_gc = PosetModules.FiltrationSpec(
        kind=:rips,
        max_dim=1,
        axes=([0.0, 0.5, 1.0, 1.5],),
        construction=OPT.ConstructionOptions(; output_stage=:graded_complex),
    )
    G = PosetModules.encode(data, spec_gc; degree=0)
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
        data = PosetModules.PointCloud([[0.0], [0.25], [0.5], [0.75], [1.0], [1.25]])
        spec = PosetModules.FiltrationSpec(
            kind=:rips,
            max_dim=2,
            construction=OPT.ConstructionOptions(; sparsify=:none, output_stage=:graded_complex),
        )
        DI._POINTCLOUD_STREAM_DIST_NONSPARSE[] = true
        G_stream = PosetModules.encode(data, spec; degree=0, stage=:graded_complex)
        DI._POINTCLOUD_STREAM_DIST_NONSPARSE[] = false
        G_packed = PosetModules.encode(data, spec; degree=0, stage=:graded_complex)
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
        data = PosetModules.PointCloud([[0.0], [0.2], [0.55], [0.9], [1.3], [1.8]])
        spec = PosetModules.FiltrationSpec(
            kind=:rips_density,
            max_dim=1,
            radius=0.75,
            density_k=2,
            nn_backend=:bruteforce,
            construction=OPT.ConstructionOptions(; sparsify=:none, output_stage=:graded_complex),
        )
        DI._POINTCLOUD_LOWDIM_RADIUS_STREAMING[] = true
        G_stream = PosetModules.encode(data, spec; degree=0, stage=:graded_complex)
        DI._POINTCLOUD_LOWDIM_RADIUS_STREAMING[] = false
        G_dense = PosetModules.encode(data, spec; degree=0, stage=:graded_complex)
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
        data = PosetModules.GraphData(
            6,
            [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (3, 5), (4, 5), (4, 6), (5, 6)],
        )
        vg = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8]
        spec = PosetModules.FiltrationSpec(
            kind=:clique_lower_star,
            max_dim=2,
            vertex_grades=vg,
            simplex_agg=:max,
            construction=OPT.ConstructionOptions(; output_stage=:graded_complex),
        )
        DI._GRAPH_CLIQUE_ENUM_MODE[] = :intersection
        G_fast = PosetModules.encode(data, spec; degree=0, stage=:graded_complex)
        DI._GRAPH_CLIQUE_ENUM_MODE[] = :combinations
        G_base = PosetModules.encode(data, spec; degree=0, stage=:graded_complex)
        DI._GRAPH_CLIQUE_ENUM_MODE[] = :auto
        G_auto = PosetModules.encode(data, spec; degree=0, stage=:graded_complex)
        @test G_fast.cells_by_dim == G_base.cells_by_dim
        @test G_fast.boundaries == G_base.boundaries
        @test G_fast.grades == G_base.grades
        @test G_auto.cells_by_dim == G_base.cells_by_dim
        @test G_auto.boundaries == G_base.boundaries
        @test G_auto.grades == G_base.grades

        w = [1.0 + 0.1 * i for i in eachindex(data.edges)]
        spec_w = PosetModules.FiltrationSpec(
            kind=:graph_weight_threshold,
            lift=:clique,
            max_dim=2,
            edge_weights=w,
            construction=OPT.ConstructionOptions(; output_stage=:graded_complex),
        )
        DI._GRAPH_CLIQUE_ENUM_MODE[] = :intersection
        Gw_fast = PosetModules.encode(data, spec_w; degree=0, stage=:graded_complex)
        DI._GRAPH_CLIQUE_ENUM_MODE[] = :combinations
        Gw_base = PosetModules.encode(data, spec_w; degree=0, stage=:graded_complex)
        DI._GRAPH_CLIQUE_ENUM_MODE[] = :auto
        Gw_auto = PosetModules.encode(data, spec_w; degree=0, stage=:graded_complex)
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
    data = PosetModules.PointCloud([[0.0], [1.0], [2.0]])
    spec = PosetModules.FiltrationSpec(
        kind=:rips,
        max_dim=1,
        construction=PosetModules.ConstructionOptions(; output_stage=:graded_complex),
    )
    G = PosetModules.encode(data, spec; degree=0)
    @test G isa PosetModules.GradedComplex
    @test !isempty(G.cells_by_dim)
end

@testset "Data pipeline: simplex-tree eps quantization parity" begin
    data = PosetModules.PointCloud([[0.0], [0.41], [0.93]])
    st = PosetModules.encode(
        data,
        PosetModules.FiltrationSpec(
            kind=:rips,
            max_dim=1,
            construction=PosetModules.ConstructionOptions(; output_stage=:simplex_tree),
        );
        degree=0,
    )
    @test st isa DI.SimplexTreeMulti

    spec_eps = PosetModules.FiltrationSpec(kind=:graded, eps=0.25)
    enc_tree = PosetModules.encode(st, spec_eps; degree=0)
    G = DI._graded_complex_from_simplex_tree(st)
    enc_grad = PosetModules.encode(G, spec_eps; degree=0)
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
    Gm = PosetModules.MultiCriticalGradedComplex(cells, [B1], grades)
    st = DI._simplex_tree_multi_from_complex(Gm)
    @test st isa DI.SimplexTreeMulti

    spec_one = PosetModules.FiltrationSpec(
        kind=:graded,
        multicritical=:one_critical,
        onecritical_selector=:lexmin,
        onecritical_enforce_boundary=true,
    )
    enc_tree = PosetModules.encode(st, spec_one; degree=0)
    enc_grad = PosetModules.encode(Gm, spec_one; degree=0)
    M_tree = _enc_module(enc_tree)
    M_grad = _enc_module(enc_grad)
    @test M_tree.dims == M_grad.dims
    for (u, v) in FF.cover_edges(M_tree.Q)
        @test Array(M_tree.edge_maps[u, v]) == Array(M_grad.edge_maps[u, v])
    end
end

@testset "Data pipeline: graded complex" begin
    cells = [Int[1]]
    boundaries = SparseMatrixCSC{Int,Int}[]
    grades = [Float64[0.0]]
    G = PosetModules.GradedComplex(cells, boundaries, grades)
    spec = PosetModules.FiltrationSpec(kind=:graded, axes=([0.0, 1.0],))
    enc = PosetModules.encode(G, spec; degree=0)
    @test _enc_dims(enc) == [1, 1]

    H = PosetModules.Workflow.fringe_presentation(DI.materialize_module(enc.M))
    Mp = IR.pmodule_from_fringe(H)
    @test Mp.dims == _enc_dims(enc)
end

@testset "Data pipeline: point cloud rips" begin
    data = PosetModules.PointCloud([[0.0], [1.0]])
    spec = PosetModules.FiltrationSpec(kind=:rips, max_dim=1, axes=([0.0, 1.0],))
    enc = PosetModules.encode(data, spec; degree=0)
    @test _enc_dims(enc) == [2, 1]
    bc = Inv.slice_barcode(_enc_module(enc), [1, 2])
    @test bc[(1, 2)] == 1
    @test bc[(1, 3)] == 1
end

@testset "Data pipeline: point cloud rips higher-dim" begin
    data = PosetModules.PointCloud([[0.0], [1.0], [2.0]])
    spec = PosetModules.FiltrationSpec(kind=:rips, max_dim=2, axes=([0.0, 1.0, 2.0],))
    enc = PosetModules.encode(data, spec; degree=0)
    @test _enc_dims(enc) == [3, 1, 1]
end

@testset "Data pipeline: point cloud dense rips d2 oracle" begin
    data = PosetModules.PointCloud([[0.0], [1.0], [3.0], [6.0]])
    spec = PosetModules.FiltrationSpec(
        kind=:rips,
        max_dim=2,
        construction=PosetModules.ConstructionOptions(; sparsify=:none, output_stage=:graded_complex),
    )
    G = PosetModules.encode(data, spec; stage=:graded_complex)
    @test G isa PosetModules.GradedComplex
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
    data = PosetModules.PointCloud([[0.0], [2.0], [5.0]])

    rips_spec = PosetModules.FiltrationSpec(
        kind=:rips,
        max_dim=1,
        construction=PosetModules.ConstructionOptions(; sparsify=:none, output_stage=:graded_complex),
    )
    G_rips = PosetModules.encode(data, rips_spec; stage=:graded_complex)
    @test G_rips isa PosetModules.GradedComplex
    @test length(G_rips.cells_by_dim[1]) == 3
    @test length(G_rips.cells_by_dim[2]) == 3
    @test G_rips.grades[1:3] == [(0.0,), (0.0,), (0.0,)]
    @test G_rips.grades[4:6] == [(2.0,), (5.0,), (3.0,)]
    Br = G_rips.boundaries[1]
    @test size(Br) == (3, 3)
    @test Br[1, 1] == -1 && Br[2, 1] == 1
    @test Br[1, 2] == -1 && Br[3, 2] == 1
    @test Br[2, 3] == -1 && Br[3, 3] == 1

    fr_spec = PosetModules.FiltrationSpec(
        kind=:function_rips,
        max_dim=1,
        vertex_values=[1.0, 4.0, 10.0],
        simplex_agg=:sum,
        construction=PosetModules.ConstructionOptions(; sparsify=:none, output_stage=:graded_complex),
    )
    G_fr = PosetModules.encode(data, fr_spec; stage=:graded_complex)
    @test G_fr.grades[1:3] == [(0.0, 1.0), (0.0, 4.0), (0.0, 10.0)]
    @test G_fr.grades[4:6] == [(2.0, 5.0), (5.0, 11.0), (3.0, 14.0)]

    rd_spec = PosetModules.FiltrationSpec(
        kind=:rips_density,
        max_dim=1,
        density_k=1,
        construction=PosetModules.ConstructionOptions(; sparsify=:none, output_stage=:graded_complex),
    )
    G_rd = PosetModules.encode(data, rd_spec; stage=:graded_complex)
    @test G_rd.grades[1:3] == [(0.0, 2.0), (0.0, 2.0), (0.0, 3.0)]
    @test G_rd.grades[4:6] == [(2.0, 2.0), (5.0, 3.0), (3.0, 3.0)]

    rh_spec = PosetModules.FiltrationSpec(
        kind=:rhomboid,
        max_dim=1,
        vertex_values=[1.0, 4.0, 10.0],
        construction=PosetModules.ConstructionOptions(; sparsify=:none, output_stage=:graded_complex),
    )
    G_rh = PosetModules.encode(data, rh_spec; stage=:graded_complex)
    @test G_rh.grades[1:3] == [(1.0, 1.0), (4.0, 4.0), (10.0, 10.0)]
    @test G_rh.grades[4:6] == [(1.0, 4.0), (1.0, 10.0), (4.0, 10.0)]
end

@testset "Data pipeline: typed filtration dispatch" begin
    data = PosetModules.PointCloud([[0.0], [1.0]])
    fspec = PosetModules.FiltrationSpec(kind=:rips, max_dim=1, axes=([0.0, 1.0],))
    ftyped = DI.RipsFiltration(max_dim=1)
    enc_spec = PosetModules.encode(data, fspec; degree=0)
    enc_typed = PosetModules.encode(data, ftyped; degree=0)
    @test _enc_dims(enc_typed) == _enc_dims(enc_spec)
    @test EC.axes_from_encoding(enc_typed.pi) == EC.axes_from_encoding(enc_spec.pi)

    g = PosetModules.GraphData(3, [(1, 2), (2, 3)])
    gfilt = DI.GraphLowerStarFiltration(vertex_values=[0.0, 1.0, 2.0], simplex_agg=:max)
    gspec = PosetModules.FiltrationSpec(kind=:graph_lower_star, vertex_values=[0.0, 1.0, 2.0], simplex_agg=:max)
    enc_g = PosetModules.encode(g, gfilt; degree=0)
    enc_gspec = PosetModules.encode(g, gspec; degree=0)
    @test _enc_dims(enc_g) == _enc_dims(enc_gspec)

    ffilt = DI.to_filtration(PosetModules.FiltrationSpec(kind=:rips_density, max_dim=1, density_k=2))
    @test ffilt isa DI.RipsDensityFiltration

    afilt = DI.to_filtration(PosetModules.FiltrationSpec(kind=:alpha, max_dim=2))
    @test afilt isa DI.AlphaFiltration
    cdfilt = DI.to_filtration(PosetModules.FiltrationSpec(kind=:core_delaunay, max_dim=2))
    @test cdfilt isa DI.CoreDelaunayFiltration
    drfilt = DI.to_filtration(PosetModules.FiltrationSpec(kind=:degree_rips, max_dim=1))
    @test drfilt isa DI.DegreeRipsFiltration
    cubfilt = DI.to_filtration(PosetModules.FiltrationSpec(kind=:cubical))
    @test cubfilt isa DI.CubicalFiltration
end

@testset "Data pipeline: Delaunay/function-Delaunay filtrations" begin
    pts = PosetModules.PointCloud([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    dspec = PosetModules.FiltrationSpec(kind=:delaunay_lower_star, vertex_values=[0.0, 1.0, 2.0], max_dim=2)
    dtyped = DI.to_filtration(dspec)
    @test dtyped isa DI.DelaunayLowerStarFiltration
    enc_d = PosetModules.encode(pts, dspec; degree=0)
    ax_d = EC.axes_from_encoding(enc_d.pi)
    @test length(ax_d) == 1
    @test ax_d[1] == [0.0, 1.0, 2.0]

    fspec = PosetModules.FiltrationSpec(kind=:function_delaunay, vertex_values=[0.0, 1.0, 2.0], simplex_agg=:max, max_dim=2)
    ftyped = DI.to_filtration(fspec)
    @test ftyped isa DI.FunctionDelaunayFiltration
    enc_f = PosetModules.encode(pts, fspec; degree=0)
    ax_f = EC.axes_from_encoding(enc_f.pi)
    @test length(ax_f) == 2
    @test 0.0 in ax_f[1]
    @test 0.0 in ax_f[2] && 2.0 in ax_f[2]

    aspec = PosetModules.FiltrationSpec(kind=:alpha, max_dim=2)
    atyped = DI.to_filtration(aspec)
    @test atyped isa DI.AlphaFiltration
    enc_a = PosetModules.encode(pts, aspec; degree=0)
    ax_a = EC.axes_from_encoding(enc_a.pi)
    @test length(ax_a) == 1
    @test 0.0 in ax_a[1]

    cdspec = PosetModules.FiltrationSpec(kind=:core_delaunay, max_dim=2)
    cdtyped = DI.to_filtration(cdspec)
    @test cdtyped isa DI.CoreDelaunayFiltration
    enc_cd = PosetModules.encode(pts, cdspec; degree=0)
    ax_cd = EC.axes_from_encoding(enc_cd.pi)
    @test length(ax_cd) == 2
    @test 0.0 in ax_cd[1]
    @test !isempty(ax_cd[2])
end

@testset "Data pipeline: Delaunay high-dimensional fallback policy" begin
    pts3d = PosetModules.PointCloud([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    spec_ls = PosetModules.FiltrationSpec(kind=:delaunay_lower_star,
                                          vertex_values=[0.0, 1.0, 2.0, 3.0],
                                          max_dim=2,
                                          highdim_policy=:rips)
    enc_ls = PosetModules.encode(pts3d, spec_ls; degree=0)
    ax_ls = EC.axes_from_encoding(enc_ls.pi)
    @test length(ax_ls) == 1
    @test 0.0 in ax_ls[1] && 3.0 in ax_ls[1]

    spec_fn = PosetModules.FiltrationSpec(kind=:function_delaunay,
                                          vertex_values=[0.0, 1.0, 2.0, 3.0],
                                          max_dim=2,
                                          simplex_agg=:max,
                                          highdim_policy=:rips)
    enc_fn = PosetModules.encode(pts3d, spec_fn; degree=0)
    ax_fn = EC.axes_from_encoding(enc_fn.pi)
    @test length(ax_fn) == 2
    @test 0.0 in ax_fn[1]
    @test 0.0 in ax_fn[2] && 3.0 in ax_fn[2]

    typed = DI.to_filtration(spec_fn)
    @test typed isa DI.FunctionDelaunayFiltration
    @test getfield(typed, :params).highdim_policy == :rips

    spec_err = PosetModules.FiltrationSpec(kind=:delaunay_lower_star,
                                           vertex_values=[0.0, 1.0, 2.0, 3.0],
                                           max_dim=2,
                                           highdim_policy=:error)
    @test_throws ErrorException PosetModules.encode(pts3d, spec_err; degree=0)

    spec_alpha = PosetModules.FiltrationSpec(kind=:alpha, max_dim=2, highdim_policy=:rips)
    enc_alpha = PosetModules.encode(pts3d, spec_alpha; degree=0)
    @test length(EC.axes_from_encoding(enc_alpha.pi)) == 1

    spec_alpha_err = PosetModules.FiltrationSpec(kind=:alpha, max_dim=2, highdim_policy=:error)
    @test_throws ErrorException PosetModules.encode(pts3d, spec_alpha_err; degree=0)

    spec_core_del = PosetModules.FiltrationSpec(kind=:core_delaunay, max_dim=2, highdim_policy=:rips)
    enc_core_del = PosetModules.encode(pts3d, spec_core_del; degree=0)
    @test length(EC.axes_from_encoding(enc_core_del.pi)) == 2

    spec_core_del_err = PosetModules.FiltrationSpec(kind=:core_delaunay, max_dim=2, highdim_policy=:error)
    @test_throws ErrorException PosetModules.encode(pts3d, spec_core_del_err; degree=0)
end

@testset "Data pipeline: core/rhomboid filtrations" begin
    function _old_point_rhomboid_emulated(data::DT.PointCloud, spec::OPT.FiltrationSpec)
        simplices = DI._rips_like_simplices_for_point_cloud(data, spec)
        vals = DI._point_vertex_values(data.points, spec)
        grades = Vector{NTuple{2,Float64}}()
        sizehint!(grades, sum(length.(simplices)))
        for s in simplices[1]
            v = vals[s[1]]
            push!(grades, (v, v))
        end
        for k in 2:length(simplices)
            for s in simplices[k]
                vmin = Float64(vals[s[1]])
                vmax = vmin
                @inbounds for t in 2:length(s)
                    v = Float64(vals[s[t]])
                    v < vmin && (vmin = v)
                    v > vmax && (vmax = v)
                end
                push!(grades, (vmin, vmax))
            end
        end
        return DI._materialize_simplicial_output(simplices, grades, spec; return_simplex_tree=true)
    end

    function _old_point_rhomboid_lowdim_emulated(data::DT.PointCloud, spec::OPT.FiltrationSpec)
        points = data.points
        n = length(points)
        construction = DI._construction_from_params(spec.params)
        max_dim = Int(get(spec.params, :max_dim, 1))
        include_edge_dim = (max_dim >= 1) || (construction.sparsify != :none)

        edges = NTuple{2,Int}[]
        edge_dists = Float64[]
        if include_edge_dim
            if construction.sparsify != :none
                edges, edge_dists, _ = DI._point_cloud_sparsify_edge_driven(points, spec, construction)
                edges, edge_dists = DI._apply_construction_collapse_edge_driven(edges, edge_dists, points, construction)
            else
                radius = haskey(spec.params, :radius) ? Float64(spec.params[:radius]) : Inf
                if isfinite(radius) && DI._POINTCLOUD_LOWDIM_RADIUS_STREAMING[]
                    edges, edge_dists = DI._point_cloud_edges_within_radius(points, radius)
                else
                    edges_all, dists_all = DI._complete_point_cloud_edges_with_dist(points)
                    if isfinite(radius)
                        for idx in eachindex(edges_all)
                            d = dists_all[idx]
                            d <= radius || continue
                            push!(edges, edges_all[idx])
                            push!(edge_dists, d)
                        end
                    else
                        edges = edges_all
                        edge_dists = dists_all
                    end
                end
            end
        end

        vals = DI._point_vertex_values(points, spec)
        grades = Vector{NTuple{2,Float64}}(undef, n + (include_edge_dim ? length(edges) : 0))
        t = 1
        @inbounds for i in 1:n
            v = Float64(vals[i])
            grades[t] = (v, v)
            t += 1
        end
        if include_edge_dim
            @inbounds for idx in eachindex(edges)
                u, v = edges[idx]
                vu = Float64(vals[u])
                vv = Float64(vals[v])
                grades[t] = (min(vu, vv), max(vu, vv))
                t += 1
            end
        end
        return DI._materialize_point_cloud_dim01(n, include_edge_dim, edges, grades, spec; return_simplex_tree=true)
    end

    g = PosetModules.GraphData(4, [(1, 2), (2, 3), (1, 3), (3, 4)])
    cspec = PosetModules.FiltrationSpec(kind=:core)
    ctyped = DI.to_filtration(cspec)
    @test ctyped isa DI.CoreFiltration
    enc_c = PosetModules.encode(g, cspec; degree=0)
    ax_c = EC.axes_from_encoding(enc_c.pi)
    @test length(ax_c) == 2
    @test 1.0 in ax_c[2] && 2.0 in ax_c[2]

    p = PosetModules.PointCloud([[0.0], [1.0], [2.0], [3.0]])
    enc_cp = PosetModules.encode(
        p,
        PosetModules.FiltrationSpec(kind=:core, knn=1, vertex_values=[0.0, 0.0, 0.0, 0.0]);
        degree=0,
    )
    @test length(EC.axes_from_encoding(enc_cp.pi)) == 2

    rspec = PosetModules.FiltrationSpec(kind=:rhomboid, max_dim=1, vertex_values=[0.0, 2.0, 4.0])
    rtyped = DI.to_filtration(rspec)
    @test rtyped isa DI.RhomboidFiltration
    g2 = PosetModules.GraphData(3, [(1, 2), (2, 3)])
    enc_r = PosetModules.encode(g2, rspec; degree=0)
    ax_r = EC.axes_from_encoding(enc_r.pi)
    @test length(ax_r) == 2
    @test 0.0 in ax_r[1] && 2.0 in ax_r[1]
    @test 0.0 in ax_r[2] && 4.0 in ax_r[2]

    # Low-dim rhomboid now routes through edges-only graph builders.
    p2 = PosetModules.PointCloud([[0.0], [0.4], [0.9], [1.5], [2.0], [2.6], [3.3]])
    vals2 = [0.2, 0.8, 0.1, 0.6, 0.9, 0.3, 0.7]
    spec_rh_rad = PosetModules.FiltrationSpec(
        kind=:rhomboid,
        max_dim=1,
        radius=0.95,
        vertex_values=vals2,
        construction=OPT.ConstructionOptions(; output_stage=:simplex_tree),
    )
    st_old_rad = first(_old_point_rhomboid_lowdim_emulated(p2, spec_rh_rad))
    st_new_rad = PosetModules.encode(p2, spec_rh_rad; degree=0, stage=:simplex_tree, cache=:auto)
    @test _canon_simplex_tree(st_new_rad) == _canon_simplex_tree(st_old_rad)

    spec_rh_knn = PosetModules.FiltrationSpec(
        kind=:rhomboid,
        max_dim=1,
        knn=3,
        nn_backend=:bruteforce,
        vertex_values=vals2,
        construction=OPT.ConstructionOptions(; sparsify=:knn, output_stage=:simplex_tree),
    )
    st_old_knn = first(_old_point_rhomboid_lowdim_emulated(p2, spec_rh_knn))
    st_new_knn = PosetModules.encode(p2, spec_rh_knn; degree=0, stage=:simplex_tree, cache=:auto)
    @test _canon_simplex_tree(st_new_knn) == _canon_simplex_tree(st_old_knn)

    # max_dim>1 now streams simplex generation/grading instead of _combinations materialization.
    spec_rh_d2 = PosetModules.FiltrationSpec(
        kind=:rhomboid,
        max_dim=2,
        vertex_values=vals2,
        construction=OPT.ConstructionOptions(; output_stage=:simplex_tree),
    )
    st_old_d2 = first(_old_point_rhomboid_emulated(p2, spec_rh_d2))
    st_new_d2 = PosetModules.encode(p2, spec_rh_d2; degree=0, stage=:simplex_tree, cache=:auto)
    @test _canon_simplex_tree(st_new_d2) == _canon_simplex_tree(st_old_d2)
end

@testset "Data pipeline: core packed dim01 + edge-only builder parity" begin
    pts = PosetModules.PointCloud([[0.0], [0.3], [0.8], [1.4], [2.1], [2.9], [3.2], [4.0]])

    # Core now routes through edge-only builders; verify edge parity vs full builders.
    spec_knn = PosetModules.FiltrationSpec(kind=:core, knn=3, nn_backend=:bruteforce)
    e_core_knn = DI._core_edges_from_point_cloud(pts.points, spec_knn)
    e_full_knn, _, _ = DI._point_cloud_knn_graph(pts.points, 3; backend=:bruteforce, approx_candidates=0)
    @test sort(e_core_knn) == sort(e_full_knn)

    spec_rad = PosetModules.FiltrationSpec(kind=:core, radius=1.25, nn_backend=:bruteforce)
    e_core_rad = DI._core_edges_from_point_cloud(pts.points, spec_rad)
    e_full_rad, _ = DI._point_cloud_radius_graph(pts.points, 1.25; backend=:bruteforce, approx_candidates=0)
    @test sort(e_core_rad) == sort(e_full_rad)

    if DI._have_pointcloud_nn_backend()
        spec_knn_nn = PosetModules.FiltrationSpec(kind=:core, knn=3, nn_backend=:nearestneighbors)
        e_core_knn_nn = DI._core_edges_from_point_cloud(pts.points, spec_knn_nn)
        e_full_knn_nn, _, _ = DI._point_cloud_knn_graph(pts.points, 3; backend=:nearestneighbors, approx_candidates=0)
        @test sort(e_core_knn_nn) == sort(e_full_knn_nn)

        spec_rad_nn = PosetModules.FiltrationSpec(kind=:core, radius=1.25, nn_backend=:nearestneighbors)
        e_core_rad_nn = DI._core_edges_from_point_cloud(pts.points, spec_rad_nn)
        e_full_rad_nn, _ = DI._point_cloud_radius_graph(pts.points, 1.25; backend=:nearestneighbors, approx_candidates=0)
        @test sort(e_core_rad_nn) == sort(e_full_rad_nn)
    end

    # Point-core path now uses packed dim01 materialization.
    pvals = [0.0, 0.2, 0.1, 0.9, 0.4, 0.8, 0.6, 0.7]
    spec_point = PosetModules.FiltrationSpec(
        kind=:core,
        knn=3,
        nn_backend=:bruteforce,
        vertex_values=pvals,
        construction=OPT.ConstructionOptions(; output_stage=:graded_complex),
    )
    Gp = PosetModules.encode(pts, spec_point; degree=0, stage=:graded_complex, cache=:auto)
    @test Gp isa DT.GradedComplex
    @test length(Gp.cells_by_dim) == 2
    @test length(Gp.cells_by_dim[1]) == length(pts.points)
    @test length(Gp.cells_by_dim[2]) == length(e_core_knn)
    @test size(Gp.boundaries[1], 1) == length(pts.points)
    @test size(Gp.boundaries[1], 2) == length(e_core_knn)

    # Graph-core path now also uses packed dim01 materialization.
    n = 9
    edges = [(1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (7, 9)]
    g = PosetModules.GraphData(n, edges)
    gvals = [Float64(i) / n for i in 1:n]
    spec_graph = PosetModules.FiltrationSpec(
        kind=:core,
        vertex_values=gvals,
        construction=OPT.ConstructionOptions(; output_stage=:graded_complex),
    )
    Gg = PosetModules.encode(g, spec_graph; degree=0, stage=:graded_complex, cache=:auto)
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
    p = PosetModules.PointCloud([[0.0], [2.0], [5.0]])
    dr_spec = PosetModules.FiltrationSpec(
        kind=:degree_rips,
        max_dim=1,
        construction=PosetModules.ConstructionOptions(; sparsify=:none, output_stage=:graded_complex),
    )
    G_dr = PosetModules.encode(p, dr_spec; stage=:graded_complex)
    @test G_dr.grades[1:3] == [(0.0, 2.0), (0.0, 2.0), (0.0, 2.0)]
    @test G_dr.grades[4:6] == [(2.0, 2.0), (5.0, 2.0), (3.0, 2.0)]

    img = PosetModules.ImageNd([0.0 1.0; 2.0 3.0])
    spec_cub = PosetModules.FiltrationSpec(kind=:cubical)
    spec_ls = PosetModules.FiltrationSpec(kind=:lower_star)
    G_cub = PosetModules.encode(img, spec_cub; stage=:graded_complex)
    G_ls = PosetModules.encode(img, spec_ls; stage=:graded_complex)
    @test G_cub.cells_by_dim == G_ls.cells_by_dim
    @test G_cub.boundaries == G_ls.boundaries
    @test G_cub.grades == G_ls.grades

    # 2D cubical fast path must be exact-parity with the generic cubical kernel.
    img2 = PosetModules.ImageNd([0.1 0.9 1.2 0.7; 0.4 1.3 0.2 1.1; 0.8 0.6 1.5 0.3])
    spec_cub2 = PosetModules.FiltrationSpec(kind=:cubical)
    spec_bi2 = PosetModules.FiltrationSpec(kind=:image_distance_bifiltration, mask=img2.data .> 0.75)
    old_fast = DI._CUBICAL_2D_FASTPATH[]
    try
        DI._CUBICAL_2D_FASTPATH[] = false
        G_cub_ref = PosetModules.encode(img2, spec_cub2; stage=:graded_complex)
        G_bi_ref = PosetModules.encode(img2, spec_bi2; stage=:graded_complex)

        DI._CUBICAL_2D_FASTPATH[] = true
        G_cub_fast = PosetModules.encode(img2, spec_cub2; stage=:graded_complex)
        G_bi_fast = PosetModules.encode(img2, spec_bi2; stage=:graded_complex)

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

    function _toy_builder(data::PosetModules.PointCloud,
                          filtration::ToyPointCloudFiltration;
                          cache::Union{Nothing,CM.EncodingCache}=nothing)
        construction = get(filtration.params, :construction, OPT.ConstructionOptions())
        return DI._graded_complex_from_data(data, DI.RipsFiltration(max_dim=1, construction=construction); cache=cache)
    end
    DI.build_graded_complex(data::PosetModules.PointCloud,
                            filtration::ToyPointCloudFiltration;
                            cache::Union{Nothing,CM.EncodingCache}=nothing) =
        _toy_builder(data, filtration; cache=cache)

    DI.register_filtration_family!(
        kind=:toy_point_cloud,
        ctor=spec -> ToyPointCloudFiltration(construction=DI._construction_from_params(spec.params)),
        builder=_toy_builder,
        arity=1,
    )

    data = PosetModules.PointCloud([[0.0], [1.0]])
    enc = PosetModules.encode(data, ToyPointCloudFiltration(); degree=0)
    @test _enc_dims(enc) == [2, 1]
    enc2 = PosetModules.encode(data, PosetModules.FiltrationSpec(kind=:toy_point_cloud); degree=0)
    @test _enc_dims(enc2) == [2, 1]
end

@testset "Data pipeline: ingestion planning protocol" begin
    data = PosetModules.PointCloud([[0.0], [1.0], [2.0]])
    filt = DI.RipsFiltration(max_dim=1, knn=2)
    spec = PosetModules.FiltrationSpec(kind=:rips, max_dim=1, knn=2)

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
    @test plan_with_preflight.preflight.n_cells_est == big(6)

    G_direct = PosetModules.encode(data, filt;
                                   degree=0,
                                   construction=construction,
                                   pipeline=pipeline,
                                   stage=:graded_complex)
    G_plan = DI.run_ingestion(plan_a; stage=:graded_complex)
    @test G_plan.cells_by_dim == G_direct.cells_by_dim
    @test G_plan.grades == G_direct.grades

    enc_direct = PosetModules.encode(data, filt;
                                     degree=0,
                                     construction=construction,
                                     pipeline=pipeline)
    enc_plan = PosetModules.encode(plan_a; degree=0)
    @test isnothing(enc_direct.H)
    @test isnothing(enc_plan.H)
    @test _enc_dims(enc_plan) == _enc_dims(enc_direct)
    @test EC.axes_from_encoding(enc_plan.pi) == EC.axes_from_encoding(enc_direct.pi)

    H_direct = PosetModules.encode(data, filt;
                                   degree=0,
                                   construction=construction,
                                   pipeline=pipeline,
                                   stage=:fringe)
    @test H_direct isa FF.FringeModule

    plan_auto = DI.plan_ingestion(data, filt;
                                  construction=construction,
                                  pipeline=pipeline,
                                  cache=:auto)
    enc_auto = PosetModules.encode(plan_auto; degree=0)
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

    @test_throws ErrorException PosetModules.encode(data, filt; stage=:not_a_stage)

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

@testset "Data pipeline: function-Rips (point cloud)" begin
    data = PosetModules.PointCloud([[0.0], [1.0]])
    spec_vals = PosetModules.FiltrationSpec(kind=:function_rips,
                                            max_dim=1,
                                            vertex_values=[0.0, 2.0],
                                            simplex_agg=:max)
    enc_vals = PosetModules.encode(data, spec_vals; degree=0)
    @test FF.nvertices(enc_vals.P) == 4
    @test MD.dim_at(_enc_module(enc_vals), EC.locate(enc_vals.pi, [0.0, 0.0])) == 1
    @test MD.dim_at(_enc_module(enc_vals), EC.locate(enc_vals.pi, [0.0, 2.0])) == 2
    @test MD.dim_at(_enc_module(enc_vals), EC.locate(enc_vals.pi, [1.0, 2.0])) == 1

    spec_fun = PosetModules.FiltrationSpec(kind=:function_rips,
                                           max_dim=1,
                                           vertex_function=(p, i) -> (i == 1 ? 0.0 : 2.0),
                                           simplex_agg=:max)
    enc_fun = PosetModules.encode(data, spec_fun; degree=0)
    @test _enc_dims(enc_fun) == _enc_dims(enc_vals)
    @test EC.axes_from_encoding(enc_fun.pi) == EC.axes_from_encoding(enc_vals.pi)
end

@testset "Data pipeline: graph vertex-values UX parity" begin
    g = PosetModules.GraphData(3, [(1, 2), (2, 3)])
    spec_old = PosetModules.FiltrationSpec(kind=:graph_lower_star,
                                           vertex_grades=[[0.0], [1.0], [2.0]],
                                           simplex_agg=:max)
    spec_new = PosetModules.FiltrationSpec(kind=:graph_lower_star,
                                           vertex_values=[0.0, 1.0, 2.0],
                                           simplex_agg=:max)
    spec_fun = PosetModules.FiltrationSpec(kind=:graph_lower_star,
                                           vertex_function=(arg, i) -> i - 1,
                                           simplex_agg=:max)
    enc_old = PosetModules.encode(g, spec_old; degree=0)
    enc_new = PosetModules.encode(g, spec_new; degree=0)
    enc_fun = PosetModules.encode(g, spec_fun; degree=0)
    @test _enc_dims(enc_new) == _enc_dims(enc_old)
    @test _enc_dims(enc_fun) == _enc_dims(enc_old)
    @test EC.axes_from_encoding(enc_new.pi) == EC.axes_from_encoding(enc_old.pi)
    @test EC.axes_from_encoding(enc_fun.pi) == EC.axes_from_encoding(enc_old.pi)
end

@testset "Data pipeline: graph centrality/geodesic/threshold filtrations" begin
    g = PosetModules.GraphData(3, [(1, 2), (2, 3)]; weights=[1.0, 2.0])

    f_cent = DI.GraphCentralityFiltration(centrality=:degree, lift=:lower_star)
    enc_cent = PosetModules.encode(g, f_cent; degree=0)
    ax_cent = EC.axes_from_encoding(enc_cent.pi)
    @test length(ax_cent) == 1
    @test ax_cent[1] == [1.0, 2.0]

    spec_cent = PosetModules.FiltrationSpec(kind=:graph_centrality, centrality=:closeness, metric=:hop, lift=:lower_star)
    typed_cent = DI.to_filtration(spec_cent)
    @test typed_cent isa DI.GraphCentralityFiltration
    enc_close = PosetModules.encode(g, spec_cent; degree=0)
    close_vals = EC.axes_from_encoding(enc_close.pi)[1]
    @test any(isapprox(v, 2 / 3; atol=1e-6) for v in close_vals)
    @test any(isapprox(v, 1.0; atol=1e-8) for v in close_vals)

    f_geo = DI.GraphGeodesicFiltration(sources=[1], metric=:hop, lift=:lower_star)
    enc_geo = PosetModules.encode(g, f_geo; degree=0)
    ax_geo = EC.axes_from_encoding(enc_geo.pi)
    @test length(ax_geo) == 1
    @test ax_geo[1] == [0.0, 1.0, 2.0]

    spec_geo = PosetModules.FiltrationSpec(kind=:graph_geodesic, sources=[1], metric=:weighted, lift=:lower_star)
    typed_geo = DI.to_filtration(spec_geo)
    @test typed_geo isa DI.GraphGeodesicFiltration
    enc_geo_w = PosetModules.encode(g, spec_geo; degree=0)
    @test EC.axes_from_encoding(enc_geo_w.pi)[1] == [0.0, 1.0, 3.0]

    spec_bi = PosetModules.FiltrationSpec(
        kind=:graph_function_geodesic_bifiltration,
        sources=[1],
        metric=:hop,
        vertex_values=[10.0, 20.0, 30.0],
        lift=:lower_star,
        simplex_agg=:max,
    )
    typed_bi = DI.to_filtration(spec_bi)
    @test typed_bi isa DI.GraphFunctionGeodesicBifiltration
    enc_bi = PosetModules.encode(g, spec_bi; degree=0)
    ax_bi = EC.axes_from_encoding(enc_bi.pi)
    @test length(ax_bi) == 2
    @test ax_bi[1] == [0.0, 1.0, 2.0]
    @test ax_bi[2] == [10.0, 20.0, 30.0]

    f_thr = DI.GraphWeightThresholdFiltration(edge_weights=[0.3, 0.8], lift=:graph)
    enc_thr = PosetModules.encode(g, f_thr; degree=0)
    @test EC.axes_from_encoding(enc_thr.pi)[1] == [0.0, 0.3, 0.8]

    gtri = PosetModules.GraphData(3, [(1, 2), (2, 3), (1, 3)]; weights=[0.3, 0.8, 0.5])
    spec_thr = PosetModules.FiltrationSpec(kind=:graph_weight_threshold, lift=:clique, max_dim=2)
    typed_thr = DI.to_filtration(spec_thr)
    @test typed_thr isa DI.GraphWeightThresholdFiltration
    enc_thr_clique = PosetModules.encode(gtri, spec_thr; degree=0)
    @test EC.axes_from_encoding(enc_thr_clique.pi)[1] == [0.0, 0.3, 0.5, 0.8]
    gtri_unsorted = PosetModules.GraphData(3, [(2, 1), (3, 2), (3, 1)]; weights=[0.3, 0.8, 0.5])
    enc_thr_clique_unsorted = PosetModules.encode(gtri_unsorted, spec_thr; degree=0)
    @test EC.axes_from_encoding(enc_thr_clique_unsorted.pi)[1] == [0.0, 0.3, 0.5, 0.8]

    est = DI.estimate_ingestion(gtri, PosetModules.FiltrationSpec(kind=:graph_centrality, lift=:clique, max_dim=2))
    @test est.cell_counts_by_dim == BigInt[3, 3, 1]
end

@testset "Data pipeline: graph new-family contracts" begin
    g = PosetModules.GraphData(3, [(1, 2), (2, 3)])
    @test_throws Exception PosetModules.encode(
        g,
        PosetModules.FiltrationSpec(kind=:graph_geodesic, sources=[1], metric=:weighted, lift=:lower_star);
        degree=0,
    )
    @test_throws Exception PosetModules.encode(
        g,
        PosetModules.FiltrationSpec(kind=:graph_weight_threshold, lift=:clique, max_dim=2);
        degree=0,
    )

    nbig = 80
    epath = [(i, j) for i in 1:(nbig - 1) for j in (i + 1):nbig]
    gpath = PosetModules.GraphData(nbig, epath; weights=fill(1.0, length(epath)))

    spec_clique_precheck = PosetModules.FiltrationSpec(
        kind=:graph_centrality,
        centrality=:degree,
        lift=:clique,
        max_dim=3,
        construction=PosetModules.ConstructionOptions(; budget=(15_000, nothing, nothing)),
    )
    err = try
        PosetModules.encode(gpath, spec_clique_precheck; degree=0)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("clique enumeration", sprint(showerror, err))

    spec_thr_clique_precheck = PosetModules.FiltrationSpec(
        kind=:graph_weight_threshold,
        lift=:clique,
        max_dim=3,
        edge_weights=fill(1.0, length(epath)),
        construction=PosetModules.ConstructionOptions(; budget=(15_000, nothing, nothing)),
    )
    err = try
        PosetModules.encode(gpath, spec_thr_clique_precheck; degree=0)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("clique enumeration", sprint(showerror, err))

    spec_graph_rhomboid_precheck = PosetModules.FiltrationSpec(
        kind=:rhomboid,
        max_dim=3,
        vertex_values=fill(0.0, nbig),
        construction=PosetModules.ConstructionOptions(; budget=(15_000, nothing, nothing)),
    )
    err = try
        PosetModules.encode(gpath, spec_graph_rhomboid_precheck; degree=0)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("clique enumeration", sprint(showerror, err))
end

@testset "Data pipeline: multi-critical graded complex" begin
    cells = [Int[1, 2], Int[1]]
    B1 = sparse([1, 2], [1, 1], [1, -1], 2, 1)
    grades = [
        [Float64[0.0, 0.0]],
        [Float64[0.0, 0.0]],
        [Float64[1.0, 0.0], Float64[0.0, 1.0]],
    ]
    G = PosetModules.MultiCriticalGradedComplex(cells, [B1], grades)
    spec = PosetModules.FiltrationSpec(kind=:graded)
    enc = PosetModules.encode(G, spec; degree=0)
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
    Gm = PosetModules.MultiCriticalGradedComplex(cells, [B1], grades)

    G1 = DI.one_criticalify(Gm)
    @test G1 isa PosetModules.GradedComplex
    @test length(G1.grades) == 3
    @test G1.grades[3] == (2.0, 2.0)  # lifted to dominate boundary-face grades

    G1_raw = DI.one_criticalify(Gm; enforce_boundary=false)
    @test G1_raw.grades[3] == (0.0, 1.0)  # default selector=:lexmin

    G1_max = DI.one_criticalify(Gm; selector=:lexmax, enforce_boundary=false)
    @test G1_max.grades[3] == (1.0, 0.0)

    Gs = PosetModules.GradedComplex(cells, [B1], [Float64[0.0, 0.0], Float64[2.0, 2.0], Float64[2.0, 2.0]])
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
    G = PosetModules.MultiCriticalGradedComplex(cells, [B1], grades)
    @test DI.criticality(G) == 3
    @test DI.criticality(DI.one_criticalify(G)) == 1

    Gn = DI.normalize_multicritical(G; keep=:minimal)
    @test DI.criticality(Gn) == 2
    @test length(Gn.grades[3]) == 2

    spec_union = PosetModules.FiltrationSpec(kind=:graded, multicritical=:union)
    spec_inter = PosetModules.FiltrationSpec(kind=:graded, multicritical=:intersection)
    spec_one = PosetModules.FiltrationSpec(kind=:graded, multicritical=:one_critical,
                                           onecritical_selector=:lexmin,
                                           onecritical_enforce_boundary=false)

    enc_union = PosetModules.encode(G, spec_union; degree=0)
    enc_inter = PosetModules.encode(G, spec_inter; degree=0)
    enc_one = PosetModules.encode(G, spec_one; degree=0)

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
        @test G isa PosetModules.MultiCriticalGradedComplex
        @test length(G.grades) == 3
        @test length(G.grades[3]) == 2
        enc = PosetModules.encode(G, PosetModules.FiltrationSpec(kind=:graded); degree=0)
        @test FF.nvertices(enc.P) == 4
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
        @test G isa PosetModules.GradedComplex
        @test length(G.cells_by_dim) == 3
        @test size(G.boundaries[1]) == (2, 2)
        @test size(G.boundaries[2]) == (2, 1)
    end
end

@testset "Data pipeline: image lower-star" begin
    img = [0.0 1.0; 2.0 3.0]
    data = PosetModules.ImageNd(img)
    spec = PosetModules.FiltrationSpec(kind=:lower_star, axes=([0.0, 1.0, 2.0, 3.0],))
    enc = PosetModules.encode(data, spec; degree=0)
    @test _enc_dims(enc) == fill(1, 4)
end

@testset "Data pipeline: 3D cubical lower-star" begin
    img = reshape(Float64.(1:8), (2, 2, 2))
    data = PosetModules.ImageNd(img)
    spec = PosetModules.FiltrationSpec(kind=:lower_star, axes=([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],))
    enc = PosetModules.encode(data, spec; degree=0)
    @test _enc_dims(enc) == fill(1, 8)
end

@testset "Data pipeline: embedded planar graph toy" begin
    verts = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
    edges = [(1, 2), (2, 3)]
    data = PosetModules.EmbeddedPlanarGraph2D(verts, edges)
    vgrades = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
    spec = PosetModules.FiltrationSpec(kind=:graph_lower_star, vertex_grades=vgrades)
    enc = PosetModules.encode(data, spec; degree=0)
    @test FF.nvertices(enc.P) > 0
    @test EC.locate(enc.pi, [0.0, 0.0]) > 0
    H = enc.H === nothing ? PosetModules.Workflow.fringe_presentation(DI.materialize_module(enc.M)) : enc.H
    @test H isa FF.FringeModule
end

@testset "Data pipeline: wing distance bifiltration" begin
    verts = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
    edges = [(1, 2), (2, 3)]
    data = PosetModules.EmbeddedPlanarGraph2D(verts, edges)
    spec = PosetModules.FiltrationSpec(
        kind=:wing_vein_bifiltration,
        grid=(8, 8),
        bbox=(0.0, 1.0, 0.0, 1.0),
        orientation=(-1, 1),
    )
    enc = PosetModules.encode(data, spec; degree=0)
    H = enc.H === nothing ? PosetModules.Workflow.fringe_presentation(DI.materialize_module(enc.M)) : enc.H
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
    G = PosetModules.GradedComplex(cells, boundaries, grades)
    spec = PosetModules.FiltrationSpec(kind=:graded, axes=([0.0, 1.0],))
    enc = PosetModules.encode(G, spec; degree=0)
    @test Inv.euler_surface(_enc_module(enc), enc.pi; opts=OPT.InvariantOptions(axes_policy=:encoding)) isa AbstractArray
    @test Inv.rank_invariant(_enc_module(enc), OPT.InvariantOptions()) isa Dict

    # Point cloud
    data = PosetModules.PointCloud([[0.0], [1.0]])
    spec = PosetModules.FiltrationSpec(kind=:rips, max_dim=1, axes=([0.0, 1.0],))
    enc = PosetModules.encode(data, spec; degree=0)
    @test Inv.euler_surface(_enc_module(enc), enc.pi; opts=OPT.InvariantOptions(axes_policy=:encoding)) isa AbstractArray
    @test Inv.rank_invariant(_enc_module(enc), OPT.InvariantOptions()) isa Dict

    # Image (2D)
    img = [0.0 1.0; 2.0 3.0]
    data = PosetModules.ImageNd(img)
    spec = PosetModules.FiltrationSpec(kind=:lower_star, axes=([0.0, 1.0, 2.0, 3.0],))
    enc = PosetModules.encode(data, spec; degree=0)
    @test Inv.euler_surface(_enc_module(enc), enc.pi; opts=OPT.InvariantOptions(axes_policy=:encoding)) isa AbstractArray

    # Graph (2D) for slice_chain
    data = PosetModules.GraphData(3, [(1, 2), (2, 3)])
    vgrades = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
    spec = PosetModules.FiltrationSpec(kind=:graph_lower_star, vertex_grades=vgrades)
    enc = PosetModules.encode(data, spec; degree=0)
    opts = OPT.InvariantOptions(axes_policy=:encoding, strict=false, box=:auto)
    chain, tvals = Inv.slice_chain(enc.pi, [0.0, 0.0], [1.0, 1.0], opts; nsteps=5)
    @test length(chain) > 0
    @test length(chain) == length(tvals)

    # Embedded planar graph (2D) for slice_chain
    verts = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
    edges = [(1, 2), (2, 3)]
    data = PosetModules.EmbeddedPlanarGraph2D(verts, edges)
    vgrades = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
    spec = PosetModules.FiltrationSpec(kind=:graph_lower_star, vertex_grades=vgrades)
    enc = PosetModules.encode(data, spec; degree=0)
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

    data = PosetModules.PointCloud([[0.0], [1.0], [2.0]])
    tf = TestTriGradeFiltration(; shift=0.5, scale=2.0)

    @test :test_trigrade in DI.available_filtrations()
    @test DI.filtration_arity(tf, data) == 3
    @test DI.filtration_kind(typeof(tf)) == :test_trigrade

    sig = DI.filtration_signature(:test_trigrade)
    @test sig.kind == :test_trigrade
    @test sig.registered == true
    @test sig.arity == 3
    @test haskey(sig.defaults, :shift)
    @test haskey(sig.defaults, :scale)

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
    tf2 = DI.to_filtration(spec_from_typed)
    @test tf2 isa TestTriGradeFiltration
    @test tf2.params.shift == 0.5
    @test tf2.params.scale == 2.0

    # Schema defaults apply when spec omits optional params.
    tf_default = DI.to_filtration(PosetModules.FiltrationSpec(kind=:test_trigrade))
    @test tf_default.params.shift == 0.0
    @test tf_default.params.scale == 1.0

    # Stage parity from typed filtration and FiltrationSpec.
    spec_direct = PosetModules.FiltrationSpec(kind=:test_trigrade, shift=0.5, scale=2.0)
    G_typed = PosetModules.encode(data, tf; stage=:graded_complex)
    G_spec = PosetModules.encode(data, spec_direct; stage=:graded_complex)
    @test G_typed.grades == G_spec.grades
    @test G_typed.cells_by_dim == G_spec.cells_by_dim

    enc_typed = PosetModules.encode(data, tf; stage=:encoding_result, degree=0)
    enc_spec = PosetModules.encode(data, spec_direct; stage=:encoding_result, degree=0)
    @test DI.module_dims(enc_typed.M) == DI.module_dims(enc_spec.M)

    # Schema contract failures.
    @test_throws ArgumentError DI.to_filtration(PosetModules.FiltrationSpec(kind=:test_trigrade, shift="bad"))
    @test_throws ArgumentError DI.to_filtration(PosetModules.FiltrationSpec(kind=:test_trigrade, scale=0.0))
end
