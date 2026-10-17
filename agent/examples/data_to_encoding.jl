import TamerOp as OP

function run_data_to_encoding_example()
    # Rows are points. The middle point needs a higher function threshold.
    points = OP.PointCloud(reshape([0.0, 1.0, 2.0], 3, 1))
    filtration = OP.FunctionRipsFiltration(
        max_dim=1, vertex_values=[0.0, 1.0, 0.0])
    encoding = OP.encode(points, filtration;
        degree=0, field=OP.CoreModules.F2())

    # Coordinates are (maximum edge length, function threshold).
    queries = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0), (2.0, 0.0)]
    components = [OP.rank_map(encoding, point, point) for point in queries]
    @assert components == [2, 3, 2, 1, 1]

    # Both initial components reach the same component after the merge.
    merged_rank = OP.rank_map(encoding, (1.0, 0.0), (1.0, 1.0))
    @assert merged_rank == 1
    radius_rank = OP.rank_map(encoding, (1.0, 0.0), (2.0, 0.0))
    @assert radius_rank == 1

    return (; encoding, queries, components, merged_rank, radius_rank)
end

if abspath(PROGRAM_FILE) == @__FILE__
    result = run_data_to_encoding_example()
    for (point, count) in zip(result.queries, result.components)
        println(point, " -> ", count, " connected components")
    end
    println("Rank across either merge: ", result.merged_rank)
end
