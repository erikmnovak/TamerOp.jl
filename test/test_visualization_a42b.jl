# A42b: maps and equations, including examples where dimensions lose information.

function _a42b_constant_module(P, d, field)
    K = CM.coeff_type(field)
    return MD.PModule{K}(P, fill(d, FF.nvertices(P)),
        Dict(edge => Matrix{K}(I, d, d) for edge in FF.cover_edges(P)); field)
end

function _a42b_matrix_equal(field, A, B)
    return field isa CM.RealField ? isapprox(A, B; atol=1e-9, rtol=1e-9) : A == B
end

@testset "A42b morphism selection and request contracts" begin
    V = TamerOp.Visualization
    P = chain_poset(3)
    M = _a42b_constant_module(P, 3, FIELD_QQ)
    N = _a42b_constant_module(P, 2, FIELD_QQ)
    A = QQ[1 1 0; 0 0 0]
    f = MD.PMorphism(M, N, [copy(A) for _ in 1:3])
    @test V.available_visuals(f) ==
        (:morphism_inspector, :naturality, :kernel_image_cokernel, :morphism_support)
    overview = V.visual_spec(f)
    @test overview.kind == :morphism_inspector
    @test overview.metadata.validation.valid
    @test !overview.interaction.clicks && !overview.interaction.hover
    @test V.check_visual_spec(overview).valid
    selected = V.visual_spec(f; vertex=2, matrix_limit=(1, 2))
    @test selected.metadata.component == A
    @test selected.metadata.rank == 1
    @test selected.metadata.selection.vertex == 2
    matrix_panels = filter(p -> any(l -> l isa V.MatrixLayer, p.layers), selected.panels)
    @test any(p -> only(p.layers).entries == ["1" "1"], matrix_panels)
    @test V.check_visual_spec(selected).valid
    selected.metadata.component[1, 1] = 7
    @test MD.component(f, 2) == A

    for options in ((; vertex=0), (; vertex=4), (; vertex=true),
                    (; pair=(1, 4)), (; pair=(true, 2)), (; pair=(1,)),
                    (; vertex=1, pair=(1, 2)), (; matrix_limit=(0, 2)),
                    (; matrix_limit=(2, false)), (; hover=true),
                    (; point=(0, 0)), (; parameter_pair=((0, 0), (1, 1))),
                    (; kind=:naturality), (; kind=:naturality, vertex=1),
                    (; kind=:kernel_image_cokernel),
                    (; kind=:kernel_image_cokernel, pair=(1, 2)))
        @test !V.check_visual_request(f; options...).valid
        @test_throws ArgumentError V.visual_spec(f; options...)
    end
    for components in ([zeros(QQ, 1, 1) for _ in 1:3], [copy(A) for _ in 1:2])
        malformed = MD.PMorphism{QQ,typeof(FIELD_QQ),Matrix{QQ}}(M, N, components)
        @test !V.check_visual_request(malformed; vertex=1).valid
        @test_throws ArgumentError V.visual_spec(malformed; vertex=1)
    end
end

@testset "A42b component and categorical equations over supported fields" begin
    V = TamerOp.Visualization
    P = chain_poset(3)
    for field in FIELDS_FULL
        K = CM.coeff_type(field)
        M = _a42b_constant_module(P, 3, field)
        N = _a42b_constant_module(P, 2, field)
        A = K[1 1 0; 0 0 0]
        f = MD.PMorphism(M, N, [copy(A) for _ in 1:3])
        spec = V.visual_spec(f; kind=:kernel_image_cokernel, vertex=2)
        data = spec.metadata.algebra
        @test spec.metadata.component == A
        @test spec.metadata.rank == 1
        @test data.kernel.dims == [2, 2, 2]
        @test data.image.dims == [1, 1, 1]
        @test data.cokernel.dims == [1, 1, 1]
        for q in 1:3
            ki = MD.component(data.inclusion_kernel, q)
            ii = MD.component(data.inclusion_image, q)
            cp = MD.component(data.projection_cokernel, q)
            @test size(ki) == (3, 2)
            @test size(ii) == (2, 1)
            @test size(cp) == (1, 2)
            @test _a42b_matrix_equal(field, A * ki, zeros(K, 2, 2))
            @test _a42b_matrix_equal(field, cp * A, zeros(K, 1, 3))
            @test _a42b_matrix_equal(field, cp * ii, zeros(K, 1, 1))
            @test FL.rank(field, ki) == 2
            @test FL.rank(field, ii) == 1
            @test FL.rank(field, cp) == 1
            @test FL.rank(field, hcat(ii, A)) == 1
        end
        @test MD.check_morphism(data.inclusion_kernel).valid
        @test MD.check_morphism(data.inclusion_image).valid
        @test MD.check_morphism(data.projection_cokernel).valid
        @test V.check_visual_spec(spec).valid

        # det = -2, so the coefficient field changes the actual categorical data.
        B = K[1 1; 1 -1]
        C = _a42b_constant_module(P, 2, field)
        g = MD.PMorphism(C, C, [copy(B) for _ in 1:3])
        expected_rank = field isa CM.PrimeField && field.p == 2 ? 1 : 2
        dependent = V.visual_spec(g; kind=:kernel_image_cokernel, vertex=1)
        @test dependent.metadata.rank == expected_rank
        @test dependent.metadata.algebra.kernel.dims == fill(2 - expected_rank, 3)
        @test dependent.metadata.algebra.image.dims == fill(expected_rank, 3)
        @test dependent.metadata.algebra.cokernel.dims == fill(2 - expected_rank, 3)

        for (source_dim, target_dim) in ((0, 0), (0, 2), (2, 0))
            source = _a42b_constant_module(P, source_dim, field)
            target = _a42b_constant_module(P, target_dim, field)
            z = MD.PMorphism(source, target,
                [zeros(K, target_dim, source_dim) for _ in 1:3])
            empty = V.visual_spec(z; kind=:kernel_image_cokernel, vertex=1)
            @test size(empty.metadata.component) == (target_dim, source_dim)
            @test empty.metadata.rank == 0
            @test empty.metadata.algebra.kernel.dims == fill(source_dim, 3)
            @test empty.metadata.algebra.image.dims == zeros(Int, 3)
            @test empty.metadata.algebra.cokernel.dims == fill(target_dim, 3)
            @test V.check_visual_spec(empty).valid
        end
    end
end

@testset "A42b support overlap does not create a morphism" begin
    V = TamerOp.Visualization
    options = TOA.EncodingOptions(; backend=:pl_backend, poset_kind=:signature, field=FIELD_QQ)
    enc = TamerOp.encode([TOA.BoxUpset([0.0, 0.0])],
        [TOA.BoxDownset([2.0, 2.0])], ones(QQ, 1, 1), options)
    M = RES.encoding_module(enc)
    P = RES.encoding_poset(enc)
    z = MD.PMorphism(M, M, [zeros(QQ, d, d) for d in M.dims])
    identity = MD.PMorphism(M, M, [Matrix{QQ}(I, d, d) for d in M.dims])
    zero_support = V.visual_spec(z; kind=:morphism_support, encoding=enc,
        box=([-1, -1], [3, 3]))
    full_support = V.visual_spec(identity; kind=:morphism_support, encoding=enc,
        box=([-1, -1], [3, 3]))
    @test zero_support.metadata.validation.valid
    @test zero_support.metadata.base === P
    @test zero_support.metadata.source_dimensions == M.dims
    @test zero_support.metadata.target_dimensions == M.dims
    @test zero_support.metadata.overlap == (M.dims .> 0)
    @test any(zero_support.metadata.overlap)
    @test zero_support.metadata.component_ranks == zeros(Int, length(M.dims))
    @test full_support.metadata.component_ranks == M.dims
    @test zero_support.metadata.support_restriction
    @test full_support.metadata.support_restriction
    @test last(zero_support.panels).metadata.quantity == :component_rank
    @test last(zero_support.panels).metadata.values == zeros(Int, length(M.dims))
    @test zero_support.legend.visible
    @test [entry.label for entry in zero_support.legend.entries] ==
        ["source", "target", "overlap", "image"]
    zero_polygons = filter(layer -> layer isa V.PolygonLayer &&
        layer.fill_color == V._VisualRole(:background), last(zero_support.panels).layers)
    @test !isempty(zero_polygons)
    @test all(layer -> layer.alpha == 1.0, zero_polygons)
    @test V.check_visual_spec(zero_support).valid
    @test V.check_visual_spec(full_support).valid

    # Isomorphic finite order and identical dimensions do not identify classifier IDs.
    n = FF.nvertices(P)
    otherP = FF.FinitePoset([FF.leq(P, a, b) for a in 1:n, b in 1:n])
    otherM = MD.PModule{QQ}(otherP, copy(M.dims),
        Dict(edge => copy(MD.structure_map(M; source=edge[1], target=edge[2]))
            for edge in FF.cover_edges(otherP)); field=FIELD_QQ)
    other = MD.PMorphism(otherM, otherM, [Matrix{QQ}(I, d, d) for d in otherM.dims])
    absent = RES.EncodingResult(P, M, nothing)
    for (morphism, opts) in ((z, (;)), (z, (; encoding=absent)),
                             (other, (; encoding=enc)),
                             (z, (; encoding=enc, vertex=1)))
        @test !V.check_visual_request(morphism; kind=:morphism_support, opts...).valid
        @test_throws ArgumentError V.visual_spec(morphism; kind=:morphism_support, opts...)
    end
end

@testset "A42b native support legend and balanced panels" begin
    V = TamerOp.Visualization
    options = TOA.EncodingOptions(; backend=:pl_backend, poset_kind=:signature, field=FIELD_QQ)
    enc = TamerOp.encode([TOA.BoxUpset([0.0, 0.0])],
        [TOA.BoxDownset([2.0, 2.0])], ones(QQ, 1, 1), options)
    M = RES.encoding_module(enc)
    zero_map = MD.PMorphism(M, M, [zeros(QQ, d, d) for d in M.dims])
    spec = V.visual_spec(zero_map; kind=:morphism_support, encoding=enc,
        box=([-1, -1], [3, 3]))
    for backend in (:cairomakie, :wglmakie)
        name = backend === :cairomakie ? "CairoMakie" : "WGLMakie"
        if Base.find_package(name) === nothing
            @test_skip false
            continue
        end
        if backend === :cairomakie
            @eval import CairoMakie
            makie = CairoMakie.Makie
        else
            @eval import WGLMakie
            makie = WGLMakie.Makie
        end
        fig = V.render(spec; backend)
        makie.update_state_before_display!(fig)
        @test count(item -> item isa makie.Legend, fig.content) == 1
        axes = filter(item -> item isa makie.Axis, fig.content)
        @test length(axes) == 4
        extents = [Tuple(makie.widths(makie.viewport(ax.scene)[])) for ax in axes]
        for coordinate in (1, 2)
            sizes = getindex.(extents, coordinate)
            @test maximum(sizes) - minimum(sizes) <= 2.0
        end
    end
end

@testset "A42b native algebra figures preserve coefficients and exports" begin
    V = TamerOp.Visualization
    AC = TamerOp.AbelianCategories
    P = chain_poset(2)
    M = _a42b_constant_module(P, 1, FIELD_QQ)
    B = _a42b_constant_module(P, 2, FIELD_QQ)
    f = MD.PMorphism(M, M, [fill(QQ(1//3), 1, 1) for _ in 1:2])
    i = MD.PMorphism(M, B, [reshape(QQ[1, 0], 2, 1) for _ in 1:2])
    p = MD.PMorphism(B, M, [QQ[0 1] for _ in 1:2])
    specs = [V.visual_spec(f; vertex=1),
        V.visual_spec(f; kind=:naturality, pair=(1, 2)),
        V.visual_spec(f; kind=:kernel_image_cokernel, vertex=1),
        V.visual_spec(AC.short_exact_sequence(i, p); vertex=1)]
    for backend in (:cairomakie, :wglmakie)
        name = backend === :cairomakie ? "CairoMakie" : "WGLMakie"
        if Base.find_package(name) === nothing
            @test_skip false
            continue
        end
        if backend === :cairomakie
            @eval import CairoMakie
            makie = CairoMakie.Makie
        else
            @eval import WGLMakie
            makie = WGLMakie.Makie
        end
        for (index, spec) in enumerate(specs)
            fig = V.render(spec; backend)
            makie.update_state_before_display!(fig)
            @test Tuple(makie.widths(makie.viewport(fig.scene)[])) == spec.metadata.figure_size
            axes = filter(item -> item isa makie.Axis, fig.content)
            @test !isempty(axes)
            @test all(ax -> all(>(0), makie.widths(makie.viewport(ax.scene)[])), axes)
            texts = [String(item.text[]) for item in fig.content if item isa makie.Label]
            if index in (1, 2)
                @test "1/3" in texts
            elseif index == 3
                @test any(text -> occursin("Empty matrix", text), texts)
            end
        end
        mktempdir() do dir
            if backend === :cairomakie
                for ext in ("png", "svg", "pdf")
                    path = joinpath(dir, "naturality.$ext")
                    V.save_visual(path, specs[2]; backend)
                    @test filesize(path) > 1000
                    bytes = read(path)
                    if ext == "png"
                        @test bytes[1:8] == UInt8[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]
                    elseif ext == "svg"
                        @test occursin("<svg", String(bytes))
                    else
                        @test String(bytes[1:4]) == "%PDF"
                    end
                end
            else
                path = joinpath(dir, "naturality.html")
                V.save_visual(path, specs[2]; backend)
                @test filesize(path) > 1000
                @test occursin("html", lowercase(read(path, String)))
            end
        end
    end
end

@testset "A42b naturality uses the two actual composites" begin
    V = TamerOp.Visualization
    P = chain_poset(2)
    for field in FIELDS_FULL
        K = CM.coeff_type(field)
        J, R = reshape(K[1, 0], 2, 1), K[0 1]
        M = MD.PModule{K}(P, [1, 2], Dict((1, 2) => J); field)
        N = MD.PModule{K}(P, [2, 1], Dict((1, 2) => R); field)
        f = MD.PMorphism(M, N, [copy(J), copy(R)])
        spec = V.visual_spec(f; kind=:naturality, pair=(1, 2))
        n = spec.metadata.naturality
        @test V.visual_spec(f; kind=:morphism_inspector, pair=(1, 2)).kind == :morphism_inspector
        @test n.defined && n.valid
        @test n.source_map == J
        @test n.target_map == R
        @test n.source_component == J
        @test n.target_component == R
        @test n.left == zeros(K, 1, 1)
        @test n.right == zeros(K, 1, 1)
        @test n.left == n.target_map * n.source_component
        @test n.right == n.target_component * n.source_map
        @test n.exact == !(field isa CM.RealField)
        @test V.check_visual_spec(spec).valid
        bad = MD.PMorphism(M, N, [copy(J), K[1 0]])
        failure = V.visual_spec(bad; kind=:naturality, pair=(1, 2))
        @test !failure.metadata.validation.valid
        @test failure.metadata.naturality.defined
        @test !failure.metadata.naturality.valid
        @test failure.metadata.naturality.left == zeros(K, 1, 1)
        @test failure.metadata.naturality.right == ones(K, 1, 1)
        @test_throws ArgumentError V.visual_spec(bad; kind=:kernel_image_cokernel, vertex=1)
    end

    P3 = chain_poset(3)
    C = _a42b_constant_module(P3, 1, FIELD_QQ)
    scalar = reshape(QQ[1//3], 1, 1)
    f = MD.PMorphism(C, C, [copy(scalar) for _ in 1:3])
    for pair in ((1, 3), (2, 2))
        data = V.visual_spec(f; kind=:naturality, pair).metadata.naturality
        @test data.defined && data.valid
        @test data.left == scalar == data.right
    end
    reverse = V.visual_spec(f; kind=:naturality, pair=(3, 1)).metadata.naturality
    @test !reverse.defined
    disconnected = disjoint_two_chains_poset()
    D = _a42b_constant_module(disconnected, 1, FIELD_QQ)
    h = MD.PMorphism(D, D, [ones(QQ, 1, 1) for _ in 1:4])
    incomparable = V.visual_spec(h; kind=:naturality, pair=(1, 3)).metadata.naturality
    @test !incomparable.defined

    for (atol, accepted) in ((1e-6, true), (1e-10, false))
        field = CM.RealField(Float64; atol, rtol=0.0)
        C = _a42b_constant_module(P, 1, field)
        numerical = MD.PMorphism(C, C, [ones(1, 1), fill(1.0 + 1e-8, 1, 1)])
        spec = V.visual_spec(numerical; kind=:naturality, pair=(1, 2))
        data = spec.metadata.naturality
        @test !data.exact
        @test data.valid == accepted
        @test isapprox(data.residual, 1e-8; atol=1e-15, rtol=0.0)
        @test data.tolerance == atol
        @test V.check_visual_spec(spec).valid
    end
end

@testset "A42b exactness requires maps and fresh evidence" begin
    V = TamerOp.Visualization
    AC = TamerOp.AbelianCategories
    P = chain_poset(2)
    for field in FIELDS_FULL
        K = CM.coeff_type(field)
        A = _a42b_constant_module(P, 1, field)
        B = _a42b_constant_module(P, 2, field)
        C = _a42b_constant_module(P, 1, field)
        J, R = reshape(K[1, 0], 2, 1), K[0 1]
        i = MD.PMorphism(A, B, [copy(J), copy(J)])
        p = MD.PMorphism(B, C, [copy(R), copy(R)])
        ses = AC.short_exact_sequence(i, p; check=false)
        @test V.available_visuals(ses) == (:exact_sequence,)
        spec = V.visual_spec(ses; vertex=1)
        @test spec.kind == :exact_sequence
        @test spec.metadata.exact
        @test spec.metadata.validation.valid
        @test spec.metadata.component.inclusion == J
        @test spec.metadata.component.projection == R
        @test spec.metadata.component.composite == zeros(K, 1, 1)
        @test V.check_visual_spec(spec).valid

        # Equal dimensions and equal ranks do not make a sequence exact.
        p_bad = MD.PMorphism(B, C, [K[1 0], K[1 0]])
        bad = AC.short_exact_sequence(i, p_bad; check=false)
        failure = V.visual_spec(bad; vertex=1)
        @test !failure.metadata.exact
        @test !failure.metadata.validation.valid
        @test failure.metadata.component.composite == ones(K, 1, 1)
        @test FL.rank(field, spec.metadata.component.inclusion) ==
            FL.rank(field, failure.metadata.component.inclusion) == 1
        @test FL.rank(field, spec.metadata.component.projection) ==
            FL.rank(field, failure.metadata.component.projection) == 1
        @test V.check_visual_spec(failure).valid

        # The container's memoized flags cannot certify mutated coefficients.
        cached = AC.short_exact_sequence(i, p)
        @test cached.checked && cached.exact
        MD.component(p, 1)[1, 1] = one(K)
        mutated = V.visual_spec(cached; vertex=1)
        @test !mutated.metadata.exact
        @test mutated.metadata.component.composite == ones(K, 1, 1)
        @test MD.component(p, 1) == K[1 1]
        for options in ((; vertex=0), (; vertex=3), (; pair=(1, 2)),
                        (; vertex=1, matrix_limit=(0, 1)), (; kind=:naturality))
            @test !V.check_visual_request(ses; options...).valid
            @test_throws ArgumentError V.visual_spec(ses; options...)
        end
    end
    for (atol, accepted) in ((1e-6, true), (1e-10, false))
        field = CM.RealField(Float64; atol, rtol=0.0)
        A = _a42b_constant_module(P, 1, field)
        B = _a42b_constant_module(P, 2, field)
        C = _a42b_constant_module(P, 1, field)
        i = MD.PMorphism(A, B, [reshape([1.0, 0.0], 2, 1) for _ in 1:2])
        p = MD.PMorphism(B, C, [[1e-8 1.0] for _ in 1:2])
        numerical = V.visual_spec(AC.short_exact_sequence(i, p; check=false); vertex=1)
        @test numerical.metadata.exact == accepted
        @test !numerical.metadata.exact_arithmetic
        @test numerical.metadata.component.equation.valid == accepted
        @test numerical.metadata.component.equation.residual == 1e-8
        @test numerical.metadata.component.equation.tolerance == atol
    end
end

@testset "A42b panel spans have a validated layout contract" begin
    V = TamerOp.Visualization
    panel = V.visual_spec(chain_poset(2); kind=:hasse)
    panels = [panel for _ in 1:4]
    positions = [(1:1, 1:3), (2:2, 1:1), (2:2, 2:2), (2:2, 3:3)]
    valid = V.VisualizationSpec(:layout_test; panels,
        metadata=(; panel_positions=positions, panel_row_weights=[2.0, 1.0]))
    @test V.check_visual_spec(valid).valid
    @test V.check_visual_spec(V.VisualizationSpec(:layout_test; panels)).valid
    for metadata in (
        (; panel_positions=positions[1:3]),
        (; panel_positions=[(1:1, 1:3), (1:1, 1:1), (2:2, 2:2), (2:2, 3:3)]),
        (; panel_positions=[(1:0, 1:3), positions[2:end]...]),
        (; panel_positions=[(0:1, 1:3), positions[2:end]...]),
        (; panel_positions=[(-1:1, 1:3), positions[2:end]...]),
        (; panel_positions=[(1, 1), (2, 1), (2, 2), (2, 3)]),
        (; panel_positions=Tuple(positions)),
        (; panel_row_weights=[2.0, 1.0]),
        (; panel_positions=positions, panel_row_weights=[1.0]),
        (; panel_positions=positions, panel_row_weights=[2.0, 0.0]),
        (; panel_positions=positions, panel_row_weights=[2.0, -1.0]),
        (; panel_positions=positions, panel_row_weights=[2.0, Inf]),
        (; panel_positions=positions, panel_row_weights=[2.0, NaN]),
        (; panel_positions=positions, panel_row_weights=[2.0, "1"]),
    )
        malformed = V.VisualizationSpec(:layout_test; panels, metadata)
        @test !V.check_visual_spec(malformed).valid
        @test_throws ArgumentError V.check_visual_spec(malformed; throw=true)
    end
end
