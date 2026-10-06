using Test
using SparseArrays

@testset "A42b selected Hom basis and zero Hom" begin
    V = TamerOp.Visualization
    field = CM.QQField()
    K = CM.coeff_type(field)
    P = FF.FinitePoset(trues(1, 1))
    M = MD.PModule{K}(P, [2], Dict{Tuple{Int,Int},Matrix{K}}(); field)
    N = MD.PModule{K}(P, [1], Dict{Tuple{Int,Int},Matrix{K}}(); field)
    H = DF.Hom(M, N)
    @test DF.dim(H) == 2
    @test V.available_visuals(H) == (:hom_basis,)
    @test !DF.hom_summary(H).basis_cached
    spec = V.visual_spec(H; basis_index=2, vertex=1)
    @test spec.kind == :hom_basis
    @test spec.metadata.basis_index == 2
    @test spec.metadata.hom_dimension == 2
    @test spec.metadata.basis_materialized
    @test DF.hom_summary(H).basis_cached
    @test MD.component(DF.basis(H)[2], 1) == reshape(K[0, 1], 1, 2)
    @test V.check_visual_spec(spec).valid
    @test_throws ArgumentError V.visual_spec(H; basis_index=0)
    @test_throws ArgumentError V.visual_spec(H; basis_index=3)
    @test_throws ArgumentError V.visual_spec(H; basis_index=true)
    @test_throws ArgumentError V.visual_spec(H; vertex=1, pair=(1, 1))
    zeroH = DF.Hom(M, MD.zero_pmodule(P; field))
    zero_cached_before = DF.hom_summary(zeroH).basis_cached
    zero_spec = V.visual_spec(zeroH)
    @test zero_spec.metadata.hom_dimension == 0
    @test zero_spec.metadata.basis_index === nothing
    @test !zero_spec.metadata.basis_materialized
    @test DF.hom_summary(zeroH).basis_cached == zero_cached_before
    @test_throws ArgumentError V.visual_spec(zeroH; basis_index=1)
end

@testset "A42b cochain lifts, homotopies and nonzero induced maps" begin
    V, MC = TamerOp.Visualization, TamerOp.ModuleComplexes
    for field in (CM.QQField(), CM.PrimeField(3), CM.RealField(Float64; atol=1e-8, rtol=1e-8))
        K = CM.coeff_type(field)
        c(x) = CM.coerce(field, x)
        P = FF.FinitePoset(trues(1, 1))
        C0 = MD.PModule{K}(P, [2], Dict{Tuple{Int,Int},Matrix{K}}(); field)
        C1 = MD.PModule{K}(P, [1], Dict{Tuple{Int,Int},Matrix{K}}(); field)
        Z = MD.zero_pmodule(P; field)
        d = MD.PMorphism(C0, C1, [reshape(K[c(0), c(1)], 1, 2)])
        C = MC.ModuleCochainComplex([C0, C1], [d]; tmin=0)
        f = MC.ModuleCochainMap(C, C, [MD.id_morphism(C0), MD.id_morphism(C1)])
        g0 = MD.PMorphism(C0, C0, [K[c(1) c(0); c(0) c(0)]])
        g = MC.ModuleCochainMap(C, C, [g0, MD.zero_morphism(C1, C1)])
        h0 = MD.zero_morphism(C0, Z)
        h1 = MD.PMorphism(C1, C0, [reshape(K[c(0), c(1)], 2, 1)])
        H = MC.ModuleCochainHomotopy(f, g, [h0, h1]; tmin=0, tmax=1)

        map_spec = V.visual_spec(f; degree=0, vertex=1, induced=true)
        @test map_spec.metadata.validation.valid
        @test map_spec.metadata.equation.valid
        @test map_spec.metadata.induced_computed
        @test MD.component(map_spec.metadata.induced_map, 1) == reshape(K[c(1)], 1, 1)
        @test map_spec.metadata.matrices.left == reshape(K[c(0), c(1)], 1, 2)
        @test map_spec.panels[1].kind == :naturality_square
        @test V.check_visual_spec(map_spec).valid

        witness_spec = V.visual_spec(H; degree=0, vertex=1, induced=true)
        @test witness_spec.metadata.validation.valid
        @test witness_spec.metadata.equation.valid
        @test witness_spec.metadata.matrices.left == K[c(0) c(0); c(0) c(1)]
        @test witness_spec.metadata.matrices.left == witness_spec.metadata.matrices.right
        @test witness_spec.metadata.induced_equality.valid
        @test witness_spec.metadata.induced_equality.exact == !(field isa CM.RealField)
        @test MD.component(witness_spec.metadata.induced_maps[1], 1) == reshape(K[c(1)], 1, 1)
        @test MD.component(witness_spec.metadata.induced_maps[2], 1) == reshape(K[c(1)], 1, 1)
        @test V.check_visual_spec(witness_spec).valid
        @test V.visual_spec(H; degree=1).metadata.matrices.left == reshape(K[c(1)], 1, 1)
        cheap = V.visual_spec(H; degree=0)
        @test !cheap.metadata.induced_computed
        @test cheap.metadata.induced_maps === nothing
        supplied = witness_spec.metadata.induced_maps
        @test V.visual_spec(H; degree=0, induced_maps=supplied).metadata.induced_equality.valid
        wrong = MD.zero_morphism(supplied[2].dom, supplied[2].cod)
        @test_throws ArgumentError V.visual_spec(H; degree=0, induced_maps=(supplied[1], wrong))
        @test_throws ArgumentError V.visual_spec(H; degree=99)
        @test_throws ArgumentError V.visual_spec(f; degree=false)
        @test_throws ArgumentError V.visual_spec(H; vertex=0)
        @test_throws ArgumentError V.visual_spec(H; matrix_limit=(0, 2))
        @test_throws ArgumentError V.visual_spec(f; induced=:yes)
        bad_h = MC.ModuleCochainHomotopy(f, g, [h0, MD.zero_morphism(C1, C0)]; check=false)
        @test_throws ArgumentError V.visual_spec(bad_h; degree=0)
        if field isa CM.RealField
            near_h1 = MD.PMorphism(C1, C0, [reshape([0.0, 1.0 + 1e-10], 2, 1)])
            near = MC.ModuleCochainHomotopy(f, g, [h0, near_h1]; check=false)
            near_spec = V.visual_spec(near; degree=0)
            @test near_spec.metadata.validation.valid
            @test !near_spec.metadata.validation.owner.valid
            @test 0 < near_spec.metadata.equation.residual < near_spec.metadata.equation.tolerance
        end
    end
end

@testset "A42b rejects a pointwise homotopy without naturality" begin
    V, MC = TamerOp.Visualization, TamerOp.ModuleComplexes
    field = CM.QQField()
    K = CM.coeff_type(field)
    P = FF.FinitePoset(Bool[1 1; 0 1])
    M = MD.PModule{K}(P, [1, 1], Dict((1, 2) => reshape(K[1], 1, 1)); field)
    Z = MD.zero_pmodule(P; field)
    C = MC.ModuleCochainComplex([M, M], [MD.zero_morphism(M, M)])
    z = MC.ModuleCochainMap(C, C, [MD.zero_morphism(M, M), MD.zero_morphism(M, M)])
    incompatible = MD.PMorphism(M, M, [reshape(K[1], 1, 1), reshape(K[2], 1, 1)])
    H = MC.ModuleCochainHomotopy(z, z, [MD.zero_morphism(M, Z), incompatible]; check=false)
    @test MC.check_module_homotopy(H).valid  # Pointwise identity alone misses this defect.
    @test_throws ArgumentError V.visual_spec(H; degree=0)
end

@testset "A42b projective resolution lifts and supplied homotopy" begin
    V = TamerOp.Visualization
    field = CM.QQField()
    K = CM.coeff_type(field)
    P = FF.FinitePoset(trues(1, 1))
    M = MD.PModule{K}(P, [1], Dict{Tuple{Int,Int},Matrix{K}}(); field)
    P0 = MD.PModule{K}(P, [2], Dict{Tuple{Int,Int},Matrix{K}}(); field)
    P1 = MD.PModule{K}(P, [1], Dict{Tuple{Int,Int},Matrix{K}}(); field)
    D = reshape(K[0, 1], 2, 1)
    d = MD.PMorphism(P1, P0, [D])
    aug = MD.PMorphism(P0, M, [reshape(K[1, 0], 1, 2)])
    res = DF.ProjectiveResolution(M, [P0, P1], [[1, 1], [1]], [d], [sparse(D)], aug)
    @test DF.check_projective_resolution(res).valid
    @test !DF.minimality_report(res; check_cover=false).minimal
    lift = [K[1 0; 0 1], reshape(K[1], 1, 1)]
    second = [K[1 0; 0 0], zeros(K, 1, 1)]
    witness = [reshape(K[0, 1], 1, 2), zeros(K, 0, 1)]
    f = MD.id_morphism(M)
    for degree in (0, 1)
        spec = V.visual_spec(res; kind=:resolution_lift, target_resolution=res, morphism=f, lift,
            comparison_lift=second, homotopy=witness, degree, vertex=1)
        @test spec.kind == :resolution_lift
        @test spec.metadata.degree_convention == :homological
        @test spec.metadata.checked_degrees == 0:1
        @test all(x -> x.equation.valid, spec.metadata.lift_checks)
        @test all(x -> x.equation.valid, spec.metadata.comparison_checks)
        @test all(x -> x.equation.valid, spec.metadata.homotopy_checks)
        @test spec.metadata.homotopy_supplied
        @test spec.metadata.induced_matrix == reshape(K[1], 1, 1)
        @test spec.metadata.lift_uniqueness == :not_asserted
        @test spec.panels[1].kind == :naturality_square
        @test V.check_visual_spec(spec).valid
    end
    minimal_view = V.visual_spec(res; kind=:resolution_lift, target_resolution=res, morphism=f, lift)
    @test minimal_view.metadata.homotopy_checks === nothing
    @test minimal_view.metadata.comparison_checks === nothing
    truncated = V.visual_spec(res; kind=:resolution_lift, target_resolution=res, morphism=f, lift=lift[1:1])
    @test truncated.metadata.checked_degrees == 0:0
    @test truncated.metadata.completeness == :only_supplied_degrees_checked
    @test V.visual_spec(res).kind == :betti_table
    @test_throws ArgumentError V.visual_spec(res; kind=:resolution_lift, target_resolution=res, morphism=f, lift, degree=2)
    @test_throws ArgumentError V.visual_spec(res; kind=:resolution_lift, target_resolution=res, morphism=f,
        lift=[zeros(K, 2, 2), zeros(K, 1, 1)])
    @test_throws ArgumentError V.visual_spec(res; kind=:resolution_lift, target_resolution=res, morphism=f,
        lift=[K[1 0; 0 1], zeros(K, 1, 1)])
    @test_throws ArgumentError V.visual_spec(res; kind=:resolution_lift, target_resolution=res, morphism=f, lift,
        comparison_lift=second, homotopy=[zeros(K, 1, 2), zeros(K, 0, 1)])
    @test_throws ArgumentError V.visual_spec(res; kind=:resolution_lift, target_resolution=res, morphism=f, lift, homotopy=witness)
    # A zero augmentation cannot certify which underlying module map is induced.
    broken = DF.ProjectiveResolution(M, [P0, P1], [[1, 1], [1]], [d], [sparse(D)], MD.zero_morphism(P0, M))
    @test_throws ArgumentError V.visual_spec(broken; kind=:resolution_lift, target_resolution=broken, morphism=f, lift)
end

@testset "A42b projective coefficient grades and forced zeros" begin
    V = TamerOp.Visualization
    field = CM.QQField()
    K = CM.coeff_type(field)
    P = FF.FinitePoset(Bool[1 1; 0 1])
    M = MD.PModule{K}(P, [1, 2], Dict((1, 2) => reshape(K[1, 0], 2, 1)); field)
    f = MD.id_morphism(M)
    res = DF.ProjectiveResolution(M, [M], [[1, 2]], MD.PMorphism{K}[],
        SparseMatrixCSC{K,Int}[], f)
    lift = [K[1 0; 0 1]]
    spec = V.visual_spec(res; kind=:resolution_lift, target_resolution=res, morphism=f, lift, vertex=2)
    coefficients = spec.panels[2]
    @test coefficients.metadata.structural_zero_mask == Bool[0 0; 1 0]
    @test coefficients.metadata.source_generators == [1, 2]
    @test only(coefficients.layers).entries[2, 1] == "0\u2020"
    @test only(coefficients.layers).entries[1, 2] == "0"
    @test only(coefficients.layers).row_labels == ["P@1 #1", "P@2 #2"]
    @test V.check_visual_spec(spec).valid
    cropped = V.visual_spec(res; kind=:resolution_lift, target_resolution=res, morphism=f, lift,
        vertex=2, matrix_limit=(1, 1)).panels[2]
    @test size(cropped.metadata.cell_roles) == (1, 1)
    @test size(cropped.metadata.structural_zero_mask) == (2, 2)
    @test cropped.metadata.truncated
    @test_throws ArgumentError V.visual_spec(res; kind=:resolution_lift, target_resolution=res, morphism=f,
        lift=[K[1 0; 1 1]], vertex=2)
end
