# Bounded algebra first-use workloads. Broader ingestion/invariant/rendering
# workloads belong to A19 after their own latency and image-size measurements.
using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    @compile_workload begin
        # Match the canonical small exact-algebra workflow, including fringe
        # conversion, resolution construction, coordinates and unit products.
        # QQ is the default algebra field; F2 covers a second scalar backend.
        for field in (CoreModules.QQField(), CoreModules.F2())
            K = CoreModules.coeff_type(field)
            P = FiniteFringe.FinitePoset(Bool[1 1; 0 1])
            S1 = IndicatorResolutions.pmodule_from_fringe(FiniteFringe.one_by_one_fringe(
                P, FiniteFringe.principal_upset(P, 1), FiniteFringe.principal_downset(P, 1),
                one(K); field=field))
            P1 = IndicatorResolutions.pmodule_from_fringe(FiniteFringe.one_by_one_fringe(
                P, FiniteFringe.principal_upset(P, 1), FiniteFringe.principal_downset(P, 2),
                one(K); field=field))
            M = Modules.direct_sum(S1, P1)
            E = Workflow.ext(M, M; maxdeg=1)
            DerivedFunctors.dim(E, 0)
            A = DerivedFunctors.ExtAlgebra(M, Options.DerivedFunctorOptions(maxdeg=1))
            x = DerivedFunctors.element(A, 0, K[1, 2, 1])
            one(A) * x
            x * one(A)
        end

        # The two-vertex resolution terminates before the nonzero higher-degree
        # map batches used on branching posets. This small diamond represents
        # the four vertex simples plus the constant module; its self-Ext
        # dimensions are (7, 4, 1), in fixed rational stalk bases. Explicit Hom
        # maps also compile conversion from basis columns to module morphisms.
        field = CoreModules.QQField()
        K = CoreModules.coeff_type(field)
        P = FiniteFringe.FinitePoset(Bool[1 1 1 1; 0 1 0 1; 0 0 1 1; 0 0 0 1])
        maps = Dict(
            (1, 2) => K[-3//44 3//11; -3//11 12//11],
            (1, 3) => K[-3//55 12//55; -3//11 12//11],
            (2, 4) => K[-2//57 10//57; -4//19 20//19],
            (3, 4) => K[-5//174 5//29; -5//29 30//29],
        )
        M = Modules.PModule{K}(P, fill(2, 4), maps; field=field)
        DerivedFunctors.basis(DerivedFunctors.Hom(M, M))
        E = DerivedFunctors.Ext(M, M, Options.DerivedFunctorOptions(maxdeg=2, model=:projective))
        DerivedFunctors.dim(E, 2)
    end
    # Retain compiled methods, not the training fixtures or native factors.
    # In particular, object-identity plan keys are process-local values.
    DerivedFunctors.Resolutions._clear_resolution_plan_caches!()
    DerivedFunctors.Functoriality._clear_functoriality_caches!()
    IndicatorResolutions._clear_indicator_prefix_caches!()
    FieldLinAlg._clear_fullcolumn_cache!()
    FieldLinAlg._clear_f2_fullcolumn_cache!()
    FieldLinAlg._clear_f3_fullcolumn_cache!()
    FieldLinAlg._reset_conversion_counters!()
end
