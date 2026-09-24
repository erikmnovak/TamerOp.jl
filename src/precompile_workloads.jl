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
