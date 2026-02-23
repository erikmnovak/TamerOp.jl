# =============================================================================
# Example 00: quickstart (one object -> one feature row)
#
# What this teaches
# -----------------
# 1) Define a typed filtration with explicit construction/pipeline controls.
# 2) Encode one object into an `EncodingResult`.
# 3) Read a couple of core invariants for sanity.
# 4) Vectorize with one canonical featurizer spec.
# 5) Save a statistics-ready feature row + reproducibility artifacts.
#
# This file is intentionally direct and linear (minimal helper indirection), so
# new users can follow the entire data -> features path at a glance.
# =============================================================================

include(joinpath(@__DIR__, "_common.jl"))

Random.seed!(20260217)
outdir = example_outdir("00_quickstart")

println("\n[00] Quickstart: one object -> one feature row")

# -----------------------------------------------------------------------------
# 1) Build one deterministic sample
# -----------------------------------------------------------------------------

# Keep n modest so quickstart stays fast on a laptop.
cloud = make_noisy_circle_cloud(12; noise=0.03, seed=11)

# -----------------------------------------------------------------------------
# 2) Define filtration + construction policy
# -----------------------------------------------------------------------------

# ConstructionOptions are the main blow-up guardrails in ingestion.
construction = PM.ConstructionOptions(
    ;
    # Build a kNN edge graph first (instead of dense pairwise expansion).
    sparsify=:knn,
    # Do not apply graph/complex collapse reductions in this quickstart.
    collapse=:none,
    # Default return stage when encode(...; stage=:auto) is used.
    output_stage=:encoding_result,
    # Hard caps: (max_simplices, max_edges, memory_budget_bytes).
    # If estimates exceed these, ingestion throws before blowing up.
    budget=(max_simplices=80_000, max_edges=20_000, memory_budget_bytes=600_000_000),
)

# Rips-density is a canonical 2-parameter point-cloud ingestion family.
filtration = PM.FiltrationSpec(
    ;
    kind=:rips_density,
    # max_dim=0 keeps only vertices in the simplicial layer (fastest quickstart).
    max_dim=0,
    # kNN graph sparsification + density coordinate both use k=4 here.
    knn=4,
    density_k=4,
    # :auto uses extension-backed nearest-neighbor backend when available.
    nn_backend=:auto,
)

# PipelineOptions control axis/orientation/quantization semantics uniformly.
pipeline = PM.PipelineOptions(
    ;
    orientation=(1, 1),
    axes_policy=:encoding,
    axis_kind=:rn,
    poset_kind=:signature,
)

# For quickstart speed and deterministic arithmetic, use F2.
field = PM.CoreModules.F2()
println("filtration type: ", typeof(filtration))

# -----------------------------------------------------------------------------
# 3) Encode
# -----------------------------------------------------------------------------

# `encode(...; degree=t)` returns the cohomology module H^t extracted
# from the constructed cochain complex (t = 0 here).
# In the simple path, `stage` is one of
# :auto | :simplex_tree | :graded_complex | :cochain | :module |
# :fringe | :flange | :encoding_result.
# We keep `stage=:auto` (default), which means "use construction.output_stage".
# `cache=:auto` gives per-call caching without requiring manual SessionCache setup.
enc = PM.encode(
    cloud,
    filtration;
    degree=0,
    field=field,
    cache=:auto,
    construction=construction,
    pipeline=pipeline,
)
println("enc type: ", typeof(enc))
# `enc.P` is the finite poset, `enc.M` the module over that poset,
# and `enc.pi` the classifier/encoding map from original grades to poset vertices.
println("enc.M type (module): ", typeof(enc.M))
println("enc.pi type (classifier): ", typeof(enc.pi))
println("Encoded poset size (before coarsen): ", length(enc.M.dims))

# Optional single-module coarsening: keeps this quickstart feature vector small
# while preserving the same user-facing workflow.
# This is safe for single-module summaries; for pairwise comparisons, coarsen jointly.
enc_small = PM.coarsen(enc; method=:uptight)
println("Encoded poset size (after coarsen):  ", length(enc_small.M.dims))

# -----------------------------------------------------------------------------
# 4) Sanity invariants
# -----------------------------------------------------------------------------

opts = PM.CoreModules.InvariantOptions(
    ;
    # Use encoding axes directly so invariants align with the encoded geometry.
    axes_policy=:encoding,
    max_axis_len=48,
    threads=false,
    strict=true,
    # PL mode contract is strict: :fast or :verified.
    pl_mode=:fast,
)

# rank_invariant returns sparse rank data over comparable poset pairs.
rank_tbl = PM.rank_invariant(enc_small; opts=opts, store_zeros=false)
# restricted_hilbert gives per-vertex dimensions dim(M_p).
rh = PM.restricted_hilbert(enc_small.M)

println("rank_invariant nonzero entries: ", length(rank_tbl))
println("restricted_hilbert nonzero count: ", count(!iszero, rh))

# -----------------------------------------------------------------------------
# 5) Canonical vectorization
# -----------------------------------------------------------------------------

# RankGridSpec gives an explicit fixed-size vector from the encoded module.
# Here feature dimension is nvertices(enc_small.P)^2 when storing full grid entries.
spec = PM.RankGridSpec(nvertices=length(enc_small.M.dims), store_zeros=false, threads=false)

vec = PM.transform(spec, enc_small; opts=opts, threaded=false)
println("nfeatures(spec): ", PM.nfeatures(spec))
println("length(transform(spec, enc_small)): ", length(vec))

# Wrap into one-row FeatureSet so export paths are identical to batch workflows.
fs = PM.FeatureSet(
    reshape(vec, 1, :),
    PM.feature_names(spec),
    ["quickstart_001"],
    (label=["circle"], note="example_00"),
)

# -----------------------------------------------------------------------------
# 6) Save outputs
# -----------------------------------------------------------------------------

paths = save_feature_bundle(outdir, "quickstart", fs)
println("Manual wide CSV: ", paths.manual_wide)
println("Manual long CSV: ", paths.manual_long)
println("Native optional outputs: ", paths.native)

# Save dataset + pipeline as reproducibility artifacts.
# Pipeline JSON stores filtration + options contract, so this run is replayable.
dset_path = joinpath(outdir, "dataset.json")
pipe_path = joinpath(outdir, "pipeline.json")
PM.save_dataset_json(dset_path, cloud)
PM.save_pipeline_json(
    pipe_path,
    cloud,
    filtration;
    degree=0,
    pipeline_opts=PM.PipelineOptions(
        orientation=pipeline.orientation,
        axes_policy=pipeline.axes_policy,
        axis_kind=pipeline.axis_kind,
        poset_kind=pipeline.poset_kind,
        field=:F2,
    ),
)

println("Saved dataset JSON: ", dset_path)
println("Saved pipeline JSON: ", pipe_path)
println("\nDone. Output directory: ", outdir)
