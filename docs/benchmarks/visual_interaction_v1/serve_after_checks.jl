# Reproduce the declared native prewarming before the browser measurements.
const A123_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
cd(A123_ROOT)
append!(ARGS, [
    "--file=test_visualization_a92.jl", "--file=test_visualization_a93.jl",
    "--file=test_visualization_a41_sessions.jl", "--file=test_visualization_a41_live.jl",
    "--file=test_visualization_a38.jl", "--file=test_visualization_a84.jl",
    "--prefix=A92", "--prefix=A93", "--prefix=A123", "--prefix=A41",
    "--prefix=A38 retained", "--prefix=A84",
    "--require-extension=TamerOpCairoMakieExt",
    "--require-extension=TamerOpWGLMakieExt",
])
include(joinpath(A123_ROOT, "test", "runtests.jl"))
module A123Browser
include(joinpath(@__DIR__, "..", "..", "..", "test", "browser", "serve.jl"))
end
A123Browser._main()
