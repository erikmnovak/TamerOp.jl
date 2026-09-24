# Canonical package-mode entrypoint for Pkg.test and focused development runs.
# The package must load normally; packaging errors are test failures.
using Test
include("runner.jl")
const _TEST_CONFIG = TamerOpTestRunner._parse_arguments(ARGS)

if _TEST_CONFIG.help
    TamerOpTestRunner._print_help()
elseif _TEST_CONFIG.list
    foreach(println, _TEST_CONFIG.files)
else
    include("prelude.jl")
    BLAS.set_num_threads(1)
    println("TamerOp source: ", pathof(TamerOp))
    println("Julia ", VERSION, "; threads=", Threads.nthreads(:default), ",",
            Threads.nthreads(:interactive), "; BLAS threads=", BLAS.get_num_threads())
    println("Shared field loops: ", join(_TEST_CONFIG.fields, ", "))
    println("Owner files: ", join(_TEST_CONFIG.files, ", "))
    println("Testset prefixes: ", isempty(_TEST_CONFIG.prefixes) ? "all" : repr(_TEST_CONFIG.prefixes))
    TamerOpTestRunner._require_extensions!(@__MODULE__, TamerOp, _TEST_CONFIG.extensions)
    const _TEST_SELECTION = TamerOpTestRunner._SelectionState(_TEST_CONFIG.prefixes)
    @testset "TamerOp" begin
        for name in _TEST_CONFIG.files
            println("RUN owner ", name)
            flush(stdout)
            elapsed = @elapsed TamerOpTestRunner._include_test_file(
                @__MODULE__, joinpath(@__DIR__, name), _TEST_SELECTION)
            println("DONE owner ", name, "; elapsed_seconds=", elapsed)
            flush(stdout)
        end
        TamerOpTestRunner._finish_selection(_TEST_SELECTION)
    end
end
