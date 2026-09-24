# Synthetic fixtures verify selection semantics without loading a second copy
# of TamerOp or duplicating the suite bootstrap.
function _runner_fixture(source, prefixes)
    target = Module(gensym(:RunnerFixture))
    Core.eval(target, :(using Test))
    Core.eval(target, :(include(path) = Base.include(@__MODULE__, path)))
    Core.eval(target, :(const events = String[]))
    state = TamerOpTestRunner._SelectionState(prefixes)
    mktempdir() do directory
        write(joinpath(directory, "helper.jl"), "fixture_helper() = 17\n")
        path = joinpath(directory, "fixture.jl")
        write(path, source)
        TamerOpTestRunner._include_test_file(target, path, state)
    end
    return Base.invokelatest(getfield, target, :events), state
end

@testset "A17 maintained test runner contracts" begin
    R = TamerOpTestRunner
    defaults = R._parse_arguments(String[])
    @test Tuple(defaults.files) == R._TEST_FILES
    @test Tuple(defaults.fields) == R._FIELD_NAMES
    @test isempty(defaults.prefixes)
    @test isempty(defaults.extensions)
    @test !defaults.list && !defaults.help
    @test length(unique(R._TEST_FILES)) == length(R._TEST_FILES)
    @test all(name -> isfile(joinpath(@__DIR__, name)), R._TEST_FILES)
    @test Set(R._TEST_FILES) == Set(filter(name -> startswith(name, "test_") && endswith(name, ".jl"),
                                         readdir(@__DIR__)))

    selected = R._parse_arguments(["--file=test_data_pipeline.jl", "--file=test_pl_backend.jl",
        "--prefix=A76", "--prefix=A78", "--fields=F3,QQ", "--require-extension=TamerOpTablesExt"])
    @test selected.files == ["test_data_pipeline.jl", "test_pl_backend.jl"]
    @test selected.prefixes == ["A76", "A78"]
    @test selected.fields == ["F3", "QQ"]
    @test selected.extensions == ["TamerOpTablesExt"]
    @test R._parse_arguments(["--list"]).list
    @test R._parse_arguments(["--help"]).help
    @test occursin("Pkg.test", sprint(R._print_help))
    for args in (
        ["--unknown"], ["test_data_pipeline.jl"], ["--file=missing.jl"],
        ["--file=../test/test_data_pipeline.jl"], ["--prefix="], ["--fields="],
        ["--fields=QQ,QQ"], ["--fields=qq"], ["--fields=F7"], ["--fields=QQ,"],
        ["--fields=QQ", "--fields=F3"], ["--list", "--list"],
        ["--help", "--list"], ["--require-extension=MissingExt"],
        ["--file=test_data_pipeline.jl", "--file=test_data_pipeline.jl"],
        ["--prefix=A76", "--prefix=A76"],
        ["--require-extension=TamerOpTablesExt", "--require-extension=TamerOpTablesExt"],
    )
        @test_throws ArgumentError R._parse_arguments(args)
    end

    source = raw"""
    include(joinpath(@__DIR__, "helper.jl"))
    const fields = ("QQ", "F3")
    for field in fields
        @testset "Parent $(field)" begin
            push!(events, "setup:" * field)
            @test fixture_helper() == 17
            @testset "Keep $(field)" begin
                push!(events, "keep:" * field)
                @test true
            end
            @testset "Discard $(field)" begin
                push!(events, "discard:" * field)
                @test true
            end
        end
    end
    @testset "Other parent" begin
        push!(events, "other:setup")
        @testset "Unrelated nested" begin
            push!(events, "other:nested")
        end
    end
    """
    events, state = _runner_fixture(source, ["Keep"])
    @test events == ["setup:QQ", "keep:QQ", "setup:F3", "keep:F3"]
    @test state.matches == Dict("Keep" => 2)
    @test R._finish_selection(state) === nothing
    events, state = _runner_fixture(source, ["Keep F3"])
    @test events == ["setup:QQ", "setup:F3", "keep:F3"]
    @test state.matches == Dict("Keep F3" => 1)
    # Parent setup is necessary to evaluate descendant names. Once selected,
    # a parent's descendants run without having to repeat their prefixes.
    events, state = _runner_fixture(source, ["Parent QQ", "Keep QQ"])
    @test events == ["setup:QQ", "keep:QQ", "discard:QQ", "setup:F3"]
    @test state.matches == Dict("Parent QQ" => 1, "Keep QQ" => 1)
    events, state = _runner_fixture(source, ["No such prefix"])
    @test isempty(events)
    @test_throws ErrorException R._finish_selection(state)
    events, state = _runner_fixture(source, ["Keep", "Missing"])
    @test state.matches == Dict("Keep" => 2)
    @test_throws ErrorException R._finish_selection(state)
    events, state = _runner_fixture(source, String[])
    @test events == ["setup:QQ", "keep:QQ", "discard:QQ", "setup:F3", "keep:F3",
                     "discard:F3", "other:setup", "other:nested"]
    @test R._finish_selection(state) === nothing

    # Dynamic names are evaluated once in the lexical scope that defines them.
    # Qualified macros, custom option placement, and local names that resemble
    # runner instrumentation must not change fixture behavior.
    dynamic = raw"""
    name = "Computed name"
    Test.@testset "$(name)" begin
        started = :fixture
        selected = :fixture
        @test started === selected
        push!(events, name)
    end
    """
    events, state = _runner_fixture(dynamic, ["Computed"])
    @test events == ["Computed name"]
    @test state.matches == Dict("Computed" => 1)
    @test_throws ArgumentError R._select_expression(Meta.parse("@testset begin @test true end"), state)
    # Quoted test syntax is data and must not be rewritten or counted.
    literal = Meta.parse("quote @testset \"Computed fake\" begin error(\"not executed\") end end")
    @test R._select_expression(literal, state) === literal

    concurrent = R._SelectionState(["parallel"])
    @sync for _ in 1:32
        Threads.@spawn R._match!(concurrent, "parallel fixture")
    end
    @test concurrent.matches == Dict("parallel" => 32)

    # The maintained runner must have one shared bootstrap and ordinary
    # package loading. In particular, no fallback may hide a load failure.
    prelude = read(joinpath(@__DIR__, "prelude.jl"), String)
    @test occursin("using TamerOp", prelude)
    @test !occursin("using .TamerOp", prelude)
    @test !occursin("include(joinpath(_TO_SRC_DIR", prelude)
    @test !isfile(joinpath(@__DIR__, "..", ".codex_focus_tests.jl"))
end

module _A79InstalledSource
    include("installation/common.jl")
end

@testset "A79 installed source identity across checkout conventions" begin
    H = _A79InstalledSource
    mktempdir() do directory
        source = joinpath(directory, "source")
        output = joinpath(directory, "evidence")
        mkpath(joinpath(source, "src"))
        mkpath(output)
        attrs = joinpath(source, ".gitattributes")
        code = joinpath(source, "src", "demo.jl")
        binary = joinpath(source, "binary.bin")
        write(attrs, "* text=auto eol=lf\n")
        write(code, "x = 17\n")
        write(binary, UInt8[0, 255, 13, 10])
        # Fixed independent `git hash-object --stdin` answers, including raw
        # binary CRLF bytes which must never undergo text normalization.
        blobs = Dict(".gitattributes" => "6313b56c57848efce05faa7aa7e901ccfc2886ea",
                     "src/demo.jl" => "d8b8a2389cde43a205eeea35ca19d687311a09d3",
                     "binary.bin" => "00822ce7dfc6f27759b94e2c7dfd26f25afbac9d")
        entries = [Dict("path" => p, "mode" => "100644", "blob" => b) for (p,b) in blobs]
        tree = repeat("0", 40) # This unit fixture exercises comparison, not Git authentication.
        inventory = joinpath(output, "inventory.toml")
        H.write_report(inventory, Dict("tree" => tree, "files" => entries))
        config = Dict("inventory" => inventory, "tree" => tree, "output" => output)
        @test isempty(H.verify_source(config, source; windows=true))
        if !Sys.iswindows()
            @test isempty(H.verify_source(config, source; windows=false))
        end
        write(attrs, "* text=auto eol=lf\r\n")
        @test H.verify_source(config, source; windows=true) == [".gitattributes: CRLF to LF"]
        redirect_stderr(devnull) do
            if !Sys.iswindows()
                @test_throws ErrorException H.verify_source(config, source; windows=false)
            end
            write(attrs, "* text=auto eol=crlf\r\n")
            @test_throws ErrorException H.verify_source(config, source; windows=true)
            write(attrs, "* text=auto eol=lf\r\n")
            write(code, "x = 17\r\n")
            @test_throws ErrorException H.verify_source(config, source; windows=true)
            write(code, "x = 18\n")
            @test_throws ErrorException H.verify_source(config, source; windows=true)
            write(code, "x = 17\n")
            write(binary, UInt8[0, 255, 10])
            @test_throws ErrorException H.verify_source(config, source; windows=true)
            rm(binary)
            @test_throws ErrorException H.verify_source(config, source; windows=true)
            write(binary, UInt8[0, 255, 13, 10])
            write(joinpath(source, "extra"), "extra")
            @test_throws ErrorException H.verify_source(config, source; windows=true)
            rm(joinpath(source, "extra"))
            mkdir(joinpath(source, "empty"))
            @test_throws ErrorException H.verify_source(config, source; windows=true)
            rm(joinpath(source, "empty"))
            if !Sys.iswindows()
                rm(code)
                symlink("../binary.bin", code)
                @test_throws ErrorException H.verify_source(config, source; windows=true)
                rm(code)
                write(code, "x = 17\n")
                write(attrs, "* text=auto eol=lf\n")
                chmod(code, 0o755)
                @test_throws ErrorException H.verify_source(config, source; windows=false)
                @test isempty(H.verify_source(config, source; windows=true))
                chmod(code, 0o644)
            end
        end
        write(attrs, "* text=auto eol=lf\n")
        @test isempty(H.verify_source(config, source; windows=true))
    end
end
