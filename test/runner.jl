# Test-only orchestration. No source-loading fallback belongs in this runner.
module TamerOpTestRunner

using TOML

const _TEST_FILES = (
    "test_contracts.jl",
    "test_test_runner.jl",
    "test_field_linalg.jl",
    "test_finite_fringe.jl",
    "test_encoding.jl",
    "test_poset_interface.jl",
    "test_zn_backend.jl",
    "test_pl_backend.jl",
    "test_geometry.jl",
    "test_data_pipeline.jl",
    "test_ordinary_persistence.jl",
    "test_synthetic_data.jl",
    "test_indicator_resolutions.jl",
    "test_derived_functors.jl",
    "test_model_independent_ext_layer.jl",
    "test_chain_complexes_homology.jl",
    "test_functoriality_ext_tor_maps.jl",
    "test_invariants.jl",
    "test_visualization.jl",
    "test_featurizers.jl",
    "test_extensions.jl",
    "test_examples.jl",
    "test_random_stress.jl",
)
const _FIELD_NAMES = ("QQ", "F2", "F3", "F5", "Real64")
const _PROJECT_FILE = normpath(joinpath(@__DIR__, "..", "Project.toml"))

struct _Configuration
    files::Vector{String}
    prefixes::Vector{String}
    fields::Vector{String}
    extensions::Vector{String}
    list::Bool
    help::Bool
end

function _parse_arguments(args; project_file=_PROJECT_FILE)
    files, prefixes, extensions = String[], String[], String[]
    fields = collect(_FIELD_NAMES)
    seen_fields = false
    list = help = false
    declared_extensions = TOML.parsefile(project_file)["extensions"]
    for arg in args
        if arg == "--list"
            list && throw(ArgumentError("--list was specified twice"))
            list = true
        elseif arg == "--help"
            help && throw(ArgumentError("--help was specified twice"))
            help = true
        else
            pair = split(arg, '='; limit=2)
            length(pair) == 2 || throw(ArgumentError("Unknown test argument: $arg; use --help"))
            key, value = pair
            isempty(value) && throw(ArgumentError("$key requires a nonempty value"))
            if key == "--file"
                value in _TEST_FILES || throw(ArgumentError("Unknown test file: $value; use --list"))
                value in files && throw(ArgumentError("Duplicate test file: $value"))
                push!(files, value)
            elseif key == "--prefix"
                value in prefixes && throw(ArgumentError("Duplicate test prefix: $value"))
                push!(prefixes, value)
            elseif key == "--fields"
                seen_fields && throw(ArgumentError("--fields was specified twice"))
                seen_fields = true
                fields = String.(split(value, ','; keepempty=true))
                all(in(_FIELD_NAMES), fields) || throw(ArgumentError(
                    "Fields must be a comma-separated subset of $(join(_FIELD_NAMES, ','))"))
                length(unique(fields)) == length(fields) || throw(ArgumentError("Duplicate field selection"))
            elseif key == "--require-extension"
                haskey(declared_extensions, value) || throw(ArgumentError("Unknown declared extension: $value"))
                value in extensions && throw(ArgumentError("Duplicate extension: $value"))
                push!(extensions, value)
            else
                throw(ArgumentError("Unknown test argument: $key; use --help"))
            end
        end
    end
    help && length(args) != 1 && throw(ArgumentError("Use --help by itself"))
    isempty(files) && append!(files, _TEST_FILES)
    return _Configuration(files, prefixes, fields, extensions, list, help)
end

function _print_help(io=stdout)
    println(io, "Usage: julia --project=. test/runtests.jl [options]")
    println(io, "  --file=test_NAME.jl            Repeat to select owner files; default: all")
    println(io, "  --prefix=TEXT                  Repeat to select testset name prefixes")
    println(io, "  --fields=QQ,F2,F3,F5,Real64    Select shared parameterized field loops")
    println(io, "  --require-extension=NAME      Load and require a declared package extension")
    println(io, "  --list                         List selected owner files without loading TamerOp")
    println(io, "  --help                         Print this help")
    println(io, "Pkg.test(test_args=[...]) accepts the same arguments. Thread counts use Julia's --threads.")
    println(io, "Fixed-field and cross-characteristic fixtures keep their explicit fields.")
end

function _require_extensions!(target::Module, package::Module, names; project_file=_PROJECT_FILE)
    declarations = TOML.parsefile(project_file)["extensions"]
    for name in names
        dependencies = declarations[name]
        dependencies isa String && (dependencies = [dependencies])
        for dependency in dependencies
            Base.find_package(dependency) === nothing && error(
                "Required extension $name needs $dependency in the active test environment")
            # Normal package import must activate the extension. An activation
            # failure is never repaired by directly including ext/*.jl.
            Core.eval(target, Expr(:import, Expr(:., Symbol(dependency))))
        end
        Base.get_extension(package, Symbol(name)) === nothing && error(
            "Required extension $name did not activate after importing its dependencies")
        println("Required extension active: ", name)
    end
    return nothing
end

mutable struct _SelectionState
    prefixes::Vector{String}
    matches::Dict{String,Int}
    lock::ReentrantLock
end
_SelectionState(prefixes) = _SelectionState(collect(String, prefixes), Dict{String,Int}(), ReentrantLock())

function _match!(state::_SelectionState, name)
    description = String(name)
    matched = false
    lock(state.lock) do
        for prefix in state.prefixes
            if startswith(description, prefix)
                state.matches[prefix] = get(state.matches, prefix, 0) + 1
                matched = true
            end
        end
    end
    return matched
end

function _finish_selection(state::_SelectionState)
    missing = filter(prefix -> !haskey(state.matches, prefix), state.prefixes)
    isempty(missing) || error("No executed testset matched requested prefix(es): " * join(missing, ", "))
    for prefix in state.prefixes
        println("Executed prefix ", repr(prefix), ": ", state.matches[prefix], " matching testsets")
    end
    return nothing
end

_is_testset(expr) = expr isa Expr && expr.head == :macrocall &&
    (expr.args[1] == Symbol("@testset") ||
     (expr.args[1] isa GlobalRef && expr.args[1].name == Symbol("@testset")) ||
     expr.args[1] == Expr(:., :Test, QuoteNode(Symbol("@testset"))))

# Named block testsets are the suite's canonical form. Accept interpolated names
# and do not silently discard unfamiliar syntax.
function _name_index(expr)
    arguments = expr.args
    body = arguments[end]
    body isa Expr && body.head == :block || throw(ArgumentError(
        "Focused selection requires a named @testset ... begin block; found $(repr(expr))"))
    candidates = Int[]
    for i in 3:(length(arguments) - 1)
        argument = arguments[i]
        (argument isa String ||
         argument isa Expr && argument.head == :string) && push!(candidates, i)
    end
    isempty(candidates) && throw(ArgumentError("Focused selection requires a named @testset"))
    # A custom testset type may precede the description; it is the last ordinary
    # argument before the body/options that supplies the name.
    return last(candidates)
end

function _possible_name(name, prefixes)
    name isa String && return any(prefix -> startswith(name, prefix), prefixes)
    if name isa Expr && name.head == :string && !isempty(name.args) && name.args[1] isa String
        leading = name.args[1]
        return any(prefix -> startswith(leading, prefix) || startswith(prefix, leading), prefixes)
    end
    return true
end

function _has_candidate(expr, prefixes)
    expr isa Expr || return false
    expr.head in (:quote, :inert) && return false
    if _is_testset(expr)
        _possible_name(expr.args[_name_index(expr)], prefixes) && return true
    end
    return any(argument -> _has_candidate(argument, prefixes), expr.args)
end

function _select_expression(expr, state::_SelectionState, inherited=false)
    expr isa Expr || return expr
    expr.head in (:quote, :inert) && return expr
    if _is_testset(expr)
        # Eliminate unrelated bodies before lowering/compilation. A runtime
        # guard alone would still compile whole owner files for tiny selections.
        inherited === false && !_has_candidate(expr, state.prefixes) && return :(nothing)
        index = _name_index(expr)
        name = expr.args[index]
        runtime_name, selected = gensym(:test_name), gensym(:test_selected)
        transformed = copy(expr)
        transformed.args[index] = Expr(:string, runtime_name)
        child_inherited = inherited === false && !_possible_name(name, state.prefixes) ? false : selected
        transformed.args[end] = _select_expression(expr.args[end], state, child_inherited)
        has_child = _has_candidate(expr.args[end], state.prefixes)
        # Evaluate the description once, in its original lexical field/loop
        # context. Matching a parent executes all its children. Selecting a
        # descendant preserves parent setup and its ordinary assertions.
        return quote
            let $runtime_name = $name,
                $selected = $(GlobalRef(@__MODULE__, :_match!))($state, $runtime_name) | $inherited
                if $selected || $has_child
                    $transformed
                end
            end
        end
    end
    return Expr(expr.head, map(argument -> _select_expression(argument, state, inherited), expr.args)...)
end

function _include_test_file(target::Module, path, state::_SelectionState)
    if isempty(state.prefixes)
        return Base.include(target, path)
    end
    return Base.include(expr -> _select_expression(expr, state), target, path)
end

end # module
