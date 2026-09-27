#!/usr/bin/env julia
"""
    julia --project=. docs/build_scripts/api_inventory.jl [--check] [OUTPUT]

Reconcile curated declarations, runtime exports, and qualified owner bindings.
OUTPUT defaults to docs/api_inventory.toml, relative to this script's project.
`--check` compares the snapshot without writing and exits nonzero if stale.

Schema 1 stores one `objects` record per function/type/module identity. Constants
are separate binding records even when their values compare equal. Each object
contains its qualified aliases and binding-level declaration/export/doc metadata.
`canonical_binding` is a deterministic documentation key, not a claim that every
method belongs to that module. `defining_module` records Julia's runtime owner;
`declaration_owners` records the curated binding tables' sources. Owner candidates
are explicitly NOT a public-API denominator: their names/docs warrant review,
not automatic promotion. Doc presence does not certify semantic or teaching
coverage, and a generic with one documented method may have undocumented methods.

Only TamerOp and stdlibs are loaded; optional integrations are not activated by
this script. The snapshot records any extensions loaded transitively. No clock,
checkout path, or Git commit is embedded, so unchanged inputs produce identical
bytes. Source fingerprints cover Project.toml, src/, ext/, and this generator.
"""
module APIInventory

import TamerOp
using SHA
using TOML

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const DEFAULT_OUTPUT = joinpath(ROOT, "docs", "api_inventory.toml")
const SCHEMA_VERSION = 1
const GENERATOR_VERSION = 1
const ROOT_MODULE = TamerOp
const ADVANCED_MODULE = TamerOp.Advanced

_module_name(m::Module) = join(string.(Base.fullname(m)), ".")
_qualified(m::Module, s::Symbol) = string(_module_name(m), '.', s)
_is_package_module(m::Module) = m === ROOT_MODULE || startswith(_module_name(m), "TamerOp.")
_identity_grouped(x) = x isa Function || x isa Type || x isa Module
_kind(x) = x isa Module ? "module" : x isa Type ? "type" : x isa Function ? "function" : "constant"

function _exports(m::Module)
    # names() includes the module's self-binding. Never count that as an API.
    return sort!([s for s in names(m; all=true, imported=true)
                  if Base.isexported(m, s) && s !== nameof(m)]; by=string)
end

function _hasdoc(m::Module, s::Symbol)
    return isdefined(m, s) && Base.Docs.hasdoc(m, s)
end

function _source_metadata()
    paths = String["Project.toml", "docs/build_scripts/api_inventory.jl"]
    for directory in ("src", "ext")
        for (dir, _, files) in walkdir(joinpath(ROOT, directory))
            for file in files
                endswith(file, ".jl") && push!(paths, relpath(joinpath(dir, file), ROOT))
            end
        end
    end
    sort!(paths)
    file_hashes = Dict(path => bytes2hex(sha256(read(joinpath(ROOT, path)))) for path in paths)
    fingerprint_input = join((string(path, '\0', file_hashes[path]) for path in paths), '\n')
    return Dict("algorithm" => "sha256", "files" => file_hashes,
                "fingerprint" => bytes2hex(sha256(fingerprint_input)))
end

function _binding!(records, modules, m::Module, s::Symbol;
                   declaration::String="", source::String="")
    key = _qualified(m, s)
    scope = m === ROOT_MODULE ? "root" : m === ADVANCED_MODULE ? "advanced" : "owner"
    record = get!(records, key) do
        defined = isdefined(m, s)
        Dict{String,Any}(
            "qualified_name" => key, "module" => _module_name(m), "name" => string(s),
            "scope" => scope, "defined" => defined, "exported" => Base.isexported(m, s),
            "declared_simple" => false, "declared_advanced" => false,
            "curated_sources" => String[], "declaration_sources" => String[],
            "has_docstring" => _hasdoc(m, s),
        )
    end
    modules[key] = (m, s)
    if declaration != ""
        push!(record["declaration_sources"], declaration)
        declaration == "SIMPLE_API" && (record["declared_simple"] = true)
        declaration == "ADVANCED_API" && (record["declared_advanced"] = true)
    end
    source != "" && push!(record["curated_sources"], source)
    return record
end

function _binding_table!(records, modules, table, target::Module, label::String)
    rows = Dict{String,Any}[]
    for (modsym, syms) in table
        isdefined(target, modsym) || error("$label owner $target.$modsym is undefined")
        owner = getfield(target, modsym)
        owner isa Module || error("$label owner $target.$modsym is not a module")
        for sym in syms
            owner_record = _binding!(records, modules, owner, sym; source=label)
            target_record = _binding!(records, modules, target, sym; source=label)
            matches = owner_record["defined"] && target_record["defined"] &&
                      getfield(owner, sym) === getfield(target, sym)
            push!(rows, Dict("table" => label, "name" => string(sym),
                             "declaration_owner" => _module_name(owner),
                             "owner_binding" => owner_record["qualified_name"],
                             "target_binding" => target_record["qualified_name"],
                             "owner_defined" => owner_record["defined"],
                             "target_matches_owner" => matches))
        end
    end
    return rows
end

function _defining_module(value)
    if value isa Module
        return _module_name(parentmodule(value))
    elseif value isa Function || value isa Type
        return _module_name(parentmodule(value))
    end
    return ""
end

function _canonical_record(group, values)
    # Prefer a curated owner binding over a facade. Within owners prefer Julia's
    # defining module, then the object's own name, then a stable lexical order.
    return first(sort(group; by=record -> begin
        key = record["qualified_name"]
        value = get(values, key, nothing)
        defining = record["defined"] ? _defining_module(value) : ""
        own_name = value isa Function || value isa Type || value isa Module ? string(nameof(value)) : ""
        (record["scope"] == "owner" ? 0 : record["scope"] == "root" ? 1 : 2,
         record["module"] == defining ? 0 : 1,
         record["name"] == own_name ? 0 : 1, key)
    end))
end

function _objects(records, modules)
    groups = Vector{Vector{Dict{String,Any}}}()
    identities = IdDict{Any,Int}()
    values = Dict{String,Any}()
    for key in sort!(collect(keys(records)))
        record = records[key]
        sort!(unique!(record["declaration_sources"]))
        sort!(unique!(record["curated_sources"]))
        m, sym = modules[key]
        if record["defined"]
            value = getfield(m, sym)
            values[key] = value
            if _identity_grouped(value)
                index = get(identities, value, 0)
                if index != 0
                    push!(groups[index], record)
                    continue
                end
                identities[value] = length(groups) + 1
            end
        end
        push!(groups, [record])
    end
    objects = Dict{String,Any}[]
    for group in groups
        canonical = _canonical_record(group, values)
        key = canonical["qualified_name"]
        value = get(values, key, nothing)
        owners = sort!(unique(String[r["module"] for r in group if r["scope"] == "owner"]))
        tiers = String[]
        any(r -> r["declared_simple"] || (r["scope"] == "root" && r["exported"]), group) && push!(tiers, "simple")
        any(r -> r["declared_advanced"] || (r["scope"] == "advanced" && r["exported"]), group) && push!(tiers, "advanced")
        !isempty(owners) && push!(tiers, "qualified")
        push!(objects, Dict(
            "canonical_binding" => key, "canonical_owner" => canonical["module"],
            "defining_module" => canonical["defined"] ? _defining_module(value) : "",
            "kind" => canonical["defined"] ? _kind(value) : "undefined",
            "aliases" => sort!(String[r["qualified_name"] for r in group]),
            "declaration_owners" => owners, "tiers" => tiers,
            "has_docstring" => any(r -> r["has_docstring"], group),
            # A preexisting unrelated target binding can prevent a table entry
            # from binding/exporting its intended owner. Keep that evidence, but
            # do not promote the unrelated object into the public denominator.
            "intended_public" => !isempty(tiers),
            "bindings" => sort!(group; by=r -> r["qualified_name"]),
        ))
    end
    return sort!(objects; by=o -> o["canonical_binding"])
end

function _owner_modules()
    found = IdDict{Module,Nothing}()
    function visit(m::Module)
        haskey(found, m) && return
        found[m] = nothing
        for s in names(m; all=true, imported=true)
            isdefined(m, s) || continue
            value = getfield(m, s)
            value isa Module || continue
            value === ADVANCED_MODULE && continue
            parentmodule(value) === m && _is_package_module(value) && visit(value)
        end
    end
    visit(ROOT_MODULE)
    return sort!([m for m in keys(found) if m !== ROOT_MODULE]; by=_module_name)
end

function _owner_candidates(records, modules, objects)
    candidates = Dict{String,Any}[]
    candidate_values = Dict{String,Any}()
    public_objects = IdDict{Any,String}()
    for object in objects
        object["intended_public"] || continue
        key = object["canonical_binding"]
        m, sym = modules[key]
        isdefined(m, sym) || continue
        value = getfield(m, sym)
        _identity_grouped(value) && (public_objects[value] = key)
    end
    for m in _owner_modules()
        for s in sort!(unique(names(m; all=true, imported=true, usings=true)); by=string)
            name = string(s)
            startswith(name, "_") && continue
            startswith(name, "#") && continue
            s in (:eval, :include) && continue
            isdefined(m, s) || continue
            defining_binding_module = try
                Base.binding_module(m, s)
            catch err
                # names(...; usings=true) also exposes ambiguous imports. They
                # have no unique binding owner and provide no local API intent.
                err isa ErrorException && err.msg == "Constant binding was imported from multiple modules" || rethrow()
                continue
            end
            imported = defining_binding_module !== m
            # An owner often fronts its own nested implementation (notably
            # CoreModules.CoeffFields). Retain those aliases as candidates, but
            # omit unrelated-owner/Base imports: using a generic is not API intent.
            child_alias = imported && startswith(_module_name(defining_binding_module), _module_name(m) * ".")
            imported && !child_alias && continue
            qualified = _qualified(m, s)
            haskey(records, qualified) && continue
            value = getfield(m, s)
            value === m && continue
            doc = _hasdoc(m, s)
            public = Base.ispublic(m, s)
            # An undocumented scalar/config constant is not an API candidate on
            # capitalization alone. Local functions/types remain review candidates.
            (_identity_grouped(value) || doc || public) || continue
            push!(candidates, Dict(
                "qualified_name" => qualified, "module" => _module_name(m),
                "name" => name, "kind" => _kind(value), "has_docstring" => doc,
                "explicit_public" => public, "exported" => Base.isexported(m, s),
                "intended_public" => false, "review_required" => true,
                "imported_from_child" => child_alias,
                "known_public_object" => _identity_grouped(value) && haskey(public_objects, value),
                "canonical_binding" => _identity_grouped(value) ? get(public_objects, value, "") : "",
                "reason" => public ? "explicit_public_outside_curated_tables" :
                            child_alias ? "child_owner_alias_outside_curated_tables" :
                            doc ? "documented_local_binding_outside_curated_tables" :
                                  "nonunderscore_local_binding_requires_intent_review",
            ))
            candidate_values[qualified] = value
        end
    end
    sort!(candidates; by=c -> c["qualified_name"])
    identities = IdDict{Any,String}()
    for candidate in candidates
        key = candidate["qualified_name"]
        value = candidate_values[key]
        candidate["candidate_identity"] = if _identity_grouped(value)
            get!(identities, value) do
                canonical = candidate["canonical_binding"]
                isempty(canonical) ? key : canonical
            end
        else
            key
        end
    end
    return candidates
end

_string_set(xs) = sort!(string.(collect(Set(xs))))

function build_inventory()
    simple = Set(TamerOp.SIMPLE_API)
    advanced = Set(TamerOp.ADVANCED_API)
    root_exports = Set(_exports(ROOT_MODULE))
    advanced_exports = Set(_exports(ADVANCED_MODULE))
    records = Dict{String,Dict{String,Any}}()
    modules = Dict{String,Tuple{Module,Symbol}}()
    for s in simple
        _binding!(records, modules, ROOT_MODULE, s; declaration="SIMPLE_API")
    end
    for s in advanced
        _binding!(records, modules, ADVANCED_MODULE, s; declaration="ADVANCED_API")
    end
    for (m, syms) in ((ROOT_MODULE, root_exports), (ADVANCED_MODULE, advanced_exports))
        for s in syms
            _binding!(records, modules, m, s)
        end
    end
    rows = _binding_table!(records, modules, TamerOp.SIMPLE_API_BINDINGS, ROOT_MODULE, "SIMPLE_API_BINDINGS")
    append!(rows, _binding_table!(records, modules, TamerOp.ADVANCED_ONLY_API_BINDINGS,
                                 ADVANCED_MODULE, "ADVANCED_ONLY_API_BINDINGS"))
    sort!(rows; by=r -> (r["table"], r["declaration_owner"], r["name"]))
    simple_bindings = Set(Symbol(r["name"]) for r in rows if r["table"] == "SIMPLE_API_BINDINGS")
    advanced_bindings = Set(Symbol(r["name"]) for r in rows if r["table"] == "ADVANCED_ONLY_API_BINDINGS")
    objects = _objects(records, modules)
    candidates = _owner_candidates(records, modules, objects)
    rec = Dict(
        "simple_declared_not_exported" => _string_set(setdiff(simple, root_exports)),
        "simple_declared_undefined" => _string_set(s for s in simple if !isdefined(ROOT_MODULE, s)),
        "advanced_declared_not_exported" => _string_set(setdiff(advanced, advanced_exports)),
        "advanced_declared_undefined" => _string_set(s for s in advanced if !isdefined(ADVANCED_MODULE, s)),
        "root_exports_absent_simple_declaration" => _string_set(setdiff(root_exports, simple)),
        "advanced_exports_absent_advanced_declaration" => _string_set(setdiff(advanced_exports, advanced)),
        "simple_binding_names_absent_declaration" => _string_set(setdiff(simple_bindings, simple)),
        "advanced_binding_names_absent_declaration" => _string_set(setdiff(advanced_bindings, advanced)),
        "simple_binding_names_not_exported" => _string_set(setdiff(simple_bindings, root_exports)),
        "advanced_binding_names_not_exported" => _string_set(setdiff(advanced_bindings, advanced_exports)),
        "advanced_binding_names_undefined" => _string_set(s for s in advanced_bindings if !isdefined(ADVANCED_MODULE, s)),
        "declared_simple_absent_simple_binding_table" => _string_set(setdiff(simple, simple_bindings)),
        "declared_advanced_absent_binding_tables" => _string_set(setdiff(advanced, union(simple_bindings, advanced_bindings))),
        "undefined_curated_owner_bindings" => sort!(unique(String[r["owner_binding"] for r in rows if !r["owner_defined"]])),
        "binding_owner_target_mismatches" => [r for r in rows if !r["target_matches_owner"]],
    )
    totals = Dict{String,Any}(
        "simple_declarations" => length(simple), "advanced_declarations" => length(advanced),
        "simple_binding_rows" => count(r -> r["table"] == "SIMPLE_API_BINDINGS", rows),
        "advanced_binding_rows" => count(r -> r["table"] == "ADVANCED_ONLY_API_BINDINGS", rows),
        "simple_binding_names" => length(simple_bindings), "advanced_binding_names" => length(advanced_bindings),
        "root_exports" => length(root_exports), "advanced_exports" => length(advanced_exports),
        "qualified_curated_owner_bindings" => count(r -> r["scope"] == "owner", values(records)),
        "canonical_objects" => length(objects), "inventoried_bindings" => length(records),
        "public_bindings" => sum(length(o["bindings"]) for o in objects if o["intended_public"]),
        "intended_public_objects" => count(o -> o["intended_public"], objects),
        "reconciliation_only_objects" => count(o -> !o["intended_public"], objects),
        "canonical_objects_with_docstrings" => count(o -> o["has_docstring"], objects),
        "intended_public_objects_with_docstrings" => count(o -> o["intended_public"] && o["has_docstring"], objects),
        "owner_candidates" => length(candidates),
    )
    for kind in ("function", "type", "module", "constant", "undefined")
        totals["canonical_" * kind * "s"] = count(o -> o["kind"] == kind, objects)
        for (label, m, syms) in (("root", ROOT_MODULE, root_exports), ("advanced", ADVANCED_MODULE, advanced_exports))
            totals[label * "_exported_" * kind * "s"] = count(s -> isdefined(m, s) ? _kind(getfield(m, s)) == kind : kind == "undefined", syms)
        end
    end
    project = TOML.parsefile(joinpath(ROOT, "Project.toml"))
    extensions = sort!(String[name for name in keys(get(project, "extensions", Dict()))
                              if Base.get_extension(ROOT_MODULE, Symbol(name)) !== nothing])
    return Dict(
        "schema_version" => SCHEMA_VERSION, "generator_version" => GENERATOR_VERSION,
        "generator" => "docs/build_scripts/api_inventory.jl",
        "package_version" => project["version"], "julia_version" => string(VERSION),
        "loaded_optional_extensions" => extensions,
        "docstring_policy" => "Docs.hasdoc, including alias and method documentation; presence is not semantic coverage",
        "candidate_policy" => "Review candidates only; nonunderscore names are not automatically public. Unrelated-owner/Base imports are omitted; aliases imported from an owner's child modules are retained.",
        "source" => _source_metadata(), "totals" => totals, "reconciliation" => rec,
        "objects" => objects, "curated_binding_rows" => rows, "owner_candidates" => candidates,
    )
end

function render_inventory(inventory)
    io = IOBuffer()
    println(io, "# Generated by docs/build_scripts/api_inventory.jl; do not edit by hand.")
    println(io, "# Coverage and editorial decisions belong in docs/api_coverage.toml.")
    TOML.print(io, inventory; sorted=true)
    return String(take!(io))
end

function _summary(inventory)
    t = inventory["totals"]
    r = inventory["reconciliation"]
    println("API declarations: simple=", t["simple_declarations"], ", advanced=", t["advanced_declarations"])
    println("Runtime exports (module self excluded): root=", t["root_exports"], ", Advanced=", t["advanced_exports"])
    println("Canonical objects=", t["canonical_objects"], " (intended public=", t["intended_public_objects"],
            "); curated owner bindings=", t["qualified_curated_owner_bindings"],
            "; separate owner candidates=", t["owner_candidates"])
    println("Advanced binding names absent declaration=", length(r["advanced_binding_names_absent_declaration"]),
            "; declared but not exported=", length(r["advanced_declared_not_exported"]),
            "; table binding names not exported=", length(r["advanced_binding_names_not_exported"]),
            "; undefined declared=", length(r["advanced_declared_undefined"]),
            "; owner/target mismatches=", length(r["binding_owner_target_mismatches"]))
end

function main(args=ARGS)
    if "--help" in args || "-h" in args
        println("Usage: julia --project=. docs/build_scripts/api_inventory.jl [--check] [OUTPUT]")
        return 0
    end
    check = "--check" in args
    paths = filter(!=("--check"), args)
    length(paths) <= 1 || error("Expected at most one output path; use --help")
    any(p -> startswith(p, "--"), paths) && error("Unknown option; use --help")
    output = isempty(paths) ? DEFAULT_OUTPUT : abspath(only(paths))
    inventory = build_inventory()
    rendered = render_inventory(inventory)
    _summary(inventory)
    if check
        if !isfile(output) || read(output, String) != rendered
            println(stderr, "API inventory is missing or stale: ", output)
            println(stderr, "Regenerate with docs/build_scripts/api_inventory.jl ", output)
            return 1
        end
        println("API inventory is current: ", output)
    else
        mkpath(dirname(output))
        write(output, rendered)
        println("Wrote API inventory: ", output)
    end
    return 0
end

end # module APIInventory

if abspath(PROGRAM_FILE) == @__FILE__
    exit(APIInventory.main())
end
