#!/usr/bin/env julia
# Validate documentation intent against the checked-in inventory; never load TamerOp.
# Usage: julia --startup-file=no --project=. docs/build_scripts/check_api_coverage.jl [coverage.toml]
using TOML

const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

require(condition, message) = condition || error(message)
function unique_index(rows, key, label)
    index = Dict{String,Any}()
    for row in rows
        value = row[key]
        require(value isa String && !isempty(value), "$label needs a nonempty $key")
        require(!haskey(index, value), "Duplicate $label: $value")
        index[value] = row
    end
    return index
end

function strings(row, key)
    values = row[key]
    require(values isa Vector && all(x -> x isa String && !isempty(x), values),
            "$key must be an array of nonempty strings")
    require(length(unique(values)) == length(values), "Duplicate entries in $key")
    return values
end

function repository_path(path; must_exist=true)
    require(path isa String && !isempty(path) && !isabspath(path), "Expected repository-relative path: $path")
    normalized = normpath(path)
    require(first(splitpath(normalized)) != "..", "Path leaves repository: $path")
    resolved = joinpath(PROJECT_ROOT, normalized)
    require(!must_exist || isfile(resolved), "Missing documentation/evidence file: $path")
    return resolved
end

function check_pages(row; family=false)
    for key in ("reference_status", "review_status")
        require(row[key] isa String && !isempty(row[key]), "Missing $key")
    end
    for path in strings(row, "reference_pages")
        repository_path(path; must_exist=row["reference_status"] != "backlog")
    end
    for key in (family ? ("guides", "example_sources") : ("example_sources",))
        foreach(repository_path, strings(row, key))
    end
    family && require(row["example_status"] isa String && !isempty(row["example_status"]), "Missing example_status")
end

function main(args)
    require(length(args) <= 1, "Usage: check_api_coverage.jl [coverage.toml]")
    coverage_path = isempty(args) ? joinpath(PROJECT_ROOT, "docs", "api_coverage.toml") : abspath(only(args))
    coverage = TOML.parsefile(coverage_path)
    require(coverage["schema_version"] == 1, "Unsupported coverage schema_version")
    require(coverage["policy"]["routing"] == "override_then_canonical_owner_then_declaration_owner", "Unsupported routing policy")
    inventory = TOML.parsefile(repository_path(coverage["inventory"]))
    require(inventory["schema_version"] == 1, "Unsupported inventory schema_version")
    families = unique_index(coverage["families"], "id", "family")
    require(!isempty(families), "No documentation families declared")
    owners = Dict{String,String}()
    for (id, family) in families
        require(family["title"] isa String && !isempty(family["title"]), "Family $id has no title")
        check_pages(family; family=true)
        for owner in strings(family, "owners")
            require(!haskey(owners, owner), "Owner assigned more than once: $owner")
            owners[owner] = id
        end
    end

    objects = unique_index(inventory["objects"], "canonical_binding", "inventory object")
    candidates = unique_index(inventory["owner_candidates"], "qualified_name", "owner candidate")
    aliases = Dict{String,String}()
    for (id, object) in objects
        for alias in strings(object, "aliases")
            require(!haskey(aliases, alias), "Alias belongs to multiple objects: $alias")
            aliases[alias] = id
        end
        require(get(aliases, id, nothing) == id, "Canonical binding absent from aliases: $id")
    end
    family_exists(id) = require(haskey(families, id), "Unknown documentation family: $id")
    public_object(id) = get(objects[id], "intended_public", false)
    overrides = Dict{String,String}()
    override_bindings = Set{String}()
    for row in get(coverage, "overrides", [])
        family_exists(row["family"])
        for binding in strings(row, "bindings")
            require(!(binding in override_bindings), "Duplicate override binding: $binding")
            push!(override_bindings, binding)
            require(haskey(aliases, binding), "Override binding not in inventory aliases: $binding")
            id = aliases[binding]
            require(public_object(id), "Override selects reconciliation-only object: $binding")
            require(get(overrides, id, row["family"]) == row["family"], "Conflicting alias overrides for $id")
            overrides[id] = row["family"]
        end
    end

    assigned = Dict{String,String}()
    for (id, object) in objects
        public_object(id) || continue
        family = get(overrides, id, get(owners, object["canonical_owner"], nothing))
        if family === nothing
            declarations = object["declaration_owners"]
            choices = unique([get(owners, owner, nothing) for owner in declarations])
            require(length(choices) == 1 && only(choices) !== nothing,
                    "Missing/ambiguous family for $id; add an override (declaration owners: $declarations)")
            family = only(choices)
        end
        assigned[id] = family
    end

    selected = unique_index(get(coverage, "qualified_apis", []), "binding", "qualified selection")
    selected_ids = Dict{String,String}()
    extra = Dict{String,String}()
    for (binding, row) in selected
        family_exists(row["family"])
        for key in ("reason", "review_status")
            require(row[key] isa String && !isempty(row[key]), "Qualified selection $binding needs $key")
        end
        evidence = strings(row, "evidence")
        require(!isempty(evidence), "Qualified selection $binding needs evidence")
        foreach(repository_path, evidence)
        id = get(aliases, binding, nothing)
        if id === nothing
            require(haskey(candidates, binding), "Unresolved qualified selection: $binding")
            candidate = candidates[binding]
            canonical = get(candidate, "canonical_binding", "")
            id = isempty(canonical) ? get(candidate, "candidate_identity", binding) : canonical
            require(isempty(canonical) || haskey(objects, id), "Candidate identity missing from inventory: $binding")
        end
        if haskey(objects, id)
            require(public_object(id), "Qualified selection points to reconciliation-only object: $binding")
            require(assigned[id] == row["family"], "Qualified selection conflicts with routed family: $binding")
        else
            require(get(extra, id, row["family"]) == row["family"], "Conflicting qualified aliases: $binding")
            extra[id] = row["family"]
        end
        selected_ids[binding] = id
    end

    first_path = coverage["first_path"]
    check_pages(first_path)
    haskey(first_path, "plan") && repository_path(first_path["plan"])
    first_bindings = strings(first_path, "bindings")
    first_ids = Set{String}()
    for binding in first_bindings
        id = get(selected_ids, binding, get(aliases, binding, nothing))
        require(id !== nothing && (haskey(assigned, id) || haskey(extra, id)), "Unresolved first-path binding: $binding")
        push!(first_ids, id)
    end

    println("Documentation assignments (not authored reference coverage):")
    for family in coverage["families"]
        id = family["id"]
        println("  ", id, ": ", count(==(id), values(assigned)), " inventory objects + ",
                count(==(id), values(extra)), " additional qualified objects; reference=", family["reference_status"])
    end
    println("Assigned intended objects: ", length(assigned), "; reconciliation-only excluded: ", length(objects) - length(assigned))
    println("Qualified selections: ", length(selected), " bindings, ", length(unique(values(selected_ids))),
            " identities; ", length(extra), " additional identities outside curated inventory")
    println("First path: ", length(first_bindings), " bindings / ", length(first_ids), " identities; reference=", first_path["reference_status"])
    println("Reference family statuses: ", join([string(status, "=", count(f -> f["reference_status"] == status, values(families)))
        for status in sort!(unique([f["reference_status"] for f in values(families)]))], ", "))
    println("Unselected owner review candidates: ", count(binding -> !haskey(selected, binding), keys(candidates)),
            " (not automatically public or reference-complete)")
    for tier in ("simple", "advanced", "qualified")
        println("  Inventory tier ", tier, ": ", count(id -> tier in objects[id]["tiers"], keys(assigned)), " objects (tiers overlap)")
    end
    for (key, value) in sort!(collect(inventory["reconciliation"]); by=first)
        println("  Reconciliation ", key, ": ", value isa Vector ? length(value) : value)
    end
    println("API coverage manifest validation passed. Assignment and docstring presence do not establish authored coverage.")
end

try
    main(ARGS)
catch err
    println(stderr, "API coverage validation failed: ", sprint(showerror, err))
    exit(1)
end
