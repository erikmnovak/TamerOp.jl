using Documenter
using TOML

reading_map = TOML.parsefile(joinpath(@__DIR__, "reading_map.toml"))
lesson_titles = Dict(node["id"] => node["title"] for node in reading_map["nodes"])

# Notebook execution/conversion precedes this step. Formatting never reruns cells.
source = joinpath(@__DIR__, ".build", "src")
isfile(joinpath(source, "tutorials", "ring.md")) ||
    error("Run python docs/build_scripts/publish.py first; see docs/README.md.")

makedocs(;
    root=@__DIR__, source, build="build", sitename="TamerOp.jl",
    remotes=nothing,
    format=Documenter.HTML(; prettyurls=false, edit_link="main",
        repolink="https://github.com/erikmnovak/TamerOp.jl",
        assets=["assets/tutorials.css", "assets/reading_map.js"]),
    pages=[
        "Home" => "index.md",
        "Choose a reading path" => "reading_map.md",
        "Start here" => [lesson_titles["install"] => "start/install.md",
                         lesson_titles["ring"] => "tutorials/ring.md",
                         lesson_titles["bridge"] => "explanations/two_parameters.md"],
        "The finite-encoding story" => [lesson_titles["modules"] => "persistence_modules.md",
            lesson_titles["encoding"] => "finite_encodings.md",
            lesson_titles["indicators"] => "indicator_presentations.md",
            lesson_titles["tameness"] => "tameness.md",
            lesson_titles["practical-tameness"] => "practical_tameness.md"],
        "Supporting guides" => [lesson_titles["ordinary"] => "ordinary_persistence.md"],
        "Implementation reference" => ["About these accounts" => "implementation/index.md",
            "Exact rational coordinates" => "implementation/qq_coordinates.md",
            "Bibliography" => "implementation/references.md"],
        "Benchmark results" => ["Overview" => "benchmarks/index.md",
            "Finite algebra: QPA" => "benchmarks/qpa.md",
            "Ordinary persistence: PHAT" => "benchmarks/phat.md",
            "QPA data dictionary" => "benchmarks/qpa_v1/README.md",
            "PHAT data dictionary" => "benchmarks/phat_v2/README.md"],
        "Contributing" => ["Writing and teaching" => "contributing/writing.md"],
    ],
)

# This unversioned preview has no deployment-generated version registry.
write(joinpath(@__DIR__, "build", "siteinfo.js"),
    "var DOCUMENTER_VERSION_SELECTOR_DISABLED = true;\n")
write(joinpath(@__DIR__, "build", "versions.js"),
    "// No published versions in this local preview.\n")
# Documenter expects versions.js one level above a versioned site. Keep this
# self-contained preview's reference inside its build folder instead.
for (dir, _, files) in walkdir(joinpath(@__DIR__, "build")), file in files
    endswith(file, ".html") || continue
    path = joinpath(dir, file)
    old = replace(relpath(joinpath(@__DIR__, "versions.js"), dir), '\\' => '/')
    new = replace(relpath(joinpath(@__DIR__, "build", "versions.js"), dir), '\\' => '/')
    write(path, replace(read(path, String), "src=\"$old\"" => "src=\"$new\""))
end

# Reading routes branch: sidebar order must not impose a different next lesson.
# Use the publication interpreter when called from publish.py (including venvs).
python = get(ENV, "TAMEROP_DOCS_PYTHON", "python")
run(`$python $(joinpath(@__DIR__, "build_scripts", "navigation.py"))`)
