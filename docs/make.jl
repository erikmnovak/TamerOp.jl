using Documenter
using TOML

# Notebook execution/conversion precedes this step. Formatting never reruns cells.
source = joinpath(@__DIR__, ".build", "src")
publication = TOML.parsefile(joinpath(@__DIR__, "publication.toml"))
all(isfile(joinpath(source, lesson["page"])) for lesson in publication["notebooks"]) ||
    error("Run python docs/build_scripts/publish.py first to capture every lesson; see docs/README.md.")

makedocs(;
    root=@__DIR__, source, build="build", sitename="TamerOp.jl",
    remotes=nothing,
    format=Documenter.HTML(; prettyurls=false, edit_link="main",
        repolink="https://github.com/erikmnovak/TamerOp.jl",
        assets=["assets/tutorials.css", "assets/home.css", "assets/reading_map.js"]),
    # The inventory and publication routes generate this catalog. Sidebar groups
    # and the separate article outline are rendered by site_shell.py below.
    pages=[entry["title"] => entry["source"] for entry in
           TOML.parsefile(joinpath(source, "catalog_pages.toml"))["pages"]],
)

# The current site is unversioned, both locally and on GitHub Pages.
write(joinpath(@__DIR__, "build", "siteinfo.js"),
    "var DOCUMENTER_VERSION_SELECTOR_DISABLED = true;\n")
write(joinpath(@__DIR__, "build", "versions.js"),
    "// This site has no published version registry.\n")
# Documenter expects versions.js one level above a versioned site. Keep this
# self-contained site's reference inside its build folder instead.
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

run(`$python $(joinpath(@__DIR__, "build_scripts", "site_shell.py"))`)
