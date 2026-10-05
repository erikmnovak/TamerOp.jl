# Execute the guide's canonical static examples and save its two public-API views.
# Run from the repository root: julia --project=docs docs/build_scripts/render_spaces_maps.jl
using LinearAlgebra
BLAS.set_num_threads(1)

docs = dirname(@__DIR__)
source = joinpath(docs, "spaces_and_maps.md")
text = read(source, String)
boundary = "\n## Explore nearby selections in a live session\n"
length(findall(boundary, text)) == 1 || error("Expected one optional live section")
static = first(split(text, boundary; limit=2))
blocks = collect(eachmatch(r"```julia\n(.*?)\n```"s, static))
isempty(blocks) && error("No static guide examples found")
for block in blocks
    include_string(Main, block.captures[1], source)
end

# Hand-computable answers in the guide, independent of finite-label numbering.
@assert OA.nvertices(P) == 4 && sort(dims) == [0, 0, 0, 1]
@assert dims[qx] == 1 && qz != 0 && z_dimension == 0
@assert A == reshape(OP.QQ[1], 1, 1)
@assert size(Z) == (0, 1)
@assert OA.locate(classifier, u) == OA.locate(classifier, v)
@assert !unordered.defined && unordered.matrix === nothing
@assert answer.rank == 0 && answer.kernel_dimension == 1
@assert H !== nothing
@assert OA.active_rows(stalk) == [1] && OA.active_columns(stalk) == [1, 2]
@assert OA.presentation_matrix(stalk) == OP.QQ[1 1]
@assert OA.presentation_summary(stalk).dimension == 1
@assert OA.image_basis(stalk) === nothing
@assert size(B) == (1, 1) && B[1, 1] != 0
@assert By * C == R * Bx

output = joinpath(docs, "assets", "guides")
mkpath(output)
OP.save_visual(joinpath(output, "spaces_map.png"), map_view; backend=:cairomakie)
OP.save_visual(joinpath(output, "spaces_presentation.png"), presentation_view;
    backend=:cairomakie)
println("Checked $(length(blocks)) static guide examples and saved two figures to $output")
