println("load")
module _M
include("../src/CoreModules.jl")
include("../src/FieldLinAlg.jl")
include("../src/ChainComplexes.jl")
using .CoreModules, .FieldLinAlg, .ChainComplexes, SparseArrays, Random
const CM = CoreModules; const FL = FieldLinAlg; const CC = ChainComplexes
function rc(rng, field)
    CM.coerce(field, rand(rng, -5:5))
end
rnc(rng, field) = (x = rc(rng, field); iszero(x) ? one(CM.coeff_type(field)) : x)
function rmat(rng, field, m, n; density=0.35)
    K = CM.coeff_type(field); A = zeros(K,m,n)
    for i in 1:m, j in 1:n
        rand(rng) < density || continue
        A[i,j] = rnc(rng, field)
    end
    A
end
function rdiffs(rng, field, dims; density=0.35)
    K = CM.coeff_type(field); out = Vector{SparseMatrixCSC{K,Int}}(); length(dims) <= 1 && return out
    push!(out, sparse(rmat(rng, field, dims[2], dims[1]; density=density)))
    for i in 2:(length(dims)-1)
        prev = Matrix(out[i-1]); left = FL.nullspace(field, transpose(prev)); r = size(left,2)
        if dims[i+1] == 0 || dims[i] == 0 || r == 0
            push!(out, spzeros(K, dims[i+1], dims[i]))
        else
            coeff = rmat(rng, field, dims[i+1], r; density=density)
            push!(out, sparse(coeff * transpose(left)))
        end
    end
    out
end
function builddc(seed=11)
    rng = MersenneTwister(seed); field = CM.QQField(); K = CM.coeff_type(field)
    dims = [rand(rng, 2:4) for _ in 1:3, __ in 1:3]
    dv = Array{SparseMatrixCSC{K,Int},2}(undef,3,3); dh = Array{SparseMatrixCSC{K,Int},2}(undef,3,3)
    for ai in 1:3
        col_dims = [dims[ai,bi] for bi in 1:3]; col_d = rdiffs(rng, field, col_dims; density=0.35)
        for bi in 1:3
            dv[ai,bi] = bi < 3 ? col_d[bi] : spzeros(K, 0, col_dims[bi])
            dh[ai,bi] = ai < 3 ? spzeros(K, dims[ai+1,bi], dims[ai,bi]) : spzeros(K, 0, dims[ai,bi])
        end
    end
    return CC.DoubleComplex{K}(0,2,0,2,dims,dv,dh)
end
function bench(f)
    f()
    GC.gc(); GC.gc(); local t=0.0; bytes = @allocated begin t=@elapsed f() end
    println("RESULT ms=", t*1000, " kib=", bytes/1024); flush(stdout)
end
probe = ARGS[1]
DC = builddc()
println("probe=", probe); flush(stdout)
if probe == "horizontal_auto"
    bench(() -> begin ss = CC.spectral_sequence(DC; first=:horizontal); sum(size(M,1)+size(M,2)+nnz(M) for M in CC.differential(ss,1)) end)
elseif probe == "horizontal_optimized"
    bench(() -> begin old = CC._spectral_exact_horizontal_filtimg_mode[]; try CC._spectral_exact_horizontal_filtimg_mode[] = :optimized; ss = CC.spectral_sequence(DC; first=:horizontal); sum(size(M,1)+size(M,2)+nnz(M) for M in CC.differential(ss,1)) finally CC._spectral_exact_horizontal_filtimg_mode[] = old end end)
elseif probe == "horizontal_legacy"
    bench(() -> begin old = CC._spectral_exact_horizontal_filtimg_mode[]; try CC._spectral_exact_horizontal_filtimg_mode[] = :legacy; ss = CC.spectral_sequence(DC; first=:horizontal); sum(size(M,1)+size(M,2)+nnz(M) for M in CC.differential(ss,1)) finally CC._spectral_exact_horizontal_filtimg_mode[] = old end end)
elseif probe == "page_terms_inf_vertical"
    bench(() -> begin ss = CC.spectral_sequence(DC; first=:vertical); sum(sq.dimH for sq in CC.page_terms(ss,:inf)) end)
elseif probe == "filtration_basis_mid_vertical"
    bench(() -> begin ss = CC.spectral_sequence(DC; first=:vertical); tmid = ss.Tot.tmin + ((ss.Tot.tmax - ss.Tot.tmin) ÷ 2); B = CC.filtration_basis(ss, ss.pmin, tmid); size(B,1)+size(B,2) end)
else
    error("unknown probe")
end
end
