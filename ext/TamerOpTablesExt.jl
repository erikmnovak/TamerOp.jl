module TamerOpTablesExt

using Tables

const PM = let pm = nothing
    if isdefined(Main, :PosetModules)
        pm = getfield(Main, :PosetModules)
    else
        @eval import PosetModules
        pm = PosetModules
    end
    pm
end

const WF = PM.Workflow
const Inv = PM.Invariants

@inline function _named_cols(names::Vector{Symbol}, cols::Vector)
    return NamedTuple{Tuple(names)}(Tuple(cols))
end

function _feature_wide_columntable(fs::WF.FeatureSet)
    ns, nf = size(fs.X)
    names = Vector{Symbol}(undef, nf + 1)
    cols = Vector{Any}(undef, nf + 1)
    names[1] = :id
    cols[1] = copy(fs.ids)
    @inbounds for j in 1:nf
        names[j + 1] = fs.names[j]
        cols[j + 1] = collect(view(fs.X, :, j))
    end
    return _named_cols(names, cols)
end

function _feature_long_columntable(fs::WF.FeatureSet)
    ns, nf = size(fs.X)
    n = ns * nf
    sample_index = Vector{Int}(undef, n)
    id = Vector{String}(undef, n)
    feat = Vector{Symbol}(undef, n)
    value = Vector{eltype(fs.X)}(undef, n)
    k = 1
    @inbounds for i in 1:ns
        ids = fs.ids[i]
        for j in 1:nf
            sample_index[k] = i
            id[k] = ids
            feat[k] = fs.names[j]
            value[k] = fs.X[i, j]
            k += 1
        end
    end
    return (sample_index=sample_index, id=id, feature=feat, value=value)
end

function _euler_surface_long_columntable(t::WF.EulerSurfaceLongTable)
    nx = length(t.x)
    ny = length(t.y)
    n = nx * ny
    id = Vector{String}(undef, n)
    x = Vector{Float64}(undef, n)
    y = Vector{Float64}(undef, n)
    value = Vector{eltype(t.values)}(undef, n)
    k = 1
    @inbounds for i in 1:nx
        xi = t.x[i]
        for j in 1:ny
            id[k] = t.id
            x[k] = xi
            y[k] = t.y[j]
            value[k] = t.values[i, j]
            k += 1
        end
    end
    return (id=id, x=x, y=y, value=value)
end

function _persistence_image_long_columntable(t::WF.PersistenceImageLongTable)
    PI = t.image
    nx = length(PI.xgrid)
    ny = length(PI.ygrid)
    n = nx * ny
    id = Vector{String}(undef, n)
    x = Vector{Float64}(undef, n)
    y = Vector{Float64}(undef, n)
    value = Vector{Float64}(undef, n)
    k = 1
    @inbounds for ix in 1:nx
        xv = PI.xgrid[ix]
        for iy in 1:ny
            id[k] = t.id
            x[k] = xv
            y[k] = PI.ygrid[iy]
            value[k] = PI.values[iy, ix]
            k += 1
        end
    end
    return (id=id, x=x, y=y, value=value)
end

function _mp_landscape_long_columntable(t::WF.MPLandscapeLongTable{D,O}) where {D,O}
    L = t.landscape
    nd, no, kmax, nt = size(L.values)
    n = nd * no * kmax * nt
    id = Vector{String}(undef, n)
    dir_index = Vector{Int}(undef, n)
    offset_index = Vector{Int}(undef, n)
    direction = Vector{D}(undef, n)
    offset = Vector{O}(undef, n)
    layer = Vector{Int}(undef, n)
    tgrid = Vector{Float64}(undef, n)
    value = Vector{Float64}(undef, n)
    weight = Vector{Float64}(undef, n)
    p = 1
    @inbounds for i in 1:nd, j in 1:no, k in 1:kmax, l in 1:nt
        id[p] = t.id
        dir_index[p] = i
        offset_index[p] = j
        direction[p] = L.directions[i]
        offset[p] = L.offsets[j]
        layer[p] = k
        tgrid[p] = L.tgrid[l]
        value[p] = L.values[i, j, k, l]
        weight[p] = L.weights[i, j]
        p += 1
    end
    return (id=id, dir_index=dir_index, offset_index=offset_index,
            direction=direction, offset=offset, layer=layer,
            t=tgrid, value=value, weight=weight)
end

function _point_signed_measure_long_columntable(t::WF.PointSignedMeasureLongTable{N,T,W}) where {N,T,W}
    pm = t.measure
    n = length(pm)
    id = fill(t.id, n)
    inds = Vector{NTuple{N,Int}}(undef, n)
    coords = Vector{NTuple{N,T}}(undef, n)
    wts = Vector{W}(undef, n)
    @inbounds for i in 1:n
        idx = pm.inds[i]
        inds[i] = idx
        coords[i] = ntuple(d -> pm.axes[d][idx[d]], N)
        wts[i] = pm.wts[i]
    end
    return (id=id, indices=inds, coords=coords, weight=wts)
end

# ---------------------------------------------------------------------------
# FeatureSet tables
# ---------------------------------------------------------------------------

Tables.istable(::Type{<:WF.FeatureSet}) = true
Tables.columnaccess(::Type{<:WF.FeatureSet}) = true
Tables.columns(fs::WF.FeatureSet) = _feature_wide_columntable(fs)
Tables.schema(fs::WF.FeatureSet) = Tables.schema(Tables.columns(fs))

Tables.istable(::Type{<:WF.FeatureSetWideTable}) = true
Tables.columnaccess(::Type{<:WF.FeatureSetWideTable}) = true
Tables.columns(t::WF.FeatureSetWideTable) = _feature_wide_columntable(t.fs)
Tables.schema(t::WF.FeatureSetWideTable) = Tables.schema(Tables.columns(t))

Tables.istable(::Type{<:WF.FeatureSetLongTable}) = true
Tables.columnaccess(::Type{<:WF.FeatureSetLongTable}) = true
Tables.columns(t::WF.FeatureSetLongTable) = _feature_long_columntable(t.fs)
Tables.schema(t::WF.FeatureSetLongTable) = Tables.schema(Tables.columns(t))

# ---------------------------------------------------------------------------
# Optional long-form invariant tables
# ---------------------------------------------------------------------------

Tables.istable(::Type{<:WF.EulerSurfaceLongTable}) = true
Tables.columnaccess(::Type{<:WF.EulerSurfaceLongTable}) = true
Tables.columns(t::WF.EulerSurfaceLongTable) = _euler_surface_long_columntable(t)
Tables.schema(t::WF.EulerSurfaceLongTable) = Tables.schema(Tables.columns(t))

Tables.istable(::Type{<:WF.PersistenceImageLongTable}) = true
Tables.columnaccess(::Type{<:WF.PersistenceImageLongTable}) = true
Tables.columns(t::WF.PersistenceImageLongTable) = _persistence_image_long_columntable(t)
Tables.schema(t::WF.PersistenceImageLongTable) = Tables.schema(Tables.columns(t))

Tables.istable(::Type{<:WF.MPLandscapeLongTable}) = true
Tables.columnaccess(::Type{<:WF.MPLandscapeLongTable}) = true
Tables.columns(t::WF.MPLandscapeLongTable) = _mp_landscape_long_columntable(t)
Tables.schema(t::WF.MPLandscapeLongTable) = Tables.schema(Tables.columns(t))

Tables.istable(::Type{<:WF.PointSignedMeasureLongTable}) = true
Tables.columnaccess(::Type{<:WF.PointSignedMeasureLongTable}) = true
Tables.columns(t::WF.PointSignedMeasureLongTable) = _point_signed_measure_long_columntable(t)
Tables.schema(t::WF.PointSignedMeasureLongTable) = Tables.schema(Tables.columns(t))

# Direct convenience: table view defaults for core invariant objects.
Tables.istable(::Type{<:Inv.PersistenceImage1D}) = true
Tables.columnaccess(::Type{<:Inv.PersistenceImage1D}) = true
Tables.columns(pi::Inv.PersistenceImage1D) = Tables.columns(WF.persistence_image_table(pi))
Tables.schema(pi::Inv.PersistenceImage1D) = Tables.schema(Tables.columns(pi))

Tables.istable(::Type{<:Inv.MPLandscape}) = true
Tables.columnaccess(::Type{<:Inv.MPLandscape}) = true
Tables.columns(L::Inv.MPLandscape) = Tables.columns(WF.mp_landscape_table(L))
Tables.schema(L::Inv.MPLandscape) = Tables.schema(Tables.columns(L))

Tables.istable(::Type{<:Inv.PointSignedMeasure}) = true
Tables.columnaccess(::Type{<:Inv.PointSignedMeasure}) = true
Tables.columns(pm::Inv.PointSignedMeasure) = Tables.columns(WF.point_signed_measure_table(pm))
Tables.schema(pm::Inv.PointSignedMeasure) = Tables.schema(Tables.columns(pm))

end # module
