# =============================================================================
# Options.jl
#
# Structured option/specification types used across ingestion, encoding,
# workflow, module, and derived-functor layers.
# =============================================================================
module Options

using ..CoreModules: AbstractCoeffField, QQField

"""
    FiltrationSpec(; kind, params...)

Lightweight filtration specification container. Stores a `kind` symbol and a
`params` named tuple.
"""
struct FiltrationSpec
    kind::Symbol
    params::NamedTuple
end

FiltrationSpec(; kind::Symbol, params...) = FiltrationSpec(kind, NamedTuple(params))

"""
    ConstructionBudget(; max_simplices=nothing, max_edges=nothing, memory_budget_bytes=nothing)

Budget controls for data-ingestion construction.
"""
struct ConstructionBudget
    max_simplices::Union{Nothing,Int}
    max_edges::Union{Nothing,Int}
    memory_budget_bytes::Union{Nothing,Int}
end

ConstructionBudget(; max_simplices::Union{Nothing,Integer}=nothing,
                   max_edges::Union{Nothing,Integer}=nothing,
                   memory_budget_bytes::Union{Nothing,Integer}=nothing) =
    ConstructionBudget(max_simplices === nothing ? nothing : Int(max_simplices),
                       max_edges === nothing ? nothing : Int(max_edges),
                       memory_budget_bytes === nothing ? nothing : Int(memory_budget_bytes))

"""
    ConstructionOptions(; sparsify=:none, collapse=:none, output_stage=:encoding_result,
                         budget=(nothing, nothing, nothing))

Canonical construction controls for data ingestion.
"""
struct ConstructionOptions
    sparsify::Symbol
    collapse::Symbol
    output_stage::Symbol
    budget::ConstructionBudget
end

function ConstructionOptions(; sparsify::Symbol=:none,
                             collapse::Symbol=:none,
                             output_stage::Symbol=:encoding_result,
                             budget=(nothing, nothing, nothing))
    sp = sparsify
    co = collapse
    os = output_stage
    (sp == :none || sp == :radius || sp == :knn || sp == :greedy_perm) ||
        error("ConstructionOptions: sparsify must be :none, :radius, :knn, or :greedy_perm.")
    (co == :none || co == :dominated_edges || co == :acyclic) ||
        error("ConstructionOptions: collapse must be :none, :dominated_edges, or :acyclic.")
    (os == :simplex_tree || os == :graded_complex || os == :cochain || os == :module ||
     os == :fringe || os == :flange || os == :encoding_result) ||
        error("ConstructionOptions: output_stage must be :simplex_tree, :graded_complex, :cochain, :module, :fringe, :flange, or :encoding_result.")
    b = if budget isa ConstructionBudget
        budget
    elseif budget isa Tuple
        length(budget) == 3 || error("ConstructionOptions: budget tuple must be (max_simplices, max_edges, memory_budget_bytes).")
        ConstructionBudget(; max_simplices=budget[1], max_edges=budget[2], memory_budget_bytes=budget[3])
    elseif budget isa NamedTuple
        ConstructionBudget(; budget...)
    else
        error("ConstructionOptions: budget must be ConstructionBudget, 3-tuple, or NamedTuple.")
    end
    return ConstructionOptions(sp, co, os, b)
end

"""
    DataFileOptions(; kind=:auto, format=:auto, header=nothing, delimiter=nothing,
                    comment_prefix='#', missing_policy=:error, cols=nothing,
                    u_col=:u, v_col=:v, weight_col=nothing)

Canonical file-ingestion parsing controls used by `DataFileIO.load_data`.
"""
struct DataFileOptions{H,D,C,ColsT,UColT,VColT,WColT}
    kind::Symbol
    format::Symbol
    header::H
    delimiter::D
    comment_prefix::C
    missing_policy::Symbol
    cols::ColsT
    u_col::UColT
    v_col::VColT
    weight_col::WColT
end

function DataFileOptions(;
    kind::Symbol=:auto,
    format::Symbol=:auto,
    header=nothing,
    delimiter=nothing,
    comment_prefix='#',
    missing_policy::Symbol=:error,
    cols=nothing,
    u_col=:u,
    v_col=:v,
    weight_col=nothing,
)
    (kind == :auto || kind == :point_cloud || kind == :graph || kind == :image || kind == :distance_matrix) ||
        error("DataFileOptions: kind must be :auto, :point_cloud, :graph, :image, or :distance_matrix.")
    (format == :auto || format == :dataset_json || format == :csv || format == :tsv || format == :txt ||
     format == :ripser_point_cloud || format == :ripser_distance || format == :ripser_lower_distance ||
     format == :ripser_upper_distance || format == :ripser_sparse_triplet ||
     format == :ripser_binary_lower_distance || format == :ripser_lower_distance_streaming) ||
        error("DataFileOptions: unsupported format $(format).")
    (missing_policy == :error || missing_policy == :drop_rows) ||
        error("DataFileOptions: missing_policy must be :error or :drop_rows.")
    return DataFileOptions{
        typeof(header),
        typeof(delimiter),
        typeof(comment_prefix),
        typeof(cols),
        typeof(u_col),
        typeof(v_col),
        typeof(weight_col),
    }(kind, format, header, delimiter, comment_prefix, missing_policy, cols, u_col, v_col, weight_col)
end

"""
    PipelineOptions(; orientation=nothing, axes_policy=:encoding, axis_kind=nothing,
                     eps=nothing, poset_kind=:signature, field=nothing, max_axis_len=nothing)

Structured pipeline controls that materially affect data-ingestion encodings and
their reproducibility in serialized pipeline artifacts.
"""
struct PipelineOptions{OrientationT,AxisKindT,EpsT,FieldT}
    orientation::OrientationT
    axes_policy::Symbol
    axis_kind::AxisKindT
    eps::EpsT
    poset_kind::Symbol
    field::FieldT
    max_axis_len::Union{Nothing,Int}
end

PipelineOptions(; orientation=nothing,
                axes_policy::Symbol=:encoding,
                axis_kind=nothing,
                eps=nothing,
                poset_kind::Symbol=:signature,
                field=nothing,
                max_axis_len::Union{Nothing,Int}=nothing) =
    PipelineOptions(orientation, axes_policy, axis_kind, eps, poset_kind, field, max_axis_len)

"""
    EncodingOptions(; backend=:auto, max_regions=nothing, strict_eps=nothing,
                    poset_kind=:signature, field=QQField())

Options controlling finite encodings.
"""
struct EncodingOptions{S,F<:AbstractCoeffField}
    backend::Symbol
    max_regions::Union{Nothing,Int}
    strict_eps::S
    poset_kind::Symbol
    field::F
end

EncodingOptions(; backend::Symbol=:auto,
                max_regions=nothing,
                strict_eps=nothing,
                poset_kind::Symbol=:signature,
                field::AbstractCoeffField=QQField()) =
    EncodingOptions{typeof(strict_eps),typeof(field)}(
        backend,
        max_regions === nothing ? nothing : Int(max_regions),
        strict_eps,
        poset_kind,
        field,
    )

"""
    ResolutionOptions(; maxlen=3, minimal=false, check=true)

Options controlling (co)resolutions.
"""
struct ResolutionOptions
    maxlen::Int
    minimal::Bool
    check::Bool
end

ResolutionOptions(; maxlen::Int=3, minimal::Bool=false, check::Bool=true) =
    ResolutionOptions(maxlen, minimal, check)

@inline function validate_pl_mode(mode::Symbol)::Symbol
    if mode === :fast
        return :fast
    elseif mode === :verified
        return :verified
    end
    throw(ArgumentError("pl_mode must be exactly :fast or :verified (got $(mode))"))
end

"""
    InvariantOptions(; axes=nothing, axes_policy=:encoding, max_axis_len=256,
                     box=nothing, threads=nothing, strict=nothing, pl_mode=:fast)

Options controlling invariant computations.
"""
struct InvariantOptions{A,B}
    axes::A
    axes_policy::Symbol
    max_axis_len::Int
    box::B
    threads::Union{Nothing,Bool}
    strict::Union{Nothing,Bool}
    pl_mode::Symbol
end

InvariantOptions(; axes=nothing,
                 axes_policy::Symbol=:encoding,
                 max_axis_len::Int=256,
                 box=nothing,
                 threads=nothing,
                 strict=nothing,
                 pl_mode::Symbol=:fast) =
    InvariantOptions{typeof(axes),typeof(box)}(
        axes,
        axes_policy,
        max_axis_len,
        box,
        threads,
        strict,
        validate_pl_mode(pl_mode),
    )

InvariantOptions(axes, axes_policy::Symbol, max_axis_len::Int, box, threads, strict) =
    InvariantOptions{typeof(axes),typeof(box)}(
        axes,
        axes_policy,
        max_axis_len,
        box,
        threads,
        strict,
        :fast,
    )

"""
    DerivedFunctorOptions(; maxdeg=3, model=:auto, canon=:auto)

Options controlling derived functor computations.
"""
struct DerivedFunctorOptions
    maxdeg::Int
    model::Symbol
    canon::Symbol
end

DerivedFunctorOptions(; maxdeg::Int=3, model::Symbol=:auto, canon::Symbol=:auto) =
    DerivedFunctorOptions(maxdeg, model, canon)

"""
    FiniteFringeOptions(; check=true, cached=true, store_sparse=false, scalar=1, poset_kind=:regions)

Options for FiniteFringe convenience entrypoints.
"""
struct FiniteFringeOptions{S}
    check::Bool
    cached::Bool
    store_sparse::Bool
    scalar::S
    poset_kind::Symbol
end

FiniteFringeOptions(; check::Bool=true,
                    cached::Bool=true,
                    store_sparse::Bool=false,
                    scalar=1,
                    poset_kind::Symbol=:regions) =
    FiniteFringeOptions{typeof(scalar)}(check, cached, store_sparse, scalar, poset_kind)

"""
    ModuleOptions(; check_sizes=true, cache=nothing)

Options for Modules convenience entrypoints.
"""
struct ModuleOptions{C}
    check_sizes::Bool
    cache::C
end

ModuleOptions(; check_sizes::Bool=true, cache=nothing) =
    ModuleOptions{typeof(cache)}(check_sizes, cache)

end # module Options
