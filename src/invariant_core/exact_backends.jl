# Private invariant-backend hook declarations shared across owners.
#
# Owners such as `DataIngestion` and `Invariants` extend these hooks with
# owner-native exact backends. Workflow-facing callers dispatch through these
# hooks rather than reaching into a specific owner module.

@inline _supports_exact_slice_barcodes(args...; kwargs...) = false
@inline _exact_slice_barcodes(args...; kwargs...) = nothing

@inline _supports_exact_euler_signed_measure(args...; kwargs...) = false
@inline _exact_euler_signed_measure(args...; kwargs...) = nothing

@inline _supports_exact_restricted_hilbert(args...; kwargs...) = false
@inline _exact_restricted_hilbert(args...; kwargs...) = nothing

@inline _supports_exact_rectangle_signed_barcode(args...; kwargs...) = false
@inline _exact_rectangle_signed_barcode(args...; kwargs...) = nothing

@inline _supports_exact_rank_signed_measure(args...; kwargs...) = false
@inline _exact_rank_signed_measure(args...; kwargs...) = nothing

@inline _supports_exact_rank_query_table(args...; kwargs...) = false
@inline _exact_rank_query_table(args...; kwargs...) = nothing
