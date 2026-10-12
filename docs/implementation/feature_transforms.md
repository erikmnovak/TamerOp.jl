# From sampled landscapes to ambient queries

A landscape stored by slice index answers a question about a line. An ambient
plot asks for a value at an original parameter. The missing step is a declared
map between those two coordinate systems. This account explains that step and
the numerical information it preserves. The [features guide](../tutorials/features.ipynb)
works through a square-supported finite module before using the same display
contract for other scalar fields.

## Preserve the line parameter before placing a height

`MPLandscape` stores an array indexed by direction, offset, landscape layer and
line parameter. Its weights support aggregation and integration over a sampled
family. A pointwise landscape height does not involve those weights.
`mp_landscape` first restricts the encoded module to chosen lines, obtains their
barcodes and evaluates ordinary landscape tents on the common parameter grid.
It retains the normalized directions used by that computation. In particular,
the workflow normalizes supplied directions in the one-norm: requesting `(1,1)`
stores `(1/2,1/2)`. Treating its parameter as either ambient coordinate would
therefore put both locations and heights on the wrong scale.

The adapter takes the actual basepoint of each stored offset. If a line has
basepoint `b`, positive direction `d`, and stored parameter `t`, its location is

```math
x=b+td.
```

The caller supplies `b` rather than asking the adapter to infer an offset
convention from a scalar or a plotting label. For each requested grid point,
`ambient_landscape` solves for `t` using the largest direction component and
checks the residual in every coordinate. An absolute tolerance governs this
geometric membership check. Coincident or tolerance-overlapping lines are
rejected because their occurrence identities could give competing answers.

The stored height is measured in units of `t`. Multiplication by
`maximum(d)` converts that height to distance along the line in the ambient
infinity norm. For equal direction components this gives the diagonal landscape
described by Vipond: its restrictions to diagonal lines are ordinary landscapes
with the appropriate parameter. Unequal positive components describe a
directional summary, which the result identifies separately. The adapter
does not average directions or identify this summary with the diagonal invariant.
[Vipond's construction](https://www.jmlr.org/papers/v21/19-054.html) supplies the
mathematical interpretation; the finite-grid adapter and its masking contract
are library choices.

## Distinguish sampling, interpolation and unavailable queries

Each query must lie in both the retained evaluation grid and the declared source
window for its line. An out-of-window query or a point without a stored line
gets `NaN` and a false coverage flag. Zero is reserved for an available zero
height. The algorithm performs no extrapolation and no interpolation between
different lines. Finite source windows may themselves reflect a restricted or
censored computation; supplying their endpoints does not undo that restriction.

Within a line, a binary search finds neighboring sample times. Both neighbors
must lie inside the declared source window; crossing that window boundary does
not justify borrowing an unavailable sample. Linear
interpolation supplies the height. If their distances from the query are
`a` and `b`, the one-Lipschitz property of an ordinary landscape gives an
interpolation error at most `2ab/(a+b)`, with zero error at a sample time.
This follows by weighting the two endpoint Lipschitz bounds by their linear
interpolation coefficients. A supplied uniform sample error is added before
conversion to ambient units. Without that supplied bound, the error against the
intended landscape is recorded as unknown. Under a nonzero line-membership
tolerance the bound concerns the projected point on the stored line.

This bound addresses numerical evaluation of the stated line landscape. It
does not establish an error for the original point cloud, a previous grid
approximation, capped essential intervals or statistical inference.
`ambient_coverage` and `ambient_errors` make the distinction inspectable.

For `Q` grid points, `S` stored offsets, `T` parameter samples and `K` layers,
the straightforward query pass costs up to `O(Q(S + log(T) + K))`, with an
additional factor for the ambient dimension in the line-membership checks.
It stores `O(QK)` values and `O(Q)` coverage/error entries. A pairwise line
uniqueness check costs `O(S²)` before evaluation. `max_points` limits the
product of grid points and retained layers before allocation. This path
consumes retained features and does not repeat barcode or rank computations.

## Keep the numerical field separate from its rendering

`ScalarField` turns declared coordinate axes and sampled values into a
visualization input. The ambient-grid overload retains its interpretation and
window metadata, with landscape layers as stable IDs. Other scalar invariants
can use the same adapter after their values have been computed; a plain field
does not claim to be a module or a finite encoding.

The [visual specification account](visual_specs.md#coordinate-sections-and-shared-scales)
explains orientation, masks, shared scales and interpolated contours. Those
contours introduce an explicitly declared display interpolation across neighboring
grid values; the ambient query error bounds do not certify the contour curves. The
[session account](inspection_sessions.md#linked-coordinate-sections) explains
how controls reuse those data. General feature-table assembly and numerical
integration are separate operations, so changing a view cannot silently change
an experiment's feature vector.

## References

See the [implementation bibliography](references.md), particularly Oliver
Vipond's multiparameter landscapes and Peter Bubenik's ordinary landscapes.
