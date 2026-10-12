# Aligned ensembles and statistical uncertainty computations

A repeated persistence computation produces several mathematical objects, not
several interchangeable vectors. Before averaging anything, we must choose an
observable and identify its arguments across the repetitions. For an encoded
module, that might be the dimension at each original parameter in a fixed grid.
The finite labels used internally by two encodings need not agree. For ordinary
persistence, it might be one landscape layer evaluated at fixed parameters.

This account follows those aligned observations into descriptive summaries and
population-mean bands. The [ensemble guide](../tutorials/uncertainty.ipynb) develops
the user workflow; the [reference](../src/reference/ensembles.md) specifies the
accepted inputs and retained contracts.

## One row is one observation

`persistence_ensemble` retains an owned array with one row per replicate and one
column per probe. Multidimensional probes use Julia's Cartesian ordering: the
first coordinate changes fastest. Coordinates and supplied values retain their
original types. Statistical reductions and drawing use `Float64`, so the
constructor rejects infinite values and coordinates that merge after conversion.
`missing` means an unavailable measurement, not zero.

The context records the observable, field, degree, selected layer, units,
normalization, essential policy, sampling protocol and population. A supplied
array declares that these choices agree across rows. The constructor cannot
establish this from numerical values alone. The scalar-field adapter checks
axes, axis labels, units and ordered layer identities. The ordinary-diagram
adapter checks fields and filtration orders, then evaluates interval tents or
active-interval counts on the requested physical grid. It retains the diagrams
for selected-replicate inspection. The essential policy is explicit; an
ordinary shortcut does not claim to have constructed a finite encoding.

A three-column example makes the representation consequential. If two rows
are `(0, 1, 4)` and `(2, 3, 8)`, their means are `(1, 2, 6)`. Reordering the
second row without changing the declared probes would change the mathematical
experiment. No sorting, interpolation, normalization or bar matching is used
to conceal such a disagreement.

## Descriptive reductions keep their denominators

`summarize_ensemble` works down columns. It computes the mean, median, sample
standard deviation, a chosen empirical interval and the fraction strictly above
a declared threshold. The default interval uses linear empirical quantiles;
`:range` uses the minimum and maximum. At each probe the available observation
count is retained. An empty column gives `NaN` summaries, and fewer than two
observations give `NaN` sample standard deviation. These values become masked
cells or gaps in curves.

Available-case summaries describe the observed values at each probe. If
missingness depends on the outcome, they need not estimate the same population
quantity at every probe. Inference therefore rejects missing observations
rather than silently changing denominators or imputing zeros.

## Conditional confidence from known bounds

For IID observations $X_{ij}$ at fixed probes $j=1,\ldots,p$, suppose known
population bounds satisfy $a_j\leq X_{ij}\leq b_j$ almost surely. With $n$ rows,
Hoeffding's inequality gives a two-sided mean radius

$$
r_j=(b_j-a_j)\sqrt{\frac{\log(2p/\alpha)}{2n}}.
$$

The union bound over probes makes the resulting intervals simultaneous with
coverage at least $1-\alpha$, conditional on the hypotheses. Coordinate
independence is unnecessary: the independence assumption concerns rows. With
`scope=:pointwise`, replace $p$ by 1, obtaining a separate statement at each
probe. The mean interval is intersected with $[a_j,b_j]$.

TamerOp checks that supplied observations respect the bounds. It cannot verify
that unseen observations will do so; the bounds must follow from the population
model or construction. Observed extrema are not an automatic substitute.
For a first landscape of an interval contained in $[0,6]$, the construction
gives the bound $[0,3]$ before any sample is seen.

The formula is a direct application of [Hoeffding's bounded-sum inequality](https://www.cs.rpi.edu/academics/courses/spring06/random/hoefding.pdf)
and a finite union bound. It does not identify the full random module, bound
all possible queries, or supply a geometric approximation certificate.

## Bootstrap whole rows

The bootstrap uses a private `Xoshiro` generator with a recorded seed. Each
resample draws $n$ row indices with replacement and computes their mean vector.
For pointwise bands, it retains the absolute mean deviation at each probe. For
simultaneous bands, it retains the maximum absolute deviation across probes.
The selected empirical order statistic, with rank $\lceil B\,\mathrm{level}\rceil$
among $B$ resamples, supplies the symmetric half-width around the original mean.
Resampling individual cells would destroy within-observation dependence, so
that route is not offered.

These are nonstudentized, finite-grid bootstrap mean bands. Their coverage is
approximate, and a small or degenerate sample can produce uninformative or
misleading widths. The contract rejects descriptive-only sampling declarations,
missing values and fewer than two rows; those checks do not certify the IID
assumption. No subject-cluster bootstrap or data-dependent probe selection is
inferred from the data.

[Confidence Bands for Multiparameter Persistence Landscapes](https://arxiv.org/abs/2504.01113)
develops a functional statistical theory for a specified landscape and sampling
model. That theory motivates the distinction between variability and mean
inference. This implementation uses a finite vector of declared observables and
does not invoke the paper's continuous-domain theorem for arbitrary slice stacks.

## Drawing and selection consume prepared results

The numerical owner is `Featurizers`; the visualization owner consumes the
prepared summary. One-dimensional figures show probes, empirical or inferential
bands, a selected replicate and its retained ordinary diagram. Connecting
segments and shading are drawing guides. Two-dimensional fields preserve
nonuniform physical coordinates and mask missing values. Shared mean/replicate
scales describe the same units; frequency has its own $[0,1]$ scale.

A selection retains the original value and coordinates alongside numerical
summary readouts. Optional source figures follow replicate IDs and are supplied
associations, not a newly computed correspondence certificate. Live selection
prepares a candidate snapshot before committing it. Invalid input leaves the
previous snapshot intact. Linked clients receive the committed state, hover is
read-only, and closing releases input/listener references while preserving the
last static snapshot. Selection never changes a bootstrap seed or reruns the
statistics. [Inspection sessions](inspection_sessions.md) explains the common
transport and lifecycle mechanics.

## Costs and domain boundaries

For $n$ observations and $p$ probes, input storage is $O(np)$; retained summary
arrays use $O(p)$. Column sorting for empirical quantiles costs
$O(pn\log n)$. A bootstrap with $B$ resamples uses $O(Bnp)$ arithmetic and
$O(Bp)$ retained deviations for pointwise bands, or $O(B)$ for simultaneous
bands. Construction, interval evaluation, bootstrap work and display have
explicit rejecting budgets. There is no hidden all-pairs distance computation.

The guarantee domain is the finite set of supplied probes. Extending a band
between nodes requires additional regularity and a separate discretization
argument. A discontinuous Betti field cannot inherit a landscape's Lipschitz
bound. Rendering a continuous strip or a colored cell changes neither fact.
Optimization may change reduction kernels and storage, but must preserve row
resampling, probe order, missing-data rules, quantile conventions and the
reported statistical target.

Statistical arithmetic is numerical. Known bounds are converted outward to
Float64, with their original declarations retained; a small outward endpoint
guard prevents a positive interval from disappearing through display rounding.
This is not an exact-arithmetic certificate for the floating reductions.

[Bibliography](references.md)
