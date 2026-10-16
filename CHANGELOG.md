# Changelog

## 0.2.0 (unreleased)

This release broadens the mathematics we can explore with TamerOp and gives the
project a clearer home for each part of that work. The library has gained new
constructions for persistence and finite modules, more ways to recover the
classes and maps behind an answer, and interactive views that let us follow
those calculations. Much of the work also went into making the existing
computations use less memory and avoid repeating algebra they had already done.

The surrounding project has changed considerably. The library, teaching site
and benchmark studies now have separate repositories, with short contribution
guides and links between them. Someone arriving with a mathematical question,
a proposed lesson or a performance improvement should have a clearer place to
begin, without having to reconstruct how the project was assembled.

The [0.1.0 account below](#010) describes the substantial foundation already in
that release. The sections here describe the additions and changes since then.

### A clearer project to use and contribute to

The three repositories have distinct responsibilities:

| Repository | What belongs there |
| --- | --- |
| [TamerOp.jl](https://github.com/erikmnovak/TamerOp.jl) | The installable Julia library, its tests, package metadata and release history. |
| [TamerOp-site](https://github.com/erikmnovak/TamerOp-site) | The lessons, guides, notebooks, figures and build tools for [tamerop.com](https://tamerop.com/). |
| [TamerOp-benchmarks](https://github.com/erikmnovak/TamerOp-benchmarks) | Runnable comparisons, frozen source versions, inputs and recorded measurements. |

Moving the website out of the library keeps its teaching material and build
environment separate from package installation. The benchmark project gives
measurements a home where their inputs and software versions can be preserved
and the computations repeated. The repository READMEs link to one another, and
the library README now concentrates on installation, a first calculation and
where to continue reading.

The website has become a teaching companion, with mathematical introductions,
worked examples, usage guides, API references and benchmark reports. Its
[Using TamerOp overview](https://tamerop.com/collections/using.html) follows inputs
into a finite encoding and then into the questions that encoding makes
accessible. Topic menus expose the individual articles, and prepared interactive
examples let readers vary a parameter or selection while following the
explanation. Published notebooks have executed examples and downloadable copies.
The site builds against the current library source.

There are now public paths for proposing library changes, teaching material and
benchmark studies. Questions and proposals share the library's
[issue tracker](https://github.com/erikmnovak/TamerOp.jl/issues) and
[Discussions](https://github.com/erikmnovak/TamerOp.jl/discussions); pull requests
go to the repository being changed. Erik reviews and integrates contributions,
including their mathematical meaning, API and final presentation. The
[contribution guide](CONTRIBUTING.md) explains the process and invites anyone
interested in becoming a maintainer to get in touch.

Testing instructions now live [beside the library tests](test/README.md), with
separate [browser testing instructions](test/browser/README.md) for changes to
interactive viewers. The [release procedure](RELEASING.md) remains with the
package. These public instructions can be followed from the corresponding
repository checkout.

### More ways to construct a filtered space

A filtration describes when cells enter a space as its parameters change. The
new input routes give us more choices about that space, while retaining the
information needed to interpret its homology.

- Alpha filtrations now accept point clouds in three-dimensional space. Exact
  predicates check the empty circumspheres that determine the cells and their
  entry values. The native construction handles lower affine dimensions and
  cospherical configurations; the optional MiniQhull extension supplies larger
  candidate triangulations that are checked by the exact geometry.
- Sparse multicover constructions provide a further way to study how many
  neighborhoods cover a location, starting from distinct Euclidean sites. They
  retain exact algebraic radii, sparse subdivision–Čech cells, multiplicative
  comparison factors and pointwise inclusion witnesses relating the sparse
  cover to the original multicover.
- Radius–degree Rips filtrations record the number of neighbors actually present
  at each radius. Component queries can then select clusters by diameter,
  neighbor count, size and observed persistence, while retaining the original
  point memberships.
- Weighted graph and distance inputs can specify vertex births, the dimension
  of cliques to include and an inclusive threshold. Landmark selection returns
  the selected points in order, their insertion radii, nearest-landmark
  assignments and covering radius, so the reduction of the point cloud can be
  inspected as part of the result.
- `filtered_complex` keeps cell geometry, original identities, entry values and
  signed boundaries together. Homology classes can be related back to those
  cells, including through retained maps from a filtered collapse. A filtered
  collapse removes paired cells while preserving the relevant homology.

### Ordinary persistence, with the classes behind the intervals

Ordinary persistence now works over prime fields beyond F₂, the field with two
elements. Signs and coefficients are retained over odd primes. When requested,
a result includes a cycle representing a class, a chain that fills it at its
death, or a persistent cocycle, which describes a class through values on cells.
These can be inspected at a chosen parameter and traced to the original source.

Several related workflows have expanded:

- Rips barcodes through homological degree two can be computed directly from
  points or dense and sparse distance data, generating the necessary cofaces
  during reduction instead of constructing all boundary matrices first. This
  route supports prime coefficients, radius restrictions, landmarks and
  persistent cocycles.
- Cubical persistence accepts values on pixels and voxels in any positive
  array dimension, extending the earlier image route. Values assigned to grid
  vertices remain a separate input convention. Both descriptions support
  periodic axes and retained source information through their general
  constructions.
- Connected-component histories record which original cells belong to each
  component and how components merge. A barcode interval can therefore be
  followed back to the component it describes.
- For retained source cycles over prime fields, representative optimization
  minimizes the sum of chosen cell weights within an explicit search budget.
  It returns a chain witnessing that the homology class is unchanged. Taking
  every weight to be one asks for a representative using the fewest cells.
- `PersistenceDiagram` results can be passed directly to bottleneck and
  Wasserstein distances, landscapes, images, silhouettes, entropy and scalar
  summaries, selecting the homological degree to analyze. These functions use
  the result's interval and filtration conventions.
- Ordinary diagrams can be saved and loaded with exact grades, coefficients,
  representatives, source geometry and retained maps. Extended diagrams have
  their own format preserving the phases described below.

The [ordinary persistence guide](https://tamerop.com/ordinary_persistence.html)
connects these results to the filtration and its barcode.

### Relative, extended and changing persistence

A class can be followed through more than a growing sequence of spaces. This
release adds several ways to specify how a space changes and which maps connect
its successive states.

- Relative persistence computes the homology over a prime field of a pair
  `(K, A)` filtered by one parameter, where chains in the chosen subcomplex `A`
  are set to zero. Cells of `A` may enter
  later in the filtration. The result retains the inclusion, quotient and
  connecting maps that relate the pair's homology to that of its two spaces.
  Spatial selection rules can choose the subcomplex from existing source cells.
- Extended persistence follows a lower-star filtration, in which vertex values
  determine when cells enter, by a relative phase.
  This gives ordinary essential classes, which never die during the first
  phase, an endpoint in the extended calculation. Ordinary, relative and
  extended intervals retain their phase labels.
- Zigzag persistence allows the maps in a finite sequence to point in either
  direction. Exact interval decompositions over rational and prime fields can
  retain representatives, including for supplied chain maps and supported
  sequences of simplex additions and deletions.
- Temporal persistence connects scalar filtrations through specified chain
  maps. It retains the time occurrences, spatial scales and directed maps, with
  a zigzag barcode at each stored scale. Input routes include unions of point
  clouds and overlapping windows of delayed observations; spatiotemporal
  landscapes summarize persistence on the resulting finite grid.
- A retained persistence state supports cell insertion, deletion, append and
  changes to filtration order. Explicit continuation of the grades records
  crossing events and interval correspondences along the path. These are the
  data needed to follow a vineyard, a family of persistence diagrams as the
  filtration changes.
- An inspectable reference reduction records the column operations that produce a
  barcode, including the original boundary matrix, its reduced form and the
  change of basis. Pairings, cycles and fillings can be read alongside the
  steps that produce them.

### Further algebra on finite modules

The finite encoding remains the common object behind the algebra. Its ordered
labels carry vector spaces and maps, so we can ask how vectors move, which
parts form submodules, and whether the whole module splits into simpler pieces.

- Vector and subspace propagation follows chosen vectors through the structure
  maps. It reports their images and can construct the submodule they generate,
  together with its inclusion and quotient.
- Exact interval multiplicities count copies of a specified interval module:
  one copy of the field on an order-convex connected support, with identity
  maps within that support. Bounded searches can find the interval summands
  and retain the inclusion and projection maps that split them from the rest.
- Indecomposable decomposition goes further, splitting a module into pieces
  that cannot themselves be written as nontrivial direct sums. Results retain
  actual splitting maps and certificates on the chosen finite poset. There is
  a specialized bipath route, for two directed paths sharing their endpoints,
  and an optional Hecke extension for rational algebra certification.
- Minimal presentations of zeroth homology can be obtained directly from
  graph connectivity. Their generators, relations and Betti data describe
  components and mergers without first building a larger general presentation.
- Selected map ranks and local Betti queries can be answered without always
  constructing the full encoded module or resolution. Betti data count the
  generators appearing at each location and degree of a minimal resolution.
- Pointwise duality replaces each vector space by its linear dual and reverses
  the arrows. It now applies consistently to finite modules, morphisms,
  presentations, complexes and projective or injective resolutions, including
  the corresponding degree reversals.
- Rank-exact resolutions organize a module using exact sequences that preserve
  the ranks of its structure maps additively. Their lower-hook terms are
  supported above one label, optionally with the region above another removed.
  The result includes the terms, differentials, Betti data and signed hook
  decomposition.

The existing Hom, Ext and Tor calculations have gained more economical basis
and coordinate queries and fuller inspection. These derived constructions are
computed in the representation category of the chosen finite poset, as
explained in the [mathematical categories guide](https://tamerop.com/math_categories.html).

### Products and invariants that retain more structure

Dimensions and ordinary ranks answer useful first questions. The additions
here let us ask about multiplication, persistence across a region, or the
organization of a module under a chosen weighting.

- Cohomology algebras retain cup products, the multiplication of cohomology
  classes. Ordered simplicial sources and supplied compatible algebra diagrams
  support restrictions, persistent product images and partition modules, which
  group products by the degrees of their factors. Cup length records how many
  positive-degree factors can have a nonzero product. Along a finite chain,
  the resulting modules can be expressed by intervals.
- Generalized rank measures how much persists coherently across a selected
  connected convex region of a finite poset. Results can retain the limit and
  colimit matrices that establish the answer. Signed summaries reconstruct
  ranks on a declared finite family of intervals.
- Generalized rank invariant landscapes (GRIL) evaluate these ranks on growing
  worm-shaped regions. The implementation computes exact critical widths at
  fixed probes of supported positive two-dimensional grid encodings, with
  boundary zigzag barcodes available for inspection.
- Harder–Narasimhan filtrations organize finite modules by slope, the ratio
  of weighted degree to positive weighted mass. Their successive factors are
  semistable: no nonzero submodule has a larger slope. Exact computations over
  rational and prime fields can retain the actual filtration maps. Skyscraper
  queries follow subspaces generated at an anchor, comparing their dimension
  there with the weighted mass of their images throughout the module. Geometric
  versions use declared region masses, and planar Cartesian models support
  chamber diagrams showing where the filtration changes. Filtered landscapes
  summarize the resulting persistence.
- For Jordan type and multirank, the maps between selected disjoint sets of
  labels form one nilpotent block operator, whose sufficiently high powers
  vanish. The ranks of its powers recover Jordan block multiplicities and give
  multiranks. Globally described parameter models also support translated
  image filtrations, their ranks and landscapes, and erosion comparisons that
  measure how much shifting is needed to compare those rank functions.
- Graphcodes express a module on a two-dimensional grid as rows of barcodes
  together with the actual linear maps between rows. The representation keeps
  coefficients and chosen bases, and supports lossless compression. Connected
  components of the graph identify direct summands.
- Two-parameter counts and birth and death curves describe finite-support
  lattice modules through the kernel and cokernel of a diagonal unit shift.
  Their locations and multiplicities use the declared lattice and its zero
  extension outside the stored grid.
- A supplied morphism between persistence modules now determines an additive
  partial matching of their intervals, with unmatched multiplicities and
  optional coefficient witnesses. This follows the given map, whereas a
  distance computation chooses a matching to minimize a cost.
- For individual finitely presented bivariate interval modules, linear
  persistence contours in nonnegative directions specify how to shift
  parameters. Exact shift dimensions and stable rank curves give the minimum
  number of generators whose submodule contains the shifted image, with
  generating witnesses available.

### Comparing modules and making features from their results

Comparisons now include explicit maps relating shifted modules. A globally
described rectilinear module records its behavior on coordinate regions,
including the region beyond a finite plotting window. Such models support
parameter shifts, checked interleaving maps, bounded searches for those maps
and bounds on interleaving distance. An interleaving uses maps in both directions
whose compositions agree with the prescribed shifts. A search distinguishes a
found interleaving, proved nonexistence and an unresolved bounded search.

The existing exact matching-distance calculation can now return its exact
value and, for a comparison window of positive area, a maximizing line and
its bottleneck assignment. Use `matching_distance_exact_2d(...; witness=true)`
to inspect the intervals responsible for that value.

Epsilon-pruning constructs the associated shifted subquotient and retains the
maps relating it to the original module. For an exploratory description by
intervals, matched diagonal slices can instead produce candidate planar
supports and error measurements on chosen lines. Those sampled errors describe
the candidate on the queried lines; they do not certify an exact splitting or
a global approximation bound.

Other additions connect persistence results to numerical analysis:

- Signed transport compares balanced signed measures, retaining their positive
  and negative contributions. Sliced transport uses selected projection
  directions, while signed convolution smooths the measure on a chosen grid
  to produce numerical features.
- Ambient landscapes place retained slice values back on a Cartesian parameter
  grid. Their values retain coverage information and, where interpolation is
  requested, its applicable error bounds.
- Ensembles align repeated results on common probes and provide means,
  empirical bands and support frequencies. Confidence intervals for bounded
  means and bootstrap bands resampling whole replicates are available when
  the declared sampling assumptions justify population inference.
- Circular coordinates turn selected persistent first cohomology classes on a
  retained source with one filtration parameter into values on a circle. The
  construction checks an integer lift of each cocycle
  before numerical smoothing; several selected classes give coordinates in a
  product of circles. Original vertex identities connect these coordinates
  to the source data.

### Seeing and following the calculation

The visualization system has expanded from plotting results to connecting
several views of the same mathematical calculation. Selecting an interval can
highlight its source cells; moving a slice can change the barcode beside it;
selecting a map can expose the matrix responsible for a rank. Static figures
and live Julia viewers draw on the same prepared mathematical records.

The available views now cover:

- Finite modules, presentations and their parameter regions, with selected
  spaces and structure maps. Algebra figures expose morphisms, Hom basis maps,
  kernels, images, cokernels, exact sequences, chain maps and homotopies.
- Resolutions and derived algebra, including Betti and Bass tables, principal
  supports, selected differentials, Ext and Tor representatives and product
  tables. Spectral-page viewers connect a differential to its kernel and image.
- Linked barcodes and persistence diagrams, with exact endpoints, repeated
  intervals and source representatives. Moving planar slices retains endpoint
  conventions, and supported encodings can certify a whole-line restriction
  independently of the drawing window.
- Filtered cells, connected components, relative pairs and multicover geometry.
  Threshold controls and parameter atlases connect selected states to their
  source cells, maps, cycles and fillings. Volume and coordinate-plane controls
  expose higher-dimensional data through chosen sections.
- Anchored rank sections, signed contributions, generalized rank and GRIL.
  Comparisons on the same queries place fields, degrees or constructions side by side;
  collections and ensemble views connect numerical features to their samples.
- Summand galleries, refinements, interleavings and pruning, with the maps
  underlying a split, transport or comparison available for inspection.
- Graphcodes, birth and death curves, stability filtrations, temporal modules
  and circular coordinates. Linked selections connect these newer invariants
  to their source labels, maps or vertices.
- Persistence-path playback and stepwise matrix reduction, so a changing answer
  or an individual column operation can be followed rather than viewed only
  at its endpoint.

A module and presentation editor checks the defining relations and shows a
conflicting path when they fail. Valid constructions can be exported as Julia
code or mathematical objects. For planar Rips and codensity examples, the source
laboratory lets us edit points and construction parameters, rebuild the
filtration, and compare the original and edited results. Undo and redo retain
the editing history.

Figure composition brings named panels together with shared scales,
annotations and checked connectors between source cycles and barcode intervals.
`VisualStyle` supplies consistent typography, colors, spacing and emphasis.
Marker shapes and labels distinguish mathematical roles in grayscale as well
as color; interval labels, image axes and narrow layouts have been revised
for readability.

Named saved views preserve exact readouts, selections and mathematical context.
Prepared states can be shared in self-contained HTML, while supported retained
sources can be reopened in Julia for further queries. Figures can be exported
as PNG, SVG or PDF. The [visualization guide](https://tamerop.com/guides/visualization.html)
explains the display and saving choices.

### Less repeated work in the computation

A recurring aim of this release was to make a requested answer determine how
much algebra is prepared. Asking for a dimension should not require every
representative, and asking another coordinate question of the same result
should be able to reuse the factorization already available.

- Homology and cohomology results retain reusable matrix factorizations and
  construct quotient representatives when requested. Exact cohomology dimension
  queries also defer boundary coordinates. Ext bases reuse retained
  representatives and can be transported together to another resolution model.
- Rational Hom constraints use sparse integer elimination to reduce repeated
  fraction arithmetic. Suitable dense rational products and factorizations use
  Nemo, the exact algebra library, while structured and sparse calculations
  retain their corresponding methods. Tables of Yoneda products, which compose
  Ext classes, reuse their lifts within the calculation.
- Ordinary persistence over F₂ reuses column storage and scratch buffers,
  and combines many binary entries in a single machine operation. Component
  merging and previously identified persistence pairs help skip unnecessary
  reductions. Exact grades and their tie order are retained.
- Eligible nonperiodic cubical barcodes in dimensions one through three work
  directly from grid grades and neighbors over F₂. Component pairings, compact
  square boundaries and local volume boundary plans reduce the remaining
  reduction. General cubical construction uses arithmetic grid indexing and
  preallocated sparse boundaries.
- Rips computations reuse neighbor lists and distance order, combine landmark
  selection with distance updates, and store triangles and reduction terms
  more compactly. Immediate cancellations at equal grades avoid filling queues
  with work that would be removed again. Eligible bounded requests can stop
  at a radius where one vertex connects to every other vertex, making the
  Rips complex a cone.
- Full rank invariants reuse map storage and record comparable pairs without
  a square array containing all pairs, including incomparable ones. Viewers
  reuse prepared source geometry,
  compatible axes and plotted objects, and batch bars, edges, polygons, faces
  and labels. Superseded interaction requests cannot overwrite a newer answer.

Advanced users can inspect retained process caches with
`Advanced.runtime_cache_summary()` and clear them between calculations with
`Advanced.clear_runtime_caches!()` while keeping their result objects.
Precompilation now includes explicit Hom maps and positive-degree Ext on a
branching poset, preparing more of that algebra before its first use.

The [benchmark reports](https://tamerop.com/benchmarks/index.html) compare finite
module algebra with QPA, boundary-matrix persistence with PHAT, Rips workflows
with Ripser.py and C++ Ripser, and image and volume persistence with CubicalRipser.
A separate study measures the time from a viewer input to the displayed result.
The reports give the workloads and observed differences; the benchmark
repository contains the procedures and measurements needed to repeat them.

### Corrections and details for existing users

- Superlevel image cells now receive the minimum of their incident vertex
  values, correcting the previous use of the sublevel maximum rule.
- Fast axis-aligned encodings retain closed faces and isolated points at shared
  birth and death thresholds. Their exact witnesses and saved encodings
  preserve these boundary strata.
- Reversed grid axes retain their structured product representation, vertex
  labels, oriented maps and coordinate lookup after saving and loading.
- Cubical construction budgets estimate sparse topology, grades and assembly
  storage. Feasible inputs are no longer rejected by an estimate based on a
  dense boundary matrix that the construction would not build.
- Wasserstein assignment refinement and exact-rational bottleneck comparisons
  involving essential intervals have been corrected. Numerical failures in
  the assignment calculation are reported rather than becoming plausible
  distance values.
- Sparse relation checks accept cancellation with explicitly stored zero
  entries. Nonfinite rational entries are rejected before shortcuts that skip
  multiplication by zero, including when a destination matrix is supplied.
- Comparisons between projective and injective Ext calculations respect the
  configured tolerances over numerical real fields. Exact coefficients retain
  exact equality checks.
- Resolution figures default to the inexpensive multiplicity view. A
  projective lift figure is selected explicitly with `kind=:resolution_lift`.
- Retaining source representatives, persistent cocycles or a mutable reduction
  is an explicit choice, so ordinary barcode requests need not store that
  additional information. Ordinary diagram JSON uses schema 2; extended
  persistence has a separate schema preserving its interval kinds and phases.
- Julia compatibility remains at 1.12 and compatible later 1.x releases.
  MiniQhull and Hecke are optional extensions; plotting and table adapters
  continue to load only when their corresponding packages are imported.

The automated checks have been reorganized into separate core, ingestion and
invariant jobs across Linux, macOS and Windows. Visualization checks now
exercise native rendering, real browser controls and saved HTML, including
coordinate correspondence, linked selection, keyboard use and viewer cleanup.
Algebraic regression checks compare known kernels, quotient coordinates,
representatives, products and comparison maps, including concurrent queries.

## 0.1.0

The [first registered release](https://github.com/erikmnovak/TamerOp.jl/releases/tag/v0.1.0)
established TamerOp's finite encoding approach to persistence and homological
algebra. A finite encoding consists of a finite partially ordered set (poset),
vector spaces and maps on that set, and an order preserving assignment from the
original parameters to its labels. This lets a module with a large or continuous
parameter space
be studied through a finite description while retaining its meaning on that
parameter space.

The original release already contained the following capabilities.

### Inputs, geometry and finite encodings

- Point clouds could be turned into Rips filtrations and density, codensity,
  function and landmark variants. Low-dimensional alpha and Delaunay
  constructions, core filtrations, degree Rips and rhomboid multicovers supplied
  further geometric choices.
- Graph inputs supported vertex functions, clique complexes, edge weights,
  geodesic distances, centrality and graph core constructions. Images supplied
  cubical filtrations and constructions using their values and distances.
- Users could supply graded complexes directly, including complexes where a
  cell has several minimal entry grades. Construction options controlled
  dimensions, geometry, intermediate results and resource limits.
- Modules could also begin algebraically: with spaces and maps on a finite
  poset, lattice presentations, or piecewise linear regions. Fringe
  presentations described a module as the image of a map from pieces supported
  on upsets to pieces supported on downsets. An upset contains everything above each
  of its points; a downset contains everything below.
- Encodings supported individual and batched parameter queries, coarsening,
  the structure of finite grids and summaries of their spaces and maps. Rational and
  real algebraic grades were retained by the constructions using them.
- Rhomboid multicovers used exact sphere predicates and physical radii.
  General polyhedral encodings retained nonconvex pieces and modules supported
  only on boundaries. Region queries included volumes, boundaries, centroids,
  moments and support measurements, with exact or sampled methods according
  to the question.

### Ordinary persistence

The ordinary reducer computed barcodes over F₂ from supplied filtered complexes
and supported input constructions. Results retained exact finite endpoints and
separated finite intervals from essential classes, which have no death in the
stored filtration.

Cubical persistence supported sublevel and superlevel filtrations and periodic
axes. Values assigned to top-dimensional cells were supported for two-dimensional
images; values assigned to vertices already worked in arrays of arbitrary
positive dimension. These were two distinct ways of describing the filtered
space.

### Spaces, maps and derived algebra

- Finite modules supported morphisms, direct sums, submodules and quotients.
  Kernels, images, cokernels and coimages came with the maps identifying them.
  Pushouts and pullbacks combined diagrams with a common source or target, while
  equalizers and coequalizers imposed agreement between two maps.
- Short exact sequences and the snake lemma supplied connecting maps between
  kernels and cokernels. These operations worked with actual module maps and
  their equations.
- Projective covers, injective hulls and projective, injective and indicator
  resolutions expressed modules through simpler building blocks. Betti and
  Bass tables recorded the multiplicities of those blocks by location and
  degree.
- Hom computed the space of module maps. Ext and Tor extended Hom and tensor
  computations through resolutions, with dimensions, bases, representatives,
  coordinates and induced maps. Yoneda composition combined Ext classes into
  multiplication tables for Ext algebras, with actions of Ext classes on Tor.
  Tor products used supplied compatible multiplicative data.
- Chain and cochain complexes supported homology, cohomology, maps, homotopies,
  mapping cones and their long exact sequences. Module complexes extended
  these calculations across the finite base.
- Derived Hom and tensor complexes, hyper-Ext, hyper-Tor and spectral sequences
  organized calculations involving complexes of modules. Spectral sequences
  exposed successive pages and differentials, with convergence and extension
  information where determined by the construction.

These derived calculations were already formulated in the category of modules
on the specified finite poset. Changing that base was an explicit mathematical
operation, supported by restriction, left and right Kan extensions and their
derived versions. Kan extensions transport a diagram along a map of posets.
Common refinements supplied shared finite descriptions and the maps needed to
compare their modules.

### Invariants, slices and features

- Rank invariants counted the dimensions of images between comparable
  parameters. Hilbert functions recorded the dimension at a parameter, and
  Euler surfaces gave alternating sums across homological degrees.
- Signed point measures and signed rectangle barcodes expressed changes in
  these quantities through positive and negative contributions. Betti and
  Bass support summaries described where the resolution data occurred.
- Restricting to a line produced an ordinary barcode. Reusable arrangements
  of slice barcodes, projected barcodes and sampled matching distances
  supported comparisons between multiparameter modules. An exact optimizer
  computed the two-dimensional matching distance within a specified finite
  rectangle.
- Bottleneck and Wasserstein distances between barcodes, persistence landscapes,
  images, silhouettes, entropy and scalar summaries were available. Sampled
  tracks across slices supplied multiparameter image and decomposition
  summaries; these tracks recorded barcode matchings rather than certified
  decompositions of the source module as a direct sum.
- Feature specifications could be combined, applied in batches and saved with
  names and parameter axes. Experiment records retained the chosen pipeline
  and numerical features for subsequent analysis.

### Arithmetic, figures and interchange

The algebra supported rational coefficients, prime fields and numerical real
fields, with explicit field choices and numerical tolerances. Exact linear
algebra used native methods and Nemo. Result summaries retained the construction
and coefficient choices, and inexpensive inspection could leave larger module
or basis calculations deferred.

A common `visualize`, `save_visual` and `save_visuals` interface covered source
geometry, encoding regions, ranks, barcodes, persistence diagrams, slice
families, signed measures and numerical features. Optional CairoMakie and
WGLMakie extensions supplied static figures and HTML output.

Datasets, pipelines, encodings, lattice and piecewise linear presentations,
feature sets and sampled multiparameter results had saving and loading
interfaces. Prescribed external file formats connected selected workflows
with other persistence and computer algebra tools. Optional integrations
included Tables, DataFrames, CSV, Arrow, Parquet, NPZ, Distances, KernelFunctions,
NearestNeighbors and DelaunayTriangulation, with Folds and ProgressLogging for
selected execution and progress reporting tasks.

The package required Julia 1.12 and included tests of the installed package and
its source contents, release instructions and software citation metadata. The
[tagged source](https://github.com/erikmnovak/TamerOp.jl/tree/v0.1.0) preserves the
implementation and documentation belonging to that release.
