# Implementing the first learning path

This is the authoring brief for the first complete route through the proposed
documentation site. Its order is fixed:

**Install → ring example → why a second parameter changes the problem → finite
encoding → inspect spaces and maps → interpret a figure.**

The intended reader knows vectors, matrices, and the idea of a hole in a shape,
but need not know Julia, posets, or persistence modules. Follow the
[writing guide](writing.md). The existing [finite-encoding introduction](finite_encodings.md)
supplies the mathematical narrative; preserve it when adapting material into
the site. The destinations below are planned paths, not published pages.

Four background chapters are now available without requiring installation.
[Persistence modules over posets](persistence_modules.md) develops the ring,
spaces and maps, and comparable parameters for pages 2 and 3 below.
[Finite encodings](finite_encodings.md) follows with the square module,
recovery by pullback, and the relationship between the nine-label and
four-label representations for pages 4 and 5.
[Indicator presentations](indicator_presentations.md) explains the square's
input regions and coefficient matrix, then develops two overlapping square
summands with inclusion and projection maps. This supplies a richer example
for later exploration without replacing the first path's square.
[Tameness and scope](tameness.md) connects constant subdivisions, finite
encodings, and finite fringe presentations, and uses a module of constant
dimension one to show why controlling stalk dimensions alone is insufficient.
Their static diagrams are available. The inspection notebook now adds linked
parameter, poset, and matrix selection; following chosen vectors and linking
source representatives remain planned.

The square's inspection lesson now has a
[canonical teaching notebook](tutorials/inspect_encoding.ipynb). It follows
original parameters into the actual finite poset, checks stalks and maps, and
exports a combined figure through the package API. Two overlapping squares
show why nonzero successive maps can have zero composite. A live WGLMakie
session then lets the reader change the selection and switch between module
and presentation views, with static snapshot export. All 23 notebook code cells
have passed under headless display capture, and the static figures have passed
visual review. The 550 inspector assertions pass across five coefficient fields,
including native callback and lifecycle checks; 116 additional checks cover
server embedding. The author reports that the two-square live-server browser
checklist passes, including pointer/keyboard controls, exact queries, linked
tabs and lifecycle behavior. The notebook frontend remains separately
unverified. This supplies the square portion of pages 4-6; connecting it
to the ring lesson and publishing
the complete site remain separate steps.

## Page briefs

### 1. Install and reopen a working environment

- **Destination:** `start/install.md`.
- **Reader's question:** How do I install TamerOp and return to the same work later?
- **Prerequisites:** Ability to open a terminal or the Julia application.
- **Starting object:** An empty working directory and a supported Julia installation.
- **Content:** Adapt the README's installation, project activation, import, and
  reopening instructions. Distinguish terminal commands from Julia commands;
  explain the purpose of `Project.toml` and `Manifest.toml`. Keep plotting
  dependencies for page 6. Verify the distribution instructions when publishing.
- **Expected conclusion:** The reader can load `TamerOp` as `OP` in their own
  environment and reopen it without reinstalling the package.
- **Transition:** What small calculation has an answer I can recognize?
- **Acceptance:** A fresh environment loads the published package through these
  instructions. Local package loading alone does not satisfy this check.

### 2. Follow a hole through the ring example

- **Destination:** `tutorials/ring.md`.
- **Reader's question:** What does a persistence interval tell me about a shape?
- **Prerequisites:** Page 1; no homology terminology assumed.
- **Starting object:** The matrix `[0 0 0; 0 5 0; 0 0 0]` interpreted as appearance
  grades of square top cells in a nonperiodic cubical sublevel filtration.
- **Content:** Explain the outer ring at grade 0 and the filled center at grade 5,
  then introduce a homology class, birth, death, and degree. Use the direct
  `cubical_persistence` route over F₂ and inspect intervals and provenance.
- **Expected conclusion:** There is one H₁ interval `[0,5)` and one essential
  H₀ class born at 0. The right endpoint is excluded because the hole is already
  filled at 5. This call returns a persistence diagram, not an `EncodingResult`.
- **Transition:** What changes if a second, independently varying parameter
  influences which cells are present?
- **Acceptance:** Predict and verify that changing the center's grade to 3 gives
  `[0,3)` while the essential component still begins at 0.

### 3. Explain why a second parameter changes the problem

- **Destination:** `explanations/two_parameters.md`.
- **Reader's question:** Why can I no longer arrange all parameter choices on one line?
- **Prerequisites:** Page 2; comparisons of pairs of real numbers.
- **Starting object:** A schematic two-parameter filtration with its coordinate
  directions explicitly chosen so that increasing either coordinate gives an
  inclusion. A density cutoff with the opposite convention needs reorientation.
- **Content:** Introduce coordinatewise order through two comparable pairs and
  an incomparable pair. Explain vector spaces and structure maps before naming
  a persistence module. Show why counts of classes do not specify the maps.
  Avoid implying that the ring's direct barcode computation has secretly
  constructed a bifiltration or that every multiparameter module has a barcode.
- **Expected conclusion:** The reader can decide when a parameter pair carries
  a structure map and explain why retaining dimensions alone loses information.
- **Transition:** Can infinitely many spaces and maps have a finite description?
- **Acceptance:** The reader identifies `(1/4,3/2)` and `(3/2,1/4)` as incomparable,
  although both lie in the square used on the next page.

### 4. Construct a finite encoding of the square module

- **Destination:** `tutorials/finite_encoding.md`.
- **Reader's question:** What finite object can recover this module on ℝ²?
- **Prerequisites:** Page 3; the meaning of an identity and a zero linear map.
- **Starting object:** Exactly the closed square-supported module from
  `finite_encodings.md`, with coefficient field ℚ and support `[0,2]²`.
- **Content:** First work through the spaces and maps below. Revisit the valid
  nine-region Cartesian model from the introduction. Then express the module
  as the image of the scalar `[1]` between one birth upset and one death downset,
  explaining those terms through their regions before using the constructors.
  Construct an actual `EncodingResult` and identify its finite poset, finite
  module, and assignment from original parameters. Explain the returned
  representation; do not demand nine labels or particular numeric label IDs.
- **Expected conclusion:** The reader can explain `M ≅ M_P ∘ π` as recovery of
  both spaces and maps. Two different finite encodings can recover this same
  module. Connect this finite-description viewpoint to Ezra Miller's theory
  without claiming every abstract tame module has an implemented encoder.
- **Transition:** How can I check what the result says at particular parameters?
- **Acceptance:** Verify the closed boundaries, the interior identity map, and
  a map leaving the support. Display the actual returned labels and their order.

### 5. Inspect spaces and maps in the returned representation

- **Destination:** `tutorials/inspect_encoding.md`.
- **Reader's question:** How do I recover a space or a map at original parameters?
- **Prerequisites:** Page 4; matrix sizes and multiplication.
- **Starting object:** The same square `EncodingResult`; do not reconstruct a
  disconnected example or substitute an identity encoding of a finite input.
- **Content:** Start with `describe`, `provenance`, and `dimensions`, then use
  `encoding_poset`, `encoding_map`, and `encoding_module`. Explain the boundary
  between the root workflow and the advanced `locate`, `leq`, `dim_at`, and
  `structure_map` queries. Explain source columns and target rows. Inspect
  maps as well as dimensions; compare two compositions to their direct map.
  For these objects, `dimensions(enc)` returns the stalk-dimension vector,
  whereas `dimensions(M)` returns a summary with a `stalks` entry; explain
  method-specific return values instead of treating them as interchangeable.
  The implemented notebook also follows `encoding_presentation` into the active
  downset rows and upset columns. It explains why the image of the restricted
  matrix is the stalk, explicitly requests embedded image bases, and checks
  the induced-map equation and the two-square zero composite. An active-zero
  block distinguishes support membership from a nonzero coefficient.
- **Expected conclusion:** The reader can recover an interior identity and a
  correctly shaped zero map, and can distinguish order in ℝ² from order among
  finite labels. Equal labels do not imply that the original points are comparable.
- **Transition:** How can a figure make this finite model easier to understand?
- **Acceptance:** Explain the table below before running its queries. An exercise
  uses an incomparable pair and asks why there is no structure map to query in
  the original module, even if both points receive the same finite label.

### 6. Interpret a figure of the same encoding

- **Destination:** `tutorials/interpret_encoding_figure.md`.
- **Reader's question:** Which parts of the module does this picture show, and
  what do I still need to query?
- **Prerequisites:** Page 5; optional renderer installation introduced here.
- **Starting object:** The same verified square encoding and its query results.
- **Content:** Prepare a parameter-plane support picture and a finite-poset
  diagram using labels obtained from the result. Mark the square's included
  boundary and show the spaces and representative maps. Reuse labels across
  panels. Explain that a dimension heatmap alone does not display the maps.
  Introduce `available_visuals`, `visualize`, and `save_visual` through the
  supported recipe actually selected and executed during implementation.
  After checking the mathematics, introduce `inspection_session`: select an
  exact parameter or finite label, preserve the query when switching between
  module and presentation views, and save `inspection_snapshot`. Explain why
  pointer coordinates are approximate and why the WGLMakie controls need live
  Julia. Keep basis computation an explicit single-stalk choice.
- **Expected conclusion:** The reader can trace a parameter through its label
  to a space, interpret an arrow, and state what the figure omits. They can
  save a figure and reopen the canonical notebook.
- **Transition:** How would data give rise to an encoding, or how would a module
  with larger spaces and nontrivial matrices behave? These are later paths.
- **Acceptance:** Executed notebook, downloaded notebook, and published figure
  agree. Captions state the field, domain, closed support, and meaning of arrows.
  Static A42a/A83 recipes and exports have passed native rendering review.
  The expanded notebook's 23 code cells pass under headless display capture;
  its selected-state exports have also passed static visual review. All 550
  inspector assertions pass across five fields, including native callback and
  lifecycle checks; 116 additional server-embedding checks pass. The author
  reports successful manual acceptance of the local live-server two-square
  inspector, including layout, exact queries, pointer/keyboard controls, view
  changes, error recovery, reset, reload, linked tabs and closure. Actual
  notebook-frontend acceptance remains separate; captured notebook displays
  do not establish that integration.

## Mathematical specification before tutorial cells

Fix coordinatewise order on ℝ² and coefficient field ℚ. Let
`U = {q : q₁ ≥ 0 and q₂ ≥ 0}` and `D = {q : q₁ ≤ 2 and q₂ ≤ 2}`.
Take the image of the scalar-one map from the indicator module of `U` to that
of `D`. Its stalk at `q` is ℚ precisely on `U ∩ D = [0,2]²`, and is zero
elsewhere. For `q ≤ r`, its map is `[1]` if both points are inside the square;
otherwise it is the unique zero matrix of size `dim(M(r)) × dim(M(q))`.

This rule respects composition: if both ends of a comparable triple lie in
the square, the middle point does too. In every other case the direct map
and the composite are zero. Endomorphisms of a zero space are `0 × 0` matrices
and serve as its identity. These statements specify the mathematical object
before any label numbering or executable lesson is chosen.

| Query | Expected answer | Reason |
| --- | --- | --- |
| Stalk at `(-1,1)` or `(3,1)` | dimension 0 | Outside the support |
| Stalk at `(0,0)`, `(2,2)`, `(0,2)`, or `(2,0)` | dimension 1 | Both boundaries are included |
| `(1/4,1/2) → (1,3/2)` | `1 × 1` identity | Comparable interior points |
| `(-1,1) → (0,1)` | `1 × 0` zero matrix | Entering the support |
| `(1,3/2) → (3,3/2)` | `0 × 1` zero matrix | Leaving the support |
| `(-1,1) → (1,1) → (3,1)` | Composite equals the direct `0 × 0` map | Functoriality through a nonzero space |
| `(1/4,3/2)` and `(3/2,1/4)` | No comparison in either direction | A shared label cannot add an ambient comparison |

For the oracle's `backend=:pl_backend, poset_kind=:signature` configuration,
explain labels by the
two bits `u(q) = [q ∈ U]` and `c(q) = [q ∉ D]`. Both bits are order preserving.
The realized signatures are `(0,0)`, `(1,0)`, `(0,1)`, and `(1,1)`, ordered
coordinatewise. Only `(1,0)` has a one-dimensional space. This gives a diamond
with four labels; it is a different valid encoding from the illustrative
nine-label grid. Numeric IDs belong to the returned representation and should
be discovered with `locate`, never used as the definition of a region.
Exterior points still receive valid labels with zero-dimensional stalks:
the square is the support, not a restriction of the query domain. A label can
also represent a disconnected set of parameters, rather than one rectangle.

The checker verifies that characterization for the selected encoder. A future
encoder could use a different finite poset: update the explanation of its
representation while retaining the stalk, map, boundary, and composition
requirements. Finite sampling checks the implementation on the example; the
argument above explains the module over its entire domain.

## Executable evidence and authoring order

The self-contained [mathematical oracle](build_scripts/check_first_encoding.jl)
checks the ring and square through public package APIs. It is a validation
fixture, not a second independently authored tutorial. It includes boundary
and exterior points, all comparable pairs and triples in its sample, and an
incomparable pair with a shared label. Run from the repository root:

```sh
julia --startup-file=no --project=. docs/build_scripts/check_first_encoding.jl
```

The verified Julia 1.12.1 run passed 8,710 checks, including 49 square query
points, 784 comparable-pair maps, and 7,056 compositions. These are checks of
the stated example, not a claim about all encoders or arbitrary modules.

The [indicator-presentation oracle](build_scripts/check_indicator_presentations.jl)
also checks the square and its two-summand extension. Its Julia 1.12.1 run
passed 43,134 checks: 49 one-square points and 81 two-square points, all
comparable-pair maps and compositions in those samples, and agreement with
the prescribed summands in compatible bases. It verifies the inclusion,
projection, and zero composite described in the third background chapter.
Run it with:

```sh
julia --startup-file=no --project=. docs/build_scripts/check_indicator_presentations.jl
```

Keep installation verification, mathematical checks, rendering checks, and
reader review separate. The background chapters, mathematical oracles, and
square inspection notebook are available. The documentation site, the linked
ring-to-square narrative, first-path reference prose, and reader review remain
to be completed.

1. Reconcile the [API inventory and backlog](api_coverage.toml); use the selected
   first-path bindings to constrain the first reference-writing pass.
2. Add the documentation environment, site build, and navigation containing
   only finished pages. Move existing explanations with links and redirects
   where needed, preserving their mathematical qualifications.
3. Integrate the canonical square inspection notebook into this route and
   connect the ring calculation to it. Generate displayed cells, outputs, and
   downloads from the notebook source; do not maintain competing examples.
4. Write only the reference entries needed for this route initially. Include
   the specific method families, field/grade conventions, options, returned
   objects, and meaningful errors. Keep other families in the manifest backlog.
5. Execute from a clean documentation environment, check installation separately,
   render and inspect the figure, then ask a reader with the stated prerequisites
   to explain the result. Publish the first path when these checks pass; the
   remaining reference backlog does not block it.
