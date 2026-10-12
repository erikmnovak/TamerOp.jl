```@raw html
<div class="home-page">
<div class="home-opening">
<header class="home-intro">
<p class="home-eyebrow">Multiparameter persistence in Julia</p>
```

# TamerOp

TamerOp is the Toolkit for Algebraic Module Encodings over ``\mathbb{R}^n``
and Other Posets, a Julia library for studying how mathematical objects change
with one or more parameters.

A persistence module records a vector space at each parameter and linear maps
that follow its vectors as the parameters change. We work with a **finite
encoding**: spaces and maps on a finite set of labels, together with an
assignment from the original parameters to those labels. The labels form a
partially ordered set, or **poset**, which records which ones can be compared.
This finite description lets us recover spaces and maps at the original
parameters and use them in further calculations.

```@raw html
</header>
<nav class="home-starts" aria-label="Ways to begin">
  <p class="home-eyebrow">Choose your starting point</p>
  <a class="home-start" href="tutorials/ring.html">
    <span class="home-start-kind">Start with an example</span>
    <strong>A hole appears and disappears <span aria-hidden="true">↗</span></strong>
    <span>Watch a shape change, then read its barcode.</span>
  </a>
  <a class="home-start" href="persistence_modules.html">
    <span class="home-start-kind">Start with definitions</span>
    <strong>Persistence modules <span aria-hidden="true">↗</span></strong>
    <span>Build the idea from spaces and the maps between them.</span>
  </a>
  <p class="home-setup">Read the lessons and saved figures without Julia.
    To run them, <a href="start/install.html">install and open a notebook</a>.</p>
</nav>
</div>
<p class="home-returning">If you already have an input in mind,
  <a href="guides/inputs_to_objects.html">construct its module and explore the maps <span aria-hidden="true">→</span></a>.
</p>

<figure class="home-workflow" aria-labelledby="home-workflow-title home-workflow-caption">
  <div class="home-figure-heading">
    <p id="home-workflow-title">Computing with a finite encoding</p>
    <span>Different inputs lead to spaces and maps that we can study together.</span>
  </div>
  <div class="home-flow">
    <div class="home-flow-inputs">
      <p class="home-flow-label">Begin with</p>
      <a class="home-flow-item" href="guides/inputs_to_objects.html"><strong>Build from filtered data <span aria-hidden="true">↗</span></strong><span>Points, images, graphs, or cells with grades</span></a>
      <a class="home-flow-item" href="guides/inputs_to_objects.html"><strong>Define a module directly <span aria-hidden="true">↗</span></strong><span>Sloped polyhedral regions, presentations, or finite posets</span></a>
    </div>
    <div class="home-flow-arrow" aria-hidden="true"><span>prepare<br> and encode</span><b>→</b></div>
    <a class="home-model" href="finite_encodings.html" aria-label="Understand finite encodings through the square example">
      <strong>A finite encoding</strong>
      <span class="home-model-subtitle">Spaces, maps, and parameter lookup</span>
      <svg viewBox="0 0 350 220" role="img" aria-labelledby="home-square-title home-square-desc">
        <title id="home-square-title">From the square to a finite model</title>
        <desc id="home-square-desc">A point inside a closed square is assigned to the highlighted vertex of a diamond with four vertices. This vertex carries the rational vector space of dimension one; the other three carry zero spaces. The diamond arrows point upward. The dotted line follows the parameter into its finite label.</desc>
        <defs>
          <marker id="home-order-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 8 4 L 0 8 z" class="home-arrowhead"/></marker>
        </defs>
        <g class="home-grid" fill="none">
          <path d="M42 25V144 M78 25V144 M114 25V144 M18 56H145 M18 92H145 M18 128H145"/>
        </g>
        <path class="home-axis" d="M18 25V144H145" fill="none"/>
        <rect class="home-square" x="42" y="56" width="72" height="72"/>
        <path class="home-lookup" d="M78 92 C140 92 149 92 206 92" fill="none"/>
        <circle class="home-point" cx="78" cy="92" r="5"/>
        <text class="home-point-label" style="font-size:22px" x="67" y="78">q</text>
        <g class="home-order" fill="none" marker-end="url(#home-order-arrow)">
          <path d="M260 139L225 104"/><path d="M276 139L310 104"/>
          <path d="M224 80L258 44"/><path d="M310 80L277 44"/>
        </g>
        <circle class="home-zero-node" cx="268" cy="32" r="16"/>
        <circle class="home-active-node" cx="216" cy="92" r="17"/>
        <circle class="home-zero-node" cx="320" cy="92" r="16"/>
        <circle class="home-zero-node" cx="268" cy="151" r="16"/>
        <g class="home-space-label" style="font-size:22px" text-anchor="middle" dominant-baseline="central">
          <text x="268" y="32">0</text><text x="216" y="92">ℚ</text>
          <text x="320" y="92">0</text><text x="268" y="151">0</text>
        </g>
        <g class="home-panel-label" style="font-size:22px" text-anchor="middle">
          <text x="81" y="182">original<tspan x="81" dy="27">parameters</tspan></text>
          <text x="268" y="182">finite<tspan x="268" dy="27">model</tspan></text>
        </g>
      </svg>
      <span class="home-model-note">The space has dimension one inside the square and zero outside.</span>
    </a>
    <div class="home-flow-arrow home-flow-arrow-out" aria-hidden="true"><span>ask</span><b>→</b></div>
    <div class="home-flow-outputs">
      <p class="home-flow-label">What next?</p>
      <a class="home-flow-item" href="guides/spaces_and_maps.html"><strong>Inspect spaces and maps <span aria-hidden="true">↗</span></strong><span>Recover a space or a map between parameters</span></a>
      <a class="home-flow-item" href="topics/algebra.html"><strong>Construct and relate modules <span aria-hidden="true">↗</span></strong><span>Use morphisms, kernels, cokernels, and resolutions</span></a>
      <a class="home-flow-item" href="topics/invariants.html"><strong>Compute summaries <span aria-hidden="true">↗</span></strong><span>Explore ranks, slices, and numerical features</span></a>
    </div>
  </div>
  <figcaption id="home-workflow-caption">In this schematic, the dotted line follows the parameter <i>q</i> into its finite label, where we recover the space ℚ of rational numbers.
    This space has dimension one over ℚ. The upward arrows record comparisons between labels;
    the <a href="tutorials/inspect_encoding.html">square lesson</a> uses them to recover maps between original parameters.</figcaption>
</figure>
<div class="home-perspectives">
<section class="home-capability" aria-labelledby="start-from-a-filtration">
```

## Start from a filtration

As we increase a threshold, we can add points, edges or cells to a shape.
This growing family is a **filtration**. Its homology records connected
components, loops around holes and their higher dimensional counterparts,
together with the maps that follow them through the growth. A finite encoding
keeps the resulting persistence module available for further study.

For example, a compatible family of linear maps between two modules is called
a **module morphism**. We can examine its kernel, the vectors it sends to
zero, or its image, the vectors it reaches. A resolution describes a module
through generators, relations and relations among those relations. These
algebraic questions apply to modules obtained from filtered data too.

When the question concerns the birth and death of classes along a single
parameter, the [direct persistence routines](../ordinary_persistence.md)
compute their intervals from the filtered complex. The ring lesson begins
with this calculation.

```@raw html
</section>
<section class="home-capability" aria-labelledby="describe-a-module-directly">
```

## Describe a module directly

We can also specify the spaces and maps directly, using geometric regions or
a finite poset. Consider a module whose space is one dimensional on the
diagonal strip ``0 \leq x+y \leq 1`` and zero elsewhere, with identity maps
between comparable parameters inside the strip. Its exact region encoding
retains the sloped boundaries and describes the spaces and maps throughout
the plane.

Once this module is encoded, the same operations on the finite model let us
study it. The starting description can therefore come from an algebraic or
geometric question about modules themselves.

```@raw html
</section>
</div>
<p class="home-capability-next">Try both routes in
  <a href="guides/inputs_to_objects.html">From inputs to computed objects</a>.
  The <a href="finite_encodings.html">explanation of finite encodings</a> develops the
  connection to Ezra Miller's theory of modules over posets.</p>
<section class="home-evidence" aria-labelledby="performance-you-can-examine">
```

## Performance you can examine

```@raw html
<div class="home-evidence-copy">
```

The [benchmark reports](../benchmarks/index.md) compare matched mathematical
requests, with independently checked answers, timings, memory measurements,
and downloadable results.

The [QPA comparison](../benchmarks/qpa.md) measures algebra on finite modules,
where TamerOp takes substantially less time for the matched requests. The
[PHAT comparison](../benchmarks/phat.md) measures complete ordinary barcodes
over the field F₂, whose arithmetic is modulo two. Its aggregate difference
is smaller, and individual cases favor each tool.

The [Ripser.py comparison](../benchmarks/ripser.md) covers ordinary Rips
persistence, including computations on selected subsets of points, called
landmarks, and retained cocycles, which evaluate homology classes. On the
1,024-point circle, the query takes 7.01 seconds with TamerOp and 30.4 seconds
with Ripser.py. Larger planar inputs also favor TamerOp; landmark results
are mixed, and some sparse and small requests favor Ripser.py.

The separate [original C++ Ripser comparison](../benchmarks/ripser_cpp.md)
studies larger requests for barcodes. TamerOp is about 1.8 times as fast
on the two circle inputs; most dense planar cases and the larger sphere
perform similarly. C++ Ripser is 1.6–1.8 times as fast on the sparse
spheres. These observations were made on a shared machine, with substantial variation
in absolute times.

The [CubicalRipser comparison](../benchmarks/cubicalripser.md) studies large
images and volumes. TamerOp is 2.18 times as fast in the aggregate for the fixed
set of requests. For a smooth `256³` array whose values specify when voxels
appear, the request takes 6.67 seconds with TamerOp and 14.2 seconds with
CubicalRipser. Binary volumes whose values are assigned to vertices favor
CubicalRipser, while smooth 2D cases are much closer. The report explains
these input conventions and how the returned barcodes were compared.

The two Ripser studies use different workloads and timing boundaries. Each
report states its inputs, versions, timing conditions, and limits, so you can
judge what the evidence means for your work.

For the algorithms behind the results, read the
[implementation accounts](../implementation/index.md).

```@raw html
</div>
</section>
<section class="home-spectrum" aria-labelledby="from-theory-to-application">
```

## From theory to application

The documentation moves from understanding the mathematics, through exploring
the library, to completing a particular task. These are independent entrances:
start with the kind of answer you need.

```@raw html
<nav class="home-explore" aria-label="Choose a kind of treatment">
  <a href="collections/mathematics.html"><strong>Mathematics <span aria-hidden="true">→</span></strong><span>Understand the objects and why the constructions matter.</span></a>
  <a href="collections/using.html"><strong>Using TamerOp <span aria-hidden="true">→</span></strong><span>Explore capabilities, compare choices, and combine operations.</span></a>
  <a href="collections/recipes.html"><strong>Task recipes <span aria-hidden="true">→</span></strong><span>Follow code and focused explanations to a defined result.</span></a>
</nav>
<p class="home-finding">Find a route through the ideas in the <a href="reading_map.html">learning map</a>,
  browse a subject in the <a href="topic_map.html">topic map</a>, or look up an operation in
  <a href="collections/api.html">API reference</a>.</p>
</section>
</div>
```
