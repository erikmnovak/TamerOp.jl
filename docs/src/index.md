# Keep the module, make its description finite

When a shape changes, we can ask which components or holes survive. One
parameter gives us a timeline. With several parameters, some choices cannot
be compared, and a timeline no longer describes the whole problem.

TamerOp works with **finite encodings**: a finite partially ordered set, vector
spaces and linear maps on it, and an assignment from original parameters to
that finite model. Under the encoding's hypotheses, these pieces recover the
represented module's spaces and maps.

Use the [linked reading map](reading_map.md) to see how the lessons and
mathematical chapters connect, choose a starting point, and find the next
question you want to pursue. The [topic map](topic_map.md) brings together
mathematical explanations, practical guides, implementation accounts and
measured comparisons about the same subject.

Start with [installation](start/install.md), then follow
[a hole that appears and disappears](tutorials/ring.md). The lesson contains
static figures and a downloadable notebook with its computed results. It leads
to the short bridge [why two parameters change the problem](../two_parameters.md),
then to [the square's finite encoding](finite_encodings.md).
Continue by [inspecting its spaces and maps](tutorials/inspect_encoding.md):
follow a parameter into the finite model, then compare maps between two
overlapping squares. This lesson also includes saved figures and an executed
notebook, with optional instructions for live inspection.

If you have already constructed an object, use the library guide
[Exploring spaces and maps](../spaces_and_maps.md) to choose dimension and map
queries, inspect retained presentation bases, and connect the results to a
static or live view.

For the full mathematical development, follow:
[persistence modules](persistence_modules.md) →
[finite encodings](finite_encodings.md) →
[indicator presentations](indicator_presentations.md) →
[tameness](tameness.md) →
[why finite computations stay tame](../practical_tameness.md).
These chapters explain the objects, constructions,
and their assumptions without requiring Julia. The practical lessons use
the same examples and link into those explanations as each question arises.

## Find the kind of answer you need

- [Mathematical lessons](collections/mathematics.md) develop the objects and the
  questions we can ask about them.
- [Using TamerOp](collections/using.md) explores workflows and the choices they offer.
- [API reference](collections/api.md) helps locate precise operation contracts.
- [Implementation](../implementation/index.md) explains how computations work.
- [Benchmark results](../benchmarks/index.md) examines measured comparisons.

For questions, reports or contributions, visit
[Contributors and contributing](contributing/index.md).
