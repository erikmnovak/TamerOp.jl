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
question you want to pursue.

Start with [installation](start/install.md), then follow
[a hole that appears and disappears](tutorials/ring.md). The lesson contains
static figures and a downloadable notebook with its computed results. It leads
to the short bridge [why two parameters change the problem](../two_parameters.md),
then to [the square's finite encoding](finite_encodings.md).

For the full mathematical development, follow:
[persistence modules](persistence_modules.md) →
[finite encodings](finite_encodings.md) →
[indicator presentations](indicator_presentations.md) →
[tameness](tameness.md) →
[why finite computations stay tame](../practical_tameness.md).
These chapters explain the objects, constructions,
and their assumptions without requiring Julia. The practical lessons use
the same examples and link into those explanations as each question arises.
