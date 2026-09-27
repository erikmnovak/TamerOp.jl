# Concurrent polyhedral computations

Use this guide when running polyhedral computations from several Julia
tasks or changing code that calls CDD, the polyhedral backend. TamerOp runs
its CDD work one operation at a time, while surrounding computations can
still run concurrently.

One internal execution lock coordinates this work across PL geometry and
incremental rhomboid construction. Exact rational and floating CDD operations
use the same protected boundary. There is no user option to disable it.

The boundary includes lazy conversion, point/facet materialization, incidence,
volume, and centroid computation. Incremental rhomboid construction copies its
facets into ordinary Julia data before continuing outside the boundary. PL
caches may retain backend representations; later library accesses are also
protected. Session and region cache locks are acquired before the CDD boundary.
Code inside that boundary does not acquire a session or region cache lock.

This contract covers calls through TamerOp. Independent calls to CDDLib or
CDD-backed Polyhedra operations in other packages do not participate in its
lock. Run such external computations in a separate process if they overlap
with TamerOp work. Keep CDD global solver, logging, and arithmetic configuration
fixed after initialization. Accessing and modifying backend objects through
raw fields is outside the supported concurrent API. Input arrays must not be
mutated while a computation reads them; clearing a PL geometry cache remains
an exclusive maintenance operation after its users finish.

The installed cddlib artifact has thread-local workspaces, but its LP statistics
still use non-atomic process globals. That concrete race surface motivates the
shared boundary; no incorrect geometric answer was observed in this review.
Dependency finalizers are unchanged: the reviewed native free routines release
object-owned allocations and do not update those globals. This is a scoped
contract for the inspected dependency versions, not a proof that arbitrary
CDD clients are safe together. The supporting source and artifact review,
including its version records and focused checks, is maintained locally and
is not included in the published library.
