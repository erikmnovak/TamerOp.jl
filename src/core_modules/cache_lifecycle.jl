# Session ownership and explicit maintenance of process-owned computational caches.

function _locked_store_count(stores, locks)
    total = 0
    for i in eachindex(stores)
        total += Base.lock(() -> length(stores[i]), locks[i])
    end
    return total
end

function _resolution_cache_counts(cache::ResolutionCache)
    return Base.lock(cache.lock) do
        (projective=length(cache.projective) + (cache.projective_primary === nothing ? 0 : length(cache.projective_primary)),
         injective=length(cache.injective) + (cache.injective_primary === nothing ? 0 : length(cache.injective_primary)),
         indicator=length(cache.indicator) + (cache.indicator_primary === nothing ? 0 : length(cache.indicator_primary)),
         ext_projective=length(cache.ext_projective), ext_injective=length(cache.ext_injective),
         ext_unified=length(cache.ext_unified), tor_first=length(cache.tor_first),
         tor_second=length(cache.tor_second), hom_bicomplex=length(cache.hom_bicomplex),
         ext_doublecomplex=length(cache.ext_doublecomplex),
         tor_doublecomplex_plan=length(cache.tor_doublecomplex_plan),
         tor_doublecomplex=length(cache.tor_doublecomplex))
    end
end

# Heavy helpers contribute their own cheap counts without making CoreModules
# depend on their mathematical owners.
_session_helper_counts(::Nothing) = NamedTuple()
function _session_helper_counts end

"""
    session_cache_summary(session)

Inspect session-owned entries without constructing missing artifacts, invoking
the garbage collector or traversing mathematical objects. Counts are per-store
observations during concurrent queries; inspect an idle session for a stable
inventory. Counts are not bytes. User-retained results, intrinsic module caches,
process-owned caches and dependency/runtime storage are outside this inventory.
"""
function session_cache_summary(session::SessionCache)
    return Base.lock(session.lock) do
        buckets = _session_cache_summary_counts(session)
        enc = (posets=0, cubical=0, region_posets=0, geometry=0)
        for cache in _session_encoding_values(session)
            counts = Base.lock(cache.lock) do
                (posets=length(cache.posets), cubical=length(cache.cubical),
                 region_posets=length(cache.region_posets), geometry=length(cache.geometry))
            end
            enc = map(+, enc, counts)
        end
        resolutions = _resolution_cache_counts(session.resolution)
        module_resolutions = map(_ -> 0, resolutions)
        payloads = 0
        for cache in _session_module_values(session)
            module_resolutions = map(+, module_resolutions, _resolution_cache_counts(cache.resolution))
            payloads += length(cache.payload)
        end
        return (kind=:session_cache, buckets=buckets, encoding=enc,
                retained_module_keys=_locked_store_count(session.module_field_keys, session.module_locks),
                module_payloads=payloads, resolutions=resolutions, module_resolutions=module_resolutions,
                hom_system=_session_helper_counts(session.hom_system),
                slice_plan=_session_helper_counts(session.slice_plan),
                retention=:until_reset_or_release, reset_boundary=:idle,
                mutation=:read_only_mathematical_objects)
    end
end

"""
    empty!(session::SessionCache)

Release all session-owned memo entries and return the reusable session. Finish
or join every query using it before calling this method. Independently retained
mathematical results remain usable; resetting one session leaves other sessions
and process-owned caches intact. It does not repair raw mutation of mathematical
objects, clear their intrinsic memo fields, unload code or force collection.
"""
function Base.empty!(session::SessionCache)
    _clear_session_cache!(session)
    return session
end

# Product dictionaries used to be the only unsynchronized session stores.
# Their mathematical construction stays outside this brief publication lock.
function _session_product_lookup(session::SessionCache, store, key)
    return Base.lock(() -> get(store, key, nothing), session.lock)
end
function _session_product_publish!(session::SessionCache, store, key, entry)
    return Base.lock(() -> get!(store, key, entry), session.lock)
end

struct _RuntimeCacheRegistration
    name::Symbol
    retention::Symbol
    count::Function
    clear::Function
    roots::Function
end
const _RUNTIME_CACHES = Dict{Symbol,_RuntimeCacheRegistration}()
const _RUNTIME_CACHES_LOCK = Base.ReentrantLock()

function _register_runtime_cache!(name::Symbol; retention::Symbol, count, clear, roots)
    Base.lock(_RUNTIME_CACHES_LOCK) do
        haskey(_RUNTIME_CACHES, name) && error("duplicate runtime cache registration: $name")
        _RUNTIME_CACHES[name] = _RuntimeCacheRegistration(name, retention, count, clear, roots)
    end
    return nothing
end

function _register_runtime_cache!(name::Symbol, store; retention::Symbol, cache_lock=nothing)
    count = cache_lock === nothing ? () -> length(store) : () -> Base.lock(() -> length(store), cache_lock)
    clear = cache_lock === nothing ? () -> empty!(store) : () -> Base.lock(() -> empty!(store), cache_lock)
    return _register_runtime_cache!(name; retention, count, clear, roots=() -> (store,))
end

function _register_runtime_cache!(name::Symbol, cache::_TaskLocalCache; retention::Symbol=:task_workspace)
    return _register_runtime_cache!(name; retention,
        count=() -> Base.lock(() -> length(cache.values), cache.lock),
        clear=() -> _clear_task_local!(cache), roots=() -> (cache,))
end

function _runtime_cache_registrations()
    return Base.lock(_RUNTIME_CACHES_LOCK) do
        sort!(collect(values(_RUNTIME_CACHES)); by=x -> String(x.name))
    end
end

"""
    runtime_cache_summary()

List registered process-owned computation caches, entry counts and retention
policies. Call at an idle boundary. A weak-owner entry count counts owner buckets,
not their children; a task-workspace count counts task buckets. This is an
explicit inventory, not a census of Julia, native libraries or user objects.
Reading it does not compute, collect garbage or estimate recursive byte sizes.
"""
function runtime_cache_summary()
    return [(name=r.name, entries=r.count(), retention=r.retention)
            for r in _runtime_cache_registrations()]
end

"""
    clear_runtime_caches!()

Release the process-owned computation entries listed by `runtime_cache_summary`.
Finish all TamerOp computations before calling: maintenance does not cancel or
synchronize active mathematical work. Existing user-owned results and explicit
sessions are preserved. Backend registrations, tuning choices, compiled code,
external dependency caches and intrinsic memo fields of retained mathematical
objects are preserved. No collection or return of allocator pages is promised.
"""
function clear_runtime_caches!()
    for registration in _runtime_cache_registrations()
        registration.clear()
    end
    return nothing
end

# Used by the bounded retention study; ordinary summaries never walk these roots.
_runtime_cache_roots() = [r.roots() for r in _runtime_cache_registrations()]
