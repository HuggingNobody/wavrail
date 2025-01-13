# API conventions

All public symbols live under `wavrail.*`. Symbols prefixed with an
underscore are private.

## Dataclasses over tuples

Configuration and result types are `frozen=True` dataclasses with
typed fields. Equality and hashing come for free; tests rely on
them for golden comparisons.

## Strict bounds

Every public function validates its inputs and raises a subclass of
`wavrail.errors.WavrailError` on failure. Validation messages name
the offending argument.

## Determinism

Functions that consume randomness accept an explicit `seed` and pass
it to a private `numpy.random.Generator` instance. The default seed
is `0`; reproducible tests rely on this default.

## Numerical tolerance

Floating-point comparisons in tests use a documented tolerance
policy. See `docs/numerical_tolerance.md`.

## Versioning

The package follows strict semver. Breaking changes require a
minor bump. Internal refactors do not bump the minor version.
