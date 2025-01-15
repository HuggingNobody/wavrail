# Numerical tolerance policy

Floating-point comparisons inside the wavrail test suite use a
two-tier tolerance system.

## Tier 1: deterministic feature outputs

For hand-derived expressions (mel filterbank construction, window
functions, framing) we compare against precomputed golden values.
The default tolerance is `atol=1e-6, rtol=1e-6`.

## Tier 2: numerically iterated algorithms

For iterative algorithms (prefix beam search with floating-point
additions, mel filterbank accumulation) we relax the tolerance to
`atol=1e-4, rtol=1e-5`. Larger deviations are documented inline at
the test site.

## Tier 3: trained acoustic model

For the optional torch acoustic model the tolerance widens to
`atol=1e-2, rtol=1e-2` because the model is initialized randomly
within a deterministic seed and may differ across PyTorch versions.

## Updating tolerances

Any widening of tolerances must include a comment linking to the
algorithmic justification. Tightening tolerances requires updating
the corresponding golden file and committing both.
