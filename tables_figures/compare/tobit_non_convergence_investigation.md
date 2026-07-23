# Tobit (censored-Gaussian) model dropped: non-convergence investigation

The design spec proposed fitting two families on the 20,000-row subsample — a
two-limit Tobit/censored-Gaussian model (`brms`, `family = gaussian()`,
`resp_cens()`) and a Beta regression — and comparing them via `loo_compare()`.

**The Tobit model could not be fit reliably.** Across every variant tried, it
produced catastrophic non-convergence: `Rhat` between 2.1 and 3.6 (target
`< 1.01`), and 95–100% of post-warmup draws flagged as divergent transitions
(target: 0, or very close to it). Point estimates were consistently close to
the values a plain (uncensored) Gaussian fit or the full-data OLS refit
produced (e.g. `curvature` ≈ 0.30–0.32 across every attempt), suggesting the
sampler was finding approximately the right region but could not explore it
reliably — the diagnostics cannot be trusted regardless of how plausible the
point estimates look.

**Diagnoses ruled out, in the order tested:**

1. **Sigma prior too tight** (`normal(0.05, 0.02)`, anchored to the published
   formula's noise scale). Ruled out: a plain Gaussian fit (no `cens()`) with
   the identical tight prior converged perfectly (`Rhat` ≈ 1.00, 0
   divergences) — the prior alone is not the problem.
2. **Sigma prior in genuine conflict with the data** (a looser
   `normal(0.05, 0.05)` prior was tried). Made no difference — still ~100%
   divergent.
3. **Poor chain initialization** (`init = "0"`, as used in `brms`'s own
   censored-regression documentation example). Made no difference.
4. **Too few boundary (exact-0/exact-1) observations to identify the
   censoring mechanism** — the original 20,000-row subsample (built by pure
   decile-stratified sampling) happened to carry only 1 exact-`accident_risk
   = 1` row (of 61 in the full 517,754-row training set). Forcing that single
   row to "uncensored" *did* restore clean convergence, which looked like the
   answer — but this only proves a single right-censored observation is
   pathological, not that "enough" censored rows fixes it. `stratified_subsample()`
   was changed to guarantee full inclusion of all boundary rows (see
   `src/features.R`'s `boundary_values` parameter and its test in
   `tests/testthat/test-features.R`) — a legitimate, kept improvement to the
   sampling scheme regardless of this investigation's outcome, since census
   representation of the rows that identify a censoring mechanism is good
   practice on its own. The rebuilt subsample carries all 608 exact-0 and all
   61 exact-1 rows.
5. **Re-tested with the corrected, boundary-rich subsample** (608 + 61
   boundary rows, both the tight and the default/loose `sigma` prior). Still
   catastrophic: `Rhat` up to 3.6, ~100% divergent transitions in every
   variant.

**Conclusion**: something about `brms`'s censored-Gaussian likelihood
geometry, combined with this dataset's scale (~20,000 rows) and censoring
rate (~3.3%), is difficult for the `rstan` backend to sample reliably even
with priors ruled out as the cause. Per explicit user direction (rather than
adding `cmdstanr` as a new dependency to test whether a different backend
resolves it, or spending further time on reparameterization), the Tobit
family is **dropped from this project**. **Beta regression is the sole
model going forward** — it converged cleanly (`Rhat` ≈ 1.00, 0 divergent
transitions) in every fold and every diagnostic test throughout this
investigation, with no tuning required.

This is reported as a genuine finding, not a silent omission: the
mechanistically "correct" model (matching the published original-data
generative process) proved intractable to fit reliably with the tooling
this project committed to: a real, honest limitation, not a hidden one.
