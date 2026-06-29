# Change Plan: Pricing-Engine Software Design

**Repo:** `jhqntdv/pricing-engine` (`main`)
**Scope:** Software-design fixes in `kernel/models/pricing_engines/` and `kernel/models/discretization_schemes/`. Math-model issues are tracked separately.
**Guarantee:** These are refactors, not model changes — correct pricing output must not change. The risk addressed here is the opposite: code that *runs and returns plausible numbers while being wrong*. Every task below therefore ships with an invariant-based test, not just a "does it run" check.

> **Why invariant-based tests:** A wrong Greek, a stale discount curve, or a re-simulated path all still produce a float. Smoke tests pass. Only analytic invariants (put–call parity, American ≥ European, code-path equivalence, call-count assertions) expose these.

---

## Existing test suite — what it covers, and its blind spot

The current `tests/` suite is **stronger on no-arbitrage invariants than it first appears**. Already covered: put–call parity (in `test_financial_relationships.py` and `test_vectorized_payoffs.py`), barrier `DI+DO=Vanilla`, `Asian ≤ Vanilla`, `Lookback ≥ Vanilla`, `Chooser ≥ max(call,put)`, forward-start exact-match to 1e-8, `American ≥ European`, Greek signs, and coupon-solver convergence/speed. Do **not** re-add these.

But the suite has a **systematic blind spot that aligns exactly with the bugs we care about** — new tests must close these, not duplicate the above:

| Gap | Evidence | Consequence |
|-----|----------|-------------|
| **Flat curve everywhere** | `test_vectorized_payoffs` uses constant `r=0.05`; `test_rate_products` uses `get_rate=λt:5.0` | A drift-vs-discount **curve mismatch is invisible** (flat ⇒ forward ≡ spot). This is why the A3 bug can't be caught today. |
| **Payoff tests bypass the real simulator** | `test_vectorized_payoffs.make_paths` generates paths with the **correct exact log-scheme**, then feeds them to `get_discounted_payoff` | The engine's actual (arithmetic-Euler) path generation is never exercised by the payoff tests. |
| **One-sided American bound** | `assert p_am >= p_eu - 0.2` | The American discounting bug biases the price *down toward* European — still passes this bound. |
| **Annual-only rate products** | coupon/swap tests use `frequency=1` | Non-annual coupon handling is untested. |
| **Greeks: presence-only** | greeks test asserts keys exist + `δ∈(0,1)` | No flag-on-vs-getter equivalence; no comparison to analytic Black–Scholes. |
| **`pre_simulated_paths` never passed to American** | no such test | A5 is uncovered. |

**Rule for new tests:** use a **non-flat (steep) curve**, **two-sided/tight** tolerances, and route through the **real engine** path generation — otherwise the new test inherits the same blind spot.

---

## A1 — Single source of truth for Greeks; serve all three usage modes

**Files:** `mc_pricing_engine.py`

**Problem.** The finite-difference logic exists twice: `get_result` inlines an optimized delta/gamma (shared up/down simulations, `epsilon_spot = 1.0`), while the standalone `get_delta` / `get_gamma` re-implement it *unshared* and are never called by `get_result`. Two implementations drift apart; the standalone getters are dead and unoptimized.

**Usage modes to support:**
1. Price **with** Greeks flag on → Greeks precomputed and stored in `PricingResults`; user retrieves via getters.
2. Price **without** Greeks → no Greek computation, no extra simulation.
3. Greeks **only** → user wants risk numbers, not the headline price.

**Change.**
- Extract one private helper per Greek as the only place the math lives: `_delta_gamma(derivative, base_price)` (returns both, sharing the up/down/base simulations), `_vega(...)`, `_rho(...)`, `_theta(...)`.
- `get_result` calls `_compute_greeks(...)` (which calls the helpers) only when the flag is on, and stores results.
- Public getters (`get_delta`, etc.) call the **same** helpers → mode 3 reuses mode 1's code.
- Pull the spot bump into a `_spot_eps(derivative)` helper so it is defined once (also lets us later make it relative instead of a hardcoded `1.0`).

**Reasoning.** One implementation = the inline/standalone versions can never disagree; the shared-simulation optimization is available to all three modes instead of only the flag-on path.

**Tests & verification.** (Existing `test_mc_pricing_engine_greeks_optimization` only checks the Greeks *exist* and `δ∈(0,1)` — and its comment actively blesses the inline duplication. None of the below exist today.)
- `test_greeks_single_source`: compute Greeks via `get_result(flag=on)` and via the standalone getters with the **same seed**; assert they are **bit-for-bit equal**. (If they differ, two implementations still exist — this is the test that pins A1.)
- `test_no_greeks_no_work`: with the flag off, assert `PricingResults` Greeks are unset/`None` **and** that `EulerScheme.simulate_paths` is called the minimal number of times (spy the call count — proves no hidden Greek simulation).
- `test_delta_gamma_share_simulations`: spy `simulate_paths`; assert delta+gamma together trigger 3 simulations (up/down/base), not 6.
- `test_finite_difference_vs_analytic`: for a vanilla European under Black–Scholes, assert MC delta/gamma/vega match closed-form Black–Scholes Greeks within MC tolerance (CRN makes this tight). The existing suite never compares a Greek to an analytic value — this catches a silently mis-wired bump or scale.

---

## A3 — Make drift and discounting provably share one curve

**Files:** `mc_pricing_engine.py`, `american_mc_pricing_engine.py`, and (Option Full only) every product's `get_discounted_payoff`.

**Problem.** Drift is built correctly and product-independently from one RFR curve in `get_stochastic_process`. But **discounting is delegated into each product's `get_discounted_payoff`**, so the engine cannot *guarantee* discounting uses the same curve as the drift. This already bit the codebase: the `_get_price` comment *"Fix: Allow passing a specific market … (resolves Rho calculation error)"* exists because, when bumping the curve for Rho, the drift moved but the product's internal discounting still used the unbumped `self.market`. The fix threaded `current_market` through by hand — but the **silent fallback `if current_market is None: current_market = self.market` means a forgotten curve produces a wrong number instead of an error**, and nothing prevents a future product from using a different rate (spot vs stepwise forwards) or convention (`exp(-rt)` vs `(1+r)^-t`).

> **Constraint discovered from the test suite:** `get_discounted_payoff` is the *contract* and is pinned tightly — `test_vectorized_payoffs.py` asserts discounted payoffs to **1e-8** (e.g. barrier `(120-100)*exp(-rT)`, forward-start `*get_discount_factor(T)`). Any change that makes products return *undiscounted* payoffs **breaks that entire file.** This forces a choice between two options.

### Option A3-Light — recommended
Keep discounting in products; **remove the silent fallback** and make the curve/market a required, explicitly-passed argument. A missing or unbumped curve then fails loudly instead of returning a silently-wrong price.
- Low risk; preserves `test_vectorized_payoffs.py` as-is.
- Fixes the actual root cause of the Rho bug (the `None`→`self.market` fallback).
- Does **not** structurally prevent a future product from choosing a different discount convention — that stays a code-review responsibility.

### Option A3-Full — cleaner, more invasive
Rename `get_discounted_payoff` → `get_payoff` (undiscounted, returning payoff + payoff time(s)); discount once in the engine using the same curve it built drift from.
- Structurally guarantees one curve and one as-of date — eliminates the bug class entirely.
- **Requires rewriting `test_vectorized_payoffs.py`** (un-discount every expected value) and touching every product. Higher risk; do it behind the invariant tests below.

**Reasoning.** A3-Light removes the silent-failure mode cheaply; A3-Full removes the *possibility* of inconsistency but pays for it in churn against the most valuable existing test file. Pick based on appetite for the rewrite.

**Tests & verification** — the existing parity tests use a **flat** curve and therefore **cannot** detect this bug. New tests must use a steep curve and the real engine:
- `test_parity_steep_curve`: put–call parity `C - P ≈ S0 - K*DF(T)` under a **non-flat term structure**, priced through the **real engine** (not hand-rolled paths), tight two-sided tolerance. Holds only if drift and discounting use the same curve.
- `test_rho_bumps_both_legs`: regression for the original bug — bump a **steep** curve and assert Rho matches a curve-consistent central difference, i.e. discounting moved with the drift.
- `test_missing_curve_raises` (Light): calling the price path without an explicit curve must **raise**, proving the silent `self.market` fallback is gone.
- `test_discount_convention`: assert engine discounting equals the product of step DFs (continuous `exp(-rt)`) under a steep curve — proves no leg silently used annual compounding `(1+r)^-t`.
- (Full only) rewrite `test_vectorized_payoffs.py` to assert **undiscounted** payoffs, plus one engine-level test that the engine reproduces the old discounted values — proving discounting merely moved, it didn't change.

---

## A4 — Remove dead code in the Euler scheme

**Files:** `euler_scheme.py` (`_simulate_one_factor`, `_simulate_two_factor`)

**Change.** Delete the unused `t = i * dt` line in both loops. (Optional: rename the `get_drift(i, x)` argument from `t` to `step_index` to stop implying it is a time.)

**Reasoning.** `t` is computed and never used; the name falsely implies continuous time when an integer step index is passed.

**Tests & verification.**
- No new test required — covered by existing simulation/pricing tests, which must remain green (proves the deletion changed nothing).

---

## A5 — American engine ignores `pre_simulated_paths` (Liskov violation)

**Files:** `american_mc_pricing_engine.py` (`_get_price`)

**Problem.** The override accepts `pre_simulated_paths` in its signature but **never references it** — it unconditionally re-simulates. The parent `MCPricingEngine._get_price` honors the parameter. This is currently latent (the coupon solver in `CallableMCPricingEngine` inherits the parent, not the American engine), but it is a real trap: a future callable American / Bermudan autocall routed through `get_coupon` would **re-simulate every bisection iteration, reintroducing Monte Carlo noise and breaking the monotonicity the root-finder relies on** — hard to diagnose because the signature advertises support.

**Change.** Mirror the parent: use `pre_simulated_paths` when provided, otherwise simulate.
```python
if pre_simulated_paths is not None:
    paths = pre_simulated_paths
else:
    paths = EulerScheme().simulate_paths(stochastic_process, self.nb_paths, self.random_seed)
```

**Reasoning.** Restores the parent's contract; makes the path-reuse optimization actually work for early-exercise products.

**Tests & verification** (note: a naive "same result twice" test will NOT catch this, because re-simulating with the same seed yields the same paths — the bug is invisible to it):
- `test_american_honors_presimulated_paths`: spy `EulerScheme.simulate_paths`; call American `_get_price` **with** `pre_simulated_paths` and assert `simulate_paths` is called **0 times**. This is the decisive check.
- `test_american_uses_provided_paths_not_seed`: pass a hand-crafted `pre_simulated_paths` array that deliberately **differs** from what the seed would produce; assert the returned price reflects the provided paths (e.g. a constructed path set with a known LSM answer), not the re-simulated ones.
- `test_callable_american_coupon_converges` (integration, guards the future case): run the coupon solver on a Bermudan/early-exercise product and assert the bisection converges and is **stable across reruns** (no MC jitter between iterations).

---

## Suggested order & risk
1. **A4** (trivial, zero-risk) and **A5** (small, isolated) first.
2. **A1** next — self-contained within `mc_pricing_engine.py`; land `test_greeks_single_source` before refactoring so it pins behavior.
3. **A3** — decide **Light vs Full** first. A3-Light is a small, low-risk change; A3-Full touches every product and requires rewriting `test_vectorized_payoffs.py`, so stage it behind the steep-curve invariant tests.

**Overarching guard.** The existing suite must stay green — but note it is a guard *only for behavior it actually exercises*. As audited above, it uses **flat curves, one-sided bounds, and bypasses the real simulator**, so "all green" does **not** by itself prove A3 (or B1-1/B1-3/B1-4) is correct. The steep-curve / two-sided / real-engine tests above are what close that gap. For A1 and A3, write the new invariant test **first** (it should pass on the current code where the current code is correct, pinning the contract), then refactor.

> **Caution — tests may pin wrong behavior.** Before trusting a green run, confirm no existing test asserts a value the math review flagged as suspect (e.g. an American price under the one-sided `>= p_eu - 0.2` bound, or an annual-only coupon-bond par price). A test that locks in a bug will *resist* the correct fix.
