# Two-Factor (Heston) Simulation & LSM Regression Upgrade Plan

> [!IMPORTANT]
> **Depends on `change_plan_euler_log_scheme.md` — merge that FIRST.** Both plans edit `EulerScheme._simulate_two_factor` and `tests/test_models.py`; see that plan's "Cross-Plan Coordination" section for the exact stitch. `IMPLEMENTATION_MASTER_ROADMAP.md` is outdated (wrong module paths, mixes the two plans) — treat these two `change_plan_*.md` files as the single source of truth.

## 1. Background & Motivation

Currently, the `EulerScheme` elegantly abstracts the underlying stochastic process (Black-Scholes vs. Heston) by returning only the simulated spot price paths (`paths[:, :, 0]`) as a 2D numpy array.

While this ensures perfect backward compatibility for standard derivatives (which only care about the spot price), it creates a critical blind spot for **American Options (LSM Engine)**. To make an optimal early exercise decision under stochastic volatility, the Longstaff-Schwartz regression must know the instantaneous variance $v_t$. Because the variance paths are discarded by `EulerScheme`, the American engine is currently forced to regress only on the spot price, leading to sub-optimal exercise boundaries and underpricing.

We will resolve this using **Option A (The `SimulationResult` Object)**, which passes the variance state to the engines that need it, without breaking the derivative payoff functions.

> **This is NOT an easy change.** The return type of `EulerScheme.simulate_paths` is consumed in **4 production call-sites** and **4 test files**. Changing it from `np.ndarray` to an object touches the whole simulation → pricing → payoff pipeline. This document is structured as **6 sequential phases** (Section 5) precisely so that each phase leaves the test suite green before the next begins. Do **not** attempt the whole change in one commit.
>
> **Verified call-site inventory (do not trust memory — this was grepped against the current tree):**
> | # | File | Method | Line | Addressed in |
> |---|------|--------|------|--------------|
> | 1 | `kernel/models/pricing_engines/mc_pricing_engine.py` | `_get_price` | ~194 | §2.C.1 |
> | 2 | `kernel/models/pricing_engines/mc_pricing_engine.py` | `_theta` (CRN branch) | ~307 | §2.C.4 ⚠️ **easy to miss** |
> | 3 | `kernel/models/pricing_engines/callable_mc_pricing_engine.py` | `get_coupon` | ~70 | §2.C.2 |
> | 4 | `kernel/models/pricing_engines/american_mc_pricing_engine.py` | `_get_price` | ~57 | §2.C.3 |
>
> Test files that read the return value as a raw array (must be fixed in Phase 2): `test_models.py`, `test_log_euler.py` (**8 call-sites**), plus the indirect regression guards `test_theta_crn.py`, `test_american_engine.py`, `test_mc_engine_greeks.py`. See §3.0.

---

## 2. Implementation Plan (Option A: `SimulationResult`)

### A. Define the Data Container
Create a new dataclass to hold simulation results comprehensively.

**Target File:** `kernel/models/discretization_schemes/simulation_result.py` (New File)
```python
from dataclasses import dataclass
import numpy as np
from typing import Optional

@dataclass
class SimulationResult:
    spot_paths: np.ndarray
    variance_paths: Optional[np.ndarray] = None
```

> **Also re-export it** so downstream imports are clean. `kernel/models/discretization_schemes/__init__.py` is currently empty — add:
> ```python
> from .simulation_result import SimulationResult
> from .euler_scheme import EulerScheme
> ```

### B. Update `EulerScheme`
Modify `simulate_paths` to return the `SimulationResult` object instead of a raw numpy array. **Also update the return type annotation** (currently `-> np.ndarray`).

**Target File:** `kernel/models/discretization_schemes/euler_scheme.py`
- `simulate_paths` return annotation: `-> SimulationResult`
- `_simulate_one_factor` returns: `SimulationResult(spot_paths=paths)`
- `_simulate_two_factor`:
  - **Keep computing the full `(nb_paths, nb_steps+1, 2)` array internally** (do NOT slice `[:, :, 0]` before returning).
  - Return: `SimulationResult(spot_paths=paths[:, :, 0], variance_paths=paths[:, :, 1])`

### C. Update Pricing Engines (The API Bridge)
The pricing engines must be updated to unpack the `SimulationResult` before passing the spot paths down to the derivatives. This protects all derivative payoff logic from breaking.

#### C.1 `MCPricingEngine._get_price`
**File:** `kernel/models/pricing_engines/mc_pricing_engine.py` (line ~190-194)

The tricky part: `pre_simulated_paths` must **still accept a raw `np.ndarray`** (the Callable engine and existing tests pass raw arrays), while a *fresh* simulation now yields a `SimulationResult`. Handle both:

```python
if pre_simulated_paths is not None:
    # Backward-compat: callers may pass a raw array OR a SimulationResult
    price_paths = getattr(pre_simulated_paths, "spot_paths", pre_simulated_paths)
else:
    scheme = EulerScheme()
    sim_result = scheme.simulate_paths(process=stochastic_process, nb_paths=self.nb_paths, seed=self.random_seed)
    price_paths = sim_result.spot_paths
# ... rest of the code is unchanged (get_discounted_payoff receives price_paths)
```

#### C.2 `CallableMCPricingEngine.get_coupon`
**File:** `kernel/models/pricing_engines/callable_mc_pricing_engine.py` (line ~69)

`simulate_paths` now returns a `SimulationResult`, but it is fed into `_get_price(..., pre_simulated_paths=...)` and ultimately into `get_discounted_payoff`. **Unpack to `spot_paths` immediately** so the raw array flows downstream:

```python
scheme = EulerScheme()
sim_result = scheme.simulate_paths(process, self.nb_paths, self.random_seed)
pre_simulated_paths = sim_result.spot_paths   # unpack early
# ... bisection loop unchanged
```

#### C.3 `AmericanMCPricingEngine._get_price` (The core fix)
**File:** `kernel/models/pricing_engines/american_mc_pricing_engine.py`

1. **Unpack (mirror C.1 logic):**
   ```python
   if pre_simulated_paths is not None:
       paths = getattr(pre_simulated_paths, "spot_paths", pre_simulated_paths)
       var_paths = getattr(pre_simulated_paths, "variance_paths", None)
   else:
       scheme = EulerScheme()
       sim_result = scheme.simulate_paths(process=stochastic_process, nb_paths=self.nb_paths, seed=self.random_seed)
       paths = sim_result.spot_paths
       var_paths = sim_result.variance_paths
   ```
   > Note: when `pre_simulated_paths` is a raw array (e.g. from unit tests), `var_paths` is `None` and the engine safely falls back to the 1-D basis.

2. **Regression basis (inside the backward-induction loop, at step `t`):** build the design matrix from the **in-the-money subset only**, applying the *same* `in_money` mask and time index `t` to both spot and variance.

   - `normalizer = derivative.strike` (moneyness scaling for spot, already present).
   - **Normalize variance too** to keep the design matrix well-conditioned (see Section 4, "Numerical Conditioning"). Use `v_norm = v / v_ref` where `v_ref = getattr(stochastic_process, "theta", None) or getattr(stochastic_process, "v0", 1.0)`.

   ```python
   normalized = paths[in_money, t] / normalizer
   basis = [np.ones(np.sum(in_money)), normalized, normalized ** 2]   # 1-D fallback

   if var_paths is not None:
       v_ref = getattr(stochastic_process, "theta", None) or getattr(stochastic_process, "v0", 1.0)
       v_in = var_paths[in_money, t] / v_ref            # SAME mask + time index as spot
       basis += [v_in, v_in ** 2, normalized * v_in]    # 5-term expanded 2-D basis

   x_matrix = np.column_stack(basis)
   coeff, _, _, _ = np.linalg.lstsq(x_matrix, y_vector, rcond=None)
   cont_val = x_matrix @ coeff                          # generic: works for 3 or 6 columns
   ```
   Using `x_matrix @ coeff` (instead of hardcoding `coeff[0] + coeff[1]*x + ...`) makes the code agnostic to the number of basis terms, so the 1-D and 2-D paths share one code line.

> **Mapping of the mathematical basis** (all normalized): $X_1 = S_t/K$, $X_2 = (S_t/K)^2$, $X_3 = v_t/v_{ref}$, $X_4 = (v_t/v_{ref})^2$, $X_5 = (S_t/K)\cdot(v_t/v_{ref})$, plus the intercept. This is the **complete second-order polynomial in two variables** (6 terms) — the textbook 2-factor Longstaff-Schwartz regression.

#### C.4 `MCPricingEngine._theta` (CRN branch — **the call-site everyone forgets**)
**File:** `kernel/models/pricing_engines/mc_pricing_engine.py` (line ~305-312)

`_theta`'s Common-Random-Numbers branch does **NOT** route through `_get_price` for its *base* simulation. It calls `simulate_paths` directly and then **slices the result** to drop the last time column:

```python
# CURRENT CODE (will break in Phase 2):
process = self.get_stochastic_process(derivative, market)
scheme = EulerScheme()
base_paths = scheme.simulate_paths(process, self.nb_paths, self.random_seed)
bumped_paths = base_paths[:, :-1]     # <-- SimulationResult is NOT subscriptable → TypeError
```

After Phase 2, `base_paths` is a `SimulationResult`, so `base_paths[:, :-1]` raises `TypeError: 'SimulationResult' object is not subscriptable`. **Fix — unpack the spot array on the same line:**

```python
base_paths = scheme.simulate_paths(process, self.nb_paths, self.random_seed).spot_paths
bumped_paths = base_paths[:, :-1]     # now a real ndarray again
```

> **This branch fires for every Heston theta and for every non-vanilla theta under Black-Scholes** (barrier, autocall) — see the `elif self.model.name == "HESTON" or not is_vanilla:` guard. The `bumped_paths` array is then passed back as `pre_simulated_paths`, which C.1 already handles, so **only this single unpack line is needed here.**
> **⚠️ This directly contradicts the claim in §4 that "Greeks are unaffected."** That claim was wrong; see the corrected §4. `test_theta_crn.py` is the regression guard that catches this (§3.0).

---

## 3. Product Compatibility & Test Coverage Plan

**ALL** derivatives in the codebase are fundamentally compatible with a Heston process, because the payoff of equity derivatives depends entirely on the spot price trajectory, which the Heston process provides. The engine-layer unpacking (Section 2.C) guarantees payoffs always receive a raw spot array.

We must **(a) fix the existing tests that consume the return value as a raw array**, and **(b) add a dedicated integration + math test file.**

### 3.0 Existing tests that WILL break (must be fixed — do not skip)

Line numbers below were verified against the current tree.

| File | Lines | Problem | Fix |
|------|-------|---------|-----|
| `tests/test_models.py` | 95, 118, 141 | `test_euler_one_factor`, `test_euler_two_factor`, `test_euler_list_drift_and_typing` do `paths = scheme.simulate_paths(...)` then read `paths.shape`, `paths[:, 0]`, `np.all(paths > 0)` directly | In all three, append `.spot_paths`: `paths = scheme.simulate_paths(...).spot_paths`. **No assertion logic changes** — the shape `(nb_paths, nb_steps+1)` and positivity asserts stay identical. |
| `tests/test_log_euler.py` | 15, 29, 42, 56, 73, 87, 103, 120 | **ALL 8 tests** do `paths = scheme.simulate_paths(...)` then consume it as a raw array (`paths[:, -1]`, `np.log(paths[:, -1])`, `np.all(paths > 0)`, `np.any(paths <= 0)`) | Append `.spot_paths` at each of the 8 call-sites: `paths = scheme.simulate_paths(...).spot_paths`. No assertion logic changes. **⚠️ This file was entirely absent from the earlier draft — it is the single most affected test file. Do not skip it.** |
| `tests/test_theta_crn.py` | *(indirect — no edit)* | `test_theta_crn_correctness_vs_analytic_bs` and `test_theta_crn_stability_barrier` price a barrier under Black-Scholes → they force the **CRN theta branch** → they hit the C.4 slice `base_paths[:, :-1]` | **No test change needed — but C.4 MUST be applied.** These two tests are the regression guard for C.4; if C.4 is skipped they fail with `TypeError`. Treat a green run here as proof C.4 landed. |
| `tests/test_american_engine.py` | 49, 65 | Passes **raw** `np.ones(...)` as `pre_simulated_paths` | No change needed IF C.3 keeps raw-array support (`getattr(..., "spot_paths", ...)`). This is a **regression guard** — it must keep passing (and confirms `var_paths is None` → 1-D fallback). |
| `tests/test_mc_engine_greeks.py` | 97 | Mocks `simulate_paths`, asserts `call_count` only | No change needed (return value unused). Regression guard. |

### 3.1 New File: `tests/test_heston_integration.py`

Tests 1–7 are ordered from "cheap sanity checks that isolate bugs" to "expensive convergence checks", so a failure points you to the layer at fault. **Test 8 is a targeted regression guard for the C.4 theta fix** and is grouped at the end regardless of cost.

#### Test 1 — Heston degenerates to Black-Scholes (structural sanity) 🔢
- **Goal:** The cleanest possible check that the two-factor path construction + unpacking are correct, **exercised through the pricing engine** (not the raw scheme).
- **⚠️ Avoid duplicating an existing test.** `tests/test_log_euler.py::test_heston_degenerates_to_bs` (line 95) already checks the *raw-scheme* degeneracy (`sigma=0, v0=theta`) directly against the BS closed form. **Do NOT copy it.** This new Test 1 must go one layer higher: price via `MCPricingEngine` with `Model.HESTON` (so `get_stochastic_process` + `SimulationResult` unpacking are on the code path), not via `EulerScheme` directly.
- **Setup:** `Model.HESTON`, engine params forced to vol-of-vol `sigma = 0` and `v0 = theta` (variance pinned at $v_0$). `S0=K=100, T=1, r=0.05, v0=0.04`.
- **Check:** A `EuropeanCallOption` priced by the engine MUST match the Black-Scholes closed form with $\sigma_{BS} = \sqrt{v_0}$, within `3 * std_dev` (use the engine's returned `std_dev`, not a hardcoded atol).
- **Why it matters:** Isolates the two-factor discretization + engine unpacking from stochastic-vol noise. If this fails but `test_heston_degenerates_to_bs` passes, the bug is in the **engine `SimulationResult` unpacking (C.1)**, not the scheme.

#### Test 2 — Put-Call Parity under Heston (no benchmark needed) 🔢
- **Goal:** Catch drift, discounting, and unpacking errors without needing any external analytical price.
- **Check:** With full stochastic Heston params, $C - P = S_0 - K e^{-rT}$ (spot form) within `3 * combined_std_dev`.
- **Why it matters:** This identity holds path-by-path regardless of the vol model, so it is the most robust regression test for the pricing pipeline.

#### Test 3 — Vanilla European vs. Semi-Closed-Form (convergence) 🔢
- **Goal:** Verify the Heston MC converges to the true price.
- **Setup (Feller-satisfying):** $S_0=100, K=100, T=1.0, r=0.05, v_0=0.04, \kappa=2.0, \theta=0.04, \sigma=0.3, \rho=-0.5$.
  - Feller check: $2\kappa\theta = 0.16 > \sigma^2 = 0.09$ ✔ (variance stays away from zero; Euler bias is manageable).
- **Check (statistical, not hardcoded):** Compare the MC price against a benchmark from the Heston characteristic-function integration. Assert the MC price is within `3 * std_dev` of the benchmark. **Use `nb_steps >= 250`** — full-truncation Euler for Heston has a discretization bias that a naive `atol=0.05` at low step-count will trip, producing a false failure that looks like a code bug.
- **Note:** If a semi-closed-form implementation is not available in the repo, hardcode the benchmark value in the test with a comment citing the parameters and the reference implementation used to produce it.

#### Test 4 — American Put under Heston: 2-D regression runs, is sane, and is accurate (the primary fix) 🔢
- **Goal:** Verify `AmericanMCPricingEngine` unpacks `variance_paths`, actually runs the 6-term regression, and produces a price consistent with a trusted benchmark.
- **How to force the 2-D path (critical):** you must drive the engine so `var_paths is not None`. Two options:
  - **(preferred)** call through `get_result` / `_get_price` with **no** `pre_simulated_paths` and a `HestonProcess`, so the engine simulates internally and `sim_result.variance_paths` is populated; **or**
  - construct a `SimulationResult(spot_paths=..., variance_paths=...)` by hand and pass it as `pre_simulated_paths` (this also unit-tests the C.3 `getattr(..., "variance_paths", None)` unpack).
  - **Add an explicit assertion that the 6-column branch was taken** (e.g. monkeypatch `np.linalg.lstsq` to record the design-matrix column count, and assert it equals 6 at least once). Without this, a silent fallback to the 1-D basis would let the test pass while testing nothing.
- **⚠️ Do NOT assert `price_2d > price_1d` strictly.** LSM on in-sample paths is a low-biased estimator; adding basis functions reduces sub-optimality but is **not monotonic** and is easily masked by MC noise. A strict `>` assertion is mathematically unjustified and will be flaky.
- **Accuracy check (primary):**
  1. Hardcode a high-accuracy Heston American-put benchmark **with a cited source** (a published reference value, or a 2-D finite-difference PDE result produced offline). Params: reuse Test 3's Feller-satisfying set. Put a comment giving the exact params and where the number came from.
  2. Assert `abs(price_2d - bench) <= tol` (choose `tol` from the engine's returned `std_dev`, e.g. `3 * std_dev`, plus a small Euler-bias allowance).
  3. Assert the 2-D price is **no further from truth than 1-D**: `abs(price_2d - bench) <= abs(price_1d - bench) + tol`.
- **❌ Out-of-sample ("regress on seed A, value on seed B") guard is NOT included.** The current `_get_price` regresses and values on the *same* path set in a single backward pass and exposes **no API** to separate the two. Implementing a genuine out-of-sample comparison requires an engine extension (persist regression coefficients, replay on a fresh path set) that is **out of scope** for this plan. Track it as a future enhancement; do not write a test that pretends the current engine supports it.
- **Lower-bound sanity (cheap, always include):** `american_put_price >= european_put_price - tol` (early-exercise premium ≥ 0) and `american_put_price >= intrinsic_at_t0`.

#### Test 5 — Black-Scholes American unchanged (regression guard) 🔢
- **Goal:** Prove the refactor did not alter the existing 1-D behavior.
- **Check:** Price a Black-Scholes `AmericanPutOption` (one-factor → `var_paths is None`) with a fixed seed and assert the price equals the pre-refactor value (record it as a golden number). This confirms the `var_paths is None` fallback path is truly a no-op relative to today.

#### Test 6 — Path-Dependent & Barrier Options complete under Heston
- **Goal:** Prove changing the `EulerScheme` return type did not break downstream exotic payoffs.
- **Check:** Run `AsianCallOption` and `UpAndOutCallOption` through `MCPricingEngine` with `Model.HESTON`. Assert the engine completes and returns a finite, non-negative price. Bonus: `UpAndOut <= Vanilla` (barrier discount holds).

#### Test 7 — Structured Products (Autocalls) complete under Heston
- **Goal:** Prove `CallableMCPricingEngine.get_coupon` handles `SimulationResult` correctly during coupon solving (this is where the C.2 early-unpack is exercised).
- **⚠️ Mind the solver:** `get_coupon`'s **default `method="analytical"`** (a linear solver), NOT bisection. So:
  - **Analytical path (default):** assert the returned coupon is finite and in `(0, 50)`, and that pricing at that coupon lands within `epsilon` of `target_price`.
  - **Bisection path:** if you want to assert "converges without hitting `max_iter` at a bound", you must **explicitly pass `method="bisection"`**; then assert the coupon is strictly inside `(lower_bound, upper_bound)` = `(0, 50)` and the round-trip price is within `epsilon`. Do not claim bisection behavior while calling the analytical default.
- **Check:** Run both under a `HestonProcess`. Both must return a finite, non-zero coupon.

#### Test 8 — Heston Theta runs (the C.4 regression guard) 🔢
- **Goal:** Directly prove the C.4 `_theta` unpack fix works — this is the exact code path that breaks if C.4 is skipped, and it is **not** covered by the vanilla-BS analytical theta tests.
- **Setup:** `Model.HESTON`, `compute_greeks=True`, `nb_steps >= 2` (CRN theta needs ≥ 2 steps or it returns 0.0 by design), a `EuropeanCallOption`. A European vanilla under Heston still routes to the CRN branch because `model.name == "HESTON"`.
- **Checks:**
  1. `get_result(...)` **completes without `TypeError`** and populates `greeks["theta"]`.
  2. `theta` is finite and has the correct sign (negative for a long ATM call under positive rates).
  3. Bonus: also run a `UpAndOutCallOption` under Heston to cover the `not is_vanilla` sub-branch.
- **Why it matters:** Without C.4, `get_result` raises `TypeError: 'SimulationResult' object is not subscriptable` at `mc_pricing_engine.py:~307`. This test turns that latent crash into an explicit, named failure.

---

## 4. Cross-Cutting Concerns & Design Notes

### Numerical Conditioning (why variance must be normalized)
The existing 1-D code normalizes spot by strike so $S/K \approx 1$, giving `np.linalg.lstsq` a well-conditioned design matrix. Raw variance breaks this:

| Column | Raw scale | Normalized scale ($v/\theta$) |
|--------|-----------|-------------------------------|
| $S/K$ | ~1 | ~1 |
| $v_t$ | ~0.04 | ~1 |
| $v_t^2$ | ~0.0016 | ~1 |
| $(S/K)\,v_t$ | ~0.04 | ~1 |

A 2–3 order-of-magnitude spread across columns yields an ill-conditioned matrix and unstable coefficients — which can make the 2-D price *worse* than 1-D. **Normalizing variance by `theta` (or `v0`) is required, not optional.**

### `pre_simulated_paths` polymorphism
Both `MCPricingEngine._get_price` and `AmericanMCPricingEngine._get_price` must accept **either** a raw `np.ndarray` **or** a `SimulationResult` for `pre_simulated_paths` (use `getattr(x, "spot_paths", x)`). This preserves the existing unit tests that inject hand-crafted arrays and keeps the Callable engine flow simple.

### Feller condition
For the benchmark params in Test 3, $2\kappa\theta > \sigma^2$ holds. Also verify the **hardcoded engine defaults** in `mc_pricing_engine.get_stochastic_process` ($\kappa=8.1471, \theta=0.0736, \sigma=0.3905$): $2\kappa\theta = 1.199 > 0.152$ ✔. Add an assertion/comment in the test so a future parameter change that violates Feller (causing frequent variance flooring) is caught.

### Greeks — MOSTLY inherit the unpacking, but `_theta` does NOT (corrected)
`_delta_gamma`, `_vega`, and `_rho` re-price exclusively through `_get_price`, so they inherit the C.1 unpacking automatically — no separate changes needed.

**`_theta` is the exception.** Its CRN branch calls `simulate_paths` **directly** and slices the return value (`base_paths[:, :-1]`) before ever reaching `_get_price`. It therefore **requires the dedicated fix in §2.C.4** and will raise `TypeError` in Phase 2 if that fix is missed. (An earlier draft of this document incorrectly stated "Greeks are unaffected / no separate changes needed" — that was wrong; §2.C.4 and Test 8 exist precisely to cover this.)

Smoke-test greeks under Heston explicitly via **Test 8**, not just "indirectly via Test 3" — Test 3 only trips the theta path if it enables `compute_greeks`, which it need not.

---

## 5. Recommended Execution Phases

Execute **strictly in order**. Each phase must leave the **entire existing test suite green** before starting the next. This is the safest ordering because it introduces the new type behind a fully backward-compatible facade first, then migrates consumers one at a time.

### Phase 0 — Prerequisite
- [ ] Confirm the core Log-Euler and Product Payoff fixes (Phase 1–5 of the previous plan) are merged to `main`. This work touches the same engine files; doing it after avoids merge conflicts.
- [ ] Run the full suite and record the current green baseline (and the golden American-BS price for Test 5).

### Phase 1 — Introduce `SimulationResult` (no behavior change)
- [ ] Create `simulation_result.py` with the dataclass.
- [ ] Add a unit test constructing a `SimulationResult` (spot only, and spot+variance).
- [ ] **Do not touch `EulerScheme` yet.** Suite stays green trivially.

### Phase 2 — Switch `EulerScheme` to return `SimulationResult` + fix its direct consumers
This is the "breaking" phase; keep it tightly scoped. **All 4 production call-sites and both array-consuming test files must change together, or the suite will not be green.**
- [ ] Create `simulation_result.py` (if not done in Phase 1) and re-export from `discretization_schemes/__init__.py` (§2.A).
- [ ] Update `_simulate_one_factor` (return `SimulationResult(spot_paths=paths)`), `_simulate_two_factor` (**stop slicing `[:,:,0]` away** — return `SimulationResult(spot_paths=paths[:,:,0], variance_paths=paths[:,:,1])`), and the `simulate_paths` return annotation.
- [ ] **Call-site 1 —** `MCPricingEngine._get_price` (C.1): unpack `.spot_paths`, keep raw-array support for `pre_simulated_paths`.
- [ ] **Call-site 2 —** `MCPricingEngine._theta` (C.4): change line ~307 to `... .spot_paths` **before** the `base_paths[:, :-1]` slice. ⚠️ **Do not skip — this is the one the earlier draft missed.**
- [ ] **Call-site 3 —** `CallableMCPricingEngine.get_coupon` (C.2): unpack `.spot_paths` early.
- [ ] **Call-site 4 —** `AmericanMCPricingEngine._get_price` (C.3): unpack spot **and** variance, but **keep the 1-D basis for now** (do not add variance terms yet — that is Phase 3).
- [ ] **Fix `tests/test_models.py`** (3 sites: lines 95, 118, 141) → append `.spot_paths`.
- [ ] **Fix `tests/test_log_euler.py`** (8 sites: lines 15, 29, 42, 56, 73, 87, 103, 120) → append `.spot_paths`. ⚠️ **Do not skip this file.**
- [ ] Confirm the indirect guards still pass unchanged: `tests/test_theta_crn.py` (proves C.4 landed), `tests/test_american_engine.py`, `tests/test_mc_engine_greeks.py`.
- [ ] Run suite → must be green. At this point behavior is identical to before; only the plumbing changed.

### Phase 3 — Add the 2-D LSM regression (the actual feature)
- [ ] Implement the expanded, variance-normalized basis in `AmericanMCPricingEngine._get_price` (C.3 step 2), guarded by `if var_paths is not None`.
- [ ] Verify Black-Scholes American price is unchanged (Test 5 golden value) — the guard must be a no-op for one-factor.
- [ ] Run suite → green.

### Phase 4 — Math & integration test suite
- [ ] Add `tests/test_heston_integration.py` with Tests 1–8 (Section 3.1). **Note Test 1 must go through the engine (not duplicate `test_log_euler.py::test_heston_degenerates_to_bs`), and Test 8 is the C.4 theta guard.**
- [ ] Tune tolerances using returned `std_dev` (statistical bands), not hardcoded atol.
- [ ] Ensure Test 3/4 use `nb_steps >= 250`.
- [ ] Test 4: assert the 6-column design matrix is actually used (guard against silent 1-D fallback). **Do not** attempt the out-of-sample regress/value split — the engine has no API for it (see Test 4 note).

### Phase 5 — Hardening & docs
- [ ] Add the Feller-condition assertion/comment.
- [ ] Confirm Greeks run under Heston — this is **Test 8** (delta/gamma/vega/rho inherit C.1; theta needs C.4). Not just an indirect check.
- [ ] Update docstrings/return annotations; note the `pre_simulated_paths` polymorphism.
- [ ] Final full-suite run + a manual Heston American vs. European sanity print.

---

## 6. Rollback / Risk Notes
- The riskiest single step is **Phase 2** (return-type change). If anything downstream unexpectedly consumes the raw array, it surfaces here. Mitigation: the `getattr(x, "spot_paths", x)` polymorphism means even a missed call-site that receives a `SimulationResult` where an array was expected fails loudly and locally, not silently. **The one place polymorphism does NOT save you is `_theta` (C.4), which slices with `[:, :-1]` directly — that raises `TypeError` immediately, which is why `test_theta_crn.py` is called out as the explicit guard.**
- Keep Phases 2 and 3 in **separate commits** so a bisect can distinguish "plumbing broke" from "regression math changed the numbers".
- The only numbers that are *expected* to change across this whole change are **Heston American option prices** (Phase 3). Every other product/model price must be byte-for-byte stable given a fixed seed.

---

# PART II — Engine Hardening Track

> This part is a **separate work stream** from the `SimulationResult`/2-D LSM feature above (Part I, Phases 0–5). It collects correctness and robustness fixes found during code review that are **independent of** the Heston two-factor change. They are grouped here so the whole engine-improvement effort lives in one document.

## 7. Scope, Coordination & Sequencing Rules

### 7.0 Coordination with Part I
- **Do Part I (Phases 0–5) FIRST, then this track.** Changes C, D and E below all edit `mc_pricing_engine.py` (`_spot_eps`/`_delta_gamma`, `_vega`, `_theta`) — the *same* file Part I touches. Landing Part I first avoids merge churn.
- Each change below is its own commit. **The full suite must be green between commits.**
- For every "math-changing" item, the commit that changes numbers must also **update/record the golden values** in the same commit, with a comment citing why the number moved.

### 7.1 Issue catalogue
| ID | Issue | Severity | File(s) | Changes prices? |
|----|-------|----------|---------|-----------------|
| **A** | Sobol sample count not a power of 2 → QMC bias (warning silently suppressed) | 🔴 High | `kernel/tools.py`, `mc_pricing_engine.py` | **Yes — only when generator = SOBOL** |
| **B** | Barrier options monitored on grid only → discrete-monitoring bias vs. continuous barrier | 🔴 High | `barrier_options.py` | **Yes — barrier products** (behind a flag) |
| **C** | Delta/Gamma finite-difference bump fixed at `1.0`, does not scale with spot → unstable Gamma at large spot | 🔴 High | `mc_pricing_engine.py` | **Yes — Greeks only when `S0 ≠ 100`** |
| **D** | Heston Vega bumps `v0` only, not `theta` → partial/mislabelled vega | 🟠 Med | `mc_pricing_engine.py` | **Yes — Heston vega only** |
| **E** | CRN Theta silently returns `0.0` when `nb_steps < 2` | 🟠 Med | `mc_pricing_engine.py` | No (adds a warning only) |
| **F** | Every Greek `deepcopy`s the whole Market (rebuilds the vol surface) → perf hotspot in `_rho` | 🟠 Med | `market.py`, `mc_pricing_engine.py` | No (must stay byte-identical) |
| **G** | LSM loop recomputes `np.sum(in_money)` / re-indexes each step | 🟡 Low | `american_mc_pricing_engine.py` | No (must stay byte-identical) |

> **Verified non-issues (do NOT "fix" — they are already correct):** the `ChooserOption` decision rule (`S_t1 > K·DF_fwd`, exact by put-call parity), the Heston Cholesky correlation sign, full-truncation on the variance, drift/discount consistency in `Market.get_fwd_rate`, and the affine autocall coupon solver. Touching these will only introduce regressions.

---

## 8. Per-Change Breakdown (executable steps + test cases)

### Change A — Snap Sobol path count to a power of 2 (inside the code) 🔴🔢
**Design decision (per stakeholder):** the *code* aligns the simulation size; the user does **not** have to pass an exact power of 2. When the generator is SOBOL, round `nb_paths` **down** to the previous power of two and actually simulate that many paths (using the full balanced point set — do **not** generate 2ᵐ then slice back, which re-breaks the balance).

**Why round down, and simulate the full set:** every consumer of the count already derives it from the array (`len(payoffs)`, `paths.shape[0]`), so returning fewer rows is safe **provided the allocation and the increments agree**. The clean, single-source place to enforce that is the engine's `nb_paths`, which flows into both `EulerScheme` (allocation) and `get_random_increments` (the Sobol draw).

**Steps:**
- **A1.** Add a pure helper (e.g. in `kernel/tools.py`): `def previous_power_of_two(n: int) -> int:` returning `1 << (n.bit_length() - 1)` (and `1` for `n <= 1`).
- **A2.** In `MCPricingEngine.__init__` (after `self.nb_paths = settings.nb_paths`), resolve the generator type once and snap:
  ```python
  gen_type = getattr(settings, "random_generator_type", "NUMPY")
  gen_type = getattr(gen_type, "value", gen_type)
  if gen_type == "SOBOL":
      snapped = previous_power_of_two(self.nb_paths)
      if snapped != self.nb_paths:
          warnings.warn(f"SOBOL requires a power-of-two sample size; rounding nb_paths "
                        f"{self.nb_paths} -> {snapped} for QMC balance.")
          self.nb_paths = snapped
  ```
  This propagates automatically to `AmericanMCPricingEngine` and `CallableMCPricingEngine` (both subclass it) and to `_theta`'s CRN base-sim (which uses `self.nb_paths`).
- **A3.** Leave `SobolRandomGenerator` as-is but keep the `catch_warnings` block as *defensive* (it will no longer fire, since `n` is now 2ᵐ). Add a one-line comment saying the power-of-two guarantee now comes from the engine.
- **A4.** (Optional, only if a caller builds a Sobol generator directly, bypassing the engine) mirror the snap inside `SobolRandomGenerator.get_standard_normal` so direct use is also safe.

**Tests — new file `tests/test_sobol_alignment.py`:**
1. `test_previous_power_of_two` — unit table: `1→1, 2→2, 3→2, 1000→512, 50000→32768, 65536→65536`.
2. `test_sobol_engine_snaps_nb_paths` — build an engine with `random_generator_type=SOBOL, nb_paths=1000`; assert `engine.nb_paths == 512` and that a `UserWarning` was raised (`pytest.warns`).
3. `test_numpy_engine_does_not_snap` — same but `NUMPY`; assert `engine.nb_paths == 1000` (regression guard: pseudo-random path untouched).
4. `test_sobol_paths_shape_matches_snapped` — simulate; assert the returned `spot_paths.shape[0] == 512` (allocation and increments agree — guards the "shape mismatch" failure mode).
5. `test_sobol_determinism` — two runs, same seed → `np.array_equal` on the paths (CRN/reproducibility preserved).
6. **🔢 Math validation** `test_sobol_prices_converge_to_bs` — European call, `Model.BLACK_SCHOLES`, `nb_paths=16384` (already 2ᵐ), SOBOL; assert price within `atol=0.02` of the BS closed form. Confirms the QMC path is still *correct*, not just balanced.
- **Golden note:** any existing test that prices with SOBOL at a non-2ᵐ count will now use more paths → its number moves. Grep for `SOBOL` in tests; re-baseline those goldens in this commit only.

---

### Change B — Continuity correction for discretely-monitored barriers 🔴🔢
**Problem:** `is_barrier_breached` uses `np.max/np.min` over grid points only ([barrier_options.py:32,52](kernel/products/options/barrier_options.py#L32)), so breaches *between* grid points are missed. Discrete monitoring records **fewer** breaches than continuous monitoring → up-and-out is priced **too high**, knock-ins too low.

**Fix (Broadie–Glasserman–Kou continuity correction).** Shift the barrier by a factor $e^{\pm\beta\sigma\sqrt{\Delta t}}$, $\beta=0.5826$, so that a *discretely* monitored option on the shifted barrier approximates the *continuously* monitored option on the true barrier:
- **Up barrier:** monitor against $B\cdot e^{-\beta\sigma\sqrt{\Delta t}}$ (lower → easier to breach → more knock-outs, matching continuous).
- **Down barrier:** monitor against $B\cdot e^{+\beta\sigma\sqrt{\Delta t}}$ (higher → closer to spot).

**Self-contained data:** the payoff already receives `market` and `paths`, so it can source everything it needs without an API change:
- $\sigma = $ `market.get_volatility(self.strike, self.maturity)` (BS: exact flat vol; Heston: BS-equivalent proxy — acceptable first-order correction),
- $\Delta t = $ `self.maturity / (paths.shape[1] - 1)`.

**Steps:**
- **B1.** Add `apply_continuity_correction: bool = True` to `AbstractBarrierOption.__init__` (keep it a flag so the raw discrete behavior remains reachable and old goldens can be reproduced with `False`).
- **B2.** Add `AbstractBarrierOption._effective_barrier(self, paths, market)`:
  ```python
  if not self.apply_continuity_correction:
      return self.barrier
  sigma = market.get_volatility(self.strike, self.maturity)
  dt = self.maturity / (paths.shape[1] - 1)
  beta = 0.5826
  shift = np.exp(beta * sigma * np.sqrt(dt))
  return self.barrier / shift if isinstance(self, UpBarrierOption) else self.barrier * shift
  ```
- **B3.** In `UpBarrierOption.is_barrier_breached` / `DownBarrierOption.is_barrier_breached`, compare against `self._effective_barrier(paths, market)` instead of `self.barrier`. **This means `is_barrier_breached` now needs `market`** — thread it through (the payoff methods that call it already hold `market`).
- **B4.** Keep the constructor validation (`barrier > strike` etc.) on the *raw* barrier.

**Tests — new file `tests/test_barrier_continuity.py`:**
1. **🔢 Primary math validation** `test_correction_matches_fine_grid` — the whole point of the correction. Price an `UpAndOutCallOption` **with** correction at `nb_steps=50`, and **without** correction at `nb_steps=5000` (the near-continuous truth). Assert they agree within `3*std_dev`. This proves both the *direction* and the *magnitude* of the shift.
2. `test_correction_lowers_up_and_out` — corrected up-and-out price `<` uncorrected price at the same (coarse) step count (more knock-outs). Sanity on direction.
3. `test_unreachable_barrier_is_noop` — barrier at 10× spot: corrected price ≈ uncorrected ≈ vanilla call (correction negligible, `atol` tight). Guards against the correction leaking into far-barrier cases.
4. `test_flag_false_reproduces_legacy` — with `apply_continuity_correction=False`, price equals the pre-change golden exactly (byte-for-byte).
5. `test_knockin_raised_by_correction` — `UpAndInCallOption` corrected price `>` uncorrected (in-out parity direction).
- **Golden / regression note:** `tests/test_theta_crn.py::test_theta_crn_stability_barrier` uses a *reachable* barrier (120) → its **theta** is checked only for sign/finiteness, so it still passes. `test_theta_crn_correctness_vs_analytic_bs` uses an unreachable barrier (10000) → no-op (test 3 above covers this). Grep any test asserting an absolute barrier price and re-baseline in this commit.

---

### Change C — Spot-relative bump for Delta/Gamma 🔴🔢
**Problem:** `_spot_eps` returns a constant `1.0` ([mc_pricing_engine.py:208](kernel/models/pricing_engines/mc_pricing_engine.py#L208)); at `S0 = 1e6` the second difference `(up + down − 2·base)/1²` is catastrophic cancellation → Gamma is noise.

**Fix:** make the bump a fixed *fraction* of spot with a floor.
- **C1.** Change `_spot_eps`:
  ```python
  def _spot_eps(self, derivative) -> float:
      S0 = self.market.underlying_asset.last_price
      return max(1e-2 * S0, 1e-8)
  ```
  `_delta_gamma` already uses `epsilon_spot` for both the bump and the denominators, so no other edit is needed.
- **C2. Backward-compat guarantee:** at `S0 = 100`, `1e-2·100 = 1.0` — **identical to today**, so `test_finite_difference_vs_analytic` (S0=100) and all existing Greek goldens are unchanged. State this in the commit message.

**Tests — add to `tests/test_mc_engine_greeks.py`:**
1. `test_spot_eps_backward_compatible` — assert `_spot_eps` returns `1.0` at `S0=100` (locks the no-op claim).
2. **🔢** `test_gamma_stable_large_spot` — `S0=K=1_000_000`, `Model.BLACK_SCHOLES`; compute Gamma and assert `np.isclose(mc_gamma, bs_gamma, rtol=0.05)` where `bs_gamma = N'(d1)/(S σ √T)`. Under the old constant `1.0` bump this assertion fails (noise); under the relative bump it passes.
3. `test_gamma_finite_small_spot` — `S0=K=1.0`; assert Gamma finite and `> 0`.
4. Regression: existing `test_finite_difference_vs_analytic` must stay green untouched.

---

### Change D — Heston Vega bumps the whole variance level (`v0` **and** `theta`) 🟠🔢
**Problem:** `_vega` Heston branch bumps only `v0` ([mc_pricing_engine.py:238-248](kernel/models/pricing_engines/mc_pricing_engine.py#L238)). That is a `v0`-sensitivity that decays with maturity, not the market-quoted vega. It is also silently mislabelled `vega`.

**Fix:** bump the *variance level* — shift both `v0` and `theta` by the same vol move, so the whole variance term structure moves in parallel (the standard "Heston vega" proxy).
- **D1.** In the Heston branch, after computing `base_vol = √v0`, bump **both**:
  ```python
  process_up.v0    = (base_vol + epsilon) ** 2
  process_up.theta = (np.sqrt(process_up.theta) + epsilon) ** 2
  process_down.v0    = (base_vol - epsilon) ** 2
  process_down.theta = (np.sqrt(process_down.theta) - epsilon) ** 2
  ```
- **D2.** Guard against `theta - epsilon²` going negative for tiny `theta` (clip at a small floor before squaring).

**Tests — new file `tests/test_heston_vega.py`:**
1. `test_heston_vega_positive_finite` — full stochastic Heston params; assert vega finite and `> 0` (long option, vol up ⇒ price up).
2. **🔢** `test_heston_vega_exceeds_v0_only` — compute vega the new way vs. an inlined `v0`-only bump; assert `vega_level >= vega_v0_only` (adding the `theta` bump adds long-dated vol sensitivity, so parallel-shift vega is larger). This documents *why* the number changed.
3. `test_heston_vega_maturity_monotone` — vega increases with maturity for an ATM option (basic term-structure sanity).
- **Golden note:** any recorded Heston vega moves in this commit — re-baseline with a comment.
- **Alternative (if stakeholders prefer minimal math change):** keep `v0`-only and instead **rename** the returned quantity `v0_vega` in `PricingResults` + docstring. Pick one; do not ship both silently.

---

### Change E — Warn instead of silently returning 0.0 Theta 🟠
**Problem:** CRN Theta returns `0.0` for `nb_steps < 2` with no signal ([mc_pricing_engine.py:299-300](kernel/models/pricing_engines/mc_pricing_engine.py#L299)).
- **E1.** Replace the bare `return 0.0` with:
  ```python
  if self.nb_steps < 2:
      warnings.warn("CRN Theta needs nb_steps >= 2; returning 0.0.")
      return 0.0
  ```
**Tests — add to `tests/test_theta_crn.py`:**
1. `test_theta_warns_low_steps` — barrier under BS with `nb_steps=1`, `compute_greeks=True`; `pytest.warns(UserWarning)` and assert `theta == 0.0`. No pricing numbers change.

---

### Change F — Cheap `_rho` without full-Market deepcopy 🟠 (no price change)
**Problem:** `bump_flat_yield_curve` / `bump_volatility` `deepcopy` the entire `Market` (including recalibrating the SVI/SSVI surface) on every Greek ([market.py:238-253](kernel/market_data/market.py#L238)). `_rho` triggers this twice.
- **F1.** Add `Market.bump_flat_yield_curve_fast(bump)` that deep-copies **only** `rate_curve` (and rebuilds it from `_raw_yield_data`), reusing the *same* (immutable during pricing) volatility surface object.
- **F2.** Point `_rho` at the fast variant. Leave `bump_volatility` as-is (vega needs the surface rebuilt anyway; but see Change D — Heston vega does not use `bump_volatility`).

**Tests — add to `tests/test_mc_engine_greeks.py`:**
1. **Byte-identical guard** `test_rho_unchanged_after_fast_bump` — assert `rho` equals the pre-change golden **exactly** (`==`, fixed seed). This is a pure optimization; any change in the number is a bug.
2. `test_rho_fast_is_faster` — soft perf check mirroring `test_fast_implied_coupon_solver`'s style (wall-clock bound), marked non-strict so CI noise doesn't flake it.

---

### Change G — LSM inner-loop micro-optimization 🟡 (no price change)
- **G1.** In `AmericanMCPricingEngine._get_price`, cache `n_itm = int(np.sum(in_money))` and `imm_itm = immediate[in_money]` once per step; reuse in the `np.ones(...)`, `x_matrix`, and exercise comparison.
**Tests:**
1. **Byte-identical guard** — extend `test_american_put_vs_binomial` (or add `test_lsm_price_unchanged_after_refactor`) asserting the price equals the pre-change golden **exactly** for a fixed seed. Pure refactor; number must not move.

---

## 9. Recommended Execution Phases (Hardening Track)

Ordered so that **zero-risk cleanups lock a baseline first**, then math changes land one isolated commit at a time, cheapest-blast-radius first, with the biggest modelling change last. Run **after** Part I is merged.

### Phase H1 — Safe, no-number-change commits (baseline lock)
- [ ] **G** (LSM micro-opt) — byte-identical guard green.
- [ ] **E** (Theta low-step warning) — warning test green, no price change.
- [ ] **F** (fast `_rho`) — `rho` byte-identical guard green.
- [ ] Run full suite → green. **Record fresh goldens for anything Part I already changed** so later phases have a clean reference.

### Phase H2 — Delta/Gamma relative bump (Change C)
- [ ] Implement C1; confirm C2 no-op at `S0=100` (existing Greek tests untouched).
- [ ] Add the large-spot / small-spot Gamma tests.
- [ ] Suite green. Only Greeks at `S0 ≠ 100` move.

### Phase H3 — Sobol power-of-two alignment (Change A)
- [ ] Implement A1–A3; add `tests/test_sobol_alignment.py`.
- [ ] Re-baseline any SOBOL-priced goldens (grep for `SOBOL` in tests).
- [ ] Suite green. Pseudo-random (NUMPY) path provably unchanged (test 3).
- [ ] **Do this before H5:** it removes QMC bias as a confound when validating the barrier correction under SOBOL.

### Phase H4 — Heston Vega semantics (Change D)
- [ ] Implement D1–D2 (or the rename alternative — decide first).
- [ ] Add `tests/test_heston_vega.py`; re-baseline Heston vega goldens.
- [ ] Suite green. Only Heston vega moves.

### Phase H5 — Barrier continuity correction (Change B) — biggest math change, do last
- [ ] Implement B1–B4 behind the `apply_continuity_correction` flag (default `True`).
- [ ] Add `tests/test_barrier_continuity.py`; the **fine-grid convergence test (B-test 1)** is the acceptance gate.
- [ ] Re-baseline barrier goldens; confirm `flag=False` reproduces legacy exactly.
- [ ] Confirm `test_theta_crn.py` still green (reachable-barrier theta is sign/finiteness only).
- [ ] Suite green.

### Phase H6 — Docs & sign-off
- [ ] Update engine docstrings: Sobol size-snapping, Heston vega definition, barrier monitoring semantics, `_spot_eps` bump policy.
- [ ] Note the `SimulationResult` memory footprint for large two-factor runs (Part I): the `(nb_paths, nb_steps+1, 2)` array is retained as long as the `SimulationResult` is alive.
- [ ] Final full-suite run.

### Risk / rollback notes (Hardening Track)
- Each phase is one commit; a bisect cleanly attributes any moved number to exactly one change.
- **Expected-to-move numbers, by phase:** H2 → Greeks at non-100 spot; H3 → SOBOL-priced values; H4 → Heston vega; H5 → barrier prices. **Everything else must stay byte-for-byte stable at a fixed seed** — treat any unexpected drift as a regression.
- Changes A, C, D, E all edit `mc_pricing_engine.py`; keep them in the phase order above to minimize conflicts, and rebase rather than merge if Part I lands in parallel.
