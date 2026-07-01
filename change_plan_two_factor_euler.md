# Two-Factor (Heston) Simulation & LSM Regression Upgrade Plan

> [!IMPORTANT]
> **Depends on `change_plan_euler_log_scheme.md` — merge that FIRST.** Both plans edit `EulerScheme._simulate_two_factor` and `tests/test_models.py`; see that plan's "Cross-Plan Coordination" section for the exact stitch. `IMPLEMENTATION_MASTER_ROADMAP.md` is outdated (wrong module paths, mixes the two plans) — treat these two `change_plan_*.md` files as the single source of truth.

## 1. Background & Motivation

Currently, the `EulerScheme` elegantly abstracts the underlying stochastic process (Black-Scholes vs. Heston) by returning only the simulated spot price paths (`paths[:, :, 0]`) as a 2D numpy array.

While this ensures perfect backward compatibility for standard derivatives (which only care about the spot price), it creates a critical blind spot for **American Options (LSM Engine)**. To make an optimal early exercise decision under stochastic volatility, the Longstaff-Schwartz regression must know the instantaneous variance $v_t$. Because the variance paths are discarded by `EulerScheme`, the American engine is currently forced to regress only on the spot price, leading to sub-optimal exercise boundaries and underpricing.

We will resolve this using **Option A (The `SimulationResult` Object)**, which passes the variance state to the engines that need it, without breaking the derivative payoff functions.

> **This is NOT an easy change.** The return type of `EulerScheme.simulate_paths` is consumed in at least 4 production call-sites and 3 test files. Changing it from `np.ndarray` to an object touches the whole simulation → pricing → payoff pipeline. This document is structured as **6 sequential phases** (Section 5) precisely so that each phase leaves the test suite green before the next begins. Do **not** attempt the whole change in one commit.

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

> **Mapping of the mathematical basis** (all normalized): $X_1 = S_t/K$, $X_2 = (S_t/K)^2$, $X_3 = v_t/v_{ref}$, $X_4 = (v_t/v_{ref})^2$, $X_5 = (S_t/K)\cdot(v_t/v_{ref})$, plus the intercept.

---

## 3. Product Compatibility & Test Coverage Plan

**ALL** derivatives in the codebase are fundamentally compatible with a Heston process, because the payoff of equity derivatives depends entirely on the spot price trajectory, which the Heston process provides. The engine-layer unpacking (Section 2.C) guarantees payoffs always receive a raw spot array.

We must **(a) fix the existing tests that consume the return value as a raw array**, and **(b) add a dedicated integration + math test file.**

### 3.0 Existing tests that WILL break (must be fixed — do not skip)
| File | Lines | Problem | Fix |
|------|-------|---------|-----|
| `tests/test_models.py` | ~95, ~115, ~135 | Uses `paths.shape` / `paths[:, 0]` directly on the return value | Change to `res = scheme.simulate_paths(...); paths = res.spot_paths` then assert on `paths` |
| `tests/test_american_engine.py` | 49, 65 | Passes **raw** `np.ones(...)` as `pre_simulated_paths` | No change needed IF C.3 keeps raw-array support (`getattr(..., "spot_paths", ...)`). This is a **regression guard** — it must keep passing. |
| `tests/test_mc_engine_greeks.py` | 97 | Mocks `simulate_paths`, asserts `call_count` only | No change needed (return value unused). Regression guard. |

### 3.1 New File: `tests/test_heston_integration.py`

The tests below are ordered from "cheap sanity checks that isolate bugs" to "expensive convergence checks", so a failure points you to the layer at fault.

#### Test 1 — Heston degenerates to Black-Scholes (structural sanity) 🔢
- **Goal:** The cleanest possible check that the two-factor path construction + unpacking are correct.
- **Setup:** Set vol-of-vol `sigma = 0` and `v0 = theta` (variance becomes a constant $= v_0$).
- **Check:** A `EuropeanCallOption` priced under this degenerate Heston MUST match the Black-Scholes closed form with $\sigma_{BS} = \sqrt{v_0}$, within `3 * std_dev` (statistical tolerance, not a hardcoded atol).
- **Why it matters:** Isolates the two-factor discretization from stochastic-vol noise. If this fails, the bug is in the path construction / unpacking, not the model.

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

#### Test 4 — American Put: 2-D regression is *not worse*, and is accurate (the primary fix) 🔢
- **Goal:** Verify `AmericanMCPricingEngine` unpacks `variance_paths`, runs the 5-term regression, and that the variance state improves the exercise boundary.
- **⚠️ Do NOT assert `price_2d > price_1d` strictly.** LSM on in-sample paths is a low-biased estimator; adding basis functions reduces sub-optimality but is **not monotonic** and is easily masked by MC noise. A strict `>` assertion is mathematically unjustified and will be flaky.
- **Preferred check (accuracy vs. benchmark):**
  1. Build a high-accuracy benchmark for the Heston American put (e.g. a 2-D finite-difference PDE, or a published reference value hardcoded with citation).
  2. Assert the 2-D LSM price is within tolerance of the benchmark.
  3. Assert the 2-D price is **at least as good** as 1-D: `price_2d >= price_1d - tol` AND `abs(price_2d - bench) <= abs(price_1d - bench) + tol` (2-D is no further from truth than 1-D).
- **Fair-comparison guard:** Regress on one seed's paths and value on an independent seed's paths (out-of-sample), so the comparison is not contaminated by in-sample high bias.
- **Lower-bound sanity (cheap, always include):** `american_put_price >= european_put_price - tol` and `american_put_price >= intrinsic_at_t0`.

#### Test 5 — Black-Scholes American unchanged (regression guard) 🔢
- **Goal:** Prove the refactor did not alter the existing 1-D behavior.
- **Check:** Price a Black-Scholes `AmericanPutOption` (one-factor → `var_paths is None`) with a fixed seed and assert the price equals the pre-refactor value (record it as a golden number). This confirms the `var_paths is None` fallback path is truly a no-op relative to today.

#### Test 6 — Path-Dependent & Barrier Options complete under Heston
- **Goal:** Prove changing the `EulerScheme` return type did not break downstream exotic payoffs.
- **Check:** Run `AsianCallOption` and `UpAndOutCallOption` through `MCPricingEngine` with `Model.HESTON`. Assert the engine completes and returns a finite, non-negative price. Bonus: `UpAndOut <= Vanilla` (barrier discount holds).

#### Test 7 — Structured Products (Autocalls) complete under Heston
- **Goal:** Prove `CallableMCPricingEngine` handles `SimulationResult` correctly during coupon solving (this is where the C.2 unpacking is exercised).
- **Check:** Solve the coupon for a `Phoenix` product using Heston. Assert it returns a finite, non-zero coupon and the bisection converges (does not hit `max_iter` at a bound).

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

### Greeks are unaffected
`_delta_gamma`, `_vega`, `_rho`, `_theta` all re-price through `_get_price`, so they inherit the unpacking automatically. No separate changes needed — but they should be smoke-tested under Heston (covered indirectly by Test 3's engine round-trip).

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
This is the "breaking" phase; keep it tightly scoped.
- [ ] Update `_simulate_one_factor`, `_simulate_two_factor` (stop slicing `[:,:,0]` away — keep variance), `simulate_paths` annotation.
- [ ] Update `MCPricingEngine._get_price` (C.1) — unpack `.spot_paths`, keep raw-array support for `pre_simulated_paths`.
- [ ] Update `CallableMCPricingEngine.get_coupon` (C.2) — unpack `.spot_paths` early.
- [ ] Update `AmericanMCPricingEngine._get_price` (C.3) — unpack spot **and** variance, but keep the 1-D basis for now (do not add variance terms yet).
- [ ] **Fix existing tests** `tests/test_models.py` (Section 3.0) to read `.spot_paths`.
- [ ] Run suite → must be green. At this point behavior is identical to before; only the plumbing changed.

### Phase 3 — Add the 2-D LSM regression (the actual feature)
- [ ] Implement the expanded, variance-normalized basis in `AmericanMCPricingEngine._get_price` (C.3 step 2), guarded by `if var_paths is not None`.
- [ ] Verify Black-Scholes American price is unchanged (Test 5 golden value) — the guard must be a no-op for one-factor.
- [ ] Run suite → green.

### Phase 4 — Math & integration test suite
- [ ] Add `tests/test_heston_integration.py` with Tests 1–7 (Section 3.1).
- [ ] Tune tolerances using returned `std_dev` (statistical bands), not hardcoded atol.
- [ ] Ensure Test 3/4 use `nb_steps >= 250` and out-of-sample seeds where specified.

### Phase 5 — Hardening & docs
- [ ] Add the Feller-condition assertion/comment.
- [ ] Confirm Greeks run under Heston (quick smoke test).
- [ ] Update docstrings/return annotations; note the `pre_simulated_paths` polymorphism.
- [ ] Final full-suite run + a manual Heston American vs. European sanity print.

---

## 6. Rollback / Risk Notes
- The riskiest single step is **Phase 2** (return-type change). If anything downstream unexpectedly consumes the raw array, it surfaces here. Mitigation: the `getattr(x, "spot_paths", x)` polymorphism means even a missed call-site that receives a `SimulationResult` where an array was expected fails loudly and locally, not silently.
- Keep Phases 2 and 3 in **separate commits** so a bisect can distinguish "plumbing broke" from "regression math changed the numbers".
- The only numbers that are *expected* to change across this whole change are **Heston American option prices** (Phase 3). Every other product/model price must be byte-for-byte stable given a fixed seed.
