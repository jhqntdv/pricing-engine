# Upgrade Euler Discretization to Log-Euler (Exact) Scheme

> [!IMPORTANT]
> **Single source of truth.** This document (together with `change_plan_two_factor_euler.md`) is authoritative. `IMPLEMENTATION_MASTER_ROADMAP.md` is **outdated / for historical reference only** — it uses wrong module paths (`kernel/engines/...` instead of `kernel/models/pricing_engines/...`) and mixes the two-factor `SimulationResult` work into the Log-Euler phases, which contradicts the required ordering. Do **not** implement from the roadmap.
>
> **Execution order across the two plans (critical):** This Log-Euler plan must be merged **BEFORE** `change_plan_two_factor_euler.md`. Both plans edit the **same** `_simulate_two_factor` method and the **same** `tests/test_models.py`. See the new "Cross-Plan Coordination" section below.

## Background & Motivation

The current `EulerScheme` applies a **Raw (Arithmetic) Euler-Maruyama** discretization to geometric asset price processes:

$$S_{t+dt} = S_t + \mu S_t \cdot dt + \sigma S_t \cdot dW$$

This linear approximation has two critical flaws:
1. **Negative prices**: Large random shocks can push $S_{t+dt}$ below zero, violating GBM's strict positivity.
2. **Discretization bias**: The linear step does not match GBM's exponential dynamics, introducing systematic pricing error that grows with $dt$.

### Evidence (Tested)
Using extreme but valid parameters ($\sigma = 200\%$, $dt = 1.0$, 100,000 paths):
```text
Minimum Price Generated: -777.82
Negative paths: 30,987 / 100,000 (30.99%)
```

### Target State
Upgrade to the **Log-Euler (Exact) scheme**:

$$S_{t+dt} = S_t \cdot \exp\!\Bigl(\bigl(\mu_{prop} - \tfrac{1}{2}\sigma_{prop}^2\bigr)dt + \sigma_{prop} \cdot dW\Bigr)$$

For Black-Scholes with constant $\sigma$, this is the **exact analytical solution** — zero discretization error regardless of step size.

---

## User Review Required

> [!IMPORTANT]
> **Design Decision: Keep existing `get_drift` / `get_volatility` interface unchanged.**
> Currently `get_drift(t, x)` returns $\mu \cdot x$ (absolute) and `get_volatility(t, x)` returns $\sigma \cdot x$ (absolute).
> Rather than changing this interface (which would break all existing processes and tests), the `EulerScheme` will **divide by $x$** internally to recover proportional rates, with `np.maximum(x, 1e-12)` protection against division-by-zero.
> This keeps the `StochasticProcess` API backward-compatible and isolates all changes to the discretization layer.

> [!WARNING]
> **Future Normal-Distributed Models**: The Log-Euler scheme assumes strictly positive asset prices. If we add models where the state variable can be negative (e.g., Vasicek interest rate model, spread models), those processes must set `is_log_process = False` to fall back to the existing Raw Euler logic. This is controlled by the new flag on `StochasticProcess`.

---

## Cross-Plan Coordination (Log-Euler ↔ Two-Factor `SimulationResult`)

This plan and `change_plan_two_factor_euler.md` **both edit two of the same artifacts**. Doing them independently will cause silent overwrites or broken tests. Coordinate as follows:

**1. Shared method: `EulerScheme._simulate_two_factor`.**
- *This plan* makes the **spot** dimension use Log-Euler (variance stays Raw Euler + Full Truncation) and returns `paths[:, :, 0]`.
- *Two-factor plan* changes the **return type** to `SimulationResult(spot_paths=paths[:, :, 0], variance_paths=paths[:, :, 1])` and stops slicing the variance away.
- **Required stitch (two-factor phase):** keep this plan's Log-Euler spot step, but return the full `(nb_paths, nb_steps+1, 2)` internally and wrap it in `SimulationResult`. The two changes are complementary, but they land on the **same lines** — merge them by hand in one commit, do not cherry-pick blindly.

**2. Shared test file: `tests/test_models.py`.**
- *This plan* adds `assert np.all(paths > 0)` on the return of `simulate_paths` (Phase 3 / sub-phase 4b–4c).
- *Two-factor plan* changes `paths.shape` → `res.spot_paths.shape` because the return becomes a `SimulationResult`.
- **Consequence:** After the two-factor merge, **every** assertion in this plan's new tests that treats the `simulate_paths` return as a raw array — the positivity asserts **and** all the new math tests below that do `paths[:, -1]` / `paths[:, 0]` — must be updated to read `res = scheme.simulate_paths(...); paths = res.spot_paths`. Each new test in this document is annotated where relevant.

**3. Ordering (non-negotiable):** Merge **this Log-Euler plan first**, then the two-factor plan. The two-factor plan's Phase 0 already states this dependency. Rationale: Log-Euler is a pure numerical-accuracy change with a stable `np.ndarray` return; introducing `SimulationResult` on top of a known-good Log-Euler baseline isolates any type-plumbing failures from any math failures.

---

## Proposed Changes

### Phase 1: Core Infrastructure

---

#### [MODIFY] [stochastic_process.py](file:///c:/Users/jms_hp26/Desktop/Proj/pricing-engine/kernel/models/stochastic_processes/stochastic_process.py)

**Change**: Add `is_log_process` flag to the base `StochasticProcess.__init__`.

```python
# Current (line 10):
def __init__(self, S0, T, nb_steps, nb_factors=1, random_generator=None):

# New:
def __init__(self, S0, T, nb_steps, nb_factors=1, random_generator=None, is_log_process=True):
    ...
    self.is_log_process = is_log_process
```

**Rationale**: Default `True` because all current processes (BS, Heston) model geometric asset prices. Future normal-distributed processes will explicitly pass `False`.

---

#### [MODIFY] [black_scholes_process.py](file:///c:/Users/jms_hp26/Desktop/Proj/pricing-engine/kernel/models/stochastic_processes/black_scholes_process.py)

**Change**: No functional change needed — it inherits the default `is_log_process=True` from the base class. However, for explicitness and documentation, add it to the `super().__init__` call:

```python
super().__init__(S0, T, nb_steps, random_generator=random_generator, is_log_process=True)
```

**No changes** to `get_drift` or `get_volatility` — they continue to return absolute values ($\mu x$, $\sigma x$).

---

#### [MODIFY] [heston_process.py](file:///c:/Users/jms_hp26/Desktop/Proj/pricing-engine/kernel/models/stochastic_processes/heston_process.py)

**Change**: Same as BS — explicitly pass `is_log_process=True`:

```python
super().__init__(S0, T, nb_steps, nb_factors=2, random_generator=random_generator, is_log_process=True)
```

---

### Phase 2: Euler Scheme Upgrade

---

#### [MODIFY] [euler_scheme.py](file:///c:/Users/jms_hp26/Desktop/Proj/pricing-engine/kernel/models/discretization_schemes/euler_scheme.py)

This is the core change. Both `_simulate_one_factor` and `_simulate_two_factor` need conditional logic based on `process.is_log_process`.

##### `_simulate_one_factor` (Black-Scholes)

```python
def _simulate_one_factor(self, process, nb_paths, seed):
    paths = np.zeros((nb_paths, process.nb_steps + 1))
    paths[:, 0] = process.S0
    dt = process.dt
    dW = process.get_random_increments(nb_paths, seed)

    for i in range(process.nb_steps):
        x = paths[:, i]
        dW_i = dW[:, i]
        drift = process.get_drift(i, x)
        vol = process.get_volatility(i, x)

        if process.is_log_process:
            # Convert absolute drift/vol to proportional rates
            safe_x = np.maximum(x, 1e-12)
            mu_prop = drift / safe_x     # mu[t]
            sig_prop = vol / safe_x      # sigma
            # Exact geometric step (Ito-corrected)
            paths[:, i + 1] = x * np.exp(
                (mu_prop - 0.5 * sig_prop ** 2) * dt + sig_prop * dW_i
            )
        else:
            # Raw Euler for normal-distributed processes
            paths[:, i + 1] = x + drift * dt + vol * dW_i

    return paths
```

**Key details**:
- `np.maximum(x, 1e-12)` prevents division-by-zero when converting absolute → proportional.
- The Ito correction term $-\frac{1}{2}\sigma^2$ is applied inside `exp()` to ensure the expected value $E[S_{t+dt}] = S_t \cdot e^{\mu \cdot dt}$ holds correctly under the risk-neutral measure.
- The `else` branch preserves the original Raw Euler logic byte-for-byte.

##### `_simulate_two_factor` (Heston)

```python
def _simulate_two_factor(self, process, nb_paths, seed):
    paths = np.zeros((nb_paths, process.nb_steps + 1, 2))
    paths[:, 0, 0] = process.S0
    paths[:, 0, 1] = process.v0
    dt = process.dt
    dW1, dW2 = process.get_random_increments(nb_paths, seed)

    for i in range(process.nb_steps):
        x = paths[:, i, 0]
        v = paths[:, i, 1]
        dW1_i = dW1[:, i]
        dW2_i = dW2[:, i]

        drift = process.get_drift(i, x)       # mu[t] * x
        vol_drift = process.get_vol_drift(i, v)  # kappa * (theta - max(v,0))
        vol_vol = process.get_vol_vol(i, v)      # sigma * sqrt(max(v,0))

        if process.is_log_process:
            # Log-Euler for spot dimension only
            safe_x = np.maximum(x, 1e-12)
            mu_prop = drift / safe_x               # mu[t]
            v_pos = np.maximum(v, 0)                # Full Truncation
            # Exact geometric step with stochastic variance
            x_next = x * np.exp(
                (mu_prop - 0.5 * v_pos) * dt + np.sqrt(v_pos) * dW1_i
            )
        else:
            x_next = x + drift * dt + np.sqrt(np.maximum(v, 0)) * x * dW1_i

        # Variance dimension: always Raw Euler with Full Truncation
        # (variance follows a CIR process which can naturally be near zero)
        v_next = v + vol_drift * dt + vol_vol * dW2_i

        paths[:, i + 1, 0] = x_next
        paths[:, i + 1, 1] = v_next

    return paths[:, :, 0]
```

**Key details**:
- Only the **spot price dimension** ($S_t$) uses Log-Euler. The **variance dimension** ($v_t$) continues to use Raw Euler with Full Truncation, because variance follows a CIR process (not geometric) and can legitimately approach zero.
- The Ito correction uses the instantaneous variance $v_t$ directly (not $\sigma^2$), because in Heston the local volatility is $\sqrt{v_t}$.

---

### Phase 3: Test Updates & New Tests

---

#### [MODIFY] [test_models.py](file:///c:/Users/jms_hp26/Desktop/Proj/pricing-engine/tests/test_models.py)

**Change**: Add positivity assertions to existing `TestEulerScheme` tests.

- `test_euler_one_factor` (line 87): Add `assert np.all(paths > 0)` after simulation.
- `test_euler_two_factor` (line 103): Add `assert np.all(paths > 0)` after simulation.
- New test `test_euler_no_negative_extreme_vol`: Reproduce the extreme scenario ($\sigma=200\%$, $dt=1.0$) and assert zero negative paths.

---

#### [NEW] Add to [test_matlab_sanity.py](file:///c:/Users/jms_hp26/Desktop/Proj/pricing-engine/tests/test_matlab_sanity.py) or new file `test_log_euler.py`

**Test 1: Log-Normal Exact Solution Consistency** (the most critical new test)

With `nb_steps=1` (single step), Log-Euler for BS is mathematically exact. Verify:
- Sample mean of $\ln(S_T)$ converges to $\ln(S_0) + (\mu - \frac{1}{2}\sigma^2)T$
- Sample variance of $\ln(S_T)$ converges to $\sigma^2 T$

```python
def test_log_euler_exact_distribution():
    """Log-Euler with 1 step must produce exact log-normal terminal distribution."""
    S0, T, r, sigma = 100.0, 1.0, 0.05, 0.30
    nb_paths = 500_000

    process = BlackScholesProcess(S0=S0, T=T, nb_steps=1, drift=[r], volatility=sigma)
    scheme = EulerScheme()
    paths = scheme.simulate_paths(process, nb_paths, seed=42)

    log_ST = np.log(paths[:, -1])
    expected_mean = np.log(S0) + (r - 0.5 * sigma**2) * T
    expected_var = sigma**2 * T

    assert np.isclose(np.mean(log_ST), expected_mean, atol=0.01)
    assert np.isclose(np.var(log_ST), expected_var, atol=0.01)
```

> [!IMPORTANT]
**Test 2: Strict Positivity Under Extreme Conditions (Pytest Implementation Required)**
We must implement `test_no_negative_prices_extreme_vol` in pytest (`tests/test_log_euler.py`) to serve as our automated regression check for positivity under extreme conditions, replacing the deleted manual script `demonstrate_negative_prices.py`.

```python
def test_no_negative_prices_extreme_vol():
    """Even with 200% vol and dt=1.0, all paths must remain strictly positive."""
    process = BlackScholesProcess(S0=100.0, T=1.0, nb_steps=1, drift=[0.0], volatility=2.0)
    scheme = EulerScheme()
    paths = scheme.simulate_paths(process, nb_paths=100_000, seed=42)

    assert np.all(paths > 0), f"Found {np.sum(paths <= 0)} non-positive prices"
```

**Test 3: Step-Size Invariance for Black-Scholes**

Log-Euler for BS should give nearly identical results regardless of `nb_steps`, since it's the exact solution at each step:

```python
def test_step_size_invariance():
    """BS Log-Euler price should be stable across different step counts."""
    S0, T, r, sigma, K = 100.0, 1.0, 0.05, 0.30, 100.0

    prices = []
    for steps in [1, 10, 50, 252]:
        process = BlackScholesProcess(S0=S0, T=T, nb_steps=steps, drift=[r]*steps, volatility=sigma)
        scheme = EulerScheme()
        paths = scheme.simulate_paths(process, nb_paths=200_000, seed=42)
        payoffs = np.maximum(paths[:, -1] - K, 0) * np.exp(-r * T)
        prices.append(np.mean(payoffs))

    # All prices should be within 0.10 of each other
    assert max(prices) - min(prices) < 0.10
```

> [!TIP]
> **Prefer a benchmark-based tolerance.** Because Log-Euler is *exact* for constant-vol BS, each per-step price should sit on top of the Black-Scholes analytical price. A stronger version of this test asserts every price is within `3 * std_dev` of the closed-form BS call, instead of only checking they agree with each other (agreement-with-each-other can hide a shared bias).

**Test 4: Risk-Neutral Martingale / Ito-Correction Check (CRITICAL — currently missing)** 🔢

This is the single most important correctness test for the Ito drift correction $-\tfrac12\sigma^2$. If the correction term is dropped, has the wrong sign, or the wrong coefficient, the mean/variance test (Test 1) can still pass while the **discounted asset price stops being a martingale**. Under the risk-neutral measure:

$$E[S_T] = S_0 \, e^{rT}$$

```python
def test_risk_neutral_martingale_bs():
    """E[S_T] must equal S0 * exp(rT) — proves the -0.5*sigma^2 Ito term is correct."""
    S0, T, r, sigma = 100.0, 1.0, 0.05, 0.40
    nb_paths = 500_000
    process = BlackScholesProcess(S0=S0, T=T, nb_steps=50, drift=[r]*50, volatility=sigma)
    scheme = EulerScheme()
    paths = scheme.simulate_paths(process, nb_paths, seed=42)   # NOTE: raw array pre-two-factor; see Cross-Plan Coordination
    ST = paths[:, -1]

    expected = S0 * np.exp(r * T)
    se = np.std(ST, ddof=1) / np.sqrt(nb_paths)
    assert abs(np.mean(ST) - expected) < 3 * se, (
        f"E[S_T]={np.mean(ST):.4f} vs expected {expected:.4f} (3*SE={3*se:.4f}) — Ito correction likely wrong"
    )


def test_risk_neutral_martingale_heston():
    """Under Heston, E[S_T] = S0*exp(rT) regardless of variance params (spot is a Q-martingale after discounting)."""
    S0, T, r = 100.0, 1.0, 0.05
    nb_paths = 500_000
    process = HestonProcess(S0=S0, v0=0.04, T=T, nb_steps=250, drift=[r]*250,
                            kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5)
    scheme = EulerScheme()
    paths = scheme.simulate_paths(process, nb_paths, seed=42)
    ST = paths[:, -1]                                            # spot dimension
    expected = S0 * np.exp(r * T)
    se = np.std(ST, ddof=1) / np.sqrt(nb_paths)
    # Full-truncation Euler introduces a small martingale bias; allow a modest multiple of SE.
    assert abs(np.mean(ST) - expected) < 5 * se
```

**Test 5: Put-Call Parity under Log-Euler BS (robust, no benchmark needed)** 🔢

```python
def test_put_call_parity_log_euler():
    """C - P = S0 - K*exp(-rT). Holds path-by-path; the most robust pipeline regression test."""
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.30
    nb_paths = 200_000
    process = BlackScholesProcess(S0=S0, T=T, nb_steps=50, drift=[r]*50, volatility=sigma)
    scheme = EulerScheme()
    paths = scheme.simulate_paths(process, nb_paths, seed=42)
    ST = paths[:, -1]
    df = np.exp(-r * T)
    call = np.mean(np.maximum(ST - K, 0)) * df
    put  = np.mean(np.maximum(K - ST, 0)) * df
    assert np.isclose(call - put, S0 - K * df, atol=0.02)
```

**Test 6: Heston degenerates to Black-Scholes (bridges both plans)** 🔢

```python
def test_heston_degenerates_to_bs():
    """vol-of-vol=0 and v0=theta => constant variance => Heston spot must match BS log-Euler exactly.
    This simultaneously validates that the Log-Euler spot step and the (later) SimulationResult change agree."""
    S0, K, T, r, v0 = 100.0, 100.0, 1.0, 0.05, 0.04
    nb_paths = 200_000
    heston = HestonProcess(S0=S0, v0=v0, T=T, nb_steps=100, drift=[r]*100,
                           kappa=2.0, theta=v0, sigma=0.0, rho=0.0)   # sigma(vol-of-vol)=0, v0=theta
    scheme = EulerScheme()
    paths = scheme.simulate_paths(heston, nb_paths, seed=42)
    price_heston = np.mean(np.maximum(paths[:, -1] - K, 0)) * np.exp(-r * T)

    # Black-Scholes closed form with sigma = sqrt(v0)
    from scipy.stats import norm
    sig = np.sqrt(v0)
    d1 = (np.log(S0/K) + (r + 0.5*sig**2)*T) / (sig*np.sqrt(T))
    d2 = d1 - sig*np.sqrt(T)
    bs = S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    assert abs(price_heston - bs) < 0.05
```

**Test 7: `is_log_process=False` fallback actually runs Raw Euler (currently an untested branch)** 🔢

The plan adds an `else` branch for future normal-distributed models, but no test exercises it — it would ship as dead, unverified code. This test forces the flag off and asserts the *old* Raw-Euler behavior returns (which, unlike Log-Euler, can produce non-positive prices under extreme vol).

```python
def test_raw_euler_fallback_when_not_log_process():
    """With is_log_process=False, the scheme must use additive Raw Euler (can go negative)."""
    process = BlackScholesProcess(S0=100.0, T=1.0, nb_steps=1, drift=[0.0], volatility=2.0)
    process.is_log_process = False          # force the fallback branch
    scheme = EulerScheme()
    paths = scheme.simulate_paths(process, nb_paths=100_000, seed=42)
    # Raw Euler with 200% vol and dt=1.0 is expected to breach zero — proving the branch is live.
    assert np.any(paths <= 0), "Fallback branch did not behave like Raw Euler (expected some non-positive paths)"
```

---

#### [MODIFY] [test_matlab_sanity.py](file:///c:/Users/jms_hp26/Desktop/Proj/pricing-engine/tests/test_matlab_sanity.py)

**Change**: After upgrade, re-run and observe whether the existing `atol=0.05` tolerance still holds. If accuracy improves significantly (likely), consider tightening to `atol=0.03`. Add a comment noting the upgrade.

---

#### [MODIFY] [test_mc_engine_greeks.py](file:///c:/Users/jms_hp26/Desktop/Proj/pricing-engine/tests/test_mc_engine_greeks.py)

**Change**: The `test_finite_difference_vs_analytic` test (line 110) compares MC Greeks against closed-form BS Greeks. After Log-Euler upgrade, the MC values should converge more tightly. Re-run and verify existing tolerances (`atol=0.01` for delta, `atol=0.005` for gamma) still pass. Tighten if appropriate.

---

#### [MODIFY] [test_american_engine.py](file:///c:/Users/jms_hp26/Desktop/Proj/pricing-engine/tests/test_american_engine.py)

**Change**: No code changes expected (the `BlackScholesProcess` constructor signature is backward-compatible). Re-run all tests to confirm they pass. The `test_american_put_vs_binomial` comparison (`atol=0.08`) may become tighter — observe and adjust.

---

#### [DELETE] [demonstrate_negative_prices.py](file:///c:/Users/jms_hp26/Desktop/Proj/pricing-engine/demonstrate_negative_prices.py)

**Change**: Already deleted from the repository. Its temporary validation purpose has been entirely replaced by the automated `test_no_negative_prices_extreme_vol` pytest described above.

---

## Complete Affected Files Summary

| File | Change Type | Description |
|---|---|---|
| [stochastic_process.py](file:///c:/Users/jms_hp26/Desktop/Proj/pricing-engine/kernel/models/stochastic_processes/stochastic_process.py) | MODIFY | Add `is_log_process` flag |
| [black_scholes_process.py](file:///c:/Users/jms_hp26/Desktop/Proj/pricing-engine/kernel/models/stochastic_processes/black_scholes_process.py) | MODIFY | Explicitly pass `is_log_process=True` |
| [heston_process.py](file:///c:/Users/jms_hp26/Desktop/Proj/pricing-engine/kernel/models/stochastic_processes/heston_process.py) | MODIFY | Explicitly pass `is_log_process=True` |
| [euler_scheme.py](file:///c:/Users/jms_hp26/Desktop/Proj/pricing-engine/kernel/models/discretization_schemes/euler_scheme.py) | MODIFY | Core Log-Euler upgrade with `np.maximum(x, 1e-12)` guard |
| [test_models.py](file:///c:/Users/jms_hp26/Desktop/Proj/pricing-engine/tests/test_models.py) | MODIFY | Add positivity assertions to existing Euler tests |
| [test_matlab_sanity.py](file:///c:/Users/jms_hp26/Desktop/Proj/pricing-engine/tests/test_matlab_sanity.py) | MODIFY | Re-calibrate tolerance after upgrade |
| [test_mc_engine_greeks.py](file:///c:/Users/jms_hp26/Desktop/Proj/pricing-engine/tests/test_mc_engine_greeks.py) | MODIFY | Re-calibrate tolerance after upgrade |
| [test_american_engine.py](file:///c:/Users/jms_hp26/Desktop/Proj/pricing-engine/tests/test_american_engine.py) | VERIFY | Re-run all tests, adjust tolerance if needed |
| [test_log_euler.py](file:///c:/Users/jms_hp26/Desktop/Proj/pricing-engine/tests/test_log_euler.py) | NEW | Log-normal exactness, positivity, step-size invariance, **risk-neutral martingale (BS+Heston)**, **put-call parity**, **Heston→BS degeneracy**, **`is_log_process=False` fallback** |
| [demonstrate_negative_prices.py](file:///c:/Users/jms_hp26/Desktop/Proj/pricing-engine/demonstrate_negative_prices.py) | DELETE | Removed from project root; functionality replaced by pytest |

---

## Verification Plan

### Automated Tests (in order)
```bash
# 1. Run the new Log-Euler specific tests
uv run pytest tests/test_log_euler.py -v

# 2. Run updated Euler scheme tests with positivity checks
uv run pytest tests/test_models.py::TestEulerScheme -v

# 3. Run all existing pricing tests (regression)
uv run pytest tests/test_matlab_sanity.py tests/test_mc_engine_greeks.py tests/test_american_engine.py tests/test_curve_consistency.py -v

# 4. Full test suite
uv run pytest tests/ -v
```

### Manual Verification
No manual verification script is needed as `demonstrate_negative_prices.py` has been deleted and converted into an automated pytest regression check.


# Volatility Surface Models Review

This document summarizes the mathematical and implementation issues identified in the volatility surface models (`svi_surface.py`, `ssvi_surface.py`, and `local_surface.py`). These findings are based on industry standards, quantitative finance literature, and common software engineering best practices.

## 1. SVI Model (`svi_surface.py`)

### A. Missing Vega Weighting in Cost Function
> [!IMPORTANT]  
> **Issue:** The cost function `cost_function_svi` calculates the `vega` array but fails to apply it when computing the Mean Squared Error (MSE). The current return value is `np.mean((SVI_total_variance - market_total_variance) ** 2)`.
> 
> **Impact:** Deep Out-of-The-Money (OTM) options have near-zero vega, meaning their prices are insensitive to implied volatility. Unweighted OLS allows noisy OTM quotes to dominate the calibration, leading to an unstable smile.
> 
> **Industry Standard / Fix:** In practice (e.g., Gatheral 2006, QuantLib's `BlackCalibrationHelper`), calibration should weight the MSE by Vega (or inverse bid-ask spread). The cost function must multiply the variance differences by the `vega` weights before computing the mean.

### B. Arbitrage via Cubic Interpolation of SVI Parameters
> [!WARNING]  
> **Issue:** In `_interpolate_parameters`, the code uses `interp1d(..., kind='cubic')` to independently interpolate the 5 SVI parameters ($a, b, \rho, m, \sigma$) across maturities.
> 
> **Impact:** Since SVI parameters are highly non-linear, naive interpolation almost guarantees the violation of $\partial w / \partial t \geq 0$ between maturities. This introduces Calendar Spread Arbitrage and potential Butterfly Arbitrage.
> 
> **Industry Standard / Fix:** As explicitly stated by Jim Gatheral and Antoine Jacquier in their 2014 paper *"Arbitrage-free SVI volatility surfaces"*, interpolating SVI parameters directly is flawed. QuantLib avoids 2D SVI interpolation for this exact reason. The fix is to transition to SSVI (Surface SVI) or interpolate on a total variance grid ensuring no-arbitrage conditions, rather than interpolating parameters directly.

---

## 2. SSVI Model (`ssvi_surface.py`)

### A. Unsorted Strikes in NumPy Interpolation
> [!CAUTION]  
> **Issue:** In `_get_market_atm_variance`, the code uses `np.interp` to estimate ATM implied volatility:  
> `atm_vol = np.interp(self.spot, option_slice["Strike"].values, option_slice["Implied Volatility"].values)`
> 
> **Impact:** According to NumPy documentation, `np.interp` requires the x-coordinates (`Strike`) to be strictly increasing. Option chain data is not guaranteed to be perfectly sorted by strike. If unsorted, `np.interp` will return garbage values, severely corrupting the ATM variance calibration.
> 
> **Fix:** Ensure the DataFrame slice is sorted before interpolation:  
> `option_slice = option_slice.sort_values("Strike")`

---

## 3. Local Volatility Model (`local_surface.py`)

### A. Excessively Large Finite Difference Step Size
> [!WARNING]  
> **Issue:** In `_finite_difference_variance`, the bump size for strike/log-moneyness is set to `dK = max(strike * 0.05, 0.01)`.
> 
> **Impact:** Bumping the strike by 5% is a massive step size for numerical differentiation. When computing the second derivative for Dupire's formula, a 5% step causes severe truncation errors ($O(h^2)$), artificially smoothing out the convexity of the volatility smile. This results in highly inaccurate and noisy local volatility values.
> 
> **Industry Standard / Fix:** In standard quant libraries (like QuantLib), the typical bump size for Greeks or finite difference derivatives is 1 bp to 10 bps ($10^{-4}$ to $10^{-3}$). The step size should be reduced to `dK = strike * 0.001` or similar.

### B. Analytical Derivatives Availability (Optimization)
> [!TIP]  
> **Suggestion:** Since the underlying implied volatility surface is parameterized by SVI/SSVI, it possesses closed-form analytical derivatives ($\frac{\partial w}{\partial k}$ and $\frac{\partial^2 w}{\partial k^2}$).
> 
> **Industry Standard / Fix:** Instead of using finite difference methods which are prone to numerical instability, the local volatility surface can directly compute the analytical derivatives of the SVI formula as provided in Gatheral's SSVI paper. QuantLib's `LocalVolSurface` relies on analytical derivatives whenever the underlying volatility structure allows it.

# Pricing Engines Review

This document summarizes the mathematical and implementation issues identified in the pricing engines (`american_mc_pricing_engine.py` and `callable_mc_pricing_engine.py`). These findings are based on industry standards, quantitative finance literature, and common software engineering best practices.

## 1. American MC Pricing Engine (`american_mc_pricing_engine.py`)

### A. Missing Variance State Variable in Longstaff-Schwartz Regression (Heston Model)
> [!IMPORTANT]  
> **Issue:** In the Longstaff-Schwartz Method (LSM) implementation `_get_price`, the regression basis `x_matrix` is constructed solely using the normalized spot price `S_t`. For a 2-factor model like Heston, the instantaneous variance `v_t` is completely ignored.
> 
> **Impact:** The state space for a Heston model is $(S_t, v_t)$. Regressing only on $S_t$ means the engine cannot accurately predict the continuation value under stochastic volatility. This causes suboptimal early exercise decisions and severely underprices American options.
> 
> **Industry Standard / Fix:** Modify the `EulerScheme` to expose the full state (e.g., returning both $S_t$ and $v_t$). The LSM regression must incorporate $v_t$, $v_t^2$, and the cross-term $S_t \cdot v_t$ into the basis functions when a Two-Factor Stochastic Process is used.

---

## 2. Callable MC Pricing Engine (`callable_mc_pricing_engine.py`)

### A. Inefficient Bisection Search for Coupon Rate
> [!WARNING]  
> **Issue:** In `get_coupon`, a Bisection (Dichotomy) method with up to 25 iterations is used to solve for the coupon rate that makes the Autocall price equal to a target price.
> 
> **Impact:** The price of a standard Autocall is a strictly linear function of the coupon rate ($Price(C) = Base\_Price + C \times PV(\text{1\% Coupon})$). Running a full bisection search with 25 iterations over an inherently linear relationship is a mathematical redundancy that unnecessarily wastes computation time, even with pre-simulated paths.
> 
> **Industry Standard / Fix:** Solve for the coupon analytically in a single step. Run the payoff evaluation exactly once to find the Base Price ($C=0$) and the Present Value of a 1% coupon. Then exactly compute $C = \frac{Target\_Price - Base\_Price}{PV_{1\%}}$.

---

# Products Payoff Review

This document summarizes the mathematical and implementation issues identified in the products payoff definitions. These findings are based on industry standards, quantitative finance literature, and common software engineering best practices.

## 1. Phoenix Autocall (`autocall_products.py`)

### A. Periodic Coupons Discounted at Wrong Time

> [!IMPORTANT]  
> **Issue:** In the Phoenix `get_discounted_payoff`, periodic coupons are accumulated into a running `cumulative_coupons` total. When the product autocalls at time $t$, or reaches maturity, the **entire** accumulated coupon sum is discounted using the discount factor at that single future time point:
> ```python
> autocall_payoff = 100.0 + cumulative_coupons + self.coupon_rate + missed_coupons
> payoffs[autocalled] = autocall_payoff[autocalled] * df  # df at time t
> ```
> 
> **Impact:** Phoenix is defined as a product where coupons are **paid to the investor at each observation date** when the coupon barrier condition is met. A coupon earned at month 1 should be discounted by `df(month_1)`, not by `df(month_6)` when the autocall eventually triggers. Applying a later (smaller) discount factor to earlier cashflows systematically **undervalues** the product. The magnitude of the error grows with the interest rate level and the product's maturity.
> 
> **Fix:** Track `discounted_cumulative_coupons` instead. At each observation, when a coupon is paid, immediately compute `pv_coupon = coupon_rate * df(t)` and add it to the discounted total. At autocall or maturity, the final payoff is: `100.0 * df(t_exit) + discounted_cumulative_coupons`.

#### Proposed Code Change

In `Phoenix.get_discounted_payoff`, replace `cumulative_coupons` with `pv_cumulative_coupons`:

```python
# Track the present value of all coupons paid so far (discounted at payment date)
pv_cumulative_coupons = np.zeros(nb_paths)
missed_coupons = np.zeros(nb_paths)

for t in range(1, num_observations):
    spot_t = obs_paths[:, t]
    discount_time = t / self.observation_frequency.value
    df = market.get_discount_factor(discount_time)

    # --- Autocall event ---
    autocalled = active & (spot_t >= self.autocall_barrier)
    if np.any(autocalled):
        # Current period coupon + any missed coupons, discounted at current time
        final_coupon_pv = (self.coupon_rate + missed_coupons[autocalled]) * df
        # Principal discounted at autocall time + all previously discounted coupons
        payoffs[autocalled] = 100.0 * df + pv_cumulative_coupons[autocalled] + final_coupon_pv
        active[autocalled] = False

    # --- Coupon event (only for still-active paths) ---
    coupon_paid = active & (spot_t >= self.coupon_barrier)
    pv_coupon = (self.coupon_rate + missed_coupons[coupon_paid]) * df
    pv_cumulative_coupons[coupon_paid] += pv_coupon
    missed_coupons[coupon_paid] = 0.0

    # --- Missed coupon accumulation (for is_plus) ---
    if self.is_plus:
        missed = active & (spot_t < self.coupon_barrier)
        missed_coupons[missed] += self.coupon_rate
```

#### Test Case: Phoenix Coupon Discount Timing

```python
def test_phoenix_coupon_discount_timing():
    """Verify that Phoenix coupons are discounted at their payment date, not at exit date.

    Setup: A Phoenix that always pays coupons (coupon_barrier=0) and never autocalls
    (autocall_barrier=999). With a steep yield curve, discounting all coupons at
    maturity vs. at each payment date produces a measurable difference.
    """
    from kernel.products.structured_products.autocall_products import Phoenix
    from kernel.tools import ObservationFrequency
    import numpy as np

    class SteepCurveMarket:
        """A market with a 10% flat rate to amplify discounting differences."""
        def get_discount_factor(self, t):
            return np.exp(-0.10 * t)  # 10% continuous rate

    market = SteepCurveMarket()
    maturity = 2.0
    coupon_rate = 5.0  # 5% per period

    phoenix = Phoenix(
        maturity=maturity,
        observation_frequency=ObservationFrequency.QUARTERLY,  # 4x/year = 8 coupons over 2 years
        capital_barrier=0.0,       # never breached (always protected)
        autocall_barrier=999.0,    # never autocalled
        coupon_rate=coupon_rate,
        coupon_barrier=0.0,        # always pays coupon
    )

    # Flat paths at 100 (all conditions always met, no autocall)
    nb_paths = 1000
    nb_steps = 252 * 2  # 2 years of daily steps
    paths = np.ones((nb_paths, nb_steps + 1)) * 100.0

    payoffs = phoenix.get_discounted_payoff(paths, market)

    # --- Expected value (analytical) ---
    # 8 quarterly coupons, each discounted at their payment time
    obs_freq = 4  # quarterly
    num_obs = int(round(maturity * obs_freq + 1))  # includes t=0
    expected_pv_coupons = sum(
        coupon_rate * np.exp(-0.10 * (t / obs_freq)) for t in range(1, num_obs)
    )
    expected_pv_principal = 100.0 * np.exp(-0.10 * maturity)
    expected_total = expected_pv_principal + expected_pv_coupons

    # --- Wrong answer (all coupons discounted at maturity) ---
    wrong_total = (100.0 + coupon_rate * (num_obs - 1)) * np.exp(-0.10 * maturity)

    mean_payoff = np.mean(payoffs)

    # The correct answer should be strictly greater than the wrong answer
    # because earlier coupons are worth more when discounted at their own (earlier) time
    assert expected_total > wrong_total, "Sanity check: correct PV > wrong PV"
    # After fix: mean_payoff should match the analytically correct value
    assert np.isclose(mean_payoff, expected_total, atol=0.10), (
        f"Phoenix payoff {mean_payoff:.4f} should match analytical {expected_total:.4f}, "
        f"not the wrong value {wrong_total:.4f}"
    )
```

---

## 2. Participation Products (`participation_products.py`)

### A. Vectorization Crash in Payoff Calculation

> [!CAUTION]
> **Issue:** In `TwinWin` and `Airbag`, the final price is extracted using `final_price = paths[-1]`. In a Monte Carlo engine, `paths` is a 2D array of shape `(nb_paths, nb_steps + 1)`. `paths[-1]` returns the entire last *path* (a 1D array of prices over time), not the last *column* (terminal prices across all paths).
> 
> **Impact:** This causes element-wise operations later on (`performance > self.upper_barrier`) to throw a fatal `ValueError: The truth value of an array with more than one element is ambiguous`, causing the entire pricing engine to crash when trying to price these products.
> 
> **Fix:** Change to `final_price = paths[:, -1]`. Furthermore, `if/elif` cannot be used on NumPy arrays in a vectorized context. The conditional payoff logic must be rewritten using `np.where` or boolean indexing.



## 2. Callable MC Pricing Engine — Analytical Coupon Solver (`callable_mc_pricing_engine.py`)

### A. Add Analytical Coupon Solver (Default), Keep Bisection as Fallback

> [!IMPORTANT]  
> **Issue:** The current `get_coupon` method uses bisection (up to 25 iterations) to find the coupon rate. Since the Autocall price is a strictly linear function of the coupon rate ($Price(C) = Price(0) + C \times \frac{Price(1) - Price(0)}{1}$), the coupon can be solved analytically in a single step.
> 
> **Design Decision:** Add an `analytical` solver method (default) alongside the existing `bisection` method. The user can switch between them via a `method` parameter. The analytical solver uses the same `pre_simulated_paths` to ensure zero MC noise between the two evaluations.
> 
> **Linearity Guarantee:** This linearity holds for **all** structured equity products in the codebase (Phoenix, Eagle, and any future Autocall variant) because the coupon rate $C$ only appears as a multiplicative scalar on indicator-function-based cashflows. The barrier/autocall/coupon conditions depend solely on the spot price path, **not** on the value of $C$ itself.

#### Proposed Code Change

```python
def get_coupon(self, derivative, process, epsilon=1e-2, max_iter=25,
               target_price=100, method="analytical"):
    """Compute the coupon rate such that the product price equals the target.

    Args:
        derivative: The autocallable product.
        process: The stochastic process.
        epsilon: Tolerance for bisection convergence.
        max_iter: Max iterations for bisection.
        target_price: Target price (typically 100 = par).
        method: "analytical" (default, exact, 2 evaluations) or
                "bisection" (legacy, up to max_iter evaluations).

    Returns:
        float: The computed coupon rate.
    """
    scheme = EulerScheme()
    pre_simulated_paths = scheme.simulate_paths(process, self.nb_paths, self.random_seed)

    if method == "analytical":
        # --- Analytical solver: exactly 2 pricing evaluations ---
        # 1. Price with C = 0
        original_coupon = derivative.coupon_rate
        derivative.coupon_rate = 0.0
        price_zero = self._get_price(derivative, process,
                                     current_market=self.market,
                                     pre_simulated_paths=pre_simulated_paths)

        # 2. Price with C = 1 (unit coupon)
        derivative.coupon_rate = 1.0
        price_one = self._get_price(derivative, process,
                                    current_market=self.market,
                                    pre_simulated_paths=pre_simulated_paths)

        # 3. Solve: target = price_zero + C * (price_one - price_zero)
        pv_per_unit = price_one - price_zero
        if abs(pv_per_unit) < 1e-12:
            derivative.coupon_rate = original_coupon
            raise ValueError("Coupon has zero sensitivity — product may have no coupon component.")
        coupon = (target_price - price_zero) / pv_per_unit
        derivative.coupon_rate = coupon
        return coupon

    elif method == "bisection":
        # --- Legacy bisection solver (unchanged) ---
        lower_bound = 0.0
        upper_bound = 50.0
        for _ in range(max_iter):
            mid_coupon = (lower_bound + upper_bound) / 2.0
            derivative.coupon_rate = mid_coupon
            price = self._get_price(derivative, process,
                                    current_market=self.market,
                                    pre_simulated_paths=pre_simulated_paths)
            if abs(price - target_price) < epsilon:
                return mid_coupon
            if price < target_price:
                lower_bound = mid_coupon
            else:
                upper_bound = mid_coupon
        return mid_coupon
```

#### Test Case: Analytical Coupon Round-Trip (All Structured Note Types)

```python
def test_analytical_coupon_roundtrip_phoenix():
    """Verify that the analytically solved coupon, when plugged back into the
    pricing engine with the SAME paths, reproduces the target price exactly."""
    # ... setup market, process, Phoenix derivative ...
    engine = CallableMCPricingEngine(market, settings)

    # Step 1: Solve for coupon analytically
    coupon = engine.get_coupon(derivative, process, method="analytical", target_price=100.0)

    # Step 2: Price with the solved coupon using the SAME paths
    scheme = EulerScheme()
    paths = scheme.simulate_paths(process, engine.nb_paths, engine.random_seed)
    derivative.coupon_rate = coupon
    reprice = engine._get_price(derivative, process, current_market=market,
                                pre_simulated_paths=paths)

    # Must match target to machine precision (same paths = zero MC noise)
    assert np.isclose(reprice, 100.0, atol=1e-6), (
        f"Round-trip price {reprice:.8f} != target 100.0 for coupon={coupon:.6f}"
    )


def test_analytical_coupon_roundtrip_eagle():
    """Same round-trip test for Eagle product."""
    # ... setup Eagle derivative ...
    # Same assertion logic as Phoenix test above


def test_analytical_vs_bisection_agreement():
    """Verify that analytical and bisection methods converge to the same coupon."""
    # ... setup ...
    coupon_analytical = engine.get_coupon(derivative, process, method="analytical")
    coupon_bisection = engine.get_coupon(derivative, process, method="bisection", epsilon=1e-4)

    assert np.isclose(coupon_analytical, coupon_bisection, atol=0.05), (
        f"Analytical coupon {coupon_analytical:.4f} vs bisection {coupon_bisection:.4f}"
    )


def test_coupon_linearity_holds():
    """Verify that the Autocall price is indeed a linear function of coupon rate.
    If linearity holds, any 3 points (C1, P1), (C2, P2), (C3, P3) must be collinear."""
    # ... setup with pre_simulated_paths ...
    coupons = [0.0, 3.0, 7.0, 15.0]
    prices = []
    for c in coupons:
        derivative.coupon_rate = c
        p = engine._get_price(derivative, process, current_market=market,
                              pre_simulated_paths=paths)
        prices.append(p)

    # Check collinearity: slope between consecutive points must be constant
    slopes = [(prices[i+1] - prices[i]) / (coupons[i+1] - coupons[i])
              for i in range(len(coupons) - 1)]
    for i in range(len(slopes) - 1):
        assert np.isclose(slopes[i], slopes[i+1], rtol=1e-6), (
            f"Non-linear! Slope {slopes[i]:.6f} vs {slopes[i+1]:.6f}"
        )
```

---

## 3. MC Theta Calculation Noise (`mc_pricing_engine.py`)

### A. Maturity-Bump Theta Breaks Grid Alignment (Especially for Barrier/Autocall Products)

> [!WARNING]
> **Issue:** In `_theta`, the current implementation computes Theta via:
> ```python
> deriv_bumped.maturity -= dt_bump          # maturity shrinks by 1 day
> process_bumped = self.get_stochastic_process(deriv_bumped, market)
> price_bumped = self._get_price(deriv_bumped, process_bumped, market)
> theta = (price_bumped - price) / dt_bump
> ```
> When `maturity` changes but `nb_steps` stays constant, every time step $dt$ shifts by a tiny amount. This causes the simulated paths to diverge from the base paths even when using the same random seed, because $\sqrt{dt}$ changes. For vanilla options this introduces moderate noise. For barrier and autocall products it is **catastrophic**: a tiny path shift can flip a barrier breach on/off, causing massive payoff jumps that dominate the finite difference.

> [!CAUTION]
> **A previously proposed "Elapsed-One-Day" fix was mathematically WRONG — do not use it.** That approach kept `maturity` / `nb_steps` / `dt` identical and instead advanced the spot by one day of carry ($S_0^{bumped} = S_0 e^{r\,dt}$). Because time-to-maturity $\tau$ is **not** reduced, the finite difference measures
> $$\frac{V(S_0 e^{r\,dt},\ \tau) - V(S_0,\ \tau)}{dt} \approx \Delta \cdot S_0 \cdot r,$$
> i.e. **delta × carry, not Theta**. For an ATM 1Y call ($S=100,r=5\%,\Delta\approx0.6$) it returns ≈ **+3/yr** while the true Theta is ≈ **−6.5/yr** (wrong sign and magnitude). Crucially, a *stability* test (low CV across seeds) does **not** catch this — a stable-but-wrong number passes. Theta requires a **correctness** test (below).

> [!TIP]
> **Correct fix — Common-Random-Numbers (CRN) forward difference.** Theta is $-\partial V/\partial\tau$ at **fixed** $S_0$, so we must genuinely reduce time-to-maturity while holding $S_0$ and the randomness fixed. The trick that removes grid noise without distorting the estimate:
> 1. Simulate the base paths once on the base grid (`nb_steps` steps of size `dt = maturity/nb_steps`).
> 2. **Reuse those exact paths, dropping the last time column:** `bumped_paths = base_paths[:, :-1]`. Because Euler builds each path forward from $S_0$, the first `nb_steps-1` columns *are* the process over $[0, \tau - dt]$ with **identical increments and identical $S_0$** — this is CRN by construction, zero re-simulation, zero $\sqrt{dt}$ mismatch.
> 3. Re-price the derivative on `bumped_paths` with `maturity = τ - dt`, then `theta = (price_bumped - price) / dt`.
>
> This is correct (τ reduced, $S_0$ fixed), noise-controlled (barrier flips only where genuine), and needs no `deepcopy(market)` or rate-curve shift. Note `dt_bump` is now one grid step (`maturity/nb_steps`), not exactly 1/365; report Theta per year by dividing by that `dt`. Use a reasonably large `nb_steps` so the forward-difference step is small.

#### Pre-Test: Demonstrate Theta Instability (Current Implementation)

```python
def test_theta_instability_current_method():
    """DIAGNOSTIC: Show that the current maturity-bump Theta is unstable across seeds.

    Run Theta calculation with 10 different seeds and measure the standard deviation.
    For a barrier option, the std should be very large relative to the mean,
    proving the method is unreliable.
    """
    from kernel.products.options.barrier_options import UpAndOutCallOption

    barrier_opt = UpAndOutCallOption(maturity=1.0, strike=100.0, barrier=120.0)
    # ... setup market with S0=100, sigma=30%, r=5% ...

    thetas = []
    for seed in range(10):
        engine = MCPricingEngine(market, settings)
        engine.random_seed = seed
        # Use the current _theta which bumps maturity
        price = engine._get_price(barrier_opt, process, market)
        theta = engine._theta(price, delta, gamma, vega, barrier_opt, market)
        thetas.append(theta)

    std_theta = np.std(thetas)
    mean_theta = np.mean(thetas)
    coeff_of_variation = abs(std_theta / mean_theta) if abs(mean_theta) > 1e-8 else float('inf')

    # BEFORE FIX: coefficient of variation should be very high (> 50%)
    # documenting the instability for the record
    print(f"Current method — Mean Theta: {mean_theta:.4f}, Std: {std_theta:.4f}, "
          f"CV: {coeff_of_variation:.2%}")
    # We expect this to be unstable (CV > 0.5)
    assert coeff_of_variation > 0.3, (
        f"Expected high instability (CV > 30%), got CV={coeff_of_variation:.2%}. "
        f"Test may need recalibration."
    )
```

#### Proposed Code Change

```python
def _theta(self, price, delta, gamma, vega, derivative, market):
    S = market.underlying_asset.last_price
    r = market.get_rate(1 / 365)

    is_vanilla = isinstance(derivative, (EuropeanCallOption, EuropeanPutOption))
    if self.model.name == "BLACK_SCHOLES" and is_vanilla:
        # Analytical BS PDE theta (unchanged)
        K = getattr(derivative, "strike", S)
        sigma = market.get_volatility(K, derivative.maturity)
        theta = -0.5 * sigma**2 * S**2 * gamma - r * S * delta + r * price

    elif self.model.name == "HESTON" or not is_vanilla:
        # Need at least 2 steps so that dropping one column leaves a valid grid.
        if self.nb_steps < 2:
            return 0.0

        dt_grid = derivative.maturity / self.nb_steps   # one grid step (the bump size)

        # --- Common-Random-Numbers forward difference (CORRECT + low noise) ---
        # 1. Simulate base paths once with the SAME seed used for `price`.
        scheme = EulerScheme()
        base = scheme.simulate_paths(process=self.get_stochastic_process(derivative, market),
                                     nb_paths=self.nb_paths, seed=self.random_seed)
        base_paths = getattr(base, "spot_paths", base)   # SimulationResult-safe (post two-factor)

        # 2. Reuse the exact same paths but drop the last time column:
        #    these first (nb_steps-1) columns ARE the process over [0, tau - dt]
        #    at fixed S0 with identical increments -> zero grid noise, true CRN.
        bumped_paths = base_paths[:, :-1]

        # 3. Re-price with time-to-maturity reduced by exactly one grid step.
        deriv_bumped = copy.deepcopy(derivative)
        deriv_bumped.maturity = derivative.maturity - dt_grid
        price_bumped = self._get_price(deriv_bumped, self.get_stochastic_process(deriv_bumped, market),
                                       current_market=market, pre_simulated_paths=bumped_paths)

        # Theta = -dV/dtau = (V(tau - dt) - V(tau)) / dt
        theta = (price_bumped - price) / dt_grid
    else:
        raise ValueError("Model not supported for calculating theta.")

    return theta
```

> [!NOTE]
> `pre_simulated_paths=bumped_paths` reuses the base draws, so the `process` argument is only needed to satisfy the signature (it is not re-simulated). Both `MCPricingEngine._get_price` and `AmericanMCPricingEngine._get_price` accept `pre_simulated_paths`, so this works for exotics and American products alike. After the two-factor merge, `bumped_paths` is a raw spot array (from `.spot_paths[:, :-1]`); `_get_price`'s `getattr(x, "spot_paths", x)` unpacking handles it either way.

#### Post-Test 1: Theta Stability After Fix (necessary but NOT sufficient)

```python
def test_theta_stability_crn_method():
    """Theta must be stable across seeds under the CRN method (low CV)."""
    from kernel.products.options.barrier_options import UpAndOutCallOption

    barrier_opt = UpAndOutCallOption(maturity=1.0, strike=100.0, barrier=120.0)
    # ... setup market with S0=100, sigma=30%, r=5%, and a reasonably large nb_steps (e.g. 100) ...

    thetas = []
    for seed in range(10):
        engine = MCPricingEngine(market, settings)
        engine.random_seed = seed
        price = engine._get_price(barrier_opt, process, market)
        theta = engine._theta(price, delta, gamma, vega, barrier_opt, market)
        thetas.append(theta)

    cv = abs(np.std(thetas) / np.mean(thetas)) if abs(np.mean(thetas)) > 1e-8 else float('inf')
    assert cv < 0.15, f"Expected stable Theta (CV < 15%), got CV={cv:.2%}"
```

#### Post-Test 2: Theta CORRECTNESS (this is the test the old plan was missing) 🔢

Stability alone is worthless if the value is wrong. Force the exotic/Heston branch but under a **degenerate Heston that equals Black-Scholes** (vol-of-vol = 0, $v_0=\theta$), so the CRN Theta can be checked against the closed-form BS Theta.

```python
def test_theta_correctness_vs_analytic_bs():
    """CRN Theta on a European call priced under (degenerate) Heston must match analytical BS Theta.
    Guards against the 'delta*carry instead of theta' class of bug."""
    from kernel.products.options.vanilla_options import EuropeanCallOption
    from scipy.stats import norm

    S0, K, T, r, v0 = 100.0, 100.0, 1.0, 0.05, 0.09   # sigma_BS = 0.30
    # Model.HESTON forces the elif branch (NOT the analytical BS-vanilla branch).
    # vol-of-vol=0, v0=theta => constant variance => equals Black-Scholes.
    # ... build engine with Model.HESTON, nb_steps>=100, nb_paths large, and a Heston process
    #     whose kappa*(theta-v)=0 path keeps v==v0 (sigma_volvol=0) ...

    call = EuropeanCallOption(maturity=T, strike=K)
    price = engine._get_price(call, process, market)
    theta_mc = engine._theta(price, delta, gamma, vega, call, market)

    # Analytical Black-Scholes call theta (per year, negative for a call):
    sig = np.sqrt(v0)
    d1 = (np.log(S0/K) + (r + 0.5*sig**2)*T) / (sig*np.sqrt(T))
    d2 = d1 - sig*np.sqrt(T)
    theta_bs = (-(S0*sig*norm.pdf(d1))/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2))

    # Must at minimum have the correct SIGN and be within a reasonable band.
    assert np.sign(theta_mc) == np.sign(theta_bs), f"Theta sign wrong: mc={theta_mc:.3f}, bs={theta_bs:.3f}"
    assert abs(theta_mc - theta_bs) < 1.0, f"Theta off: mc={theta_mc:.3f}, bs={theta_bs:.3f}"
```

#### Post-Test 3: Vanilla BS branch unchanged (guard)

```python
def test_theta_vanilla_unchanged():
    """Vanilla European under Model.BLACK_SCHOLES still uses the analytical BS PDE formula (untouched path)."""
    from kernel.products.options.vanilla_options import EuropeanCallOption
    call = EuropeanCallOption(maturity=1.0, strike=100.0)
    # ... setup BS model ...
    theta = engine._theta(price, delta, gamma, vega, call, market)
    assert np.isclose(theta, analytical_bs_theta, atol=0.5)
```

---

# Clarity Review

Issues found during a full read-through of this document that should be clarified before implementation begins.

## 1. Ambiguous `_get_price` Signature Between Engines

The `callable_mc_pricing_engine.py` calls `self._get_price(derivative, process, current_market=self.market, pre_simulated_paths=...)`. But `_get_price` in the base `MCPricingEngine` has a different default behavior for `current_market` (it defaults to `None` and uses `self.market` internally). The proposed analytical coupon solver code passes `current_market=self.market` explicitly. This is correct, but the document should note:

> [!NOTE]
> Both the analytical and bisection branches must pass `current_market=self.market` to `_get_price` when using `pre_simulated_paths`, because the derivative's `get_discounted_payoff` needs the market's discount curve. The `AmericanMCPricingEngine._get_price` raises an error if `current_market` is `None`, so this pattern is already enforced there but not in the base class.

## 2. Phoenix Proposed Code — Missing Maturity Handling

The proposed Phoenix fix (Section "Products Payoff Review §1A") shows the loop body but omits the maturity handling block (`# --- Paths still active at maturity ---`). This block also needs the same `pv_cumulative_coupons` treatment:
- `above_capital` payoff should be `100.0 * df_final + pv_cumulative_coupons[above_capital] + missed_coupons[above_capital] * df_final`
- `below_capital` payoff must also add `pv_cumulative_coupons[below_capital]`

The current proposed snippet is incomplete without this — the implementer might miss it.

## 3. Theta — resolved by the CRN method (obsolete concern)

An earlier draft used an "elapsed-one-day" spot bump and raised a concern about whether the rate curve should also be shifted forward. **That concern is now moot:** the corrected **CRN method** (see Pricing Engines Review §3) reuses the base simulation's own paths (`base_paths[:, :-1]`) and only reduces `maturity` by one grid step. No spot bump, no `deepcopy(market)`, no rate-curve shift is involved, so there is no such approximation to reconcile. The only residual approximation is the standard forward-difference truncation error, controlled by using a sufficiently large `nb_steps` (small `dt`).

## 4. Document Structure

The file has grown to cover 4 separate review topics (Euler scheme, Volatility surfaces, Pricing engines, Products payoff) but uses inconsistent heading hierarchy. Consider renaming the top-level `#` headings to clearly separate them:
- `# Part 1: Euler Discretization Upgrade`
- `# Part 2: Volatility Surface Models Review`
- `# Part 3: Pricing Engines Review`
- `# Part 4: Products Payoff Review`
- `# Part 5: Implementation Roadmap` (this section, below)

This is cosmetic but helps future readers navigate the 800+ line document.

---

# Implementation Roadmap

> [!CAUTION]
> All changes in this document touch **core mathematical modeling code**. A single misplaced sign, missing Ito correction, or broken backward-compatibility will silently corrupt every price in the system. The phased approach below is designed so that **each phase is independently testable and reversible**. Never proceed to the next phase until all gate tests pass with zero failures.

## Guiding Principles

1. **One concern per phase.** Each phase modifies exactly one logical component. If something breaks, you know exactly where.
2. **Test before and after.** Run the full existing test suite (`uv run pytest tests/ -v`) before touching any code. Record the baseline. After each phase, run the full suite again and diff.
3. **Git branch per phase.** Each phase gets its own feature branch (e.g., `fix/phoenix-coupon-discount`). Merge to `main` only after gate tests pass. This way any phase can be reverted independently.
4. **No silent regressions.** If an existing test tolerance needs to change (e.g., `atol=0.05` → `atol=0.03`), the change must be documented in the commit message with the old vs. new observed value.

---

## Phase 0: Baseline Snapshot (No Code Changes)

**Goal:** Record the current state of all tests so we can detect any regression.

**Actions:**
- [ ] Run `uv run pytest tests/ -v` and save the full output to `tests/baseline_results.txt`
- [ ] Record the count: X passed, Y failed, Z skipped
- [ ] Commit this baseline file

**Gate Criteria:**
- All currently-passing tests are documented
- Any currently-failing tests are noted (they must not get worse)

**Estimated Effort:** 5 minutes

---

## Phase 1: Products Payoff Fixes (Isolated Product Changes)

**Why first:** These are self-contained changes touching only product definition files (`autocall_products.py`, `participation_products.py`). They do NOT touch the simulation engine. If they break, only specific products are affected — and we have tests to catch them immediately.

**Branch:** `fix/product-payoff-bugs`

**Actions:**
- [ ] **Autocall:** Modify `Phoenix.get_discounted_payoff` to track `pv_cumulative_coupons` and discount at payment date.
- [ ] **Participation:** Fix `TwinWin` and `Airbag` by using `paths[:, -1]` and rewriting `if/elif` statements with `np.where`.
- [ ] Add `test_phoenix_coupon_discount_timing` to `tests/test_structured_products.py`
- [ ] Add vectorized execution tests for `TwinWin` and `Airbag` in `tests/test_vectorized_payoffs.py`
- [ ] Run existing structured product tests to verify no regressions

**Gate Tests (must ALL pass before proceeding):**
```bash
# 1. New product tests pass
uv run pytest tests/test_structured_products.py tests/test_vectorized_payoffs.py -v

# 2. Full regression — nothing else broke
uv run pytest tests/ -v
```

**Estimated Effort:** 2-3 hours

---

## Phase 2: Analytical Coupon Solver (Isolated Engine Change)

**Why second:** This only touches `callable_mc_pricing_engine.py`. It adds a new method (`method="analytical"`) while keeping the existing bisection code completely untouched. Zero risk of regression — the old code path is preserved byte-for-byte.

**Branch:** `fix/analytical-coupon-solver`

**Actions:**
- [ ] Modify `get_coupon` in `callable_mc_pricing_engine.py`:
  - Add `method` parameter (default `"analytical"`)
  - Implement the analytical solver branch
  - Keep bisection branch unchanged
- [ ] Add `test_coupon_linearity_holds` to `tests/test_structured_products.py`
- [ ] Add `test_analytical_coupon_roundtrip_phoenix` to `tests/test_structured_products.py`
- [ ] Add `test_analytical_coupon_roundtrip_eagle` to `tests/test_structured_products.py`
- [ ] Add `test_analytical_vs_bisection_agreement` to `tests/test_structured_products.py`

**Gate Tests:**
```bash
# 1. Linearity proof (this can run even BEFORE the analytical solver is written)
uv run pytest tests/test_structured_products.py::test_coupon_linearity_holds -v

# 2. Round-trip tests
uv run pytest tests/test_structured_products.py -k "coupon" -v

# 3. Full regression
uv run pytest tests/ -v
```

**Estimated Effort:** 1-2 hours

---

## Phase 3: Theta CRN Fix (Isolated Engine Change)

**Why third:** This only touches the `_theta` method in `mc_pricing_engine.py`. The analytical BS PDE path (vanilla options) is completely unchanged. Only the Heston/exotic branch is modified.

**Branch:** `fix/theta-crn`

**Actions:**
- [ ] **Pre-test first (before any code change):** Write and run `test_theta_instability_current_method` to record the baseline CV. Commit the test with the recorded values.
- [ ] Modify `_theta` in `mc_pricing_engine.py`:
  - Replace `deriv_bumped.maturity -= dt_bump` (+ re-simulation) with the **CRN forward difference**: reduce `maturity` by one grid step, reuse `base_paths[:, :-1]` via `pre_simulated_paths`.
  - Do **NOT** use the delta×carry "elapsed-one-day" spot bump (it computes the wrong Greek — see Pricing Engines Review §3 CAUTION).
- [ ] Write `test_theta_stability_crn_method` (post-fix stability test)
- [ ] Write `test_theta_correctness_vs_analytic_bs` (**correctness** test — degenerate Heston vs analytical BS Theta; this is the one that catches the delta×carry bug)
- [ ] Write `test_theta_vanilla_unchanged` (guard test)

**Gate Tests:**
```bash
# 1. Pre-test proves the old method was indeed unstable (run BEFORE code change)
uv run pytest tests/test_mc_engine_greeks.py::test_theta_instability_current_method -v

# 2. After fix: stability test passes
uv run pytest tests/test_mc_engine_greeks.py::test_theta_stability_crn_method -v

# 2b. After fix: CORRECTNESS test passes (right sign + right magnitude)
uv run pytest tests/test_mc_engine_greeks.py::test_theta_correctness_vs_analytic_bs -v

# 3. Vanilla Theta unchanged
uv run pytest tests/test_mc_engine_greeks.py::test_theta_vanilla_unchanged -v

# 4. All existing Greeks tests still pass
uv run pytest tests/test_mc_engine_greeks.py -v

# 5. Full regression
uv run pytest tests/ -v
```

**Estimated Effort:** 2-3 hours

---

## Phase 4: Log-Euler Discretization Upgrade (Core Math — Highest Risk)

**Why fourth (not first):** This is the most dangerous change because it modifies the simulation engine that feeds into every single pricing path in the system. By doing Phases 1-3 first, we:
- Have already fixed independent bugs that could mask or amplify Log-Euler issues
- Have a clean, fully-passing test suite as our regression baseline
- Can isolate any Phase 4 failures as purely Log-Euler related

**Branch:** `feature/log-euler-scheme`

**Sub-phase 4a: Infrastructure (zero behavioral change)**
- [ ] Add `is_log_process` flag to `StochasticProcess.__init__` (default `True`)
- [ ] Add explicit `is_log_process=True` to `BlackScholesProcess.__init__`
- [ ] Add explicit `is_log_process=True` to `HestonProcess.__init__`
- [ ] **Run full test suite — must be IDENTICAL to Phase 3 baseline** (no behavioral change)

**Sub-phase 4b: One-Factor Log-Euler (Black-Scholes only)**
- [ ] Modify `_simulate_one_factor` in `euler_scheme.py` with `is_log_process` branch
- [ ] Write `test_log_euler_exact_distribution` (log-normal mean/variance check)
- [ ] Write `test_no_negative_prices_extreme_vol` (positivity under σ=200%)
- [ ] Write `test_step_size_invariance` (BS price stable across nb_steps)
- [ ] Add `assert np.all(paths > 0)` to existing `test_euler_one_factor`
- [ ] Run the new `test_no_negative_prices_extreme_vol` pytest → expect 0 negatives (passes successfully)
- [ ] Run `test_matlab_sanity.py` — observe if `atol` needs tightening
- [ ] Run `test_mc_engine_greeks.py` — observe if tolerances shift

**Sub-phase 4c: Two-Factor Log-Euler (Heston)**
- [ ] Modify `_simulate_two_factor` in `euler_scheme.py` with `is_log_process` branch (spot only)
- [ ] Add `assert np.all(paths > 0)` to existing `test_euler_two_factor`
- [ ] Run all Heston-related tests

**Gate Tests (after all sub-phases):**
```bash
# 1. New Log-Euler specific tests
uv run pytest tests/test_log_euler.py -v

# 2. Euler scheme shape/positivity tests
uv run pytest tests/test_models.py::TestEulerScheme -v

# 3. Pricing regression — prices should improve or stay within tolerance
uv run pytest tests/test_matlab_sanity.py -v
uv run pytest tests/test_mc_engine_greeks.py -v
uv run pytest tests/test_american_engine.py -v

# 4. Structured products (must still work with new paths)
uv run pytest tests/test_structured_products.py tests/test_vectorized_payoffs.py tests/test_simulation_grids.py -v

# 5. Full regression — ZERO failures
uv run pytest tests/ -v
```

**Estimated Effort:** 3-4 hours

---

## Phase 5: Tolerance Re-calibration & Documentation

**Why last:** Only after all math changes are in place can we observe the final steady-state accuracy and calibrate tolerances.

**Branch:** `chore/recalibrate-tolerances`

**Actions:**
- [ ] Compare new test outputs against Phase 0 baseline
- [ ] For each test where tolerance changed:
  - Document old value, new value, and the reason (e.g., "Log-Euler eliminates discretization bias, MC call price moved from 10.42 to 10.45, closer to BS analytical 10.4506")
  - Tighten `atol` where accuracy improved
- [x] Remove `demonstrate_negative_prices.py` from project root (replaced by `test_no_negative_prices_extreme_vol` in pytest - Completed)
- [ ] Update `change_plan_euler_log_scheme.md` with a "Completed" status for each item

**Gate Tests:**
```bash
# Final full suite with tightened tolerances
uv run pytest tests/ -v
```

**Estimated Effort:** 1 hour

---

## Phase Summary

| Phase | Component | Risk | Files Touched | Dependencies |
|---|---|---|---|---|
| 0 | Baseline snapshot | None | 0 | — |
| 1 | Products Payoff Fixes | Low | 2 products + 2 tests | None |
| 2 | Analytical coupon solver | Low | 1 engine + 1 test | None |
| 3 | Theta elapsed-one-day | Medium | 1 engine + 1 test | None |
| 4 | Log-Euler discretization | **High** | 3 processes + 1 scheme + 3 tests | Phases 1-3 clean |
| 5 | Tolerance re-calibration | Low | Test files only | Phase 4 complete |

> [!IMPORTANT]
> **Critical rule: Phases 1, 2, and 3 are independent of each other and can be done in any order.** Phase 4 must come after all three are merged and passing. Phase 5 must come last.

> [!TIP]
> **Recommended git workflow:**
> ```
> main ← fix/product-payoff-bugs (Phase 1)
>      ← fix/analytical-coupon-solver (Phase 2)
>      ← fix/theta-elapsed-day (Phase 3)
>      ← feature/log-euler-scheme (Phase 4, after 1-3 merged)
>      ← chore/recalibrate-tolerances (Phase 5, after 4 merged)
> ```
