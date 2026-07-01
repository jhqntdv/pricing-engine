# Two-Factor (Heston) Simulation & LSM Regression Upgrade Plan

## 1. Background & Motivation

Currently, the `EulerScheme` elegantly abstracts the underlying stochastic process (Black-Scholes vs. Heston) by returning only the simulated spot price paths (`paths[:, :, 0]`) as a 2D numpy array. 

While this ensures perfect backward compatibility for standard derivatives (which only care about the spot price), it creates a critical blind spot for **American Options (LSM Engine)**. To make an optimal early exercise decision under stochastic volatility, the Longstaff-Schwartz regression must know the instantaneous variance $v_t$. Because the variance paths are discarded by `EulerScheme`, the American engine is currently forced to regress only on the spot price, leading to sub-optimal exercise boundaries and underpricing.

We will resolve this using **Option A (The `SimulationResult` Object)**, which passes the variance state to the engines that need it, without breaking the derivative payoff functions.

---

## 2. Implementation Plan (Option A: `SimulationResult`)

### A. Define the Data Container
Create a new dataclass to hold simulation results comprehensively.

**Target File:** `kernel/models/discretization_schemes/simulation_result.py` (New File) or append to `euler_scheme.py`
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
Modify `simulate_paths` to return the `SimulationResult` object instead of a raw numpy array.

**Target File:** `kernel/models/discretization_schemes/euler_scheme.py`
- `_simulate_one_factor` returns: `SimulationResult(spot_paths=paths)`
- `_simulate_two_factor` returns: `SimulationResult(spot_paths=paths[:, :, 0], variance_paths=paths[:, :, 1])`

### C. Update Pricing Engines (The API Bridge)
The pricing engines must be updated to unpack the `SimulationResult` before passing the spot paths down to the derivatives. This protects all derivative payoff logic from breaking.

**Target Files:**
1. **`MCPricingEngine`**: 
   ```python
   sim_result = scheme.simulate_paths(...)
   paths = sim_result.spot_paths
   # ... rest of the code is unchanged
   ```
2. **`CallableMCPricingEngine`**: 
   Ensure `pre_simulated_paths` handling supports `SimulationResult` or unpacks it early.
3. **`AmericanMCPricingEngine`**:
   - Extract `paths = sim_result.spot_paths`
   - Extract `var_paths = sim_result.variance_paths`
   - **Regression Logic Update**: When building `x_matrix`, if `var_paths` is not `None`:
     - $X_1 = S_t / K$
     - $X_2 = (S_t / K)^2$
     - $X_3 = v_t$
     - $X_4 = v_t^2$
     - $X_5 = (S_t / K) \times v_t$
     - Perform regression on this expanded 2D basis.

---

## 3. Product Compatibility & Test Coverage Plan

**ALL** derivatives in the codebase are fundamentally compatible with a Heston process, because the payoff of equity derivatives depends entirely on the spot price trajectory, which the Heston process provides.

We must add a dedicated test file to guarantee this integration.

**New File:** `tests/test_heston_integration.py`

### Test 1: Vanilla Options (European)
- **Goal:** Verify that a standard `EuropeanCallOption` prices correctly under `HestonProcess`.
- **Check (Mathematical Exactness):** Price a European Call with standard parameters (e.g., $S_0=100, K=100, T=1.0, r=0.05, v_0=0.04, \kappa=2.0, \theta=0.04, \sigma=0.3, \rho=-0.5$). The test MUST compare the Monte Carlo result against a hardcoded benchmark price derived from the Heston semi-closed form analytical solution (characteristic function integration). Ensure they match within an acceptable MC noise tolerance (`atol=0.05` for 100k paths).

### Test 2: American Options (The Primary Fix)
- **Goal:** Verify that `AmericanMCPricingEngine` successfully unpacks `variance_paths` and executes the 5-term regression, and that this 2D regression is mathematically superior.
- **Check (Mathematical Superiority):** 
  1. Price a Heston `AmericanPutOption` using the upgraded 2D regression (using $S_t, v_t$).
  2. Price the exact same option using a forced 1D regression (only $S_t$).
  3. **Assertion:** The 2D regression price MUST be strictly greater than the 1D regression price (`price_2d > price_1d + noise_margin`). This proves that incorporating the variance state leads to a more optimal early exercise boundary, hence a higher theoretical option value. 

### Test 3: Path-Dependent & Barrier Options
- **Goal:** Prove that modifying the `EulerScheme` return type did not break downstream exotic payoffs.
- **Check:** Run `AsianCallOption` and `UpAndOutCallOption` through the `MCPricingEngine` with `Model.HESTON`. Assert the engine completes successfully.

### Test 4: Structured Products (Autocalls)
- **Goal:** Prove that the `CallableMCPricingEngine` handles `SimulationResult` correctly during its bisection/analytical coupon solving.
- **Check:** Price a `Phoenix` product using `HestonProcess`. Assert that it successfully returns a non-zero price and completes the coupon calculation.

---

## 4. Execution Guidelines
This task should be executed **after** the core Log-Euler and Product Payoff fixes (Phase 1-5 of the previous plan) are merged to `main`, to avoid merge conflicts on the Pricing Engines and Euler Scheme files.
