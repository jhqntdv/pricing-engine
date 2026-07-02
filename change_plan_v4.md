# Change Plan V4: Advanced Optimization & Codebase Hardening

This document outlines the findings from the deep-dive technical audit of the pricing engine, categorizing potential mathematical and performance optimizations, and providing concrete test case designs to prevent scaling or logic errors in the future.

## 1. Findings and Suggestions

### 1.1 Greeks Computation Engine (Finite Differences)
- **Severity**: Medium (Optimization / Math Issue)
- **Finding**: The Monte Carlo engine computes Greeks using a naive finite difference (bump-and-reprice) method with a hardcoded absolute bump size (`epsilon = 1.0` for spot, `0.01` for vol). For underlying assets with large spot prices (e.g., SPX at 5768), `1.0` is microscopically small. For discontinuous payoffs (Barrier, Digital, Autocall), this tiny bump causes exactly zero paths to cross the barrier differently, resulting in `0.0000` for Delta, Gamma, and Vega.
- **Suggestion**: 
  - Change `_spot_eps` to scale dynamically with the spot price (e.g., `epsilon = S0 * 1e-4`).
  - *Advanced*: Implement **Pathwise Derivatives** or the **Likelihood Ratio Method (LRM)** for discontinuous exotic products, avoiding finite difference noise entirely.

### 1.2 Heston Process Variance Simulation Scheme
- **Severity**: Medium (Optimization / Math Issue)
- **Finding**: The codebase currently uses the "Full Truncation" Euler scheme (`np.maximum(v, 0)`) for the variance dimension to prevent negative variance. While safe, it induces a positive bias in the variance process, especially for large time steps (`dt`) or when the Feller condition ($2\kappa\theta > \sigma^2$) is violated.
- **Suggestion**: Upgrade the variance discretization to the **Quadratic-Exponential (QE) scheme** (Andersen, 2008). This drastically improves convergence and eliminates the truncation bias.

### 1.3 American Option Regression Basis (Longstaff-Schwartz)
- **Severity**: Low (Optimization)
- **Finding**: The 2D regression basis currently relies on raw monomials (powers of $S$ and $v$ up to degree 2). Raw monomials can lead to severe multicollinearity and ill-conditioned matrices in `np.linalg.lstsq` when higher polynomial degrees are used.
- **Suggestion**: Replace raw monomials with **Orthogonal Polynomials** (e.g., Laguerre or Hermite polynomials). Allow the user to configure the polynomial degree dynamically.

### 1.4 Quasi-Monte Carlo (Sobol Sequences)
- **Severity**: Low (Not Implemented)
- **Finding**: `PricingSettings` includes a `random_generator_type = "SOBOL"`, but the engine still heavily relies on NumPy's standard pseudo-random number generator, meaning the convergence rate remains at standard Monte Carlo levels: $O(1/\sqrt{N})$.
- **Suggestion**: Integrate `scipy.stats.qmc.Sobol` and map it into the `AbstractRandomGenerator` interface to achieve faster $O(1/N)$ convergence rates.

### 1.5 Callable Coupon Solver
- **Severity**: Low (Optimization)
- **Finding**: The analytical coupon solver in `CallableMCPricingEngine` evaluates exactly two points (`coupon=0` and `coupon=1.0`) and extrapolates. This assumes a perfectly linear relationship. While true for standard Autocalls, it will break if future products include conditional reinvestments or complex compounding.
- **Suggestion**: Upgrade the solver to use **Secant** or **Brent's method**, guaranteeing rapid convergence even for non-linear structures.

### 1.6 Loop Vectorization and Naming Conventions
- **Severity**: None (Optimal As-Is)
- **Finding**: 
  - Time-stepping `for` loops in Euler simulation and Longstaff-Schwartz backward induction are fundamentally sequential (Markov property) and mathematically cannot be vectorized over time. 
  - Autocall payoff evaluation loops have a very small time dimension ($N \le 12$) and run fast; fully vectorizing them over time would bloat memory allocation and reduce cache locality.
  - Naming conventions use PEP 8 strictly, with standard industry exceptions for quantitative finance variables (e.g., `S0`, `T`, `K`, `dW`), which is considered a Best Practice.
- **Suggestion**: Leave these as-is.

---

## 2. Test Cases to Address Finite Difference Greeks Error

To prevent silent failures where Autocalls or Barrier options report `0.0000` for Greeks and Standard Deviation due to parameter scaling errors (e.g., passing `1.0` instead of `100.0` for a barrier when paths are 100-based), the following explicit test cases must be added.

### Test Case Design: `test_autocall_greeks_validity`
**Location**: `tests/test_mc_engine_greeks.py`

```python
import numpy as np
from kernel.market_data import Market
from kernel.tools import ObservationFrequency, Model
from utils.pricing_settings import PricingSettings
from kernel.products.structured_products.autocall_products import Phoenix
from kernel.models.pricing_engines.callable_mc_pricing_engine import CallableMCPricingEngine

def test_autocall_greeks_and_stddev_not_zero():
    """
    Ensures that structured products (like Phoenix) yield non-zero standard deviations 
    and valid Greeks, catching cases where extreme parameters (e.g., 1-based instead of 100-based barriers)
    force all paths into a single fixed deterministic payoff.
    """
    # 1. Setup Market and Settings
    market = DummyMarket(spot=100.0, rate=0.05)
    settings = PricingSettings(
        nb_paths=5000, 
        nb_steps=50, 
        random_seed=42, 
        compute_greeks=True,
        model=Model.BLACK_SCHOLES
    )
    
    # 2. Setup a standard Autocall with correctly scaled (100-based) parameters
    phoenix = Phoenix(
        maturity=3.0,
        observation_frequency=ObservationFrequency.SEMIANNUAL,
        capital_barrier=70.0,    # 100-based
        autocall_barrier=100.0,  # 100-based
        coupon_barrier=80.0,     # 100-based
        coupon_rate=8.0
    )
    
    # 3. Price the product
    engine = CallableMCPricingEngine(market, settings)
    engine.compute_coupon = False # We want the price and Greeks, not solving for coupon
    res = engine.get_result(phoenix)
    
    # 4. Assertions
    # If std_dev is exactly 0.0, every path yielded identical results (likely a barrier scale error)
    assert res.std_dev is not None and res.std_dev > 0.0, \
        f"StdDev is {res.std_dev}. Payoff collapsed to a single deterministic value!"
        
    # Greeks should be computed and non-zero
    assert res.greeks is not None, "Greeks dictionary is missing!"
    
    delta = res.greeks.get("delta", 0.0)
    gamma = res.greeks.get("gamma", 0.0)
    vega = res.greeks.get("vega", 0.0)
    
    # Under normal market conditions, an Autocall has non-zero risk exposure
    assert abs(delta) > 1e-6, f"Delta is practically zero: {delta}"
    assert abs(vega) > 1e-6, f"Vega is practically zero: {vega}"
```

### Action Items for Next Development Phase
1. Add `test_autocall_greeks_and_stddev_not_zero` into the regression test suite.
2. Refactor `_spot_eps()` in `MCPricingEngine` to return `self.market.underlying_asset.last_price * 0.001` (i.e., a proportional 10 bps bump) instead of the hardcoded `1.0`.
