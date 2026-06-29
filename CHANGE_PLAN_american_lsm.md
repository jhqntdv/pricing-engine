# Change Plan: American LSM Pricing Engine

**Repo:** `jhqntdv/pricing-engine` (`main`)
**Scope:** Correctness, robustness, and hygiene fixes in the Longstaff–Schwartz (LSM) Monte Carlo engine for American/Bermudan options — `kernel/models/pricing_engines/american_mc_pricing_engine.py`.
**Guarantee:** The headline change (L1) **deliberately changes pricing output** — it fixes a discounting bug that currently biases American prices downward. Every other task is behavior-preserving. Because L1 changes numbers, it ships with *invariant-based* tests (analytic/benchmark comparison, two-sided early-exercise premium), not "does it run" checks.

> **Status:** Documentation only. Nothing in this plan has been applied to the code yet.

> **Overlap with `CHANGE_PLAN_engine_design.md`:** Tasks **L2** (honor `pre_simulated_paths`) and **L3** (remove the silent `current_market` fallback) restate, *for this file specifically*, the same defects tracked generally as **A5** and **A3** in the engine-design plan. Do them once; treat whichever plan you execute from as the source of truth and delete the duplicate task in the other to avoid conflicting instructions.

---

## Background — how `_get_price` works today

`american_mc_pricing_engine.py::_get_price` runs a backward-induction LSM:

```python
CF = derivative.intrinsic_payoff(paths[:, -1])              # terminal node N (= nb_steps)
for t in range(self.nb_steps - 2, -1, -1):                  # t = N-2 .. 0
    df_forward = current_market.get_fwd_discount_factor(dt*(t+1), dt*(t+2))
    discounted_CF = CF * df_forward
    CF = discounted_CF.copy()
    if exercise_indices is None or t in exercise_indices:
        immediate = derivative.intrinsic_payoff(paths[:, t])
        in_money = (immediate > 0)
        if np.any(in_money):
            paths_in_money = paths[in_money, t]
            x_matrix = np.column_stack([np.ones(...), paths_in_money, paths_in_money**2])
            coeff, *_ = np.linalg.lstsq(x_matrix, discounted_CF[in_money], rcond=None)
            cont_val = coeff[0] + coeff[1]*paths_in_money + coeff[2]*paths_in_money**2
            exercise = immediate[in_money] >= cont_val
            CF[in_money] = np.where(exercise, immediate[in_money], discounted_CF[in_money])

df_first = current_market.get_discount_factor(dt)
price = np.mean(df_first * CF)
```

Verified market semantics (from `market.py`): `get_discount_factor(t) = exp(-r(t)·t)` is the `[0,t]` factor; `get_fwd_discount_factor(a,b) = DF(b)/DF(a)` is the `[a,b]` factor (discounts a value at `b` back to `a`). Path columns are time-aligned: `paths[:,0] = S0` at `t=0`, `paths[:,N]` at maturity `T = N·dt`.

---

## L1 — Fix the one-step discounting misalignment in backward induction (**changes pricing**)

**Files:** `american_mc_pricing_engine.py` (`_get_price`).

**Problem.** The exercise decision compares values that sit **one time step apart**:

- On the first iteration (`t = N-2`), `df_forward = DF[(N-1)dt, N·dt]`, so `discounted_CF` is valued at node **N-1**. But `immediate` is taken at `paths[:, t] = paths[:, N-2]`, valued at node **N-2**. The regression therefore fits a continuation value living at `t+1` against the spot `S_t` at `t`, and the rule `immediate(t) >= cont_val(t+1)` compares across one step.
- Consequence 1 — the continuation value is **under-discounted (overstated)** relative to immediate → the engine **under-exercises**.
- Consequence 2 — for a path that *does* exercise at node `k`, the realised payoff (fixed at `paths[:,k]`) is subsequently multiplied by one extra step factor. Telescoping the loop factors plus the final `df_first` gives an effective discount of `DF[0,(k+1)dt]` instead of the correct `DF[0,k·dt]` → early-exercise cashflows are **over-discounted by one step**.
- Both effects bias the American price **downward, toward the European value**.
- Secondary defect: the loop starts at `nb_steps - 2`, so the **last pre-maturity node `N-1` is never evaluated for exercise**.

**Why the current tests miss it.** For a path that never exercises early, the loop factors telescope to `DF[dt, N·dt]` and the trailing `df_first = DF[0,dt]` completes a correct total `DF[0,T]`. So the *aggregate* discount magnitude is right; only the *per-step alignment* is wrong. The existing American check is the loose one-sided bound `p_am >= p_eu - 0.2` (see `CHANGE_PLAN_engine_design.md`), and the payoff tests use a flat curve — neither can see a downward bias of this shape.

**Change.** Discount the running cashflow by exactly one step to the **current** node before the exercise decision, so continuation and immediate values are aligned at the same node, and include node `N-1`:

```python
CF = derivative.intrinsic_payoff(paths[:, -1])              # valued at node N
for t in range(self.nb_steps - 1, 0, -1):                   # nodes N-1 .. 1
    df_step = current_market.get_fwd_discount_factor(dt * t, dt * (t + 1))
    CF = CF * df_step                                        # now CF is valued at node t
    if exercise_indices is None or t in exercise_indices:
        immediate = derivative.intrinsic_payoff(paths[:, t])
        in_money = immediate > 0
        if np.any(in_money):
            # ... regress CF[in_money] (valued at t) on basis(S_t) ...
            exercise = immediate[in_money] >= cont_val
            CF[in_money] = np.where(exercise, immediate[in_money], CF[in_money])

price = np.mean(CF * current_market.get_discount_factor(dt))  # node 1 -> 0
```

- Now both sides of `immediate >= cont_val` are valued at node `t`, and an exercised payoff at node `k` is discounted by exactly `DF[0,k·dt]`.
- **Edge case to decide explicitly:** exercise at node `t=0` (the valuation date). Standard LSM reports the holding value at `t=0`; if exercise-at-valuation must be supported, wrap the final value as `max(intrinsic(S0), holding_value)`. State the chosen convention in the docstring (ties into L6).

**Tests & verification** (none of these exist today; all must use a real engine run, not hand-rolled paths):
- `test_american_put_vs_binomial`: price an American put under Black–Scholes and compare to a CRR/binomial-tree reference within MC tolerance. The current code prices **below** the tree; the fix should land within tolerance. This is the decisive regression for the downward bias.
- `test_deep_itm_american_put_equals_intrinsic`: a deep ITM American put under a non-trivial positive rate should price ≈ its immediate intrinsic `K − S0` (optimal to exercise now). The one-step over-discounting mis-prices this; the fixed engine should match `K − S0` tightly.
- `test_early_exercise_premium_positive`: under a **steep** curve, assert `p_am − p_eu` is positive and within a two-sided band around a benchmark — replacing the loose one-sided `>= p_eu − 0.2`.
- `test_exercise_at_last_step_considered`: a Bermudan whose only/optimal exercise date is the last pre-maturity node `N-1` must reflect that exercise (guards the loop-start fix).
- Run the full existing suite — it must stay green; if any existing test asserts a value that depended on the buggy bound (e.g. an American price under `>= p_eu − 0.2`), confirm it is not silently pinning the bug before trusting "all green".

---

## L2 — Honor `pre_simulated_paths` (Liskov; duplicate of engine-plan A5)

**Files:** `american_mc_pricing_engine.py` (`_get_price`).

**Problem.** The override accepts `pre_simulated_paths` but never references it — it always re-simulates ([line 32-33]). The parent `MCPricingEngine._get_price` honors the parameter. Latent today, but a future callable/Bermudan routed through the coupon solver would re-simulate every bisection iteration, reintroducing MC noise and breaking root-finder monotonicity.

**Change.** Mirror the parent:
```python
if pre_simulated_paths is not None:
    paths = pre_simulated_paths
else:
    paths = EulerScheme().simulate_paths(process=stochastic_process, nb_paths=self.nb_paths, seed=self.random_seed)
```

**Tests.** Spy `EulerScheme.simulate_paths`; calling `_get_price` **with** `pre_simulated_paths` must call it **0 times**. (A "same result twice" test will NOT catch this — same seed reproduces the same paths.) See engine-plan A5 for the full test set; do not duplicate.

---

## L3 — Remove the silent `current_market` fallback (duplicate of engine-plan A3-Light)

**Files:** `american_mc_pricing_engine.py` (`_get_price`).

**Problem.** `if current_market is None: current_market = self.market` ([line 29-30]) means a forgotten/unbumped curve yields a silently-wrong price instead of an error — the same failure mode that caused the Rho bug in the base engine.

**Change.** Make the market/curve an explicitly-required argument and fail loudly when missing (align with the base engine's resolution of A3-Light). Keep `test_vectorized_payoffs.py` semantics intact.

**Tests.** `test_american_missing_curve_raises`: the price path without an explicit curve must raise, not fall back. (Coordinate with engine-plan A3 so the base and American engines are fixed consistently.)

---

## L4 — Snap exercise times to the grid with rounding, not truncation

**Files:** `american_mc_pricing_engine.py` (`_get_price`).

**Problem.** `exercise_indices = set(int(t_ex / dt) for t_ex in derivative.exercise_times)` ([line 40]) truncates toward zero, so an exercise date that falls between grid nodes snaps to the **earlier** node and can drift by up to one step (worse for coarse `nb_steps`).

**Change.** Use `int(round(t_ex / dt))` and clip into `[0, nb_steps]`. Document the snapping convention.

**Tests.** `test_exercise_time_snapping`: with a deliberately off-grid exercise date, assert the chosen index is the nearest node. Behavior-affecting only when exercise dates are off-grid; otherwise a no-op covered by the existing suite.

---

## L5 — Numerical stability of the LSM regression basis

**Files:** `american_mc_pricing_engine.py` (`_get_price`).

**Problem.** The design matrix uses raw monomials `[1, S, S²]`. With spot in the ~100 range, `S²` is ~10⁴, so the columns span two orders of magnitude and the matrix is poorly conditioned. `np.linalg.lstsq` (SVD) tolerates this, so it is a **stability**, not a correctness, issue — but coefficients degrade and the problem worsens if the basis is ever extended (S³, S⁴).

**Change.** Normalize the regressor before building the basis — cheapest effective option is **moneyness** `x = S / K` (or `S / S0`), giving columns near unit scale. Alternatively center-and-scale, or switch to orthogonal polynomials (Laguerre/Chebyshev, as in the original Longstaff–Schwartz paper). Keep regression restricted to in-the-money paths (already correct today).

**Tests.** `test_lsm_regression_conditioning` (optional): assert the design-matrix condition number is bounded (e.g. < 1e6) after normalization. Pricing output should be within MC noise of the pre-change value on well-conditioned cases — pair with L1's benchmark test so a basis change can't silently move the price.

---

## L6 — Code hygiene (behavior-preserving)

**Files:** `american_mc_pricing_engine.py`.

- **Unused imports:** remove `pandas as pd`, `ObservationFrequency`, `BlackScholesProcess`, `HestonProcess`, and `AmericanPutOption` (only `AmericanAbstractOption` is used).
- **Wrong class docstring:** the class doc is a copy-paste of a generic MC engine ("classic financial derivatives (no barrier, no asian payoff …)"), which misdescribes an American LSM engine. Rewrite it to describe Longstaff–Schwartz least-squares Monte Carlo and add a docstring to `_get_price` (state the discounting convention from L1 and the t=0 edge case).
- **Naming (PEP8):** `CF` / `discounted_CF` are non-snake_case; worse, `discounted_CF` and the later `discounted_cf` differ only by case and are easy to misread. Rename to distinct lower_snake_case names (e.g. `cashflow`, `cont_cashflow`).

**Tests.** None — covered by the existing suite remaining green (proves hygiene changed nothing).

---

## Suggested execution order

Principle: pin the behavior with a benchmark test *before* the behavior-changing fix; do the risky numeric change in isolation; batch the hygiene last. (No git repo and no CI here, so each checkpoint = `uv run pytest -q`.)

1. **L6 + L4** first (lowest risk: hygiene + a localized rounding fix). Checkpoint: suite green — proves no behavior change.
2. **L2 + L3** next (small, isolated; coordinate with engine-plan A5/A3 so they aren't done twice). Add the spy/raise tests. Checkpoint: green.
3. **L1** — write `test_american_put_vs_binomial` and `test_deep_itm_american_put_equals_intrinsic` **first** (they should *fail* on current code, exposing the downward bias), then apply the discounting fix and the loop-start fix until they pass. Add the steep-curve / last-step tests. This is the only step that legitimately changes numbers.
4. **L5** last — apply the basis normalization behind L1's benchmark test so the price cannot move silently; optionally add the conditioning assertion.

## Risks
- **L1 changes pricing output by design.** The risk is masking the change as "all green" against the existing loose/flat-curve tests — mitigated by the benchmark and two-sided tests written *before* the fix. Confirm no existing test is pinning the buggy value.
- **L2/L3 overlap** with the engine-design plan; doing both creates conflicting edits. Pick one source of truth.
- **L5** can move the price within noise; always pair with L1's benchmark so a regression is visible.
- Everything in L4/L6 must be a no-op on pricing — the existing suite is the regression guard.
