# Change Plan: Docstring Coverage & Error-Handling Consistency

**Repo:** `jhqntdv/pricing-engine` (`main`)
**Scope:** Two non-functional refactors — (1) docstring coverage, (2) error-handling consistency.
**Guarantee:** No pricing numbers change. All 10 existing tests in `tests/` must still pass after each change.

---

## Conventions to follow

- **Docstring style:** Google style (`Args:` / `Returns:` / `Raises:` blocks).
- **Language:** English only.
    - Scope note: `"avec"` is **not** the only non-English content. The codebase contains substantial **Traditional Chinese** *inline comments* (notably the Greeks/Theta blocks in `mc_pricing_engine.py`, e.g. `修正：Vega 應該…`, `用時間有限差分法`), and `README.md` / `examples.py` are written mostly in Traditional Chinese.
    - This plan's "English-ification" applies **to docstrings only**. Translating the Chinese inline comments and the README/examples narration is a *separate* effort that this plan does **not** cover (the only runtime-string fix in scope is the French `"avec"`, Task C-5).
- **Coverage bar:** Every public method, every `@abstractmethod`, and every class must have a docstring. Private (`_`-prefixed) methods only need one when the logic is non-obvious (numerical tricks, discounting, discretization).

Template:
```python
def get_delta(self, derivative: AbstractOption, epsilon: float = 1.0) -> float:
    """Compute the option Delta via central finite difference.

    Args:
        derivative: The option to value.
        epsilon: Spot bump size in price points. Defaults to 1.0.

    Returns:
        The Delta of the option.

    Raises:
        UnsupportedModelError: If the configured model is not supported.
    """
```

---

## Part 1 — Docstring coverage

Current state: ~189 methods total, ~89 missing docstrings (~53% coverage).

### Setup
- **Task 1.0** — First add `ruff` to the dev dependency group in `pyproject.toml` (currently it lists only `pytest`). Then enable the `ruff` `D` rules with `convention = "google"`. Run once with `--exit-zero` to capture the baseline. **CI note:** the repo has no CI today (no `.github/workflows`), so "gate regressions in CI" is not actionable as-is — run `ruff check` locally for now, and *if* a CI pipeline is added later, add the docstring lint to it then.

### Tier 1 — files with 0% coverage (do first)
| File | What to add |
|------|-------------|
| `utils/pricing_settings.py` | Class + `__init__`. Public config object integrators touch directly — do this first. |
| `utils/pricing_results.py` | Class + `lower_bound`, `upper_bound`, `set_greek`, `__str__`, `get_aggregated_results`. This is the source of the JSON output. |
| `utils/day_counter.py` | Class + 5 methods. Also fix the bug in Task C-3. |
| `kernel/models/discretization_schemes/euler_scheme.py` | Class + `simulate_paths`, `_simulate_one_factor`, `_simulate_two_factor`. Document the numerical scheme. |
| `kernel/products/options/american_options.py` | 12 methods + 4 of 5 classes (American/Bermudan call & put). |
| `kernel/products/structured_products/autocall_products.py` | 6 methods (Phoenix/Eagle `__init__` and `get_discounted_payoff`). |
| `kernel/tools.py` | `AbstractRandomGenerator` + Numpy/Sobol generators' `get_standard_normal`. Enums: document only the non-obvious ones. |

### Tier 2 — heavy gaps (classes documented, method overrides missing)
| File | What to add |
|------|-------------|
| `kernel/models/pricing_engines/mc_pricing_engine.py` | All 13 methods, incl. `get_delta/gamma/vega/rho/theta` (document the `epsilon` default's unit/meaning). |
| `kernel/models/pricing_engines/discounting_pricing_engine.py` | All 8 methods, incl. `_price_bond`, `_price_swap`. |
| `kernel/models/pricing_engines/american_mc_pricing_engine.py` | Both methods. |
| `kernel/products/options/path_dependent_options.py` | 11 of 12 (`get_discounted_payoff` / `__init__` overrides). |
| `kernel/products/options/barrier_options.py` | 10 of 12 (all `get_discounted_payoff` overrides). |
| `kernel/products/rate/bond.py` | 4 of 6 (`__init__` ×3, `AbstractBond.payoff`). |
| `kernel/models/stochastic_processes/stochastic_process.py` | 5 of 7 (abstract `get_drift`, `get_vol_drift`, `get_vol_vol`). |

### Tier 3 — small gaps (finish to 100%)
| File | What to add |
|------|-------------|
| `kernel/models/pricing_engines/abstract_pricing_engine.py` | The **4** `@abstractmethod` `calculate_*` methods (`calculate_option`, `calculate_strategy`, `calculate_rate_product`, `calculate_structured_product`). (`_get_price` is also abstract but already documented; `get_results` is concrete and already documented.) **Do these early** — they are the contract every engine implements, and the docstrings should state which exception to raise when unsupported (ties into Part 2). |
| `kernel/models/pricing_engines/callable_mc_pricing_engine.py` | `__init__`, `calculate_structured_product`. |
| `kernel/models/stochastic_processes/black_scholes_process.py` | `get_drift`, `get_volatility`. |
| `kernel/market_data/market.py` | `bump_volatility`, `bump_flat_yield_curve`. |
| `kernel/market_data/data_loader.py` | `__init__`. |
| `kernel/market_data/volatility_surface/local_surface.py` | `__init__`. |
| `kernel/products/options/abstract_option.py` | `__init__`, `accept`. |
| `kernel/products/options/vanilla_options.py` | 2 × `get_discounted_payoff`. |
| `kernel/products/options/binary_options.py` | 2 × `AssetOrNothing*` `get_discounted_payoff`. |
| `kernel/products/options_strategies/abstract_option_strategy.py` | `accept`. |
| `kernel/products/rate/abstract_rate_product.py` | `__init__`. |
| `kernel/products/rate/vanilla_swap.py` | `__init__`, `payoff`. |
| `kernel/products/structured_products/abstract_structured_product.py` | `__init__`, `accept` (see also Task C-4). |
| `kernel/products/structured_products/participation_products.py` | `TwinWin.description`, `Airbag.description`. |
| `kernel/pricing_launcher.py` | `__init__`. |

### Already well-documented — leave as-is
`market.py` (near-complete), `underlying_asset.py`, `rate_curve.py`, `nelson_siegel_interpolator.py`, `svensson_interpolator.py`, `abstract_volatility_surface.py`, `svi_surface.py`, `heston_process.py`, `abstract_derivative.py`, `options_strategies.py`, `multi_assets_options.py`.

---

## Part 2 — Error-handling consistency

Today the codebase mixes `NotImplementedError`, `ValueError`, a hand-raised `ZeroDivisionError`, and an unwrapped `KeyError`. Integrators have no single exception to catch.

### Task 2.0 — Add `kernel/exceptions.py`
Define one hierarchy with a common root. The multiple inheritance from built-ins is deliberate: it keeps existing `except ValueError` / `except KeyError` code working (no breaking change) while letting new code catch `PricingEngineError` for everything.

```text
PricingEngineError(Exception)                                       # common root
├── ConfigurationError
│   ├── UnsupportedModelError(ConfigurationError, ValueError)
│   └── UnsupportedEngineTypeError(ConfigurationError, KeyError)
├── UnsupportedProductError(PricingEngineError, NotImplementedError)
├── InvalidProductInputError(PricingEngineError, ValueError)
├── IndeterminateValuationError(PricingEngineError, ZeroDivisionError)
└── CalibrationError(PricingEngineError)                            # move existing one here
```

### Task 2.1 — Replace existing raises
| Location | Change to |
|----------|-----------|
| `discounting.calculate_option` / `calculate_strategy` / `calculate_structured_product`, and `mc.calculate_rate_product` | `UnsupportedProductError(f"{engine} does not support {product_category}.")` — these are *deliberately unsupported* engine×product combinations, not "TODO". |
| `discounting.calculate_rate_product` (unknown sub-type) | `UnsupportedProductError(...)` with a message that clearly says "unknown rate product sub-type" (distinct from the line above). |
| `mc.get_stochastic_process` | `UnsupportedModelError(...)`. |
| `discounting._price_bond` (×2) | `InvalidProductInputError("You must provide either ytm or the price")`. Keep the original message string verbatim (it includes "the"); both occurrences are at the `ZeroCouponBond` and `CouponBond` branches. |
| `discounting._price_swap` | `IndeterminateValuationError("Annuity is zero; par rate indeterminate")` — replaces the misleading hand-raised `ZeroDivisionError`. |
| `pricing_launcher.calculate` (enum lookup `PricingEngineType[...]`) | Wrap in try/except, re-raise as `UnsupportedEngineTypeError(...)`. Note: in the supported flow `settings.pricing_engine_type` is already a `PricingEngineType` enum member, so `PricingEngineType[...name]` always resolves and a `KeyError` does not actually occur. This wrap is **defensive** — it guards against a wrong/foreign value being passed (which would more likely raise `AttributeError` on `.name` than `KeyError`); the except clause should therefore cover both `KeyError` and `AttributeError`. |
| `abstract_interpolator.CalibrationError` (existing) | Re-parent to `PricingEngineError`. |

### Task 2.2 — Keep `NotImplementedError` only where genuinely "not built yet"
Features the README marks as pending (multi-asset correlation simulation, TwinWin/Airbag vectorization) may keep `NotImplementedError`, but add a `# TODO(issue#):` reference. Everything that is "unsupported by design" moves to `UnsupportedProductError`.

### Task 2.3 — Update README integration section
The relevant README section is written in Traditional Chinese: "## 8. 開發者協作注意事項 (Developer Integration Tips)". The current guidance there says the core model may raise `ValueError` and recommends wrapping in `try-except` at the API layer (`建議在 API 層面套用 try-except 包裝`). Replace that recommendation with: catch `PricingEngineError` for all engine errors, and list the subclasses.
**Language:** the README body is Traditional Chinese — keep the rewrite in Chinese for consistency (do not insert a standalone English paragraph).

### Task 2.4 — Add `tests/test_error_handling.py`
Assert each unsupported combination, invalid input, and unknown engine type raises the correct exception type and message. Include compatibility regression checks: e.g. `UnsupportedModelError` is still caught by `except ValueError`.

---

## Part 3 — Content fixes found during the audit
These are real defects, not just missing docstrings — fix alongside the relevant file.

| ID | File | Issue | Fix |
|----|------|-------|-----|
| C-1 | `ssvi_surface.py` | `SSVIVolatilitySurface` class docstring is empty (`""" """`). | Write real content. |
| C-2 | `multi_assets_options.py` | `BasketPutOption.payoff` ends with the inline comment `# Payoff for a call` even though it is a put (copy-paste from `BasketCallOption`); the docstring is also generic ("Calculates the basket option payoff."). | Fix the trailing comment to say put, and specialize the docstring for the put. |
| C-3 | `day_counter.py` | `get_year_fraction(start_date=datetime.now())` — the default expression is evaluated **once at import** (a stale-timestamp bug), *not* a mutable default. Note: no current caller relies on the default (all callers pass `start_date` explicitly), so this is a latent defect. | Change to `start_date=None`, assign inside the function. Behavior change (bug fix) — cover with a test. |
| C-4 | `abstract_structured_product.py` | `description`'s `@abstractmethod` is commented out, so it's a concrete no-op. | Restore `@abstractmethod`, **provided no concrete subclass relies on the no-op** — verify `Phoenix`, `Eagle`, `TwinWin`, `Airbag` all implement `description`; if any doesn't, give it one first so the abstract restore doesn't break instantiation. |
| C-5 | `abstract_option_strategy.py` | `__str__` returns a string containing the French word `"avec"`. | Change to English. |

---

## Suggested execution order

Principle: build the safety net first, then do the high-risk / dependency-bearing work, and leave the pure-documentation sweep for last. The behavior-changing content fixes (C-3/C-4/C-5) are pulled *out* of the big docstring pass and isolated, so that if the existing tests go red it is obvious which change caused it. (This repo is not a git repo and has no CI, so each "checkpoint" below means `uv run pytest -q` — there is no pipeline to gate on.)

### Phase 0 — Safety net (do first)
1. ~~Run the existing suite and confirm green: `uv run pytest -q`. This is the regression guard for Parts 2 & 3.~~
2. ~~**Task 1.0** — add `ruff` to the dev deps, enable the `D` rules (`convention = "google"`), then `uv run ruff check --select D --exit-zero`. The output is the *precise* missing-docstring list — drive Part 1 from it rather than the "~89" estimate.~~
   - *Rationale:* no logic changes; gives a repeatable baseline before anything moves.

### Phase 1 — Exception foundation (Part 2 core — first, because other steps depend on it)
3. ~~**Task 2.0** — add `kernel/exceptions.py` (the multiple-inheritance hierarchy).~~
4. ~~Tier 3 `abstract_pricing_engine.py` contract docstrings — write the 4 `calculate_*` `Raises:` blocks together with the Part 2 exception semantics.~~
5. ~~**Task 2.1** — swap the raises (engines, launcher, interpolator). First grep existing `except` clauses to confirm the multiple-inheritance hierarchy keeps them working.~~
6. ~~**Task 2.4** — add `tests/test_error_handling.py`, including the backward-compat checks (e.g. `except ValueError` still catches `UnsupportedModelError`). Checkpoint: suite green.~~
7. ~~**Task 2.3** — update the README integration section (keep it in Chinese).~~
   - *Rationale:* the abstract contract docstrings need to state *which* exception is raised, so the exception module must exist first; type changes are the highest-risk item, so pin them with the compat tests earliest.

### Phase 2 — Content fixes (Part 3 — behavior-changing ones isolated, each with its own test)
8. ~~**C-3** `day_counter.py` default → `None` + test. (No caller uses the default today; latent fix.)~~
9. ~~**C-4** `abstract_structured_product.py` (highest risk in this batch): **first** add `description()` to `Phoenix` and `Eagle` (which currently lack it), **then** restore `@abstractmethod`; add an instantiation test for all four subclasses. Grep tests for `Phoenix`/`Eagle` instantiation before changing.~~
10. ~~**C-5** `abstract_option_strategy.py` `__str__` `avec` → English (confirm no test asserts on "avec").~~
11. ~~**C-1 / C-2** (doc/comment only — low risk): SSVI empty class docstring; `BasketPutOption` trailing comment + put-specific docstring.~~
   - *Rationale:* C-3/C-4/C-5 alter behavior; isolating them from the bulk docstring sweep makes any red test trivial to attribute. C-4 ordering is mandatory or `Phoenix`/`Eagle` can no longer be instantiated.

### Phase 3 — Docstring coverage (Part 1 — pure documentation, lowest risk, last)
12. ~~Fill docstrings Tier 1 → Tier 2 → Tier 3, driven by the Phase 0 `ruff D` list. Do C-1/C-2 here if not already done.~~
13. ~~`uv run ruff check --select D` (no `--exit-zero`) → target 0 warnings; `uv run pytest -q` → green.~~
   - *Rationale:* largest volume, zero logic risk; the existing tests guarantee pricing output is unchanged.

### Phase 4 — Close out
14. Final: full suite green + `ruff D` clean.
15. CI deferred — none exists; rely on local `ruff check`. Add docstring lint to CI only if/when a pipeline is created.

## Risks
- Exception-type changes are a potential breaking change — mitigated by the multiple-inheritance hierarchy (Task 2.0) and the compatibility tests (Task 2.4).
- C-3 and C-4 change behavior (a bug fix and an abstract-method restore) — both need their own tests; C-4 must verify all subclasses before restoring.
- Everything else is documentation-only and must not alter pricing output — the existing 10 tests are the regression guard.
