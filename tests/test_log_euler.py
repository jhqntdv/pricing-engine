import pytest
import numpy as np
from kernel.models.stochastic_processes.black_scholes_process import BlackScholesProcess
from kernel.models.stochastic_processes.heston_process import HestonProcess
from kernel.models.discretization_schemes.euler_scheme import EulerScheme


def test_log_euler_exact_distribution():
    """Log-Euler with 1 step must produce exact log-normal terminal distribution."""
    S0, T, r, sigma = 100.0, 1.0, 0.05, 0.30
    nb_paths = 500_000

    process = BlackScholesProcess(S0=S0, T=T, nb_steps=1, drift=np.array([r]), volatility=sigma)
    scheme = EulerScheme()
    paths = scheme.simulate_paths(process, nb_paths, seed=42).spot_paths

    log_ST = np.log(paths[:, -1])
    expected_mean = np.log(S0) + (r - 0.5 * sigma**2) * T
    expected_var = sigma**2 * T

    assert np.isclose(np.mean(log_ST), expected_mean, atol=0.01)
    assert np.isclose(np.var(log_ST), expected_var, atol=0.01)


def test_no_negative_prices_extreme_vol():
    """Even with 200% vol and dt=1.0, all paths must remain strictly positive."""
    process = BlackScholesProcess(S0=100.0, T=1.0, nb_steps=1, drift=np.array([0.0]), volatility=2.0)
    scheme = EulerScheme()
    paths = scheme.simulate_paths(process, nb_paths=100_000, seed=42).spot_paths

    assert np.all(paths > 0), f"Found {np.sum(paths <= 0)} non-positive prices"


def test_step_size_invariance():
    """BS Log-Euler price should be stable across different step counts."""
    S0, T, r, sigma, K = 100.0, 1.0, 0.05, 0.30, 100.0

    prices = []
    for steps in [1, 10, 50, 252]:
        process = BlackScholesProcess(S0=S0, T=T, nb_steps=steps, drift=np.array([r]*steps), volatility=sigma)
        scheme = EulerScheme()
        paths = scheme.simulate_paths(process, nb_paths=200_000, seed=42).spot_paths
        payoffs = np.maximum(paths[:, -1] - K, 0) * np.exp(-r * T)
        prices.append(np.mean(payoffs))

    # All prices should be within MC noise of each other (SE of diff is ~0.06, use 4*SE)
    assert max(prices) - min(prices) < 0.25


def test_risk_neutral_martingale_bs():
    """E[S_T] must equal S0 * exp(rT) — proves the -0.5*sigma^2 Ito term is correct."""
    S0, T, r, sigma = 100.0, 1.0, 0.05, 0.40
    nb_paths = 500_000
    process = BlackScholesProcess(S0=S0, T=T, nb_steps=50, drift=np.array([r]*50), volatility=sigma)
    scheme = EulerScheme()
    paths = scheme.simulate_paths(process, nb_paths, seed=42).spot_paths
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
    process = HestonProcess(S0=S0, v0=0.04, T=T, nb_steps=250, drift=np.array([r]*250),
                            kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5)
    scheme = EulerScheme()
    paths = scheme.simulate_paths(process, nb_paths, seed=42).spot_paths
    ST = paths[:, -1]
    expected = S0 * np.exp(r * T)
    se = np.std(ST, ddof=1) / np.sqrt(nb_paths)
    # Full-truncation Euler introduces a small martingale bias; allow a modest multiple of SE.
    assert abs(np.mean(ST) - expected) < 5 * se


def test_put_call_parity_log_euler():
    """C - P = S0 - K*exp(-rT). Holds path-by-path; the most robust pipeline regression test."""
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.30
    nb_paths = 200_000
    process = BlackScholesProcess(S0=S0, T=T, nb_steps=50, drift=np.array([r]*50), volatility=sigma)
    scheme = EulerScheme()
    paths = scheme.simulate_paths(process, nb_paths, seed=42).spot_paths
    ST = paths[:, -1]
    df = np.exp(-r * T)
    call = np.mean(np.maximum(ST - K, 0)) * df
    put  = np.mean(np.maximum(K - ST, 0)) * df
    assert np.isclose(call - put, S0 - K * df, atol=0.02)


def test_heston_degenerates_to_bs():
    """vol-of-vol=0 and v0=theta => constant variance => Heston spot must match BS log-Euler exactly.
    This simultaneously validates that the Log-Euler spot step and the (later) SimulationResult change agree."""
    S0, K, T, r, v0 = 100.0, 100.0, 1.0, 0.05, 0.04
    nb_paths = 200_000
    heston = HestonProcess(S0=S0, v0=v0, T=T, nb_steps=100, drift=np.array([r]*100),
                           kappa=2.0, theta=v0, sigma=0.0, rho=0.0)   # sigma(vol-of-vol)=0, v0=theta
    scheme = EulerScheme()
    paths = scheme.simulate_paths(heston, nb_paths, seed=42).spot_paths
    price_heston = np.mean(np.maximum(paths[:, -1] - K, 0)) * np.exp(-r * T)

    # Black-Scholes closed form with sigma = sqrt(v0)
    from scipy.stats import norm
    sig = np.sqrt(v0)
    d1 = (np.log(S0/K) + (r + 0.5*sig**2)*T) / (sig*np.sqrt(T))
    d2 = d1 - sig*np.sqrt(T)
    bs = S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    assert abs(price_heston - bs) < 0.05


def test_raw_euler_fallback_when_not_log_process():
    """With is_log_process=False, the scheme must use additive Raw Euler (can go negative)."""
    process = BlackScholesProcess(S0=100.0, T=1.0, nb_steps=1, drift=np.array([0.0]), volatility=2.0)
    process.is_log_process = False          # force the fallback branch
    scheme = EulerScheme()
    paths = scheme.simulate_paths(process, nb_paths=100_000, seed=42).spot_paths
    # Raw Euler with 200% vol and dt=1.0 is expected to breach zero — proving the branch is live.
    assert np.any(paths <= 0), "Fallback branch did not behave like Raw Euler (expected some non-positive paths)"
