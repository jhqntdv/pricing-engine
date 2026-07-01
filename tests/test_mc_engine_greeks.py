import pytest
import numpy as np
from unittest.mock import patch, call
from kernel.market_data.market import Market
from kernel.models.pricing_engines.mc_pricing_engine import MCPricingEngine
from kernel.products.options.vanilla_options import EuropeanCallOption
from kernel.models.stochastic_processes.black_scholes_process import BlackScholesProcess
from utils.pricing_settings import PricingSettings
from kernel.tools import Model
import pandas as pd
from scipy.stats import norm

class DummyUnderlying:
    def __init__(self, spot: float):
        self.last_price = spot

class DummyMarket(Market):
    def __init__(self, spot: float = 100.0, rate: float = 0.05, vol: float = 0.20):
        self.underlying_asset = DummyUnderlying(spot)
        self.rate = rate
        self.vol = vol

    def get_discount_factor(self, maturity: float) -> float:
        return np.exp(-self.rate * maturity)

    def get_fwd_discount_factor(self, start: float, end: float) -> float:
        return np.exp(-self.rate * (end - start))

    def get_rate(self, maturity: float) -> float:
        return self.rate

    def get_fwd_rate(self, start: float, end: float) -> float:
        return self.rate

    def get_volatility(self, strike: float, maturity: float) -> float:
        return self.vol

    def bump_flat_yield_curve(self, bump: float):
        return DummyMarket(self.underlying_asset.last_price, self.rate + bump, self.vol)

def test_greeks_single_source():
    """A1: compute Greeks via get_result(flag=on) and standalone getters, assert bit-for-bit equal."""
    market = DummyMarket()
    
    settings_on = PricingSettings(nb_paths=1000, nb_steps=10, compute_greeks=True, random_seed=42)
    settings_on.model = Model.BLACK_SCHOLES
    engine_on = MCPricingEngine(market, settings_on)
    derivative = EuropeanCallOption(strike=100.0, maturity=1.0)
    
    res = engine_on.get_result(derivative)
    
    settings_off = PricingSettings(nb_paths=1000, nb_steps=10, compute_greeks=False, random_seed=42)
    settings_off.model = Model.BLACK_SCHOLES
    engine_off = MCPricingEngine(market, settings_off)
    
    standalone_delta = engine_off.get_delta(derivative)
    standalone_gamma = engine_off.get_gamma(derivative)
    standalone_vega = engine_off.get_vega(derivative)
    standalone_rho = engine_off.get_rho(derivative)
    standalone_theta = engine_off.get_theta(
        res.price, standalone_delta, standalone_gamma, standalone_vega, derivative, market
    )
    
    assert res.greeks["delta"] == standalone_delta
    assert res.greeks["gamma"] == standalone_gamma
    assert res.greeks["vega"] == standalone_vega
    assert res.greeks["rho"] == standalone_rho
    assert res.greeks["theta"] == standalone_theta

def test_no_greeks_no_work():
    """A1: flag off means no Greeks and minimal simulate_paths calls."""
    market = DummyMarket()
    settings_off = PricingSettings(nb_paths=100, nb_steps=10, compute_greeks=False, random_seed=42)
    settings_off.model = Model.BLACK_SCHOLES
    engine_off = MCPricingEngine(market, settings_off)
    derivative = EuropeanCallOption(strike=100.0, maturity=1.0)
    
    with patch.object(engine_off, "_get_price", wraps=engine_off._get_price) as mock_get_price:
        res = engine_off.get_result(derivative)
        # Check Greeks unset
        assert res.greeks.get("delta") is None
        assert res.greeks.get("gamma") is None
        assert res.greeks.get("vega") is None
        # Should be called exactly once for base price
        assert mock_get_price.call_count == 1

def test_delta_gamma_share_simulations():
    """A1: delta+gamma together trigger 3 simulations (up/down/base)."""
    market = DummyMarket()
    settings_off = PricingSettings(nb_paths=10, nb_steps=2, compute_greeks=False)
    settings_off.model = Model.BLACK_SCHOLES
    engine = MCPricingEngine(market, settings_off)
    derivative = EuropeanCallOption(strike=100.0, maturity=1.0)
    
    base_price = engine._get_price(derivative, engine.get_stochastic_process(derivative, market), market)
    
    with patch("kernel.models.pricing_engines.mc_pricing_engine.EulerScheme.simulate_paths") as mock_sim:
        engine._delta_gamma(derivative, base_price)
        # Should be called exactly twice (up and down)
        assert mock_sim.call_count == 2

def _bs_call_greeks(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    return delta, gamma, vega

def test_finite_difference_vs_analytic():
    """A1: assert MC delta/gamma/vega match closed-form BS Greeks."""
    market = DummyMarket()
    # Need many paths and steps for Euler convergence to analytic
    settings = PricingSettings(nb_paths=100000, nb_steps=50, compute_greeks=True, random_seed=42)
    settings.model = Model.BLACK_SCHOLES
    engine = MCPricingEngine(market, settings)
    derivative = EuropeanCallOption(strike=100.0, maturity=1.0)
    
    res = engine.get_result(derivative)
    
    mc_delta = res.greeks["delta"]
    mc_gamma = res.greeks["gamma"]
    mc_vega = res.greeks["vega"]
    
    S = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.2
    
    bs_delta, bs_gamma, bs_vega = _bs_call_greeks(S, K, T, r, sigma)
    
    # Tolerances are relatively loose due to Monte Carlo noise, but should be close
    assert np.isclose(mc_delta, bs_delta, atol=0.01)
    assert np.isclose(mc_gamma, bs_gamma, atol=0.005)
    # Note: BS vega is typically scaled by 0.01 for a 1% move, but our formula yields the raw derivative.
    # We must ensure mc_vega matches raw or scaled. Our mc_vega gives raw derivative (price_up - price_down) / (2 * 0.01).
    # wait, in mc_pricing_engine: vega = (price_up - price_down) / (2 * 0.01) which approximates dP/dSigma exactly.
    # bs_vega is dP/dSigma. Let's assert raw.
    assert np.isclose(mc_vega, bs_vega, rtol=0.05)
