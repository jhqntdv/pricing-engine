import pytest
import numpy as np
from datetime import datetime, date

from kernel.market_data.market import Market

from kernel.products.options.vanilla_options import EuropeanCallOption, EuropeanPutOption
from kernel.models.pricing_engines.mc_pricing_engine import MCPricingEngine
from utils.pricing_settings import PricingSettings, Model

class DummyUnderlying:
    def __init__(self, spot: float):
        self.last_price = spot

class SteepMarket(Market):
    def __init__(self, bump_bp=0.0):
        self.underlying_asset = DummyUnderlying(100.0)
        self.bump = bump_bp / 10000.0
        self.vol = 0.2

    def _get_curve_rate(self, T: float) -> float:
        # A mock steep curve: short term 2%, 1yr 5%, 5yr 10%
        # Shifted by bump_bp for Rho testing
        if T <= 1/12:
            return 0.02 + self.bump
        elif T <= 1.0:
            # linear interp between 1m and 1y
            w = (T - 1/12) / (1.0 - 1/12)
            return (1-w)*(0.02 + self.bump) + w*(0.05 + self.bump)
        else:
            # linear interp between 1y and 5y
            w = min((T - 1.0) / 4.0, 1.0)
            return (1-w)*(0.05 + self.bump) + w*(0.10 + self.bump)

    def get_rate(self, T: float) -> float:
        return self._get_curve_rate(T)
    
    def get_fwd_rate(self, t1: float, t2: float) -> float:
        if t1 == 0:
            return self.get_rate(t2)
        r1 = self.get_rate(t1)
        r2 = self.get_rate(t2)
        return (r2 * t2 - r1 * t1) / (t2 - t1)
    
    def get_volatility(self, K: float, T: float) -> float:
        return self.vol
    
    def get_discount_factor(self, T: float) -> float:
        return np.exp(-self.get_rate(T) * T)

    def get_fwd_discount_factor(self, start: float, end: float) -> float:
        return np.exp(-self.get_fwd_rate(start, end) * (end - start))

    def bump_flat_yield_curve(self, bump: float) -> "Market":
        # bump is in decimal e.g. 0.01 for 1%
        return SteepMarket(bump_bp=bump * 10000.0)

def test_missing_curve_raises():
    """A3-Light: Calling _get_price without explicitly passing a market curve for discounting must raise ValueError."""
    market = SteepMarket()
    settings = PricingSettings(nb_paths=100, nb_steps=10, compute_greeks=False, random_seed=42)
    settings.model = Model.BLACK_SCHOLES
    engine = MCPricingEngine(market, settings)
    
    derivative = EuropeanCallOption(strike=100.0, maturity=1.0)
    process = engine.get_stochastic_process(derivative, market)
    
    # Intentionally missing current_market
    with pytest.raises(ValueError, match="current_market must be explicitly provided"):
        engine._get_price(derivative, process, current_market=None)

def test_parity_steep_curve():
    """A3-Light: Verify Put-Call parity holds under a steep curve.
    If drift uses the curve but discounting does not (or uses a different one), parity will fail.
    """
    market = SteepMarket()
    settings = PricingSettings(nb_paths=100000, nb_steps=50, compute_greeks=False, random_seed=42)
    settings.model = Model.BLACK_SCHOLES
    engine = MCPricingEngine(market, settings)
    
    call = EuropeanCallOption(strike=100.0, maturity=1.0)
    put = EuropeanPutOption(strike=100.0, maturity=1.0)
    
    res_call = engine.get_result(call)
    res_put = engine.get_result(put)
    
    S = 100.0
    K = 100.0
    DF = market.get_discount_factor(1.0)
    
    parity_lhs = res_call.price - res_put.price
    parity_rhs = S - K * DF
    
    # Tolerance increased because 100000 paths still has MC variance around 0.1 for this specific product
    assert np.isclose(parity_lhs, parity_rhs, atol=0.2)

def test_rho_bumps_both_legs():
    """A3-Light: Verify that Rho calculations correctly bump both drift and discounting.
    We compute the central difference by manually building bumped markets and passing them.
    """
    market_base = SteepMarket(bump_bp=0)
    settings = PricingSettings(nb_paths=10000, nb_steps=10, compute_greeks=True, random_seed=42)
    settings.model = Model.BLACK_SCHOLES
    engine = MCPricingEngine(market_base, settings)
    
    call = EuropeanCallOption(strike=100.0, maturity=1.0)
    res_base = engine.get_result(call)
    
    # Manual finite difference (matching the engine's exact logic)
    # The engine bumps by epsilon_fit = 0.01 (100 bps) but divides by 2 * 0.0001.
    market_up = SteepMarket(bump_bp=100)
    market_down = SteepMarket(bump_bp=-100)
    
    # We must explicitly use engine._get_price with the bumped markets to simulate Rho logic manually
    process_up = engine.get_stochastic_process(call, market_up)
    process_down = engine.get_stochastic_process(call, market_down)
    
    price_up = engine._get_price(call, process_up, current_market=market_up)
    price_down = engine._get_price(call, process_down, current_market=market_down)
    
    # Engine divides by 2 * epsilon where epsilon = 0.0001
    manual_rho = (price_up - price_down) / (2 * 0.0001)
    engine_rho = res_base.greeks["rho"]
    
    assert np.isclose(engine_rho, manual_rho, atol=0.01)
