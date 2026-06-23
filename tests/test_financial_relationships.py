"""
Comprehensive test suite verifying financial and mathematical relationships
for various option types and their numerical Greeks.

These tests rely on logical boundaries, monotonicity, and parity relationships
rather than analytical formulas to verify the robustness of Monte Carlo engines.
"""

import numpy as np
import pytest
from kernel.market_data.market import Market
from utils.pricing_settings import PricingSettings
from kernel.models.pricing_engines.mc_pricing_engine import MCPricingEngine
from kernel.models.pricing_engines.american_mc_pricing_engine import AmericanMCPricingEngine
from kernel.products.options.vanilla_options import EuropeanCallOption, EuropeanPutOption
from kernel.products.options.path_dependent_options import (
    AsianCallOption, AsianPutOption,
    LookbackCallOption, LookbackPutOption,
    ForwardStartCallOption, ForwardStartPutOption,
    ChooserOption
)
from kernel.products.options.barrier_options import (
    DownAndInCallOption, DownAndOutCallOption
)
from kernel.products.options.american_options import AmericanCallOption
from kernel.products.options.binary_options import AssetOrNothingCallOption


# ---------------------------------------------------------------------------
# Dummy Market Setup
# ---------------------------------------------------------------------------
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


class DummyModel:
    def __init__(self):
        self.name = "BLACK_SCHOLES"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@pytest.fixture
def mc_engine():
    market = DummyMarket()
    settings = PricingSettings()
    settings.nb_paths = 10000
    settings.nb_steps = 100
    settings.random_seed = 42
    settings.compute_greeks = True
    settings.model = DummyModel()
    return MCPricingEngine(market, settings)

@pytest.fixture
def american_engine():
    market = DummyMarket()
    settings = PricingSettings()
    settings.nb_paths = 10000
    settings.nb_steps = 100
    settings.random_seed = 42
    settings.compute_greeks = False
    settings.model = DummyModel()
    return AmericanMCPricingEngine(market, settings)


# ===========================================================================
# 1. Vanilla Bounds and Relationships
# ===========================================================================
def test_call_strike_monotonicity(mc_engine):
    """Lower strike call > Higher strike call"""
    call1 = EuropeanCallOption(maturity=1.0, strike=90.0)
    call2 = EuropeanCallOption(maturity=1.0, strike=110.0)
    res1 = mc_engine.calculate_option(call1)
    res2 = mc_engine.calculate_option(call2)
    assert res1.price > res2.price

def test_time_value_monotonicity(mc_engine):
    """Longer maturity > Shorter maturity for ATM call"""
    call1 = EuropeanCallOption(maturity=0.5, strike=100.0)
    call2 = EuropeanCallOption(maturity=1.5, strike=100.0)
    res1 = mc_engine.calculate_option(call1)
    res2 = mc_engine.calculate_option(call2)
    assert res2.price > res1.price

def test_put_call_parity_with_engine(mc_engine):
    """C - P = S - K * exp(-rT)"""
    T = 1.0
    K = 100.0
    S = mc_engine.market.underlying_asset.last_price
    r = mc_engine.market.get_rate(T)
    call = EuropeanCallOption(maturity=T, strike=K)
    put = EuropeanPutOption(maturity=T, strike=K)
    c_price = mc_engine.calculate_option(call).price
    p_price = mc_engine.calculate_option(put).price
    expected_diff = S - K * np.exp(-r * T)
    assert abs((c_price - p_price) - expected_diff) < 0.5


# ===========================================================================
# 2. Exotics vs Vanilla
# ===========================================================================
def test_asian_vs_vanilla(mc_engine):
    """Asian Call <= European Call"""
    asian = AsianCallOption(maturity=1.0, strike=100.0)
    vanilla = EuropeanCallOption(maturity=1.0, strike=100.0)
    p_asian = mc_engine.calculate_option(asian).price
    p_vanilla = mc_engine.calculate_option(vanilla).price
    assert p_asian <= p_vanilla

def test_lookback_vs_vanilla(mc_engine):
    """Lookback Call >= European Call"""
    lookback = LookbackCallOption(maturity=1.0, strike=100.0)
    vanilla = EuropeanCallOption(maturity=1.0, strike=100.0)
    p_lookback = mc_engine.calculate_option(lookback).price
    p_vanilla = mc_engine.calculate_option(vanilla).price
    assert p_lookback >= p_vanilla

def test_chooser_vs_max_call_put(mc_engine):
    """Chooser Option >= max(Call, Put)"""
    chooser = ChooserOption(maturity=1.0, strike=100.0, chooser_time=0.5)
    call = EuropeanCallOption(maturity=1.0, strike=100.0)
    put = EuropeanPutOption(maturity=1.0, strike=100.0)
    
    p_chooser = mc_engine.calculate_option(chooser).price
    p_call = mc_engine.calculate_option(call).price
    p_put = mc_engine.calculate_option(put).price
    
    # Chooser gives optimal choice at t=0.5, so it must be worth at least the max of standalone call and put.
    assert p_chooser >= max(p_call, p_put) - 0.5

def test_asset_or_nothing_vs_vanilla(mc_engine):
    """Asset-or-Nothing Call >= European Call"""
    aon = AssetOrNothingCallOption(maturity=1.0, strike=100.0)
    vanilla = EuropeanCallOption(maturity=1.0, strike=100.0)
    p_aon = mc_engine.calculate_option(aon).price
    p_vanilla = mc_engine.calculate_option(vanilla).price
    assert p_aon >= p_vanilla

def test_barrier_in_out_parity(mc_engine):
    """Down&In + Down&Out = Vanilla Call"""
    di = DownAndInCallOption(maturity=1.0, strike=100.0, barrier=80.0)
    do = DownAndOutCallOption(maturity=1.0, strike=100.0, barrier=80.0)
    vanilla = EuropeanCallOption(maturity=1.0, strike=100.0)
    
    p_di = mc_engine.calculate_option(di).price
    p_do = mc_engine.calculate_option(do).price
    p_vanilla = mc_engine.calculate_option(vanilla).price
    
    assert abs((p_di + p_do) - p_vanilla) < 0.5


# ===========================================================================
# 3. Forward Start Options
# ===========================================================================
def test_forward_start_strike_percentage(mc_engine):
    """Forward Start: lower strike_percentage -> higher price (for calls)"""
    # 0.9 means ITM forward start, 1.0 means ATM forward start
    fwd_itm = ForwardStartCallOption(maturity=1.0, forward_start_time=0.5, strike_percentage=0.9)
    fwd_atm = ForwardStartCallOption(maturity=1.0, forward_start_time=0.5, strike_percentage=1.0)
    
    p_itm = mc_engine.calculate_option(fwd_itm).price
    p_atm = mc_engine.calculate_option(fwd_atm).price
    
    assert p_itm > p_atm


# ===========================================================================
# 4. American vs European
# ===========================================================================
def test_american_vs_european(american_engine, mc_engine):
    """American Call >= European Call"""
    am_call = AmericanCallOption(maturity=1.0, strike=100.0)
    eu_call = EuropeanCallOption(maturity=1.0, strike=100.0)
    
    p_am = american_engine.calculate_option(am_call).price
    p_eu = mc_engine.calculate_option(eu_call).price
    
    # American should be at least European. Might be equal if no dividends.
    assert p_am >= p_eu - 0.2


# ===========================================================================
# 5. Greeks Directional Check
# ===========================================================================
def test_call_greeks_directional(mc_engine):
    """Verify Delta > 0, Gamma > 0, Vega > 0, Theta < 0 for an ATM Call"""
    call = EuropeanCallOption(maturity=1.0, strike=100.0)
    res = mc_engine.calculate_option(call)
    
    assert 0 < res.greeks.get("delta", 0.0) < 1.0, f"Delta {res.greeks.get('delta')} not in (0, 1)"
    assert res.greeks.get("gamma", 0.0) > 0, "Gamma must be positive"
    assert res.greeks.get("vega", 0.0) > 0, "Vega must be positive"
    assert res.greeks.get("theta", 0.0) < 0, "Theta must be negative"
