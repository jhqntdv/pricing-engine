import pytest
import numpy as np
from kernel.market_data.market import Market
from kernel.models.pricing_engines.mc_pricing_engine import MCPricingEngine
from kernel.products.options.vanilla_options import EuropeanCallOption, EuropeanPutOption
from utils.pricing_settings import PricingSettings, Model

class DummyUnderlying:
    def __init__(self, spot: float):
        self.last_price = spot

class MatlabFlatMarket(Market):
    """A dummy market that returns flat rates and volatilities matching the MATLAB example."""
    def __init__(self, spot: float, rate: float, vol: float):
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
        return MatlabFlatMarket(self.underlying_asset.last_price, self.rate + bump, self.vol)


def test_matlab_european_options():
    """
    Sanity check against a standard MATLAB Black-Scholes example:
    S0 = 100, K = 95, T = 0.25 (3 months), r = 0.10 (10%), sigma = 0.50 (50%)
    Expected Call = 13.6953
    Expected Put = 6.3497
    """
    # MATLAB example parameters
    S0 = 100.0
    K = 95.0
    T = 0.25
    r = 0.10
    sigma = 0.50
    
    # Expected analytical results from MATLAB
    expected_call = 13.6953
    expected_put = 6.3497

    # Setup the market with the flat parameters
    market = MatlabFlatMarket(spot=S0, rate=r, vol=sigma)
    
    # Setup the engine (Euler scheme needs sufficient steps to approximate GBM accurately)
    settings = PricingSettings(nb_paths=1000000, nb_steps=100, random_seed=42)
    settings.model = Model.BLACK_SCHOLES
    engine = MCPricingEngine(market, settings)

    # Price the Call Option
    call_option = EuropeanCallOption(maturity=T, strike=K)
    process_call = engine.get_stochastic_process(call_option, market)
    call_price = engine._get_price(call_option, process_call, current_market=market)
    
    # Price the Put Option
    put_option = EuropeanPutOption(maturity=T, strike=K)
    process_put = engine.get_stochastic_process(put_option, market)
    put_price = engine._get_price(put_option, process_put, current_market=market)

    # Since it is Monte Carlo, we check within a small tolerance (e.g. 5 cents)
    assert np.isclose(call_price, expected_call, atol=0.05), \
        f"Call price {call_price:.4f} did not match MATLAB {expected_call:.4f}"
        
    assert np.isclose(put_price, expected_put, atol=0.05), \
        f"Put price {put_price:.4f} did not match MATLAB {expected_put:.4f}"
