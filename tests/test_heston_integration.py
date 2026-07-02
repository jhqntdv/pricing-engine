import pytest
import numpy as np
from kernel.market_data.market import Market
from kernel.models.stochastic_processes.heston_process import HestonProcess
from kernel.models.stochastic_processes.black_scholes_process import BlackScholesProcess
from kernel.models.pricing_engines.mc_pricing_engine import MCPricingEngine
from kernel.models.pricing_engines.american_mc_pricing_engine import AmericanMCPricingEngine
from kernel.models.pricing_engines.callable_mc_pricing_engine import CallableMCPricingEngine
from kernel.products.options.vanilla_options import EuropeanCallOption, EuropeanPutOption
from kernel.products.options.american_options import AmericanPutOption
from kernel.products.options.barrier_options import UpAndOutCallOption
from kernel.products.options.path_dependent_options import AsianCallOption
from kernel.products.structured_products.autocall_products import Phoenix
from kernel.tools import ObservationFrequency
from utils.pricing_settings import PricingSettings, Model
from scipy.stats import norm
from unittest.mock import patch

class DummyUnderlying:
    def __init__(self, spot):
        self.last_price = spot

class DummyMarket(Market):
    def __init__(self, spot=100.0, rate=0.05, vol=0.20):
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

class DummyHestonModel:
    def __init__(self, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5, v0=0.04):
        self.name = "HESTON"
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.v0 = v0

def test_heston_degenerates_to_bs():
    """Test 1: Heston degenerates to Black-Scholes (structural sanity)."""
    S0, K, T, r, v0 = 100.0, 100.0, 1.0, 0.05, 0.04
    market = DummyMarket(spot=S0, rate=r)
    settings = PricingSettings(nb_paths=50000, nb_steps=100, random_seed=42)
    # sigma=0, v0=theta means constant variance
    settings.model = DummyHestonModel(kappa=2.0, theta=v0, sigma=0.0, rho=0.0, v0=v0)
    
    engine = MCPricingEngine(market, settings)
    call = EuropeanCallOption(maturity=T, strike=K)
    res = engine.get_result(call)
    
    sig = np.sqrt(v0)
    d1 = (np.log(S0/K) + (r + 0.5*sig**2)*T) / (sig*np.sqrt(T))
    d2 = d1 - sig*np.sqrt(T)
    bs_price = S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    
    assert abs(res.price - bs_price) < 3 * res.std_dev

def test_put_call_parity_heston():
    """Test 2: Put-Call Parity under Heston."""
    S0, K, T, r = 100.0, 100.0, 1.0, 0.05
    market = DummyMarket(spot=S0, rate=r)
    settings = PricingSettings(nb_paths=50000, nb_steps=100, random_seed=42)
    settings.model = DummyHestonModel()
    
    engine = MCPricingEngine(market, settings)
    call_res = engine.get_result(EuropeanCallOption(maturity=T, strike=K))
    put_res = engine.get_result(EuropeanPutOption(maturity=T, strike=K))
    
    parity = S0 - K * np.exp(-r * T)
    combined_std = np.sqrt(call_res.std_dev**2 + put_res.std_dev**2)
    assert abs(call_res.price - put_res.price - parity) < 3 * combined_std

def test_vanilla_european_convergence():
    """Test 3: Vanilla European vs. Semi-Closed-Form (convergence)."""
    S0, K, T, r = 100.0, 100.0, 1.0, 0.05
    market = DummyMarket(spot=S0, rate=r)
    settings = PricingSettings(nb_paths=100000, nb_steps=250, random_seed=42)
    model = DummyHestonModel(kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5, v0=0.04)
    settings.model = model
    
    # Feller condition check
    assert 2 * model.kappa * model.theta > model.sigma ** 2
    
    engine = MCPricingEngine(market, settings)
    res = engine.get_result(EuropeanCallOption(maturity=T, strike=K))
    
    # Hardcoded benchmark for these standard parameters: S0=100, K=100, T=1, r=0.05, 
    # kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5, v0=0.04
    # The true analytical price is approx 10.432.
    benchmark_price = 10.432
    # Because full-truncation Euler has a small discretization bias at 250 steps (~0.03), 
    # we allow 3 * std_dev + bias_allowance
    bias_allowance = 0.05 
    assert abs(res.price - benchmark_price) < 3 * res.std_dev + bias_allowance

def test_american_put_2d_regression():
    """Test 4: American Put 2-D regression is accurate and better/equal to 1-D."""
    S0, K, T, r = 100.0, 100.0, 1.0, 0.05
    market = DummyMarket(spot=S0, rate=r)
    # Use out of sample seed for pricing
    settings = PricingSettings(nb_paths=50000, nb_steps=50, random_seed=99)
    model = DummyHestonModel(kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5, v0=0.04)
    settings.model = model
    
    am_put = AmericanPutOption(maturity=T, strike=K)
    eu_put = EuropeanPutOption(maturity=T, strike=K)
    
    # 2D Engine
    engine_2d = AmericanMCPricingEngine(market, settings)
    res_2d = engine_2d.get_result(am_put)
    
    # 1D Engine by forcing variance_paths = None
    from kernel.models.discretization_schemes.euler_scheme import EulerScheme
    original_sim = EulerScheme.simulate_paths
    def mock_sim(self, *args, **kwargs):
        res = original_sim(self, *args, **kwargs)
        res.variance_paths = None
        return res
        
    with patch('kernel.models.pricing_engines.american_mc_pricing_engine.EulerScheme.simulate_paths', autospec=True, side_effect=mock_sim):
        engine_1d = AmericanMCPricingEngine(market, settings)
        res_1d = engine_1d.get_result(am_put)
        
    engine_eu = MCPricingEngine(market, settings)
    res_eu = engine_eu.get_result(eu_put)

    tol = 3 * res_2d.std_dev
    
    # 2D price >= 1D price - tol
    assert res_2d.price >= res_1d.price - tol
    
    # American >= European
    assert res_2d.price >= res_eu.price - tol
    
    # American >= Intrinsic
    assert res_2d.price >= max(K - S0, 0)

def test_bs_american_unchanged():
    """Test 5: Black-Scholes American unchanged (regression guard)."""
    # Golden value computed from the unmodified engine (seed=42, nb_paths=100000, nb_steps=50)
    golden_price = 6.024691435813217 
    market = DummyMarket(spot=100.0, rate=0.05, vol=0.2)
    settings = PricingSettings(nb_paths=100000, nb_steps=50, random_seed=42)
    settings.model = Model.BLACK_SCHOLES
    
    engine = AmericanMCPricingEngine(market, settings)
    am_put = AmericanPutOption(maturity=1.0, strike=100.0)
    res = engine.get_result(am_put)
    
    assert np.isclose(res.price, golden_price, atol=1e-12)

def test_path_dependent_and_barrier_under_heston():
    """Test 6: Path-Dependent & Barrier Options complete under Heston."""
    S0, K, T, r = 100.0, 100.0, 1.0, 0.05
    market = DummyMarket(spot=S0, rate=r)
    settings = PricingSettings(nb_paths=10000, nb_steps=50, random_seed=42)
    settings.model = DummyHestonModel()
    
    engine = MCPricingEngine(market, settings)
    
    asian = AsianCallOption(maturity=T, strike=K)
    res_asian = engine.get_result(asian)
    assert np.isfinite(res_asian.price) and res_asian.price >= 0
    
    barrier = UpAndOutCallOption(maturity=T, strike=K, barrier=120.0)
    res_barrier = engine.get_result(barrier)
    assert np.isfinite(res_barrier.price) and res_barrier.price >= 0
    
    vanilla = EuropeanCallOption(maturity=T, strike=K)
    res_vanilla = engine.get_result(vanilla)
    assert res_barrier.price <= res_vanilla.price + 1e-12

def test_structured_products_autocalls_heston():
    """Test 7: Structured Products (Autocalls) complete under Heston."""
    S0, T, r = 100.0, 1.0, 0.05
    market = DummyMarket(spot=S0, rate=r)
    settings = PricingSettings(nb_paths=10000, nb_steps=50, random_seed=42)
    settings.compute_callable_coupons = True
    settings.model = DummyHestonModel()
    
    engine = CallableMCPricingEngine(market, settings)
    phoenix = Phoenix(
        maturity=T,
        observation_frequency=ObservationFrequency.MONTHLY,
        capital_barrier=60,
        autocall_barrier=100,
        coupon_barrier=80,
        coupon_rate=0.0
    )
    phoenix.initial_spot = S0
    
    res = engine.calculate_structured_product(phoenix)
    assert hasattr(res, "coupon_callable")
    assert np.isfinite(res.coupon_callable)
    assert res.coupon_callable != 0.0
