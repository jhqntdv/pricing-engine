import pytest
from kernel.tools import ObservationFrequency
from kernel.products.structured_products.autocall_products import Phoenix, Eagle
from kernel.products.structured_products.participation_products import TwinWin, Airbag

def test_structured_products_instantiation():
    """
    Verify that all structured products can be successfully instantiated.
    This acts as a regression test to ensure that no subclass fails to implement
    the @abstractmethod `description()` defined in AbstractStructuredProduct.
    """
    try:
        # Phoenix
        p1 = Phoenix(
            maturity=5.0, 
            observation_frequency=ObservationFrequency.ANNUAL,
            capital_barrier=60.0, 
            autocall_barrier=100.0, 
            coupon_rate=5.0, 
            coupon_barrier=80.0
        )
        
        # Eagle
        p2 = Eagle(
            maturity=3.0,
            observation_frequency=ObservationFrequency.SEMIANNUAL,
            capital_barrier=70.0,
            autocall_barrier=105.0,
            coupon_rate=8.0
        )
        
        # TwinWin
        p3 = TwinWin(
            maturity=2.0,
            upper_barrier=120.0,
            lower_barrier=80.0,
            rebate=5.0,
            leverage=2.0
        )
        
        # Airbag
        p4 = Airbag(
            maturity=2.0,
            upper_barrier=120.0,
            lower_barrier=80.0,
            rebate=5.0,
            leverage=2.0
        )
        
        # Ensure description is callable and returns a string
        assert isinstance(p1.description(), str)
        assert isinstance(p2.description(), str)
        assert isinstance(p3.description(), str)
        assert isinstance(p4.description(), str)

    except TypeError as e:
        pytest.fail(f"Failed to instantiate structured product, possibly missing abstract method: {e}")

def test_phoenix_coupon_discount_timing():
    """Verify that Phoenix coupons are discounted at their payment date, not at exit date.

    Setup: A Phoenix that always pays coupons (coupon_barrier=0) and never autocalls
    (autocall_barrier=999). With a steep yield curve, discounting all coupons at
    maturity vs. at each payment date produces a measurable difference.
    """
    import numpy as np
    from kernel.products.structured_products.autocall_products import Phoenix
    from kernel.tools import ObservationFrequency

    class SteepCurveMarket:
        """A market with a 10% flat rate to amplify discounting differences."""
        def get_discount_factor(self, t):
            return np.exp(-0.10 * t)  # 10% continuous rate

    market = SteepCurveMarket()
    maturity = 2.0
    coupon_rate = 5.0  # 5% per period

    phoenix = Phoenix(
        maturity=maturity,
        observation_frequency=ObservationFrequency.QUARTERLY,  # 4x/year = 8 coupons over 2 years
        capital_barrier=0.0,       # never breached (always protected)
        autocall_barrier=999.0,    # never autocalled
        coupon_rate=coupon_rate,
        coupon_barrier=0.0,        # always pays coupon
    )
    phoenix.initial_spot = 100.0

    # Flat paths at 100 (all conditions always met, no autocall)
    nb_paths = 1000
    nb_steps = 252 * 2  # 2 years of daily steps
    paths = np.ones((nb_paths, nb_steps + 1)) * 100.0

    payoffs = phoenix.get_discounted_payoff(paths, market)

    # --- Expected value (analytical) ---
    # 8 quarterly coupons, each discounted at their payment time
    obs_freq = 4  # quarterly
    num_obs = int(round(maturity * obs_freq + 1))  # includes t=0
    expected_pv_coupons = sum(
        coupon_rate * np.exp(-0.10 * (t / obs_freq)) for t in range(1, num_obs)
    )
    expected_pv_principal = 100.0 * np.exp(-0.10 * maturity)
    expected_total = expected_pv_principal + expected_pv_coupons

    # --- Wrong answer (all coupons discounted at maturity) ---
    wrong_total = (100.0 + coupon_rate * (num_obs - 1)) * np.exp(-0.10 * maturity)

    mean_payoff = np.mean(payoffs)
import pytest
from kernel.tools import ObservationFrequency
from kernel.products.structured_products.autocall_products import Phoenix, Eagle
from kernel.products.structured_products.participation_products import TwinWin, Airbag

def test_structured_products_instantiation():
    """
    Verify that all structured products can be successfully instantiated.
    This acts as a regression test to ensure that no subclass fails to implement
    the @abstractmethod `description()` defined in AbstractStructuredProduct.
    """
    try:
        # Phoenix
        p1 = Phoenix(
            maturity=5.0, 
            observation_frequency=ObservationFrequency.ANNUAL,
            capital_barrier=60.0, 
            autocall_barrier=100.0, 
            coupon_rate=5.0, 
            coupon_barrier=80.0
        )
        
        # Eagle
        p2 = Eagle(
            maturity=3.0,
            observation_frequency=ObservationFrequency.SEMIANNUAL,
            capital_barrier=70.0,
            autocall_barrier=105.0,
            coupon_rate=8.0
        )
        
        # TwinWin
        p3 = TwinWin(
            maturity=2.0,
            upper_barrier=120.0,
            lower_barrier=80.0,
            rebate=5.0,
            leverage=2.0
        )
        
        # Airbag
        p4 = Airbag(
            maturity=2.0,
            upper_barrier=120.0,
            lower_barrier=80.0,
            rebate=5.0,
            leverage=2.0
        )
        
        # Ensure description is callable and returns a string
        assert isinstance(p1.description(), str)
        assert isinstance(p2.description(), str)
        assert isinstance(p3.description(), str)
        assert isinstance(p4.description(), str)

    except TypeError as e:
        pytest.fail(f"Failed to instantiate structured product, possibly missing abstract method: {e}")

def test_phoenix_coupon_discount_timing():
    """Verify that Phoenix coupons are discounted at their payment date, not at exit date.

    Setup: A Phoenix that always pays coupons (coupon_barrier=0) and never autocalls
    (autocall_barrier=999). With a steep yield curve, discounting all coupons at
    maturity vs. at each payment date produces a measurable difference.
    """
    import numpy as np
    from kernel.products.structured_products.autocall_products import Phoenix
    from kernel.tools import ObservationFrequency

    class SteepCurveMarket:
        """A market with a 10% flat rate to amplify discounting differences."""
        def get_discount_factor(self, t):
            return np.exp(-0.10 * t)  # 10% continuous rate

    market = SteepCurveMarket()
    maturity = 2.0
    coupon_rate = 5.0  # 5% per period

    phoenix = Phoenix(
        maturity=maturity,
        observation_frequency=ObservationFrequency.QUARTERLY,  # 4x/year = 8 coupons over 2 years
        capital_barrier=0.0,       # never breached (always protected)
        autocall_barrier=999.0,    # never autocalled
        coupon_rate=coupon_rate,
        coupon_barrier=0.0,        # always pays coupon
    )
    phoenix.initial_spot = 100.0

    # Flat paths at 100 (all conditions always met, no autocall)
    nb_paths = 1000
    nb_steps = 252 * 2  # 2 years of daily steps
    paths = np.ones((nb_paths, nb_steps + 1)) * 100.0

    payoffs = phoenix.get_discounted_payoff(paths, market)

    # --- Expected value (analytical) ---
    # 8 quarterly coupons, each discounted at their payment time
    obs_freq = 4  # quarterly
    num_obs = int(round(maturity * obs_freq + 1))  # includes t=0
    expected_pv_coupons = sum(
        coupon_rate * np.exp(-0.10 * (t / obs_freq)) for t in range(1, num_obs)
    )
    expected_pv_principal = 100.0 * np.exp(-0.10 * maturity)
    expected_total = expected_pv_principal + expected_pv_coupons

    # --- Wrong answer (all coupons discounted at maturity) ---
    wrong_total = (100.0 + coupon_rate * (num_obs - 1)) * np.exp(-0.10 * maturity)

    mean_payoff = np.mean(payoffs)

    # The correct answer should be strictly greater than the wrong answer
    # because earlier coupons are worth more when discounted at their own (earlier) time
    assert expected_total > wrong_total, "Sanity check: correct PV > wrong PV"
    # After fix: mean_payoff should match the analytically correct value
    assert np.isclose(mean_payoff, expected_total, atol=0.10), (
        f"Phoenix payoff {mean_payoff:.4f} should match analytical {expected_total:.4f}, "
        f"not the wrong value {wrong_total:.4f}"
    )

def test_coupon_linearity_holds():
    """Prove that structured product price is exactly linear w.r.t the coupon rate."""
    from kernel.products.structured_products.autocall_products import Phoenix
    from kernel.tools import ObservationFrequency
    import numpy as np

    class DummyMarket:
        def get_discount_factor(self, t): return np.exp(-0.05 * t)
        def get_rate(self, t): return 5.0
        def get_fwd_rate(self, t1, t2): return 5.0
        def get_volatility(self, k, t): return 0.2
        class Asset:
            last_price = 100.0
        underlying_asset = Asset()

    phoenix = Phoenix(
        maturity=2.0, observation_frequency=ObservationFrequency.SEMIANNUAL,
        capital_barrier=80.0, autocall_barrier=100.0, coupon_rate=0.0, coupon_barrier=80.0
    )
    phoenix.initial_spot = 100.0
    market = DummyMarket()

    # Generate 100 paths
    np.random.seed(42)
    paths = 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.02, (100, 5)), axis=1))

    # Calculate price at coupon = 0
    p0 = np.mean(phoenix.get_discounted_payoff(paths, market))

    # Calculate price at coupon = 1
    phoenix.coupon_rate = 1.0
    p1 = np.mean(phoenix.get_discounted_payoff(paths, market))

    # Calculate price at coupon = 5
    phoenix.coupon_rate = 5.0
    p5 = np.mean(phoenix.get_discounted_payoff(paths, market))

    # Check linearity: p5 should equal p0 + 5 * (p1 - p0)
    expected_p5 = p0 + 5.0 * (p1 - p0)
    assert np.isclose(p5, expected_p5, atol=1e-12), "Price is not perfectly linear in coupon_rate"

def test_analytical_coupon_roundtrip_phoenix():
    import numpy as np
    from kernel.products.structured_products.autocall_products import Phoenix
    from kernel.tools import ObservationFrequency
    from kernel.models.pricing_engines.callable_mc_pricing_engine import CallableMCPricingEngine
    from utils.pricing_settings import PricingSettings, Model

    class DummyMarket:
        def get_discount_factor(self, t): return np.exp(-0.05 * t)
        def get_rate(self, t): return 5.0
        def get_fwd_rate(self, t1, t2): return 5.0
        def get_volatility(self, k, t): return 0.2
        class Asset:
            last_price = 100.0
        underlying_asset = Asset()

    market = DummyMarket()
    settings = PricingSettings(compute_callable_coupons=True, nb_paths=1000, nb_steps=50, model=Model.BLACK_SCHOLES)
    engine = CallableMCPricingEngine(market, settings)
    
    phoenix = Phoenix(
        maturity=2.0, observation_frequency=ObservationFrequency.SEMIANNUAL,
        capital_barrier=80.0, autocall_barrier=100.0, coupon_rate=0.0, coupon_barrier=80.0
    )
    
    coupon = engine.get_coupon(phoenix, engine.get_stochastic_process(phoenix, market), target_price=100.0, method="analytical")
    phoenix.coupon_rate = coupon
    
    # By setting compute_coupon=False, calculate_structured_product calculates the price instead of solving the coupon.
    engine.compute_coupon = False
    price = engine.calculate_structured_product(phoenix).price
    
    assert np.isclose(price, 100.0, atol=0.1), f"Roundtrip failed: price={price} with coupon={coupon}"

def test_analytical_coupon_roundtrip_eagle():
    import numpy as np
    from kernel.products.structured_products.autocall_products import Eagle
    from kernel.tools import ObservationFrequency
    from kernel.models.pricing_engines.callable_mc_pricing_engine import CallableMCPricingEngine
    from utils.pricing_settings import PricingSettings, Model

    class DummyMarket:
        def get_discount_factor(self, t): return np.exp(-0.05 * t)
        def get_rate(self, t): return 5.0
        def get_fwd_rate(self, t1, t2): return 5.0
        def get_volatility(self, k, t): return 0.2
        class Asset:
            last_price = 100.0
        underlying_asset = Asset()

    market = DummyMarket()
    settings = PricingSettings(compute_callable_coupons=True, nb_paths=1000, nb_steps=50, model=Model.BLACK_SCHOLES)
    engine = CallableMCPricingEngine(market, settings)
    
    eagle = Eagle(
        maturity=2.0, observation_frequency=ObservationFrequency.SEMIANNUAL,
        capital_barrier=80.0, autocall_barrier=100.0, coupon_rate=0.0
    )
    
    coupon = engine.get_coupon(eagle, engine.get_stochastic_process(eagle, market), target_price=100.0, method="analytical")
    eagle.coupon_rate = coupon
    
    engine.compute_coupon = False
    price = engine.calculate_structured_product(eagle).price
    
    assert np.isclose(price, 100.0, atol=0.1), f"Roundtrip failed: price={price} with coupon={coupon}"

def test_analytical_vs_bisection_agreement():
    import numpy as np
    from kernel.products.structured_products.autocall_products import Phoenix
    from kernel.tools import ObservationFrequency
    from kernel.models.pricing_engines.callable_mc_pricing_engine import CallableMCPricingEngine
    from utils.pricing_settings import PricingSettings, Model

    class DummyMarket:
        def get_discount_factor(self, t): return np.exp(-0.05 * t)
        def get_rate(self, t): return 5.0
        def get_fwd_rate(self, t1, t2): return 5.0
        def get_volatility(self, k, t): return 0.2
        class Asset:
            last_price = 100.0
        underlying_asset = Asset()

    market = DummyMarket()
    settings = PricingSettings(compute_callable_coupons=True, nb_paths=1000, nb_steps=50, model=Model.BLACK_SCHOLES)
    engine = CallableMCPricingEngine(market, settings)
    
    phoenix = Phoenix(
        maturity=2.0, observation_frequency=ObservationFrequency.SEMIANNUAL,
        capital_barrier=80.0, autocall_barrier=100.0, coupon_rate=0.0, coupon_barrier=80.0
    )
    
    process = engine.get_stochastic_process(phoenix, market)
    
    coupon_analytical = engine.get_coupon(phoenix, process, target_price=100.0, method="analytical")
    coupon_bisection = engine.get_coupon(phoenix, process, target_price=100.0, method="bisection", epsilon=1e-3, max_iter=50)
    
    assert np.isclose(coupon_analytical, coupon_bisection, atol=0.05), (
        f"Solver disagreement: analytical={coupon_analytical}, bisection={coupon_bisection}"
    )
