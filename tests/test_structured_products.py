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
