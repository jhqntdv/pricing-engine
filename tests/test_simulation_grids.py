import pytest
import numpy as np
from kernel.products.options.path_dependent_options import ForwardStartCallOption, ChooserOption
from kernel.products.structured_products.autocall_products import Phoenix, Eagle
from kernel.tools import ObservationFrequency

class DummyMarket:
    def get_discount_factor(self, t):
        return 1.0
    def get_fwd_discount_factor(self, t1, t2):
        return 1.0

def test_grid_frequencies():
    """Verify standard financial simulation steps map to exact indices without crashing."""
    dt_steps = [250, 252, 256, 360, 365, 52, 12, 4, 2, 1]
    maturity = 1.0
    market = DummyMarket()
    
    for steps in dt_steps:
        # Create a mock simulation paths array with shape (10, steps + 1)
        paths = np.ones((10, steps + 1)) * 100.0
        
        # 1. Path Dependent Options
        fsc = ForwardStartCallOption(maturity=maturity, forward_start_time=0.5)
        ch = ChooserOption(maturity=maturity, strike=100.0, chooser_time=0.5)
        
        res_fsc = fsc.get_discounted_payoff(paths, market)
        res_ch = ch.get_discounted_payoff(paths, market)
        
        assert not np.isnan(res_fsc).any()
        assert not np.isnan(res_ch).any()
        assert res_fsc.shape == (10,)
        
        # 2. Autocalls
        phx = Phoenix(
            maturity=maturity, observation_frequency=ObservationFrequency.MONTHLY,
            capital_barrier=80.0, autocall_barrier=100.0, coupon_rate=5.0, coupon_barrier=80.0
        )
        eagle = Eagle(
            maturity=maturity, observation_frequency=ObservationFrequency.MONTHLY,
            capital_barrier=80.0, autocall_barrier=100.0, coupon_rate=5.0
        )
        
        res_phx = phx.get_discounted_payoff(paths, market)
        res_eagle = eagle.get_discounted_payoff(paths, market)
        
        assert not np.isnan(res_phx).any()
        assert not np.isnan(res_eagle).any()
        assert res_phx.shape == (10,)

def test_extreme_array_bounds():
    """Verify that a simulation matrix with only 1 point (t=0) does not raise IndexError."""
    market = DummyMarket()
    
    # 0 steps (only initial spot), shape (10, 1)
    paths_zero_steps = np.ones((10, 1)) * 100.0
    
    fsc = ForwardStartCallOption(maturity=1.0, forward_start_time=0.5)
    phx = Phoenix(
        maturity=1.0, observation_frequency=ObservationFrequency.MONTHLY,
        capital_barrier=80.0, autocall_barrier=100.0, coupon_rate=5.0, coupon_barrier=80.0
    )
    
    # Should not raise IndexError
    res_fsc = fsc.get_discounted_payoff(paths_zero_steps, market)
    assert not np.isnan(res_fsc).any()
    
    res_phx = phx.get_discounted_payoff(paths_zero_steps, market)
    assert not np.isnan(res_phx).any()

def test_zero_initial_spot_nan_protection():
    """Verify that a spot path starting at 0.0 does not produce NaNs in Autocalls."""
    market = DummyMarket()
    
    # Paths where initial spot is strictly 0.0
    paths = np.zeros((10, 252))
    
    phx = Phoenix(
        maturity=1.0, observation_frequency=ObservationFrequency.MONTHLY,
        capital_barrier=80.0, autocall_barrier=100.0, coupon_rate=5.0, coupon_barrier=80.0
    )
    eagle = Eagle(
        maturity=1.0, observation_frequency=ObservationFrequency.MONTHLY,
        capital_barrier=80.0, autocall_barrier=100.0, coupon_rate=5.0
    )
    
    res_phx = phx.get_discounted_payoff(paths, market)
    res_eagle = eagle.get_discounted_payoff(paths, market)
    
    # Assert NaN cascade is prevented by the safe division logic
    assert not np.isnan(res_phx).any()
    assert not np.isnan(res_eagle).any()
