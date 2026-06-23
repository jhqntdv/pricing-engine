import unittest
import numpy as np
import pandas as pd
from kernel.market_data.volatility_surface.svi_surface import SVIVolatilitySurface
from kernel.market_data.volatility_surface.local_surface import LocalVolatilitySurface

class DummyRateCurve:
    def __init__(self, rate=0.05):
        self.rate = rate
    def get_rate(self, maturity: float) -> float:
        return self.rate * 100  # Returns in percentage like original implementation

class TestLocalVolatility(unittest.TestCase):
    def setUp(self):
        # Create Dummy SPX Data
        spot = 5000.0
        maturities = [0.1, 0.5, 1.0, 5.0]
        strikes = np.linspace(3000, 7000, 11)
        
        data = []
        for t in maturities:
            for k in strikes:
                # Flat 20% volatility for testing
                data.append({"Strike": k, "Maturity": t, "Spot": spot, "Implied Volatility": 20.0})
                
        self.option_data = pd.DataFrame(data)
        self.rate_curve = DummyRateCurve(0.05)
        
        self.svi = SVIVolatilitySurface(self.option_data, self.rate_curve)
        self.svi.calibrate_surface()

    def test_short_dated_options_stability(self):
        """
        Extremely short-dated options should be handled stably using the new local volatility model
        without crashing or returning NaNs.
        """
        local_vol_surface = LocalVolatilitySurface(self.option_data, self.rate_curve, self.svi)

        strike = 5000.0
        maturity = 0.005  # ~1.2 days
        
        vol = local_vol_surface.get_volatility(strike, maturity)
        
        # Volatility should be stable and bounded by the 5% floor and 350% cap
        self.assertTrue(vol >= 0.05)
        self.assertTrue(vol <= 3.5)
        self.assertFalse(np.isnan(vol))
        
    def test_deep_otm_numerical_stability(self):
        """
        Deep OTM options should be handled gracefully, staying within the bounds
        and avoiding numerical blowups.
        """
        local_vol_surface = LocalVolatilitySurface(self.option_data, self.rate_curve, self.svi)

        strike = 15000.0  # Extreme OTM strike
        maturity = 1.0
        
        vol = local_vol_surface.get_volatility(strike, maturity)
        
        # The result must be stable and properly bounded
        self.assertTrue(0.05 <= vol <= 3.5)
        self.assertFalse(np.isnan(vol))

    def test_caps_and_floors(self):
        """
        Ensure that the volatility floor of 5% (0.05) and cap of 350% (3.5) are strictly respected.
        """
        local_vol_surface = LocalVolatilitySurface(self.option_data, self.rate_curve, self.svi)
        
        # SVI is calibrated to a flat 20% volatility. Let's test a dense grid of strikes and maturities.
        test_strikes = np.linspace(1000, 20000, 50)
        test_maturities = np.linspace(0.001, 10.0, 50)
        
        for t in test_maturities:
            for K in test_strikes:
                vol = local_vol_surface.get_volatility(K, t)
                self.assertTrue(vol >= 0.05, f"Floor violated: vol={vol} for K={K}, t={t}")
                self.assertTrue(vol <= 3.5, f"Cap violated: vol={vol} for K={K}, t={t}")
                self.assertFalse(np.isnan(vol))

if __name__ == '__main__':
    unittest.main()
