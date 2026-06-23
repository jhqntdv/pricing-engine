import unittest
import numpy as np
import pandas as pd
from kernel.market_data import RateCurve
from kernel.market_data.volatility_surface import SVIVolatilitySurface, SSVIVolatilitySurface, LocalVolatilitySurface

class DummyRateCurve:
    def __init__(self):
        pass
    
    def get_rate(self, maturity: float) -> float:
        return 5.0  # Flat 5% rate for testing

class TestVolatilitySurfaces(unittest.TestCase):
    
    def setUp(self):
        # Create dummy option data that perfectly fits a smile
        spot = 100.0
        maturities = [0.25, 0.5, 1.0, 2.0]
        strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
        
        data = []
        for T in maturities:
            for K in strikes:
                moneyness = np.log(K / spot)
                vol = 0.20 + 0.05 * moneyness**2
                data.append({
                    "Spot": spot,
                    "Maturity": T,
                    "Strike": K,
                    "Implied Volatility": vol * 100
                })
        
        self.option_data = pd.DataFrame(data)
        self.rate_curve = DummyRateCurve()

    def test_svi_calibration(self):
        svi_surface = SVIVolatilitySurface(self.option_data, self.rate_curve)
        svi_surface.calibrate_surface()
        self.assertTrue(svi_surface.is_calibrated)
        
        # Test ATM volatility retrieval
        vol = svi_surface.get_volatility(strike=100.0, maturity=1.0)
        self.assertTrue(0.15 < vol < 0.25)  # Should be close to 20%

    def test_ssvi_calibration(self):
        ssvi_surface = SSVIVolatilitySurface(self.option_data, self.rate_curve)
        ssvi_surface.calibrate_surface()
        self.assertTrue(ssvi_surface.is_calibrated)
        
        # Test ATM volatility retrieval
        vol = ssvi_surface.get_volatility(strike=100.0, maturity=1.0)
        self.assertTrue(0.15 < vol < 0.25)  # Should be close to 20%

    def test_local_volatility(self):
        svi_surface = SVIVolatilitySurface(self.option_data, self.rate_curve)
        # Local Vol implicitly calibrates SVI
        local_surface = LocalVolatilitySurface(self.option_data, self.rate_curve, svi_surface)
        
        self.assertTrue(local_surface.is_calibrated)
        self.assertTrue(svi_surface.is_calibrated)
        
        # Test volatility retrieval
        vol = local_surface.get_volatility(strike=100.0, maturity=1.0)
        self.assertTrue(vol > 0.0)

    def test_ssvi_heston_parametrization_math(self):
        # We build a dummy market with an exact Heston term structure
        # Target parameters: kappa=2.0, v0=0.04, v_inf=0.09
        target_kappa = 2.0
        target_v0 = 0.04
        target_v_inf = 0.09
        
        spot = 100.0
        maturities = np.linspace(0.1, 3.0, 10)
        
        data = []
        for T in maturities:
            # Heston ATM variance formula
            theta_t = (target_v0 - target_v_inf) / target_kappa * (1 - np.exp(-target_kappa * T)) + target_v_inf * T
            atm_vol = np.sqrt(theta_t / T)
            
            # Create a simple slice centered at spot
            for K in [90.0, 100.0, 110.0]:
                data.append({
                    "Spot": spot,
                    "Maturity": T,
                    "Strike": K,
                    "Implied Volatility": atm_vol * 100 
                })
                
        heston_data = pd.DataFrame(data)
        
        # Instantiate surface
        ssvi_surface = SSVIVolatilitySurface(heston_data, self.rate_curve)
        
        # Extract the target implied ATM variances from the dummy market
        atm_market_variance = np.array([ssvi_surface._get_market_atm_variance(maturity) for maturity in maturities])
        
        # Now, pass the EXACT parameters into the cost function
        target_params = np.array([target_kappa, target_v0, target_v_inf])
        cost = ssvi_surface._ssvi_atm_cost_function(target_params, maturities, atm_market_variance)
        
        # The cost (Mean Squared Error) should be virtually 0.0, proving the math matches perfectly!
        self.assertAlmostEqual(cost, 0.0, places=15)

if __name__ == '__main__':
    unittest.main()
