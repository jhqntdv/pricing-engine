import unittest
import numpy as np
from kernel.models.stochastic_processes.black_scholes_process import BlackScholesProcess
from kernel.models.stochastic_processes.heston_process import HestonProcess
from kernel.models.discretization_schemes.euler_scheme import EulerScheme
from kernel.tools import NumpyRandomGenerator, SobolRandomGenerator

class TestRandomGenerators(unittest.TestCase):
    def test_sobol_generator(self):
        generator = SobolRandomGenerator()
        
        # Test one factor
        z1 = generator.get_standard_normal(nb_paths=1000, nb_steps=252, nb_factors=1)
        self.assertEqual(z1.shape, (1000, 252))
        self.assertAlmostEqual(np.mean(z1), 0.0, places=1)
        
        # Test two factors
        z1_2, z2_2 = generator.get_standard_normal(nb_paths=1000, nb_steps=252, nb_factors=2)
        self.assertEqual(z1_2.shape, (1000, 252))
        self.assertEqual(z2_2.shape, (1000, 252))
        
        # The correlation between the two independent factors should be near zero
        corr = np.corrcoef(z1_2.flatten(), z2_2.flatten())[0, 1]
        self.assertAlmostEqual(corr, 0.0, places=1)

class TestStochasticProcesses(unittest.TestCase):
    def setUp(self):
        self.S0 = 100.0
        self.T = 1.0
        self.nb_steps = 252
        self.nb_paths = 1000
        
        # Drift vector matching the steps
        self.drift = np.full(self.nb_steps, 0.05)
        
    def test_black_scholes_process_increments(self):
        process = BlackScholesProcess(
            S0=self.S0, 
            T=self.T, 
            nb_steps=self.nb_steps, 
            drift=self.drift, 
            volatility=0.2
        )
        increments = process.get_random_increments(self.nb_paths)
        
        # Increments should match the shape (nb_paths, nb_steps)
        self.assertEqual(increments.shape, (self.nb_paths, self.nb_steps))
        
        # Mean should be close to 0
        self.assertAlmostEqual(np.mean(increments), 0.0, places=2)
        
    def test_heston_process_increments(self):
        process = HestonProcess(
            S0=self.S0, 
            v0=0.04, 
            T=self.T, 
            nb_steps=self.nb_steps, 
            drift=self.drift,
            kappa=2.0, 
            theta=0.04, 
            sigma=0.3, 
            rho=-0.5
        )
        Z1, Z3 = process.get_random_increments(self.nb_paths)
        
        self.assertEqual(Z1.shape, (self.nb_paths, self.nb_steps))
        self.assertEqual(Z3.shape, (self.nb_paths, self.nb_steps))
        
        # Calculate empirical correlation
        flat_Z1 = Z1.flatten()
        flat_Z3 = Z3.flatten()
        correlation = np.corrcoef(flat_Z1, flat_Z3)[0, 1]
        
        # Empirical correlation should be close to rho (-0.5)
        self.assertAlmostEqual(correlation, -0.5, places=1)


class TestEulerScheme(unittest.TestCase):
    def setUp(self):
        self.S0 = 100.0
        self.T = 1.0
        self.nb_steps = 252
        self.nb_paths = 1000
        self.drift = np.full(self.nb_steps, 0.05)
        self.scheme = EulerScheme()

    def test_euler_one_factor(self):
        process = BlackScholesProcess(
            S0=self.S0, 
            T=self.T, 
            nb_steps=self.nb_steps, 
            drift=self.drift, 
            volatility=0.2
        )
        paths = self.scheme.simulate_paths(process, self.nb_paths)
        
        # Shape should be (nb_paths, nb_steps + 1)
        self.assertEqual(paths.shape, (self.nb_paths, self.nb_steps + 1))
        
        # Initial spot check
        np.testing.assert_array_equal(paths[:, 0], np.full(self.nb_paths, self.S0))

        # Log-Euler must keep prices strictly positive
        self.assertTrue(np.all(paths > 0), "Log-Euler must keep prices strictly positive")

    def test_euler_two_factor(self):
        process = HestonProcess(
            S0=self.S0, 
            v0=0.04, 
            T=self.T, 
            nb_steps=self.nb_steps, 
            drift=self.drift,
            kappa=2.0, 
            theta=0.04, 
            sigma=0.3, 
            rho=-0.5
        )
        paths = self.scheme.simulate_paths(process, self.nb_paths)
        
        # Shape for two-factor (spot price is returned, variance is internal)
        # The EulerScheme implementation returns paths[:, :, 0] meaning just spot
        self.assertEqual(paths.shape, (self.nb_paths, self.nb_steps + 1))
        
        # Initial spot check
        np.testing.assert_array_equal(paths[:, 0], np.full(self.nb_paths, self.S0))

        # Log-Euler spot must remain strictly positive
        self.assertTrue(np.all(paths > 0), "Log-Euler spot must remain strictly positive")

    def test_euler_list_drift_and_typing(self):
        # Passing drift as a list instead of ndarray, which happens in mc_pricing_engine.py
        list_drift = [0.05] * self.nb_steps
        
        process = BlackScholesProcess(
            S0=self.S0, 
            T=self.T, 
            nb_steps=self.nb_steps, 
            drift=list_drift, 
            volatility=0.2
        )
        paths = self.scheme.simulate_paths(process, self.nb_paths)
        self.assertEqual(paths.shape, (self.nb_paths, self.nb_steps + 1))
        
        # Verify get_random_increments typing output correctly matches Union logic
        inc = process.get_random_increments(self.nb_paths)
        self.assertIsInstance(inc, np.ndarray)
        
        heston = HestonProcess(
            S0=self.S0, v0=0.04, T=self.T, nb_steps=self.nb_steps, drift=list_drift,
            kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5
        )
        z1, z2 = heston.get_random_increments(self.nb_paths)
        self.assertIsInstance(z1, np.ndarray)
        self.assertIsInstance(z2, np.ndarray)

from kernel.market_data.market import Market
from utils.pricing_settings import PricingSettings
from kernel.models.pricing_engines.callable_mc_pricing_engine import CallableMCPricingEngine
from kernel.products.structured_products.autocall_products import Phoenix
from kernel.tools import ObservationFrequency

class TestCallablePricingEngine(unittest.TestCase):
    def setUp(self):
        # Setup a dummy market and settings
        import pandas as pd
        yield_df = pd.DataFrame({"Maturity": ["1M", "6M", "1Y", "2Y", "5Y"], "Rate": [0.04, 0.045, 0.05, 0.055, 0.06]})
        underlying_df = pd.DataFrame({"Security Label": ["S&P"], "Ticker": ["^SPX"], "ISIN": ["123"], "Is Index": [True], "Last Price": [100.0]})
        
        # SVI calibration requires at least 4 maturities for cubic interpolation
        maturities = ["1M", "3M", "6M", "1Y"]
        strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
        option_df = pd.DataFrame([{"Maturity": m, "Strike": k, "Implied Volatility": 0.2} for m in maturities for k in strikes])
        
        self.market = Market(
            underlying_name='^SPX',
            yield_curve_data=yield_df,
            underlying_data=underlying_df,
            option_data=option_df
        )
        self.market.underlying_asset.last_price = 100.0 
            
        self.settings = PricingSettings()
        self.settings.nb_paths = 1000
        self.settings.nb_steps = 252
        self.engine = CallableMCPricingEngine(market=self.market, settings=self.settings)

    def test_get_coupon_root_finding(self):
        # Create a dummy phoenix autocall
        phoenix = Phoenix(
            maturity=1.0, 
            observation_frequency=ObservationFrequency.ANNUAL,
            capital_barrier=60, 
            autocall_barrier=100, 
            coupon_barrier=80, 
            coupon_rate=0.0
        )
        phoenix.initial_spot = 100.0
        
        # We test that get_coupon modifies the derivative coupon to target a specific price
        process = BlackScholesProcess(
            S0=100.0, 
            T=1.0, 
            nb_steps=self.settings.nb_steps, 
            drift=np.full(self.settings.nb_steps, 0.0), 
            volatility=0.2
        )
        
        # Test root finding converges to target price 100
        coupon = self.engine.get_coupon(derivative=phoenix, process=process, target_price=100.0, epsilon=0.5)
        self.assertTrue(0.0 <= coupon <= 50.0)

    def test_fast_implied_coupon_solver(self):
        import time
        from kernel.products.structured_products.autocall_products import Phoenix
        
        phoenix = Phoenix(
            maturity=1.0, 
            observation_frequency=ObservationFrequency.ANNUAL,
            capital_barrier=60, 
            autocall_barrier=100, 
            coupon_barrier=80, 
            coupon_rate=0.0
        )
        phoenix.initial_spot = 100.0
        
        process = BlackScholesProcess(
            S0=100.0, 
            T=1.0, 
            nb_steps=self.settings.nb_steps, 
            drift=np.full(self.settings.nb_steps, 0.0), 
            volatility=0.2
        )
        
        start_time = time.time()
        coupon = self.engine.get_coupon(derivative=phoenix, process=process, target_price=100.0, epsilon=0.01)
        elapsed_time = time.time() - start_time
        
        # Verify that pre-simulation makes the solver very fast
        self.assertTrue(elapsed_time < 0.5, f"Implied coupon solver is too slow: {elapsed_time:.4f} seconds")
        self.assertTrue(0.0 <= coupon <= 50.0)


    def test_mc_pricing_engine_with_structured_product(self):
        # Create a dummy phoenix autocall
        phoenix = Phoenix(
            maturity=1.0, 
            observation_frequency=ObservationFrequency.ANNUAL,
            capital_barrier=60, 
            autocall_barrier=100, 
            coupon_barrier=80, 
            coupon_rate=5.0
        )
        phoenix.initial_spot = 100.0
        
        # Test that MCPricingEngine can correctly price an Autocall 
        # (It should no longer crash with a tuple mismatch thanks to decoupled get_discounted_payoff)
        from kernel.models.pricing_engines.mc_pricing_engine import MCPricingEngine
        
        # Mock get_stochastic_process to avoid market data lookups
        mc_engine = MCPricingEngine(market=self.market, settings=self.settings)
        process = BlackScholesProcess(
            S0=100.0, 
            T=1.0, 
            nb_steps=self.settings.nb_steps, 
            drift=np.full(self.settings.nb_steps, 0.0), 
            volatility=0.2
        )
        mc_engine.get_stochastic_process = lambda derivative, market: process
        
        result = mc_engine.calculate_structured_product(phoenix)
        
        self.assertIsNotNone(result.price)
        self.assertTrue(result.price > 0.0)

    def test_mc_pricing_engine_greeks_optimization(self):
        # Test that MCPricingEngine computes Greeks properly and that the new inline optimization works
        from kernel.models.pricing_engines.mc_pricing_engine import MCPricingEngine
        from kernel.products.options.vanilla_options import EuropeanCallOption
        
        call_option = EuropeanCallOption(strike=100.0, maturity=1.0)
        
        # Enable greeks calculation
        self.settings.compute_greeks = True
        mc_engine = MCPricingEngine(market=self.market, settings=self.settings)
        from kernel.tools import Model
        mc_engine.model = Model.BLACK_SCHOLES
        
        # Mock get_volatility and get_rate
        self.market.get_volatility = lambda K, T: 0.2
        self.market.get_rate = lambda T: 0.05
        
        # Mock get_stochastic_process
        mc_engine.get_stochastic_process = lambda derivative, market: BlackScholesProcess(
            S0=100.0, 
            T=1.0, 
            nb_steps=self.settings.nb_steps, 
            drift=np.full(self.settings.nb_steps, 0.0), 
            volatility=0.2
        )
        
        # The engine should calculate delta, gamma, vega, rho, theta using the inline optimized logic
        result = mc_engine.get_result(call_option)
        
        self.assertIsNotNone(result.price)
        self.assertTrue(result.price > 0.0)
        
        # Ensure all greeks were populated
        greeks = result.greeks
        self.assertIn("delta", greeks)
        self.assertIn("gamma", greeks)
        self.assertIn("vega", greeks)
        self.assertIn("rho", greeks)
        self.assertIn("theta", greeks)
        
        # Verify delta is somewhat sensible for an ATM call
        self.assertTrue(0.0 < greeks["delta"] < 1.0)

from kernel.pricing_launcher import PricingLauncher
from kernel.models.pricing_engines.enum_pricing_engine import PricingEngineType

class TestPricingLauncher(unittest.TestCase):
    def test_market_injection(self):
        # Setup settings
        settings = PricingSettings()
        settings.pricing_engine_type = PricingEngineType.MC
        
        # Test that launcher correctly uses injected market and doesn't recreate it
        class DummyMarket:
            def __init__(self):
                self.was_used = True
        
        dummy_market = DummyMarket()
        
        # Initialize launcher with dummy market
        launcher = PricingLauncher(pricing_settings=settings, market=dummy_market)
        
        # Assert the launcher kept the market
        self.assertEqual(launcher.market, dummy_market)
        self.assertTrue(launcher.market.was_used)

if __name__ == "__main__":
    unittest.main()
