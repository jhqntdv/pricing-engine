"""
Tests for PricingResults: verifies __str__, lower/upper bounds, std_dev, 
confidence interval, Greek aggregation, and integration with MC engine.
"""
import unittest
import numpy as np
from utils.pricing_results import PricingResults


class TestPricingResultsStr(unittest.TestCase):
    """Test that PricingResults __str__ does not crash and formats correctly."""

    def test_str_with_all_fields(self):
        """Full result with price, std_dev, greeks — should format without errors."""
        result = PricingResults(price=10.5, std_dev=0.25, confidence_level=0.95)
        result.set_greek("delta", 0.55)
        result.set_greek("gamma", 0.02)
        output = str(result)
        self.assertIn("10.5", output)
        self.assertIn("0.25", output)
        self.assertIn("95%", output)
        self.assertIn("delta", output)

    def test_str_with_price_only(self):
        """Result with only price, no std_dev — should not crash."""
        result = PricingResults(price=42.0)
        output = str(result)
        self.assertIn("42.0", output)
        self.assertIn("N/A", output)  # std_dev is None

    def test_str_with_no_fields(self):
        """Completely empty result — should not crash."""
        result = PricingResults()
        output = str(result)
        self.assertIn("N/A", output)

    def test_str_with_greeks_only(self):
        """Result with greeks but no price — should not crash."""
        result = PricingResults()
        result.set_greek("vega", 1.5)
        output = str(result)
        self.assertIn("vega", output)

    def test_str_with_coupon(self):
        """Result with coupon_callable — str should still work."""
        result = PricingResults(coupon_callable=7.82)
        output = str(result)
        self.assertIsInstance(output, str)

    def test_str_with_rate(self):
        """Result with rate (bond/swap) — str should still work."""
        result = PricingResults(price=10000.0, rate=0.04)
        output = str(result)
        self.assertIn("10000", output)


class TestPricingResultsConfidenceInterval(unittest.TestCase):
    """Test confidence interval calculations (lower/upper bounds)."""

    def test_bounds_with_std_dev(self):
        """Bounds should be price ± 1.96 * std_dev."""
        result = PricingResults(price=100.0, std_dev=2.0)
        self.assertAlmostEqual(result.lower_bound, 100.0 - 1.96 * 2.0, places=4)
        self.assertAlmostEqual(result.upper_bound, 100.0 + 1.96 * 2.0, places=4)

    def test_bounds_without_std_dev(self):
        """Bounds should be None when std_dev is not set."""
        result = PricingResults(price=100.0)
        self.assertIsNone(result.lower_bound)
        self.assertIsNone(result.upper_bound)

    def test_bounds_without_price(self):
        """Bounds should be None when price is not set."""
        result = PricingResults(std_dev=1.0)
        self.assertIsNone(result.lower_bound)
        self.assertIsNone(result.upper_bound)

    def test_confidence_level_default(self):
        """Default confidence level should be 0.95."""
        result = PricingResults()
        self.assertEqual(result.confidence_level, 0.95)

    def test_custom_confidence_level(self):
        """Custom confidence level should be stored correctly."""
        result = PricingResults(confidence_level=0.99)
        self.assertEqual(result.confidence_level, 0.99)

    def test_zero_std_dev(self):
        """Zero std_dev should produce equal lower and upper bounds."""
        result = PricingResults(price=50.0, std_dev=0.0)
        self.assertAlmostEqual(result.lower_bound, 50.0)
        self.assertAlmostEqual(result.upper_bound, 50.0)


class TestPricingResultsAggregation(unittest.TestCase):
    """Test strategy aggregation logic."""

    def test_aggregate_prices(self):
        """Aggregated price should be the sum of component prices."""
        r1 = PricingResults(price=10.0)
        r2 = PricingResults(price=20.0)
        agg = PricingResults.get_aggregated_results([r1, r2])
        self.assertAlmostEqual(agg.price, 30.0)

    def test_aggregate_greeks(self):
        """Aggregated greeks should sum across components."""
        r1 = PricingResults(price=10.0)
        r1.set_greek("delta", 0.5)
        r1.set_greek("gamma", 0.01)
        r2 = PricingResults(price=5.0)
        r2.set_greek("delta", -0.3)
        r2.set_greek("vega", 1.0)
        agg = PricingResults.get_aggregated_results([r1, r2])
        self.assertAlmostEqual(agg.greeks["delta"], 0.2)
        self.assertAlmostEqual(agg.greeks["gamma"], 0.01)
        self.assertAlmostEqual(agg.greeks["vega"], 1.0)

    def test_aggregate_empty_list(self):
        """Aggregating empty list should produce zero price."""
        agg = PricingResults.get_aggregated_results([])
        self.assertEqual(agg.price, 0)

    def test_aggregate_with_none_price(self):
        """Components with None price should be skipped in aggregation."""
        r1 = PricingResults(price=10.0)
        r2 = PricingResults()  # price is None
        agg = PricingResults.get_aggregated_results([r1, r2])
        self.assertAlmostEqual(agg.price, 10.0)


class TestPricingResultsIntegrationWithMC(unittest.TestCase):
    """Integration test: verify MC engine populates std_dev in PricingResults."""

    def test_mc_engine_populates_std_dev(self):
        """MCPricingEngine.get_result() should populate std_dev in results."""
        from kernel.tools import RateCurveType, Model, CalendarConvention, ObservationFrequency
        from kernel.models.pricing_engines.enum_pricing_engine import PricingEngineType
        from kernel.products.options.vanilla_options import EuropeanCallOption
        from kernel.market_data import Market
        from kernel.market_data.rate_curve_data.enums_interpolators import InterpolationType
        from kernel.market_data.volatility_surface.enums_volatility import VolatilitySurfaceType
        from utils.pricing_settings import PricingSettings
        from kernel.pricing_launcher import PricingLauncher

        settings = PricingSettings()
        settings.underlying_name = "SPX"
        settings.rate_curve_type = RateCurveType.RF_US_TREASURY
        settings.interpolation_type = InterpolationType.CUBIC
        settings.volatility_surface_type = VolatilitySurfaceType.SVI
        settings.day_count_convention = CalendarConvention.ACT_360
        settings.obs_frequency = ObservationFrequency.ANNUAL
        settings.model = Model.BLACK_SCHOLES
        settings.pricing_engine_type = PricingEngineType.MC
        settings.compute_greeks = False
        settings.nb_paths = 5000
        settings.nb_steps = 50
        settings.random_seed = 42

        from kernel.market_data.data_loader import MarketDataLoader
        data_loader = MarketDataLoader()
        underlying_df = data_loader.get_underlying_info(settings.underlying_name)
        options_df = data_loader.get_option_data(settings.underlying_name)
        yield_df = data_loader.get_yield_curve(settings.rate_curve_type.value)

        market = Market(
            underlying_name=settings.underlying_name,
            yield_curve_data=yield_df,
            underlying_data=underlying_df,
            option_data=options_df,
            rate_curve_type=settings.rate_curve_type,
            interpolation_type=settings.interpolation_type,
            volatility_surface_type=VolatilitySurfaceType.SVI,
            calendar_convention=settings.day_count_convention,
            obs_frequency=settings.obs_frequency,
        )

        launcher = PricingLauncher(pricing_settings=settings, market=market)
        spot = market.underlying_asset.last_price
        product = EuropeanCallOption(maturity=1.0, strike=spot)
        result = launcher.calculate(product)

        # Core checks
        self.assertIsNotNone(result.price)
        self.assertIsNotNone(result.std_dev)
        self.assertGreater(result.std_dev, 0)
        self.assertIsNotNone(result.lower_bound)
        self.assertIsNotNone(result.upper_bound)
        self.assertLess(result.lower_bound, result.price)
        self.assertGreater(result.upper_bound, result.price)

        # str() should work without error
        output = str(result)
        self.assertIsInstance(output, str)
        self.assertNotIn("N/A", output.split("\n")[1])  # std_dev line should have a value

        print(f"\n--- MC Integration Test Result ---")
        print(result)


if __name__ == "__main__":
    unittest.main()
