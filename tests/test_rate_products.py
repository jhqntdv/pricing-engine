import unittest
from datetime import datetime
from kernel.tools import CalendarConvention
from kernel.products.rate.bond import ZeroCouponBond, CouponBond
from kernel.products.rate.vanilla_swap import InterestRateSwap
from kernel.models.pricing_engines.discounting_pricing_engine import DiscountingPricingEngine
from kernel.market_data.market import Market
from utils.pricing_settings import PricingSettings
import numpy as np

class TestRateProducts(unittest.TestCase):
    def setUp(self):
        # Setup dummy market
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

        # Mock rate curves for discounting
        self.market.get_rate = lambda t: 5.0
        self.market.get_discount_factor = lambda t: 1.0 / ((1 + 0.05) ** t) if t > 0 else 1.0
        self.market.get_fwd_rate = lambda t1, t2: 5.0

        self.issue_date = datetime(2023, 1, 1)
        self.maturity = datetime(2024, 1, 1)
        
        self.settings = PricingSettings()
        self.settings.valuation_date = self.issue_date
        self.engine = DiscountingPricingEngine(market=self.market, settings=self.settings)

    def test_zero_coupon_bond_pricing(self):
        zcb = ZeroCouponBond(
            notional=100.0, 
            issue_date=self.issue_date, 
            maturity=self.maturity, 
            calendar_convention=CalendarConvention.ACT_365, 
            ytm=0.05
        )
        
        results = self.engine.calculate_rate_product(zcb)
        
        # t = 1 year, ytm = 5% => price = 100 / 1.05 = 95.238
        self.assertAlmostEqual(results.price, 100.0 / 1.05, places=3)
        self.assertEqual(results.rate, 0.05)
        
    def test_zero_coupon_bond_ytm(self):
        zcb = ZeroCouponBond(
            notional=100.0, 
            issue_date=self.issue_date, 
            maturity=self.maturity, 
            calendar_convention=CalendarConvention.ACT_365, 
            price=95.238095
        )
        
        results = self.engine.calculate_rate_product(zcb)
        
        # t = 1 year, price = 95.238095 => ytm = 5%
        self.assertAlmostEqual(results.rate, 0.05, places=3)

    def test_coupon_bond_pricing(self):
        cb = CouponBond(
            notional=100.0, 
            issue_date=self.issue_date, 
            maturity=self.maturity,
            coupon_rate=0.05, 
            frequency=1, 
            calendar_convention=CalendarConvention.ACT_365, 
            ytm=0.05
        )
                    
        results = self.engine.calculate_rate_product(cb)
        
        # 1 year out, 1 coupon of 5%, ytm = 5%. PV = 5/1.05 + 100/1.05 = 105/1.05 = 100
        self.assertAlmostEqual(results.price, 100.0, places=3)
        self.assertEqual(results.rate, 0.05)

    def test_interest_rate_swap(self):
        swap = InterestRateSwap(
            notional=100.0, 
            issue_date=self.issue_date, 
            maturity=self.maturity, 
            calendar_convention=CalendarConvention.ACT_365, 
            float_spread=0.0, 
            frequency=1
        )
                            
        results = self.engine.calculate_rate_product(swap)
        
        # At start date, par swap price should be 0
        self.assertAlmostEqual(results.price, 0.0, places=3)
        # Fixed rate should be non-zero
        self.assertTrue(results.rate > 0)
        
    def test_abstract_rate_product_payoff(self):
        # Ensure abstract rate product properly throws NotImplementedError for MC
        zcb = ZeroCouponBond(
            notional=100.0, 
            issue_date=self.issue_date, 
            maturity=self.maturity, 
            calendar_convention=CalendarConvention.ACT_365, 
            ytm=0.05
        )
        with self.assertRaises(NotImplementedError):
            zcb.get_discounted_payoff(np.array([]), self.market)

if __name__ == "__main__":
    unittest.main()
