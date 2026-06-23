import time
import numpy as np
from datetime import datetime, timedelta

from kernel.tools import ObservationFrequency, Model, CalendarConvention, RateCurveType
from kernel.models.pricing_engines.enum_pricing_engine import PricingEngineType
from kernel.products.structured_products.autocall_products import Phoenix
from kernel.products.options.vanilla_options import EuropeanCallOption, EuropeanPutOption
from kernel.products.options.barrier_options import DownAndInCallOption
from kernel.products.options.path_dependent_options import AsianCallOption
from kernel.products.options.american_options import AmericanCallOption, AmericanPutOption
from kernel.products.rate.bond import ZeroCouponBond, CouponBond
from kernel.products.rate.vanilla_swap import InterestRateSwap
from utils.pricing_settings import PricingSettings
from kernel.pricing_launcher import PricingLauncher
from kernel.market_data import Market, VolatilitySurfaceType
from kernel.market_data.data_loader import MarketDataLoader
from kernel.market_data.rate_curve_data.enums_interpolators import InterpolationType

def run_examples():
    # Common Settings
    settings = PricingSettings()
    settings.underlying_name = "SPX"
    settings.rate_curve_type = RateCurveType.RF_US_TREASURY
    settings.interpolation_type = InterpolationType.CUBIC
    settings.day_count_convention = CalendarConvention.ACT_360
    settings.obs_frequency = ObservationFrequency.ANNUAL
    settings.model = Model.BLACK_SCHOLES
    settings.volatility_surface_type = VolatilitySurfaceType.SVI
    settings.compute_greeks = False
    
    # 50,000 paths and 100 steps for accurate but reasonably fast Monte Carlo
    settings.nb_paths = 50000  
    settings.nb_steps = 100
    
    print("=" * 60)
    print("INITIALIZING MARKET DATA (Rates, Volatility Surface)")
    print("=" * 60)
    t0 = time.time()
    
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
        volatility_surface_type=settings.volatility_surface_type,
        calendar_convention=settings.day_count_convention,
        obs_frequency=settings.obs_frequency
    )
    t1 = time.time()
    print(f"Market init time: {t1 - t0:.4f} seconds\n")
    
    spot = market.underlying_asset.last_price
    print(f"Current Underlying Spot Price: {spot:.2f}\n")
    
    today = datetime.today()
    
    # 3-4 Examples per Engine Type
    examples = {
        PricingEngineType.MC: [
            ("European Call Option", EuropeanCallOption(maturity=1.0, strike=spot)),
            ("European Put Option", EuropeanPutOption(maturity=1.0, strike=spot)),
            ("Asian Call Option", AsianCallOption(maturity=1.0, strike=spot)),
            ("Down-And-In Call (Barrier=80%)", DownAndInCallOption(maturity=1.0, strike=spot, barrier=spot*0.8))
        ],
        PricingEngineType.AMERICAN_MC: [
            ("American Call Option", AmericanCallOption(maturity=1.0, strike=spot)),
            ("American Put Option", AmericanPutOption(maturity=1.0, strike=spot))
        ],
        PricingEngineType.CALLABLE_MC: [
            ("Phoenix Autocall (3Y)", Phoenix(
                maturity=3.0, 
                observation_frequency=ObservationFrequency.ANNUAL, 
                capital_barrier=60, 
                autocall_barrier=100, 
                coupon_barrier=80, 
                coupon_rate=5.0
            ))
        ],
        PricingEngineType.RATE: [
            ("Zero Coupon Bond (1Y)", ZeroCouponBond(
                issue_date=today, 
                maturity=today + timedelta(days=365), 
                calendar_convention=CalendarConvention.ACT_360,
                notional=1000,
                ytm=0.04
            )),
            ("Coupon Bond (5Y, 4% coupon)", CouponBond(
                issue_date=today, 
                maturity=today + timedelta(days=365*5), 
                calendar_convention=CalendarConvention.ACT_360,
                notional=1000, 
                coupon_rate=0.04, 
                frequency=2,
                ytm=0.045
            )),
            ("Interest Rate Swap (2Y)", InterestRateSwap(
                issue_date=today, 
                maturity=today + timedelta(days=365*2), 
                calendar_convention=CalendarConvention.ACT_360,
                notional=10000, 
                fixed_rate=0.03, 
                float_spread=0.0, 
                frequency=1
            ))
        ]
    }
    
    total_time = 0
    for engine_type, products in examples.items():
        print("=" * 60)
        print(f"ENGINE: {engine_type.name}")
        print("=" * 60)
        settings.pricing_engine_type = engine_type
        launcher = PricingLauncher(pricing_settings=settings, market=market)
        
        for name, product in products:
            t_start = time.time()
            res = launcher.calculate(product)
            t_end = time.time()
            elapsed = t_end - t_start
            total_time += elapsed
            
            print(f"> {name}")
            print(f"  Time taken: {elapsed:.4f}s")
            
            if res.price is not None:
                print(f"  Price:      {res.price:.4f}")
            else:
                print("  Price:      N/A")
                
            if res.std_dev is not None and res.std_dev > 0:
                print(f"  Std Dev:    {res.std_dev:.4f}")
                print(f"  95% CI:     [{res.lower_bound:.4f}, {res.upper_bound:.4f}]")
            else:
                print("  Std Dev:    N/A (Analytical / Deterministic)")
            print("-" * 40)
        print()
        
    print(f"Total Pricing Time (excluding market init): {total_time:.4f} seconds")

if __name__ == '__main__':
    run_examples()
