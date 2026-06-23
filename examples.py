import time
from kernel.market_data.data_loader import MarketDataLoader
from kernel.market_data.market import Market
from kernel.market_data.volatility_surface.enums_volatility import VolatilitySurfaceType
from kernel.market_data.rate_curve_data.enums_interpolators import InterpolationType
from kernel.tools import CalendarConvention, ObservationFrequency, RateCurveType, Model
from utils.pricing_settings import PricingSettings
from kernel.pricing_launcher import PricingLauncher
from kernel.models.pricing_engines.enum_pricing_engine import PricingEngineType

# Products
from kernel.products.options.vanilla_options import EuropeanCallOption, EuropeanPutOption
from kernel.products.options.barrier_options import DownAndInCallOption, UpAndOutPutOption
from kernel.products.options.binary_options import BinaryCallOption
from kernel.products.options.path_dependent_options import AsianCallOption, LookbackPutOption
from kernel.products.options_strategies.options_strategies import Straddle, BullSpread
from kernel.products.options.american_options import AmericanCallOption, BermudanPutOption
from kernel.products.structured_products.autocall_products import Phoenix, Eagle
from kernel.products.rate.bond import ZeroCouponBond, CouponBond
from kernel.products.rate.vanilla_swap import InterestRateSwap
from datetime import datetime

def print_result(name, result, t_elapsed):
    print(f"> {name}")
    print(f"  Time taken: {t_elapsed:.4f}s")
    print(f"  Price:      {result.price:.4f}")
    if result.std_dev is not None:
        print(f"  Std Dev:    {result.std_dev:.4f}")
        print(f"  95% CI:     [{result.lower_bound:.4f}, {result.upper_bound:.4f}]")
    else:
        print("  Std Dev:    N/A (Analytical / Deterministic)")
        print("  95% CI:     N/A")
    if result.greeks:
        print(f"  Greeks:     {result.greeks}")
    else:
        print("  Greeks:     None")
    
    if hasattr(result, 'coupon_callable') and result.coupon_callable is not None:
        print(f"  Coupon Call: {result.coupon_callable}")
    
    if hasattr(result, 'rate') and result.rate is not None:
        print(f"  Rate:       {result.rate:.4f}")
    print("-" * 60)

def main():
    print("=" * 60)
    print("INITIALIZING MARKET DATA (Rates, Volatility Surface)")
    print("=" * 60)
    
    t0 = time.time()
    settings = PricingSettings()
    settings.underlying_name = "SPX"
    settings.rate_curve_type = RateCurveType.RF_US_TREASURY
    settings.interpolation_type = InterpolationType.CUBIC
    settings.day_count_convention = CalendarConvention.ACT_360
    settings.obs_frequency = ObservationFrequency.ANNUAL
    settings.model = Model.BLACK_SCHOLES
    settings.volatility_surface_type = VolatilitySurfaceType.SVI
    settings.compute_greeks = True
    settings.nb_paths = 10000  
    settings.nb_steps = 100

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
    
    launcher = PricingLauncher(pricing_settings=settings, market=market)

    products = [
        ("ENGINE: MC - Vanilla & Exotics", [
            ("European Call", EuropeanCallOption(strike=spot, maturity=1.0)),
            ("European Put", EuropeanPutOption(strike=spot*0.9, maturity=1.0)),
            ("Down-And-In Call", DownAndInCallOption(strike=spot, maturity=1.0, barrier=spot*0.8)),
            ("Up-And-Out Put", UpAndOutPutOption(strike=spot, maturity=1.0, barrier=spot*1.2)),
            ("Cash-Or-Nothing Call", BinaryCallOption(strike=spot, maturity=1.0, coupon=100)),
            ("Asian Call", AsianCallOption(strike=spot, maturity=1.0)),
            ("Lookback Put", LookbackPutOption(strike=spot, maturity=1.0)),
            ("Straddle", Straddle(strike=spot, maturity=1.0, position_call=True, position_put=True)),
            ("Bull Spread", BullSpread(maturity=1.0, strike_low=spot*0.9, strike_high=spot*1.1, position_low=True, position_high=False)),
        ]),
        ("ENGINE: AMERICAN_MC - Early Exercise", [
            ("American Call", AmericanCallOption(strike=spot, maturity=1.0)),
            ("Bermudan Put", BermudanPutOption(strike=spot, maturity=1.0, exercise_times=[0.25, 0.5, 0.75, 1.0])),
        ]),
        ("ENGINE: CALLABLE_MC - Structured Products", [
            ("Phoenix Autocall", Phoenix(maturity=3.0, observation_frequency=ObservationFrequency.SEMIANNUAL, capital_barrier=0.7, autocall_barrier=1.0, coupon_barrier=0.8, coupon_rate=0.08)),
            ("Eagle Autocall", Eagle(maturity=3.0, observation_frequency=ObservationFrequency.SEMIANNUAL, capital_barrier=0.7, autocall_barrier=1.0, coupon_rate=0.08)),
        ]),
        ("ENGINE: RATE - Fixed Income", [
            ("Zero Coupon Bond (1Y)", ZeroCouponBond(notional=100.0, issue_date=datetime.now(), maturity=datetime.now().replace(year=datetime.now().year + 1), calendar_convention=CalendarConvention.ACT_360, ytm=0.03)),
            ("Coupon Bond (5Y)", CouponBond(notional=100.0, issue_date=datetime.now(), maturity=datetime.now().replace(year=datetime.now().year + 5), coupon_rate=0.04, frequency=2, calendar_convention=CalendarConvention.ACT_360, ytm=0.03)),
            ("Interest Rate Swap (2Y)", InterestRateSwap(notional=10000.0, issue_date=datetime.now(), maturity=datetime.now().replace(year=datetime.now().year + 2), calendar_convention=CalendarConvention.ACT_360, fixed_rate=0.03, frequency=2)),
        ])
    ]

    total_t0 = time.time()
    for category, items in products:
        print("=" * 60)
        print(category)
        print("=" * 60)
        
        # Determine engine type from category name
        if "MC - Vanilla" in category:
            settings.pricing_engine_type = PricingEngineType.MC
        elif "AMERICAN_MC" in category:
            settings.pricing_engine_type = PricingEngineType.AMERICAN_MC
        elif "CALLABLE_MC" in category:
            settings.pricing_engine_type = PricingEngineType.CALLABLE_MC
        elif "RATE" in category:
            settings.pricing_engine_type = PricingEngineType.RATE
            
        for name, prod in items:
            start = time.time()
            res = launcher.calculate(prod)
            elapsed = time.time() - start
            print_result(name, res, elapsed)

    print(f"\nTotal Pricing Time (excluding market init): {time.time() - total_t0:.4f} seconds")

if __name__ == "__main__":
    main()
