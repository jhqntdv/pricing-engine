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

import pandas as pd

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

    european_products = [
        ("European Call", EuropeanCallOption(strike=spot, maturity=1.0)),
        ("European Put", EuropeanPutOption(strike=spot*0.9, maturity=1.0)),
        ("Cash-Or-Nothing Call", BinaryCallOption(strike=spot, maturity=1.0, coupon=100)),
        ("Straddle", Straddle(strike=spot, maturity=1.0, position_call=True, position_put=True)),
        ("Bull Spread", BullSpread(maturity=1.0, strike_low=spot*0.9, strike_high=spot*1.1, position_low=True, position_high=False)),
    ]
    
    path_dependent_products = [
        ("Down-And-In Call", DownAndInCallOption(strike=spot, maturity=1.0, barrier=spot*0.8)),
        ("Up-And-Out Put", UpAndOutPutOption(strike=spot, maturity=1.0, barrier=spot*1.2)),
        ("Asian Call", AsianCallOption(strike=spot, maturity=1.0)),
        ("Lookback Put", LookbackPutOption(strike=spot, maturity=1.0)),
    ]
    
    american_products = [
        ("American Call", AmericanCallOption(strike=spot, maturity=1.0)),
        ("Bermudan Put", BermudanPutOption(strike=spot, maturity=1.0, exercise_times=[0.25, 0.5, 0.75, 1.0])),
    ]
    
    structured_products = [
        ("Phoenix Autocall", Phoenix(maturity=3.0, observation_frequency=ObservationFrequency.SEMIANNUAL, capital_barrier=70.0, autocall_barrier=100.0, coupon_barrier=80.0, coupon_rate=8.0)),
        ("Eagle Autocall", Eagle(maturity=3.0, observation_frequency=ObservationFrequency.SEMIANNUAL, capital_barrier=70.0, autocall_barrier=100.0, coupon_rate=8.0)),
    ]

    results_list = []
    
    def run_batch(products, engine_type, model_type):
        settings.pricing_engine_type = engine_type
        settings.model = model_type
        for name, prod in products:
            start = time.time()
            res = launcher.calculate(prod)
            elapsed = time.time() - start
            
            row = {
                "Product": name,
                "Model": model_type.name,
                "Price": f"{res.price:.4f}",
                "Time(s)": f"{elapsed:.4f}",
            }
            if res.std_dev is not None:
                row["StdDev"] = f"{res.std_dev:.4f}"
            else:
                row["StdDev"] = "N/A"
                
            if res.greeks:
                row["Delta"] = f"{res.greeks.get('delta', 0):.4f}"
                row["Gamma"] = f"{res.greeks.get('gamma', 0):.6f}"
                row["Vega"] = f"{res.greeks.get('vega', 0):.4f}"
            else:
                row["Delta"] = "N/A"
                row["Gamma"] = "N/A"
                row["Vega"] = "N/A"
                
            results_list.append(row)

    total_t0 = time.time()
    
    # Run European under Black-Scholes and Heston
    run_batch(european_products, PricingEngineType.MC, Model.BLACK_SCHOLES)
    run_batch(european_products, PricingEngineType.MC, Model.HESTON)
    
    # Run Path Dependent under Black-Scholes and Heston
    run_batch(path_dependent_products, PricingEngineType.MC, Model.BLACK_SCHOLES)
    run_batch(path_dependent_products, PricingEngineType.MC, Model.HESTON)
    
    # Run Early Exercise under Black-Scholes and Heston
    run_batch(american_products, PricingEngineType.AMERICAN_MC, Model.BLACK_SCHOLES)
    run_batch(american_products, PricingEngineType.AMERICAN_MC, Model.HESTON)
    
    # Run Structured Products under Black-Scholes and Heston
    run_batch(structured_products, PricingEngineType.CALLABLE_MC, Model.BLACK_SCHOLES)
    run_batch(structured_products, PricingEngineType.CALLABLE_MC, Model.HESTON)

    df = pd.DataFrame(results_list)
    
    # Side-by-side table for ALL Products (BS vs Heston)
    # Pivot to get models side-by-side
    pivot_df = df.pivot(index='Product', columns='Model', values=['Price', 'Time(s)', 'StdDev', 'Delta', 'Gamma', 'Vega'])
    
    # Flatten columns: 'Price' 'BLACK_SCHOLES' -> 'Price (BS)'
    new_cols = []
    for val_col, model_col in pivot_df.columns:
        model_name = "BS" if model_col == "BLACK_SCHOLES" else "Heston"
        new_cols.append(f"{val_col} ({model_name})")
    pivot_df.columns = new_cols
    pivot_df = pivot_df.reset_index()
    
    # Reorder the rows to match the order they were defined (since pivot sorts alphabetically)
    product_order = [p[0] for p in european_products + path_dependent_products + american_products + structured_products]
    pivot_df['Product'] = pd.Categorical(pivot_df['Product'], categories=product_order, ordered=True)
    pivot_df = pivot_df.sort_values('Product')
    
    print("=" * 120)
    print("ALL PRODUCTS: BLACK-SCHOLES vs HESTON (Side-by-Side Comparison)")
    print("=" * 120)
    print(pivot_df.set_index('Product').T.to_string())
    
    print(f"\nTotal Pricing Time (excluding market init): {time.time() - total_t0:.4f} seconds")

if __name__ == "__main__":
    main()
