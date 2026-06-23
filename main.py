import os
import time
os.environ["MPLBACKEND"] = "Agg"

from kernel.tools import ObservationFrequency, Model, CalendarConvention, RateCurveType
from kernel.models.pricing_engines.enum_pricing_engine import PricingEngineType
from kernel.products.structured_products.autocall_products import Phoenix
from kernel.products.options.vanilla_options import EuropeanCallOption
from kernel.products.options.barrier_options import DownAndInCallOption
from utils.pricing_settings import PricingSettings
from kernel.pricing_launcher import PricingLauncher
from kernel.market_data import Market, VolatilitySurfaceType
from kernel.market_data.data_loader import MarketDataLoader
from kernel.market_data.rate_curve_data.enums_interpolators import InterpolationType

def main():
    start_time = time.time()
    
    # Configure common pricing settings
    settings = PricingSettings()
    settings.underlying_name = "SPX"
    settings.rate_curve_type = RateCurveType.RF_US_TREASURY
    settings.interpolation_type = InterpolationType.SVENSSON
    settings.day_count_convention = CalendarConvention.ACT_360
    settings.obs_frequency = ObservationFrequency.ANNUAL
    settings.pricing_engine_type = PricingEngineType.CALLABLE_MC
    settings.compute_greeks = True
    settings.nb_paths = 50000
    settings.nb_steps = 250
    # settings.random_generator_type = RandomGeneratorType.SOBOL

    autocall = Phoenix(
        maturity=5,
        observation_frequency=ObservationFrequency.ANNUAL,
        capital_barrier=60,
        autocall_barrier=100,
        coupon_barrier=80,
        coupon_rate=5.0
    )

    print("=" * 79)
    print("                           STRUCTURED PRODUCTS PRICING                          ")
    print("=" * 79)
    print(f"Product Type  : Phoenix Autocall")
    print(f"Maturity      : {autocall.maturity} Years")
    print(f"Observation   : {autocall.observation_frequency.name}")
    print(f"Barriers      : Capital: {autocall.capital_barrier}% | Autocall: {autocall.autocall_barrier}% | Coupon: {autocall.coupon_barrier}%")
    print(f"Coupon Rate   : {autocall.coupon_rate}%")
    print(f"Paths / Steps : {settings.nb_paths} / {settings.nb_steps}")
    print("=" * 79)

    # We will run pricing for both models and all vol surface types
    models = [Model.BLACK_SCHOLES, Model.HESTON]
    vol_surfaces = [VolatilitySurfaceType.SVI, VolatilitySurfaceType.SSVI, VolatilitySurfaceType.LOCAL]

    # Load market data once using the new DataLoader
    data_loader = MarketDataLoader()
    underlying_df = data_loader.get_underlying_info(settings.underlying_name)
    options_df = data_loader.get_option_data(settings.underlying_name)
    yield_df = data_loader.get_yield_curve(settings.rate_curve_type.value)

    # Pre-build Market objects for each Volatility Surface to avoid I/O bottlenecks during pricing
    markets = {}
    for vol_surface in vol_surfaces:
        markets[vol_surface] = Market(
            underlying_name=settings.underlying_name,
            yield_curve_data=yield_df,
            underlying_data=underlying_df,
            option_data=options_df,
            rate_curve_type=settings.rate_curve_type,
            interpolation_type=settings.interpolation_type, 
            volatility_surface_type=vol_surface,
            calendar_convention=settings.day_count_convention,
            obs_frequency=settings.obs_frequency
        )

    for model in models:
        header_text = f" MODEL: {model.value.upper()}"
        print(f"\n+{'-' * 77}+")
        print(f"|{header_text:<77}|")
        print(f"+{'-' * 77}+")
        print(f"| Vol Surface |  Price  |  Delta  |  Gamma  |   Vega   |   Theta  |    Rho    |")
        print(f"+-------------+---------+---------+---------+----------+----------+-----------+")

        for vol_surface in vol_surfaces:
            settings.model = model
            settings.volatility_surface_type = vol_surface

            launcher = PricingLauncher(pricing_settings=settings, market=markets[vol_surface])
            res = launcher.calculate(autocall)

            price = res.price if res.price is not None else 0.0
            greeks = res.greeks if res.greeks else {}

            # Retrieve greeks, convert from numpy types if needed, default to 0.0
            delta = float(greeks.get("delta", 0.0))
            gamma = float(greeks.get("gamma", 0.0))
            vega = float(greeks.get("vega", 0.0))
            theta = float(greeks.get("theta", 0.0))
            rho = float(greeks.get("rho", 0.0))

            print(f"| {vol_surface.name:^11} | {price:7.3f} | {delta:7.4f} | {gamma:7.4f} | {vega:8.4f} | {theta:8.3f} | {rho:9.4f} |")

        print(f"+-------------+---------+---------+---------+----------+----------+-----------+")

    print()

    # -------------------------------------------------------------------------
    # STANDARD OPTIONS PRICING COMPARISON
    # -------------------------------------------------------------------------
    # Set pricing engine type to standard Monte Carlo for options
    settings.pricing_engine_type = PricingEngineType.MC

    # Fetch spot price to set option strike and barrier (using SVI market from our cache)
    spot = markets[VolatilitySurfaceType.SVI].underlying_asset.last_price

    options_to_price = [
        ("Vanilla Call Option", EuropeanCallOption(maturity=5.0, strike=spot)),
        ("Down-and-In Call Option", DownAndInCallOption(maturity=5.0, strike=spot, barrier=spot * 0.8))
    ]

    print("=" * 79)
    print("                              STANDARD OPTIONS PRICING                          ")
    print("=" * 79)
    print(f"Spot Price    : {spot}")
    print(f"Maturity      : 5.0 Years")
    print(f"Strike Price  : {spot} (At-The-Money)")
    print(f"Paths / Steps : {settings.nb_paths} / {settings.nb_steps}")
    print("=" * 79)

    for name, option in options_to_price:
        desc = f"{name} (Strike: {option.strike:.1f}"
        if hasattr(option, "barrier"):
            desc += f", Barrier: {option.barrier:.1f}"
        desc += ")"

        print(f"\n+-----------------------------------------------------------------------------+")
        print(f"| {desc:<75} |")
        print(f"+-----------------------------------------------------------------------------+")

        for model in models:
            header_text = f" MODEL: {model.value.upper()}"
            print(f"+{'-' * 77}+")
            print(f"|{header_text:<77}|")
            print(f"+{'-' * 77}+")
            print(f"| Vol Surface |  Price  | Delta |  Gamma  |   Vega   |  Theta   |     Rho     |")
            print(f"+-------------+---------+-------+---------+----------+----------+-------------+")

            for vol_surface in vol_surfaces:
                settings.model = model
                settings.volatility_surface_type = vol_surface

                launcher = PricingLauncher(pricing_settings=settings, market=markets[vol_surface])
                res = launcher.calculate(option)

                price = res.price if res.price is not None else 0.0
                greeks = res.greeks if res.greeks else {}

                delta = float(greeks.get("delta", 0.0))
                gamma = float(greeks.get("gamma", 0.0))
                vega = float(greeks.get("vega", 0.0))
                theta = float(greeks.get("theta", 0.0))
                rho = float(greeks.get("rho", 0.0))

                print(f"| {vol_surface.name:^11} | {price:7.2f} | {delta:5.4f} | {gamma:7.6f} | {vega:8.2f} | {theta:8.2f} | {rho:11.2f} |")

            print(f"+-------------+---------+-------+---------+----------+----------+-------------+")

        print()

    # -------------------------------------------------------------------------
    # QUASI-MONTE CARLO (SOBOL) PRICING COMPARISON
    # -------------------------------------------------------------------------
    # We will adjust nb_paths to a power of 2 for Sobol, which is mathematically optimal
    nb_paths_sobol = 2**16  # 65536 paths
    settings.nb_paths = nb_paths_sobol
    settings.random_generator_type = "SOBOL"

    print("=" * 79)
    print("                      QUASI-MONTE CARLO (SOBOL) PRICING                 ")
    print("=" * 79)
    print(f"Spot Price    : {spot}")
    print(f"Maturity      : 5.0 Years")
    print(f"Strike Price  : {spot} (At-The-Money)")
    print(f"Paths / Steps : {settings.nb_paths} (2^16) / {settings.nb_steps}")
    print("=" * 79)

    option = EuropeanCallOption(maturity=5.0, strike=spot)
    desc = f"Vanilla Call Option (Sobol Sequence, {nb_paths_sobol} paths)"

    print(f"\n+-----------------------------------------------------------------------------+")
    print(f"| {desc:<75} |")
    print(f"+-----------------------------------------------------------------------------+")

    for model in models:
        header_text = f" MODEL: {model.value.upper()}"
        print(f"+{'-' * 77}+")
        print(f"|{header_text:<77}|")
        print(f"+{'-' * 77}+")
        print(f"| Vol Surface |  Price  | Delta |  Gamma  |   Vega   |  Theta   |     Rho     |")
        print(f"+-------------+---------+-------+---------+----------+----------+-------------+")

        for vol_surface in vol_surfaces:
            settings.model = model
            settings.volatility_surface_type = vol_surface

            launcher = PricingLauncher(pricing_settings=settings, market=markets[vol_surface])
            res = launcher.calculate(option)

            price = res.price if res.price is not None else 0.0
            greeks = res.greeks if res.greeks else {}

            delta = float(greeks.get("delta", 0.0))
            gamma = float(greeks.get("gamma", 0.0))
            vega = float(greeks.get("vega", 0.0))
            theta = float(greeks.get("theta", 0.0))
            rho = float(greeks.get("rho", 0.0))

            print(f"| {vol_surface.name:^11} | {price:7.2f} | {delta:5.4f} | {gamma:7.6f} | {vega:8.2f} | {theta:8.2f} | {rho:11.2f} |")

        print(f"+-------------+---------+-------+---------+----------+----------+-------------+")

    print()
    
    end_time = time.time()
    print(f"===============================================================================")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    print(f"===============================================================================")

if __name__ == "__main__":
    main()
    # uv run pytest tests/test_financial_relationships.py
    # uv run pytest -v -s

