import pytest
from kernel.market_data.data_loader import MarketDataLoader
from kernel.market_data.market import Market
from kernel.market_data.volatility_surface.enums_volatility import VolatilitySurfaceType
from kernel.market_data.rate_curve_data.enums_interpolators import InterpolationType
from kernel.tools import CalendarConvention, ObservationFrequency, RateCurveType, Model
from utils.pricing_settings import PricingSettings
from kernel.pricing_launcher import PricingLauncher
from kernel.products.options.vanilla_options import EuropeanCallOption
from kernel.products.rate.bond import ZeroCouponBond
from datetime import datetime

def test_end_to_end_pricing():
    # 1. Load Data
    data_loader = MarketDataLoader()
    underlying_name = "SPX"
    rate_curve_type = RateCurveType.RF_US_TREASURY
    
    underlying_df = data_loader.get_underlying_info(underlying_name)
    options_df = data_loader.get_option_data(underlying_name)
    yield_df = data_loader.get_yield_curve(rate_curve_type.value)

    # 2. Build Market
    market = Market(
        underlying_name=underlying_name,
        yield_curve_data=yield_df,
        underlying_data=underlying_df,
        option_data=options_df,
        rate_curve_type=rate_curve_type,
        interpolation_type=InterpolationType.SVENSSON,
        volatility_surface_type=VolatilitySurfaceType.SVI,
        calendar_convention=CalendarConvention.ACT_360,
        obs_frequency=ObservationFrequency.ANNUAL
    )

    # 3. Settings
    from kernel.models.pricing_engines.enum_pricing_engine import PricingEngineType
    settings = PricingSettings()
    settings.underlying_name = underlying_name
    settings.model = Model.BLACK_SCHOLES
    settings.volatility_surface_type = VolatilitySurfaceType.SVI
    settings.pricing_engine_type = PricingEngineType.MC
    settings.compute_greeks = True
    settings.nb_paths = 1000
    settings.nb_steps = 10

    # 4. Price an Option
    launcher = PricingLauncher(pricing_settings=settings, market=market)
    
    call_option = EuropeanCallOption(strike=5000, maturity=1.0)
    res_call = launcher.calculate(call_option)
    assert res_call.price > 0
    assert "delta" in res_call.greeks

    # 5. Price a Rate Product
    settings.pricing_engine_type = PricingEngineType.RATE
    bond = ZeroCouponBond(
        notional=100.0,
        issue_date=datetime.now(),
        maturity=datetime.now().replace(year=datetime.now().year + 2),
        calendar_convention=CalendarConvention.ACT_360,
        ytm=0.05
    )
    res_bond = launcher.calculate(bond)
    assert res_bond.price > 0
