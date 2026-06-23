import threading
from typing import Dict
from kernel.market_data.market import Market
from kernel.market_data.volatility_surface.enums_volatility import VolatilitySurfaceType
from kernel.market_data.rate_curve_data.enums_interpolators import InterpolationType
from kernel.market_data.data_loader import MarketDataLoader
from kernel.tools import CalendarConvention, ObservationFrequency, RateCurveType

class MarketCache:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(MarketCache, cls).__new__(cls)
                cls._instance._initialize_markets()
            return cls._instance

    def _initialize_markets(self):
        print("Initializing Global Market Data Cache...")
        self.markets: Dict[VolatilitySurfaceType, Market] = {}
        
        data_loader = MarketDataLoader()
        underlying_name = "SPX"
        rate_curve_type = RateCurveType.RF_US_TREASURY
        
        # Load flat files once
        underlying_df = data_loader.get_underlying_info(underlying_name)
        options_df = data_loader.get_option_data(underlying_name)
        yield_df = data_loader.get_yield_curve(rate_curve_type.value)

        vol_surfaces = [VolatilitySurfaceType.SVI, VolatilitySurfaceType.SSVI, VolatilitySurfaceType.LOCAL]
        
        for vol_surface in vol_surfaces:
            self.markets[vol_surface] = Market(
                underlying_name=underlying_name,
                yield_curve_data=yield_df,
                underlying_data=underlying_df,
                option_data=options_df,
                rate_curve_type=rate_curve_type,
                interpolation_type=InterpolationType.SVENSSON,
                volatility_surface_type=vol_surface,
                calendar_convention=CalendarConvention.ACT_360,
                obs_frequency=ObservationFrequency.ANNUAL
            )
        print("Market Data Initialization Complete.")

    def get_market(self, vol_surface_type: VolatilitySurfaceType) -> Market:
        return self.markets.get(vol_surface_type)
