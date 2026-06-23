from kernel.market_data import Market
from kernel.models.pricing_engines.enum_pricing_engine import PricingEngineType
from kernel.products.abstract_derivative import AbstractDerivative
from utils.pricing_results import PricingResults
from kernel.products.structured_products import AbstractStructuredProduct
from utils.pricing_settings import PricingSettings 
from kernel.models.pricing_engines.mc_pricing_engine import MCPricingEngine  
from kernel.market_data.data_loader import MarketDataLoader
from typing import Optional


class PricingLauncher:
    """
    The pricing launcher defines the objects used for the pricing as follow:
        - Based on the selected diffusion (BS, Heston...) the associated stochastic process is defined
    """

    def __init__(self, pricing_settings: PricingSettings, market: Optional[Market] = None):
        self.settings = pricing_settings
        if market is not None:
            self.market = market
        else:
            self._init_market()

    def _init_market(self):
        """
        Initializes the market object with the given settings.
        """
        data_loader = MarketDataLoader()
        underlying_df = data_loader.get_underlying_info(self.settings.underlying_name)
        options_df = data_loader.get_option_data(self.settings.underlying_name)
        yield_df = data_loader.get_yield_curve(self.settings.rate_curve_type.value)

        self.market = Market(underlying_name=self.settings.underlying_name,
                             yield_curve_data=yield_df,
                             underlying_data=underlying_df,
                             option_data=options_df,
                             rate_curve_type=self.settings.rate_curve_type,
                             interpolation_type=self.settings.interpolation_type, 
                             volatility_surface_type=self.settings.volatility_surface_type,
                             calendar_convention=self.settings.day_count_convention,
                             obs_frequency=self.settings.obs_frequency)

    def calculate(self, derivative: AbstractDerivative) -> PricingResults:
        """
        Main method to perform the pricing calculation and return results.
        """
        
        # Initialize pricer
        engine = PricingEngineType[self.settings.pricing_engine_type.name].value(market=self.market,settings=self.settings)
        
        # Compute and return the results directly
        return engine.get_results(derivative=derivative)


