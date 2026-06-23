from enum import Enum
from .callable_mc_pricing_engine import CallableMCPricingEngine
from .american_mc_pricing_engine  import AmericanMCPricingEngine
from .mc_pricing_engine import MCPricingEngine
from .discounting_pricing_engine import DiscountingPricingEngine

class PricingEngineType(Enum):
    CALLABLE_MC = CallableMCPricingEngine
    MC = MCPricingEngine
    AMERICAN_MC= AmericanMCPricingEngine
    RATE = DiscountingPricingEngine
