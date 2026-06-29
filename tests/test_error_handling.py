import pytest
from datetime import date
from kernel.exceptions import (
    PricingEngineError,
    ConfigurationError,
    UnsupportedModelError,
    UnsupportedEngineTypeError,
    UnsupportedProductError,
    InvalidProductInputError,
    IndeterminateValuationError
)
from kernel.pricing_launcher import PricingLauncher
from utils.pricing_settings import PricingSettings
from kernel.models.pricing_engines.enum_pricing_engine import PricingEngineType
from kernel.market_data import InterpolationType, VolatilitySurfaceType
from kernel.models.pricing_engines.discounting_pricing_engine import DiscountingPricingEngine
from kernel.models.pricing_engines.mc_pricing_engine import MCPricingEngine
from kernel.products.options.vanilla_options import EuropeanCallOption
from kernel.products.rate.bond import ZeroCouponBond
from kernel.market_data.market import Market
import pandas as pd

def test_unsupported_engine_type_error_compatibility():
    """Verify that UnsupportedEngineTypeError can be caught by KeyError."""
    try:
        raise UnsupportedEngineTypeError("Test engine error")
    except KeyError:
        pass  # Expected
    except Exception as e:
        pytest.fail(f"Expected KeyError to catch it, but got {type(e)}")

def test_unsupported_model_error_compatibility():
    """Verify that UnsupportedModelError can be caught by ValueError."""
    try:
        raise UnsupportedModelError("Test model error")
    except ValueError:
        pass  # Expected
    except Exception as e:
        pytest.fail(f"Expected ValueError to catch it, but got {type(e)}")

def test_unsupported_product_error_compatibility():
    """Verify that UnsupportedProductError can be caught by NotImplementedError."""
    try:
        raise UnsupportedProductError("Test product error")
    except NotImplementedError:
        pass  # Expected
    except Exception as e:
        pytest.fail(f"Expected NotImplementedError to catch it, but got {type(e)}")

def test_invalid_product_input_error_compatibility():
    """Verify that InvalidProductInputError can be caught by ValueError."""
    try:
        raise InvalidProductInputError("Test input error")
    except ValueError:
        pass  # Expected
    except Exception as e:
        pytest.fail(f"Expected ValueError to catch it, but got {type(e)}")

def test_indeterminate_valuation_error_compatibility():
    """Verify that IndeterminateValuationError can be caught by ZeroDivisionError."""
    try:
        raise IndeterminateValuationError("Test valuation error")
    except ZeroDivisionError:
        pass  # Expected
    except Exception as e:
        pytest.fail(f"Expected ZeroDivisionError to catch it, but got {type(e)}")

def test_pricing_launcher_raises_unsupported_engine_type():
    """Verify PricingLauncher raises UnsupportedEngineTypeError for unknown engine."""
    class DummyType:
        name = "INVALID_ENGINE_NAME"

    settings = PricingSettings(underlying_name="AAPL")
    settings.pricing_engine_type = DummyType()

    launcher = PricingLauncher(pricing_settings=settings, market="dummy")
    
    with pytest.raises(UnsupportedEngineTypeError):
        # Pass dummy derivative just to trigger the engine init
        launcher.calculate(derivative=None)

def test_mc_pricing_engine_raises_unsupported_product():
    """Verify MCPricingEngine raises UnsupportedProductError for rate products."""
    settings = PricingSettings()
    engine = MCPricingEngine(market=None, settings=settings)
    
    with pytest.raises(UnsupportedProductError, match="does not support rate products"):
        engine.calculate_rate_product(derivative=None)

def test_discounting_pricing_engine_raises_unsupported_product():
    """Verify DiscountingPricingEngine raises UnsupportedProductError for options."""
    settings = PricingSettings()
    engine = DiscountingPricingEngine(market=None, settings=settings)
    
    option = EuropeanCallOption(maturity=1.0, strike=100)
    with pytest.raises(UnsupportedProductError, match="does not support options"):
        engine.calculate_option(derivative=option)

from kernel.tools import CalendarConvention
from datetime import datetime

def test_discounting_pricing_engine_invalid_bond_input():
    """Verify DiscountingPricingEngine raises InvalidProductInputError on missing bond inputs."""
    settings = PricingSettings()
    engine = DiscountingPricingEngine(market=None, settings=settings)
    
    # Missing price and ytm
    bond = ZeroCouponBond(issue_date=datetime(2023,1,1), maturity=datetime(2024,1,1), notional=100, calendar_convention=CalendarConvention.ACT_360)
    bond.price = None
    bond.ytm = None
    
    with pytest.raises(InvalidProductInputError, match="You must provide either ytm or the price"):
        engine._price_bond(bond)
