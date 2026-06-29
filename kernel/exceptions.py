"""Exception hierarchy for the pricing engine."""

class PricingEngineError(Exception):
    """Base class for all pricing engine errors."""
    pass

class ConfigurationError(PricingEngineError):
    """Base class for configuration-related errors."""
    pass

class UnsupportedModelError(ConfigurationError, ValueError):
    """Raised when a pricing model is not supported."""
    pass

class UnsupportedEngineTypeError(ConfigurationError, KeyError):
    """Raised when an unknown engine type is specified."""
    pass

class UnsupportedProductError(PricingEngineError, NotImplementedError):
    """Raised when a pricing engine does not support a specific product type."""
    pass

class InvalidProductInputError(PricingEngineError, ValueError):
    """Raised when a product is initialized with invalid inputs."""
    pass

class IndeterminateValuationError(PricingEngineError, ZeroDivisionError):
    """Raised when a valuation formula encounters an indeterminate form (e.g. 0/0)."""
    pass

class CalibrationError(PricingEngineError):
    """Raised when model or curve calibration fails."""
    pass
