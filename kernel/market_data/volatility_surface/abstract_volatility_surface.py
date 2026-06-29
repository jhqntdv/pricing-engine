import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from kernel.market_data import RateCurve

class AbstractVolatilitySurface(ABC):
    """Abstract base class for all volatility surfaces.
    It encapsulates shared attributes, calibration status, and visualization methods
    to ensure DRY (Don't Repeat Yourself) principles across all models.
    """

    def __init__(self, option_data: pd.DataFrame, rate_curve: RateCurve):
        """Initializes the base surface with market data and rate curve.
        """
        self.option_data = option_data
        self.rate_curve = rate_curve
        self.spot = option_data["Spot"].values[0]
        self.is_calibrated = False

    @abstractmethod
    def calibrate_surface(self) -> None:
        """Calibrates the model parameters to the market data."""
        pass

    @abstractmethod
    def get_volatility(self, strike: float, maturity: float) -> float:
        """Returns the model-implied volatility for a given strike and maturity."""
        pass
