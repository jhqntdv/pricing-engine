import numpy as np
from abc import ABC, abstractmethod

class CalibrationError(Exception):
    """Exception raised when an interpolator fails to calibrate."""
    pass

class Interpolator(ABC):

    def __init__(self, maturities: np.ndarray, rates: np.ndarray):
        """
        Initializes the interpolator with observed market rates and calibrates it.

        Parameters:
            maturities (np.ndarray): Array of maturities (in years).
            rates (np.ndarray): Array of observed yield rates corresponding to the maturities.
        """
        self.maturities = maturities
        self.rates = rates

    @abstractmethod
    def calibrate(self):
        """
        Calibrates the interpolator to fit the observed market rates.
        """
        pass

    @abstractmethod
    def interpolate(self, t: float) -> float:
        """
        Interpolates the yield for a given maturity.

        Parameters:
            t (float): Maturity at which to estimate the yield.

        Returns:
            float: Estimated yield rate for the given maturity.
        """
        pass
