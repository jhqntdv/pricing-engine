import numpy as np
from .abstract_interpolator import Interpolator
from scipy.interpolate import CubicSpline

class CubicInterpolator(Interpolator):

    def __init__(self, maturities: np.ndarray, rates: np.ndarray):
        """
        Initializes the CubicInterpolator with market maturities and corresponding rates.

        Parameters:
            maturities (np.ndarray): List of bond maturities.
            rates (np.ndarray): List of observed yield rates corresponding to the maturities.
        """
        super().__init__(maturities, rates)
        self.interpolator = None

    def calibrate(self):
        """
        Calibrates the CubicInterpolator by fitting the observed market rates.
        """
        self.interpolator = CubicSpline(self.maturities, self.rates, bc_type='natural', extrapolate=True)

    def interpolate(self, t: float) -> float:
        """
        Interpolates the yield for a given maturity using the calibrated CubicSpline model.

        Parameters:
            t (float): Maturity at which to estimate the yield.

        Returns:
            float: Estimated yield rate for the given maturity.
        """
        if self.interpolator is None:
            raise ValueError("Interpolator has not been calibrated. Please call the 'calibrate' method first.")
        return self.interpolator(t)