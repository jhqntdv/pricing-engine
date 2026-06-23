from .abstract_interpolator import Interpolator
from scipy.optimize import curve_fit
import numpy as np


class SvenssonInterpolator(Interpolator):
    """
    Implements the Svensson model for yield curve interpolation, an extension of the Nelson-Siegel model.
    The Svensson model provides a smooth parametric representation of the yield curve using six parameters.

    The yield curve is defined as:
        r(t) = beta0 + term1 + term2 + term3
    where:
        - term1 accounts for short-term behavior,
        - term2 accounts for medium-term behavior,
        - term3 accounts for long-term behavior.

    The model ensures flexibility in fitting different shapes of yield curves, including humps and slopes.
    """

    def __init__(self, maturities: np.ndarray, rates: np.ndarray):
        """
        Initializes the SvenssonInterpolator with market maturities and corresponding rates.

        Parameters:
            maturities (np.ndarray): List of bond maturities.
            rates (np.ndarray): List of observed yield rates corresponding to the maturities.
        """
        super().__init__(maturities, rates)
        
        self.params = None

    @staticmethod
    def _svensson(t, beta0: float, beta1: float, beta2: float, beta3: float, tau1: float, tau2: float) -> float:
        """
        Computes the Svensson yield curve function at a given maturity t.

        Parameters:
            t (float): Maturity at which to compute the yield.
            beta0, beta1, beta2, beta3, tau1, tau2 (floats): Svensson model parameters

        Returns:
            float: Yield rate for the given maturity.
        """

        term1 = beta1 * (1 - np.exp(-t / tau1)) / (t / tau1)
        term2 = beta2 * ((1 - np.exp(-t / tau1)) / (t / tau1) - np.exp(-t / tau1))
        term3 = beta3 * ((1 - np.exp(-t / tau2)) / (t / tau2) - np.exp(-t / tau2))

        return beta0 + term1 + term2 + term3

    def calibrate(self) -> np.ndarray:
        """
        Calibrates the Svensson model parameters by fitting the yield curve to observed market rates.

        Uses non-linear least squares optimization to estimate beta0, beta1, beta2, beta3, tau1, tau2.

        Returns:
            np.ndarray: Estimated parameters (beta0, beta1, beta2, beta3, tau1, tau2).
        """
        p0 = [0.02, -0.02, 0.02, 0.01, 1.0, 2.0] 
        bounds = ([0, -np.inf, -np.inf, -np.inf, 0.01, 0.01], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
        
        params, _ = curve_fit(self._svensson, self.maturities, self.rates, p0=p0, bounds=bounds)
        self.params =  np.array(params)

    def interpolate(self, t: float) -> float:
        """
        Interpolates the yield for a given maturity using the calibrated Svensson model.

        Parameters:
            t (float): Maturity at which to estimate the yield.

        Returns:
            float: Estimated yield rate for the given maturity.
        """
        if self.params is None:
            raise ValueError("Interpolator has not been calibrated. Please call the 'calibrate' method first.")
        return self._svensson(t, *self.params)
