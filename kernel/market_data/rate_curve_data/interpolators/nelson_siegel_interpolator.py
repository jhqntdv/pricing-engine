from .abstract_interpolator import Interpolator, CalibrationError
import numpy as np
from scipy.optimize import curve_fit

class NelsonSiegelInterpolator(Interpolator):
    """
    Implements the Nelson-Siegel yield curve model for yield curve fitting and interpolation.

    This model provides a parsimonious representation of the yield curve using four parameters.
    It is widely used in fixed income markets to fit and interpolate interest rates across different maturities.
    """

    def __init__(self, maturities: np.ndarray, rates: np.ndarray):
        """
        Initializes the interpolator with observed market rates and calibrates the Nelson-Siegel model.

        Parameters:
            maturities (np.ndarray): A list or array of maturities (in years).
            rates (np.ndarray): A list or array of observed yield rates corresponding to the maturities.
        """
        super().__init__(maturities, rates)

        self.params = None

    @staticmethod
    def _nelson_siegel(t, beta0: float, beta1: float, beta2: float, tau: float ) -> float:
        """
        Computes the yield at a given maturity using the Nelson-Siegel model formula.

        Parameters:
            t (float): Maturity in years.
            beta0, beta1, beta2, tau (floats): Model parameters 

        Returns:
            float: Yield rate for the given maturity.
        """
        return beta0 + beta1 * (1 - np.exp(-t / tau)) / (t / tau) + beta2 * (
            (1 - np.exp(-t / tau)) / (t / tau) - np.exp(-t / tau))

    def calibrate(self) -> np.ndarray:
        """
        Calibrates the Nelson-Siegel parameters by fitting the model to observed yield data.

        Returns:
            np.ndarray: Estimated parameters [beta0, beta1, beta2, tau].
        """
        if len(self.maturities) < 4:
            raise CalibrationError("Insufficient data points for Nelson-Siegel calibration (requires at least 4).")
        p0 = [0.02, -0.02, 0.02, 1.0]  # Initial parameter guess
        try:
            params, _ = curve_fit(self._nelson_siegel, self.maturities, self.rates, p0=p0,
                                  bounds=([-np.inf, -np.inf, -np.inf, 0.01], [np.inf, np.inf, np.inf, np.inf]))
            self.params = np.array(params)
        except Exception as e:
            raise CalibrationError(f"Nelson-Siegel calibration failed: {str(e)}")

    def interpolate(self, t: float) -> float:
        """
        Interpolates the yield for a given maturity using the calibrated Nelson-Siegel model.

        Parameters:
            t (float): Maturity at which to estimate the yield.

        Returns:
            float: Estimated yield rate for the given maturity.
        """
        if self.params is None:
            raise ValueError("Interpolator has not been calibrated. Please call the 'calibrate' method first.")
        return self._nelson_siegel(t, *self.params)
