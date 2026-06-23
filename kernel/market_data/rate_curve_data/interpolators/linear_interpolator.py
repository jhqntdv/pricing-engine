from .abstract_interpolator import Interpolator
from scipy.interpolate import interp1d

class LinearInterpolator(Interpolator):

    def __init__(self, maturities, rates):
        """
        Initializes the LinearInterpolator with market maturities and corresponding rates.

        Parameters:
            maturities (np.ndarray): List of bond maturities.
            rates (np.ndarray): List of observed yield rates corresponding to the maturities.
        """
        super().__init__(maturities, rates)
        self.interpolator = None

    def calibrate(self):
        """
        Calibrates the LinearInterpolator by fitting the observed market rates.
        """
        self.interpolator = interp1d(self.maturities, self.rates, kind='linear', fill_value='extrapolate')
        
    def interpolate(self, t: float) -> float:
        """
        Interpolates the yield for a given maturity using the calibrated linear model.

        Parameters:
            t (float): Maturity at which to estimate the yield.

        Returns:
            float: Estimated yield rate for the given maturity.
        """
        if self.interpolator is None:
            raise ValueError("Interpolator has not been calibrated. Please call the 'calibrate' method first.")
        return float(self.interpolator(t))