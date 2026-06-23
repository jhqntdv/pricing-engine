import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from kernel.market_data import RateCurve
from . import AbstractVolatilitySurface


class SVIVolatilitySurface(AbstractVolatilitySurface):
    """
    Defines the SVI raw parametrisation defined by Jim Gatheral (2004) to fit an arbitrage free Implied Volatility surface.

    We choose the "raw" parametrisation as it is the simplest SVI form by we note that "natural" or "jump wings"
    parametrisation can we derived from the raw form to insure parameters interpretability.
    All of those parametrisation are strictly equivalent.

    SVI models the total implied variance defined by :
        w(k, t) = σ(k, t)^2 * t.
    Where :
        - t is the maturity
        - k is the log moneyness (not the strike)

    And the implied variance is derived as :
        v(k, t) = σ(k, t)^2 = w(k, t) / t

    We shall refer to the two-dimensional map (k, t) → w(k, t) as the volatility surface,
    and for any fixed maturity t > 0, the function k → w(k, t) will represent a slice (volatility smile).
    For a given maturity slice, we shall use the notation w(k; χ) where χ represents a set of parameters, and drop the t-dependence.

    The parametrisation we use is defined by the five following parameters giving the total implied variance for any k :
        - a : level of variance, defines a smile vertical translation
        - b : smile slope level
        - p : counter-clockwise smile rotation
        - m : smile right translation
        - sigma : ATM smile curvature
    """

    def __init__(self, option_data: pd.DataFrame, rate_curve: RateCurve):
        """
        Parameters:
            option_data (pd.DataFrame): option market data, must contain the following columns : 'Strike', 'Spot', 'Maturity', 'Implied Volatility'
            rate_curve (RateCurve): rate curve object already calibrated
        """
        super().__init__(option_data, rate_curve)
        self.svi_params = None
        self.svi_params_by_maturity = {}
        self.interpolators = {}

    @staticmethod
    def svi_total_variance(k: np.ndarray, svi_params: np.ndarray) -> float:
        """
        Defines the SVI total implied variance w(k).

        Parameters:
            k (np.ndarray): log moneyness
            svi_params (np.ndarray): five parameters of the SVI [a, b, p, m, sigma]

        Returns:
            float: total implied variance for this log moneyness level
        """
        a, b, rho, m, sigma = svi_params
        return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))

    def compute_weighting_vega(self, spot: float, maturities: np.ndarray,
                               vols: np.ndarray, strikes: np.ndarray) -> np.ndarray:
        """
        Compute vegas to weight the cost function in order to fit better the ATM options.

        Parameters:
            spot (np.ndarray): market data spot
            maturities (np.ndarray): market data maturities
            vols (np.ndarray): market data Implied Volatility from the option data historic
            strikes (np.ndarray): market data strikes

        Returns:
            np.ndarray: options vega
        """
        try:
            r = np.array([self.rate_curve.get_rate(t) for t in maturities])/100
        except Exception:
            r = 0

        vols = vols / 100
        d1 = (np.log(spot / strikes) + (r + 0.5 * vols ** 2) * maturities) / (vols * np.sqrt(maturities))
        return spot * norm.pdf(d1) * np.sqrt(maturities)

    def cost_function_svi(self, svi_params: np.ndarray, log_moneyness : np.ndarray,
                          maturities: np.ndarray, market_implied_vol: np.ndarray,
                          vega: np.ndarray) -> float:
        """
        Defines the MSE cost function for the optimization problem.
        We want to minimize the SVI fitting error :
            i.e. the gap between SVI total implied variance and market data total implied variance.

        Parameters:
            svi_params (np.ndarray): given set of svi parameters [a, b, p, m, sigma]
            log_moneyness (np.ndarray): market data log moneyness
            maturities (np.ndarray): market data maturities
            market_implied_vol (np.ndarray): market data Implied Volatility from the option data historic
            vega (np.ndarray): vega to weight the MSE by putting more importance on ATM options

        Returns:
            float: Mean Squared Error between market data and SVI total implied variance
        """
        # Conversion from market data Implied Volatility to market data total implied variance
        market_total_variance = (market_implied_vol ** 2) * maturities

        # SVI total implied variance
        SVI_total_variance = np.array(self.svi_total_variance(log_moneyness, svi_params))

        return float(np.mean((SVI_total_variance - market_total_variance) ** 2))

    def calibrate_surface(self) -> None:
        """
        Calibrate the volatility surface by fitting SVI parameters for each maturity slice,
        then interpolate the parameters across maturities.
        """
        unique_maturities = self.option_data["Maturity"].unique()
        self.svi_params_by_maturity = {}

        for maturity in unique_maturities:
            # Filter data for the current maturity
            slice_data = self.option_data[self.option_data["Maturity"] == maturity]
            log_moneyness = np.log(slice_data['Strike'].values / self.spot)
            market_vols = slice_data["Implied Volatility"].values / 100
            strikes = slice_data["Strike"].values

            # Compute vega weights
            vega = self.compute_weighting_vega(self.spot,
                                               slice_data["Maturity"].values,
                                               market_vols,
                                               strikes)

            # Initial parameters and bounds
            initial_params = np.array([0.1, 0.1, 0.0, 0.0, 0.1])
            bounds = [(0, None), (0, None), (-1, 1), (None, None), (0, None)]

            # Perform optimization
            result = minimize(self.cost_function_svi, initial_params, method="L-BFGS-B", bounds=bounds,
                              args=(log_moneyness, slice_data["Maturity"].values, market_vols, vega))

            if result.success:
                self.svi_params_by_maturity[maturity] = result.x
                self.is_calibrated = True
            else:
                raise Exception(f"SVI calibration failed for maturity {maturity}: {result.message}")

        # Interpolate SVI parameters across maturities
        self._interpolate_parameters(unique_maturities)

    def _interpolate_parameters(self, maturities: np.ndarray) -> None:
        """
        Interpolate SVI parameters across maturities.
        """
        svi_params_array = np.array([self.svi_params_by_maturity[m] for m in maturities])
        self.interpolators = {
            i: interp1d(maturities, svi_params_array[:, i], kind='cubic', fill_value="extrapolate")
            for i in range(5)
        }

    def get_volatility(self, strike: float, maturity: float) -> float:
        """
        Get the volatility interpolated by the volatility surface at this specific point (Strike * Maturity).

        Parameters:
            strike (float): strike price
            maturity (float): maturity in years

        Returns:
            float: implied volatility
        """
        if not self.interpolators:
            raise Exception("SVI surface not calibrated yet!")

        # Handle flat extrapolation for maturities outside the calibrated range
        min_maturity = min(self.svi_params_by_maturity.keys())
        max_maturity = max(self.svi_params_by_maturity.keys())

        if maturity < min_maturity:
            interpolated_params = np.array([self.svi_params_by_maturity[min_maturity][i] for i in range(5)])
        elif maturity > max_maturity:
            interpolated_params = np.array([self.svi_params_by_maturity[max_maturity][i] for i in range(5)])
        else:
            interpolated_params = np.array([self.interpolators[i](maturity) for i in range(5)])

        log_moneyness = np.log(strike / self.spot)
        total_variance = self.svi_total_variance(log_moneyness, interpolated_params)
        return np.sqrt(total_variance / maturity)


