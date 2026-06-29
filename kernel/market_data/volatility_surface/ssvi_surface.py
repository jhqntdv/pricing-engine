import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from kernel.market_data import RateCurve
from . import AbstractVolatilitySurface


class SSVIVolatilitySurface(AbstractVolatilitySurface):
    """Surface Stochastic Volatility Inspired (SSVI) volatility surface model.
    """

    def __init__(self, option_data: pd.DataFrame, rate_curve: RateCurve):
        """Parameters:
        option_data (pd.DataFrame): option market data, must contain the following columns : 'Strike', 'Spot', 'Maturity', 'Implied Volatility'
        rate_curve (RateCurve): rate curve object already calibrated
        """
        super().__init__(option_data, rate_curve)
        self.ssvi_params = None
        self.ssvi_ATM_params = None

    @staticmethod
    def _ssvi_atm_variance(maturity: np.ndarray, ssvi_atm_params: np.ndarray) -> float:
        """SSVI total variance function.

        Parameters:
            maturity (np.ndarray): maturity in years
            ssvi_atm_params (np.ndarray): SSVI ATM variance parametrization parameters

        Returns:
            np.ndarray: total variance
        """
        kappa, v0, v_inf = ssvi_atm_params

        # return (((1 - np.exp(-kappa * maturity))/kappa * maturity) * (v0 - v_inf) + v_inf) * maturity
        return (((1 - np.exp(-kappa * maturity))/(kappa * maturity)) * (v0 - v_inf) + v_inf) * maturity # Heston ATM Variance parametrization defined in Section 4.1
    
    @staticmethod
    def _ssvi_total_variance(k: float, atm_variance: float, ssvi_params: np.ndarray) -> float:
        """SSVI total variance function.

        Parameters:
            k (float): log-moneyness
            atm_variance (float): ATM variance
            ssvi_params (np.ndarray): SSVI parameters [rho, eta, gamma]

        Returns:
            float: SSVI total variance
        """
        rho, eta, gamma = ssvi_params
        
        # Gatheral 2014 Power-Law parametrization
        phi = lambda theta: eta / ((theta ** gamma) * ((1 + theta) ** (1 - gamma)))
        
        return 0.5 * atm_variance * ( 1 + rho * phi(atm_variance) * k + np.sqrt((phi(atm_variance) * k + rho)**2 + (1 - rho **2)))
    
    def _get_market_atm_variance(self, maturity: float) -> float:
        """Get the ATM variance for a given maturity from the option data.
        This function checks if an ATM option is available in the data.
        If not, it interpolates the ATM variance from the available options of the slice.

        Parameters:
            maturity (float): maturity in years

        Returns:
            float: ATM variance
        """
        # Filter the option data for the given maturity
        option_slice = self.option_data[self.option_data["Maturity"] == maturity]
        if option_slice.empty:
            raise ValueError(f"No option data available for maturity {maturity}")

        # If an ATM option is available, use its implied volatility
        atm_option = option_slice[option_slice["Strike"] == self.spot]
        if not atm_option.empty:
            return maturity * (atm_option["Implied Volatility"].values[0] / 100) ** 2
        
        # Otherwise, interpolate the ATM variance from the option slice data
        else:
            atm_vol = np.interp(self.spot, option_slice["Strike"].values, option_slice["Implied Volatility"].values)
            return maturity * (atm_vol / 100) ** 2
    
    def _get_atm_variance(self, maturity: float) -> float:
        """Get the ATM variance for a given maturity from the option data.

        Parameters:
            maturity (float): maturity in years

        Returns:
            float: ATM variance
        """
        if self.ssvi_ATM_params is None:
            raise ValueError("SSVI ATM parameters are not calibrated. Please call calibrate_atm_variance() first.")
        
        return self._ssvi_atm_variance(maturity, self.ssvi_ATM_params)
    
    def _ssvi_atm_cost_function(self, ssvi_atm_params: np.ndarray, maturities: np.ndarray, implied_ATM_variance: np.ndarray) -> float:
        """Cost function for the SSVI ATM calibration.
        The calibration is done by minimizing the mean squared error between the market implied volatility and the SSVI model implied volatility.

        Parameters:
            ssvi_atm_params (np.ndarray): SSVI ATM variance parametrization parameters
            maturities (np.ndarray): maturities in years
            implied_ATM_variance (np.ndarray): market implied ATM variance

        Returns:
            float: mean squared error between the market implied volatility and the SSVI model implied volatility for ATM options
        """
        ssvi_ATM_variance = self._ssvi_atm_variance(maturities, ssvi_atm_params)

        return np.mean((implied_ATM_variance - ssvi_ATM_variance) ** 2)
    
    def _ssvi_objective_function(self, ssvi_params: np.ndarray, option_data: pd.DataFrame) -> float:
        """Objective function for the SSVI calibration.
        The calibration is done by minimizing the mean squared error between the market implied volatility and the SSVI model implied volatility.
        For each maturity, the ATM variance is computed and used to compute the SSVI implied volatility.

        Parameters:
            ssvi_params (np.ndarray): SSVI parameters
            option_data (pd.DataFrame): option market data, must contain the following columns : 'Strike', 'Spot', 'Maturity', 'Implied Volatility'

        Returns:
            float: mean squared error between the market implied volatility and the SSVI model implied volatility
        """
        ssvi_total_variance = []
        market_total_variance = []
        for maturity in option_data["Maturity"].unique():
            # Filter the option data for the given maturity
            slice_options = option_data[option_data["Maturity"] == maturity]

            # For each maturity, compute the ATM variance and the SSVI implied volatility
            atm_variance = self._get_atm_variance(maturity)
            
            k = np.log(slice_options["Strike"].values / self.spot)
            ssvi_total_variance.append(self._ssvi_total_variance(k, atm_variance, ssvi_params))

            # Also compute the market data total implied variance
            market_total_variance.append(((slice_options["Implied Volatility"].values / 100) ** 2) * maturity)

        # Flatten the list of total variance arrays and market total variance arrays
        ssvi_total_variance = np.concatenate(ssvi_total_variance)
        market_total_variance = np.concatenate(market_total_variance)

        # Scale MSE by 1e6 to ensure SLSQP has large enough gradients to descend
        return np.mean((market_total_variance - ssvi_total_variance) ** 2) * 1e6

    def calibrate_atm_variance(self):
        """Calibrate the ATM variance using the option data.
        This function uses the least squares method to minimize the difference between the market implied volatility and the SSVI model implied volatility at ATM.
        """
        maturities = self.option_data["Maturity"].unique()
        atm_market_variance = np.array([self._get_market_atm_variance(maturity) for maturity in maturities])

        # Calibrate the ATM variance wit ATM options
        initial_values = [0.1, 0.2, 0.2]  # Initial guess for the ATM parameters [kappa, v0, v_inf]

        res = minimize(
            self._ssvi_atm_cost_function,
            initial_values,
            args=(maturities, atm_market_variance),
            method="Nelder-Mead",
        )
        
        if res.success:
            self.ssvi_ATM_params = res.x
        else:
            raise Exception(f"SSVI ATM parametrization calibration failed : {res.message}")

    def calibrate_surface(self):
        """Calibrate the SSVI parameters using the option data.
        This function uses the least squares method to minimize the difference between the market implied volatility and the SSVI model implied volatility.
        
        The calibration is done for the whole maturity range of the option data.
        First we calibrate the ATM variance and then we calibrate the SSVI parameters.
        """
        # Calibrate the ATM variance first
        self.calibrate_atm_variance()

        # Initial guess for the SSVI parameters [rho, eta, gamma]
        initial_values = [0.1, 0.1, 0.1]
        
        # Arbitrage-free bounds for Power-Law SSVI:
        # rho in (-1, 1), eta > 0, gamma in (0, 0.5]
        bounds = [(-0.999, 0.999), (1e-5, None), (1e-5, 0.5)]

        # Arbitrage-free inequality constraint: eta * (1 + |rho|) <= 2
        # For SLSQP, inequality constraints must be >= 0. So we rewrite as: 2 - eta * (1 + |rho|) >= 0
        constraints = [
            {'type': 'ineq', 'fun': lambda x: 2.0 - x[1] * (1.0 + np.abs(x[0]))}
        ]

        res = minimize(
            self._ssvi_objective_function,
            initial_values,
            args=(self.option_data),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )

        if res.success:
            self.ssvi_params = res.x
            self.is_calibrated = True
        else:
            raise Exception(f"SSVI calibration failed : {res.message}")
        
    def get_volatility(self, strike: float, maturity: float) -> float:
        """Get the volatility interpolated by the SSVI model for a given strike and maturity.

        Parameters:
            strike (float): strike price
            maturity (float): maturity in years

        Returns:
            float: implied volatility
        """
        if self.ssvi_params is None:
            raise ValueError("SSVI parameters are not calibrated. Please call calibrate_surface() first.")

        atm_variance = self._get_atm_variance(maturity)

        k = np.log(strike / self.spot)
        ssvi_total_variance = self._ssvi_total_variance(k, atm_variance, self.ssvi_params)

        return np.sqrt(ssvi_total_variance / maturity)
    
