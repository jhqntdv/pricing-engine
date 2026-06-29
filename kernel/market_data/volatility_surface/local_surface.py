import numpy as np
import pandas as pd
from scipy.stats import norm
from . import AbstractVolatilitySurface
from kernel.market_data import RateCurve

class LocalVolatilitySurface(AbstractVolatilitySurface):
    """Defines the Local Volatility Surface based on Gatheral's formulation of Dupire's formula.
    
    Rather than differentiating Call prices (which is numerically unstable for OTM options
    due to small prices), this class computes local volatility by differentiating the
    Total Implied Variance surface w(k, t) = sigma_imp(k, t)^2 * t, where:
        - k is the log-moneyness: k = ln(K/S)
        - t is the maturity (time to expiration in years)
    
    Gatheral's formula for local variance is:
        sigma_local^2(k, t) = (dw/dt) / [1 - (k/w)*(dw/dk) + 0.25*(-0.25 - 1/w + k^2/w^2)*(dw/dk)^2 + 0.5*(d2w/dk2)]
    
    Attributes:
        option_data (pd.DataFrame): Market option data containing columns: 'Strike', 'Spot',
                                    'Maturity', and 'Implied Volatility'.
        rate_curve (RateCurve): Object providing the risk-free yield curve.
        svi_surface (SVIVolatilitySurface): Calibrated SVI surface to evaluate implied volatility.
        spot (float): Spot price of the underlying asset.
    """

    def __init__(self, option_data: pd.DataFrame, rate_curve: RateCurve, svi_surface: 'SVIVolatilitySurface'): # type: ignore
        """Initialize the Local Volatility Surface.

        Args:
            option_data: Market option data.
            rate_curve: The risk-free rate curve.
            svi_surface: A calibrated SVI volatility surface.
        """
        super().__init__(option_data, rate_curve)
        
        # Ensure that the SVI surface is calibrated before usage
        if not svi_surface.is_calibrated:
            svi_surface.calibrate_surface()
            
        self.svi_surface = svi_surface
        self.is_calibrated = True  # Local Volatility is implicitly calibrated via SVI

    def calibrate_surface(self):
        """No calibration is needed for the Local Volatility Surface as it is directly
        derived from the calibrated SVI Implied Volatility surface.
        """
        pass

    def _option_price(self, strike: float, maturity: float) -> float:
        """Compute the price of a call option using the Black-Scholes formula.
        Used for visualization purposes.
        """
        S = self.spot
        sigma = self.svi_surface.get_volatility(strike, maturity)
        r = self.rate_curve.get_rate(maturity) / 100.0

        d1 = (np.log(S / strike) + (r + 0.5 * sigma ** 2) * maturity) / (sigma * np.sqrt(maturity))
        d2 = d1 - sigma * np.sqrt(maturity)

        return S * norm.cdf(d1) - strike * np.exp(-r * maturity) * norm.cdf(d2)

    def _total_variance(self, k: float, t: float) -> float:
        """Compute the Total Implied Variance w(k, t) = sigma_imp^2(K, T) * t
        where k = ln(K/S).
        """
        K = self.spot * np.exp(k)
        sigma_imp = self.svi_surface.get_volatility(K, t)
        return (sigma_imp ** 2) * t

    def _finite_difference_variance(self, k: float, t: float) -> tuple:
        """Compute the partial derivatives of the Total Variance surface w(k, t)
        using custom user-specified finite difference steps.
        
        Log-moneyness bump:
            - dK = max(K * 0.05, 0.01) where K is the strike
            - dk = ln(1 + dK / K) to define a symmetric log-moneyness bump
            
        Time bump:
            - dt = 0.004 (approx. 1 trading day)
            - If t < 0.004, dt = 0.0001
        """
        # Determine the strike and strike bump
        strike = self.spot * np.exp(k)
        dK = max(strike * 0.05, 0.01)
        
        # Translate to symmetric log-moneyness bump
        dk = np.log(1.0 + dK / strike)

        # Determine time bump
        dt = 0.004
        if t < 0.004:
            dt = 0.0001

        # Evaluate total variance at the evaluation point
        w = self._total_variance(k, t)

        # Log-moneyness derivatives (Symmetric Central Difference)
        w_up_k = self._total_variance(k + dk, t)
        w_dn_k = self._total_variance(k - dk, t)
        dw_dk = (w_up_k - w_dn_k) / (2 * dk)
        d2w_dk2 = (w_up_k - 2 * w + w_dn_k) / (dk ** 2)

        # Time derivative (Central Difference with Forward Difference near boundary t=0)
        if t - dt < 1e-8:
            w_up_t = self._total_variance(k, t + dt)
            dw_dt = (w_up_t - w) / dt
        else:
            w_up_t = self._total_variance(k, t + dt)
            w_dn_t = self._total_variance(k, t - dt)
            dw_dt = (w_up_t - w_dn_t) / (2 * dt)

        return w, dw_dk, d2w_dk2, dw_dt

    def get_volatility(self, strike: float, maturity: float) -> float:
        """Compute the local volatility using Gatheral's Total Variance Dupire Formula.
        
        Volatility values are capped at 350% (3.5) and floored at 5% (0.05).
        """
        k = np.log(strike / self.spot)
        t = maturity
        
        # Extrapolate flat volatility if maturity is virtually zero
        if t <= 1e-6:
            t = 1e-6
            
        w, dw_dk, d2w_dk2, dw_dt = self._finite_difference_variance(k, t)

        if w <= 1e-12:
            return 0.05  # Floor

        term1 = 1.0
        term2 = - (k / w) * dw_dk
        term3 = 0.25 * (-0.25 - (1.0 / w) + (k ** 2) / (w ** 2)) * (dw_dk ** 2)
        term4 = 0.5 * d2w_dk2

        denominator = term1 + term2 + term3 + term4
        numerator = dw_dt

        # Capped at 350% Volatility, Floored at 5% Volatility.
        if denominator <= 1e-8 or numerator < 0:
            fallback_var = numerator / max(denominator, 1e-8)
            if fallback_var < 0:
                return 0.05
            return min(max(np.sqrt(fallback_var), 0.05), 3.5)

        local_var = numerator / denominator
        local_vol = np.sqrt(local_var)
        
        return min(max(local_vol, 0.05), 3.5)

