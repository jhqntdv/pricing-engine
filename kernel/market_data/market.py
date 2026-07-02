import os
import numpy as np
import pandas as pd
from kernel.tools import RateCurveType, CalendarConvention, ObservationFrequency
from .rate_curve_data.enums_interpolators import InterpolationType
from .rate_curve_data.rate_curve import RateCurve
from .underlying_asset import UnderlyingAsset
from .volatility_surface.enums_volatility import VolatilitySurfaceType
from .volatility_surface.svi_surface import SVIVolatilitySurface
from .volatility_surface.ssvi_surface import SSVIVolatilitySurface
from .volatility_surface.local_surface import LocalVolatilitySurface
import re

import copy


class Market:
    """Represents a financial market environment, providing tools to fetch and use yield curves,
    compute interest rates, and discount factors.

    Attributes:
        rate_curve_type (RateCurveType): The type of rate curve to use (e.g., RF_US_TREASURY)
        interpolation_type (InterpolationType): The interpolation method for the rate curve (e.g., CUBIC)
        volatility_surface_type (VolatilitySurfaceType): The type of volatility surface to use (e.g., SVI)
        calendar_convention (CalendarConvention): The calendar convention for calculations (e.g., ACT_360)
        rate_curve (RateCurve): The rate curve object containing yield curve data and interpolation logic
    """

    def __init__(self, underlying_name: str, 
                 yield_curve_data: pd.DataFrame,
                 underlying_data: pd.DataFrame,
                 option_data: pd.DataFrame,
                 rate_curve_type: RateCurveType = RateCurveType.RF_US_TREASURY, 
                 interpolation_type: InterpolationType = InterpolationType.CUBIC,
                 volatility_surface_type: VolatilitySurfaceType = VolatilitySurfaceType.SVI,
                 calendar_convention: CalendarConvention = CalendarConvention.ACT_360,
                 obs_frequency: ObservationFrequency = ObservationFrequency.ANNUAL):
        """Initializes the Market object with specified configurations and injected data.

        Parameters:
            underlying_name (str): Ticker of the underlying asset
            yield_curve_data (pd.DataFrame): Dataframe containing the yield curve
            underlying_data (pd.DataFrame): Dataframe row containing the underlying asset info
            option_data (pd.DataFrame): Dataframe containing option chain
            rate_curve_type (RateCurveType): The type of rate curve to use
            interpolation_type (InterpolationType): The interpolation method for the rate curve
            volatility_surface_type (VolatilitySurfaceType): The type of volatility surface to use
            calendar_convention (CalendarConvention): The calendar convention for calculations
        """
        self.rate_curve_type = rate_curve_type
        self.interpolation_type = interpolation_type
        self.volatility_surface_type = volatility_surface_type
        self.calendar_convention = calendar_convention
        self.obs_frequency = obs_frequency
        
        self.rate_curve = None
        # Store raw data for bump methods
        self._raw_yield_data = yield_curve_data
        self._raw_option_data = option_data
        
        self._build_yield_curves(yield_curve_data)

        self.underlying_asset = UnderlyingAsset(underlying_name)
        self._build_underlying_info(underlying_data)

        self.volatility_surface = None
        self._build_volatility_surface(option_data)

    @staticmethod
    def _convert_maturities(maturity: str) -> float:
        """Converts a maturity string (e.g., '10Y', '6M', '3W') into a numerical value in years.

        Parameters:
            maturity (str): Maturity in standard financial format (e.g., '10Y', '6M', '3W').

        Returns:
            float: Maturity expressed in years.
        """
        match = re.match(r"(\d+)([MWY])", maturity)
        if not match:
            raise ValueError(f"Invalid maturity: {maturity}")

        value, unit = int(match.group(1)), match.group(2)
        
        if unit == "W":
            return value / 52  # Convert weeks to years
        elif unit == "M":
            return value / 12  # Convert months to years
        elif unit == "Y":
            return value       # Already in years
        else:
            raise ValueError(f"Unrecognized unit: {unit}")
        
    def _build_yield_curves(self, data_curve: pd.DataFrame, bump: float = 0.0):
        """Builds and calibrates the rate curve from the provided DataFrame.
        """
        required_columns = ["Maturity", "Rate"]
        missing_columns = [col for col in required_columns if col not in data_curve.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        
        data_curve = data_curve.copy()
        data_curve["Maturity"] = data_curve["Maturity"].astype(str).apply(self._convert_maturities)
        data_curve["Rate"] = data_curve["Rate"].astype(float) + bump
        rate_curve = RateCurve(data_curve=data_curve, interpolation_type=self.interpolation_type)
        rate_curve.calibrate()

        self.rate_curve = rate_curve

    def _build_underlying_info(self, asset_info: pd.DataFrame):
        """Initializes the underlying asset data from the provided DataFrame row.
        """
        if asset_info.empty:
            raise ValueError(f"No data found for security name: {self.underlying_asset.name}")
        
        self.underlying_asset.load_underlying_info(asset_info)

    def _build_volatility_surface(self, option_data: pd.DataFrame, bump: float = 0.0):
        """Builds and calibrates the volatility surface from the provided Option DataFrame.
        """
        required_columns = ["Maturity", "Implied Volatility", "Strike"]
        missing_columns = [col for col in required_columns if col not in option_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        
        option_data = option_data.copy()
        # Conversions
        option_data['Implied Volatility'] = option_data['Implied Volatility'].astype(float) + bump
        option_data['Strike'] = option_data['Strike'].astype(float)
        option_data["Maturity"] = option_data["Maturity"].astype(str).apply(self._convert_maturities)
        option_data["Spot"] = self.underlying_asset.last_price
        
        if self.rate_curve is None:
            raise ValueError("Rate curve must be built before volatility surface")
        
        if self.volatility_surface_type.name in ["LOCAL"]:
            svi_surface = SVIVolatilitySurface(option_data=option_data, rate_curve=self.rate_curve)
            svi_surface.calibrate_surface()
            volatility_surface = self.volatility_surface_type.value(option_data=option_data, rate_curve=self.rate_curve, svi_surface=svi_surface)
        else:
            volatility_surface = self.volatility_surface_type.value(option_data=option_data, rate_curve=self.rate_curve)

        volatility_surface.calibrate_surface()
        self.volatility_surface = volatility_surface

    def get_rate(self, maturity: float) -> float:
        """Retrieves the interest rate for a given maturity.

        Parameters:
            maturity (float): Desired maturity in years

        Returns:
            float: Interpolated yield rate
        """
        if not self.rate_curve:
            raise ValueError("Rate curve not initialized")
        return self.rate_curve.get_rate(maturity) / 100
    
    def get_fwd_rate(self, start: float, end: float) -> float:
        """Computes the implied forward rate between two maturities.

        Parameters:
            start (float): Start maturity in years (e.g. 1.0 for 1 year)
            end (float): End maturity in years (must be > start)

        Returns:
            float: Forward rate
        """
        if not self.rate_curve:
            self._fetch_yield_curves()
    
        if end <= start:
            raise ValueError("End maturity must be greater than start maturity")
        if start == 0.0:
            return self.get_rate(end)
        
        r1 = self.get_rate(start)
        r2 = self.get_rate(end)

        fwd_rate = (r2 * end - r1 * start) / (end - start)
        return fwd_rate
    
    def get_discount_factor(self, maturity: float) -> float:
        """Computes the discount factor for a given maturity using the interpolated yield.

        Parameters:
            maturity (float): Desired maturity in years

        Returns:
            float: Discount factor
        """
        rate = self.get_rate(maturity)
        return np.exp(-rate * maturity)
    def get_fwd_discount_factor(self, start: float, end: float) -> float:
        """Computes the forward discount factor between two future dates.

        Parameters:
            start (float): Start maturity in years (e.g., 1.0 for 1 year)
            end (float): End maturity in years (must be > start)

        Returns:
            float: Forward discount factor between start and end
        """
        if not self.rate_curve:
            self._fetch_yield_curves()

        if end <= start:
            raise ValueError("End maturity must be greater than start maturity")
        if start == 0.0:
            return self.get_discount_factor(end)
        df_start = self.get_discount_factor(start)
        df_end = self.get_discount_factor(end)

        return df_end / df_start

    def get_volatility(self, strike: float, maturity: float) -> float:
        """Get the volatility interpolated by the volatility surface at this specific point (Strike * Maturity).
        Params:
            strike (float): option strike
            maturity (float): option maturity in year

        Returns:
            float: volatility at this point of the surface
        """
        if not self.volatility_surface:
            self._fetch_volatility_surface()
        return self.volatility_surface.get_volatility(strike, maturity)
    
    def bump_volatility(self, bump: float) -> "Market":
        """Create a new Market instance with bumped volatility.

        Args:
            bump: The absolute bump to apply to implied volatilities.

        Returns:
            A new Market instance with the bumped volatility surface.
        """
        bumped_market = copy.deepcopy(self)
        bumped_market._build_volatility_surface(bumped_market._raw_option_data, bump=bump)
        return bumped_market
    
    def bump_flat_yield_curve(self, bump: float) -> "Market":
        """Create a new Market instance with a bumped flat yield curve.

        Args:
            bump: The absolute bump to apply to the yield curve rates.

        Returns:
            A new Market instance with the bumped yield curve.
        """
        bumped_market = copy.deepcopy(self)
        bumped_market._build_yield_curves(bumped_market._raw_yield_data, bump=bump)
        return bumped_market

    def bump_flat_yield_curve_fast(self, bump: float) -> "Market":
        """Create a new Market instance with a bumped flat yield curve.
        Uses a shallow copy to reuse the already calibrated volatility surface.

        Args:
            bump: The absolute bump to apply to the yield curve rates.

        Returns:
            A new Market instance with the bumped yield curve.
        """
        if not hasattr(self, '_raw_yield_data'):
            return self.bump_flat_yield_curve(bump)
        bumped_market = copy.copy(self)
        bumped_market._build_yield_curves(bumped_market._raw_yield_data, bump=bump)
        return bumped_market
