import re
import numpy as np
import pandas as pd
from typing import Tuple

class RateCurve:
    """
    Defines a yield curve model based on market data and an interpolation method.

    This class processes market rate data, interpolates missing values, and provides methods to compute yields
    and discount factors. The interpolation can be based on different models such as Svensson, Nelson-Siegel...
    """

    def __init__(self, data_curve: pd.DataFrame, interpolation_type: 'InterpolationType'):  # type: ignore
        """
        Initializes the rate curve with market data and an interpolation method.

        Parameters:
            data_curve (pd.DataFrame): Market yield data with 'Maturity'and 'Rate' columns.
            interpolation_type (InterpolationType): The interpolation method to use for yield curve fitting.
        """
        self.data_curve = data_curve
        maturities, rates = np.array(self.data_curve["Maturity"]), np.array(self.data_curve["Rate"])

        self.interpolator = interpolation_type.value(maturities, rates)

    def calibrate(self) -> None:
        """
        Calibrates the interpolator to the market data.
        """
        self.interpolator.calibrate()

    def get_rate(self, maturity: float) -> float:
        """
        Retrieves the interpolated yield rate for a given maturity.

        Parameters:
            maturity (float): Desired maturity in years.

        Returns:
            float: Interpolated yield rate.
        """
        return self.interpolator.interpolate(maturity)
