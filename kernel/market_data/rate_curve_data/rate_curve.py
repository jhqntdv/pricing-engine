import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
import seaborn as sns

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

    def display_curve(self) -> None:
        """
        Plots the yield curve based on market data and the chosen interpolated yield.
        """
        maturities = np.linspace(0, 30, 500)
        yield_curve = np.array([self.get_rate(mat) for mat in maturities])

        sns.set(style="whitegrid")
        palette = sns.color_palette("coolwarm", 2)
        plt.figure(figsize=(10, 6))
        plt.scatter(self.data_curve['Maturity'], self.data_curve['Rate'], color=palette[0], label='Market yields', zorder=5)
        plt.plot(maturities, yield_curve, label='Interpolated yield curve', color=palette[1], linewidth=2)
        plt.xlabel('Maturity (Years)', fontsize=12)
        plt.ylabel('Yield (%)', fontsize=12)
        plt.title('Implied Yield Curve & Market Rate Points', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()