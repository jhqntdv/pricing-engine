import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from kernel.market_data import RateCurve

class AbstractVolatilitySurface(ABC):
    """
    Abstract base class for all volatility surfaces.
    It encapsulates shared attributes, calibration status, and visualization methods
    to ensure DRY (Don't Repeat Yourself) principles across all models.
    """

    def __init__(self, option_data: pd.DataFrame, rate_curve: RateCurve):
        """
        Initializes the base surface with market data and rate curve.
        """
        self.option_data = option_data
        self.rate_curve = rate_curve
        self.spot = option_data["Spot"].values[0]
        self.is_calibrated = False

    @abstractmethod
    def calibrate_surface(self) -> None:
        """Calibrates the model parameters to the market data."""
        pass

    @abstractmethod
    def get_volatility(self, strike: float, maturity: float) -> float:
        """Returns the model-implied volatility for a given strike and maturity."""
        pass

    def display_smiles(self, model_name: str = "Model", reference_surface=None, ref_name: str = "Reference") -> None:
        """
        Displays the volatility smiles for all maturities in the option data.
        
        Parameters:
            model_name (str): The name of the model to display on the y-axis.
            reference_surface (AbstractVolatilitySurface, optional): Another surface to plot alongside for comparison.
            ref_name (str): Label for the reference surface.
        """
        if not self.is_calibrated:
            raise Exception("Surface is not calibrated yet. Please call calibrate_surface() first.")

        unique_maturities = np.sort(self.option_data["Maturity"].unique())
        num_maturities = len(unique_maturities)

        cols = 4
        rows = (num_maturities + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
        
        # Handle case where subplots returns a single Axes object
        if num_maturities == 1:
            axes = np.array([axes])
        else:
            axes = axes.flatten()

        strikes_range = np.linspace(self.spot / 2, self.spot * 2, 500)

        for i, T in enumerate(unique_maturities):
            ax = axes[i]
            
            # Compute model volatilities (handling potential domain errors gracefully)
            model_vols = []
            for K in strikes_range:
                try:
                    vol = self.get_volatility(K, T) * 100
                except Exception:
                    vol = np.nan
                model_vols.append(vol)

            # Plot Model Smile
            moneyness = (strikes_range / self.spot) * 100
            ax.plot(moneyness, model_vols, label=f'{model_name} Smile', color='orange')

            # Plot Reference Smile if provided (e.g., used by Local Vol to overlay SVI)
            if reference_surface:
                ref_vols = []
                for K in strikes_range:
                    try:
                        vol = reference_surface.get_volatility(K, T) * 100
                    except Exception:
                        vol = np.nan
                    ref_vols.append(vol)
                ax.plot(moneyness, ref_vols, label=ref_name, color='green', linestyle='--')

            # Plot Market Data
            market_data = self.option_data[self.option_data["Maturity"] == T]
            ax.scatter((market_data['Strike'] / self.spot) * 100, market_data['Implied Volatility'], 
                       color='blue' if not reference_surface else 'red', label='Market Data', s=20)

            # Formatting
            ax.set_title(f"Maturity: {int(T * 252)} days", fontsize=12, fontweight='bold')
            ax.set_xlabel('Moneyness (% ATM)', fontsize=10)
            ax.set_ylabel(f'{model_name} Implied Volatility (%)', fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, linestyle='--', alpha=0.7)

        # Hide unused subplots
        for j in range(len(unique_maturities), len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    def display_surface(self, model_name: str = "Model Implied") -> None:
        """
        Displays the volatility surface in 3D.
        """
        if not self.is_calibrated:
            raise Exception("Surface is not calibrated yet. Please call calibrate_surface() first.")

        strikes = np.linspace(self.spot / 2, self.spot * 2, 50)
        maturities = np.linspace(self.option_data["Maturity"].min(), self.option_data["Maturity"].max(), 50)

        # Build surface grid
        vol_surface = np.zeros((len(strikes), len(maturities)))
        for i, K in enumerate(strikes):
            for j, T in enumerate(maturities):
                try:
                    vol_surface[i, j] = self.get_volatility(K, T) * 100
                except Exception:
                    vol_surface[i, j] = np.nan

        X, Y = np.meshgrid(maturities * 252, (strikes / self.spot) * 100)
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, vol_surface, cmap='viridis', edgecolor='k', alpha=0.8)

        # Scatter market options
        market_strikes = (self.option_data["Strike"] / self.spot) * 100
        market_maturities = self.option_data["Maturity"] * 252
        market_vols = self.option_data["Implied Volatility"]
        ax.scatter(market_maturities, market_strikes, market_vols, color='red', label='Market Options', s=20)

        # Formatting
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label('Volatility (%)', fontsize=12)
        ax.set_xlabel('Maturity (days)', fontsize=12, labelpad=10)
        ax.set_ylabel('Moneyness (% ATM)', fontsize=12, labelpad=10)
        ax.set_zlabel('Volatility (%)', fontsize=12, labelpad=10)
        ax.set_title(f'{model_name} Volatility Surface', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()