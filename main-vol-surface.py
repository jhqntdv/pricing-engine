import os
import matplotlib.pyplot as plt

os.environ["MPLBACKEND"] = "Agg"

from kernel.market_data import Market, InterpolationType, VolatilitySurfaceType
from kernel.market_data.data_loader import MarketDataLoader
from kernel.tools import RateCurveType

def main():
    print("Initializing Market Data...")
    
    surfaces_to_plot = [
        VolatilitySurfaceType.SVI,
        VolatilitySurfaceType.SSVI,
        VolatilitySurfaceType.LOCAL
    ]

    
    # The output directory for the artifacts (now saving directly to your project folder)
    output_dir = r"c:\AppPy\structured-products-engine-only"

    # Load market data
    data_loader = MarketDataLoader()
    underlying_df = data_loader.get_underlying_info("SPX")
    options_df = data_loader.get_option_data("SPX")
    yield_df = data_loader.get_yield_curve(RateCurveType.RF_US_TREASURY.value)

    for vol_surface_type in surfaces_to_plot:
        print(f"Calibrating {vol_surface_type.name} Surface...")
        
        market = Market(
            underlying_name="SPX",
            yield_curve_data=yield_df,
            underlying_data=underlying_df,
            option_data=options_df,
            rate_curve_type=RateCurveType.RF_US_TREASURY,
            interpolation_type=InterpolationType.SVENSSON,
            volatility_surface_type=vol_surface_type
        )
        
        # We need to monkey-patch plt.show() to save the file instead of hanging the terminal
        original_show = plt.show
        def save_and_close():
            output_path = os.path.join(output_dir, f"{vol_surface_type.name}_Surface.png")
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Saved: {output_path}")
            plt.close()
        
        plt.show = save_and_close
        
        # Display Surface triggers the plot and our monkey-patched save
        market.volatility_surface.display_surface()
        
        # Restore plt.show just in case
        plt.show = original_show

if __name__ == "__main__":
    main()
