import os
import pandas as pd

class MarketDataLoader:
    """Data Access Layer to abstract all file I/O operations from the Market kernel.
    It natively loads and processes flat CSV formats.
    """
    def __init__(self, base_dir: str = None):
        """Initialize the MarketDataLoader.

        Args:
            base_dir: Base directory for data. Defaults to 'data' in project root.
        """
        if base_dir is None:
            # Default to the data directory in the project root
            self.base_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "data"
            )
        else:
            self.base_dir = base_dir

    def get_underlying_info(self, ticker: str) -> pd.DataFrame:
        """Loads the underlying data from a CSV and filters for the specified ticker.
        """
        file_path = os.path.join(self.base_dir, "underlying_data.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Underlying data file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        asset_info = df[df["Ticker"] == ticker]
        if asset_info.empty:
            raise ValueError(f"No data found for security name: {ticker}")
        
        return asset_info

    def get_yield_curve(self, rate_curve_type_value: str) -> pd.DataFrame:
        """Loads the yield curve data. Prioritizes .csv format.
        """
        base_name = os.path.splitext(rate_curve_type_value)[0]
        file_path = os.path.join(self.base_dir, "yield_curves", f"{base_name}.csv")
        
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
            
        raise FileNotFoundError(f"Yield curve file not found: {file_path}")

    def get_option_data(self, ticker: str) -> pd.DataFrame:
        """Loads flat option CSV data (e.g., options_AAPL.csv or options_SPX.csv).
        Handles both comma and semicolon separators, and standardizes column names.
        """
        file_path = os.path.join(self.base_dir, "option_data", f"options_{ticker}.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Option data file not found: {file_path}")
        
        # Try reading as comma-separated
        df = pd.read_csv(file_path)
        
        # If it seems like it didn't parse properly (only 1 giant column), try semicolon
        if len(df.columns) <= 1 or ('strike' not in df.columns and 'Strike' not in df.columns):
            df = pd.read_csv(file_path, sep=';')

        # Drop the empty leading column generated if the CSV starts with a semicolon (like AAPL)
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)
            
        # Standardize column names
        col_map = {
            'expiration': 'Maturity',
            'strike': 'Strike',
            'implied_volatility': 'Implied Volatility',
            'ImpliedVolatility': 'Implied Volatility'
        }
        df.rename(columns=col_map, inplace=True)

        required_columns = ["Maturity", "Implied Volatility", "Strike"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in option data: {', '.join(missing_columns)}")
        
        return df
