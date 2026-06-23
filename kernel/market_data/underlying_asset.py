import pandas as pd
import os

class UnderlyingAsset:
    """
    Represents an underlying asset in the financial market.

    Attributes:
        name (str): Name of the underlying asset
        ticker (str): Ticker of the underlying asset
        isin (str): ISIN code of the underlying asset
        is_index (bool): Boolean indicating if the underlying asset is an index
        last_price (float): Last known price of the underlying asset
    """

    def __init__(self, name: str):
        """
        Initialize an underlying asset.

        Parameters:
            name (str): Name of the underlying asset
        """
        self.name = name

        self.ticker: str= None
        self.isin: str = None
        self.is_index: bool = None
        self.last_price: float = None

    def load_underlying_info(self, asset_info: pd.Series):
        """
        Load the underlying informations from the loaded file.
        """
        self.ticker = str(asset_info["Ticker"].iloc[0])
        self.isin = str(asset_info["ISIN"].iloc[0])
        self.is_index = bool(asset_info["Is Index"].iloc[0])
        self.last_price = float(asset_info["Last Price"].iloc[0])
