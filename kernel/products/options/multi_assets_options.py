from .abstract_option import AbstractOption
import numpy as np

class AbstractMultiAssetOption(AbstractOption):
    """Abstract class for options with multiple underlyings.
    """
    def __init__(self, maturity: float, strike: float, weights:np.ndarray=None) -> None:
        """Initializes a multi-underlying option.

        Args:
            maturity (float): Maturity of the option.
            strike (float): Strike price of the option.
            weights (np.ndarray, optional): Weight of the underlyings. By default, equal weights.
        """
        super().__init__(maturity, strike)
        self.weights = weights

    def weighted_average(self, paths: np.ndarray) -> float:
        """Calculates the weighted average of the underlyings.

        Args:
            paths (np.ndarray): Paths of underlying prices.

        Returns:
            float: Weighted average of underlyings.
        """
        if self.weights is None:
            self.weights = np.ones(paths.shape[0]) / paths.shape[0]  # Equal weights by default
        return np.dot(self.weights, paths)
    
class BasketCallOption(AbstractMultiAssetOption):
    """Class representing a basket option with fixed weights for a purchase option.
    """
    def payoff(self, paths: np.ndarray) -> float:
        """Calculates the basket option payoff.

        Args:
            paths (np.ndarray): Paths of underlying prices.

        Returns:
            float: The payoff of the basket option.
        """
        basket_price = self.weighted_average(paths[:, -1])  # Weighted average of final prices
        return max(0, basket_price - self.strike)  # Payoff for a call
    
class BasketPutOption(AbstractMultiAssetOption):
    """Class representing a basket option with fixed weights for a put option.
    """
    def payoff(self, paths: np.ndarray) -> float:
        """Calculates the basket option payoff for a put.

        Args:
            paths (np.ndarray): Paths of underlying prices.

        Returns:
            float: The payoff of the basket put option.
        """
        basket_price = self.weighted_average(paths[:, -1])  # Weighted average of final prices
        return max(0, self.strike - basket_price)  # Payoff for a put

class BestOfCallOption(AbstractMultiAssetOption):
    """Class representing a Best-Of Call option.
    The payoff is based on the best underlying on the final date.
    """
    def payoff(self, paths: np.ndarray) -> float:
        """Calculates the payoff for the Best-Of Call option.

        Args:
            paths (np.ndarray): Paths of underlying prices.

        Returns:
            float: The payoff of the Best-Of Call option.
        """
        best_performance = np.max(paths[:, -1])  # Best final price among underlyings
        return max(0, best_performance - self.strike)  # Payoff for a call


class BestOfPutOption(AbstractMultiAssetOption):
    """Class representing a Best-Of Put option.
    The payoff is based on the best underlying on the final date.
    """
    def payoff(self, paths: np.ndarray) -> float:
        """Calculates the payoff of the Best-Of Put option.

        Args:
            paths (np.ndarray): Paths of underlying prices.

        Returns:
            float: The payoff of the Best-Of Put option.
        """
        best_performance = np.max(paths[:, -1])  # Best final price among underlyings
        return max(0, self.strike - best_performance)  # Payoff for a put


class WorstOfCallOption(AbstractMultiAssetOption):
    """Class representing a Worst-Of Call option.
    The payoff is based on the underlying worst on the final date.
    """
    def payoff(self, paths: np.ndarray) -> float:
        """Calculates the payoff for the Worst-Of Call option.

        Args:
            paths (np.ndarray): Paths of underlying prices.

        Returns:
            float: The payoff of the Worst-Of Call option.
        """
        worst_performance = np.min(paths[:, -1])  # Worst final price among underlyings
        return max(0, worst_performance - self.strike)  # Payoff for a call


class WorstOfPutOption(AbstractMultiAssetOption):
    """Class representing a Worst-Of Put option.
    The payoff is based on the underlying worst on the final date.
    """
    def payoff(self, paths: np.ndarray) -> float:
        """Calculates the payoff of the Worst-Of Put option.

        Args:
            paths (np.ndarray): Paths of underlying prices.

        Returns:
            float: The payoff of the Worst-Of Put option.
        """
        worst_performance = np.min(paths[:, -1])  # Worst final price among underlyings
        return max(0, self.strike - worst_performance)  # Payoff for a put