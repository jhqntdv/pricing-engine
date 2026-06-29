from .abstract_option import AbstractOption
import numpy as np

class EuropeanCallOption(AbstractOption):
    """Class representing a European call option.
    """
    def get_discounted_payoff(self, paths: np.ndarray, market: 'Market') -> np.ndarray:
        """Calculate the discounted payoff for a European call option.

        Args:
            paths: Array of simulated asset prices.
            market: The market data containing the discount curve.

        Returns:
            An array of discounted payoffs for each path.
        """
        # paths[:, -1] is the final price for all paths; np.maximum vectorizes over all paths
        intrinsic = np.maximum(0.0, paths[:, -1] - self.strike)
        return intrinsic * market.get_discount_factor(self.maturity)

class EuropeanPutOption(AbstractOption):
    """Class representing a European put option.
    """
    def get_discounted_payoff(self, paths: np.ndarray, market: 'Market') -> np.ndarray:
        """Calculate the discounted payoff for a European put option.

        Args:
            paths: Array of simulated asset prices.
            market: The market data containing the discount curve.

        Returns:
            An array of discounted payoffs for each path.
        """
        intrinsic = np.maximum(0.0, self.strike - paths[:, -1])
        return intrinsic * market.get_discount_factor(self.maturity)
