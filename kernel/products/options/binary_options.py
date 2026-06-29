from .abstract_option import AbstractOption
import numpy as np


class AbstractBinaryOption(AbstractOption):
    """Abstract class representing the different binary (digital) options.
    """
    def __init__(self, maturity: float, strike: float, coupon: float) -> None:
        """Initializes a binary option with a maturity, a strike price and a fixed coupon.
        """
        super().__init__(maturity, strike)
        self.coupon = coupon


class BinaryCallOption(AbstractBinaryOption):
    """Class representing a binary (digital) call option.
    Pays a fixed coupon if the final spot is above the strike, else pays 0.
    """
    def get_discounted_payoff(self, paths: np.ndarray, market: 'Market') -> np.ndarray:
        """Vectorized payoff: returns the fixed coupon for all paths where final spot > strike.

        Args:
            paths (np.ndarray): Shape (nb_paths, nb_steps+1).
            market (Market): Market for discount factor.

        Returns:
            np.ndarray: Discounted payoffs of shape (nb_paths,).
        """
        payoffs = np.where(paths[:, -1] > self.strike, self.coupon, 0.0)
        return payoffs * market.get_discount_factor(self.maturity)


class BinaryPutOption(AbstractBinaryOption):
    """Class representing a binary (digital) put option.
    Pays a fixed coupon if the final spot is below the strike, else pays 0.
    """
    def get_discounted_payoff(self, paths: np.ndarray, market: 'Market') -> np.ndarray:
        """Vectorized payoff: returns the fixed coupon for all paths where final spot < strike.

        Args:
            paths (np.ndarray): Shape (nb_paths, nb_steps+1).
            market (Market): Market for discount factor.

        Returns:
            np.ndarray: Discounted payoffs of shape (nb_paths,).
        """
        payoffs = np.where(paths[:, -1] < self.strike, self.coupon, 0.0)
        return payoffs * market.get_discount_factor(self.maturity)


class AssetOrNothingCallOption(AbstractOption):
    """Class representing an asset-or-nothing call option.
    Pays the final spot price if the final spot is above the strike, else pays 0.
    """
    def get_discounted_payoff(self, paths: np.ndarray, market: 'Market') -> np.ndarray:
        """Calculate the discounted payoff for an asset-or-nothing call option.

        Args:
            paths: Array of simulated asset prices.
            market: The market data containing the discount curve.

        Returns:
            An array of discounted payoffs for each path.
        """
        payoffs = np.where(paths[:, -1] > self.strike, paths[:, -1], 0.0)
        return payoffs * market.get_discount_factor(self.maturity)


class AssetOrNothingPutOption(AbstractOption):
    """Class representing an asset-or-nothing put option.
    Pays the final spot price if the final spot is below the strike, else pays 0.
    """
    def get_discounted_payoff(self, paths: np.ndarray, market: 'Market') -> np.ndarray:
        """Calculate the discounted payoff for an asset-or-nothing put option.

        Args:
            paths: Array of simulated asset prices.
            market: The market data containing the discount curve.

        Returns:
            An array of discounted payoffs for each path.
        """
        payoffs = np.where(paths[:, -1] < self.strike, paths[:, -1], 0.0)
        return payoffs * market.get_discount_factor(self.maturity)
