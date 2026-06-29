from kernel.products.options.abstract_option import AbstractOption
import numpy as np
from abc import ABC, abstractmethod
from typing import Union, List

class AmericanAbstractOption(AbstractOption):
    """Abstract base class for American-style options."""

    def __init__(self, strike: float, maturity: float) -> None:
        """Initialize the American option.

        Args:
            strike: The strike price of the option.
            maturity: The maturity of the option in years.
        """
        super().__init__(strike=strike, maturity=maturity)
        self.exercise_times = None

    def payoff(self, path: np.ndarray) -> float:
        """Calculate the payoff of the option at maturity.

        Args:
            path: Array of simulated asset prices.

        Returns:
            The option payoff.
        """
        pass

    @abstractmethod
    def intrinsic_payoff(self, S: np.ndarray)-> np.ndarray:
        """Calculate the intrinsic payoff for given asset prices.

        Args:
            S: Array of asset prices.

        Returns:
            An array of intrinsic payoffs.
        """
        pass
    
class AmericanCallOption(AmericanAbstractOption):
    """Class representing an American Call option."""

    def __init__(self, strike:float, maturity:float) -> None:
        """Initialize the American Call option.

        Args:
            strike: The strike price.
            maturity: The maturity in years.
        """
        super().__init__(strike=strike, maturity=maturity)

    def intrinsic_payoff(self, S):
        """Calculate the intrinsic payoff for an American Call.

        Args:
            S: Array of asset prices.

        Returns:
            Array of max(S - strike, 0).
        """
        return np.maximum(S - self.strike, 0)
    
class AmericanPutOption(AmericanAbstractOption):
    """Class representing an American Put option."""

    def __init__(self, strike:float, maturity:float) -> None:
        """Initialize the American Put option.

        Args:
            strike: The strike price.
            maturity: The maturity in years.
        """
        super().__init__(strike=strike, maturity=maturity)
    
    def intrinsic_payoff(self, S:np.ndarray):
        """Calculate the intrinsic payoff for an American Put.

        Args:
            S: Array of asset prices.

        Returns:
            Array of max(strike - S, 0).
        """
        return np.maximum(self.strike - S, 0)

class BermudanCallOption(AmericanCallOption):
    """Class representing a Bermudan Call option."""

    def __init__(self, strike, maturity, exercise_times) -> None:
        """Initialize the Bermudan Call option.

        Args:
            strike: The strike price.
            maturity: The maturity in years.
            exercise_times: List of allowed exercise times.
        """
        super().__init__(strike=strike, maturity=maturity)
        self.exercise_times = exercise_times
    
    def intrinsic_payoff(self, S):
        """Calculate the intrinsic payoff for a Bermudan Call.

        Args:
            S: Array of asset prices.

        Returns:
            Array of max(S - strike, 0).
        """
        return np.maximum(S - self.strike, 0)
    

class BermudanPutOption(AmericanPutOption):
    """Class representing a Bermudan Put option."""
    def __init__(self, strike:float, maturity:float, exercise_times:List[float]) -> None:
        """Initialize the Bermudan Put option.

        Args:
            strike: The strike price.
            maturity: The maturity in years.
            exercise_times: List of allowed exercise times.
        """
        super().__init__(strike=strike, maturity=maturity)
        self.exercise_times = exercise_times
    
    def intrinsic_payoff(self, S:np.ndarray):
        """Calculate the intrinsic payoff for a Bermudan Put.

        Args:
            S: Array of asset prices.

        Returns:
            Array of max(strike - S, 0).
        """
        return np.maximum(self.strike - S, 0)