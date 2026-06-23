from typing import List, Tuple
import numpy as np
from ..abstract_derivative import AbstractDerivative
from ..options.abstract_option import AbstractOption

class AbstractOptionStrategy(AbstractDerivative):
    """
    Class representing an optional strategy composed of several options.
    """
    def __init__(self, options: List[Tuple[AbstractOption, bool]] = None):
        """
        Initializes an optional strategy.

        Args:
            options (List[Tuple[AbstractOption, bool]]): List of tuples containing an option and a boolean
                indicating whether the option is bought (True) or sold (False).
        """
        self.options: List[Tuple[AbstractOption, bool]] = options if options is not None else []

    def add_option(self, option: AbstractOption, is_long: bool):
        """
        Adds an option to the policy.

        Args:
            option (AbstractOption): An option instance to add.
            is_long (bool): Indicates whether the option is bought (True) or sold (False).
        """
        self.options.append((option, is_long))

    def get_discounted_payoff(self, path: np.ndarray, market: 'Market') -> np.ndarray:
        """
        Calculates the total discounted payoff of the strategy for a given path.

        Args:
            path (np.ndarray): Path of underlying prices or values.
            market (Market): The market object to calculate discount factors.

        Returns:
            np.ndarray: The total discounted payoff of the strategy for each point in the path.
        """
        total_payoff = 0
        for option, is_long in self.options:
            payoff = option.get_discounted_payoff(path, market)
            total_payoff += payoff if is_long else -payoff
        return total_payoff

    def accept(self, engine: 'AbstractPricingEngine'):
        return engine.calculate_strategy(self)

    def __str__(self) -> str:
        """
        Textual representation of the strategy.

        Returns:
            str: Description of the strategy and the options it contains.
        """
        return f"AbstractOptionStrategy avec {len(self.options)} options: " + \
               f"{[(str(option), is_long) for option, is_long in self.options]}"
