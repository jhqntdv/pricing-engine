from abc import ABC, abstractmethod
import numpy as np 

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from kernel.models.pricing_engines.abstract_pricing_engine import AbstractPricingEngine

class AbstractDerivative(ABC):
    """Abstract class representing the different derivatives.
    """
    @abstractmethod
    def get_discounted_payoff(self, path: np.ndarray, market: 'Market') -> float:
        """Calculates the discounted payoff of the derivative based on the price path.

        Args:
            path (np.ndarray): Array representing the path of prices or underlying values.
            market (Market): The market object to retrieve discount factors.

        Returns:
            float: The discounted payoff of the derivative.
        """
        pass

    @abstractmethod
    def accept(self, engine: 'AbstractPricingEngine'):
        """Accepts a pricing engine to dispatch the correct calculation logic.
        """
        pass