from abc import ABC, abstractmethod
import numpy as np

from ..abstract_derivative import AbstractDerivative

class AbstractStructuredProduct(AbstractDerivative):
    """Abstract class representing a structured product.
    """
    def __init__(self, maturity : float, initial_spot: float = None):
        """Initialize the abstract structured product.

        Args:
            maturity: The maturity of the product in years.
            initial_spot: The initial spot price. Defaults to None.
        """
        self.validate_inputs(maturity)
        self.maturity = maturity
        self.initial_spot = initial_spot

    def validate_inputs(self, maturity : float):
        """Validates entries for maturity and nominal.

        Args:
            maturity (float): The maturity of the structured product.
            notional (float): The nominal value of the structured product.

        Reasons:
            ValueError: If maturity or nominal are not positive.
        """
        if maturity <= 0:
            raise ValueError("Maturity must be positive.")

    @abstractmethod
    def get_discounted_payoff(self, paths: np.ndarray, market: 'Market') -> float:
        """Calculates the discounted payoff of the structured product.

        Args:
            paths (np.ndarray): Paths of underlying prices.
            market (Market): The market object to calculate discount factors.

        Returns:
            float: The discounted payoff of the structured product.
        """
        pass

    @abstractmethod
    def description(self) -> str:
        """Returns a textual description of the structured product.

        Returns:
            str: Structured product description.
        """
        pass

    def accept(self, engine: 'AbstractPricingEngine'):
        """Accept a pricing engine using the visitor pattern.

        Args:
            engine: The pricing engine.

        Returns:
            Pricing results from the engine.
        """
        return engine.calculate_structured_product(self)