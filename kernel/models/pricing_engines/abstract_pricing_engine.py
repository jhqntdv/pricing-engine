from abc import ABC, abstractmethod
from kernel.market_data.market import Market
from kernel.products.abstract_derivative import AbstractDerivative
from utils.pricing_results import PricingResults
from kernel.models.stochastic_processes import StochasticProcess

class AbstractPricingEngine(ABC):
    """Abstract base class for pricing engines.
    """
    def __init__(self, market: Market):
        self.market = market

    def get_results(self, derivative: AbstractDerivative) -> 'PricingResults':
        """Compute the pricing results of the financial product using double dispatch.
        """
        return derivative.accept(self)

    @abstractmethod
    def calculate_option(self, derivative: 'AbstractOption') -> 'PricingResults':
        """Compute the pricing results of an option.

        Args:
            derivative: The option to price.

        Returns:
            The pricing results (price and greeks).

        Raises:
            UnsupportedProductError: If the pricing engine does not support this type of option.
        """
        pass

    @abstractmethod
    def calculate_strategy(self, derivative: 'AbstractOptionStrategy') -> 'PricingResults':
        """Compute the pricing results of an option strategy.

        Args:
            derivative: The option strategy to price.

        Returns:
            The aggregated pricing results.

        Raises:
            UnsupportedProductError: If the pricing engine does not support option strategies.
        """
        pass

    @abstractmethod
    def calculate_rate_product(self, derivative: 'AbstractRateProduct') -> 'PricingResults':
        """Compute the pricing results of a rate product.

        Args:
            derivative: The rate product to price.

        Returns:
            The pricing results.

        Raises:
            UnsupportedProductError: If the pricing engine does not support this rate product.
        """
        pass

    @abstractmethod
    def calculate_structured_product(self, derivative: 'AbstractStructuredProduct') -> 'PricingResults':
        """Compute the pricing results of a structured product.

        Args:
            derivative: The structured product to price.

        Returns:
            The pricing results.

        Raises:
            UnsupportedProductError: If the pricing engine does not support structured products.
        """
        pass

    @abstractmethod
    def _get_price(self, derivative: AbstractDerivative,stochastic_process : StochasticProcess) -> float:
        """Abstract method to compute the price of the financial product.
        """
        pass
