from abc import abstractmethod
import numpy as np
from datetime import datetime
from kernel.tools import CalendarConvention
from kernel.products.abstract_derivative import AbstractDerivative
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from kernel.models.pricing_engines.abstract_pricing_engine import AbstractPricingEngine
    from kernel.market_data.market import Market

class AbstractRateProduct(AbstractDerivative):
    """
    Abstract base class for all rate products.
    """
    def __init__(self, notional: float, issue_date: datetime, maturity: datetime, calendar_convention: CalendarConvention = None, frequency: int = None):
        super().__init__()
        self.notional = notional
        self.start = issue_date
        self.end = maturity
        self.date = None
        
        # frequency is passed as an int (e.g. 1, 2, 4, 12)
        self.frequency = frequency
        self.convention = calendar_convention.value if calendar_convention else None

    def set_market(self, market: 'Market'):
        """
        Sets the market object containing the interest rate curves.
        Override in subclasses if needed.
        """
        self.market = market

    @abstractmethod
    def calculate(self, valuation_date: datetime) -> Tuple[float, float]:
        """
        Calculate the price and the market rate of the rate product.
        """
        pass

    @abstractmethod
    def payoff(self) -> float:
        """
        Calculate the terminal payoff of the rate product.
        """
        pass

    def get_discounted_payoff(self, paths: np.ndarray, market: 'Market') -> np.ndarray:
        """
        Rate products use deterministic discounting via the DiscountingPricingEngine.
        This method is required by AbstractDerivative but should not be called for rate products.
        """
        raise NotImplementedError("Rate products do not use Monte Carlo simulation.")

    def accept(self, engine: 'AbstractPricingEngine'):
        """
        Visitor pattern accept method to delegate calculation to the pricing engine.
        """
        return engine.calculate_rate_product(self)