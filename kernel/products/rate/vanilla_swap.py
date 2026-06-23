import numpy as np
from kernel.products.rate.abstract_rate_product import AbstractRateProduct
from kernel.market_data import Market
from kernel.tools import CalendarConvention
from datetime import datetime
from dateutil.relativedelta import relativedelta
from utils.day_counter import DayCounter
from typing import Tuple, List  

class InterestRateSwap(AbstractRateProduct):
    """
    Standard Interest Rate Swap (Vanilla Swap).
    """
    def __init__(self, notional: float, issue_date: datetime, maturity: datetime, calendar_convention: CalendarConvention = None,
                 fixed_rate: float = None, float_spread: float = 0.0, frequency: int = 1,
                 price: float = None):
        
        super().__init__(notional, issue_date, maturity, calendar_convention, frequency)
        self.float_spread = float_spread
        self.interval = 12 / self.frequency if self.frequency else 0
        self.day_counter = DayCounter(self.convention) if self.convention else None
        self.fixed_rate = fixed_rate
        self.price = price
        self.dates = self.generate_payment_dates()

    def generate_payment_dates(self) -> List[datetime]:
        """
        Generates payment dates based on the payment interval.
        """
        dates = []
        if self.interval == 0:
            return [self.end]
            
        current = self.start + relativedelta(months=int(self.interval))
        while current < self.end:
            dates.append(current)
            current += relativedelta(months=int(self.interval))
        dates.append(self.end)
        return dates

    def payoff(self) -> float:
        return 0.0

    def calculate(self, valuation_date: datetime) -> Tuple[float, float]:
        """
        Pricing logic is handled by DiscountingPricingEngine.
        """
        self.date = valuation_date if valuation_date is not None else self.start
        return self.price, self.fixed_rate
