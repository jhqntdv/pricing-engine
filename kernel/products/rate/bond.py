import numpy as np
from kernel.products.rate.abstract_rate_product import AbstractRateProduct
from kernel.market_data import Market
from kernel.tools import CalendarConvention
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Tuple, List

class AbstractBond(AbstractRateProduct):
    """Base class for bond products.
    """
    def __init__(self, notional: float, issue_date: datetime, maturity: datetime,
                 calendar_convention: CalendarConvention, frequency: int = None,
                 price: float = None, ytm: float = None):
        """Initialize a base bond product.

        Args:
            notional: The face value of the bond.
            issue_date: The date the bond was issued.
            maturity: The date the bond matures.
            calendar_convention: The day count convention for interest.
            frequency: Coupon frequency per year. Defaults to None.
            price: Current market price. Defaults to None.
            ytm: Yield to maturity. Defaults to None.
        """
        super().__init__(notional, issue_date, maturity, calendar_convention, frequency)
        self.price = price
        self.ytm = ytm

    def payoff(self) -> float:
        """Calculate the payoff at maturity (notional amount).

        Returns:
            The bond's notional value.
        """
        return self.notional

    def calculate(self, valuation_date: datetime) -> Tuple[float, float]:
        """Pricing logic is handled by DiscountingPricingEngine.
        """
        self.date = valuation_date if valuation_date is not None else self.start
        return self.price, self.ytm


class CouponBond(AbstractBond):
    """Standard fixed-rate coupon bond.
    """
    def __init__(self, notional: float, issue_date: datetime, maturity: datetime,
                 coupon_rate: float, frequency: int,
                 calendar_convention: CalendarConvention, price: float = None, ytm: float = None):
        """Initialize a coupon bond.

        Args:
            notional: The face value of the bond.
            issue_date: The date the bond was issued.
            maturity: The date the bond matures.
            coupon_rate: The annual coupon rate.
            frequency: Coupon frequency per year.
            calendar_convention: The day count convention for interest.
            price: Current market price. Defaults to None.
            ytm: Yield to maturity. Defaults to None.
        """
        super().__init__(notional, issue_date, maturity, calendar_convention, frequency, price, ytm)
        self.coupon_rate = coupon_rate
        self.interval = 12 / self.frequency if self.frequency else 0
        self.coupons = self.generate_coupon_dates()
        
    def generate_coupon_dates(self) -> List[datetime]:
        """Generates coupon payment dates respecting the payment interval.
        Does not generate any coupons too close to issue or just before maturity.
        """
        dates = []
        if self.interval == 0:
            return [self.end]
            
        current = self.start + relativedelta(months=int(self.interval))
        while current <= self.end - relativedelta(months=int(self.interval)):
            dates.append(current)
            current += relativedelta(months=int(self.interval))

        dates.append(self.end)
        return dates


class ZeroCouponBond(AbstractBond):
    """Zero-coupon bond.
    """
    def __init__(self, notional: float, issue_date: datetime, maturity: datetime,
                 calendar_convention: CalendarConvention,
                 price: float = None, ytm: float = None):
        """Initialize a zero-coupon bond.

        Args:
            notional: The face value of the bond.
            issue_date: The date the bond was issued.
            maturity: The date the bond matures.
            calendar_convention: The day count convention for interest.
            price: Current market price. Defaults to None.
            ytm: Yield to maturity. Defaults to None.
        """
        super().__init__(notional, issue_date, maturity, calendar_convention, frequency=None, price=price, ytm=ytm)
        self.coupons = []
        if self.date == self.start and self.ytm is None:
            raise ValueError("On the issue date, you must provide the YTM.")

