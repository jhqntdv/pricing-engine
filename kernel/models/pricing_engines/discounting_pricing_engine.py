import numpy as np
from scipy.optimize import brentq
import calendar
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Tuple

from kernel.exceptions import (
    UnsupportedProductError,
    InvalidProductInputError,
    IndeterminateValuationError
)

from kernel.models.pricing_engines.abstract_pricing_engine import AbstractPricingEngine
from kernel.market_data.market import Market
from utils.pricing_settings import PricingSettings
from utils.pricing_results import PricingResults
from kernel.products.rate.abstract_rate_product import AbstractRateProduct
from kernel.products.rate.bond import AbstractBond, CouponBond, ZeroCouponBond
from kernel.products.rate.vanilla_swap import InterestRateSwap
from kernel.models.stochastic_processes import StochasticProcess
from kernel.tools import CalendarConvention

class DiscountingPricingEngine(AbstractPricingEngine):
    """Pricing engine for interest rate products.
    """

    def __init__(self, market: Market, settings: PricingSettings):
        """Initialize the discounting pricing engine.

        Args:
            market: The market data containing the rate curves.
            settings: The pricing settings.
        """
        super().__init__(market)
        self.settings = settings
        self.valuation_date = settings.valuation_date

    def calculate_rate_product(self, derivative: AbstractRateProduct) -> PricingResults:
        """Calculate the price and yield/rate of a rate product.

        Args:
            derivative: The rate product to price.

        Returns:
            The pricing results containing the price and rate.

        Raises:
            UnsupportedProductError: If the rate product sub-type is unknown.
        """
        derivative.set_market(self.market)
        derivative.date = self.valuation_date if self.valuation_date is not None else derivative.start

        if isinstance(derivative, AbstractBond):
            price, rate = self._price_bond(derivative)
        elif isinstance(derivative, InterestRateSwap):
            price, rate = self._price_swap(derivative)
        else:
            raise UnsupportedProductError("DiscountingPricingEngine does not support unknown rate product sub-type.")

        derivative.price = price
        if hasattr(derivative, 'ytm'):
            derivative.ytm = rate
        elif hasattr(derivative, 'fixed_rate'):
            derivative.fixed_rate = rate
            
        return PricingResults(price=price, rate=rate)

    def _price_bond(self, bond: AbstractBond) -> Tuple[float, float]:
        """Price a bond and compute its yield to maturity (YTM).

        Args:
            bond: The bond to price.

        Returns:
            A tuple of (price, ytm).

        Raises:
            InvalidProductInputError: If neither price nor YTM is provided.
        """
        if isinstance(bond, ZeroCouponBond):
            if bond.price is None and bond.ytm is None:
                 raise InvalidProductInputError("You must provide either ytm or the price")
            
            if bond.price is None:
                t = (bond.end - bond.date).days / 365
                bond.price = bond.notional / ((1 + bond.ytm) ** t) if t > 0 else bond.notional
            elif bond.ytm is None:
                t = (bond.end - bond.date).days / 365
                if t > 0:
                    bond.ytm = (bond.notional / bond.price) ** (1/t) - 1
                else:
                    bond.ytm = 0.0

            return bond.price, bond.ytm
            
        elif isinstance(bond, CouponBond):
            if bond.price is None and bond.ytm is None:
                 raise InvalidProductInputError("You must provide either ytm or the price")
                 
            def _accrued_interest() -> float:
                past_coupons = [c for c in bond.coupons if c <= bond.date]
                if past_coupons:
                    last_coupon = max(past_coupons) + relativedelta(days=1)
                else:
                    last_coupon = bond.start
                    
                next_coupon = min((c for c in bond.coupons if c > bond.date), default=bond.end)
                days_since = (bond.date - last_coupon).days
                total_days = (next_coupon - last_coupon).days

                if bond.convention == CalendarConvention.ACT_360.value:
                    total_days = 360
                elif bond.convention == CalendarConvention.ACT_365.value:
                    total_days = 365
                elif bond.convention == CalendarConvention.THIRTY_360.value:
                    d1, d2 = min(last_coupon.day, 30), min(bond.date.day, 30)
                    days_since = 360 * (bond.date.year - last_coupon.year) + 30 * (bond.date.month - last_coupon.month) + (d2 - d1)
                    d2 = min(next_coupon.day, 30)
                    total_days = 360 * (next_coupon.year - last_coupon.year) + 30 * (next_coupon.month - last_coupon.month) + (d2 - d1)
                elif bond.convention == CalendarConvention.ACT_ACT.value:
                    total_days = 366 if calendar.isleap(last_coupon.year) else 365

                return days_since / total_days if total_days else 0

            def _present_value(ytm: float) -> float:
                future_coupons = [c for c in bond.coupons if c > bond.date]
                accrued = _accrued_interest()
                times = np.array([(1 - accrued) + i / bond.frequency for i in range(len(future_coupons))])
                pv_coupons = sum((bond.coupon_rate * bond.notional) / (1 + ytm) ** t for t in times)
                pv_principal = bond.notional / (1 + ytm) ** times[-1] if times.size > 0 else 0
                return pv_coupons + pv_principal

            if bond.price is None:
                bond.price = _present_value(bond.ytm)
            elif bond.ytm is None:
                func = lambda ytm: _present_value(ytm) - bond.price
                bond.ytm = brentq(func, -0.5, 1.0)
                
            return bond.price, bond.ytm


    def _price_swap(self, swap: InterestRateSwap) -> Tuple[float, float]:
        """Price an interest rate swap and compute its par fixed rate.

        Args:
            swap: The swap to price.

        Returns:
            A tuple of (price, fixed_rate).

        Raises:
            IndeterminateValuationError: If the annuity is zero.
        """
        def _get_annuities() -> float:
            annuities = 0
            prev_date = swap.start
            for d in swap.dates:
                if d > swap.date:
                    yf = swap.day_counter.get_year_fraction(prev_date, d)
                    t = (d - swap.date).days / 365
                    df = self.market.get_discount_factor(t)
                    annuities += yf * df
                prev_date = d
            return annuities
            
        def _float_leg_value() -> float:
            pv = 0
            prev_date = swap.start
            for d in swap.dates:
                if d > swap.date:
                    t1 = (prev_date - swap.date).days / 365
                    t2 = (d - swap.date).days / 365
                    yf = swap.day_counter.get_year_fraction(prev_date, d)

                    if t1 <= 0:
                        forward_rate = self.market.get_rate(t2)
                    else:
                        forward_rate = self.market.get_fwd_rate(t1, t2)

                    forward_rate += swap.float_spread / 10000.0
                    df = self.market.get_discount_factor(t2)
                    cashflow = swap.notional * forward_rate * yf
                    pv += cashflow * df
                prev_date = d
            return pv

        def _fixed_leg_value(fixed_rate) -> float:
            return fixed_rate * swap.notional * _get_annuities()

        if swap.fixed_rate is None:
            annuities = _get_annuities()
            if annuities == 0:
                raise IndeterminateValuationError("Annuity is zero; par rate indeterminate")
            swap.fixed_rate = _float_leg_value() / (swap.notional * annuities)

        if swap.price is None:
            if swap.date == swap.start:
                swap.price = 0.0
            else:
                fixed_pv = _fixed_leg_value(swap.fixed_rate)
                float_pv = _float_leg_value()
                swap.price = float_pv - fixed_pv
                
        return swap.price, swap.fixed_rate

    def calculate_option(self, derivative: 'AbstractOption') -> PricingResults:
        """Calculate the price of an option (Unsupported).

        Args:
            derivative: The option.

        Raises:
            UnsupportedProductError: Engine does not support options.
        """
        raise UnsupportedProductError("DiscountingPricingEngine does not support options.")

    def calculate_strategy(self, derivative: 'AbstractOptionStrategy') -> PricingResults:
        """Calculate the price of an option strategy (Unsupported).

        Args:
            derivative: The strategy.

        Raises:
            UnsupportedProductError: Engine does not support strategies.
        """
        raise UnsupportedProductError("DiscountingPricingEngine does not support strategies.")

    def calculate_structured_product(self, derivative: 'AbstractStructuredProduct') -> PricingResults:
        """Calculate the price of a structured product (Unsupported).

        Args:
            derivative: The structured product.

        Raises:
            UnsupportedProductError: Engine does not support structured products.
        """
        raise UnsupportedProductError("DiscountingPricingEngine does not support structured products.")

    def _get_price(self, derivative: AbstractRateProduct, process : StochasticProcess) -> float:
        """Get the price (abstract engine hook).

        Args:
            derivative: The rate product.
            process: The stochastic process.

        Returns:
            The product price.
        """
        return derivative.price
