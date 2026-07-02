from .mc_pricing_engine import MCPricingEngine
from kernel.tools import ObservationFrequency
from kernel.products.structured_products import AbstractAutocall
from kernel.market_data import Market
from kernel.models.stochastic_processes import StochasticProcess
from utils.pricing_settings import PricingSettings
from utils.pricing_results import PricingResults
from kernel.models.discretization_schemes.euler_scheme import EulerScheme

import numpy as np


class CallableMCPricingEngine(MCPricingEngine):
    """A Monte Carlo pricing engine for callable financial derivatives.

    This class uses Monte Carlo simulation to compute the price of derivatives
    and can be extended to compute Greeks or other risk measures.
    """

    def __init__(self, market: Market, settings: PricingSettings) -> None: # type: ignore
        """Initialize the Callable MC pricing engine.

        Args:
            market: The market data containing underlying and rates.
            settings: Configuration settings for the simulation.
        """
        super().__init__(market, settings)
        self.obs_frequency = settings.obs_frequency
        self.compute_coupon = settings.compute_callable_coupons

    def calculate_structured_product(self, derivative: AbstractAutocall) -> PricingResults: 
        """Calculate the price or coupon for a structured callable product.

        Args:
            derivative: The autocallable structured product.

        Returns:
            The pricing results containing the computed coupon or price.
        """
        if self.compute_coupon:
            result = PricingResults()
            if hasattr(derivative, "initial_spot") and getattr(derivative, "initial_spot", None) is None:
                derivative.initial_spot = self.market.underlying_asset.last_price
            process = self.get_stochastic_process(derivative=derivative, market=self.market)
            
            coupon = self.get_coupon(derivative, process) #see for parameterization of the target price
            result.coupon_callable = coupon
            return result
        else:
            # Fallback to the base optimized MC engine pricing
            return super().get_result(derivative)

    def get_coupon(self, derivative: 'CallableProduct', process: StochasticProcess, epsilon: float = 1e-2, max_iter: int = 25, target_price: float = 100, method: str = "analytical") -> float: # type: ignore
        """Computes the coupon of the structured product such that the price equals the target price (e.g., initial capital).

        Parameters:
            derivative (CallableProduct): The derivative for which to compute the coupon.
            process (StochasticProcess): The stochastic process simulating the underlying asset.
            epsilon (float): The tolerance for the price difference.
            max_iter (int): The maximum number of iterations for the dichotomy method.
            target_price (float): The target price.
            method (str): "analytical" (fast linear solver) or "bisection" (iterative).

        Returns:
            float: The computed coupon.
        """
        # Pre-simulate paths once to avoid re-simulating inside the root-finding loop.
        # This drastically improves performance and eliminates Monte Carlo noise between iterations.
        scheme = EulerScheme()
        sim_result = scheme.simulate_paths(process, self.nb_paths, self.random_seed)
        pre_simulated_paths = sim_result.spot_paths

        if method == "analytical":
            # 1. Price with coupon = 0
            derivative.coupon_rate = 0.0
            p0 = self._get_price(derivative, process, current_market=self.market, pre_simulated_paths=pre_simulated_paths)
            
            # 2. Price with coupon = 1.0 (1%)
            derivative.coupon_rate = 1.0
            p1 = self._get_price(derivative, process, current_market=self.market, pre_simulated_paths=pre_simulated_paths)
            
            # 3. Compute delta price per 1% coupon (the slope)
            dp_dc = p1 - p0
            
            # 4. Solve for target coupon
            # p0 + c * dp_dc = target_price => c = (target_price - p0) / dp_dc
            if abs(dp_dc) < 1e-12:
                return 0.0  # Avoid division by zero if coupon has no effect
            
            return float((target_price - p0) / dp_dc)

        # Fallback: Bisection method
        # Define the bounds for the coupon (%)
        lower_bound = 0.0
        upper_bound = 50.0

        for _ in range(max_iter):
            mid_coupon = (lower_bound + upper_bound) / 2.0

            # Set the coupon in the derivative
            derivative.coupon_rate = mid_coupon

            # Compute the price for the current coupon using pre-simulated paths
            price = self._get_price(derivative, process, current_market=self.market, pre_simulated_paths=pre_simulated_paths)

            # Check if the price is close enough to the target price
            if abs(price - target_price) < epsilon:
                return mid_coupon
            
            if price < target_price:
                lower_bound = mid_coupon
            else:
                upper_bound = mid_coupon

        return mid_coupon