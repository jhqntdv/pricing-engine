from .abstract_pricing_engine import AbstractPricingEngine
from ..stochastic_processes import StochasticProcess
from kernel.products.options.abstract_option import AbstractOption
from kernel.market_data.market import Market
from kernel.products.options.american_options import AmericanAbstractOption, AmericanPutOption
from kernel.tools import ObservationFrequency
from utils.pricing_settings import PricingSettings
from utils.pricing_results import PricingResults
from kernel.models.stochastic_processes import BlackScholesProcess, HestonProcess
from kernel.models.discretization_schemes.euler_scheme import EulerScheme
from .mc_pricing_engine import MCPricingEngine
import numpy as np
import pandas as pd


class AmericanMCPricingEngine(MCPricingEngine):
    """A Monte Carlo pricing engine for classic financial derivatives (no barrier, no asian payoff ...)

    This class uses Monte Carlo simulation to compute the price of derivatives
    and can be extended to compute Greeks or other risk measures.
    """
    def __init__(self, market: Market, settings: PricingSettings) -> None: # type: ignore
        """Initialize the American Monte Carlo pricing engine.

        Args:
            market: The market data.
            settings: The pricing settings.
        """
        super().__init__(market, settings)



    def _get_price(self, derivative: AmericanAbstractOption, stochastic_process: StochasticProcess, current_market: Market = None, pre_simulated_paths: np.ndarray = None, return_std: bool = False):
        """Calculate the price of an American-style option using Longstaff-Schwartz.

        Args:
            derivative: The American option to evaluate.
            stochastic_process: The simulated stochastic process.
            current_market: Overridden market data for simulations.
            pre_simulated_paths: Optional paths to use directly.
            return_std: Whether to return the standard deviation.

        Returns:
            The option price (and standard deviation if requested).
        """
        if current_market is None:
            current_market = self.market
        
        scheme = EulerScheme()
        paths = scheme.simulate_paths(process=stochastic_process, nb_paths=self.nb_paths, seed=self.random_seed)


        dt = derivative.maturity / self.nb_steps

        exercise_indices = None
        if derivative.exercise_times is not None:
            exercise_indices = set(int(t_ex / dt) for t_ex in derivative.exercise_times)

        CF = derivative.intrinsic_payoff(paths[:, -1])
        for t in range(self.nb_steps - 2, -1, -1):
            df_forward = current_market.get_fwd_discount_factor(dt * (t + 1), dt * (t + 2))
            discounted_CF = CF * df_forward
            CF = discounted_CF.copy()
            
            if exercise_indices is None or t in exercise_indices: 
                immediate = derivative.intrinsic_payoff(paths[:, t])
                in_money = (immediate > 0)

                if np.any(in_money):
                    paths_in_money = paths[in_money, t]
                    x_matrix = np.column_stack([
                        np.ones(np.sum(in_money)),
                        paths_in_money,
                        paths_in_money ** 2
                    ])
                    y_vector = discounted_CF[in_money]
                    coeff, _, _, _ = np.linalg.lstsq(x_matrix, y_vector, rcond=None)
                    cont_val = coeff[0] + coeff[1] * paths_in_money + coeff[2] * (paths_in_money ** 2)
                    exercise = immediate[in_money] >= cont_val
                    CF[in_money] = np.where(exercise, immediate[in_money], discounted_CF[in_money])

        df_first = current_market.get_discount_factor(dt)
        discounted_cf = df_first * CF
        price = np.mean(discounted_cf)
        if return_std:
            std_dev = np.std(discounted_cf, ddof=1) / np.sqrt(len(discounted_cf))
            return price, std_dev
        return price
