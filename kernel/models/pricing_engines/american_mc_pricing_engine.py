from .abstract_pricing_engine import AbstractPricingEngine
from ..stochastic_processes import StochasticProcess
from kernel.products.options.abstract_option import AbstractOption
from kernel.market_data.market import Market
from kernel.products.options.american_options import AmericanAbstractOption
from utils.pricing_settings import PricingSettings
from utils.pricing_results import PricingResults
from kernel.models.discretization_schemes.euler_scheme import EulerScheme
from .mc_pricing_engine import MCPricingEngine
import numpy as np


class AmericanMCPricingEngine(MCPricingEngine):
    """A Monte Carlo pricing engine for American and Bermudan options using the Longstaff-Schwartz algorithm.

    This engine uses least-squares Monte Carlo (LSM) backward induction to determine the optimal early
    exercise strategy and compute the price of derivatives with early exercise features.

    # TBD (Future Enhancements):
    # 1. Dividend Support: The engine currently relies on the base stochastic process which does not natively 
    #    support continuous dividend yields (q) or discrete cash dividends. This is particularly important for 
    #    pricing American Call options, which are never optimally exercised early without a dividend yield.
    # 2. American Barrier Options: The backward induction loop does not currently check for barrier knock-outs. 
    #    Support needs to be added to zero out the continuation/immediate values for paths that breach a barrier.
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
            pre_simulated_paths: Optional paths to use directly. Accepts a raw np.ndarray or a SimulationResult.
            return_std: Whether to return the standard deviation.

        Returns:
            The option price (and standard deviation if requested).
        """
        if current_market is None:
            raise ValueError("current_market must be explicitly provided to ensure drift and discounting share the same curve.")
        
        if pre_simulated_paths is not None:
            paths = getattr(pre_simulated_paths, "spot_paths", pre_simulated_paths)
            var_paths = getattr(pre_simulated_paths, "variance_paths", None)
        else:
            scheme = EulerScheme()
            sim_result = scheme.simulate_paths(process=stochastic_process, nb_paths=self.nb_paths, seed=self.random_seed)
            paths = sim_result.spot_paths
            var_paths = sim_result.variance_paths


        dt = derivative.maturity / self.nb_steps

        exercise_indices = None
        if derivative.exercise_times is not None:
            exercise_indices = set(int(round(t_ex / dt)) for t_ex in derivative.exercise_times)

        # L5: Normalize regression basis by strike (moneyness) for numerical stability.
        # Using x = S/K keeps all polynomial terms near unit scale, improving
        # the condition number of the design matrix for np.linalg.lstsq.
        normalizer = derivative.strike if hasattr(derivative, 'strike') else 1.0

        cashflow = derivative.intrinsic_payoff(paths[:, -1])
        for t in range(self.nb_steps - 1, 0, -1):
            df_step = current_market.get_fwd_discount_factor(dt * t, dt * (t + 1))
            cashflow = cashflow * df_step
            
            if exercise_indices is None or t in exercise_indices: 
                immediate = derivative.intrinsic_payoff(paths[:, t])
                in_money = (immediate > 0)

                if np.any(in_money):
                    paths_in_money = paths[in_money, t]
                    normalized = paths_in_money / normalizer
                    basis = [np.ones(np.sum(in_money)), normalized, normalized ** 2]
                    
                    if var_paths is not None:
                        v_ref = getattr(stochastic_process, "theta", None) or getattr(stochastic_process, "v0", 1.0)
                        v_in = var_paths[in_money, t] / v_ref
                        basis += [v_in, v_in ** 2, normalized * v_in]

                    x_matrix = np.column_stack(basis)
                    y_vector = cashflow[in_money]
                    coeff, _, _, _ = np.linalg.lstsq(x_matrix, y_vector, rcond=None)
                    cont_val = x_matrix @ coeff
                    exercise = immediate[in_money] >= cont_val
                    cashflow[in_money] = np.where(exercise, immediate[in_money], cashflow[in_money])

        df_first = current_market.get_discount_factor(dt)
        discounted_cashflow = df_first * cashflow
        price = np.mean(discounted_cashflow)
        
        # L1 Edge case: Support exercise at valuation date (t=0)
        # If the option is already deeply ITM, immediate exercise might be better than the holding value.
        initial_spot = np.array([paths[0, 0]])
        immediate_t0 = derivative.intrinsic_payoff(initial_spot)[0]
        if immediate_t0 > price:
            price = immediate_t0
            
        if return_std:
            std_dev = np.std(discounted_cashflow, ddof=1) / np.sqrt(len(discounted_cashflow))
            return price, std_dev
        return price
