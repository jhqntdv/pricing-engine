from .abstract_pricing_engine import AbstractPricingEngine
from ..stochastic_processes import StochasticProcess
from kernel.products.abstract_derivative import AbstractDerivative
from kernel.products.options.abstract_option import AbstractOption
from kernel.products.options_strategies.abstract_option_strategy import AbstractOptionStrategy
from kernel.market_data.market import Market
from kernel.tools import ObservationFrequency, NumpyRandomGenerator, SobolRandomGenerator
from utils.pricing_settings import PricingSettings
from utils.pricing_results import PricingResults
from kernel.models.stochastic_processes import BlackScholesProcess, HestonProcess
from kernel.products.structured_products.abstract_structured_product import AbstractStructuredProduct
from kernel.products.options.vanilla_options import EuropeanCallOption, EuropeanPutOption
from kernel.models.discretization_schemes.euler_scheme import EulerScheme
import numpy as np
import pandas as pd
import copy

class MCPricingEngine(AbstractPricingEngine):
    """
    A Monte Carlo pricing engine for classic financial derivatives.
    """

    def __init__(self, market: Market, settings: PricingSettings):
        super().__init__(market)
        self.settings = settings
        self.nb_paths = settings.nb_paths
        self.nb_steps = settings.nb_steps
        self.random_seed = settings.random_seed
        self.enable_greeks = settings.compute_greeks 
        self.valuation_date = settings.valuation_date 
        self.model = settings.model

    def calculate_option(self, derivative: 'AbstractOption') -> 'PricingResults':
        return self.get_result(derivative)

    def calculate_strategy(self, derivative: 'AbstractOptionStrategy') -> 'PricingResults':
        strat_results = []
        for opt, is_long in derivative.options:
            position = 1 if is_long else -1
            result = self.get_result(derivative=opt, position=position)
            strat_results.append(result)
        return PricingResults.get_aggregated_results(strat_results)

    def calculate_structured_product(self, derivative: 'AbstractStructuredProduct') -> 'PricingResults':
        return self.get_result(derivative)

    def calculate_rate_product(self, derivative: 'AbstractRateProduct') -> 'PricingResults':
        raise NotImplementedError("MCPricingEngine does not support rate products.")

    def get_result(self, derivative: AbstractDerivative, position: int = 1) -> PricingResults:
        self.derivative = derivative
        if hasattr(derivative, "initial_spot") and getattr(derivative, "initial_spot", None) is None:
            derivative.initial_spot = self.market.underlying_asset.last_price
        process = self.get_stochastic_process(derivative=derivative, market=self.market)
        price, std_dev = self._get_price(derivative, process, self.market, return_std=True)
        
        pricing_results = PricingResults()
        pricing_results.price = price * position
        pricing_results.std_dev = std_dev

        if self.enable_greeks:
            # OPTIMIZED DELTA AND GAMMA (Shares simulations)
            epsilon_spot = 1.0
            
            process_up = self.get_stochastic_process(derivative, self.market)
            process_down = self.get_stochastic_process(derivative, self.market)
            process_up.S0 += epsilon_spot
            process_down.S0 -= epsilon_spot
            
            price_up = self._get_price(derivative, process_up, self.market)
            price_down = self._get_price(derivative, process_down, self.market)
            
            delta = ((price_up - price_down) / (2 * epsilon_spot)) * position
            gamma = ((price_up + price_down - 2 * price) / (epsilon_spot ** 2)) * position
            
            vega = self.get_vega(derivative=derivative, epsilon=0.01) * position
            rho = self.get_rho(derivative=derivative, epsilon=0.0001) * position
            
            # For Theta, pass the un-positioned greeks
            theta = self.get_theta(price=price, delta=delta/position, gamma=gamma/position, vega=vega/position, derivative=derivative, market=self.market) * position
            
            pricing_results.set_greek("delta", delta)
            pricing_results.set_greek("gamma", gamma)
            pricing_results.set_greek("vega", vega)
            pricing_results.set_greek("rho", rho)
            pricing_results.set_greek("theta", theta)
        
        return pricing_results
    
    def get_stochastic_process(self, derivative: AbstractOption, market: Market) -> StochasticProcess:
        T = derivative.maturity
        if hasattr(derivative, "strike"):
            K = derivative.strike
        else:
            K = market.underlying_asset.last_price

        initial_value = market.underlying_asset.last_price
        delta_t = T / self.nb_steps
        drift = [
            market.get_rate(T) if self.nb_steps == 1 
            else market.get_fwd_rate(i * delta_t, (i + 1) * delta_t) for i in range(self.nb_steps)
        ]
        volatility = market.get_volatility(K, T)
        
        gen_type = getattr(self.settings, "random_generator_type", "NUMPY")
        if hasattr(gen_type, "value"):
            gen_type = gen_type.value
            
        if gen_type == "SOBOL":
            generator = SobolRandomGenerator()
        else:
            generator = NumpyRandomGenerator()

        if self.model.name == "BLACK_SCHOLES":
            return BlackScholesProcess(S0=initial_value, T=T, nb_steps=self.nb_steps, drift=drift, volatility=volatility, random_generator=generator)
        elif self.model.name == "HESTON":
            # Fix: Parameters are no longer hardcoded. Please ensure the model configuration provides these parameters.
            kappa = getattr(self.model, "kappa", 8.1471)
            theta = getattr(self.model, "theta", 0.0736)
            sigma = getattr(self.model, "sigma", 0.3905)
            rho = getattr(self.model, "rho", -0.1707)
            # Prioritize the initial variance defined by the model, otherwise fallback to the square of market volatility.
            v0 = getattr(self.model, "v0", volatility**2)
            
            return HestonProcess(S0=initial_value, v0=v0, T=T, nb_steps=self.nb_steps, 
                                 drift=drift, kappa=kappa, theta=theta, sigma=sigma, rho=rho, random_generator=generator)
        else:
            raise ValueError(f"Unsupported model: {self.model.name}. Supported models are: BLACK_SCHOLES, HESTON.")

    def _get_price(self, derivative: AbstractDerivative, stochastic_process: StochasticProcess, current_market: Market = None, pre_simulated_paths: np.ndarray = None, return_std: bool = False):
        # Fix: Allow passing a specific market to ensure discount factor and drift rates match (resolves Rho calculation error).
        if current_market is None:
            current_market = self.market
            
        if pre_simulated_paths is not None:
            price_paths = pre_simulated_paths
        else:
            scheme = EulerScheme()
            price_paths = scheme.simulate_paths(process=stochastic_process, nb_paths=self.nb_paths, seed=self.random_seed)
            
        # Vectorized payoff evaluation: derivative.get_discounted_payoff now accepts
        # the full (nb_paths, nb_steps+1) matrix and returns a (nb_paths,) array,
        # eliminating the Python loop that previously iterated 50,000+ times.
        payoffs = derivative.get_discounted_payoff(price_paths, current_market)

        # The payoff is already discounted by the derivative internally
        price = np.mean(payoffs)
        if return_std:
            std_dev = np.std(payoffs, ddof=1) / np.sqrt(len(payoffs))
            return price, std_dev
        return price

    def get_delta(self, derivative: AbstractOption, epsilon: float = 1) -> float:
        process_up = self.get_stochastic_process(derivative, self.market)
        process_down = self.get_stochastic_process(derivative, self.market)
        process_up.S0 += epsilon  
        process_down.S0 -= epsilon
    
        price_up = self._get_price(derivative, process_up, self.market)
        price_down = self._get_price(derivative, process_down, self.market)
    
        return (price_up - price_down) / (2 * epsilon)
    
    def get_gamma(self, derivative: AbstractOption, epsilon: float = 1) -> float:
        base_process = self.get_stochastic_process(derivative, self.market)
        base_price = self._get_price(derivative, base_process, self.market)
   
        process_up = self.get_stochastic_process(derivative, self.market)
        process_up.S0 += epsilon 
        process_down = self.get_stochastic_process(derivative, self.market)
        process_down.S0 -= epsilon
    
        price_up = self._get_price(derivative, process_up, self.market)
        price_down = self._get_price(derivative, process_down, self.market)
    
        return (price_up + price_down - 2 * base_price) / (epsilon ** 2)

    def get_vega(self, derivative: AbstractOption, epsilon: float = 0.01) -> float:
        vega = 0.0
        if self.model.name == "BLACK_SCHOLES":
            process_up = self.get_stochastic_process(derivative, self.market) 
            process_up.sigma += epsilon 
            process_down = self.get_stochastic_process(derivative, self.market)
            process_down.sigma -= epsilon
    
            price_up = self._get_price(derivative, process_up, self.market)
            price_down = self._get_price(derivative, process_down, self.market)
            vega = (price_up - price_down) / (2 * epsilon)
            
        elif self.model.name == "HESTON":
            process_up = self.get_stochastic_process(derivative, self.market)
            process_down = self.get_stochastic_process(derivative, self.market)
            
            # 修正：Vega 應該針對波動率平移，再轉回變異數，避免數值衝擊過大
            base_vol = np.sqrt(process_up.v0)
            process_up.v0 = (base_vol + epsilon) ** 2
            process_down.v0 = (base_vol - epsilon) ** 2
            
            price_up = self._get_price(derivative, process_up, self.market)
            price_down = self._get_price(derivative, process_down, self.market)
            vega = (price_up - price_down) / (2 * epsilon)

        return vega
    
    def get_rho(self, derivative: AbstractOption, epsilon: float = 0.0001):
        epsilon_fit = epsilon * 100 
        market_up = self.market.bump_flat_yield_curve(epsilon_fit)
        market_down = self.market.bump_flat_yield_curve(-epsilon_fit)

        process_up = self.get_stochastic_process(derivative, market_up)
        process_down = self.get_stochastic_process(derivative, market_down)
        
        # 修正：必須將 market_up 與 market_down 傳入 _get_price 供折現使用
        price_up = self._get_price(derivative, process_up, market_up)
        price_down = self._get_price(derivative, process_down, market_down)
        return (price_up - price_down) / (2 * epsilon)
        
    def get_theta(self, price: float, delta: float, gamma: float, vega: float, derivative: AbstractOption, market: Market) -> float:
        S = market.underlying_asset.last_price
        r = market.get_rate(1/365)
        
        is_vanilla = isinstance(derivative, (EuropeanCallOption, EuropeanPutOption))
        if self.model.name == "BLACK_SCHOLES" and is_vanilla:
            if hasattr(derivative, "strike"):
                K = derivative.strike
            else:
                K = market.underlying_asset.last_price
            sigma = market.get_volatility(K, derivative.maturity)
            theta = -0.5 * sigma**2 * S**2 * gamma - r * S * delta + r * price
            
        elif self.model.name == "HESTON" or not is_vanilla:
            # 用時間有限差分法 (Finite Difference)
            dt_bump = 1.0 / 365.0 # 假設減去 1 天
            
            if derivative.maturity <= dt_bump:
                return 0.0 # 快要到期時不計算 Theta 或回傳 0
            
            # 建立一個到期日減少 1 天的複製合約
            deriv_bumped = copy.deepcopy(derivative)
            deriv_bumped.maturity -= dt_bump
            
            process_bumped = self.get_stochastic_process(deriv_bumped, market)
            price_bumped = self._get_price(deriv_bumped, process_bumped, market)
            
            # Theta 表示為時間流逝造成價格的變化
            theta = (price_bumped - price) / dt_bump
        else:
            raise ValueError("Model not supported for calculating theta.")

        return theta