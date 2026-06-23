from kernel.models.stochastic_processes.stochastic_process import StochasticProcess,TwoFactorStochasticProcess
from kernel.tools import AbstractRandomGenerator
import numpy as np
from typing import Tuple, Union, List

class HestonProcess(TwoFactorStochasticProcess):
    """
    Class representing a Heston stochastic volatility process under the risk-neutral measure Q.
    """

    def __init__(self, S0: float, v0: float, T: float, nb_steps: int, drift: Union[List[float], np.ndarray],
                 kappa: float, theta: float, sigma: float, rho: float, random_generator: AbstractRandomGenerator = None):
        """
        Initializes the Heston process.

        Parameters:
            S0 (float): Initial spot price
            v0 (float): Initial variance
            T (float): Maturity
            nb_steps (int): Number of steps
            drift (Union[List[float], np.ndarray]): Array or list of drift rates under Q (risk-free or forward rates)
            kappa (float): Mean reversion speed of variance
            theta (float): Long-run variance mean
            sigma (float): Volatility of volatility
            rho (float): Correlation between spot and volatility processes
        """
        super().__init__(S0, T, nb_steps, nb_factors=2, random_generator=random_generator)
        self.v0 = v0
        self.mu = drift
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho

    def get_drift(self, t: int, x: np.ndarray) -> np.ndarray:
        """Drift of the spot price under the risk-neutral measure Q."""
        return self.mu[t] * x

    def get_vol_drift(self, t: int, v: np.ndarray) -> np.ndarray:
        """Drift of the variance under Q (mean reversion component)."""
        return self.kappa * (self.theta - np.maximum(v, 0)) # Full Truncation Scheme, 2021

    def get_vol_vol(self, t: int, v: np.ndarray) -> np.ndarray:
        """Volatility of the volatility (variance process)."""
        return self.sigma * np.sqrt(np.maximum(v, 0))  # Ensure positivity

    def get_random_increments(self, nb_paths: int, seed: int = 4012) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates the two correlated brownian motions of the Heston process with the Cholesky decomposition.

        Parameters:
            nb_paths (int): The number of paths to simulate
            seed (int): The seed for the random number generator. Default is 4012
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: The generated increments for the two correlated brownian motions
        """
        # We use the injected random generator to generate the independent brownian motions
        Z1, Z2 = super().get_random_increments(nb_paths, seed)

        # Apply the Cholesky decomposition to get the correlated brownian motions 
        Z3 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2

        return Z1 * np.sqrt(self.dt), Z3 * np.sqrt(self.dt)
        