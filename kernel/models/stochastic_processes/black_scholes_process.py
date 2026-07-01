from kernel.models.stochastic_processes.stochastic_process import StochasticProcess,OneFactorStochasticProcess
from kernel.tools import AbstractRandomGenerator
import numpy as np
from typing import Union, List

class BlackScholesProcess(OneFactorStochasticProcess):
    """Class representing a Black-Scholes process.
    Inherits from StochasticProcess.
    """

    def __init__(self, S0: float, T: float, nb_steps: int, drift: Union[List[float], np.ndarray], volatility: float, random_generator: AbstractRandomGenerator = None):
        """Initializes the stochastic process.

        Parameters:
            S0 (float): The initial value of the process
            T (float): The maturity of the process
            nb_steps (int): The number of steps to simulate
            drift (float): The drift of the process
            volatility (float): The volatility of the process
        """
        super().__init__(S0, T, nb_steps, random_generator=random_generator, is_log_process=True)
        self.mu = drift
        self.sigma = volatility 
    
    def get_drift(self, t:int, x:np.ndarray) -> np.ndarray:
        """Calculate the drift of the Black-Scholes process.

        Args:
            t: The time step index.
            x: Array of current state values.

        Returns:
            Array of drift values.
        """
        return self.mu[t] * x

    def get_volatility(self, t:int, x:np.ndarray) -> np.ndarray:
        """Calculate the volatility of the Black-Scholes process.

        Args:
            t: The time step index.
            x: Array of current state values.

        Returns:
            Array of volatility values.
        """
        return self.sigma * x
    
    def get_random_increments(self, nb_paths : int, seed :int = 4012) -> np.ndarray:
        """Generates random increments of the brownian motion of Black-Scholes process.

        Parameters:
            nb_paths (int): The number of paths to simulate
            seed (int): The seed for the random number generator. Default is 4012
        
        Returns:
            np.ndarray: The generated increments for the brownian motion
        """
        Z = super().get_random_increments(nb_paths, seed)
        return Z * np.sqrt(self.dt)