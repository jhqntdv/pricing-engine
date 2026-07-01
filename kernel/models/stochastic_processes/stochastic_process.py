from abc import ABC, abstractmethod
import numpy as np
from typing import Union, Tuple
from kernel.tools import AbstractRandomGenerator, NumpyRandomGenerator

class StochasticProcess(ABC):
    """Abstract class representing a stochastic process.
    """

    def __init__(self, S0: float, T: float, nb_steps: int, nb_factors: int = 1, random_generator: AbstractRandomGenerator = None, is_log_process: bool = True) -> None:
        """Initialize the base stochastic process.

        Args:
            S0 (float): Initial value of the process (e.g., spot price).
            T (float): Time to maturity in years.
            nb_steps (int): Number of discretization steps.
            nb_factors (int, optional): Number of driving Brownian motions. Defaults to 1.
            random_generator (RandomGenerator, optional): Source of random numbers. 
                Defaults to NumpyRandomGenerator if None is provided.
            is_log_process (bool, optional): Whether the process represents a strictly positive
                geometric process (e.g., asset prices) that should use exact Log-Euler discretization. 
                Defaults to True. False indicates a normal/additive process (e.g., Vasicek) that uses Raw Euler.
        """
        self.S0 = S0
        self.T = T
        self.nb_steps = nb_steps
        self.dt = T / nb_steps if nb_steps > 0 else 0.0
        self.nb_factors = nb_factors
        self.random_generator = random_generator or NumpyRandomGenerator()
        self.is_log_process = is_log_process

    @abstractmethod
    def get_random_increments(self, nb_paths: int, seed: int = 4012) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """Generates random increments of the brownian motion(s).

        Parameters:
            nb_paths (int): The number of paths to simulate
            seed (int): The seed for the random number generator. Default is 4012
        
        Returns:
            np.ndarray: The generated increments for the brownian motion
                or
            tuple(np.ndarray): The generated increments for the brownian motions if the process has multiple sources of randomness
        """
        return self.random_generator.get_standard_normal(nb_paths, self.nb_steps, self.nb_factors, seed)

class OneFactorStochasticProcess(StochasticProcess):
    """Abstract class for one-factor processes (e.g. Black-Scholes).
    """

    @abstractmethod
    def get_drift(self, t: int, x: np.ndarray) -> np.ndarray:
        """Calculate the drift for a one-factor process.

        Args:
            t: Time step index.
            x: Array of current state values.

        Returns:
            An array of drift values.
        """
        pass

    @abstractmethod
    def get_volatility(self, t: int, x: np.ndarray) -> np.ndarray:
        """Calculate the volatility for a one-factor process.

        Args:
            t: Time step index.
            x: Array of current state values.

        Returns:
            An array of volatility values.
        """
        pass


class TwoFactorStochasticProcess(StochasticProcess):
    @abstractmethod
    def get_drift(self, t: int, x: np.ndarray) -> np.ndarray:
        """Calculate the drift of the underlying asset for a two-factor process.

        Args:
            t: Time step index.
            x: Array of current state values.

        Returns:
            An array of drift values.
        """
        pass

    @abstractmethod
    def get_vol_drift(self, t: int, v: np.ndarray) -> np.ndarray:
        """Calculate the drift of the variance process.

        Args:
            t: Time step index.
            v: Array of current variance values.

        Returns:
            An array of variance drift values.
        """
        pass

    @abstractmethod
    def get_vol_vol(self, t: int, v: np.ndarray) -> np.ndarray:
        """Calculate the volatility of the variance process (vol-of-vol).

        Args:
            t: Time step index.
            v: Array of current variance values.

        Returns:
            An array of vol-of-vol values.
        """
        pass
