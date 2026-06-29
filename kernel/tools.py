from enum import Enum

class RateCurveType(Enum):
    """Enumeration of different rate curve types used in pricing."""
    # Risk free rate_curve yield curves
    RF_US_TREASURY = "RateCurve_temp.xlsx"
    RF_OAT = ""
    # Swap curves
    SWAP_EURIBOR = ""
    SWAP_SOFR = ""
    # OIS curves
    OIS_SOFR = ""
    OIS_ESTER = ""
    # Credit spread curves
    CREDIT_IG = ""
    CREDIT_HY = ""

class CalendarConvention(Enum):
    """Enumeration of standard day count conventions for interest calculations."""

    ACT_360 = "Actual/360"
    ACT_365 = "Actual/365"
    ACT_ACT = "Actual/Actual"
    THIRTY_360 = "30/360"

class ObservationFrequency(Enum):
    """Enum mapping observation frequencies to the number of periods in a year.
    """
    ANNUAL = 1          # Annual
    SEMIANNUAL = 2      # Semi-annual
    QUARTERLY = 4       # Quarterly
    MONTHLY = 12        # Monthly
    
class Model(Enum):
    """Enum mapping model names to their respective classes.
    """
    BLACK_SCHOLES = "Black-Scholes"
    HESTON = "Heston"
    
class RandomGeneratorType(Enum):
    """Enumeration of supported random number generator types."""
    NUMPY = "NUMPY"
    SOBOL = "SOBOL"

from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import norm
import warnings
from scipy.stats.qmc import Sobol
from typing import Union, Tuple

class AbstractRandomGenerator(ABC):
    """Abstract base class for random number generators."""

    @abstractmethod
    def get_standard_normal(self, nb_paths: int, nb_steps: int, nb_factors: int = 1, seed: int = 4012) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """Generate standard normal random increments.

        Args:
            nb_paths: Number of Monte Carlo paths.
            nb_steps: Number of simulation steps.
            nb_factors: Number of factors (e.g., 1 for Black-Scholes, 2 for Heston). Defaults to 1.
            seed: Seed for the generator. Defaults to 4012.

        Returns:
            An array of shape (nb_paths, nb_steps) if nb_factors is 1, 
            or a tuple of such arrays if nb_factors > 1.
        """
        pass

class NumpyRandomGenerator(AbstractRandomGenerator):
    """Random number generator using numpy's default RNG (pseudo-random)."""

    def get_standard_normal(self, nb_paths: int, nb_steps: int, nb_factors: int = 1, seed: int = 4012) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """Generate standard normal random increments using NumPy."""
        rng = np.random.default_rng(seed)
        if nb_factors == 1:
            return rng.standard_normal(size=(nb_paths, nb_steps))
        
        # We generate directly an array of shape (nb_factors, nb_paths, nb_steps)
        Z = rng.standard_normal(size=(nb_factors, nb_paths, nb_steps))
        # Since the code expects a tuple when nb_factors > 1, we unpack it.
        return tuple(Z[i] for i in range(nb_factors))

class SobolRandomGenerator(AbstractRandomGenerator):
    """Random number generator using Sobol sequences (quasi-random)."""

    def get_standard_normal(self, nb_paths: int, nb_steps: int, nb_factors: int = 1, seed: int = 4012) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """Generate standard normal random increments using a Sobol sequence."""
        dim = nb_steps * nb_factors
        sampler = Sobol(d=dim, scramble=True, seed=seed)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") # Ignore power of 2 warning
            sample = sampler.random(n=nb_paths)
            
        sample = np.clip(sample, 1e-10, 1 - 1e-10)
        Z = norm.ppf(sample)
        
        Z = Z.reshape((nb_paths, nb_factors, nb_steps))
        Z = Z.transpose((1, 0, 2))
        
        if nb_factors == 1:
            return Z[0]
        return tuple(Z[i] for i in range(nb_factors))
class EquityGreeksName(Enum):
    """Enumeration of standard Equity Greeks."""
    DELTA = "delta"
    VEGA = "vega"
    GAMMA = "gamma"