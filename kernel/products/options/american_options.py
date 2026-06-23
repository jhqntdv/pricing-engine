from kernel.products.options.abstract_option import AbstractOption
import numpy as np
from abc import ABC, abstractmethod
from typing import Union, List

class AmericanAbstractOption(AbstractOption):
    """
    Class representing an American option.
    """
    def __init__(self, strike: float, maturity: float) -> None:
        super().__init__(strike=strike, maturity=maturity)
        self.exercise_times = None

    def payoff(self, path: np.ndarray) -> float:
        pass

    @abstractmethod
    def intrinsic_payoff(self, S: np.ndarray)-> np.ndarray:
        pass
    
class AmericanCallOption(AmericanAbstractOption):

    def __init__(self, strike:float, maturity:float) -> None:
        super().__init__(strike=strike, maturity=maturity)

    def intrinsic_payoff(self, S):
        return np.maximum(S - self.strike, 0)
    
class AmericanPutOption(AmericanAbstractOption):

    def __init__(self, strike:float, maturity:float) -> None:
        super().__init__(strike=strike, maturity=maturity)
    
    def intrinsic_payoff(self, S:np.ndarray):
        return np.maximum(self.strike - S, 0)

class BermudanCallOption(AmericanCallOption):

    def __init__(self, strike, maturity, exercise_times) -> None:
        super().__init__(strike=strike, maturity=maturity)
        self.exercise_times = exercise_times
    
    def intrinsic_payoff(self, S):
        return np.maximum(S - self.strike, 0)
    

class BermudanPutOption(AmericanPutOption):
    def __init__(self, strike:float, maturity:float, exercise_times:List[float]) -> None:
        super().__init__(strike=strike, maturity=maturity)
        self.exercise_times = exercise_times
    
    def intrinsic_payoff(self, S:np.ndarray):
        return np.maximum(self.strike - S, 0)