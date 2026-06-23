from ..abstract_derivative import AbstractDerivative
import numpy as np

class AbstractOption(AbstractDerivative):
    """
    Abstract class representing the different options.
    """
    def __init__(self, maturity: float, strike: float = None) -> None:
        super().__init__()
        if maturity <= 0:
            raise ValueError("Maturity must be positive.")
        self.maturity = maturity
        self.strike = strike
    
    def get_discounted_payoff(self, paths: np.ndarray, market: 'Market') -> np.ndarray:
        """
        Calculates the discounted payoffs for all Monte Carlo paths simultaneously.

        Vectorized signature: accepts the full (nb_paths, nb_steps+1) price matrix
        and returns a 1D array of shape (nb_paths,), eliminating the per-path Python
        loop in the pricing engine for a significant speedup.

        Args:
            paths (np.ndarray): 2D array of shape (nb_paths, nb_steps+1) containing
                                price paths for all simulated scenarios.
            market (Market): The market object used to retrieve discount factors.

        Returns:
            np.ndarray: 1D array of shape (nb_paths,) containing discounted payoffs.
        """
        raise NotImplementedError("Subclasses must implement vectorized get_discounted_payoff.")

    def accept(self, engine: 'AbstractPricingEngine'):
        return engine.calculate_option(self)