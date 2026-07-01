import numpy as np
from .abstract_structured_product import AbstractStructuredProduct

class AbstractParticipationProduct(AbstractStructuredProduct):
    """Abstract class for participation products.
    """
    def __init__(self, maturity: float, rebate: float, leverage: float):
        """Initializes a participation product.

        Args:
            maturity (float): Maturity of the product.
            notional (float): Nominal of the product.
            rebate (float, optional): Fixed refund in case of specific conditions. Defaults to 0.
            leverage (float, optional): Leverage factor to amplify gains or losses. By default, 1.
        """
        super().__init__(maturity)
        self.rebate: float = rebate
        self.leverage: float = leverage

class TwinWin(AbstractParticipationProduct):
    """Twin Win structured product with upper and lower barriers, rebate and lever.
    """
    def __init__(self, maturity: float, upper_barrier: float, lower_barrier: float, rebate: float = 0, leverage: float = 100):
        """Initializes a Twin Win product.

        Args:
            maturity (float): Maturity of the product.
            upper_barrier (float): Upper barrier.
            lower_barrier (float): Lower barrier.
            rebate (float, optional): Fixed refund if the upper barrier is crossed. Defaults to 0.
            leverage (float, optional): Leverage factor. By default, 1.
        """
        super().__init__(maturity = maturity, rebate=rebate, leverage=leverage)
        if upper_barrier <= lower_barrier:
            raise ValueError("The upper barrier must be strictly greater than the lower barrier.")
        self.upper_barrier = upper_barrier
        self.lower_barrier = lower_barrier

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        """Calculates the Twin Win payoff.

        Args:
            paths (np.ndarray): Paths of underlying prices (nb_paths, nb_steps+1).

        Returns:
            np.ndarray: The Twin Win payoff for each path.
        """
        paths = np.atleast_2d(paths)
        final_price = paths[:, -1] # Final price of the underlying
        initial_price = self.initial_spot if getattr(self, "initial_spot", None) is not None else paths[:, 0]
        performance = (final_price / initial_price) * 100
        
        payoff = np.full(performance.shape, 100.0)
        
        # If the upper barrier is crossed
        upper_mask = performance > self.upper_barrier
        payoff[upper_mask] = 100.0 + self.rebate
        
        # If the lower barrier is crossed
        lower_mask = performance < self.lower_barrier
        payoff[lower_mask] = 100.0 + self.leverage * (performance[lower_mask] - 100.0)
        
        # Participation within the range defined by the barriers
        range_mask = ~(upper_mask | lower_mask)
        payoff[range_mask] = self.leverage * np.abs(performance[range_mask] - 100.0) + 100.0

        return payoff

    def get_discounted_payoff(self, paths: np.ndarray, market: 'Market') -> np.ndarray:
        """Calculate the discounted payoff for the Twin Win product.

        Args:
            paths: Array of simulated asset prices.
            market: The market data containing the discount curve.

        Returns:
            An array of discounted payoffs for each path.
        """
        undiscounted = self.payoff(paths)
        df = market.get_discount_factor(self.maturity)
        return undiscounted * df

    def description(self) -> str:
        """Return a description of the Twin Win product.

        Returns:
            A string describing the product's features.
        """
        if self.upper_barrier:
            return (f"Twin Win with upper barrier at {self.upper_barrier}, lower barrier at {self.lower_barrier}, "
                     f"rebate of {self.rebate}, and leverage of {self.leverage}.")
        else:
            return (f"Twin Win with no upper barrier, lower barrier at {self.lower_barrier}, "
                    f"rebate of {self.rebate}, and leverage of {self.leverage}.")
    

class Airbag(AbstractParticipationProduct):
    """AirBag structured product with upper and lower barriers, rebate and lever.
    """
    def __init__(self, maturity: float, upper_barrier: float, lower_barrier: float, rebate: float = 0, leverage: float = 1):
        """Initializes an AirBag product.

        Args:
            maturity (float): Maturity of the product.
            notional (float): Nominal of the product.
            upper_barrier (float): Upper barrier.
            lower_barrier (float): Lower barrier.
            rebate (float, optional): Fixed refund if the upper barrier is crossed. Defaults to 0.
            leverage (float, optional): Leverage factor. By default, 1.
        """
        super().__init__(maturity = maturity, rebate=rebate, leverage=leverage)
        if upper_barrier <= lower_barrier:
            raise ValueError("The upper barrier must be strictly greater than the lower barrier.")
        self.upper_barrier = upper_barrier
        self.lower_barrier = lower_barrier

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        """Calculates the Airbag payoff.

        Args:
            paths (np.ndarray): Paths of underlying prices (nb_paths, nb_steps+1).

        Returns:
            np.ndarray: The Airbag payoff for each path.
        """
        paths = np.atleast_2d(paths)
        final_price = paths[:, -1]  # Final price of the underlying
        initial_price = self.initial_spot if getattr(self, "initial_spot", None) is not None else paths[:, 0]
        performance = (final_price / initial_price) * 100
        
        payoff = np.full(performance.shape, 100.0)
        
        upper_mask = performance > self.upper_barrier
        lower_mask = performance < self.lower_barrier
        mid_mask = (performance < 100.0) & (~lower_mask)
        range_mask = ~(upper_mask | lower_mask | mid_mask)
        
        # If the upper barrier is crossed
        payoff[upper_mask] = 100.0 + self.rebate
        # If the lower barrier is crossed
        payoff[lower_mask] = 100.0 + self.leverage * (performance[lower_mask] - 100.0)
        # performance < 100 (but >= lower_barrier)
        payoff[mid_mask] = 100.0
        # Participation within the range defined by the barriers
        payoff[range_mask] = self.leverage * (performance[range_mask] - 100.0) + 100.0

        return payoff

    def get_discounted_payoff(self, paths: np.ndarray, market: 'Market') -> np.ndarray:
        """Calculate the discounted payoff for the Airbag product.

        Args:
            paths: Array of simulated asset prices.
            market: The market data containing the discount curve.

        Returns:
            An array of discounted payoffs for each path.
        """
        undiscounted = self.payoff(paths)
        df = market.get_discount_factor(self.maturity)
        return undiscounted * df

    def description(self) -> str:
        """Return a description of the Airbag product.

        Returns:
            A string describing the product's features.
        """
        if self.upper_barrier:
            return (f"Airbag with upper barrier at {self.upper_barrier}, lower barrier at {self.lower_barrier}, "
                    f"rebate of {self.rebate}, and leverage of {self.leverage}.")
        else:
            return (f"Airbag with no upper barrier, lower barrier at {self.lower_barrier}, "
                    f"rebate of {self.rebate}, and leverage of {self.leverage}.")