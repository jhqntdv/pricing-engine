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

    def payoff(self, paths: np.ndarray) -> float:
        """Calculates the Twin Win payoff.

        Args:
            paths (np.ndarray): Paths of underlying prices.

        Returns:
            float: The Twin Win payoff.
        """
        final_price: float = paths[-1] # Final price of the underlying
        initial_price : float = self.initial_spot if getattr(self, "initial_spot", None) is not None else paths[0]
        performance: float = (final_price / initial_price) * 100
        payoff = 100
        # If the upper barrier is crossed
        if performance > self.upper_barrier:
            payoff = 100 + self.rebate  # Fixed refund
        # If the lower barrier is crossed
        elif performance < self.lower_barrier:
            # Loss similar to a Put Down-and-In
            loss = self.leverage * (performance - 100)
            payoff =  100 + loss  # Loss
        else:
            # Participation within the range defined by the barriers
            payoff = self.leverage * abs(performance - 100) + 100

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

    def payoff(self, paths: np.ndarray) -> float:
        """Calculates the Airbag payoff.

        Args:
            paths (np.ndarray): Paths of underlying prices.

        Returns:
            float: The Airbag payoff.
        """
        final_price: float = paths[-1]  # Final price of the underlying
        initial_price: float = self.initial_spot if getattr(self, "initial_spot", None) is not None else paths[0]
        performance: float =  (final_price / initial_price) * 100
        payoff = 100
        # If the upper barrier is crossed
        if performance > self.upper_barrier:
            payoff =  100 + self.rebate  # Fixed refund
        # If the lower barrier is crossed
        elif performance < self.lower_barrier:
            # Loss similar to a Put Down-and-In
            loss = self.leverage  * (performance - 100)
            payoff = loss + 100  # Loss
        elif performance < 100:
            payoff =  100
        else:
            # Participation within the range defined by the barriers
            payoff = self.leverage * (performance - 100) + 100

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