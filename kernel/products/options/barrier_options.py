import numpy as np
from .abstract_option import AbstractOption


class AbstractBarrierOption(AbstractOption):
    """Abstract class representing the different barrier options.
    """
    def __init__(self, maturity: float, strike: float, barrier: float) -> None:
        """Initializes a barrier option with a maturity, a strike price and a barrier.
        """
        super().__init__(maturity, strike)
        self.barrier = barrier


class UpBarrierOption(AbstractBarrierOption):
    """Abstract class for options with a high (up) barrier.
    """
    def __init__(self, maturity: float, strike: float, barrier: float) -> None:
        super().__init__(maturity, strike, barrier)
        if self.barrier <= self.strike:
            raise ValueError("The barrier must be greater than the strike for a high barrier.")

    def is_barrier_breached(self, paths: np.ndarray) -> np.ndarray:
        """Vectorized barrier check across all paths.

        Args:
            paths (np.ndarray): Shape (nb_paths, nb_steps+1).

        Returns:
            np.ndarray: Boolean array of shape (nb_paths,); True if the up-barrier was hit.
        """
        return np.max(paths, axis=1) > self.barrier


class DownBarrierOption(AbstractBarrierOption):
    """Abstract class for options with a low (down) barrier.
    """
    def __init__(self, maturity: float, strike: float, barrier: float) -> None:
        super().__init__(maturity, strike, barrier)
        if self.barrier >= self.strike:
            raise ValueError("The barrier must be lower than the strike for a low barrier.")

    def is_barrier_breached(self, paths: np.ndarray) -> np.ndarray:
        """Vectorized barrier check across all paths.

        Args:
            paths (np.ndarray): Shape (nb_paths, nb_steps+1).

        Returns:
            np.ndarray: Boolean array of shape (nb_paths,); True if the down-barrier was hit.
        """
        return np.min(paths, axis=1) < self.barrier


class UpAndOutCallOption(UpBarrierOption):
    """Class representing a call barrier option with an Up-And-Out barrier.
    Knocked out (pays 0) if the spot ever rises above the barrier.
    """
    def get_discounted_payoff(self, paths: np.ndarray, market: 'Market') -> np.ndarray:
        """Calculate the discounted payoff for an Up-And-Out call option.

        Args:
            paths: Array of simulated asset prices.
            market: The market data containing the discount curve.

        Returns:
            An array of discounted payoffs for each path.
        """
        breached = self.is_barrier_breached(paths)
        intrinsic = np.maximum(0.0, paths[:, -1] - self.strike)
        # Barrier breach kills the payoff
        payoffs = np.where(breached, 0.0, intrinsic)
        return payoffs * market.get_discount_factor(self.maturity)


class UpAndInCallOption(UpBarrierOption):
    """Class representing a call barrier option with an Up-And-In barrier.
    Knocked in (pays intrinsic) only if the spot ever rises above the barrier.
    """
    def get_discounted_payoff(self, paths: np.ndarray, market: 'Market') -> np.ndarray:
        """Calculate the discounted payoff for an Up-And-In call option.

        Args:
            paths: Array of simulated asset prices.
            market: The market data containing the discount curve.

        Returns:
            An array of discounted payoffs for each path.
        """
        breached = self.is_barrier_breached(paths)
        intrinsic = np.maximum(0.0, paths[:, -1] - self.strike)
        payoffs = np.where(breached, intrinsic, 0.0)
        return payoffs * market.get_discount_factor(self.maturity)


class DownAndInCallOption(DownBarrierOption):
    """Class representing a call barrier option with a Down-And-In barrier.
    Knocked in (pays intrinsic) only if the spot ever falls below the barrier.
    """
    def get_discounted_payoff(self, paths: np.ndarray, market: 'Market') -> np.ndarray:
        """Calculate the discounted payoff for a Down-And-In call option.

        Args:
            paths: Array of simulated asset prices.
            market: The market data containing the discount curve.

        Returns:
            An array of discounted payoffs for each path.
        """
        breached = self.is_barrier_breached(paths)
        intrinsic = np.maximum(0.0, paths[:, -1] - self.strike)
        payoffs = np.where(breached, intrinsic, 0.0)
        return payoffs * market.get_discount_factor(self.maturity)


class DownAndOutCallOption(DownBarrierOption):
    """Class representing a call barrier option with a Down-And-Out barrier.
    Knocked out (pays 0) if the spot ever falls below the barrier.
    """
    def get_discounted_payoff(self, paths: np.ndarray, market: 'Market') -> np.ndarray:
        """Calculate the discounted payoff for a Down-And-Out call option.

        Args:
            paths: Array of simulated asset prices.
            market: The market data containing the discount curve.

        Returns:
            An array of discounted payoffs for each path.
        """
        breached = self.is_barrier_breached(paths)
        intrinsic = np.maximum(0.0, paths[:, -1] - self.strike)
        payoffs = np.where(breached, 0.0, intrinsic)
        return payoffs * market.get_discount_factor(self.maturity)


class UpAndInPutOption(UpBarrierOption):
    """Class representing a put barrier option with an Up-And-In barrier.
    Knocked in (pays intrinsic) only if the spot ever rises above the barrier.
    """
    def get_discounted_payoff(self, paths: np.ndarray, market: 'Market') -> np.ndarray:
        """Calculate the discounted payoff for an Up-And-In put option.

        Args:
            paths: Array of simulated asset prices.
            market: The market data containing the discount curve.

        Returns:
            An array of discounted payoffs for each path.
        """
        breached = self.is_barrier_breached(paths)
        intrinsic = np.maximum(0.0, self.strike - paths[:, -1])
        payoffs = np.where(breached, intrinsic, 0.0)
        return payoffs * market.get_discount_factor(self.maturity)


class UpAndOutPutOption(UpBarrierOption):
    """Class representing a put barrier option with an Up-And-Out barrier.
    Knocked out (pays 0) if the spot ever rises above the barrier.
    """
    def get_discounted_payoff(self, paths: np.ndarray, market: 'Market') -> np.ndarray:
        """Calculate the discounted payoff for an Up-And-Out put option.

        Args:
            paths: Array of simulated asset prices.
            market: The market data containing the discount curve.

        Returns:
            An array of discounted payoffs for each path.
        """
        breached = self.is_barrier_breached(paths)
        intrinsic = np.maximum(0.0, self.strike - paths[:, -1])
        payoffs = np.where(breached, 0.0, intrinsic)
        return payoffs * market.get_discount_factor(self.maturity)


class DownAndInPutOption(DownBarrierOption):
    """Class representing a put barrier option with a Down-And-In barrier.
    Knocked in (pays intrinsic) only if the spot ever falls below the barrier.
    """
    def get_discounted_payoff(self, paths: np.ndarray, market: 'Market') -> np.ndarray:
        """Calculate the discounted payoff for a Down-And-In put option.

        Args:
            paths: Array of simulated asset prices.
            market: The market data containing the discount curve.

        Returns:
            An array of discounted payoffs for each path.
        """
        breached = self.is_barrier_breached(paths)
        intrinsic = np.maximum(0.0, self.strike - paths[:, -1])
        payoffs = np.where(breached, intrinsic, 0.0)
        return payoffs * market.get_discount_factor(self.maturity)


class DownAndOutPutOption(DownBarrierOption):
    """Class representing a put barrier option with a Down-And-Out barrier.
    Knocked out (pays 0) if the spot ever falls below the barrier.
    """
    def get_discounted_payoff(self, paths: np.ndarray, market: 'Market') -> np.ndarray:
        """Calculate the discounted payoff for a Down-And-Out put option.

        Args:
            paths: Array of simulated asset prices.
            market: The market data containing the discount curve.

        Returns:
            An array of discounted payoffs for each path.
        """
        breached = self.is_barrier_breached(paths)
        intrinsic = np.maximum(0.0, self.strike - paths[:, -1])
        payoffs = np.where(breached, 0.0, intrinsic)
        return payoffs * market.get_discount_factor(self.maturity)