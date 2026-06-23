import numpy as np
from .abstract_option import AbstractOption


class AsianCallOption(AbstractOption):
    """
    Asian call option where the payoff depends on the average path price.
    """
    def get_discounted_payoff(self, paths: np.ndarray, market: 'Market') -> np.ndarray:
        """
        Vectorized payoff: computes the path average for all paths simultaneously.

        Args:
            paths (np.ndarray): Shape (nb_paths, nb_steps+1).
            market (Market): Market for discount factor.

        Returns:
            np.ndarray: Discounted payoffs of shape (nb_paths,).
        """
        avg_price = np.mean(paths, axis=1)
        return np.maximum(0.0, avg_price - self.strike) * market.get_discount_factor(self.maturity)


class AsianPutOption(AbstractOption):
    """
    Asian put option where the payoff depends on the average path price.
    """
    def get_discounted_payoff(self, paths: np.ndarray, market: 'Market') -> np.ndarray:
        avg_price = np.mean(paths, axis=1)
        return np.maximum(0.0, self.strike - avg_price) * market.get_discount_factor(self.maturity)


class LookbackCallOption(AbstractOption):
    """
    Lookback call option; payoff based on the maximum price reached by the underlying.
    """
    def get_discounted_payoff(self, paths: np.ndarray, market: 'Market') -> np.ndarray:
        max_price = np.max(paths, axis=1)
        return np.maximum(0.0, max_price - self.strike) * market.get_discount_factor(self.maturity)


class LookbackPutOption(AbstractOption):
    """
    Lookback put option; payoff based on the minimum price reached by the underlying.
    """
    def get_discounted_payoff(self, paths: np.ndarray, market: 'Market') -> np.ndarray:
        min_price = np.min(paths, axis=1)
        return np.maximum(0.0, self.strike - min_price) * market.get_discount_factor(self.maturity)


class FloatingStrikeCallOption(AbstractOption):
    """
    Floating strike call option; strike is the path-average price.
    """
    def get_discounted_payoff(self, paths: np.ndarray, market: 'Market') -> np.ndarray:
        floating_strike = np.mean(paths, axis=1)
        return np.maximum(0.0, paths[:, -1] - floating_strike) * market.get_discount_factor(self.maturity)


class FloatingStrikePutOption(AbstractOption):
    """
    Floating strike put option; strike is the path-average price.
    """
    def get_discounted_payoff(self, paths: np.ndarray, market: 'Market') -> np.ndarray:
        floating_strike = np.mean(paths, axis=1)
        return np.maximum(0.0, floating_strike - paths[:, -1]) * market.get_discount_factor(self.maturity)


class ForwardStartCallOption(AbstractOption):
    """
    Forward start call option; strike is set to a percentage of the spot at a specified forward start time.
    """
    def __init__(self, maturity: float, forward_start_time: float, strike_percentage: float = 1.0):
        super().__init__(maturity=maturity, strike=0.0) # Strike dynamically determined
        if forward_start_time <= 0 or forward_start_time >= maturity:
            raise ValueError("forward_start_time must be strictly between 0 and maturity.")
        if strike_percentage < 0:
            raise ValueError("strike_percentage must be non-negative.")
        self.forward_start_time = forward_start_time
        self.strike_percentage = strike_percentage

    def get_discounted_payoff(self, paths: np.ndarray, market: 'Market') -> np.ndarray:
        nb_steps = paths.shape[1] - 1
        idx = int(round((self.forward_start_time / self.maturity) * nb_steps))
        idx = max(0, min(idx, nb_steps))
        
        forward_start_spot = paths[:, idx]
        strike = forward_start_spot * self.strike_percentage
        return np.maximum(0.0, paths[:, -1] - strike) * market.get_discount_factor(self.maturity)


class ForwardStartPutOption(AbstractOption):
    """
    Forward start put option; strike is set to a percentage of the spot at a specified forward start time.
    """
    def __init__(self, maturity: float, forward_start_time: float, strike_percentage: float = 1.0):
        super().__init__(maturity=maturity, strike=0.0) # Strike dynamically determined
        if forward_start_time <= 0 or forward_start_time >= maturity:
            raise ValueError("forward_start_time must be strictly between 0 and maturity.")
        if strike_percentage < 0:
            raise ValueError("strike_percentage must be non-negative.")
        self.forward_start_time = forward_start_time
        self.strike_percentage = strike_percentage

    def get_discounted_payoff(self, paths: np.ndarray, market: 'Market') -> np.ndarray:
        nb_steps = paths.shape[1] - 1
        idx = int(round((self.forward_start_time / self.maturity) * nb_steps))
        idx = max(0, min(idx, nb_steps))
        
        forward_start_spot = paths[:, idx]
        strike = forward_start_spot * self.strike_percentage
        return np.maximum(0.0, strike - paths[:, -1]) * market.get_discount_factor(self.maturity)


class ChooserOption(AbstractOption):
    """
    Chooser option: gives the holder the right to choose whether the option is a call or put at chooser_time.
    """
    def __init__(self, maturity: float, strike: float, chooser_time: float):
        super().__init__(maturity=maturity, strike=strike)
        if chooser_time <= 0 or chooser_time >= maturity:
            raise ValueError("chooser_time must be strictly between 0 and maturity.")
        self.chooser_time = chooser_time

    def get_discounted_payoff(self, paths: np.ndarray, market: 'Market') -> np.ndarray:
        nb_steps = paths.shape[1] - 1
        idx = int(round((self.chooser_time / self.maturity) * nb_steps))
        idx = max(0, min(idx, nb_steps))
        
        S_t1 = paths[:, idx]
        S_T = paths[:, -1]
        
        df_fwd = market.get_fwd_discount_factor(self.chooser_time, self.maturity)
        
        # Rational choice: choose call if S_t1 > K * DF_fwd
        choose_call = S_t1 > self.strike * df_fwd
        
        call_payoff = np.maximum(0.0, S_T - self.strike)
        put_payoff = np.maximum(0.0, self.strike - S_T)
        
        final_payoff = np.where(choose_call, call_payoff, put_payoff)
        return final_payoff * market.get_discount_factor(self.maturity)