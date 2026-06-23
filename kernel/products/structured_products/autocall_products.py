from abc import ABC, abstractmethod
import numpy as np
from ...tools import ObservationFrequency
from . import AbstractStructuredProduct


class AbstractAutocall(AbstractStructuredProduct):
    """
    Abstract class for Autocall products.

    Attributes:
        maturity (float): Maturity of the product in years.
        observation_frequency (ObservationFrequency): Frequency of observations.
        capital_barrier (float): Level of capital protection (as % of initial spot).
        autocall_barrier (float): Early callback level as % of initial spot.
        coupon_rate (float): Coupon rate paid per period (as absolute value, e.g. 5.0).
        is_security (bool): Improved protection in case of loss.
        is_plus (bool): "Plus" option which accumulates unpaid coupons.
    """
    def __init__(self, maturity: float, observation_frequency: ObservationFrequency,
                 capital_barrier: float, autocall_barrier: float, coupon_rate: float,
                 is_security: bool = False, is_plus: bool = False, initial_spot: float = None):

        super().__init__(maturity, initial_spot)
        self.observation_frequency = observation_frequency
        self.capital_barrier = capital_barrier
        self.autocall_barrier = autocall_barrier
        self.coupon_rate = coupon_rate
        self.is_security = is_security
        self.is_plus = is_plus

    @abstractmethod
    def get_discounted_payoff(self, paths: np.ndarray, market: 'Market') -> np.ndarray:
        pass


class Phoenix(AbstractAutocall):
    """
    Phoenix product: pays periodic coupons if the underlying is above a coupon barrier.
    - Recalled automatically if the underlying exceeds the autocall barrier on any observation date.
    - At maturity, capital protection applies based on the capital barrier level.

    Vectorized implementation: operates on (nb_paths, nb_steps+1) simultaneously.
    The loop over observation dates is O(num_observations) which is very small (<=12 per year).
    At each observation, boolean masks select which paths autocall or earn coupons.
    """
    def __init__(self, maturity, observation_frequency,
                 capital_barrier, autocall_barrier,
                 coupon_rate, coupon_barrier,
                 is_security=False, is_plus=False, initial_spot=None):

        super().__init__(maturity, observation_frequency, capital_barrier, autocall_barrier, coupon_rate, is_security, is_plus, initial_spot)
        self.coupon_barrier = coupon_barrier

    def get_discounted_payoff(self, paths: np.ndarray, market: 'Market') -> np.ndarray:
        nb_paths = paths.shape[0]
        nb_steps = paths.shape[1] - 1

        # Select the observation time indices uniformly across the path
        num_observations = int(round(self.maturity * self.observation_frequency.value + 1))
        target_times = np.linspace(0, self.maturity, num_observations)
        index_observations = np.round((target_times / self.maturity) * nb_steps).astype(int)
        index_observations = np.clip(index_observations, 0, nb_steps)

        # Normalize paths to % of initial spot (100-based)
        initial_spot = self.initial_spot if getattr(self, "initial_spot", None) is not None else paths[:, 0]
        safe_initial_spot = np.where(initial_spot == 0.0, 1e-9, initial_spot)
        
        # initial_spot may be a scalar (set from market) or per-path array (paths[:, 0])
        if np.isscalar(initial_spot):
            obs_paths = (paths[:, index_observations] / safe_initial_spot) * 100  # (nb_paths, num_obs)
        else:
            obs_paths = (paths[:, index_observations] / safe_initial_spot[:, None]) * 100

        # --- Vectorized backward accumulation ---
        # payoffs[i] = final discounted payoff for path i
        payoffs = np.zeros(nb_paths)
        # active[i] = True if path i has not yet been autocalled
        active = np.ones(nb_paths, dtype=bool)
        # Accumulated coupon per path (may include missed coupons for is_plus)
        cumulative_coupons = np.zeros(nb_paths)
        missed_coupons = np.zeros(nb_paths)

        for t in range(1, num_observations):
            spot_t = obs_paths[:, t]
            discount_time = t / self.observation_frequency.value
            df = market.get_discount_factor(discount_time)

            # --- Autocall event ---
            autocalled = active & (spot_t >= self.autocall_barrier)
            if np.any(autocalled):
                autocall_payoff = 100.0 + cumulative_coupons + self.coupon_rate + missed_coupons
                payoffs[autocalled] = autocall_payoff[autocalled] * df
                active[autocalled] = False

            # --- Coupon event (only for still-active paths, not already autocalled) ---
            coupon_paid = active & (spot_t >= self.coupon_barrier)
            cumulative_coupons[coupon_paid] += self.coupon_rate + missed_coupons[coupon_paid]
            missed_coupons[coupon_paid] = 0.0

            # --- Missed coupon accumulation (for is_plus) ---
            if self.is_plus:
                missed = active & (spot_t < self.coupon_barrier)
                missed_coupons[missed] += self.coupon_rate

        # --- Paths still active at maturity ---
        final_discount_time = (num_observations - 1) / self.observation_frequency.value
        df_final = market.get_discount_factor(final_discount_time)
        final_spot = obs_paths[:, -1]

        if np.any(active):
            above_capital = active & (final_spot >= self.capital_barrier)
            below_capital = active & (final_spot < self.capital_barrier)

            # Above capital barrier: return 100 + accumulated coupons
            payoffs[above_capital] = (100.0 + cumulative_coupons[above_capital] + missed_coupons[above_capital]) * df_final

            # Below capital barrier
            if np.any(below_capital):
                if self.is_security:
                    gearing = 100.0 / self.capital_barrier
                    loss = (self.capital_barrier - final_spot[below_capital]) * gearing
                    payoffs[below_capital] = np.maximum(0.0, 100.0 - loss + cumulative_coupons[below_capital]) * df_final
                else:
                    payoffs[below_capital] = np.maximum(0.0, final_spot[below_capital] + cumulative_coupons[below_capital]) * df_final

        return payoffs


class Eagle(AbstractAutocall):
    """
    Eagle product: variation of the Phoenix with a simpler coupon structure.
    - Coupon at autocall is proportional to the number of periods elapsed.
    - No separate coupon barrier; recall triggers a full coupon payment.

    Vectorized implementation mirrors Phoenix logic above.
    """
    def __init__(self, maturity, observation_frequency,
                 capital_barrier, autocall_barrier,
                 coupon_rate,
                 is_security=False, is_plus=False, initial_spot=None):

        super().__init__(maturity, observation_frequency, capital_barrier, autocall_barrier, coupon_rate, is_security, is_plus, initial_spot)

    def get_discounted_payoff(self, paths: np.ndarray, market: 'Market') -> np.ndarray:
        nb_paths = paths.shape[0]
        nb_steps = paths.shape[1] - 1

        num_observations = int(round(self.maturity * self.observation_frequency.value + 1))
        target_times = np.linspace(0, self.maturity, num_observations)
        index_observations = np.round((target_times / self.maturity) * nb_steps).astype(int)
        index_observations = np.clip(index_observations, 0, nb_steps)

        initial_spot = self.initial_spot if getattr(self, "initial_spot", None) is not None else paths[:, 0]
        safe_initial_spot = np.where(initial_spot == 0.0, 1e-9, initial_spot)
        
        if np.isscalar(initial_spot):
            obs_paths = (paths[:, index_observations] / safe_initial_spot) * 100
        else:
            obs_paths = (paths[:, index_observations] / safe_initial_spot[:, None]) * 100

        payoffs = np.zeros(nb_paths)
        active = np.ones(nb_paths, dtype=bool)

        for t in range(1, num_observations):
            spot_t = obs_paths[:, t]
            discount_time = t / self.observation_frequency.value
            df = market.get_discount_factor(discount_time)

            autocalled = active & (spot_t >= self.autocall_barrier)
            if np.any(autocalled):
                # Eagle pays coupon proportional to time elapsed
                payoffs[autocalled] = (100.0 + t * self.coupon_rate) * df
                active[autocalled] = False

        # --- Paths still active at maturity ---
        final_discount_time = (num_observations - 1) / self.observation_frequency.value
        df_final = market.get_discount_factor(final_discount_time)
        final_spot = obs_paths[:, -1]

        if np.any(active):
            above_capital = active & (final_spot >= self.capital_barrier)
            below_capital = active & (final_spot < self.capital_barrier)

            if self.is_plus:
                payoffs[above_capital] = (100.0 + (num_observations - 1) * self.coupon_rate) * df_final
            else:
                payoffs[above_capital] = 100.0 * df_final

            if np.any(below_capital):
                if self.is_security:
                    gearing = 100.0 / self.capital_barrier
                    loss = (self.capital_barrier - final_spot[below_capital]) * gearing
                    payoffs[below_capital] = np.maximum(0.0, 100.0 - loss) * df_final
                else:
                    payoffs[below_capital] = np.maximum(0.0, final_spot[below_capital]) * df_final

        return payoffs