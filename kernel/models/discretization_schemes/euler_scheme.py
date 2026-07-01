import numpy as np
from ..stochastic_processes.stochastic_process import StochasticProcess,OneFactorStochasticProcess,TwoFactorStochasticProcess

class EulerScheme:
    """Euler-Maruyama discretization scheme for simulating stochastic processes."""
    
    def simulate_paths(self, process: StochasticProcess, nb_paths: int, seed: int = 4012) -> np.ndarray:
        """Simulate paths for a given stochastic process using the Euler scheme.

        Args:
            process: The stochastic process to simulate.
            nb_paths: The number of Monte Carlo paths to generate.
            seed: Seed for the random number generator. Defaults to 4012.

        Returns:
            An array containing the simulated paths.

        Raises:
            NotImplementedError: If the process is neither OneFactor nor TwoFactor.
        """
        if isinstance(process, OneFactorStochasticProcess):
            return self._simulate_one_factor(process, nb_paths, seed)
        elif isinstance(process, TwoFactorStochasticProcess):
            return self._simulate_two_factor(process, nb_paths, seed)
        else:
            raise NotImplementedError("Only OneFactor or TwoFactor processes are supported.")

    def _simulate_one_factor(self, process: OneFactorStochasticProcess, nb_paths: int, seed: int) -> np.ndarray:
        """Simulate paths for a one-factor stochastic process.

        Args:
            process: The one-factor stochastic process.
            nb_paths: The number of paths to generate.
            seed: Seed for the random number generator.

        Returns:
            A 2D array of shape (nb_paths, nb_steps + 1) containing the simulated paths.
        """
        paths = np.zeros((nb_paths, process.nb_steps + 1))

        paths[:, 0] = process.S0
        dt = process.dt
        dW = process.get_random_increments(nb_paths, seed)

        for i in range(process.nb_steps):
            x = paths[:, i]
            dW_i = dW[:, i]
            drift = process.get_drift(i, x)
            vol = process.get_volatility(i, x)

            if getattr(process, "is_log_process", False):
                # Convert absolute drift/vol to proportional rates
                safe_x = np.maximum(x, 1e-12)
                mu_prop = drift / safe_x     # mu[t]
                sig_prop = vol / safe_x      # sigma
                # Exact geometric step (Ito-corrected)
                paths[:, i + 1] = x * np.exp(
                    (mu_prop - 0.5 * sig_prop ** 2) * dt + sig_prop * dW_i
                )
            else:
                # Raw Euler for normal-distributed processes
                paths[:, i + 1] = x + drift * dt + vol * dW_i
        return paths

    def _simulate_two_factor(self, process: TwoFactorStochasticProcess, nb_paths: int, seed: int) -> np.ndarray:
        """Simulate paths for a two-factor stochastic process (e.g., Heston).

        Args:
            process: The two-factor stochastic process.
            nb_paths: The number of paths to generate.
            seed: Seed for the random number generator.

        Returns:
            A 2D array of shape (nb_paths, nb_steps + 1) containing the primary asset paths.
        """
        paths = np.zeros((nb_paths, process.nb_steps + 1, 2))
        paths[:, 0, 0] = process.S0
        paths[:, 0, 1] = process.v0
        dt = process.dt
        dW1, dW2 = process.get_random_increments(nb_paths, seed)
        for i in range(process.nb_steps):
            x = paths[:, i, 0]
            v = paths[:, i, 1]
            dW1_i = dW1[:, i]
            dW2_i = dW2[:, i]

            drift = process.get_drift(i, x)       # mu[t] * x
            vol_drift = process.get_vol_drift(i, v)  # kappa * (theta - max(v,0))
            vol_vol = process.get_vol_vol(i, v)      # sigma * sqrt(max(v,0))

            if getattr(process, "is_log_process", False):
                # Log-Euler for spot dimension only
                safe_x = np.maximum(x, 1e-12)
                mu_prop = drift / safe_x               # mu[t]
                v_pos = np.maximum(v, 0)                # Full Truncation
                # Exact geometric step with stochastic variance
                x_next = x * np.exp(
                    (mu_prop - 0.5 * v_pos) * dt + np.sqrt(v_pos) * dW1_i
                )
            else:
                x_next = x + drift * dt + np.sqrt(np.maximum(v, 0)) * x * dW1_i

            # Variance dimension: always Raw Euler with Full Truncation
            v_next = v + vol_drift * dt + vol_vol * dW2_i

            paths[:, i + 1, 0] = x_next
            paths[:, i + 1, 1] = v_next

        return paths[:, :, 0] 
