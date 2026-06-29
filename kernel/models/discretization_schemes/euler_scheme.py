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
            t = i * dt
            x = paths[:, i]
            dW_i = dW[:, i]
            drift = process.get_drift(i, x)
            vol = process.get_volatility(i, x)

            paths[:, i + 1] = x + drift * dt + vol  * dW_i
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
            t = i * dt
            x = paths[:, i, 0]
            v = paths[:, i, 1]
            dW1_i = dW1[:, i]
            dW2_i = dW2[:, i]

            drift = process.get_drift(i, x)
            vol_drift = process.get_vol_drift(i, v)
            vol_vol = process.get_vol_vol(i, v)

            x_next = x + drift * dt + np.sqrt(np.maximum(v, 0)) * x * dW1_i
            v_next = v + vol_drift * dt + vol_vol * dW2_i

            paths[:, i + 1, 0] = x_next
            paths[:, i + 1, 1] = v_next

        return paths[:, :, 0] 
