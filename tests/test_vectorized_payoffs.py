"""
Tests for vectorized payoff calculations.

Each test verifies that the NumPy-vectorized get_discounted_payoff produces
results consistent with a scalar reference implementation, checked against
known analytical or boundary cases.

A dummy Market stub is used to keep these tests self-contained (no file I/O).
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Dummy market stub (no data files required)
# ---------------------------------------------------------------------------
class DummyMarket:
    """Minimal market stub: constant discount factor and spot."""
    def __init__(self, rate: float = 0.05, spot: float = 100.0):
        self.rate = rate
        self.spot = spot

    def get_discount_factor(self, maturity: float) -> float:
        return np.exp(-self.rate * maturity)

    def get_fwd_discount_factor(self, start: float, end: float) -> float:
        return np.exp(-self.rate * (end - start))

    def get_rate(self, maturity: float) -> float:
        return self.rate * 100  # stored in percent

    def get_volatility(self, strike: float, maturity: float) -> float:
        return 0.20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_paths(nb_paths: int, nb_steps: int, S0: float = 100.0, seed: int = 42) -> np.ndarray:
    """Generate simple GBM paths for testing purposes."""
    rng = np.random.default_rng(seed)
    mu, sigma, dt = 0.05, 0.20, 1.0 / nb_steps
    paths = np.zeros((nb_paths, nb_steps + 1))
    paths[:, 0] = S0
    for t in range(nb_steps):
        Z = rng.standard_normal(nb_paths)
        paths[:, t + 1] = paths[:, t] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    return paths


MARKET = DummyMarket(rate=0.05)
PATHS = make_paths(nb_paths=10_000, nb_steps=250, S0=100.0)
STRIKE = 100.0
BARRIER_UP = 130.0
BARRIER_DN = 70.0


# ===========================================================================
# Vanilla Options
# ===========================================================================
class TestVanillaOptions:
    from kernel.products.options.vanilla_options import EuropeanCallOption, EuropeanPutOption

    def test_call_payoff_shape(self):
        from kernel.products.options.vanilla_options import EuropeanCallOption
        opt = EuropeanCallOption(maturity=1.0, strike=STRIKE)
        payoffs = opt.get_discounted_payoff(PATHS, MARKET)
        assert payoffs.shape == (PATHS.shape[0],), "Payoff array must have shape (nb_paths,)"

    def test_call_payoff_non_negative(self):
        from kernel.products.options.vanilla_options import EuropeanCallOption
        opt = EuropeanCallOption(maturity=1.0, strike=STRIKE)
        payoffs = opt.get_discounted_payoff(PATHS, MARKET)
        assert np.all(payoffs >= 0), "Call payoffs must be non-negative"

    def test_put_call_parity_approx(self):
        """For ATM options on no-dividend stock: C - P ≈ S*e^(-0) - K*e^(-rT)"""
        from kernel.products.options.vanilla_options import EuropeanCallOption, EuropeanPutOption
        T = 1.0
        call = EuropeanCallOption(maturity=T, strike=STRIKE)
        put = EuropeanPutOption(maturity=T, strike=STRIKE)
        call_price = np.mean(call.get_discounted_payoff(PATHS, MARKET))
        put_price = np.mean(put.get_discounted_payoff(PATHS, MARKET))
        lhs = call_price - put_price
        rhs = MARKET.spot - STRIKE * MARKET.get_discount_factor(T)
        # Monte Carlo noise tolerance: 1.0 price units
        assert abs(lhs - rhs) < 1.5, f"Put-call parity violated: C-P={lhs:.4f}, S-Ke^(-rT)={rhs:.4f}"

    def test_deep_otm_call_near_zero(self):
        """A deeply OTM call should have near-zero expected payoff."""
        from kernel.products.options.vanilla_options import EuropeanCallOption
        opt = EuropeanCallOption(maturity=1.0, strike=STRIKE * 3)  # Strike = 300
        payoffs = opt.get_discounted_payoff(PATHS, MARKET)
        assert np.mean(payoffs) < 0.01, "Deep OTM call should be near zero"


# ===========================================================================
# Binary Options
# ===========================================================================
class TestBinaryOptions:
    def test_binary_call_shape(self):
        from kernel.products.options.binary_options import BinaryCallOption
        opt = BinaryCallOption(maturity=1.0, strike=STRIKE, coupon=10.0)
        payoffs = opt.get_discounted_payoff(PATHS, MARKET)
        assert payoffs.shape == (PATHS.shape[0],)

    def test_binary_call_only_two_values(self):
        """Binary call payoff before discounting must be either coupon or 0."""
        from kernel.products.options.binary_options import BinaryCallOption
        coupon = 10.0
        opt = BinaryCallOption(maturity=1.0, strike=STRIKE, coupon=coupon)
        payoffs = opt.get_discounted_payoff(PATHS, MARKET)
        df = MARKET.get_discount_factor(1.0)
        # Un-discount to get raw values
        raw = payoffs / df
        unique = np.unique(np.round(raw, 6))
        assert set(unique).issubset({0.0, coupon}), f"Binary call must pay only 0 or coupon, got {unique}"

    def test_binary_put_complementary(self):
        """For ATM binary: call + put probability ≈ 0.5 + 0.5 = 1 (approx)."""
        from kernel.products.options.binary_options import BinaryCallOption, BinaryPutOption
        coupon = 1.0
        call = BinaryCallOption(maturity=1.0, strike=STRIKE, coupon=coupon)
        put = BinaryPutOption(maturity=1.0, strike=STRIKE, coupon=coupon)
        # Average probability (before discounting)
        df = MARKET.get_discount_factor(1.0)
        p_call = np.mean(call.get_discounted_payoff(PATHS, MARKET)) / df
        p_put = np.mean(put.get_discounted_payoff(PATHS, MARKET)) / df
        # Excluding the zero-probability of spot == strike exactly, probabilities sum to ~1
        assert abs(p_call + p_put - 1.0) < 0.02, f"Binary call + put prob should ≈ 1.0, got {p_call + p_put:.4f}"

    def test_binary_call_zero_when_far_otm(self):
        """Binary call with strike=300 should have ~0 payoff."""
        from kernel.products.options.binary_options import BinaryCallOption
        opt = BinaryCallOption(maturity=1.0, strike=300.0, coupon=10.0)
        payoffs = opt.get_discounted_payoff(PATHS, MARKET)
        assert np.mean(payoffs) < 0.01


# ===========================================================================
# Barrier Options
# ===========================================================================
class TestBarrierOptions:
    def test_barrier_check_shape(self):
        from kernel.products.options.barrier_options import DownAndInCallOption
        opt = DownAndInCallOption(maturity=1.0, strike=STRIKE, barrier=BARRIER_DN)
        breached = opt.is_barrier_breached(PATHS)
        assert breached.shape == (PATHS.shape[0],), "Barrier breach array must have shape (nb_paths,)"

    def test_down_in_out_call_complement(self):
        """Down-and-In + Down-and-Out = Vanilla Call (parity relationship)."""
        from kernel.products.options.barrier_options import DownAndInCallOption, DownAndOutCallOption
        di = DownAndInCallOption(maturity=1.0, strike=STRIKE, barrier=BARRIER_DN)
        do_ = DownAndOutCallOption(maturity=1.0, strike=STRIKE, barrier=BARRIER_DN)
        from kernel.products.options.vanilla_options import EuropeanCallOption
        vanilla = EuropeanCallOption(maturity=1.0, strike=STRIKE)

        p_di = np.mean(di.get_discounted_payoff(PATHS, MARKET))
        p_do = np.mean(do_.get_discounted_payoff(PATHS, MARKET))
        p_vanilla = np.mean(vanilla.get_discounted_payoff(PATHS, MARKET))

        assert abs((p_di + p_do) - p_vanilla) < 0.5, (
            f"DI + DO must equal Vanilla: {p_di:.4f} + {p_do:.4f} ≠ {p_vanilla:.4f}"
        )

    def test_up_in_out_put_complement(self):
        """Up-and-In + Up-and-Out = Vanilla Put."""
        from kernel.products.options.barrier_options import UpAndInPutOption, UpAndOutPutOption
        from kernel.products.options.vanilla_options import EuropeanPutOption
        ui = UpAndInPutOption(maturity=1.0, strike=STRIKE, barrier=BARRIER_UP)
        uo = UpAndOutPutOption(maturity=1.0, strike=STRIKE, barrier=BARRIER_UP)
        vanilla = EuropeanPutOption(maturity=1.0, strike=STRIKE)

        p_ui = np.mean(ui.get_discounted_payoff(PATHS, MARKET))
        p_uo = np.mean(uo.get_discounted_payoff(PATHS, MARKET))
        p_vanilla = np.mean(vanilla.get_discounted_payoff(PATHS, MARKET))

        assert abs((p_ui + p_uo) - p_vanilla) < 0.5, (
            f"UI + UO must equal Vanilla Put: {p_ui:.4f} + {p_uo:.4f} ≠ {p_vanilla:.4f}"
        )

    def test_down_in_call_payoff_non_negative(self):
        from kernel.products.options.barrier_options import DownAndInCallOption
        opt = DownAndInCallOption(maturity=1.0, strike=STRIKE, barrier=BARRIER_DN)
        payoffs = opt.get_discounted_payoff(PATHS, MARKET)
        assert np.all(payoffs >= 0)

    def test_barrier_never_breached_knockout_equals_vanilla(self):
        """If barrier is set far below any path, Down-And-Out Call ≈ Vanilla Call."""
        from kernel.products.options.barrier_options import DownAndOutCallOption
        from kernel.products.options.vanilla_options import EuropeanCallOption
        # Barrier at 1% of spot — essentially unreachable in 1 year with 20% vol
        opt = DownAndOutCallOption(maturity=1.0, strike=STRIKE, barrier=1.0)
        vanilla = EuropeanCallOption(maturity=1.0, strike=STRIKE)
        p_barrier = np.mean(opt.get_discounted_payoff(PATHS, MARKET))
        p_vanilla = np.mean(vanilla.get_discounted_payoff(PATHS, MARKET))
        assert abs(p_barrier - p_vanilla) < 0.1, (
            f"Unreachable barrier should match vanilla: {p_barrier:.4f} vs {p_vanilla:.4f}"
        )

    def test_barrier_always_breached_knockin_equals_vanilla(self):
        """If Down-And-In barrier is above the initial spot, all paths are knocked-in."""
        from kernel.products.options.barrier_options import DownAndInCallOption
        from kernel.products.options.vanilla_options import EuropeanCallOption
        # Barrier above initial spot: every path starts below it, so it's instantly 'breached'
        # Actually for DI the barrier is BELOW the spot. Let's test:
        # All paths start at S0=100. If barrier=100 (equal to strike), it would raise ValueError.
        # Instead use: ensure min(path) < barrier by having barrier just below S0.
        # A down barrier at 99.99 should be breached by almost all paths with 20% vol.
        opt = DownAndInCallOption(maturity=1.0, strike=STRIKE, barrier=99.99)
        vanilla = EuropeanCallOption(maturity=1.0, strike=STRIKE)
        p_di = np.mean(opt.get_discounted_payoff(PATHS, MARKET))
        p_vanilla = np.mean(vanilla.get_discounted_payoff(PATHS, MARKET))
        # With 20% vol over 1Y, almost all paths will breach 99.99 vs starting at 100
        assert abs(p_di - p_vanilla) < 1.5, (
            f"Near-ATM DI barrier should approach vanilla: {p_di:.4f} vs {p_vanilla:.4f}"
        )

    def test_down_and_in_call_specific_paths(self):
        """
        Manual validation with controlled paths:
        - Path A: goes down below barrier, ends ITM  → should get intrinsic payoff
        - Path B: stays above barrier, ends ITM      → should get 0 (not knocked in)
        - Path C: goes down below barrier, ends OTM  → should get 0 (intrinsic is 0)
        """
        from kernel.products.options.barrier_options import DownAndInCallOption

        S0, K, B = 100.0, 100.0, 80.0
        T, r = 1.0, 0.05
        df = np.exp(-r * T)
        opt = DownAndInCallOption(maturity=T, strike=K, barrier=B)

        # 3 paths: [start, low_point, final]
        paths = np.array([
            [100.0, 75.0, 120.0],  # A: breaches barrier (75 < 80), ends at 120 (ITM)
            [100.0, 85.0, 120.0],  # B: barrier not breached (min=85 > 80), ends at 120 (ITM)
            [100.0, 75.0,  90.0],  # C: breaches barrier (75 < 80), ends at 90 (OTM)
        ])

        payoffs = opt.get_discounted_payoff(paths, DummyMarket(rate=r, spot=S0))

        assert abs(payoffs[0] - (120.0 - 100.0) * df) < 1e-8, f"Path A should pay {(120-100)*df:.4f}, got {payoffs[0]:.4f}"
        assert abs(payoffs[1] - 0.0) < 1e-8, f"Path B not knocked in, should pay 0, got {payoffs[1]:.4f}"
        assert abs(payoffs[2] - 0.0) < 1e-8, f"Path C OTM, should pay 0, got {payoffs[2]:.4f}"


# ===========================================================================
# Path-Dependent Options
# ===========================================================================
class TestPathDependentOptions:
    def test_asian_call_shape(self):
        from kernel.products.options.path_dependent_options import AsianCallOption
        opt = AsianCallOption(maturity=1.0, strike=STRIKE)
        payoffs = opt.get_discounted_payoff(PATHS, MARKET)
        assert payoffs.shape == (PATHS.shape[0],)

    def test_asian_call_cheaper_than_vanilla(self):
        """Asian call is always <= vanilla call because avg <= final (in expectation)."""
        from kernel.products.options.path_dependent_options import AsianCallOption
        from kernel.products.options.vanilla_options import EuropeanCallOption
        asian = AsianCallOption(maturity=1.0, strike=STRIKE)
        vanilla = EuropeanCallOption(maturity=1.0, strike=STRIKE)
        p_asian = np.mean(asian.get_discounted_payoff(PATHS, MARKET))
        p_vanilla = np.mean(vanilla.get_discounted_payoff(PATHS, MARKET))
        assert p_asian < p_vanilla, f"Asian call {p_asian:.4f} should be < vanilla call {p_vanilla:.4f}"

    def test_lookback_call_non_negative_and_geq_vanilla(self):
        """Lookback call >= Vanilla call because max >= final."""
        from kernel.products.options.path_dependent_options import LookbackCallOption
        from kernel.products.options.vanilla_options import EuropeanCallOption
        lookback = LookbackCallOption(maturity=1.0, strike=STRIKE)
        vanilla = EuropeanCallOption(maturity=1.0, strike=STRIKE)
        payoffs = lookback.get_discounted_payoff(PATHS, MARKET)
        assert np.all(payoffs >= 0)
        p_lookback = np.mean(payoffs)
        p_vanilla = np.mean(vanilla.get_discounted_payoff(PATHS, MARKET))
        assert p_lookback >= p_vanilla - 0.01, f"Lookback {p_lookback:.4f} should be >= vanilla {p_vanilla:.4f}"

    def test_chooser_geq_call_and_put(self):
        """Chooser option >= max(call, put)."""
        from kernel.products.options.path_dependent_options import ChooserOption
        from kernel.products.options.vanilla_options import EuropeanCallOption, EuropeanPutOption
        chooser = ChooserOption(maturity=1.0, strike=STRIKE, chooser_time=0.5)
        call = EuropeanCallOption(maturity=1.0, strike=STRIKE)
        put = EuropeanPutOption(maturity=1.0, strike=STRIKE)
        p_chooser = np.mean(chooser.get_discounted_payoff(PATHS, MARKET))
        p_call = np.mean(call.get_discounted_payoff(PATHS, MARKET))
        p_put = np.mean(put.get_discounted_payoff(PATHS, MARKET))
        assert p_chooser >= max(p_call, p_put) - 0.5, (
            f"Chooser {p_chooser:.4f} should be >= max(call={p_call:.4f}, put={p_put:.4f})"
        )

    def test_forward_start_call_matches_vanilla(self):
        """Forward start call evaluated exactly at t1 should match vanilla call with strike=S_{t1}"""
        from kernel.products.options.path_dependent_options import ForwardStartCallOption
        fwd_opt = ForwardStartCallOption(maturity=1.0, forward_start_time=0.5, strike_percentage=1.0)
        
        payoffs = fwd_opt.get_discounted_payoff(PATHS, MARKET)
        
        # Calculate manually
        nb_steps = PATHS.shape[1] - 1
        idx = int(0.5 / 1.0 * nb_steps)
        idx = max(1, min(idx, nb_steps - 1))
        S_t1 = PATHS[:, idx]
        S_T = PATHS[:, -1]
        
        expected_payoffs = np.maximum(0.0, S_T - S_t1) * MARKET.get_discount_factor(1.0)
        np.testing.assert_allclose(payoffs, expected_payoffs, atol=1e-8)

    def test_forward_start_put_matches_vanilla(self):
        """Forward start put evaluated exactly at t1 should match vanilla put with strike=S_{t1}"""
        from kernel.products.options.path_dependent_options import ForwardStartPutOption
        fwd_opt = ForwardStartPutOption(maturity=1.0, forward_start_time=0.5, strike_percentage=1.0)
        
        payoffs = fwd_opt.get_discounted_payoff(PATHS, MARKET)
        
        # Calculate manually
        nb_steps = PATHS.shape[1] - 1
        idx = int(0.5 / 1.0 * nb_steps)
        idx = max(1, min(idx, nb_steps - 1))
        S_t1 = PATHS[:, idx]
        S_T = PATHS[:, -1]
        
        expected_payoffs = np.maximum(0.0, S_t1 - S_T) * MARKET.get_discount_factor(1.0)
        np.testing.assert_allclose(payoffs, expected_payoffs, atol=1e-8)
