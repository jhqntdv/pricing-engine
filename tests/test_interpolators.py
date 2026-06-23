import pytest
import numpy as np

from kernel.market_data.rate_curve_data.interpolators.abstract_interpolator import CalibrationError
from kernel.market_data.rate_curve_data.interpolators.nelson_siegel_interpolator import NelsonSiegelInterpolator
from kernel.market_data.rate_curve_data.interpolators.svensson_interpolator import SvenssonInterpolator
from kernel.market_data.rate_curve_data.interpolators.linear_interpolator import LinearInterpolator
from kernel.market_data.rate_curve_data.interpolators.cubic_interpolator import CubicInterpolator

class TestNelsonSiegelInterpolator:
    def test_mathematical_round_trip(self):
        maturities = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
        # t, beta0, beta1, beta2, tau
        rates = np.array([NelsonSiegelInterpolator._nelson_siegel(t, 0.03, -0.02, 0.01, 1.5) for t in maturities])
        
        interp = NelsonSiegelInterpolator(maturities, rates)
        interp.calibrate()
        
        np.testing.assert_allclose(interp.params, [0.03, -0.02, 0.01, 1.5], atol=1e-5)

    def test_asymptotic_behavior(self):
        maturities = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
        rates = np.array([NelsonSiegelInterpolator._nelson_siegel(t, 0.03, -0.02, 0.01, 1.5) for t in maturities])
        
        interp = NelsonSiegelInterpolator(maturities, rates)
        interp.calibrate()
        
        # As t -> 0, yield should converge to beta0 + beta1 = 0.01
        assert interp.interpolate(1e-8) == pytest.approx(0.01, abs=1e-4)
        
        # As t -> infinity, yield should converge to beta0 = 0.03
        assert interp.interpolate(1000.0) == pytest.approx(0.03, abs=1e-4)

    def test_real_world_fitting(self):
        maturities = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
        rates = np.array([0.01, 0.015, 0.02, 0.025, 0.03, 0.032, 0.035])
        
        interp = NelsonSiegelInterpolator(maturities, rates)
        interp.calibrate()
        
        fitted_rates = np.array([interp.interpolate(t) for t in maturities])
        rmse = np.sqrt(np.mean((rates - fitted_rates) ** 2))
        assert rmse < 1e-3
        
    def test_negative_interest_rates(self):
        maturities = np.array([1.0, 2.0, 5.0, 10.0])
        rates = np.array([-0.005, -0.003, -0.001, 0.005])
        
        interp = NelsonSiegelInterpolator(maturities, rates)
        interp.calibrate()
        
        rate_3y = interp.interpolate(3.0)
        assert isinstance(rate_3y, float)
        assert -0.005 <= rate_3y <= 0.005


class TestSvenssonInterpolator:
    def test_mathematical_round_trip(self):
        maturities = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
        rates = np.array([SvenssonInterpolator._svensson(t, 0.025, -0.015, 0.01, 0.005, 1.2, 2.5) for t in maturities])
        
        interp = SvenssonInterpolator(maturities, rates)
        interp.calibrate()
        
        np.testing.assert_allclose(interp.params, [0.025, -0.015, 0.01, 0.005, 1.2, 2.5], atol=1e-5)

    def test_asymptotic_behavior(self):
        maturities = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
        rates = np.array([SvenssonInterpolator._svensson(t, 0.025, -0.015, 0.01, 0.005, 1.2, 2.5) for t in maturities])
        
        interp = SvenssonInterpolator(maturities, rates)
        interp.calibrate()
        
        assert interp.interpolate(1e-8) == pytest.approx(0.01, abs=1e-4)
        assert interp.interpolate(1000.0) == pytest.approx(0.025, abs=1e-4)

    def test_real_world_fitting(self):
        maturities = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
        rates = np.array([0.01, 0.015, 0.02, 0.025, 0.03, 0.032, 0.035])
        
        interp = SvenssonInterpolator(maturities, rates)
        interp.calibrate()
        
        fitted_rates = np.array([interp.interpolate(t) for t in maturities])
        rmse = np.sqrt(np.mean((rates - fitted_rates) ** 2))
        assert rmse < 1e-3


class TestLinearCubicInterpolators:
    def test_exact_node_matching(self):
        maturities = np.array([1.0, 2.0, 5.0, 10.0])
        rates = np.array([0.01, 0.02, 0.03, 0.04])
        
        linear_interp = LinearInterpolator(maturities, rates)
        linear_interp.calibrate()
        
        cubic_interp = CubicInterpolator(maturities, rates)
        cubic_interp.calibrate()
        
        for t, r in zip(maturities, rates):
            assert linear_interp.interpolate(t) == pytest.approx(r)
            assert cubic_interp.interpolate(t) == pytest.approx(r)

    def test_extrapolation_behavior(self):
        maturities = np.array([1.0, 2.0, 5.0, 10.0])
        rates = np.array([0.01, 0.02, 0.03, 0.04])
        
        linear_interp = LinearInterpolator(maturities, rates)
        linear_interp.calibrate()
        
        cubic_interp = CubicInterpolator(maturities, rates)
        cubic_interp.calibrate()
        
        # Test extrapolation below min maturity
        assert isinstance(linear_interp.interpolate(0.5), float)
        assert isinstance(cubic_interp.interpolate(0.5), float)
        
        # Test extrapolation above max maturity
        assert isinstance(linear_interp.interpolate(15.0), float)
        assert isinstance(cubic_interp.interpolate(15.0), float)


class TestInterpolatorEdgeCases:
    def test_uncalibrated_access_raises_error(self):
        maturities = np.array([1.0, 2.0])
        rates = np.array([0.01, 0.02])
        
        interps = [
            NelsonSiegelInterpolator(maturities, rates),
            SvenssonInterpolator(maturities, rates),
            LinearInterpolator(maturities, rates),
            CubicInterpolator(maturities, rates)
        ]
        
        for interp in interps:
            with pytest.raises(ValueError, match="calibrated|calibrate"):
                interp.interpolate(1.5)

    def test_insufficient_data_points(self):
        maturities = np.array([1.0, 2.0])
        rates = np.array([0.01, 0.02])
        
        ns = NelsonSiegelInterpolator(maturities, rates)
        with pytest.raises(CalibrationError):
            ns.calibrate()
            
        sv = SvenssonInterpolator(maturities, rates)
        with pytest.raises(CalibrationError):
            sv.calibrate()
