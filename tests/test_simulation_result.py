import numpy as np
from kernel.models.discretization_schemes.simulation_result import SimulationResult

def test_simulation_result_spot_only():
    spot_paths = np.array([[100, 105], [100, 95]])
    res = SimulationResult(spot_paths=spot_paths)
    assert np.array_equal(res.spot_paths, spot_paths)
    assert res.variance_paths is None

def test_simulation_result_spot_and_variance():
    spot_paths = np.array([[100, 105], [100, 95]])
    var_paths = np.array([[0.04, 0.05], [0.04, 0.03]])
    res = SimulationResult(spot_paths=spot_paths, variance_paths=var_paths)
    assert np.array_equal(res.spot_paths, spot_paths)
    assert np.array_equal(res.variance_paths, var_paths)
