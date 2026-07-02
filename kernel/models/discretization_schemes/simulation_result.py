from dataclasses import dataclass
import numpy as np
from typing import Optional

@dataclass
class SimulationResult:
    spot_paths: np.ndarray
    variance_paths: Optional[np.ndarray] = None
