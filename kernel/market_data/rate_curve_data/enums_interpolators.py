from enum import Enum
from .interpolators import LinearInterpolator, CubicInterpolator, NelsonSiegelInterpolator, SvenssonInterpolator

class InterpolationType(Enum):
    LINEAR = LinearInterpolator
    CUBIC = CubicInterpolator
    NELSON_SIEGEL = NelsonSiegelInterpolator
    SVENSSON = SvenssonInterpolator