from enum import Enum
from .local_surface import LocalVolatilitySurface
from .svi_surface import SVIVolatilitySurface
from .ssvi_surface import SSVIVolatilitySurface

class VolatilitySurfaceType(Enum):
    LOCAL = LocalVolatilitySurface
    SVI = SVIVolatilitySurface
    SSVI = SSVIVolatilitySurface

