from datetime import datetime
from typing import Optional
#from kernel.models.pricing_engines.enum_pricing_engine import PricingEngineTypeBis
from kernel.tools import RateCurveType, ObservationFrequency, Model
from kernel.market_data import InterpolationType, VolatilitySurfaceType

class PricingSettings:
    def __init__(
        self,
        underlying_name: Optional[str] = None,
        rate_curve_type: Optional[RateCurveType] = None,
        interpolation_type: Optional[InterpolationType] = None,
        volatility_surface_type: Optional[VolatilitySurfaceType] = None,
        obs_frequency: Optional[ObservationFrequency] = None,
        day_count_convention: Optional[str] = None,
        model: Optional[Model] = None,
        pricing_engine_type: Optional[str] = None,
        nb_paths: Optional[int] = None,
        nb_steps: Optional[int] = None,
        random_seed: Optional[int] = 4012,
        compute_greeks: bool = False,
        valuation_date: Optional[datetime] = None,
        compute_callable_coupons: bool = False,
        random_generator_type: str = "NUMPY",
    ):
        self.underlying_name = underlying_name
        self.rate_curve_type = rate_curve_type
        self.interpolation_type = interpolation_type
        self.volatility_surface_type = volatility_surface_type
        self.obs_frequency = obs_frequency
        self.day_count_convention = day_count_convention
        self.model = model
        self.pricing_engine_type = pricing_engine_type
        self.nb_paths = nb_paths
        self.nb_steps = nb_steps
        self.random_seed = random_seed
        self.compute_greeks = compute_greeks
        self.valuation_date = valuation_date
        self.compute_callable_coupons = compute_callable_coupons
        self.random_generator_type = random_generator_type

