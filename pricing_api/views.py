import json
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from kernel.tools import ObservationFrequency, Model, CalendarConvention, RateCurveType
from kernel.market_data.volatility_surface.enums_volatility import VolatilitySurfaceType
from kernel.models.pricing_engines.enum_pricing_engine import PricingEngineType
from kernel.pricing_launcher import PricingLauncher
from utils.pricing_settings import PricingSettings

# Import all products to allow dynamic evaluation
from kernel.products.options.vanilla_options import EuropeanCallOption, EuropeanPutOption
from kernel.products.options.american_options import AmericanCallOption, AmericanPutOption
from kernel.products.options.barrier_options import DownAndInCallOption, DownAndOutCallOption, UpAndInPutOption, UpAndOutPutOption
from kernel.products.options.binary_options import AssetOrNothingCallOption, AssetOrNothingPutOption, CashOrNothingCallOption, CashOrNothingPutOption
from kernel.products.options.path_dependent_options import AsianCallOption, AsianPutOption, LookbackCallOption, LookbackPutOption, ChooserOption
from kernel.products.structured_products.autocall_products import Phoenix
from kernel.products.structured_products.reverse_convertible import ReverseConvertible
from kernel.products.rate.bond import ZeroCouponBond, CouponBond
from kernel.products.rate.vanilla_swap import InterestRateSwap

from pricing_api.market_cache import MarketCache

class PriceEndpoint(APIView):
    def post(self, request):
        try:
            data = request.data
            engine_settings_data = data.get('engine_settings', {})
            product_details_data = data.get('product_details', {})
            
            if not product_details_data:
                return Response({"error": "Missing product_details in payload"}, status=status.HTTP_400_BAD_REQUEST)
                
            # 1. Parse Settings
            settings = PricingSettings()
            
            # Map enums from string
            model_str = engine_settings_data.get('model', 'BLACK_SCHOLES')
            settings.model = getattr(Model, model_str, Model.BLACK_SCHOLES)
            
            vol_str = engine_settings_data.get('volatility_surface_type', 'SVI')
            settings.volatility_surface_type = getattr(VolatilitySurfaceType, vol_str, VolatilitySurfaceType.SVI)
            
            engine_str = engine_settings_data.get('pricing_engine_type', 'MC')
            settings.pricing_engine_type = getattr(PricingEngineType, engine_str, PricingEngineType.MC)
            
            obs_str = engine_settings_data.get('obs_frequency', 'ANNUAL')
            settings.obs_frequency = getattr(ObservationFrequency, obs_str, ObservationFrequency.ANNUAL)
            
            settings.compute_greeks = engine_settings_data.get('compute_greeks', False)
            settings.nb_paths = engine_settings_data.get('nb_paths', 10000)
            settings.nb_steps = engine_settings_data.get('nb_steps', 100)
            
            # 2. Parse Product
            product_type_str = product_details_data.get('type')
            if not product_type_str:
                return Response({"error": "Missing 'type' in product_details"}, status=status.HTTP_400_BAD_REQUEST)
                
            # Remove 'type' before instantiating the class
            params = {k: v for k, v in product_details_data.items() if k != 'type'}
            
            # Helper to map ObservationFrequency strings inside product params
            if 'observation_frequency' in params and isinstance(params['observation_frequency'], str):
                params['observation_frequency'] = getattr(ObservationFrequency, params['observation_frequency'], ObservationFrequency.ANNUAL)
                
            try:
                product_class = globals()[product_type_str]
                product = product_class(**params)
            except KeyError:
                return Response({"error": f"Unknown product type: {product_type_str}"}, status=status.HTTP_400_BAD_REQUEST)
            except Exception as e:
                return Response({"error": f"Error instantiating product {product_type_str}: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)

            # 3. Retrieve Cached Market
            cache = MarketCache()
            market = cache.get_market(settings.volatility_surface_type)
            if not market:
                return Response({"error": f"Market data not cached for volatility surface: {settings.volatility_surface_type}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # 4. Execute Pricing
            launcher = PricingLauncher(pricing_settings=settings, market=market)
            result = launcher.calculate(product)
            
            # 5. Build Response
            response_data = {
                "price": result.price,
                "greeks": result.greeks if result.greeks else {}
            }
            if result.std_dev is not None:
                response_data["std_dev"] = result.std_dev
                response_data["confidence_interval"] = {
                    "lower": result.lower_bound,
                    "upper": result.upper_bound,
                    "level": result.confidence_level
                }
                
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response({"error": f"An unexpected error occurred: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
