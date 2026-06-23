from django.apps import AppConfig


class PricingApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'pricing_api'

    def ready(self):
        # Initialize the global market cache so the cold start happens
        # when the container boots, not during the first user request.
        from pricing_api.market_cache import MarketCache
        MarketCache()
