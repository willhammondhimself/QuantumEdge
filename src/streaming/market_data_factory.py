"""
Market Data Factory with multi-provider support and fallback chain.

This module provides a factory for creating market data sources with
automatic fallback between providers, health monitoring, and caching.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum

from ..api.config import settings
from .market_data_source import MarketDataSource
from .yahoo_finance_source import YahooFinanceDataSource

logger = logging.getLogger(__name__)

try:
    from .alpha_vantage_source import AlphaVantageDataSource

    ALPHA_VANTAGE_AVAILABLE = True
except ImportError:
    ALPHA_VANTAGE_AVAILABLE = False
    AlphaVantageDataSource = None


class ProviderType(str, Enum):
    """Supported market data providers."""

    YAHOO = "yahoo"
    ALPHA_VANTAGE = "alpha_vantage"
    IEX = "iex"
    POLYGON = "polygon"
    FINNHUB = "finnhub"


class MarketDataFactory:
    """Factory for creating and managing market data sources."""

    _instances: Dict[str, MarketDataSource] = {}
    _health_status: Dict[str, Dict] = {}

    @classmethod
    def create_provider(
        cls, provider_type: ProviderType, **kwargs
    ) -> Optional[MarketDataSource]:
        """
        Create a market data provider instance.

        Args:
            provider_type: Type of provider to create
            **kwargs: Provider-specific configuration

        Returns:
            MarketDataSource instance or None if unavailable
        """
        if provider_type == ProviderType.YAHOO:
            return YahooFinanceDataSource(**kwargs)

        elif provider_type == ProviderType.ALPHA_VANTAGE:
            if not ALPHA_VANTAGE_AVAILABLE:
                logger.warning("Alpha Vantage provider not available")
                return None

            api_key = kwargs.get("api_key") or settings.alpha_vantage_api_key
            if not api_key:
                logger.warning("Alpha Vantage API key not configured")
                return None

            return AlphaVantageDataSource(api_key=api_key, **kwargs)

        elif provider_type == ProviderType.IEX:
            logger.warning("IEX provider not yet implemented")
            return None

        elif provider_type == ProviderType.POLYGON:
            logger.warning("Polygon provider not yet implemented")
            return None

        elif provider_type == ProviderType.FINNHUB:
            logger.warning("Finnhub provider not yet implemented")
            return None

        else:
            logger.error(f"Unknown provider type: {provider_type}")
            return None

    @classmethod
    def get_or_create_provider(
        cls, provider_type: ProviderType, **kwargs
    ) -> Optional[MarketDataSource]:
        """Get existing provider instance or create new one."""
        provider_key = f"{provider_type}_{hash(frozenset(kwargs.items()))}"

        if provider_key not in cls._instances:
            provider = cls.create_provider(provider_type, **kwargs)
            if provider:
                cls._instances[provider_key] = provider
            return provider

        return cls._instances[provider_key]

    @classmethod
    async def get_health_status(
        cls, provider_type: ProviderType, **kwargs
    ) -> Dict[str, Any]:
        """Get health status for a provider."""
        provider = cls.get_or_create_provider(provider_type, **kwargs)
        if not provider:
            return {
                "provider": provider_type,
                "healthy": False,
                "error": "Provider not available or not configured",
                "timestamp": datetime.now().isoformat(),
            }

        try:
            if hasattr(provider, "get_health_status"):
                return provider.get_health_status()
            else:
                # Basic health check for providers without detailed health monitoring
                return {
                    "provider": provider_type,
                    "healthy": True,
                    "status": "basic_check",
                    "timestamp": datetime.now().isoformat(),
                }
        except Exception as e:
            logger.error(f"Health check failed for {provider_type}: {e}")
            return {
                "provider": provider_type,
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }


class FallbackMarketDataSource(MarketDataSource):
    """
    Market data source with automatic fallback between providers.

    Attempts to fetch data from primary provider, falls back to
    secondary providers if primary fails.
    """

    def __init__(
        self,
        primary_provider: ProviderType = ProviderType.YAHOO,
        fallback_providers: List[ProviderType] = None,
        **provider_kwargs,
    ):
        """
        Initialize fallback market data source.

        Args:
            primary_provider: Primary provider to use
            fallback_providers: List of fallback providers
            **provider_kwargs: Keyword arguments for provider creation
        """
        self.primary_provider = primary_provider
        self.fallback_providers = fallback_providers or [ProviderType.ALPHA_VANTAGE]
        self.provider_kwargs = provider_kwargs

        # Create provider instances
        self.providers = {}
        self._create_providers()

        # Health monitoring
        self.provider_health = {}
        self.last_health_check = {}

        self._running = False

    def _create_providers(self):
        """Create all configured provider instances."""
        all_providers = [self.primary_provider] + self.fallback_providers

        for provider_type in all_providers:
            try:
                provider = MarketDataFactory.create_provider(
                    provider_type, **self.provider_kwargs
                )
                if provider:
                    self.providers[provider_type] = provider
                    logger.info(f"Created {provider_type} provider")
            except Exception as e:
                logger.error(f"Failed to create {provider_type} provider: {e}")

    async def start(self):
        """Start all providers."""
        self._running = True

        for provider_type, provider in self.providers.items():
            try:
                await provider.start()
                logger.info(f"Started {provider_type} provider")
            except Exception as e:
                logger.error(f"Failed to start {provider_type} provider: {e}")

    async def stop(self):
        """Stop all providers."""
        self._running = False

        for provider_type, provider in self.providers.items():
            try:
                await provider.stop()
            except Exception as e:
                logger.error(f"Failed to stop {provider_type} provider: {e}")

    async def _try_providers(self, method_name: str, *args, **kwargs):
        """Try method on providers in order until one succeeds."""
        providers_to_try = [self.primary_provider] + self.fallback_providers

        for provider_type in providers_to_try:
            if provider_type not in self.providers:
                continue

            provider = self.providers[provider_type]

            try:
                method = getattr(provider, method_name)
                result = await method(*args, **kwargs)

                if result is not None:
                    logger.debug(f"Successfully got data from {provider_type}")
                    return result
                else:
                    logger.warning(f"{provider_type} returned None for {method_name}")

            except Exception as e:
                logger.error(f"{provider_type} failed for {method_name}: {e}")
                continue

        logger.error(f"All providers failed for {method_name}")
        return None

    async def fetch_real_price(self, symbol: str) -> Optional[Dict]:
        """Fetch real-time price with fallback."""
        return await self._try_providers("fetch_real_price", symbol)

    async def get_market_data(self) -> List[Dict]:
        """Get market data with fallback."""
        result = await self._try_providers("get_market_data")
        return result or []

    async def get_historical_data(self, symbol: str, period: str = "1y") -> List[Dict]:
        """Get historical data with fallback."""
        result = await self._try_providers("get_historical_data", symbol, period)
        return result or []

    async def get_health_status_all(self) -> Dict[str, Any]:
        """Get health status for all providers."""
        health_status = {}

        for provider_type, provider in self.providers.items():
            try:
                if hasattr(provider, "get_health_status"):
                    health_status[provider_type.value] = provider.get_health_status()
                else:
                    health_status[provider_type.value] = {
                        "provider": provider_type.value,
                        "healthy": True,
                        "status": "basic_check",
                        "timestamp": datetime.now().isoformat(),
                    }
            except Exception as e:
                health_status[provider_type.value] = {
                    "provider": provider_type.value,
                    "healthy": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }

        return {
            "providers": health_status,
            "primary_provider": self.primary_provider.value,
            "fallback_providers": [p.value for p in self.fallback_providers],
            "overall_healthy": any(
                h.get("healthy", False) for h in health_status.values()
            ),
            "timestamp": datetime.now().isoformat(),
        }


def create_market_data_source_with_fallback(
    use_simulation: bool = False,
    primary_provider: str = None,
    fallback_providers: List[str] = None,
) -> Union[MarketDataSource, FallbackMarketDataSource]:
    """
    Create market data source with fallback support.

    Args:
        use_simulation: Whether to use simulated data (for backwards compatibility)
        primary_provider: Primary provider name
        fallback_providers: List of fallback provider names

    Returns:
        Market data source instance
    """
    if use_simulation:
        # For backwards compatibility, return simulated source
        from .market_data_source import SimulatedMarketDataSource

        return SimulatedMarketDataSource()

    # Use configuration or defaults
    primary = primary_provider or settings.market_data_provider
    fallbacks = fallback_providers or settings.market_data_fallback_providers

    try:
        primary_type = ProviderType(primary)
    except ValueError:
        logger.warning(f"Unknown primary provider '{primary}', using Yahoo Finance")
        primary_type = ProviderType.YAHOO

    fallback_types = []
    for fallback in fallbacks:
        try:
            fallback_types.append(ProviderType(fallback))
        except ValueError:
            logger.warning(f"Unknown fallback provider '{fallback}', skipping")

    # If only one provider configured, use it directly
    if not fallback_types:
        return MarketDataFactory.get_or_create_provider(primary_type)

    # Use fallback source for multiple providers
    return FallbackMarketDataSource(
        primary_provider=primary_type, fallback_providers=fallback_types
    )
