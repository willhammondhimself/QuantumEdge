"""
Configuration management for QuantumEdge API.
"""

import os
from typing import List, Optional
from pydantic import Field

try:
    from pydantic_settings import BaseSettings
except ImportError:
    try:
        from pydantic import BaseSettings
    except ImportError:
        # Fallback for environments without pydantic
        class BaseSettings:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)


class Settings(BaseSettings):
    """Application settings."""

    # App metadata
    app_name: str = "QuantumEdge"
    app_version: str = "0.1.0"
    app_description: str = (
        "Quantum-inspired portfolio optimization with crisis-proof robustness"
    )

    # Environment
    environment: str = Field(
        default="production", env="ENVIRONMENT"
    )  # Changed to production for real data
    debug: bool = Field(default=False, env="DEBUG")  # Disabled debug for production

    # API settings
    api_prefix: str = "/api/v1"
    docs_url: Optional[str] = "/docs"
    redoc_url: Optional[str] = "/redoc"

    # CORS settings
    allowed_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        env="ALLOWED_ORIGINS",
    )
    allowed_methods: List[str] = ["GET", "POST", "PUT", "DELETE"]
    allowed_headers: List[str] = ["*"]

    # Database settings
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    postgres_url: str = Field(
        default="postgresql://quantumedge:quantumedge123@localhost:5432/quantumedge",
        env="POSTGRES_URL",
    )

    # External services
    influxdb_url: str = Field(default="http://localhost:8086", env="INFLUXDB_URL")
    influxdb_token: str = Field(default="quantumedge-token", env="INFLUXDB_TOKEN")
    influxdb_org: str = Field(default="quantumedge", env="INFLUXDB_ORG")
    influxdb_bucket: str = Field(default="metrics", env="INFLUXDB_BUCKET")

    kafka_bootstrap_servers: str = Field(
        default="localhost:9092", env="KAFKA_BOOTSTRAP_SERVERS"
    )

    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Optimization settings
    max_optimization_time: int = Field(
        default=300, description="Max optimization time in seconds"
    )
    max_concurrent_optimizations: int = Field(
        default=5, description="Max concurrent optimizations"
    )
    default_risk_free_rate: float = Field(
        default=0.02, description="Default risk-free rate"
    )

    # Market Data Configuration
    market_data_provider: str = Field(
        default="yahoo", env="MARKET_DATA_PROVIDER"
    )  # yahoo, alpha_vantage, iex, polygon
    market_data_cache_ttl: int = Field(
        default=300, env="MARKET_DATA_CACHE_TTL"
    )  # Cache TTL in seconds
    market_data_rate_limit: int = Field(
        default=120, env="MARKET_DATA_RATE_LIMIT"
    )  # Requests per minute
    market_data_timeout: int = Field(
        default=30, env="MARKET_DATA_TIMEOUT"
    )  # Request timeout in seconds
    market_data_retry_attempts: int = Field(default=3, env="MARKET_DATA_RETRY_ATTEMPTS")
    market_data_retry_delay: float = Field(
        default=1.0, env="MARKET_DATA_RETRY_DELAY"
    )  # Base delay in seconds

    # Market Data Provider API Keys
    alpha_vantage_api_key: Optional[str] = Field(
        default=None, env="ALPHA_VANTAGE_API_KEY"
    )
    iex_cloud_api_key: Optional[str] = Field(default=None, env="IEX_CLOUD_API_KEY")
    polygon_api_key: Optional[str] = Field(default=None, env="POLYGON_API_KEY")
    finnhub_api_key: Optional[str] = Field(default=None, env="FINNHUB_API_KEY")

    # Market Data Fallback Chain
    market_data_fallback_providers: List[str] = Field(
        default=["yahoo", "alpha_vantage", "iex"], env="MARKET_DATA_FALLBACK_PROVIDERS"
    )

    # Market Data Health Monitoring
    market_data_health_check_interval: int = Field(
        default=300, env="MARKET_DATA_HEALTH_CHECK_INTERVAL"
    )  # seconds
    market_data_failure_threshold: int = Field(
        default=5, env="MARKET_DATA_FAILURE_THRESHOLD"
    )  # consecutive failures
    market_data_circuit_breaker_timeout: int = Field(
        default=900, env="MARKET_DATA_CIRCUIT_BREAKER_TIMEOUT"
    )  # seconds

    # Security
    secret_key: str = Field(
        default="dev-secret-key-change-in-production", env="SECRET_KEY"
    )
    access_token_expire_minutes: int = 30

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
