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
    app_description: str = "Quantum-inspired portfolio optimization with crisis-proof robustness"
    
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")
    
    # API settings
    api_prefix: str = "/api/v1"
    docs_url: Optional[str] = "/docs"
    redoc_url: Optional[str] = "/redoc"
    
    # CORS settings
    allowed_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        env="ALLOWED_ORIGINS"
    )
    allowed_methods: List[str] = ["GET", "POST", "PUT", "DELETE"]
    allowed_headers: List[str] = ["*"]
    
    # Database settings
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    postgres_url: str = Field(
        default="postgresql://quantumedge:quantumedge123@localhost:5432/quantumedge",
        env="POSTGRES_URL"
    )
    
    # External services
    influxdb_url: str = Field(default="http://localhost:8086", env="INFLUXDB_URL")
    influxdb_token: str = Field(default="quantumedge-token", env="INFLUXDB_TOKEN")
    influxdb_org: str = Field(default="quantumedge", env="INFLUXDB_ORG")
    influxdb_bucket: str = Field(default="metrics", env="INFLUXDB_BUCKET")
    
    kafka_bootstrap_servers: str = Field(
        default="localhost:9092", 
        env="KAFKA_BOOTSTRAP_SERVERS"
    )
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Optimization settings
    max_optimization_time: int = Field(default=300, description="Max optimization time in seconds")
    max_concurrent_optimizations: int = Field(default=5, description="Max concurrent optimizations")
    default_risk_free_rate: float = Field(default=0.02, description="Default risk-free rate")
    
    # Security
    secret_key: str = Field(default="dev-secret-key-change-in-production", env="SECRET_KEY")
    access_token_expire_minutes: int = 30
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()