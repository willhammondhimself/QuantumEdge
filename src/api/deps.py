"""
Dependency injection for QuantumEdge API.
"""

import logging
from typing import Generator, Optional

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

from .config import settings

logger = logging.getLogger(__name__)


def get_redis_client() -> Optional[object]:
    """
    Get Redis client for caching and session management.

    Returns:
        Redis client or None if not available
    """
    if not REDIS_AVAILABLE:
        logger.warning("Redis not available - caching disabled")
        return None

    try:
        client = redis.from_url(settings.redis_url)
        # Test connection
        client.ping()
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        return None


def get_logger(name: str) -> logging.Logger:
    """
    Get configured logger.

    Args:
        name: Logger name

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(settings.log_format)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, settings.log_level.upper()))

    return logger


class OptimizationManager:
    """Manages running optimizations and resources."""

    def __init__(self):
        self.running_optimizations = {}
        self.max_concurrent = settings.max_concurrent_optimizations

    def can_start_optimization(self) -> bool:
        """Check if new optimization can be started."""
        return len(self.running_optimizations) < self.max_concurrent

    def start_optimization(self, optimization_id: str) -> bool:
        """
        Start tracking an optimization.

        Args:
            optimization_id: Unique optimization identifier

        Returns:
            True if started successfully
        """
        if not self.can_start_optimization():
            return False

        self.running_optimizations[optimization_id] = {
            "status": "running",
            "start_time": None,  # Will be set by actual optimizer
        }
        return True

    def finish_optimization(self, optimization_id: str):
        """Finish tracking an optimization."""
        self.running_optimizations.pop(optimization_id, None)

    def get_optimization_status(self, optimization_id: str) -> Optional[dict]:
        """Get status of a running optimization."""
        return self.running_optimizations.get(optimization_id)


# Global optimization manager
optimization_manager = OptimizationManager()


def get_optimization_manager() -> OptimizationManager:
    """Get the global optimization manager."""
    return optimization_manager
