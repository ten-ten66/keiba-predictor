"""
Response Caching Layer
======================
Simple in-memory cache with TTL for API responses.
Supports both sync and async functions.
"""

import time
import logging
from functools import wraps
from typing import Optional

logger = logging.getLogger(__name__)

_cache = {}


def cached(ttl_seconds: int = 300, key_func=None):
    """Decorator to cache API responses.

    Args:
        ttl_seconds: Time-to-live in seconds (default: 5 minutes).
        key_func: Optional function to generate cache key from args/kwargs.
                  If None, uses function name + str(args) + str(kwargs).
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if key_func is not None:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{args}:{kwargs}"
            now = time.time()
            if cache_key in _cache:
                val, ts = _cache[cache_key]
                if now - ts < ttl_seconds:
                    return val
            result = await func(*args, **kwargs)
            _cache[cache_key] = (result, now)
            return result
        return wrapper
    return decorator


def invalidate(pattern: Optional[str] = None):
    """Clear cache entries matching pattern, or all if pattern is None."""
    if pattern is None:
        count = len(_cache)
        _cache.clear()
        logger.info("Cache cleared: %d entries removed", count)
    else:
        keys_to_remove = [k for k in _cache if pattern in k]
        for k in keys_to_remove:
            del _cache[k]
        if keys_to_remove:
            logger.info("Cache invalidated: %d entries matching '%s'", len(keys_to_remove), pattern)


def cache_stats() -> dict:
    """Return cache statistics."""
    now = time.time()
    total = len(_cache)
    expired = sum(1 for _, (_, ts) in _cache.items() if now - ts >= 300)
    return {
        "total_entries": total,
        "expired_entries": expired,
        "active_entries": total - expired,
    }
