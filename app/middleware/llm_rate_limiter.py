"""
LLM Rate Limiter - Queue-based concurrency control for LLM API calls.

This module provides rate limiting for LLM API calls to prevent 529 "Overloaded"
errors from providers like OpenRouter. It implements:

1. **Global concurrency limit**: Max parallel LLM calls across all users
2. **Request queuing**: Calls wait their turn instead of failing
3. **Retry with backoff**: Automatic retry on transient 529/503 errors

Usage:
    from app.middleware.llm_rate_limiter import wrap_model_with_rate_limiting

    # Wrap any LangChain chat model or runnable
    base_model = ChatOpenAI(model="claude-3.5-sonnet", ...)
    model = wrap_model_with_rate_limiting(base_model)

    # Use as normal - rate limiting is transparent
    response = await model.ainvoke(messages)
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, AsyncIterator, Optional, Union

from aiolimiter import AsyncLimiter
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

logger = logging.getLogger(__name__)


def is_retryable_error(exception: Exception) -> bool:
    """
    Check if an exception is retryable (transient API errors).

    Retryable errors:
    - 529: Overloaded (provider capacity)
    - 503: Service Unavailable
    - 502: Bad Gateway
    - 504: Gateway Timeout
    - Rate limit errors
    """
    error_str = str(exception).lower()
    retryable_indicators = [
        "529",
        "overload",
        "503",
        "502",
        "504",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
        "rate limit",
        "too many requests",
        "capacity",
        "temporarily unavailable",
    ]
    return any(indicator in error_str for indicator in retryable_indicators)


class LLMConcurrencyManager:
    """
    Singleton manager for LLM API call concurrency.

    Provides a global semaphore and rate limiter that all LLM calls share.
    This ensures that even when multiple sub-agents run in parallel,
    their combined API calls stay within safe limits.
    """

    _instance: Optional[LLMConcurrencyManager] = None
    _lock: asyncio.Lock

    def __init__(
        self,
        max_concurrent: int = 5,
        requests_per_second: float = 3.0,
    ):
        """
        Initialize the concurrency manager.

        Args:
            max_concurrent: Maximum parallel LLM API calls (default: 5)
            requests_per_second: Max requests per second (default: 3.0)
        """
        self.max_concurrent = max_concurrent
        self.requests_per_second = requests_per_second

        # Semaphore for concurrent request limiting
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # Token bucket rate limiter for smoothing request rate
        self._rate_limiter = AsyncLimiter(
            max_rate=requests_per_second,
            time_period=1.0,
        )

        # Stats for monitoring
        self._active_calls = 0
        self._total_calls = 0
        self._retried_calls = 0

        logger.info(
            f"LLM Concurrency Manager initialized: "
            f"max_concurrent={max_concurrent}, rps={requests_per_second}"
        )

    @classmethod
    def _get_lock(cls) -> asyncio.Lock:
        """Get or create the class-level lock."""
        if not hasattr(cls, '_lock') or cls._lock is None:
            cls._lock = asyncio.Lock()
        return cls._lock

    @classmethod
    async def get_instance(
        cls,
        max_concurrent: int = 5,
        requests_per_second: float = 3.0,
    ) -> LLMConcurrencyManager:
        """Get or create the singleton instance."""
        if cls._instance is None:
            lock = cls._get_lock()
            async with lock:
                if cls._instance is None:
                    cls._instance = cls(
                        max_concurrent=max_concurrent,
                        requests_per_second=requests_per_second,
                    )
        return cls._instance

    @classmethod
    def get_instance_sync(
        cls,
        max_concurrent: int = 5,
        requests_per_second: float = 3.0,
    ) -> LLMConcurrencyManager:
        """Get or create the singleton instance (sync version for init)."""
        if cls._instance is None:
            cls._instance = cls(
                max_concurrent=max_concurrent,
                requests_per_second=requests_per_second,
            )
        return cls._instance

    async def acquire(self) -> None:
        """
        Acquire a slot for making an LLM API call.

        This method:
        1. Adds random jitter to prevent thundering herd
        2. Waits for a semaphore slot (concurrency limit)
        3. Waits for rate limiter (requests per second)
        """
        # Add jitter to stagger concurrent requests (0-100ms)
        jitter = random.uniform(0, 0.1)
        await asyncio.sleep(jitter)

        # Wait for semaphore slot
        await self._semaphore.acquire()
        self._active_calls += 1
        self._total_calls += 1

        # Wait for rate limiter
        await self._rate_limiter.acquire()

        logger.debug(
            f"LLM call acquired: active={self._active_calls}/{self.max_concurrent}, "
            f"total={self._total_calls}"
        )

    def release(self) -> None:
        """Release the slot after an LLM API call completes."""
        self._semaphore.release()
        self._active_calls -= 1
        logger.debug(f"LLM call released: active={self._active_calls}/{self.max_concurrent}")

    def record_retry(self) -> None:
        """Record that a call was retried."""
        self._retried_calls += 1

    def get_stats(self) -> dict[str, Any]:
        """Get current stats for monitoring."""
        return {
            "active_calls": self._active_calls,
            "max_concurrent": self.max_concurrent,
            "total_calls": self._total_calls,
            "retried_calls": self._retried_calls,
            "requests_per_second": self.requests_per_second,
        }


class RateLimitedChatModel(Runnable):
    """
    Wrapper that adds rate limiting and retry to any LangChain chat model or Runnable.

    This wrapper:
    1. Queues LLM calls through a global concurrency manager
    2. Retries on transient errors (529, 503, etc.) with exponential backoff
    3. Is transparent to the rest of the application

    Works with BaseChatModel, RunnableBinding, and any other Runnable.
    """

    def __init__(
        self,
        model: Union[BaseChatModel, Runnable, Any],
        max_retries: int = 3,
        min_retry_wait: float = 1.0,
        max_retry_wait: float = 30.0,
    ):
        """
        Initialize the rate-limited model wrapper.

        Args:
            model: The underlying chat model/runnable to wrap
            max_retries: Maximum retry attempts for transient errors
            min_retry_wait: Minimum wait between retries (seconds)
            max_retry_wait: Maximum wait between retries (seconds)
        """
        self.wrapped_model = model
        self.max_retries = max_retries
        self.min_retry_wait = min_retry_wait
        self.max_retry_wait = max_retry_wait

    @property
    def InputType(self) -> type:
        """Return the input type of the wrapped model."""
        return getattr(self.wrapped_model, 'InputType', Any)

    @property
    def OutputType(self) -> type:
        """Return the output type of the wrapped model."""
        return getattr(self.wrapped_model, 'OutputType', Any)

    def invoke(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Any:
        """Synchronous invocation - delegates to wrapped model."""
        return self.wrapped_model.invoke(input, config=config, **kwargs)

    async def ainvoke(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Async invocation with rate limiting and retry.

        This is the main entry point for async LLM calls.
        """
        manager = await LLMConcurrencyManager.get_instance()

        @retry(
            retry=retry_if_exception(is_retryable_error),
            wait=wait_random_exponential(
                multiplier=1,
                min=self.min_retry_wait,
                max=self.max_retry_wait,
            ),
            stop=stop_after_attempt(self.max_retries),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        async def _call_with_retry() -> Any:
            await manager.acquire()
            try:
                return await self.wrapped_model.ainvoke(input, config=config, **kwargs)
            except Exception as e:
                if is_retryable_error(e):
                    manager.record_retry()
                    logger.warning(f"Retryable LLM error: {e}")
                raise
            finally:
                manager.release()

        return await _call_with_retry()

    async def astream(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """
        Async streaming with rate limiting and retry.

        For streaming, we acquire the rate limit slot before starting
        and release it when the stream completes or errors.
        """
        manager = await LLMConcurrencyManager.get_instance()
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries):
            await manager.acquire()
            try:
                async for chunk in self.wrapped_model.astream(input, config=config, **kwargs):
                    yield chunk
                # Stream completed successfully
                return
            except Exception as e:
                last_error = e
                if is_retryable_error(e) and attempt < self.max_retries - 1:
                    manager.record_retry()
                    wait_time = min(
                        self.max_retry_wait,
                        self.min_retry_wait * (2 ** attempt) + random.uniform(0, 1),
                    )
                    logger.warning(
                        f"Retryable streaming error (attempt {attempt + 1}/{self.max_retries}): {e}. "
                        f"Retrying in {wait_time:.1f}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise
            finally:
                manager.release()

        if last_error:
            raise last_error

    def bind_tools(self, *args: Any, **kwargs: Any) -> "RateLimitedChatModel":
        """Bind tools to the wrapped model."""
        bound = self.wrapped_model.bind_tools(*args, **kwargs)
        return RateLimitedChatModel(
            model=bound,
            max_retries=self.max_retries,
            min_retry_wait=self.min_retry_wait,
            max_retry_wait=self.max_retry_wait,
        )

    def with_config(self, *args: Any, **kwargs: Any) -> "RateLimitedChatModel":
        """Add config to the wrapped model."""
        configured = self.wrapped_model.with_config(*args, **kwargs)
        return RateLimitedChatModel(
            model=configured,
            max_retries=self.max_retries,
            min_retry_wait=self.min_retry_wait,
            max_retry_wait=self.max_retry_wait,
        )

    def with_structured_output(self, *args: Any, **kwargs: Any) -> "RateLimitedChatModel":
        """Add structured output to the wrapped model."""
        structured = self.wrapped_model.with_structured_output(*args, **kwargs)
        return RateLimitedChatModel(
            model=structured,
            max_retries=self.max_retries,
            min_retry_wait=self.min_retry_wait,
            max_retry_wait=self.max_retry_wait,
        )

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the wrapped model."""
        return getattr(self.wrapped_model, name)


def wrap_model_with_rate_limiting(
    model: Union[BaseChatModel, Runnable, Any],
    max_retries: int = 3,
    min_retry_wait: float = 1.0,
    max_retry_wait: float = 30.0,
) -> RateLimitedChatModel:
    """
    Wrap a chat model or runnable with rate limiting.

    Args:
        model: The chat model or runnable to wrap
        max_retries: Maximum retry attempts (default: 3)
        min_retry_wait: Minimum retry wait in seconds (default: 1.0)
        max_retry_wait: Maximum retry wait in seconds (default: 30.0)

    Returns:
        A RateLimitedChatModel wrapping the original model
    """
    return RateLimitedChatModel(
        model=model,
        max_retries=max_retries,
        min_retry_wait=min_retry_wait,
        max_retry_wait=max_retry_wait,
    )


def configure_global_rate_limits(
    max_concurrent: int = 5,
    requests_per_second: float = 3.0,
) -> None:
    """
    Configure the global rate limits before creating any models.

    This should be called once at application startup.

    Args:
        max_concurrent: Maximum parallel LLM API calls
        requests_per_second: Maximum requests per second
    """
    LLMConcurrencyManager.get_instance_sync(
        max_concurrent=max_concurrent,
        requests_per_second=requests_per_second,
    )
    logger.info(
        f"Global LLM rate limits configured: "
        f"max_concurrent={max_concurrent}, rps={requests_per_second}"
    )
