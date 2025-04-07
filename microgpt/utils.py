"""
Utility functions and decorators for the MicroGPT framework.

This module provides decorators, logging utilities, and token tracking
functionality to enhance and monitor agent behavior.
"""

import functools
import inspect
import logging
import os
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

# Configure logging
logging.basicConfig(
    level=os.getenv("MICROGPT_LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class TokenTracker:
    """
    Track token usage for language model interactions.
    
    This class provides utilities for estimating and tracking token
    usage to help monitor costs and prevent exceeding limits.
    """
    
    # Approximate tokens per character for different languages
    # These are rough estimates and may vary by model
    TOKENS_PER_CHAR = {
        "english": 0.25,  # ~4 characters per token for English
        "chinese": 1.0,   # ~1 character per token for Chinese
        "japanese": 0.5,  # ~2 characters per token for Japanese
        "korean": 0.5,    # ~2 characters per token for Korean
        "default": 0.25,  # Default to English ratio
    }
    
    # Model-specific costs per 1000 tokens (approximate, may change)
    MODEL_COSTS = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-32k": {"input": 0.06, "output": 0.12},
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        "default": {"input": 0.0015, "output": 0.002},  # Default to gpt-3.5-turbo
    }
    
    def __init__(self, model: str = "default", language: str = "default"):
        """
        Initialize a new TokenTracker.
        
        Args:
            model (str, optional): The model name to track tokens for.
            language (str, optional): The primary language to optimize for.
        """
        self.model = model
        self.language = language
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.history = []
        
        # Get token-to-char ratio for the specified language
        self.tokens_per_char = self.TOKENS_PER_CHAR.get(
            language, self.TOKENS_PER_CHAR["default"]
        )
        
        # Get cost rates for the specified model
        self.costs = self.MODEL_COSTS.get(model, self.MODEL_COSTS["default"])
        
        self.logger = logging.getLogger(f"microgpt.TokenTracker.{model}")
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text string.
        
        This is a rough estimate based on character count and language.
        For precise counts, use the tokenizer from the specific model.
        
        Args:
            text (str): The text to estimate tokens for.
            
        Returns:
            int: Estimated token count.
        """
        return int(len(text) * self.tokens_per_char)
    
    def track_usage(self, prompt: str, response: str) -> Dict[str, Any]:
        """
        Track token usage for a prompt-response pair.
        
        Args:
            prompt (str): The input prompt.
            response (str): The output response.
            
        Returns:
            dict: Usage statistics including token counts and cost.
        """
        input_tokens = self.estimate_tokens(prompt)
        output_tokens = self.estimate_tokens(response)
        
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        
        # Calculate costs (in dollars)
        input_cost = (input_tokens / 1000) * self.costs["input"]
        output_cost = (output_tokens / 1000) * self.costs["output"]
        total_cost = input_cost + output_cost
        
        # Record usage
        usage = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost": total_cost,
        }
        
        self.history.append(usage)
        
        self.logger.debug(
            f"Usage: {input_tokens} input tokens, {output_tokens} output tokens, "
            f"${total_cost:.6f} cost"
        )
        
        return usage
    
    def get_total_usage(self) -> Dict[str, Any]:
        """
        Get total token usage statistics.
        
        Returns:
            dict: Total usage statistics.
        """
        total_tokens = self.total_input_tokens + self.total_output_tokens
        total_cost = (
            (self.total_input_tokens / 1000) * self.costs["input"] +
            (self.total_output_tokens / 1000) * self.costs["output"]
        )
        
        return {
            "model": self.model,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "history_entries": len(self.history),
        }
    
    def reset(self) -> None:
        """Reset token tracking history and counts."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.history = []


# Agent decorators

def log_calls(func: Callable) -> Callable:
    """
    Decorator to log agent method calls.
    
    This decorator logs the method name, arguments, and return value
    of agent method calls for debugging and monitoring.
    
    Args:
        func: The function to decorate.
        
    Returns:
        The wrapped function.
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        logger = logging.getLogger(f"microgpt.{self.__class__.__name__}")
        
        # Log the function call
        arg_str = ", ".join([repr(a) for a in args])
        kwarg_str = ", ".join([f"{k}={repr(v)}" for k, v in kwargs.items()])
        all_args = ", ".join(filter(None, [arg_str, kwarg_str]))
        
        logger.debug(f"Calling {func.__name__}({all_args})")
        
        # Measure execution time
        start_time = time.time()
        
        try:
            # Call the original function
            result = func(self, *args, **kwargs)
            
            # Log the result
            exec_time = time.time() - start_time
            result_repr = repr(result)
            if len(result_repr) > 100:
                result_repr = result_repr[:97] + "..."
            
            logger.debug(
                f"{func.__name__} completed in {exec_time:.3f}s: {result_repr}"
            )
            
            return result
            
        except Exception as e:
            # Log any exceptions
            exec_time = time.time() - start_time
            logger.exception(
                f"{func.__name__} failed after {exec_time:.3f}s: {type(e).__name__}: {e}"
            )
            raise
    
    return wrapper


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator to retry a function on failure.
    
    This decorator retries the function if an exception is raised,
    with exponential backoff between attempts.
    
    Args:
        max_attempts (int, optional): Maximum number of attempts. Defaults to 3.
        delay (float, optional): Initial delay between attempts in seconds. Defaults to 1.0.
        backoff (float, optional): Backoff factor for delay. Defaults to 2.0.
        
    Returns:
        The decorator function.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            logger = logging.getLogger(f"microgpt.{self.__class__.__name__}")
            
            last_exception = None
            current_delay = delay
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(self, *args, **kwargs)
                    
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_attempts:
                        logger.warning(
                            f"Attempt {attempt}/{max_attempts} for {func.__name__} "
                            f"failed: {type(e).__name__}: {e}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"All {max_attempts} attempts for {func.__name__} failed. "
                            f"Last error: {type(e).__name__}: {e}"
                        )
            
            # If all attempts failed, re-raise the last exception
            raise last_exception
        
        return wrapper
    
    return decorator


def cache_result(ttl: Optional[float] = None):
    """
    Decorator to cache function results.
    
    This decorator caches the results of a function call for a specified
    time-to-live (TTL) to avoid redundant computation or API calls.
    
    Args:
        ttl (float, optional): Time-to-live in seconds. If None, cache never expires.
        
    Returns:
        The decorator function.
    """
    def decorator(func: Callable) -> Callable:
        # Use class attribute to share cache across instances
        cache_attr_name = f"__{func.__name__}_cache"
        timestamps_attr_name = f"__{func.__name__}_timestamps"
        
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Initialize cache if it doesn't exist
            if not hasattr(self.__class__, cache_attr_name):
                setattr(self.__class__, cache_attr_name, {})
                setattr(self.__class__, timestamps_attr_name, {})
            
            cache = getattr(self.__class__, cache_attr_name)
            timestamps = getattr(self.__class__, timestamps_attr_name)
            
            # Create a cache key from args and kwargs
            # This is a simplified approach; may need more robust key generation
            key = str((args, sorted(kwargs.items())))
            
            # Check if result is in cache and not expired
            current_time = time.time()
            if key in cache:
                if ttl is None or current_time - timestamps[key] < ttl:
                    return cache[key]
            
            # Call the function and cache the result
            result = func(self, *args, **kwargs)
            cache[key] = result
            timestamps[key] = current_time
            
            return result
        
        return wrapper
    
    return decorator


def track_tokens(func: Callable) -> Callable:
    """
    Decorator to track token usage for language model interactions.
    
    This decorator works with the TokenTracker class to monitor
    token usage for cost estimation and limits.
    
    Args:
        func: The function to decorate.
        
    Returns:
        The wrapped function.
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Check if the instance has a token_tracker attribute
        if not hasattr(self, "token_tracker"):
            if hasattr(self, "llm_provider") and self.llm_provider:
                model = getattr(self.llm_provider, "model", "default")
                self.token_tracker = TokenTracker(model=model)
            else:
                self.token_tracker = TokenTracker()
        
        # Call the original function
        result = func(self, *args, **kwargs)
        
        # Track token usage if the function matches expected signature
        sig = inspect.signature(func)
        if "prompt" in sig.parameters:
            # This appears to be a function that takes a prompt and returns a response
            prompt = next((a for i, a in enumerate(args) if list(sig.parameters.keys())[i] == "prompt"), 
                          kwargs.get("prompt", ""))
            
            if prompt and result and isinstance(result, str):
                self.token_tracker.track_usage(prompt, result)
        
        return result
    
    return wrapper