"""
MicroGPT: A lightweight, modular framework for building AI agents.

This package provides tools and utilities for creating small, composable,
single-purpose AI agents powered by language models like GPT.
"""

__version__ = "0.1.0"

from .agent import MicroAgent, LLMAgent
from .core import run_agent, AgentPipeline, AgentNetwork
from .llm import LLMProvider, OpenAIProvider
from .memory import Memory, InMemoryDict, JSONFileStorage, PickleStorage
from .web import WebClient, WebAgent
from .utils import (
    TokenTracker, log_calls, retry, cache_result, track_tokens
)

__all__ = [
    # Agent classes
    "MicroAgent", 
    "LLMAgent",
    "WebAgent",
    
    # Core functions and classes 
    "run_agent", 
    "AgentPipeline", 
    "AgentNetwork",
    
    # LLM providers
    "LLMProvider",
    "OpenAIProvider",
    
    # Memory classes
    "Memory",
    "InMemoryDict",
    "JSONFileStorage",
    "PickleStorage",
    
    # Web classes
    "WebClient",
    
    # Utility classes and decorators
    "TokenTracker",
    "log_calls",
    "retry",
    "cache_result",
    "track_tokens"
]