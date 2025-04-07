"""
Language model integration for MicroGPT.

This module provides classes and functions for interacting with
language models like GPT through APIs.
"""

import os
from typing import Dict, List, Optional, Union
import openai
from dotenv import load_dotenv

from .utils import log_calls, retry, track_tokens, TokenTracker

# Load API keys from environment variables
load_dotenv()

class LLMProvider:
    """Base class for language model providers."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the language model provider.
        
        Args:
            api_key (str, optional): API key for the provider.
                If not provided, will attempt to load from environment variables.
        """
        self.api_key = api_key
        
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from the language model.
        
        Args:
            prompt (str): The input prompt to send to the model.
            **kwargs: Additional parameters for the generation.
            
        Returns:
            str: The generated text.
            
        Raises:
            NotImplementedError: If the subclass doesn't override this method.
        """
        raise NotImplementedError("Subclasses must implement the generate method")


class OpenAIProvider(LLMProvider):
    """OpenAI API integration for accessing models like GPT-4."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        """
        Initialize the OpenAI provider.
        
        Args:
            api_key (str, optional): OpenAI API key.
                If not provided, will attempt to load from OPENAI_API_KEY environment variable.
            model (str, optional): The model to use. Defaults to "gpt-4".
        """
        super().__init__(api_key)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Either pass it directly or "
                "set the OPENAI_API_KEY environment variable."
            )
        self.model = model
        openai.api_key = self.api_key
        
        # Initialize token tracker for this model
        self.token_tracker = TokenTracker(model=model)
        
    @log_calls
    @retry(max_attempts=3, delay=1.0, backoff=2.0)
    @track_tokens
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using OpenAI's API.
        
        Args:
            prompt (str): The input prompt to send to the model.
            **kwargs: Additional parameters to pass to the OpenAI API.
                Common options include:
                - max_tokens (int): Maximum number of tokens to generate.
                - temperature (float): Controls randomness (0.0 to 1.0).
                - top_p (float): Controls diversity via nucleus sampling.
                
        Returns:
            str: The generated text response.
        """
        # Set default parameters if not provided
        params = {
            "model": self.model,
            "temperature": 0.7,
            "max_tokens": 500,
        }
        # Update with any provided kwargs
        params.update(kwargs)
        
        # Create messages for the chat completion
        messages = [{"role": "user", "content": prompt}]
        
        # Call the OpenAI API
        response = openai.chat.completions.create(
            messages=messages,
            **params
        )
        
        # Extract and return the generated text
        return response.choices[0].message.content