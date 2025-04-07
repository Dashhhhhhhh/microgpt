"""
Web integration module for MicroGPT agents.

This module provides classes and utilities for web-related operations,
such as making HTTP requests and processing web content.
"""

import json
import os
from typing import Any, Dict, List, Optional, Union
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

from .agent import MicroAgent
from .utils import log_calls, retry, cache_result


class WebClient:
    """
    A client for making HTTP requests and processing web content.
    
    This class provides methods for fetching web content, parsing HTML,
    and interacting with REST APIs.
    """
    
    def __init__(self, 
                 base_url: Optional[str] = None, 
                 headers: Optional[Dict[str, str]] = None,
                 timeout: int = 10):
        """
        Initialize a new WebClient.
        
        Args:
            base_url (str, optional): A base URL to prepend to all requests.
            headers (dict, optional): Default headers to include in all requests.
            timeout (int, optional): Default timeout in seconds for requests.
        """
        self.base_url = base_url
        self.headers = headers or {}
        self.timeout = timeout
        self.session = requests.Session()
        
        # Set default user agent if not provided
        if 'User-Agent' not in self.headers:
            self.headers['User-Agent'] = (
                'MicroGPT/0.1.0 (Python Web Client; '
                '+https://github.com/example/microgpt)'
            )
        
        # Apply headers to session
        self.session.headers.update(self.headers)
    
    def _build_url(self, url: str) -> str:
        """
        Build the full URL for a request.
        
        Args:
            url (str): The URL path or full URL.
            
        Returns:
            str: The full URL.
        """
        if self.base_url and not urlparse(url).netloc:
            # If url doesn't have a domain and we have a base_url, prepend it
            if url.startswith('/'):
                return f"{self.base_url}{url}"
            else:
                return f"{self.base_url}/{url}"
        return url
    
    @log_calls
    @retry(max_attempts=3, delay=2.0, backoff=2.0)
    def get(self, 
            url: str, 
            params: Optional[Dict[str, Any]] = None, 
            headers: Optional[Dict[str, str]] = None,
            timeout: Optional[int] = None) -> requests.Response:
        """
        Make a GET request.
        
        Args:
            url (str): The URL to request.
            params (dict, optional): Query parameters.
            headers (dict, optional): Additional headers.
            timeout (int, optional): Request timeout in seconds.
            
        Returns:
            requests.Response: The response object.
        """
        full_url = self._build_url(url)
        merged_headers = {**self.headers, **(headers or {})}
        return self.session.get(
            full_url, 
            params=params, 
            headers=merged_headers,
            timeout=timeout or self.timeout
        )
    
    @log_calls
    @retry(max_attempts=3, delay=2.0, backoff=2.0)
    def post(self, 
             url: str, 
             data: Optional[Dict[str, Any]] = None,
             json_data: Optional[Dict[str, Any]] = None,
             headers: Optional[Dict[str, str]] = None,
             timeout: Optional[int] = None) -> requests.Response:
        """
        Make a POST request.
        
        Args:
            url (str): The URL to request.
            data (dict, optional): Form data.
            json_data (dict, optional): JSON data.
            headers (dict, optional): Additional headers.
            timeout (int, optional): Request timeout in seconds.
            
        Returns:
            requests.Response: The response object.
        """
        full_url = self._build_url(url)
        merged_headers = {**self.headers, **(headers or {})}
        return self.session.post(
            full_url, 
            data=data, 
            json=json_data,
            headers=merged_headers,
            timeout=timeout or self.timeout
        )
    
    @log_calls
    @retry(max_attempts=2)
    def fetch_text(self, url: str, **kwargs) -> str:
        """
        Fetch text content from a URL.
        
        Args:
            url (str): The URL to fetch.
            **kwargs: Additional arguments to pass to get().
            
        Returns:
            str: The text content.
            
        Raises:
            requests.RequestException: If the request fails.
        """
        response = self.get(url, **kwargs)
        response.raise_for_status()
        return response.text
    
    @log_calls
    @retry(max_attempts=2)
    def fetch_json(self, url: str, **kwargs) -> Dict[str, Any]:
        """
        Fetch JSON content from a URL.
        
        Args:
            url (str): The URL to fetch.
            **kwargs: Additional arguments to pass to get().
            
        Returns:
            dict: The parsed JSON data.
            
        Raises:
            requests.RequestException: If the request fails.
            json.JSONDecodeError: If the response is not valid JSON.
        """
        response = self.get(url, **kwargs)
        response.raise_for_status()
        return response.json()
    
    @log_calls
    @retry(max_attempts=2)
    def fetch_html(self, url: str, **kwargs) -> BeautifulSoup:
        """
        Fetch and parse HTML content from a URL.
        
        Args:
            url (str): The URL to fetch.
            **kwargs: Additional arguments to pass to get().
            
        Returns:
            BeautifulSoup: The parsed HTML.
            
        Raises:
            requests.RequestException: If the request fails.
        """
        response = self.get(url, **kwargs)
        response.raise_for_status()
        return BeautifulSoup(response.text, 'html.parser')
    
    @cache_result(ttl=3600)  # Cache for 1 hour
    def extract_text(self, html: Union[str, BeautifulSoup]) -> str:
        """
        Extract plain text from HTML.
        
        Args:
            html (str or BeautifulSoup): The HTML content.
            
        Returns:
            str: The extracted text.
        """
        if isinstance(html, str):
            soup = BeautifulSoup(html, 'html.parser')
        else:
            soup = html
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text and clean it up
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        return '\n'.join(chunk for chunk in chunks if chunk)


class WebAgent(MicroAgent):
    """
    A MicroAgent with web capabilities.
    
    This agent can fetch and process web content, make API requests,
    and extract useful information from websites.
    """
    
    def __init__(self, 
                 name=None, 
                 description=None, 
                 llm_provider=None,
                 memory=None,
                 web_client=None):
        """
        Initialize a new WebAgent.
        
        Args:
            name (str, optional): A name for this agent.
            description (str, optional): A description of this agent's purpose.
            llm_provider (LLMProvider, optional): A language model provider.
            memory (Memory, optional): A memory storage system.
            web_client (WebClient, optional): A web client for making requests.
                If None, a default WebClient will be created.
        """
        super().__init__(name, description, llm_provider, memory)
        self.web_client = web_client or WebClient()
    
    @log_calls
    @retry(max_attempts=2, delay=1.0)
    def fetch_url(self, url: str, **kwargs) -> str:
        """
        Fetch content from a URL and return it as text.
        
        Args:
            url (str): The URL to fetch.
            **kwargs: Additional arguments to pass to the web client.
            
        Returns:
            str: The fetched content as text.
        """
        try:
            return self.web_client.fetch_text(url, **kwargs)
        except Exception as e:
            return f"Error fetching URL: {e}"
    
    @log_calls
    @retry(max_attempts=2, delay=1.0)
    def fetch_json_api(self, url: str, **kwargs) -> Dict[str, Any]:
        """
        Fetch JSON from an API endpoint.
        
        Args:
            url (str): The URL to fetch.
            **kwargs: Additional arguments to pass to the web client.
            
        Returns:
            dict: The parsed JSON data.
            
        Raises:
            requests.RequestException: If the request fails.
            json.JSONDecodeError: If the response is not valid JSON.
        """
        return self.web_client.fetch_json(url, **kwargs)
    
    @log_calls
    def search_webpage(self, url: str, query: str, **kwargs) -> str:
        """
        Search a webpage for information related to a query.
        
        This method uses the language model to extract relevant information
        from the webpage content based on the query.
        
        Args:
            url (str): The URL of the webpage to search.
            query (str): The search query.
            **kwargs: Additional arguments to pass to the language model.
            
        Returns:
            str: The relevant information extracted from the webpage.
            
        Raises:
            ValueError: If no language model provider is configured.
        """
        if not self.llm_provider:
            raise ValueError("No language model provider configured for this agent")
        
        try:
            # Fetch and parse the webpage
            html = self.web_client.fetch_html(url)
            
            # Extract text content
            text = self.web_client.extract_text(html)
            
            # If the text is too long, we need to truncate it
            # (This is a simple approach, could be improved with chunking)
            if len(text) > 4000:
                text = text[:4000] + "...[content truncated]"
            
            # Use the language model to extract relevant information
            prompt = (
                f"The following is the content of a webpage. "
                f"Extract information relevant to the query: '{query}'\n\n"
                f"Webpage content:\n{text}\n\n"
                f"Relevant information:"
            )
            
            return self.llm_provider.generate(prompt, **kwargs)
        except Exception as e:
            return f"Error searching webpage: {e}"
    
    @log_calls
    def run(self, input_data: Union[str, Dict[str, Any]]) -> str:
        """
        Process the input data and return a result.
        
        This method should be overridden by subclasses to implement
        specific web agent functionality.
        
        Args:
            input_data: The data to be processed.
            
        Returns:
            str: The processing result.
            
        Raises:
            NotImplementedError: If the subclass doesn't override this method.
        """
        raise NotImplementedError(
            "WebAgent is an abstract base class. Create a subclass "
            "that implements the run method with your specific logic."
        )