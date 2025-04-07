"""
This module defines the core MicroAgent class for the MicroGPT framework.
"""

from typing import Optional, Dict, Any, Union, List
from .llm import LLMProvider, OpenAIProvider
from .memory import Memory, InMemoryDict
from .utils import log_calls, retry, cache_result, track_tokens

class MicroAgent:
    """
    A base class for building small, modular AI agents.
    
    This class provides the fundamental structure for creating agents
    that can process inputs and generate outputs, potentially using
    language models like GPT.
    """
    
    def __init__(self, name=None, description=None, llm_provider=None, memory=None):
        """
        Initialize a new MicroAgent.
        
        Args:
            name (str, optional): A name for this agent. Defaults to None.
            description (str, optional): A description of this agent's purpose. Defaults to None.
            llm_provider (LLMProvider, optional): A language model provider to use.
                If None, language model features will be unavailable.
            memory (Memory, optional): A memory storage system.
                If None, a default InMemoryDict will be used.
        """
        self.name = name or self.__class__.__name__
        self.description = description or "A MicroGPT agent"
        self.llm_provider = llm_provider
        
        # Set up memory
        self.memory = memory if memory is not None else InMemoryDict()
        
    @log_calls
    def run(self, input_data: Any) -> Any:
        """
        Process the input data and return a result.
        
        This is the main method that should be overridden by subclasses
        to implement specific agent functionality.
        
        Args:
            input_data: The data to be processed by this agent.
            
        Returns:
            The processed result.
            
        Raises:
            NotImplementedError: If the subclass doesn't override this method.
        """
        raise NotImplementedError("Subclasses must implement the run method")
    
    @log_calls
    @retry(max_attempts=2)
    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text using the configured language model.
        
        Args:
            prompt (str): The prompt to send to the language model.
            **kwargs: Additional parameters to pass to the language model.
            
        Returns:
            str: The generated text.
            
        Raises:
            ValueError: If no language model provider is configured.
        """
        if not self.llm_provider:
            raise ValueError("No language model provider configured for this agent")
        return self.llm_provider.generate(prompt, **kwargs)
    
    # Memory management methods
    
    @log_calls
    def remember(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Store a value in the agent's memory.
        
        Args:
            key (str): The key to store the value under.
            value (Any): The value to store.
            metadata (dict, optional): Additional metadata to store with the value.
        """
        self.memory.store(key, value, metadata)
    
    @log_calls
    @cache_result(ttl=60)  # Cache results for 60 seconds
    def recall(self, key: str) -> Any:
        """
        Retrieve a value from the agent's memory.
        
        Args:
            key (str): The key of the value to retrieve.
            
        Returns:
            The stored value, or None if the key doesn't exist.
        """
        return self.memory.retrieve(key)
    
    @log_calls
    def recall_with_metadata(self, key: str) -> Dict[str, Any]:
        """
        Retrieve a value and its metadata from the agent's memory.
        
        Args:
            key (str): The key of the value to retrieve.
            
        Returns:
            A dictionary containing the value and its metadata,
            or None if the key doesn't exist.
        """
        return self.memory.retrieve_with_metadata(key)
    
    @log_calls
    def has_memory(self, key: str) -> bool:
        """
        Check if a key exists in the agent's memory.
        
        Args:
            key (str): The key to check.
            
        Returns:
            bool: True if the key exists, False otherwise.
        """
        return self.memory.contains(key)
    
    @log_calls
    def forget(self, key: str) -> bool:
        """
        Remove a value from the agent's memory.
        
        Args:
            key (str): The key of the value to remove.
            
        Returns:
            bool: True if the key was removed, False if it didn't exist.
        """
        return self.memory.remove(key)
    
    @log_calls
    def forget_all(self) -> None:
        """Remove all values from the agent's memory."""
        self.memory.clear()
    
    @log_calls
    def memories(self) -> List[str]:
        """
        Get all keys in the agent's memory.
        
        Returns:
            list: A list of all keys in memory.
        """
        return self.memory.get_keys()


class LLMAgent(MicroAgent):
    """
    A MicroAgent that uses a language model to process input.
    
    This provides a simple wrapper around a language model,
    converting input into a prompt and returning the model's response.
    """
    
    def __init__(self, 
                 name=None, 
                 description=None, 
                 llm_provider=None,
                 prompt_template=None,
                 memory=None,
                 include_memory_in_prompts=False):
        """
        Initialize a new LLMAgent.
        
        Args:
            name (str, optional): A name for this agent. Defaults to None.
            description (str, optional): A description of this agent's purpose. Defaults to None.
            llm_provider (LLMProvider, optional): A language model provider.
                If None, will create a default OpenAIProvider.
            prompt_template (str, optional): A template for formatting prompts.
                Use {input} as a placeholder for the input data.
                Defaults to a simple template that includes the agent's description.
            memory (Memory, optional): A memory storage system.
                If None, a default InMemoryDict will be used.
            include_memory_in_prompts (bool, optional): Whether to include the agent's 
                memories in prompts sent to the language model. Defaults to False.
        """
        super().__init__(name, description, llm_provider, memory)
        
        # Create a default OpenAI provider if none is provided
        if not self.llm_provider:
            try:
                self.llm_provider = OpenAIProvider()
            except ValueError as e:
                # If OpenAI API key is missing, warn but continue
                self.llm_provider = None
                print(f"Warning: {e}")
                print("LLMAgent created without a language model provider.")
        
        # Set up the prompt template
        if prompt_template:
            self.prompt_template = prompt_template
        else:
            self.prompt_template = (
                f"You are {self.name}, {self.description}\n\n"
                "Input: {input}\n\n"
                "Response:"
            )
        
        self.include_memory_in_prompts = include_memory_in_prompts
    
    def _format_memories_for_prompt(self) -> str:
        """
        Format the agent's memories for inclusion in a prompt.
        
        Returns:
            str: A formatted string of memories, or an empty string if there are none.
        """
        if not self.memories():
            return ""
        
        memory_str = "Relevant information from memory:\n"
        for key in self.memories():
            value = self.recall(key)
            memory_str += f"- {key}: {value}\n"
        
        return memory_str + "\n"
    
    @log_calls
    @track_tokens
    def run(self, input_data: Union[str, Dict[str, Any]]) -> str:
        """
        Process the input with the language model and return the response.
        
        Args:
            input_data: Either a string to use directly in the prompt,
                       or a dictionary of values to format into the prompt template.
            
        Returns:
            str: The generated response from the language model.
            
        Raises:
            ValueError: If no language model provider is configured,
                       or if input_data is a dictionary but doesn't contain
                       all the keys needed by the prompt template.
        """
        if not self.llm_provider:
            raise ValueError("No language model provider configured for this agent")
        
        # Format the prompt
        if isinstance(input_data, dict):
            # If input_data is a dictionary, use it for template formatting
            try:
                prompt = self.prompt_template.format(**input_data)
            except KeyError as e:
                raise ValueError(f"Prompt template requires key {e} which is not in input_data")
        else:
            # Otherwise, assume it's a string and use it as {input}
            prompt = self.prompt_template.format(input=input_data)
        
        # Include memories if configured to do so
        if self.include_memory_in_prompts:
            memories = self._format_memories_for_prompt()
            if memories:
                # Insert memories before the "Response:" part
                if "Response:" in prompt:
                    parts = prompt.split("Response:", 1)
                    prompt = parts[0] + memories + "Response:" + parts[1]
                else:
                    prompt += "\n" + memories
        
        # Store the input in memory
        self.remember("last_input", input_data)
        
        # Generate the response
        response = self.llm_provider.generate(prompt)
        
        # Store the response in memory
        self.remember("last_response", response)
        
        return response