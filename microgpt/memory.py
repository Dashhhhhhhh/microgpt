"""
Memory management system for MicroGPT agents.

This module provides classes and utilities for storing and retrieving
information that agents can use across interactions.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import json
import os
import pickle
import time
from datetime import datetime


class Memory(ABC):
    """
    Abstract base class for MicroGPT memory storage systems.
    
    This class defines the interface that all memory implementations must follow,
    allowing for different storage backends to be used interchangeably.
    """
    
    @abstractmethod
    def store(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Store a value in memory.
        
        Args:
            key (str): The key to store the value under.
            value (Any): The value to store.
            metadata (dict, optional): Additional metadata to store with the value.
        """
        pass
    
    @abstractmethod
    def retrieve(self, key: str) -> Any:
        """
        Retrieve a value from memory.
        
        Args:
            key (str): The key of the value to retrieve.
            
        Returns:
            The stored value, or None if the key doesn't exist.
        """
        pass
    
    @abstractmethod
    def retrieve_with_metadata(self, key: str) -> Dict[str, Any]:
        """
        Retrieve a value and its metadata from memory.
        
        Args:
            key (str): The key of the value to retrieve.
            
        Returns:
            A dictionary containing the value and its metadata,
            or None if the key doesn't exist.
        """
        pass
    
    @abstractmethod
    def contains(self, key: str) -> bool:
        """
        Check if a key exists in memory.
        
        Args:
            key (str): The key to check.
            
        Returns:
            bool: True if the key exists, False otherwise.
        """
        pass
    
    @abstractmethod
    def remove(self, key: str) -> bool:
        """
        Remove a value from memory.
        
        Args:
            key (str): The key of the value to remove.
            
        Returns:
            bool: True if the key was removed, False if it didn't exist.
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Remove all values from memory."""
        pass
    
    @abstractmethod
    def get_keys(self) -> List[str]:
        """
        Get all keys in memory.
        
        Returns:
            list: A list of all keys in memory.
        """
        pass


class InMemoryDict(Memory):
    """
    A simple in-memory dictionary-based implementation of Memory.
    
    This implementation stores all data in a Python dictionary,
    which means the data is lost when the program exits.
    """
    
    def __init__(self):
        """Initialize a new in-memory storage."""
        self._storage = {}  # key -> (value, metadata) mapping
    
    def store(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Store a value in memory with optional metadata."""
        if metadata is None:
            metadata = {}
        
        # Add timestamp if not provided
        if "timestamp" not in metadata:
            metadata["timestamp"] = datetime.now().isoformat()
        
        self._storage[key] = (value, metadata)
    
    def retrieve(self, key: str) -> Any:
        """Retrieve a value from memory."""
        if key in self._storage:
            return self._storage[key][0]  # Just the value
        return None
    
    def retrieve_with_metadata(self, key: str) -> Dict[str, Any]:
        """Retrieve a value and its metadata from memory."""
        if key in self._storage:
            value, metadata = self._storage[key]
            return {"value": value, "metadata": metadata}
        return None
    
    def contains(self, key: str) -> bool:
        """Check if a key exists in memory."""
        return key in self._storage
    
    def remove(self, key: str) -> bool:
        """Remove a value from memory."""
        if key in self._storage:
            del self._storage[key]
            return True
        return False
    
    def clear(self) -> None:
        """Remove all values from memory."""
        self._storage.clear()
    
    def get_keys(self) -> List[str]:
        """Get all keys in memory."""
        return list(self._storage.keys())


class JSONFileStorage(Memory):
    """
    A file-based implementation of Memory using JSON for storage.
    
    This implementation persists data to a JSON file on disk,
    so the data is retained between program runs.
    """
    
    def __init__(self, file_path: str):
        """
        Initialize a new JSON file storage.
        
        Args:
            file_path (str): Path to the JSON file to use for storage.
        """
        self.file_path = file_path
        self._storage = {}
        
        # Create the file if it doesn't exist
        if not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump({}, f)
        else:
            # Load existing data
            try:
                with open(file_path, 'r') as f:
                    self._storage = json.load(f)
            except json.JSONDecodeError:
                # If the file is corrupt, create a new empty storage
                self._storage = {}
    
    def _save(self) -> None:
        """Save the current storage to the file."""
        with open(self.file_path, 'w') as f:
            json.dump(self._storage, f, indent=2)
    
    def store(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Store a value in memory with optional metadata."""
        if metadata is None:
            metadata = {}
        
        # Add timestamp if not provided
        if "timestamp" not in metadata:
            metadata["timestamp"] = datetime.now().isoformat()
        
        # Store the value and metadata
        # Note: This assumes the value is JSON-serializable
        self._storage[key] = {
            "value": value,
            "metadata": metadata
        }
        
        # Save to file
        self._save()
    
    def retrieve(self, key: str) -> Any:
        """Retrieve a value from memory."""
        if key in self._storage:
            return self._storage[key]["value"]
        return None
    
    def retrieve_with_metadata(self, key: str) -> Dict[str, Any]:
        """Retrieve a value and its metadata from memory."""
        if key in self._storage:
            return self._storage[key]
        return None
    
    def contains(self, key: str) -> bool:
        """Check if a key exists in memory."""
        return key in self._storage
    
    def remove(self, key: str) -> bool:
        """Remove a value from memory."""
        if key in self._storage:
            del self._storage[key]
            self._save()
            return True
        return False
    
    def clear(self) -> None:
        """Remove all values from memory."""
        self._storage.clear()
        self._save()
    
    def get_keys(self) -> List[str]:
        """Get all keys in memory."""
        return list(self._storage.keys())


class PickleStorage(Memory):
    """
    A file-based implementation of Memory using pickle for storage.
    
    This implementation persists data to a pickle file on disk,
    allowing for storage of arbitrary Python objects, not just JSON-serializable ones.
    """
    
    def __init__(self, file_path: str):
        """
        Initialize a new pickle file storage.
        
        Args:
            file_path (str): Path to the pickle file to use for storage.
        """
        self.file_path = file_path
        self._storage = {}
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Load existing data if the file exists
        if os.path.exists(file_path):
            try:
                with open(file_path, 'rb') as f:
                    self._storage = pickle.load(f)
            except (pickle.PickleError, EOFError):
                # If the file is corrupt, create a new empty storage
                self._storage = {}
    
    def _save(self) -> None:
        """Save the current storage to the file."""
        with open(self.file_path, 'wb') as f:
            pickle.dump(self._storage, f)
    
    def store(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Store a value in memory with optional metadata."""
        if metadata is None:
            metadata = {}
        
        # Add timestamp if not provided
        if "timestamp" not in metadata:
            metadata["timestamp"] = datetime.now().isoformat()
        
        # Store the value and metadata
        self._storage[key] = {
            "value": value,
            "metadata": metadata
        }
        
        # Save to file
        self._save()
    
    def retrieve(self, key: str) -> Any:
        """Retrieve a value from memory."""
        if key in self._storage:
            return self._storage[key]["value"]
        return None
    
    def retrieve_with_metadata(self, key: str) -> Dict[str, Any]:
        """Retrieve a value and its metadata from memory."""
        if key in self._storage:
            return self._storage[key]
        return None
    
    def contains(self, key: str) -> bool:
        """Check if a key exists in memory."""
        return key in self._storage
    
    def remove(self, key: str) -> bool:
        """Remove a value from memory."""
        if key in self._storage:
            del self._storage[key]
            self._save()
            return True
        return False
    
    def clear(self) -> None:
        """Remove all values from memory."""
        self._storage.clear()
        self._save()
    
    def get_keys(self) -> List[str]:
        """Get all keys in memory."""
        return list(self._storage.keys())